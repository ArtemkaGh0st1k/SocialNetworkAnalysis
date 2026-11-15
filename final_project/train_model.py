import json
from os.path import join
from pathlib import Path
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLEAN_DATASET_PATH = Path(join("final_project", "data", "posts_clean.csv"))
RESULTS_DIR = Path(join("final_project", "results", "model"))
MODELS_DIR = Path(join("final_project", "models"))

STATE = 42
VALID_START = "2026-03"         # обучение - до марта 2026, проверка - март-июнь
TEST_START = "2026-07"          # тест - с июля 2026, на нём модели оцениваются один раз
TARGET_LIMIT = 3                # отклик больше чем в e^3 = 20 раз считается выбросом
SVD_COMPONENTS = 64
RIDGE_ALPHAS = [1, 3, 10, 30, 100]
DECILES = 10

TARGET = "log_relative_engagement"
NUMERIC_FEATURES = \
[
    "n_photos", "text_length", "n_lines", "n_hashtags", "n_links", "n_emoji",
    "n_exclamations", "hour", "weekday", "month_of_year",
]
BOOLEAN_FEATURES = ["has_video", "has_link", "has_poll", "has_doc", "has_question"]
CATEGORICAL_FEATURES = ["group"]
RIDGE_CATEGORICAL_FEATURES = ["group", "hour", "weekday", "month_of_year"]
COUNT_FEATURES = ["text_length", "n_lines", "n_hashtags", "n_links", "n_emoji", "n_exclamations"]

COLOR_MAIN = "#2a78d6"
COLOR_BASELINE = "#898781"
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"


class EngagementModelTrainer():

    def __init__(self, dataset_path=CLEAN_DATASET_PATH):

        posts = pd.read_csv(dataset_path, encoding="utf-8-sig")
        posts["lemmas"] = posts["lemmas"].fillna("")
        posts["month_of_year"] = posts["month"].str[5:].astype(int)
        posts[TARGET] = posts[TARGET].clip(-TARGET_LIMIT, TARGET_LIMIT)

        # разбиение по времени: модель учится на прошлом и проверяется на будущем
        self.train = posts[posts["month"] < VALID_START]
        self.valid = posts[(posts["month"] >= VALID_START) & (posts["month"] < TEST_START)]
        self.test = posts[posts["month"] >= TEST_START]

        self.predictions = {}
        self.metrics = {}

        print(f"Постов: обучение {len(self.train)}, проверка {len(self.valid)}, тест {len(self.test)}")


    def evaluate(self, name, predicted, data):
        """Считает метрики модели на выборке"""

        actual = data[TARGET].values

        # у постоянного прогноза ранговая корреляция не определена
        spearman = None if np.std(predicted) == 0 else round(float(spearmanr(actual, predicted).statistic), 4)

        return \
        {
            "MAE" : round(float(mean_absolute_error(actual, predicted)), 4),
            "R2" : round(float(r2_score(actual, predicted)), 4),
            "Spearman" : spearman,
            # угадан ли знак: пост выше или ниже типичного для своего сообщества
            "sign_accuracy" : round(float(np.mean((actual > 0) == (predicted > 0))), 4),
        }


    def add_result(self, name, valid_predicted, test_predicted):
        """Запоминает прогнозы и метрики модели"""

        self.predictions[name] = test_predicted
        self.metrics[name] = \
        {
            "valid" : self.evaluate(name, valid_predicted, self.valid),
            "test" : self.evaluate(name, test_predicted, self.test),
        }
        print(f"{name}: тест {self.metrics[name]['test']}")


    def fit_baselines(self):
        """Простые решения, с которыми сравниваются модели"""

        # типичный пост: отклонение 0, то есть прогноз «как обычно»
        self.add_result("Типичный пост", np.zeros(len(self.valid)), np.zeros(len(self.test)))

        # медиана по формату поста из обучающей выборки
        def post_format(data):
            return np.select(
                [data["has_poll"], data["has_video"], data["n_photos"] >= 5, data["n_photos"] >= 2,
                 data["n_photos"] == 1, data["has_link"]],
                ["poll", "video", "photos_5", "photos_2", "photo_1", "link"], default="text"
            )

        medians = self.train.groupby(post_format(self.train))[TARGET].median()
        self.add_result(
            "Медиана по формату",
            pd.Series(post_format(self.valid)).map(medians).fillna(0).values,
            pd.Series(post_format(self.test)).map(medians).fillna(0).values,
        )


    def build_text_vectorizer(self):
        """Создаёт TF-IDF векторизатор лемм"""

        return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5,
                               max_features=50000, sublinear_tf=True)


    def tabular_matrix(self, data):
        """Числовые признаки для линейной модели: счётчики сжимаются логарифмом"""

        table = data[NUMERIC_FEATURES + BOOLEAN_FEATURES].astype(float).copy()
        table[COUNT_FEATURES] = np.log1p(table[COUNT_FEATURES])

        return table.drop(columns=["hour", "weekday", "month_of_year"]).values


    def ridge_matrix(self, data, parts):
        """Признаки для линейной модели: слова, оформление и категории"""

        return hstack([
            parts["vectorizer"].transform(data["lemmas"]),
            parts["scaler"].transform(self.tabular_matrix(data)),
            parts["encoder"].transform(data[RIDGE_CATEGORICAL_FEATURES]),
        ]).tocsr()


    def fit_ridge(self):
        """Линейная модель на TF-IDF и признаках оформления"""

        parts = \
        {
            "vectorizer" : self.build_text_vectorizer().fit(self.train["lemmas"]),
            "scaler" : StandardScaler().fit(self.tabular_matrix(self.train)),
            "encoder" : OneHotEncoder(handle_unknown="ignore").fit(self.train[RIDGE_CATEGORICAL_FEATURES]),
        }

        train_x = self.ridge_matrix(self.train, parts)
        valid_x = self.ridge_matrix(self.valid, parts)
        test_x = self.ridge_matrix(self.test, parts)

        # силу регуляризации выбираем на проверочной выборке, тест не трогаем
        scores = {}
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha).fit(train_x, self.train[TARGET])
            scores[alpha] = mean_absolute_error(self.valid[TARGET], model.predict(valid_x))

        best_alpha = min(scores, key=scores.get)
        model = Ridge(alpha=best_alpha).fit(train_x, self.train[TARGET])

        print(f"Ridge: лучшая alpha = {best_alpha}")
        self.add_result("Ridge (TF-IDF)", model.predict(valid_x), model.predict(test_x))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({**parts, "model" : model}, MODELS_DIR / "ridge.joblib")


    def catboost_matrix(self, data, svd, vectorizer):
        """Признаки для CatBoost: оформление, время, сообщество и сжатый текст"""

        table = data[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES].copy()
        table[BOOLEAN_FEATURES] = table[BOOLEAN_FEATURES].astype(int)

        text = svd.transform(vectorizer.transform(data["lemmas"]))
        text = pd.DataFrame(text, index=data.index, columns=[f"svd_{i}" for i in range(text.shape[1])])

        return pd.concat([table, text], axis=1)


    def fit_catboost(self):
        """Градиентный бустинг на признаках и сжатом TF-IDF"""

        vectorizer = self.build_text_vectorizer().fit(self.train["lemmas"])
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=STATE).fit(
            vectorizer.transform(self.train["lemmas"])
        )

        train_x = self.catboost_matrix(self.train, svd, vectorizer)
        valid_x = self.catboost_matrix(self.valid, svd, vectorizer)
        test_x = self.catboost_matrix(self.test, svd, vectorizer)

        model = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, loss_function="RMSE",
                                  random_seed=STATE, verbose=0, cat_features=CATEGORICAL_FEATURES,
                                  allow_writing_files=False)

        # число деревьев подбирается по проверочной выборке
        model.fit(train_x, self.train[TARGET], eval_set=(valid_x, self.valid[TARGET]),
                  early_stopping_rounds=200)

        print(f"CatBoost: деревьев {model.get_best_iteration() + 1}")
        self.add_result("CatBoost", model.predict(valid_x), model.predict(test_x))

        importance = pd.Series(model.get_feature_importance(), index=train_x.columns)
        text_importance = importance[importance.index.str.startswith("svd_")].sum()
        importance = importance[~importance.index.str.startswith("svd_")]
        importance["текст (все компоненты)"] = text_importance
        self.metrics["CatBoost"]["feature_importance"] = importance.sort_values(ascending=False).round(2).to_dict()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(MODELS_DIR / "catboost.cbm")
        joblib.dump({"vectorizer" : vectorizer, "svd" : svd}, MODELS_DIR / "text_features.joblib")


    def style_axes(self, ax, title, xlabel=None, ylabel=None, grid_axis="x"):
        """Оформляет оси в едином стиле"""

        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=COLOR_INK)
        if xlabel:
            ax.set_xlabel(xlabel, color=COLOR_SECONDARY)
        if ylabel:
            ax.set_ylabel(ylabel, color=COLOR_SECONDARY)

        ax.set_facecolor(COLOR_SURFACE)
        ax.grid(axis=grid_axis, color=COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=COLOR_SECONDARY, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(COLOR_AXIS)


    def plot_comparison(self):
        """Сравнение моделей на тестовой выборке"""

        names = list(self.metrics)
        colors = [COLOR_BASELINE if name in ("Типичный пост", "Медиана по формату") else COLOR_MAIN
                  for name in names]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained", facecolor=COLOR_SURFACE)
        fig.suptitle("Сравнение моделей на тесте (июль-ноябрь 2026)", x=0.01, ha="left",
                     fontsize=14, fontweight="bold", color=COLOR_INK)

        for ax, metric, title in ((axes[0], "MAE", "Средняя ошибка (MAE), ниже лучше"),
                                  (axes[1], "Spearman", "Ранговая корреляция, выше лучше")):
            values = [self.metrics[name]["test"][metric] for name in names]
            positions = np.arange(len(names))[::-1]
            ax.barh(positions, [value or 0 for value in values], color=colors, height=0.6)
            ax.set_yticks(positions, names)
            for position, value in zip(positions, values):
                label = "не определена" if value is None else f"{value:.3f}"
                ax.annotate(label, (value or 0, position), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=9, color=COLOR_SECONDARY)
            self.style_axes(ax, title)

        fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=200, facecolor=COLOR_SURFACE)
        plt.close(fig)


    def plot_deciles(self):
        """Реальный отклик постов, разбитых по прогнозу CatBoost на 10 групп"""

        data = self.test[[TARGET]].copy()
        data["predicted"] = self.predictions["CatBoost"]
        data["decile"] = pd.qcut(data["predicted"], DECILES, labels=False) + 1

        stats = data.groupby("decile").agg(actual=(TARGET, "median"), predicted=("predicted", "median"))
        actual_percent = (np.exp(stats["actual"]) - 1) * 100
        predicted_percent = (np.exp(stats["predicted"]) - 1) * 100

        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained", facecolor=COLOR_SURFACE)
        ax.bar(stats.index, actual_percent, color=COLOR_MAIN, width=0.7, label="Реальный отклик")
        ax.plot(stats.index, predicted_percent, "o", color=COLOR_INK, markersize=7, label="Прогноз")
        ax.axhline(0, color=COLOR_AXIS, linewidth=1)
        ax.set_xticks(stats.index)
        ax.legend(frameon=False, labelcolor=COLOR_SECONDARY)
        self.style_axes(ax, "Посты теста по прогнозу CatBoost: от худших к лучшим",
                        xlabel="Группа по прогнозу (1 - самый низкий прогноз)",
                        ylabel="Медианное отклонение от типичного поста, %", grid_axis="y")

        fig.savefig(RESULTS_DIR / "deciles.png", dpi=200, facecolor=COLOR_SURFACE)
        plt.close(fig)

        self.metrics["CatBoost"]["test_deciles_actual_percent"] = actual_percent.round(1).to_dict()


    def run(self):
        """Обучает все модели, сравнивает их и сохраняет результаты"""

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        self.fit_baselines()
        self.fit_ridge()
        self.fit_catboost()

        with plt.rc_context({"font.family" : ["Segoe UI", "DejaVu Sans"]}):
            self.plot_comparison()
            self.plot_deciles()

        with open(RESULTS_DIR / "metrics.json", mode="w", encoding="utf-8") as file:
            json.dump(self.metrics, file, ensure_ascii=False, indent=4)

        print(f"Метрики успешно сохранены в файл: {RESULTS_DIR / 'metrics.json'}")

        return self.metrics


if __name__ == "__main__":

    EngagementModelTrainer().run()
