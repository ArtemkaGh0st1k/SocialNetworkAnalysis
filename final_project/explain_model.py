import contextlib
import io
import json
from os.path import join
from pathlib import Path
from catboost import CatBoostRegressor, Pool
import joblib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_project.train_model import (CATEGORICAL_FEATURES, MODELS_DIR, TARGET,
                                       EngagementModelTrainer)


RESULTS_DIR = Path(join("final_project", "results", "explain"))

STATE = 42
TOP_FEATURES_COUNT = 10
TOP_WORDS_COUNT = 15
MIN_WORD_POSTS = 30             # слово должно встречаться хотя бы в стольких постах теста
WATERFALL_ITEMS = 7

TEXT_FEATURE = "Слова текста"
FEATURE_NAMES = \
{
    "n_photos" : "Число фото",
    "text_length" : "Длина текста",
    "n_lines" : "Число строк",
    "n_hashtags" : "Хэштеги",
    "n_links" : "Ссылки в тексте",
    "n_emoji" : "Эмодзи",
    "n_exclamations" : "Восклицательные знаки",
    "hour" : "Час публикации",
    "weekday" : "День недели",
    "month_of_year" : "Месяц",
    "has_video" : "Видео",
    "has_link" : "Вложение-ссылка",
    "has_poll" : "Опрос",
    "has_doc" : "Документ",
    "has_question" : "Вопрос в тексте",
    "group" : "Сообщество",
}
DEPENDENCE_FEATURES = ["n_photos", "text_length", "n_emoji", "hour", "n_links", "n_lines"]

COLOR_POSITIVE = "#2a78d6"
COLOR_NEGATIVE = "#e34948"
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#cde2fb", "#2a78d6", "#0d366b"])


def to_percent(log_value):
    """Переводит вклад в логарифм отклика в изменение отклика, %"""

    return (np.exp(log_value) - 1) * 100


class ModelExplainer():

    def __init__(self):

        # разбиение и признаки строятся так же, как при обучении
        with contextlib.redirect_stdout(io.StringIO()):
            self.trainer = EngagementModelTrainer()

        self.test = self.trainer.test
        self.catboost = CatBoostRegressor()
        self.catboost.load_model(MODELS_DIR / "catboost.cbm")
        self.text_parts = joblib.load(MODELS_DIR / "text_features.joblib")
        self.ridge = joblib.load(MODELS_DIR / "ridge.joblib")

        self.features = self.trainer.catboost_matrix(
            self.test, self.text_parts["svd"], self.text_parts["vectorizer"]
        )
        self.summary = {"test_posts" : len(self.test)}

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    def compute_catboost_shap(self):
        """Считает SHAP-значения CatBoost. Компоненты текста по отдельности
        не имеют смысла, поэтому их вклады складываются в один признак"""

        pool = Pool(self.features, cat_features=CATEGORICAL_FEATURES)
        values = self.catboost.get_feature_importance(pool, type="ShapValues")

        shap = pd.DataFrame(values[:, :-1], columns=self.features.columns, index=self.test.index)
        self.base_value = values[0, -1]

        # сумма вкладов и базового значения должна совпасть с прогнозом
        prediction = self.catboost.predict(self.features)
        assert np.allclose(shap.sum(axis=1) + self.base_value, prediction, atol=1e-4)

        text_columns = [column for column in shap.columns if column.startswith("svd_")]
        shap[TEXT_FEATURE] = shap[text_columns].sum(axis=1)
        self.shap = shap.drop(columns=text_columns).rename(columns=FEATURE_NAMES)
        self.prediction = pd.Series(prediction, index=self.test.index)


    def style_axes(self, ax, title=None, xlabel=None, ylabel=None, grid_axis="x"):
        """Оформляет оси в едином стиле"""

        if title:
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=COLOR_INK)
        if xlabel:
            ax.set_xlabel(xlabel, color=COLOR_SECONDARY)
        if ylabel:
            ax.set_ylabel(ylabel, color=COLOR_SECONDARY)

        ax.set_facecolor(COLOR_SURFACE)
        if grid_axis:
            ax.grid(axis=grid_axis, color=COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=COLOR_SECONDARY, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(COLOR_AXIS)


    def new_figure(self, title, subtitle, **kwargs):
        """Создаёт рисунок с заголовком и пояснением"""

        fig = plt.figure(facecolor=COLOR_SURFACE, layout="constrained", **kwargs)
        fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=COLOR_INK)
        fig.text(0.01, 0.945, subtitle, fontsize=10, color=COLOR_SECONDARY, va="top")
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))

        return fig


    def save(self, fig, filename):
        """Сохраняет рисунок в файл"""

        fig.savefig(RESULTS_DIR / filename, dpi=200, facecolor=COLOR_SURFACE)
        plt.close(fig)
        print(f"График успешно сохранен в файл: {filename}")


    def plot_importance(self):
        """Средний вклад каждого признака в прогноз"""

        importance = to_percent(self.shap.abs().mean()).sort_values()

        fig = self.new_figure("Что влияет на прогноз",
                              "Средний по модулю вклад признака в прогноз CatBoost (SHAP), "
                              "переведённый в изменение отклика, %. Тестовая выборка.",
                              figsize=(10, 6.5))
        ax = fig.subplots()

        colors = [COLOR_POSITIVE if name == TEXT_FEATURE else COLOR_MUTED for name in importance.index]
        positions = np.arange(len(importance))
        ax.barh(positions, importance.values, color=colors, height=0.65)
        ax.set_yticks(positions, importance.index)
        for position, value in zip(positions, importance.values):
            ax.annotate(f"{value:.1f}%", (value, position), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=COLOR_SECONDARY)
        self.style_axes(ax, xlabel="Типичное изменение отклика из-за признака, %")

        self.summary["importance_percent"] = importance[::-1].round(2).to_dict()
        self.save(fig, "shap_importance.png")


    def plot_beeswarm(self):
        """Каждая точка - пост: насколько признак сдвинул его прогноз и какое у признака значение"""

        features = (self.shap.drop(columns=[TEXT_FEATURE, "Сообщество"]).abs().mean()
                    .sort_values(ascending=False).head(TOP_FEATURES_COUNT).index)
        source = {FEATURE_NAMES[column] : column for column in FEATURE_NAMES}
        rng = np.random.default_rng(STATE)

        fig = self.new_figure("Как значения признаков сдвигают прогноз",
                              "Каждая точка - пост теста. Положение - вклад признака (SHAP) в отклик, %. "
                              "Цвет - значение признака: светлее - меньше, темнее - больше.",
                              figsize=(11, 7))
        ax = fig.subplots()

        for position, feature in enumerate(features[::-1]):
            values = to_percent(self.shap[feature].values)
            raw = self.test[source[feature]].astype(float)
            color = raw.rank(pct=True).values

            # разброс по вертикали только для читаемости: точки с одинаковым вкладом не сливаются
            jitter = rng.uniform(-0.3, 0.3, len(values))
            ax.scatter(values, position + jitter, c=color, cmap=SEQUENTIAL, s=6, alpha=0.7,
                       linewidths=0, rasterized=True)

        ax.axvline(0, color=COLOR_AXIS, linewidth=1)
        ax.set_yticks(range(len(features)), features[::-1])
        self.style_axes(ax, xlabel="Вклад признака в отклик, %")

        colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQUENTIAL), ax=ax, shrink=0.6, pad=0.01,
                                ticks=[0, 1])
        colorbar.ax.set_yticklabels(["меньше", "больше"])
        colorbar.set_label("Значение признака", color=COLOR_SECONDARY)
        colorbar.outline.set_visible(False)

        self.save(fig, "shap_beeswarm.png")


    def plot_dependence(self):
        """Зависимость вклада признака от его значения"""

        fig = self.new_figure("Вклад отдельных признаков",
                              "Точки - посты теста, линия - медианный вклад при данном значении признака. "
                              "Учтены и другие признаки поста.",
                              figsize=(13, 7.5))
        axes = fig.subplots(2, 3)

        effects = {}
        for ax, column in zip(axes.flat, DEPENDENCE_FEATURES):
            name = FEATURE_NAMES[column]
            values = self.test[column].astype(float)
            contribution = pd.Series(to_percent(self.shap[name].values), index=self.test.index)

            ax.scatter(values, contribution, s=5, alpha=0.25, color=COLOR_POSITIVE,
                       linewidths=0, rasterized=True)

            # для непрерывных признаков медиана считается по группам равного размера
            if values.nunique() > 25:
                bins = pd.qcut(values, 12, duplicates="drop")
                line = contribution.groupby(bins, observed=True).median()
                centers = values.groupby(bins, observed=True).median()
            else:
                counts = values.value_counts()
                kept = counts[counts >= 30].index
                line = contribution[values.isin(kept)].groupby(values).median()
                centers = line.index.values

            ax.plot(centers, line.values, color=COLOR_INK, linewidth=2)
            ax.axhline(0, color=COLOR_AXIS, linewidth=1)
            if column == "text_length":
                ax.set_xscale("symlog", linthresh=100)
                ax.set_xlim(left=0)

            self.style_axes(ax, title=name, ylabel="Вклад, %" if ax in axes[:, 0] else None, grid_axis="y")
            effects[name] = {str(round(float(center), 1)) : round(float(value), 1)
                             for center, value in zip(centers, line.values)}

        self.summary["dependence_percent"] = effects
        self.save(fig, "shap_dependence.png")


    def compute_word_effects(self):
        """Вклад слов по линейной модели. Для линейной модели SHAP-значение признака
        равно коэффициенту, умноженному на отклонение признака от среднего"""

        vectorizer = self.ridge["vectorizer"]
        words = vectorizer.get_feature_names_out()
        coefficients = self.ridge["model"].coef_[:len(words)]

        matrix = vectorizer.transform(self.test["lemmas"]).tocsc()
        means = np.asarray(matrix.mean(axis=0)).ravel()

        rows = []
        for index, word in enumerate(words):
            column = matrix[:, index]
            if column.nnz < MIN_WORD_POSTS:
                continue

            # средний вклад слова в постах, где оно есть
            contribution = coefficients[index] * (column.data - means[index])
            rows.append({"word" : word, "posts" : int(column.nnz),
                         "effect" : float(to_percent(contribution.mean()))})

        effects = pd.DataFrame(rows).sort_values("effect")
        effects.to_csv(RESULTS_DIR / "word_effects.csv", index=False, encoding="utf-8-sig")

        return effects


    def plot_words(self):
        """Слова, которые сильнее всего повышают и понижают прогноз"""

        effects = self.compute_word_effects()
        selected = pd.concat([effects.head(TOP_WORDS_COUNT), effects.tail(TOP_WORDS_COUNT)])

        fig = self.new_figure("Слова, которые меняют прогноз",
                              f"Средний вклад слова (SHAP линейной модели) в постах теста, где оно есть. "
                              f"Слова и пары слов из {MIN_WORD_POSTS}+ постов. В скобках - число постов.",
                              figsize=(10, 10))
        ax = fig.subplots()

        positions = np.arange(len(selected))
        colors = [COLOR_POSITIVE if value >= 0 else COLOR_NEGATIVE for value in selected["effect"]]
        ax.barh(positions, selected["effect"], color=colors, height=0.65)
        ax.axvline(0, color=COLOR_AXIS, linewidth=1)
        ax.set_yticks(positions, selected["word"])
        for position, value, count in zip(positions, selected["effect"], selected["posts"]):
            ax.annotate(f"{value:+.0f}%  ({count})", (value, position),
                        xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                        ha="left" if value >= 0 else "right", va="center", fontsize=9, color=COLOR_SECONDARY)

        left, right = ax.get_xlim()
        margin = (right - left) * 0.2
        ax.set_xlim(left - margin, right + margin)
        self.style_axes(ax, xlabel="Вклад слова в отклик, %")

        self.summary["top_words"] = effects.tail(TOP_WORDS_COUNT)[::-1]["word"].tolist()
        self.summary["bottom_words"] = effects.head(TOP_WORDS_COUNT)["word"].tolist()
        self.save(fig, "ridge_words.png")


    def draw_waterfall(self, ax, index, title):
        """Рисует разложение прогноза одного поста на вклады признаков"""

        contributions = self.shap.loc[index]
        order = contributions.abs().sort_values(ascending=False).index
        top = contributions[order[:WATERFALL_ITEMS]]
        rest = contributions[order[WATERFALL_ITEMS:]].sum()
        items = list(top.items()) + [("Остальные признаки", rest)]

        start = self.base_value
        for position, (name, value) in enumerate(items):
            color = COLOR_POSITIVE if value >= 0 else COLOR_NEGATIVE
            ax.barh(position, value, left=start, color=color, height=0.6)
            ax.annotate(f"{to_percent(value):+.0f}%", (start + max(value, 0), position),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=9,
                        color=COLOR_SECONDARY)
            start += value

        ax.axvline(self.base_value, color=COLOR_AXIS, linewidth=1)
        ax.axvline(start, color=COLOR_INK, linewidth=1.2)
        ax.set_yticks(range(len(items)), [name for name, _ in items])
        ax.invert_yaxis()

        # ось в логарифме отклика подписывается в процентах
        ticks = ax.get_xticks()
        ax.set_xticks(ticks, [f"{to_percent(tick):+.0f}%" for tick in ticks])

        actual = to_percent(self.test.loc[index, TARGET])
        self.style_axes(ax, title=f"{title}: прогноз {to_percent(start):+.0f}%, факт {actual:+.0f}%",
                        xlabel="Отклик относительно типичного поста")


    def plot_examples(self):
        """Объяснение прогноза для постов с самым высоким и самым низким прогнозом"""

        high = self.prediction.idxmax()
        low = self.prediction.idxmin()

        fig = self.new_figure("Почему модель так решила",
                              "Разложение прогноза на вклады признаков (SHAP). Серая линия - типичный пост, "
                              "чёрная - итоговый прогноз.",
                              figsize=(13, 5.5))
        axes = fig.subplots(1, 2)
        self.draw_waterfall(axes[0], high, "Самый высокий прогноз")
        self.draw_waterfall(axes[1], low, "Самый низкий прогноз")
        self.save(fig, "shap_examples.png")

        self.summary["examples"] = {
            name : {
                "group" : self.test.loc[index, "group"],
                "text" : self.test.loc[index, "text"][:200],
                "predicted_percent" : round(float(to_percent(self.prediction[index])), 1),
                "actual_percent" : round(float(to_percent(self.test.loc[index, TARGET])), 1),
            }
            for name, index in (("high", high), ("low", low))
        }


    def run(self):
        """Считает SHAP-значения, строит графики и сохраняет сводку"""

        print("Начало объяснения моделей ...")

        self.compute_catboost_shap()

        with plt.rc_context({"font.family" : ["Segoe UI", "DejaVu Sans"]}):
            self.plot_importance()
            self.plot_beeswarm()
            self.plot_dependence()
            self.plot_words()
            self.plot_examples()

        with open(RESULTS_DIR / "summary.json", mode="w", encoding="utf-8") as file:
            json.dump(self.summary, file, ensure_ascii=False, indent=4)

        print(f"Сводка успешно сохранена в файл: {RESULTS_DIR / 'summary.json'}")

        return self.summary


if __name__ == "__main__":

    ModelExplainer().run()
