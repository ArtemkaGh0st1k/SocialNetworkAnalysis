import json
import os
from pathlib import Path
import sys

# Streamlit запускает этот файл как отдельный скрипт, поэтому корень проекта
# добавляется в пути поиска модулей и делается рабочей папкой
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from datetime import date
from os.path import join
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit import runtime
from streamlit.web import cli


CLEAN_DATASET_PATH = Path(join("final_project", "data", "posts_clean.csv"))
MODELS_DIR = Path(join("final_project", "models"))
RESULTS_DIR = Path(join("final_project", "results"))

POST_COLUMNS = \
[
    "group", "timestamp", "month", "likes", "views", "engagement_rate", "relative_engagement",
    "hour", "weekday", "n_photos", "has_video", "has_link", "has_poll", "text_length", "lemmas",
]
MIN_CELL_POSTS = 30             # меньше постов в ячейке тепловой карты - медиана ненадёжна
MIN_WORD_POSTS = 20             # меньше постов со словом - вывод ненадёжен
LENGTH_BINS = 10
TOP_CONTRIBUTIONS = 8
TOP_WORDS_COUNT = 5

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
FORMAT_ORDER = ["Видео", "5 и больше фото", "2-4 фото", "1 фото", "Ссылка без фото", "Только текст", "Опрос"]
PAGES = ["Обзор", "Сообщества", "Когда публиковать", "Формат и текст", "Модель", "Проверить пост"]

COLOR_MAIN = "#2a78d6"
COLOR_POSITIVE = "#2a78d6"
COLOR_NEGATIVE = "#e34948"
COLOR_NEUTRAL = "#f0efec"
COLOR_MUTED = "#898781"
DIVERGING_SCALE = [[0, COLOR_NEGATIVE], [0.5, COLOR_NEUTRAL], [1, COLOR_POSITIVE]]


def to_percent(log_value):
    """Переводит логарифм относительного отклика в изменение отклика, %"""

    return (np.exp(log_value) - 1) * 100


@st.cache_data(show_spinner="Загрузка постов ...")
def load_posts():
    """Загружает обработанные посты один раз за запуск"""

    posts = pd.read_csv(CLEAN_DATASET_PATH, encoding="utf-8-sig", usecols=POST_COLUMNS)
    posts["lemmas"] = posts["lemmas"].fillna("")
    posts["date"] = pd.to_datetime(posts["timestamp"], unit="s")
    posts["format"] = np.select(
        [posts["has_poll"], posts["has_video"], posts["n_photos"] >= 5, posts["n_photos"] >= 2,
         posts["n_photos"] == 1, posts["has_link"]],
        FORMAT_ORDER[-1:] + FORMAT_ORDER[:4] + FORMAT_ORDER[4:5], default="Только текст"
    )

    return posts


@st.cache_data
def load_json(path):
    """Загружает сводку результатов"""

    with open(path, mode="r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(path):
    """Загружает таблицу результатов"""

    return pd.read_csv(path, encoding="utf-8-sig")


class PostPredictor():
    """Прогноз отклика для нового поста и его объяснение"""

    def __init__(self):

        # тяжёлые модули загружаются только для страницы прогноза
        from catboost import CatBoostRegressor, Pool
        import joblib
        from final_project.explain_model import FEATURE_NAMES, TEXT_FEATURE
        from final_project.preprocess import PostsPreprocessor
        from final_project.train_model import CATEGORICAL_FEATURES, EngagementModelTrainer

        self.pool_class = Pool
        self.categorical_features = CATEGORICAL_FEATURES
        self.feature_names = FEATURE_NAMES
        self.text_feature = TEXT_FEATURE

        self.catboost = CatBoostRegressor()
        self.catboost.load_model(MODELS_DIR / "catboost.cbm")
        self.text_parts = joblib.load(MODELS_DIR / "text_features.joblib")
        self.ridge = joblib.load(MODELS_DIR / "ridge.joblib")

        # нужны только методы построения признаков, без загрузки всего набора данных
        self.preprocessor = object.__new__(PostsPreprocessor)
        self.trainer = object.__new__(EngagementModelTrainer)


    def build_post(self, group, text, n_photos, has_video, has_link, has_poll, has_doc, hour, post_date):
        """Строит строку признаков нового поста так же, как при обучении"""

        attachments = ["photo"] * n_photos + ["video"] * has_video + ["link"] * has_link \
                      + ["poll"] * has_poll + ["doc"] * has_doc

        self.preprocessor.posts = pd.DataFrame([{
            "group" : group, "text" : text, "attachments" : ",".join(attachments),
        }])
        self.preprocessor.add_format_features()

        post = self.preprocessor.posts
        post["hour"] = hour
        post["weekday"] = post_date.weekday()
        post["month_of_year"] = post_date.month
        post["lemmas"] = self.preprocessor.lemmatize(text)

        return post


    def predict(self, **post_params):
        """Возвращает прогноз, вклады признаков и вклады слов"""

        post = self.build_post(**post_params)
        features = self.trainer.catboost_matrix(post, self.text_parts["svd"], self.text_parts["vectorizer"])

        pool = self.pool_class(features, cat_features=self.categorical_features)
        values = self.catboost.get_feature_importance(pool, type="ShapValues")[0]
        contributions = pd.Series(values[:-1], index=features.columns)

        text_columns = [column for column in contributions.index if column.startswith("svd_")]
        contributions[self.text_feature] = contributions[text_columns].sum()
        contributions = contributions.drop(text_columns).rename(self.feature_names)

        # вклад слова в линейной модели: коэффициент, умноженный на вес слова в тексте
        vectorizer = self.ridge["vectorizer"]
        row = vectorizer.transform(post["lemmas"])
        coefficients = self.ridge["model"].coef_[:row.shape[1]]
        words = pd.Series(coefficients[row.indices] * row.data,
                          index=vectorizer.get_feature_names_out()[row.indices])

        return values.sum(), contributions, words


@st.cache_resource(show_spinner="Загрузка модели ...")
def load_predictor():
    """Загружает модели один раз за запуск"""

    return PostPredictor()


class Dashboard():

    def __init__(self):

        self.posts = load_posts()
        self.eda = load_json(RESULTS_DIR / "eda" / "summary.json")
        self.metrics = load_json(RESULTS_DIR / "model" / "metrics.json")
        self.explain = load_json(RESULTS_DIR / "explain" / "summary.json")


    def style_figure(self, fig, height=420):
        """Оформляет график в едином стиле. Цвета текста и сетки не задаются:
        их подбирает Streamlit под светлую или тёмную тему"""

        fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                          font=dict(family="Segoe UI, sans-serif", size=13))

        return fig


    def diverging_bar(self, labels, values, counts=None, height=420):
        """Горизонтальные столбцы: вверх синим, вниз красным"""

        text = [f"{value:+.0f}%" + (f" ({count})" if counts is not None else "")
                for value, count in zip(values, counts if counts is not None else values)]

        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h", text=text, textposition="outside", cliponaxis=False,
            marker_color=[COLOR_POSITIVE if value >= 0 else COLOR_NEGATIVE for value in values],
            hovertemplate="%{y}: %{x:+.1f}%<extra></extra>",
        ))
        fig.update_xaxes(title="Отклонение от типичного поста, %", zeroline=True, zerolinecolor=COLOR_MUTED)

        return self.style_figure(fig, height)


    def filter_posts(self):
        """Фильтры по сообществам и периоду в боковой панели"""

        groups = sorted(self.posts["group"].unique())
        selected = st.sidebar.multiselect("Сообщества", groups, default=groups)

        months = sorted(self.posts["month"].unique())
        start, end = st.sidebar.select_slider("Период", options=months, value=(months[0], months[-1]))

        posts = self.posts[self.posts["group"].isin(selected)
                           & self.posts["month"].between(start, end)]

        st.sidebar.caption(f"Выбрано постов: {len(posts)}")

        return posts


    def page_overview(self, posts):
        """Главные числа и выводы"""

        st.header("Какой пост наберёт отклик?")
        st.write("Посты 9 сообществ российских университетов во «ВКонтакте» за 3 года. "
                 "Отклик - доля лайков от просмотров относительно типичного поста того же сообщества и месяца.")

        columns = st.columns(4)
        columns[0].metric("Постов", f"{len(posts):,}".replace(",", " "))
        columns[1].metric("Сообществ", posts["group"].nunique())
        columns[2].metric("Период", f"{posts['date'].min():%m.%y}-{posts['date'].max():%m.%y}")
        columns[3].metric("Медиана лайков на просмотры", f"{posts['engagement_rate'].median() * 100:.2f}%")

        test = self.metrics["CatBoost"]["test"]
        st.subheader("Главное")
        st.markdown(
            f"- **Формат.** Видео получает {self.eda['format_effect_percent']['Видео']:+.0f}%, "
            f"посты только с текстом {self.eda['format_effect_percent']['Только текст']:+.0f}% "
            "к типичному посту.\n"
            f"- **Содержание.** Больше реакций у постов со словами: {', '.join(self.explain['top_words'][:5])}. "
            f"Меньше - со словами: {', '.join(self.explain['bottom_words'][:5])}.\n"
            f"- **Модель.** CatBoost верно определяет, будет ли пост выше или ниже типичного, "
            f"в {test['sign_accuracy'] * 100:.0f}% случаев, ранговая корреляция на тесте {test['Spearman']:.2f}.\n"
            "- **Важно.** Это связи в данных, а не причины."
        )


    def page_communities(self, posts):
        """Сравнение сообществ"""

        st.header("Сообщества")

        stats = posts.groupby("group").agg(
            posts=("group", "size"), engagement=("engagement_rate", "median"),
            views=("views", "median"), likes=("likes", "median"),
        ).sort_values("engagement")

        fig = px.bar(stats, x=stats["engagement"] * 100, y=stats.index, orientation="h",
                     text=[f"{value * 100:.2f}%" for value in stats["engagement"]])
        fig.update_traces(marker_color=COLOR_MAIN, textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}: %{x:.2f}%<extra></extra>")
        fig.update_layout(xaxis_title="Медиана лайков на просмотры, %", yaxis_title=None)
        st.subheader("Доля реакции")
        st.plotly_chart(self.style_figure(fig), width="stretch")

        st.subheader("Динамика по месяцам")
        metric = st.radio("Показатель", ["Число постов", "Медиана лайков на просмотры, %"], horizontal=True)

        if metric == "Число постов":
            monthly = posts.groupby(["group", "month"]).size().rename("value").reset_index()
        else:
            monthly = (posts.groupby(["group", "month"])["engagement_rate"].median() * 100).rename("value").reset_index()

        fig = px.line(monthly, x="month", y="value", facet_col="group", facet_col_wrap=3,
                      facet_row_spacing=0.08, color_discrete_sequence=[COLOR_MAIN])
        fig.for_each_annotation(lambda annotation : annotation.update(text=annotation.text.split("=")[-1]))
        fig.update_xaxes(title=None, showticklabels=True, nticks=4)
        fig.update_yaxes(title=None)
        st.plotly_chart(self.style_figure(fig, height=650), width="stretch")

        table = stats.sort_values("engagement", ascending=False).rename(columns={
            "posts" : "Постов", "engagement" : "Лайки на просмотры", "views" : "Медиана просмотров",
            "likes" : "Медиана лайков"})
        table["Лайки на просмотры"] = (table["Лайки на просмотры"] * 100).round(2).astype(str) + "%"
        st.dataframe(table, width="stretch")


    def page_time(self, posts):
        """Когда публиковать"""

        st.header("Когда публиковать")
        st.write(f"Отклик относительно типичного поста, %. Местное время сообщества. "
                 f"Пустые ячейки: меньше {MIN_CELL_POSTS} постов.")

        grouped = posts.groupby(["weekday", "hour"])["relative_engagement"]
        medians = grouped.median().unstack().reindex(index=range(7), columns=range(24))
        counts = grouped.size().unstack().reindex(index=range(7), columns=range(24)).fillna(0)
        effect = ((medians - 1) * 100).where(counts >= MIN_CELL_POSTS)

        if effect.isna().all().all():
            st.info("Слишком мало постов для выбранных фильтров.")
            return

        limit = float(np.nanmax(np.abs(effect.values)))
        fig = px.imshow(effect.values, x=list(range(24)), y=WEEKDAYS, aspect="auto",
                        color_continuous_scale=DIVERGING_SCALE, zmin=-limit, zmax=limit,
                        labels=dict(x="Час публикации", y="", color="Отклик, %"))
        fig.update_traces(customdata=counts.values,
                          hovertemplate="%{y}, %{x}:00<br>Отклик: %{z:+.1f}%<br>Постов: %{customdata}<extra></extra>")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(self.style_figure(fig, height=380), width="stretch")

        by_hour = posts.groupby("hour")["relative_engagement"].agg(["median", "size"])
        by_hour = by_hour[by_hour["size"] >= MIN_CELL_POSTS]
        fig = px.bar(x=by_hour.index, y=(by_hour["median"] - 1) * 100,
                     labels=dict(x="Час публикации", y="Отклонение от типичного поста, %"))
        fig.update_traces(marker_color=[COLOR_POSITIVE if value >= 1 else COLOR_NEGATIVE
                                        for value in by_hour["median"]],
                          customdata=by_hour["size"],
                          hovertemplate="%{x}:00: %{y:+.1f}%<br>Постов: %{customdata}<extra></extra>")
        fig.update_xaxes(dtick=1)
        st.subheader("По часам")
        st.plotly_chart(self.style_figure(fig, height=330), width="stretch")
        st.caption("Вечером чаще публикуют поздравления, которые сами по себе собирают больше реакций. "
                   "С учётом содержания влияние времени слабее: см. страницу «Модель».")


    def page_content(self, posts):
        """Формат поста, длина текста и слова"""

        st.header("Формат и текст")

        stats = posts.groupby("format")["relative_engagement"].agg(["median", "size"])
        stats = stats.reindex([name for name in FORMAT_ORDER[::-1] if name in stats.index])
        st.subheader("Формат поста")
        st.plotly_chart(self.diverging_bar(stats.index, (stats["median"] - 1) * 100, stats["size"], height=360),
                        width="stretch")

        texts = posts[posts["text_length"] > 0]
        if len(texts) >= LENGTH_BINS * MIN_CELL_POSTS:
            bins = pd.qcut(texts["text_length"], LENGTH_BINS, duplicates="drop")
            lengths = texts.groupby(bins, observed=True).agg(
                length=("text_length", "median"), effect=("relative_engagement", "median"))
            fig = px.line(lengths, x="length", y=(lengths["effect"] - 1) * 100, markers=True, log_x=True,
                          labels=dict(length="Длина текста, символов (логарифмическая шкала)",
                                      y="Отклонение от типичного поста, %"))
            fig.update_traces(line_color=COLOR_MAIN)
            fig.add_hline(y=0, line_color=COLOR_MUTED)
            st.subheader("Длина текста")
            st.plotly_chart(self.style_figure(fig, height=330), width="stretch")

        st.subheader("Найти слово")
        from final_project.preprocess import lemmatize_word

        word = st.text_input("Слово в любой форме", value="поздравление").strip().lower()
        if word:
            lemma = lemmatize_word(word)
            mask = posts["lemmas"].str.split().apply(lambda lemmas : lemma in lemmas)

            if mask.sum() < MIN_WORD_POSTS:
                st.info(f"«{lemma}» встречается в {mask.sum()} постах: этого мало для вывода.")
            else:
                effect = (posts.loc[mask, "relative_engagement"].median() - 1) * 100
                columns = st.columns(3)
                columns[0].metric("Начальная форма", lemma)
                columns[1].metric("Постов со словом", int(mask.sum()))
                columns[2].metric("Отклик постов со словом", f"{effect:+.0f}%")

        words = load_csv(RESULTS_DIR / "eda" / "word_effects.csv")
        st.subheader("Слова-лидеры и слова-аутсайдеры по всем данным")
        selected = pd.concat([words.head(10), words.tail(10)])
        st.plotly_chart(self.diverging_bar(selected["word"], selected["effect"], selected["posts"], height=600),
                        width="stretch")


    def page_model(self, posts):
        """Качество модели и её объяснение"""

        st.header("Модель")
        st.write("Разбиение по времени: обучение до марта 2026, проверка март-июнь, тест июль-ноябрь 2026.")

        table = pd.DataFrame({name : values["test"] for name, values in self.metrics.items()}).T
        table = table.rename(columns={"sign_accuracy" : "Знак угадан", "Spearman" : "Ранговая корреляция"})
        table["Знак угадан"] = (table["Знак угадан"] * 100).round(0).astype(int).astype(str) + "%"
        st.subheader("Качество на тесте")
        st.dataframe(table, width="stretch")

        deciles = pd.Series(self.metrics["CatBoost"]["test_deciles_actual_percent"])
        fig = px.bar(x=deciles.index.astype(int), y=deciles.values,
                     labels=dict(x="Группа по прогнозу (1 - самый низкий прогноз)",
                                 y="Реальное отклонение от типичного поста, %"))
        fig.update_traces(marker_color=[COLOR_POSITIVE if value >= 0 else COLOR_NEGATIVE for value in deciles],
                          hovertemplate="Группа %{x}: %{y:+.0f}%<extra></extra>")
        st.subheader("Посты теста по прогнозу: от худших к лучшим")
        st.plotly_chart(self.style_figure(fig, height=360), width="stretch")

        importance = pd.Series(self.explain["importance_percent"]).sort_values()
        fig = px.bar(x=importance.values, y=importance.index, orientation="h",
                     labels=dict(x="Типичное изменение отклика из-за признака, %", y=""))
        fig.update_traces(marker_color=COLOR_MAIN, hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
        st.subheader("Что влияет на прогноз (SHAP)")
        st.plotly_chart(self.style_figure(fig, height=500), width="stretch")

        st.subheader("Как значения признаков сдвигают прогноз")
        st.image(str(RESULTS_DIR / "explain" / "shap_beeswarm.png"))


    def page_predict(self, posts):
        """Прогноз отклика для нового поста"""

        st.header("Проверить пост")
        st.write("Введите текст и параметры поста. Модель оценит отклик относительно типичного поста "
                 "выбранного сообщества и покажет, что повлияло на прогноз.")

        if not (MODELS_DIR / "catboost.cbm").exists():
            st.warning("Модель не найдена. Сначала запустите обучение: python -m final_project.train_model")
            return

        with st.form("post"):
            group = st.selectbox("Сообщество", sorted(self.posts["group"].unique()))
            text = st.text_area("Текст поста", height=180,
                                value="Дорогие первокурсники! Поздравляем вас с началом учебного года!")
            columns = st.columns(3)
            n_photos = columns[0].number_input("Число фото", min_value=0, max_value=10, value=1)
            post_date = columns[1].date_input("Дата публикации", value=date.today())
            hour = columns[2].slider("Час публикации", 0, 23, 18)
            columns = st.columns(4)
            has_video = columns[0].checkbox("Видео")
            has_link = columns[1].checkbox("Ссылка")
            has_poll = columns[2].checkbox("Опрос")
            has_doc = columns[3].checkbox("Документ")
            submitted = st.form_submit_button("Оценить пост")

        if not submitted:
            return

        prediction, contributions, words = load_predictor().predict(
            group=group, text=text, n_photos=int(n_photos), has_video=has_video, has_link=has_link,
            has_poll=has_poll, has_doc=has_doc, hour=hour, post_date=post_date,
        )

        effect = to_percent(prediction)
        st.metric("Прогноз отклика", f"{effect:+.0f}% к типичному посту")
        if effect >= 0:
            st.success(f"Пост должен собрать больше реакций, чем типичный пост сообщества {group}.")
        else:
            st.warning(f"Пост, вероятно, соберёт меньше реакций, чем типичный пост сообщества {group}.")

        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index[:TOP_CONTRIBUTIONS])
        st.subheader("Что повлияло на прогноз")
        st.plotly_chart(self.diverging_bar(top.index[::-1], to_percent(top.values[::-1]), height=380),
                        width="stretch")

        if len(words):
            columns = st.columns(2)
            columns[0].markdown("**Слова, которые повышают прогноз**")
            columns[0].write(", ".join(words[words > 0].sort_values(ascending=False).index[:TOP_WORDS_COUNT])
                             or "нет")
            columns[1].markdown("**Слова, которые понижают прогноз**")
            columns[1].write(", ".join(words[words < 0].sort_values().index[:TOP_WORDS_COUNT]) or "нет")

        st.caption("Прогноз хорошо сравнивает варианты поста между собой, но точное число реакций "
                   "предсказывает с большой ошибкой.")


    def run(self):
        """Строит выбранную страницу дашборда"""

        st.sidebar.title("Отклик постов VK")
        page = st.sidebar.radio("Раздел", PAGES)
        posts = self.filter_posts()

        if posts.empty:
            st.info("Нет постов для выбранных фильтров.")
            return

        pages = \
        {
            "Обзор" : self.page_overview,
            "Сообщества" : self.page_communities,
            "Когда публиковать" : self.page_time,
            "Формат и текст" : self.page_content,
            "Модель" : self.page_model,
            "Проверить пост" : self.page_predict,
        }
        pages[page](posts)


def run_dashboard():
    """Отрисовывает дашборд. Вызывается, когда файл выполняет Streamlit"""

    st.set_page_config(page_title="Отклик постов VK", page_icon="📊", layout="wide")

    if not CLEAN_DATASET_PATH.exists():
        st.error("Данные не найдены. Сначала запустите pipeline: python -m final_project.pipeline")
        return

    Dashboard().run()


if __name__ == "__main__":

    if runtime.exists():
        # файл выполняет Streamlit: рисуем страницу
        run_dashboard()
    else:
        # обычный запуск через python: просим Streamlit запустить этот же файл
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(cli.main())
