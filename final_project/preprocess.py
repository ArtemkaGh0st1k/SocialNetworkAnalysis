from functools import lru_cache
from os.path import join
from pathlib import Path
import re
import emoji
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pymorphy3


DATASET_PATH = Path(join("final_project", "data", "posts.csv"))
CLEAN_DATASET_PATH = Path(join("final_project", "data", "posts_clean.csv"))

FRESH_POST_DAYS = 7             # у более свежих постов счётчики ещё растут
MAX_ZERO_LIKES_SHARE = 0.5      # если без лайков больше половины постов месяца, лайки были отключены
DEFAULT_UTC_OFFSET = 3          # московское время
UTC_OFFSETS = \
{
    "samara_university" : 4,
    "nsu24" : 7,
}

CUSTOM_STOP_WORDS = \
{
    "год", "это", "который", "свой", "весь", "всё", "ещё", "еще", "уже", "также",
    "наш", "ваш", "очень", "именно", "кроме", "мочь", "стать", "каждый", "самый",
}

LINK_PATTERN = re.compile(r"https?://\S+|www\.\S+|vk\.cc/\S+")
WIKI_LINK_PATTERN = re.compile(r"\[[^\[\]|]*\|([^\[\]]*)\]")    # [club1|Текст] -> Текст
HASHTAG_PATTERN = re.compile(r"#[\w@]+")
WORD_PATTERN = re.compile(r"[а-яё]+")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

MORPH = pymorphy3.MorphAnalyzer()
STOP_WORDS = set(stopwords.words("russian")) | CUSTOM_STOP_WORDS


@lru_cache(maxsize=None)
def lemmatize_word(word):
    """Возвращает начальную форму слова. Кеш нужен, потому что слова часто повторяются"""

    return MORPH.parse(word)[0].normal_form


class PostsPreprocessor():

    def __init__(self, dataset_path=DATASET_PATH):

        self.posts = pd.read_csv(dataset_path, encoding="utf-8-sig")
        self.posts["text"] = self.posts["text"].fillna("")
        self.posts["attachments"] = self.posts["attachments"].fillna("")


    def filter_posts(self):
        """Оставляет посты, для которых отклик можно честно предсказать"""

        posts = self.posts
        before = len(posts)

        is_fresh = posts["collected_at"] - posts["timestamp"] < FRESH_POST_DAYS * 86400

        # в отдельные месяцы сообщество отключало лайки: у большинства постов их 0
        month = pd.to_datetime(posts["timestamp"], unit="s").dt.to_period("M")
        zero_likes_share = (posts["likes"] == 0).groupby([posts["group"], month]).transform("mean")
        likes_disabled = zero_likes_share > MAX_ZERO_LIKES_SHARE

        # у репоста свой текст пуст, а отклик определяется исходной записью
        keep = ~posts["is_repost"] & ~posts["is_ad"] & ~is_fresh & ~likes_disabled & (posts["views"] > 0)
        self.posts = posts[keep].copy()

        print(f"Удалено постов: репостов {posts['is_repost'].sum()}, свежих {is_fresh.sum()}, "
              f"в месяцы без лайков {likes_disabled.sum()}, "
              f"всего {before - len(self.posts)}; осталось {len(self.posts)}")


    def add_target(self):
        """Считает долю реакции и её значение относительно сообщества и месяца.
        Отклик различается между сообществами и меняется со временем, поэтому
        доля реакции делится на медиану того же сообщества за тот же месяц"""

        posts = self.posts

        posts["engagement_rate"] = posts["likes"] / posts["views"]
        posts["month"] = pd.to_datetime(posts["timestamp"], unit="s").dt.to_period("M").astype(str)

        group_month_median = posts.groupby(["group", "month"])["engagement_rate"].transform("median")
        posts["relative_engagement"] = posts["engagement_rate"] / group_month_median
        posts["log_relative_engagement"] = np.log(posts["relative_engagement"].clip(lower=1e-3))


    def add_time_features(self):
        """Добавляет время публикации по местному времени сообщества"""

        posts = self.posts
        offsets = posts["group"].map(UTC_OFFSETS).fillna(DEFAULT_UTC_OFFSET)

        local_time = pd.to_datetime(posts["timestamp"] + offsets * 3600, unit="s")
        posts["hour"] = local_time.dt.hour
        posts["weekday"] = local_time.dt.weekday


    def add_format_features(self):
        """Добавляет признаки оформления поста и его вложений"""

        posts = self.posts
        text = posts["text"]
        attachments = posts["attachments"].str.split(",")

        posts["n_photos"] = attachments.apply(lambda items : items.count("photo"))
        for kind in ("video", "link", "poll", "doc"):
            posts[f"has_{kind}"] = attachments.apply(lambda items : kind in items)

        posts["text_length"] = text.str.len()
        posts["n_lines"] = text.str.count("\n") + (posts["text_length"] > 0)
        posts["n_hashtags"] = text.str.count(HASHTAG_PATTERN)
        posts["n_links"] = text.str.count(LINK_PATTERN) + text.str.count(WIKI_LINK_PATTERN)
        posts["n_emoji"] = text.apply(emoji.emoji_count)
        posts["n_exclamations"] = text.str.count("!")
        posts["has_question"] = text.str.contains(r"\?", regex=True)


    def lemmatize(self, text):
        """Возвращает строку лемм без ссылок, эмодзи и стоп-слов"""

        text = WIKI_LINK_PATTERN.sub(r"\1", text)
        text = LINK_PATTERN.sub(" ", text)
        text = HASHTAG_PATTERN.sub(" ", text)

        lemmas = []
        for word in WORD_PATTERN.findall(text.lower()):
            if len(word) < 3:
                continue

            lemma = lemmatize_word(word)
            if lemma not in STOP_WORDS:
                lemmas.append(lemma)

        return " ".join(lemmas)


    def run(self):
        """Выполняет все шаги обработки и сохраняет таблицу"""

        print("Начало обработки постов ...")

        self.filter_posts()
        self.add_target()
        self.add_time_features()
        self.add_format_features()

        print("Лемматизация текстов ...")
        self.posts["lemmas"] = self.posts["text"].apply(self.lemmatize)

        self.posts.to_csv(CLEAN_DATASET_PATH, index=False, encoding="utf-8-sig")
        print(f"Таблица успешно сохранена в файл: {CLEAN_DATASET_PATH}")

        return self.posts


if __name__ == "__main__":

    PostsPreprocessor().run()
