from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import pymorphy3
from os.path import join
from pathlib import Path


SAVE_PATH_DIR = Path(join("lab2", "results"))

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
    

def lemmatize_texts(cleaned_texts):
    """Выполняет лемматизацию и удаляет стоп-слова.
    Возвращает список строк из лемм (по одной строке на каждый пост)"""

    # инициализация морфологического анализитора и стоп-слова
    morph = pymorphy3.MorphAnalyzer()
    stop_words = set(stopwords.words("russian"))

    # Дополнительные специфичные стоп-слова (опционально)
    custom_stop_words = \
    {
        "год",
        "это",
        "который",
        "свой",
        "весь",
        "то",
        "так",
        "один",
        "еще",
        "уже",
    }
    stop_words.update(custom_stop_words)

    lemmatized_texts = []

    print("Начало обработки и лемматизации текста...")

    for text in cleaned_texts:
        # приведение к нижнему
        words = re.findall(r"[а-яёa-z]+", text.lower())

        lemmas = []
        for word in words:
            # короткие слова пропускаем
            if len(word) < 2: continue

            # лемматизация (получение нач.формы)
            normal_form = morph.parse(word)[0].normal_form

            if normal_form in stop_words: continue

            lemmas.append(normal_form)

        lemmatized_texts.append(" ".join(lemmas))

    return lemmatized_texts


def analyze_word_frequrncy(lemmatized_texts, output="word_frequency", format=".csv"):
    """Считает частоту слов по всем постам и сохраняет результат в файл"""

    # подсчёт частоты вхождений
    word_counts = Counter()
    for text in lemmatized_texts:
        word_counts.update(text.split())

    df_freq = pd.DataFrame(word_counts.most_common(),
                           columns=['word', 'frequency'])

    SAVE_PATH_DIR.mkdir(parents=True, exist_ok=True)
    file_name = output + format

    if format.__contains__(".csv"):
        df_freq.to_csv(SAVE_PATH_DIR / file_name, index=False, encoding="utf-8-sig")
        print(f"Результат сохранен в csv")
    elif format.__contains__(".json"):
        df_freq.to_json(SAVE_PATH_DIR / file_name, indent=4)
        print(f"Результат сохранен в json")

    return df_freq