from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import pymorphy2
from os.path import join
from pathlib import Path


SAVE_PATH_DIR = Path(join("lab2", "results"))

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
    

def analyze_word_frequrncy(cleaned_texts, output="word_frequency", format=".csv"):
    """Выполняет лемаатизациб, удаляет стоп-слова и считает частоту слова"""

    # инициализация морфологического анализитора и стоп-слова
    morph = pymorphy2.MorphAnalyzer()
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

    lemmatized_words = []

    print("Начало обработки и лемматизации текста...")

    for text in cleaned_texts:
        # приведение к нижнему 
        words = re.findall(r"[а-яёa-z]+", text.lower())

        for word in words:
            # короткие слова пропускаем
            if len(word) < 2: continue

            # лемматизация (получение нач.формы)
            parsed_word = morph.parse(word)[0]
            normal_form = parsed_word.normal_form

            # проверяем часть речи (оставляем сущ, прилаг, глагол и тд)
            pos = parsed_word.tag.POS

            if normal_form in stop_words: continue

            lemmatized_words.append(normal_form)

        # подсчёт частоты вхождений
        word_counts = Counter(lemmatized_words)

        df_freq = pd.DataFrame(word_counts.most_common(),
                               columns=['word', 'frequency'])

        file_name = output + format

        if format.__contains__(".csv"):
            df_freq.to_csv(SAVE_PATH_DIR / file_name, index=False, encoding="utf-8-sig")
            print(f"Результат сохранен в csv")
        elif format.__contains__(".json"):
            df_freq.to_json(SAVE_PATH_DIR / file_name, indent=4)
            print(f"Результат сохранен в json")

        return df_freq