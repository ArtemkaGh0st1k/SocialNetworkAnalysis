import pandas as pd
import re

PATH_JSON = "lab1" / "results" / "samara_university_august_2026.json"
PATH_CSV = "lab1" / "results" / "samara_university_august_2026.csv"

def load_and_clean_posts(file_path : str = PATH_CSV):
    """ Загружает данные постов, извлекает текст и очищает его от мусора,
     ссылок и дубликатов"""

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")

    print(f"Исходное кол-во записей: {len(df)}")

    # проверка наличии колонки 'text' 
    if "text" not in df.columns:
        raise KeyError("В файле отсутствует колонка 'text'.")

    texts = df["text"].dropna().astype(str).tolist()

    cleaned_texts = []
    for text in texts:
        if not text.strip():
            continue  # пропускаем пустые строки

        # очистка url и ссылок
        text_cleaned = re.sub(
            r"https?://\S+|www\.\S+", "", text, flags=re.MULTILINE
        )

        # замена переносов строк и множественных пробелов на один пробел
        text_cleaned = re.sub(r"\s+", " ", text_cleaned).strip()  

        if text_cleaned:
            cleaned_texts.append(text_cleaned)

    original_count = len(cleaned_texts)
    cleaned_texts = list(dict.fromkeys(cleaned_texts))  # удаление дубликатов
    duplicate_removed = original_count - len(cleaned_texts)

    print(f"Удалено дубликатов: {duplicate_removed}")
    print(f"Итоговое кол-во чистых текстов: {len(cleaned_texts)}")

    return cleaned_texts