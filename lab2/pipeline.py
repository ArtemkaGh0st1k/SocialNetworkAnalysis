from lab2.clean_text import load_and_clean_posts
from lab2.word_frequency import analyze_word_frequrncy, lemmatize_texts
from lab2.text_clustering import perform_clustering_and_visualization


def run_lab2_pipeline():
    cleaned_texts = load_and_clean_posts()
    lemmatized_texts = lemmatize_texts(cleaned_texts)
    df_freq = analyze_word_frequrncy(lemmatized_texts)
    df_res = perform_clustering_and_visualization(cleaned_texts, lemmatized_texts)
