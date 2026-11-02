import matplotlib.pyplot as plt
from os.path import join
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import (AgglomerativeClustering,
                             KMeans,
                             MiniBatchKMeans)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

STATE = 42
TOP_TERMS_COUNT = 10
SAVE_PATH_DIR = Path(join("lab2", "results"))


def perform_clustering_and_visualization(cleaned_texts, lemmatized_texts, n_cluster=4):
    """Выполняет TF-IDF векторизацию лемм, кластеризацию (3 алгоритма),
    сохраняет распределение документов по кластерам и 2D-визуалзиацию через PCA"""

    SAVE_PATH_DIR.mkdir(parents=True, exist_ok=True)

    if len(cleaned_texts) < n_cluster:
        print("Внимание: кол-во текстов меньше выбранного числа кластеров")
        n_cluster = max(2, len(cleaned_texts) // 2)

    print("Векторизация текстов с помощью TF_IDF ...")

    vectorizer = TfidfVectorizer(max_features=1000,
                                 min_df=2,
                                 max_df=0.85,
                                 ngram_range=(1, 2))

    X_sparse = vectorizer.fit_transform(lemmatized_texts)
    X_dense = X_sparse.toarray()    # для алгоритмов, требующих плотную матрицу

    print("Подбор числа кластеров по силуэтному коэффициенту (KMeans) ...")

    for k in range(2, min(9, len(cleaned_texts))):
        labels = KMeans(
            n_clusters=k, random_state=STATE, n_init=10
        ).fit_predict(X_sparse)
        score = silhouette_score(X_sparse, labels, metric="cosine")
        print(f"k = {k}: силуэт = {score:.3f}")

    print("Обучение алгоритмов кластеризации ...")

    models = \
    {
        "KMeans" : KMeans(
            n_clusters=n_cluster, random_state=STATE, n_init=10
        ).fit_predict(X_sparse),
        "MiniBatchKMeans" : MiniBatchKMeans(
            n_clusters=n_cluster, random_state=STATE, n_init=10,
        ).fit_predict(X_sparse),
        "Agglomerative" : AgglomerativeClustering(
            n_clusters=n_cluster
        ).fit_predict(X_dense)
    }

    # распределение документов по кластерам для каждого алгоритма
    df_clusters = pd.DataFrame({"text" : cleaned_texts, **models})
    df_clusters.to_csv(
        SAVE_PATH_DIR / "cluster_assignments.csv", index=False, encoding="utf-8-sig"
    )

    # количество документов в каждом кластере и сравнение алгоритмов
    sizes = []
    for name, labels in models.items():
        score = silhouette_score(X_sparse, labels, metric="cosine")
        print(f"{name}: силуэт = {score:.3f}")

        for cluster, count in pd.Series(labels).value_counts().sort_index().items():
            sizes.append({"algorithm" : name, "cluster" : cluster, "documents" : count})

    df_sizes = pd.DataFrame(sizes)
    df_sizes.to_csv(
        SAVE_PATH_DIR / "cluster_sizes.csv", index=False, encoding="utf-8-sig"
    )
    print(df_sizes.to_string(index=False))

    # выбираем основные метки для дальнейшей визуализации
    cluster_labels = models["KMeans"]

    # самые весомые слова каждого кластера для описания его темы
    terms = vectorizer.get_feature_names_out()
    top_terms = []
    for cluster in sorted(set(cluster_labels)):
        weights = X_dense[cluster_labels == cluster].mean(axis=0)
        words = terms[np.argsort(weights)[::-1][:TOP_TERMS_COUNT]]
        top_terms.append({"cluster" : cluster, "top_terms" : ", ".join(words)})
        print(f"Кластер {cluster}: {', '.join(words)}")

    pd.DataFrame(top_terms).to_csv(
        SAVE_PATH_DIR / "cluster_top_terms.csv", index=False, encoding="utf-8-sig"
    )

    print("Снижение размерности с помощью PCA до 2D ...")

    pca = PCA(n_components=2, random_state=STATE)
    coords = pca.fit_transform(X_dense)

    df_plot = pd.DataFrame(
        {
            "x" : coords[:, 0],
            "y" : coords[:, 1],
            "cluster" : cluster_labels,
            "text" : cleaned_texts
        }
    )

    plt.figure(figsize=(12 , 8))
    sns.set_theme(style='whitegrid')

    scatter = sns.scatterplot(
        x="x", 
        y = "y",
        hue = "cluster",
        palette="tab10",
        data=df_plot,
        legend="full",
        s=80,
        alpha=0.8
    )

    plt.title(
        "Кластеризация постов сообщества (TF-IDF + KMeans + PCA)",
        fontsize=14, fontweight="bold"
    )
    plt.xlabel("Главная компонента 1 (PCA 1)", fontsize=12)
    plt.ylabel("Главная компонента 2 (PCA 2)", fontsize=12)
    plt.legend(
        title="Кластеры",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=11,
    )
    plt.tight_layout()

    plot_filename = "clusters_visualization.png"
    plt.savefig(SAVE_PATH_DIR / plot_filename, dpi=300)
    print(f"График успешно сохранен в файл: {plot_filename}")

    plt.show()

    return df_plot