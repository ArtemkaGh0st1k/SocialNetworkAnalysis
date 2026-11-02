import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import (AgglomerativeClustering,
                             KMeans,
                             MiniBatchKMeans)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

STATE = 42
SAVE_PATH_DIR = "lab2" / "results"


def perform_clustering_and_visualization(cleaned_texts, n_cluster=4):
    """Выполняет TF-IDF векторизацию, кластеризацию (3 алгоритма)
    и 2D-визуалзиацию через PCA"""

    if len(cleaned_texts) < n_cluster:
        print("Внимание: кол-во текстов меньше выбранного числа кластеров")
        n_cluster = max(2, len(cleaned_texts) // 2)

    print("Векторизация текстов с помощью TF_IDF ...")

    vectorizer = TfidfVectorizer(max_features=1000,
                                 min_df=2,
                                 max_df=0.85,
                                 ngram_range=(1, 2))

    X_sparse = vectorizer.fit_transform(cleaned_texts)
    X_dense = X_sparse.toarray()    # для алгоритмов, требующих плотную матрицу

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

    # выбираем основные метки для дальнейшей визуализации
    cluster_labels = models["KMeans"]

    print("Снижение размерности с помощью PCA до 2D ...")

    pca = PCA(n_components=2, random_state=STATE)
    coords = pca.fit_transform(X_dense)

    df_plot = pd.DataFrame
    (
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
        palatte="tab10",
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