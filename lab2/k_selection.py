from itertools import combinations
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import (AgglomerativeClustering,
                             KMeans,
                             MiniBatchKMeans)
from sklearn.metrics import (adjusted_rand_score,
                             davies_bouldin_score,
                             silhouette_score)

from lab2.text_clustering import build_tfidf_vectorizer

K_RANGE = range(2, 16)
SEEDS = range(20)
MIN_CLUSTER_SIZE = 2
SAVE_PATH_DIR = Path(join("lab2", "results"))

COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
ALGORITHM_STYLES = \
{
    "KMeans" : {"color" : "#2a78d6", "marker" : "o"},
    "MiniBatchKMeans" : {"color" : "#eb6834", "marker" : "s"},
    "Agglomerative" : {"color" : "#1baf7a", "marker" : "^"},
}


def fit_predict(algorithm, k, seed, X_dense):
    """Обучает выбранный алгоритм и возвращает метки кластеров"""

    if algorithm == "KMeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    elif algorithm == "MiniBatchKMeans":
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=k)

    return model.fit_predict(X_dense)


def evaluate_k(lemmatized_texts):
    """Считает метрики качества кластеризации для каждого k и алгоритма"""

    X_dense = build_tfidf_vectorizer().fit_transform(lemmatized_texts).toarray()
    rng = np.random.default_rng(42)

    rows = []
    for k in K_RANGE:
        for algorithm in ("KMeans", "MiniBatchKMeans", "Agglomerative"):
            # агломеративная кластеризация детерминирована, ей хватает одного запуска
            seeds = [0] if algorithm == "Agglomerative" else SEEDS
            runs = [fit_predict(algorithm, k, seed, X_dense) for seed in seeds]

            silhouettes = [silhouette_score(X_dense, labels, metric="cosine")
                           for labels in runs]
            davies_bouldin = [davies_bouldin_score(X_dense, labels) for labels in runs]
            min_sizes = [np.bincount(labels).min() for labels in runs]
            # насколько совпадают разбиения при разных случайных запусках
            stability = [adjusted_rand_score(a, b) for a, b in combinations(runs, 2)]

            rows.append({
                "k" : k,
                "algorithm" : algorithm,
                "silhouette" : np.mean(silhouettes),
                "silhouette_std" : np.std(silhouettes),
                "davies_bouldin" : np.mean(davies_bouldin),
                "min_cluster_size" : np.mean(min_sizes),
                "stability_ari" : np.mean(stability) if stability else 1.0,
            })

        # базовая линия: те же данные без структуры (столбцы перемешаны независимо)
        null_scores = []
        for seed in SEEDS:
            X_null = rng.permuted(X_dense, axis=0)
            labels = fit_predict("KMeans", k, seed, X_null)
            null_scores.append(silhouette_score(X_null, labels, metric="cosine"))

        inertia = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_dense).inertia_
        rows[-3].update({"inertia" : inertia,
                         "null_silhouette" : np.mean(null_scores),
                         "null_silhouette_std" : np.std(null_scores)})

    df = pd.DataFrame(rows)
    SAVE_PATH_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_PATH_DIR / "k_selection_metrics.csv", index=False, encoding="utf-8-sig")

    return df


def select_optimal_k(df):
    """Выбирает k с максимальным силуэтом KMeans среди разбиений без мелких кластеров.
    Без этого ограничения силуэт растёт за счёт кластеров из одного поста"""

    kmeans = df[df["algorithm"] == "KMeans"]
    allowed = kmeans[kmeans["min_cluster_size"] >= MIN_CLUSTER_SIZE]

    return int(allowed.loc[allowed["silhouette"].idxmax(), "k"])


def plot_metric(ax, df, column, algorithms):
    """Рисует линию метрики по k для каждого алгоритма"""

    for algorithm in algorithms:
        style = ALGORITHM_STYLES[algorithm]
        data = df[df["algorithm"] == algorithm]
        ax.plot(data["k"], data[column], label=algorithm, color=style["color"],
                marker=style["marker"], linewidth=1.5, markersize=6,
                markeredgecolor=COLOR_SURFACE, markeredgewidth=1)


def format_axes(ax, title, ylabel, best_k):
    """Оформляет оси и отмечает выбранное k"""

    ax.axvline(best_k, color=COLOR_MUTED, linewidth=1, zorder=0)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=COLOR_INK)
    ax.set_xlabel("Число кластеров k", color=COLOR_SECONDARY)
    ax.set_ylabel(ylabel, color=COLOR_SECONDARY)
    ax.set_xticks(list(K_RANGE))
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLOR_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_AXIS)


def plot_k_selection(df, best_k):
    """Строит графики метрик по k и сохраняет их в файл"""

    algorithms = list(ALGORITHM_STYLES)
    kmeans = df[df["algorithm"] == "KMeans"]

    with plt.rc_context({"font.family" : ["Segoe UI", "DejaVu Sans"]}):
        fig, axes = plt.subplot_mosaic("AAB;CDE", figsize=(15, 9), layout="constrained",
                                       facecolor=COLOR_SURFACE)

        ax = axes["A"]
        plot_metric(ax, df, "silhouette", algorithms)
        ax.plot(kmeans["k"], kmeans["null_silhouette"], color=COLOR_MUTED, linewidth=1.5,
                label="Данные без структуры (KMeans)")
        ax.axhline(0, color=COLOR_AXIS, linewidth=1, zorder=0)
        format_axes(ax, "Силуэт растёт с k, но остаётся низким: кластеры выражены слабо",
                    "Силуэт (косинусное расстояние), выше лучше", best_k)
        ax.annotate(f"выбрано k = {best_k}", xy=(best_k, ax.get_ylim()[1]),
                    xytext=(5, -12), textcoords="offset points", color=COLOR_SECONDARY)
        ax.legend(frameon=False, loc="center right", labelcolor=COLOR_SECONDARY)

        ax = axes["B"]
        ax.plot(kmeans["k"], kmeans["inertia"], color=ALGORITHM_STYLES["KMeans"]["color"],
                marker="o", linewidth=1.5, markersize=6,
                markeredgecolor=COLOR_SURFACE, markeredgewidth=1)
        format_axes(ax, "Метод локтя: выраженного излома нет",
                    "Инерция KMeans, ниже лучше", best_k)

        ax = axes["C"]
        plot_metric(ax, df, "davies_bouldin", algorithms)
        format_axes(ax, "Индекс Дэвиса–Болдина не даёт минимума",
                    "Индекс Дэвиса–Болдина, ниже лучше", best_k)
        ax.legend(frameon=False, labelcolor=COLOR_SECONDARY)

        ax = axes["D"]
        plot_metric(ax, df, "min_cluster_size", algorithms)
        ax.axhline(MIN_CLUSTER_SIZE, color=COLOR_INK, linewidth=1, zorder=0)
        ax.annotate(f"порог: {MIN_CLUSTER_SIZE} поста", xy=(K_RANGE[-1], MIN_CLUSTER_SIZE),
                    xytext=(0, 5), textcoords="offset points", ha="right", color=COLOR_INK)
        format_axes(ax, "При большом k кластеры сжимаются до одного поста",
                    "Размер наименьшего кластера, постов", best_k)
        ax.legend(frameon=False, labelcolor=COLOR_SECONDARY)

        # агломеративная кластеризация детерминирована, её устойчивость всегда равна 1
        ax = axes["E"]
        plot_metric(ax, df, "stability_ari", ["KMeans", "MiniBatchKMeans"])
        format_axes(ax, "MiniBatchKMeans неустойчив между запусками",
                    "Совпадение разбиений (ARI), выше лучше", best_k)
        ax.legend(frameon=False, labelcolor=COLOR_SECONDARY)

        for ax in (axes["B"], axes["C"], axes["D"], axes["E"]):
            ax.set_xticks(list(K_RANGE)[::2])

        plt.savefig(SAVE_PATH_DIR / "k_selection.png", dpi=200)

    print("График успешно сохранен в файл: k_selection.png")
    plt.show()


def run_k_selection(lemmatized_texts):
    """Проводит опыт при разных k и возвращает оптимальное значение"""

    print("Опыт при разных k ...")

    df = evaluate_k(lemmatized_texts)
    best_k = select_optimal_k(df)
    plot_k_selection(df, best_k)

    print(f"Оптимальное число кластеров: {best_k}")

    return best_k
