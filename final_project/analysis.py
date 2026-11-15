from collections import Counter
import json
from os.path import join
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLEAN_DATASET_PATH = Path(join("final_project", "data", "posts_clean.csv"))
SAVE_PATH_DIR = Path(join("final_project", "results", "eda"))

MIN_CELL_POSTS = 30             # меньше постов в ячейке - медиана ненадёжна
MIN_HOUR_POSTS = 300            # порог для выводов о лучшем и худшем часе
MIN_WORD_POSTS = 200            # слово должно встречаться хотя бы в стольких постах
MIN_WORD_GROUPS = 4             # и в стольких сообществах, иначе это примета одного вуза
TOP_WORDS_COUNT = 12
LENGTH_BINS = 10

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

COLOR_MAIN = "#2a78d6"
COLOR_POSITIVE = "#2a78d6"
COLOR_NEGATIVE = "#e34948"
COLOR_NEUTRAL = "#f0efec"
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"


class PostsAnalyzer():

    def __init__(self, dataset_path=CLEAN_DATASET_PATH):

        self.posts = pd.read_csv(dataset_path, encoding="utf-8-sig")
        self.posts["lemmas"] = self.posts["lemmas"].fillna("")
        self.summary = {"posts" : len(self.posts), "groups" : self.posts["group"].nunique()}

        SAVE_PATH_DIR.mkdir(parents=True, exist_ok=True)


    def style_axes(self, ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
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
        fig.text(0.01, 0.955, subtitle, fontsize=10, color=COLOR_SECONDARY, va="top")
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))

        return fig


    def save(self, fig, filename):
        """Сохраняет рисунок в файл"""

        fig.savefig(SAVE_PATH_DIR / filename, dpi=200, facecolor=COLOR_SURFACE)
        plt.close(fig)
        print(f"График успешно сохранен в файл: {filename}")


    def to_percent(self, relative):
        """Переводит относительный отклик в отклонение от типичного поста, %"""

        return (relative - 1) * 100


    def plot_activity(self):
        """Число постов по месяцам для каждого сообщества"""

        posts = self.posts
        months = sorted(posts["month"].unique())
        groups = posts.groupby("group").size().sort_values(ascending=False).index

        fig = self.new_figure("Активность сообществ",
                              "Число постов в месяц, без репостов. Шкала общая для всех сообществ.",
                              figsize=(13, 8))
        axes = fig.subplots(3, 3, sharex=True, sharey=True)

        for ax, group in zip(axes.flat, groups):
            counts = posts[posts["group"] == group].groupby("month").size().reindex(months, fill_value=0)
            ax.bar(range(len(months)), counts.values, color=COLOR_MAIN, width=0.85)
            self.style_axes(ax, title=f"{group} ({counts.sum()})")

        ticks = [i for i, month in enumerate(months) if month.endswith("-01")]
        for ax in axes[-1]:
            ax.set_xticks(ticks, [months[i][:4] for i in ticks])

        self.save(fig, "activity.png")


    def plot_engagement_by_group(self):
        """Доля реакции по сообществам: медиана и межквартильный размах"""

        stats = self.posts.groupby("group")["engagement_rate"].quantile([0.25, 0.5, 0.75]).unstack() * 100
        stats = stats.sort_values(0.5)
        positions = np.arange(len(stats))

        fig = self.new_figure("Доля реакции по сообществам",
                              "Лайки на просмотры, %. Точка - медиана, линия - половина постов вокруг неё.",
                              figsize=(10, 5.5))
        ax = fig.subplots()

        ax.hlines(positions, stats[0.25], stats[0.75], color=COLOR_MAIN, linewidth=2.5, alpha=0.35)
        ax.plot(stats[0.5], positions, "o", color=COLOR_MAIN, markersize=8,
                markeredgecolor=COLOR_SURFACE, markeredgewidth=1.5)
        for position, median in zip(positions, stats[0.5]):
            ax.annotate(f"{median:.2f}%", (median, position), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9, color=COLOR_SECONDARY)

        ax.set_yticks(positions, stats.index)
        self.style_axes(ax, xlabel="Лайки на просмотры, %", grid_axis="x")

        self.summary["engagement_rate_median_percent"] = stats[0.5].round(3).to_dict()
        self.save(fig, "engagement_by_group.png")


    def plot_time_heatmap(self):
        """Отклик в зависимости от дня недели и часа публикации"""

        posts = self.posts
        grouped = posts.groupby(["weekday", "hour"])["relative_engagement"]
        medians = grouped.median().unstack().reindex(index=range(7), columns=range(24))
        counts = grouped.size().unstack().reindex(index=range(7), columns=range(24)).fillna(0)

        effect = self.to_percent(medians).where(counts >= MIN_CELL_POSTS)
        limit = np.nanmax(np.abs(effect.values))

        cmap = LinearSegmentedColormap.from_list("diverging", [COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE])
        cmap.set_bad(COLOR_SURFACE)

        fig = self.new_figure("Когда публиковать",
                              f"Отклик относительно типичного поста того же сообщества и месяца, %. "
                              f"Местное время. Пустые ячейки: меньше {MIN_CELL_POSTS} постов.",
                              figsize=(13, 5))
        ax = fig.subplots()

        image = ax.imshow(effect.values, cmap=cmap, aspect="auto",
                          norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit))
        ax.set_xticks(range(24), [str(hour) for hour in range(24)])
        ax.set_yticks(range(7), WEEKDAYS)
        self.style_axes(ax, xlabel="Час публикации", grid_axis=None)
        ax.spines["bottom"].set_visible(False)

        colorbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.01)
        colorbar.set_label("Отклонение от типичного поста, %", color=COLOR_SECONDARY)
        colorbar.outline.set_visible(False)

        by_hour = posts.groupby("hour")["relative_engagement"].agg(["median", "size"])
        # ночью постов мало и среди них много поздравлений, поэтому для выводов нужен порог выше
        by_hour = by_hour[by_hour["size"] >= MIN_HOUR_POSTS]
        by_weekday = posts.groupby("weekday")["relative_engagement"].median()

        self.summary["best_hour"] = int(by_hour["median"].idxmax())
        self.summary["worst_hour"] = int(by_hour["median"].idxmin())
        self.summary["hour_effect_percent"] = self.to_percent(by_hour["median"]).round(1).to_dict()
        self.summary["weekday_effect_percent"] = {
            WEEKDAYS[day] : round(value, 1) for day, value in self.to_percent(by_weekday).items()
        }
        self.save(fig, "time_heatmap.png")


    def get_post_format(self, post):
        """Определяет основной формат поста"""

        if post["has_poll"]:
            return "Опрос"
        if post["has_video"]:
            return "Видео"
        if post["n_photos"] >= 5:
            return "5 и больше фото"
        if post["n_photos"] >= 2:
            return "2-4 фото"
        if post["n_photos"] == 1:
            return "1 фото"
        if post["has_link"]:
            return "Ссылка без фото"

        return "Только текст"


    def plot_diverging_bars(self, ax, labels, values, counts):
        """Рисует отклонения вверх и вниз от нуля с подписями"""

        colors = [COLOR_POSITIVE if value >= 0 else COLOR_NEGATIVE for value in values]
        positions = np.arange(len(values))

        ax.barh(positions, values, color=colors, height=0.65)
        ax.axvline(0, color=COLOR_AXIS, linewidth=1)
        ax.set_yticks(positions, labels)

        for position, value, count in zip(positions, values, counts):
            ax.annotate(f"{value:+.0f}%  ({count})", (value, position),
                        xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                        ha="left" if value >= 0 else "right", va="center", fontsize=9, color=COLOR_SECONDARY)

        left, right = ax.get_xlim()
        margin = (right - left) * 0.18
        ax.set_xlim(left - margin, right + margin)


    def plot_formats(self):
        """Отклик в зависимости от формата поста"""

        posts = self.posts
        posts["format"] = posts.apply(self.get_post_format, axis=1)

        stats = posts.groupby("format")["relative_engagement"].agg(["median", "size"])
        stats["effect"] = self.to_percent(stats["median"])
        stats = stats.sort_values("effect")

        fig = self.new_figure("Формат поста",
                              "Отклик относительно типичного поста того же сообщества и месяца, медиана. "
                              "В скобках - число постов.",
                              figsize=(10, 5))
        ax = fig.subplots()
        self.plot_diverging_bars(ax, stats.index, stats["effect"], stats["size"])
        self.style_axes(ax, xlabel="Отклонение от типичного поста, %", grid_axis="x")

        self.summary["format_effect_percent"] = stats["effect"].round(1).to_dict()
        self.summary["format_counts"] = stats["size"].to_dict()
        self.save(fig, "formats.png")


    def plot_text_length(self):
        """Отклик в зависимости от длины текста"""

        posts = self.posts[self.posts["text_length"] > 0].copy()
        posts["length_bin"] = pd.qcut(posts["text_length"], LENGTH_BINS, duplicates="drop")

        stats = posts.groupby("length_bin", observed=True).agg(
            length=("text_length", "median"),
            median=("relative_engagement", "median"),
            low=("relative_engagement", lambda values : values.quantile(0.25)),
            high=("relative_engagement", lambda values : values.quantile(0.75)),
        )

        fig = self.new_figure("Длина текста",
                              f"Посты разбиты на {LENGTH_BINS} равных по числу групп. "
                              "Линия - медиана, полоса - половина постов вокруг неё.",
                              figsize=(10, 5))
        ax = fig.subplots()

        ax.fill_between(stats["length"], self.to_percent(stats["low"]), self.to_percent(stats["high"]),
                        color=COLOR_MAIN, alpha=0.15, linewidth=0)
        ax.plot(stats["length"], self.to_percent(stats["median"]), "-o", color=COLOR_MAIN, linewidth=2,
                markersize=7, markeredgecolor=COLOR_SURFACE, markeredgewidth=1.5)
        ax.axhline(0, color=COLOR_AXIS, linewidth=1)
        ax.set_xscale("log")
        self.style_axes(ax, xlabel="Длина текста, символов (логарифмическая шкала)",
                        ylabel="Отклонение от типичного поста, %")

        self.summary["text_length_effect_percent"] = {
            int(length) : round(value, 1)
            for length, value in zip(stats["length"], self.to_percent(stats["median"]))
        }
        self.save(fig, "text_length.png")


    def plot_trend(self):
        """Изменение доли реакции со временем"""

        posts = self.posts
        groups = posts.groupby("group").size().sort_values(ascending=False).index
        months = sorted(posts["month"].unique())

        fig = self.new_figure("Доля реакции со временем",
                              "Медиана лайков на просмотры по месяцам, %. Шкала общая для всех сообществ.",
                              figsize=(13, 8))
        axes = fig.subplots(3, 3, sharex=True, sharey=True)

        for ax, group in zip(axes.flat, groups):
            monthly = posts[posts["group"] == group].groupby("month")["engagement_rate"].median() * 100
            monthly = monthly.reindex(months)
            ax.plot(range(len(months)), monthly.values, color=COLOR_MAIN, linewidth=1.8)
            self.style_axes(ax, title=group)

        ticks = [i for i, month in enumerate(months) if month.endswith("-01")]
        for ax in axes[-1]:
            ax.set_xticks(ticks, [months[i][:4] for i in ticks])

        first_year = posts["month"] < months[12]
        last_year = posts["month"] >= months[-12]
        self.summary["engagement_change_percent"] = (
            (posts[last_year].groupby("group")["engagement_rate"].median()
             / posts[first_year].groupby("group")["engagement_rate"].median() - 1) * 100
        ).round(1).to_dict()
        self.save(fig, "trend.png")


    def find_word_effects(self):
        """Считает отклик постов, в которых встречается слово.
        Учитываются только слова, которые встречаются в нескольких сообществах"""

        posts = self.posts
        word_sets = posts["lemmas"].str.split().apply(set)

        post_counts = Counter(word for words in word_sets for word in words)
        frequent = {word for word, count in post_counts.items() if count >= MIN_WORD_POSTS}

        rows = []
        for word in frequent:
            mask = word_sets.apply(lambda words : word in words)
            if posts.loc[mask, "group"].nunique() < MIN_WORD_GROUPS:
                continue

            rows.append({
                "word" : word,
                "posts" : int(mask.sum()),
                "effect" : self.to_percent(posts.loc[mask, "relative_engagement"].median()),
            })

        return pd.DataFrame(rows).sort_values(["effect", "word"])


    def plot_words(self):
        """Слова, связанные с более высоким и более низким откликом"""

        words = self.find_word_effects()
        words.to_csv(SAVE_PATH_DIR / "word_effects.csv", index=False, encoding="utf-8-sig")

        selected = pd.concat([words.head(TOP_WORDS_COUNT), words.tail(TOP_WORDS_COUNT)])

        fig = self.new_figure("Слова и отклик",
                              f"Медианный отклик постов со словом относительно типичного поста. "
                              f"Слова из {MIN_WORD_POSTS}+ постов и {MIN_WORD_GROUPS}+ сообществ. "
                              "Это связь, а не причина.",
                              figsize=(10, 9))
        ax = fig.subplots()
        self.plot_diverging_bars(ax, selected["word"], selected["effect"], selected["posts"])
        self.style_axes(ax, xlabel="Отклонение от типичного поста, %", grid_axis="x")

        self.summary["words_checked"] = len(words)
        self.summary["top_words"] = words.tail(TOP_WORDS_COUNT)[::-1]["word"].tolist()
        self.summary["bottom_words"] = words.head(TOP_WORDS_COUNT)["word"].tolist()
        self.save(fig, "words.png")


    def compute_correlations(self):
        """Считает ранговую корреляцию признаков с откликом"""

        features = ["text_length", "n_lines", "n_photos", "n_emoji", "n_hashtags", "n_links",
                    "n_exclamations", "has_question", "has_video", "has_link", "has_poll", "hour"]

        correlations = self.posts[features].astype(float).corrwith(
            self.posts["log_relative_engagement"], method="spearman"
        ).sort_values()

        correlations.to_csv(SAVE_PATH_DIR / "correlations.csv", header=["spearman"], encoding="utf-8-sig")
        self.summary["spearman"] = correlations.round(3).to_dict()


    def run(self):
        """Строит все графики и сохраняет сводку"""

        print("Начало анализа постов ...")

        with plt.rc_context({"font.family" : ["Segoe UI", "DejaVu Sans"]}):
            self.plot_activity()
            self.plot_engagement_by_group()
            self.plot_time_heatmap()
            self.plot_formats()
            self.plot_text_length()
            self.plot_trend()
            self.plot_words()

        self.compute_correlations()

        with open(SAVE_PATH_DIR / "summary.json", mode="w", encoding="utf-8") as file:
            json.dump(self.summary, file, ensure_ascii=False, indent=4, default=str)

        print(f"Сводка успешно сохранена в файл: {SAVE_PATH_DIR / 'summary.json'}")

        return self.summary


if __name__ == "__main__":

    PostsAnalyzer().run()
