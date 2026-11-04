import json
from os.path import join
from pathlib import Path
import sys
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


SAVE_PATH_DIR = Path(join("lab3", "results"))
PLOT_FILENAME = "friends_graph.png"

STATE = 42
LAYOUT_ITERATIONS = 200
LAYOUT_SPREAD = 0.35            # расстояние между вершинами в spring_layout
OUTER_RING_RADIUS = 1.2         # радиус кольца для малых компонент
COMPONENT_RADIUS = 0.05         # радиус размещения вершин малой компоненты
MIN_NODE_SIZE = 100
MAX_NODE_SIZE = 1000
MIN_EDGE_WIDTH = 0.5
MAX_EDGE_WIDTH = 4
UNKNOWN_EDGE_WIDTH = 0.4
TOP_USERS_COUNT = 5
MIN_COLORED_COMMUNITY = 5       # сообщества меньше этого размера рисуются серым

COMMUNITY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
COLOR_OTHER = "#898781"
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_SECONDARY = "#52514e"
COLOR_EDGE_KNOWN = "#6f6e69"
COLOR_EDGE_UNKNOWN = "#d5d4cc"


class FriendsGraphVisualizer():

    def __init__(self, graph_path):

        # без node_type id вершин будут прочитаны как строки
        self.graph = nx.read_graphml(graph_path, node_type=int)

        self.source_id = next(
            user_id for user_id, attrs in self.graph.nodes(data=True) if attrs["is_source"]
        )
        self.graph_path = Path(graph_path)

        # исходный пользователь связан со всеми: он размыл бы сообщества и стянул укладку к центру
        self.friends_graph = self.graph.subgraph(
            [user_id for user_id in self.graph if user_id != self.source_id]
        )

        self.communities = []
        self.positions = {}


    def detect_communities(self):
        """Выделяет сообщества методом Лувена и сортирует их по размеру.
        Веса не используются: у рёбер закрытых профилей вес 0 означает «неизвестно»,
        и с весами каждый закрытый профиль становится отдельным сообществом"""

        communities = nx.community.louvain_communities(
            self.friends_graph, weight=None, seed=STATE
        )
        self.communities = sorted(communities, key=len, reverse=True)

        return self.communities


    def compute_layout(self):
        """Считает координаты вершин. Укладка строится только для наибольшей компоненты:
        несвязанные вершины spring_layout отталкивает далеко и сжимает основную часть.
        Малые компоненты ставятся по кольцу вокруг неё, исходный пользователь - в центр"""

        components = sorted(nx.connected_components(self.friends_graph), key=len, reverse=True)

        self.positions = nx.spring_layout(
            self.friends_graph.subgraph(components[0]), k=LAYOUT_SPREAD, weight=None,
            seed=STATE, iterations=LAYOUT_ITERATIONS
        )

        small_components = components[1:]
        for number, component in enumerate(small_components):
            angle = 2 * np.pi * number / len(small_components)
            center = OUTER_RING_RADIUS * np.array([np.cos(angle), np.sin(angle)])

            for index, user_id in enumerate(component):
                user_angle = 2 * np.pi * index / len(component)
                offset = np.array([np.cos(user_angle), np.sin(user_angle)])
                self.positions[user_id] = center + COMPONENT_RADIUS * offset

        self.positions[self.source_id] = np.array([0.0, 0.0])

        return self.positions


    def normalize(self, values, low, high):
        """Приводит значения к диапазону от low до high"""

        values = np.asarray(values, dtype=float)

        if values.max() == values.min():
            return np.full(len(values), low, dtype=float)

        return low + (high - low) * (values - values.min()) / (values.max() - values.min())


    def compute_node_sizes(self):
        """Считает размеры вершин по числу друзей в логарифмической шкале:
        разброс слишком велик, и иначе почти все вершины были бы минимальными.
        Закрытым профилям назначается минимальный размер: их число друзей неизвестно"""

        known = [
            user_id for user_id, attrs in self.graph.nodes(data=True) if attrs["available"]
        ]
        counts = [self.graph.nodes[user_id]["friends_count"] for user_id in known]
        sizes = self.normalize(np.log1p(counts), MIN_NODE_SIZE, MAX_NODE_SIZE)

        result = {user_id : MIN_NODE_SIZE for user_id in self.graph}
        result.update(zip(known, sizes))

        return result


    def split_edges(self):
        """Разделяет рёбра: от исходного пользователя, с известным и неизвестным весом"""

        source_edges, known_edges, unknown_edges = [], [], []

        for user_a, user_b, attrs in self.graph.edges(data=True):
            if self.source_id in (user_a, user_b):
                source_edges.append((user_a, user_b))
            elif attrs["weight_known"]:
                known_edges.append((user_a, user_b))
            else:
                unknown_edges.append((user_a, user_b))

        return source_edges, known_edges, unknown_edges


    def get_community_styles(self):
        """Возвращает цвет каждой вершины и подписи для легенды"""

        colors = {user_id : COLOR_OTHER for user_id in self.friends_graph}
        legend_items = []
        colored_count = 0

        for number, community in enumerate(self.communities, start=1):
            if colored_count == len(COMMUNITY_COLORS) or len(community) < MIN_COLORED_COMMUNITY:
                break

            color = COMMUNITY_COLORS[colored_count]
            colors.update({user_id : color for user_id in community})
            legend_items.append((color, f"Сообщество {number}: {len(community)} чел."))
            colored_count += 1

        other_count = sum(map(len, self.communities[colored_count:]))
        if other_count:
            legend_items.append((COLOR_OTHER, f"Малые сообщества: {other_count} чел."))

        return colors, legend_items


    def draw_edges(self, ax, known_edges, unknown_edges, source_edges):
        """Рисует рёбра, толщина соответствует весу только там, где вес известен"""

        nx.draw_networkx_edges(self.graph, self.positions, edgelist=source_edges, ax=ax,
                               width=UNKNOWN_EDGE_WIDTH, edge_color=COLOR_EDGE_UNKNOWN, alpha=0.35)
        nx.draw_networkx_edges(self.graph, self.positions, edgelist=unknown_edges, ax=ax,
                               width=UNKNOWN_EDGE_WIDTH, edge_color=COLOR_EDGE_UNKNOWN)

        weights = [self.graph.edges[edge]["weight"] for edge in known_edges]
        widths = self.normalize(weights, MIN_EDGE_WIDTH, MAX_EDGE_WIDTH)
        nx.draw_networkx_edges(self.graph, self.positions, edgelist=known_edges, ax=ax,
                               width=widths, edge_color=COLOR_EDGE_KNOWN, alpha=0.55)

        return weights


    def draw_nodes(self, ax, sizes, colors):
        """Рисует вершины: закрытые профили полыми, исходного пользователя ромбом"""

        friends = list(self.friends_graph)
        opened = [user_id for user_id in friends if self.graph.nodes[user_id]["available"]]
        closed = [user_id for user_id in friends if not self.graph.nodes[user_id]["available"]]

        nx.draw_networkx_nodes(self.graph, self.positions, nodelist=closed, ax=ax,
                               node_size=[sizes[user_id] for user_id in closed],
                               node_color=COLOR_SURFACE,
                               edgecolors=[colors[user_id] for user_id in closed],
                               linewidths=1.5)
        nx.draw_networkx_nodes(self.graph, self.positions, nodelist=opened, ax=ax,
                               node_size=[sizes[user_id] for user_id in opened],
                               node_color=[colors[user_id] for user_id in opened],
                               edgecolors=COLOR_SURFACE, linewidths=1)
        nx.draw_networkx_nodes(self.graph, self.positions, nodelist=[self.source_id], ax=ax,
                               node_size=sizes[self.source_id], node_color=COLOR_INK,
                               node_shape="D", edgecolors=COLOR_SURFACE, linewidths=1.5)


    def marker(self, label, color=COLOR_OTHER, size=MIN_NODE_SIZE, face=None, shape="o"):
        """Создаёт маркер для легенды, размер задаётся как у вершин графа"""

        return Line2D([], [], linestyle="none", marker=shape, label=label,
                      markersize=np.sqrt(size), markeredgecolor=color,
                      markerfacecolor=face or color, markeredgewidth=1.2)


    def add_legends(self, fig, legend_items, sizes, weights):
        """Добавляет легенды цвета, размера вершин и толщины рёбер"""

        def add_legend(handles, title, top, labelspacing=0.9):
            fig.legend(handles=handles, title=title, loc="upper left",
                       bbox_to_anchor=(0.79, top), frameon=False, alignment="left",
                       labelspacing=labelspacing, labelcolor=COLOR_SECONDARY, fontsize=11,
                       title_fontproperties={"weight" : "bold", "size" : 12})

        groups = [self.marker(label, color) for color, label in legend_items]
        groups.append(self.marker("Закрытый профиль", face=COLOR_SURFACE))
        groups.append(self.marker("Исходный пользователь", COLOR_INK, shape="D"))
        add_legend(groups, "Вершины", 0.93)

        known = [user_id for user_id, attrs in self.graph.nodes(data=True) if attrs["available"]]
        known.sort(key=lambda user_id : self.graph.nodes[user_id]["friends_count"])
        samples = [known[0], known[len(known) // 2], known[-1]]
        add_legend(
            [self.marker(str(self.graph.nodes[user_id]["friends_count"]), size=sizes[user_id])
             for user_id in samples],
            "Число друзей", 0.60, labelspacing=2
        )

        low, high = min(weights), max(weights)
        add_legend(
            [Line2D([], [], color=COLOR_EDGE_KNOWN, linewidth=MIN_EDGE_WIDTH, label=str(low)),
             Line2D([], [], color=COLOR_EDGE_KNOWN, linewidth=MAX_EDGE_WIDTH, label=str(high)),
             Line2D([], [], color=COLOR_EDGE_UNKNOWN, linewidth=1, label="Вес неизвестен")],
            "Общие друзья (рёбра)", 0.33
        )


    def draw(self):
        """Строит рисунок графа и сохраняет его в файл"""

        sizes = self.compute_node_sizes()
        colors, legend_items = self.get_community_styles()
        source_edges, known_edges, unknown_edges = self.split_edges()

        with plt.rc_context({"font.family" : ["Segoe UI", "DejaVu Sans"]}):
            fig = plt.figure(figsize=(16, 10.5), facecolor=COLOR_SURFACE)
            ax = fig.add_axes([0.01, 0.01, 0.77, 0.90])
            ax.set_axis_off()

            weights = self.draw_edges(ax, known_edges, unknown_edges, source_edges)
            self.draw_nodes(ax, sizes, colors)
            self.add_legends(fig, legend_items, sizes, weights)

            fig.text(0.015, 0.965, "Граф друзей пользователя VK и его сообщества",
                     fontsize=16, fontweight="bold", color=COLOR_INK)
            fig.text(0.015, 0.935,
                     f"Вершин: {self.graph.number_of_nodes()}, рёбер: {self.graph.number_of_edges()}. "
                     "Сообщества: метод Лувена. Укладка: spring_layout. "
                     "Рёбра исходного пользователя показаны бледно.",
                     fontsize=11, color=COLOR_SECONDARY)

            SAVE_PATH_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(SAVE_PATH_DIR / PLOT_FILENAME, dpi=200)

        print(f"График успешно сохранен в файл: {PLOT_FILENAME}")
        plt.show(block=True)


    def summarize_communities(self):
        """Описывает сообщества и сохраняет описание в JSON"""

        details = []

        for number, community in enumerate(self.communities, start=1):
            subgraph = self.friends_graph.subgraph(community)
            top_users = sorted(community, key=self.friends_graph.degree, reverse=True)

            details.append({
                "community" : number,
                "users" : len(community),
                "closed_profiles" : sum(
                    not self.graph.nodes[user_id]["available"] for user_id in community
                ),
                "inner_edges" : subgraph.number_of_edges(),
                "density" : round(nx.density(subgraph), 4),
                "most_connected_users" : top_users[:TOP_USERS_COUNT],
            })

        summary = \
        {
            "modularity" : round(
                nx.community.modularity(self.friends_graph, self.communities, weight=None), 4
            ),
            "communities_count" : len(self.communities),
            "communities" : details,
        }

        path = self.graph_path.with_name(f"{self.graph_path.stem}_communities.json")
        with open(path, mode="w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=4)

        print(f"Модулярность: {summary['modularity']}, сообществ: {summary['communities_count']}")
        print(f"Размеры сообществ: {[item['users'] for item in details]}")
        print(f"Описание сообществ успешно сохранено в файл: {path}")

        return summary


    def run(self):
        """Выделяет сообщества, строит рисунок и сохраняет описание"""

        self.detect_communities()
        self.compute_layout()
        summary = self.summarize_communities()
        self.draw()

        return summary


if __name__ == "__main__":

    # python -m lab3.visualize_graph [путь к файлу]; без пути берётся единственный файл графа
    graph_files = sys.argv[1:] or list(SAVE_PATH_DIR.glob("graph_*.graphml"))

    if len(graph_files) != 1:
        raise SystemExit(f"Укажите один файл графа, найдено: {len(graph_files)}")

    FriendsGraphVisualizer(graph_files[0]).run()
