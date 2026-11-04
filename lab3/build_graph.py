from itertools import combinations
import json
from os.path import join
from pathlib import Path
import sys
import networkx as nx


SAVE_PATH_DIR = Path(join("lab3", "results"))
TOP_USERS_COUNT = 5


class FriendsGraphBuilder():

    def __init__(self, data_path):

        with open(data_path, mode="r", encoding="utf-8") as file:
            data = json.load(file)

        self.source_id = data["user_id"]
        self.friends = data["friends"]
        self.unavailable = set(data["unavailable"])

        # ключи JSON всегда строки, а id в списках - числа
        self.friend_lists = {
            int(user_id) : set(friends)
            for user_id, friends in data["friends_of_friends"].items()
        }

        # без этой проверки несобранные списки выглядели бы как пустые
        missing = [user_id for user_id in self.friends if user_id not in self.friend_lists]
        if missing:
            raise ValueError(f"Сбор не завершён: нет списков для {len(missing)} друзей")

        self.friend_lists[self.source_id] = set(self.friends)
        self.graph = nx.Graph()


    def is_available(self, user_id):
        """Проверяет, известен ли список друзей пользователя"""

        return user_id not in self.unavailable


    def are_friends(self, user_a, user_b):
        """Проверяет дружбу с обеих сторон: закрытый профиль виден в списке открытого"""

        return user_b in self.friend_lists[user_a] or user_a in self.friend_lists[user_b]


    def count_common_friends(self, user_a, user_b):
        """Считает общих друзей пары. Исходный пользователь не учитывается:
        он есть в списке каждого друга и завысил бы вес всех рёбер на 1"""

        common = self.friend_lists[user_a] & self.friend_lists[user_b]

        return len(common - {self.source_id})


    def add_nodes(self):
        """Добавляет исходного пользователя и его друзей.
        У недоступного профиля friends_count = 0 означает, что число неизвестно"""

        for user_id in [self.source_id, *self.friends]:
            self.graph.add_node(
                user_id,
                friends_count=len(self.friend_lists[user_id]),
                available=self.is_available(user_id),
                is_source=user_id == self.source_id,
            )


    def add_edge(self, user_a, user_b):
        """Добавляет ребро с весом, равным числу общих друзей.
        При недоступном профиле вес неизвестен: weight_known = False"""

        self.graph.add_edge(
            user_a,
            user_b,
            weight=self.count_common_friends(user_a, user_b),
            weight_known=self.is_available(user_a) and self.is_available(user_b),
        )


    def build(self):
        """Строит граф: вершины - пользователи, рёбра - дружеские связи"""

        print("Начало построения графа ...")

        self.add_nodes()

        for friend_id in self.friends:
            self.add_edge(self.source_id, friend_id)

        for user_a, user_b in combinations(self.friends, 2):
            if self.are_friends(user_a, user_b):
                self.add_edge(user_a, user_b)

        print(
            f"Граф построен: вершин {self.graph.number_of_nodes()}, "
            f"рёбер {self.graph.number_of_edges()}"
        )

        return self.graph


    def describe_graph(self, graph):
        """Считает метрики графа. Диаметр определён только для связного графа,
        поэтому он считается по наибольшей компоненте связности"""

        components = list(nx.connected_components(graph))
        largest = graph.subgraph(max(components, key=len))

        return \
        {
            "nodes" : graph.number_of_nodes(),
            "edges" : graph.number_of_edges(),
            "density" : round(nx.density(graph), 4),
            "average_clustering" : round(nx.average_clustering(graph), 4),
            "components" : len(components),
            "isolated_nodes" : nx.number_of_isolates(graph),
            "largest_component_nodes" : largest.number_of_nodes(),
            "largest_component_diameter" : nx.diameter(largest),
        }


    def compute_metrics(self):
        """Считает метрики полного графа и графа без исходного пользователя.
        Исходный пользователь связан со всеми, поэтому в полном графе
        диаметр не больше 2, а коэффициент кластеризации завышен"""

        without_source = self.graph.subgraph(self.friends)

        top_users = sorted(self.friends, key=without_source.degree, reverse=True)
        unavailable_count = len(self.unavailable)

        return \
        {
            "friends" : len(self.friends),
            "unavailable_friends" : unavailable_count,
            # связь внутри пары неизвестна, если закрыты оба профиля
            "unknown_pairs" : unavailable_count * (unavailable_count - 1) // 2,
            "total_pairs" : len(self.friends) * (len(self.friends) - 1) // 2,
            "full_graph" : self.describe_graph(self.graph),
            "without_source" : self.describe_graph(without_source),
            "most_connected_users" : [
                {
                    "id" : user_id,
                    "friends_in_network" : without_source.degree(user_id),
                    "friends_count" : self.graph.nodes[user_id]["friends_count"],
                }
                for user_id in top_users[:TOP_USERS_COUNT]
            ],
        }


    def save(self, metrics):
        """Сохраняет граф в GraphML, а метрики в JSON"""

        SAVE_PATH_DIR.mkdir(parents=True, exist_ok=True)

        graph_path = SAVE_PATH_DIR / f"graph_{self.source_id}.graphml"
        metrics_path = SAVE_PATH_DIR / f"graph_{self.source_id}_metrics.json"

        nx.write_graphml(self.graph, graph_path)

        with open(metrics_path, mode="w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=4)

        print(f"Граф успешно сохранен в файл: {graph_path}")
        print(f"Метрики успешно сохранены в файл: {metrics_path}")


    def run(self):
        """Строит граф, считает метрики и сохраняет результат"""

        self.build()
        metrics = self.compute_metrics()
        self.save(metrics)

        print(json.dumps(metrics, ensure_ascii=False, indent=4))

        return self.graph, metrics


if __name__ == "__main__":

    # python -m lab3.build_graph [путь к файлу]; без пути берётся единственный файл с друзьями
    data_files = sys.argv[1:] or list(SAVE_PATH_DIR.glob("friends_*.json"))

    if len(data_files) != 1:
        raise SystemExit(f"Укажите один файл с друзьями, найдено: {len(data_files)}")

    FriendsGraphBuilder(data_files[0]).run()
