from os.path import join

from lab3.collect_friends import VKFriendsCollector
from lab3.build_graph import FriendsGraphBuilder


def run_lab3_pipeline():

    # 1е задание

    #collector = VKFriendsCollector()
    #friends_data = collector.collect()

    # 2е задание

    friends_json_path = join("lab3", "results", "friends_279910828.json")
    FriendsGraphBuilder(friends_json_path).run()
