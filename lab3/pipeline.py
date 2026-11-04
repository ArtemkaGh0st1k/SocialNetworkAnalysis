from lab3.collect_friends import VKFriendsCollector


def run_lab3_pipeline():
    collector = VKFriendsCollector()
    friends_data = collector.collect()
