from lab3.collect_friends import VKFriendsCollector


# id или короткое имя пользователя; None - владелец токена
SOURCE_USER = None


def run_lab3_pipeline():
    collector = VKFriendsCollector()
    friends_data = collector.collect(SOURCE_USER)
