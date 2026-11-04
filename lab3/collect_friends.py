import json
import os
from os.path import join
from pathlib import Path
import time
from dotenv import load_dotenv
import vk_api
from vk_api.exceptions import ApiError


################ =============================== ################

# Чтобы собрать со своего аккаунта данные с помощью библиотеки
# vk_api нужно настроить приватность своего профиля:

#
# В настройки: 
# 1) Приватность:
#   1.1) Закрытый профиль -> Открытый
#   1.2) Кому в интернете видна моя страница -> Все пользователи 
#   1.3) Кто видит основную информацию моей страницы -> Все пользователи
#
# Также информация не обо всех пользователях может быть собрана 
# из-за их собственных настроек приватности


################ =============================== ################



ENV_PATH = join("lab1", ".env")
SAVE_PATH_DIR = Path(join("lab3", "results"))

MAX_FRIENDS_BY_REQUEST = 5000   # макс. кол-во друзей за один запрос
REQUEST_PAUSE = 1               # пауза между запросами, сек
RETRY_PAUSE = 5                 # базовая пауза перед повтором запроса, сек
MAX_RETRIES = 3
SAVE_EVERY = 10                 # как часто сохранять прогресс, друзей

RATE_LIMIT_CODES = (6, 29)      # превышен лимит: данные есть, запрос нужно повторить
UNAVAILABLE_CODES = (15, 18, 30)    # доступ запрещён, страница удалена, профиль закрыт


class VKFriendsCollector():

    def __init__(self):

        load_dotenv(ENV_PATH)

        self.token = os.getenv("API_TOKEN")
        if not self.token:
            raise ValueError(f"API_TOKEN не найден в файле {ENV_PATH}!")

        self.user = "three_nines999"
        vk_session = vk_api.VkApi(token=self.token)
        self.vk = vk_session.get_api()


    def resolve_user_id(self, user=None):
        """Возвращает числовой id пользователя по id или короткому имени.
        Без аргумента возвращает id владельца токена"""

        response = self.vk.users.get(user_ids=user) if user else self.vk.users.get()

        return response[0]["id"]


    def request_friends(self, **params):
        """Вызывает friends.get и повторяет запрос при превышении лимита"""

        for attempt in range(MAX_RETRIES + 1):
            try:
                return self.vk.friends.get(**params)
            except ApiError as error:
                if error.code not in RATE_LIMIT_CODES or attempt == MAX_RETRIES:
                    raise

                pause = RETRY_PAUSE * (attempt + 1)
                print(f"Превышен лимит запросов (код {error.code}), ждём {pause} сек ...")
                time.sleep(pause)


    def get_friends(self, user_id):
        """Возвращает полный список id друзей пользователя"""

        friends = []

        while True:
            response = self.request_friends(
                user_id=user_id, count=MAX_FRIENDS_BY_REQUEST, offset=len(friends)
            )

            items = response.get("items", [])
            friends.extend(items)

            if not items or len(friends) >= response.get("count", 0):
                return friends

            time.sleep(REQUEST_PAUSE)


    def load_progress(self, path : Path):
        """Загружает результаты прошлого запуска, если они есть"""

        if not path.exists():
            return None

        with open(path, mode="r", encoding="utf-8") as file:
            return json.load(file)


    def save_progress(self, data, path : Path):
        """Сохраняет данные через временный файл, чтобы сбой не испортил результат"""

        SAVE_PATH_DIR.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix(".tmp")
        with open(temp_path, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        os.replace(temp_path, path)


    def collect(self, user=None):
        """Собирает друзей пользователя и друзей каждого из его друзей.
        При повторном запуске продолжает сбор с места остановки"""

        if not user:
            user = self.user

        user_id = self.resolve_user_id(user)
        path = SAVE_PATH_DIR / f"friends_{user_id}.json"

        data = self.load_progress(path)

        if data:
            print(f"Найден файл {path}, продолжаем сбор ...")
        else:
            # ошибка на исходном пользователе не обрабатывается: без него собирать нечего
            data = \
            {
                "user_id" : user_id,
                "friends" : self.get_friends(user_id),
                "friends_of_friends" : {},
                "unavailable" : [],
            }
            time.sleep(REQUEST_PAUSE)

        friends = data["friends"]
        friends_of_friends = data["friends_of_friends"]

        print(f"Начало сбора друзей друзей, всего друзей: {len(friends)} ...")

        # прогресс сохраняется и при ошибке, которая прерывает сбор
        try:
            for number, friend_id in enumerate(friends, start=1):
                key = str(friend_id)    # ключи JSON всегда строки
                if key in friends_of_friends:
                    continue

                try:
                    friends_of_friends[key] = self.get_friends(friend_id)
                except ApiError as error:
                    if error.code not in UNAVAILABLE_CODES:
                        raise

                    friends_of_friends[key] = []
                    data["unavailable"].append(friend_id)

                if number % SAVE_EVERY == 0:
                    self.save_progress(data, path)
                    print(f"Обработано друзей: {number} из {len(friends)} ...")

                time.sleep(REQUEST_PAUSE)
        finally:
            self.save_progress(data, path)

        unavailable_count = len(data["unavailable"])

        print(f"Сбор завершен! Недоступных профилей: {unavailable_count} из {len(friends)}")
        print(f"Данные успешно сохранены в файл: {path}")

        if friends and unavailable_count == len(friends):
            print("Внимание: недоступны все профили, проверьте права токена")

        return data
