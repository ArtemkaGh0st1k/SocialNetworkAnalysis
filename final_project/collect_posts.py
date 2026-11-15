from datetime import datetime, timedelta, timezone
import json
import os
from os.path import join
from pathlib import Path
import time
from dotenv import load_dotenv
import pandas as pd
import vk_api
from vk_api.exceptions import ApiError


ENV_PATH = join("lab1", ".env")
RAW_DATA_DIR = Path(join("final_project", "data", "raw"))
DATASET_PATH = Path(join("final_project", "data", "posts.csv"))

GROUPS = \
[
    "samara_university",
    "msu_official",
    "hse_university",
    "itmoru",
    "bmstu1830",
    "nsu24",
    "mephi_official",
    "rudn_university",
    "mgimo",
]
PERIOD_YEARS = 3

MAX_POSTS_BY_REQUEST = 100      # макс. кол-во постов за один запрос
REQUEST_PAUSE = 0.4             # пауза между запросами, сек
RETRY_PAUSE = 5                 # базовая пауза перед повтором запроса, сек
MAX_RETRIES = 3
RATE_LIMIT_CODES = (6, 29)      # превышен лимит: запрос нужно повторить


class VKPostsCollector():

    def __init__(self, groups=GROUPS, period_years=PERIOD_YEARS):

        load_dotenv(ENV_PATH)

        self.token = os.getenv("API_TOKEN")
        if not self.token:
            raise ValueError(f"API_TOKEN не найден в файле {ENV_PATH}!")

        vk_session = vk_api.VkApi(token=self.token)
        self.vk = vk_session.get_api()

        self.groups = groups
        self.collected_at = datetime.now(timezone.utc)
        start_date = self.collected_at - timedelta(days=365 * period_years)
        self.start_timestamp = int(start_date.timestamp())


    def request(self, method, **params):
        """Вызывает метод API и повторяет запрос при превышении лимита"""

        for attempt in range(MAX_RETRIES + 1):
            try:
                return method(**params)
            except ApiError as error:
                if error.code not in RATE_LIMIT_CODES or attempt == MAX_RETRIES:
                    raise

                pause = RETRY_PAUSE * (attempt + 1)
                print(f"Превышен лимит запросов (код {error.code}), ждём {pause} сек ...")
                time.sleep(pause)


    def get_members_count(self, domain):
        """Возвращает число подписчиков сообщества на момент сбора"""

        response = self.request(self.vk.groups.getById, group_id=domain, fields="members_count")

        # в новых версиях API ответ приходит словарём с ключом groups
        groups = response["groups"] if isinstance(response, dict) else response

        return groups[0].get("members_count", 0)


    def parse_post(self, post, domain):
        """Извлекает из поста поля, нужные для анализа"""

        attachments = [att.get("type") for att in post.get("attachments", [])]

        return \
        {
            "group" : domain,
            "post_id" : post.get("id"),
            "owner_id" : post.get("owner_id"),
            "timestamp" : post.get("date", 0),
            "text" : post.get("text", ""),
            "likes" : post.get("likes", {}).get("count", 0),
            "comments" : post.get("comments", {}).get("count", 0),
            "reposts" : post.get("reposts", {}).get("count", 0),
            "views" : post.get("views", {}).get("count"),
            "attachments" : ",".join(attachments),
            "is_pinned" : bool(post.get("is_pinned")),
            "is_ad" : bool(post.get("marked_as_ads")),
            "is_repost" : bool(post.get("copy_history")),
            "post_type" : post.get("post_type", ""),
        }


    def load_progress(self, path : Path):
        """Загружает результаты прошлого запуска, если они есть"""

        if not path.exists():
            return None

        with open(path, mode="r", encoding="utf-8") as file:
            return json.load(file)


    def save_progress(self, data, path : Path):
        """Сохраняет данные через временный файл, чтобы сбой не испортил результат"""

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix(".tmp")
        with open(temp_path, mode="w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

        os.replace(temp_path, path)


    def collect_group(self, domain):
        """Собирает посты сообщества за заданный период.
        При повторном запуске продолжает сбор с места остановки"""

        path = RAW_DATA_DIR / f"{domain}.json"
        data = self.load_progress(path)

        if data and data["finished"]:
            print(f"{domain}: уже собрано {len(data['posts'])} постов")
            return data

        if not data:
            data = \
            {
                "group" : domain,
                "members_count" : self.get_members_count(domain),
                "collected_at" : int(self.collected_at.timestamp()),
                "offset" : 0,
                "finished" : False,
                "posts" : [],
            }

        # за время сбора появляются новые посты и сдвигают ленту, поэтому id проверяются на повтор
        known_ids = {post["post_id"] for post in data["posts"]}

        try:
            while not data["finished"]:
                response = self.request(
                    self.vk.wall.get, domain=domain,
                    count=MAX_POSTS_BY_REQUEST, offset=data["offset"]
                )
                items = response.get("items", [])

                for post in items:
                    # закреплённый пост стоит первым независимо от даты и не должен остановить сбор
                    if post.get("is_pinned"):
                        continue

                    if post.get("date", 0) < self.start_timestamp:
                        data["finished"] = True
                        break

                    if post.get("id") not in known_ids:
                        known_ids.add(post.get("id"))
                        data["posts"].append(self.parse_post(post, domain))

                if not items:
                    data["finished"] = True

                data["offset"] += MAX_POSTS_BY_REQUEST
                self.save_progress(data, path)
                time.sleep(REQUEST_PAUSE)
        finally:
            self.save_progress(data, path)

        print(f"{domain}: собрано {len(data['posts'])} постов")

        return data


    def build_dataset(self):
        """Объединяет посты всех сообществ в одну таблицу"""

        frames = []

        for domain in self.groups:
            data = self.load_progress(RAW_DATA_DIR / f"{domain}.json")

            frame = pd.DataFrame(data["posts"])
            frame["members_count"] = data["members_count"]
            frame["collected_at"] = data["collected_at"]
            frames.append(frame)

        dataset = pd.concat(frames, ignore_index=True)
        dataset.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")

        print(f"Таблица успешно сохранена в файл: {DATASET_PATH}, постов: {len(dataset)}")

        return dataset


    def run(self):
        """Собирает посты всех сообществ и сохраняет общую таблицу"""

        print(f"Начало сбора постов, сообществ: {len(self.groups)} ...")

        for domain in self.groups:
            self.collect_group(domain)

        return self.build_dataset()


if __name__ == "__main__":

    VKPostsCollector().run()
