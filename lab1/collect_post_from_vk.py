import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import json
import csv
import vk_api
from vk_api import VkApi
from vk_api.execute import VkApiMethod 


class VKParser():

    def __init__(self):

        load_dotenv()

        self.token = os.getenv("API_TOKEN")
        if not self.token:
            raise ValueError("API_TOKEN не найден в переменных окружении!")
    
        self.group_domain = "samara_university"
        self.start_date = datetime(2026, 8, 1, 0, 0 , 0, tzinfo=timezone.utc)
        self.end_date = datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)

        self.start_timestamp = int(self.start_date.timestamp())
        self.end_timestamp = int(self.end_date.timestamp())


    def get_all_posts(self, vk : VkApiMethod, domain : str = None, start_ts = None, end_ts = None):
        """Собирает посты стены сообщества в заданном временном диапазоне"""

        if not domain:
            domain = self.group_domain
        if not start_ts:
            start_ts = self.start_timestamp
        if not end_ts:
            end_ts = self.end_timestamp
        
        posts_data = []
        offset = 0
        max_posts_by_request = 100  # макс. кол-во постов за один запрос

        print("Начало сбора постов ...")

        while True:
            # нужно отрубать VPN для корректного запроса !!!
            response = vk.wall.get(domain=domain, count=max_posts_by_request, offset=offset)

            items = response.get("items", [])
            if not items:
                break

            reached_older_posts = False

            for post in items:
                post_date = post.get("date", 0)

                # если пост новее нужного диапазона, пропускаем его
                if post_date > end_ts:
                    continue

                # если пост старее нужного останавливаем сбор
                if post_date < start_ts:
                    reached_older_posts = True
                    break

                date_str = datetime.fromtimestamp\
                    (
                        post_date, 
                        tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S")

                # типы вложений (картинка, видео и тп)
                attachments = [
                    att.get("type") for att in post.get("attachments", [])
                ]

                post_info = \
                {
                    "id" : post.get("id"),
                    "date" : date_str,
                    "timestamp" : post_date,
                    "text" : post.get("text", "").replace("\n", " "),
                    "likes" : post.get("likes", {}).get("count", 0),
                    "comments" : post.get("comments", {}).get("count", 0),
                    "reposts" : post.get("reposts", {}).get("count", 0),
                    "views" : post.get("views", {}).get("count", 0),
                    "link" : f"https://vk.com/wall{post.get('owner_id')}_{post.get('id')}",
                    "attachments" : ", ".join(attachments) if attachments else "None"
                }

                posts_data.append(post_info)

            if reached_older_posts:
                break

            offset += max_posts_by_request

            print(f"Обработано постов: {offset} ...")

        print(
            f"Сбор постов завершен! Всего найдено {len(posts_data)}"
        )

        return posts_data


    def save_to_csv(self, data,  filename="samara_university_august_2026.csv"):
        """Сохраняет список постов в CSV-файл"""

        if not data:
            print("Нет данных для сохранения в CSV")

        fields = \
        [
            "id",
            "date",
            "text",
            "likes",
            "comments",
            "reposts",
            "views",
            "link",
            "attachments",
        ]

        full_path = f"lab1/results/{filename}"
        with open(full_path, mode='w', encoding='utf-8-sig', newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=fields, extrasaction='ignore'
            )
            writer.writeheader()
            writer.writerows(data)

        print(f"Данные успешно сохранены в файл: {full_path}")


    def save_to_json(self, data, filename="samara_university_august_2026.json"):
        """Сохраняет список постов в JSON-файл."""

        if not data:
            print("Нет данных для сохранения в JSON.")
            return
        
        full_path = f"lab1/results/{filename}" 
        with open(full_path, mode="w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Данные успешно сохранены в файл: {filename}")