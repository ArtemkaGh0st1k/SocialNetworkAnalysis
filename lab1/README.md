### Архитектура и принцип работы 

1. Авторизация VK API
2. Цикл запросов `wall.get` с шагом 100 постов
- Пост свежее 31.08.2026? --> Игнорируем
- Пост в прделах? --> Сохраняем данные
- Пост старее 01.08.2026? --> Прерываем сбор
3. Сохранение массива в `CSV` или `JSON`

**Логика работы:**

1. Подготовка и конвертация дат в Timestamp
ВКонтакте хранит время публикации постов в формате Unix Timestamp (количество секунд, прошедших с 1 января 1970 года).

``` python
START_DATE = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)

START_TIMESTAMP = int(START_DATE.timestamp())
END_TIMESTAMP = int(END_DATE.timestamp())
```

Перевод дат в Timestamp позволяет сравнивать время публикации обычными операциями сравнения чисел (< и >), что работает моментально.

2. Инициализация и подключение к AP

``` python
vk_session = vk_api.VkApi(token=ACCESS_TOKEN)
vk = vk_session.get_api()
```

- vk_session создаёт обёртку над сетевыми запросами с вашим токеном.

- vk = vk_session.get_api() превращает методы VK API в обычные функции Python. Например, метод VK API wall.get становится доступен как vk.wall.get().

3. Механизм постраничной навигации (Пагинация)
VK API возвращает максимум 100 постов за один запрос. Чтобы прочитать всю стену, используется пагинация через сдвиг `offset`:

```python

offset = 0
count = 100

while True:
    response = vk.wall.get(domain=domain, count=count, offset=offset)
    items = response.get("items", [])
    ...
    offset += count
```

При первом запросе (offset=0) VK возвращает посты 1–100.
При втором запросе (offset=100) VK возвращает посты 101–200.
Цикл продолжается, пока не будут обработаны все записи.

4. Фильтрация постов по временному диапазону
Посты на стене VK всегда отсортированы от новых к старым. Это позволяет эффективнее фильтровать данные:

```python
for post in items:
    post_date = post.get("date", 0)

    # 1. Пост опубликован ПОСЛЕ августа 2026
    if post_date > end_ts:
        continue  # Пропускаем, ждем более старые посты

    # 2. Пост опубликован ДО августа 2026
    if post_date < start_ts:
        reached_older_posts = True
        break  # Выходим из цикла, так как далее идут только еще более старые посты
```

5. Извлечение и защита от отсутствующих полей
Структура ответа VK API представляет собой вложенные словари. В разных постах могут отсутствовать просмотры, вложения или текст. Чтобы скрипт не упал с ошибкой `KeyError`, используется метод .`get()` со значениями по умолчанию:

```python
post_info = {
    "id": post.get("id"),
    "date": date_str,
    "text": post.get("text", "").replace("\n", " "),
    "likes": post.get("likes", {}).get("count", 0),
    "comments": post.get("comments", {}).get("count", 0),
    "reposts": post.get("reposts", {}).get("count", 0),
    "views": post.get("views", {}).get("count", 0),
    "link": f"https://vk.com/wall{post.get('owner_id')}_{post.get('id')}",
}
```

6. Сохранение результатов
- В CSV `save_to_csv`: Используется `csv.DictWriter` с кодировкой utf-8-sig. Спецификация sig добавляет UTF-8 BOM-заголовок, что позволяет файлу корректно открываться в Excel.
- В JSON `save_to_json`: Используется `json.dump` с параметром `ensure_ascii=False`, чтобы кириллические символы сохранялись в читаемом виде, а не в кодировке \u0430....