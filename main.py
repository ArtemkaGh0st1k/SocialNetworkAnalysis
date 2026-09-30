import vk_api
from lab1.collect_post_from_vk import VKParser


if __name__ == "__main__":

    vkParser = VKParser()

    vk_session = vk_api.VkApi(token=vkParser.token)
    vk = vk_session.get_api()

    posts = vkParser.get_all_posts(vk)

    vkParser.save_to_csv(posts)
    vkParser.save_to_json(posts)