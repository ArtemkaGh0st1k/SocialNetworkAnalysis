from final_project.analysis import PostsAnalyzer
from final_project.collect_posts import VKPostsCollector
from final_project.explain_model import ModelExplainer
from final_project.preprocess import PostsPreprocessor
from final_project.train_model import EngagementModelTrainer


def run_final_project_pipeline():

    # 1. Сбор постов (уже собранные сообщества повторно не запрашиваются)

    VKPostsCollector().run()

    # 2. Предобработка: очистка, целевая переменная, признаки

    PostsPreprocessor().run()

    # 3. Исследовательский анализ

    PostsAnalyzer().run()

    # 4. Обучение и сравнение моделей

    EngagementModelTrainer().run()

    # 5. Объяснение моделей через SHAP

    ModelExplainer().run()

    # Дашборд запускается отдельно: python -m final_project.dashboard


if __name__ == "__main__":

    run_final_project_pipeline()
