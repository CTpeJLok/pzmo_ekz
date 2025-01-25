import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_user_profile_on_the_fly(user_interactions, article_metadata, article_embeddings, embedding_dim):
    """
    Создает профиль пользователя на основе взаимодействий.
    """
    # Проверяем, пуст ли массив взаимодействий
    if user_interactions.size == 0:
        return np.zeros(embedding_dim)

    user_embeddings = []
    strengths = []
    for content_id, strength in user_interactions:
        try:
            idx = article_metadata[article_metadata['contentId'] == content_id].index[0]
            user_embeddings.append(article_embeddings[idx])
            strengths.append(strength)
        except IndexError:
            continue  # Пропускаем, если contentId не найден

    user_embeddings = np.array(user_embeddings)
    strengths = np.array(strengths).reshape(-1, 1)

    weighted_sum = np.sum(user_embeddings * strengths, axis=0)
    return weighted_sum / np.sum(strengths)


def recommend_items(user_profile, article_embeddings, article_metadata, topn=10):
    """
    Рекомендует статьи на основе косинусного сходства.
    """
    cosine_similarities = cosine_similarity(user_profile.reshape(1, -1), article_embeddings).flatten()
    top_indices = cosine_similarities.argsort()[-topn:][::-1]
    recommendations = article_metadata.iloc[top_indices].copy()  # Добавляем .copy() для создания новой копии DataFrame
    recommendations.loc[:, 'similarity'] = cosine_similarities[top_indices]
    return recommendations


def get_recommendations(user_id, interactions_df, topn=10):
    """
    Рекомендации для пользователя:
    <5 взаимодействий -> популярные статьи.
    >=5 взаимодействий -> рекомендации через BERT.
    """
    article_embeddings = np.load('article_embeddings.npy')

    with open('article_metadata.pkl', 'rb') as meta_file:
        article_metadata = pickle.load(meta_file)

    with open('bert_model_params.pkl', 'rb') as params_file:
        bert_model_params = pickle.load(params_file)

    # Загружаем BERT-модель
    embedding_dim = bert_model_params['embedding_dim']

    user_profile = build_user_profile_on_the_fly(
        interactions_df[['contentId', 'eventStrength']].values,
        article_metadata,
        article_embeddings,
        embedding_dim,
    )
    return recommend_items(user_profile, article_embeddings, article_metadata, topn=topn)
