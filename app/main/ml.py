import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_user_profile_on_the_fly(
    user_interactions, article_metadata, article_embeddings, embedding_dim
):
    if user_interactions.size == 0:
        print("Нет взаимодействий у пользователя.")
        return np.zeros(embedding_dim)

    user_embeddings = []
    strengths = []

    for content_id, strength in user_interactions:
        idx = article_metadata[article_metadata["contentId"] == content_id].index
        if not idx.empty and idx[0] < len(article_embeddings):
            embedding = article_embeddings[idx[0]]
            user_embeddings.append(embedding)
            strengths.append(strength)
        else:
            continue

    if not user_embeddings:
        print("Не найдено ни одного валидного эмбеддинга.")
        return np.zeros(embedding_dim)

    user_embeddings = np.array(user_embeddings)
    strengths = np.array(strengths).reshape(-1, 1)
    weighted_sum = np.sum(user_embeddings * strengths, axis=0)
    return weighted_sum / np.sum(strengths)


def recommend_items(
    user_profile, article_embeddings, article_metadata, contentIds, topn=10
):
    """
    Рекомендует статьи на основе косинусного сходства.
    """
    cosine_similarities = cosine_similarity(
        user_profile.reshape(1, -1), article_embeddings
    ).flatten()

    # Убираем из предсказанных статей статьи, которые уже были взаимодействиями
    cosine_similarities = cosine_similarities[
        ~article_metadata["contentId"].isin(contentIds)
    ]

    top_indices = cosine_similarities.argsort()[-topn:][::-1]
    recommendations = article_metadata.iloc[top_indices].copy()
    recommendations.loc[:, "similarity"] = cosine_similarities[top_indices]
    return recommendations


def get_recommendations(interactions_df, topn=10):
    """
    Рекомендации для пользователя
    """
    article_embeddings = np.load("article_embeddings.npy")

    with open("article_metadata.pkl", "rb") as meta_file:
        article_metadata = pickle.load(meta_file)

    with open("bert_model_params.pkl", "rb") as params_file:
        bert_model_params = pickle.load(params_file)

    # Ограничиваем article_metadata длиной article_embeddings
    valid_content_ids = article_metadata.iloc[: len(article_embeddings)]["contentId"]
    # Оставляем только те записи, которые соответствуют корректным эмбеддингам
    article_metadata = article_metadata[
        article_metadata["contentId"].isin(valid_content_ids)
    ].reset_index(drop=True)

    # Пересоздаем article_embeddings
    filtered_indices = (
        article_metadata.index.to_numpy()
    )  # Теперь индексы будут последовательными
    article_embeddings = article_embeddings[filtered_indices]

    # Загружаем BERT-модель
    embedding_dim = bert_model_params["embedding_dim"]

    user_profile = build_user_profile_on_the_fly(
        interactions_df[["contentId", "eventStrength"]].values,
        article_metadata,
        article_embeddings,
        embedding_dim,
    )

    return recommend_items(
        user_profile,
        article_embeddings,
        article_metadata,
        contentIds=interactions_df["contentId"],
        topn=topn,
    )
