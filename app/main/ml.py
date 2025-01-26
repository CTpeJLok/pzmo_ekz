import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

article_embeddings = np.load("article_embeddings.npy")
with open("article_metadata.pkl", "rb") as meta_file:
    article_metadata = pickle.load(meta_file)

with open("bert_model_params.pkl", "rb") as params_file:
    bert_model_params = pickle.load(params_file)

valid_content_ids = article_metadata.iloc[: len(article_embeddings)]["contentId"]
article_metadata = article_metadata[
    article_metadata["contentId"].isin(valid_content_ids)
].reset_index(drop=True)
filtered_indices = article_metadata.index.to_numpy()
article_embeddings = article_embeddings[filtered_indices]

embedding_dim = bert_model_params["embedding_dim"]

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
tfidf_matrix = sp.load_npz("tfidf_matrix.npz")
with open("item_ids.pkl", "rb") as f:
    item_ids = pickle.load(f)

bert_content_ids = set(article_metadata["contentId"])
tfidf_content_ids = set(item_ids)
common_content_ids = bert_content_ids.intersection(tfidf_content_ids)
print(
    f"До согласования: BERT={len(bert_content_ids)}, TF-IDF={len(tfidf_content_ids)}."
)
print(f"Общее количество статей в пересечении: {len(common_content_ids)}")

article_metadata = article_metadata[
    article_metadata["contentId"].isin(common_content_ids)
].reset_index(drop=True)
cid_to_idx = {cid: i for i, cid in enumerate(item_ids)}
new_indices = [cid_to_idx[cid] for cid in article_metadata["contentId"]]
tfidf_matrix = tfidf_matrix[new_indices, :]
item_ids_filtered = article_metadata["contentId"].tolist()
item_ids = item_ids_filtered
print(
    f"После согласования: article_metadata={len(article_metadata)},",
    f"tfidf_matrix.shape={tfidf_matrix.shape}, item_ids={len(item_ids)}",
)


def build_user_profile_on_the_fly(user_interactions):
    if user_interactions.size == 0:
        return np.zeros(embedding_dim)
    user_embeddings = []
    strengths = []
    for content_id, strength in user_interactions:
        idx = article_metadata[article_metadata["contentId"] == content_id].index
        if not idx.empty:
            user_embeddings.append(article_embeddings[idx[0]])
            strengths.append(strength)
    if not user_embeddings:
        return np.zeros(embedding_dim)
    user_embeddings = np.array(user_embeddings)
    strengths = np.array(strengths).reshape(-1, 1)
    weighted_sum = np.sum(user_embeddings * strengths, axis=0)
    return weighted_sum / np.sum(strengths)


def recommend_items_bert(user_profile, topn=10):
    cosim = cosine_similarity(user_profile.reshape(1, -1), article_embeddings).flatten()
    top_idx = cosim.argsort()[-topn:][::-1]
    recs = article_metadata.iloc[top_idx].copy()
    recs["similarity"] = cosim[top_idx]
    return recs[["contentId", "similarity"]]


def get_recommendations_bert(user_id, interactions_df, topn=10):
    user_interactions = interactions_df[interactions_df["personId"] == user_id]
    if len(user_interactions) < 5:
        print("Мало взаимодействий (BERT). Возвращаем популярные.")
        popular_articles = pd.read_csv("popularity_top_50.csv").head(topn)
        popular_articles = popular_articles[["contentId"]].copy()
        popular_articles["similarity"] = 0.0
        return popular_articles
    user_profile = build_user_profile_on_the_fly(
        user_interactions[["contentId", "eventStrength"]].values
    )
    return recommend_items_bert(user_profile, topn=topn)


def build_user_profile_on_the_fly_tfidf(user_interactions_df):
    if user_interactions_df.empty:
        print("Нет взаимодействий (TF-IDF).")
        return np.zeros((1, tfidf_matrix.shape[1]))

    item_id_to_idx = {cid: i for i, cid in enumerate(item_ids)}
    profiles = []
    strengths = []
    for content_id, strength in zip(
        user_interactions_df["contentId"], user_interactions_df["eventStrength"]
    ):
        if content_id in item_id_to_idx:
            idx = item_id_to_idx[content_id]
            profiles.append(tfidf_matrix[idx])
            strengths.append(strength)

    if not profiles:
        print("profiles пуст — ни одна статья не найдена (TF-IDF).")
        return np.zeros((1, tfidf_matrix.shape[1]))

    stacked = sp.vstack(profiles)
    strengths = np.array(strengths).reshape(-1, 1)
    weighted = stacked.multiply(strengths)
    sum_profile = weighted.sum(axis=0)
    sum_profile = sp.csr_matrix(sum_profile)
    norm_value = np.sqrt(sum_profile.multiply(sum_profile).sum())
    print("[TF-IDF DEBUG] Сумма профиля (норма) =", norm_value)

    if strengths.sum() == 0:
        return np.zeros((1, tfidf_matrix.shape[1]))
    profile_norm = sum_profile.multiply(1.0 / strengths.sum())
    return profile_norm


def recommend_items_tfidf(user_profile, topn=10):
    cosim = cosine_similarity(tfidf_matrix, user_profile).flatten()
    top_idx = cosim.argsort()[-topn:][::-1]
    rec_ids = [item_ids[i] for i in top_idx]
    rec_df = pd.DataFrame({"contentId": rec_ids, "similarity": cosim[top_idx]})
    return rec_df[["contentId", "similarity"]]


def get_recommendations_tfidf(user_id, interactions_df, topn=10):
    user_interactions = interactions_df[interactions_df["personId"] == user_id]
    if len(user_interactions) < 5:
        print("Мало взаимодействий (TF-IDF). Возвращаем популярные.")
        popular_articles = pd.read_csv("popularity_top_50.csv").head(topn)
        popular_articles = popular_articles[["contentId"]].copy()
        popular_articles["similarity"] = 0.0
        return popular_articles
    profile = build_user_profile_on_the_fly_tfidf(
        user_interactions[["contentId", "eventStrength"]]
    )
    return recommend_items_tfidf(profile, topn=topn)


article_embeddings_w2v = np.load("article_embeddings_w2v.npy")
with open("item_ids_w2v.pkl", "rb") as f:
    item_ids_w2v = pickle.load(f)

# Предположим, articles_w2v_df.pkl мы тоже загружаем, если нужно
articles_w2v_df = pd.read_pickle("articles_w2v_df.pkl")
# или, при желании, вы можете не загружать, если article_metadata
# итак совпадает. Главное согласовать их.

# Создадим словарь contentId -> индекс
w2v_id_to_idx = {cid: i for i, cid in enumerate(item_ids_w2v)}
w2v_vector_size = article_embeddings_w2v.shape[1]  # 100


def build_user_profile_on_the_fly_word2vec(user_interactions_df):
    """
    Аналог BERT, но для Word2Vec. Берём средневзвешенные вектора статей.
    """
    if user_interactions_df.empty:
        print("Нет взаимодействий (Word2Vec).")
        return np.zeros(w2v_vector_size)

    user_embeddings = []
    strengths = []
    for content_id, strength in zip(
        user_interactions_df["contentId"], user_interactions_df["eventStrength"]
    ):
        if content_id in w2v_id_to_idx:
            idx = w2v_id_to_idx[content_id]
            emb = article_embeddings_w2v[idx]  # (vector_size,)
            user_embeddings.append(emb)
            strengths.append(strength)

    if not user_embeddings:
        print("profiles пуст (Word2Vec) - нет подходящих статей.")
        return np.zeros(w2v_vector_size)

    user_embeddings = np.array(user_embeddings)
    strengths = np.array(strengths).reshape(-1, 1)
    weighted_sum = np.sum(user_embeddings * strengths, axis=0)
    return weighted_sum / np.sum(strengths)


def recommend_items_word2vec(user_profile, topn=10):
    """
    Рекомендуем статьи, используя article_embeddings_w2v.
    user_profile: shape (vector_size,) - профайл пользователя
    """
    cosim = cosine_similarity(
        user_profile.reshape(1, -1), article_embeddings_w2v
    ).flatten()
    top_idx = cosim.argsort()[-topn:][::-1]
    rec_ids = [item_ids_w2v[i] for i in top_idx]  # восстанавливаем contentId
    rec_df = pd.DataFrame({"contentId": rec_ids, "similarity": cosim[top_idx]})
    return rec_df[["contentId", "similarity"]]


def get_recommendations_word2vec(user_id, interactions_df, topn=10):
    user_interactions = interactions_df[interactions_df["personId"] == user_id]
    if len(user_interactions) < 5:
        print("Мало взаимодействий (Word2Vec). Возвращаем популярные.")
        popular_articles = pd.read_csv("popularity_top_50.csv").head(topn)
        popular_articles = popular_articles[["contentId"]].copy()
        popular_articles["similarity"] = 0.0
        return popular_articles

    user_profile = build_user_profile_on_the_fly_word2vec(
        user_interactions[["contentId", "eventStrength"]]
    )
    return recommend_items_word2vec(user_profile, topn=topn)


def show_top_tokens_for_user_profile_tf(user_profile, tfidf_vectorizer, top_n=20):
    """
    user_profile: (1, n_features) sparse или (n_features,) dense
    tfidf_vectorizer: ваш TfidfVectorizer
    Возвращаем DataFrame с колонками [token, relevance].
    """
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Приводим профиль к плотному виду (1D numpy array)
    if hasattr(user_profile, "toarray"):
        profile_array = user_profile.toarray().flatten()
    else:
        profile_array = user_profile.flatten()  # если уже ndarray

    tokens_with_weights = list(zip(feature_names, profile_array))
    tokens_with_weights.sort(key=lambda x: x[1], reverse=True)
    top_tokens = tokens_with_weights[:top_n]

    return pd.DataFrame(top_tokens, columns=["token", "relevance"])
