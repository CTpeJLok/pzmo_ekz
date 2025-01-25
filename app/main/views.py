import pandas as pd
from datetime import datetime
from random import randint

from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.contrib import messages

from core.settings import BASE_DIR

from .models import Article, UserInteract

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

EVENTS = {
    "v": "VIEW",
    "l": "LIKE",
    "b": "BOOKMARK",
    "f": "FOLLOW",
    "c": "COMMENT CREATED",
}

EVENT_WEIGHT = {
    "VIEW": 1.0,
    "LIKE": 2.0,
    "BOOKMARK": 2.5,
    "FOLLOW": 3.0,
    "COMMENT CREATED": 4.0,
}

article_embeddings = np.load('article_embeddings.npy')

with open('article_metadata.pkl', 'rb') as meta_file:
    article_metadata = pickle.load(meta_file)

with open('bert_model_params.pkl', 'rb') as params_file:
    bert_model_params = pickle.load(params_file)

# Загружаем BERT-модель
model = SentenceTransformer(bert_model_params['model_name'])
embedding_dim = bert_model_params['embedding_dim']


def build_user_profile_on_the_fly(user_interactions, article_metadata, article_embeddings):
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
    user_interactions = interactions_df[interactions_df['personId'] == user_id]
    user_interactions_count = len(user_interactions)

    print(f"User {user_id} has >=5 interactions. Generating recommendations using BERT.")
    user_profile = build_user_profile_on_the_fly(
        user_interactions[['contentId', 'eventStrength']].values,
        article_metadata,
        article_embeddings
    )
    recommendations = recommend_items(user_profile, article_embeddings, article_metadata, topn=topn)
    return recommendations


def generate_session_id():
    id = randint(0, 9223372036854775807)
    return "{:018d}".format(id)


@login_required(login_url="login")
def home(request):
    # articles = Article.objects.prefetch_related("interacts").order_by("-timestamp")[
    #     :250
    # ]

    articles = Article.objects.annotate(i_c=Count('interacts')).order_by('-i_c', '-timestamp')[:250]


    response = render(
        request,
        "main/home.html",
        {"articles": articles, "user": request.user},
    )

    if not request.COOKIES.get("session_id"):
        response.set_cookie(key="session_id", value=generate_session_id())

    return response


@login_required(login_url='login')
def new_session(request):
    articles = Article.objects.annotate(i_c=Count('interacts')).order_by('-i_c', '-timestamp')[:250]

    response = render(
        request,
        "main/home.html",
        {"articles": articles, "user": request.user},
    )
    
    response.set_cookie(key="session_id", value=generate_session_id())
    return response


@login_required(login_url="login")
def article(request, article_id):
    article = Article.objects.get(contentId=article_id)
    UserInteract.objects.create(
        eventType="VIEW",
        contentId=article,
        personId=request.user,
        sessionId=request.COOKIES["session_id"],
    )

    response = render(
        request,
        "main/home.html",
        {
            "article": article,
            "user": request.user,
        },
    )

    response.set_cookie(key="session_id", value=request.COOKIES["session_id"])

    return response


@login_required(login_url="login")
def article_action(request, article_id, action):
    article = Article.objects.get(contentId=article_id)

    if action in ["l", "b", "f"]:
        interaction = UserInteract.objects.filter(
            personId=request.user, contentId=article, eventType=EVENTS[action]
        ).first()
        if interaction:
            interaction.delete()
            return redirect("home")

    UserInteract.objects.create(
        eventType=EVENTS[action],
        contentId=article,
        personId=request.user,
        sessionId=request.COOKIES["session_id"],
    )

    return redirect("home")


@login_required(login_url="login")
def my_articles(request):
    interactions_df = UserInteract.objects.filter(personId=request.user).order_by('contentId')
    if interactions_df.count() < 5:
        messages.add_message(request, messages.INFO, 'Вы новый пользователь, поэтому мы не можем составить для вас рекомендации')
        return redirect('home')

    file_path_articles = "shared_articles.csv"
    articles_df = pd.read_csv(file_path_articles)
    articles_df = articles_df[articles_df["eventType"] == "CONTENT SHARED"]
    articles_df.drop(
        columns=["authorUserAgent", "authorRegion", "authorCountry"], inplace=True
    )

    interactions_df = list(interactions_df.values())
    interactions_df = pd.DataFrame(interactions_df)
    interactions_df.drop(columns=["id"], inplace=True)
    interactions_df["timestamp"] = (
        pd.to_datetime(interactions_df["timestamp"]).astype(int) // 10**9
    )
    interactions_df = interactions_df.rename(
        columns={
            "contentId_id": "contentId",
            "personId_id": "personId",
        }
    )

    print(interactions_df.head(100))

    # Веса взаимодействий
    event_type_strength = {
        "VIEW": 1.0,
        "LIKE": 2.0,
        "BOOKMARK": 2.5,
        "FOLLOW": 3.0,
        "COMMENT CREATED": 4.0,
    }
    interactions_df["eventStrength"] = interactions_df["eventType"].apply(
        lambda x: event_type_strength[x]
    )
    interactions_df = interactions_df.groupby(
        ["personId", "contentId"], as_index=False
    ).agg({"eventStrength": "sum"})

    print(interactions_df.head(100))
  
    recommended_articles = get_recommendations(
        request.user.id,
        interactions_df,
        topn=10,
    )
    
    article_ids = recommended_articles["contentId"].tolist()
    articles = Article.objects.filter(contentId__in=article_ids)

    return render(request, "main/home.html", {"user": request.user, 'articles': articles, 'is_rec': True})


@login_required(login_url="login")
def load_data(request):
    df = pd.read_csv(BASE_DIR / "shared_articles.csv")
    df = df.map(lambda x: None if pd.isna(x) else x)

    Article.objects.all().delete()

    objs = []
    for _, row in df.iterrows():
        if row["eventType"] != "CONTENT SHARED":
            continue

        objs.append(
            Article(
                timestamp=str(datetime.fromtimestamp(row["timestamp"])),
                contentId=row["contentId"],
                authorPersonId=row["authorPersonId"],
                authorSessionId=row["authorSessionId"],
                authorUserAgent=row["authorUserAgent"],
                authorRegion=row["authorRegion"],
                authorCountry=row["authorCountry"],
                contentType=row["contentType"],
                url=row["url"],
                title=row["title"],
                text=row["text"],
                lang=row["lang"],
            )
        )

    Article.objects.bulk_create(objs)

    return render(request, "main/home.html")
