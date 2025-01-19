import pandas as pd
from datetime import datetime
from random import randint
import pickle

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from core.settings import BASE_DIR

from .models import Article, UserInteract


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

# with open(BASE_DIR / "fm_recommender_model.pkl", "rb") as f:
#     fm_recommender_model = pickle.load(f)

# with open(BASE_DIR / "cf_recommender_model2.pkl", "rb") as f:
#     cf_recommender_model = pickle.load(f)


def generate_session_id():
    id = randint(0, 9999999999999999999)
    return "{:019d}".format(id)


@login_required(login_url="login")
def home(request):
    articles = Article.objects.prefetch_related("interacts").order_by("-timestamp")[
        :250
    ]

    response = render(
        request,
        "main/home.html",
        {"articles": articles, "user": request.user},
    )

    response.set_cookie(key="session_id", value=generate_session_id())

    return response


@login_required(login_url="login")
def article_action(request, article_id, action):
    UserInteract.objects.create(
        eventType=EVENTS[action],
        contentId=Article.objects.get(pk=article_id),
        personId=request.user,
        sessionId=request.COOKIES["session_id"],
    )

    articles = Article.objects.prefetch_related("interacts").order_by("-timestamp")[:50]

    return render(
        request, "main/home.html", {"articles": articles, "user": request.user}
    )


@login_required(login_url="login")
def my_articles(request):

    return render(request, "main/home.html", {"user": request.user})


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
