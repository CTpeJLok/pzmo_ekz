from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("new-session", views.new_session, name="new_session"),
    path("my-bert", views.my_articles_bert, name="my_articles_bert"),
    path("my-tf-idf", views.my_articles_tf_idf, name="my_articles_tf_idf"),
    path("my-word2vec", views.my_articles_word2vec, name="my_articles_word2vec"),
    path("load", views.load_data, name="load"),
    path(
        "action/<str:article_id>/<str:action>",
        views.article_action,
        name="article_action",
    ),
    path("article/<str:article_id>", views.article, name="article"),
]
