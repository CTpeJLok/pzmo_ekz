from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("new-session", views.new_session, name="new_session"),
    path("my", views.my_articles, name="my_articles"),
    path("load", views.load_data, name="load"),
    path("action/<str:article_id>/<str:action>", views.article_action, name="article_action"),
    path("article/<str:article_id>", views.article, name="article"),
]
