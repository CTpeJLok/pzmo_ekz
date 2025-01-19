from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("my", views.my_articles, name="my_articles"),
    path("load", views.load_data, name="load"),
    path("<str:article_id>/<str:action>", views.article_action, name="article_action"),
]
