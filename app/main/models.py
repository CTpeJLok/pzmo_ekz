from django.db import models

from django.contrib.auth.models import User


class Article(models.Model):
    timestamp = models.DateTimeField(
        "Время",
    )

    contentId = models.BigIntegerField(
        "ID Контента",
        primary_key=True,
    )

    authorPersonId = models.BigIntegerField(
        "ID Автора",
    )

    authorSessionId = models.BigIntegerField(
        "ID Сессии",
    )

    authorUserAgent = models.TextField(
        "User Agent",
        null=True,
    )

    authorRegion = models.CharField(
        "Регион",
        max_length=10,
        null=True,
    )

    authorCountry = models.CharField(
        "Страна",
        max_length=50,
        null=True,
    )

    contentType = models.CharField(
        "Тип контента",
        max_length=100,
        null=True,
    )

    url = models.TextField(
        "URL",
        null=True,
    )

    title = models.TextField(
        "Название",
        null=True,
    )

    text = models.TextField(
        "Текст",
        null=True,
    )

    lang = models.CharField(
        "Язык",
        max_length=10,
        null=True,
    )

    def __str__(self) -> str:
        return f"{self.title}"

    class Meta:
        verbose_name = "Статья"
        verbose_name_plural = "Статьи"


class UserInteract(models.Model):
    timestamp = models.DateTimeField(
        "Время",
        auto_now_add=True,
    )

    eventType = models.CharField(
        "Тип события",
        max_length=100,
    )

    contentId = models.ForeignKey(
        to=Article,
        on_delete=models.CASCADE,
        related_name="interacts",
        verbose_name="Статья",
    )

    personId = models.ForeignKey(
        to=User,
        on_delete=models.CASCADE,
        related_name="interacts",
        verbose_name="Пользователь",
    )

    sessionId = models.BigIntegerField(
        "ID Сессии",
    )

    class Meta:
        verbose_name = "Взаимодействие"
        verbose_name_plural = "Взаимодействия"
