from django.contrib import admin

from .models import Article, UserInteract


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = (
        "timestamp",
        "title",
        "contentId",
        "authorPersonId",
        "authorSessionId",
        "authorRegion",
        "authorCountry",
        "contentType",
    )
    search_fields = (
        "title",
        "contentId",
        "authorPersonId",
        "authorSessionId",
        "authorUserAgent",
        "authorRegion",
        "authorCountry",
        "contentType",
        "url",
    )
    list_filter = (
        "timestamp",
        "authorRegion",
        "authorCountry",
        "contentType",
    )
    ordering = (
        "timestamp",
        "authorPersonId",
        "authorSessionId",
        "authorUserAgent",
        "authorRegion",
        "authorCountry",
        "contentType",
    )


@admin.register(UserInteract)
class UserInteractAdmin(admin.ModelAdmin):
    list_display = (
        "timestamp",
        "eventType",
        "contentId",
        "personId",
        "sessionId",
    )
    search_fields = (
        "eventType",
        "contentId",
        "personId",
        "sessionId",
    )
    list_filter = (
        "timestamp",
        "eventType",
        "personId",
    )
