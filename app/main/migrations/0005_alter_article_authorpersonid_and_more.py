# Generated by Django 5.1.5 on 2025-01-19 16:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_alter_article_contentid'),
    ]

    operations = [
        migrations.AlterField(
            model_name='article',
            name='authorPersonId',
            field=models.BigIntegerField(verbose_name='ID Автора'),
        ),
        migrations.AlterField(
            model_name='article',
            name='authorSessionId',
            field=models.BigIntegerField(verbose_name='ID Сессии'),
        ),
    ]
