# Generated by Django 5.1.5 on 2025-01-19 16:06

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Article',
            fields=[
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('contentId', models.IntegerField(primary_key=True, serialize=False, verbose_name='Content ID')),
                ('authorPersonId', models.IntegerField(verbose_name='Author Person ID')),
                ('authorSessionId', models.IntegerField(verbose_name='Author Session ID')),
                ('authorUserAgent', models.TextField(verbose_name='Author User Agent')),
                ('authorRegion', models.CharField(max_length=10, verbose_name='Author Region')),
                ('authorCountry', models.CharField(max_length=50, verbose_name='Author Country')),
                ('contentType', models.CharField(max_length=100, verbose_name='Content Type')),
                ('url', models.URLField(verbose_name='URL')),
                ('title', models.TextField(verbose_name='Title')),
                ('text', models.TextField(verbose_name='Text')),
                ('lang', models.CharField(max_length=10, verbose_name='Language')),
            ],
            options={
                'verbose_name': 'Статья',
                'verbose_name_plural': 'Статьи',
            },
        ),
    ]
