#!/bin/sh

echo "Waiting for Postgresql..."

while ! nc -z $POSTGRES_HOST $PGPORT; do
  sleep 0.1
done

echo "PostgreSQL started"

python manage.py makemigrations

exec "$@"
