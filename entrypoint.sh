#!/bin/bash

# Migrate database
python manage.py migrate  2>&1

# Fire up server
exec python manage.py runserver 0.0.0.0:8000