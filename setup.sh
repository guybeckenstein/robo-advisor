##!/bin/bash -ex
# The -e option would make our script exit with an error if any command
# fails while the -x option makes verbosely it output what it does

# Install Pipenv, the -n option makes sudo fail instead of asking for a
# password if we don't have sufficient privileges to run it
sudo -n dnf install -y pipenv

cd /vagrant
# Install dependencies with Pipenv
pipenv sync --dev
pipenv install django-crispy-forms
pipenv install django-bootstrap4
pipenv install django-phonenumber-field[phonenumberslite]
pipenv install django-phonenumber-field[phonenumbers]
pipenv install pillow

# run our app. setsid, the parentheses and "&" are used to perform a "double
# fork" so that our app stays up after the setup script finishes.
# The app logs are redirected to the 'runserver.log' file.
(setsid pipenv run \
	python manage.py runserver 0.0.0.0:8080 > runserver.log 2>&1 &)
