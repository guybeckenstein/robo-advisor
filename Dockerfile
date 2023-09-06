# pull official base image
FROM python:3.11.4-slim-buster

# Install netcat
RUN apt-get update && apt-get install -y netcat && apt-get install -y locales

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app

# Create a directory for fonts
RUN mkdir -p /usr/src/app/fonts

# Copy the 'fonts' directory from your project into the container
COPY static/fonts/ /usr/src/app/static/fonts/

# copy entrypoint.sh
COPY ./entrypoint.sh /usr/src/app/entrypoint.sh
RUN sed -i 's/\r$//g' /usr/src/app/entrypoint.sh
RUN chmod a+x /usr/src/app/entrypoint.sh

# run entrypoint.sh
# ENTRYPOINT ["/usr/src/app/entrypoint.sh"]

RUN python manage.py flush --no-input
RUN python manage.py makemigrations
RUN python manage.py migrate