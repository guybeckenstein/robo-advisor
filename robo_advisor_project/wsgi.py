import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'robo_advisor_project.settings')
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'robo_advisor_project.production')

application = get_wsgi_application()
