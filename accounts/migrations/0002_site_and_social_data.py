import os
from pathlib import Path

import environ
from allauth.socialaccount.models import SocialApp
from django.db import migrations, transaction

from django.contrib.sites.models import Site
from django.db.models import QuerySet

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load and read .env file
# OS environment variables take precedence over variables from .env
env = environ.Env()
env.read_env(os.path.join(BASE_DIR, '../.env.oauth'))

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


class Migration(migrations.Migration):
    dependencies = [
        ('accounts', '0001_initial'),
        ('socialaccount', '0001_initial'),
    ]

    def generate_site_data(apps, schema_editor):
        site_data: list[tuple] = [
            ('http://127.0.0.1:8000/', 'RoboAdvisor',),
        ]
        with transaction.atomic():
            for domain, name in site_data:
                site: Site = Site(
                    domain=domain,
                    name=name,
                )
                site.save()

    def generate_socialapp_data(apps, schema_editor):
        socialapp_data: list[tuple] = [
            ('facebook', 'facebook', env("FACEBOOK_CLIENT_ID"), env("FACEBOOK_CLIENT_SECRET"),),
            ('google', 'google', env("GMAIL_CLIENT_ID"), env("GMAIL_CLIENT_SECRET"),),
            ('github', 'github', env("GITHUB_CLIENT_ID"), env("GITHUB_CLIENT_SECRET"),),
        ]
        with transaction.atomic():
            for i, (provider, name, client_id, secret_key) in enumerate(socialapp_data):
                social_app: SocialApp = SocialApp(
                    provider=provider,
                    name=name,
                    client_id=client_id,
                    secret=secret_key,
                )
                social_app.save()

                sites: QuerySet[Site] = Site.objects.all()
                for site in sites:
                    social_app.sites.add(site)

    operations = [
        migrations.RunPython(generate_site_data),
        migrations.RunPython(generate_socialapp_data),
    ]
