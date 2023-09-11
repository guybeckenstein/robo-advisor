import os

from allauth.socialaccount.models import SocialApp

from django.db import migrations, transaction

from django.contrib.sites.models import Site
from django.db.models import QuerySet


class Migration(migrations.Migration):
    dependencies = [
        ('accounts', '0001_initial'),
        ('socialaccount', '0001_initial'),
    ]

    def generate_site_data(apps, schema_editor):
        site_data: list[tuple] = [
            (
                f'http://{os.environ.get("WEB_IP", "localhost")}:8000/',
                'RoboAdvisor',
            ),
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
            (
                'facebook',
                'Facebook',
                os.environ.get("FACEBOOK_CLIENT_ID", ValueError),
                os.environ.get("FACEBOOK_CLIENT_SECRET", ValueError),
                '',
                'facebook',
            ),
            (
                'google',
                'Google',
                os.environ.get("GMAIL_CLIENT_ID", ValueError),
                os.environ.get("GMAIL_CLIENT_SECRET", ValueError),
                '',
                'google',
            ),
            (
                'github',
                'GitHub',
                os.environ.get("GITHUB_CLIENT_ID", ValueError),
                os.environ.get("GITHUB_CLIENT_SECRET", ValueError),
                '',
                'github',
            ),
        ]
        with transaction.atomic():
            for i, (provider, name, client_id, secret_key, key, provider_id) in enumerate(socialapp_data):
                social_app: SocialApp = SocialApp(
                    provider=provider,
                    name=name,
                    client_id=client_id,
                    secret=secret_key,
                    key=key,
                    provider_id=provider_id,
                )
                social_app.save()

                sites: QuerySet[Site] = Site.objects.all()
                for site in sites:
                    social_app.sites.add(site)

    operations = [
        migrations.RunPython(generate_site_data),
        migrations.RunPython(generate_socialapp_data),
    ]
