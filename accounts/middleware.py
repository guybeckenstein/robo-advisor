from django.contrib.sites.models import Site
from django.conf import settings

"""
This file is mainly for fixing the potential issue when entering the login page (GET request), because of django-allauth
The value of SITE_ID matters, thus wrong value will make this error:
`django.contrib.sites.models.Site.DoesNotExist: Site matching query does not exist.`
Using this file (that is linked directly to `settings.py`), we fix the problem.
"""


class DynamicSiteIDMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            current_site = Site.objects.get(domain=request.get_host())
            settings.SITE_ID = current_site.id
        except Site.DoesNotExist:
            settings.SITE_ID = 1

        response = self.get_response(request)
        return response
