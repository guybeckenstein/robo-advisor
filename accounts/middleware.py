from django.contrib.sites.models import Site
from django.conf import settings


class DynamicSiteIDMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            current_site = Site.objects.get(domain=request.get_host())
            settings.SITE_ID = current_site.id
        except Site.DoesNotExist:
            settings.SITE_ID = 1  # Default to SITE_ID 1 if site doesn't exist

        response = self.get_response(request)
        return response
