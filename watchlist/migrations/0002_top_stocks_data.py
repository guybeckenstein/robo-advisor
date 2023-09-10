from service.util import helpers
from watchlist.models import TopStock

from django.db import migrations, transaction


class Migration(migrations.Migration):
    dependencies = [
        ('watchlist', '0001_initial'),
    ]

    def generate_watchlist_data(apps, schema_editor):
        test_data: list = helpers.get_sectors_names_list()
        with transaction.atomic():
            for i, sector_name in enumerate(test_data):
                team_member = TopStock(
                    id=i + 1,
                    sector_name=sector_name,
                    sector_as_variable=sector_name.lower().replace(' ', '-')
                )
                team_member.save()
            team_member = TopStock(
                id=i + 1,
                sector_name='All',
            )
            team_member.save()

    operations = [
        migrations.RunPython(generate_watchlist_data),
    ]
