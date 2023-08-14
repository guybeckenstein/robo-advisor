from service.util import helpers
from watchlist.models import TopStock

from django.db import migrations, transaction


class Migration(migrations.Migration):
    dependencies = [
        ('watchlist', '0001_initial'),
    ]

    def generate_team_members_data(apps, schema_editor):
        test_data: list = helpers.get_sectors_names_list()
        with transaction.atomic():
            for i, sector_name in enumerate(test_data):
                team_member = TopStock(
                    id=i + 1,
                    sector_name=sector_name,
                    img_src='',
                    img_alt=f'{sector_name} Sector Image',
                )
                team_member.save()
            team_member = TopStock(
                id=i + 1,
                sector_name='All',
                img_src='',
                img_alt=f'All Sector Image',
            )
            team_member.save()

    operations = [
        migrations.RunPython(generate_team_members_data),
    ]
