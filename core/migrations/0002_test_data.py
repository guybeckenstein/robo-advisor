from core.models import TeamMember

from django.db import migrations, transaction


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0001_initial'),
    ]

    def generate_team_members_data(apps, schema_editor):
        test_data = ['Guy Beckenstein', 'Yarden Agami', 'Yarden Gazit', 'Hagai Levy']
        with transaction.atomic():
            for team_member_full_name in test_data:
                team_member = TeamMember(
                    alt="".join(team_member_full_name.split()),
                    full_name=team_member_full_name,
                    img=f'img/{"".join(team_member_full_name.split())}.jpg',
                )
                team_member.save()

    operations = [
        migrations.RunPython(generate_team_members_data),
    ]
