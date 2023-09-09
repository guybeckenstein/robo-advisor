from core.models import TeamMember

from django.db import migrations, transaction


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0001_initial'),
    ]

    @staticmethod
    def generate_team_members_data():
        test_data: list[tuple[str, str]] = [
            ('Guy Beckenstein', 'guybeckenstein',),
            ('Yarden Agami', 'yardet',),
            ('Hagai Levy', 'hagailevy',),
            ('Yarden Gazit', 'Yardengz',),
        ]
        with transaction.atomic():
            for team_member_full_name, github_username in test_data:
                team_member: TeamMember = TeamMember(
                    alt="".join(team_member_full_name.split()),
                    full_name=team_member_full_name,
                    github_username=github_username,
                    img=f'img/{"".join(team_member_full_name.split())}.jpg',
                )
                team_member.save()

    operations = [
        migrations.RunPython(generate_team_members_data),
    ]
