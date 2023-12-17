from core.models import QuestionnaireA, QuestionnaireB, TeamMember
from rest_framework import serializers


class QuestionnaireASerializer(serializers.ModelSerializer):
    class Meta:
        model = QuestionnaireA
        fields = '__all__'


class CreateQuestionnaireASerializer(serializers.ModelSerializer):
    class Meta:
        model = QuestionnaireA
        fields = ('ml_answer', 'model_answer',)


class QuestionnaireBSerializer(serializers.ModelSerializer):
    class Meta:
        model = QuestionnaireB
        fields = '__all__'


class CreateQuestionnaireBSerializer(serializers.ModelSerializer):
    class Meta:
        model = QuestionnaireB
        fields = ('answer_1', 'answer_2', 'answer_3',)

class TeamMemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = TeamMember
        fields = '__all__'
