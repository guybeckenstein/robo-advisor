import datetime

import pytz
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import QuerySet

from accounts.models import CustomUser
from core.api import serializers
from core.models import TeamMember, QuestionnaireA


class TeamMemberListView(generics.ListAPIView):
    queryset = TeamMember.objects.all()
    serializer_class = serializers.TeamMemberSerializer


class CreateQuestionnaireAView(APIView):
    serializer_class = serializers.CreateQuestionnaireASerializer

    def post(self, request, format=None):
        if not self.request.session.exists(self.request.session.session_key):
            self.request.session.create()

        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            user_session_key = self.request.session.session_key
            ml_answer = serializer.data.get('ml_answer')
            model_answer = serializer.data.get('model_answer')
            user: CustomUser = CustomUser.objects.filter(session_key=user_session_key).first()
            questionnaire_queryset: QuerySet[QuestionnaireA] = QuestionnaireA.objects.filter(user=user)
            if questionnaire_queryset.exists():
                questionnaire: QuestionnaireA = questionnaire_queryset.first()
                questionnaire.ml_answer = ml_answer
                questionnaire.model_answer = model_answer
                questionnaire.date = datetime.datetime.now(tz=pytz.timezone('Asia/Jerusalem'))
                questionnaire.save(update_fields=['ml_answer', 'model_answer', 'date'])
            else:
                questionnaire: QuestionnaireA = QuestionnaireA(
                    user=user, ml_answer=ml_answer, model_answer=model_answer
                )
                questionnaire.save()

        return Response(serializers.QuestionnaireASerializer(questionnaire).data, status=status.HTTP_200_OK)
