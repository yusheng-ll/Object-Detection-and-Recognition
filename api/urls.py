# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_crowd, name='predict_crowd'),
    path('clear-records/', views.clear_records, name='clear_records'),
    path('ai_chat/', views.ai_chat, name='ai_chat'),
    path('get_chat_history/', views.get_chat_history, name='get_chat_history'),
    path('clear_chat_history/', views.clear_chat_history, name='clear_chat_history'),
    path('realtime_detect/', views.realtime_detect, name='realtime_detect'),
    path('video_detect/', views.video_detect, name='video_detect'),
    path('ai_analysis/', views.ai_analysis, name='ai_analysis'),
]