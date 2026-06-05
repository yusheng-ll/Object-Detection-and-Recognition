from django.urls import path
from django.contrib.auth.views import LogoutView
from . import views

urlpatterns = [
    path('', views.index, name='webui_index'),
    path('detect/', views.detect_page, name='webui_detect'),
    path('record/', views.record_page, name='webui_record'),
    path('agent/', views.agent_page, name='webui_agent'),
    path('realtime/', views.realtime_page, name='webui_realtime'),
    path('video/', views.video_page, name='webui_video'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('logout/', LogoutView.as_view(), name='logout'),
]