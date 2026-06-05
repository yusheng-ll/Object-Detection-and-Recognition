# api/urls.py
from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    # ✅ 图片检测路由
    path('detect/image/', views.predict_crowd, name='detect_image'),

    # ✅ 视频检测路由（注意路径）
    path('detect/video/', views.video_detect, name='video_detect'),
    # 或者如果你的前端请求的是这个：
    path('video_detect/', views.video_detect, name='video_detect_old'),

    # 其他路由
    path('records/', views.get_records, name='get_records'),
    path('records/delete/', views.clear_records, name='clear_records'),
    path('ai_chat/', views.ai_chat, name='ai_chat'),
    path('get_chat_history/', views.get_chat_history, name='get_chat_history'),
    path('clear_chat_history/', views.clear_chat_history, name='clear_chat_history'),
    path('realtime_detect/', views.realtime_detect, name='realtime_detect'),
]