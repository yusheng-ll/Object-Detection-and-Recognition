# api/admin.py
from django.contrib import admin
from .models import DetectionRecord, ChatRecord  # ✅ 正确的类名

@admin.register(DetectionRecord)
class DetectionRecordAdmin(admin.ModelAdmin):
    list_display = ['id', 'filename', 'total_count', 'density_value', 'is_abnormal', 'create_time']
    list_filter = ['is_abnormal', 'create_time']
    search_fields = ['filename', 'class_data']
    ordering = ['-create_time']
    readonly_fields = ['create_time']

@admin.register(ChatRecord)
class ChatRecordAdmin(admin.ModelAdmin):
    list_display = ['id', 'message', 'reply', 'create_time']
    search_fields = ['message', 'reply']
    ordering = ['-create_time']