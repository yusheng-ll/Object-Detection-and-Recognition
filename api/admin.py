from django.contrib import admin
from .models import DetectRecord
from django.contrib import admin
from .models import DetectRecord, ChatRecord

admin.site.register(DetectRecord)

admin.site.register(ChatRecord)