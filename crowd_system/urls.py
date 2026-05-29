from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

admin.site.site_header = '公共场所人群密度估计与异常聚集检测系统'
admin.site.site_title = '人群密度系统后台'
admin.site.index_title = '系统管理后台'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # 注册api路由
    path('', include('webui.urls')),    # 注册前端webui路由
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)