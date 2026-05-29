import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-crowd-system-2025'
DEBUG = True
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']

LANGUAGE_CODE = 'zh-hans'  # 改成中文

TIME_ZONE = 'Asia/Shanghai'  # 改成中国时区
USE_TZ = True

# 关键：补全所有必须的应用
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'api',
    'webui',
]

# 关键：补全中间件
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'crowd_system.urls'

# 关键：补全模板配置
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'crowd_system.wsgi.application'

# 数据库配置
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'crowd_system',
        'USER': 'root',
        'PASSWORD': '***********',
        'HOST': '127.0.0.1',
        'PORT': '3306',
    }
}

# 修复模型主键警告
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# 上传文件配置
MEDIA_ROOT = os.path.join(BASE_DIR, 'uploads')
MEDIA_URL = '/uploads/'

# 模型路径
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

STATIC_URL = '/static/'
