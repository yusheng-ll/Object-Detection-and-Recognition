# webui/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required

# 引入数据库模型 (注：根据之前的迁移记录，模型名应为 DetectionRecord)
from api.models import DetectionRecord


# ==================== 🔐 认证相关视图 ====================

def login_view(request):
    """登录页面"""
    # 已登录用户直接跳首页
    if request.user.is_authenticated:
        return redirect('/')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('/')  # ✅ 修复：使用绝对路径 '/' 避免 NoReverseMatch
        else:
            messages.error(request, '用户名或密码错误')

    return render(request, 'webui/login.html')


def register_view(request):
    """注册页面"""
    if request.user.is_authenticated:
        return redirect('/')

    if request.method == 'POST':
        username = request.POST.get('username')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if password1 != password2:
            messages.error(request, '两次密码输入不一致')
            return render(request, 'webui/register.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, '用户名已存在')
            return render(request, 'webui/register.html')

        user = User.objects.create_user(username=username, password=password1)
        user.save()
        login(request, user)  # 注册后自动登录
        return redirect('/')  # ✅ 修复：使用绝对路径 '/'

    return render(request, 'webui/register.html')


# ==================== 🛡️ 业务页面视图（全部加上登录保护） ====================

@login_required(login_url='/login/')
def index(request):
    """首页"""
    return render(request, 'webui/index.html')


@login_required(login_url='/login/')
def detect_page(request):
    """图片检测页面"""
    return render(request, 'webui/detect.html')


@login_required(login_url='/login/')
def video_page(request):
    """视频检测页面"""
    return render(request, 'webui/video.html')


@login_required(login_url='/login/')
def record_page(request):
    """历史记录页面"""
    records = DetectionRecord.objects.all().order_by('-create_time')
    return render(request, 'webui/record.html', {"records": records})


@login_required(login_url='/login/')
def agent_page(request):
    """AI智能体页面"""
    return render(request, 'webui/agent.html')


@login_required(login_url='/login/')
def realtime_page(request):
    """实时摄像头检测页面"""
    return render(request, 'webui/realtime.html')