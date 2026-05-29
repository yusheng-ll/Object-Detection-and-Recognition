from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import os
from django.conf import settings
from ultralytics import YOLO
import numpy as np
import json
import requests
from .models import DetectRecord
import cv2
import tempfile
from . import views

# 加载模型
model = YOLO("best.pt")

UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


@csrf_exempt
def predict_crowd(request):
    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        img = Image.open(img_file).convert('RGB')
        w, h = img.size
        img_area = w * h

        # YOLO 推理
        results = model(img)
        boxes = results[0].boxes
        count = len(boxes)

        # ====================== 【针对小目标优化的密度算法】 ======================
        if count == 0:
            density_score = 0.0
            abnormal = False
        else:
            # 1. 过滤边缘误检
            valid_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                if (x1 > w * 0.03 and x2 < w * 0.97 and
                        y1 > h * 0.03 and y2 < h * 0.97):
                    valid_boxes.append(box)

            valid_count = len(valid_boxes) 
            if valid_count == 0:
                valid_count = count

            # 2. 核心优化：按“人头数量”计算密度，专门适配小目标
            base_area = 1000000  # 100万像素基准
            people_per_million = (valid_count / img_area) * base_area

            density_score = round(people_per_million * 1.2, 1)
            density_score = min(density_score, 100)

            # 4. 拥挤判断
            abnormal = (valid_count > 40) or (density_score > 70)

        # 保存原始图片
        original_img_path = os.path.join(UPLOAD_DIR, img_file.name)
        img.save(original_img_path)

        # 保存带框图片
        res_img = results[0].plot()
        res_img = Image.fromarray(res_img[..., ::-1])
        detected_img_path = os.path.join(UPLOAD_DIR, "detected_" + img_file.name)
        res_img.save(detected_img_path)

        # ====================== ✅ 【保存记录到 MySQL】 ======================
        DetectRecord.objects.create(
            filename=img_file.name,
            file_path=f"/uploads/detected_{img_file.name}",
            crowd_count=count,
            density_level=str(density_score),
            is_abnormal=abnormal
        )

        return JsonResponse({
            "code": 200,
            "count": count,
            "density": density_score,
            "abnormal": abnormal,
            "detected_img_url": f"/uploads/detected_{img_file.name}"
        })

    return JsonResponse({"code": 400, "error": "Please upload image"}, status=400)


# ====================== 【清空所有历史记录接口】 ======================
@csrf_exempt
def clear_records(request):
    if request.method == "POST":
        # 删除数据库所有记录
        DetectRecord.objects.all().delete()
        return JsonResponse({"code": 200, "msg": "清空成功"})
    return JsonResponse({"code": 400, "msg": "请求方法错误"})


# ====================== 阿里云百炼配置 只改这里 ======================
ALI_API_KEY = "sk-d7586de1dc7e4034905569d1f209e3b4"
ALI_MODEL = "qwen-turbo"
ALI_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
# =================================================================

@csrf_exempt
def ai_chat(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_msg = data.get("message", "").strip()

            # 获取最新一条检测记录，给大模型做参考
            latest = DetectRecord.objects.order_by("-create_time").first()

            # 构造系统角色 + 带入最新检测数据
            system_prompt = """
你是智能人群密度监测系统专属AI助手，专业解答人群检测、拥挤判定、安全疏散、系统使用问题。
当前系统规则：人数超过40人 或 密度大于70分 判定为拥挤。
"""
            if latest:
                system_prompt += f"""
最新一次检测数据：
检测人数：{latest.crowd_count} 人
密度分值：{latest.density_level}
是否拥挤：{"是" if latest.is_abnormal else "否"}
请基于以上真实检测数据，专业、简洁回答用户问题，并可以给出疏散和管控建议。
"""

            headers = {
                "Authorization": f"Bearer {ALI_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": ALI_MODEL,
                "input": {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ]
                },
                "parameters": {"temperature": 0.7}
            }

            resp = requests.post(ALI_URL, headers=headers, json=payload, timeout=20)
            res_json = resp.json()
            answer = res_json["output"]["text"].strip()

            # ========== 关键：保存对话到数据库 ==========
            from .models import ChatRecord
            ChatRecord.objects.create(
                message=user_msg,
                reply=answer
            )

            return JsonResponse({"answer": answer})

        except Exception as e:
            return JsonResponse({"answer": f"AI服务异常：{str(e)}"})

    return JsonResponse({"answer": "仅支持POST请求"})

# 获取AI历史聊天记录
@csrf_exempt
def get_chat_history(request):
    from .models import ChatRecord
    records = ChatRecord.objects.all()[:20]  # 取最近20条
    data = []
    for item in records:
        data.append({
            "user_msg": item.message,
            "ai_reply": item.reply,
            "time": item.create_time.strftime("%Y-%m-%d %H:%M:%S")
        })
    # 倒序，最新在最后
    data.reverse()
    return JsonResponse({"list": data})

# 清空 AI 聊天记录
@csrf_exempt
def clear_chat_history(request):
    from .models import ChatRecord
    if request.method == "POST":
        ChatRecord.objects.all().delete()
        return JsonResponse({"code": 200, "msg": "清空成功"})
    return JsonResponse({"code": 400, "msg": "请求错误"})


@csrf_exempt
def realtime_detect(request):
    if request.method == 'POST' and request.FILES.get('frame'):
        frame_file = request.FILES['frame']
        img = Image.open(frame_file).convert('RGB')

        # 严格对齐前端 480*320
        target_w, target_h = 480, 320
        img = img.resize((target_w, target_h))

        results = model(img)
        boxes = results[0].boxes
        count = len(boxes)

        box_list = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_list.append([float(x1), float(y1), float(x2), float(y2)])

        # 密度计算
        img_area = target_w * target_h
        valid_count = len(box_list)

        base_area = 1000000
        density_score = round((valid_count / img_area) * base_area * 1.2, 1)
        density_score = min(density_score, 100)
        abnormal = (valid_count > 20) or (density_score > 70)

        return JsonResponse({
            'count': count,
            'density': density_score,
            'abnormal': abnormal,
            'boxes': box_list,
            'img_w': target_w,
            'img_h': target_h
        })
    # 异常兜底返回
    return JsonResponse({
        'count': 0,
        'density': 0,
        'abnormal': False,
        'boxes': [],
        'img_w': 480,
        'img_h': 320
    })

@csrf_exempt
def video_detect(request):
    if request.method == 'POST' and request.FILES.get('video'):
        print("收到视频检测请求")
        video_file = request.FILES['video']
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        for chunk in video_file.chunks():
            temp_video.write(chunk)
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        filename_only = os.path.basename(temp_video.name).rsplit('.', 1)[0]
        out_video_name = f"detect_{filename_only}.mp4"
        out_video_path = os.path.join(UPLOAD_DIR, out_video_name)
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        people_list = []
        density_list = []
        realtime_counts = []
        sample_interval = 3
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            w, h = img.size
            img_area = w * h

            results = model(img)
            boxes = results[0].boxes
            current_count = len(boxes)

            valid_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if (x1 > w * 0.03 and x2 < w * 0.97 and
                    y1 > h * 0.03 and y2 < h * 0.97):
                    valid_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            valid_count = len(valid_boxes) if valid_boxes else current_count

            for (x1, y1, x2, y2) in valid_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                cv2.putText(frame, "person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)

            out.write(frame)

            base_area = 1000000
            people_per_million = (valid_count / img_area) * base_area
            density = round(people_per_million * 1.2, 1)
            density = min(density, 100)

            people_list.append(valid_count)
            density_list.append(density)
            realtime_counts.append(valid_count)
            frame_idx += 1

        cap.release()
        out.release()
        os.unlink(temp_video.name)

        if not people_list:
            avg_people = 0
            avg_density = 0.0
        else:
            avg_people = int(np.mean(people_list))
            avg_density = float(np.mean(density_list))

        is_abnormal = (avg_people > 40) or (avg_density > 70)
        video_url = f"/uploads/{out_video_name}"

        DetectRecord.objects.create(
            filename=out_video_name,
            file_path=video_url,
            crowd_count=avg_people,
            density_level=str(round(avg_density, 1)),
            is_abnormal=is_abnormal,
            is_video=True
        )

        print(f"视频检测完成，平均人数：{avg_people}")
        return JsonResponse({
            "avg_people": avg_people,
            "avg_density": round(avg_density, 1),
            "is_crowd": is_abnormal,
            "video_url": video_url,
            "realtime_counts": realtime_counts
        })

    return JsonResponse({"code": 400, "error": "请上传视频文件"})

@csrf_exempt
def ai_analysis(request):
    if request.method == "POST":
        data = json.loads(request.body)
        count = data.get("count", 0)
        density = data.get("density", 0)
        is_abnormal = data.get("is_abnormal", False)

        # 传给大模型
        prompt = f"""
你是人群安全监测专家。

当前检测数据：
- 人数：{count} 人
- 密度评分：{density}
- 是否拥挤：{"是" if is_abnormal else "否"}

请用简洁、专业、安全的语言给出：
1. 当前人群状态
2. 拥挤风险等级
3. 管理/疏散建议

不要使用markdown，直接自然语言回答，控制在100字内。
"""

        # 调用你已有的通义千问
        headers = {
            "Authorization": f"Bearer {ALI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": ALI_MODEL,
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {"temperature": 0.5}
        }
        resp = requests.post(ALI_URL, headers=headers, json=payload)
        res_json = resp.json()
        answer = res_json["output"]["text"].strip()

        return JsonResponse({"analysis": answer})
    return JsonResponse({"analysis": "AI 服务异常"})