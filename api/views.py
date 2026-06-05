# api/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import time
from PIL import Image
import os
from django.conf import settings
from ultralytics import YOLO
import numpy as np
import json
import requests
from .models import DetectionRecord, ChatRecord
import cv2
import tempfile

# ====================== 全局配置 ======================
# 🔑 场景面积估算（平方米）- 根据实际摄像头覆盖范围调整
SCENE_AREA_SQM = 100.0

# 上传目录
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔑 加载 best1.pt 模型
try:
    model = YOLO("best1.pt")
    print(f"✅ YOLO 模型加载成功: best1.pt")
    print(f"📦 支持类别: {list(model.names.values())[:10]}...")
except Exception as e:
    print(f"⚠️ 模型加载失败: {e}")
    model = None

# 阿里云百炼配置
ALI_API_KEY = "sk-d7586de1dc7e4034905569d1f209e3b4"
ALI_MODEL = "qwen-turbo"
ALI_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


# ====================== 🎯 图片检测接口 ======================
@csrf_exempt
def predict_crowd(request):
    """目标检测与密度统计 API"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            img_file = request.FILES['image']
            img = Image.open(img_file).convert('RGB')
            w, h = img.size

            if model is None:
                return JsonResponse({"code": 500, "error": "模型未加载"}, status=500)

            # 🔍 YOLO 推理
            results = model(img, verbose=False)
            boxes = results[0].boxes
            total_count = len(boxes)

            # 📊 类别统计
            class_stats = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                class_stats[cls_name] = class_stats.get(cls_name, 0) + 1

            # 📏 真实密度计算
            density = round(total_count / SCENE_AREA_SQM, 2)

            # ⚠️ 拥挤判定
            is_abnormal = (density > 3.0) or (total_count > 40)

            # 💾 保存带框结果图
            res_img = results[0].plot()
            res_img = Image.fromarray(res_img[..., ::-1])
            detected_filename = f"detected_{img_file.name}"
            detected_img_path = os.path.join(UPLOAD_DIR, detected_filename)
            res_img.save(detected_img_path)

            # 🤖 生成 AI 分析
            ai_analysis = generate_ai_analysis(total_count, density, class_stats)

            # 🗄️ 保存到数据库
            DetectionRecord.objects.create(
                filename=img_file.name,
                file_path=f"/uploads/{detected_filename}",
                total_count=total_count,  # ✅ 使用新字段
                density_value=density,  # ✅ 使用新字段
                class_data=class_stats,
                is_abnormal=is_abnormal
            )
            print(f"✅ 检测完成: {total_count}个目标, 密度{density}个/㎡")

            return JsonResponse({
                "code": 200,
                "total_count": total_count,
                "density": density,
                "is_abnormal": is_abnormal,
                "class_stats": class_stats,
                "detected_img_url": f"/uploads/{detected_filename}",
                "ai_analysis": ai_analysis
            })

        except Exception as e:
            import traceback
            print(f"❌ 检测错误: {e}\n{traceback.format_exc()}")
            return JsonResponse({"code": 500, "error": str(e)}, status=500)

    return JsonResponse({"code": 400, "error": "请上传图片"}, status=400)


# ====================== 🤖 AI 分析生成函数 ======================
def generate_ai_analysis(count, density, class_stats):
    """根据检测结果生成专业建议"""
    top_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join([f"{cls}({cnt})" for cls, cnt in top_classes]) if top_classes else "无"

    if density > 4.0:
        risk = "高风险"
        suggestions = [
            f"⚠️ 密度过高({density}个/㎡)，建议立即疏导或限流",
            "开启备用出口，增派工作人员引导",
            "启动应急预案，避免踩踏风险"
        ]
    elif density > 2.5:
        risk = "中风险"
        suggestions = [
            "密度接近警戒值，建议加强现场引导",
            "优化空间布局，避免局部聚集",
            "关注高峰期人流变化趋势"
        ]
    else:
        risk = "低风险"
        suggestions = [
            "当前目标分布均匀，系统运行正常",
            "建议保持定期巡检",
            "可考虑优化空间利用率"
        ]

    return {
        "risk_level": risk,
        "summary": f"共检测 {count} 个目标，密度 {density} 个/㎡。主要类别：{top_str}。",
        "suggestions": suggestions,
        "density": density
    }


# ====================== 🗑️ 清空历史记录 ======================
@csrf_exempt
def clear_records(request):
    if request.method == "POST":
        try:
            count = DetectionRecord.objects.count()
            DetectionRecord.objects.all().delete()
            return JsonResponse({"code": 200, "msg": f"已清空 {count} 条记录"})
        except Exception as e:
            return JsonResponse({"code": 500, "error": str(e)}, status=500)
    return JsonResponse({"code": 400, "msg": "仅支持 POST 请求"}, status=400)


# ====================== 💬 AI 智能对话 ======================
@csrf_exempt
def ai_chat(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_msg = data.get("message", "").strip()
            if not user_msg:
                return JsonResponse({"answer": "请输入问题"})

            # 获取最新检测记录作为上下文
            latest = DetectionRecord.objects.order_by("-create_time").first()

            system_prompt = """你是目标检测与识别系统专属AI助手，专业解答目标识别、密度分析、安全预警问题。
当前系统规则：密度>3个/㎡ 或 总目标数>40 判定为异常。请用简洁专业的中文回答。"""

            if latest:
                # ✅ 修复：使用正确的字段名 total_count 和 density_value
                system_prompt += f"\n【最新检测】总数:{latest.total_count}, 密度:{latest.density_value}个/㎡, 异常:{'是' if latest.is_abnormal else '否'}"

            headers = {"Authorization": f"Bearer {ALI_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": ALI_MODEL,
                "input": {"messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ]},
                "parameters": {"temperature": 0.7}
            }

            resp = requests.post(ALI_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                return JsonResponse({"answer": f"AI服务异常: {resp.status_code}"})

            res_json = resp.json()
            answer = res_json.get("output", {}).get("text", "AI未返回内容").strip()

            # 保存对话记录
            ChatRecord.objects.create(message=user_msg, reply=answer)

            return JsonResponse({"answer": answer})

        except Exception as e:
            import traceback
            print(f"❌ AI对话错误: {e}\n{traceback.format_exc()}")
            return JsonResponse({"answer": f"系统异常: {str(e)}"})
    return JsonResponse({"answer": "仅支持POST请求"}, status=400)


@csrf_exempt
def get_chat_history(request):
    records = ChatRecord.objects.all().order_by('-create_time')[:20]
    data = [{"user_msg": r.message, "ai_reply": r.reply, "time": r.create_time.strftime("%Y-%m-%d %H:%M:%S")} for r in
            records]
    data.reverse()
    return JsonResponse({"list": data})


@csrf_exempt
def clear_chat_history(request):
    if request.method == "POST":
        try:
            count = ChatRecord.objects.count()
            ChatRecord.objects.all().delete()
            return JsonResponse({"code": 200, "msg": f"已清空 {count} 条记录"})
        except Exception as e:
            return JsonResponse({"code": 500, "error": str(e)}, status=500)
    return JsonResponse({"code": 400, "msg": "请求错误"}, status=400)


# ====================== 📹 实时帧检测 ======================
@csrf_exempt
def realtime_detect(request):
    if request.method == 'POST' and request.FILES.get('frame'):
        try:
            frame_file = request.FILES['frame']
            img = Image.open(frame_file).convert('RGB').resize((480, 320))

            if model is None:
                return JsonResponse({'count': 0, 'density': 0, 'abnormal': False, 'boxes': []}, status=500)

            results = model(img, verbose=False)
            count = len(results[0].boxes)
            box_list = [box.xyxy[0].cpu().numpy().tolist() for box in results[0].boxes]

            density = round(count / SCENE_AREA_SQM, 2)
            abnormal = (count > 20) or (density > 3.0)

            return JsonResponse({
                'count': count,
                'density': density,
                'abnormal': abnormal,
                'boxes': box_list,
                'img_w': 480,
                'img_h': 320
            })
        except Exception as e:
            return JsonResponse({'count': 0, 'density': 0, 'abnormal': False, 'boxes': [], 'error': str(e)}, status=500)
    return JsonResponse({'count': 0, 'density': 0, 'abnormal': False, 'boxes': []}, status=400)


# ====================== 🎬 视频检测 ======================
@csrf_exempt
def video_detect(request):
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            video_file = request.FILES['video']
            print(f"📥 收到视频文件: {video_file.name}")

            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=UPLOAD_DIR)
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            temp_video.close()
            temp_path = temp_video.name

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                os.unlink(temp_path)
                return JsonResponse({"code": 400, "error": "无法读取视频文件"}, status=400)

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            out_name = f"detect_{os.path.basename(video_file.name).replace('.mp4', '')}_{int(time.time())}.mp4"
            out_path = os.path.join(UPLOAD_DIR, out_name)

            # 尝试多种编码器
            # ✅ 修复：直接使用 mp4v 编码器，避免 libopenh264 版本冲突
            # mp4v 是 OpenCV 最基础支持的编码，虽然浏览器兼容性稍差，但一定能生成文件
            # ✅ 尝试多种编码器，优先使用 H.264 (avc1)，如果不行则回退到 mp4v
            fourcc_codes = ['avc1', 'H264', 'mp4v', 'XVID']
            fourcc = None
            out = None

            for code in fourcc_codes:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*code)
                    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        print(f"✅ 成功使用编码器: {code}")
                        break
                    else:
                        out.release()
                        out = None
                except Exception as e:
                    continue

            if not out or not out.isOpened():
                raise Exception("无法创建输出视频文件，请检查 OpenCV 是否支持 H.264/mp4v 编码")

            if not out or not out.isOpened():
                raise Exception("无法创建输出视频文件")

            # 🔑 数据收集（确保在循环前初始化）
            people_list, density_list, timeline = [], [], []
            all_class_stats = {}
            sample_interval = max(1, int(fps * 0.5))
            frame_idx = 0
            processed_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                w, h = img.size

                if model:
                    results = model(img, verbose=False)

                    # 1. 过滤边缘框
                    valid_boxes = []
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        if (x1 > w * 0.03 and x2 < w * 0.97 and
                                y1 > h * 0.03 and y2 < h * 0.97):
                            valid_boxes.append(box)

                    valid_count = len(valid_boxes)

                    # 2. 绘制检测框（每一帧都画）
                    for box in valid_boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                        cls_name = model.names.get(int(box.cls[0]), "obj")
                        cv2.putText(frame, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)

                    out.write(frame)
                    processed_frames += 1

                    # 3. 【关键】只在采样帧里统计类别和记录数据
                    if frame_idx % sample_interval == 0:
                        density = round(valid_count / SCENE_AREA_SQM, 2)
                        timestamp = frame_idx / fps

                        people_list.append(valid_count)
                        density_list.append(density)
                        timeline.append({
                            'time': round(timestamp, 2),
                            'count': valid_count,
                            'density': density
                        })

                        # ✅ 只在采样时累加类别统计
                        for box in valid_boxes:
                            cls_id = int(box.cls[0])
                            cls_name = model.names.get(cls_id, f"class_{cls_id}")
                            all_class_stats[cls_name] = all_class_stats.get(cls_name, 0) + 1

                frame_idx += 1
            # ... (后面的代码保持不变)
            cap.release()
            out.release()
            if os.path.exists(temp_path):
                os.unlink(temp_path)

            if not os.path.exists(out_path):
                raise Exception(f"输出视频文件未生成: {out_path}")

            avg_people = int(np.mean(people_list)) if people_list else 0
            avg_density = float(np.mean(density_list)) if density_list else 0.0
            is_abnormal = (avg_density > 3.0) or (avg_people > 40)
            video_url = f"/uploads/{out_name}"

            sampled_frames = len(timeline)
            avg_class_stats = {
                cls: round(count / sampled_frames, 2)
                for cls, count in all_class_stats.items()
            } if sampled_frames > 0 else {}

            ai_analysis = generate_ai_analysis(avg_people, avg_density, all_class_stats)

            DetectionRecord.objects.create(
                filename=video_file.name,
                file_path=video_url,
                total_count=avg_people,
                density_value=round(avg_density, 2),
                class_data=all_class_stats,
                is_abnormal=is_abnormal
            )

            return JsonResponse({
                "code": 200,
                "avg_people": avg_people,
                "avg_density": round(avg_density, 2),
                "is_crowd": is_abnormal,
                "video_url": video_url,
                "sample_count": len(timeline),
                "processed_frames": processed_frames,
                "video_info": {"duration": round(duration, 2), "fps": round(fps, 2)},
                "class_stats_total": all_class_stats,
                "class_stats_avg": avg_class_stats,
                "total_detections": sum(all_class_stats.values()),
                "ai_analysis": ai_analysis,
                "density_timeline": timeline
            })

        except Exception as e:
            import traceback
            print(f"❌ 视频检测错误: {e}\n{traceback.format_exc()}")
            return JsonResponse({"code": 500, "error": str(e)}, status=500)

    return JsonResponse({"code": 400, "error": "请上传视频文件"}, status=400)


# ====================== 🧠 独立 AI 分析接口 ======================
@csrf_exempt
def ai_analysis(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            count = data.get("count", 0)
            density = data.get("density", 0)
            is_abnormal = data.get("is_abnormal", False)
            class_stats = data.get("class_stats", {})

            top_classes = ", ".join(
                [f"{k}({v})" for k, v in sorted(class_stats.items(), key=lambda x: x[1], reverse=True)[:3]])
            prompt = f"""你是目标检测安全专家。
检测数据：总数{count}个，密度{density}个/㎡，主要类别:{top_classes or '无'}，异常:{'是' if is_abnormal else '否'}。
请用专业简洁的中文（≤100字）回答：1.当前状态 2.风险等级 3.管理建议。"""

            headers = {"Authorization": f"Bearer {ALI_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": ALI_MODEL,
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {"temperature": 0.5}
            }
            resp = requests.post(ALI_URL, headers=headers, json=payload, timeout=20)
            res_json = resp.json()
            answer = res_json.get("output", {}).get("text", "分析失败").strip()

            return JsonResponse({"analysis": answer})
        except Exception as e:
            return JsonResponse({"analysis": f"AI服务异常: {str(e)}"})
    return JsonResponse({"analysis": "仅支持POST请求"}, status=400)


# ====================== 📋 获取历史记录接口 ======================
@csrf_exempt
def get_records(request):
    """获取检测历史记录（支持分页）"""
    try:
        from django.core.paginator import Paginator

        page = int(request.GET.get('page', 1))
        limit = int(request.GET.get('limit', 10))

        records = DetectionRecord.objects.all().order_by('-create_time')
        paginator = Paginator(records, limit)
        page_obj = paginator.get_page(page)

        data = []
        for record in page_obj.object_list:
            data.append({
                'id': record.id,
                'filename': record.filename,
                'file_path': record.file_path,
                'total_count': record.total_count,
                'density_value': record.density_value,
                'class_data': record.class_data,
                'is_abnormal': record.is_abnormal,
                'create_time': record.create_time.strftime('%Y-%m-%d %H:%M:%S')
            })

        return JsonResponse({
            'success': True,
            'total': paginator.count,
            'page': page_obj.number,
            'limit': limit,
            'data': data
        })
    except Exception as e:
        import traceback
        print(f"❌ 获取记录失败: {e}\n{traceback.format_exc()}")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)