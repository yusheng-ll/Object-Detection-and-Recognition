import torch
import cv2
import numpy as np
import os
from django.conf import settings
from .crowd_model import CSRNet

# ======================================
# 加载 人群密度模型
# ======================================
model = CSRNet()
model_path = os.path.join(settings.BASE_DIR, "best.pt")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
else:
    print("⚠️  未找到预训练权重，使用随机初始化模型")

model.eval()

# 图片预处理
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.unsqueeze(0)

def detect_crowd(file_path):
    # 1. 读取图片
    img = cv2.imread(file_path)
    if img is None:
        raise Exception("无法读取图片")

    h, w = img.shape[:2]

    # 2. 预处理
    img_tensor = preprocess(img)

    # 3. 真实预测！
    with torch.no_grad():
        density_map = model(img_tensor)
        count = int(density_map.sum().item())  # 真实人数

    # 4. 密度等级
    if count < 20:
        level = "低密度"
        abnormal = False
    elif count < 50:
        level = "中密度"
        abnormal = False
    else:
        level = "高密度（异常聚集）"
        abnormal = True

    # 5. 绘制结果
    cv2.putText(img, f"COUNT: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    # 6. 保存结果图
    fname = os.path.basename(file_path)
    save_path = os.path.join(settings.MEDIA_ROOT, f"result_{fname}")
    cv2.imwrite(save_path, img)

    return count, level, abnormal, save_path