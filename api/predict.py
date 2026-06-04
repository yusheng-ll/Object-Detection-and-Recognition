import torch
import cv2
import numpy as np
import os
from django.conf import settings
from .crowd_model import CSRNet


# 1. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ℹ️  人群密度模型运行设备: {device}")

# 2. 加载 人群密度模型
model = CSRNet().to(device)
model_path = os.path.join(settings.BASE_DIR, "best.pt")

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 成功加载预训练权重 best.pt")
    except Exception as e:
        print(f"⚠️  权重加载失败: {e}，使用随机初始化模型")
else:
    print("⚠️  未找到预训练权重 best.pt，使用随机初始化模型")

model.eval()

# 3. 图片预处理
def preprocess(img, target_size=(640, 480)):
    """
    统一预处理：
    - 转RGB
    - 归一化
    - 转Tensor
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)  # 统一尺寸，提升稳定性
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.unsqueeze(0).to(device)

# 4. 人群密度检测主函数
def detect_crowd(file_path):
    """
    输入图片路径，返回人数、密度等级、异常状态、结果图路径
    """
    # 1. 读取图片
    img = cv2.imread(file_path)
    if img is None:
        raise Exception("❌ 无法读取图片，请检查路径是否正确")

    # 2. 预处理
    img_tensor = preprocess(img)

    # 3. 模型预测（关闭梯度，加速推理）
    with torch.no_grad():
        density_map = model(img_tensor)
        count = int(density_map.sum().item())  # 真实人数

    # 4. 密度等级判定
    if count < 20:
        level = "低密度"
        abnormal = False
    elif count < 50:
        level = "中密度"
        abnormal = False
    else:
        level = "高密度（异常聚集）"
        abnormal = True

    # 5. 绘制结果（标注人数）
    cv2.putText(
        img,
        f"COUNT: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3
    )

    # 6. 保存结果图
    fname = os.path.basename(file_path)
    save_path = os.path.join(settings.MEDIA_ROOT, f"result_{fname}")
    cv2.imwrite(save_path, img)

    return count, level, abnormal, save_path
