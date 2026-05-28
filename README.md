# YOLOv11 + DeepSORT 人物追踪

## 快速开始
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare VOC dataset
mkdir -p data/VOCdevkit
# Extract VOC2007/2012 to data/VOCdevkit/

# 3. Convert dataset
cd scripts && python voc2yolo.py && cd ..

# 4. Train detection model
cd src && python train.py && cd ..

# 5. Run video tracking
cd src && python track.py --source ../test_video.mp4
```

数据集信息：
目标检测：VOC 数据集（自动转换为 YOLO 格式）将vocal数据集放在data目录下 地址：https://aistudio.baidu.com/datasetdetail/134811

目标跟踪：DeepSORT，使用在 Market-1501 上预训练的 Re-ID 模型

测试视频：将 test_video.mp4 放置在项目根目录下 视频链接：https://github.com/yusheng-ll/Object-Detection-and-Recognition/releases/tag/v1

模型与演示
实验voc数据集训练的YOLO模型
 [下载训练权重 best.pt](https://github.com/yusheng-ll/Object-Detection-and-Recognition/releases/tag/v1.0)
 [测试结果视频](同上链接)

通过voc2yolo.py将voc数据集自动转换为yolo格式。
