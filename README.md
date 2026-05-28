# YOLOv11 + DeepSORT Person Tracking

## Quick Start
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

## Dataset Info
- **Detection**: VOC dataset (auto-converted to YOLO format)
- **Tracking**: DeepSORT with pre-trained Re-ID model on Market-1501
- **Test Video**: Place test_video.mp4 in project root
