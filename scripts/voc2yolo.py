#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
import shutil
import yaml
import random
from pathlib import Path

VOC_ROOT = Path("../data/VOCdevkit/VOC2007")
OUT_ROOT = Path("../data/yolo_dataset")
TRAIN_RATIO = 0.8

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W, H = int(size.find("width").text), int(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bb = obj.find("bndbox")
        xmin, ymin = int(bb.find("xmin").text), int(bb.find("ymin").text)
        xmax, ymax = int(bb.find("xmax").text), int(bb.find("ymax").text)
        cx, cy = (xmin + xmax) / 2 / W, (ymin + ymax) / 2 / H
        w, h = (xmax - xmin) / W, (ymax - ymin) / H
        objs.append((name, cx, cy, w, h))
    return objs

def convert():
    img_dir = VOC_ROOT / "JPEGImages"
    ann_dir = VOC_ROOT / "Annotations"
    imgs = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg",".jpeg",".png"]])
    random.seed(42)
    random.shuffle(imgs)
    split = int(len(imgs) * TRAIN_RATIO)
    train_imgs, val_imgs = imgs[:split], imgs[split:]
    all_classes = set()
    for img in imgs:
        xml = ann_dir / (img.stem + ".xml")
        if not xml.exists(): continue
        for name, *_ in parse_xml(xml): all_classes.add(name)
    class_list = sorted(list(all_classes))
    class_map = {n: i for i, n in enumerate(class_list)}
    print(f"Found {len(class_list)} classes: {class_list}")
    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img in split_imgs:
            shutil.copy(img, OUT_ROOT / "images" / split_name / img.name)
            xml = ann_dir / (img.stem + ".xml")
            if not xml.exists(): continue
            lines = []
            for name, cx, cy, w, h in parse_xml(xml):
                if name not in class_map: continue
                lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if lines:
                (OUT_ROOT / "labels" / split_name / (img.stem + ".txt")).write_text("\n".join(lines))
    (OUT_ROOT / "data.yaml").write_text(yaml.dump({
        "path": str(OUT_ROOT.resolve()),
        "train": "images/train", "val": "images/val",
        "names": class_list
    }, default_flow_style=False, sort_keys=False))
    print(f"Conversion done! Train:{len(train_imgs)} | Val:{len(val_imgs)}")

if __name__ == "__main__": convert()
