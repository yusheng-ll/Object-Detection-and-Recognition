#!/usr/bin/env python3
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/yolo_dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print("Starting training...")
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project="../yolo/detect", name="voc_person",
        exist_ok=True, patience=15, close_mosaic=10, amp=True,
        lr0=0.01, weight_decay=0.0005, warmup_epochs=3.0,
        box=7.5, cls=0.5, dfl=1.5
    )
    print("Training done! Weights: yolo/detect/voc_person/weights/best.pt")

if __name__ == "__main__": main()
