import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import random


def get_color(track_id):
    """为每个 ID 生成固定且不重复的颜色（防视觉疲劳）"""
    random.seed(int(track_id) * 137)  # 质数种子保证颜色分布均匀
    return tuple(random.randint(50, 255) for _ in range(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="../test_video1.mp4")
    parser.add_argument("--model", type=str, default=r"../runs/detect/voc_person/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    args = parser.parse_args()

    print(f" 加载模型: {args.model}")
    model = YOLO(args.model)

    # ✅ 优化1：使用 stream=True 流式推理，避免长视频内存溢出，且跟踪状态更连续
    print("🎬 启动 ByteTrack 跟踪... (按 q 退出)")
    results = model.track(
        source=args.source,
        tracker="bytetrack.yaml",  # 内置跟踪器，无需手动放 yaml 文件
        conf=args.conf,
        persist=True,
        stream=True,  # 关键：逐帧生成，不占内存
        verbose=False
    )

    # 获取视频属性用于保存输出
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # ✅ 优化2：防止部分视频返回 0 导致报错
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out = cv2.VideoWriter("output_custom.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for r in results:
        frame = r.orig_img.copy()  # 获取原始帧用于自定义绘制

        if r.boxes.id is not None:
            for box, tid, cls_id, conf in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_name = model.names[int(cls_id)]
                track_id = int(tid)

                label = f"{class_name}-{track_id}"
                color = get_color(track_id)  # ✅ 优化3：按 ID 分配专属颜色

                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # ✅ 优化4：添加文字背景框，防止标签被复杂背景遮挡
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Tracking", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️ 用户中断录制")
            break

    out.release()
    cv2.destroyAllWindows()
    print("✅ 渲染完成！结果已保存至: output_custom.mp4")


if __name__ == "__main__":
    main()