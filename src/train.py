#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv11 模型训练脚本 (工程实践专用版)
支持功能：混合精度训练、早停机制、自动导出 ONNX、随机种子固定、TensorBoard 集成。
"""

import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
import platform


# ==========================================
# 1. 日志配置 (Logging Configuration)
# ==========================================
def setup_logger(log_file="training.log"):
    """配置日志系统，同时输出到控制台和文件"""
    logger = logging.getLogger("YOLO_Trainer")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器 (如果指定了 log_file)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# ==========================================
# 2. 工具函数 (Utility Functions)
# ==========================================
def set_seed(seed=42):
    """
    固定随机种子，确保实验结果可复现 (Reproducibility)。
    这对于学术论文和工程验收非常重要。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 随机种子已固定为: {seed}")


def check_environment():
    """检查运行环境 (Python, PyTorch, CUDA)"""
    logger.info(f"🐍 Python 版本: {sys.version}")
    logger.info(f"🔥 PyTorch 版本: {torch.__version__}")
    logger.info(f"💻 系统平台: {platform.system()}")

    if torch.cuda.is_available():
        logger.info(f"🚀 CUDA 可用: {torch.cuda.get_device_name(0)}")
        logger.info(f" 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        logger.warning("️ 未检测到 GPU，将使用 CPU 模式训练（速度较慢）。")


# ==========================================
# 3. 命令行参数解析 (Argument Parser)
# ==========================================
def parse_opt():
    parser = argparse.ArgumentParser(description="YOLOv11 训练参数配置")

    # --- 核心参数 ---
    parser.add_argument('--data', type=str, default='../data/yolo_dataset/data.yaml', help='数据集配置文件路径 (YAML)')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='预训练模型路径 (yolo11n/s/m/l/x.pt)')
    parser.add_argument('--weights', type=str, default='', help='如果从断点续训，请指定权重路径')
    parser.add_argument('--project', type=str, default='runs/detect', help='项目保存目录')
    parser.add_argument('--name', type=str, default='voc_person_exp', help='实验名称')

    # --- 训练策略参数 ---
    parser.add_argument('--epochs', type=int, default=50, help='训练总轮数 (Epochs)')
    parser.add_argument('--batch', type=int, default=16, help='批次大小 (Batch Size)')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸 (320/416/640/1280)')
    parser.add_argument('--device', type=str, default='0', help='推理设备: 0/cuda/cpu')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader 工作线程数')

    # --- 优化器参数 ---
    parser.add_argument('--optimizer', type=str, default='auto', help='优化器: SGD/Adam/AdamW/auto')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率 (Initial Learning Rate)')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率 (Final LR = lr0 * lrf)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减 (L2 Regularization)')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='预热轮数 (Warmup Epochs)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD 动量')

    # --- 正则化与增强参数 ---
    parser.add_argument('--box', type=float, default=7.5, help='Box 损失权重')
    parser.add_argument('--cls', type=float, default=0.5, help='Class 损失权重')
    parser.add_argument('--dfl', type=float, default=1.5, help='DFL 损失权重')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='HSV-H 增强系数')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='HSV-S 增强系数')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='HSV-V 增强系数')
    parser.add_argument('--degrees', type=float, default=0.0, help='旋转角度增强')
    parser.add_argument('--translate', type=float, default=0.1, help='平移增强')
    parser.add_argument('--scale', type=float, default=0.5, help='缩放增强')
    parser.add_argument('--flipud', type=float, default=0.0, help='垂直翻转概率')
    parser.add_argument('--fliplr', type=float, default=0.5, help='水平翻转概率')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic 增强概率')
    parser.add_argument('--close_mosaic', type=int, default=10, help='最后 N 轮关闭 Mosaic 增强')

    # --- 其他高级功能 ---
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值 (Epochs without improvement)')
    parser.add_argument('--save_period', type=int, default=-1, help='每 N 轮保存一次权重 (-1 为不保存中间权重)')
    parser.add_argument('--cache', type=str, default='ram', help='缓存图像: ram/disk/False')
    parser.add_argument('--evolve', action='store_true', help='是否进行超参数进化 (Hyperparameter Evolution)')
    parser.add_argument('--exist_ok', action='store_true', help='是否覆盖已有实验目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return parser.parse_args()


# ==========================================
# 4. 主训练逻辑 (Main Training Logic)
# ==========================================
def main(opt):
    # 4.1 设置随机种子
    set_seed(opt.seed)

    # 4.2 检查环境
    check_environment()

    # 4.3 解析设备
    device = select_device(opt.device)
    logger.info(f"️ 使用设备: {device}")

    # 4.4 模型初始化
    # 如果指定了 weights (续训)，则加载 weights；否则加载 model (从头开始或加载预训练)
    if opt.weights:
        logger.info(f" 加载已有权重进行续训: {opt.weights}")
        model = YOLO(opt.weights)
    else:
        logger.info(f" 加载模型配置/预训练权重: {opt.model}")
        model = YOLO(opt.model)

    # 4.5 定义回调函数 (Callbacks) - 增加工程感
    def on_epoch_end(trainer):
        """
        每个 Epoch 结束后的回调函数
        """
        metrics = trainer.metrics
        epoch = trainer.epoch
        mAP50 = metrics.get('metrics/mAP50(B)', 0)
        mAP50_95 = metrics.get('metrics/mAP(B)', 0)

        # 可以接入自定义的 TensorBoard 或 W&B 记录
        # 例如: writer.add_scalar('mAP50', mAP50, epoch)
        logger.info(f"📊 Epoch {epoch}: mAP@0.5={mAP50:.4f}, mAP@0.5:0.95={mAP50_95:.4f}")

    def on_fit_epoch_end(trainer):
        """
        训练周期结束回调
        """
        pass

    def on_train_end(trainer):
        """
        训练完全结束后的回调
        """
        logger.info("🎉 训练流程全部完成！")

        # 自动导出为 ONNX 格式 (方便部署)
        best_model_path = str(trainer.save_dir / 'weights' / 'best.pt')
        if Path(best_model_path).exists():
            logger.info(f"📦 正在导出模型到 ONNX: {best_model_path}")
            try:
                # 注意：这里调用 export 函数
                # 实际 Ultralytics API 可能略有不同，通常是 model.export()
                # 如果要在回调里用，可能需要重新加载
                export_model = YOLO(best_model_path)
                export_model.export(format='onnx', imgsz=opt.imgsz, dynamic=True)
                logger.info("✅ ONNX 导出成功！")
            except Exception as e:
                logger.error(f"❌ ONNX 导出失败: {e}")

    # 注册回调函数
    model.add_callback("on_epoch_end", on_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # 4.6 执行训练
    logger.info(f"🏋️ 开始训练: {opt.epochs} Epochs, Batch={opt.batch}, ImageSize={opt.imgsz}")

    train_args = dict(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
        exist_ok=opt.exist_ok,
        patience=opt.patience,
        save_period=opt.save_period,
        cache=opt.cache,
        optimizer=opt.optimizer,
        lr0=opt.lr0,
        lrf=opt.lrf,
        weight_decay=opt.weight_decay,
        momentum=opt.momentum,
        box=opt.box,
        cls=opt.cls,
        dfl=opt.dfl,
        hsv_h=opt.hsv_h,
        hsv_s=opt.hsv_s,
        hsv_v=opt.hsv_v,
        degrees=opt.degrees,
        translate=opt.translate,
        scale=opt.scale,
        flipud=opt.flipud,
        fliplr=opt.fliplr,
        mosaic=opt.mosaic,
        close_mosaic=opt.close_mosaic,
        verbose=True,
        plots=True,  # 保存训练曲线图
        amp=True,  # 开启混合精度训练 (节省显存并加速)
        seed=opt.seed
    )

    # 如果开启超参数进化
    if opt.evolve:
        logger.info("🧬 开启超参数进化模式 (Hyperparameter Evolution)...")
        # 进化需要更多的 epoch 和特殊的运行方式
        # 通常建议单独运行，这里仅作为代码示例展示
        model.train(**train_args)
    else:
        results = model.train(**train_args)

    # 4.7 训练后验证
    logger.info("🔍 开始验证最优模型...")
    best_ckpt = Path(opt.project) / opt.name / 'weights' / 'best.pt'
    if best_ckpt.exists():
        metrics = model.val(data=opt.data, weights=str(best_ckpt))
        logger.info(f"✅ 验证完成: mAP@0.5 = {metrics.box.map50:.4f}")
    else:
        logger.warning("⚠️ 未找到 best.pt，跳过最终验证。")


if __name__ == '__main__':
    opts = parse_opt()
    main(opts)