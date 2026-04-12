#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard 模型训练启动脚本

基于 XGuard-Train-Open-200K 数据集训练 YuFeng-XGuard-Reason-0.6B 模型

使用方法:
    # 使用默认配置训练
    python train.py
    
    # 指定配置文件
    python train.py --config configs/train_config.yaml
    
    # 覆盖部分配置
    python train.py --mode response_safety --epochs 5
"""

import os
import sys
import argparse

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.training.train_core import train, load_config, merge_config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="XGuard 模型训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["prompt_safety", "response_safety", "reasoning"],
        help="训练模式 (覆盖配置文件)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数 (覆盖配置文件)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小 (覆盖配置文件)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="学习率 (覆盖配置文件)",
    )
    parser.add_argument(
        "--use-lora",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=None,
        help="是否使用 LoRA (覆盖配置文件)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (覆盖配置文件)",
    )
    args = parser.parse_args()
    
    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 构建覆盖列表
    overrides = []
    if args.mode is not None:
        overrides.append(f"data.mode={args.mode}")
    if args.epochs is not None:
        overrides.append(f"training.num_train_epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"training.per_device_train_batch_size={args.batch_size}")
    if args.learning_rate is not None:
        overrides.append(f"training.learning_rate={args.learning_rate}")
    if args.use_lora is not None:
        overrides.append(f"lora.enabled={args.use_lora}")
    if args.output_dir is not None:
        overrides.append(f"training.output_dir={args.output_dir}")
    
    # 应用覆盖
    if overrides:
        config = merge_config(config, overrides)
        logger.info(f"应用配置覆盖: {overrides}")
    
    # 执行训练
    logger.info("=" * 60)
    logger.info("开始训练 XGuard 模型")
    logger.info("=" * 60)
    
    try:
        final_dir = train(config)
        logger.info("=" * 60)
        logger.info(f"训练完成! 模型保存在: {final_dir}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
