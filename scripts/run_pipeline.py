#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard 一键训练流水线

自动执行: 数据下载 -> 数据预处理 -> 模型训练

使用方法:
    # 完整流水线
    python scripts/run_pipeline.py

    # 仅下载数据
    python scripts/run_pipeline.py --step download

    # 仅训练 (数据已下载)
    python scripts/run_pipeline.py --step train

    # 指定配置
    python scripts/run_pipeline.py --config configs/train_config.yaml
"""

import os
import sys
import argparse

from loguru import logger

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def step_download(output_dir: str = "data/raw") -> str:
    """步骤1: 下载数据集"""
    logger.info("=" * 60)
    logger.info("步骤 1/3: 下载 XGuard-Train-Open-200K 数据集")
    logger.info("=" * 60)

    from scripts.download_dataset import download_dataset
    csv_path = download_dataset(output_dir)

    logger.info(f"数据集已下载: {csv_path}")
    return csv_path


def step_preprocess(raw_data_path: str, mode: str = "prompt_safety") -> str:
    """步骤2: 预处理数据"""
    logger.info("=" * 60)
    logger.info("步骤 2/3: 预处理训练数据")
    logger.info("=" * 60)

    from src.data.dataset import preprocess_raw_data

    processed_path = os.path.join("data/processed", f"xguard_{mode}.json")
    preprocess_raw_data(raw_data_path, processed_path, mode)

    logger.info(f"数据已预处理: {processed_path}")
    return processed_path


def step_train(config_path: str) -> str:
    """步骤3: 训练模型"""
    logger.info("=" * 60)
    logger.info("步骤 3/3: 训练模型")
    logger.info("=" * 60)

    from src.training.train import load_config, train

    config = load_config(config_path)
    final_dir = train(config)

    logger.info(f"模型已保存: {final_dir}")
    return final_dir


def run_pipeline(
    config_path: str = "configs/train_config.yaml",
    skip_download: bool = False,
    skip_preprocess: bool = False,
):
    """
    运行完整流水线

    Args:
        config_path: 训练配置文件路径
        skip_download: 是否跳过下载步骤
        skip_preprocess: 是否跳过预处理步骤
    """
    import yaml

    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_data_path = config["data"]["raw_data_path"]
    mode = config["data"]["mode"]

    # 步骤1: 下载数据
    if not skip_download and not os.path.exists(raw_data_path):
        step_download(os.path.dirname(os.path.dirname(raw_data_path)))
    else:
        if os.path.exists(raw_data_path):
            logger.info(f"数据已存在，跳过下载: {raw_data_path}")
        else:
            logger.warning(f"数据不存在且跳过下载: {raw_data_path}")

    # 步骤2: 预处理数据
    processed_path = os.path.join("data/processed", f"xguard_{mode}.json")
    if not skip_preprocess and not os.path.exists(processed_path):
        step_preprocess(raw_data_path, mode)
    else:
        if os.path.exists(processed_path):
            logger.info(f"预处理数据已存在，跳过: {processed_path}")
        else:
            logger.warning(f"预处理数据不存在且跳过: {processed_path}")

    # 步骤3: 训练
    final_dir = step_train(config_path)

    logger.info("=" * 60)
    logger.info("流水线执行完成!")
    logger.info(f"最终模型: {final_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="XGuard 一键训练流水线")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["download", "preprocess", "train", "all"],
        default="all",
        help="执行步骤 (默认: all)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过数据下载步骤",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳过数据预处理步骤",
    )
    args = parser.parse_args()

    if args.step == "download":
        step_download()
    elif args.step == "preprocess":
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        step_preprocess(config["data"]["raw_data_path"], config["data"]["mode"])
    elif args.step == "train":
        step_train(args.config)
    else:
        run_pipeline(
            config_path=args.config,
            skip_download=args.skip_download,
            skip_preprocess=args.skip_preprocess,
        )


if __name__ == "__main__":
    main()
