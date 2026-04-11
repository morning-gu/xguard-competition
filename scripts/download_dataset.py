#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard-Train-Open-200K 数据集下载脚本

数据集信息:
- 数据集名称: XGuard-Train-Open-200K
- 数据集 ID: Alibaba-AAIG/XGuard-Train-Open-200K
- 来源: ModelScope 魔搭社区
- 用途: XGuard 安全护栏模型训练数据

使用方法:
    python scripts/download_dataset.py [--output_dir data/raw]
"""

import os
import argparse
from modelscope.msdatasets import MsDataset


def download_dataset(output_dir: str = "data/raw") -> str:
    """
    下载 XGuard-Train-Open-200K 数据集

    Args:
        output_dir: 数据集保存目录

    Returns:
        数据集保存路径
    """
    dataset_id = "Alibaba-AAIG/XGuard-Train-Open-200K"

    print(f"正在下载数据集: {dataset_id}")
    print("数据集较大(200K条)，请耐心等待...")

    os.makedirs(output_dir, exist_ok=True)

    # 使用 ModelScope SDK 下载数据集
    ds = MsDataset.load(
        dataset_id,
        split="train",
        cache_dir=output_dir,
    )

    # 将数据集保存为本地文件
    save_path = os.path.join(output_dir, "xguard_train_200k")
    os.makedirs(save_path, exist_ok=True)

    # 转换为 pandas DataFrame 并保存
    df = ds.to_pandas()
    csv_path = os.path.join(save_path, "train.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\n数据集下载完成!")
    print(f"保存路径: {csv_path}")
    print(f"数据量: {len(df)} 条")
    print(f"列名: {list(df.columns)}")

    # 打印前几行预览
    print(f"\n数据预览:")
    print(df.head(3).to_string())

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 XGuard-Train-Open-200K 数据集")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="数据集保存目录 (默认: data/raw)",
    )
    args = parser.parse_args()

    download_dataset(args.output_dir)
