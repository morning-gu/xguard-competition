#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard 校验集评估脚本（命令行入口）

这是评估模块的命令行入口，实际评估逻辑已迁移至 src/evaluation/evaluator.py。

用法:
    # 使用默认预训练模型评估
    python scripts/evaluate.py

    # 使用微调后的模型评估
    python scripts/evaluate.py --model_path models/checkpoints/xguard-finetuned

    # 指定校验集路径
    python scripts/evaluate.py --test_data data/test_dataset/xguard_test_open_1k.jsonl

    # 输出详细结果到文件
    python scripts/evaluate.py --output_dir results/eval
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_with_guardrail

# 默认校验集路径
DEFAULT_TEST_DATA = PROJECT_ROOT / "data" / "test_dataset" / "xguard_test_open_1k.jsonl"


def main():
    parser = argparse.ArgumentParser(description="XGuard 校验集评估脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径 (默认使用预训练模型)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help=f"校验集文件路径 (默认：{DEFAULT_TEST_DATA})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/eval",
        help="结果输出目录 (默认：results/eval)",
    )
    parser.add_argument(
        "--enable_reasoning",
        action="store_true",
        default=False,
        help="是否开启归因分析 (默认关闭，评估时通常不需要)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制评测样本数量 (默认评测全部)",
    )
    args = parser.parse_args()

    evaluate_with_guardrail(
        model_path=args.model_path,
        test_data=args.test_data or str(DEFAULT_TEST_DATA),
        output_dir=args.output_dir,
        enable_reasoning=args.enable_reasoning,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
