"""测试集推理启动脚本

用法:
    python scripts/run_inference.py
    python scripts/run_inference.py --model_path outputs/checkpoints --enable_reasoning
"""

import os
import sys

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.run_inference import run_inference_on_test_set

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XGuard 测试集推理")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B",
        help="模型路径",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="test_dataset/xguard_test_open_1k.jsonl",
        help="测试数据路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inference_results.json",
        help="推理结果输出路径",
    )
    parser.add_argument(
        "--enable_reasoning",
        action="store_true",
        help="是否开启归因分析",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大推理样本数 (调试用)",
    )

    args = parser.parse_args()

    run_inference_on_test_set(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_path=args.output,
        enable_reasoning=args.enable_reasoning,
        max_samples=args.max_samples,
    )
