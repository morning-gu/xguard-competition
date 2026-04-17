"""评估启动脚本

用法:
    python scripts/evaluate.py --model_path outputs/checkpoints
    python scripts/evaluate.py --model_path Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B
"""

import os
import sys
import argparse
import logging

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import Guardrail
from src.data.dataset import load_xguard_test_data
from src.evaluation.evaluate import evaluate_on_test_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="XGuard 模型评估")
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
        default="outputs/eval_results.json",
        help="评估结果输出路径",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU 设备 ID",
    )

    args = parser.parse_args()

    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    guardrail = Guardrail(args.model_path, args.device_id)

    # 加载测试数据
    logger.info(f"加载测试数据: {args.test_data}")
    test_data = load_xguard_test_data(args.test_data)

    # 评估
    logger.info("开始评估...")
    results = evaluate_on_test_set(
        guardrail=guardrail,
        test_data=test_data,
        output_path=args.output,
    )

    logger.info(f"\n评估完成!")
    logger.info(f"F1 (binary): {results['f1_binary']:.4f}")
    logger.info(f"F1 (macro): {results['f1_macro']:.4f}")
    logger.info(f"平均耗时: {results['avg_time']:.4f}s")


if __name__ == "__main__":
    main()
