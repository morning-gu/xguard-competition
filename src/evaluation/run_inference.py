"""基于测试数据集的批量推理脚本

读取 test_dataset/xguard_test_open_1k.jsonl,
使用 Guardrail 模型进行推理,
输出每条样本的推理结果。
"""

import json
import os
import sys
import time
import argparse
import logging

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import Guardrail

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_inference_on_test_set(
    model_path: str,
    test_data_path: str,
    output_path: str,
    enable_reasoning: bool = False,
    max_samples: int = None,
    device_id: int = 0,
):
    """在测试集上运行推理

    Args:
        model_path: 模型路径
        test_data_path: 测试数据 jsonl 路径
        output_path: 推理结果输出路径
        enable_reasoning: 是否开启归因分析
        max_samples: 最大推理样本数 (用于调试)
        device_id: GPU 设备 ID
    """
    # 1. 加载模型
    logger.info(f"加载模型: {model_path}")
    guardrail = Guardrail(model_path, device_id)

    # 2. 加载测试数据
    logger.info(f"加载测试数据: {test_data_path}")
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))
                if max_samples and len(test_data) >= max_samples:
                    break

    logger.info(f"测试样本数: {len(test_data)}")

    # 3. 逐条推理
    results = []
    total_time = 0.0

    for i, item in enumerate(test_data):
        # 构建 messages
        messages = _build_messages(item)

        # 推理
        result = guardrail.infer(
            messages=messages,
            policy=None,
            enable_reasoning=enable_reasoning,
        )

        # 记录结果
        result_entry = {
            "id": item.get("id", ""),
            "stage": item.get("stage", ""),
            "true_label": item.get("label", ""),
            "risk_score": result["risk_score"],
            "risk_tag": result["risk_tag"],
            "explanation": result["explanation"],
            "time": result["time"],
        }
        results.append(result_entry)
        total_time += result["time"]

        if (i + 1) % 50 == 0:
            avg_time = total_time / (i + 1)
            logger.info(
                f"进度: {i+1}/{len(test_data)}, "
                f"当前样本耗时: {result['time']:.4f}s, "
                f"平均耗时: {avg_time:.4f}s"
            )

    # 4. 统计结果
    avg_time = total_time / len(results) if results else 0
    logger.info(f"\n推理完成!")
    logger.info(f"总样本数: {len(results)}")
    logger.info(f"总耗时: {total_time:.2f}s")
    logger.info(f"平均耗时: {avg_time:.4f}s")

    # 5. 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_samples": len(results),
                    "total_time": total_time,
                    "avg_time": avg_time,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"结果已保存到: {output_path}")

    return results


def _build_messages(item: dict) -> list:
    """从测试数据构建 messages"""
    stage = item.get("stage", "q")
    prompt = item.get("prompt", "") or ""
    response = item.get("response", "") or ""

    messages = []
    if stage == "q":
        messages.append({"role": "user", "content": prompt})
    elif stage == "r":
        messages.append({"role": "assistant", "content": response})
    elif stage == "qr":
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})

    return messages


if __name__ == "__main__":
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
