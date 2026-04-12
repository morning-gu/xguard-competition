#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard 校验集评估脚本

使用 data/test_dataset/xguard_test_open_1k.jsonl 校验集对模型进行评估,
计算 F1 分数、精确率、召回率及平均推理耗时等指标。

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
import json
import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from inference import Guardrail


# 默认校验集路径
DEFAULT_TEST_DATA = PROJECT_ROOT / "data" / "test_dataset" / "xguard_test_open_1k.jsonl"

# 安全标签: "sec" 对应 safe, 其余均为 unsafe
SAFE_TAG = "sec"


def load_test_dataset(test_file: str) -> list:
    """加载 JSONL 格式的校验集"""
    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"加载校验集: {test_file} ({len(samples)} 条样本)")
    return samples


def build_messages(sample: dict) -> list:
    """根据样本的 stage 字段构造 messages 输入

    Args:
        sample: 校验集样本, 包含 prompt, response, stage 字段

    Returns:
        messages: 符合 OpenAI API 格式的消息列表
    """
    stage = sample["stage"]

    if stage == "q":
        # 仅输入 prompt
        return [{"role": "user", "content": sample["prompt"]}]
    elif stage == "r":
        # 仅输入 response
        return [{"role": "assistant", "content": sample["response"]}]
    elif stage == "qr":
        # 输入 prompt + response 对话对
        return [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]
    else:
        raise ValueError(f"未知的 stage 类型: {stage}")


def risk_tag_to_label(risk_tag: str) -> str:
    """将细粒度风险标签映射为二分类标签

    Args:
        risk_tag: 细粒度风险标签 (如 "sec", "pc", "dw" 等)

    Returns:
        "safe" 或 "unsafe"
    """
    return "safe" if risk_tag == SAFE_TAG else "unsafe"


def evaluate(
    model_path: str = None,
    test_data: str = None,
    output_dir: str = None,
    enable_reasoning: bool = False,
    limit: int = None,
):
    """执行校验集评估

    Args:
        model_path: 模型路径, None 则使用默认路径
        test_data: 校验集文件路径
        output_dir: 结果输出目录, None 则不保存文件
        enable_reasoning: 是否开启归因分析 (评估时通常为 False)
        limit: 限制评测样本数量, None 则评测全部
    """
    # 1. 加载校验集
    test_file = test_data or str(DEFAULT_TEST_DATA)
    if not os.path.exists(test_file):
        print(f"错误: 校验集文件不存在: {test_file}")
        sys.exit(1)
    samples = load_test_dataset(test_file)

    # 限制评测数量
    if limit is not None and limit > 0:
        samples = samples[:limit]
        print(f"限制评测数量: {limit} 条")

    # 2. 初始化模型
    print(f"正在加载模型...")
    guardrail = Guardrail(model_path=model_path)
    print("模型加载完成!")

    # 3. 逐条推理
    predictions = []
    ground_truths = []
    inference_times = []
    risk_tags = []
    risk_scores = []
    errors = []

    print(f"\n开始推理 ({len(samples)} 条样本)...")
    for i, sample in enumerate(tqdm(samples, desc="推理进度")):
        try:
            messages = build_messages(sample)
            result = guardrail.infer(
                messages=messages,
                policy=None,
                enable_reasoning=enable_reasoning,
            )

            pred_label = risk_tag_to_label(result["risk_tag"])
            predictions.append(pred_label)
            ground_truths.append(sample["label"])
            inference_times.append(result["time"])
            risk_tags.append(result["risk_tag"])
            risk_scores.append(result["risk_score"])

        except Exception as e:
            errors.append({"index": i, "id": sample.get("id", ""), "error": str(e)})
            # 推理失败时默认预测为 safe
            predictions.append("safe")
            ground_truths.append(sample["label"])
            inference_times.append(0.0)
            risk_tags.append("sec")
            risk_scores.append(1.0)

    # 4. 计算评估指标
    # 将标签转为数值: unsafe=1 (正类), safe=0
    y_true = np.array([1 if l == "unsafe" else 0 for l in ground_truths])
    y_pred = np.array([1 if l == "unsafe" else 0 for l in predictions])

    # 手动计算指标 (避免 sklearn 依赖)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # unsafe 预测为 unsafe
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # safe 预测为 unsafe
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # unsafe 预测为 safe
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))  # safe 预测为 safe

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    avg_time = float(np.mean(inference_times))

    # 分类报告 (手动生成)
    safe_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    safe_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    safe_f1 = 2 * safe_precision * safe_recall / (safe_precision + safe_recall) if (safe_precision + safe_recall) > 0 else 0.0
    safe_support = tn + fp
    unsafe_support = tp + fn

    report = (
        f"              precision    recall  f1-score   support\n"
        f"\n"
        f"        safe     {safe_precision:.4f}    {safe_recall:.4f}    {safe_f1:.4f}       {safe_support}\n"
        f"      unsafe     {precision:.4f}    {recall:.4f}    {f1:.4f}       {unsafe_support}\n"
        f"\n"
        f"    accuracy                         {accuracy:.4f}       {len(y_true)}\n"
        f"   macro avg     {(safe_precision+precision)/2:.4f}    {(safe_recall+recall)/2:.4f}    {(safe_f1+f1)/2:.4f}       {len(y_true)}\n"
        f"weighted avg     {(safe_precision*safe_support+precision*unsafe_support)/len(y_true):.4f}    {(safe_recall*safe_support+recall*unsafe_support)/len(y_true):.4f}    {(safe_f1*safe_support+f1*unsafe_support)/len(y_true):.4f}       {len(y_true)}\n"
    )

    # 按 stage 分组统计
    stage_stats = {}
    for sample, pred, gt, t in zip(samples, predictions, ground_truths, inference_times):
        stage = sample["stage"]
        if stage not in stage_stats:
            stage_stats[stage] = {"correct": 0, "total": 0, "times": []}
        stage_stats[stage]["total"] += 1
        if pred == gt:
            stage_stats[stage]["correct"] += 1
        stage_stats[stage]["times"].append(t)

    # 5. 输出结果
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"校验集: {test_file}")
    print(f"样本数: {len(samples)}")
    print(f"模型路径: {model_path or '默认'}")
    print(f"")
    print(f"--- 核心指标 ---")
    print(f"F1 Score (unsafe):  {f1:.4f}")
    print(f"Precision (unsafe): {precision:.4f}")
    print(f"Recall (unsafe):    {recall:.4f}")
    print(f"平均推理耗时:       {avg_time:.4f} 秒")
    print(f"")
    print(f"--- 分类报告 ---")
    print(report)
    print(f"--- 按 stage 分组 ---")
    for stage, stats in stage_stats.items():
        acc = stats["correct"] / stats["total"]
        avg_t = np.mean(stats["times"])
        print(f"  stage={stage}: 准确率={acc:.4f} ({stats['correct']}/{stats['total']}), 平均耗时={avg_t:.4f}s")

    if errors:
        print(f"\n--- 推理错误 ({len(errors)} 条) ---")
        for err in errors[:10]:
            print(f"  样本 {err['id']}: {err['error']}")

    # 6. 保存详细结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 汇总结果
        summary = {
            "test_data": str(test_file),
            "model_path": str(model_path) if model_path else "default",
            "num_samples": len(samples),
            "f1_unsafe": float(f1),
            "precision_unsafe": float(precision),
            "recall_unsafe": float(recall),
            "avg_inference_time": float(avg_time),
            "stage_stats": {
                k: {
                    "accuracy": v["correct"] / v["total"],
                    "total": v["total"],
                    "avg_time": float(np.mean(v["times"])),
                }
                for k, v in stage_stats.items()
            },
            "num_errors": len(errors),
        }
        with open(output_path / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 逐条预测结果
        details = []
        for sample, pred, gt, t, tag, score in zip(
            samples, predictions, ground_truths, inference_times, risk_tags, risk_scores
        ):
            details.append({
                "id": sample["id"],
                "stage": sample["stage"],
                "ground_truth": gt,
                "prediction": pred,
                "risk_tag": tag,
                "risk_score": float(score),
                "inference_time": float(t),
                "correct": pred == gt,
            })
        with open(output_path / "eval_details.jsonl", "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # 错误样本 (预测错误的样本, 用于分析)
        wrong_samples = [d for d in details if not d["correct"]]
        with open(output_path / "eval_errors.jsonl", "w", encoding="utf-8") as f:
            for d in wrong_samples:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"\n结果已保存到: {output_path}")
        print(f"  - eval_summary.json  (汇总指标)")
        print(f"  - eval_details.jsonl (逐条预测结果)")
        print(f"  - eval_errors.jsonl  (预测错误样本, {len(wrong_samples)} 条)")


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
        help=f"校验集文件路径 (默认: {DEFAULT_TEST_DATA})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/eval",
        help="结果输出目录 (默认: results/eval)",
    )
    parser.add_argument(
        "--enable_reasoning",
        action="store_true",
        default=False,
        help="是否开启归因分析 (默认关闭, 评估时通常不需要)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制评测样本数量 (默认评测全部)",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        test_data=args.test_data,
        output_dir=args.output_dir,
        enable_reasoning=args.enable_reasoning,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
