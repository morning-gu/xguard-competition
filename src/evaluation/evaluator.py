#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集校验模块

提供测试集/验证集的评估功能，支持：
- 加载测试集
- 批量推理
- 计算评估指标（F1、精确率、召回率等）
- 生成评估报告
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm
from loguru import logger

# 安全标签
SAFE_TAG = "sec"


def load_test_dataset(test_file: str) -> List[Dict]:
    """加载 JSONL 格式的测试集"""
    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info(f"加载测试集：{test_file} ({len(samples)} 条样本)")
    return samples


def build_messages(sample: Dict) -> List[Dict]:
    """
    根据样本的 stage 字段构造 messages 输入
    
    Args:
        sample: 测试集样本，包含 prompt, response, stage 字段
    
    Returns:
        messages: 符合 OpenAI API 格式的消息列表
    """
    stage = sample["stage"]
    
    if stage == "q":
        return [{"role": "user", "content": sample["prompt"]}]
    elif stage == "r":
        return [{"role": "assistant", "content": sample["response"]}]
    elif stage == "qr":
        return [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["response"]},
        ]
    else:
        raise ValueError(f"未知的 stage 类型：{stage}")


def risk_tag_to_label(risk_tag: str) -> str:
    """将细粒度风险标签映射为二分类标签"""
    return "safe" if risk_tag == SAFE_TAG else "unsafe"


def _calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    """
    计算评估指标
    
    Args:
        predictions: 预测标签列表
        ground_truths: 真实标签列表
    
    Returns:
        包含各项指标的字典
    """
    y_true = np.array([1 if l == "unsafe" else 0 for l in ground_truths])
    y_pred = np.array([1 if l == "unsafe" else 0 for l in predictions])
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    
    # 分类报告
    safe_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    safe_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    safe_f1 = 2 * safe_precision * safe_recall / (safe_precision + safe_recall) if (safe_precision + safe_recall) > 0 else 0.0
    safe_support = tn + fp
    unsafe_support = tp + fn
    
    report = (
        f"              precision    recall  f1-score   support\n\n"
        f"        safe     {safe_precision:.4f}    {safe_recall:.4f}    {safe_f1:.4f}       {safe_support}\n"
        f"      unsafe     {precision:.4f}    {recall:.4f}    {f1:.4f}       {unsafe_support}\n\n"
        f"    accuracy                         {accuracy:.4f}       {len(y_true)}\n"
        f"   macro avg     {(safe_precision+precision)/2:.4f}    {(safe_recall+recall)/2:.4f}    {(safe_f1+f1)/2:.4f}       {len(y_true)}\n"
        f"weighted avg     {(safe_precision*safe_support+precision*unsafe_support)/len(y_true):.4f}    "
        f"{(safe_recall*safe_support+recall*unsafe_support)/len(y_true):.4f}    "
        f"{(safe_f1*safe_support+f1*unsafe_support)/len(y_true):.4f}       {len(y_true)}\n"
    )
    
    return {
        "f1_unsafe": float(f1),
        "precision_unsafe": float(precision),
        "recall_unsafe": float(recall),
        "accuracy": float(accuracy),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "classification_report": report,
    }


def _save_results(
    output_dir: str,
    samples: List[Dict],
    predictions: List[str],
    ground_truths: List[str],
    inference_times: List[float],
    risk_tags: List[str],
    risk_scores: List[float],
    results: Dict,
):
    """保存评估结果到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
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
    
    wrong_samples = [d for d in details if not d["correct"]]
    with open(output_path / "eval_errors.jsonl", "w", encoding="utf-8") as f:
        for d in wrong_samples:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    logger.info(f"\n结果已保存到：{output_path}")
    logger.info(f"  - eval_summary.json  (汇总指标)")
    logger.info(f"  - eval_details.jsonl (逐条预测结果)")
    logger.info(f"  - eval_errors.jsonl  (预测错误样本，{len(wrong_samples)} 条)")


def evaluate_with_guardrail(
    model_path: str = None,
    test_data: str = None,
    output_dir: str = None,
    enable_reasoning: bool = False,
    limit: int = None,
) -> Dict:
    """
    使用 Guardrail 类执行评估（供 scripts/evaluate.py 调用）
    
    Args:
        model_path: 模型路径
        test_data: 测试集文件路径
        output_dir: 结果输出目录
        enable_reasoning: 是否开启归因分析
        limit: 限制评测样本数量
    
    Returns:
        评估结果字典
    """
    from inference import Guardrail
    
    # 1. 加载测试集
    if not os.path.exists(test_data):
        raise FileNotFoundError(f"测试集文件不存在：{test_data}")
    
    samples = load_test_dataset(test_data)
    
    # 限制评测数量
    if limit is not None and limit > 0:
        samples = samples[:limit]
        logger.info(f"限制评测数量：{limit} 条")
    
    # 2. 初始化模型
    logger.info("正在加载模型...")
    guardrail = Guardrail(model_path=model_path)
    logger.info("模型加载完成!")
    
    # 3. 逐条推理
    predictions = []
    ground_truths = []
    inference_times = []
    risk_tags = []
    risk_scores = []
    errors = []
    
    logger.info(f"\n开始推理 ({len(samples)} 条样本)...")
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
            predictions.append("safe")
            ground_truths.append(sample["label"])
            inference_times.append(0.0)
            risk_tags.append("sec")
            risk_scores.append(1.0)
    
    # 4. 计算评估指标
    metrics = _calculate_metrics(predictions, ground_truths)
    avg_time = float(np.mean(inference_times))
    
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
    
    # 构建结果
    results = {
        "test_file": test_data,
        "model_path": model_path or "default",
        "num_samples": len(samples),
        **metrics,
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
    
    # 5. 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)
    logger.info(f"测试集：{test_data}")
    logger.info(f"样本数：{len(samples)}")
    logger.info("")
    logger.info("--- 核心指标 ---")
    logger.info(f"F1 Score (unsafe):  {results['f1_unsafe']:.4f}")
    logger.info(f"Precision (unsafe): {results['precision_unsafe']:.4f}")
    logger.info(f"Recall (unsafe):    {results['recall_unsafe']:.4f}")
    logger.info(f"Accuracy:           {results['accuracy']:.4f}")
    logger.info(f"平均推理耗时：       {avg_time:.4f} 秒")
    logger.info("")
    logger.info("--- 分类报告 ---")
    logger.info(results["classification_report"])
    logger.info("--- 按 stage 分组 ---")
    for stage, stats in stage_stats.items():
        acc = stats["correct"] / stats["total"]
        avg_t = np.mean(stats["times"])
        logger.info(f"  stage={stage}: 准确率={acc:.4f} ({stats['correct']}/{stats['total']}), 平均耗时={avg_t:.4f}s")
    
    if errors:
        logger.warning(f"\n推理错误 ({len(errors)} 条)")
        for err in errors[:10]:
            logger.warning(f"  样本 {err['id']}: {err['error']}")
    
    # 6. 保存详细结果
    if output_dir:
        _save_results(output_dir, samples, predictions, ground_truths, 
                      inference_times, risk_tags, risk_scores, results)
    
    return results


def evaluate(
    model,
    tokenizer,
    test_file: str,
    output_dir: Optional[str] = None,
    enable_reasoning: bool = False,
    limit: Optional[int] = None,
) -> Dict:
    """
    执行测试集评估（直接使用 model 和 tokenizer）
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        test_file: 测试集文件路径
        output_dir: 结果输出目录
        enable_reasoning: 是否开启归因分析
        limit: 限制评测样本数量
    
    Returns:
        评估结果字典
    """
    from src.inference.engine import infer_with_timing
    
    # 1. 加载测试集
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试集文件不存在：{test_file}")
    
    samples = load_test_dataset(test_file)
    
    # 限制评测数量
    if limit is not None and limit > 0:
        samples = samples[:limit]
        logger.info(f"限制评测数量：{limit} 条")
    
    # 2. 逐条推理
    predictions = []
    ground_truths = []
    inference_times = []
    risk_tags = []
    risk_scores = []
    errors = []
    
    logger.info(f"\n开始推理 ({len(samples)} 条样本)...")
    for i, sample in enumerate(tqdm(samples, desc="推理进度")):
        try:
            messages = build_messages(sample)
            result = infer_with_timing(
                model,
                tokenizer,
                messages,
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
            predictions.append("safe")
            ground_truths.append(sample["label"])
            inference_times.append(0.0)
            risk_tags.append("sec")
            risk_scores.append(1.0)
    
    # 3. 计算评估指标
    metrics = _calculate_metrics(predictions, ground_truths)
    avg_time = float(np.mean(inference_times))
    
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
    
    # 构建结果
    results = {
        "test_file": test_file,
        "num_samples": len(samples),
        **metrics,
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
    
    # 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)
    logger.info(f"测试集：{test_file}")
    logger.info(f"样本数：{len(samples)}")
    logger.info("")
    logger.info("--- 核心指标 ---")
    logger.info(f"F1 Score (unsafe):  {results['f1_unsafe']:.4f}")
    logger.info(f"Precision (unsafe): {results['precision_unsafe']:.4f}")
    logger.info(f"Recall (unsafe):    {results['recall_unsafe']:.4f}")
    logger.info(f"Accuracy:           {results['accuracy']:.4f}")
    logger.info(f"平均推理耗时：       {avg_time:.4f} 秒")
    logger.info("")
    logger.info("--- 分类报告 ---")
    logger.info(results["classification_report"])
    logger.info("--- 按 stage 分组 ---")
    for stage, stats in stage_stats.items():
        acc = stats["correct"] / stats["total"]
        avg_t = np.mean(stats["times"])
        logger.info(f"  stage={stage}: 准确率={acc:.4f} ({stats['correct']}/{stats['total']}), 平均耗时={avg_t:.4f}s")
    
    if errors:
        logger.warning(f"\n推理错误 ({len(errors)} 条)")
        for err in errors[:10]:
            logger.warning(f"  样本 {err['id']}: {err['error']}")
    
    # 保存详细结果
    if output_dir:
        _save_results(output_dir, samples, predictions, ground_truths,
                      inference_times, risk_tags, risk_scores, results)
    
    return results


if __name__ == "__main__":
    import argparse
    from src.model.loader import load_model_and_tokenizer
    
    parser = argparse.ArgumentParser(description="测试集校验评估")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/test_dataset/xguard_test_open_1k.jsonl",
        help="测试集文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/eval",
        help="结果输出目录",
    )
    parser.add_argument(
        "--enable_reasoning",
        action="store_true",
        default=False,
        help="是否开启归因分析",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制评测样本数量",
    )
    args = parser.parse_args()
    
    logger.info("正在加载模型...")
    model, tokenizer = load_model_and_tokenizer(model_path=args.model_path)
    logger.info("模型加载完成!")
    
    evaluate(
        model=model,
        tokenizer=tokenizer,
        test_file=args.test_data,
        output_dir=args.output_dir,
        enable_reasoning=args.enable_reasoning,
        limit=args.limit,
    )
