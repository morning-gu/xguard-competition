"""模型评估模块

在测试集上评估模型的 F1 分数和推理耗时。
"""

import json
import time
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, classification_report

logger = logging.getLogger(__name__)


def evaluate_on_test_set(
    guardrail,
    test_data: List[Dict],
    output_path: Optional[str] = None,
) -> Dict:
    """在测试集上评估模型

    Args:
        guardrail: Guardrail 实例 (包含 infer 方法)
        test_data: 测试数据列表
        output_path: 评估结果保存路径 (可选)

    Returns:
        评估结果字典, 包含 F1 分数、平均耗时等
    """
    predictions = []
    labels = []
    times = []
    details = []

    for i, item in enumerate(test_data):
        # 构建 messages
        messages = _build_messages(item)

        # 推理
        result = guardrail.infer(
            messages=messages,
            policy=None,
            enable_reasoning=False,
        )

        # 解析预测结果
        pred_label = _parse_risk_tag(result["risk_tag"])
        true_label = item.get("label", "")

        predictions.append(pred_label)
        labels.append(true_label)
        times.append(result["time"])

        details.append({
            "id": item.get("id", ""),
            "true_label": true_label,
            "pred_label": pred_label,
            "risk_score": result["risk_score"],
            "risk_tag": result["risk_tag"],
            "time": result["time"],
        })

        if (i + 1) % 100 == 0:
            logger.info(f"已评估 {i+1}/{len(test_data)} 条样本")

    # 计算指标
    # 将标签转为二分类: safe vs unsafe
    pred_binary = ["safe" if p == "safe" else "unsafe" for p in predictions]
    label_binary = ["safe" if l == "safe" else "unsafe" for l in labels]

    f1_binary = f1_score(label_binary, pred_binary, pos_label="unsafe", average="binary")
    f1_macro = f1_score(label_binary, pred_binary, average="macro")
    avg_time = np.mean(times)

    report = classification_report(
        label_binary, pred_binary, target_names=["safe", "unsafe"], digits=4
    )

    results = {
        "f1_binary": float(f1_binary),
        "f1_macro": float(f1_macro),
        "avg_time": float(avg_time),
        "total_samples": len(test_data),
        "classification_report": report,
    }

    logger.info(f"\n{report}")
    logger.info(f"F1 (binary, unsafe): {f1_binary:.4f}")
    logger.info(f"F1 (macro): {f1_macro:.4f}")
    logger.info(f"平均推理耗时: {avg_time:.4f}s")

    # 保存详细结果
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": results,
                "details": details,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"评估结果已保存到: {output_path}")

    return results


def _build_messages(item: Dict) -> List[Dict]:
    """从测试数据构建 messages 格式

    Args:
        item: 测试数据条目, 包含 prompt, response, stage

    Returns:
        OpenAI API 格式的 messages 列表
    """
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


def _parse_risk_tag(risk_tag: str) -> str:
    """解析 risk_tag 为 safe/unsafe 二分类

    Args:
        risk_tag: 细粒度风险类别标签

    Returns:
        "safe" 或 "unsafe"
    """
    if risk_tag == "sec" or risk_tag == "Safe-Safe" or risk_tag.lower() == "safe":
        return "safe"
    return "unsafe"
