#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard 训练数据预处理脚本

将 XGuard-Train-Open-200K JSONL 数据转换为训练所需的 JSON 格式
"""

import os
import json
import argparse
from pathlib import Path
from loguru import logger


def preprocess_jsonl_data(
    input_path: str,
    output_path: str,
    mode: str = "prompt_safety",
    max_samples: int = None,
) -> str:
    """
    预处理 JSONL 数据集,转换为训练格式
    
    Args:
        input_path: 原始 JSONL 数据路径
        output_path: 处理后 JSON 数据保存路径
        mode: 训练模式
            - prompt_safety: 仅检测用户输入安全性
            - response_safety: 检测用户输入+模型响应的安全性
            - reasoning: 带解释的风险归因训练
        max_samples: 最大处理样本数(用于测试)
    
    Returns:
        保存路径
    """
    logger.info(f"预处理数据: {input_path} -> {output_path}, 模式: {mode}")
    
    processed_data = []
    total_count = 0
    skipped_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_samples and line_idx >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                raw_item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_idx+1} 行 JSON 解析失败: {e}")
                skipped_count += 1
                continue
            
            # 处理单条数据
            item = _process_item(raw_item, mode)
            
            if item is not None:
                processed_data.append(item)
                total_count += 1
            else:
                skipped_count += 1
    
    logger.info(f"处理完成: 总计 {total_count} 条, 跳过 {skipped_count} 条")
    
    # 保存处理后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据已保存: {output_path}")
    return output_path


def _process_item(raw_item: dict, mode: str) -> dict | None:
    """
    处理单条数据
    
    根据数据集的实际字段,提取 messages 和 label
    """
    # 提取字段
    prompt = raw_item.get("prompt")
    response = raw_item.get("response")
    label = raw_item.get("label")
    explanation = raw_item.get("explanation")
    sample_type = raw_item.get("sample_type", "general")
    stage = raw_item.get("stage", "q")
    
    # 根据数据类型和阶段判断如何构建训练样本
    # stage: "q" 表示仅用户输入, "qr" 表示用户输入+模型响应
    
    # 构建 messages
    messages = []
    
    # 添加用户输入
    if prompt and prompt.strip():
        messages.append({"role": "user", "content": prompt.strip()})
    elif response and response.strip() and stage == "qr":
        # 如果没有 prompt 但有 response,可能是纯响应评估
        # 这种情况下我们跳过,因为必须有要有用户输入
        return None
    else:
        # 既没有 prompt 也没有有效的 response
        return None
    
    # 根据模式添加 assistant 响应
    if mode in ("response_safety", "reasoning"):
        if response and response.strip():
            messages.append({"role": "assistant", "content": response.strip()})
    
    if not messages:
        return None
    
    # 构建训练样本
    item = {
        "messages": messages,
        "label": label if label else "sec",  # 默认安全
    }
    
    # reasoning 模式添加解释
    if mode == "reasoning" and explanation and explanation.strip():
        item["explanation"] = explanation.strip()
    
    return item


def main():
    parser = argparse.ArgumentParser(description="XGuard 训练数据预处理")
    parser.add_argument(
        "--input",
        type=str,
        default="data/XGuard-Train-Open-200K/xguard_train_open_200k.jsonl",
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/xguard_train.json",
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prompt_safety",
        choices=["prompt_safety", "response_safety", "reasoning"],
        help="训练模式",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大处理样本数(用于测试)",
    )
    args = parser.parse_args()
    
    # 处理数据
    preprocess_jsonl_data(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
