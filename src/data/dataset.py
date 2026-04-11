#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard 训练数据集处理模块

将 XGuard-Train-Open-200K 原始数据转换为模型微调所需的格式:
- 构建对话消息 (messages)
- 应用 chat_template 生成训练输入
- 支持不同训练模式: prompt_safety / response_safety / reasoning
"""

import os
import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from loguru import logger


# 风险类别 ID 到名称的映射
RISK_ID_MAP = {
    "sec": "Safe-Safe",
    "pc": "Crimes and Illegal Activities-Pornographic Contraband",
    "dc": "Crimes and Illegal Activities-Drug Crimes",
    "dw": "Crimes and Illegal Activities-Dangerous Weapons",
    "pi": "Crimes and Illegal Activities-Property Infringement",
    "ec": "Crimes and Illegal Activities-Economic Crimes",
    "ac": "Hate Speech-Abusive Curses",
    "def": "Hate Speech-Defamation",
    "ti": "Hate Speech-Threats and Intimidation",
    "cy": "Hate Speech-Cyberbullying",
    "ph": "Physical and Mental Health-Physical Health",
    "mh": "Physical and Mental Health-Mental Health",
    "se": "Ethics and Morality-Social Ethics",
    "sci": "Ethics and Morality-Science Ethics",
    "pp": "Data Privacy-Personal Privacy",
    "cs": "Data Privacy-Commercial Secret",
    "acc": "Cybersecurity-Access Control",
    "mc": "Cybersecurity-Malicious Code",
    "ha": "Cybersecurity-Hacker Attack",
    "ps": "Cybersecurity-Physical Security",
    "ter": "Extremism-Violent Terrorist Activities",
    "sd": "Extremism-Social Disruption",
    "ext": "Extremism-Extremist Ideological Trends",
    "fin": "Inappropriate Suggestions-Finance",
    "med": "Inappropriate Suggestions-Medicine",
    "law": "Inappropriate Suggestions-Law",
    "cm": "Risks Involving Minors-Corruption of Minors",
    "ma": "Risks Involving Minors-Minor Abuse and Exploitation",
    "md": "Risks Involving Minors-Minor Delinquency",
}


class XGuardTrainDataset(Dataset):
    """
    XGuard 训练数据集

    将原始数据转换为模型可训练的格式:
    - 输入: chat_template 渲染后的文本
    - 标签: 风险类别 token 的 ID
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        mode: str = "prompt_safety",
    ):
        """
        Args:
            data_path: 处理后的数据文件路径 (JSON 格式)
            tokenizer: 分词器
            max_length: 最大序列长度
            mode: 训练模式
                - prompt_safety: 仅检测用户输入安全性
                - response_safety: 检测用户输入+模型响应的安全性
                - reasoning: 带解释的风险归因训练
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        logger.info(f"加载数据: {data_path}, 模式: {mode}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logger.info(f"数据量: {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        label = item.get("label", "sec")

        # 应用 chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # 构建标签: 对于因果语言模型训练，标签 = input_ids (shift 在模型内部完成)
        # 但我们只对生成部分计算 loss，所以将 prompt 部分的标签设为 -100
        labels = [-100] * len(input_ids)

        # 获取标签 token ID
        if label in self.tokenizer.get_vocab():
            label_token_id = self.tokenizer.get_vocab()[label]
        else:
            label_token_id = -100

        # 在序列末尾添加标签 token
        input_ids.append(label_token_id)
        attention_mask.append(1)
        labels.append(label_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def preprocess_raw_data(
    input_path: str,
    output_path: str,
    mode: str = "prompt_safety",
) -> str:
    """
    预处理原始数据集，转换为训练格式

    Args:
        input_path: 原始 CSV 数据路径
        output_path: 处理后 JSON 数据保存路径
        mode: 训练模式

    Returns:
        保存路径
    """
    logger.info(f"预处理数据: {input_path} -> {output_path}, 模式: {mode}")

    df = pd.read_csv(input_path)
    logger.info(f"原始数据量: {len(df)} 条, 列: {list(df.columns)}")

    processed_data = []

    for _, row in df.iterrows():
        item = _process_row(row, mode)
        if item is not None:
            processed_data.append(item)

    logger.info(f"处理后数据量: {len(processed_data)} 条")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    logger.info(f"数据已保存: {output_path}")
    return output_path


def _process_row(row: pd.Series, mode: str) -> dict | None:
    """
    处理单条数据

    根据数据集的实际列名，提取 messages 和 label
    支持多种可能的列名格式
    """
    item = {}

    # 尝试从不同列名中提取数据
    # 常见列名: prompt, query, user_input, instruction
    user_content = (
        row.get("prompt")
        or row.get("query")
        or row.get("user_input")
        or row.get("instruction")
        or row.get("input")
    )

    # 常见列名: response, answer, assistant_response, output
    assistant_content = (
        row.get("response")
        or row.get("answer")
        or row.get("assistant_response")
        or row.get("output")
    )

    # 常见列名: label, risk_label, risk_category, category
    label = (
        row.get("label")
        or row.get("risk_label")
        or row.get("risk_category")
        or row.get("category")
        or row.get("risk_type")
    )

    # 常见列名: explanation, reason, reasoning
    explanation = (
        row.get("explanation")
        or row.get("reason")
        or row.get("reasoning")
    )

    if user_content is None or pd.isna(user_content):
        return None

    # 构建 messages
    messages = [{"role": "user", "content": str(user_content)}]

    if mode in ("response_safety", "reasoning") and assistant_content and not pd.isna(assistant_content):
        messages.append({"role": "assistant", "content": str(assistant_content)})

    item["messages"] = messages

    # 设置标签
    if label and not pd.isna(label):
        item["label"] = str(label)
    else:
        item["label"] = "sec"  # 默认安全

    # 添加解释 (reasoning 模式)
    if mode == "reasoning" and explanation and not pd.isna(explanation):
        item["explanation"] = str(explanation)

    return item


def load_and_preprocess(
    raw_data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    mode: str = "prompt_safety",
    cache_dir: str = "data/processed",
) -> XGuardTrainDataset:
    """
    一站式数据加载: 预处理 + 构建数据集

    Args:
        raw_data_path: 原始 CSV 数据路径
        tokenizer: 分词器
        max_length: 最大序列长度
        mode: 训练模式
        cache_dir: 预处理数据缓存目录

    Returns:
        XGuardTrainDataset 实例
    """
    # 预处理数据
    processed_path = os.path.join(
        cache_dir, f"xguard_{mode}.json"
    )

    if not os.path.exists(processed_path):
        preprocess_raw_data(raw_data_path, processed_path, mode)

    # 构建数据集
    dataset = XGuardTrainDataset(
        data_path=processed_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode=mode,
    )

    return dataset
