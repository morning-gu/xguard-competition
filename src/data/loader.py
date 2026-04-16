#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集加载模块

提供统一的数据集加载接口，支持：
- 从 ModelScope 下载训练数据集到本地缓存
- 加载本地缓存的训练数据
- 加载测试集/验证集
- 数据预处理和格式转换
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from loguru import logger


# 默认数据集配置
DEFAULT_TRAIN_DATASET_ID = "Alibaba-AAIG/XGuard-Train-Open-200K"
DEFAULT_TEST_DATASET_PATH = "data/test_dataset/xguard_test_open_1k.jsonl"

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


def get_dataset_cache_dir() -> Path:
    """获取数据集本地缓存目录"""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "XGuard-Train-Open-200K"


def download_train_dataset(dataset_id: str = None) -> str:
    """
    下载训练数据集到本地缓存
    
    Args:
        dataset_id: 数据集 ID (可选，默认使用 XGuard-Train-Open-200K)
    
    Returns:
        数据集本地路径
    """
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        raise ImportError("请安装 modelscope: pip install modelscope")
    
    dataset_id = dataset_id or DEFAULT_TRAIN_DATASET_ID
    local_dir = get_dataset_cache_dir()
    output_file = local_dir / "xguard_train_open_200k.jsonl"
    
    logger.info(f"正在下载数据集：{dataset_id}")
    logger.info(f"本地保存路径：{local_dir}")
    
    # 创建下载目录
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用 ModelScope SDK 加载数据集
    logger.info("[1/3] 正在加载数据集...")
    ds = MsDataset.load(dataset_id, split='train')
    logger.info(f"[OK] 数据集加载成功！大小：{len(ds)} 条数据")
    
    # 保存数据到本地文件
    logger.info(f"[2/3] 正在保存数据到：{output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(ds):
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
            if (idx + 1) % 10000 == 0:
                logger.info(f"  已处理 {idx + 1} 条数据...")
    
    logger.info(f"[OK] 数据保存完成！总计保存 {len(ds)} 条数据")
    logger.info(f"数据集已下载到：{local_dir}")
    
    return str(local_dir)


def load_train_dataset(
    data_path: str = None,
    auto_download: bool = True,
) -> List[Dict]:
    """
    加载训练数据集
    
    Args:
        data_path: 数据文件路径 (JSONL 格式)，None 则使用默认路径
        auto_download: 如果数据不存在是否自动下载
    
    Returns:
        数据列表，每项为一个字典
    """
    if data_path is None:
        local_dir = get_dataset_cache_dir()
        data_path = local_dir / "xguard_train_open_200k.jsonl"
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        if auto_download:
            logger.info(f"数据文件不存在：{data_path}，开始自动下载")
            download_train_dataset()
        else:
            raise FileNotFoundError(f"数据文件不存在：{data_path}")
    
    logger.info(f"加载训练数据：{data_path}")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    logger.info(f"训练数据加载完成，共 {len(data)} 条")
    return data


def load_test_dataset(test_path: str = None) -> List[Dict]:
    """
    加载测试集/验证集
    
    Args:
        test_path: 测试集文件路径 (JSONL 格式)
    
    Returns:
        数据列表，每项为一个字典
    """
    project_root = Path(__file__).resolve().parents[2]
    
    if test_path is None:
        test_path = project_root / DEFAULT_TEST_DATASET_PATH
    
    test_path = Path(test_path)
    
    if not test_path.exists():
        raise FileNotFoundError(f"测试集文件不存在：{test_path}")
    
    logger.info(f"加载测试数据：{test_path}")
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    logger.info(f"测试数据加载完成，共 {len(data)} 条")
    return data


class XGuardTrainDataset(Dataset):
    """
    XGuard 训练数据集
    
    将原始数据转换为模型可训练的格式:
    - 输入：chat_template 渲染后的文本
    - 标签：风险类别 token 的 ID
    """

    def __init__(
        self,
        data: Union[List[Dict], str],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        mode: str = "prompt_safety",
    ):
        """
        Args:
            data: 数据列表或数据文件路径 (JSON 格式)
            tokenizer: 分词器
            max_length: 最大序列长度
            mode: 训练模式
                - prompt_safety: 仅检测用户输入安全性
                - response_safety: 检测用户输入 + 模型响应的安全性
                - reasoning: 带解释的风险归因训练
                - all: 训练所有样本，根据 stage 自动判断输入格式
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # 加载数据
        if isinstance(data, str):
            logger.info(f"加载数据：{data}, 模式：{mode}")
            with open(data, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            logger.info(f"使用传入的数据列表，模式：{mode}")
            self.data = data
        
        logger.info(f"数据量：{len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 如果是原始数据，先处理成训练格式
        if "messages" not in item:
            processed = _process_raw_item(item, self.mode)
            if processed is None:
                # 返回一个空样本（会被 collator 过滤）
                return self.__getitem__((idx + 1) % len(self.data))
            messages = processed["messages"]
            label = processed.get("label", "sec")
        else:
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
            max_length=self.max_length - 1,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # 构建标签
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


def _process_raw_item(raw_item: dict, mode: str) -> Optional[dict]:
    """
    处理原始数据项
    
    Args:
        raw_item: 原始数据项
        mode: 训练模式
    
    Returns:
        处理后的数据项，包含 messages 和 label
    """
    prompt = raw_item.get("prompt")
    response = raw_item.get("response")
    label = raw_item.get("label")
    explanation = raw_item.get("explanation")
    stage = raw_item.get("stage", "q")
    
    messages = []
    
    if mode == "all":
        # all 模式：根据 stage 自动判断输入格式
        if stage == "q":
            if prompt and prompt.strip():
                messages.append({"role": "user", "content": prompt.strip()})
            else:
                return None
        elif stage == "r":
            if response and response.strip():
                messages.append({"role": "assistant", "content": response.strip()})
            else:
                return None
        elif stage == "qr":
            if prompt and prompt.strip():
                messages.append({"role": "user", "content": prompt.strip()})
            if response and response.strip():
                messages.append({"role": "assistant", "content": response.strip()})
            if not messages:
                return None
        else:
            return None
    else:
        # 原有逻辑
        if prompt and prompt.strip():
            messages.append({"role": "user", "content": prompt.strip()})
        elif response and response.strip() and stage == "qr":
            return None
        else:
            return None
        
        if mode in ("response_safety", "reasoning"):
            if response and response.strip():
                messages.append({"role": "assistant", "content": response.strip()})
    
    if not messages:
        return None
    
    item = {
        "messages": messages,
        "label": label if label else "sec",
    }
    
    if mode == "reasoning" and explanation and explanation.strip():
        item["explanation"] = explanation.strip()
    
    return item


def preprocess_data(
    input_path: str,
    output_path: str,
    mode: str = "prompt_safety",
) -> str:
    """
    预处理原始数据集，转换为训练格式
    
    Args:
        input_path: 原始数据路径 (JSONL 格式)
        output_path: 处理后 JSON 数据保存路径
        mode: 训练模式
    
    Returns:
        保存路径
    """
    logger.info(f"预处理数据：{input_path} -> {output_path}, 模式：{mode}")
    
    processed_data = []
    total_count = 0
    skipped_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                raw_item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_idx+1} 行 JSON 解析失败：{e}")
                skipped_count += 1
                continue
            
            item = _process_raw_item(raw_item, mode)
            if item is not None:
                processed_data.append(item)
                total_count += 1
            else:
                skipped_count += 1
    
    logger.info(f"原始数据量：{total_count} 条，跳过 {skipped_count} 条")
    logger.info(f"处理后数据量：{len(processed_data)} 条")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据已保存：{output_path}")
    return output_path


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """
    创建优化的数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批大小
        num_workers: 数据加载线程数
        prefetch_factor: 预取因子
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader 实例
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        shuffle=shuffle,
    )
    
    return dataloader


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("数据集加载模块示例")
    print("=" * 60)
    
    # 示例 1: 下载并加载训练数据
    print("\n示例 1: 下载训练数据集")
    try:
        data_dir = download_train_dataset()
        print(f"数据集下载到：{data_dir}")
    except Exception as e:
        print(f"下载失败：{e}")
    
    # 示例 2: 加载训练数据
    print("\n示例 2: 加载训练数据")
    data = load_train_dataset(auto_download=True)
    print(f"加载了 {len(data)} 条训练数据")
    if data:
        print(f"第一条数据预览：{list(data[0].keys())}")
    
    # 示例 3: 加载测试数据
    print("\n示例 3: 加载测试数据")
    try:
        test_data = load_test_dataset()
        print(f"加载了 {len(test_data)} 条测试数据")
    except FileNotFoundError as e:
        print(f"测试集不存在：{e}")
