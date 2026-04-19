"""XGuard 数据集加载模块

支持加载:
- XGuard-Train-Open-200K 训练数据集 (从 ModelScope/HuggingFace)
- XGuard-Test-Open-1k 测试数据集 (本地 jsonl)
"""

import json
import os
from typing import List, Dict, Optional, Iterator
from torch.utils.data import Dataset


class XGuardDataset(Dataset):
    """XGuard 训练数据集的 PyTorch Dataset 封装"""

    def __init__(self, data: List[Dict], tokenizer=None, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.tokenizer is not None:
            return self._tokenize(item)
        return item

    def _tokenize(self, item: Dict):
        """对单条数据进行 tokenize"""
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # 拼接 input 和 output
        full_text = input_text + output_text

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # 构建 labels: input 部分用 -100 mask 掉
        input_encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_len = len(input_encodings["input_ids"])

        labels = encodings["input_ids"].copy()
        labels[:input_len] = [-100] * input_len

        encodings["labels"] = labels
        return encodings


def load_xguard_train_data(
    source: str = "modelscope",
    local_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """加载 XGuard-Train-Open-200K 训练数据集

    Args:
        source: 数据来源, "modelscope" 或 "huggingface" 或 "local"
        local_path: 本地数据路径 (source="local" 时使用)
        max_samples: 最大加载样本数 (用于调试)

    Returns:
        原始数据列表
    """
    # 优先使用本地路径 (即使 source 不是 "local", 只要指定了 local_path 就从本地加载)
    if local_path and os.path.exists(local_path):
        print(f"从本地加载数据: {local_path}")
        return _load_from_jsonl(local_path, max_samples)

    if source == "local" and local_path:
        return _load_from_jsonl(local_path, max_samples)

    if source == "modelscope":
        try:
            from modelscope.msdatasets import MsDataset
            raw_dataset = MsDataset.load(
                "Alibaba-AAIG/XGuard-Train-Open-200K",
                subset_name="xguard_train_open_200k",
            )
            data = []
            for item in raw_dataset["train"]:
                data.append(dict(item))
                if max_samples and len(data) >= max_samples:
                    break
            return data
        except Exception as e:
            print(f"从 ModelScope 加载失败: {e}")
            print("尝试从 HuggingFace 加载...")
            # 不要修改 source 变量, 直接执行 HuggingFace 加载

    # HuggingFace 加载 (作为 fallback 或首选)
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset(
            "Alibaba-AAIG/XGuard-Train-Open-200K",
            "xguard_train_open_200k",
            split="train",
            trust_remote_code=True,  # 信任远程代码, 解决 LargeList 等类型问题
        )
        data = []
        for item in raw_dataset:
            data.append(dict(item))
            if max_samples and len(data) >= max_samples:
                break
        return data
    except Exception as e:
        print(f"从 HuggingFace 加载失败: {e}")
        raise ValueError(f"无法从 ModelScope 或 HuggingFace 加载数据集。请使用本地数据: --local_data_path <path>")


def load_xguard_test_data(
    test_path: str = "test_dataset/xguard_test_open_1k.jsonl",
) -> List[Dict]:
    """加载 XGuard 测试数据集

    Args:
        test_path: 测试数据 jsonl 文件路径

    Returns:
        测试数据列表
    """
    return _load_from_jsonl(test_path)


def _load_from_jsonl(
    filepath: str, max_samples: Optional[int] = None
) -> List[Dict]:
    """从 jsonl 文件加载数据"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break
    return data
