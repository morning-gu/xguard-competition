"""模型加载模块

支持加载:
- YuFeng-XGuard-Reason-0.6B 基座模型
- 微调后的自定义模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple


def load_model_and_tokenizer(
    model_path: str,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    use_flash_attention: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型和 tokenizer

    Args:
        model_path: 模型路径 (本地路径或 HuggingFace/ModelScope model ID)
        torch_dtype: 数据类型, "auto", "float16", "bfloat16" 等
        device_map: 设备映射策略
        use_flash_attention: 是否使用 Flash Attention 2

    Returns:
        (model, tokenizer) 元组
    """
    # 解析 torch_dtype
    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = "auto"

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 加载模型
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model = model.eval()

    return model, tokenizer


def load_model_for_training(
    model_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """加载模型用于训练 (非 eval 模式)

    Args:
        model_path: 模型路径
        torch_dtype: 数据类型
        device_map: 设备映射策略

    Returns:
        模型实例 (train 模式)
    """
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    return model
