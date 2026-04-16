#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载模块

提供统一的模型加载接口，支持：
- 自动从 ModelScope/HuggingFace 下载模型到本地缓存
- 从本地缓存加载模型
- 支持量化加载（4-bit/8-bit）以节省显存
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logger.warning("bitsandbytes 未安装，量化加载不可用")


# 默认模型配置
DEFAULT_MODEL_ID = "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"
MODEL_VERSIONS = {
    "0.6B": "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B",
    "8B": "Alibaba-AAIG/YuFeng-XGuard-Reason-8B",
}


def get_model_cache_dir(model_id: str) -> Path:
    """
    获取模型本地缓存目录
    
    Args:
        model_id: 模型 ID (如 Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B)
    
    Returns:
        本地缓存目录路径
    """
    project_root = Path(__file__).resolve().parents[2]  # src/model/ -> 项目根目录
    model_name = model_id.split("/")[-1]
    org_name = model_id.split("/")[0] if "/" in model_id else ""
    
    if org_name:
        cache_dir = project_root / "models" / "pretrained" / org_name / model_name
    else:
        cache_dir = project_root / "models" / "pretrained" / model_name
    
    return cache_dir


def download_model(model_id: str = None, model_version: str = "0.6B") -> str:
    """
    下载模型到本地缓存
    
    Args:
        model_id: 完整模型 ID (可选，如指定则覆盖 model_version)
        model_version: 模型版本 ("0.6B" 或 "8B")
    
    Returns:
        模型本地路径
    """
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError("请安装 modelscope: pip install modelscope")
    
    # 确定模型 ID
    if model_id:
        final_model_id = model_id
    elif model_version in MODEL_VERSIONS:
        final_model_id = MODEL_VERSIONS[model_version]
    else:
        raise ValueError(f"不支持的模型版本：{model_version}, 支持的版本：{list(MODEL_VERSIONS.keys())}")
    
    # 获取本地缓存目录
    local_dir = get_model_cache_dir(final_model_id)
    
    logger.info(f"正在下载模型：{final_model_id}")
    logger.info(f"本地保存路径：{local_dir}")
    
    # 确保目标目录存在
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载模型
    model_dir = snapshot_download(
        final_model_id,
        local_dir=str(local_dir),
        revision="master"
    )
    
    logger.info(f"模型下载完成！路径：{model_dir}")
    return model_dir


def load_model_and_tokenizer(
    model_path: str = None,
    model_version: str = "0.6B",
    use_4bit: bool = False,
    use_8bit: bool = False,
    trust_remote_code: bool = True,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径，可以是：
            - 本地绝对路径
            - HuggingFace/ModelScope 模型 ID
            - None (使用默认模型，自动检查本地缓存或下载)
        model_version: 模型版本 (仅当 model_path=None 时使用)
        use_4bit: 是否使用 4-bit 量化加载
        use_8bit: 是否使用 8-bit 量化加载
        trust_remote_code: 是否信任远程代码
        device_map: 设备映射策略
    
    Returns:
        (model, tokenizer) 元组
    """
    # 解析模型路径
    if model_path is None:
        # 使用默认模型，检查本地缓存
        model_id = MODEL_VERSIONS.get(model_version, DEFAULT_MODEL_ID)
        local_cache = get_model_cache_dir(model_id)
        
        if local_cache.exists():
            logger.info(f"使用本地缓存模型：{local_cache}")
            model_path = str(local_cache)
        else:
            logger.info(f"本地缓存不存在，开始下载模型：{model_id}")
            model_path = download_model(model_id=model_id)
    elif "/" in model_path and not os.path.isdir(model_path):
        # 看起来是模型 ID 格式，检查本地缓存
        local_cache = get_model_cache_dir(model_path)
        if local_cache.exists():
            logger.info(f"使用本地缓存模型：{local_cache}")
            model_path = str(local_cache)
        else:
            logger.info(f"本地缓存不存在，开始下载模型：{model_path}")
            model_path = download_model(model_id=model_path)
    else:
        # 本地路径
        logger.info(f"使用本地模型：{model_path}")
    
    logger.info(f"加载模型：{model_path}")
    
    # 构建量化配置
    quantization_config = None
    if use_4bit and BNB_AVAILABLE:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("使用 4-bit 量化加载")
    elif use_8bit and BNB_AVAILABLE:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("使用 8-bit 量化加载")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"分词器加载完成，词表大小：{len(tokenizer)}")
    
    # 加载模型
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["device_map"] = device_map
    
    # Flash Attention 2 支持
    try:
        from flash_attn import flash_attn_func
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("使用 Flash Attention 2")
    except ImportError:
        logger.debug("Flash Attention 2 未安装，使用默认注意力实现")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型加载完成，参数量：{param_count / 1e9:.2f}B")
    
    return model, tokenizer


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("模型加载模块示例")
    print("=" * 60)
    
    # 示例 1: 使用默认模型（自动检查缓存或下载）
    print("\n示例 1: 加载默认模型")
    model, tokenizer = load_model_and_tokenizer()
    print(f"模型类型：{type(model).__name__}")
    print(f"分词器词表大小：{len(tokenizer)}")
    
    # 示例 2: 指定模型版本
    print("\n示例 2: 加载 0.6B 模型")
    model, tokenizer = load_model_and_tokenizer(model_version="0.6B")
    
    # 示例 3: 使用量化加载
    print("\n示例 3: 使用 4-bit 量化加载")
    if BNB_AVAILABLE:
        model, tokenizer = load_model_and_tokenizer(use_4bit=True)
    else:
        print("bitsandbytes 未安装，跳过量化加载示例")
