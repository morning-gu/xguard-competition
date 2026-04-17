#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载模块

提供统一的模型加载接口，支持：
- 自动从 ModelScope 下载模型到本地缓存
- 支持量化加载（4-bit/8-bit）以节省显存
"""

import os
from pathlib import Path
from typing import Optional, Tuple

# 在导入 transformers 之前设置 ModelScope 缓存目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
cache_dir = PROJECT_ROOT / "models" / "pretrained"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['MODELSCOPE_CACHE'] = str(cache_dir)

# 设置 HuggingFace 镜像为 ModelScope，防止 transformers 库回退到 HF
os.environ['HF_ENDPOINT'] = 'https://modelscope.cn'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logger.warning("bitsandbytes 未安装，量化加载不可用")

# 缓存目录设置（仅在首次导入时以 debug 级别输出，避免多进程数据加载时刷屏）
logger.debug(f"ModelScope 缓存目录设置为: {cache_dir}")
logger.debug(f"HF_ENDPOINT 设置为：{os.environ.get('HF_ENDPOINT')}")


# 默认模型配置
DEFAULT_MODEL_ID = "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"
MODEL_VERSIONS = {
    "0.6B": "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B",
    "8B": "Alibaba-AAIG/YuFeng-XGuard-Reason-8B",
}


def download_model(model_id: str = None, model_version: str = "0.6B", cache_dir: str = None) -> str:
    """
    下载模型到本地缓存
    
    Args:
        model_id: 完整模型 ID (可选，如指定则覆盖 model_version)
        model_version: 模型版本 ("0.6B" 或 "8B")
        cache_dir: 缓存目录路径，默认为当前项目下的 models 目录
    
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
    
    logger.info(f"正在下载模型：{final_model_id}")
    
    # 设置缓存目录为当前项目下的 models 目录
    if cache_dir is None:
        # 获取项目根目录
        project_root = Path(__file__).resolve().parents[2]
        cache_dir = project_root / "models" / "pretrained"
    
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"模型将下载到：{cache_dir}")
    
    # 下载模型到指定目录（使用 local_dir 参数）
    # 注意：snapshot_download 会自动处理 ModelScope 到本地路径的映射
    # transformers 加载时使用本地路径，不会再次联网
    logger.info(f"正在从 ModelScope 下载模型：{final_model_id}")
    model_dir = snapshot_download(
        final_model_id,
        revision="master",
        local_dir=str(cache_dir / final_model_id.replace("/", "_"))
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
    use_flash_attention_2: bool = True,
    set_eval_mode: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径，可以是：
            - 本地绝对路径
            - ModelScope 模型 ID
            - None (使用默认模型)
        model_version: 模型版本 (仅当 model_path=None 时使用)
        use_4bit: 是否使用 4-bit 量化加载
        use_8bit: 是否使用 8-bit 量化加载
        trust_remote_code: 是否信任远程代码
        device_map: 设备映射策略
        use_flash_attention_2: 是否使用 Flash Attention 2
        set_eval_mode: 是否设置为评估模式（训练时应设为 False）
    
    Returns:
        (model, tokenizer) 元组
    """
    # 解析模型路径
    if model_path is None:
        # 使用默认模型 - 先尝试从 ModelScope 下载到本地，然后使用本地路径
        model_id = MODEL_VERSIONS.get(model_version, DEFAULT_MODEL_ID)
        logger.info(f"使用默认模型：{model_id}")
        # 检查本地缓存是否存在
        project_root = Path(__file__).resolve().parents[2]
        local_cache_dir = project_root / "models" / "pretrained"
        local_model_path = local_cache_dir / model_id.replace("/", "_")
        
        if local_model_path.exists():
            logger.info(f"使用本地缓存模型：{local_model_path}")
            model_path = str(local_model_path)
        else:
            # 需要下载模型
            logger.info(f"本地缓存不存在，正在从 ModelScope 下载...")
            model_path = download_model(model_id=model_id, cache_dir=str(local_cache_dir))
    elif "/" in model_path and not os.path.isdir(model_path):
        # 看起来是模型 ID 格式 - 先下载到本地再加载
        logger.info(f"检测到 ModelScope 模型 ID：{model_path}")
        logger.info("为避免访问 HuggingFace，将先从 ModelScope 下载到本地...")
        project_root = Path(__file__).resolve().parents[2]
        local_cache_dir = project_root / "models" / "pretrained"
        model_path = download_model(model_id=model_path, cache_dir=str(local_cache_dir))
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
    if use_flash_attention_2:
        try:
            from flash_attn import flash_attn_func
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("使用 Flash Attention 2")
        except ImportError:
            logger.debug("Flash Attention 2 未安装，使用默认注意力实现")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    # 根据参数决定是否设置为评估模式
    if set_eval_mode:
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
