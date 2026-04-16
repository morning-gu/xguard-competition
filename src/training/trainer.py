#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块（兼容层）

提供统一的模型训练接口，所有核心逻辑委托给 train_core 模块。
保留此文件以保持向后兼容的导入路径。
"""

import yaml
from typing import Optional, List
from loguru import logger

from .train_core import (
    load_config,
    merge_config,
    load_model_and_tokenizer_for_training as load_model_and_tokenizer,
    apply_lora,
    prepare_data,
    build_training_args,
    train,
)


def train_from_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> str:
    """
    从配置文件启动训练

    Args:
        config_path: 配置文件路径
        overrides: 配置覆盖列表 (格式：["key.subkey=value"])

    Returns:
        最终模型保存路径
    """
    config = load_config(config_path)

    if overrides:
        config = merge_config(config, overrides)

    logger.info(f"配置：{yaml.dump(config, default_flow_style=False)}")

    return train(config)
