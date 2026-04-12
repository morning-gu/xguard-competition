#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard 模型微调训练脚本

基于 XGuard-Train-Open-200K 数据集，对 YuFeng-XGuard-Reason 模型进行微调训练。
支持全参数微调和 LoRA 参数高效微调。

使用方法:
    # 使用默认配置训练
    python src/training/train.py

    # 指定配置文件
    python src/training/train.py --config configs/train_config.yaml

    # 覆盖部分配置
    python src/training/train.py --config configs/train_config.yaml --training.num_train_epochs 5
"""

import os
import sys
import yaml
import argparse
from typing import Optional
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from loguru import logger

# LoRA 支持
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft 未安装，LoRA 微调不可用。请安装: pip install peft")

# 量化支持
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import XGuardTrainDataset, preprocess_raw_data, load_and_preprocess


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def merge_config(config: dict, overrides: list[str]) -> dict:
    """
    合并命令行覆盖到配置

    用法: --key.subkey value
    例如: --training.num_train_epochs 5
    """
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
        else:
            # 下一个参数是值的情况由 argparse 处理
            continue

        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        # 尝试转换类型
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.lower() == "null":
                    value = None
        d[keys[-1]] = value

    return config


def _resolve_model_path(model_name: str) -> str:
    """
    解析模型路径：如果传入的是 HuggingFace 模型 ID 格式（包含 /），
    则尝试在本地 models/pretrained 目录下查找对应的本地路径。

    Args:
        model_name: 模型名称或路径

    Returns:
        解析后的模型路径（本地绝对路径或原始模型 ID）
    """
    # 如果已经是本地绝对路径且目录存在，直接返回
    if os.path.isdir(model_name):
        return model_name

    # 尝试在项目根目录下查找本地模型
    project_root = Path(__file__).resolve().parents[2]  # src/training/ -> 项目根目录
    local_path = project_root / "models/pretrained" / model_name
    if local_path.is_dir():
        logger.info(f"使用本地模型路径: {local_path}")
        return str(local_path)

    # 本地未找到，回退到远程模型 ID
    logger.info(f"本地未找到模型，将使用远程模型 ID: {model_name}")
    return model_name


def load_model_and_tokenizer(config: dict):
    """
    加载基座模型和分词器

    支持全精度、4-bit、8-bit 量化加载
    """
    model_cfg = config["model"]
    model_name = model_cfg["base_model"]

    # 解析本地模型路径
    model_name = _resolve_model_path(model_name)

    logger.info(f"加载基座模型: {model_name}")

    # 量化配置
    quantization_config = None
    if model_cfg.get("use_4bit") and BNB_AVAILABLE:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("使用 4-bit 量化加载")
    elif model_cfg.get("use_8bit") and BNB_AVAILABLE:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("使用 8-bit 量化加载")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        padding_side="right",
    )

    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model_kwargs = {
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, tokenizer


def apply_lora(model, config: dict):
    """
    对模型应用 LoRA 适配器

    Args:
        model: 基座模型
        config: 配置字典

    Returns:
        包装后的 PEFT 模型
    """
    if not PEFT_AVAILABLE:
        raise ImportError("LoRA 需要 peft 库，请安装: pip install peft")

    lora_cfg = config["lora"]
    logger.info(f"应用 LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def prepare_data(config: dict, tokenizer: AutoTokenizer):
    """
    准备训练和验证数据集

    Returns:
        train_dataset, eval_dataset
    """
    data_cfg = config["data"]

    raw_data_path = data_cfg["raw_data_path"]
    if not os.path.exists(raw_data_path):
        logger.error(f"原始数据不存在: {raw_data_path}")
        logger.info("请先运行数据下载脚本: python scripts/download_dataset.py")
        sys.exit(1)

    # 预处理数据
    mode = data_cfg["mode"]
    processed_path = os.path.join(
        data_cfg["processed_dir"], f"xguard_{mode}.json"
    )

    if not os.path.exists(processed_path):
        preprocess_raw_data(raw_data_path, processed_path, mode)

    # 加载完整数据集
    full_dataset = XGuardTrainDataset(
        data_path=processed_path,
        tokenizer=tokenizer,
        max_length=data_cfg["max_length"],
        mode=mode,
    )

    # 划分训练集和验证集
    train_ratio = data_cfg.get("train_ratio", 0.9)
    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)

    torch.manual_seed(data_cfg.get("seed", 42))
    indices = torch.randperm(total_len).tolist()

    train_dataset = torch.utils.data.Subset(full_dataset, indices[:train_len])
    eval_dataset = torch.utils.data.Subset(full_dataset, indices[train_len:])

    logger.info(f"训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条")

    return train_dataset, eval_dataset


def build_training_args(config: dict) -> TrainingArguments:
    """构建 TrainingArguments"""
    train_cfg = config["training"]
    eval_cfg = config.get("evaluation", {})

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=eval_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    return args


def train(config: dict):
    """
    执行训练流程

    1. 加载模型和分词器
    2. (可选) 应用 LoRA
    3. 准备数据集
    4. 配置训练参数
    5. 启动训练
    6. 保存模型
    """
    logger.info("=" * 60)
    logger.info("XGuard 模型微调训练")
    logger.info("=" * 60)

    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(config)

    # 2. 应用 LoRA (如果启用)
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        model = apply_lora(model, config)

    # 3. 准备数据集
    train_dataset, eval_dataset = prepare_data(config, tokenizer)

    # 4. 构建训练参数
    training_args = build_training_args(config)

    # 5. 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config["data"]["max_length"],
        return_tensors="pt",
    )

    # 6. 早停回调
    callbacks = []
    patience = config["training"].get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    # 7. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # 8. 启动训练
    logger.info("开始训练...")
    trainer.train()

    # 9. 保存最终模型
    output_dir = config["training"]["output_dir"]
    final_dir = os.path.join(output_dir, "final")
    logger.info(f"保存最终模型到: {final_dir}")

    if lora_cfg.get("enabled", False):
        # LoRA: 只保存适配器权重
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info("LoRA 适配器已保存")
    else:
        # 全参数: 保存完整模型
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

    logger.info("训练完成!")
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="XGuard 模型微调训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="配置覆盖 (格式: key.subkey=value)",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    if args.overrides:
        config = merge_config(config, args.overrides)

    logger.info(f"配置: {yaml.dump(config, default_flow_style=False)}")

    # 执行训练
    train(config)


if __name__ == "__main__":
    main()
