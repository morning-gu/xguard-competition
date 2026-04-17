"""模型训练模块

基于 HuggingFace Transformers + PEFT (LoRA) 进行微调训练。
支持全量微调和 LoRA 微调两种模式。
"""

import os
import json
import logging
from typing import Optional, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

from src.data.dataset import XGuardDataset
from src.data.preprocess import preprocess_for_training, load_prompt_template
from src.data.dataset import load_xguard_train_data

logger = logging.getLogger(__name__)


class TrainConfig:
    """训练配置"""

    # 模型配置
    base_model_path: str = "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"
    model_max_length: int = 2048

    # 数据配置
    data_source: str = "modelscope"  # "modelscope", "huggingface", "local"
    local_data_path: Optional[str] = None
    max_train_samples: Optional[int] = None
    prompt_template_path: Optional[str] = None

    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None  # None 表示自动选择

    # 训练超参数
    output_dir: str = "outputs/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.warning(f"未知配置项: {k}={v}")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def setup_lora(
    model: AutoModelForCausalLM,
    config: TrainConfig,
) -> AutoModelForCausalLM:
    """为模型添加 LoRA 适配器

    Args:
        model: 基座模型
        config: 训练配置

    Returns:
        添加 LoRA 后的模型
    """
    target_modules = config.lora_target_modules
    if target_modules is None:
        # Qwen3 架构默认 LoRA 目标模块
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize 处理函数"""
    inputs = examples["input"]
    outputs = examples["output"]

    full_texts = [inp + outp for inp, outp in zip(inputs, outputs)]

    model_inputs = tokenizer(
        full_texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    # 构建 labels: mask 掉 input 部分
    input_only = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    labels = []
    for i, (input_ids, input_len) in enumerate(
        zip(model_inputs["input_ids"], input_only["input_ids"])
    ):
        label = input_ids.copy()
        label[:len(input_len)] = [-100] * len(input_len)
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs


def train(config: Optional[TrainConfig] = None, **kwargs) -> str:
    """执行训练

    Args:
        config: 训练配置对象
        **kwargs: 配置参数 (用于覆盖默认配置)

    Returns:
        模型输出路径
    """
    if config is None:
        config = TrainConfig(**kwargs)

    logger.info(f"训练配置: {json.dumps(config.to_dict(), indent=2, ensure_ascii=False)}")

    # 1. 加载 tokenizer 和模型
    logger.info(f"加载基座模型: {config.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. 添加 LoRA
    if config.use_lora:
        logger.info("添加 LoRA 适配器...")
        model = setup_lora(model, config)

    # 3. 启用 gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # 4. 加载和预处理数据
    logger.info("加载训练数据...")
    raw_data = load_xguard_train_data(
        source=config.data_source,
        local_path=config.local_data_path,
        max_samples=config.max_train_samples,
    )

    # 加载 prompt template
    instruction_template = None
    if config.prompt_template_path:
        instruction_template = load_prompt_template(config.prompt_template_path)

    processed_data = preprocess_for_training(
        raw_data,
        instruction_template=instruction_template,
    )

    # 5. Tokenize
    logger.info("Tokenize 训练数据...")
    from datasets import Dataset as HFDataset
    hf_dataset = HFDataset.from_list(processed_data)

    tokenized_dataset = hf_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.model_max_length),
        batched=True,
        remove_columns=hf_dataset.column_names,
        desc="Tokenizing",
    )

    # 6. 配置训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # 7. 数据 collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=config.model_max_length,
    )

    # 8. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info("开始训练...")
    trainer.train()

    # 9. 保存模型
    logger.info(f"保存模型到: {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # 保存训练配置
    with open(os.path.join(config.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("训练完成!")
    return config.output_dir
