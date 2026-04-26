"""模型训练模块

基于 HuggingFace Transformers + PEFT (LoRA) 进行微调训练。
支持全量微调和 LoRA 微调两种模式。
"""

import os
import json
import logging
from typing import Optional, Dict, Any

import torch

# 修复: PyTorch 2.6+ 默认 weights_only=True, 但 HuggingFace Trainer
# 保存的 rng_state.pth 包含 numpy/_codecs 等非默认安全全局对象,
# 导致从 checkpoint 恢复训练时 _pickle.UnpicklingError.
# 解决方案: monkey-patch Trainer._load_rng_state 使用 weights_only=False
# (rng_state.pth 由 Trainer 自身保存, 来源可信)
try:
    from transformers.trainer import Trainer as _Trainer
    import os as _os

    _orig_load_rng_state = _Trainer._load_rng_state

    def _patched_load_rng_state(self, checkpoint):
        rng_file = _os.path.join(checkpoint, "rng_state.pth")
        if not _os.path.isfile(rng_file):
            return
        try:
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        except Exception:
            return
        if "random" in checkpoint_rng_state:
            import random
            random.setstate(checkpoint_rng_state["random"])
        if "numpy" in checkpoint_rng_state:
            import numpy as np
            np.random.set_state(checkpoint_rng_state["numpy"])
        if "torch" in checkpoint_rng_state:
            torch.set_rng_state(checkpoint_rng_state["torch"])
        if "cuda" in checkpoint_rng_state:
            try:
                torch.cuda.set_rng_state_all(checkpoint_rng_state["cuda"])
            except Exception:
                pass

    _Trainer._load_rng_state = _patched_load_rng_state
except Exception:
    pass

from modelscope import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 修复: 禁用 PEFT 对 awq 的自动检测
# awq 包与 transformers>=4.51 不兼容 (shard_checkpoint 已移除),
# 但 PEFT 的 LoRA dispatcher 仍会尝试 from awq.modules.linear import WQLinear_GEMM,
# 导致 ImportError. 直接将 dispatch_awq 替换为空操作来跳过 awq 检测.
try:
    import peft.tuners.lora.model as _peft_lora_model
    _peft_lora_model.dispatch_awq = lambda target, adapter_name, **kwargs: None
except Exception:
    pass

# 修复: transformers<4.52 不支持 Qwen3 架构
# Qwen3 在 transformers 4.52.0 才加入, 但赛事要求 transformers==4.51.0.
# 解决方案: 将 Qwen3 注册为 Qwen2 的别名 (Qwen3 架构与 Qwen2 高度相似).
# 这需要在 AutoConfig/AutoModel 映射中添加 qwen3 -> Qwen2 的映射.
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    if "qwen3" not in CONFIG_MAPPING:
        # 使用 register 方法注册 Qwen3Config 为 Qwen2Config 的别名
        CONFIG_MAPPING.register("qwen3", Qwen2Config, exist_ok=True)
        # 注册 Qwen3ForCausalLM 为 Qwen2ForCausalLM 的别名
        MODEL_FOR_CAUSAL_LM_MAPPING.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)
        logging.getLogger(__name__).info("已注册 Qwen3 为 Qwen2 别名 (transformers<4.52 兼容)")
except Exception as e:
    logging.getLogger(__name__).debug(f"Qwen3 注册跳过: {e}")

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

    # 评测配置
    eval_after_train: bool = True
    test_data_path: str = "test_dataset/xguard_test_open_1k.jsonl"

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
    # padding="max_length" + max_length: 显式截断到max_length并padding,
    # 避免 "max_length is ignored when padding=True" 的警告
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config.model_max_length,
    )

    # 8. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 检查是否有可恢复的checkpoint
    checkpoint = None
    if os.path.exists(config.output_dir):
        checkpoints = [
            d for d in os.listdir(config.output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(config.output_dir, d))
        ]
        if checkpoints:
            # 选择最新的checkpoint
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint = os.path.join(config.output_dir, latest)
            logger.info(f"检测到checkpoint,将从断点恢复训练: {checkpoint}")

    logger.info("开始训练...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # 9. 保存模型
    logger.info(f"保存模型到: {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # 保存训练配置
    with open(os.path.join(config.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

    # 10. 释放训练对象占用的 GPU 显存
    # 训练完成后模型和优化器不再需要，显式删除并清理缓存，
    # 避免后续评估阶段加载推理模型时因显存不足导致 OOM。
    logger.info("释放训练显存...")
    del trainer, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("训练完成!")
    return config.output_dir
