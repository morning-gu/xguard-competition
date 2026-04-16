# 代码重复分析报告

## 执行摘要

经过全面分析，发现仓库中存在**多处严重的代码重复**，主要分布在以下模块：

1. **数据模块** (`src/data/`): `loader.py` 和 `dataset.py` 存在大量重复
2. **训练模块** (`src/training/`): `trainer.py` 和 `train_core.py` 高度重复
3. **推理模块** (`src/inference/`): `engine.py` 和 `infer_yufeng_xguard.py` 存在重复
4. **评估脚本** (`scripts/evaluate.py`): 已与 `src/evaluation/evaluator.py` 重构

---

## 1. 数据模块重复 (src/data/)

### 重复文件
- `src/data/loader.py` (465 行)
- `src/data/dataset.py` (420 行)

### 重复内容

| 重复项 | loader.py | dataset.py | 说明 |
|--------|-----------|------------|------|
| `RISK_ID_MAP` | 第 30-60 行 | 第 21-51 行 | 完全相同的风险类别映射字典 |
| `XGuardTrainDataset` 类 | 第 185-283 行 | 第 54-135 行 | 核心 Dataset 类，逻辑基本相同 |
| `_process_raw_item` / `_process_jsonl_item` | 第 285-349 行 | 第 207-285 行 | 数据处理函数，逻辑相同 |
| `preprocess_data` / `preprocess_raw_data` | 第 352-402 行 | 第 138-204 行 | 数据预处理函数 |
| `create_dataloader` / `create_optimized_dataloader` | 第 405-435 行 | 第 394-419 行 | 数据加载器创建函数 |

### 建议重构方案

**保留 `src/data/loader.py` 作为主模块**，删除 `dataset.py` 中的重复代码：

```python
# src/data/__init__.py 应改为：
from .loader import (
    RISK_ID_MAP,
    download_train_dataset,
    load_train_dataset,
    load_test_dataset,
    XGuardTrainDataset,
    preprocess_data,
    create_dataloader,
)

__all__ = [
    "RISK_ID_MAP",
    "download_train_dataset",
    "load_train_dataset",
    "load_test_dataset",
    "XGuardTrainDataset",
    "preprocess_data",
    "create_dataloader",
]
```

**理由**: `loader.py` 功能更完整，包含数据集下载、加载、预处理的全流程支持。

---

## 2. 训练模块重复 (src/training/)

### 重复文件
- `src/training/trainer.py` (400 行)
- `src/training/train_core.py` (416 行)

### 重复内容

| 重复项 | trainer.py | train_core.py | 说明 |
|--------|------------|---------------|------|
| `load_config` | 第 44-49 行 | 第 60-65 行 | 完全相同 |
| `merge_config` | 第 51-86 行 | 第 67-101 行 | 几乎相同 |
| `_resolve_model_path` | 第 153-169 行 | 第 103-128 行 | 完全相同 |
| `load_model_and_tokenizer` | 第 88-151 行 | 第 130-193 行 | 几乎相同 |
| `apply_lora` | 第 171-193 行 | 第 195-217 行 | 几乎相同 |
| `train` 函数 | 第 279-357 行 | 第 312-416 行 | 核心训练逻辑重复 |

### 建议重构方案

**保留 `src/training/train_core.py` 作为主模块**，将 `trainer.py` 简化为兼容层或删除：

```python
# 方案 A: 删除 trainer.py，统一使用 train_core.py
# 方案 B: 将 trainer.py 改为薄封装层，导入 train_core 的实现

# trainer.py (简化后)
from .train_core import (
    load_config,
    merge_config,
    load_model_and_tokenizer,
    apply_lora,
    train,
)

__all__ = [
    "load_config",
    "merge_config", 
    "load_model_and_tokenizer",
    "apply_lora",
    "train",
]
```

**理由**: `train_core.py` 是更新的版本，有更完整的文档字符串和类型注解。

---

## 3. 推理模块重复 (src/inference/)

### 重复文件
- `src/inference/engine.py` (345 行)
- `src/inference/infer_yufeng_xguard.py` (285 行)

### 重复内容

| 重复项 | engine.py | infer_yufeng_xguard.py | 说明 |
|--------|-----------|------------------------|------|
| `_resolve_model_path` | - | 第 23-48 行 | `engine.py` 无此函数，使用 `src/model/loader.py` |
| `load_model_and_tokenizer` | - | 第 50-76 行 | `engine.py` 使用 `src/model/loader.py` |
| `infer` 函数 | 第 22-129 行 | 第 79-163 行 | **核心推理逻辑完全相同** |
| `build_messages` | 第 211-244 行 | - | `infer_yufeng_xguard.py` 无此独立函数 |

### 建议重构方案

**删除 `src/inference/infer_yufeng_xguard.py`**，其功能已完全由以下模块覆盖：
- 模型加载 → `src/model/loader.py`
- 推理引擎 → `src/inference/engine.py`
- 评估 → `src/evaluation/evaluator.py`

**理由**: `infer_yufeng_xguard.py` 是一个独立的推理脚本，但所有核心功能都已在其他模块中实现，且 `engine.py` 提供了更完善的接口。

---

## 4. 评估脚本 (已完成重构)

### 现状
- `scripts/evaluate.py`: 已简化为命令行入口 (约 70 行)
- `src/evaluation/evaluator.py`: 核心评估模块 (约 490 行)

### 说明
该部分已在之前的重构中完成，`scripts/evaluate.py` 不再是重复代码，而是作为 CLI 入口存在。

---

## 5. 模型加载重复

### 重复内容

| 位置 | 函数 | 说明 |
|------|------|------|
| `src/model/loader.py` | `load_model_and_tokenizer` | 主模型加载模块 |
| `src/inference/infer_yufeng_xguard.py` | `load_model_and_tokenizer` | 重复实现 |
| `src/training/trainer.py` | `load_model_and_tokenizer` | 针对训练的变体 |
| `src/training/train_core.py` | `load_model_and_tokenizer` | 针对训练的变体 |

### 建议
训练模块的 `load_model_and_tokenizer` 可以接受，因为需要处理配置字典和训练特定参数。但应移除 `infer_yufeng_xguard.py` 中的重复版本。

---

## 优先级排序

| 优先级 | 任务 | 预计影响 |
|--------|------|----------|
| 🔴 **高** | 删除 `src/inference/infer_yufeng_xguard.py` | 消除 285 行重复代码 |
| 🔴 **高** | 合并 `src/data/dataset.py` 到 `loader.py` | 消除 ~300 行重复代码 |
| 🟡 **中** | 简化 `src/training/trainer.py` | 消除 ~350 行重复代码 |
| 🟢 **低** | 提取公共工具函数到 `src/utils/` | 提高代码复用性 |

---

## 建议的新目录结构

```
src/
├── data/
│   ├── __init__.py
│   ├── loader.py          # 统一数据加载模块 (保留)
│   └── preprocess_xguard_data.py
│
├── model/
│   ├── __init__.py
│   └── loader.py          # 统一模型加载模块
│
├── training/
│   ├── __init__.py
│   ├── train.py           # CLI 入口
│   ├── train_core.py      # 核心训练逻辑 (保留)
│   └── trainer.py         # 简化为兼容层或删除
│
├── inference/
│   ├── __init__.py
│   └── engine.py          # 统一推理引擎 (保留)
│
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py       # 统一评估模块
│
└── utils/                 # 新增：公共工具函数
    ├── __init__.py
    ├── config.py          # 配置相关工具
    └── common.py          # 通用工具函数
```

---

## 下一步行动

1. **立即执行**: 删除 `src/inference/infer_yufeng_xguard.py`
2. **立即执行**: 合并 `src/data/dataset.py` 到 `loader.py`
3. **计划执行**: 简化 `src/training/trainer.py`
4. **可选**: 创建 `src/utils/` 模块提取公共函数

