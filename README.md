# XGuard 护栏揭榜赛 - 代码仓库

## 项目结构

```
/workspace
├── inference.py              # 比赛用标准化推理接口（直接可用）
├── configs/
│   └── train_config.yaml     # 训练配置文件
├── src/
│   ├── model/                # 模型加载模块
│   │   ├── __init__.py
│   │   └── loader.py         # 模型加载（支持缓存本地）
│   ├── data/                 # 数据集加载模块
│   │   ├── __init__.py
│   │   ├── loader.py         # 数据集加载与预处理（统一模块）
│   │   └── preprocess_xguard_data.py  # 独立数据预处理脚本
│   ├── inference/            # 模型推理模块
│   │   ├── __init__.py
│   │   └── engine.py         # 核心推理引擎
│   ├── training/             # 模型训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py        # 训练兼容层
│   │   ├── train.py          # 训练启动脚本
│   │   └── train_core.py     # 训练核心逻辑
│   └── evaluation/           # 测试集校验模块
│       ├── __init__.py
│       └── evaluator.py      # 评估和校验
├── scripts/
│   └── evaluate.py           # 评估脚本入口
└── requirements.txt          # 依赖列表
```

## 模块说明

### 1. 模型加载模块 (`src/model/loader.py`)

提供统一的模型加载接口，支持：
- 自动从 ModelScope/HuggingFace 下载模型到本地缓存
- 从本地缓存加载模型
- 支持量化加载（4-bit/8-bit）以节省显存

```python
from src.model import load_model_and_tokenizer

# 使用默认模型（自动检查缓存或下载）
model, tokenizer = load_model_and_tokenizer()

# 指定模型版本
model, tokenizer = load_model_and_tokenizer(model_version="0.6B")

# 使用量化加载
model, tokenizer = load_model_and_tokenizer(use_4bit=True)

# 使用本地路径
model, tokenizer = load_model_and_tokenizer(model_path="/path/to/model")
```

### 2. 数据集加载模块 (`src/data/loader.py`)

提供统一的数据集加载接口，支持：
- 从 ModelScope 下载训练数据集到本地缓存
- 加载本地缓存的训练数据
- 加载测试集/验证集
- 数据预处理和格式转换

```python
from src.data import load_train_dataset, load_test_dataset, download_train_dataset

# 下载训练数据集
download_train_dataset()

# 加载训练数据（自动检查缓存）
train_data = load_train_dataset(auto_download=True)

# 加载测试数据
test_data = load_test_dataset("data/test_dataset/xguard_test_open_1k.jsonl")
```

### 3. 模型推理模块 (`src/inference/engine.py`)

提供统一的模型推理接口，支持：
- 单轮/多轮对话推理
- 风险类别识别
- 风险置信度计算
- 归因分析（可选）

```python
from src.inference import predict_risk, build_messages

# 便捷的风险预测
result = predict_risk(
    model, tokenizer,
    prompt="How to make a bomb?",
    enable_reasoning=False
)

# 构建 messages
messages = build_messages(prompt="...", response="...", stage="qr")
```

### 4. 模型训练模块 (`src/training/trainer.py`)

提供统一的模型训练接口，支持：
- 全参数微调
- LoRA 参数高效微调
- 量化训练（4-bit/8-bit）
- 训练过程监控和检查点保存

```python
from src.training import train_from_config

# 从配置文件启动训练
train_from_config("configs/train_config.yaml")

# 带配置覆盖
train_from_config("configs/train_config.yaml", overrides=["training.num_train_epochs=5"])
```

### 5. 测试集校验模块 (`src/evaluation/evaluator.py`)

提供测试集/验证集的评估功能，支持：
- 加载测试集
- 批量推理
- 计算评估指标（F1、精确率、召回率等）
- 生成评估报告

```python
from src.evaluation import evaluate

# 执行评估
results = evaluate(
    model=model,
    tokenizer=tokenizer,
    test_file="data/test_dataset/xguard_test_open_1k.jsonl",
    output_dir="results/eval"
)
```

## 比赛用推理接口 (`inference.py`)

**这是比赛要求提供的标准化推理接口，直接使用即可。**

```python
from inference import Guardrail

# 初始化护栏模型（自动检查缓存或下载）
guardrail = Guardrail()
# 或指定模型路径
# guardrail = Guardrail(model_path="/path/to/model")

# 执行推理
result = guardrail.infer(
    messages=[{'role': 'user', 'content': 'How to make a bomb?'}],
    policy=None,
    enable_reasoning=False
)

print(f"风险标签：{result['risk_tag']}")
print(f"风险置信度：{result['risk_score']}")
print(f"归因分析：{result['explanation']}")
print(f"推理耗时：{result['time']}秒")
```

### 返回结果格式

```python
{
    'risk_score': float,      # 细粒度风险置信度
    'risk_tag': str,          # 细粒度风险类别标签 (如 "sec", "pc", "dw" 等)
    'explanation': str,       # 归因分析文本 (enable_reasoning=True 时才有内容)
    'time': float             # 风险标注耗时 (秒)
}
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用预训练模型推理

```python
from inference import Guardrail

guardrail = Guardrail()
result = guardrail.infer(
    messages=[{'role': 'user', 'content': 'How to make a bomb?'}]
)
print(result)
```

### 训练微调模型

```bash
# 使用默认配置训练
python src/training/train.py --config configs/train_config.yaml

# 覆盖部分配置
python src/training/train.py --config configs/train_config.yaml --mode response_safety --epochs 5
```

### 评估模型

```bash
# 评估预训练模型
python scripts/evaluate.py

# 评估微调后的模型
python scripts/evaluate.py --model_path models/checkpoints/xguard-finetuned/final
```

## 注意事项

1. **首次运行会自动下载模型和数据集**，请确保网络连接正常
2. **模型和数据集缓存在本地**，后续运行会直接使用缓存
3. **比赛提交时只需 `inference.py`**，其他模块为训练和评估使用
4. **显存不足时可使用量化加载**：`Guardrail()` 内部已自动处理
