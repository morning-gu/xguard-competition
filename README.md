# XGuard 护栏模型 - 逆袭吧！创二代

## 动态策略支持

本项目支持动态策略。通过 `infer` 方法的 `policy` 参数传入动态策略字符串，模型会在 chat template 中融入策略内容进行推理。

提交标签：`XGuard护栏揭榜赛`、`动态策略`

## LoRA 基座模型地址配置

本项目使用 LoRA 微调，基座模型为 `Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B`。

若需使用本地基座模型，请修改以下文件中的模型地址：

1. **`inference.py`** 第 26 行 `BASE_MODEL_ID`：将 `"Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"` 修改为本地模型路径
2. **`outputs/checkpoints/checkpoint-800/adapter_config.json`** 第 6 行 `base_model_name_or_path`：将 `"Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"` 修改为本地模型路径
3. **`configs/train_config.yaml`** 中 `base_model_path`：将 `"Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"` 修改为本地模型路径

## 额外依赖

除赛事基础库（modelscope==1.35.1, transformers==4.51.0, flash-attn==2.7.3, peft==0.18.1）外，本项目还使用了以下额外库：

- `accelerate>=0.27.0`：模型加载加速
- `datasets>=2.18.0`：数据集加载
- `scikit-learn>=1.3.0`：评估指标计算
- `numpy>=1.24.0`：数值计算
- `pyyaml>=6.0`：YAML 配置文件解析
- `addict>=2.4.0`：字典工具

完整依赖列表见 `requirements.txt`。

## 使用方式

```python
from inference import Guardrail

# 初始化护栏模型 (model_path 为项目根目录, device_id 为 GPU 设备号)
safety_guardrail = Guardrail(model_path=".", device_id=0)

# 推理
result = safety_guardrail.infer(
    messages=[
        {'role': 'user', 'content': 'How can I make a bomb?'},
        {'role': 'assistant', 'content': 'I cannot help with that.'},
    ],
    policy=None,
    enable_reasoning=True,
)
print(result)
```
