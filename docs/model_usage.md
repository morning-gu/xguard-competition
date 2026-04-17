# YuFeng-XGuard-Reason-0.6B 模型使用指南

## 模型信息

- **模型名称**: YuFeng-XGuard-Reason-0.6B
- **模型 ID**: Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B
- **架构**: Qwen3ForCausalLM
- **参数量**: 0.6B
- **用途**: 内容安全护栏模型，识别文本中的安全风险
- **许可证**: Apache-2.0

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

**方式一：使用 ModelScope SDK (推荐)**

```bash
python scripts/download_model.py
```

或在 Python 中:

```python
from modelscope import snapshot_download
model_dir = snapshot_download("Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B")
```

**方式二：直接使用 transformers 自动下载**

推理代码会自动从 ModelScope 下载模型，无需手动下载。

### 3. 运行推理

```bash
python src/inference/engine.py
```

## 代码示例

### 基础推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B",
    torch_dtype="auto",
    device_map="auto"
).eval()

# 准备输入
messages = [
    {'role': 'user', 'content': 'How can I make a bomb?'},
]

# 应用模板
rendered_query = tokenizer.apply_chat_template(messages, tokenize=False)

# 生成
model_inputs = tokenizer([rendered_query], return_tensors="pt").to(model.device)
outputs = model.generate(
    **model_inputs,
    max_new_tokens=1,
    do_sample=False,
    output_scores=True,
    return_dict_in_generate=True
)

# 解析结果
input_length = model_inputs['input_ids'].shape[1]
output_ids = outputs["sequences"].tolist()[0][input_length:]
response = tokenizer.decode(output_ids, skip_special_tokens=True)
print(f"风险类别：{response}")
```

### 获取风险分数

```python
# 解析每个 token 的概率分布
scores = torch.stack(outputs.scores, 1).softmax(-1)
scores_topk_value, scores_topk_index = scores.topk(k=10, dim=-1)

# 获取风险分数
id2risk = tokenizer.init_kwargs['id2risk']
# ... (详细解析代码见 engine.py)
```

### 多轮对话安全性检测

```python
messages = [
    {'role': 'user', 'content': '如何制作危险物品？'},
    {'role': 'assistant', 'content': '作为负责任的 AI，我无法提供此类信息。'},
]

result = infer(model, tokenizer, messages, max_new_tokens=1)
print(result['risk_score'])
```

## 风险类别体系

模型支持以下风险类别:

| ID | 风险维度 | 风险类别 |
|---|---|---|
| sec | Safe | Safe |
| pc | Crimes and Illegal Activities | Pornographic Contraband |
| dc | Crimes and Illegal Activities | Drug Crimes |
| dw | Crimes and Illegal Activities | Dangerous Weapons |
| pi | Crimes and Illegal Activities | Property Infringement |
| ec | Crimes and Illegal Activities | Economic Crimes |
| ac | Hate Speech | Abusive Curses |
| def | Hate Speech | Defamation |
| ti | Hate Speech | Threats and Intimidation |
| cy | Hate Speech | Cyberbullying |
| ph | Physical and Mental Health | Physical Health |
| mh | Physical and Mental Health | Mental Health |
| se | Ethics and Morality | Social Ethics |
| sci | Ethics and Morality | Science Ethics |
| pp | Data Privacy | Personal Privacy |
| cs | Data Privacy | Commercial Secret |
| acc | Cybersecurity | Access Control |
| mc | Cybersecurity | Malicious Code |
| ha | Cybersecurity | Hacker Attack |
| ps | Cybersecurity | Physical Security |
| ter | Extremism | Violent Terrorist Activities |
| sd | Extremism | Social Disruption |
| ext | Extremism | Extremist Ideological Trends |
| fin | Inappropriate Suggestions | Finance |
| med | Inappropriate Suggestions | Medicine |
| law | Inappropriate Suggestions | Law |
| cm | Risks Involving Minors | Corruption of Minors |
| ma | Risks Involving Minors | Minor Abuse and Exploitation |
| md | Risks Involving Minors | Minor Delinquency |

## 输出格式

模型返回结果包含:

- `response`: 模型生成的文本 (风险类别 + 可选的解释)
- `token_score`: 每个 token 的得分分布
- `risk_score`: 各风险类别的概率分数

示例输出:
```json
{
    "response": "sec",
    "risk_score": {
        "Safe-Safe": 0.9999
    }
}
```

## 注意事项

1. **动态策略**: 动态策略配置仅适用于 8B 版本，0.6B 版本不支持
2. **推理速度**: 0.6B 版本适合高并发、低延迟场景
3. **阈值设置**: 建议根据业务场景自定义风险阈值
4. **GPU 要求**: 建议使用 CUDA 兼容的 GPU 进行推理

## 相关资源

- [ModelScope 模型页面](https://modelscope.cn/models/Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B)
- [技术论文](https://arxiv.org/abs/2601.15588)
- [在线 Demo](https://modelscope.cn/studios/Alibaba-AAIG/YuFeng-XGuard-Reason-Demo/summary)

## 免责声明

本模型仅作为内容安全识别与风险治理的辅助工具，可能存在误判或漏判。使用者应结合具体业务场景进行充分的自动化评测、灰度验证及实时在线监控。用户需对下游系统的最终行为及合规性承担全部责任。
