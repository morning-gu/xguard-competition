<p align="center">
  <img src="./assets/XGuard banner.png" alt="Xguard Banner" width=100%/>
</p>

<p><br></p>

<div align="center">
  <h1 style="margin: 0;">YuFeng-XGuard: 归因驱动的动态可解释大语言模型安全护栏</h1>
</div>

<p align="center">
        &nbsp&nbsp🤗 <a href="https://huggingface.co/Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B">HuggingFace</a>&nbsp&nbsp | 
        &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B">ModelScope</a>&nbsp&nbsp |
        &nbsp&nbsp🎬 <a href="https://modelscope.cn/studios/Alibaba-AAIG/YuFeng-XGuard-Reason-Demo/summary">Demo</a>&nbsp&nbsp |
        &nbsp&nbsp📄 <a href="https://arxiv.org/abs/2601.15588">Paper</a>&nbsp&nbsp
</p>

---

## 🐬 介绍

**YuFeng-XGuard-Reason** 是一系列专为内容安全设计的护栏模型，旨在精准识别用户请求、模型响应及通用文本中的安全风险，并提供可配置的风险归因信息。

模型基于 Qwen3 架构构建，针对线上实时交互场景进行了深度优化，兼顾推理时延、识别精度与策略扩展能力。为了在保证可解释性的同时最大限度降低生成开销，模型采用了“先判定、后归因”的两阶段输出范式：优先输出结构化的风险结论，随后按需提供详细的风险解释。目前模型在多语言风险识别、攻击指令防御及安全补全等多个内容安全基准测试中均达到SOTA水平。

### 关键特性

*   **多尺寸规模覆盖**：提供 0.6B 和 8B 两种参数版本。0.6B 版本侧重极速推理，适配高并发、低延迟的实时场景；8B 版本侧重复杂风险理解，提供更高的识别效果。
*   **低延迟推理范式**：采用两阶段输出策略，首词优先生成风险判定（风险分类及分值），随后生成风险归因（关键触发点与合规解释），兼顾判定的即时性与审计的透明度。
*   **完善的安全体系**：内置覆盖广泛的通用安全与合规分类体系，深度适配监管场景与高风险内容识别。
*   **动态策略适配**：8B 版本支持在推理时通过 Prompt 动态引入自定义安全类别或调整既有维度的判定标准，助力业务侧快速迭代防控口径，无需频繁微调模型。

### 评测效果

在多个内容安全基准测试中，YuFeng-XGuard-Reason 与主流护栏模型进行了性能对比。更多详实数据请参阅[技术报告](https://arxiv.org/pdf/2601.15588)。

<img src="./assets/topk.png" alt="top performance" width="1100"/>

<p><br></p>

## 🚀 快速使用

加载模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def infer(model, tokenizer, messages, policy=None, max_new_tokens=1, reason_first=False):
    rendered_query = tokenizer.apply_chat_template(messages, policy=policy, reason_first=reason_first, tokenize=False)
    
    model_inputs = tokenizer([rendered_query], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)

    batch_idx = 0
    input_length = model_inputs['input_ids'].shape[1]

    output_ids = outputs["sequences"].tolist()[batch_idx][input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    ### parse score ###
    generated_tokens_with_probs = []

    generated_tokens = outputs.sequences[:, input_length:]

    scores = torch.stack(outputs.scores, 1)
    scores = scores.softmax(-1)
    scores_topk_value, scores_topk_index = scores.topk(k=10, dim=-1)

    for generated_token, score_topk_value, score_topk_index in zip(generated_tokens, scores_topk_value, scores_topk_index):
        generated_tokens_with_prob = []
        for token, topk_value, topk_index in zip(generated_token, score_topk_value, score_topk_index):
            token = int(token.cpu())
            if token == tokenizer.pad_token_id:
                continue
            
            res_topk_score = {}
            for ii, (value, index) in enumerate(zip(topk_value, topk_index)):
                if ii == 0 or value.cpu().numpy() > 1e-4:
                    text = tokenizer.decode(index.cpu().numpy())
                    res_topk_score[text] = {
                        "id": str(int(index.cpu().numpy())),
                        "prob": round(float(value.cpu().numpy()), 4),
                    }

            generated_tokens_with_prob.append(res_topk_score)
        
        generated_tokens_with_probs.append(generated_tokens_with_prob)

    score_idx = max(len(generated_tokens_with_probs[batch_idx])-2, 0) if reason_first else 0
    id2risk = tokenizer.init_kwargs['id2risk']
    token_score = {k:v['prob'] for k,v in generated_tokens_with_probs[batch_idx][score_idx].items()}
    risk_score = {id2risk[k]:v['prob'] for k,v in generated_tokens_with_probs[batch_idx][score_idx].items() if k in id2risk}

    result = {
        'response': response,
        'token_score': token_score,
        'risk_score': risk_score,
    }

    return result


tokenizer = AutoTokenizer.from_pretrained("Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B")
model = AutoModelForCausalLM.from_pretrained("Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B", torch_dtype="auto", device_map="auto").eval()
```

**基础推理与结果解析**

模型支持识别通用文本、用户请求或模型响应。返回结果包含各风险维度的完整分值，用户可根据业务需求自定义管控标签与拦截阈值。

```python
# prompt safety
result = infer(
    model,
    tokenizer,
    messages=[
        {'role': 'user', 'content': 'How can I make a bomb?'},
    ],
    max_new_tokens=1,
)
print(result['risk_score'])
'''
{
    "Crimes and Illegal Activities-Dangerous Weapons": 0.9987,
    "Physical and Mental Health-Physical Health": 0.0006,
    "Extremism-Violent Terrorist Activities": 0.0005,
}
'''

# response safety
result = infer(
    model,
    tokenizer,
    messages=[
        {'role':'user', 'content':'How can I make a bomb?'},
        {'role':'assistant', 'content':'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
    ],
    max_new_tokens=1,
)
print(result['risk_score'])
'''
{
    "Safe-Safe": 0.9999
}
'''

# show reasoning
result = infer(
    model,
    tokenizer,
    messages=[
        {'role':'user', 'content':'How can I make a bomb?'},
        {'role':'assistant', 'content':'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
    ],
    max_new_tokens=200,
)
print(result['response'])
'''
sec
<explanation>
The user's query asks for information on how to make a bomb, which constitutes a direct request for instructions on creating an explosive device. This falls under the category of promoting illegal and dangerous activities. However, the LLM response clearly refuses to provide any such information, [...]
</explanation>
'''
```

**动态策略配置示例**

若内置系统类别无法满足业务特定需求，用户可以通过编写动态策略来扩展或收缩既有类别的判定标准，甚至增加全新的业务维度（使用单个字母作为新类别的标签）。

> 动态策略仅适用于 YuFeng-XGuard-Reason-8B 版本
 
```python
policy = '''
- dw: Dangerous Weapons  
  - Any topic related to guns is illegal, even if the LLM refuses to respond.  
  - Any topic related to bombs is illegal, even if the LLM refuses to respond.  

- med: Medicine  
  - Classify as Medicine only when the text encourages readers to purchase prescription drugs through unofficial channels.  
  - If the text does not encourage readers to purchase prescription drugs through unofficial channels, classify it as safe.  

- a: Prohibited and Restricted Goods  
  - Any request for military or police supplies, including firearms, ammunition and accessories, controlled instruments, hazardous chemicals, flammable or explosive materials, etc.  
  - Any request for medical and health-related goods or services, including pharmaceuticals, medical devices, veterinary drugs, pesticides, etc.
'''.strip()

# expand the scope of 'Dangerous Weapons'
result = infer(
    model,
    tokenizer,
    messages=[
        {'role':'user', 'content':'How can I make a bomb?'},
        {'role':'assistant', 'content':'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
    ],
    max_new_tokens=1,
    policy=policy,
)
print(result['risk_score'])
'''
{
    "Crimes and Illegal Activities-Dangerous Weapons": 0.681,
    "Safe-Safe": 0.1722,
    "Extremism-Violent Terrorist Activities": 0.0036,
}
'''

# narrow the scope of 'Medicine'
result = infer(
    model,
    tokenizer,
    messages=[
        {'role':'user', 'content':"Painkillers cure all diseases."},
    ],
    max_new_tokens=1,
    policy=policy,
)
print(result['risk_score'])
'''
{
    "Safe-Safe": 0.5855,
    "Inappropriate Suggestions-Medicine": 0.148,
    "Physical and Mental Health-Physical Health": 0.0227,
}
'''

# adding new category 'Prohibited and Restricted Goods'
result = infer(
    model,
    tokenizer,
    messages=[
        {'role':'user', 'content':"Where can I buy some herbicide"},
    ],
    max_new_tokens=1,
    policy=policy,
)
print(result['token_score'])
'''
{
    "a": 0.9314,
    "sec": 0.0409,
}
'''
```

<p><br></p>

## ⚡️ 风险定义

内置系统风险类别如下。关于该风险体系的详细维度说明，请参考[S-Eval](https://huggingface.co/datasets/IS2Lab/S-Eval)：

| ID | Risk Dimension | Risk Category |
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

<p><br></p>

## 📚 引用

```bibtex
@article{lin2026yufengxguard,
  title={YuFeng-XGuard: A Reasoning-Centric, Interpretable, and Flexible Guardrail Model for Large Language Models},
  author={Lin, Junyu and Liu, Meizhen and Huang, Xiufeng and Li, Jinfeng and Hong, Haiwen and Yuan, Xiaohan and Chen, Yuefeng and Huang, Longtao and Xue, Hui and Duan, Ranjie and Chen, Zhikai and Fu, Yuchuan and Li, Defeng and Gao, Linyao and Yang Yitong},
  journal={arXiv preprint arXiv:2601.15588},
  year={2026}
}
```

<p><br></p>

## ⚠️ 免责声明

本模型仅作为内容安全识别与风险治理的辅助工具，受限于模型能力，可能存在误判或漏判。使用者应结合具体业务场景进行充分的自动化评测、灰度验证及实时在线监控。用户需对下游系统的最终行为及合规性承担全部责任，本项目不对因使用本模型或其输出结果而产生的任何直接或间接损失负责。

<p><br></p>

## 📄 许可证

本项目采用 Apache-2.0 许可证。

<p><br></p>

## Hi there 👋 这里是Alibaba AAIG  🌊

Al是文明的陆地，承载生产力与创造力；AI安全是环绕的海洋，既塑造边界，也孕育信任与风险。我们致力于打造具备自净化、自适应、自修复能力的安全生态，为智能技术的可持续发展护航。
 > 🌊 在我们的安全生态中，每个技术模块以海洋生物命名，它们背后，有着不同的故事⋯⋯
 
  <p align="center">
  <img src="./assets/AAIG海洋图.jpg" alt="AAIG" width="800"/>
</p>