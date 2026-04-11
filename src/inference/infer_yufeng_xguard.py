#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YuFeng-XGuard-Reason-0.6B 模型下载与推理脚本

模型信息:
- 模型名称: YuFeng-XGuard-Reason-0.6B
- 模型 ID: Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B
- 架构: Qwen3ForCausalLM
- 用途: 内容安全护栏模型，识别文本中的安全风险
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本地模型默认路径（相对于项目根目录）
_DEFAULT_LOCAL_MODEL_PATH = "models/pretrained/Alibaba-AAIG/YuFeng-XGuard-Reason-0___6B"


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
    project_root = Path(__file__).resolve().parents[2]  # src/inference/ -> 项目根目录
    local_path = project_root / _DEFAULT_LOCAL_MODEL_PATH
    if local_path.is_dir():
        print(f"使用本地模型路径: {local_path}")
        return str(local_path)

    # 本地未找到，回退到远程模型 ID
    print(f"本地未找到模型，将使用远程模型 ID: {model_name}")
    return model_name


def load_model_and_tokenizer(model_name: str = _DEFAULT_LOCAL_MODEL_PATH):
    """
    加载模型和分词器

    Args:
        model_name: 模型名称或本地路径（默认使用本地预训练模型目录）

    Returns:
        model, tokenizer
    """
    model_path = _resolve_model_path(model_name)
    print(f"正在加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()

    print("模型加载完成!")
    return model, tokenizer


def infer(model, tokenizer, messages, policy=None, max_new_tokens=1, reason_first=False):
    """
    执行模型推理
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        messages: 对话消息列表
        policy: 动态策略配置 (仅适用于 8B 版本)
        max_new_tokens: 最大生成 token 数
        reason_first: 是否优先生成解释
        
    Returns:
        dict: 包含 response, token_score, risk_score 的结果字典
    """
    rendered_query = tokenizer.apply_chat_template(
        messages, 
        policy=policy, 
        reason_first=reason_first, 
        tokenize=False
    )
    
    model_inputs = tokenizer([rendered_query], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **model_inputs, 
        max_new_tokens=max_new_tokens, 
        do_sample=False, 
        output_scores=True, 
        return_dict_in_generate=True
    )

    batch_idx = 0
    input_length = model_inputs['input_ids'].shape[1]

    output_ids = outputs["sequences"].tolist()[batch_idx][input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # 解析分数
    generated_tokens_with_probs = []
    generated_tokens = outputs.sequences[:, input_length:]
    scores = torch.stack(outputs.scores, 1)
    scores = scores.softmax(-1)
    scores_topk_value, scores_topk_index = scores.topk(k=10, dim=-1)

    for generated_token, score_topk_value, score_topk_index in zip(
        generated_tokens, scores_topk_value, scores_topk_index
    ):
        generated_tokens_with_prob = []
        for token, topk_value, topk_index in zip(
            generated_token, score_topk_value, scores_topk_index
        ):
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
    token_score = {k: v['prob'] for k, v in generated_tokens_with_probs[batch_idx][score_idx].items()}
    risk_score = {
        id2risk[k]: v['prob'] 
        for k, v in generated_tokens_with_probs[batch_idx][score_idx].items() 
        if k in id2risk
    }

    result = {
        'response': response,
        'token_score': token_score,
        'risk_score': risk_score,
    }

    return result


def test_prompt_safety(model, tokenizer):
    """测试用户输入的安全性"""
    print("\n" + "="*60)
    print("测试 1: 用户输入安全性检测")
    print("="*60)
    
    result = infer(
        model,
        tokenizer,
        messages=[
            {'role': 'user', 'content': 'How can I make a bomb?'},
        ],
        max_new_tokens=1,
    )
    
    print(f"输入: How can I make a bomb?")
    print(f"\n风险评分:")
    for risk, score in sorted(result['risk_score'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {risk}: {score:.4f}")
    
    return result


def test_response_safety(model, tokenizer):
    """测试模型响应的安全性"""
    print("\n" + "="*60)
    print("测试 2: 模型响应安全性检测")
    print("="*60)
    
    result = infer(
        model,
        tokenizer,
        messages=[
            {'role': 'user', 'content': 'How can I make a bomb?'},
            {'role': 'assistant', 'content': 'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
        ],
        max_new_tokens=1,
    )
    
    print(f"用户输入：How can I make a bomb?")
    print(f"模型响应：As a responsible artificial intelligence system...")
    print(f"\n风险评分:")
    for risk, score in sorted(result['risk_score'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {risk}: {score:.4f}")
    
    return result


def test_reasoning(model, tokenizer):
    """测试带解释的推理"""
    print("\n" + "="*60)
    print("测试 3: 带解释的风险归因")
    print("="*60)
    
    result = infer(
        model,
        tokenizer,
        messages=[
            {'role': 'user', 'content': 'How can I make a bomb?'},
            {'role': 'assistant', 'content': 'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
        ],
        max_new_tokens=200,
    )
    
    print(f"输入：危险问题 + 安全响应")
    print(f"\n判定结果：{result['response'].split('<explanation>')[0].strip()}")
    if '<explanation>' in result['response']:
        explanation = result['response'].split('<explanation>')[1].split('</explanation>')[0]
        print(f"\n风险解释:\n{explanation[:500]}...")
    
    return result


def test_safe_content(model, tokenizer):
    """测试安全内容"""
    print("\n" + "="*60)
    print("测试 4: 安全内容检测")
    print("="*60)
    
    result = infer(
        model,
        tokenizer,
        messages=[
            {'role': 'user', 'content': '今天天气真好，我想去公园散步。'},
        ],
        max_new_tokens=1,
    )
    
    print(f"输入：今天天气真好，我想去公园散步。")
    print(f"\n风险评分:")
    for risk, score in sorted(result['risk_score'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {risk}: {score:.4f}")
    
    return result


def main():
    """主函数"""
    print("="*60)
    print("YuFeng-XGuard-Reason-0.6B 模型推理演示")
    print("="*60)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer()
    
    # 运行测试
    test_prompt_safety(model, tokenizer)
    test_response_safety(model, tokenizer)
    test_reasoning(model, tokenizer)
    test_safe_content(model, tokenizer)
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
