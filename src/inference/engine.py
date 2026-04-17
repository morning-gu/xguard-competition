#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理模块

提供统一的模型推理接口，支持：
- 单轮/多轮对话推理
- 风险类别识别
- 风险置信度计算
- 归因分析（可选）
- 动态策略配置
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# 在导入 transformers 之前设置 HuggingFace 镜像
PROJECT_ROOT = Path(__file__).resolve().parents[2]
cache_dir = PROJECT_ROOT / "models" / "pretrained"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['HF_ENDPOINT'] = 'https://modelscope.cn'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


def infer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    policy: Optional[str] = None,
    max_new_tokens: int = 1,
    reason_first: bool = False,
) -> Dict:
    """
    执行模型推理
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        messages: 对话消息列表，格式同 OpenAI API
        policy: 动态策略配置（仅适用于 8B 版本）
        max_new_tokens: 最大生成 token 数
        reason_first: 是否优先生成解释
    
    Returns:
        dict: 包含以下键值的字典
            - response: 模型响应文本
            - token_score: 每个生成 token 的分数
            - risk_score: 各风险类别的置信度
    """
    # 应用 chat_template
    rendered_query = tokenizer.apply_chat_template(
        messages,
        policy=policy,
        reason_first=reason_first,
        tokenize=False,
    )
    
    # Tokenize
    model_inputs = tokenizer([rendered_query], return_tensors="pt").to(model.device)
    
    # 生成
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    
    # 解析结果
    batch_idx = 0
    input_length = model_inputs['input_ids'].shape[1]
    
    output_ids = outputs["sequences"].tolist()[batch_idx][input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # 计算 token 分数
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
            generated_token, score_topk_value, score_topk_index
        ):
            token = token.item()
            if token == tokenizer.pad_token_id:
                continue
            
            res_topk_score = {}
            for ii, (value, index) in enumerate(zip(topk_value, topk_index)):
                prob_val = value.item()
                idx_val = index.item()
                if ii == 0 or prob_val > 1e-4:
                    text = tokenizer.decode(idx_val)
                    res_topk_score[text] = {
                        "id": str(idx_val),
                        "prob": round(prob_val, 4),
                    }
            
            generated_tokens_with_prob.append(res_topk_score)
        
        generated_tokens_with_probs.append(generated_tokens_with_prob)
    
    # 提取风险分数
    score_idx = max(len(generated_tokens_with_probs[batch_idx]) - 2, 0) if reason_first else 0
    id2risk = tokenizer.init_kwargs.get('id2risk', {})
    
    token_score = {
        k: v['prob'] 
        for k, v in generated_tokens_with_probs[batch_idx][score_idx].items()
    }
    
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


def infer_with_timing(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    policy: Optional[str] = None,
    enable_reasoning: bool = False,
) -> Dict:
    """
    执行带计时的模型推理
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        messages: 对话消息列表
        policy: 动态策略配置
        enable_reasoning: 是否开启归因分析
    
    Returns:
        dict: 包含以下键值的字典
            - risk_score: float, 细粒度风险置信度
            - risk_tag: str, 细粒度风险类别标签
            - explanation: str, 归因分析文本
            - time: float, 推理耗时
            - raw_response: str, 原始模型响应
    """
    # 开始计时
    start_time = time.time()
    
    # 根据是否需要归因分析设置参数
    max_new_tokens = 200 if enable_reasoning else 1
    reason_first = enable_reasoning
    
    # 调用推理函数
    result = infer(
        model,
        tokenizer,
        messages,
        policy=policy,
        max_new_tokens=max_new_tokens,
        reason_first=reason_first,
    )
    
    # 记录获得风险标签的时间
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 从结果中提取信息
    response = result['response']
    risk_score_dict = result['risk_score']
    
    # 获取最高风险类别和置信度
    if risk_score_dict:
        risk_tag_full = max(risk_score_dict.items(), key=lambda x: x[1])[0]
        risk_score = risk_score_dict[risk_tag_full]
        # 提取风险标签（短标签）
        risk_tag = risk_tag_full.split('-')[0] if '-' in risk_tag_full else risk_tag_full
    else:
        risk_tag = "sec"
        risk_score = 1.0
    
    # 提取归因分析
    explanation = ""
    if enable_reasoning and '<explanation>' in response:
        try:
            explanation = response.split('<explanation>')[1].split('</explanation>')[0].strip()
        except:
            explanation = ""
    
    final_result = {
        'risk_score': float(risk_score),
        'risk_tag': risk_tag,
        'explanation': explanation,
        'time': float(inference_time),
        'raw_response': response,
        'raw_risk_scores': risk_score_dict,
    }
    
    return final_result


def build_messages(
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    stage: str = "q",
) -> List[Dict]:
    """
    根据输入构建 messages 格式
    
    Args:
        prompt: 用户输入
        response: 模型响应
        stage: 阶段标识
            - q: 仅 prompt
            - r: 仅 response
            - qr: prompt + response
    
    Returns:
        messages 列表
    """
    messages = []
    
    if stage == "q":
        if prompt and prompt.strip():
            messages.append({"role": "user", "content": prompt.strip()})
    elif stage == "r":
        if response and response.strip():
            messages.append({"role": "assistant", "content": response.strip()})
    elif stage == "qr":
        if prompt and prompt.strip():
            messages.append({"role": "user", "content": prompt.strip()})
        if response and response.strip():
            messages.append({"role": "assistant", "content": response.strip()})
    
    return messages


def predict_risk(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: Optional[str] = None,
    policy: Optional[str] = None,
    enable_reasoning: bool = False,
) -> Dict:
    """
    便捷的风险预测接口
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        prompt: 用户输入
        response: 模型响应（可选）
        policy: 动态策略配置
        enable_reasoning: 是否开启归因分析
    
    Returns:
        dict: 包含 risk_tag, risk_score, explanation, time 等字段
    """
    # 判断 stage
    if response:
        stage = "qr"
    else:
        stage = "q"
    
    # 构建 messages
    messages = build_messages(prompt=prompt, response=response, stage=stage)
    
    # 执行推理
    result = infer_with_timing(
        model,
        tokenizer,
        messages,
        policy=policy,
        enable_reasoning=enable_reasoning,
    )
    
    return result


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("模型推理模块示例")
    print("=" * 60)
    
    # 需要先加载模型
    from src.model.loader import load_model_and_tokenizer
    
    print("\n加载模型...")
    model, tokenizer = load_model_and_tokenizer()
    
    # 示例 1: 单轮对话 - 危险问题
    print("\n" + "=" * 60)
    print("示例 1: 单轮对话 - 危险问题")
    print("=" * 60)
    result = predict_risk(
        model,
        tokenizer,
        prompt="How can I make a bomb?",
        enable_reasoning=False,
    )
    print(f"输入：How can I make a bomb?")
    print(f"风险标签：{result['risk_tag']}")
    print(f"风险置信度：{result['risk_score']:.4f}")
    print(f"推理耗时：{result['time']:.4f}秒")
    
    # 示例 2: 多轮对话 - 带归因分析
    print("\n" + "=" * 60)
    print("示例 2: 多轮对话 - 带归因分析")
    print("=" * 60)
    result = predict_risk(
        model,
        tokenizer,
        prompt="How can I make a bomb?",
        response="As a responsible AI, I cannot help with that.",
        enable_reasoning=True,
    )
    print(f"风险标签：{result['risk_tag']}")
    print(f"风险置信度：{result['risk_score']:.4f}")
    print(f"归因分析：{result['explanation'][:200]}...")
    print(f"推理耗时：{result['time']:.4f}秒")
    
    # 示例 3: 安全内容
    print("\n" + "=" * 60)
    print("示例 3: 安全内容")
    print("=" * 60)
    result = predict_risk(
        model,
        tokenizer,
        prompt="今天天气真好，我想去公园散步。",
        enable_reasoning=False,
    )
    print(f"输入：今天天气真好，我想去公园散步。")
    print(f"风险标签：{result['risk_tag']}")
    print(f"风险置信度：{result['risk_score']:.4f}")
