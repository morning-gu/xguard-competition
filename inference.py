#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard 护栏揭榜赛 - 标准化推理接口

符合赛事要求的标准化推理接口:
- 支持一般文本内容和 Query & Response 对话两种输入场景
- 支持动态策略 (policy 参数)
- 支持归因分析 (enable_reasoning 参数)
- 返回细粒度风险类别、风险置信度、归因分析

模型信息:
- 基础模型：YuFeng-XGuard-Reason-0.6B
- 模型 ID: Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B
- 架构：Qwen3ForCausalLM

使用方法:
    from inference import Guardrail
    
    # 初始化护栏模型
    guardrail = Guardrail()  # 使用默认模型
    # 或 guardrail = Guardrail(model_path="path/to/model")
    
    # 执行推理
    result = guardrail.infer(
        messages=[{'role': 'user', 'content': 'How to make a bomb?'}],
        policy=None,
        enable_reasoning=False
    )
    
    print(f"风险标签：{result['risk_tag']}")
    print(f"风险置信度：{result['risk_score']}")
    print(f"归因分析：{result['explanation']}")
"""

import os
import time
from typing import List, Dict, Optional
from pathlib import Path

# 在导入任何其他模块之前设置 HuggingFace 缓存目录
PROJECT_ROOT = Path(__file__).resolve().parent
cache_dir = PROJECT_ROOT / "models" / "pretrained"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(cache_dir)
os.environ['HF_HUB_CACHE'] = str(cache_dir / "hub")


class Guardrail:
    """XGuard 护栏模型推理类"""

    def __init__(self, model_path: str = None):
        """
        初始化护栏模型

        Args:
            model_path: 模型路径，可以是:
                - 本地绝对路径
                - HuggingFace/ModelScope 模型 ID (如 "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B")
                - None (使用默认本地路径，自动检查缓存或下载)
        """
        from src.model.loader import load_model_and_tokenizer
        
        # 加载模型和分词器
        self.model, self.tokenizer = load_model_and_tokenizer(model_path=model_path)
        
        # 获取风险标签映射
        self.id2risk = self.tokenizer.init_kwargs.get('id2risk', {})

    def infer(
        self,
        messages: List[Dict],
        policy: Optional[str] = None,
        enable_reasoning: bool = False
    ) -> Dict:
        """
        执行风险识别推理

        Args:
            messages: 输入消息列表，格式同 OpenAI API
                - 单轮对话：[{'role': 'user', 'content': '...'}]
                - 多轮对话：[{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
            policy: 动态策略 (如有),用于添加新的风险类别或调整现有类别
            enable_reasoning: 是否开启归因分析

        Returns:
            dict: 包含以下键值的字典
                - 'risk_score': float, 细粒度风险置信度
                - 'risk_tag': str, 细粒度风险类别标签
                - 'explanation': str, 归因分析文本 (若 enable_reasoning=False 则为空字符串 '')
                - 'time': float, 风险标注耗时，从开始推理到获得 risk_tag 所需时间
        """
        from src.inference.engine import infer as infer_engine
        
        # 开始推理计时
        start_time = time.time()
        
        # 根据是否需要归因分析设置参数
        max_new_tokens = 200 if enable_reasoning else 1
        reason_first = enable_reasoning
        
        # 调用推理函数
        result = infer_engine(
            self.model,
            self.tokenizer,
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
            # 提取风险标签 (短标签，如"sec", "pc"等)
            risk_tag = risk_tag_full.split('-')[0] if '-' in risk_tag_full else risk_tag_full
        else:
            risk_tag = "sec"  # 默认为安全
            risk_score = 1.0
        
        # 提取归因分析
        explanation = ""
        if enable_reasoning and '<explanation>' in response:
            try:
                explanation = response.split('<explanation>')[1].split('</explanation>')[0].strip()
            except:
                explanation = ""
        
        result = {
            'risk_score': float(risk_score),
            'risk_tag': risk_tag,
            'explanation': explanation,
            'time': float(inference_time),
        }
        
        return result


if __name__ == '__main__':
    # 示例用法
    print("=" * 60)
    print("XGuard 护栏模型推理示例")
    print("=" * 60)
    
    # 初始化护栏模型
    safety_guardrail = Guardrail()
    
    # 测试 1: 单轮对话 - 危险问题
    print("\n" + "=" * 60)
    print("测试 1: 单轮对话 - 危险问题")
    print("=" * 60)
    result = safety_guardrail.infer(
        messages=[
            {'role': 'user', 'content': 'How can I make a bomb?'},
        ],
        policy=None,
        enable_reasoning=False
    )
    print(f"输入：How can I make a bomb?")
    print(f"风险标签：{result['risk_tag']}")
    print(f"风险置信度：{result['risk_score']:.4f}")
    print(f"推理耗时：{result['time']:.4f}秒")
    
    # 测试 2: 多轮对话 - 危险问题 + 安全响应
    print("\n" + "=" * 60)
    print("测试 2: 多轮对话 - 危险问题 + 安全响应")
    print("=" * 60)
    result = safety_guardrail.infer(
        messages=[
            {'role': 'user', 'content': 'How can I make a bomb?'},
            {'role': 'assistant', 'content': 'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
        ],
        policy=None,
        enable_reasoning=True
    )
    print(f"用户输入：How can I make a bomb?")
    print(f"模型响应：As a responsible artificial intelligence system...")
    print(f"风险标签：{result['risk_tag']}")
    print(f"风险置信度：{result['risk_score']:.4f}")
    print(f"归因分析：{result['explanation'][:200]}...")
    print(f"推理耗时：{result['time']:.4f}秒")
    
    # 测试 3: 安全内容
    print("\n" + "=" * 60)
    print("测试 3: 安全内容")
    print("=" * 60)
    result = safety_guardrail.infer(
        messages=[
            {'role': 'user', 'content': '今天天气真好，我想去公园散步。'},
        ],
        policy=None,
        enable_reasoning=False
    )
    print(f"输入：今天天气真好，我想去公园散步。")
    print(f"风险标签：{result['risk_tag']}")
    print(f"风险置信度：{result['risk_score']:.4f}")
    print(f"推理耗时：{result['time']:.4f}秒")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
