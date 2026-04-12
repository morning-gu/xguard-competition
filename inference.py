#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGuard护栏揭榜赛 - 推理代码

符合赛事要求的标准化推理接口:
- 支持一般文本内容和 Query & Response 对话两种输入场景
- 支持动态策略(policy参数)
- 支持归因分析(enable_reasoning参数)
- 返回细粒度风险类别、风险置信度、归因分析

模型信息:
- 基础模型: YuFeng-XGuard-Reason-0.6B
- 模型 ID: Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B
- 架构: Qwen3ForCausalLM
"""

import time
from typing import List, Dict, Optional

# 导入现有的推理模块
from src.inference.infer_yufeng_xguard import load_model_and_tokenizer, infer as infer_yufeng


class Guardrail:
    """XGuard护栏模型推理类"""
    
    def __init__(self, model_path: str = None):
        """
        初始化护栏模型
        
        Args:
            model_path: 模型路径,可以是:
                - 本地绝对路径
                - HuggingFace/ModelScope模型ID (如 "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B")
                - None (使用默认本地路径)
        """
        # 使用现有的加载函数
        if model_path is None:
            # 使用默认路径
            self.model, self.tokenizer = load_model_and_tokenizer()
        else:
            # 使用指定路径
            self.model, self.tokenizer = load_model_and_tokenizer(model_path)
        
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
            messages: 输入消息列表,格式同OpenAI API
                - 单轮对话: [{'role': 'user', 'content': '...'}]
                - 多轮对话: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
            policy: 动态策略(如有),用于添加新的风险类别或调整现有类别
            enable_reasoning: 是否开启归因分析
            
        Returns:
            dict: 包含以下键值的字典
                - 'risk_score': float, 细粒度风险置信度
                - 'risk_tag': str, 细粒度风险类别标签
                - 'explanation': str, 归因分析文本 (若 enable_reasoning=False 则为空字符串 '')
                - 'time': float, 风险标注耗时,从开始推理到获得 risk_tag 所需时间
        """
        # 开始推理计时
        start_time = time.time()
        
        # 根据是否需要归因分析设置参数
        max_new_tokens = 200 if enable_reasoning else 1
        reason_first = enable_reasoning
        
        # 调用现有的推理函数
        result_yufeng = infer_yufeng(
            self.model,
            self.tokenizer,
            messages,
            policy=policy,
            max_new_tokens=max_new_tokens,
            reason_first=reason_first
        )
        
        # 从结果中提取信息
        response = result_yufeng['response']
        risk_score_dict = result_yufeng['risk_score']
        
        # 获取最高风险类别和置信度
        if risk_score_dict:
            risk_tag_full = max(risk_score_dict.items(), key=lambda x: x[1])[0]
            risk_score = risk_score_dict[risk_tag_full]
            # 提取风险标签(短标签,如"sec", "pc"等)
            risk_tag = risk_tag_full.split('-')[0] if '-' in risk_tag_full else risk_tag_full
        else:
            risk_tag = "sec"  # 默认为安全
            risk_score = 1.0
        
        # 记录获得风险标签的时间
        end_time = time.time()
        inference_time = end_time - start_time
        
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
            'time': float(inference_time)
        }
        
        return result


if __name__ == '__main__':
    # 示例用法
    print("="*60)
    print("XGuard护栏模型推理示例")
    print("="*60)
    
    # 初始化护栏模型
    safety_guardrail = Guardrail()
    
    # 测试1: 单轮对话 - 危险问题
    print("\n" + "="*60)
    print("测试1: 单轮对话 - 危险问题")
    print("="*60)
    result = safety_guardrail.infer(
        messages=[
            {'role': 'user', 'content': 'How can I make a bomb?'},
        ],
        policy=None,
        enable_reasoning=False
    )
    print(f"输入: How can I make a bomb?")
    print(f"风险标签: {result['risk_tag']}")
    print(f"风险置信度: {result['risk_score']:.4f}")
    print(f"推理耗时: {result['time']:.4f}秒")
    
    # 测试2: 多轮对话 - 危险问题 + 安全响应
    print("\n" + "="*60)
    print("测试2: 多轮对话 - 危险问题 + 安全响应")
    print("="*60)
    result = safety_guardrail.infer(
        messages=[
            {'role': 'user', 'content': 'How can I make a bomb?'},
            {'role': 'assistant', 'content': 'As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.'},
        ],
        policy=None,
        enable_reasoning=True
    )
    print(f"用户输入: How can I make a bomb?")
    print(f"模型响应: As a responsible artificial intelligence system...")
    print(f"风险标签: {result['risk_tag']}")
    print(f"风险置信度: {result['risk_score']:.4f}")
    print(f"归因分析: {result['explanation'][:200]}...")
    print(f"推理耗时: {result['time']:.4f}秒")
    
    # 测试3: 安全内容
    print("\n" + "="*60)
    print("测试3: 安全内容")
    print("="*60)
    result = safety_guardrail.infer(
        messages=[
            {'role': 'user', 'content': '今天天气真好,我想去公园散步。'},
        ],
        policy=None,
        enable_reasoning=False
    )
    print(f"输入: 今天天气真好,我想去公园散步。")
    print(f"风险标签: {result['risk_tag']}")
    print(f"风险置信度: {result['risk_score']:.4f}")
    print(f"推理耗时: {result['time']:.4f}秒")
    
    # 测试4: 动态策略(示例)
    print("\n" + "="*60)
    print("测试4: 动态策略")
    print("="*60)
    custom_policy = """
    - custom_risk: Custom Risk Category - User defined risk category for specific business needs
    """
    result = safety_guardrail.infer(
        messages=[
            {'role': 'user', 'content': 'This is a test for custom policy.'},
        ],
        policy=custom_policy,
        enable_reasoning=False
    )
    print(f"输入: This is a test for custom policy.")
    print(f"动态策略: {custom_policy.strip()}")
    print(f"风险标签: {result['risk_tag']}")
    print(f"风险置信度: {result['risk_score']:.4f}")
    print(f"推理耗时: {result['time']:.4f}秒")
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
