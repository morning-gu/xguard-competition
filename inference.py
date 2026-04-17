"""XGuard 护栏模型推理代码

比赛要求的标准化推理接口, 遵循赛事说明书规定的 Guardrail 类和 infer 方法签名。
支持:
- 一般文本内容 (stage=q/r) 和 Query & Response 对话 (stage=qr)
- 细粒度风险类别、细粒度风险置信度、归因分析
- 动态策略 (通过 policy 参数)
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional


class Guardrail:
    """XGuard 安全护栏模型

    遵循赛事规定的接口:
    - __init__(model_path): 加载模型和 tokenizer
    - infer(messages, policy, enable_reasoning): 执行推理
    """

    def __init__(self, model_path: str):
        """初始化护栏模型

        Args:
            model_path: 模型路径 (本地路径或 HuggingFace/ModelScope model ID)
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # 获取 id2risk 映射 (YuFeng-XGuard 模型内置)
        self.id2risk = getattr(self.tokenizer, "init_kwargs", {}).get("id2risk", {})

    def infer(
        self,
        messages: List[Dict],
        policy: Optional[str] = None,
        enable_reasoning: bool = False,
    ) -> Dict:
        """执行安全护栏推理

        Args:
            messages: List[Dict], 输入消息列表, 格式同 OpenAI API
                - stage=q: [{'role':'user', 'content':'...'}]
                - stage=r: [{'role':'assistant', 'content':'...'}]
                - stage=qr: [{'role':'user', 'content':'...'}, {'role':'assistant', 'content':'...'}]
            policy: str, 动态策略 (如有)
            enable_reasoning: bool, 是否开启归因分析

        Returns:
            dict: 包含以下键值的字典
                - 'risk_score': float, 细粒度风险置信度
                - 'risk_tag': str, 细粒度风险类别标签
                - 'explanation': str, 归因分析文本 (若 enable_reasoning=False 则为空字符串 '')
                - 'time': float, 风险标注耗时, 从开始推理到获得 risk_tag 所需时间
        """
        start_time = time.time()

        # 确定 max_new_tokens: 仅判定时为 1, 需要归因时为 512
        max_new_tokens = 512 if enable_reasoning else 1

        # 使用 chat template 渲染输入
        rendered_query = self.tokenizer.apply_chat_template(
            messages,
            policy=policy,
            reason_first=False,
            tokenize=False,
        )

        model_inputs = self.tokenizer(
            [rendered_query],
            return_tensors="pt",
        ).to(self.model.device)

        # 模型生成
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        batch_idx = 0
        input_length = model_inputs["input_ids"].shape[1]

        # 解码生成的 token
        output_ids = outputs["sequences"].tolist()[batch_idx][input_length:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 解析风险标签 (第一个 token)
        risk_tag = self._parse_risk_tag(output_ids)

        # 记录得到 risk_tag 的时间
        end_time = time.time()
        infer_time = end_time - start_time

        # 解析风险置信度
        risk_score = self._parse_risk_score(outputs, input_length, batch_idx)

        # 解析归因分析
        explanation = ""
        if enable_reasoning and max_new_tokens > 1:
            explanation = self._parse_explanation(response)

        result = {
            "risk_score": risk_score,
            "risk_tag": risk_tag,
            "explanation": explanation,
            "time": infer_time,
        }
        return result

    def _parse_risk_tag(self, output_ids: list) -> str:
        """从生成的 token IDs 中解析风险标签

        Args:
            output_ids: 生成的 token ID 列表

        Returns:
            风险类别标签字符串
        """
        if not output_ids:
            return "sec"

        # 解码第一个 token 作为风险标签
        first_token_id = output_ids[0]
        risk_tag = self.tokenizer.decode([first_token_id], skip_special_tokens=True).strip()

        return risk_tag

    def _parse_risk_score(
        self,
        outputs,
        input_length: int,
        batch_idx: int = 0,
    ) -> float:
        """从模型输出中解析风险置信度

        Args:
            outputs: 模型 generate 输出
            input_length: 输入长度
            batch_idx: batch 索引

        Returns:
            最高风险类别的置信度分数
        """
        try:
            generated_tokens = outputs.sequences[:, input_length:]
            scores = torch.stack(outputs.scores, 1)
            scores = scores.softmax(-1)

            # 取第一个生成 token 的概率分布
            first_token_scores = scores[batch_idx, 0]

            # 获取生成的第一个 token
            generated_token = generated_tokens[batch_idx, 0]
            token_id = int(generated_token.cpu())

            # 获取该 token 的概率作为 risk_score
            risk_score = float(first_token_scores[token_id].cpu())

            # 如果有 id2risk 映射, 计算所有风险类别的总概率
            if self.id2risk:
                risk_total = 0.0
                for risk_id in self.id2risk:
                    if risk_id in self.tokenizer.get_vocab():
                        tid = self.tokenizer.get_vocab()[risk_id]
                        risk_total += float(first_token_scores[tid].cpu())
                # 如果有非 safe 的风险, 用 1 - safe 概率作为 risk_score
                if "sec" in self.id2risk and "sec" in self.tokenizer.get_vocab():
                    sec_id = self.tokenizer.get_vocab()["sec"]
                    risk_score = 1.0 - float(first_token_scores[sec_id].cpu())

            return round(risk_score, 4)
        except Exception:
            return 0.0

    def _parse_explanation(self, response: str) -> str:
        """从模型输出中解析归因分析文本

        Args:
            response: 模型生成的完整文本

        Returns:
            归因分析文本
        """
        # 提取 <explanation>...</explanation> 之间的内容
        start_tag = "<explanation>"
        end_tag = "</explanation>"

        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)

        if start_idx != -1 and end_idx != -1:
            explanation = response[start_idx + len(start_tag):end_idx].strip()
            return explanation

        # 如果没有找到标签, 返回整个 response (去掉风险标签行)
        lines = response.strip().split("\n")
        if len(lines) > 1:
            return "\n".join(lines[1:]).strip()

        return response.strip()


if __name__ == "__main__":
    # 示例: 使用 YuFeng-XGuard-Reason-0.6B 进行推理
    model_path = "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"

    safety_guardrail = Guardrail(model_path)

    # 示例 1: Query & Response 对话场景 (stage=qr)
    result = safety_guardrail.infer(
        messages=[
            {"role": "user", "content": "How can I make a bomb?"},
            {
                "role": "assistant",
                "content": "As a responsible artificial intelligence system, I regret that I am unable to fulfill your request.",
            },
        ],
        policy=None,
        enable_reasoning=True,
    )
    print("=== 示例 1: QR 对话 ===")
    print(result)

    # 示例 2: 仅 Query 场景 (stage=q)
    result = safety_guardrail.infer(
        messages=[
            {"role": "user", "content": "How can I make a bomb?"},
        ],
        policy=None,
        enable_reasoning=True,
    )
    print("\n=== 示例 2: 仅 Query ===")
    print(result)

    # 示例 3: 仅 Response 场景 (stage=r)
    result = safety_guardrail.infer(
        messages=[
            {
                "role": "assistant",
                "content": "To make a bomb, you need to mix chemicals A and B...",
            },
        ],
        policy=None,
        enable_reasoning=False,
    )
    print("\n=== 示例 3: 仅 Response (无归因) ===")
    print(result)
