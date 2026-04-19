"""XGuard 护栏模型推理代码

比赛要求的标准化推理接口, 遵循赛事说明书规定的 Guardrail 类和 infer 方法签名。
支持:
- 一般文本内容 (stage=q/r) 和 Query & Response 对话 (stage=qr)
- 细粒度风险类别、细粒度风险置信度、归因分析
- 动态策略 (通过 policy 参数)
"""

import os
import time
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional

# 修复: transformers<4.52 不支持 Qwen3 架构
# Qwen3 在 transformers 4.52.0 才加入, 但赛事要求 transformers==4.51.0.
# 解决方案: 将 Qwen3 注册为 Qwen2 的别名 (Qwen3 架构与 Qwen2 高度相似).
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    if "qwen3" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("qwen3", Qwen2Config, exist_ok=True)
        MODEL_FOR_CAUSAL_LM_MAPPING.register(Qwen2Config, Qwen2ForCausalLM, exist_ok=True)
except Exception:
    pass

# 修复: 禁用 PEFT 对 awq 的自动检测
# awq 包与 transformers>=4.51 不兼容 (shard_checkpoint 已移除),
# 但 PEFT 的 LoRA dispatcher 仍会尝试 from awq.modules.linear import WQLinear_GEMM,
# 导致 ImportError. 直接将 dispatch_awq 替换为空操作来跳过 awq 检测.
try:
    import peft.tuners.lora.model as _peft_lora_model
    _peft_lora_model.dispatch_awq = lambda target, adapter_name, **kwargs: None
except Exception:
    pass


class Guardrail:
    """XGuard 安全护栏模型

    遵循赛事规定的接口:
    - __init__(model_path, device_id): 加载模型和 tokenizer
    - infer(messages, policy, enable_reasoning): 执行推理
    """

    # LoRA checkpoint 中基座模型的远程 ID
    BASE_MODEL_ID = "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"

    def __init__(self, model_path: str, device_id: int = 0):
        """初始化护栏模型

        Args:
            model_path: 模型路径 (项目根目录, 本地路径或 HuggingFace/ModelScope model ID)
                支持完整模型路径和 LoRA checkpoint 路径
            device_id: GPU 设备 ID, 所有模型加载到 cuda:{device_id}
        """
        # 确定 device
        if torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
            print(f"[Guardrail] CUDA 可用, 使用设备: {self.device}")
        else:
            self.device = "cpu"
            print(f"[Guardrail] CUDA 不可用, 使用 CPU")

        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            # LoRA checkpoint: 先在 CPU 加载基座模型, 合并 LoRA 后再移到指定 GPU
            from peft import PeftModel

            print(f"[Guardrail] 检测到 LoRA checkpoint, 加载基座模型: {self.BASE_MODEL_ID}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            print(f"[Guardrail] 加载 LoRA 适配器: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print(f"[Guardrail] 合并 LoRA 并移至 {self.device}")
            self.model = self.model.merge_and_unload()
            self.model = self.model.to(self.device)
            self.model = self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.BASE_MODEL_ID,
                trust_remote_code=True,
            )
        else:
            # 完整模型: 直接加载到指定设备
            print(f"[Guardrail] 加载完整模型: {model_path} -> {self.device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        # 打印模型设备信息
        if hasattr(self.model, 'device'):
            print(f"[Guardrail] 模型实际设备: {self.model.device}")
        else:
            # 检查第一个参数的设备
            try:
                first_param = next(self.model.parameters())
                print(f"[Guardrail] 模型参数设备: {first_param.device}")
            except StopIteration:
                print(f"[Guardrail] 模型无参数")

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

        # 模型生成 - 优化: 关闭 output_scores, 启用 KV cache 加速
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = model_inputs["input_ids"].shape[1]

        # 解码生成的 token
        output_ids = outputs[0].tolist()[input_length:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 解析风险标签 (第一个 token)
        risk_tag = self._parse_risk_tag(output_ids)

        # 记录得到 risk_tag 的时间
        end_time = time.time()
        infer_time = end_time - start_time

        # 解析风险置信度 - 简化版: 基于首 token 置信度
        risk_score = self._parse_risk_score_simple(output_ids)

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

    def _parse_risk_score_simple(self, output_ids: list) -> float:
        """简化版风险置信度 (不依赖 output_scores)

        Args:
            output_ids: 生成的 token ID 列表

        Returns:
            风险置信度分数 (基于是否为 safe 标签)
        """
        if not output_ids:
            return 0.0

        # 解码首 token
        first_token_id = output_ids[0]
        risk_tag = self.tokenizer.decode([first_token_id], skip_special_tokens=True).strip()

        # 如果是 safe 标签, risk_score 较低; 否则较高
        if risk_tag == "sec" or risk_tag.lower() == "safe":
            return 0.1  # safe 的风险分数
        else:
            return 0.9  # unsafe 的风险分数

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
    device_id = 0

    safety_guardrail = Guardrail(model_path, device_id)

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
