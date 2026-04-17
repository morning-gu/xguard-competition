"""XGuard 数据预处理模块

将原始训练数据转换为模型训练所需的 input/output 格式,
遵循 YuFeng-XGuard-Reason 的 prompt template 格式。
"""

import json
import os
from typing import List, Dict, Optional


# XGuard 风险类别定义
RISK_CATEGORIES = {
    "sec": "Safe",
    "pc": "Pornographic Contraband",
    "dc": "Drug Crimes",
    "dw": "Dangerous Weapons",
    "pi": "Property Infringement",
    "ec": "Economic Crimes",
    "ac": "Abusive Curses",
    "def": "Defamation",
    "ti": "Threats and Intimidation",
    "cy": "Cyberbullying",
    "ph": "Physical Health",
    "mh": "Mental Health",
    "se": "Social Ethics",
    "sci": "Science Ethics",
    "pp": "Personal Privacy",
    "cs": "Commercial Secret",
    "acc": "Access Control",
    "mc": "Malicious Code",
    "ha": "Hacker Attack",
    "ps": "Physical Security",
    "ter": "Violent Terrorist Activities",
    "sd": "Social Disruption",
    "ext": "Extremist Ideological Trends",
    "fin": "Finance",
    "med": "Medicine",
    "law": "Law",
    "cm": "Corruption of Minors",
    "ma": "Minor Abuse and Exploitation",
    "md": "Minor Delinquency",
}

# ID 到风险全名的映射 (用于 risk_tag 输出)
ID2RISK_FULL = {
    k: f"{dim}-{cat}"
    for k, cat in RISK_CATEGORIES.items()
    for dim in [
        "Safe" if k == "sec" else
        "Crimes and Illegal Activities" if k in ("pc", "dc", "dw", "pi", "ec") else
        "Hate Speech" if k in ("ac", "def", "ti", "cy") else
        "Physical and Mental Health" if k in ("ph", "mh") else
        "Ethics and Morality" if k in ("se", "sci") else
        "Data Privacy" if k in ("pp", "cs") else
        "Cybersecurity" if k in ("acc", "mc", "ha", "ps") else
        "Extremism" if k in ("ter", "sd", "ext") else
        "Inappropriate Suggestions" if k in ("fin", "med", "law") else
        "Risks Involving Minors"
    ]
}

# 默认 instruction template
DEFAULT_INSTRUCTION_TEMPLATE = """You are an expert in content safety evaluation, possessing strong language comprehension and risk identification skills.
Your task is to classify the provided input text into the most appropriate category from the list below as part of a safety audit project.

# Category List
{category_list}

{dynamic_policy_section}
# Instructions
- Identify the single most relevant category ID for the input text.
- On the next line, provide a concise justification for your choice, placing it between <explanation> and </explanation> tags.

---

Input Text: {input_text}"""


def build_category_list() -> str:
    """构建风险类别列表字符串"""
    lines = []
    for cat_id, cat_name in RISK_CATEGORIES.items():
        lines.append(f"- {cat_id}: {cat_name}")
    return "\n".join(lines)


def preprocess_single(
    raw_data: Dict,
    instruction_template: Optional[str] = None,
) -> Dict:
    """预处理单条训练数据

    Args:
        raw_data: 原始数据, 包含 id, sample_type, prompt, response, stage, policy, label, explanation
        instruction_template: 自定义 instruction template

    Returns:
        处理后的数据, 包含 id, input, output
    """
    if instruction_template is None:
        instruction_template = DEFAULT_INSTRUCTION_TEMPLATE

    sample_type = raw_data.get("sample_type", "general")
    prompt = raw_data.get("prompt", "") or ""
    response = raw_data.get("response", "") or ""
    stage = raw_data.get("stage", "q")
    policy = raw_data.get("policy", "") or ""
    label = raw_data.get("label", "")
    explanation = (raw_data.get("explanation", "") or "").strip()

    # 构建动态策略部分
    if sample_type == "dynamic_policy" and policy:
        dynamic_policy_section = "\n# Dynamic Policy\n" + policy.strip() + "\n"
    else:
        dynamic_policy_section = ""

    # 构建输入文本
    if stage == "qr":
        input_text = f"[User Query] {prompt.strip()}\n\n[LLM Response] {response.strip()}"
    elif stage == "q":
        input_text = prompt.strip()
    elif stage == "r":
        input_text = response.strip()
    else:
        input_text = prompt.strip()

    # 构建完整 input
    category_list = build_category_list()
    input_str = instruction_template.format(
        category_list=category_list,
        dynamic_policy_section=dynamic_policy_section,
        input_text=input_text,
    )

    # 构建输出: label + explanation
    output_str = f"{label}\n<explanation>\n{explanation}\n</explanation>"

    return {
        "id": raw_data.get("id", ""),
        "input": input_str,
        "output": output_str,
    }


def preprocess_for_training(
    raw_data_list: List[Dict],
    instruction_template: Optional[str] = None,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """批量预处理训练数据

    Args:
        raw_data_list: 原始数据列表
        instruction_template: 自定义 instruction template
        output_path: 处理后数据保存路径 (可选)

    Returns:
        处理后的数据列表
    """
    processed = []
    for raw_data in raw_data_list:
        item = preprocess_single(raw_data, instruction_template)
        processed.append(item)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return processed


def load_prompt_template(template_path: str) -> str:
    """加载自定义 prompt template"""
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()
