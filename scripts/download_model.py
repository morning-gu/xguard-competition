# 使用 ModelScope SDK 下载 YuFeng-XGuard-Reason 模型

"""
本脚本演示如何使用 ModelScope SDK 下载模型

安装依赖:
    pip install modelscope

使用方法:
    # 下载默认模型 (0.6B)
    python download_model.py

    # 下载 8B 模型
    python download_model.py --model 8B

    # 下载 0.6B 模型
    python download_model.py --model 0.6B

    # 指定自定义模型 ID
    python download_model.py --model_id Alibaba-AAIG/YuFeng-XGuard-Reason-8B
"""

import os
import argparse
from pathlib import Path
from modelscope import snapshot_download


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 支持的模型版本
MODEL_VERSIONS = {
    "0.6B": "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B",
    "8B": "Alibaba-AAIG/YuFeng-XGuard-Reason-8B",
}


def download_model(model_id: str = None, model_version: str = "0.6B"):
    """
    下载 YuFeng-XGuard-Reason 模型
    
    Args:
        model_id: 完整模型 ID (如 Alibaba-AAIG/YuFeng-XGuard-Reason-8B)
        model_version: 模型版本 (0.6B 或 8B)
    """
    # 确定模型 ID
    if model_id:
        final_model_id = model_id
    elif model_version in MODEL_VERSIONS:
        final_model_id = MODEL_VERSIONS[model_version]
    else:
        print(f"错误: 不支持的模型版本 '{model_version}'")
        print(f"支持的版本: {list(MODEL_VERSIONS.keys())}")
        return None
    
    # 解析模型名称 (如 Alibaba-AAIG/YuFeng-XGuard-Reason-8B -> YuFeng-XGuard-Reason-8B)
    model_name = final_model_id.split("/")[-1]
    org_name = final_model_id.split("/")[0] if "/" in final_model_id else ""
    
    # 本地保存目录
    if org_name:
        local_dir = PROJECT_ROOT / "models" / "pretrained" / org_name / model_name
    else:
        local_dir = PROJECT_ROOT / "models" / "pretrained" / model_name
    
    print(f"正在下载模型：{final_model_id}")
    print(f"本地保存路径：{local_dir}")
    print("这可能需要一些时间，取决于您的网络连接...")
    
    # 确保目标目录存在
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载模型到指定目录
    model_dir = snapshot_download(
        final_model_id,
        local_dir=str(local_dir),
        revision="master"
    )
    
    print(f"\n模型下载完成!")
    print(f"模型路径：{model_dir}")
    
    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 YuFeng-XGuard-Reason 模型")
    parser.add_argument(
        "--model",
        type=str,
        default="0.6B",
        choices=["0.6B", "8B"],
        help="模型版本 (默认: 0.6B)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="完整模型 ID (覆盖 --model 参数)",
    )
    args = parser.parse_args()
    
    model_path = download_model(model_id=args.model_id, model_version=args.model)
    
    if model_path:
        print(f"\n现在可以使用以下代码加载模型:")
        print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{model_path}")
model = AutoModelForCausalLM.from_pretrained("{model_path}", torch_dtype="auto", device_map="auto")
        """)
