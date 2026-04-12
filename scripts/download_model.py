# 使用 ModelScope SDK 下载 YuFeng-XGuard-Reason-0.6B 模型

"""
本脚本演示如何使用 ModelScope SDK 下载模型

安装依赖:
    pip install modelscope

使用方法:
    python download_model.py
"""

import os
from pathlib import Path
from modelscope import snapshot_download


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型配置
MODEL_ID = "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"
# 本地保存目录 (保持原始模型名称, 不替换 . 为 _)
LOCAL_DIR = PROJECT_ROOT / "models" / "pretrained" / "Alibaba-AAIG" / "YuFeng-XGuard-Reason-0.6B"


def download_model():
    """下载 YuFeng-XGuard-Reason-0.6B 模型"""
    
    print(f"正在下载模型：{MODEL_ID}")
    print(f"本地保存路径：{LOCAL_DIR}")
    print("这可能需要一些时间，取决于您的网络连接...")
    
    # 确保目标目录存在
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 下载模型到指定目录
    model_dir = snapshot_download(
        MODEL_ID,
        local_dir=str(LOCAL_DIR),
        revision="master"
    )
    
    print(f"\n模型下载完成!")
    print(f"模型路径：{model_dir}")
    
    return model_dir


if __name__ == "__main__":
    model_path = download_model()
    print(f"\n现在可以使用以下代码加载模型:")
    print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{model_path}")
model = AutoModelForCausalLM.from_pretrained("{model_path}", torch_dtype="auto", device_map="auto")
    """)
