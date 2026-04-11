# 使用 ModelScope SDK 下载 YuFeng-XGuard-Reason-0.6B 模型

"""
本脚本演示如何使用 ModelScope SDK 下载模型

安装依赖:
    pip install modelscope

使用方法:
    python download_model.py
"""

from modelscope import snapshot_download


def download_model():
    """下载 YuFeng-XGuard-Reason-0.6B 模型"""
    
    model_id = "Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B"
    
    print(f"正在下载模型：{model_id}")
    print("这可能需要一些时间，取决于您的网络连接...")
    
    # 下载模型到本地缓存目录
    model_dir = snapshot_download(model_id)
    
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
