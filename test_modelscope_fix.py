#!/usr/bin/env python3
"""测试 ModelScope 修复脚本"""

import os
from pathlib import Path

print("=" * 60)
print("测试 ModelScope 修复")
print("=" * 60)

# 1. 检查环境变量设置
print("\n1. 检查环境变量:")
print(f"   HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
print(f"   MODELSCOPE_CACHE: {os.environ.get('MODELSCOPE_CACHE', '未设置')}")

# 2. 测试导入
print("\n2. 测试模块导入:")
try:
    from src.model.loader import load_model_and_tokenizer, download_model
    print("   ✓ src.model.loader 导入成功")
except Exception as e:
    print(f"   ✗ src.model.loader 导入失败：{e}")

try:
    from inference import Guardrail
    print("   ✓ inference.Guardrail 导入成功")
except Exception as e:
    print(f"   ✗ inference.Guardrail 导入失败：{e}")

# 3. 检查代码中是否设置了 HF_ENDPOINT
print("\n3. 检查关键文件中的 HF_ENDPOINT 设置:")

files_to_check = [
    '/workspace/src/model/loader.py',
    '/workspace/inference.py',
    '/workspace/src/inference/engine.py',
    '/workspace/src/data/loader.py',
    '/workspace/src/training/train_core.py',
]

for filepath in files_to_check:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if "HF_ENDPOINT" in content and "modelscope.cn" in content:
                print(f"   ✓ {filepath} - 已设置 HF_ENDPOINT")
            else:
                print(f"   ✗ {filepath} - 未设置 HF_ENDPOINT")
    except Exception as e:
        print(f"   ✗ {filepath} - 读取失败：{e}")

# 4. 检查 load_model_and_tokenizer 函数的逻辑
print("\n4. 检查模型加载逻辑:")
import inspect
from src.model.loader import load_model_and_tokenizer
source = inspect.getsource(load_model_and_tokenizer)
if "download_model" in source and "local_model_path" in source:
    print("   ✓ 模型加载函数包含本地缓存检查和下载逻辑")
else:
    print("   ✗ 模型加载函数缺少本地缓存检查和下载逻辑")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print("\n修复说明:")
print("1. 在所有导入 transformers 的文件开头设置了 HF_ENDPOINT=https://modelscope.cn")
print("2. 修改了 load_model_and_tokenizer 函数，优先使用本地缓存")
print("3. 当传入模型 ID 时，会先通过 snapshot_download 从 ModelScope 下载到本地")
print("4. 下载完成后使用本地路径加载，避免 transformers 访问 HuggingFace")
