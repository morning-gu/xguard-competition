"""
XGuard-Train-Open-200K 数据集下载脚本
数据集来源: https://modelscope.cn/datasets/Alibaba-AAIG/XGuard-Train-Open-200K

使用方法:
1. 安装依赖: pip install modelscope
2. 运行脚本: python download_xguard_dataset.py
"""

import os
import json
from modelscope.msdatasets import MsDataset

# 配置
DATASET_ID = 'Alibaba-AAIG/XGuard-Train-Open-200K'
DOWNLOAD_DIR = './data/XGuard-Train-Open-200K'
OUTPUT_FILE = os.path.join(DOWNLOAD_DIR, 'xguard_train_open_200k.jsonl')

def download_dataset():
    """下载并保存数据集"""
    
    # 创建下载目录
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print("=" * 70)
    print("XGuard-Train-Open-200K 数据集下载工具")
    print("=" * 70)
    print(f"数据集ID: {DATASET_ID}")
    print(f"下载目录: {os.path.abspath(DOWNLOAD_DIR)}")
    print(f"输出文件: {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 70)
    print()
    
    try:
        # 使用 ModelScope SDK 加载数据集
        print("[1/3] 正在加载数据集...")
        ds = MsDataset.load(
            DATASET_ID,
            split='train'
        )
        print(f"[OK] 数据集加载成功!")
        print(f"[OK] 数据集大小: {len(ds)} 条数据")
        print()
        
        # 查看数据集结构
        print("[2/3] 数据集字段:")
        if len(ds) > 0:
            first_item = ds[0]
            for key in first_item.keys():
                print(f"  - {key}")
        print()
        
        # 保存数据到本地文件
        print(f"[3/3] 正在保存数据到: {OUTPUT_FILE}")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(ds):
                # 将每条数据写入 JSONL 文件
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
                
                if (idx + 1) % 10000 == 0:
                    print(f"  已处理 {idx + 1} 条数据...")
        
        print(f"[OK] 数据保存完成!")
        print(f"[OK] 总计保存 {len(ds)} 条数据")
        print()
        print("=" * 70)
        print(f"数据集已下载到: {os.path.abspath(DOWNLOAD_DIR)}")
        print("=" * 70)
        
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("请确保已安装 modelscope:")
        print("  pip install modelscope")
        print()
        print("如果需要登录,请先执行:")
        print("  from modelscope.hub.api import HubApi")
        print("  api = HubApi()")
        print("  api.login('your_access_token')")


def preview_data(num_samples=3):
    """预览已下载的数据"""
    
    if not os.path.exists(OUTPUT_FILE):
        print(f"数据文件不存在: {OUTPUT_FILE}")
        return
    
    print("=" * 70)
    print("数据预览")
    print("=" * 70)
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for i in range(num_samples):
            line = f.readline()
            if not line:
                break
            
            data = json.loads(line)
            print(f"\n[样本 {i+1}]")
            print(f"ID: {data.get('id', 'N/A')}")
            print(f"样本类型: {data.get('sample_type', 'N/A')}")
            print(f"阶段: {data.get('stage', 'N/A')}")
            print(f"标签: {data.get('label', 'N/A')}")
            
            prompt = data.get('prompt', 'N/A')
            if prompt and prompt != 'N/A':
                print(f"Prompt: {prompt[:100]}...")
            else:
                print(f"Prompt: {prompt}")
            
            response = data.get('response', 'N/A')
            if response and response != 'N/A':
                print(f"Response: {response[:100]}...")
            else:
                print(f"Response: {response}")
    
    print()
    print("=" * 70)


if __name__ == '__main__':
    # 下载数据集
    download_dataset()
    
    # 预览数据
    print()
    preview_data(num_samples=2)
