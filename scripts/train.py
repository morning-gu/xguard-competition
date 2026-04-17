"""训练启动脚本

用法:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --base_model_path Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B --use_lora true
"""

import os
import sys
import argparse
import yaml
import logging

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import train, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="XGuard 模型微调训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="训练配置文件路径",
    )
    args, remaining = parser.parse_known_args()

    # 加载 YAML 配置
    config_kwargs = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_kwargs = yaml.safe_load(f) or {}
        print(f"已加载配置: {args.config}")

    # 命令行参数覆盖
    override_parser = argparse.ArgumentParser()
    for key, value in TrainConfig().__dict__.items():
        if isinstance(value, bool):
            override_parser.add_argument(f"--{key}", type=str, default=None)
        elif isinstance(value, int):
            override_parser.add_argument(f"--{key}", type=int, default=None)
        elif isinstance(value, float):
            override_parser.add_argument(f"--{key}", type=float, default=None)
        elif isinstance(value, str):
            override_parser.add_argument(f"--{key}", type=str, default=None)

    override_args, _ = override_parser.parse_known_args(remaining)
    for key, value in vars(override_args).items():
        if value is not None:
            if isinstance(getattr(TrainConfig(), key), bool):
                config_kwargs[key] = value.lower() in ("true", "1", "yes")
            else:
                config_kwargs[key] = value

    # 执行训练
    config = TrainConfig(**config_kwargs)
    output_dir = train(config)
    print(f"\n训练完成! 模型保存到: {output_dir}")


if __name__ == "__main__":
    main()
