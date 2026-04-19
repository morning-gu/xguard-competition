"""训练启动脚本

用法:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --base_model_path Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B --use_lora true
    python scripts/train.py --eval_after_train  # 训练后自动评估
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
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, test_data_path: str = "test_dataset/xguard_test_open_1k.jsonl"):
    """评估模型

    Args:
        model_path: 模型路径
        test_data_path: 测试数据路径
    """
    logger.info(f"\n{'='*60}")
    logger.info("开始评估模型...")
    logger.info(f"{'='*60}")

    # 加载测试数据
    from src.data.dataset import load_xguard_test_data
    test_data = load_xguard_test_data(test_data_path)
    logger.info(f"加载测试数据: {len(test_data)} 条")

    # 加载模型
    from inference import Guardrail
    import torch
    device_id = 0 if torch.cuda.is_available() else -1
    guardrail = Guardrail(model_path, device_id=device_id)
    logger.info(f"模型加载完成: {model_path}")

    # 评估
    from src.evaluation.evaluate import evaluate_on_test_set
    output_path = os.path.join(model_path, "eval_results.json")
    results = evaluate_on_test_set(guardrail, test_data, output_path=output_path)

    logger.info(f"\n评估完成!")
    logger.info(f"F1 (binary): {results['f1_binary']:.4f}")
    logger.info(f"F1 (macro): {results['f1_macro']:.4f}")
    logger.info(f"平均耗时: {results['avg_time']:.4f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="XGuard 模型微调训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--eval_after_train",
        action="store_true",
        help="训练完成后自动评估",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="test_dataset/xguard_test_open_1k.jsonl",
        help="测试数据路径 (用于评估)",
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

    # 训练后评估
    if args.eval_after_train:
        evaluate_model(output_dir, args.test_data_path)


if __name__ == "__main__":
    main()
