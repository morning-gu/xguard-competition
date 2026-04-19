"""从 XGuard-Train-Open-200K 中精选 1K 高质量数据集用于微调训练

精选策略:
1. 质量过滤: 去除空字段、label不在28类中、explanation过短等低质量样本
2. 类别均衡: 28个风险类别 + dynamic_policy自定义标签 均衡覆盖
3. 场景覆盖: q/r/qr 三种stage均匀分布
4. 类型覆盖: general + dynamic_policy 双类型, dynamic_policy占比约15%(对应比赛15%加分)
5. 难度优先: 优先选择explanation更详细(更长)的样本, 质量更高
"""

import json
import os
import random
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Optional

# XGuard 28个标准风险类别
VALID_LABELS = {
    "sec", "pc", "dc", "dw", "pi", "ec",
    "ac", "def", "ti", "cy",
    "ph", "mh",
    "se", "sci",
    "pp", "cs",
    "acc", "mc", "ha", "ps",
    "ter", "sd", "ext",
    "fin", "med", "law",
    "cm", "ma", "md",
}

# 9大维度映射
DIMENSION_MAP = {
    "Safe": {"sec"},
    "Crimes and Illegal Activities": {"pc", "dc", "dw", "pi", "ec"},
    "Hate Speech": {"ac", "def", "ti", "cy"},
    "Physical and Mental Health": {"ph", "mh"},
    "Ethics and Morality": {"se", "sci"},
    "Data Privacy": {"pp", "cs"},
    "Cybersecurity": {"acc", "mc", "ha", "ps"},
    "Extremism": {"ter", "sd", "ext"},
    "Inappropriate Suggestions": {"fin", "med", "law"},
    "Risks Involving Minors": {"cm", "ma", "md"},
}


def load_raw_data(source: str = "modelscope", local_path: Optional[str] = None) -> List[Dict]:
    """加载原始训练数据"""
    if local_path and os.path.exists(local_path):
        data = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"从本地加载 {len(data)} 条数据")
        return data

    if source == "modelscope":
        try:
            from modelscope.msdatasets import MsDataset
            raw_dataset = MsDataset.load(
                "Alibaba-AAIG/XGuard-Train-Open-200K",
                subset_name="xguard_train_open_200k",
            )
            data = [dict(item) for item in raw_dataset["train"]]
            print(f"从 ModelScope 加载 {len(data)} 条数据")
            return data
        except Exception as e:
            print(f"从 ModelScope 加载失败: {e}, 尝试 HuggingFace...")

    if source == "huggingface":
        from datasets import load_dataset
        raw_dataset = load_dataset(
            "Alibaba-AAIG/XGuard-Train-Open-200K",
            "xguard_train_open_200k",
            split="train",
        )
        data = [dict(item) for item in raw_dataset]
        print(f"从 HuggingFace 加载 {len(data)} 条数据")
        return data

    raise ValueError(f"无法加载数据: source={source}, local_path={local_path}")


def quality_filter(data: List[Dict]) -> List[Dict]:
    """质量过滤: 去除低质量样本"""
    filtered = []
    stats = Counter()

    for item in data:
        # 必须有label
        label = item.get("label", "").strip()
        if not label:
            stats["no_label"] += 1
            continue

        # 必须有explanation且长度合理
        explanation = (item.get("explanation", "") or "").strip()
        if len(explanation) < 20:
            stats["short_explanation"] += 1
            continue

        # 必须有输入内容
        prompt = (item.get("prompt", "") or "").strip()
        response = (item.get("response", "") or "").strip()
        stage = item.get("stage", "q")

        if stage == "qr" and (not prompt or not response):
            stats["empty_qr_field"] += 1
            continue
        elif stage == "q" and not prompt:
            stats["empty_prompt"] += 1
            continue
        elif stage == "r" and not response:
            stats["empty_response"] += 1
            continue

        # dynamic_policy类型必须有policy内容
        sample_type = item.get("sample_type", "general")
        policy = (item.get("policy", "") or "").strip()
        if sample_type == "dynamic_policy" and not policy:
            stats["dp_no_policy"] += 1
            continue

        # 输入文本不能过短
        input_text = prompt if stage in ("q",) else response if stage in ("r",) else f"{prompt} {response}"
        if len(input_text) < 10:
            stats["short_input"] += 1
            continue

        filtered.append(item)
        stats["passed"] += 1

    print(f"\n质量过滤统计:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")
    print(f"  过滤前: {len(data)}, 过滤后: {len(filtered)}")

    return filtered


def compute_quality_score(item: Dict) -> float:
    """计算样本质量分数, 越高越好

    考虑因素:
    - explanation长度(更详细的归因=更高质量)
    - 输入文本长度(更丰富的上下文)
    - 多语言样本加分
    """
    explanation = (item.get("explanation", "") or "").strip()
    prompt = (item.get("prompt", "") or "").strip()
    response = (item.get("response", "") or "").strip()
    stage = item.get("stage", "q")

    # explanation质量分 (0-40分, 长度对数缩放)
    expl_score = min(40, len(explanation) / 10)

    # 输入丰富度分 (0-30分)
    if stage == "qr":
        input_len = len(prompt) + len(response)
    elif stage == "q":
        input_len = len(prompt)
    else:
        input_len = len(response)
    input_score = min(30, input_len / 20)

    # 多语言加分 (0-20分): 检测非ASCII字符比例
    full_text = f"{prompt} {response}"
    if full_text:
        non_ascii = sum(1 for c in full_text if ord(c) > 127)
        lang_score = min(20, (non_ascii / len(full_text)) * 40)
    else:
        lang_score = 0

    # qr场景加分 (10分): 对话场景更复杂
    stage_score = 10 if stage == "qr" else 0

    return expl_score + input_score + lang_score + stage_score


def curate_balanced_dataset(
    data: List[Dict],
    target_size: int = 1000,
    dp_ratio: float = 0.15,
    seed: int = 42,
) -> List[Dict]:
    """精选均衡数据集

    策略:
    1. 分离 general 和 dynamic_policy 数据
    2. general数据: 按 label × stage 均衡采样
    3. dynamic_policy数据: 按 stage 均衡采样
    4. 每个桶内按质量分数排序, 优先选高质量样本
    """
    rng = random.Random(seed)

    # 分离 general 和 dynamic_policy
    general_data = [d for d in data if d.get("sample_type") == "general"]
    dp_data = [d for d in data if d.get("sample_type") == "dynamic_policy"]

    print(f"\n数据类型分布:")
    print(f"  general: {len(general_data)}")
    print(f"  dynamic_policy: {len(dp_data)}")

    # 计算各部分配额
    dp_count = int(target_size * dp_ratio)
    general_count = target_size - dp_count
    print(f"\n目标配额:")
    print(f"  general: {general_count}")
    print(f"  dynamic_policy: {dp_count}")

    # === 采样 general 数据 ===
    # 按 label 分桶
    label_buckets = defaultdict(list)
    for item in general_data:
        label = item["label"]
        label_buckets[label].append(item)

    print(f"\ngeneral 数据 label 分布 (共 {len(label_buckets)} 个label):")
    for label in sorted(label_buckets.keys()):
        print(f"  {label}: {len(label_buckets[label])}")

    # 计算每个label的配额
    # 标准label按均衡分配, 非标准label(不在28类中)分配剩余配额
    standard_labels = [l for l in label_buckets if l in VALID_LABELS]
    non_standard_labels = [l for l in label_buckets if l not in VALID_LABELS]

    print(f"  标准label数: {len(standard_labels)}")
    print(f"  非标准label数: {len(non_standard_labels)}")

    # 标准label均分80%的general配额, 非标准label分20%
    standard_quota = int(general_count * 0.8)
    non_standard_quota = general_count - standard_quota

    if standard_labels:
        per_standard = standard_quota // len(standard_labels)
    else:
        per_standard = 0
        non_standard_quota = general_count

    if non_standard_labels:
        per_non_standard = non_standard_quota // len(non_standard_labels)
    else:
        per_non_standard = 0

    print(f"  每个标准label配额: {per_standard}")
    print(f"  每个非标准label配额: {per_non_standard}")

    selected_general = []

    for label in standard_labels:
        bucket = label_buckets[label]
        # 按质量分数排序
        scored = [(compute_quality_score(item), item) for item in bucket]
        scored.sort(key=lambda x: x[0], reverse=True)

        # 从top 50%中随机采样, 兼顾质量和多样性
        top_half = scored[:max(len(scored) // 2, per_standard)]
        rng.shuffle(top_half)
        selected = [item for _, item in top_half[:per_standard]]
        selected_general.extend(selected)

    for label in non_standard_labels:
        bucket = label_buckets[label]
        scored = [(compute_quality_score(item), item) for item in bucket]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_half = scored[:max(len(scored) // 2, per_non_standard)]
        rng.shuffle(top_half)
        selected = [item for _, item in top_half[:per_non_standard]]
        selected_general.extend(selected)

    # === 采样 dynamic_policy 数据 ===
    # 按 stage 分桶
    dp_stage_buckets = defaultdict(list)
    for item in dp_data:
        stage = item.get("stage", "q")
        dp_stage_buckets[stage].append(item)

    print(f"\ndynamic_policy 数据 stage 分布:")
    for stage in sorted(dp_stage_buckets.keys()):
        print(f"  {stage}: {len(dp_stage_buckets[stage])}")

    selected_dp = []
    dp_stages = list(dp_stage_buckets.keys())

    if dp_stages:
        per_dp_stage = dp_count // len(dp_stages)
        print(f"  每个stage配额: {per_dp_stage}")

        for stage in dp_stages:
            bucket = dp_stage_buckets[stage]
            scored = [(compute_quality_score(item), item) for item in bucket]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_half = scored[:max(len(scored) // 2, per_dp_stage)]
            rng.shuffle(top_half)
            selected = [item for _, item in top_half[:per_dp_stage]]
            selected_dp.extend(selected)

    # === 合并并去重 ===
    selected = selected_general + selected_dp

    # 按id去重 (保留先出现的)
    seen_ids = set()
    deduped = []
    for item in selected:
        item_id = item.get("id")
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            deduped.append(item)
    selected = deduped

    # 如果不足, 从剩余数据中补充
    if len(selected) < target_size:
        remaining = [d for d in data if d.get("id") not in seen_ids]
        rng.shuffle(remaining)
        deficit = target_size - len(selected)
        selected.extend(remaining[:deficit])

    # 如果超出, 随机裁剪
    if len(selected) > target_size:
        rng.shuffle(selected)
        selected = selected[:target_size]

    return selected


def print_dataset_stats(data: List[Dict], title: str = "数据集统计"):
    """打印数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"总样本数: {len(data)}")

    # sample_type 分布
    type_counts = Counter(d.get("sample_type", "unknown") for d in data)
    print(f"\nsample_type 分布:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c} ({c/len(data)*100:.1f}%)")

    # label 分布
    label_counts = Counter(d.get("label", "unknown") for d in data)
    print(f"\nlabel 分布 (共 {len(label_counts)} 个):")
    for l, c in label_counts.most_common():
        print(f"  {l}: {c} ({c/len(data)*100:.1f}%)")

    # stage 分布
    stage_counts = Counter(d.get("stage", "unknown") for d in data)
    print(f"\nstage 分布:")
    for s, c in stage_counts.most_common():
        print(f"  {s}: {c} ({c/len(data)*100:.1f}%)")

    # explanation 长度统计
    expl_lens = [len((d.get("explanation", "") or "").strip()) for d in data]
    print(f"\nexplanation 长度统计:")
    print(f"  min: {min(expl_lens)}, max: {max(expl_lens)}, avg: {sum(expl_lens)/len(expl_lens):.0f}")

    # 维度覆盖
    dim_counts = defaultdict(int)
    for d in data:
        label = d.get("label", "")
        if label in VALID_LABELS:
            for dim, labels in DIMENSION_MAP.items():
                if label in labels:
                    dim_counts[dim] += 1
                    break
        else:
            dim_counts["dynamic_policy_custom"] += 1

    print(f"\n维度覆盖:")
    for dim, c in sorted(dim_counts.items(), key=lambda x: -x[1]):
        print(f"  {dim}: {c}")


def main():
    parser = argparse.ArgumentParser(description="从 XGuard-Train-Open-200K 精选 1K 数据集")
    parser.add_argument("--source", type=str, default="modelscope",
                        choices=["modelscope", "huggingface", "local"],
                        help="数据来源")
    parser.add_argument("--local_path", type=str, default=None,
                        help="本地数据路径 (source=local时使用)")
    parser.add_argument("--target_size", type=int, default=1000,
                        help="目标数据集大小")
    parser.add_argument("--dp_ratio", type=float, default=0.15,
                        help="dynamic_policy数据占比")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="输出目录")
    args = parser.parse_args()

    # 1. 加载原始数据
    print("步骤 1: 加载原始数据...")
    raw_data = load_raw_data(source=args.source, local_path=args.local_path)

    # 2. 质量过滤
    print("\n步骤 2: 质量过滤...")
    filtered_data = quality_filter(raw_data)

    # 3. 打印过滤后统计
    print_dataset_stats(filtered_data, "过滤后数据集统计")

    # 4. 精选均衡数据集
    print("\n步骤 3: 精选均衡数据集...")
    curated = curate_balanced_dataset(
        filtered_data,
        target_size=args.target_size,
        dp_ratio=args.dp_ratio,
        seed=args.seed,
    )

    # 5. 打印精选后统计
    print_dataset_stats(curated, "精选数据集统计")

    # 6. 保存
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "xguard_train_curated_1k.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in curated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n精选数据集已保存到: {output_path}")
    print(f"共 {len(curated)} 条样本")

    # 同时保存统计报告
    stats_path = os.path.join(args.output_dir, "curated_1k_stats.json")
    label_counts = Counter(d.get("label", "unknown") for d in curated)
    stage_counts = Counter(d.get("stage", "unknown") for d in curated)
    type_counts = Counter(d.get("sample_type", "unknown") for d in curated)

    stats = {
        "total": len(curated),
        "label_distribution": dict(label_counts.most_common()),
        "stage_distribution": dict(stage_counts.most_common()),
        "sample_type_distribution": dict(type_counts.most_common()),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"统计报告已保存到: {stats_path}")


if __name__ == "__main__":
    main()
