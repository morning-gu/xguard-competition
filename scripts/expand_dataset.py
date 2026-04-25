"""扩展数据集: 使用大模型检测语言并翻译为多种语言

功能:
1. 语言检测: 调用 OpenAI 兼容接口检测数据集中每条数据的语言
2. 多语言翻译: 将数据翻译为配置的多种目标语言, 扩展数据集
3. 支持增量处理: 跳过已检测/已翻译的数据, 支持断点续传
4. 支持并发: 使用异步并发加速 API 调用

使用示例:
  # 检测语言
  python scripts/expand_dataset.py detect-language \\
    --input data/xguard_train_curated_1k.jsonl \\
    --output data/xguard_train_curated_1k_with_lang.jsonl

  # 翻译扩展 (翻译为中文、日语、韩语)
  python scripts/expand_dataset.py translate \\
    --input data/xguard_train_curated_1k_with_lang.jsonl \\
    --output data/xguard_train_curated_1k_expanded.jsonl \\
    --target-languages zh ja ko

  # 一键完成: 检测语言 + 翻译扩展
  python scripts/expand_dataset.py expand \\
    --input data/xguard_train_curated_1k.jsonl \\
    --output data/xguard_train_curated_1k_expanded.jsonl \\
    --target-languages zh ja ko fr de es ar ru
"""

import json
import os
import re
import argparse
import asyncio
import hashlib
import time
from typing import List, Dict, Optional, Set
from collections import Counter, defaultdict

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# ============================================================
# 语言代码 -> 语言名称映射 (用于 prompt)
# ============================================================
LANGUAGE_NAMES = {
    "zh": "Chinese (Simplified)",
    "zh-Hant": "Chinese (Traditional)",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "cs": "Czech",
    "he": "Hebrew",
    "id": "Indonesian",
    "ms": "Malay",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "el": "Greek",
}


# ============================================================
# 数据加载与保存
# ============================================================

def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"从 {path} 加载 {len(data)} 条数据")
    return data


def save_jsonl(data: List[Dict], path: str):
    """保存 JSONL 文件"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存 {len(data)} 条数据到 {path}")


# ============================================================
# 文本提取工具
# ============================================================

def extract_texts(item: Dict) -> Dict[str, str]:
    """从数据条目中提取需要检测/翻译的文本字段

    Returns:
        dict: {"prompt": ..., "response": ..., "explanation": ...}
              仅包含非空字段
    """
    texts = {}
    for field in ("prompt", "response", "explanation"):
        val = (item.get(field) or "").strip()
        if val:
            texts[field] = val
    return texts


def update_item_texts(item: Dict, translated_texts: Dict[str, str]) -> Dict:
    """用翻译后的文本更新数据条目 (不修改原条目, 返回新条目)"""
    new_item = dict(item)
    for field, text in translated_texts.items():
        if field in ("prompt", "response", "explanation"):
            new_item[field] = text
    return new_item


# ============================================================
# OpenAI 兼容接口调用
# ============================================================

def get_client(base_url: str, api_key: str) -> "AsyncOpenAI":
    """创建 AsyncOpenAI 客户端"""
    if AsyncOpenAI is None:
        raise ImportError("请安装 openai 库: pip install openai")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


async def call_llm(
    client: "AsyncOpenAI",
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[str]:
    """调用大模型, 带重试"""
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                print(f"  API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}, {wait:.1f}s 后重试...")
                await asyncio.sleep(wait)
            else:
                print(f"  API 调用失败 (已耗尽重试): {e}")
                return None


# ============================================================
# 语言检测
# ============================================================

LANG_DETECT_SYSTEM = """You are a language detection assistant. Your task is to identify the primary language of the given text.

Respond with ONLY the ISO 639-1 language code (two letters). Common codes:
- en: English
- zh: Chinese
- ja: Japanese
- ko: Korean
- fr: French
- de: German
- es: Spanish
- it: Italian
- pt: Portuguese
- ru: Russian
- ar: Arabic
- hi: Hindi
- th: Thai
- vi: Vietnamese
- nl: Dutch
- pl: Polish
- tr: Turkish
- cs: Czech
- he: Hebrew
- id: Indonesian

If the text contains multiple languages, identify the dominant one.
Respond with ONLY the two-letter code, nothing else."""


async def detect_language_for_item(
    client: "AsyncOpenAI",
    model: str,
    item: Dict,
    semaphore: asyncio.Semaphore,
) -> str:
    """检测单条数据的语言"""
    texts = extract_texts(item)
    if not texts:
        return "unknown"

    # 将所有文本拼接后检测 (取主要语言)
    combined = "\n\n".join(texts.values())
    # 截断过长文本, 避免超出 token 限制
    if len(combined) > 2000:
        combined = combined[:2000]

    async with semaphore:
        result = await call_llm(
            client, model,
            LANG_DETECT_SYSTEM,
            f"Detect the language of this text:\n\n{combined}",
        )

    if result:
        # 提取两字母语言代码
        match = re.match(r'^[a-z]{2}(-[A-Za-z]+)?$', result.strip().lower())
        if match:
            return match.group()
    return "unknown"


async def detect_languages(
    data: List[Dict],
    client: "AsyncOpenAI",
    model: str,
    concurrency: int = 10,
) -> List[Dict]:
    """批量检测数据集语言, 在每条数据中添加 "detected_lang" 字段"""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    for i, item in enumerate(data):
        # 跳过已检测的
        if item.get("detected_lang"):
            tasks.append(asyncio.coroutine(lambda item=item: item)())
            continue
        tasks.append(detect_language_for_item(client, model, item, semaphore))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    lang_stats = Counter()
    for i, (item, result) in enumerate(zip(data, results)):
        if isinstance(result, Exception):
            print(f"  条目 {i} 检测失败: {result}")
            item["detected_lang"] = "unknown"
        elif isinstance(result, Dict):
            # 已经有 detected_lang 的条目
            pass
        else:
            item["detected_lang"] = result
        lang_stats[item.get("detected_lang", "unknown")] += 1
        output.append(item)

    print(f"\n语言检测统计:")
    for lang, count in lang_stats.most_common():
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        print(f"  {lang} ({lang_name}): {count}")

    return output


# ============================================================
# 翻译
# ============================================================

TRANSLATE_SYSTEM = """You are a professional translator. Translate the given text to {target_language}.

IMPORTANT RULES:
1. Translate naturally and fluently, preserving the original meaning and tone.
2. Do NOT add any explanations, notes, or commentary - output ONLY the translation.
3. Preserve the structure: if the input has paragraphs, lists, or formatting, maintain them.
4. For technical terms, use the standard {target_language} terminology.
5. If the text is already in {target_language}, return it unchanged.
6. Do NOT translate proper nouns, URLs, code snippets, or mathematical expressions."""


async def translate_text(
    client: "AsyncOpenAI",
    model: str,
    text: str,
    target_lang: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """翻译单段文本"""
    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    system = TRANSLATE_SYSTEM.format(target_language=target_name)

    async with semaphore:
        result = await call_llm(client, model, system, text)

    return result


async def translate_item(
    client: "AsyncOpenAI",
    model: str,
    item: Dict,
    target_lang: str,
    semaphore: asyncio.Semaphore,
    skip_same_lang: bool = True,
) -> Optional[Dict]:
    """翻译单条数据到目标语言, 返回翻译后的新条目 (或 None 表示跳过)"""
    source_lang = item.get("detected_lang", "unknown")

    # 如果源语言与目标语言相同, 跳过
    if skip_same_lang and source_lang == target_lang:
        return None

    texts = extract_texts(item)
    if not texts:
        return None

    # 逐字段翻译
    translated_texts = {}
    for field, text in texts.items():
        # 截断过长文本
        truncated = text[:4000] if len(text) > 4000 else text
        result = await translate_text(client, model, truncated, target_lang, semaphore)
        if result:
            translated_texts[field] = result
        else:
            # 任一字段翻译失败则整条跳过
            return None

    # 构建新条目
    new_item = update_item_texts(item, translated_texts)
    # 更新元信息
    new_item["id"] = f"{item.get('id', 'unknown')}_{target_lang}"
    new_item["detected_lang"] = target_lang
    new_item["translated_from"] = source_lang
    new_item["translation_target"] = target_lang

    return new_item


async def translate_dataset(
    data: List[Dict],
    target_languages: List[str],
    client: "AsyncOpenAI",
    model: str,
    concurrency: int = 5,
    skip_same_lang: bool = True,
) -> List[Dict]:
    """将数据集翻译为多种目标语言, 返回原始数据 + 所有翻译数据"""
    all_expanded = list(data)  # 保留原始数据
    translation_stats = defaultdict(int)

    for target_lang in target_languages:
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        print(f"\n翻译到 {target_lang} ({target_name})...")

        semaphore = asyncio.Semaphore(concurrency)
        tasks = []
        valid_items = []

        for item in data:
            source_lang = item.get("detected_lang", "unknown")
            if skip_same_lang and source_lang == target_lang:
                translation_stats[f"{target_lang}_skipped_same"] += 1
                continue
            valid_items.append(item)
            tasks.append(translate_item(client, model, item, target_lang, semaphore, skip_same_lang))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        fail_count = 0
        for item, result in zip(valid_items, results):
            if isinstance(result, Exception):
                fail_count += 1
                if fail_count <= 3:
                    print(f"  翻译失败: {result}")
            elif result is not None:
                all_expanded.append(result)
                success_count += 1
            else:
                # 跳过 (源语言与目标语言相同, 或翻译返回 None)
                pass

        translation_stats[f"{target_lang}_success"] = success_count
        translation_stats[f"{target_lang}_fail"] = fail_count
        print(f"  {target_lang}: 成功 {success_count}, 失败 {fail_count}")

    print(f"\n翻译统计汇总:")
    for k, v in sorted(translation_stats.items()):
        print(f"  {k}: {v}")

    return all_expanded


# ============================================================
# 增量处理: 检查点保存与加载
# ============================================================

def save_checkpoint(data: List[Dict], path: str):
    """保存检查点 (与 save_jsonl 相同, 但打印不同信息)"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"检查点已保存: {path} ({len(data)} 条)")


# ============================================================
# 统计与报告
# ============================================================

def print_expand_stats(data: List[Dict], title: str = "扩展数据集统计"):
    """打印扩展数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"总样本数: {len(data)}")

    # 语言分布
    lang_counts = Counter(d.get("detected_lang", "unknown") for d in data)
    print(f"\n语言分布 (共 {len(lang_counts)} 种):")
    for lang, count in lang_counts.most_common():
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        print(f"  {lang} ({lang_name}): {count} ({count/len(data)*100:.1f}%)")

    # 翻译来源分布
    from_counts = Counter(d.get("translated_from", "original") for d in data)
    print(f"\n翻译来源分布:")
    for src, count in from_counts.most_common():
        print(f"  {src}: {count} ({count/len(data)*100:.1f}%)")

    # label 分布
    label_counts = Counter(d.get("label", "unknown") for d in data)
    print(f"\nlabel 分布 (共 {len(label_counts)} 个):")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")
    if len(label_counts) > 10:
        print(f"  ... 还有 {len(label_counts)-10} 个类别")

    # stage 分布
    stage_counts = Counter(d.get("stage", "unknown") for d in data)
    print(f"\nstage 分布:")
    for s, count in stage_counts.most_common():
        print(f"  {s}: {count} ({count/len(data)*100:.1f}%)")

    # 原始 vs 翻译
    original_count = sum(1 for d in data if "translated_from" not in d)
    translated_count = len(data) - original_count
    print(f"\n原始数据: {original_count} ({original_count/len(data)*100:.1f}%)")
    print(f"翻译数据: {translated_count} ({translated_count/len(data)*100:.1f}%)")


# ============================================================
# 子命令: detect-language
# ============================================================

async def cmd_detect_language(args):
    """检测数据集中每条数据的语言"""
    data = load_jsonl(args.input)

    # 统计已有语言标注
    already_detected = sum(1 for d in data if d.get("detected_lang"))
    if already_detected:
        print(f"已有 {already_detected}/{len(data)} 条数据标注了语言, 将跳过这些")

    client = get_client(args.base_url, args.api_key)
    data = await detect_languages(data, client, args.model, concurrency=args.concurrency)

    save_jsonl(data, args.output)
    print_expand_stats(data, "语言检测结果统计")


# ============================================================
# 子命令: translate
# ============================================================

async def cmd_translate(args):
    """将数据集翻译为多种目标语言"""
    data = load_jsonl(args.input)

    # 检查是否有语言标注
    has_lang = sum(1 for d in data if d.get("detected_lang"))
    if has_lang == 0:
        print("警告: 数据中没有 detected_lang 字段, 将无法跳过同语言数据")
        print("建议先运行 detect-language 命令")
    else:
        print(f"已有 {has_lang}/{len(data)} 条数据标注了语言")

    client = get_client(args.base_url, args.api_key)
    expanded = await translate_dataset(
        data, args.target_languages,
        client, args.model,
        concurrency=args.concurrency,
        skip_same_lang=not args.no_skip_same,
    )

    save_jsonl(expanded, args.output)
    print_expand_stats(expanded, "翻译扩展结果统计")


# ============================================================
# 子命令: expand (一键完成)
# ============================================================

async def cmd_expand(args):
    """一键完成: 检测语言 + 翻译扩展"""
    data = load_jsonl(args.input)

    client = get_client(args.base_url, args.api_key)

    # 步骤 1: 检测语言
    print("\n" + "="*60)
    print("  步骤 1/2: 检测语言")
    print("="*60)
    data = await detect_languages(data, client, args.model, concurrency=args.concurrency)

    # 保存中间结果 (检查点)
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        save_checkpoint(data, checkpoint_path)

    # 步骤 2: 翻译扩展
    print("\n" + "="*60)
    print("  步骤 2/2: 翻译扩展")
    print("="*60)
    expanded = await translate_dataset(
        data, args.target_languages,
        client, args.model,
        concurrency=args.concurrency,
        skip_same_lang=not args.no_skip_same,
    )

    save_jsonl(expanded, args.output)
    print_expand_stats(expanded, "扩展数据集最终统计")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="扩展数据集: 语言检测 + 多语言翻译",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 公共参数
    def add_common_args(p):
        p.add_argument("--input", type=str, required=True,
                        help="输入 JSONL 文件路径")
        p.add_argument("--output", type=str, required=True,
                        help="输出 JSONL 文件路径")
        p.add_argument("--base-url", type=str,
                        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        help="OpenAI 兼容 API base URL (默认: OPENAI_BASE_URL 环境变量或 https://api.openai.com/v1)")
        p.add_argument("--api-key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""),
                        help="API Key (默认: OPENAI_API_KEY 环境变量)")
        p.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="模型名称 (默认: gpt-4o-mini)")
        p.add_argument("--concurrency", type=int, default=10,
                        help="并发请求数 (默认: 10)")

    # detect-language 子命令
    p_detect = subparsers.add_parser("detect-language",
                                      help="检测数据集中每条数据的语言")
    add_common_args(p_detect)

    # translate 子命令
    p_translate = subparsers.add_parser("translate",
                                         help="将数据集翻译为多种目标语言")
    add_common_args(p_translate)
    p_translate.add_argument("--target-languages", type=str, nargs="+",
                              required=True,
                              help="目标语言代码列表, 如: zh ja ko fr de")
    p_translate.add_argument("--no-skip-same", action="store_true",
                              help="不跳过源语言与目标语言相同的数据")

    # expand 子命令
    p_expand = subparsers.add_parser("expand",
                                      help="一键完成: 检测语言 + 翻译扩展")
    add_common_args(p_expand)
    p_expand.add_argument("--target-languages", type=str, nargs="+",
                           required=True,
                           help="目标语言代码列表, 如: zh ja ko fr de es ar ru")
    p_expand.add_argument("--no-skip-same", action="store_true",
                           help="不跳过源语言与目标语言相同的数据")
    p_expand.add_argument("--checkpoint", type=str, default=None,
                           help="中间检查点保存路径 (语言检测后保存, 可选)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if not args.api_key:
        print("错误: 未提供 API Key. 请通过 --api-key 参数或 OPENAI_API_KEY 环境变量提供")
        return

    # 运行对应子命令
    if args.command == "detect-language":
        asyncio.run(cmd_detect_language(args))
    elif args.command == "translate":
        asyncio.run(cmd_translate(args))
    elif args.command == "expand":
        asyncio.run(cmd_expand(args))


if __name__ == "__main__":
    main()
