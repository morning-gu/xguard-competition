"""扩展数据集: 使用大模型检测语言并翻译为多种语言

功能:
1. 语言检测: 调用 OpenAI 兼容接口检测数据集中每条数据的语言 (仅基于 prompt 字段)
2. 多语言翻译: 将数据翻译为配置的多种目标语言, 扩展数据集
3. 支持增量处理: 跳过已检测/已翻译的数据, 支持断点续传
4. 支持并发: 使用异步并发加速 API 调用
5. 实时写入: 每完成一条原始数据的翻译就追加写入文件, 翻译失败也写入原始数据
6. 多模型随机选择: 支持指定多个模型名称, 每次调用随机选择一个模型

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

  # 使用多个模型 (随机选择)
  python scripts/expand_dataset.py expand \\
    --input data/xguard_train_curated_1k.jsonl \\
    --output data/xguard_train_curated_1k_expanded.jsonl \\
    --model gpt-4o-mini qwen2.5-72b-instruct deepseek-chat \\
    --target-languages zh ja ko fr de es ar ru
"""

import json
import os
import re
import random
import argparse
import asyncio
from typing import List, Dict, Optional
from collections import Counter, defaultdict

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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


def append_jsonl(items: List[Dict], path: str):
    """追加写入 JSONL 文件 (逐条追加, 用于实时保存翻译结果)"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# 文本提取工具
# ============================================================

def extract_texts(item: Dict) -> Dict[str, str]:
    """从数据条目中提取需要翻译的文本字段

    Returns:
        dict: {"prompt": ..., "response": ..., "policy": ..., "explanation": ...}
              仅包含非空字段
    """
    texts = {}
    for field in ("prompt", "response", "policy", "explanation"):
        val = (item.get(field) or "").strip()
        if val:
            texts[field] = val
    return texts


def update_item_texts(item: Dict, translated_texts: Dict[str, str]) -> Dict:
    """用翻译后的文本更新数据条目 (不修改原条目, 返回新条目)

    保持与原始数据格式一致: 仅替换 prompt/response/policy/explanation 文本,
    其他字段 (id, sample_type, stage, label 等) 原样保留.
    """
    new_item = dict(item)
    for field, text in translated_texts.items():
        if field in ("prompt", "response", "policy", "explanation"):
            new_item[field] = text
    return new_item


# ============================================================
# 模型选择: 支持多模型随机选择
# ============================================================

def pick_model(models: List[str]) -> str:
    """从模型列表中随机选择一个模型

    Args:
        models: 模型名称列表, 仅一个时直接返回

    Returns:
        随机选中的模型名称
    """
    if len(models) == 1:
        return models[0]
    return random.choice(models)


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
    max_tokens: int = 4096,
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
                max_tokens=max_tokens,
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
# 带进度条的异步 gather
# ============================================================

async def _gather_with_progress(
    coros: list,
    desc: str = "处理中",
    unit: str = "条",
) -> list:
    """带进度条的 asyncio.gather, 保持结果顺序, 实时显示进度"""
    total = len(coros)
    if total == 0:
        return []

    pbar = tqdm(total=total, desc=desc, unit=unit, smoothing=0.1) if tqdm else None
    results = [None] * total

    async def _wrapped(i, coro):
        try:
            result = await coro
            results[i] = result
        except Exception as e:
            results[i] = e
        finally:
            if pbar:
                pbar.update(1)

    await asyncio.gather(*(_wrapped(i, c) for i, c in enumerate(coros)))

    if pbar:
        pbar.close()

    return results


# ============================================================
# 语言检测 (仅基于 prompt 字段)
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
    models: List[str],
    item: Dict,
    semaphore: asyncio.Semaphore,
) -> str:
    """检测单条数据的语言 (基于 prompt 字段, prompt 为空时使用 response 字段)"""
    text = (item.get("prompt") or "").strip()
    if not text:
        text = (item.get("response") or "").strip()
    if not text:
        return "unknown"

    # 截断过长文本, 避免超出 token 限制
    text = text[:2000]

    model = pick_model(models)
    async with semaphore:
        result = await call_llm(
            client, model,
            LANG_DETECT_SYSTEM,
            f"Detect the language of this text:\n\n{text}",
            max_tokens=5,
        )

    if result:
        match = re.match(r'^[a-z]{2}(-[A-Za-z]+)?$', result.strip().lower())
        if match:
            return match.group()
    return "unknown"


async def detect_languages(
    data: List[Dict],
    client: "AsyncOpenAI",
    models: List[str],
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
        tasks.append(detect_language_for_item(client, models, item, semaphore))

    results = await _gather_with_progress(tasks, desc="语言检测")

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
    models: List[str],
    text: str,
    target_lang: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """翻译单段文本"""
    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    system = TRANSLATE_SYSTEM.format(target_language=target_name)

    model = pick_model(models)
    async with semaphore:
        result = await call_llm(client, model, system, text)

    return result


async def translate_item(
    client: "AsyncOpenAI",
    models: List[str],
    item: Dict,
    target_lang: str,
    semaphore: asyncio.Semaphore,
    skip_same_lang: bool = True,
) -> Optional[Dict]:
    """翻译单条数据到目标语言, 返回翻译后的新条目 (或 None 表示跳过)

    翻译结果保持与原始数据格式一致, 仅替换文本字段 (prompt/response/explanation),
    不添加额外元信息字段.
    """
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
        result = await translate_text(client, models, truncated, target_lang, semaphore)
        if result:
            translated_texts[field] = result
        else:
            # 任一字段翻译失败则整条跳过
            return None

    # 构建新条目 (保持与原始数据格式一致, 仅替换翻译后的文本字段)
    new_item = update_item_texts(item, translated_texts)

    return new_item


async def translate_item_all_langs(
    client: "AsyncOpenAI",
    models: List[str],
    item: Dict,
    target_languages: List[str],
    semaphore: asyncio.Semaphore,
    skip_same_lang: bool = True,
) -> List[Dict]:
    """翻译单条原始数据到所有目标语言

    Returns:
        list: 翻译成功的结果列表 (不含原始数据本身, 不含跳过的)
    """
    results = []
    for target_lang in target_languages:
        translated = await translate_item(
            client, models, item, target_lang, semaphore, skip_same_lang
        )
        if translated is not None:
            results.append(translated)
    return results


async def translate_dataset(
    data: List[Dict],
    target_languages: List[str],
    client: "AsyncOpenAI",
    models: List[str],
    concurrency: int = 5,
    skip_same_lang: bool = True,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """将数据集翻译为多种目标语言, 返回所有翻译数据 (不含原始数据)

    以原始数据为粒度并发: 每条原始数据并发翻译到所有目标语言,
    完成后立即将翻译结果追加写入文件 (如果提供 output_path).

    Args:
        output_path: 输出文件路径, 提供时每完成一条原始数据就追加写入
    """
    semaphore = asyncio.Semaphore(concurrency)
    translation_stats = defaultdict(int)

    # 预计算跳过统计
    for target_lang in target_languages:
        for item in data:
            source_lang = item.get("detected_lang", "unknown")
            if skip_same_lang and source_lang == target_lang:
                translation_stats[f"{target_lang}_skipped_same"] += 1

    # 如果提供了 output_path, 先清空文件
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            pass

    # 以原始数据为粒度并发
    async def _process_one_item(item):
        """处理一条原始数据: 翻译到所有目标语言, 返回 (原始数据, 翻译结果列表)"""
        translated = await translate_item_all_langs(
            client, models, item, target_languages, semaphore, skip_same_lang
        )
        return item, translated

    pbar = tqdm(total=len(data), desc="翻译扩展", unit="条", smoothing=0.1) if tqdm else None

    all_expanded = []
    success_items = 0
    fail_items = 0

    # 使用 as_completed 实现逐条完成逐条写入
    tasks = [_process_one_item(item) for item in data]
    for coro in asyncio.as_completed(tasks):
        try:
            original, translated_list = await coro
        except Exception as e:
            fail_items += 1
            if fail_items <= 3:
                print(f"  翻译失败: {e}")
            if pbar:
                pbar.update(1)
            continue

        # 仅写入翻译结果, 不包含原始数据
        batch = translated_list
        if translated_list:
            success_items += 1
        else:
            fail_items += 1

        all_expanded.extend(batch)

        # 实时追加写入文件
        if output_path:
            append_jsonl(batch, output_path)

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    print(f"\n翻译统计: 成功 {success_items}, 失败 {fail_items}")
    for k, v in sorted(translation_stats.items()):
        print(f"  {k}: {v}")

    return all_expanded


# ============================================================
# 增量处理: 检查点保存与加载
# ============================================================

def save_checkpoint(data: List[Dict], path: str):
    """保存检查点"""
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


# ============================================================
# 子命令: detect-language
# ============================================================

async def cmd_detect_language(args):
    """检测数据集中每条数据的语言"""
    data = load_jsonl(args.input)

    already_detected = sum(1 for d in data if d.get("detected_lang"))
    if already_detected:
        print(f"已有 {already_detected}/{len(data)} 条数据标注了语言, 将跳过这些")

    client = get_client(args.base_url, args.api_key)
    models = args.model
    print(f"使用模型: {models} (共 {len(models)} 个, 随机选择)")
    data = await detect_languages(data, client, models, concurrency=args.concurrency)

    save_jsonl(data, args.output)
    print_expand_stats(data, "语言检测结果统计")


# ============================================================
# 子命令: translate
# ============================================================

async def cmd_translate(args):
    """将数据集翻译为多种目标语言"""
    data = load_jsonl(args.input)

    has_lang = sum(1 for d in data if d.get("detected_lang"))
    if has_lang == 0:
        print("警告: 数据中没有 detected_lang 字段, 将无法跳过同语言数据")
        print("建议先运行 detect-language 命令")
    else:
        print(f"已有 {has_lang}/{len(data)} 条数据标注了语言")

    client = get_client(args.base_url, args.api_key)
    models = args.model
    print(f"使用模型: {models} (共 {len(models)} 个, 随机选择)")
    expanded = await translate_dataset(
        data, args.target_languages,
        client, models,
        concurrency=args.concurrency,
        skip_same_lang=not args.no_skip_same,
        output_path=args.output,
    )

    print_expand_stats(expanded, "翻译扩展结果统计")


# ============================================================
# 子命令: expand (一键完成)
# ============================================================

async def cmd_expand(args):
    """一键完成: 检测语言 + 翻译扩展"""
    data = load_jsonl(args.input)

    client = get_client(args.base_url, args.api_key)
    models = args.model
    print(f"使用模型: {models} (共 {len(models)} 个, 随机选择)")

    # 步骤 1: 检测语言
    print("\n" + "="*60)
    print("  步骤 1/2: 检测语言")
    print("="*60)
    data = await detect_languages(data, client, models, concurrency=args.concurrency)

    if args.checkpoint:
        save_checkpoint(data, args.checkpoint)

    # 步骤 2: 翻译扩展
    print("\n" + "="*60)
    print("  步骤 2/2: 翻译扩展")
    print("="*60)
    expanded = await translate_dataset(
        data, args.target_languages,
        client, models,
        concurrency=args.concurrency,
        skip_same_lang=not args.no_skip_same,
        output_path=args.output,
    )

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

    def add_common_args(p):
        p.add_argument("--input", type=str, required=True,
                        help="输入 JSONL 文件路径")
        p.add_argument("--output", type=str, required=True,
                        help="输出 JSONL 文件路径")
        p.add_argument("--base-url", type=str,
                        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        help="OpenAI 兼容 API base URL")
        p.add_argument("--api-key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""),
                        help="API Key")
        p.add_argument("--model", type=str, nargs="+", default=["gpt-4o-mini"],
                        help="模型名称, 支持多个 (空格分隔), 每次调用随机选择 (默认: gpt-4o-mini)")
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

    if args.command == "detect-language":
        asyncio.run(cmd_detect_language(args))
    elif args.command == "translate":
        asyncio.run(cmd_translate(args))
    elif args.command == "expand":
        asyncio.run(cmd_expand(args))


if __name__ == "__main__":
    main()
