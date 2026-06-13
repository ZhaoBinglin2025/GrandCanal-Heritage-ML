#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CanalEcho Phase 2: 文本均衡化脚本 (修复版 v2)
===============================================
修复: 原版本一次性拼接500条文本导致内存溢出(8.11GB)。
      新版本逐条提取关键词再聚合，内存占用<<100MB。

输入: enriched_heritage_full_allright.json (500条全量遗产数据)
输出: balanced_text.json (保留全部原始字段 + balanced_text + dimension_breakdown)

执行流程:
  Step 1: KeyBERT逐条提取关键词 → 聚合权重 → 人工交互命名维度
  Step 2: Sentence-BERT段落切分 + 维度归类
  Step 3: 等字数缩句 (<200字抛弃, 最少字数为基准, 规则式提取)

依赖安装:
  pip install keybert sentence-transformers jieba numpy pandas scikit-learn
"""

import json
import os
import re
import sys
from collections import defaultdict

import jieba
import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer

# ========================== 用户可修改配置 ==========================
INPUT_FILE = r"D:\PHD\find\Projects\CanalEcho\output\enriched_heritage_full_allright.json"   # 输入文件路径
OUTPUT_DIR = r"D:\PHD\find\Projects\CanalEcho\output"                                   # 输出目录
DIMENSION_CANDIDATES_FILE = os.path.join(OUTPUT_DIR, "dimension_candidates.md")
DIMENSION_DICT_FILE = os.path.join(OUTPUT_DIR, "dimension_dict.json")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "balanced_text.json")

MIN_CHAR_THRESHOLD = 200       # 维度抛弃阈值 (字)
# ===================================================================


def chinese_char_count(text: str) -> int:
    """统计中文字符数量（作为"字数"标准）"""
    if not text:
        return 0
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def split_sentences(text: str) -> list:
    """按中文标点分割文本为句子列表"""
    if not text:
        return []
    parts = re.split(r'([。！？；\n])', text)
    sentences = []
    i = 0
    while i < len(parts):
        sent = parts[i]
        if i + 1 < len(parts):
            sent += parts[i + 1]
            i += 2
        else:
            i += 1
        sent = sent.strip()
        if sent:
            sentences.append(sent)
    return sentences


def trim_text_to_target(text: str, target_chars: int) -> str:
    """
    规则式缩句：保留首尾句 + 均匀采样中间句，使中文字数接近 target_chars
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    if len(sentences) <= 2:
        result = ""
        for s in sentences:
            if chinese_char_count(result) + chinese_char_count(s) <= target_chars:
                result += s
            else:
                break
        return result

    first = sentences[0]
    last = sentences[-1]
    first_c = chinese_char_count(first)
    last_c = chinese_char_count(last)

    if first_c + last_c >= target_chars:
        if first_c >= target_chars:
            return first[:int(target_chars * 1.2)]
        return first

    middle_budget = target_chars - first_c - last_c
    middle = sentences[1:-1]
    if not middle:
        return first + last

    avg_len = sum(chinese_char_count(s) for s in middle) / max(len(middle), 1)
    needed_count = max(1, int(middle_budget / max(avg_len, 1)))
    needed_count = min(needed_count, len(middle))

    if needed_count >= len(middle):
        selected = middle
    else:
        indices = [int(i * (len(middle) - 1) / max(needed_count - 1, 1)) for i in range(needed_count)]
        selected = [middle[i] for i in indices]

    selected_text = "".join(selected)
    while chinese_char_count(selected_text) > middle_budget and len(selected) > 1:
        selected = selected[:-1]
        selected_text = "".join(selected)

    return first + selected_text + last


# ========================== Step 1: 全局维度识别 (修复版) ==========================

def step1_extract_keywords_batch(data: list, kw_model: KeyBERT) -> list:
    """
    修复版: 逐条提取关键词再聚合，避免一次性拼接全部文本导致内存溢出。
    原版本: 500条拼接后46661个候选词，MMR矩阵8.11GB。
    新版本: 每条提取Top-10，聚合权重取平均，内存<<100MB。
    """
    print("\n[Step 1/3] 全局维度识别 —— KeyBERT逐条提取关键词（内存优化版）...")
    print("  逐条处理500条遗产，每条约1-2秒...")

    def zh_tokenizer(t):
        return jieba.lcut(t)

    vectorizer = CountVectorizer(tokenizer=zh_tokenizer, token_pattern=None)

    # 逐条提取关键词
    all_keywords = defaultdict(float)
    keyword_counts = defaultdict(int)

    for idx, item in enumerate(data, 1):
        text = item.get("full_text", "")
        if not text or chinese_char_count(text) < 50:
            continue

        try:
            keywords = kw_model.extract_keywords(
                text,
                vectorizer=vectorizer,
                keyphrase_ngram_range=(1, 2),
                stop_words=None,
                top_n=10,
                use_mmr=True,
                diversity=0.5
            )
            for kw, score in keywords:
                all_keywords[kw] += score
                keyword_counts[kw] += 1
        except Exception as e:
            print(f"    警告: 第{idx}条提取失败: {e}")
            continue

        if idx % 100 == 0:
            print(f"    已处理 {idx}/{len(data)} 条...")

    # 聚合：按出现次数加权平均
    aggregated = []
    for kw, total_score in all_keywords.items():
        avg_score = total_score / keyword_counts[kw]
        aggregated.append((kw, avg_score, keyword_counts[kw]))

    # 按平均权重排序，取Top-60
    aggregated.sort(key=lambda x: -x[1])
    top_keywords = aggregated[:60]

    # 生成 Markdown 报告
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = [
        "# CanalEcho Phase 2 —— 全局维度候选关键词报告（逐条聚合版）",
        "",
        f"基于 **{len(data)}** 条遗产逐条提取并聚合的 Top-60 候选关键词：",
        "",
        "| 排名 | 关键词 | 平均权重 | 出现次数 |",
        "|------|--------|----------|----------|",
    ]
    for i, (kw, score, cnt) in enumerate(top_keywords, 1):
        report_lines.append(f"| {i} | {kw} | {score:.4f} | {cnt} |")

    report_lines += [
        "",
        "---",
        "",
        "## 请人工命名全局维度（5-8个）",
        "",
        "根据上方关键词，归纳出 **5-8 个全局语义维度**。",
        "维度应覆盖遗产文本的主要语义方向，例如：",
        "",
        "- 工程技艺（水利原理、砌筑、灌溉、防洪）",
        "- 历史叙事（朝代、漕运、战争、移民、考古）",
        "- 景观审美（风景、诗词、游览、四季、倒影）",
        "- 信仰仪式（祭祀、祈福、龙王庙、镇水兽、民间传说）",
        "- 日常生活（饮水、航运、洗衣、邻里交往）",
        "- 考古研究（发掘、文化层、出土遗物、断代）",
        "",
        "## 操作方式",
        "",
        "**方式A（推荐）：** 直接在本脚本运行时的命令行中输入维度名称（逗号分隔），",
        "脚本会自动保存为 `dimension_dict.json` 并继续执行均衡化。",
        "",
        "**方式B：** 关闭脚本，手动创建 `dimension_dict.json`，格式如下：",
        "```json",
        '{"dimensions": ["工程技艺", "历史叙事", "景观审美", "信仰仪式", "考古研究"]}',
        "```",
        "然后重新运行本脚本。",
        "",
    ]

    with open(DIMENSION_CANDIDATES_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n  ✓ 关键词报告已保存: {DIMENSION_CANDIDATES_FILE}")
    print(f"\n  Top-20 关键词预览:")
    for kw, score, cnt in top_keywords[:20]:
        print(f"    {score:.4f}  ({cnt}次)  →  {kw}")
    print("")

    return top_keywords


def interactive_dimension_naming() -> list:
    """交互式命名维度"""
    print("-" * 60)
    print("请根据关键词报告命名全局维度（5-8个）")
    print("示例输入: 工程技艺,历史叙事,景观审美,信仰仪式,考古研究")
    print("输入 'skip' 可退出，稍后手动创建 dimension_dict.json 再运行")
    print("-" * 60)
    user_input = input("维度名称（逗号分隔）> ").strip()

    if user_input.lower() == "skip":
        print("\n已跳过。请查看报告并手动创建 dimension_dict.json 后重新运行。")
        sys.exit(0)

    dimension_names = [d.strip() for d in user_input.split(",") if d.strip()]
    if len(dimension_names) < 3:
        print(f"\n错误: 仅输入了 {len(dimension_names)} 个维度，至少需要 3 个。请重新运行脚本。")
        sys.exit(1)

    dim_dict = {
        "dimensions": dimension_names,
        "count": len(dimension_names),
        "source": "user_defined"
    }
    with open(DIMENSION_DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(dim_dict, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ 维度词典已保存: {DIMENSION_DICT_FILE}")
    print(f"  ✓ 共 {len(dimension_names)} 个维度: {dimension_names}")
    return dimension_names


# ========================== Step 2: 段落切分与维度归类 ==========================

def step2_classify_paragraphs(data: list, dimension_names: list, encoder: SentenceTransformer) -> list:
    """将每条遗产的句子按语义相似度归入最近维度"""
    print(f"\n[Step 2/3] 段落切分与维度归类 —— 处理 {len(data)} 条遗产...")

    dim_embeddings = encoder.encode(dimension_names, convert_to_tensor=True)

    classified = []
    for idx, item in enumerate(data, 1):
        text = item.get("full_text", "")
        if not text:
            classified.append({
                "heritage_id": item.get("序号", f"unknown_{idx}"),
                "name": item.get("遗产点名称", "unknown"),
                "dimensions": {},
                "text_quality": "Low"
            })
            continue

        sentences = split_sentences(text)
        if not sentences:
            classified.append({
                "heritage_id": item.get("序号", f"unknown_{idx}"),
                "name": item.get("遗产点名称", "unknown"),
                "dimensions": {},
                "text_quality": "Low"
            })
            continue

        sent_embeddings = encoder.encode(sentences, convert_to_tensor=True)
        similarities = util.cos_sim(sent_embeddings, dim_embeddings)
        sim_np = similarities.cpu().numpy()

        dim_texts = defaultdict(list)
        for i, sent in enumerate(sentences):
            best_dim_idx = int(np.argmax(sim_np[i]))
            dim_name = dimension_names[best_dim_idx]
            dim_texts[dim_name].append(sent)

        dimensions = {}
        for dim_name, sents in dim_texts.items():
            dim_full = "".join(sents)
            dimensions[dim_name] = {
                "raw_text": dim_full,
                "char_count": chinese_char_count(dim_full),
                "sentence_count": len(sents)
            }

        classified.append({
            "heritage_id": item.get("序号", f"unknown_{idx}"),
            "name": item.get("遗产点名称", "unknown"),
            "dimensions": dimensions,
            "text_quality": "OK"
        })

        if idx % 100 == 0:
            print(f"    已处理 {idx}/{len(data)} 条...")

    dim_coverage = defaultdict(int)
    for c in classified:
        for d in c["dimensions"]:
            dim_coverage[d] += 1
    print(f"\n  维度覆盖统计:")
    for d, cnt in sorted(dim_coverage.items(), key=lambda x: -x[1]):
        print(f"    {d}: {cnt} 条遗产包含")

    return classified


# ========================== Step 3: 等字数缩句 ==========================

def step3_balance_texts(classified_data: list, min_threshold: int = MIN_CHAR_THRESHOLD) -> list:
    """等字数缩句：抛弃<<阈值维度，以最少字数为基准，规则式缩句"""
    print(f"\n[Step 3/3] 等字数缩句 —— 阈值={min_threshold}字...")

    balanced_results = []
    stats = {"ok": 0, "low": 0}

    for item in classified_data:
        dims = item["dimensions"]

        valid_dims = {}
        dropped = []
        for dim_name, dim_info in dims.items():
            if dim_info["char_count"] >= min_threshold:
                valid_dims[dim_name] = dim_info
            else:
                dropped.append(f"{dim_name}({dim_info['char_count']}字)")

        if not valid_dims:
            balanced_results.append({
                **item,
                "balanced_text": "",
                "dimension_breakdown": {},
                "text_quality": "Low",
                "dropped_dimensions": dropped
            })
            stats["low"] += 1
            continue

        base_count = min(d["char_count"] for d in valid_dims.values())

        balanced_parts = []
        breakdown = {}

        for dim_name, dim_info in valid_dims.items():
            raw_text = dim_info["raw_text"]
            raw_count = dim_info["char_count"]

            if raw_count == base_count:
                final_text = raw_text
                was_trimmed = False
            else:
                final_text = trim_text_to_target(raw_text, base_count)
                was_trimmed = True

            balanced_parts.append(final_text)
            breakdown[dim_name] = {
                "original_chars": raw_count,
                "final_chars": chinese_char_count(final_text),
                "was_trimmed": was_trimmed,
                "base_count": base_count
            }

        balanced_text = "".join(balanced_parts)

        balanced_results.append({
            **item,
            "balanced_text": balanced_text,
            "dimension_breakdown": breakdown,
            "text_quality": "OK",
            "dropped_dimensions": dropped
        })
        stats["ok"] += 1

    print(f"\n  缩句完成: {stats['ok']} 条 OK, {stats['low']} 条 Low")
    print(f"  Low原因: 全部维度缺失或字数 < {min_threshold}")
    return balanced_results


# ========================== 主函数 ==========================

def main():
    print("=" * 60)
    print("CanalEcho Phase 2: 文本均衡化 (修复版 v2)")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"\n错误: 找不到输入文件 '{INPUT_FILE}'")
        print("请将 enriched_heritage_full_allright.json 放置到脚本同级目录，")
        print("或修改脚本顶部的 INPUT_FILE 变量为正确路径。")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n加载数据: {len(data)} 条遗产")

    print("\n加载 Sentence-BERT 模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
    print("  首次运行需下载约 400MB 模型文件，请耐心等待...")
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    kw_model = KeyBERT(model=encoder)
    print("  ✓ 模型加载完成")

    # ---------- Step 1 ----------
    if not os.path.exists(DIMENSION_DICT_FILE):
        step1_extract_keywords_batch(data, kw_model)
        dimension_names = interactive_dimension_naming()
    else:
        with open(DIMENSION_DICT_FILE, "r", encoding="utf-8") as f:
            dim_dict = json.load(f)
        dimension_names = dim_dict["dimensions"]
        print(f"\n[Step 1/3] 读取已有维度词典: {dimension_names}")

    # ---------- Step 2 ----------
    classified = step2_classify_paragraphs(data, dimension_names, encoder)

    # ---------- Step 3 ----------
    balanced = step3_balance_texts(classified, min_threshold=MIN_CHAR_THRESHOLD)

    # ---------- 输出 ----------
    print("\n" + "=" * 60)
    print("合并原始字段并输出...")

    final_output = []
    for i, item in enumerate(data):
        if i < len(balanced):
            bal = balanced[i]
            final_item = {
                **item,
                "balanced_text": bal.get("balanced_text", ""),
                "dimension_breakdown": bal.get("dimension_breakdown", {}),
                "text_quality": bal.get("text_quality", "Unknown"),
                "dropped_dimensions": bal.get("dropped_dimensions", [])
            }
        else:
            final_item = item
        final_output.append(final_item)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ 输出完成: {OUTPUT_FILE}")
    print(f"  ✓ 总记录数: {len(final_output)}")
    print(f"  ✓ 新增字段: balanced_text, dimension_breakdown, text_quality, dropped_dimensions")
    print(f"\n下一步: 运行 02b_topic_discovery.py (BERTopic主题涌现)")
    print("=" * 60)


if __name__ == "__main__":
    main()