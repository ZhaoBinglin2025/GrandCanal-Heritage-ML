#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CanalEcho Phase 3: BERTopic无监督主题涌现 (v3.1 - 修复字段名)
=====================================================================
修复: 统一使用"类型"字段名，避免KeyError

确认可用的中文字体:
  - simhei.ttf (黑体, 9.7MB)
  - simsun.ttc (宋体, 18MB)
  - simkai.ttf (楷体, 11.7MB)
  - simfang.ttf (仿宋, 10.5MB)
  - msyh.ttc (微软雅黑, 19.7MB)
  - NotoSansSC-VF.ttf (17.7MB)

输入: output/balanced_text_aggressive.json
输出: topic_report.md, topic_visualization.png, topic_assignment.csv
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

# ========================== 中文字体强制设置 ==========================
FONT_PATHS = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\simfang.ttf",
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
]

font_prop = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            font_prop = font_manager.FontProperties(fname=fp)
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', fontproperties=font_prop, fontsize=12, ha='center')
            plt.close(fig)
            print(f"✓ 中文字体已加载: {fp}")
            break
        except Exception as e:
            print(f"  ✗ {fp} 加载失败: {e}")
            continue

if font_prop is None:
    print("\n✗ 错误: 所有中文字体加载失败!")
    sys.exit(1)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ========================== 用户可修改配置 ==========================
INPUT_FILE = r"D:\PHD\find\Projects\CanalEcho\output\balanced_text_aggressive.json"
OUTPUT_DIR = r"D:\PHD\find\Projects\CanalEcho\output"
REPORT_FILE = os.path.join(OUTPUT_DIR, "topic_report.md")
VIZ_FILE = os.path.join(OUTPUT_DIR, "topic_visualization.png")
ASSIGNMENT_FILE = os.path.join(OUTPUT_DIR, "topic_assignment.csv")

MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 5
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 4
# ===================================================================


def load_balanced_texts(filepath: str) -> tuple:
    """加载数据，过滤Low质量，返回(texts, metadata)"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    metadata = []
    low_count = 0

    for item in data:
        if item.get("text_quality") == "Low":
            low_count += 1
            continue

        bt = item.get("balanced_text", "").strip()
        if not bt:
            low_count += 1
            continue

        texts.append(bt)
        # 统一字段名，确保"类型"字段存在
        meta_item = {
            "序号": item.get("序号", ""),
            "遗产点名称": item.get("遗产点名称", ""),
            "省份": item.get("省份", ""),
            "经度": item.get("经度1", ""),
            "纬度": item.get("纬度1", ""),
            "类型": item.get("类型", "未知类型"),  # 提供默认值避免空值
        }
        metadata.append(meta_item)

    print(f"  加载: {len(data)} 条")
    print(f"  过滤 Low/空文本: {low_count} 条")
    print(f"  有效输入: {len(texts)} 条")
    return texts, metadata, data


def run_bertopic(texts: list, embeddings: np.ndarray) -> tuple:
    """运行BERTopic，不限制类别数"""
    print("\n[2/4] 配置BERTopic...")
    print(f"  UMAP: n_neighbors={UMAP_NEIGHBORS}, n_components={UMAP_COMPONENTS}")
    print(f"  HDBSCAN: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}")
    print(f"  策略: 不限制类别数，允许离群点(-1)")

    umap_model = UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True
    )

    print("\n[3/4] 运行主题涌现（可能需要1-3分钟）...")
    topics, probs = topic_model.fit_transform(texts, embeddings)

    return topic_model, topics, probs


def generate_visualization(embeddings: np.ndarray, topics: list, topic_model: BERTopic, 
                           output_path: str, font_prop):
    """生成UMAP可视化散点图 - 强制使用指定中文字体"""
    print("  生成可视化图表...")

    umap_viz = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
    embeddings_2d = umap_viz.fit_transform(embeddings)

    unique_topics = sorted(set(topics))
    n_colors = len(unique_topics)
    cmap = plt.cm.get_cmap('tab20', max(n_colors, 20))

    fig, ax = plt.subplots(figsize=(16, 12))

    for i, tid in enumerate(unique_topics):
        mask = np.array(topics) == tid
        color = '#999999' if tid == -1 else cmap(i % 20)
        label = f"离群点(-1)" if tid == -1 else f"主题{tid}"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=label,
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_title("CanalEcho Phase 3 —— BERTopic主题涌现可视化\n(UMAP 2D投影，按主题着色)", 
                 fontsize=16, fontweight='bold', fontproperties=font_prop)
    ax.set_xlabel("UMAP-1", fontsize=12, fontproperties=font_prop)
    ax.set_ylabel("UMAP-2", fontsize=12, fontproperties=font_prop)

    n_legend_cols = 3 if n_colors > 10 else 2
    legend = ax.legend(loc='best', fontsize=10, ncol=n_legend_cols, markerscale=1.5,
                       title="主题分布", title_fontsize=12, prop=font_prop)
    plt.setp(legend.get_title(), fontproperties=font_prop)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 可视化已保存: {output_path}")


def generate_report(topic_model: BERTopic, texts: list, metadata: list, topics: list) -> str:
    """生成Markdown主题涌现报告"""
    print("\n[4/4] 生成主题报告...")

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info.Topic != -1])
    n_outliers = topic_info[topic_info.Topic == -1]["Count"].values[0] if -1 in topic_info.Topic.values else 0

    lines = [
        "# CanalEcho Phase 3 —— BERTopic无监督主题涌现报告",
        "",
        f"**输入数据**: {len(texts)} 条均衡化遗产文本",
        f"**涌现主题数**: {n_topics} 个（不含离群点）",
        f"**离群点数量**: {n_outliers} 条（标签-1）",
        f"**HDBSCAN参数**: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}",
        "",
        "---",
        "",
        "## 一、主题概览",
        "",
        "| 主题ID | 数量 | 占比 | Top-10关键词 |",
        "|--------|------|------|--------------|",
    ]

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        count = row["Count"]
        pct = f"{count/len(texts)*100:.1f}%"

        if tid == -1:
            keywords = "[离群点，无主题词]"
        else:
            kw_list = topic_model.get_topic(tid)
            if kw_list:
                keywords = "; ".join([f"{w}({s:.3f})" for w, s in kw_list[:10]])
            else:
                keywords = "[无关键词]"

        lines.append(f"| {tid} | {count} | {pct} | {keywords} |")

    lines += [
        "",
        "---",
        "",
        "## 二、各主题代表性文本片段",
        "",
        "> 以下从每个主题中随机抽取3条代表性遗产的 balanced_text 前200字，",
        "> 供人工审阅语义可解释性。",
        "",
    ]

    # 修复: 统一使用"类型"作为键名
    topic_texts = defaultdict(list)
    for i, tid in enumerate(topics):
        topic_texts[tid].append({
            "idx": i,
            "text": texts[i][:200],
            "name": metadata[i]["遗产点名称"],
            "类型": metadata[i]["类型"]  # 统一使用"类型"
        })

    for tid in sorted(topic_texts.keys()):
        items = topic_texts[tid]
        lines.append(f"### 主题 {tid}（{len(items)}条）")
        lines.append("")

        sample = items[:3] if len(items) > 3 else items
        for s in sample:
            # 修复: 使用"类型"键
            lines.append(f"**{s['name']}**（{s['类型']}）")
            lines.append(f"> {s['text']}...")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines += [
        "## 三、人工命名指引",
        "",
        "请根据上述关键词和代表性文本，为每个主题赋予学术化命名。",
        "命名建议格式：",
        "",
        "```",
        "主题0 → [历史叙事型] —— 朝代更迭、漕运记忆、考古断代",
        "主题1 → [工程技艺型] —— 水利原理、砌筑技术、灌溉防洪",
        "主题2 → [景观审美型] —— 风景诗词、游览价值、四季变换",
        "...",
        "```",
        "",
        "命名完成后，请手动创建 `topic_names.json` 或直接告诉我命名结果，",
        "我将据此生成《标注规则手册》（annotation_rulebook.md）。",
        "",
        "---",
        "",
        "## 四、方法论说明",
        "",
        f"- 输入文本为Phase 2均衡化后的 `balanced_text`（已消除篇幅偏见）。",
        f"- HDBSCAN未预设类别数，实际涌现{n_topics}个簇 + {n_outliers}个离群点。",
        f"- 若某主题簇语义不可解释（如按地理区域或文本长度伪聚类），请告诉我，可调整UMAP/HDBSCAN参数重跑。",
        "",
    ]

    return "\n".join(lines)


def save_assignment_csv(metadata: list, topics: list, probs: np.ndarray, output_path: str):
    """保存主题分配表"""
    df_data = []
    for i, meta in enumerate(metadata):
        row = {
            "序号": meta["序号"],
            "遗产点名称": meta["遗产点名称"],
            "省份": meta["省份"],
            "经度": meta["经度"],
            "纬度": meta["纬度"],
            "类型": meta["类型"],
            "主题ID": topics[i],
            "主题概率": float(np.max(probs[i])) if probs is not None and len(probs) > i else None
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ 分配表已保存: {output_path}")


# ========================== 主函数 ==========================

def main():
    print("=" * 60)
    print("CanalEcho Phase 3: BERTopic无监督主题涌现 (v3.1)")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"\n错误: 找不到输入文件 '{INPUT_FILE}'")
        print("请先运行 01d_aggressive_clean.py 生成 balanced_text_aggressive.json")
        sys.exit(1)

    print("\n[1/4] 加载均衡化文本...")
    texts, metadata, raw_data = load_balanced_texts(INPUT_FILE)

    if len(texts) < 50:
        print(f"\n错误: 有效文本仅 {len(texts)} 条，不足以运行BERTopic。请检查数据质量。")
        sys.exit(1)

    print("\n加载 Sentence-BERT 编码器...")
    encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("  编码均衡化文本...")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    print(f"  ✓ 嵌入矩阵: {embeddings.shape}")

    topic_model, topics, probs = run_bertopic(texts, embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = topics.count(-1)
    print(f"\n  主题涌现完成!")
    print(f"  主题数量: {n_topics} 个")
    print(f"  离群点: {n_outliers} 条 ({n_outliers/len(texts)*100:.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = generate_report(topic_model, texts, metadata, topics)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  ✓ 报告已保存: {REPORT_FILE}")

    generate_visualization(embeddings, topics, topic_model, VIZ_FILE, font_prop)

    save_assignment_csv(metadata, topics, probs, ASSIGNMENT_FILE)

    model_path = os.path.join(OUTPUT_DIR, "bertopic_model")
    topic_model.save(model_path)
    print(f"  ✓ 模型已保存: {model_path}")

    print("\n" + "=" * 60)
    print("Phase 3 完成!")
    print(f"  请查看报告: {REPORT_FILE}")
    print(f"  请查看可视化: {VIZ_FILE}")
    print(f"  请查看分配表: {ASSIGNMENT_FILE}")
    print("\n下一步: 人工审阅主题簇 → 命名 → 生成《标注规则手册》")
    print("=" * 60)


if __name__ == "__main__":
    main()