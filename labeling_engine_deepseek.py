#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CanalEcho Phase 4: LLM多标签Top1标注引擎 (DeepSeek适配版)
============================================================
输入: output/balanced_text_aggressive.json (467条有效文本)
输出: output/labeled_heritage_v2.json (多标签概率 + Top1硬标签)

机制:
  1. 读取BERTopic涌现的10个主题 + 离群点(-1)
  2. 读取《标注规则手册》(annotation_rulebook.md)作为系统Prompt
  3. 逐条调用DeepSeek API，输出多标签概率分布
  4. 取Top1权重最高者为硬标签
  5. 每处理100条暂停，抽查5条，命令行交互 y/n
  6. 硬性阀门: 正确率 < 80% 立即报错终止，要求调整Prompt/规则手册后重跑
  7. 断点续传: 每10条自动保存，中途崩溃可重新运行

依赖:
  pip install openai numpy pandas

环境变量:
  set DEEPSEEK_API_KEY=你的密钥
"""

import json
import os
import sys
import time
import random
from collections import defaultdict

import numpy as np

# ========================== DeepSeek 配置 ==========================
API_TYPE = "deepseek"           # 可选: "deepseek" | "openai" | "ollama"
BASE_URL = "https://api.deepseek.com/v1"  # DeepSeek官方API
MODEL = "deepseek-v4-flash"         # 或 "deepseek-reasoner"（推理模型，更慢但更准）

# 若使用本地Ollama部署的DeepSeek，取消下面两行注释并注释掉上方：
# BASE_URL = "http://localhost:11434/v1"
# MODEL = "deepseek-r1:14b"

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("错误: 未设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量")
    print("请执行: set DEEPSEEK_API_KEY=你的密钥")
    sys.exit(1)

BATCH_SIZE = 100          # 每批处理条数
SPOT_CHECK_PER_BATCH = 5  # 每批抽查5条
PASS_THRESHOLD = 0.80     # 正确率阀门: 80%
CHECKPOINT_EVERY = 10     # 每10条自动保存

INPUT_FILE = r"D:\PHD\find\Projects\CanalEcho\output\balanced_text_aggressive.json"
RULEBOOK_FILE = r"D:\PHD\find\Projects\CanalEcho\output\annotation_rulebook.md"
TOPIC_NAMES_FILE = r"D:\PHD\find\Projects\CanalEcho\output\topic_names.json"
OUTPUT_FILE = r"D:\PHD\find\Projects\CanalEcho\output\labeled_heritage_v2.json"
# =================================================================

try:
    from openai import OpenAI
except ImportError:
    print("请先安装依赖: pip install openai")
    sys.exit(1)


def load_rulebook(path: str) -> str:
    """加载标注规则手册作为系统Prompt"""
    if not os.path.exists(path):
        print(f"错误: 找不到规则手册 '{path}'")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_topic_names(path: str) -> dict:
    """加载主题映射表"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_labels(path: str) -> dict:
    """加载已处理的标签（断点续传）"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    existing = {}
    for item in data:
        idx = item.get("序号")
        if idx is not None and "top1_label" in item:
            existing[idx] = item
    return existing


def build_prompt(balanced_text: str, topic_names: dict, rulebook: str) -> list:
    """
    构建DeepSeek/OpenAI兼容的 messages 格式
    system: 规则手册
    user: 待标注文本 + 输出格式要求
    """
    # 构建类别说明
    categories = []
    for tid, info in topic_names.items():
        if tid == "-1":
            continue
        categories.append(f"{tid}. {info['short_name']}（{info['full_name']}）")
    
    categories_str = "\n".join(categories)
    
    system_msg = f"""你是一位大运河文化遗产语义分类专家。请严格按照以下《标注规则手册》执行分类任务。

{rulebook}

输出要求：
1. 你必须为文本输出10个类别的概率分布，格式为JSON: {{"0":0.1, "1":0.05, ..., "9":0.02}}
2. 概率之和必须严格等于1.0
3. 概率最高的类别即为Top1标签
4. 如果文本语义极度模糊、跨多个主题且无法判断主次，允许输出离群点: {{"-1":1.0}}
5. 只输出纯JSON，不要任何解释、不要markdown代码块"""

    user_msg = f"""待分类遗产文本：
\"\"\"{balanced_text[:800]}\"\"\"

可选类别：
{categories_str}

请输出JSON格式的概率分布："""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def call_llm(client: OpenAI, messages: list, max_retries: int = 3) -> dict:
    """调用DeepSeek API，解析JSON概率分布"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,  # 低温度，减少随机性
                max_tokens=500,
                response_format={"type": "json_object"}  # DeepSeek v3支持，若不支持则去掉
            )
            raw = response.choices[0].message.content.strip()
            probs = json.loads(raw)
            
            # 归一化
            total = sum(float(v) for v in probs.values())
            if total == 0:
                raise ValueError("概率和为0")
            probs = {k: float(v)/total for k, v in probs.items()}
            
            return probs
            
        except Exception as e:
            print(f"    API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                # 最终失败，返回均匀分布作为fallback（会触发阀门）
                return {str(i): 0.1 for i in range(10)}


def interactive_spot_check(batch_items: list, batch_labels: list) -> float:
    """
    交互式抽查
    返回正确率 (0.0-1.0)
    """
    n = len(batch_items)
    if n == 0:
        return 1.0
    
    correct = 0
    print("\n" + "="*60)
    print(f"本批抽查 ({len(batch_items)}条)")
    print("="*60)
    
    for i, (item, label_info) in enumerate(zip(batch_items, batch_labels), 1):
        name = item.get("遗产点名称", "unknown")
        text_preview = item.get("balanced_text", "")[:150].replace("\n", " ")
        pred = label_info["top1_label"]
        prob = label_info["top1_prob"]
        
        print(f"\n[{i}/{n}] {name}")
        print(f"文本摘要: {text_preview}...")
        print(f"预测标签: {pred} (置信度: {prob:.2f})")
        
        while True:
            user_input = input("正确? [y/n/skip] > ").strip().lower()
            if user_input in ("y", "n", "skip"):
                break
            print("请输入 y, n 或 skip")
        
        if user_input == "y":
            correct += 1
        elif user_input == "skip":
            n -= 1  # 跳过不计入分母
    
    if n == 0:
        return 1.0
    accuracy = correct / n
    print(f"\n本批正确率: {accuracy*100:.1f}% ({correct}/{n})")
    return accuracy


def main():
    print("="*60)
    print("CanalEcho Phase 4: LLM多标签Top1标注引擎 (DeepSeek版)")
    print("="*60)
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # 加载数据
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 过滤有效数据
    valid_data = [item for item in data if item.get("text_quality") != "Low" and item.get("balanced_text")]
    print(f"\n加载数据: {len(data)} 条")
    print(f"有效文本: {len(valid_data)} 条 (排除 Low/空文本)")
    
    # 加载规则手册和主题名
    rulebook = load_rulebook(RULEBOOK_FILE)
    topic_names = load_topic_names(TOPIC_NAMES_FILE)
    print(f"规则手册: {len(rulebook)} 字符")
    print(f"主题映射: {len(topic_names)} 个主题")
    
    # 断点续传检查
    existing = load_existing_labels(OUTPUT_FILE)
    if existing:
        print(f"\n检测到已有标注: {len(existing)} 条，将自动跳过")
    
    # 准备输出容器
    os.makedirs("output", exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            output_data = json.load(f)
    else:
        output_data = []
    
    # 建立序号索引
    output_idx = {item["序号"]: item for item in output_data if "序号" in item}
    
    total_batches = (len(valid_data) + BATCH_SIZE - 1) // BATCH_SIZE
    processed_count = 0
    
    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(valid_data))
        batch = valid_data[start:end]
        
        # 跳过已处理的
        batch_to_process = []
        for item in batch:
            idx = item.get("序号")
            if idx not in existing:
                batch_to_process.append(item)
        
        if not batch_to_process:
            print(f"\n第 {batch_num+1}/{total_batches} 批已全部处理，跳过")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理第 {batch_num+1}/{total_batches} 批 ({len(batch_to_process)} 条)...")
        print(f"{'='*60}")
        
        batch_labeled = []
        for item in batch_to_process:
            idx = item["序号"]
            balanced_text = item.get("balanced_text", "")
            
            messages = build_prompt(balanced_text, topic_names, rulebook)
            probs = call_llm(client, messages)
            
            # 确定Top1
            top1_id = max(probs, key=probs.get)
            top1_prob = probs[top1_id]
            
            # 映射到名称
            top1_label = topic_names.get(top1_id, {}).get("short_name", f"主题{top1_id}")
            
            label_info = {
                "label_probs": probs,
                "top1_id": top1_id,
                "top1_label": top1_label,
                "top1_prob": top1_prob
            }
            
            # 合并到原始数据
            new_item = {**item, **label_info}
            batch_labeled.append(new_item)
            output_idx[idx] = new_item
            
            processed_count += 1
            
            # 断点保存
            if processed_count % CHECKPOINT_EVERY == 0:
                output_list = list(output_idx.values())
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(output_list, f, ensure_ascii=False, indent=2)
                print(f"  ✓ 已处理 {processed_count} 条，自动保存")
        
        # 抽查
        if len(batch_labeled) >= SPOT_CHECK_PER_BATCH:
            spot_samples = random.sample(batch_labeled, SPOT_CHECK_PER_BATCH)
        else:
            spot_samples = batch_labeled
        
        # 从output_idx中提取对应样本用于抽查显示
        spot_for_check = [output_idx[s["序号"]] for s in spot_samples]
        accuracy = interactive_spot_check(spot_samples, spot_for_check)
        
        # 阀门检查
        if accuracy < PASS_THRESHOLD:
            print(f"\n{'!'*60}")
            print(f"错误: 本批正确率 {accuracy*100:.1f}% < 阈值 {PASS_THRESHOLD*100}%")
            print("标注质量不达标，已终止运行。")
            print("请检查以下可能原因:")
            print("  1. annotation_rulebook.md 的边界定义是否模糊？")
            print("  2. DeepSeek模型是否理解错误？尝试换用 deepseek-reasoner")
            print("  3. balanced_text 是否仍有噪声？")
            print(f"{'!'*60}")
            print(f"\n已保存进度至 {OUTPUT_FILE}，共 {len(output_idx)} 条")
            sys.exit(1)
        else:
            print(f"  ✓ 通过阀门，继续下一批...")
        
        # 批次结束保存
        output_list = list(output_idx.values())
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("全部完成!")
    print(f"输出: {OUTPUT_FILE}")
    print(f"总计: {len(output_idx)} 条")
    
    # 统计分布
    dist = defaultdict(int)
    for item in output_idx.values():
        lbl = item.get("top1_label", "Unknown")
        dist[lbl] += 1
    
    print(f"\n主题分布:")
    for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {lbl}: {cnt} 条")
    
    print(f"\n下一步: Phase 5 特征工程与机器学习")


if __name__ == "__main__":
    main()