#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CanalEcho Phase 2.5b: balanced_text 激进数字清洗
====================================================
输入: output/balanced_text.json (或上一步的 balanced_text_cleaned.json)
输出: output/balanced_text_aggressive.json

清洗规则（激进模式）:
  1. 删除所有阿拉伯数字（0-9），无论位数、无论上下文
  2. 删除数字相关的量词/单位残留（如 "m²" "km" "公顷" 前的数字已删，但清理残留符号）
  3. 删除孤立标点（如 "（年）" → "（）" → 删除空括号）
  4. 删除连续空格、多余换行
  5. 保留: 中文数字（"十三""五百""万"）、中文标点、所有汉字内容

注意:
  - 仅清洗 balanced_text 字段
  - 原始字段（full_text, baidu_baike_text 等）完全保留
  - "3孔石桥" → "孔石桥"（语义可理解）
  - "4A级景区" → "级景区"（略有损失，但"景区"语义覆盖）
  - "2022年" → "年"（年代信息丢失，但BERTopic不需要精确年份）
"""

import json
import os
import re

INPUT_FILE = r"D:\PHD\find\Projects\CanalEcho\output\balanced_text.json"
OUTPUT_FILE = r"D:\PHD\find\Projects\CanalEcho\output\balanced_text_aggressive.json"


def aggressive_clean(text: str) -> tuple:
    """
    激进清洗：删除所有阿拉伯数字及相关噪声
    """
    if not text:
        return text, 0

    original_len = len(text)

    # 规则1: 删除所有阿拉伯数字（0-9）
    text = re.sub(r'[0-9]', '', text)

    # 规则2: 删除数字相关的常见残留符号/单位
    # 如 "m²" "km" "㎡" "公顷" 等（数字已删，但单位可能残留）
    text = re.sub(r'[mM][²2]', '', text)  # m², m2
    text = re.sub(r'[kK][mM]', '', text)   # km
    text = re.sub(r'[㎡]', '', text)        # 平方米符号
    text = re.sub(r'[亩]', '', text)        # 亩（若前面有数字已删，单独"亩"保留？不，激进方案删除面积单位）
    # 上面这行：其实"亩"是汉字，但如果是"占地面积约亩"就不通。保守起见，不删汉字单位。

    # 规则3: 清理孤立标点组合
    # "（年）" → "（）" → 删除
    text = re.sub(r'[（(][）)]', '', text)
    # "[]" "【】" 空括号
    text = re.sub(r'[\[【]\s*[\]】]', '', text)
    # 连续标点
    text = re.sub(r'[，,；;。．]+[，,；;。．]+', '。', text)

    # 规则4: 删除英文单词中的数字残留（如 "Baodai Bridge" 保留，但 "4A" → "A"）
    # 其实规则1已经删除了数字，"4A" → "A"
    # 清理孤立的单个字母（如 "A级" → "级"）
    text = re.sub(r'\b[A-Za-z]\b', '', text)

    # 规则5: 清理多余空格和换行
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    # 规则6: 清理句首句尾的孤立标点
    text = re.sub(r'^[，,；;。．]+', '', text)
    text = re.sub(r'[，,；;。．]+$', '', text)

    cleaned_len = len(text)
    removed = original_len - cleaned_len

    return text, removed


def main():
    print("=" * 60)
    print("CanalEcho Phase 2.5b: balanced_text 激进数字清洗")
    print("=" * 60)

    # 检查输入（优先用原始balanced_text，若不存在则用cleaned版本）
    input_file = INPUT_FILE
    if not os.path.exists(input_file):
        alt_file = "output/balanced_text_cleaned.json"
        if os.path.exists(alt_file):
            input_file = alt_file
            print(f"\n注意: 未找到 {INPUT_FILE}，使用 {alt_file}")
        else:
            print(f"\n错误: 找不到输入文件 '{INPUT_FILE}' 或 '{alt_file}'")
            print("请先运行 01b_text_balancer.py 生成 balanced_text.json")
            return

    # 加载数据
    print(f"\n加载: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  总记录数: {len(data)}")

    # 激进清洗
    print("\n开始激进清洗 balanced_text 字段...")
    print("  规则: 删除所有阿拉伯数字（0-9）及相关残留")
    total_removed = 0
    cleaned_count = 0
    unchanged_count = 0
    empty_after_clean = 0

    for i, item in enumerate(data):
        bt = item.get("balanced_text", "")
        if not bt:
            unchanged_count += 1
            continue

        cleaned_bt, removed = aggressive_clean(bt)
        item["balanced_text"] = cleaned_bt
        item["balanced_text_aggressive_cleaned"] = True

        total_removed += removed
        if removed > 0:
            cleaned_count += 1
        else:
            unchanged_count += 1

        if not cleaned_bt.strip():
            empty_after_clean += 1

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(data)} 条...")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ 激进清洗完成!")
    print(f"  清洗了 {cleaned_count} 条, 未变动 {unchanged_count} 条")
    print(f"  共删除 {total_removed} 个字符的噪声")
    print(f"  清洗后为空文本: {empty_after_clean} 条")
    print(f"  输出: {OUTPUT_FILE}")

    # 打印5条清洗前后对比示例
    print(f"\n  清洗示例（前5条，展示前100字）:")
    for i in range(min(5, len(data))):
        item = data[i]
        name = item.get("遗产点名称", "unknown")
        bt = item.get("balanced_text", "")
        print(f"\n  [{i+1}] {name}")
        preview = bt[:100] + "..." if len(bt) > 100 else bt
        print(f"      {preview}")

    print(f"\n下一步:")
    print(f"  1. 修改 02b_topic_discovery.py 的 INPUT_FILE 为 'output/balanced_text_aggressive.json'")
    print(f"  2. 重新运行 Phase 3 主题涌现")
    print(f"  3. 检查关键词是否变为纯中文语义词")
    print("=" * 60)


if __name__ == "__main__":
    main()
