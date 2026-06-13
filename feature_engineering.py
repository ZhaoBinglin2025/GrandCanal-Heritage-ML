#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CanalEcho Phase 5 v2: 4超级主题特征工程与机器学习
================================================
输入: output/labeled_heritage_v2.json (467条，含10维概率)
输出: 
  - output/features_umap_v2.csv (6维特征矩阵 + 4超级主题标签)
  - output/umap_plot_v2.png (4主题UMAP可视化)
  - output/classification_report_v2.csv
  - output/xgb_model_v2.pkl / rf_model_v2.pkl
  - output/X_train_v2.npy 等 (供Phase 6 SHAP分析直接加载)

核心修正:
  1. 基于label_probs重新加总为4超级主题概率（非硬映射）
  2. 4分类目标: 工程水利型/信仰祭祀型/考古遗址型/人居建筑型
  3. 排除离群点(-1)与缺失坐标记录
"""

import json
import os
import sys
from collections import defaultdict

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ========================== 配置 ==========================
INPUT_FILE = r"D:\PHD\find\Projects\CanalEcho\output\labeled_heritage_v2.json"
MERGE_MAP_FILE = r"D:\PHD\find\Projects\CanalEcho\output\topic_merge_map.json"
OUTPUT_DIR = r"D:\PHD\find\Projects\CanalEcho\output"

UMAP_N_COMPONENTS = 4
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

TEST_SIZE = 0.2
RANDOM_STATE = 42
SBERT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
# ========================================================


def set_chinese_font():
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/msyh.ttc',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            from matplotlib import font_manager
            font_prop = font_manager.FontProperties(fname=fp)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return font_prop
    return None


def load_merge_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_merge_data(input_path: str, merge_map: dict):
    """加载数据，基于label_probs合并为4超级主题概率"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    valid = []
    excluded = {"outlier": 0, "no_coords": 0, "no_text": 0, "no_probs": 0}
    
    for item in data:
        # 排除离群点
        if item.get("top1_id") == "-1":
            excluded["outlier"] += 1
            continue
        
        # 排除缺失坐标
        lat = item.get("纬度1")
        lon = item.get("经度1")
        if not lat or not lon or float(lat) == 0 or float(lon) == 0:
            excluded["no_coords"] += 1
            continue
        
        # 排除缺失文本
        if not item.get("balanced_text"):
            excluded["no_text"] += 1
            continue
        
        # 排除缺失概率
        probs = item.get("label_probs")
        if not probs:
            excluded["no_probs"] += 1
            continue
        
        # 合并概率
        super_probs = defaultdict(float)
        for old_id_str, prob in probs.items():
            rule = merge_map["merge_rules"].get(old_id_str)
            if rule and rule["super_id"] != -1:
                super_probs[rule["super_id"]] += float(prob)
        
        total = sum(super_probs.values())
        if total == 0:
            continue
        
        super_probs = {k: v / total for k, v in super_probs.items()}
        super_top1_id = str(max(super_probs, key=super_probs.get))
        super_top1_prob = super_probs[int(super_top1_id)]
        super_top1_name = merge_map["super_topics"][super_top1_id]["name"]
        
        # 添加到item
        item["super_probs"] = super_probs
        item["super_top1_id"] = super_top1_id
        item["super_top1_name"] = super_top1_name
        item["super_top1_prob"] = super_top1_prob
        item["lat"] = float(lat)
        item["lon"] = float(lon)
        
        valid.append(item)
    
    print(f"加载数据: {len(data)} 条")
    print(f"  排除离群点: {excluded['outlier']}")
    print(f"  排除缺失坐标: {excluded['no_coords']}")
    print(f"  排除缺失文本: {excluded['no_text']}")
    print(f"  排除缺失概率: {excluded['no_probs']}")
    print(f"有效数据: {len(valid)} 条")
    
    return valid


def encode_texts(data: list, model: SentenceTransformer):
    texts = [item["balanced_text"] for item in data]
    print(f"\nSentence-BERT编码中... ({SBERT_MODEL})")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"  嵌入矩阵: {embeddings.shape}")
    return embeddings


def reduce_umap(embeddings: np.ndarray):
    print(f"\nUMAP降维: {embeddings.shape[1]}维 → {UMAP_N_COMPONENTS}维")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_STATE,
        metric='cosine'
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"  降维后: {reduced.shape}")
    return reducer, reduced


def build_feature_matrix(reduced: np.ndarray, data: list):
    features = []
    for i, item in enumerate(data):
        features.append([
            reduced[i, 0], reduced[i, 1], reduced[i, 2], reduced[i, 3],
            item["lon"], item["lat"]
        ])
    features = np.array(features)
    print(f"\n特征矩阵: {features.shape} (UMAP4 + Lon + Lat)")
    return features


def prepare_labels(data: list, merge_map: dict):
    labels = [item["super_top1_name"] for item in data]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"\n4超级主题分布:")
    for i, cls in enumerate(le.classes_):
        count = sum(1 for l in labels if l == cls)
        info = merge_map["super_topics"][str(list(merge_map["super_topics"].keys())[list(merge_map["super_topics"].values()).index(next(v for v in merge_map["super_topics"].values() if v["name"]==cls))])]
        print(f"  {cls}: {count} 条 | {info['definition'][:40]}...")
    return y, le


def split_data(features: np.ndarray, y: np.ndarray, data: list):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_idx, test_idx in sss.split(features, y):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        data_train = [data[i] for i in train_idx]
        data_test = [data[i] for i in test_idx]
    
    print(f"\n数据划分: 训练集 {len(y_train)} / 测试集 {len(y_test)}")
    return X_train, X_test, y_train, y_test, train_idx, test_idx


def train_models(X_train, y_train):
    print(f"\n训练 XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, eval_metric='mlogloss'
    )
    xgb.fit(X_train, y_train)
    print("  ✓ XGBoost完成")
    
    print(f"\n训练 RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=3,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("  ✓ RandomForest完成")
    
    return xgb, rf


def evaluate_models(xgb, rf, X_test, y_test, le: LabelEncoder):
    print(f"\n{'='*60}")
    print("模型评估 (4超级主题)")
    print(f"{'='*60}")
    
    y_pred_xgb = xgb.predict(X_test)
    print("\n--- XGBoost ---")
    report_xgb = classification_report(y_test, y_pred_xgb, target_names=le.classes_, output_dict=True)
    print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))
    
    y_pred_rf = rf.predict(X_test)
    print("\n--- RandomForest ---")
    report_rf = classification_report(y_test, y_pred_rf, target_names=le.classes_, output_dict=True)
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    
    return report_xgb, report_rf


def save_outputs(features, y, data, le, xgb, rf, reducer, report_xgb, report_rf, train_idx, test_idx):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 特征矩阵CSV
    df = pd.DataFrame(features, columns=['UMAP_1', 'UMAP_2', 'UMAP_3', 'UMAP_4', 'Lon', 'Lat'])
    df['super_label'] = [item["super_top1_name"] for item in data]
    df['super_label_id'] = y
    df['name'] = [item.get("遗产点名称", "") for item in data]
    df['lat'] = [item["lat"] for item in data]
    df['lon'] = [item["lon"] for item in data]
    df.to_csv(os.path.join(OUTPUT_DIR, "features_umap_v2.csv"), index=False, encoding='utf-8-sig')
    print(f"\n  ✓ features_umap_v2.csv")
    
    # 2. 分类报告
    rows = []
    for k in list(report_xgb.keys())[:-3]:
        rows.append({
            'class': k,
            'xgb_precision': report_xgb[k]['precision'],
            'xgb_recall': report_xgb[k]['recall'],
            'xgb_f1': report_xgb[k]['f1-score'],
            'rf_precision': report_rf[k]['precision'],
            'rf_recall': report_rf[k]['recall'],
            'rf_f1': report_rf[k]['f1-score'],
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "classification_report_v2.csv"), index=False, encoding='utf-8-sig')
    print(f"  ✓ classification_report_v2.csv")
    
    # 3. 保存模型与划分数据
    joblib.dump(xgb, os.path.join(OUTPUT_DIR, "xgb_model_v2.pkl"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf_model_v2.pkl"))
    joblib.dump(reducer, os.path.join(OUTPUT_DIR, "umap_reducer_v2.pkl"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder_v2.pkl"))
    
    np.save(os.path.join(OUTPUT_DIR, "X_train_v2.npy"), features[train_idx])
    np.save(os.path.join(OUTPUT_DIR, "X_test_v2.npy"), features[test_idx])
    np.save(os.path.join(OUTPUT_DIR, "y_train_v2.npy"), y[train_idx])
    np.save(os.path.join(OUTPUT_DIR, "y_test_v2.npy"), y[test_idx])
    
    # 保存数据索引供SHAP使用
    with open(os.path.join(OUTPUT_DIR, "shap_data_meta_v2.json"), "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": ['UMAP_1', 'UMAP_2', 'UMAP_3', 'UMAP_4', 'Lon', 'Lat'],
            "super_classes": list(le.classes_),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "random_state": RANDOM_STATE
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 模型与数据文件已保存至 {OUTPUT_DIR}")


def plot_umap(reduced: np.ndarray, y: np.ndarray, le: LabelEncoder, merge_map: dict):
    print(f"\n生成UMAP可视化...")
    font_prop = set_chinese_font()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # 红蓝绿紫
    
    for i, cls in enumerate(le.classes_):
        mask = y == i
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=[colors[i]], label=cls, alpha=0.75, s=60,
                   edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('UMAP-1', fontsize=12)
    ax.set_ylabel('UMAP-2', fontsize=12)
    ax.set_title('CanalEcho Phase 5 v2 —— 4超级主题UMAP投影', fontsize=14)
    
    if font_prop:
        ax.legend(title='超级主题', loc='best', prop=font_prop, title_fontproperties=font_prop)
    else:
        ax.legend(title='超级主题', loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "umap_plot_v2.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ umap_plot_v2.png")
    plt.close()


def main():
    print("="*60)
    print("CanalEcho Phase 5 v2: 4超级主题特征工程与机器学习")
    print("="*60)
    
    merge_map = load_merge_map(MERGE_MAP_FILE)
    data = load_and_merge_data(INPUT_FILE, merge_map)
    if len(data) == 0:
        print("错误: 无有效数据")
        sys.exit(1)
    
    print(f"\n加载Sentence-BERT: {SBERT_MODEL}")
    sbert = SentenceTransformer(SBERT_MODEL)
    embeddings = encode_texts(data, sbert)
    
    reducer, reduced = reduce_umap(embeddings)
    features = build_feature_matrix(reduced, data)
    y, le = prepare_labels(data, merge_map)
    
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(features, y, data)
    xgb, rf = train_models(X_train, y_train)
    report_xgb, report_rf = evaluate_models(xgb, rf, X_test, y_test, le)
    
    save_outputs(features, y, data, le, xgb, rf, reducer, report_xgb, report_rf, train_idx, test_idx)
    plot_umap(reduced, y, le, merge_map)
    
    print(f"\n{'='*60}")
    print("Phase 5 v2 完成!")
    print(f"{'='*60}")
    print("\n下一步: python 06_shap_analysis.py")


if __name__ == "__main__":
    main()