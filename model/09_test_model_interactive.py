#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CanalEcho 模型交互式测试工具
==============================
用法:
    python test_model_interactive.py
    
功能:
    1. 自动加载训练好的模型（xgb_model_v2.pkl + umap_reducer_v2.pkl + label_encoder_v2.pkl）
    2. 交互式输入：遗产名称、百科文本、经纬度
    3. 自动完成：文本→Sentence-BERT→UMAP降维→拼接经纬度→XGBoost预测
    4. 输出：预测主题 + 置信度 + 4类概率分布
    
依赖安装:
    pip install joblib numpy pandas sentence-transformers umap-learn xgboost scikit-learn
"""

import os
import sys
import re

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import umap

# ========================== 配置路径 ==========================
# 模型文件路径
MODEL_DIR = r"D:\PHD\find\Projects\CanalEcho\output\ml"

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model_v2.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_v2.pkl")
UMAP_REDUCER_PATH = os.path.join(MODEL_DIR, "umap_reducer_v2.pkl")

# Sentence-BERT模型名称
SBERT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# 特征顺序（和训练时一致）
FEATURE_NAMES = ['UMAP_1', 'UMAP_2', 'UMAP_3', 'UMAP_4', 'Lon', 'Lat']
# ============================================================


class CanalEchoPredictor:
    """CanalEcho遗产主题预测器"""
    
    def __init__(self):
        self.xgb_model = None
        self.label_encoder = None
        self.umap_reducer = None
        self.sbert = None
        self.is_ready = False
        
    def load_models(self):
        """加载所有模型和工具"""
        print("=" * 60)
        print("正在加载模型和工具...")
        print("=" * 60)
        
        # 1. 检查模型文件是否存在
        for path, name in [(XGB_MODEL_PATH, "XGBoost模型"),
                          (LABEL_ENCODER_PATH, "标签编码器"),
                          (UMAP_REDUCER_PATH, "UMAP降维器")]:
            if not os.path.exists(path):
                print(f"❌ 错误: 找不到 {name}: {path}")
                print("请确认 MODEL_DIR 路径是否正确")
                return False
        
        # 2. 加载XGBoost模型
        print(f"\n[1/4] 加载 XGBoost 模型...")
        self.xgb_model = joblib.load(XGB_MODEL_PATH)
        print(f"  ✓ 已加载: {os.path.basename(XGB_MODEL_PATH)}")
        
        # 3. 加载标签编码器
        print(f"\n[2/4] 加载标签编码器...")
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.classes = self.label_encoder.classes_
        print(f"  ✓ 可预测主题: {list(self.classes)}")
        
        # 4. 加载UMAP降维器
        print(f"\n[3/4] 加载 UMAP 降维器...")
        self.umap_reducer = joblib.load(UMAP_REDUCER_PATH)
        print(f"  ✓ 已加载: {os.path.basename(UMAP_REDUCER_PATH)}")
        
        # 5. 加载Sentence-BERT（首次会自动下载）
        print(f"\n[4/4] 加载 Sentence-BERT 模型 ({SBERT_MODEL})...")
        print("  ⚠ 首次运行需要下载约400MB模型文件，请耐心等待...")
        self.sbert = SentenceTransformer(SBERT_MODEL)
        print(f"  ✓ Sentence-BERT 加载完成")
        
        self.is_ready = True
        print(f"\n{'=' * 60}")
        print(" 所有模型加载成功，可以开始预测")
        print(f"{'=' * 60}")
        return True
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理：和训练时一致的清洗规则"""
        if not text:
            return ""
        
        # 删除所有阿拉伯数字（和01d_aggressive_clean.py一致）
        text = re.sub(r'[0-9]', '', text)
        
        # 删除数字相关残留符号
        text = re.sub(r'[mM][²2]', '', text)
        text = re.sub(r'[kK][mM]', '', text)
        text = re.sub(r'[㎡]', '', text)
        
        # 清理孤立标点
        text = re.sub(r'[（(][）)]', '', text)
        text = re.sub(r'[\[【]\s*[\]】]', '', text)
        text = re.sub(r'[，,；;。．]+[，,；;。．]+', '。', text)
        
        # 删除孤立单个英文字母
        text = re.sub(r'\b[A-Za-z]\b', '', text)
        
        # 清理空格和换行
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        # 清理句首句尾标点
        text = re.sub(r'^[，,；;。．]+', '', text)
        text = re.sub(r'[，,；;。．]+$', '', text)
        
        return text
    
    def predict(self, name: str, raw_text: str, latitude: float, longitude: float) -> dict:
        """
        预测一条遗产的主题类型
        
        参数:
            name: 遗产点名称
            raw_text: 原始百科文本（会自动清洗）
            latitude: 纬度（如 33.1487609552021）
            longitude: 经度（如 119.339826451795）
        
        返回:
            dict: 包含预测结果的字典
        """
        if not self.is_ready:
            return {"error": "模型未加载，请先调用 load_models()"}
        
        # Step 1: 文本清洗
        cleaned_text = self.preprocess_text(raw_text)
        if not cleaned_text or len(cleaned_text) < 50:
            return {
                "error": f"清洗后文本过短（{len(cleaned_text)}字），无法预测。请提供更详细的百科介绍。"
            }
        
        # Step 2: Sentence-BERT编码 → 384维
        print(f"\n  [处理中] {name}")
        print(f"    文本清洗后: {len(cleaned_text)} 字")
        print(f"    Step 1/3: Sentence-BERT 编码...")
        embedding_384 = self.sbert.encode([cleaned_text])
        
        # Step 3: UMAP降维 → 4维
        print(f"    Step 2/3: UMAP 降维 (384→4)...")
        embedding_4 = self.umap_reducer.transform(embedding_384)
        
        # Step 4: 拼接经纬度 → 6维特征
        print(f"    Step 3/3: 拼接经纬度 → 6维特征...")
        features_6 = np.hstack([embedding_4, [[longitude, latitude]]])
        
        # 打印特征值（方便调试）
        print(f"\n    6维特征值:")
        for i, fname in enumerate(FEATURE_NAMES):
            print(f"      {fname}: {features_6[0, i]:+.4f}")
        
        # Step 5: XGBoost预测
        pred_class_num = self.xgb_model.predict(features_6)[0]
        pred_probs = self.xgb_model.predict_proba(features_6)[0]
        
        # 转换回主题名称
        pred_class_name = self.label_encoder.inverse_transform([pred_class_num])[0]
        confidence = pred_probs[pred_class_num]
        
        # 构建概率分布字典
        prob_dict = {
            cls: float(prob) 
            for cls, prob in zip(self.classes, pred_probs)
        }
        
        return {
            "name": name,
            "predicted_class": pred_class_name,
            "confidence": float(confidence),
            "all_probabilities": prob_dict,
            "features": {
                fname: float(features_6[0, i]) 
                for i, fname in enumerate(FEATURE_NAMES)
            },
            "text_length": len(cleaned_text)
        }


def get_float_input(prompt: str) -> float:
    """安全获取浮点数输入"""
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  请输入有效的数字！")


def main():
    """主函数：交互式预测"""
    
    # 初始化预测器
    predictor = CanalEchoPredictor()
    
    # 加载模型
    if not predictor.load_models():
        print("\n 模型加载失败，请检查路径后重试")
        sys.exit(1)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   CanalEcho 遗产主题预测器                    ║
║                                                              ║
║  使用方法：                                                   ║
║  1. 输入遗产点名称（如：刘堡减水闸）                            ║
║  2. 输入该遗产的百度百科介绍文本（粘贴即可）                     ║
║  3. 输入纬度（如：33.1488）                                    ║
║  4. 输入经度（如：119.3398）                                   ║
║  5. 查看预测结果                                              ║
║                                                              ║
║  预测任务完成，按 'q' 或 'quit' 退出程序                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\n" + "=" * 60)
        
        # 输入遗产名称
        name = input("\n 遗产点名称 (或 q 退出): ").strip()
        if name.lower() in ('q', 'quit', 'exit'):
            print("\n 感谢使用，再见！")
            break
        
        if not name:
            print("   名称不能为空，请重新输入")
            continue
        
        # 输入百科文本
        print("\n 请输入百科介绍文本（输入空行结束）:")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "" and len(lines) > 0:
                    break
                lines.append(line)
            except EOFError:
                break
        
        raw_text = "\n".join(lines).strip()
        if not raw_text:
            print("   文本不能为空，请重新输入")
            continue
        
        # 输入经纬度
        print("\n 请输入地理位置:")
        latitude = get_float_input("  纬度 (如 33.1488): ")
        longitude = get_float_input("  经度 (如 119.3398): ")
        
        # 执行预测
        print(f"\n{'=' * 60}")
        print(" 开始预测...")
        
        result = predictor.predict(name, raw_text, latitude, longitude)
        
        # 检查结果
        if "error" in result:
            print(f"\n 预测失败: {result['error']}")
            continue
        
        # 打印结果
        print(f"\n{'=' * 60}")
        print(" 预测结果")
        print(f"{'=' * 60}")
        print(f"\n  遗产名称: {result['name']}")
        print(f"  文本长度: {result['text_length']} 字")
        print(f"\n   预测主题: 【{result['predicted_class']}】")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"\n   各类别概率分布:")
        
        # 按概率排序
        sorted_probs = sorted(
            result['all_probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for cls, prob in sorted_probs:
            marker = " ← 预测结果" if cls == result['predicted_class'] else ""
            bar = "█" * int(prob * 20)  # 简单进度条
            print(f"    {cls:12s}: {prob:.2%} {bar}{marker}")
        
        print(f"\n{'=' * 60}")
        print(" 预测完成！输入下一条或按 q 退出")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()