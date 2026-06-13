#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CanalEcho Phase 7 v3: GWLR地理加权逻辑回归
===========================================
输入: output/features_umap_v2.csv
输出:
  - gwlr_coefficients_v3.csv (每个点的局部回归系数)
  - gwlr_heatmap_*_v3.png (4张，各类别空间系数图)
  - gwlr_local_stats_v3.csv (局部统计摘要)

核心:
  1. 4个独立二分类逻辑回归（每个超级主题一个）
  2. 自变量: Lat, Lon, UMAP_1, UMAP_2, UMAP_3, UMAP_4
  3. 自适应带宽: k-NN高斯核，AICc优化k值
  4. 自动去重 (name+Lat+Lon)
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist

OUTPUT_DIR = r"D:\PHD\find\Projects\CanalEcho\output\ml"
INPUT_CSV = os.path.join(OUTPUT_DIR, "features_umap_v2.csv")

# GWLR候选k值（最近邻数量，覆盖稀疏到密集）
K_CANDIDATES = [15, 25, 40, 60, 90]


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


def load_and_dedup():
    print(f"加载数据: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    n_before = len(df)
    
    # 按名称+Lat+Lon去重
    df = df.drop_duplicates(subset=['name', 'Lat', 'Lon'], keep='first')
    n_after = len(df)
    
    print(f"  去重前: {n_before} 条, 去重后: {n_after} 条 (删除 {n_before - n_after} 条)")
    
    # 特征列
    feature_cols = ['Lat', 'Lon', 'UMAP_1', 'UMAP_2', 'UMAP_3', 'UMAP_4']
    X = df[feature_cols].values
    coords = df[['Lon', 'Lat']].values
    
    classes = df['super_label'].unique().tolist()
    print(f"  超级主题: {classes}")
    
    return df, X, coords, classes, feature_cols


def gaussian_weights(distances, bw):
    """高斯核权重"""
    if bw <= 0:
        bw = 1e-6
    return np.exp(-0.5 * (distances / bw) ** 2)


def local_logit_fit(X, y, weights):
    """加权逻辑回归，返回参数和AIC"""
    try:
        # 检查是否全0或全1（完美分离）
        if y.sum() == 0 or y.sum() == len(y):
            return np.full(X.shape[1] + 1, np.nan), np.inf
        
        X_const = sm.add_constant(X, has_constant='add')
        # 权重归一化，避免数值过小
        w_norm = weights / (weights.max() + 1e-10)
        model = sm.GLM(y, X_const, family=sm.families.Binomial(), freq_weights=w_norm)
        result = model.fit()
        return result.params, result.aic
    except Exception:
        return np.full(X.shape[1] + 1, np.nan), np.inf


def compute_aicc_for_k(X, y, coords, k):
    """计算给定k值的平均AIC"""
    n = len(y)
    dist_matrix = cdist(coords, coords, metric='euclidean')
    aics = []
    
    for i in range(n):
        dists = dist_matrix[i]
        sorted_dists = np.sort(dists)
        bw = sorted_dists[min(k, n - 1)]
        weights = gaussian_weights(dists, bw)
        weights[i] = 1.0
        
        _, aic = local_logit_fit(X, y, weights)
        if not np.isinf(aic):
            aics.append(aic)
    
    return np.mean(aics) if aics else np.inf


def select_bandwidth(X, y, coords, k_candidates):
    """通过AICc选择最优k"""
    print(f"\nAICc自适应带宽选择...")
    best_aic = np.inf
    best_k = k_candidates[0]
    
    for k in k_candidates:
        aic = compute_aicc_for_k(X, y, coords, k)
        print(f"  k={k:3d}: 平均AIC={aic:.2f}")
        if aic < best_aic:
            best_aic = aic
            best_k = k
    
    print(f"  最优k={best_k} (最小AIC={best_aic:.2f})")
    return best_k


def run_gwlr_for_class(df, X, coords, class_name, feature_cols, k_opt):
    """对单个超级主题运行GWLR"""
    print(f"\n{'='*50}")
    print(f"GWLR: {class_name} (k={k_opt})")
    print(f"{'='*50}")
    
    y = (df['super_label'] == class_name).astype(int).values
    n = len(y)
    
    dist_matrix = cdist(coords, coords, metric='euclidean')
    coeff_names = ['Intercept'] + feature_cols
    coeffs = np.zeros((n, len(coeff_names)))
    aics = []
    
    for i in range(n):
        dists = dist_matrix[i]
        sorted_dists = np.sort(dists)
        bw = sorted_dists[min(k_opt, n - 1)]
        if bw == 0:
            bw = 1e-6
        weights = gaussian_weights(dists, bw)
        weights[i] = 1.0
        
        params, aic = local_logit_fit(X, y, weights)
        coeffs[i] = params
        aics.append(aic)
    
    # 构建结果DataFrame
    result_df = df[['name', 'Lat', 'Lon', 'super_label']].copy()
    for j, cn in enumerate(coeff_names):
        result_df[f'coef_{cn}'] = coeffs[:, j]
    
    result_df['local_aic'] = aics
    result_df['is_target'] = y
    
    # 打印全局系数摘要
    print(f"\n局部系数均值:")
    for j, cn in enumerate(coeff_names):
        mean_coef = np.nanmean(coeffs[:, j])
        print(f"  {cn:15s}: {mean_coef:+.4f}")
    
    return result_df, coeffs, coeff_names


def plot_gwlr_heatmap(df_result, class_name, coeff_names, feature_cols):
    """绘制GWLR空间系数热力图"""
    print(f"\n  生成 {class_name} 空间系数图...")
    font_prop = set_chinese_font()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(feature_cols):
        ax = axes[idx]
        coef_col = f'coef_{feat}'
        
        vmin, vmax = -1.5, 1.5
        # 对Lat/Lon系数范围可能更大
        if feat in ['Lat', 'Lon']:
            vmin, vmax = -3.0, 3.0
        
        scatter = ax.scatter(df_result['Lon'], df_result['Lat'], 
                           c=df_result[coef_col], cmap='RdBu_r', 
                           s=60, alpha=0.85, edgecolors='white', linewidth=0.3,
                           vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
        title = f'{feat} 局部系数'
        if font_prop:
            ax.set_title(title, fontsize=11, fontproperties=font_prop)
        else:
            ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, shrink=0.8)
    
    # 第6个子图：局部AIC
    ax = axes[5]
    aic_vals = df_result['local_aic'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(aic_vals) > 0:
        vmin, vmax = aic_vals.quantile(0.05), aic_vals.quantile(0.95)
        scatter = ax.scatter(df_result['Lon'], df_result['Lat'], 
                           c=df_result['local_aic'], cmap='YlOrRd', 
                           s=60, alpha=0.85, edgecolors='white', linewidth=0.3,
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
        title = '局部AIC'
        if font_prop:
            ax.set_title(title, fontsize=11, fontproperties=font_prop)
        else:
            ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, shrink=0.8)
    
    fig.suptitle(f'GWLR: {class_name} — 空间回归系数分布', fontsize=16, y=1.02)
    if font_prop:
        fig.suptitle(f'GWLR: {class_name} — 空间回归系数分布', 
                    fontsize=16, y=1.02, fontproperties=font_prop)
    
    plt.tight_layout()
    safe_name = class_name.replace('/', '_')
    fname = f"gwlr_heatmap_{safe_name}_v3.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    print(f"  ✓ {fname}")
    plt.close()


def save_gwlr_results(all_results, classes):
    """保存所有GWLR结果"""
    print(f"\n保存GWLR系数...")
    
    combined = []
    for cls, (df_res, _, _) in zip(classes, all_results):
        df_res['target_class'] = cls
        combined.append(df_res)
    
    full_df = pd.concat(combined, ignore_index=True)
    full_df.to_csv(os.path.join(OUTPUT_DIR, "gwlr_coefficients_v3.csv"), 
                   index=False, encoding='utf-8-sig')
    print(f"  ✓ gwlr_coefficients_v3.csv")
    
    # 局部统计摘要
    stats_rows = []
    for cls, (df_res, coeffs, cnames) in zip(classes, all_results):
        row = {'class': cls, 'n': len(df_res)}
        for j, cn in enumerate(cnames):
            col = f'coef_{cn}'
            row[f'{cn}_mean'] = df_res[col].mean()
            row[f'{cn}_std'] = df_res[col].std()
        stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "gwlr_local_stats_v3.csv"), 
                    index=False, encoding='utf-8-sig')
    print(f"  ✓ gwlr_local_stats_v3.csv")


def main():
    print("="*60)
    print("CanalEcho Phase 7 v3: GWLR地理加权逻辑回归")
    print("="*60)
    
    df, X, coords, classes, feature_cols = load_and_dedup()
    
    # 用第一个类别快速搜索最优k，所有类别共享（节省计算）
    y_temp = (df['super_label'] == classes[0]).astype(int).values
    k_opt = select_bandwidth(X, y_temp, coords, K_CANDIDATES)
    
    all_results = []
    for cls in classes:
        df_res, coeffs, cnames = run_gwlr_for_class(df, X, coords, cls, feature_cols, k_opt)
        plot_gwlr_heatmap(df_res, cls, cnames, feature_cols)
        all_results.append((df_res, coeffs, cnames))
    
    save_gwlr_results(all_results, classes)
    
    print(f"\n{'='*60}")
    print("Phase 7 v3 完成!")
    print(f"{'='*60}")
    print("\n核心产出:")
    print("  - gwlr_coefficients_v3.csv (全部局部系数)")
    print("  - gwlr_local_stats_v3.csv (统计摘要)")
    print("  - gwlr_heatmap_*_v3.png (4张空间系数图)")


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    main()