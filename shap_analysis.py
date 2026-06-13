#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CanalEcho Phase 6 v4.1: SHAP可解释性与多分界线跃迁检测（v4修正版）
===============================================================
基于v4版本修正，核心改进:
  1. 删除ci_excludes_zero列（置换差异CI必然包含0，无判定意义）
  2. strong_combined改为两重标准: p<0.05 and |jump|>0.1
  3. 置换p值作为主要显著性标准，固定标准作为辅助参考
  4. 热力图图例同步更新，去掉"CI≠0"描述
  5. CSV输出17列（删除ci_excludes_zero）

输入: output/*_v2.pkl / *_v2.npy (Phase 5 v2产出)
输出:
  - shap_summary_v4.1.png
  - shap_dependence_lat_v4.1.png / lon_v4.1.png
  - shap_transition_multizone_v4.1.png
  - transition_zone_stats_v4.1.csv (17列)
  - transition_zone_heatmap_v4.1.png
"""

import json
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

OUTPUT_DIR = r"D:\PHD\find\Projects\CanalEcho\output\ml"


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


def deduplicate_by_coords(X_test, y_test, feature_names):
    """按Lat+Lon去重，保留首条"""
    lat_idx = feature_names.index("Lat")
    lon_idx = feature_names.index("Lon")
    seen = set()
    keep = []
    for i in range(len(X_test)):
        key = (round(float(X_test[i, lat_idx]), 6), round(float(X_test[i, lon_idx]), 6))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    if len(keep) < len(X_test):
        print(f"  测试集去重: {len(X_test)} -> {len(keep)} 条")
    return X_test[keep], y_test[keep]


def load_phase5_outputs():
    xgb = joblib.load(os.path.join(OUTPUT_DIR, "xgb_model_v2.pkl"))
    le = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder_v2.pkl"))
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test_v2.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test_v2.npy"))

    with open(os.path.join(OUTPUT_DIR, "shap_data_meta_v2.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    super_classes = meta["super_classes"]

    X_test, y_test = deduplicate_by_coords(X_test, y_test, feature_names)

    print(f"加载模型: XGBoost")
    print(f"测试集: {X_test.shape}")
    print(f"特征: {feature_names}")
    print(f"类别: {super_classes}")

    return xgb, le, X_test, y_test, feature_names, super_classes


def compute_shap(xgb, X_test, feature_names):
    print(f"\n计算SHAP值 (全部测试集={len(X_test)}条)...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    print(f"  SHAP矩阵: {np.array(shap_values).shape}")
    return explainer, shap_values


def permutation_test(group_a, group_b, n_permutations=100000, random_state=42):
    """
    对两组数据做置换检验，检验均值差异是否显著。

    参数:
        group_a: 数组，南侧组SHAP值
        group_b: 数组，北侧组SHAP值
        n_permutations: 置换次数（默认100000次）
        random_state: 随机种子，确保可重复

    返回:
        obs_diff: 观测差异（北-南）
        p_value: 双侧p值
        ci_lower: 95%CI下限
        ci_upper: 95%CI上限
    """
    obs_diff = np.mean(group_b) - np.mean(group_a)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    n_total = len(combined)

    rng = np.random.RandomState(random_state)
    perm_diffs = []

    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_combined = combined[perm]
        perm_a = perm_combined[:n_a]
        perm_b = perm_combined[n_a:]
        perm_diffs.append(np.mean(perm_b) - np.mean(perm_a))

    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    ci_lower = np.percentile(perm_diffs, 2.5)
    ci_upper = np.percentile(perm_diffs, 97.5)

    return obs_diff, p_value, ci_lower, ci_upper


def classify_jump_type(south_mean, north_mean):
    """
    根据南北均值符号和幅度，自动分类跃迁类型。

    分类规则:
        - 反转型跃迁: 南北均值严格异号（一正一负）
        - 同向加深型跃迁: 同号，且北侧绝对值 > 南侧绝对值
        - 同向减弱型跃迁: 同号，且北侧绝对值 <= 南侧绝对值
        - 无法判定: 任一侧为NaN

    注意: 若一侧均值为0，乘积为0，不满足严格异号，归入同向型。
    """
    if np.isnan(south_mean) or np.isnan(north_mean):
        return "无法判定"
    if south_mean * north_mean < 0:
        return "反转型跃迁"
    elif abs(north_mean) > abs(south_mean):
        return "同向加深型跃迁"
    else:
        return "同向减弱型跃迁"


def plot_summary_manual(shap_values, X_test, feature_names, super_classes):
    """手动绘制SHAP全局摘要图"""
    print(f"\n生成SHAP全局摘要图（手动绘制）...")
    font_prop = set_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for i, cls in enumerate(super_classes):
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[..., i]
        importance = np.mean(np.abs(sv), axis=0)
        idx = np.argsort(importance)[::-1]

        y_pos = np.arange(len(feature_names))
        bars = axes[i].barh(y_pos, importance[idx], color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels([feature_names[j] for j in idx], fontsize=10)
        axes[i].invert_yaxis()
        axes[i].set_xlabel('mean(|SHAP value|)', fontsize=11)

        title = f'{cls} — SHAP特征重要性'
        if font_prop:
            axes[i].set_title(title, fontsize=12, fontproperties=font_prop)
        else:
            axes[i].set_title(title, fontsize=12)
        axes[i].grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, importance[idx]):
            axes[i].text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_v4.1.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ shap_summary_v4.1.png")
    plt.close()


def plot_lat_dependence(shap_values, X_test, feature_names, super_classes):
    print(f"\n生成纬度SHAP依赖图...")
    font_prop = set_chinese_font()
    lat_idx = feature_names.index("Lat")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for i, cls in enumerate(super_classes):
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[..., i]
        lat_shap = sv[:, lat_idx]
        lat_vals = X_test[:, lat_idx]

        axes[i].scatter(lat_vals, lat_shap, c=colors[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        for line in [31.0, 34.0, 37.0]:
            axes[i].axvline(x=line, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[i].axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[i].set_xlabel('Latitude (°N)', fontsize=11)
        axes[i].set_ylabel('SHAP value (Lat)', fontsize=11)

        title = f'{cls} — 纬度边际效应'
        if font_prop:
            axes[i].set_title(title, fontsize=12, fontproperties=font_prop)
        else:
            axes[i].set_title(title, fontsize=12)

        z = np.polyfit(lat_vals, lat_shap, 2)
        p = np.poly1d(z)
        lat_sort = np.sort(lat_vals)
        axes[i].plot(lat_sort, p(lat_sort), "k--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_lat_v4.1.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ shap_dependence_lat_v4.1.png")
    plt.close()


def plot_lon_dependence(shap_values, X_test, feature_names, super_classes):
    print(f"\n生成经度SHAP依赖图...")
    font_prop = set_chinese_font()
    lon_idx = feature_names.index("Lon")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for i, cls in enumerate(super_classes):
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[..., i]
        lon_shap = sv[:, lon_idx]
        lon_vals = X_test[:, lon_idx]

        axes[i].scatter(lon_vals, lon_shap, c=colors[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        axes[i].axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[i].set_xlabel('Longitude (°E)', fontsize=11)
        axes[i].set_ylabel('SHAP value (Lon)', fontsize=11)

        title = f'{cls} — 经度边际效应'
        if font_prop:
            axes[i].set_title(title, fontsize=12, fontproperties=font_prop)
        else:
            axes[i].set_title(title, fontsize=12)

        z = np.polyfit(lon_vals, lon_shap, 2)
        p = np.poly1d(z)
        lon_sort = np.sort(lon_vals)
        axes[i].plot(lon_sort, p(lon_sort), "k--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_lon_v4.1.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ shap_dependence_lon_v4.1.png")
    plt.close()


def analyze_multizone_transition(shap_values, X_test, feature_names, super_classes):
    """多分界线跃迁检测：31°N, 34°N, 37°N（v4.1修正版）

    修正要点:
        1. 删除ci_excludes_zero列（置换差异CI必然包含0，无判定意义）
        2. strong_combined改为两重标准: p<0.05 and |jump|>0.1
        3. 置换p值作为主要显著性标准，固定标准作为辅助参考
        4. CSV输出17列（删除ci_excludes_zero）
    """
    print(f"\n{'='*60}")
    print("多分界线跃迁检测 v4.1 (31°N / 34°N / 37°N)")
    print("  [修正] strong_combined改为两重标准(p<0.05 + |jump|>0.1)")
    print("  [修正] 删除ci_excludes_zero列，CSV输出17列")
    print(f"{'='*60}")

    lat_idx = feature_names.index("Lat")
    lat_vals = X_test[:, lat_idx]
    boundaries = [31.0, 34.0, 37.0]
    results = []
    valid_jumps = []

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for i, cls in enumerate(super_classes):
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[..., i]
        lat_shap = sv[:, lat_idx]

        for j, boundary in enumerate(boundaries):
            ax = axes[i, j]

            bins = np.arange(boundary - 5, boundary + 6, 1.0)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            means, stds = [], []

            for b in range(len(bins) - 1):
                mask = (lat_vals >= bins[b]) & (lat_vals < bins[b+1])
                if mask.sum() > 0:
                    means.append(lat_shap[mask].mean())
                    stds.append(lat_shap[mask].std())
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)
            stds = np.array(stds)
            valid = ~np.isnan(means)

            ax.plot(bin_centers[valid], means[valid], 'o-', color=colors[i], linewidth=2, markersize=6)
            ax.fill_between(bin_centers[valid], 
                           (means - stds)[valid], 
                           (means + stds)[valid], 
                           alpha=0.2, color=colors[i])
            ax.axvline(x=boundary, color='black', linestyle='--', linewidth=2, label=f'{boundary}°N')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Latitude (°N)', fontsize=10)
            ax.set_ylabel('Mean SHAP (Lat)', fontsize=10)

            if j == 0:
                if font_prop:
                    ax.set_ylabel(f'{cls}\nMean SHAP (Lat)', fontsize=11, fontproperties=font_prop)
                else:
                    ax.set_ylabel(f'{cls}\nMean SHAP (Lat)', fontsize=11)

            if i == 0:
                ax.set_title(f'{boundary}°N 过渡带', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # ==================== 跃迁检测 v4.1 核心逻辑 ====================

            mask_south = (lat_vals >= boundary - 1) & (lat_vals < boundary)
            mask_north = (lat_vals >= boundary) & (lat_vals < boundary + 1)

            south_n = mask_south.sum()
            north_n = mask_north.sum()

            # [样本量检查] 每侧至少3个点
            if south_n < 3 or north_n < 3:
                print(f"\n[{cls}] @ {boundary}°N")
                print(f"  样本量: 南侧n={south_n}, 北侧n={north_n}")
                print(f"  >>> 样本不足（每侧需≥3个点），跳过置换检验")

                results.append({
                    "class": cls,
                    "boundary": boundary,
                    "south_n": int(south_n),
                    "north_n": int(north_n),
                    "south_mean": lat_shap[mask_south].mean() if south_n > 0 else np.nan,
                    "north_mean": lat_shap[mask_north].mean() if north_n > 0 else np.nan,
                    "jump": np.nan,
                    "jump_type": "样本不足",
                    "p_value": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "significant_fixed": False,
                    "significant_permutation": False,
                    "significant_adaptive": False,
                    "strong_fixed": False,
                    "strong_combined": False,
                    "strong_adaptive": False,
                })
                continue

            # [置换检验]
            south_shap = lat_shap[mask_south]
            north_shap = lat_shap[mask_north]

            obs_jump, p_value, ci_lower, ci_upper = permutation_test(
                south_shap, north_shap, n_permutations=100000, random_state=42
            )

            # [跃迁类型分类]
            south_mean = south_shap.mean()
            north_mean = north_shap.mean()
            jump_type = classify_jump_type(south_mean, north_mean)

            valid_jumps.append(abs(obs_jump))

            # [多重显著性判定]
            significant_fixed = abs(obs_jump) > 0.05
            strong_fixed = abs(obs_jump) > 0.1

            significant_permutation = p_value < 0.05
            # [v4.1修正] strong_combined改为两重标准: p<0.05 and |jump|>0.1
            strong_combined = significant_permutation and strong_fixed

            # [稳健性检验] 逐个剔除南侧点，检验结果稳定性
            # 仅对主检验p<0.05的组合做稳健性检验
            robust_results = []
            all_significant = None
            all_strong = None
            robust_flag = "未做稳健性检验（p≥0.05或样本不足）"
            
            if significant_permutation and south_n >= 2:
                for idx in range(south_n):
                    robust_south = np.delete(south_shap, idx)
                    robust_jump, robust_p, _, _ = permutation_test(
                        robust_south, north_shap, n_permutations=100000, random_state=42
                    )
                    robust_results.append({
                        "removed_idx": idx,
                        "robust_jump": robust_jump,
                        "robust_p": robust_p,
                        "still_significant": robust_p < 0.05,
                        "still_strong": (robust_p < 0.05) and (abs(robust_jump) > 0.1)
                    })
                
                all_significant = all(r["still_significant"] for r in robust_results)
                all_strong = all(r["still_strong"] for r in robust_results)
                robust_flag = "稳健" if all_significant else "不稳定（某点主导）"

            significant_adaptive = False
            strong_adaptive = False

            result = {
                "class": cls,
                "boundary": boundary,
                "south_n": int(south_n),
                "north_n": int(north_n),
                "south_mean": south_mean,
                "north_mean": north_mean,
                "jump": obs_jump,
                "jump_type": jump_type,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "significant_fixed": significant_fixed,
                "significant_permutation": significant_permutation,
                "significant_adaptive": significant_adaptive,
                "strong_fixed": strong_fixed,
                "strong_combined": strong_combined,
                "strong_adaptive": strong_adaptive,
                "robust_flag": robust_flag,
                "robust_all_significant": all_significant,
                "robust_all_strong": all_strong,
            }
            results.append(result)

            

            # [控制台输出 v4.1]
            print(f"\n[{cls}] @ {boundary}°N")
            print(f"  样本量: 南侧n={south_n}, 北侧n={north_n}")
            print(f"  {boundary-1}-{boundary}°N 平均SHAP: {south_mean:.4f}")
            print(f"  {boundary}-{boundary+1}°N 平均SHAP: {north_mean:.4f}")
            print(f"  跃迁幅度 (北-南): {obs_jump:.4f}")
            print(f"  跃迁类型: {jump_type}")
            print(f"  置换检验p值: {p_value:.4f} (100000次置换, seed=42)")
            print(f"  95%CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"  固定标准: 显著={significant_fixed}, 强跃迁={strong_fixed}")
            print(f"  置换标准: 显著={significant_permutation}, 强跃迁={strong_combined}")
            print(f"  >>> 综合判定: {'强跃迁' if strong_combined else ('显著跃迁' if significant_permutation else '不显著')}")
            if significant_permutation and south_n >= 2:
                print(f"  [稳健性检验] 逐个剔除南侧{int(south_n)}个点:")
                for r in robust_results:
                    status = "✓显著" if r["still_significant"] else "✗不显著"
                    print(f"    剔除点{r['removed_idx']}: jump={r['robust_jump']:.4f}, p={r['robust_p']:.5f} {status}")
                print(f"  >>> 整体稳健性: {robust_flag}")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_transition_multizone_v4.1.png"), dpi=300, bbox_inches='tight')
    print(f"\n  ✓ shap_transition_multizone_v4.1.png")
    plt.close()

    # ==================== 自适应分位数计算 ====================

    if len(valid_jumps) > 0:
        adaptive_sig_threshold = np.percentile(valid_jumps, 50)
        adaptive_strong_threshold = np.percentile(valid_jumps, 75)

        print(f"\n{'='*60}")
        print("自适应分位数阈值（基于有效样本）")
        print(f"  显著阈值 (50th percentile): {adaptive_sig_threshold:.4f}")
        print(f"  强跃迁阈值 (75th percentile): {adaptive_strong_threshold:.4f}")
        print(f"  有效样本数: {len(valid_jumps)} / {len(results)}")
        print(f"{'='*60}")

        for r in results:
            if r["jump_type"] != "样本不足" and not np.isnan(r["jump"]):
                r["significant_adaptive"] = abs(r["jump"]) > adaptive_sig_threshold
                r["strong_adaptive"] = (r["p_value"] < 0.05) and (abs(r["jump"]) > adaptive_strong_threshold)
    else:
        print("\n[警告] 无有效跃迁样本，自适应分位数无法计算")

    # 保存CSV（17列）
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "transition_zone_stats_v4.1.csv"), index=False, encoding='utf-8-sig')
    print(f"\n  ✓ transition_zone_stats_v4.1.csv (17列)")

    # 图B: 综合热力图
    plot_transition_heatmap_v41(df_results, super_classes, boundaries)

    return results


def plot_transition_heatmap_v41(df_results, super_classes, boundaries):
    """跃迁幅度综合热力图（v4.1修正版）

    视觉编码:
        - 背景色: RdBu_r色阶，表示jump幅度
        - 边框颜色: 表示综合判定结果
            * 深红粗边框 = 强跃迁（strong_combined=True, p<0.05且|jump|>0.1）
            * 红细边框 = 显著但非强（significant_permutation=True, strong_combined=False）
            * 蓝细边框 = 不显著
            * 灰背景 + "N/A" = 样本不足
        - 文字: jump数值 + 跃迁类型缩写

    [v4.1修正] 图例说明同步更新，去掉"CI≠0"描述
    """
    print(f"\n生成跃迁综合热力图 (v4.1)...")
    font_prop = set_chinese_font()

    pivot = df_results.pivot(index='class', columns='boundary', values='jump')
    pivot = pivot.reindex(index=super_classes, columns=boundaries)

    pivot_strong = df_results.pivot(index='class', columns='boundary', values='strong_combined')
    pivot_strong = pivot_strong.reindex(index=super_classes, columns=boundaries)

    pivot_sig = df_results.pivot(index='class', columns='boundary', values='significant_permutation')
    pivot_sig = pivot_sig.reindex(index=super_classes, columns=boundaries)

    pivot_type = df_results.pivot(index='class', columns='boundary', values='jump_type')
    pivot_type = pivot_type.reindex(index=super_classes, columns=boundaries)

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)

    ax.set_xticks(np.arange(len(boundaries)))
    ax.set_yticks(np.arange(len(super_classes)))
    ax.set_xticklabels([f'{b}°N' for b in boundaries], fontsize=11)

    if font_prop:
        ax.set_yticklabels(super_classes, fontsize=11, fontproperties=font_prop)
    else:
        ax.set_yticklabels(super_classes, fontsize=11)

    for i in range(len(super_classes)):
        for j in range(len(boundaries)):
            val = pivot.values[i, j]
            is_strong = pivot_strong.values[i, j] if not pd.isna(pivot_strong.values[i, j]) else False
            is_sig = pivot_sig.values[i, j] if not pd.isna(pivot_sig.values[i, j]) else False
            jtype = pivot_type.values[i, j] if not pd.isna(pivot_type.values[i, j]) else "样本不足"

            if jtype == "样本不足":
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, 
                                    facecolor='#cccccc', edgecolor='none', alpha=0.7)
                ax.add_patch(rect)
                text = "N/A"
                text_color = '#666666'
            elif is_strong:
                rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9, fill=False, 
                                    edgecolor='#8B0000', linewidth=4)
                ax.add_patch(rect)
                text = f'{val:.3f}'
                text_color = 'white' if abs(val) > 0.08 else 'black'
            elif is_sig:
                rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9, fill=False, 
                                    edgecolor='#e41a1c', linewidth=2)
                ax.add_patch(rect)
                text = f'{val:.3f}'
                text_color = 'white' if abs(val) > 0.08 else 'black'
            else:
                rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9, fill=False, 
                                    edgecolor='#377eb8', linewidth=2)
                ax.add_patch(rect)
                text = f'{val:.3f}'
                text_color = 'white' if abs(val) > 0.08 else 'black'

            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=12, fontweight='bold')

            if jtype not in ["样本不足", "无法判定"]:
                type_short = {
                    "反转型跃迁": "反转",
                    "同向加深型跃迁": "加深",
                    "同向减弱型跃迁": "减弱"
                }.get(jtype, jtype)
                ax.text(j, i + 0.25, type_short, ha='center', va='center',
                       color=text_color, fontsize=8, alpha=0.8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('跃迁幅度 (北-南 SHAP差)', fontsize=11)

    # [v4.1修正] 图例说明同步更新，去掉"CI≠0"描述
    legend_elements = [
        plt.Rectangle((0,0),1,1, fill=False, edgecolor='#8B0000', linewidth=4, 
                       label='强跃迁 (p<0.05, |jump|>0.1)'),
        plt.Rectangle((0,0),1,1, fill=False, edgecolor='#e41a1c', linewidth=2, 
                       label='显著跃迁 (p<0.05)'),
        plt.Rectangle((0,0),1,1, fill=False, edgecolor='#377eb8', linewidth=2, 
                       label='不显著'),
        plt.Rectangle((0,0),1,1, facecolor='#cccccc', edgecolor='none', alpha=0.7, 
                       label='样本不足'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.25, 1), fontsize=9)

    title = '多分界线SHAP跃迁幅度综合热力图 (v4.1)'
    if font_prop:
        ax.set_title(title, fontsize=14, fontproperties=font_prop)
    else:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "transition_zone_heatmap_v4.1.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ transition_zone_heatmap_v4.1.png")
    plt.close()


def save_shap_values(shap_values, X_test, feature_names, super_classes):
    print(f"\n保存SHAP值...")
    for i, cls in enumerate(super_classes):
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[..., i]
        df = pd.DataFrame(sv, columns=feature_names)
        df['Lat'] = X_test[:, feature_names.index("Lat")]
        df['Lon'] = X_test[:, feature_names.index("Lon")]
        safe_name = cls.replace('/', '_')
        df.to_csv(os.path.join(OUTPUT_DIR, f"shap_values_{safe_name}_v4.1.csv"), index=False, encoding='utf-8-sig')
    print(f"  ✓ 各类别SHAP值已保存")


def main():
    print("="*60)
    print("CanalEcho Phase 6 v4.1: SHAP可解释性与多分界线跃迁检测（修正版）")
    print("="*60)

    xgb, le, X_test, y_test, feature_names, super_classes = load_phase5_outputs()
    explainer, shap_values = compute_shap(xgb, X_test, feature_names)

    plot_summary_manual(shap_values, X_test, feature_names, super_classes)
    plot_lat_dependence(shap_values, X_test, feature_names, super_classes)
    plot_lon_dependence(shap_values, X_test, feature_names, super_classes)
    results = analyze_multizone_transition(shap_values, X_test, feature_names, super_classes)
    save_shap_values(shap_values, X_test, feature_names, super_classes)

    print(f"\n{'='*60}")
    print("Phase 6 v4.1 完成!")
    print(f"{'='*60}")
    print("\n核心产出:")
    print("  - shap_summary_v4.1.png")
    print("  - shap_dependence_lat_v4.1.png / lon_v4.1.png")
    print("  - shap_transition_multizone_v4.1.png")
    print("  - transition_zone_stats_v4.1.csv (17列)")
    print("  - transition_zone_heatmap_v4.1.png")
    print("\nv4.1修正要点:")
    print("  ✓ strong_combined改为两重标准(p<0.05 + |jump|>0.1)")
    print("  ✓ 删除ci_excludes_zero列，CSV输出17列")
    print("  ✓ 置换p值作为主要显著性标准")
    print("  ✓ 热力图图例同步更新")


if __name__ == "__main__":
    font_prop = set_chinese_font()
    main()
