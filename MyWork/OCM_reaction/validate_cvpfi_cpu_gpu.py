# -*- coding: utf-8 -*-
"""
CPU版とGPU版のCVPFI結果を比較検証するスクリプト
"""

import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# モジュールを直接インポート
import importlib.util

# CPU版のインポート
cpu_path = os.path.join(os.path.dirname(__file__), 'OCM_cvpfi.py')
spec_cpu = importlib.util.spec_from_file_location("OCM_cvpfi", cpu_path)
ocm_cvpfi_cpu = importlib.util.module_from_spec(spec_cpu)
spec_cpu.loader.exec_module(ocm_cvpfi_cpu)

# GPU版のインポート
gpu_path = os.path.join(os.path.dirname(__file__), 'OCM_cvpfi_GPU.py')
spec_gpu = importlib.util.spec_from_file_location("OCM_cvpfi_GPU", gpu_path)
ocm_cvpfi_gpu = importlib.util.module_from_spec(spec_gpu)
spec_gpu.loader.exec_module(ocm_cvpfi_gpu)

calcCVPFI_CPU = ocm_cvpfi_cpu.calcCVPFI
calcCVPFI_GPU = ocm_cvpfi_gpu.calcCVPFI

def validate_cvpfi_results(y_name, x_name, model_name):
    """
    CPU版とGPU版のCVPFI結果を比較検証する関数
    
    Parameters:
    -----------
    y_name : str
        目的変数名
    x_name : str
        説明変数名
    model_name : str
        モデル名（GPR_3, GPR_4, GPR_11, GPR_12など）
    """
    print("=" * 80)
    print(f"検証開始: y={y_name}, x={x_name}, model={model_name}")
    print("=" * 80)
    
    # データセットの読み込み（data_for_articleフォルダを使用）
    # 現在のディレクトリがMyWorkの場合と、親ディレクトリの場合の両方に対応
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(base_dir) == 'MyWork':
        data_path = os.path.join(base_dir, 'datasets', 'larger_data', 'data_for_article', f'{x_name}.csv')
    else:
        data_path = os.path.join(base_dir, 'MyWork', 'datasets', 'larger_data', 'data_for_article', f'{x_name}.csv')
    
    # パスが存在しない場合は、相対パスで試す
    if not os.path.exists(data_path):
        data_path = os.path.join('datasets', 'larger_data', 'data_for_article', f'{x_name}.csv')
    try:
        data_cpu = pd.read_csv(data_path, index_col=0, header=0, encoding='utf-8-sig')
        print(f"CPU版データパス: {data_path}")
        # 最初の50行のみ使用
        data_cpu = data_cpu.head(50)
        print(f"データを最初の50行に制限: {data_cpu.shape[0]}行")
    except Exception as e:
        print(f"CPU版データ読み込みエラー: {e}")
        return None
    
    try:
        data_gpu = pd.read_csv(data_path, index_col=0, header=0, encoding='utf-8-sig')
        print(f"GPU版データパス: {data_path}")
        # 最初の50行のみ使用
        data_gpu = data_gpu.head(50)
        print(f"データを最初の50行に制限: {data_gpu.shape[0]}行")
    except Exception as e:
        print(f"GPU版データ読み込みエラー: {e}")
        return None
    
    # データの前処理（inf/nanの処理）
    for data in [data_cpu, data_gpu]:
        if np.inf in data.values or np.nan in data.values:
            data.replace(np.inf, np.nan, inplace=True)
            dropped_columns = data.columns[data.isnull().any()].tolist()
            data.dropna(axis=1, how='any', inplace=True)
            if dropped_columns:
                print(f"inf or nan found and dropped: {dropped_columns}")
    
    # データが同じか確認
    if not data_cpu.equals(data_gpu):
        print("警告: CPU版とGPU版で使用するデータが異なる可能性があります")
        # 共通の列のみを使用
        common_cols = data_cpu.columns.intersection(data_gpu.columns)
        data_cpu = data_cpu[common_cols]
        data_gpu = data_gpu[common_cols]
        print(f"共通列数: {len(common_cols)}")
    
    # データセットの準備
    x_train_cpu = data_cpu.iloc[:, 1:].copy()
    y_train_cpu = pd.DataFrame(data_cpu[y_name], columns=[y_name])
    
    x_train_gpu = data_gpu.iloc[:, 1:].copy()
    y_train_gpu = pd.DataFrame(data_gpu[y_name], columns=[y_name])
    
    # var()==0の列を削除
    for x_train in [x_train_cpu, x_train_gpu]:
        zero_var_cols = x_train.columns[np.where(x_train.var() == 0)]
        if len(zero_var_cols) > 0:
            x_train.drop(zero_var_cols, axis=1, inplace=True)
    
    # 共通の列のみを使用
    common_x_cols = x_train_cpu.columns.intersection(x_train_gpu.columns)
    x_train_cpu = x_train_cpu[common_x_cols]
    x_train_gpu = x_train_gpu[common_x_cols]
    
    # autoscale
    autoscaled_x_train_cpu = (x_train_cpu - x_train_cpu.mean(axis=0)) / x_train_cpu.std(axis=0, ddof=1)
    autoscaled_y_train_cpu = (y_train_cpu - y_train_cpu.mean()) / y_train_cpu.std(ddof=1)
    
    autoscaled_x_train_gpu = (x_train_gpu - x_train_gpu.mean(axis=0)) / x_train_gpu.std(axis=0, ddof=1)
    autoscaled_y_train_gpu = (y_train_gpu - y_train_gpu.mean()) / y_train_gpu.std(ddof=1)
    
    # fold_number
    fold_number = x_train_cpu.shape[0]
    
    print(f"\nデータ形状: {x_train_cpu.shape}")
    print(f"特徴量数: {x_train_cpu.shape[1]}")
    print(f"サンプル数: {x_train_cpu.shape[0]}")
    print(f"fold_number: {fold_number}")
    
    # CPU版のCVPFI計算
    print("\n" + "-" * 80)
    print("CPU版のCVPFI計算を開始...")
    print("-" * 80)
    try:
        # コピーを作成して渡す（calcCVPFI内で列名が変更されるため）
        autoscaled_x_train_cpu_copy = autoscaled_x_train_cpu.copy()
        autoscaled_y_train_cpu_copy = autoscaled_y_train_cpu.copy()
        importances_mean_cpu, importances_std_cpu, importances_cpu = \
            calcCVPFI_CPU(x_name, model_name, y_name, x_train_cpu, y_train_cpu, 
                         autoscaled_x_train_cpu_copy, autoscaled_y_train_cpu_copy, fold_number)
        print("CPU版のCVPFI計算完了")
    except Exception as e:
        print(f"CPU版のCVPFI計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # GPU版のCVPFI計算
    print("\n" + "-" * 80)
    print("GPU版のCVPFI計算を開始...")
    print("-" * 80)
    try:
        # コピーを作成して渡す（calcCVPFI内で列名が変更されるため）
        autoscaled_x_train_gpu_copy = autoscaled_x_train_gpu.copy()
        autoscaled_y_train_gpu_copy = autoscaled_y_train_gpu.copy()
        importances_mean_gpu, importances_std_gpu, importances_gpu = \
            calcCVPFI_GPU(x_name, model_name, y_name, x_train_gpu, y_train_gpu, 
                         autoscaled_x_train_gpu_copy, autoscaled_y_train_gpu_copy, fold_number)
        print("GPU版のCVPFI計算完了")
    except Exception as e:
        print(f"GPU版のCVPFI計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 結果の比較
    print("\n" + "=" * 80)
    print("結果の比較")
    print("=" * 80)
    
    # 重要度の平均値の比較
    print("\n【重要度の平均値の比較】")
    print(f"CPU版: mean={np.mean(importances_mean_cpu):.6f}, std={np.std(importances_mean_cpu):.6f}")
    print(f"GPU版: mean={np.mean(importances_mean_gpu):.6f}, std={np.std(importances_mean_gpu):.6f}")
    
    # 相関係数
    if len(importances_mean_cpu) == len(importances_mean_gpu):
        correlation = np.corrcoef(importances_mean_cpu, importances_mean_gpu)[0, 1]
        print(f"相関係数: {correlation:.6f}")
        
        # 差分
        diff = importances_mean_cpu - importances_mean_gpu
        print(f"差分: mean={np.mean(diff):.6f}, std={np.std(diff):.6f}, max_abs={np.max(np.abs(diff)):.6f}")
        
        # 相対誤差
        relative_error = np.abs(diff) / (np.abs(importances_mean_cpu) + 1e-10)
        print(f"相対誤差: mean={np.mean(relative_error):.6f}, max={np.max(relative_error):.6f}")
    
    # 結果をDataFrameにまとめる
    if len(importances_mean_cpu) == len(importances_mean_gpu) and len(x_train_cpu.columns) == len(importances_mean_cpu):
        comparison_df = pd.DataFrame({
            'feature': x_train_cpu.columns[:len(importances_mean_cpu)],
            'importance_cpu': importances_mean_cpu,
            'importance_gpu': importances_mean_gpu,
            'std_cpu': importances_std_cpu,
            'std_gpu': importances_std_gpu,
            'diff': importances_mean_cpu - importances_mean_gpu,
            'relative_error': np.abs(importances_mean_cpu - importances_mean_gpu) / (np.abs(importances_mean_cpu) + 1e-10)
        })
        
        # 結果をCSVに保存
        output_path = f'result/cvpfi/comparison_{y_name}_{x_name}_{model_name}.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        comparison_df.to_csv(output_path, encoding='cp932', index=False)
        print(f"\n比較結果を保存: {output_path}")
        
        # 上位10個の特徴量を表示
        print("\n【上位10個の特徴量の比較】")
        top10_cpu = comparison_df.nlargest(10, 'importance_cpu')
        print("\nCPU版の上位10個:")
        print(top10_cpu[['feature', 'importance_cpu', 'importance_gpu', 'diff']].to_string())
        
        # 可視化
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(importances_mean_cpu, importances_mean_gpu, alpha=0.6)
        plt.plot([importances_mean_cpu.min(), importances_mean_cpu.max()], 
                [importances_mean_cpu.min(), importances_mean_cpu.max()], 'r--', lw=2)
        plt.xlabel('CPU版 重要度')
        plt.ylabel('GPU版 重要度')
        plt.title(f'重要度の比較 (相関係数: {correlation:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(diff, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('差分 (CPU - GPU)')
        plt.ylabel('頻度')
        plt.title('重要度の差分の分布')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'result/cvpfi/comparison_{y_name}_{x_name}_{model_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"比較プロットを保存: {plot_path}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("検証完了")
    print("=" * 80)
    
    return {
        'importances_mean_cpu': importances_mean_cpu,
        'importances_mean_gpu': importances_mean_gpu,
        'importances_std_cpu': importances_std_cpu,
        'importances_std_gpu': importances_std_gpu,
        'correlation': correlation if len(importances_mean_cpu) == len(importances_mean_gpu) else None,
        'comparison_df': comparison_df if len(importances_mean_cpu) == len(importances_mean_gpu) else None
    }


if __name__ == '__main__':
    # 検証する設定
    y_name = 'Y(C2), %'
    x_name = 'x4_noLa'
    model_name = 'GPR_12'
    
    # 検証実行
    result = validate_cvpfi_results(y_name, x_name, model_name)
    
    if result is not None:
        print("\n検証結果サマリー:")
        if result['correlation'] is not None:
            print(f"  相関係数: {result['correlation']:.6f}")
            if result['correlation'] > 0.99:
                print("  ✓ 高い相関が確認されました（結果はほぼ一致）")
            elif result['correlation'] > 0.95:
                print("  ⚠ 中程度の相関（若干の差異あり）")
            else:
                print("  ✗ 低い相関（大きな差異あり）")

