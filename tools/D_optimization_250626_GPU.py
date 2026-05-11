import pandas as pd
import numpy as np
from scipy.stats import zscore
import itertools
import os

# GPU計算用のライブラリ
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU (CuPy) が利用可能です")
except ImportError:
    print("⚠️ CuPy が見つかりません。CPUモードで実行します")
    print("GPU使用には: pip install cupy-cuda11x または pip install cupy-cuda12x")
    GPU_AVAILABLE = False
    import numpy as cp  # fallback to numpy

# GPU対応の行列式計算
def gpu_det(X):
    """GPU対応の行列式計算"""
    if GPU_AVAILABLE:
        return cp.linalg.det(X)
    else:
        # CPUフォールバック
        from scipy.linalg import det
        return det(X)

# GPU対応のD最適性評価
def d_optimality_criterion_gpu(X):
    """
    GPU対応：情報行列 X^T X の行列式を計算してD最適性を評価
    """
    if GPU_AVAILABLE:
        X_gpu = cp.asarray(X)
        info_matrix = cp.dot(X_gpu.T, X_gpu)
        return float(gpu_det(info_matrix))
    else:
        info_matrix = np.dot(X.T, X)
        return float(gpu_det(info_matrix))

def fedorov_exchange_gpu(X, design_indices, max_iter=1000):
    """
    GPU対応のFedorov交換法により、候補行列 X からD最適性が最大となる実験点群を探索
    
    Parameters:
        X: 標準化・エンコード済みの候補行列（各行が1点）
        design_indices: 初期デザイン点のインデックス（numpyの1次元配列）
        max_iter: 最大反復回数
        
    Returns:
        design_indices: D最適性が最大となる実験点群のインデックス
        current_det: 最適なデザインのD最適性（行列式の値）
    """
    n_design = len(design_indices)
    candidate_indices = np.arange(X.shape[0])
    
    # GPUにデータを転送
    if GPU_AVAILABLE:
        X_gpu = cp.asarray(X)
        print(f"GPU メモリ使用量: {cp.get_default_memory_pool().used_bytes() / 1024**2:.1f} MB")
    else:
        X_gpu = X
    
    current_design = X_gpu[design_indices, :]
    current_det = d_optimality_criterion_gpu(current_design)
    
    print(f"初期 D最適性: {current_det:.6e}")
    
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        print(f"反復 {iteration}/{max_iter}")
        
        for i in range(n_design):
            if i % 1 == 0:  # 進捗表示
                print(f"  デザイン点 {i + 1}/{n_design}")
            
            # 候補点をバッチで処理
            batch_size = 1000 if GPU_AVAILABLE else 100
            best_det = current_det
            best_j = None
            
            for batch_start in range(0, len(candidate_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(candidate_indices))
                batch_candidates = candidate_indices[batch_start:batch_end]
                
                # 現在のデザインインデックスに含まれていない候補のみを処理
                valid_candidates = [j for j in batch_candidates if j not in design_indices]
                
                if not valid_candidates:
                    continue
                
                # バッチ処理で複数の候補を同時に評価
                for j in valid_candidates:
                    new_design_indices = design_indices.copy()
                    new_design_indices[i] = j
                    new_design = X_gpu[new_design_indices, :]
                    new_det = d_optimality_criterion_gpu(new_design)
                    
                    if new_det > best_det:
                        best_det = new_det
                        best_j = j
            
            # 改善があった場合は更新
            if best_j is not None:
                design_indices[i] = best_j
                current_design = X_gpu[design_indices, :]
                current_det = best_det
                improved = True
                print(f"    改善: D最適性 = {current_det:.6e}")
                break  # 内側ループ終了
                
        if improved:
            print(f"反復 {iteration} で改善されました")
        else:
            print(f"反復 {iteration} で改善なし - 収束")
    
    return design_indices, current_det

# SCC要望条件で元素種と組成を生成する関数（CPUのまま）
def generate_sample(metal_list):
    # 1) 必須元素
    required_metals = ['Rh', 'Fe', 'Mn']
    # まず、必須元素 'Rh','Fe','Mn' とその他からランダムに2種を選択（初期候補5種）
    other_candidates = list(set(metal_list) - set(required_metals))
    selected_others = np.random.choice(other_candidates, size=2, replace=False).tolist()
    metals = ['Rh', 'Fe', 'Mn'] + selected_others  # 例：['Rh', 'Fe', 'Mn', 'Pt', 'Pd']
    # 0.01刻みの組成を、5元素で生成（np.random.multinomial で100点を分配）
    p = np.random.dirichlet(np.ones(5))
    counts = np.random.multinomial(100, p)  # 合計100の整数
    ratios = counts / 100.0  # 0.01刻みの比率
    # 条件チェック（必須条件）
    # 1. Rh (index0) の割合が 0.05～0.7
    if not (0.05 <= ratios[0] <= 0.7):
        return None
    # 2. Fe (index1) と Mn (index2) はそれぞれ 0～0.5
    if not (ratios[1] <= 0.5 and ratios[2] <= 0.5):
        return None
    # 3. Fe+Mn が 0.05～0.7（両方とも 0 でなければチェック）
    if not (0.05 <= (ratios[1] + ratios[2]) <= 0.7):
        return None                    

    # --- 追加処理: FeまたはMnが 0 の場合、補充する ---
    # Fe の index = 1, Mn の index = 2
    missing_indices = []
    if ratios[1] == 0:
        missing_indices.append(1)
    if ratios[2] == 0:
        missing_indices.append(2)
    
    if missing_indices:
        # 現在の候補から、該当インデックスの元素と比率を除外
        metals_keep = [m for i, m in enumerate(metals) if i not in missing_indices]
        ratios_keep = [r for i, r in enumerate(ratios) if i not in missing_indices]
        current_sum = sum(ratios_keep)
        # 補充すべき比率の合計
        #remaining_ratio = 1 - current_sum
        # 補充する元素の数
        num_missing = len(missing_indices)
        # 補充する各元素は最低 0.01 以上でなければならない
        #if remaining_ratio < 0.01 * num_missing:
        #    return None  # 補充不可能ならサンプル無効

        # remaining_ratio を 0.01刻みの整数値に変換
        #total_missing_counts = int(round(remaining_ratio * 100))
        # 各新元素に最低 1 を割り当て、残りをランダムに分割する
        base = [1] * num_missing
        #remaining_counts = total_missing_counts - num_missing
        #if remaining_counts > 0:
        #    # ランダムに分割（多項分布を利用）
        #    extra = np.random.multinomial(remaining_counts, [1/num_missing]*num_missing)
        #    added_counts = [base[i] + extra[i] for i in range(num_missing)]
        #else:
        added_counts = base
        added_ratios = [c / 100.0 for c in added_counts]
        # 追加する元素は、既に採用している元素以外から選ぶ
        available = list(set(metal_list) - set(metals_keep))
        if len(available) < num_missing:
            return None
        new_metals = np.random.choice(available, size=num_missing, replace=False).tolist()
        # 追加した元素の比率が 0.01 以上になる（すでに added_ratios は最低1/100 なのでOK）
        # 最終的な候補リストは、元々の metals_keep に新たな元素を追加して合計5元素にする
        metals = metals_keep + new_metals
        ratios = ratios_keep + added_ratios
        # ※順序は特に問わなければこのままでよいが、必要に応じて並び替え可能です

    # ここで、最終的な metals, ratios の長さは必ず 5 になっているはず
    if len(metals) != 5 or len(ratios) != 5:
        return None
    # また、各比率は 0.01 以上（補充した元素は確実に >=0.01）
    if any(r < 0.01 for r in ratios):
        return None

    # 4. Te制約: Teが含まれている場合、その組成は0.4以下でなければならない
    for i, metal in enumerate(metals):
        if metal == 'Te' and ratios[i] > 0.4:
            return None

    # DataFrame用の辞書を作成（列名：metal1～metal5, ratio1～ratio5）
    data = {metal: 0 for metal in sorted(metal_list)}
    for m, r in zip(metals, ratios):
        data[m] = r
        
    return data

# メイン実行部分
if __name__ == "__main__":
    print("GPU対応 D最適化実験設計を開始します...")
    
    # パラメータ設定（GPU使用時はより多くの候補点を扱える）
    if GPU_AVAILABLE:
        n_candidates = 1500   # GPU使用時は候補点数を増やす
        n_design = 100         # 選ぶ実験点数を設定
    else:
        n_candidates = 1000   # CPU使用時は候補点数を抑える
        n_design = 5
    
    np.random.seed(42)
    print(f"候補点数: {n_candidates}, 設計点数: {n_design}")

    # ======= condition_dict の定義 =======
    condition_dict = {
        '前処理還元炉温℃':[350, 400, 450],
        'temp':[250, 350, 380],
        'flow_base':[0.1, 1, 5, 10],
        'flow_slurry':[10, 20, 30],
        'flow_red':[50, 75, 100],
        'ターゲット担持量':[1, 3, 5, 10],
        'pressure':[25, 30, 35],
        'support':['support_Al2O3_AKPG15'],  #supportはあえてcondition_dictに残す
    }   

    constant_condition_dict = {
        '評価反応炉温℃':[260],    
        'wash':[1],
        'conc_ETA':[25],
        'conc_base':[2.5],
        '前担持':[1],
    }

    # base 変数については、どれか一つが2.5、他は0となる
    base_names = ['base_NaOH','base_KOH','base_LiOH','base_urea']
    def sample_base():
        chosen = np.random.choice(base_names)
        return {bn: 2.5 if bn == chosen else 0 for bn in base_names}

    # 組成（x2）の生成：
    metal_list = ['Rh','Fe','Mn','Mg','Al','Ca','Sc','V','Cr','Co','Ni','Cu','Zn','Ga','Se','Sr','Y','Zr','Mo','Ru','Pd','In',\
        'Sn','Te','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Yb','Lu','Hf','W','Ir','Pt','Au','Pb','Re']

    # 候補点の生成
    print("候補点を生成中...")
    candidate_rows = []
    # 試行回数を増やして十分な候補数を確保
    for i in range(n_candidates * 3):  # 余裕を持って3倍の試行回数
        if i % 1000 == 0:
            print(f"  候補点生成進捗: {i}/{n_candidates * 3}")
        
        row = {}
        # (flow_red + flow_base) / flow_slurry < 8.09 の条件を満たすもののみ採用
        threshold_of_flow = 8.09
        for key, values in condition_dict.items():
            row[key] = np.random.choice(values)
        if (row['flow_red'] + row['flow_base']) / row['flow_slurry'] >= threshold_of_flow:
            continue
        # base変数
        row.update(sample_base())
        # 組成（x2）：各元素の組成（0.01刻み、合計1）
        metal_compo = generate_sample(metal_list)
        if metal_compo is None:
            continue
        for key, value in metal_compo.items():
            row[key] = value
        candidate_rows.append(row)
        
        # 目標候補数に達したら終了
        if len(candidate_rows) >= n_candidates:
            break

    # 候補数チェックと警告表示
    if len(candidate_rows) < n_design:
        print(f"⚠️ 警告: 生成された候補数が不足しています ({len(candidate_rows)} < {n_design})")
        print("条件を緩和するか、試行回数を増やすことを検討してください")
    elif len(candidate_rows) < n_candidates:
        print(f"ℹ️ 情報: 目標候補数より少ないですが実行可能です ({len(candidate_rows)} < {n_candidates})")
    else:
        print(f"✅ 十分な候補数が生成されました ({len(candidate_rows)} >= {n_candidates})")

    # 候補点の DataFrame
    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.loc[:, (candidates!=0).any(axis=0)]
    print("サンプリングした候補点数:", len(candidates))

    # ---------------------------
    # 前処理：各説明変数の変換
    # ---------------------------
    print("データの前処理中...")
    
    # ① 連続変数の標準化
    continuous_cols = list(condition_dict.keys())
    continuous_cols = [c for c in continuous_cols if not c == 'support']
    continuous_data = candidates[continuous_cols].copy()
    cont_means = continuous_data.mean()   # 逆変換用
    cont_stds = continuous_data.std()       # 逆変換用

    # 分散が0に近い列を除外してから標準化
    valid_cols = cont_stds[cont_stds > 1e-8].index
    if len(valid_cols) < len(continuous_cols):
        print(f"警告: 分散が0に近い列を除外しました: {set(continuous_cols) - set(valid_cols)}")
    continuous_data_valid = continuous_data[valid_cols]
    continuous_scaled = continuous_data_valid.apply(zscore)

    # ② カテゴリ変数（触媒）：文字列化してから one-hot エンコーディング（dtype=int指定）
    categorical_cols = ['support']
    categorical_data = candidates[categorical_cols].astype(str).copy()
    categorical_encoded = pd.get_dummies(categorical_data, drop_first=False, dtype=int)

    # ③ base変数：そのまま使用
    base_cols = base_names

    # ④ 組成（x2）の標準化
    composition_cols = metal_list
    composition_data = candidates[composition_cols].copy()
    composition_means = composition_data.mean()  # 後のデスケーリング用
    composition_stds = composition_data.std()      # 後のデスケーリング用

    # 分散が0に近い組成列を除外してから標準化
    comp_valid_cols = composition_stds[composition_stds > 1e-8].index
    if len(comp_valid_cols) < len(composition_cols):
        print(f"警告: 分散が0に近い組成列を除外しました: {len(composition_cols) - len(comp_valid_cols)} 列")
    composition_data_valid = composition_data[comp_valid_cols]
    composition_scaled = composition_data_valid.apply(zscore)

    # ⑤ デザイン行列の構築：各グループを結合
    design_matrix = pd.concat([
        continuous_scaled,       # 連続変数（標準化済み）
        categorical_encoded,       # カテゴリ変数（ダミー変数）
        composition_scaled,        # 組成（標準化済み）
        candidates[base_cols]      # base変数（そのまま）
    ], axis=1).values

    print(f"デザイン行列のサイズ: {design_matrix.shape}")

    # GPU最適化の実行
    print("\nGPU最適化を開始します...")
    import time
    start_time = time.time()
    
    # 初期実験点数
    initial_design_indices = np.random.choice(np.arange(design_matrix.shape[0]), n_design, replace=False)
    opt_indices, opt_det = fedorov_exchange_gpu(design_matrix, initial_design_indices.copy())
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n✅ GPU最適化完了！")
    print(f"実行時間: {elapsed_time:.2f} 秒")
    print(f"最適なD最適性基準（行列式）の値: {opt_det:.6e}")
    print("最適なデザインの候補インデックス:", opt_indices)

    # ---------------------------
    # 最適な実験条件の復元と組成の調整
    # ---------------------------
    print("\n実験条件を復元中...")
    
    # ① 連続変数：逆標準化
    optimal_continuous_scaled = continuous_scaled.iloc[opt_indices]
    optimal_continuous_original = optimal_continuous_scaled * cont_stds[valid_cols] + cont_means[valid_cols]

    # ② カテゴリ変数、base変数はそのまま候補から抽出
    optimal_categorical = candidates[categorical_cols].iloc[opt_indices]
    optimal_base = candidates[base_cols].iloc[opt_indices]

    # ③ 組成：逆標準化して元のスケールに戻す
    optimal_composition_scaled = composition_scaled.iloc[opt_indices]
    optimal_composition_original = optimal_composition_scaled * composition_stds[comp_valid_cols] + composition_means[comp_valid_cols]

    # ④ 組成の丸めと調整（各行について0.01刻みに丸め、合計が1にならない場合は最も大きい値を調整）
    def adjust_composition(row):
        rounded = row.round(2)
        total = rounded.sum()
        if total < 1:
            idx = rounded.idxmax()
            rounded[idx] += 0.01
        elif total > 1:
            idx = rounded.idxmax()
            rounded[idx] -= 0.01
        return rounded

    optimal_composition_adjusted = optimal_composition_original.apply(adjust_composition, axis=1)

    # ⑤ 組成を metal1-5, ratio1-5 形式に変換
    def extract_top5_metals(row):
        """各行から組成の上位5つの元素を抽出してmetal1-5, ratio1-5形式に変換（アルファベット順）"""
        # 0でない組成のみを取得
        non_zero_metals = row[row > 0].sort_values(ascending=False)
        
        # 上位5つを取得（不足する場合は利用可能な分だけ）
        top5 = non_zero_metals.head(5)
        
        # 元素名と組成のペアを作成してアルファベット順にソート
        metal_ratio_pairs = [(metal, ratio) for metal, ratio in zip(top5.index, top5.values)]
        metal_ratio_pairs.sort(key=lambda x: x[0])  # 元素名でアルファベット順ソート
        
        result = {}
        # metal1-5を先に追加（アルファベット順）
        for i in range(5):
            if i < len(metal_ratio_pairs):
                result[f'metal{i+1}'] = metal_ratio_pairs[i][0]
            else:
                result[f'metal{i+1}'] = ''
        
        # ratio1-5を後に追加（対応する組成比）
        for i in range(5):
            if i < len(metal_ratio_pairs):
                result[f'ratio{i+1}'] = round(metal_ratio_pairs[i][1], 2)
            else:
                result[f'ratio{i+1}'] = 0.0
        
        return pd.Series(result)

    # 各行に対して上位5つの元素を抽出
    metal_ratio_df = optimal_composition_adjusted.apply(extract_top5_metals, axis=1)

    # ⑥ 最終的な実験条件の結合（metal1-5, ratio1-5は最後に配置）
    optimal_design_base = pd.concat([
        optimal_continuous_original,
        optimal_categorical,
        optimal_base
    ], axis=1)

    # その他微調整等
    optimal_design_base[valid_cols] = optimal_design_base[valid_cols].round(0)
    for key, value in constant_condition_dict.items():
        optimal_design_base[key] = value * len(optimal_design_base)

    # metal1-5, ratio1-5を最後に追加
    optimal_design = pd.concat([
        optimal_design_base,
        metal_ratio_df  # metal1-5, ratio1-5を最後に追加
    ], axis=1)

    # 結果の保存
    output_dir = os.path.join('result', 'inverse_analysis')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Al2O3_AKPG15.csv')
    optimal_design.to_csv(output_file, encoding='utf-8-sig', index=False)
    
    print(f"\n📊 最終結果:")
    print(f"実験条件数: {len(optimal_design)}")
    print(f"保存先: {output_file}")
    print("\n実験条件のサンプル:")
    print(optimal_design.head())
    
    # 組成部分を分けて表示
    print("\n📈 選択された元素と組成:")
    metal_columns = [f'metal{i}' for i in range(1, 6)]
    ratio_columns = [f'ratio{i}' for i in range(1, 6)]
    metal_ratio_columns = metal_columns + ratio_columns
    print(optimal_design[metal_ratio_columns].head())
    
    if GPU_AVAILABLE:
        # GPUメモリクリーンアップ
        cp.get_default_memory_pool().free_all_blocks()
        print("\n🧹 GPUメモリをクリーンアップしました")

    print('\n🎉 GPU最適化D最適実験設計が完了しました！')