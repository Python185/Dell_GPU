import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.linalg import det
import itertools

# 元素組成からプロセス条件までをD最適化基準で作成する
def d_optimality_criterion(X):
    """
    情報行列 X^T X の行列式を計算してD最適性を評価
    """
    return det(X.T @ X)

def fedorov_exchange(X, design_indices, max_iter=1000):
    """
    Fedorov交換法により、候補行列 X からD最適性が最大となる実験点群（インデックス）を探索する。
    
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
    current_design = X[design_indices, :]
    current_det = d_optimality_criterion(current_design)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(n_design):
            print(i + 1, "/", n_design)
            for j in candidate_indices:
                if j in design_indices:
                    continue
                new_design_indices = design_indices.copy()
                new_design_indices[i] = j
                new_design = X[new_design_indices, :]
                new_det = d_optimality_criterion(new_design)
                if new_det > current_det:
                    design_indices = new_design_indices
                    current_design = new_design
                    current_det = new_det
                    improved = True
                    break  # 内側ループ終了
            if improved:
                break  # 改善があれば再探索
    return design_indices, current_det

# SCC要望条件で元素種と組成を生成する関数
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

# x2_process_dataをD最適化基準で最適化する
#x2 = transform_x2(metal_data)
n_candidates = 10000  # 候補点数
n_design = 5  # 選ぶ実験点数（テスト用に縮小）
np.random.seed(42)

# ======= condition_dict の定義 =======
condition_dict = {
    '前処理還元炉温℃':[350, 400, 450],
    'temp':[250, 350, 380],
    'flow_base':[0.1, 1, 5, 10],
    'flow_slurry':[10, 20, 30],
    'flow_red':[50, 75, 100],
    'pressure':[25],
    'ターゲット担持量':[1, 3, 5, 10],
    #'support':['support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100'],
    'support':['support_SiO2_CARIACTg-10'],
    }   

constant_condition_dict = {
    '評価反応炉温℃':[260],    
    'wash':[1],
    'conc_ETA':[25],
    'conc_base':[2.5],
    '前担持':[1],
    }

# (1) その他条件：温度、圧力、触媒
#temperature_list = [300, 350, 400]
#pressure_list = [1.0, 1.5, 2.0]
#catalyst_list = ["A", "B", "C"]

# base 変数については、どれか一つが2.5、他は0となる
#base_names = ['base_NaOH', 'base_LiOH', 'base_KOH', 'base_Na2CO3', 'base_urea']
base_names = ['base_urea']
def sample_base():
    chosen = np.random.choice(base_names)
    return {bn: 2.5 if bn == chosen else 0 for bn in base_names}

# (3) 組成（x2）の生成：
# Li, K, Rb, Cs, Tiが消えていることに注意
metal_list = ['Rh','Fe','Mn','Mg','Al','Ca','Sc','V','Cr','Co','Ni','Cu','Zn','Ga','Se','Sr','Y','Zr','Mo','Ru','Pd','In',\
    'Sn','Te','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Yb','Lu','Hf','W','Ir','Pt','Au','Pb','Re']



# (4) 各候補点の条件をランダムサンプリングして辞書を作成
candidate_rows = []
# 試行回数を増やして十分な候補数を確保
for i in range(n_candidates * 2):
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
print(candidates.head())

# ---------------------------
# 前処理：各説明変数の変換
# ---------------------------
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

# 初期実験点数（例として10点）
#np.random.seed(42)
initial_design_indices = np.random.choice(np.arange(design_matrix.shape[0]), n_design, replace=False)
opt_indices, opt_det = fedorov_exchange(design_matrix, initial_design_indices.copy())
print("\n最適なD最適性基準（行列式）の値:", opt_det)
print("最適なデザインの候補インデックス:", opt_indices)

# ---------------------------
# 最適な実験条件の復元と組成の調整
# ---------------------------
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

# ⑤ 最終的な実験条件の結合
optimal_design = pd.concat([
    optimal_continuous_original,
    optimal_categorical,
    optimal_composition_adjusted,
    optimal_base
], axis=1)

# その他微調整等
optimal_design[valid_cols] = optimal_design[valid_cols].round(0)
for key, value in constant_condition_dict.items():
    optimal_design[key] = value * len(optimal_design)

optimal_design = optimal_design.loc[:, (optimal_design!=0).any(axis=0)]

# 実際に使用されているsupportの値でフィルタリング
# 現在は'support_SiO2_CARIACTg-10'のみが定義されている
optimal_design_sio2 = optimal_design[optimal_design['support'] == 'support_SiO2_CARIACTg-10']

# SiO2サポートのデータを保存（テスト実行のため一時的にコメントアウト）
# optimal_design_sio2.to_csv('result/virtual_data/candidates_sio2.csv', encoding='utf-8-sig')
print(f"最終的な実験条件数: {len(optimal_design_sio2)}")
print("最終結果のサンプル:")
print(optimal_design_sio2.head())

# 他のサポートが必要な場合は、condition_dictで定義してから使用する
# optimal_design_ce = optimal_design[optimal_design['support'] == 'support_CeO2_HS']
# optimal_design_ti = optimal_design[optimal_design['support'] == 'support_TiO2_SSP-M'] 
# optimal_design_zr = optimal_design[optimal_design['support'] == 'support_ZrO2_RC100']

print('End')