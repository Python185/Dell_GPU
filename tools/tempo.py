import pandas as pd
import numpy as np
import matplotlib
import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from libs.boruta import BorutaPy
from itertools import product
import math

# 日本語フォント設定（Windows用）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'Hiragino Sans', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def Boruta_Apply2(target, df, perc, corr_threshold=0.97):
    y = df[target].copy()
    x = df.drop(target, axis= 1).copy()
    if '触媒ロット' in x.columns:
        x = x.drop('触媒ロット', axis= 1)
    
    # 同じ値を多く持つ変数を削除
    variables_to_delete = []
    for variable_name in x.columns:
        value_counts = x[variable_name].value_counts()
        if max(value_counts) >= x[variable_name].count() - 1:
            variables_to_delete.append(variable_name)
    
    # 相関係数の高い変数を削除
    corr_matrix = x.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_variables = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    
    # 削除対象変数をまとめる（重複を避ける）
    all_variables_to_delete = list(set(variables_to_delete + high_corr_variables))
    
    x.drop(columns = all_variables_to_delete, inplace = True)
    autoscaled_y = (y - y.mean(axis = 0)) / y.std(axis = 0, ddof = 1)
    autoscaled_x = (x - x.mean(axis = 0)) / x.std(axis = 0, ddof = 1)
    # RandomForestRegressorでBorutaを実行
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, perc= perc)
    feat_selector.fit(autoscaled_x.values, autoscaled_y.values)
    # 選択された特徴量を確認
    selected = feat_selector.support_
    print('選択された特徴量の数: %d' % np.sum(selected))
    print(x.columns[selected])
    #上で選択した説明変数のみを残す。
    selected_features= x.columns[selected]
    rf_selected = RandomForestRegressor(n_jobs=-1, max_depth= 5, random_state= 1)
    rf_selected.fit(autoscaled_x[selected_features], autoscaled_y.values)
    selected_importance = rf_selected.feature_importances_    
    df_b= df.loc[:, selected_features]
    df_b = pd.concat([df.loc[:, :'base_urea'], df_b], axis= 1)
    #df_b = pd.concat([df.loc[:, :'support_CeO2_HS'], df_b], axis= 1)
    df_b = df_b.loc[:, ~df_b.columns.duplicated()]
    df_b = df_b.loc[:, (df_b != 0).any(axis=0)]
    importance_df = pd.DataFrame(index= selected_features, data= selected_importance, columns= ['Importance'])
    importance_df = importance_df.sort_values(by= 'Importance', ascending= False)    
    
    return df_b, importance_df

# x2をmetal_xに変換する
def create_metalx(row):
    sorted_elements = row[row > 0].sort_values(ascending= False)
    metals = sorted_elements.index.tolist()
    ratios = sorted_elements.values.tolist()
    metals.extend([0.0] * (5 - len(metals)))
    ratios.extend([0.0] * (5 - len(ratios)))
    return pd.Series(metals + ratios, index= ['metal1','metal2','metal3','metal4','metal5','ratio1','ratio2','ratio3','ratio4','ratio5'])

# x2_41, x1_41を担体ごと、前担持、後担持ごとに分離する
def create_support_df(x2_41, x1_41):
    base_cols = [c for c in x2_41.columns if c.startswith('base_')]
    x2_41[base_cols] = x2_41[base_cols].replace({True: 1, False: 0})
    x1_41[base_cols] = x1_41[base_cols].replace({True: 1, False: 0})
    x2_m = x2_41[x2_41['前担持'] == 1].copy()
    x2_a = x2_41[x2_41['前担持'] == 0].copy()
    
    x2_m_ti = x2_m[(x2_m['support_TiO2_SSP-M'] == 1) | (x2_m['support_TiO2_TK805'] == 1)]
    x2_m_ti.insert(17, 'support_TiO2', 1)
    x2_m_ti = x2_m_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)
    x2_a_ti = x2_a[(x2_a['support_TiO2_SSP-M'] == 1) | (x2_a['support_TiO2_TK805'] == 1)]
    x2_a_ti.insert(17, 'support_TiO2', 1)
    x2_a_ti = x2_a_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)

    x2_m_ce = x2_m[x2_m['support_CeO2_HS'] == 1]
    x2_m_si = x2_m[x2_m['support_SiO2_CARIACTg-10'] == 1]
    x2_a_ce = x2_a[x2_a['support_CeO2_HS'] == 1]
    x2_a_si = x2_a[x2_a['support_SiO2_CARIACTg-10'] == 1]
    
    x2_m_al = x2_m[(x2_m['support_Al2O3_A-11'] == 1) | (x2_m['support_Al2O3_AKP-G15'] == 1) | (x2_m['support_Al2O3_KC501'] == 1)]
    x2_m_al.insert(17, 'support_Al2O3', 1)
    x2_m_al = x2_m_al.drop(['support_Al2O3_A-11', 'support_Al2O3_AKP-G15', 'support_Al2O3_KC501'], axis=1)
    x2_a_al = x2_a[(x2_a['support_Al2O3_A-11'] == 1) | (x2_a['support_Al2O3_AKP-G15'] == 1) | (x2_a['support_Al2O3_KC501'] == 1)]
    x2_a_al.insert(17, 'support_Al2O3', 1)
    x2_a_al = x2_a_al.drop(['support_Al2O3_A-11', 'support_Al2O3_AKP-G15', 'support_Al2O3_KC501'], axis=1)
    
    x1_ti = x1_41[(x1_41['support_TiO2_SSP-M'] == 1) | (x1_41['support_TiO2_TK805'] == 1)]
    x1_ti.insert(17, 'support_TiO2', 1)
    x1_ti = x1_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)
    x1_al = x1_41[(x1_41['support_Al2O3_A-11'] == 1) | (x1_41['support_Al2O3_AKP-G15'] == 1) | (x1_41['support_Al2O3_KC501'] == 1)]
    x1_al.insert(17, 'support_Al2O3', 1)
    x1_al = x1_al.drop(['support_Al2O3_A-11', 'support_Al2O3_AKP-G15', 'support_Al2O3_KC501'], axis=1)

    x1_m_ti = x1_ti[x1_ti.index.isin(x2_m_ti.index)].copy()
    x1_a_ti = x1_ti[x1_ti.index.isin(x2_a_ti.index)].copy()
    x1_m_ce = x1_41[x1_41.index.isin(x2_m_ce.index)].copy()
    x1_m_si = x1_41[x1_41.index.isin(x2_m_si.index)].copy()
    x1_a_ce = x1_41[x1_41.index.isin(x2_a_ce.index)].copy()
    x1_a_si = x1_41[x1_41.index.isin(x2_a_si.index)].copy()
    x1_m_al = x1_al[x1_al.index.isin(x2_m_al.index)].copy()
    x1_a_al = x1_al[x1_al.index.isin(x2_a_al.index)].copy()

    return x2_m_al, x2_m_ce, x2_m_ti, x2_m_si, x2_a_al, x2_a_ce, x2_a_ti, x2_a_si, x1_m_al, x1_m_ce, x1_m_ti, x1_m_si, x1_a_al, x1_a_ce, x1_a_ti, x1_a_si

# 重複データの処理と重複グループ番号の追加
def add_duplicate_group_numbers(df, support_name):
    """
    重複データに重複グループ番号を追加する関数
    
    Parameters:
    df: DataFrame - 重複データ
    support_name: str - 担体名（'Al2O3', 'CeO2', 'TiO2'）
    
    Returns:
    DataFrame - 重複グループ番号が追加されたデータ
    """
    print(f"\n=== {support_name}担体の重複グループ番号追加 ===")
    
    # ターゲット変数と触媒ロットを特定
    target_cols = ['選択率NPA', '収率NPA'] if all(col in df.columns for col in ['選択率NPA', '収率NPA']) else []
    catalyst_cols = ['触媒ロット'] if '触媒ロット' in df.columns else []
    
    # 特徴量列を特定（ターゲット変数と触媒ロット、support列を除く）
    feature_cols = [col for col in df.columns 
                   if col not in target_cols + catalyst_cols + ['support']]
    
    print(f"特徴量列数: {len(feature_cols)}")
    print(f"ターゲット列: {target_cols}")
    
    # 特徴量で重複グループを作成
    df_with_group = df.copy()
    
    # 特徴量の組み合わせでグループ化
    grouped = df_with_group.groupby(feature_cols)
    group_sizes = grouped.size()
    duplicate_groups = group_sizes[group_sizes > 1]
    
    print(f"重複グループ数: {len(duplicate_groups)}")
    print(f"重複が含まれる行数: {duplicate_groups.sum()}")
    
    # 重複グループ番号を追加
    group_number = 1
    duplicate_group_numbers = []
    
    for feature_values, group_df in grouped:
        group_size = len(group_df)
        if group_size > 1:
            # 重複グループの場合
            duplicate_group_numbers.extend([group_number] * group_size)
            print(f"グループ{group_number}: {group_size}個の重複データ")
            if len(target_cols) > 0:
                # ターゲット値の統計を表示
                for target_col in target_cols:
                    target_values = group_df[target_col]
                    print(f"  {target_col}: 平均={target_values.mean():.3f}, 範囲={target_values.min():.3f}-{target_values.max():.3f}")
            group_number += 1
        else:
            # 重複なしの場合は0を割り当て
            duplicate_group_numbers.append(0)
    
    # 重複グループ番号を列として追加
    df_with_group['duplicate_group'] = duplicate_group_numbers
    
    print(f"最大グループ番号: {max(duplicate_group_numbers)}")
    print(f"重複なしデータ数: {duplicate_group_numbers.count(0)}")
    
    return df_with_group

# 重複グループ番号をsupport列の次に移動
def move_duplicate_group_after_support(df):
    """重複グループ列をsupport列の次に移動"""
    if 'duplicate_group' in df.columns and 'support' in df.columns:
        # duplicate_group列を削除して取得
        duplicate_group_col = df.pop('duplicate_group')
        # support列の位置を取得
        support_index = df.columns.get_loc('support')
        # support列の次に挿入
        df.insert(support_index + 1, 'duplicate_group', duplicate_group_col)
    return df

def count_combinations(params, base_sum=None, base_prefix='base_'):
    # 値リストを抽出
    values = {k: list(v[1]) for k, v in params.items() if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] == 'list'}
    # 制約なしの全組合せ数
    lens = [len(v) for v in values.values()]
    total_no_constraint = math.prod(lens)

    # base系の制約適用
    base_keys = [k for k in values if k.startswith(base_prefix)]
    if not base_keys or base_sum is None:
        return total_no_constraint, None, None, None

    base_lists = [values[k] for k in base_keys]
    base_all = math.prod(len(lst) for lst in base_lists)

    valid_base = 0
    for combo in product(*base_lists):
        if abs(sum(combo) - base_sum) < 1e-9:
            valid_base += 1

    non_base_count = total_no_constraint // base_all if base_all else 0
    total_with_constraint = non_base_count * valid_base
    return total_no_constraint, base_all, valid_base, total_with_constraint



"""
# 逆解析データの内容確認
data_al = pd.read_csv('result/inverse_analysis/Al2O3_candidates100.csv', encoding='utf-8-sig', index_col=0, header=0)
data_si = pd.read_csv('result/inverse_analysis/SiO2_candidates100.csv', encoding='utf-8-sig', index_col=0, header=0)
data_al = data_al.loc[:, 'Al': 'Zr']
data_si = data_si.loc[:, 'Al': 'Zr']

# 元素種の使用状況を分析
print("=== Al2O3担体データの元素種分析 ===")
print(f"データ形状: {data_al.shape}")
print(f"使用されている元素種: {data_al.columns.tolist()}")
print(f"各元素の非ゼロ値の数:")
for element in data_al.columns:
    non_zero_count = (data_al[element] != 0).sum()
    print(f"  {element}: {non_zero_count}個")

print("\n=== SiO2担体データの元素種分析 ===")
print(f"データ形状: {data_si.shape}")
print(f"使用されている元素種: {data_si.columns.tolist()}")
print(f"各元素の非ゼロ値の数:")
for element in data_si.columns:
    non_zero_count = (data_si[element] != 0).sum()
    print(f"  {element}: {non_zero_count}個")

# ヒストグラムの作成
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Al2O3担体のヒストグラム
al_non_zero_counts = [(data_al[element] != 0).sum() for element in data_al.columns]
axes[0].bar(data_al.columns, al_non_zero_counts, color='skyblue', alpha=0.7)
axes[0].set_title('Al2O3担体データの各元素種の使用頻度', fontsize=18, fontweight='bold')
axes[0].set_xlabel('元素種', fontsize=16)
axes[0].set_ylabel('非ゼロ値の数', fontsize=16)
axes[0].tick_params(axis='x', rotation=45, labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].grid(True, alpha=0.3)

# SiO2担体のヒストグラム
si_non_zero_counts = [(data_si[element] != 0).sum() for element in data_si.columns]
axes[1].bar(data_si.columns, si_non_zero_counts, color='lightcoral', alpha=0.7)
axes[1].set_title('SiO2担体データの各元素種の使用頻度', fontsize=18, fontweight='bold')
axes[1].set_xlabel('元素種', fontsize=16)
axes[1].set_ylabel('非ゼロ値の数', fontsize=16)
axes[1].tick_params(axis='x', rotation=45, labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('result/inverse_analysis/element_usage_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

print('\nヒストグラムを保存しました: result/inverse_analysis/element_usage_histogram.png')
print('End')
"""

 
"""
# x4を作成
x4_base = pd.read_csv('datasets/v669/desc_for_boruta_all.csv', encoding='utf-8-sig', index_col=0, header=0)
x0 = pd.read_csv('datasets/v669/x0.csv', encoding='utf-8-sig', index_col=0, header=0)
target = ['選択率NPA', '収率NPA']
support_al = ['support_Al2O3_A-11', 'support_Al2O3_AKP-G15', 'support_Al2O3_KC501']
support_ti = ['support_TiO2_SSP-M', 'support_TiO2_TK805']

x4_base.insert(15, 'support_al', x4_base[support_al].sum(axis=1))
x4_base.insert(15, 'support_ti', x4_base[support_ti].sum(axis=1))
x4_base = x4_base.drop(support_al + support_ti, axis=1)

# x0とx4_baseの差を抽出
diff_df = x0[x0.index.isin(x4_base.index) == False]

x4_= Boruta_Apply2(target, x4_base, perc=80)[0]
x4_.to_csv('datasets/v669/x4.csv', encoding='utf-8-sig')
"""


# 前後担持、担体はカテゴリ変数、アルカリ元素は閾値により変数を与える
x1_41 = pd.read_csv('datasets/v669/x1_41.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_41 = pd.read_csv('datasets/v669/x2_41.csv', encoding='utf-8-sig', index_col=0, header=0)
original_data = pd.read_excel('file/★データセット_251003_v669.xlsx', sheet_name='データセット', index_col=0, header=0)
target = ['選択率NPA', '収率NPA']
support_al = ['support_Al2O3_A-11', 'support_Al2O3_AKP-G15', 'support_Al2O3_KC501']
support_ti = ['support_TiO2_SSP-M', 'support_TiO2_TK805']
alkali_metals = ['Li', 'Na', 'K', 'Rb', 'Cs']
alkali_threshold = 0.05
base_cols = [c for c in x1_41.columns if c.startswith('base_')]
x1_41[base_cols] = x1_41[base_cols].replace({True: 1, False: 0})
x2_41[base_cols] = x2_41[base_cols].replace({True: 1, False: 0})

x1_41.insert(15, 'support_Al2O3', x1_41[support_al].sum(axis=1))
x2_41.insert(15, 'support_Al2O3', x2_41[support_al].sum(axis=1))
x1_41.insert(16, 'support_TiO2', x1_41[support_ti].sum(axis=1))
x2_41.insert(16, 'support_TiO2', x2_41[support_ti].sum(axis=1))
x1_41 = x1_41.drop(support_al + support_ti, axis=1)
x2_41 = x2_41.drop(support_al + support_ti, axis=1)

original_data = original_data[original_data.index.isin(x2_41.index)]
x2_elements = x2_41.loc[:, 'Al':'Zr'].columns.tolist()
fx_elements = original_data.loc[:, 'Li': 'Bi'].columns.tolist()
x2_fx = x2_41.copy()
x2_process = x2_fx.drop(x2_elements, axis=1)
x2_fx = pd.concat([x2_process, original_data[fx_elements]], axis=1)
nan_check = x2_fx.isnull().sum()
x2_fx = x2_fx.dropna(axis=0, how='any')

# x2_fxの組成を0/1置換する
x2_fx_a = original_data[fx_elements].copy()
x2_fx_a = x2_fx_a.applymap(lambda x: 1 if x != 0 else 0)
x2_fx_a = pd.concat([x2_process, x2_fx_a], axis=1)

# アルカリ金属の処理 -1は過剰であることを意味する
x2_fx_b = x2_fx.copy()

# アルカリ金属以外の列を0/1変換
for col in x2_fx_b.loc[:, fx_elements].columns:
    if col not in alkali_metals:
        x2_fx_b[col] = x2_fx_b[col].apply(lambda x: 1 if x != 0 else 0)

# アルカリ金属列の3値ダミー変数化（0: なし, 1: 適量, -1: 過剰）
for metal in alkali_metals:
    if metal in x2_fx_b.columns:
        # 3つのダミー変数を作成
        x2_fx_b[f'{metal}_none'] = (x2_fx_b[metal] == 0).astype(int)  # なし
        x2_fx_b[f'{metal}_normal'] = ((x2_fx_b[metal] > 0) & (x2_fx_b[metal] < alkali_threshold)).astype(int)  # 適量
        x2_fx_b[f'{metal}_excess'] = (x2_fx_b[metal] >= alkali_threshold).astype(int)  # 過剰
        
        # 元の列を削除
        x2_fx_b = x2_fx_b.drop(columns=[metal])

x2_fx_b.to_csv('datasets/v669/x2_fx_b.csv', encoding='utf-8-sig')
x2_fx.to_csv('datasets/v669/x2_fx.csv', encoding='utf-8-sig')
x2_fx_a.to_csv('datasets/v669/x2_fx_a.csv', encoding='utf-8-sig')





"""
# x4にプロセス交差項を追加する
x1 = pd.read_csv('datasets/v669/x1_41.csv', encoding='utf-8-sig', index_col=0, header=0)
x2 = pd.read_csv('datasets/v669/x2_41.csv', encoding='utf-8-sig', index_col=0, header=0)
x4 = pd.read_csv('datasets/v669/desc_for_boruta_all.csv', encoding='utf-8-sig', index_col=0, header=0)
target = ['選択率NPA', '収率NPA']

x1_m = x1[x1['前担持'] == 1].copy()
x2_m = x2[x2['前担持'] == 1].copy()
x4_m = x4[x4['前担持'] == 1].copy()

# 担体はAl2O3とSiO2
support_al = ['support_Al2O3_A-11', 'support_Al2O3_AKP-G15', 'support_Al2O3_KC501']
x4_m_al = x4_m[x4_m[support_al].sum(axis=1) == 1].copy()
x4_m_si = x4_m[x4_m['support_SiO2_CARIACTg-10'] == 1].copy()

# プロセス変数と元素由来変数の交差項を作成
def create_interaction_terms(df, target_cols):
    
    #プロセス変数と元素由来変数の交差項を作成する関数
    
    df_work = df.copy()
    
    # プロセス変数（連続値のみ、カテゴリ変数と触媒ロットを除く）
    process_vars = ['前処理還元炉温℃', '評価反応炉温℃', 'temp', 'flow_base', 'flow_slurry', 
                   'flow_red', 'pressure', 'wash', '前担持', 'conc_ETA', 'ターゲット担持量wt%', 'conc_base']
    
    # 実際に存在するプロセス変数のみを選択
    process_vars = [var for var in process_vars if var in df_work.columns]
    
    # 元素由来変数（ave_, var_, gmean_, hmean_, max_, min_, maxcompo_で始まる変数）
    element_vars = [col for col in df_work.columns 
                   if any(col.startswith(prefix) for prefix in ['ave_', 'var_', 'gmean_', 'hmean_', 'max_', 'min_', 'maxcompo_'])]
    
    print(f"プロセス変数数: {len(process_vars)}")
    print(f"元素由来変数数: {len(element_vars)}")
    
    # 交差項を作成
    interaction_count = 0
    for process_var in process_vars:
        for element_var in element_vars:
            if process_var in df_work.columns and element_var in df_work.columns:
                interaction_name = f"{process_var}_x_{element_var}"
                df_work[interaction_name] = df_work[process_var] * df_work[element_var]
                interaction_count += 1
    
    print(f"作成された交差項数: {interaction_count}")
    print(f"元データ列数: {df.shape[1]}, 交差項追加後列数: {df_work.shape[1]}")
    
    return df_work

# x4_mに交差項を追加
x4_m_with_interactions = create_interaction_terms(x4_m_si, target)

# Boruta_Apply2で変数選択を実行
print("\n=== Boruta変数選択を実行中 ===")
x4_m_selected, importance_df = Boruta_Apply2(target, x4_m_with_interactions, perc=80)

print(f"\n選択された変数を含むデータ形状: {x4_m_selected.shape}")
print(f"選択された変数の重要度:")
print(importance_df.head(20))

# 結果を保存
x4_m_selected.to_csv('datasets/v669/x4_mx_si.csv', encoding='utf-8-sig')
importance_df.to_csv('datasets/v669/x4_mx_si_importance.csv', encoding='utf-8-sig')

print("\n結果ファイルを保存しました:")
print("- datasets/v669/x4_mx_si.csv")
print("- datasets/v669/x4_mx_si_importance.csv")
"""


"""
# 前担持、担体ごとにx1を作成する
x1_41 = pd.read_csv('datasets/v669/x1_41.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_41 = pd.read_csv('datasets/v669/x2_41.csv', encoding='utf-8-sig', index_col=0, header=0)
target = ['選択率NPA', '収率NPA']
support_al = ['support_Al2O3_A-11', 'support_Al2O3_AKP-G15', 'support_Al2O3_KC501']
support_ti = ['support_TiO2_SSP-M', 'support_TiO2_TK805']

x2_m_al, x2_m_ce, x2_m_ti, x2_m_si, x2_a_al, x2_a_ce, x2_a_ti, x2_a_si, x1_m_al, x1_m_ce, x1_m_ti, x1_m_si, x1_a_al, x1_a_ce, x1_a_ti, x1_a_si = create_support_df(x2_41, x1_41)
x2_m_al.to_csv('datasets/v669/x2_m_al.csv', encoding='utf-8-sig')
x2_m_ce.to_csv('datasets/v669/x2_m_ce.csv', encoding='utf-8-sig')
x2_m_ti.to_csv('datasets/v669/x2_m_ti.csv', encoding='utf-8-sig')
x2_m_si.to_csv('datasets/v669/x2_m_si.csv', encoding='utf-8-sig')
x2_a_al.to_csv('datasets/v669/x2_a_al.csv', encoding='utf-8-sig')
x2_a_ce.to_csv('datasets/v669/x2_a_ce.csv', encoding='utf-8-sig')
x2_a_ti.to_csv('datasets/v669/x2_a_ti.csv', encoding='utf-8-sig')
x2_a_si.to_csv('datasets/v669/x2_a_si.csv', encoding='utf-8-sig')

x1_m_al = Boruta_Apply2(target, x1_m_al, perc=80)[0]
x1_m_al.to_csv('datasets/v669/x1_m_al.csv', encoding='utf-8-sig')
x1_m_ce = Boruta_Apply2(target, x1_m_ce, perc=80)[0]
x1_m_ce.to_csv('datasets/v669/x1_m_ce.csv', encoding='utf-8-sig')
x1_m_ti = Boruta_Apply2(target, x1_m_ti, perc=80)[0]
x1_m_ti.to_csv('datasets/v669/x1_m_ti.csv', encoding='utf-8-sig')
x1_m_si = Boruta_Apply2(target, x1_m_si, perc=80)[0]
x1_m_si.to_csv('datasets/v669/x1_m_si.csv',encoding='utf-8-sig')
"""

"""
# データの縮小
al_data = pd.read_csv('result/inverse_analysis/Al2O3_pareto_opt.csv', encoding='utf-8-sig', index_col=0, header=0)
si_data = pd.read_csv('result/inverse_analysis/SiO2_pareto_opt.csv', encoding='utf-8-sig', index_col=0, header=0)
al_data_small = al_data.sample(n=100, random_state=18)
si_data_small = si_data.sample(n=100, random_state=18)
al_data_small = al_data_small.sort_values(by='one_metrics', ascending=False)
si_data_small = si_data_small.sort_values(by='one_metrics', ascending=False)
al_data_small.to_csv('result/inverse_analysis/Al2O3_candidates100.csv', encoding='utf-8-sig')
si_data_small.to_csv('result/inverse_analysis/SiO2_candidates100.csv', encoding='utf-8-sig')
"""

"""
# DFTデータの担体ごと分離、Boruta等
x2_7 = pd.read_csv('matlantis_descriptors/datasets/Akashi#2/x2_7.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_7_1 = pd.read_csv('matlantis_descriptors/datasets/Akashi#2/x2_7_1.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_7_2 = pd.read_csv('matlantis_descriptors/datasets/Akashi#2/x2_7_2.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_7_3 = pd.read_csv('matlantis_descriptors/datasets/Akashi#2/x2_7_3.csv', encoding='utf-8-sig', index_col=0, header=0)

x2_7_ti = x2_7[(x2_7['support_TiO2_SSP-M'] == 1) | (x2_7['support_TiO2_TK805'] == 1)].copy()
x2_7_ti.insert(17, 'support_TiO2', 1)
x2_7_ti = x2_7_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)
x2_7_ti.to_csv('matlantis_descriptors/datasets/Akashi#2/x2_7_ti.csv', encoding='utf-8-sig')

x2_7_1_ti = x2_7_1[(x2_7_1['support_TiO2_SSP-M'] == 1) | (x2_7_1['support_TiO2_TK805'] == 1)].copy()
x2_7_1_ti.insert(17, 'support_TiO2', 1)
x2_7_1_ti = x2_7_1_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)
x2_7_1_ti.to_csv('matlantis_descriptors/datasets/Akashi#2/x2_7_1_ti.csv', encoding='utf-8-sig')

x2_7_2_ti = x2_7_2[(x2_7_2['support_TiO2_SSP-M'] == 1) | (x2_7_2['support_TiO2_TK805'] == 1)].copy()
x2_7_2_ti.insert(17, 'support_TiO2', 1)
x2_7_2_ti = x2_7_2_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)
x2_7_2_ti.to_csv('matlantis_descriptors/datasets/Akashi#2/x2_7_2_ti.csv', encoding='utf-8-sig')

x2_7_3_ti = x2_7_3[(x2_7_3['support_TiO2_SSP-M'] == 1) | (x2_7_3['support_TiO2_TK805'] == 1)].copy()
x2_7_3_ti.insert(17, 'support_TiO2', 1)
x2_7_3_ti = x2_7_3_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)
x2_7_3_ti.to_csv('matlantis_descriptors/datasets/Akashi#2/x2_7_3_ti.csv', encoding='utf-8-sig')


x1_7 = pd.read_csv('matlantis_descriptors/datasets/Akashi#2/x1_7.csv', encoding='utf-8-sig', index_col=0, header=0)
x1_7_3 = pd.read_csv('matlantis_descriptors/datasets/Akashi#2/x1_7_3.csv', encoding='utf-8-sig', index_col=0, header=0)
target = ['選択率NPA', '収率NPA']

x1_7_ce = x1_7[x1_7['support_CeO2_HS'] == 1].copy()
x1_7_zr = x1_7[x1_7['support_ZrO2_RC100'] == 1].copy()
x1_7_ti = x1_7[(x1_7['support_TiO2_SSP-M'] == 1) | (x1_7['support_TiO2_TK805'] == 1)].copy()
x1_7_ti.insert(17, 'support_TiO2', 1)
x1_7_ti = x1_7_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)

# データ数不整合あるため、確認
check_df = pd.concat([x1_7_ce, x1_7_zr, x1_7_ti], axis=0)
diff_df = x1_7[~x1_7.index.isin(check_df.index)]

x1_7_3_ce = x1_7_3[x1_7_3['support_CeO2_HS'] == 1].copy()
x1_7_3_zr = x1_7_3[x1_7_3['support_ZrO2_RC100'] == 1].copy()
x1_7_3_ti = x1_7_3[(x1_7_3['support_TiO2_SSP-M'] == 1) | (x1_7_3['support_TiO2_TK805'] == 1)].copy()
x1_7_3_ti.insert(17, 'support_TiO2', 1)
x1_7_3_ti = x1_7_3_ti.drop(['support_TiO2_SSP-M', 'support_TiO2_TK805'], axis=1)

x1_7_ti = Boruta_Apply2(target, x1_7_ti, perc=70)[0]
x1_7_ti.to_csv('matlantis_descriptors/datasets/Akashi#2/x1_7_ti.csv', encoding='utf-8-sig')
x1_7_3_ti = Boruta_Apply2(target, x1_7_3_ti, perc=70)[0]
x1_7_3_ti.to_csv('matlantis_descriptors/datasets/Akashi#2/x1_7_3_ti.csv', encoding='utf-8-sig')
#x1_7_zr = Boruta_Apply2(target, x1_7_zr, perc=50)[0]
#x1_7_zr.to_csv('matlantis_descriptors/datasets/Akashi#2/x1_7_zr.csv', encoding='utf-8-sig')
#x1_7_3_zr = Boruta_Apply2(target, x1_7_3_zr, perc=50)[0]
#x1_7_3_zr.to_csv('matlantis_descriptors/datasets/Akashi#2/x1_7_3_zr.csv', encoding='utf-8-sig')
#x1_7_ce = Boruta_Apply2(target, x1_7_ce, perc=90)[0]
#x1_7_ce.to_csv('matlantis_descriptors/datasets/Akashi#2/x1_7_ce.csv', encoding='utf-8-sig')
#x1_7_3_ce = Boruta_Apply2(target, x1_7_3_ce, perc=90)[0]
#x1_7_3_ce.to_csv('matlantis_descriptors/datasets/Akashi#2/x1_7_3_ce.csv', encoding='utf-8-sig')
"""

print('End')