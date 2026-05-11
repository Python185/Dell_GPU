# GMR, VBGMR で逆解析を行い合成条件を提案するプログラム
import time
import pandas as pd
import numpy as np
import os
import shutil
from dcekit.generative_model import GMR
from dcekit.generative_model import VBGMR
import matplotlib.pyplot as plt
# import matplotlib.figure as figure
import random
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy.special import logit, expit
from deap import base
from deap import creator
from deap import tools
import warnings
from libs.libs_for_GA_v2.util_ga import GAutility
#from libs.Calculator_x26 import MetalFeaturizers
from libs.Calculator_x import Calc_desc
from libs.InverseAnalysisUtility import MIutility
#from pandasgui import show

warnings.simplefilter('ignore')  # warning の非表示
plt.rcParams["font.family"] = "MS Gothic"
ga_util = GAutility()

##### 設定 ここから #####
y_targets = [20, 5]  # y の目標値 [選択率NPAの目標値, 収率NPAの目標値]
y_names = ['選択率NPA', '収率NPA']
method = 'GMR'  # 'GMR' or 'VBGMR'

# 金属元素数：３の場合
#numbers_of_used_metals = [3]            # 金属種の候補
#numbers_of_metals = 3                   # GA:金属の数を選択
#Composition_upper_limit = 9             # 小数点以下 0.1 単位の場合
#Composition_upper_limit_div = 10        # 小数点以下 0.1 単位の場合
# 金属元素数：５の場合
numbers_of_used_metals = [2, 3, 4, 5]  # 金属種の候補
numbers_of_metals = 5                  # GA:金属の数を選択
Composition_upper_limit = 99           # 小数点以下 0.01 単位の場合
Composition_upper_limit_div = 100      # 小数点以下 0.01 単位の場合

Composition_round = 2  # 金属の構成割合は0.01刻み、小数点2桁
# Composition_round = 1  # 金属の構成割合は0.1刻み、小数点1桁

number_of_iterations = 30  # 提案する合成条件の数
number_of_population = 100  # GA の染色体数
number_of_generation = 100  # GA の世代数
convergence_generation = 10   # 収束の世代数(n世代にわたりbest_fitの値(小数点以下5桁)が変化しない場合break)

#number_of_iterations = 30  # 提案する合成条件の数
#number_of_population = 100  # GA の染色体数
#number_of_generation = 100  # GA の世代数
#convergence_generation = 10   # 収束の世代数

hyperparameter_optimization = 'cv_bo'  
# gridserch + cv ⇒ cv_opt
# cv + Bayesian optimization ⇒ cv_bo
# 評価条件・合成条件の設定
metal_ratio_list = ['metal1', 'ratio1', 'metal2', 'ratio2',
                    'metal3', 'ratio3', 'metal4', 'ratio4', 'metal5', 'ratio5']
metal_ratio_list_2 = ['metal1', 'metal2', 'metal3', 'metal4', 'metal5',
                      'ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5']
metal_ratio_list_3 = ['metal1', 'metal2', 'metal3',
                        'ratio1', 'ratio2', 'ratio3']
supports_list = ['support_ZrO2_RC100']
#supports_list = ['support_Al2O3_A-11', 'support_CeO2_HS',
#                 'support_TiO2_SSP-M', 'support_ZrO2_RC100']

inverse_analysis_condition_dict = {
    '前処理還元炉温℃': ['list', [350, 400, 450]],
    '評価反応炉温℃': ['list', [260]],
    'temp': ['list', [250, 350, 380]],
    'flow_base': ['list', [0.1, 1, 5, 10]],      # x2_dft1_b9で全て同じ値のためlistから外す
    'flow_slurry': ['list', [10, 20, 30]],
    'flow_red': ['list', [50, 75, 100]],
    'wash':['list', [1]],
    'pressure':['list', [25, 30, 35]],
    'conc_ETA':['list', [25]],
    'base_NaOH':['list', [0, 2.5]],
    'base_LiOH':['list', [0, 2.5]],
    'base_KOH':['list', [0, 2.5]],
    #'base_Na2CO3':['list', [0, 2.5]],  # baseの合計=2.5の制約必要
    'base_urea':['list', [0, 2.5]],
    'ターゲット担持量':['list', [1, 3, 5, 10]],
    '前担持': ['list', [1]],
    'support': ['list', supports_list],
}

# 追加の制約用
#(flow_red + flow_base) / flow_slurry >= 8.09 のサンプルを削除

probability_of_crossover = 0.5
probability_of_mutation = 0.6
fold_number = 5  #DCVのfold数に合わせる
covariance_types = ['full', 'diag', 'tied', 'spherical']  # 分散共分散行列
# numbers_of_components = np.arange(1, 11, 1) # 正規分布の数
numbers_of_components = np.arange(2, 22, 2)  # 正規分布の数
weight_concentration_prior_types = [
    'dirichlet_process', 'dirichlet_distribution']  # 正規分布の重みの事前分布
# 正規分布の重みの事前分布のパラメータ
weight_concentration_priors = 10 ** np.arange(-4, 2, 2, dtype=float)

# 相関係数の設定
threshold_of_r = 0.95
##### 設定 ここまで #####

def autoscaling(data, base_data):
    # data と base_data を数値型に変換
    data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce')
    base_data = pd.DataFrame(base_data).apply(pd.to_numeric, errors='coerce')
    # 欠損値が発生した場合にエラーメッセージを出す
    if data.isnull().values.any():
        raise ValueError("data contains non-numeric values that couldn't be converted to floats.")
    if base_data.isnull().values.any():
        raise ValueError("base_data contains non-numeric values that couldn't be converted to floats.")
    # autoscaling実行
    autoscaled_data = (data - base_data.mean(axis=0)) / \
        base_data.std(axis=0, ddof=1)
    return autoscaled_data


def rescaling(data, base_data):
    rescaled_data = data * \
        base_data.std(axis=0, ddof=1) + base_data.mean(axis=0)
    return rescaled_data


def log_transform(arg_data):
    data = arg_data.copy()
    data[data == 0] = (data[data != 0].nsmallest(1) / 2).iloc[0]
    data = logit((data / 100).values)
    return data


def log_inverse_transform(data):
    data = expit(data) * 100
    return data

# dfの行名を分解して列名の方に加える
def rename_index_col(df):
    # 行名（インデックス）を分割して新しいデータフレームを作成
    df.index = df.index.str.split('_', n=1, expand=True)
    # 列名を新しい形式に変換
    new_columns = []
    for col in df.columns:
        for idx in df.index.levels[1]:
            new_columns.append(f'{col}_{idx}')
    # 新しいデータフレームを作成
    df_new = pd.DataFrame(index=df.index.levels[0], columns=new_columns)
    # データを新しいデータフレームに移動
    for element, group in df.groupby(level=0):
        if element == 'Rh':
            # 元素がRhの場合の特別処理
            for col in df.columns:
                for idx in group.index.get_level_values(1):
                    new_col_name = f'{col}_{idx}'
                    df_new.at[element, new_col_name] = group.loc[(element, idx), col]
        else:
            # 通常の処理
            for col in df.columns:
                for idx in df.index.levels[1]:
                    new_col_name = f'{col}_{idx}'
                    df_new.at[element, new_col_name] = group.loc[element, idx][col]
    return df_new

def make_matlantis_desc(individual_df, inverse_df):
    # データの読込み（ファイルが存在しない場合はエラーハンドリング）
    try:
        E_CO = pd.read_csv('matlantis_descriptors/#2/20240830_E_ads_CO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        E_ETA = pd.read_csv('matlantis_descriptors/#2/20240830_E_ads_ETA_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        E_HCO = pd.read_csv('matlantis_descriptors/#2/20240830_E_ads_HCO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        E_EtCO = pd.read_csv('matlantis_descriptors/#2/20240830_E_ads_EtCO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        NEB_HCO = pd.read_csv('matlantis_descriptors/#2/20240830_NEB_HCO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        NEB_CO = pd.read_csv('matlantis_descriptors/#2/20240830_NEB_EtCO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
    except FileNotFoundError as e:
        print(f"警告: Matlantis記述子ファイルが見つかりません: {e}")
        print("記述子なしで処理を続行します")
        return inverse_df
    matlantis_metals = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ir', 'Pt', 'Au', 'V', 'Mo', 'W']
    # 読込みデータの整形
    E_CO = E_CO.drop('Doped_metal', axis= 1)
    E_ETA = E_ETA.drop('Doped_metal', axis= 1)
    E_HCO = E_HCO.drop('Doped_metal', axis= 1)
    E_EtCO = E_EtCO.drop('Doped_metal', axis= 1)
    NEB_CO = NEB_CO.drop('Doped_metal', axis= 1)
    NEB_HCO = NEB_HCO.drop('Doped_metal', axis= 1)
    # 各ファイル内容の展開
    E_ads_CO = rename_index_col(E_CO)
    E_ads_ETA = rename_index_col(E_ETA)
    E_ads_HCO = rename_index_col(E_HCO)
    E_ads_EtCO = rename_index_col(E_EtCO)
    NEB_CO_ = rename_index_col(NEB_CO)
    NEB_HCO_ = rename_index_col(NEB_HCO)
    mtlnts_desc = pd.concat([E_ads_CO, E_ads_ETA, E_ads_HCO, E_ads_EtCO, NEB_CO_, NEB_HCO_], axis= 1)
    # matlantis_descの追加
    individual_df = individual_df.loc[:, 'metal1':]
    # インデックスを列に変換してから melt
    individual_df_reset = individual_df.reset_index()
    metal_melted = pd.melt(individual_df_reset, id_vars=['index'], value_vars=['metal1', 'metal2', 'metal3', 'metal4', 'metal5'], 
                        var_name='metal_num', value_name='element')
    ratio_melted = pd.melt(individual_df_reset, id_vars=['index'], value_vars=['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5'], 
                        var_name='ratio_num', value_name='composition')
    # 列番号（1-5）を対応させる
    metal_melted['ratio_num'] = metal_melted['metal_num'].str.extract('(\d+)')
    ratio_melted['ratio_num'] = ratio_melted['ratio_num'].str.extract('(\d+)')
    # 合体して元素と組成を結びつける
    df_combined = pd.merge(metal_melted, ratio_melted, on=['index', 'ratio_num']).drop(columns=['metal_num', 'ratio_num'])
    # 元素名でpivotして新しいdf_bを作成
    x2 = df_combined.pivot(index='index', columns='element', values='composition')
    # インデックスを戻す
    x2.index.name = None  # index列の名前を元に戻す
    x2 = x2.replace(np.nan, 0)

    for index, row in x2.iterrows():
        # 元素列のみを対象とする（2文字以下の列名）
        element_columns = [col for col in x2.columns if isinstance(col, str) and len(col) <= 2 and col != '']
        el_ratio_list = [[el, row[el]] for el in element_columns if el in row.index and row[el] != 0]
        el_ratio_list = [el for el in el_ratio_list if el[0] in matlantis_metals]
        el_ratio_dic = {el[0]: el[1] for el in el_ratio_list}
        if len(el_ratio_dic) >= 1:
            for prop in mtlnts_desc.columns:
                prop_value = mtlnts_desc.loc[el_ratio_dic.keys(), prop].values
                weight_values = np.array(list(el_ratio_dic.values())).astype(float)
                weighted_average = np.average(prop_value,weights=weight_values)
                x2.at[index, f'{prop}_weighted'] = weighted_average
        elif row['Rh'] != 0:
            for prop in mtlnts_desc.columns:
                x2.at[index, f'{prop}_weighted'] = mtlnts_desc.loc['Rh', prop]
        else:
            print('Error! '+index+'にはRh, ドープ元素ともありません')
            break
        
    descriptors = inverse_df.loc[:, 'Fe':].columns.tolist()
    existing_col = [c for c in descriptors if c in x2.columns]
    missing_col = [c for c in descriptors if c not in x2.columns]
    inverse_df[existing_col] = x2[existing_col]    
    inverse_df[missing_col] = 0
    
    return inverse_df    
    
def make_dft_desc(individual_df, inverse_df):
    try:
        dft_data = pd.read_csv('matlantis_descriptors/DFT_DATA_single_impurity_onRh(Akashi20240910) .csv', encoding= 'utf-8-sig', index_col= 1, header= 0)
    except FileNotFoundError as e:
        print(f"警告: DFT記述子ファイルが見つかりません: {e}")
        print("記述子なしで処理を続行します")
        return inverse_df
    # Matlantisにて計算した対象金属
    dft_metals = dft_data.index.tolist()
    dft_metals = [s for s in dft_metals if s not in ['Ag','Os']]
    dft_data = dft_data.dropna(how= 'all', axis= 0)
    dft_data = dft_data.drop('Unnamed: 0', axis= 1)

    # dft_descの追加
    individual_df = individual_df.loc[:, 'metal1':]
    # インデックスを列に変換してから melt
    individual_df_reset = individual_df.reset_index()
    metal_melted = pd.melt(individual_df_reset, id_vars=['index'], value_vars=['metal1', 'metal2', 'metal3', 'metal4', 'metal5'], 
                        var_name='metal_num', value_name='element')
    ratio_melted = pd.melt(individual_df_reset, id_vars=['index'], value_vars=['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5'], 
                        var_name='ratio_num', value_name='composition')
    # 列番号（1-5）を対応させる
    metal_melted['ratio_num'] = metal_melted['metal_num'].str.extract('(\d+)')
    ratio_melted['ratio_num'] = ratio_melted['ratio_num'].str.extract('(\d+)')
    # 合体して元素と組成を結びつける
    df_combined = pd.merge(metal_melted, ratio_melted, on=['index', 'ratio_num']).drop(columns=['metal_num', 'ratio_num'])
    # 元素名でpivotして新しいdf_bを作成
    x2 = df_combined.pivot(index='index', columns='element', values='composition')
    # インデックスを戻す
    x2.index.name = None  # index列の名前を元に戻す
    x2 = x2.replace(np.nan, 0)
    
    for index, row in x2.iterrows():
        metal_col = [c for c in x2.columns if isinstance(c, str) and len(c) <= 2 and c != '']
        el_ratio_list = [[el, row[el]] for el in metal_col if el in row.index and row[el] != 0]
        el_ratio_list = [el for el in el_ratio_list if el[0] in dft_metals]
        el_ratio_dic = {el[0]: el[1] for el in el_ratio_list}
        if len(el_ratio_dic) >= 1:
            for prop in dft_data.columns:
                prop_value = dft_data.loc[el_ratio_dic.keys(), prop].values
                weight_values = np.array(list(el_ratio_dic.values())).astype(float)
                weighted_average = np.average(prop_value,weights=weight_values)
                x2.at[index, f'{prop}_weighted'] = weighted_average
        elif row['Rh'] != 0:
            for prop in dft_data.columns:
                x2.at[index, f'{prop}_weighted'] = dft_data.loc['Rh', prop]
        else:
            print('Error! '+index+'にはRh, ドープ元素ともありません')
            break    

    descriptors = inverse_df.loc[:, 'Cr':].columns.tolist()
    existing_col = [c for c in descriptors if c in x2.columns]
    missing_col = [c for c in descriptors if c not in x2.columns]
    inverse_df[existing_col] = x2[existing_col]    
    inverse_df[missing_col] = 0
    
    return inverse_df

def make_x1_ave_desc(individual_df, inverse_df):
    #x1の作成
    metal_x = individual_df.loc[:, 'metal1':]
    metal_cols = [f'metal{j}' for j in range(1, 6)]
    ratio_cols = [f'ratio{j}' for j in range(1, 6)]    
    metal_x_col = metal_cols + ratio_cols
    metal_x = metal_x[metal_x_col]
    xenonpy_merge = pd.read_csv('results/xenonpy_element_data240515.csv', index_col=0)

    # x1_metaldesc の初期化（列名は 'ave_' + 元の列名）
    x1_metaldesc_columns = [f'ave_{col}' for col in xenonpy_merge.columns]
    x1_metaldesc = pd.DataFrame(index=metal_x.index, columns=x1_metaldesc_columns)
    
    # 各行について処理
    for idx, row in metal_x.iterrows():
        # 金属と比率を取得
        metals = row[metal_cols]
        ratios = row[ratio_cols].astype(float)
        # 有効な（金属名が欠損していない）データを抽出
        valid_m = metals.notna()
        valid_r = ratios.notna()
        metals_valid = metals[valid_m].values
        ratios_valid = ratios[valid_r].values
        
        if len(metals_valid) == 0:
            raise Exception('有効なmetalが1つもありません。')
        # 金属の記述子を取得
        try:
            mt = xenonpy_merge.loc[metals_valid].values
        except KeyError:
            # xenonpy_merge に金属が存在しない場合
            x1_metaldesc.loc[idx] = np.nan
            continue
        # 重み付き平均の計算
        weighted_avg = np.dot(ratios_valid, mt) / ratios_valid.sum()
        # 結果を x1_metaldesc に代入
        x1_metaldesc.loc[idx] = weighted_avg

    x1_metaldesc = x1_metaldesc.replace([np.inf, -np.inf], np.nan)
    #ここでnanが生まれる可能性あり！！
    x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]   #全てnanの列を削除
    
    descriptors = inverse_df.loc[:, 'ave_atomic_number':].columns.tolist()
    existing_col = [c for c in descriptors if c in x1_metaldesc.columns]
    missing_col = [c for c in descriptors if c not in x1_metaldesc.columns]
    inverse_df[existing_col] = x1_metaldesc[existing_col]    
    inverse_df[missing_col] = 0
    
    return inverse_df

def evaluate(x_name, ssc_condition, individual, individual_metal, variables, condition_dict, xenonpy_element_data,
             autoscaled_estimated_means, autoscaled_estimated_covariances, weights):
    individual_df = pd.DataFrame(
        np.concatenate(
            [np.array(individual), np.array(individual_metal)], axis=1),
        columns=list(inverse_analysis_condition_dict.keys()) + metal_ratio_list[:len(individual_metal[0])])

    # 逆解析用データの作成
    individual_inverse_analysis_x_df = \
        pd.DataFrame(np.zeros(
            (individual_df.shape[0], variables.shape[1])), columns=variables.columns)
    # 遺伝子をxに変換
    rtn_message, individual_inverse_analysis_x_df = \
        ind_to_xdata(x_name, ssc_condition, individual_df, individual_inverse_analysis_x_df,
                     condition_dict, xenonpy_element_data, return_metal=False)
    individual_inverse_analysis_x_df = \
        individual_inverse_analysis_x_df.loc[:, variables.columns]
    autoscaled_individual_inverse_analysis_x_df = autoscaling(individual_inverse_analysis_x_df, variables)
    autoscaled_individual_inverse_analysis_x_df.drop(columns=y_names, inplace=True)
    log_probality = []
    tmps = []
    try:
        for i in range(autoscaled_estimated_covariances.shape[0]):
            tmp = np.log(weights[i, 0]) + \
                multivariate_normal.logpdf(autoscaled_individual_inverse_analysis_x_df.values,
                                           mean=autoscaled_estimated_means[i, 0, :],
                                           cov=autoscaled_estimated_covariances[i, :, :])
            tmps.append(tmp.tolist())
        tmps = np.array(tmps)
        for i in range(len(individual)):
            log_probality.append((logsumexp(tmps[:, i]),))
        log_probality = [(-10 ** 100,) if 'invalid' in rtn_message[i] else log_probality[i] for i in
                         range(len(individual))]
    except:
        return [(-10 ** 100,) for _ in range(len(individual))]
    return log_probality


def ind_to_xdata(x_name, ssc_condition, individual_df, inverse_df, condition_dict, xenonpy_element_data, return_metal=False):
    rtn_message = ['' for i in range(len(individual_df))]
    metal_list = [[] for i in range(len(individual_df))]
    metal_component_list = [[] for i in range(len(individual_df))]
    for idx in range(len(individual_df)):
        rtn_message[idx] = 'success'
        for i, col in enumerate(list(condition_dict.keys())):
            # list の場合はリストのインデックスを設定
            idx_list = int(
                int(individual_df.loc[idx, col]) / (100 / len(condition_dict[col][1])))
            ind_list_selection = condition_dict[col][1][idx_list]
            if col == 'support':
                if ind_list_selection in inverse_df.columns:
                    inverse_df.loc[idx, ind_list_selection] = 1
            else:
                if col in inverse_df.columns:
                    inverse_df.loc[idx, col] = ind_list_selection
        metal_ratio_list = individual_df.columns.to_list()[len(condition_dict):]
        for metal_name, metal_ratio in zip(metal_ratio_list[0::2], metal_ratio_list[1::2]):
            # x2_41の時に、inverse_dfに元素を追加
            if x_name == 'x2_41':
                inverse_df.loc[idx, individual_df.loc[idx, metal_name]] = float(individual_df.loc[idx, metal_ratio])
            metal_list[idx].append(individual_df.loc[idx, metal_name])
            metal_component_list[idx].append(float(individual_df.loc[idx, metal_ratio]))
        if 'flow_red' in inverse_df.columns and 'flow_base' in inverse_df.columns and 'flow_slurry' in inverse_df.columns:
            if (inverse_df.loc[idx, 'flow_red'] + inverse_df.loc[idx, 'flow_base']) / \
                    inverse_df.loc[idx, 'flow_slurry'] >= 8.09:
                # (flow_red + flow_base) / flow_slurry >= 8.09　は除外
                rtn_message[idx] = 'invalid'
                #rtn_message[idx] = 'invalid condition'
        # 金属組成の正規化
        metal_component_list[idx] = [round(x / sum(metal_component_list[idx]), Composition_round)
                                     for x in metal_component_list[idx]]
            
    # 統計量の計算
    if x_name == 'x1':
        xenonpy_col_name = ga_util.get_xenonpyList(xenonpy_element_data)
        x_metaldesc = pd.DataFrame(columns=xenonpy_col_name)
        for idx in range(len(individual_df)):
            # xenonpy記述子の計算
            metal_to_xenonpy_array = np.zeros((len(metal_list[idx]), xenonpy_element_data.shape[1]))
            for i, metal in enumerate(metal_list[idx]):
                # 金属種データの取得
                metal_to_xenonpy_array[i, :] = xenonpy_element_data.loc[metal, :].values
            metal_component_array = np.array(metal_component_list[idx])
            #x_metaldesc = pd.DataFrame(columns=xenonpy_col_name)
            for desc in range(xenonpy_element_data.shape[1]):
                d_name = xenonpy_element_data.columns[desc]

                if np.isnan(metal_to_xenonpy_array[:, desc]).any():
                    x_metaldesc.loc[i, f'ave_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'var_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'gmean_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'hmean_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'max_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'min_{d_name}'] = np.nan
                    continue
                x_metaldesc_array = np.array([
                    np.dot(metal_to_xenonpy_array[:, desc],metal_component_array) / np.sum(metal_component_array),
                    np.dot((metal_to_xenonpy_array[:, desc] - np.average(metal_to_xenonpy_array[:, desc]))**2 , metal_component_array),
                    np.prod(metal_to_xenonpy_array[:, desc]**metal_component_array)**(1/sum(metal_component_array)),
                    sum(metal_component_array)/sum((1/metal_to_xenonpy_array[:, desc])*metal_component_array),
                    max(metal_to_xenonpy_array[:, desc]),
                    min(metal_to_xenonpy_array[:, desc])
                    ])
                x_metaldesc_array = pd.Series(x_metaldesc_array, index=[f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}', f'min_{d_name}'])
                x_metaldesc.loc[i, [f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}', f'min_{d_name}']] = x_metaldesc_array
            x_metaldesc = x_metaldesc.replace([np.inf, -np.inf], np.nan)
            # inverse_df[x_metaldesc.columns] = x_metaldesc.values
            last_support = [i for i in inverse_df.loc[idx:idx+1, :].columns.tolist() if 'support_' in i][-1]
            idx_of_first_metaldesc = inverse_df.loc[idx:idx+1, :].columns.tolist().index(last_support) + 1
            inverse_df.loc[idx, inverse_df.loc[idx:idx+1, :].columns[idx_of_first_metaldesc:]] = \
                x_metaldesc[inverse_df.loc[idx:idx+1, :].columns[idx_of_first_metaldesc:]].values.astype(float)

    elif x_name == 'x2y_41':
        # 元素は上部で付加
        featurizers2 = Calc_desc()
        inverse_df, metal_list, metal_component_list = featurizers2.get_x2_forGA(individual_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition)
    
    elif x_name in ['x1_ce_am','x1_ti_am','x1_zr_am']:
        inverse_df = make_x1_ave_desc(individual_df, inverse_df)

    elif x_name == 'xx2_mat1_b9':
        inverse_df = make_matlantis_desc(individual_df, inverse_df)
        
    elif x_name == 'x2_dft1_b9':
        inverse_df = make_dft_desc(individual_df, inverse_df)

    else:
        raise Exception(f"{x_name}に対するGA逆解析が未実装です")
            
    if return_metal:
        return rtn_message, inverse_df, metal_list, metal_component_list
    else:
        return rtn_message, inverse_df

def Tournament(individuals, k, pop_metal, tournsize):
    chosen = []
    chosen_metal = []
    for i in range(k):
        choice = [random.randint(0, k - 1) for _ in range(tournsize)]
        aspirants = [individuals[i] for i in choice]
        aspirant_metals = [pop_metal[i] for i in choice]
        fitness_list = [item.fitness for item in aspirants]
        max_idx = fitness_list.index(max(fitness_list))
        chosen.append(aspirants[max_idx])
        chosen_metal.append(aspirant_metals[max_idx])
    return chosen, chosen_metal


def create_pop_metal(num_of_population, metal_list, num_metal, Ir_Rh_none_flag='0'):
    """
    metal_name, metal_ratioの作成
    """
    #  使用元素リストnum_of_metal個
    #  組成比が合計100%となるように選出（表示は0~0.99)
    inverse_metal_x_data = []
    if Ir_Rh_none_flag == '4':
        while len(inverse_metal_x_data) < num_of_population / 2:
            metal_candidate = random.sample(metal_list, num_metal)
            if len(metal_candidate) == len(set(metal_candidate)):
                # 各組成の計算 : 組成は、下限0.01、上限0.99、間隔0.01　で作成する
                ratio_x = [random.randint(1, Composition_upper_limit) for _ in range(num_metal)]
                metal_ratio = [round(i / sum(ratio_x), Composition_round) for i in ratio_x]
                while (0.0 in metal_ratio):
                    ratio_x = [random.randint(1, Composition_upper_limit) for _ in range(num_metal)]
                    metal_ratio = [round(i / sum(ratio_x), Composition_round) for i in ratio_x]
                metal_ratio_pair = []
                for a, b in zip(metal_candidate, metal_ratio):
                    metal_ratio_pair = metal_ratio_pair + [a, b]
                inverse_metal_x_data.append(metal_ratio_pair)  
        while len(inverse_metal_x_data) < num_of_population:
            metal_candidate = random.sample([m for m in metal_list if m != 'Rh'], num_metal - 1)
            metal_candidate.append('Rh')
            # Rhの比率を0.1から0.7の間でランダムに設定
            rh_ratio = random.uniform(0.1, 0.7)
            rh_ratio = round(rh_ratio, Composition_round)
            # 残りのメタルの比率を計算
            remaining_ratio_total = 1 - rh_ratio
            if num_metal > 1:
                # Rh以外の比率を生成
                remaining_ratios = [random.randint(1, Composition_upper_limit) for _ in range(num_metal - 1)]
                sum_remaining_ratios = sum(remaining_ratios)
                metal_ratio = [round((ratio / sum_remaining_ratios) * remaining_ratio_total, Composition_round) for ratio in remaining_ratios]
            else:
                # Rh以外のメタルがない場合
                metal_ratio = []
            # Rhの比率を追加
            metal_ratio.append(rh_ratio)
            # メタルと比率をペアにしてリストに追加
            metal_ratio_pair = []
            for a, b in zip(metal_candidate, metal_ratio):
                metal_ratio_pair += [a, b]
            inverse_metal_x_data.append(metal_ratio_pair)
      
    else:
        # 元素数に応じた元素の選択
        while len(inverse_metal_x_data) < num_of_population:
            metal_candidate = random.sample(metal_list, num_metal)
            if len(metal_candidate) == len(set(metal_candidate)):
                # 各組成の計算 : 組成は、下限0.01、上限0.99、間隔0.01　で作成する
                #ratio_x = [random.randint(1, 99) for _ in range(num_metal)]
                #metal_ratio = [round(i / sum(ratio_x), Composition_round) for i in ratio_x]
                # 各組成の計算 : 組成は、下限0.1、上限0.9、間隔0.1　で作成する
                ratio_x = [random.randint(1, Composition_upper_limit) for _ in range(num_metal)]
                metal_ratio = [round(i / sum(ratio_x), Composition_round) for i in ratio_x]
                while (0.0 in metal_ratio):
                    #ratio_x = [random.randint(1, 99) for _ in range(num_metal)]
                    #metal_ratio = [round(i / sum(ratio_x), Composition_round) for i in ratio_x]
                    ratio_x = [random.randint(1, Composition_upper_limit) for _ in range(num_metal)]
                    metal_ratio = [round(i / sum(ratio_x), Composition_round) for i in ratio_x]
                # ratio_x = np.random.rand(self.num_metal) * (99 - 1) + 1  # 乱数を生成
                # metal_ratio = np.round(ratio_x / ratio_x.sum(), decimals=2)  # 割って0.01～0.99にする
                metal_ratio_pair = []
                for a, b in zip(metal_candidate, metal_ratio):
                    metal_ratio_pair = metal_ratio_pair + [a, b]
                inverse_metal_x_data.append(metal_ratio_pair)
        # inverse_metal_x_data = pd.DataFrame([random.sample(metal_list, self.num_metal) for i in range(3)], columns=metal_list)
        # # 各組成の計算 : 組成は、下限0.01、上限0.99、間隔0.01　で作成する
    return inverse_metal_x_data


def mutant_metal(ssc_condition, individual_metal, metal_list, indpb):
    metal_name_list = individual_metal[::2]
    new_metal = individual_metal.copy()
    for idx in range(len(new_metal)):
        if random.random() < indpb:
            if idx % 2 == 0:  # metal_name
                new_metal_name = metal_list[random.randint(0, len(metal_list) - 1)]
                while len(set(new_metal[::2] + [new_metal_name])) == len(set(new_metal[::2])):
                    # new_metal_name　は、重複するmetal_name
                    new_metal_name = metal_list[random.randint(0, len(metal_list) - 1)]
                new_metal[idx] = new_metal_name
            else:  # ratio
                new_ratio = random.randint(1, Composition_upper_limit) / Composition_upper_limit_div
                #new_ratio = random.randint(1, 99) / 100
                new_metal[idx] = new_ratio
    metal_ratio_list = new_metal[1::2]
    metal_ratio_list = [round(i / sum(metal_ratio_list), Composition_round)for i in metal_ratio_list]
    #metal_ratio_list = [round(i / sum(metal_ratio_list), Composition_round)for i in metal_ratio_list]
    new_metal[1::2] = metal_ratio_list
    return new_metal


def select_best_ind(x_name, ssc_condition, pop, pop_metal, variables, condition_dict, xenonpy_element_data):
    # ベストの遺伝子を選択
    fitness = [each.fitness.values[0] for each in pop]
    best_ind_index = fitness.index(max(fitness))
    best_pop = np.array(pop[best_ind_index]).tolist()
    best_pop_metal = pop_metal[best_ind_index]
    # 個体の取得
    best_individual_df = pd.DataFrame([best_pop + best_pop_metal],
                                      columns=list(condition_dict.keys()) + metal_ratio_list[:len(best_pop_metal)])
    # 遺伝子をxに変換
    best_individual_inverse_analysis_x_df = pd.DataFrame(np.zeros((1, variables.shape[1])),
                                                         columns=variables.columns)
    rtn_message, best_individual_inverse_analysis_x_df = \
        ind_to_xdata(x_name, ssc_condition, best_individual_df, best_individual_inverse_analysis_x_df, condition_dict,
                     xenonpy_element_data, return_metal=False)
    # ベスト遺伝子の金属の組み合わせと組成を保存
    inverse_metal_list = best_pop_metal[::2]
    inverse_metal_component_list = best_pop_metal[1::2]
    inverse_metal_list = inverse_metal_list + \
        [np.nan] * (5 - len(inverse_metal_list))
    inverse_metal_component_list = inverse_metal_component_list + \
        [np.nan] * (5 - len(inverse_metal_component_list))
    best_individual_inverse_analysis_metal_df = pd.DataFrame(
        np.array(inverse_metal_list + inverse_metal_component_list).reshape((1, -1)))
    return best_individual_inverse_analysis_x_df, best_individual_inverse_analysis_metal_df


def inverse_analysis_ga(x_name, ssc_condition, variables, x_names, metal_list,
                        synthesis_metal_x_data, used_element_data, 
                        xenonpy_element_data, 
                        autoscaled_estimated_means, autoscaled_estimated_covariances, weights, Ir_Rh_none_flag='0'):

    training_elements = list(
        set(np.ndarray.flatten(np.array((synthesis_metal_x_data.iloc[:, :int(synthesis_metal_x_data.shape[1] / 2)])))))
    training_elements.remove(np.nan)
    element_candidates = list(used_element_data.columns)
    element_candidates_np = np.array(element_candidates)
    all_element_candidates = list(set(training_elements + element_candidates))

    # GA の設定関係
    numbers_of_candidates = []
    for key in inverse_analysis_condition_dict.keys():
        condition_list = inverse_analysis_condition_dict[key]
        numbers_of_candidates.append(len(condition_list[1]))
    numbers_of_candidates.append(len(numbers_of_used_metals))  # 金属種の個数

    probs_ga = np.zeros([number_of_iterations, 1])
    xs_ga = np.zeros([number_of_iterations, len(x_names) + max(numbers_of_used_metals) * 2])
    xs_ga = xs_ga.astype(object)
    # gmr_flag = True
    for iteration_number in range(number_of_iterations):
        print('Inverse analysis ... ', iteration_number +
              1, '/', number_of_iterations)

        # GA
        # for minimization, set weights as (-1.0,)
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        # 遺伝子を生成する関数"attr_gene"を登録
        toolbox.register("attr_gene", random.randint, 0, 99)
        num_ind = len(inverse_analysis_condition_dict)
        toolbox.register("individual", tools.initRepeat,
                         creator.Individual, toolbox.attr_gene, num_ind)
        toolbox.register('population', tools.initRepeat,
                         list, toolbox.individual)
        # 遺伝的操作の設定
        toolbox.register('evaluate', evaluate)  # 評価関数の設定
        toolbox.register('mate', tools.cxTwoPoint)  # 交叉の設定
        toolbox.register('mutate', tools.mutUniformInt,
                         low=0, up=99, indpb=0.3)  # 突然変異の設定
        toolbox.register("select", Tournament, tournsize=3)
        # 個体集団の生成
        pop = toolbox.population(n=number_of_population)
        # 金属の数を選択
        metal_num = numbers_of_metals
        #metal_num = random.choice([2, 3, 4, 5])
        pop_metal = create_pop_metal(number_of_population, metal_list, metal_num, Ir_Rh_none_flag)
        #pop_metal = create_pop_metal(number_of_population, metal_list.tolist(),
        #                             element_candidates_omitN, element_candidates_omitY, metal_num, element_constraints)
        print("Start of evolution")

        # 個体集団の適応度の評価
        fitnesses = toolbox.evaluate(x_name, ssc_condition, pop, pop_metal, variables, inverse_analysis_condition_dict, 
                                     xenonpy_element_data,
                                     autoscaled_estimated_means, autoscaled_estimated_covariances, weights)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        print("  Evaluated %i individuals" % len(pop))

        # 適応度の抽出
        fits = [ind.fitness.values[0] for ind in pop]

        # 進化ループ開始
        df_fitness = pd.DataFrame(np.zeros((number_of_population, number_of_generation + 1)),
                                  columns=[f'{i}-gen' for i in range(number_of_generation + 1)])
        df_metal = pd.DataFrame([[[]] * (number_of_generation + 1)] * number_of_population,
                                columns=[f'{i}-gen' for i in range(number_of_generation + 1)])
        best_fit_list = [sorted(fits, reverse=True)[0]]
        df_fitness.iloc[:, 0] = sorted(fits, reverse=True)
        for i in range(number_of_population):
            df_metal.iat[i, 0] = pop_metal[i]
        for generation in range(number_of_generation):
            # print('-- Generation {0} --'.format(generation + 1))
            # 次世代個体の選択・複製
            offspring, pop_metal_new = toolbox.select(pop, len(pop), pop_metal)
            offspring = list(map(toolbox.clone, offspring))
            pop_metal_new = [item.copy() for item in pop_metal_new]
            changed_individulas = set()
            # 交叉
            num_of_mate_offspring = 0
            for idx1, idx2 in zip(range(len(offspring))[::2], range(len(offspring))[1::2]):
                # 交叉させる個体を選択
                if random.random() < probability_of_crossover:
                    changed_individulas.add(idx1)
                    changed_individulas.add(idx2)
                    toolbox.mate(offspring[idx1], offspring[idx2])
                    num_of_mate_offspring += 2
                    # 交叉させた個体は適応度を削除する
                    del offspring[idx1].fitness.values
                    del offspring[idx2].fitness.values
            # 変異
            num_of_mutate_offspring = 0
            for idx in range(len(offspring)):
                # 変異させる個体を選択
                if random.random() < probability_of_mutation:
                    changed_individulas.add(idx)
                    toolbox.mutate(offspring[idx])
                    num_of_mutate_offspring += 1
                    # 変異させた個体は適応度を削除する
                    del offspring[idx].fitness.values
            # 変異_metal
            num_of_mutate_metal = 0
            for idx in range(len(pop_metal_new)):
                # 変異させる個体を選択
                if random.random() < probability_of_mutation:
                    changed_individulas.add(idx)
                    new_mutant = mutant_metal(ssc_condition, pop_metal_new[idx], metal_list, indpb=0.3)
                    #new_mutant = mutant_metal(pop_metal_new[idx], metal_list.tolist(), element_candidates_omitN,
                    #                          element_candidates_omitY, element_constraints, indpb=0.3)
                    pop_metal_new[idx] = new_mutant
                    num_of_mutate_metal += 1
            fitnesses = toolbox.evaluate(x_name, ssc_condition, offspring, pop_metal_new, variables, inverse_analysis_condition_dict,
                                         xenonpy_element_data,
                                         autoscaled_estimated_means, autoscaled_estimated_covariances, weights)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            print(f"\r\t-- iteration_number:{iteration_number + 1} Generation:{generation + 1} --  Evaluated {len(changed_individulas)} ,  "
                  f"Changed(mate):{num_of_mate_offspring}, Changed(mutate):{num_of_mutate_offspring}, "
                  f"Changed(metal):{num_of_mutate_metal}", end=' ')

            # 個体集団を新世代個体集団で更新
            pop[:] = offspring
            pop_metal[:] = pop_metal_new
            # 新世代の全個体の適応度の抽出
            fits = [ind.fitness.values[0] for ind in pop]
            best_fit_list.append(sorted(fits, reverse=True)[0])
            df_fitness.iloc[:, generation + 1] = sorted(fits, reverse=True)
            for i in range(number_of_population):
                df_metal.iat[i, generation + 1] = pop_metal[i]
            # n世代にわたりbest_fitの値(小数点以下5桁)が変化していないならば、ほほ収束したと判断し、ループから抜け出す
            #convergence_generation = 5
            if len(best_fit_list) >= convergence_generation:
                ###best_fit_list_round = [i.round(5) for i in best_fit_list]
                best_fit_list_round = [round(i, 5) for i in best_fit_list]
                if len(set(best_fit_list_round[-convergence_generation:])) == 1:
                    break

        best_fit_df = pd.DataFrame(best_fit_list, columns=['fitness'],
                                   index=[f"{i}世代" for i in range(len(best_fit_list))])
                                   #index=[f"{i}世代" for i in range(number_of_generation + 1)])
        best_fit_df.plot()
        savefig_path = f"result/debug_gmm_ga/fitness_progress_({iteration_number}).png"
        plt.savefig(savefig_path, bbox_inches='tight')
        # plt.show()
        best_fit_df.to_excel(
            f'result/debug_gmm_ga/best_fitness_progress_({iteration_number}).xlsx')
        df_fitness.to_excel(
            f'result/debug_gmm_ga/df_fitness_({iteration_number}).xlsx')
        df_metal.to_excel(
            f'result/debug_gmm_ga/df_metal_({iteration_number}).xlsx')
        print('-- End of (successful) evolution --')

        # bestな遺伝子を選択
        mi_util = MIutility() #データ作成クラス生成
        best_individual_inverse_analysis_x_df, best_individual_inverse_analysis_metal_df = \
            select_best_ind(x_name, ssc_condition, pop, pop_metal, variables,
                            inverse_analysis_condition_dict, xenonpy_element_data)
        best_individual_inverse_analysis_metal_df.columns = metal_ratio_list_2
        #best_individual_inverse_analysis_metal_df.columns = metal_ratio_list
        
        #ratio1～ratio5の合計が「1.00」になっていない行を修正する
        #best_individual_inverse_analysis_metal_df = mi_util.checkMetalRatioTotalForRatio3(best_individual_inverse_analysis_metal_df) 
        best_individual_inverse_analysis_metal_df = mi_util.checkMetalRatioTotal(best_individual_inverse_analysis_metal_df)  
        #alphabet順に入れ替え
        best_individual_inverse_analysis_metal_df = mi_util.checkMetalAlphabetOrder(best_individual_inverse_analysis_metal_df)
        best_individual_inverse_analysis_metal_df = best_individual_inverse_analysis_metal_df.replace(['', None, 'nan'], np.nan)
        
        best_individual_inverse_analysis_x_df = \
            best_individual_inverse_analysis_x_df.loc[:, variables.columns]
        
        autoscaled_best_individual_inverse_analysis_x_df = autoscaling(
            best_individual_inverse_analysis_x_df, variables)
        autoscaled_best_individual_inverse_analysis_x_df.drop(
            columns=y_names, inplace=True)
        tmps = []
        for i in range(autoscaled_estimated_covariances.shape[0]):             
            tmp = np.log(weights[i, 0]) + \
                multivariate_normal.logpdf(autoscaled_best_individual_inverse_analysis_x_df.values,
                                           mean=autoscaled_estimated_means[i, 0, :],
                                           cov=autoscaled_estimated_covariances[i, :, :])
            tmps.append(tmp)
        value = logsumexp(tmps)
        probs_ga[iteration_number, 0] = value
        conditon_list = list(inverse_analysis_condition_dict.keys())
        conditon_list.remove('support')
        if(ssc_condition=='scc_1'):
            # x26_NPA2は対象外
            #best_individual_inverse_analysis_x_df.insert(2, '評価反応炉温℃', 300)
            
            """
            best_individual_inverse_analysis_x_df.insert(2, '前処理還元炉温℃', 450)
            best_individual_inverse_analysis_x_df.insert(4, 'temp', 350)
            best_individual_inverse_analysis_x_df.insert(5, 'flow_NaOH', 10)
            best_individual_inverse_analysis_x_df.insert(6, 'flow_slurry', 20)
            best_individual_inverse_analysis_x_df.insert(7, 'flow_red', 100)
            best_individual_inverse_analysis_x_df.insert(8, 'wash', 1)
            best_individual_inverse_analysis_x_df.insert(9, 'support_Al2O3_A-11', 1)
            best_individual_inverse_analysis_x_df.insert(10, 'support_CeO2_HS', 0)
            best_individual_inverse_analysis_x_df.insert(11, 'support_TiO2_SSP-M', 0)
            best_individual_inverse_analysis_x_df.insert(12, 'support_ZrO2_RC100', 0)
            """

        if numbers_of_metals == 3:
            best_individual_inverse_analysis_metal_df = best_individual_inverse_analysis_metal_df.drop(['metal4', 'metal5', 'ratio4', 'ratio5'],axis=1)
            
        valid_support = [c for c in best_individual_inverse_analysis_x_df.columns if 'support' in c]
        
        xs_ga[iteration_number, :] = \
            np.c_[best_individual_inverse_analysis_x_df.loc[:, conditon_list + valid_support].values,
                  best_individual_inverse_analysis_metal_df.values]
            
        #xs_ga[iteration_number, :] = \
        #    np.c_[best_individual_inverse_analysis_x_df.loc[:, conditon_list + supports_list].values,
        #          best_individual_inverse_analysis_metal_df.values]

    return probs_ga, xs_ga


def createModelGMR(autoscaled_variables, numbers_of_x, numbers_of_y):
    if method == 'GMR':
        model = GMR()  # GMR モデルの宣言
        if hyperparameter_optimization == 'cv_opt':  # グリッドサーチ+クロスバリデーションによるハイパーパラメータの最適化
            model.cv_opt(autoscaled_variables, numbers_of_x, numbers_of_y, covariance_types, numbers_of_components,
                         fold_number)
        elif hyperparameter_optimization == 'cv_bo':  # クロスバリデーション+ベイズ最適化によるハイパーパラメータの最適化
            model.cv_bo(autoscaled_variables, numbers_of_x, numbers_of_y, covariance_types, numbers_of_components,
                        fold_number)
    elif method == 'VBGMR':
        model = VBGMR()  # VBGMR モデルの宣言
        if hyperparameter_optimization == 'cv_opt':  # グリッドサーチ+クロスバリデーションによるハイパーパラメータの最適化
            model.cv_opt(autoscaled_variables, numbers_of_x, numbers_of_y, covariance_types, numbers_of_components,
                         weight_concentration_prior_types, weight_concentration_priors, fold_number)
        elif hyperparameter_optimization == 'cv_bo':  # クロスバリデーション+ベイズ最適化によるハイパーパラメータの最適化
            model.cv_bo(autoscaled_variables, numbers_of_x, numbers_of_y, covariance_types, numbers_of_components,
                        weight_concentration_prior_types, weight_concentration_priors, fold_number)

    return model


def createMetalList(synthesis_metal_x_data):

    synthesis_metal_x_data_tmp = synthesis_metal_x_data.reset_index()
    synthesis_metal_x_data_tmp.drop_duplicates(inplace=True)
    synthesis_metal_x_data = synthesis_metal_x_data_tmp.set_index(
        '触媒ロット', drop=True)

    synthesis_metal_x_data.sort_index(inplace=True)
    met1 = synthesis_metal_x_data['metal1'].unique()
    met2 = synthesis_metal_x_data['metal2'].unique()
    met3 = synthesis_metal_x_data['metal3'].unique()
    met4 = synthesis_metal_x_data['metal4'].unique()
    met5 = synthesis_metal_x_data['metal5'].unique()
    metal_list = np.concatenate([met1, met2, met3, met4, met5])
    metal_list = pd.Series(metal_list)
    metal_list = metal_list.dropna()
    metal_list = metal_list.unique()
    metal_list = np.sort(metal_list)

    return metal_list


def checkDefaultX0(synthesis_metal_x_data):
    rd = 'results\\'
    evaluation_lot_data = pd.read_excel(
        rd + 'evaluation_lot_data.xlsx', index_col=0, header=0, engine='openpyxl')
    evaluation_y_x_data_with_dummy_variables = pd.read_excel(rd + 'evaluation_y_x_data_with_dummy_variables.xlsx',
                                                             index_col=0, header=0, engine='openpyxl')
    synthesis_condition_x_data_with_dummy_variables = pd.read_excel(
        rd + 'synthesis_condition_x_data_with_dummy_variables.xlsx', index_col=0, header=0, engine='openpyxl')

    # x1の作成
    x1 = evaluation_y_x_data_with_dummy_variables.copy()
    x1['触媒ロット'] = evaluation_lot_data.loc[evaluation_y_x_data_with_dummy_variables.index, '触媒ロット']
    x1['評価ロット数字'] = x1.index
    # 目的変数の内行は削除
    x1.dropna(subset=['選択率NPA※※'], axis=0, inplace=True)
    # tantai_col = [i for i in synthesis_condition_x_data_with_dummy_variables.columns if '担体種類' in i]
    x1 = pd.merge(x1, synthesis_condition_x_data_with_dummy_variables,
                  right_index=True, left_on='触媒ロット')
    x1.drop_duplicates(subset='評価ロット数字', inplace=True)

    synthesis_metal_x_data_tmp = synthesis_metal_x_data.reset_index()
    synthesis_metal_x_data_tmp.drop_duplicates(inplace=True)
    synthesis_metal_x_data = synthesis_metal_x_data_tmp.set_index(
        '触媒ロット', drop=True)

    synthesis_metal_x_data.sort_index(inplace=True)
    met1 = synthesis_metal_x_data['metal1'].unique()
    met2 = synthesis_metal_x_data['metal2'].unique()
    met3 = synthesis_metal_x_data['metal3'].unique()
    met4 = synthesis_metal_x_data['metal4'].unique()
    met5 = synthesis_metal_x_data['metal5'].unique()
    metal_list = np.concatenate([met1, met2, met3, met4, met5])
    metal_list = pd.Series(metal_list)
    metal_list = metal_list.dropna()
    metal_list = metal_list.unique()
    metal_list = np.sort(metal_list)
    synthesis_metal_x_data_modified = pd.DataFrame(
        index=synthesis_metal_x_data.index, columns=metal_list)

    # 触媒組成のデータの元素、組成を上のdf形式へ変換
    synthesis_metal_x_data_modified['dummy'] = np.nan
    for index, row in synthesis_metal_x_data.iterrows():
        content1 = row[0], row[5]
        content2 = row[1], row[6]
        content3 = row[2], row[7]
        content4 = row[3], row[8]
        content5 = row[4], row[9]
        if content3[0] is np.nan:
            content3 = list(content3)
            content3[0] = 'dummy'
        if content4[0] is np.nan:
            content4 = list(content4)
            content4[0] = 'dummy'
        if content5[0] is np.nan:
            content5 = list(content5)
            content5[0] = 'dummy'
        synthesis_metal_x_data_modified.iat[
            synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(
                content1[0])] = content1[1]
        synthesis_metal_x_data_modified.iat[
            synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(
                content2[0])] = content2[1]
        synthesis_metal_x_data_modified.iat[
            synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(
                content3[0])] = content3[1]
        synthesis_metal_x_data_modified.iat[
            synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(
                content4[0])] = content4[1]
        synthesis_metal_x_data_modified.iat[
            synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(
                content5[0])] = content5[1]
    synthesis_metal_x_data_modified.drop('dummy', axis=1, inplace=True)
    synthesis_metal_x_data_modified.replace(np.nan, 0, inplace=True)

    vari_0_numbers = np.where(synthesis_metal_x_data_modified.var() == 0)[0]
    synthesis_metal_x_data_modified = synthesis_metal_x_data_modified.drop(
        synthesis_metal_x_data_modified.columns[vari_0_numbers], axis=1)  # 分散が 0 の変数を削除

    x_conditions_with_y = x1.copy()
    x_conditions_with_y.drop(['触媒ロット', '評価ロット数字'], axis=1, inplace=True)
    x_names = list(x_conditions_with_y.columns[len(y_targets):])

    x1 = pd.merge(x1, synthesis_metal_x_data_modified,
                  right_index=True, left_on='触媒ロット')
    x1['評価ロット数字'] = x1.index
    x1.drop_duplicates(subset='評価ロット数字', inplace=True)
    x1.replace(np.nan, 0, inplace=True)

    # x1 = remove_all_zero_col(x1)
    x1.drop('評価ロット数字', axis=1, inplace=True)

    # check = x1.iloc[:,14:]
    x1 = x1[x1.iloc[:, 14:].sum(axis=1) > 0]  # 分析値が0のデータを削除
    x1.sort_index(inplace=True)
    x1.columns = [i.replace('※', '') if '※' in i else i for i in x1.columns]
    x1.drop(['触媒ロット'], axis=1, inplace=True)

    return x1, x_names

def checkXfile(x_name):
    data = pd.read_csv('datasets/v630_Ti/'+x_name+'.csv', index_col=0)  # データの読み込み
    #data = pd.read_csv('matlantis_descriptors/datasets/Akashi#1/'+x_name+'.csv', index_col=0)  # データの読み込み    

    
    x_data = data.copy()
    if x_name in ['x1_ce_am','x1_ti','x1_zr_am']:
        data_ato = pd.read_csv('datasets/v630_Ti/x1_41.csv', index_col=0)
        data_ato = data_ato[data_ato['前担持'] == 0]
        if x_name == 'x1_ce_am':
            data_ato = data_ato[data_ato['support_CeO2_HS'] == 1]
        elif x_name == 'x1_ti':
            data_ato = data_ato[data_ato['support_TiO2_SSP-M'] == 1]
        elif x_name == 'x1_zr_am':
            data_ato = data_ato[data_ato['support_ZrO2_RC100'] == 1]
        x_data = pd.concat([data, data_ato], axis=0)    
        x_data = x_data.drop(['触媒ロット'], axis=1)    
        x_data = x_data.drop(x_data.columns[np.where(x_data.var() < 1e-10)], axis=1)
    else:
        data.drop(['触媒ロット'], axis=1, inplace=True)
        x_data = x_data.drop(x_data.columns[np.where(x_data.var() < 1e-10)], axis=1)
        
    col = x_data.columns.tolist()
    support_indices = [i for i, c in enumerate(col) if c.startswith('support')]
    if len(support_indices) != 0:
        last_support = max(support_indices)
        x_names = col[len(y_targets): last_support + 1]
    elif '前担持' in col:
        mae_index = [i for i, c in enumerate(col) if c == '前担持']
        x_names = col[len(y_targets): mae_index[0] + 1]
    elif 'wash' in col:
        wash_index = [i for i, c in enumerate(col) if c == 'wash']
        x_names = col[len(y_targets): wash_index[0] + 1]
    else:
        red_index = [i for i, c in enumerate(col) if c == 'flow_red']
        x_names = col[len(y_targets): red_index[0] + 1]

    return x_data, x_names

def checkProbsGaResult(probs_ga, xs_ga, x_names, synthesis_metal_x_data, ssc_condition):
    probs_ga = pd.DataFrame(probs_ga, columns=['probs_ga'])
    probs_ga = probs_ga.sort_values('probs_ga', ascending=False)
    if(ssc_condition=='scc_1'):
        xs_ga = pd.DataFrame(xs_ga, columns=x_names + metal_ratio_list_2)    #5元素用
        #xs_ga = pd.DataFrame(xs_ga, columns=x_names + metal_ratio_list_3)   #3元素用
    else:
        xs_ga = pd.DataFrame(xs_ga, columns=x_names + metal_ratio_list_2)
    #xs_ga = pd.DataFrame(xs_ga, columns=x_names + list(synthesis_metal_x_data.columns))
    xs_ga.replace('nan', '', inplace=True)
    xs_ga = xs_ga.loc[probs_ga.index]
    # xs_gaで重複しているものを除く。
    duplicates_data = xs_ga[xs_ga.duplicated()]
    duplicates_data.to_csv(r'result/debug_gmm_ga/duplicates_data.csv', encoding='cp932')
    index = xs_ga.drop_duplicates(keep='first').index
    xs_ga = xs_ga.loc[index, :]
    probs_ga = probs_ga.loc[index, :]

    xs_ga_prob = pd.concat([xs_ga, probs_ga], axis=1)

    return xs_ga_prob


def createDebugDir():
    if os.path.exists('result/debug_gmm_ga'):
        shutil.rmtree('result/debug_gmm_ga')
    if not os.path.exists('result/debug_gmm_ga'):
        os.mkdir('result/debug_gmm_ga')

def saveDebugInfoForModel(model,autoscaled_variables_test,numbers_of_y,numbers_of_x,variables):
    autoscaled_estimated_x = model.predict_rep(autoscaled_variables_test, numbers_of_y, numbers_of_x)
    estimated_x = pd.DataFrame(autoscaled_estimated_x, columns=variables.iloc[:, numbers_of_x].columns) * \
                  pd.DataFrame([variables.iloc[:, numbers_of_x].std(axis=0, ddof=1)]) + \
                  pd.DataFrame([variables.iloc[:, numbers_of_x].mean(axis=0)])
    estimated_x = estimated_x.round(2)
    estimated_x.to_csv('result/debug_gmm_ga/gmm_estimated_x_values.csv', encoding='cp932', index=False)
    autoscaled_estimated_means, autoscaled_estimated_covariances, weights = \
        model.predict_mog(autoscaled_variables_test, numbers_of_y, numbers_of_x)
    # autoscaled_estimated_meansとweightsを出力する
    df_estimated_means = pd.DataFrame(index=range(len(weights)),
                                      columns=['weight'] + variables.iloc[:, numbers_of_x].columns.to_list())
    for idx in range(len(weights)):
        df_estimated_means.iloc[idx, 0] = weights[idx][0]
        estimated_x = pd.DataFrame(autoscaled_estimated_means[idx],
                                   columns=variables.iloc[:, numbers_of_x].columns) * \
                      pd.DataFrame([variables.iloc[:, numbers_of_x].std(axis=0, ddof=1)]) + \
                      pd.DataFrame([variables.iloc[:, numbers_of_x].mean(axis=0)])
        df_estimated_means.iloc[idx, 1:] = estimated_x
    df_estimated_means = df_estimated_means.astype('float64').round(2)
    df_estimated_means.to_csv('result/debug_gmm_ga/gmm_estimated_x_means.csv', encoding='cp932')
    

def inverse_GMM(x_name, Ir_Rh_none_flag, ssc_condition):
    start = time.time()
    # debug フォルダ
    createDebugDir()
    rd = os.path.join('results', '')
    
    # xのチェックnumber_of_iterations
    print('■ Xの設定')
    print('x_name = ', x_name)
    print('ssc_condition = ', ssc_condition)
    print('Ir_Rh_none_flag = ', Ir_Rh_none_flag)
    print('■ 実行時のparameter')
    print('method = ', method)

    print('◇GA設定')
    print('number_of_iterations = ', number_of_iterations)
    print('number_of_population = ', number_of_population)
    print('number_of_generation = ', number_of_generation)
    print('convergence_generation = ', convergence_generation)
    
    # resultフォルダからデータ取得（ファイル存在確認）
    try:
        xenonpy_element_data = pd.read_csv('results/xenonpy_element_data240515.csv', index_col=0)
    except FileNotFoundError:
        print("エラー: results/xenonpy_element_data240515.csv が見つかりません")
        return
    
    try:
        synthesis_metal_x_data = pd.read_excel(rd + 'synthesis_metal_x_data.xlsx', index_col=0, header=0, engine='openpyxl')
    except FileNotFoundError:
        print(f"エラー: {rd}synthesis_metal_x_data.xlsx が見つかりません")
        return
        
    # 住友指定の元素を取得
    if x_name == 'xx2_mat1_b9':
        used_element_data = pd.DataFrame(columns= ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ir', 'Pt', 'Au', 'V', 'Mo', 'W', 'Rh'])
    elif x_name == 'x2_dft1_b9':
        used_element_data = pd.DataFrame(columns= ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Ru', 'Pd', 'Ir', 'Pt', 'Au', 'V', 'Rh'])
    else:
        used_element_data = pd.read_csv(rd + '20230316_使用元素リスト.csv', index_col=None, encoding='cp932')
        # 削減元素を除く
        drop_elements = ['Li', 'K', 'Rb', 'Cs', 'Ti', 'Se']
        used_element_data = used_element_data.drop(
            [el for el in drop_elements if el in used_element_data.columns], 
            axis=1
        )

    #Ir、RhなしFlag導入
    if (Ir_Rh_none_flag == '1'):
        used_element_data = used_element_data
        
    # xのmetal_list
    if (x_name == 'x0'):
        # X、実験条件のカラム名を取得
        x, x_names = checkDefaultX0(synthesis_metal_x_data)
        # default(Kaneko's sample)
        metal_list = list(createMetalList(synthesis_metal_x_data))
    else:
        # X、実験条件のカラム名を取得
        x, x_names = checkXfile(x_name)
        #x = x.iloc[: 50, :]
        # 'x0'以外の場合
        metal_list = list(used_element_data.columns)
    
    
    # モデル用データ：variables
    variables = x.copy()
    variables = variables.sort_index()
    variables.columns = [i.replace('※', '') if '※' in i else i for i in variables.columns]

    # モデリング
    print('Modeling　')
    numbers_of_x = list(range(len(y_targets), variables.shape[1]))
    numbers_of_y = list(range(0, len(y_targets)))
    autoscaled_variables = (variables - variables.mean(axis=0)) / variables.std(axis=0, ddof=1)
    
    # nan列は削除
    autoscaled_variables = autoscaled_variables.dropna(axis=1)
    numbers_of_x = list(range(len(y_targets), autoscaled_variables.shape[1]))
    # variables : var()==0のものを削除する
    variables = variables.drop(variables.columns[np.where(variables.var() < 1e-10)], axis=1)
    
    # モデル構築
    print('モデル構築 : NPA選択率 & NPA収率')
    model = createModelGMR(autoscaled_variables, numbers_of_x, numbers_of_y)
    model.fit(autoscaled_variables)

    # Y の目標値の設定
    print('Y の目標値の設定')
    variables_test = np.array(y_targets).reshape([1, len(y_targets)])  # Y の目標値
    variables_test = pd.DataFrame(variables_test, columns=variables.columns[numbers_of_y])
    # オートスケーリング
    autoscaled_variables_test = (variables_test - variables.iloc[:, numbers_of_y].mean(axis=0)) / \
                                variables.iloc[:, numbers_of_y].std(axis=0, ddof=1)
                                    
    print(f'model:{model}')
    
    ### Debug Info : 
    # 参考のために、modelとmodel.predict_rep,model.predict_mogをdebugフォルダに出力しておく
    # 参考のためにmodel.predict_repによるxの値を出力する
    saveDebugInfoForModel(model,autoscaled_variables_test,numbers_of_y,numbers_of_x,variables)
    
    # 直接的逆解析
    print('直接的逆解析')
    autoscaled_estimated_means, autoscaled_estimated_covariances, weights = \
        model.predict_mog(autoscaled_variables_test, numbers_of_y, numbers_of_x)

    # GA実行
    print('GA実行')
    probs_ga, xs_ga = inverse_analysis_ga(x_name, ssc_condition, variables, x_names, metal_list,
                                synthesis_metal_x_data, used_element_data,
                                xenonpy_element_data,
                                autoscaled_estimated_means, autoscaled_estimated_covariances, weights, Ir_Rh_none_flag)

    # 確率密度関数から上位を提案
    print('確率密度関数から上位を提案')
    xs_ga_prob = checkProbsGaResult(probs_ga, xs_ga, x_names, synthesis_metal_x_data, ssc_condition)
    #metal1-metal5の組合せ被り、metal1-5が0であるものを除外する
    xs_ga_prob = xs_ga_prob.drop_duplicates(subset= ['metal1','metal2','metal3','metal4','metal5'], keep= 'first')
    xs_ga_prob = xs_ga_prob[(xs_ga_prob['ratio1']!=0)&(xs_ga_prob['ratio2']!=0)&(xs_ga_prob['ratio3']!=0)&(xs_ga_prob['ratio4']!=0)&(xs_ga_prob['ratio5']!=0)]
    xs_ga_prob.to_csv(
        r'result/inverse_GMM_{0}_{1}.csv'.format(x_name, method), encoding='cp932')

    elapsed_time = time.time() - start
    print("計算時間: :{0}".format(np.round(elapsed_time / 60)) + '[分]')


if __name__ == '__main__':

    # 直接的逆解析の対象
    x_name = 'x1_zr_am'     # ５元素対応
    #x_name = 'x2_dft1_b9'
    #x_name = 'x2_17'     # ５元素対応
        
    # 住友化学の条件
    #Ir、RhなしFlag導入
    # '0':Ir_Rh_あり、2:Rhあり、4:Rh0.3-0.7とrandomが1:1
    Ir_Rh_none_flag = '2'  
    # scc_1:固定
    #       評価反応炉温℃:300℃
    ssc_condition = 'scc_0'
    #ssc_condition = 'scc_0' # scc_0:特になし、

    # ---------------------------
    # その他、考慮点
    # ---------------------------
    # methodを選択  # 'GMR' or 'VBGMR'
    #

    #
    # 使用元素リスト
    # # 金属元素数：３の場合
    #     used_element_data =  pd.DataFrame(columns=['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt', 'Au', 'In', 'Sn'])
    #
    # 元素数　3か5かコード上部にある下記４定数を変更すること
    # # 金属元素数：３の場合
    #     numbers_of_used_metals = [3]            # 金属種の候補
    #     numbers_of_metals = 3                   # GA:金属の数を選択
    #     Composition_upper_limit = 9             # 小数点以下 0.1 単位の場合
    #     Composition_upper_limit_div = 10        # 小数点以下 0.1 単位の場合
    #
    # Composition_round = 1  # 金属の構成割合は0.1刻み、小数点1桁
    
    inverse_GMM(x_name, Ir_Rh_none_flag, ssc_condition)
