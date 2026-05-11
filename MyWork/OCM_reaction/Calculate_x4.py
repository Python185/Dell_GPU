# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from libs.boruta import BorutaPy
from BorutaShap import BorutaShap
import matplotlib.pyplot as plt
import pycalphad
from pycalphad import Database, Model, calculate, equilibrium, variables as v
import chardet
import sympy
from sympy import srepr, Mul
import itertools
import codecs
import seaborn as sns

def Boruta_Apply(target, df, perc):
    y= df[target].copy()
    #y = df['転化率']
    x= df.drop(target, axis= 1).copy()
    if '触媒ロット' in x.columns.to_list():
        x = x.drop('触媒ロット', axis= 1)

    # RandomForestRegressorでBorutaを実行
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state= 1)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state= 1, perc= perc)
    feat_selector.fit(x.values, y.values)

    # 選択された特徴量を確認
    selected = feat_selector.support_
    print('選択された特徴量の数: %d' % np.sum(selected))
    print(x.columns[selected])

    #上で選択した説明変数のみを残す。
    selected_features= x.columns[selected]
    rf_selected = RandomForestRegressor(n_jobs=-1, max_depth= 5, random_state= 1)
    rf_selected.fit(x[selected_features], y.values)
    selected_importance = rf_selected.feature_importances_
    df_b = df.loc[:, selected_features]
    df_b = pd.concat([df[target], df_b], axis= 1)
    importance_df = pd.DataFrame(index= selected_features, data= selected_importance, columns= ['Importance'])
    importance_df = importance_df.sort_values(by= 'Importance', ascending= False)
    
    return df_b, importance_df

def Boruta_Apply2(target, df, perc):
    y = df[target].copy()
    x = df.drop(target, axis= 1).copy()
    if '触媒ロット' in x.columns:
        x = x.drop('触媒ロット', axis= 1)
    # 同じ値を多く持つ変数を削除
    deleting_variable_numbers = []
    for x_variable_name in x.columns:
        same_value_number = x[x_variable_name].value_counts()
        if max(same_value_number) >= x[x_variable_name].count() - 1:
            deleting_variable_numbers.append(x_variable_name)
    x.drop(columns = deleting_variable_numbers, inplace = True)
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
    df_b = pd.concat([df[target], df_b], axis= 1)
    importance_df = pd.DataFrame(index= selected_features, data= selected_importance, columns= ['Importance'])
    importance_df = importance_df.sort_values(by= 'Importance', ascending= False)    
    
    return df_b, importance_df

def BorutaShap_Apply(target, df, perc):
    datatype = df.dtypes
    y= df[target].copy()
    x = df.drop(target, axis= 1).copy()
    if '触媒ロット' in x.columns:
        x = x.drop('触媒ロット', axis= 1)
    # BorutaShpaの実行 defaultではXGB
    #model = lgb.LGBMRegressor()
    model = xgb.XGBRFRegressor()
    Feature_Selector = BorutaShap(model= model, importance_measure= 'shap', classification= False, percentile= perc)        
    Feature_Selector.fit(X= x, y= y, n_trials= 50) 
    #Feature_Selector.plot(which_features= 'all')
    #上で選択した説明変数のみを残す。
    importance_df = Feature_Selector.Subset()
    selected_features= importance_df.columns.to_list()
    df_b= df.loc[:, selected_features]
    df_c = pd.concat([df[target], df_b], axis= 1)   
    df_c = df_c.loc[:, ~df_c.columns.duplicated()]
    shap_values = Feature_Selector.feature_importance(normalize= False)[0]
    accepted_features = Feature_Selector.accepted
    importance_means = pd.DataFrame(index= selected_features, data= shap_values[:len(selected_features)], columns= ['Importance'])
    return df_c, importance_means

#交差項のみの関数
def addCross(df: pd.DataFrame, first_element):
    df_elements = df.loc[:, first_element:]    
    columns = df_elements.columns

    for i, c1 in enumerate(columns):
        #df[c1 + '^2'] = df[c1] ** 2
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = df[c1] * df[c2]
    return df

#metal_xの列順を組成順に入替える
def sort_metals_ratios(row):
    # メタルと比率の列名を動的に特定
    metal_cols = [col for col in row.index if 'metal' in col]
    ratio_cols = [col for col in row.index if 'ratio' in col]
    
    # メタルと比率の列をペアで取得
    pairs = list(zip(row[metal_cols], row[ratio_cols]))
    # 比率に従って降順にソート
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    # ソートされた値を分解して新しい行に割り当て
    sorted_metals, sorted_ratios = zip(*sorted_pairs)
    # 新しい行をSeriesとして返す
    return pd.Series(sorted_metals + sorted_ratios)

#metal_xの列名を修正する
def metal_x_columns_name(df):
    # 列数が偶数かどうか確認
    if len(df.columns) % 2 != 0:
        raise ValueError("metal_xの列数は偶数です。")
    # 列数の半分
    n = len(df.columns) // 2
    # 新しい列名の生成
    new_columns = [f'metal{i+1}' if i < n else f'ratio{i-n+1}' for i in range(2*n)]
    # DataFrameの列名の更新
    df.columns = new_columns
    return df    

#相関係数の大きい特徴量を削除する
def delete_high_corr(df, threshold):
    corr_matrix = df.corr().abs()
    upper_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k= 1).astype(np.bool_))
    drop_col = [col for col in upper_matrix.columns if any(upper_matrix[col] > threshold)]
    reduced_df = df.drop(drop_col, axis= 1)
    return reduced_df    

def calc_EME(element_pair, element_ratio, plot_fig):
    if 'Rh' in element_pair and 'Mn' in element_pair:
        return np.nan, np.nan
    elif 'Mn' in element_pair and 'K' in element_pair:
        return np.nan, np.nan
    #elif 'Mn' in element_pair and 'W' in element_pair:
    #    return np.nan, np.nan    
    elif 'La' in element_pair and 'K' in element_pair:
        return np.nan, np.nan        
    else:
        #定数の定義
        T = 298.15
        P = 101325  #標準状態とする
        #composition_at_calculating_mixing_energy = 0.01
        #データ読込みとphase element_state_dbの作成
        element_state_db = pd.read_csv('tdb_files/single_element_unary.csv', na_values=[], keep_default_na=False)
        element_state_db = element_state_db[element_state_db['Version'] == 'ELEMENT']
        element_state_db.index = element_state_db['5']
        element_state_db = element_state_db.loc[:, 'of']
        element_state_db = element_state_db.iloc[2:]
        element_state_db.name = 'phase'
        #element_pairに相当したDBを選択する まずは選択される蓋然性の高いpairのみ
        element_pair = [element.upper() for element in element_pair]
        COST_list = ['AL','CE','CR','CU','FE','MO','NI','SI']
        mc_al_list = ['AL','CU','MG','SI','ZN','ZR']
        mc_fe_list = ['CO','CR','FE','HF','LA','MN','NB','NI','TI','V','W']
        if 'CE' in element_pair and 'Y' in element_pair:
            tdb = 'Ce_Y.tdb'
        if element_pair[0] in mc_al_list and element_pair[1] in mc_al_list:
            tdb = 'mc_al_v2.032.pycalphad.tdb'
        if 'MN' in element_pair and 'RH' not in element_pair:
            tdb = 'PrecHiMn-04_2.pycalphad.tdb'
        if element_pair[0] in mc_fe_list and element_pair[1] in mc_fe_list:
            tdb = 'mc_fe_v2.059.pycalphad.tdb'
        if 'MN' in element_pair and 'W' in element_pair:
            tdb = 'mc_fe_v2.059.pycalphad.tdb'
        if 'MN' in element_pair and 'NA' in element_pair:
            tdb = 'SGTE-unary-1991-2010.tdb'
        if 'NI' in element_pair:
            tdb = 'mc_ni_v2.034.pycalphad.tdb'
        if 'CE' in element_pair and 'Y' not in element_pair:
            tdb = 'COST507_modified.tdb'
        if 'LA' in element_pair:
            tdb = 'SGTE-unary1991-2010.tdb'
        else:
            tdb = 'SGTE-unary1991-2010.tdb'
        #tdb = 'mc_al_v2.032.pycalphad.tdb'
        if tdb == 'mc_fe_v2.059.pycalphad.tdb':
            with open('tdb_files/'+tdb, 'r', encoding= 'utf-8') as f:
                dbf = Database(f.read().upper())
        else:
            dbf = Database('tdb_files/'+tdb)
        #element_permutation = list(itertools.permutations(elements_in_db, 2))
        #diff_index = [''.join(element) for element in element_permutation]
        #diff_df = pd.DataFrame(index= diff_index, columns= ['diff'])
        #element_permutation = [list(combi) + ['VA'] for combi in element_permutation]
        #element_ratioに基づいてelement_pairの並び替え
        element_ratio = [s/sum(element_ratio) for s in element_ratio]
        if element_ratio[0] < element_ratio[1]:
            element_ratio = [element_ratio[1], element_ratio[0]]
            element_pair = [element_pair[1], element_pair[0]]
        #EMEの計算
        #for comps in element_permutation:
        comps = element_pair + ['VA']
        phase = element_state_db.loc[comps[0]]
        if tdb == 'Ce_Y.tdb' and phase == 'FCC_A1':
            phase = 'FCC'
        elif tdb == 'Ce_Y.tdb' and phase == 'HCP_A3':
            phase = 'HCP'                  
        if phase == 'DIAMOND_A4':
            phase = 'SI_DIAMOND_A4'
        phases = [phase, 'LIQUID']
        conditions = {v.X(comps[1]):(0,1,0.001), v.T: T, v.P:P}
        #自由エネルギーの計算
        #GM, HM, SM, CPM, GM_MIX etc.を出力可能 但し符号の意味はGM以外説明されていない
        #GMは自由エネルギー, HMはエンタルピー、SMはエントロピー、CPMは定圧比熱と推定 Mはモル単位の意味
        #result_FCC = calculate(dbf, comps, 'FCC_A1', P= P, T= T, output= 'GM')
        #result_LIQ = calculate(dbf, comps, 'LIQUID', P= P, T= T, output= 'GM')
        #データが出たときは計算、図作成、出ないときはskip
        result_mix = calculate(dbf, comps, phase, P= P, T= T, output= 'GM_MIX')
        #result_LIQ_mix = calculate(dbf, comps, 'LIQUID', P = P, T = T, output= 'GM_MIX')
        data_GMmix = result_mix.GM_MIX.values.flatten()
        data_X = result_mix.X.values[..., 0].flatten()
        df_GMmix = pd.DataFrame(index= data_X, data= data_GMmix, columns= ['GMmix'])
        try:
            aiming_index = np.abs(df_GMmix.index - element_ratio[1]).argmin()
            compo_before = df_GMmix.index[aiming_index - 1]
            value_before = df_GMmix.iloc[aiming_index - 1]['GMmix']
            compo_after = df_GMmix.index[aiming_index + 1]
            value_after = df_GMmix.iloc[aiming_index + 1]['GMmix']
            diff = (value_after - value_before) / (compo_after - compo_before)
            EME_value = df_GMmix.iloc[aiming_index]['GMmix']
            #diff_df.at[(comps[0]+comps[1]), 'diff'] = diff
            if plot_fig:
                fig = plt.figure(figsize= (9, 6))
                ax = fig.gca()
                ax.scatter(result_mix.X.sel(component= comps[1]), result_mix.GM_MIX, marker='.', s=5, color= 'b')
                #ax.scatter(result_LIQ_mix.X.sel(component= comps[1]), result_LIQ_mix.GM_MIX, marker='.', s=5, color= 'r')
                ax.set_xlabel('X'+comps[1], fontsize= 16)
                ax.set_ylabel('GM_MIX', fontsize= 16)
                ax.tick_params(axis= 'x', labelsize= 16)
                ax.tick_params(axis= 'y', labelsize= 16)
                ax.set_xlim((0, 1))
                ax.set_title(comps[0]+ '-' + comps[1], fontsize= 16)
                plt.tight_layout()
                plt.show()
        except:
            print(comps[0]+'-'+comps[1]+'の系はデータがありません。')
            diff = np.nan
            EME_value = np.nan
        """
        #平衡計算:Xは平衡時の各相組成を与える
        eq_result = equilibrium(dbf, comps, phases, conditions)
        eq_compo = eq_result.X.where(eq_result.Phase == 'FCC_A1').sel(P=P, T=T).values
        eq_compo = eq_compo[~np.isnan(eq_compo)]

        #excess_mixing_energyの計算
        model= Model(dbf, comps, phase_name= 'FCC_A1')
        EME = model.excess_mixing_energy(dbf)
        EME = np.array(EME)
        print(EME)
        """
        #元素名を大文字から小文字へ変換
        #diff_df_index = diff_df.index.to_list()
        #new_index = ['-'.join([item[i:i+2].capitalize() for i in range(0, len(item), 2)]) for item in diff_df_index]
        #diff_df.index = new_index
        return diff, EME_value

#x1に相互作用項を付与したようなx3を作成する
def Calc_x3(y, x2, metal_x, xenonpy_original, xenonpy_additional, selector, compo, perc, plot_fig):
    #metal_xの列名の変更
    if metal_x.columns.to_list()[0] != 'metal1':
        metal_x = metal_x_columns_name(metal_x)
    #xenonpy_baseの作成
    metal_columns = [col for col in metal_x.columns if 'metal' in col]
    element_list = pd.unique(metal_x[metal_columns].values.ravel('K'))
    element_list = list(set(element_list))
    element_list = sorted([s for s in element_list if s != 0 and not pd.isna(s)])
    xenonpy_base = xenonpy_original.loc[element_list]
    #xenonpy_additional = xenonpy_additional.loc[element_list].drop('oxide', axis= 1)
    #xenonpy_additional = xenonpy_additional.drop('Te')
    
    xenonpy_base = pd.merge(xenonpy_base, xenonpy_additional, left_index= True, right_index= True)
    xenonpy_base = xenonpy_base.dropna(how= 'any', axis= 1)
    #xenonpy_base = xenonpy_base.drop(['hhi_p', 'hhi_r', 'period'], axis= 1)
    #相関の高い変数を削除する
    xenonpy_base = delete_high_corr(xenonpy_base, threshold= 0.95)
    
    #metal_xを組成順に並べる
    metal_x = metal_x.replace(np.nan, 0)
    new_columns = metal_x.columns.to_list()
    # applyを使って各行に対してソート関数を適用し、列名を更新
    metal_x[new_columns] = metal_x.apply(sort_metals_ratios, axis=1)
    metal_x.columns = new_columns    
    if xenonpy_base.isna().any().any():
        print('dataにnanが含まれています')
        sys.exit()
    #各レイヤーの作成
    weighted_average_name = list() # 加重平均の index 名
    weighted_variance_name = list() # 加重分散の index 名
    geometric_mean_name = list() # 幾何平均の index 名
    harmonic_mean_name = list() # 調和平均の index 名
    max_pooling_name = list() # 最大値の index 名
    min_pooling_name = list() # 最小値の index 名
    max_composition_name = list() #最大組成値の index 名
    for j in xenonpy_base.columns:
        weighted_average_name.append(f'ave_{j}')
        weighted_variance_name.append(f'var_{j}')
        geometric_mean_name.append(f'gmean_{j}')
        harmonic_mean_name.append(f'hmean_{j}')
        max_pooling_name.append(f'max_{j}')
        min_pooling_name.append(f'min_{j}')
        max_composition_name.append(f'maxcompo_{j}')

    x3_metaldesc = pd.DataFrame(
        index=metal_x.index, dtype= float,
        columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name+max_composition_name
        )
    for i in range(metal_x.shape[0]):
        # メタルと比率の列名を動的に特定
        metal_cols = [col for col in metal_x.columns if 'metal' in col]
        ratio_cols = [col for col in metal_x.columns if 'ratio' in col]
        # 各行からメタルと比率の値を取得
        metals = metal_x.loc[metal_x.index[i], metal_cols].values
        ratios = metal_x.loc[metal_x.index[i], ratio_cols].values
        # 無効なメタル（値が0または'na'）とそれに対応する比率を除外
        valid_indices = [index for index, metal in enumerate(metals) if metal not in (0, 'na', 'NaN', np.nan)]
        valid_metals = [metals[index] for index in valid_indices]
        valid_ratios = [ratios[index] for index in valid_indices]
        # xenonpy_baseから対応するメタルの記述子を取得し、NumPy配列に格納
        metal_descs = [xenonpy_base.loc[metal, :].values for metal in valid_metals]
        mt = np.array(metal_descs)
        mr = np.array(valid_ratios)

        for desc in range(xenonpy_base.shape[1]):
            d_name = xenonpy_base.columns[desc]
            x3_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
            x3_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)/np.sum(mr)
            x3_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
            x3_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
            x3_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
            x3_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])
            if len(mr) == 1:
                x3_metaldesc[f'maxcompo_{d_name}'].iloc[i] = max(mt[:, desc])
            elif mr[0] != mr[1]:
                x3_metaldesc[f'maxcompo_{d_name}'].iloc[i] = mt[0, desc]
            else:
                x3_metaldesc[f'maxcompo_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)      
    #descの値が負であるものはgmeanでnanとなるため、その列はdropする(負の値のものは計算しないようにするのはかなり面倒)
    x3_metaldesc = x3_metaldesc.dropna(how= 'any', axis= 1)      
                  
    #x2を用いて交差項元素を決定する
    first_element = next((s for s in x2.columns.to_list() if s in element_list), None)
    x2_cross = addCross(x2, first_element)
    cross_elements = [s for s in x2_cross.columns.to_list() if '*' in s] 
    x3_crossdesc = pd.DataFrame(index= metal_x.index)    
    # cross_elementsの各要素とxenonpy_baseの列を合わせた列をx3_crossdescに作成
    # 加重平均値を計算する関数
    def calculate_weighted_avg(existing_elements, col):
        if existing_elements:
            values = [xenonpy_base.at[el, col] for el, ratio in existing_elements]
            ratios = [ratio for el, ratio in existing_elements]
            return np.average(values, weights=ratios)
        return np.nan

    # 列の作成と値の計算
    for element_pair in cross_elements:
        elements = element_pair.split(' * ')
        for col in xenonpy_base.columns:
            col_p = f"{element_pair.replace(' * ', '*')}_{col}_p"
            col_n = f"{element_pair.replace(' * ', '*')}_{col}_n"
            x3_crossdesc[col_p] = np.nan
            x3_crossdesc[col_n] = np.nan
        x3_crossdesc[element_pair.replace(' * ', '*')+'_EME_diff'] = np.nan
        x3_crossdesc[element_pair.replace(' * ', '*')+'_EME'] = np.nan

    # 相互作用項の計算
    for index, row in metal_x.iterrows():
        print(index, len(metal_x))
        metal_list = [row[f'metal{i}'] for i in range(1, len(row)//2+1) if pd.notna(row[f'metal{i}'])]
        ratio_list = [row[f'ratio{i}'] for i in range(1, len(row)//2+1) if pd.notna(row[f'ratio{i}'])]
        existing_elements = [(metal_list[i], ratio_list[i]) for i in range(len(metal_list)) if metal_list[i] in xenonpy_base.index]
        
        for element_pair in cross_elements:
            elements = element_pair.split(' * ')
            # element_pairに一致する要素があるか確認
            matched_elements = [el for el in elements if el in [el[0] for el in existing_elements]]
            matched_list = [el for el in existing_elements if el[0] in matched_elements]    
            #EME(ExcessMixingEnergy)を入力する
            if len(matched_elements) == len(elements):
                matched_compo = [compo for el, compo in existing_elements if el in matched_elements]
                diff, EME_value = calc_EME(matched_elements, matched_compo, plot_fig)
                x3_crossdesc.at[index, element_pair.replace(' * ', '*')+'_EME_diff'] = diff
                x3_crossdesc.at[index, element_pair.replace(' * ', '*')+'_EME'] = EME_value    
            else:
                x3_crossdesc.at[index, element_pair.replace(' * ', '*')+'_EME_diff'] = 0
                x3_crossdesc.at[index, element_pair.replace(' * ', '*')+'_EME'] = 0 

        for element_pair in cross_elements:
            elements = element_pair.split(' * ')
            # element_pairに一致する要素があるか確認
            matched_elements = [el for el in elements if el in [el[0] for el in existing_elements]]
            matched_list = [el for el in existing_elements if el[0] in matched_elements]            
            
            for col in xenonpy_base.columns:
                col_p = f"{element_pair.replace(' * ', '*')}_{col}_p"
                col_n = f"{element_pair.replace(' * ', '*')}_{col}_n"
                # 2つとも一致するときはスケーリングしたうえで加重平均と相互作用を計算する
                if len(matched_elements) == len(elements):
                        value_array = np.array([xenonpy_base.at[el, col] for el, ratio in existing_elements]).reshape(-1, 1)
                        ratio_array = np.array([ratio for el, ratio in existing_elements])
                        matched_value = np.array([xenonpy_base.at[el, col] for el, ratio in matched_list]).reshape(-1, 1)
                        # Scaling
                        scaler = StandardScaler()
                        standardized_values = scaler.fit_transform(value_array).flatten()
                        # スケーリングした値で計算
                        weighted_avg = np.average(standardized_values, weights= ratio_array)
                        matched_info = [(element, weight) for element, weight in existing_elements if element in matched_elements]
                        matched_indices = [i for i, (element, _) in enumerate(existing_elements) if element in matched_elements]
                        matched_values = standardized_values[matched_indices]
                        matched_weights = np.array([weight for _, weight in matched_info])
                        if compo == 'relative':
                            matched_weights = matched_weights/ matched_weights.sum()    
                        products = matched_values * matched_weights
                        interaction = np.prod(products)
                        # スケーリングを戻す
                        weighted_avg_rescaled = weighted_avg * scaler.scale_[0] + scaler.mean_[0]
                        interaction_rescaled = interaction * scaler.var_[0] + scaler.mean_[0] * matched_weights[0] * matched_weights[1]\
                            * (matched_value[0][0] + matched_value[1][0] - scaler.mean_[0])
                        interaction_rescaled = np.sqrt(interaction_rescaled)  #組成の積になっているため√をとってdimensionを戻す
                        x3_crossdesc.at[index, col_p] = weighted_avg_rescaled + interaction_rescaled
                        x3_crossdesc.at[index, col_n] = weighted_avg_rescaled - interaction_rescaled                   
                else:  # element_pairと完全一致でない場合
                    weighted_avg = calculate_weighted_avg(existing_elements, col)  # 存在する全要素の加重平均
                    x3_crossdesc.at[index, col_p] = weighted_avg
                    x3_crossdesc.at[index, col_n] = weighted_avg
                        
    first_element_loc = x2.columns.get_loc(first_element)
    final_desc = pd.concat([x3_metaldesc, x3_crossdesc], axis= 1)
    nan_check = final_desc.isna().sum()
    if final_desc.isna().any().any():
        nan_number = final_desc.isna().any().sum()
        final_desc = final_desc.dropna(axis= 1)
        print('Dropped columns containing nan:' + str(nan_number))  
    print('number of columns before boruta:' + str(final_desc.shape[1]))  
    
    desc_for_boruta = pd.concat([x2.iloc[:, : first_element_loc], final_desc], axis= 1)
    desc_for_boruta.to_csv('datasets/v453/desc_for_boruta.csv', encoding= 'utf-8-sig')
    if selector == 'Boruta':
        desc_boruta, importance_df = Boruta_Apply(target, desc_for_boruta, perc= perc)
    elif selector == 'Boruta2':
        desc_boruta, importance_df = Boruta_Apply2(target, desc_for_boruta, perc= perc)
    elif selector == 'BorutaShap':
        desc_boruta, importance_df = BorutaShap_Apply(target, desc_for_boruta, perc= perc)
    first_desc = next((s for s in desc_boruta.columns.to_list() if s in final_desc), None)    
    final_df = pd.concat([x2.iloc[:, :first_element_loc], desc_boruta.loc[:, first_desc:]], axis= 1)
    duplicated_list = [s for s in final_df.columns if '.1' in s]
    if len(duplicated_list) != 0:
        final_df = final_df.drop(duplicated_list, axis= 1)
    
    return final_df


"""
#tdbに収録されている元素種全てを取得
tdb_DB = pd.read_csv('tdb_files/tdb_DB.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
tdb_DB = tdb_DB[tdb_DB['readable'] == True]
tdb_DB = tdb_DB.drop(['readable', 'temperature/K'], axis= 1)
tdb_elements = tdb_DB.values.flatten()
tdb_elements = list(set(list(tdb_elements)))
drop_list = ['O','N','P','S','AR','C','B','BR','CL','F','H','HE','I','KR','NE','XE']
tdb_elements = pd.Series(tdb_elements)
tdb_elements = tdb_elements.dropna().tolist()
#tdb_elements = [s for s in tdb_elements if s not in drop_list and not np.isnan(s)] #240604現在27元素
tdb_elements.sort()
"""
"""
#tdbの読込み確認
#with open('tdb_files/steel_database_fix.tdb', 'rb') as f:
#    result = chardet.detect(f.read())
#print("検出されたエンコーディング:", result['encoding'])
#db = pd.read_csv('tdb_files/steel_database_fix.tdb', encoding= 'utf-8', sep='\t')
#db.to_csv('tdb_files/steel_database_fix.tdb', encoding= 'utf-8')
#dbf = Database('tdb_files/steel_database_fix.tdb')
with open('tdb_files/Cr_Ir_unreadable.tdb', 'r',encoding= 'utf-8') as f:
    dbf = Database(f.read().upper())
"""
"""
#EME関数の動作確認
element_pair = ['W','Mn']
element_ratio = [0.4, 0.28]
EME = calc_EME(element_pair, element_ratio, plot_fig= True)
"""
"""
#Borutaのみ動かして説明変数の数を減らす
x4_old = pd.read_csv('datasets/v453/x4a_9r.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
#x4_old = x4_old.iloc[:600, :]
target= ['選択率NPA', '収率NPA'] 
x4_new, importance_df = Boruta_Apply2(target, x4_old, perc= 99)
x4_new_first_feature = [s for s in x4_new.columns if s.startswith('ave') or s.startswith('var')]
x4_new = pd.concat([x4_old.loc[:, :'support_ZrO2_RC100'], x4_new.loc[:, x4_new_first_feature[0]:]], axis= 1)
x4_new.to_csv('datasets/v453/x4a_99r.csv', encoding= 'utf-8-sig')
"""
"""
#Borutaのみ動かして説明変数の数を減らす percだけでなくimportanceを利用
#相関係数で絞ろうかとしたが、説明変数間の相関係数はそう大きくなかった。
x4_old = pd.read_csv('datasets/v453/x4a_9r.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
#x4_old = x4_old.iloc[:600, :]
target= ['選択率NPA', '収率NPA'] 
x4_new, importance_df = Boruta_Apply2(target, x4_old, perc= 99)
threshold = 50 #importance_dfの上位50%を抽出
threshold_index = int(len(importance_df) * (threshold / 100))
selected_features = importance_df.iloc[: threshold_index].index.tolist()
x4_new = pd.concat([x4_old.loc[:, : 'support_ZrO2_RC100'], x4_new.loc[:, selected_features]], axis= 1)
x4_new = x4_new.loc[:, ~x4_new.columns.duplicated()]
x4_new.to_csv('datasets/v453/x4a_3r.csv', encoding= 'utf-8-sig')
"""
"""
#データセット内容の確認
x4a_9r = pd.read_csv('datasets/v453/x4a_9r.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x4_5r = pd.read_csv('datasets/v453/x4_5r.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
columns_x4a = set(x4a_9r.columns)
columns_x4b = set(x4_5r.columns)
common_columns = list(columns_x4a & columns_x4b)
only_in_x4a = list(columns_x4a - columns_x4b)
only_in_x4b = list(columns_x4b - columns_x4a)
"""

xenonpy_original = pd.read_csv('results/xenonpy_element_data240515.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
xenonpy_additional = pd.read_csv('results/xenonpy_additional.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
webelements_data = pd.read_csv('results/webelements_data_normal_strings_0515.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
element_data = pd.read_csv('results/element_data_normal_string240527.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
thermo_data = pd.read_csv('results/thermo_data.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
metal_x = pd.read_csv('MyWork/datasets/metal_x.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x2 = pd.read_csv('MyWork/datasets/x2.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
reactive_elements_list = ['Li','Na','K','Rb','Cs','Be','Mg','Ca','Sr','Ba','Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
#NaMn_data = pd.read_csv('MyWork/datasets/NaMn_data.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
target= ['C2 yield'] 
metal_x = metal_x.replace('na', 0)
metal_col = [s for s in metal_x.columns if 'Amount' not in s]
metal_list = pd.unique(metal_x[metal_col].values.flatten())
metal_list = pd.Series(metal_list).dropna().to_list()
metal_list.remove(0)
#element_data内容の確認
element_data = element_data.loc[element_data.index.isin(metal_list)]
element_data = element_data.loc[:, element_data.isnull().mean() < 0.2]
check_nan_a = element_data.isna().sum()
element_data = element_data.dropna(axis= 1)
element_data = element_data.drop(['space group', 'structure','oxide','hhi_p','hhi_r'], axis= 1)
#element_data['calc'] = element_data['CpH of solid'] / element_data['H 298.15-H 0 of solid']
thermo_data = thermo_data.loc[thermo_data.index.isin(metal_list)]
thermo_data = thermo_data.loc[:, : 'Stockmayer parameter, K']
thermo_data = thermo_data.drop(['CAS', 'Heat of formation, J/mol'], axis= 1)
thermo_data = thermo_data.loc[:, thermo_data.isnull().mean() < 0.2]
check_nan_b = thermo_data.isna().sum()
thermo_data = thermo_data.dropna(axis= 1)
element_data = delete_high_corr(element_data, threshold= 0.95)
thermo_data = delete_high_corr(thermo_data, threshold= 0.95)
xenonpy_original =element_data
xenonpy_additional = thermo_data
"""
# 2元素dfのときのmetal_xを作成
metal_x = pd.DataFrame(index= NaMn_data.index, columns= ['metal1','metal2','ratio1','ratio2'])
metal_x['metal1'] = 'Na'
metal_x['metal2'] = 'Mn'
metal_x['ratio1'] = NaMn_data['Na']
metal_x['ratio2'] = NaMn_data['Mn']
# NaWのデータ形式の調整
new_columns = [c for c in NaMn_data.columns if c not in ['Na','Mn']] + ['Na','Mn']
NaMn_data.columns = new_columns
"""
# active_metalはないものとして扱う
drop_list = list(set(metal_list) & set(reactive_elements_list))
x2 = x2.drop(drop_list, axis= 1)
x2_compo = x2.loc[:, 'Co': ]
def adjust_compo(row):
    total = row.sum()
    if total != 1.0:
        row = row / total
    return row
x2_compo = x2_compo.apply(adjust_compo, axis= 1)
x2 = pd.concat([x2.loc[:, : 'Support_SiO5'], x2_compo], axis= 1)    
# metal_xもactive_metalは削除する
def create_metalx(row):
    sorted_elements = row[row > 0].sort_values(ascending= False)
    metals = sorted_elements.index.tolist()
    ratios = sorted_elements.values.tolist()
    metals.extend([0.0] * (3 - len(metals)))
    ratios.extend([0.0] * (3 - len(ratios)))
    return pd.Series(metals + ratios, index= ['metal1','metal2','metal3','ratio1','ratio2','ratio3'])
metal_x = x2_compo.apply(create_metalx, axis= 1)    
    
#metal_x = metal_x.iloc[152:160, :]
#x2 = x2.iloc[152: 160, :]
#scalingは、standard(標準化)とnormal(正規化)、compoは、relative(相対値)とabsolute(絶対値)から選択する
#selectorは、Boruta, Boruta2, BorutaShapの3つから選択する
#x3 = Calc_x3(target, x2, metal_x, xenonpy_original, xenonpy_additional, selector= 'BorutaShap', compo= 'absolute', perc= 90)
#Rh_EMEはEME計算をRhを含むものについてはomitするか否かを決めるもの Rh_EME= False とすると、Rhを含まないものでEME計算する
x4 = Calc_x3(target, x2, metal_x, xenonpy_original, xenonpy_additional, selector = 'Boruta', compo= 'relative', perc= 99, plot_fig= False)
x4 = x4.dropna(axis= 0)
x4 = x4.loc[:, ~x4.columns.duplicated()]
x4.to_csv('MyWork/datasets/x4_wo_acmet_99.csv', encoding= 'utf-8-sig')

print('hello')