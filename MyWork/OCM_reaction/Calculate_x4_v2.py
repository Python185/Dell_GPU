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

def remove_correlated_features(df, threshold):
    corr_matrix = df.corr().abs()  # 絶対値の相関行列
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # 上三角行列のみを取得
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]  # 閾値を超える変数を取得
    return df.drop(columns=to_drop)  # 高相関の変数を削除

def Boruta_Apply(target, df, perc):
    y= df[target].copy()
    #y = df['転化率']
    x = df.drop(target, axis= 1).copy()
    # 相関係数大きいものは削除
    x = remove_correlated_features(x, threshold=0.95)
    
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

def BorutaShap_Apply(target, df, perc):
    datatype = df.dtypes
    y= df[target].copy()
    x = df.drop(target, axis= 1).copy()
    # 相関係数大きいものは削除
    x = remove_correlated_features(x, threshold=0.95)
    
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
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            sorted_cols = sorted([c1, c2])
            df[sorted_cols[0] + ' * ' + sorted_cols[1]] = df[c1] * df[c2]
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

# x2をmetal_xに変換する
def create_metalx(row):
    sorted_elements = row[row > 0].sort_values(ascending= False)
    metals = sorted_elements.index.tolist()
    ratios = sorted_elements.values.tolist()
    metals.extend([0.0] * (3 - len(metals)))
    ratios.extend([0.0] * (3 - len(ratios)))
    return pd.Series(metals + ratios, index= ['metal1','metal2','metal3','ratio1','ratio2','ratio3'])

# metal_xのmetal1-3, ratio1-3をアルファベット順に変更する
def sort_metal_ratios(row):
    metals = row[['metal1', 'metal2', 'metal3']].tolist()
    ratios = row[['ratio1', 'ratio2', 'ratio3']].tolist()
    sorted_pairs = sorted(zip(metals, ratios), key=lambda x: (str(x[0]), x[1]))
    sorted_metals, sorted_ratios = zip(*sorted_pairs)
    row[['metal1', 'metal2', 'metal3']] = sorted_metals
    row[['ratio1', 'ratio2', 'ratio3']] = sorted_ratios
    return row

# ReactiveElementsをOmitする
def RemoveReactiveMetal(df, reactive_elements_list):
    elements_cols = [c for c in df.columns if len(c) <= 2]
    df_elements = df.loc[:, elements_cols].copy()
    drop_elements = list(set(elements_cols) & set(reactive_elements_list))
    df_elements = df_elements.drop(drop_elements, axis= 1)
    def adjust_compo(row):
        total = row.sum()
        if total != 1.0:
            row = row / total
        return row
    df_elements = df_elements.apply(adjust_compo, axis= 1)
    return_df = pd.concat([df.drop(elements_cols, axis= 1), df_elements], axis= 1)
    return return_df

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
def Calc_x3(y, x2, metal_x, element_data, oxide_data, selector, cross_term, compo, perc):
    #metal_xの列名の変更
    if metal_x.columns.to_list()[0] != 'metal1':
        metal_x = metal_x_columns_name(metal_x)
    #xenonpy_baseの作成 全てreactive metalであればxenonpy_baseは空となる
    metal_columns = [col for col in metal_x.columns if 'metal' in col]
    element_list = pd.unique(metal_x[metal_columns].values.ravel('K'))
    element_list = list(set(element_list))
    element_list = sorted([s for s in element_list if s != 0 and not pd.isna(s)])
    xenonpy_base = element_data.loc[~element_data.index.isin(oxide_data.index.tolist())]
    if xenonpy_base.isna().any().any():
        print('dataにnanが含まれています')
        sys.exit()
    
    #metal_xを組成順に並べる
    metal_x = metal_x.replace(np.nan, 0)
    new_columns = metal_x.columns.to_list()
    # applyを使って各行に対してソート関数を適用し、列名を更新
    metal_x[new_columns] = metal_x.apply(sort_metals_ratios, axis=1)
    metal_x.columns = new_columns    

    reactive_metals = oxide_data.index.tolist()
    # metal_xをmetal_xとoxide_xに分割する
    def split_metal_x(row, reactive_metals):
        metal_gr = {f'm_metal{i+1}': row[f'metal{i+1}'] for i in range(3) if row[f'metal{i+1}'] not in reactive_metals}
        metal_gr.update({f'm_ratio{i+1}': row[f'ratio{i+1}'] for i in range(3) if row[f'metal{i+1}'] not in reactive_metals})

        oxide_gr = {f'o_metal{i+1}': row[f'metal{i+1}'] for i in range(3) if row[f'metal{i+1}'] in reactive_metals}
        oxide_gr.update({f'o_ratio{i+1}': row[f'ratio{i+1}'] for i in range(3) if row[f'metal{i+1}'] in reactive_metals})
        
        return pd.Series({**metal_gr, **oxide_gr})

    # metal_xの各行を分割
    split_df = metal_x.apply(split_metal_x, axis=1, reactive_metals= reactive_metals)
    # 不要なNaN列を削除してグループを抽出
    metal_x = split_df[[col for col in split_df.columns if col.startswith('m_')]].dropna(axis=1, how='all')
    oxide_x = split_df[[col for col in split_df.columns if col.startswith('o_')]].dropna(axis=1, how='all')
    if 'm_metal3' not in metal_x.columns:
        metal_x[['m_metal3, m_ratio3']] = np.nan
        col = ['m_metal'+str(c) for c in range(1,4)] + ['m_ratio'+str(c) for c in range(1,4)]
        metal_x = metal_x[col]
    for col in ['o_metal1', 'o_metal2', 'o_metal3']:
        if col not in oxide_x.columns:
            oxide_x[col] = np.nan
    for col in ['o_ratio1', 'o_ratio2', 'o_ratio3']:
        if col not in oxide_x.columns:
            oxide_x[col] = np.nan
    col = ['o_metal'+str(c) for c in range(1,4)] + ['o_ratio'+str(c) for c in range(1,4)]
    oxide_x = oxide_x[col]
    
    if metal_x.shape[1] != 6:
        expected_columns = [f'm_metal{i}' for i in range(1, 4)] + [f'm_ratio{i}' for i in range(1, 4)]
        for col in expected_columns:
            if col not in metal_x.columns:
                metal_x[col] = np.nan
        metal_x = metal_x.reindex(columns=expected_columns)
            
    # metal_x, oxide_xの変形(元素を詰める)
    m_col = [c[2:] for c in metal_x.columns]
    o_col = [c[2:] for c in oxide_x.columns]
    metal_x.columns = m_col
    oxide_x.columns = o_col

    # dfに欠損値あるため左詰め
    for index, row in metal_x.iterrows():
        metals = row[['metal1', 'metal2', 'metal3']].tolist()
        ratios = row[['ratio1', 'ratio2', 'ratio3']].tolist()
        # 有効なmetalとratioのペアを保持
        filled_metals = []
        filled_ratios = []
        for metal, ratio in zip(metals, ratios):
            if pd.notna(metal) and pd.notna(ratio):
                filled_metals.append(metal)
                filled_ratios.append(ratio)
        # 補完された値で行を再構成
        filled_metals += [np.nan] * (len(metals) - len(filled_metals))
        filled_ratios += [np.nan] * (len(ratios) - len(filled_ratios))
        # 元のDataFrameに戻す
        metal_x.loc[index, ['metal1', 'metal2', 'metal3']] = filled_metals
        metal_x.loc[index, ['ratio1', 'ratio2', 'ratio3']] = filled_ratios

    for index, row in oxide_x.iterrows():
        metals = row[['metal1', 'metal2', 'metal3']].tolist()
        ratios = row[['ratio1', 'ratio2', 'ratio3']].tolist()
        # 有効なmetalとratioのペアを保持
        filled_metals = []
        filled_ratios = []
        for metal, ratio in zip(metals, ratios):
            if pd.notna(metal) and pd.notna(ratio):
                filled_metals.append(metal)
                filled_ratios.append(ratio)
        # 補完された値で行を再構成
        filled_metals += [np.nan] * (len(metals) - len(filled_metals))
        filled_ratios += [np.nan] * (len(ratios) - len(filled_ratios))
        # 元のDataFrameに戻す
        oxide_x.loc[index, ['metal1', 'metal2', 'metal3']] = filled_metals
        oxide_x.loc[index, ['ratio1', 'ratio2', 'ratio3']] = filled_ratios        
    
    # metal_xにデータがないときはmetaldescの計算はskipする
    metal_no_data = (metal_x.isna() | (metal_x ==0)).all().all()
    oxide_no_data = (oxide_x.isna() | (oxide_x ==0)).all().all()
    
    #if metal_no_data == False:
    #    metal_x.to_csv('MyWork/datasets/metal_x_Sn.csv', encoding= 'utf-8')
    #if oxide_no_data == False:
    #    oxide_x.to_csv('MyWork/datasets/oxide_x_Sn.csv', encoding= 'utf-8')
    
        
    #各レイヤーの作成
    if metal_no_data == False:
        if metal_x['metal2'].replace(np.nan, 0).sum() == 0:
            x3_metaldesc = pd.DataFrame(index= metal_x.index, dtype= float, columns=xenonpy_base.columns)
            for idx, row in x3_metaldesc.iterrows():
                x3_metaldesc.loc[idx, :] = xenonpy_base.loc[metal_x.loc[idx, 'metal1'], :]
        else:
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
                #valid_indices = [index for index, metal in enumerate(metals) if metal not in (0,'na','NaN','nan',None,np.nan)]
                valid_indices = [index for index, metal in enumerate(metals) if not pd.isna(metal) and metal not in (0, 'na', 'NaN', 'nan', None)]
                valid_metals = [metals[index] for index in valid_indices]
                valid_ratios = [ratios[index] for index in valid_indices]
                
                # metalがないときはnp.nanを入力
                if valid_metals == []:
                    x3_metaldesc.iloc[i] = np.nan
                else:
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
            x3_metaldesc = x3_metaldesc.dropna(how= 'all', axis= 1)   
                  
    #oxide:各レイヤーの作成
    if oxide_no_data == False:
        weighted_average_oxide = list() # 加重平均の index 名
        weighted_variance_oxide = list() # 加重分散の index 名
        geometric_mean_oxide = list() # 幾何平均の index 名
        harmonic_mean_oxide = list() # 調和平均の index 名
        max_pooling_oxide = list() # 最大値の index 名
        min_pooling_oxide = list() # 最小値の index 名
        max_composition_oxide = list() #最大組成値の index 名
        for j in oxide_data.columns:
            weighted_average_oxide.append(f'ave_{j}')
            weighted_variance_oxide.append(f'var_{j}')
            geometric_mean_oxide.append(f'gmean_{j}')
            harmonic_mean_oxide.append(f'hmean_{j}')
            max_pooling_oxide.append(f'max_{j}')
            min_pooling_oxide.append(f'min_{j}')
            max_composition_oxide.append(f'maxcompo_{j}')

        x3_oxidedesc = pd.DataFrame(
            index=oxide_x.index, dtype= float,
            columns=weighted_average_oxide+weighted_variance_oxide+geometric_mean_oxide+harmonic_mean_oxide+max_pooling_oxide+min_pooling_oxide+max_composition_oxide
            )
        for i in range(oxide_x.shape[0]):
            # メタルと比率の列名を動的に特定
            metal_cols = [col for col in oxide_x.columns if 'metal' in col]
            ratio_cols = [col for col in oxide_x.columns if 'ratio' in col]
            # 各行からメタルと比率の値を取得
            metals = oxide_x.loc[oxide_x.index[i], metal_cols].values
            ratios = oxide_x.loc[oxide_x.index[i], ratio_cols].values
            # 無効なメタル（値が0または'na'）とそれに対応する比率を除外
            #valid_indices = [index for index, metal in enumerate(metals) if metal not in (0, 'na', 'NaN', np.nan)]
            valid_indices = [index for index, metal in enumerate(metals) if not pd.isna(metal) and metal not in (0, 'na', 'NaN', 'nan', None)]
            valid_metals = [metals[index] for index in valid_indices]
            valid_ratios = [ratios[index] for index in valid_indices]
            # xenonpy_baseから対応するメタルの記述子を取得し、NumPy配列に格納
            metal_descs = [oxide_data.loc[metal, :].values for metal in valid_metals]
            mt = np.vstack(metal_descs)
            #mt = np.array(metal_descs)
            mr = np.array(valid_ratios)

            for desc in range(oxide_data.shape[1]):
                d_name = oxide_data.columns[desc]
                x3_oxidedesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
                x3_oxidedesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)/np.sum(mr)
                x3_oxidedesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
                x3_oxidedesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
                x3_oxidedesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
                x3_oxidedesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])
                if len(mr) == 1:
                    x3_oxidedesc[f'maxcompo_{d_name}'].iloc[i] = max(mt[:, desc])
                elif mr[0] != mr[1]:
                    x3_oxidedesc[f'maxcompo_{d_name}'].iloc[i] = mt[0, desc]
                else:
                    x3_oxidedesc[f'maxcompo_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)      
        #descの値が負であるものはgmeanでnanとなるため、その列はdropする(負の値のものは計算しないようにするのはかなり面倒)
        x3_oxidedesc = x3_oxidedesc.dropna(how= 'any', axis= 1)      

    if metal_no_data:
        x3_metaldesc = pd.DataFrame(index= metal_x.index)
    if oxide_no_data:
        x3_oxidedesc = pd.DataFrame(index= oxide_x.index)
        
    # metal, oxideごとに相互作用項を計算する EMEは計算止める
    # 組成比が8%以上を対象として、2元素の相互作用を計算する　3元素系は組成上位の2元素について計算する
    # metalまたはoxideデータがなくなる行については、その行を削除する(コードには反映していない)

    if cross_term:
        x3_oxidecross = pd.DataFrame(index= oxide_x.index)
        def calculate_weighted_avg(existing_elements, col):
            if existing_elements:
                values = [oxide_data.at[el, col] for el, ratio in existing_elements]
                ratios = [ratio for el, ratio in existing_elements]
                return np.average(values, weights=ratios)
            return np.nan        
        
        # 列の作成
        for col in oxide_data.columns:
            col_p = f"{col}_p"
            col_n = f"{col}_n"
            x3_oxidecross[col_p] = np.nan
            x3_oxidecross[col_n] = np.nan

        # 相互作用項の計算
        for index, row in oxide_x.iterrows():
            metal_list = [row[f'metal{i}'] for i in range(1, len(row)//2+1) if pd.notna(row[f'metal{i}'])]
            ratio_list = [row[f'ratio{i}'] for i in range(1, len(row)//2+1) if pd.notna(row[f'ratio{i}'])]
            existing_elements = [(metal_list[i], ratio_list[i]) for i in range(len(metal_list)) if metal_list[i] in oxide_data.index]

            def calc_interaction():
                value_array = np.array([oxide_data.at[el, col] for el, ratio in existing_elements]).reshape(-1, 1)
                ratio_array = np.array([ratio for el, ratio in existing_elements])
                # Scaling
                scaler = StandardScaler()
                standardized_values = scaler.fit_transform(value_array).flatten()
                # スケーリングした値で計算
                weighted_avg = np.average(standardized_values, weights= ratio_array)
                (el1, ratio1), (el2, ratio2) = existing_elements
                value1, value2 = standardized_values[:2]
                if compo == 'relative':
                    total_ratio = ratio1 + ratio2
                    ratio1 /= total_ratio
                    ratio2 /= total_ratio    
                products = np.array([ratio1 * value1, ratio2 * value2])
                interaction = np.prod(products)
                # スケーリングを戻す
                weighted_avg_rescaled = weighted_avg * scaler.scale_[0] + scaler.mean_[0]
                interaction_rescaled_ = interaction * scaler.var_[0] + scaler.mean_[0] * ratio1 * ratio2 \
                    * (value1 + value2 - scaler.mean_[0])
                interaction_rescaled = np.sqrt(np.abs(interaction_rescaled_))
                    
                return weighted_avg_rescaled ,interaction_rescaled
                    
            # existing_elementsの数により処理を分岐
            for col in oxide_data.columns:
                col_p = f"{col}_p"
                col_n = f"{col}_n"
                if len(existing_elements) == 1:
                    weighted_avg = calculate_weighted_avg(existing_elements, col)
                    x3_oxidecross.at[index, col_p] = weighted_avg
                    x3_oxidecross.at[index, col_n] = weighted_avg

                elif len(existing_elements) == 2:
                    if existing_elements[0][1] / existing_elements[1][1] > 0.08 or existing_elements[1][1] / existing_elements[0][1] > 0.08:
                        weighted_avg_rescaled, interaction_rescaled = calc_interaction()
                        x3_oxidecross.at[index, col_p] = weighted_avg_rescaled + interaction_rescaled
                        x3_oxidecross.at[index, col_n] = weighted_avg_rescaled - interaction_rescaled   
                    
                elif len(existing_elements) >= 3:
                    existing_elements = sorted(existing_elements, key=lambda x: x[1], reverse=True)[:2]
                    if existing_elements[0][1] / existing_elements[1][1] > 0.08 or existing_elements[1][1] / existing_elements[0][1] > 0.08:
                        weighted_avg_rescaled, interaction_rescaled = calc_interaction()
                        x3_oxidecross.at[index, col_p] = weighted_avg_rescaled + interaction_rescaled
                        x3_oxidecross.at[index, col_n] = weighted_avg_rescaled - interaction_rescaled

    else:
        pass
    
    first_element = next((s for s in x2.columns if s in oxide_data.index), None)                    
    first_element_loc = x2.columns.get_loc(first_element)
    if not x3_metaldesc.empty and not x3_oxidedesc.empty:
        final_desc = pd.concat([x3_metaldesc, x3_oxidedesc], axis= 1)
    elif not x3_oxidedesc.empty:
        final_desc = x3_oxidedesc
    else:
        final_desc = x3_metaldesc
        
    if cross_term:
        final_desc = pd.concat([final_desc, x3_oxidecross], axis= 1)
    else:
        if metal_no_data == False and oxide_no_data == False:
            final_desc = pd.concat([x3_metaldesc, x3_oxidedesc], axis= 1)
        
    nan_check = final_desc.isna().sum()
    if final_desc.isna().any().any():
        nan_number = final_desc.isna().any().sum()
        final_desc = final_desc.dropna(axis= 1)
        print('Dropped columns containing nan:' + str(nan_number))  
    print('number of columns before boruta:' + str(final_desc.shape[1])) 
    
    desc_for_boruta = pd.concat([x2.iloc[:, : first_element_loc], final_desc], axis= 1)
    desc_for_boruta = desc_for_boruta.reindex(sorted(desc_for_boruta.columns), axis= 1)
    if selector == 'Boruta':
        desc_boruta, importance_df = Boruta_Apply(target, desc_for_boruta, perc= perc)
    elif selector == 'Boruta2':
        desc_boruta, importance_df = Boruta_Apply2(target, desc_for_boruta, perc= perc)
    elif selector == 'BorutaShap':
        desc_boruta, importance_df = BorutaShap_Apply(target, desc_for_boruta, perc= perc)
    first_desc = next((s for s in desc_boruta.columns.to_list() if s in final_desc), None)    
    final_df = pd.concat([x2.iloc[:, :first_element_loc], desc_boruta.loc[:, first_desc:]], axis= 1)
    # final_dfの形式整備
    duplicated_list = [s for s in final_df.columns if '.1' in s]
    if len(duplicated_list) != 0:
        final_df = final_df.drop(duplicated_list, axis= 1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    
    return final_df, importance_df

def Calc_support(x4, oxides_data):
    support_col = [c for c in x4.columns if 'Support' in c]
    support_x = [s.split('_')[1] for s in support_col]
    oxides_data = oxides_data[oxides_data['formula'].isin(support_x)]
    oxides_data = oxides_data.dropna(axis=1)
    oxides_data = oxides_data.loc[:, oxides_data.nunique() > 1]
    oxides_data.index = oxides_data['formula']
    oxides_data = oxides_data.drop('formula', axis=1)
    support_df = x4[support_col]
    # support_oxide_dataの作成
    support_oxide_data = pd.DataFrame(index=x4.index, columns=oxides_data.columns)
    for index, row in support_df.iterrows():
        sup = row[row == 1].index
        sup_mat = [c.split('_')[1] for c in sup]
        support_oxide_data.loc[index, :] = oxides_data.loc[sup_mat[0], :]
    new_col_name = ['Support_'+c for c in support_oxide_data.columns]
    support_oxide_data.columns = new_col_name
    # supportを含まない最初の説明変数名を取得
    col_name = pd.Series(x4.columns)
    is_support = col_name.str.contains('Support')
    transition = (~is_support) & is_support.shift(1).fillna(False)
    first_feature = col_name[transition].iloc[0]
    # x4の再構築
    x4 = pd.concat([x4.loc[:,:'Temp'], support_oxide_data, x4.loc[:, first_feature:]], axis=1)
    
    return x4

def Add_interaction(x2, metal_x, oxide_x, reactive_elements_list, reactive_interaction):
    interaction = pd.read_csv('results/diatomic_df.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
    # 相互作用追加の方針：reactive_elementsはOとの相互作用を追加、stable_elementsはその集合内の相互作用を追加
    # diatomic_dfを必要元素のみとする
    used_elements = x2.loc[:, 'Ba':].columns.tolist()
    reactive_elements_list = [el for el in reactive_elements_list if el in used_elements]
    stable_elements = [el for el in used_elements if el not in reactive_elements_list]  
    reactive_elements = reactive_elements_list + ['O']

    # 相互作用項作成
    if reactive_interaction:
        combinations_r = [tuple(sorted(comb)) for comb in itertools.combinations_with_replacement(reactive_elements, 2)]
    else:
        combinations_r = [tuple(sorted([el, 'O'])) for el in reactive_elements if el != 'O']
    cross_elements_r = [f"{x}_{y}" for x, y in combinations_r]
    combinations_s = [tuple(sorted(comb)) for comb in itertools.combinations_with_replacement(stable_elements, 2)]
    cross_elements_s = [f"{x}_{y}" for x, y in combinations_s]

    # interactionの不要な列を削除
    drop_list = ['data_dir_path','vasprun','outcar','no_error','converged','converged_electronic','converged_ionic','stabilized',
                'atomic_symbol_1','atomic_symbol_2','potcar_symbol_1','potcar_symbol_2','calc_stat']
    interaction = interaction.drop(drop_list, axis=1)
    interaction = interaction.set_index('system_name')

    # interactionのsystem_nameは原子番号順になっているので、alphabet順に更新
    interaction.index = pd.Index(["_".join(sorted(idx.split("_"))) for idx in interaction.index])
    # interactionの必要な行だけ抽出
    interaction = interaction[interaction.index.isin(cross_elements_r + cross_elements_s)]
    interaction = interaction.sort_index()
    #cross_elements = [f'{x}_{y}' for x, y in combinations]
    # nanがある列はdrop
    interaction = interaction.dropna(how='any', axis=1)
    # 必要元素の列名を作成
    index_list = interaction.index.tolist()
    col_list = interaction.columns.tolist()
    new_col_list = [f"{idx}_{col}" for idx in index_list for col in col_list]

    # x2に元素の組合せを追加
    cross_df = pd.DataFrame(index=x2.index, columns=new_col_list)    
    cross_df = cross_df.replace(np.nan, 0)

    metal_x = metal_x.replace(np.nan, 0)
    #cross_df = cross_df[cross_df.index.isin(oxide_x.index)]
    
    # metal_x, oxide_x内容を元にinteraction内容を代入していく
    for index, row in cross_df.iterrows():
        if pd.notna(oxide_x.at[index, 'metal3']) is True:
            if reactive_interaction is False:
                elements = sorted([oxide_x.at[index, 'metal1'], oxide_x.at[index, 'metal2'], oxide_x.at[index, 'metal3']])
                combos = [tuple(sorted([el, 'O'])) for el in elements]
                for el1, el2 in combos:
                    for item in row.index:
                        cross_df.loc[index, item] = interaction.loc[el1 + '_' + el2, item.split('_', 2)[2]]
            else:
                elements = sorted([oxide_x.at[index, 'metal1'], oxide_x.at[index, 'metal2'], oxide_x.at[index, 'metal3']])
                combos = [(elements[0], elements[1]), (elements[0], elements[2]), (elements[1], elements[2])]
                for el1, el2 in combos:
                    for item in row.index:
                        cross_df.loc[index, item] = interaction.loc[el1 + '_' + el2, item.split('_', 2)[2]]
        elif pd.notna(oxide_x.at[index, 'metal2']) is True:
            if reactive_interaction is False:
                elements = sorted([oxide_x.at[index, 'metal1'], oxide_x.at[index, 'metal2']])
                combs = [tuple(sorted([el, 'O'])) for el in elements]
                for el1, el2 in combs:
                    for item in row.index:
                        cross_df.loc[index, item] = interaction.loc[el1 + '_' + el2, item.split('_', 2)[2]]
            else:
                elements = sorted([oxide_x.at[index, 'metal1'], oxide_x.at[index, 'metal2']])
                for item in row.index:
                    cross_df.loc[index, item] = interaction.loc[elements[0] + '_' + elements[1], item.split('_', 2)[2]]

    cross_df = cross_df.loc[:, (cross_df != 0).any(axis=0)]   #全て0の列は削除
    x2_ita = pd.concat([x2.loc[:, : 'Support_SiO5'], cross_df], axis=1)    
    
    return x2_ita
    
    
# データの読込み
element_data = pd.read_csv('results/element_data_241220.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
thermo_data = pd.read_csv('results/thermo_data.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
oxides_data = pd.read_csv('results/oxides_data_241106.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
metal_x = pd.read_csv('MyWork/datasets/metal_x.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
#x2 = pd.read_csv('MyWork/datasets/x2.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)  # C2 yield=0を含む
x2 = pd.read_csv('MyWork/datasets/x2_0.csv', encoding='utf-8-sig', index_col=0, header=0)    #C2 yield=0を削除
lanthanoid = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
#reactive_elements_list = ['Ca','Y','Sc','Mg','Sr','Li','Ba','Hf','Al','Zr','Ti','Si','Na','Mn','Cs','K','Cr','Rb','Ga','Zn','In'] + lanthanoid
#reactive_elements_list = ['Ca','Y','Sc','Mg','Sr','Li','Ba','Hf','Al','Zr','Ti','Si','Na','Mn','Cs','K','Cr','Rb','Ga','Zn','In','V','Mo','Sn'] + lanthanoid
reactive_elements_list = ['Ca','Y','Sc','Mg','Sr','Li','Ba','Hf','Al','Zr','Ti','Si','Na','Mn','Cs','K','Cr','Rb','Ga','Zn','In','V','Mo','Sn','Fe','W','Co','Ni','Pb','Cu'] + lanthanoid

target= ['C2 yield'] 
"""
# x2に間違いあったため修正 この部位は2度実行する必要はない。
x2_error = x2[x2.loc[:, 'Ba':].sum(axis=1) != 1]
x2_error.loc[:, ['Cr', 'V']] = x2_error.loc[:, ['Cr', 'V']].replace(0.7, 1)
x2_fixed = x2_error
x2_no_error = x2[~x2.index.isin(x2_error.index)]
x2 = pd.concat([x2_no_error, x2_fixed], axis= 0)
x2 = x2.sort_index()
x2_error = x2[x2.loc[:, 'Ba':].sum(axis=1) != 1]
x2.to_csv('MyWork/datasets/x2.csv', encoding= 'utf-8-sig')
"""

# metal_list作成
# C2 yield=0のデータを使用するときは下の1文を使用すること
metal_x = metal_x[metal_x.index.isin(x2.index)]
metal_x = metal_x.replace('na', 0)
metal_col = [s for s in metal_x.columns if 'Amount' not in s]
metal_list = pd.unique(metal_x[metal_col].values.flatten())
metal_list = pd.Series(metal_list).dropna().to_list()
metal_list.remove(0)

# reactive_elements_listの更新
reactive_elements_list = [s for s in reactive_elements_list if s in metal_list]

# element_dataとthermo_dataの統合
element_data = pd.concat([element_data, thermo_data], axis= 1)
element_data = element_data.loc[element_data.index.isin(metal_list)]
element_data = element_data.loc[:, element_data.isnull().mean() < 0.2]
check_nan = element_data.isna().sum()
element_data = element_data.dropna(axis= 1)
element_data = element_data.drop(['space group', 'structure','oxide','hhi_p','hhi_r','CAS'], axis= 1)
element_data = delete_high_corr(element_data, threshold= 0.95)

# oxides_dataもmetal_listとreactive_elemtentsにある元素に限定する
oxides_data.index = oxides_data['formula']
oxides_data = oxides_data.drop('formula', axis= 1)
oxides_data_index = [s[:2] for s in oxides_data.index]
oxides_data_index = [s[:1] if '2' in s else s for s in oxides_data_index]
oxides_data_index = [s[:1] if 'O' in s else s for s in oxides_data_index]
oxides_data.index = oxides_data_index
oxides_data = oxides_data.loc[oxides_data.index.isin(reactive_elements_list)]
#oxides_data = oxides_data.loc[oxides_data.index.isin(metal_list)]

# nanのある列, 全て同じ値である列はdropする
oxides_data = oxides_data.dropna(how='any',axis=1)
oxides_data = oxides_data.loc[:, oxides_data.nunique() > 1]

# reactive_elementsを決めたとき、それを含まないデータは分割する必要がある
# 逆にreactive_elementsのみから成るデータも分割する必要がある
# ここでreactive_elementsのみからなるデータ(_r)とそうでないデータ(_s)に分割する
# _rのデータはx4計算時にmetal_xがnanばかりになるデータということ
x2_r = x2[x2[reactive_elements_list].sum(axis=1) == 1]
stable_elements = list(set(metal_list) - set(reactive_elements_list))
x2_s = x2[x2[stable_elements].sum(axis=1) != 0]
# metal_x内容も更新
metal_xr = metal_x[metal_x.index.isin(x2_r.index)]
metal_xs = metal_x[metal_x.index.isin(x2_s.index)]
#metal_x = metal_x.iloc[300:360, :]

#scalingは、standard(標準化)とnormal(正規化)、compoは、relative(相対値)とabsolute(絶対値)から選択する
#selectorは、Boruta, Boruta2, BorutaShapの3つから選択する
#x3 = Calc_x3(target, x2, metal_x, xenonpy_original, xenonpy_additional, selector= 'BorutaShap', compo= 'absolute', perc= 90)
#Rh_EMEはEME計算をRhを含むものについてはomitするか否かを決めるもの Rh_EME= False とすると、Rhを含まないものでEME計算する
x4, importance_df = Calc_x3(
    target,
    x2,
    metal_x,
    element_data,
    oxides_data,
    selector = 'Boruta',
    cross_term = False,
    compo= 'relative',
    perc= 80,
    )

x4.to_csv('MyWork/datasets/x4_h0.csv', encoding= 'utf-8-sig')
importance_df.to_csv('MyWork/datasets/x4_h0_importance.csv', encoding= 'utf-8-sig')

"""
metal_x_Sn = pd.read_csv('MyWork/datasets/metal_x_Sn.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
oxide_x_Sn = pd.read_csv('MyWork/datasets/oxide_x_Sn.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
#metal_x_Sn = metal_x_Sn.iloc[300:360, :]
#oxide_x_Sn = oxide_x_Sn.iloc[300:360, :]
x5 = Add_interaction(x2, metal_x_Sn, oxide_x_Sn, reactive_elements_list, reactive_interaction= False)
x5.to_csv('MyWork/datasets/x5_c_o.csv', encoding= 'utf-8-sig')
"""
"""
# Boruta絞込み
x5_b_o = pd.read_csv('MyWork/datasets/x5_c_o.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x5, importance_df = Boruta_Apply2(target, x5_b_o, perc= 50)
x5.to_csv('MyWork/datasets/x5_c.csv', encoding= 'utf-8-sig')
importance_df.to_csv('MyWork/datasets/x5_c_importance.csv', encoding= 'utf-8-sig')
"""
"""
# oxides_dataをsupport変数へ変換
x4 = pd.read_csv('MyWork/datasets/x4_e.csv', encoding='utf-8-sig', index_col=0, header=0)
oxides_data = pd.read_csv('results/oxides_data_241106.csv', encoding='utf-8-sig', index_col=0, header=0)
x4 = x4.drop(['Support_Al2O4','Support_Al2O5','Support_Al2O6','Support_SiO3','Support_SiO4','Support_SiO5'], axis=1)
x4 = x4[x4[['Support_Al2O3','Support_SiO2']].sum(axis=1) == 1]
x4 = Calc_support(x4, oxides_data)
x4.to_csv('MyWork/datasets/x4_f.csv', encoding='utf-8-sig')
"""
"""
# Borutaの結果から上位を抽出
x5_b_o = pd.read_csv('MyWork/datasets/x5_b_o.csv', encoding='utf-8-sig', index_col=0, header=0)
x5 = pd.read_csv('MyWork/datasets/x5_b.csv', encoding='utf-8-sig', index_col=0, header=0)
x5_importance = pd.read_csv('MyWork/datasets/x5_b_importance.csv', encoding='utf-8-sig', index_col=0, header=0)
x5 = pd.concat([x5_b_o.loc[:, :'Support_SiO5'], x5.loc[:, x5_importance.index.tolist()[:30]]], axis=1)
x5 = x5.loc[:, ~x5.columns.duplicated()]
x5.to_csv('MyWork/datasets/x5_b1.csv', encoding='utf-8-sig')
"""
"""
# C2 yield=0のデータを除外
xo1 = pd.read_csv('MyWork/datasets/xo1.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x2 = pd.read_csv('MyWork/datasets/x2.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)

xo1 = xo1[xo1['C2 yield'] != 0]
x2 = x2[x2['C2 yield'] != 0]

xo1.to_csv('MyWork/datasets/xo1_0.csv', encoding= 'utf-8-sig')
x2.to_csv('MyWork/datasets/x2_0.csv', encoding= 'utf-8-sig')
"""
print('End')