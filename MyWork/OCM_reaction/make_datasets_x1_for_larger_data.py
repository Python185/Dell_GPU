import pandas as pd
import numpy as np
import os, sys
import itertools
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from BorutaShap import BorutaShap
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer, BaseFeaturizer
from matminer.featurizers.composition.alloy import Miedema, YangSolidSolution, WenAlloys
from matminer.featurizers.composition.ion import OxidationStates, IonProperty, ElectronAffinity, ElectronegativityDiff
from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital
from matminer.featurizers.composition.composite import ElementProperty, Meredig
from matminer.featurizers.composition.element import BandCenter

#交差項と自乗項の関数
def addSquareCross(df: pd.DataFrame):
    columns = df.columns
    for i, c1 in enumerate(columns):
        df[c1 + '^2'] = df[c1] ** 2
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = df[c1] * df[c2]
    return df

#交差項のみの関数
def addCross(df: pd.DataFrame):
    columns = df.columns
    for i, c1 in enumerate(columns):
        #df[c1 + '^2'] = df[c1] ** 2
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = df[c1] * df[c2]
    return df

#3元素交差項の関数 3乗項はなし 2乗項は含む
def addTripleCross(df: pd.DataFrame):
    columns = df.columns
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i < j:
                continue
            for k, c3 in enumerate(columns):
                if j < k:
                    continue
                if c1== c2 and c1== c3:
                    continue
                df[c1 +' * ' + c2 + ' * ' +c3] = df[c1] * df[c2] * df[c3]
    return df

#3元素交差項の関数 2乗項も含まない純粋な3元素項
def addTripleCross2(df: pd.DataFrame):
    columns = df.columns
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            for k, c3 in enumerate(columns):
                if j <= k:
                    continue
                df[c1 +' * ' + c2 + ' * ' +c3] = df[c1] * df[c2] * df[c3]
    return df

def remove_correlated_features(df, threshold):
    corr_matrix = df.corr().abs()  # 絶対値の相関行列
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # 上三角行列のみを取得
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]  # 閾値を超える変数を取得
    return df.drop(columns=to_drop)  # 高相関の変数を削除

def Boruta_Apply(target, df, perc):
    y= df[target]
    x= df.drop(target, axis= 1).copy()
    # 相関係数の高い特徴量を削除
    x = remove_correlated_features(x, threshold= 0.95)
    
    # RandomForestRegressorでBorutaを実行
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, perc= perc)
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

def BorutaShap_Apply(df):
    y= df['C2 yield']
    x= df.drop('C2 yield', axis= 1).copy()
    # 相関係数の高い特徴量を削除
    x = remove_correlated_features(x, threshold= 0.95)
        
    # BorutaShpaの実行 defaultではXGB
    Feature_Selector = BorutaShap(importance_measure= 'shap', classification= False)        
    Feature_Selector.fit(X= x, y= y, n_trials= 100) 
    Feature_Selector.plot(which_features= 'all')
    
    #上で選択した説明変数のみを残す。
    important_df = Feature_Selector.Subset()
    selected_features= important_df.columns.to_list()
    df_a = df.loc[:, :'Support_SiO5']
    df_b= df.loc[:, selected_features]
    df_c = pd.concat([df_a, df_b], axis= 1)    

    return df_c

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

#相関係数の大きい特徴量を削除する
def delete_high_corr(df, threshold):
    corr_matrix = df.corr().abs()
    upper_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k= 1).astype(np.bool_))
    drop_col = [col for col in upper_matrix.columns if any(upper_matrix[col] > threshold)]
    reduced_df = df.drop(drop_col, axis= 1)
    return reduced_df    


# x1の作成
def Calc_x1(x2, metal_x, element_data):
    # metal_xの変形
    metal_x = metal_x.rename(columns={c: c.replace('Cation ', 'metal') for c in metal_x.columns})
    metal_x = metal_x.rename(columns={c: c.replace('metal', 'ratio') for c in metal_x.columns if 'mol%' in c})
    metal_x = metal_x.rename(columns={c: c.replace('mol%', '') for c in metal_x.columns if 'mol%' in c})
    metal_x = metal_x.rename(columns={c: c.replace(' ', '') for c in metal_x.columns if ' ' in c})
    metal_x_col = [f'metal{i}' for i in range(1, len(metal_x.columns) // 2 + 1)] + [f'ratio{i}' for i in range(1, len(metal_x.columns) // 2 + 1)]
    metal_x = metal_x[metal_x_col]

    # element_listの作成
    metal_col = [s for s in metal_x.columns if 'metal' in s]
    metal_list = pd.unique(metal_x[metal_col].values.flatten())
    element_list = pd.Series(metal_list).dropna().to_list()    
    
    #xenonpy_baseの作成
    element_list = sorted([s for s in element_list if s != 0 and not pd.isna(s) and s != 0.0])
    xenonpy_base = element_data.loc[element_list]
    xenonpy_base = xenonpy_base.dropna(how= 'any', axis= 1)
    xenonpy_base = xenonpy_base.drop(['hhi_p', 'hhi_r'], axis= 1)
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
    for j in xenonpy_base.columns:
        weighted_average_name.append(f'ave_{j}')
        weighted_variance_name.append(f'var_{j}')
        geometric_mean_name.append(f'gmean_{j}')
        harmonic_mean_name.append(f'hmean_{j}')
        max_pooling_name.append(f'max_{j}')
        min_pooling_name.append(f'min_{j}')

    x3_metaldesc = pd.DataFrame(
        index=metal_x.index, dtype= float,
        columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
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
            idx = x3_metaldesc.index[i]
            x3_metaldesc.loc[idx, f'ave_{d_name}'] = np.dot(mt[:, desc],mr) / np.sum(mr)
            x3_metaldesc.loc[idx, f'var_{d_name}'] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)/np.sum(mr)
            x3_metaldesc.loc[idx, f'gmean_{d_name}'] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
            x3_metaldesc.loc[idx, f'hmean_{d_name}'] = sum(mr)/sum((1/mt[:, desc])*mr)
            x3_metaldesc.loc[idx, f'max_{d_name}'] = max(mt[:, desc])
            x3_metaldesc.loc[idx, f'min_{d_name}'] = min(mt[:, desc])
   
    #descの値が負であるものはgmeanでnanとなるため、その列はdropする(負の値のものは計算しないようにするのはかなり面倒)
    x3_metaldesc = x3_metaldesc.dropna(how= 'any', axis= 1)      
    x3_metaldesc = x3_metaldesc.replace([np.inf, -np.inf], np.nan).dropna(how= 'any', axis= 1)
    x1 = pd.concat([x2.loc[:, : 'Contact time, s'], x3_metaldesc], axis= 1)
    
    return x1


#データ読込み
x2 = pd.read_csv('Mywork/datasets/larger_data/x2.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
metal_x = pd.read_csv('Mywork/datasets/larger_data/metal_x.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
element_data = pd.read_csv('results/xenonpy_element_data240515.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
target = ['Y(C2), %']

x1 = Calc_x1(x2, metal_x, element_data)
x1, importance = Boruta_Apply(target, x1, perc=90)
x1 = pd.concat([x2.loc[:, :'Contact time, s'], x1], axis= 1)
x1 = x1.loc[:, ~x1.columns.duplicated()]
x1.to_csv('Mywork/datasets/larger_data/x1.csv', encoding= 'utf-8-sig')
importance.to_csv('Mywork/datasets/larger_data/x1_importance.csv', encoding= 'utf-8-sig')

print('End')