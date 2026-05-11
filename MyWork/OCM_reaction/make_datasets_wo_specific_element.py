import pandas as pd
import numpy as np
import sys
import os
import itertools
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from boruta import BorutaPy
from BorutaShap import BorutaShap
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer, BaseFeaturizer
from matminer.featurizers.composition.alloy import Miedema, YangSolidSolution, WenAlloys
from matminer.featurizers.composition.ion import OxidationStates, IonProperty, ElectronAffinity, ElectronegativityDiff
from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital
from matminer.featurizers.composition.composite import ElementProperty, Meredig
from matminer.featurizers.composition.element import BandCenter

#Sequential Backward Selection(逐次後退選択)アルゴリズムの作成 by ChatGPT
class SBS():
    def __init__(self, estimator, k_features, scoring=r2_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, 
                                                            random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train.iloc[:, list(indices)], y_train)
        y_pred = self.estimator.predict(X_test.iloc[:, list(indices)])
        score = self.scoring(y_test, y_pred)
        return score

#3元素交差項の関数 2乗項も含まない純粋な3元素項
def addTripleCross2(df: pd.DataFrame):
    df_elements = df.loc[:, 'Ba':]       
    columns = df_elements.columns
    for i, c1 in enumerate(columns):
        print(i)
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            for k, c3 in enumerate(columns):
                if j <= k:
                    continue
                df[c1 +' * ' + c2 + ' * ' +c3] = df[c1] * df[c2] * df[c3]
    return df
#2元素交差項と3元素交差項を作成
def addTripleDoubleCross(df: pd.DataFrame):
    #try:
    #    df_elements = df.loc[:, 'Ba':]  
    #except KeyError:
    #    df_elements = df.loc[:, 'Co':]
    df_elements = df.loc[:, 'Temp':]
    columns = df_elements.columns
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = df[c1] * df[c2]
    for i, c1 in enumerate(columns):
        print(i)
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            for k, c3 in enumerate(columns):
                if j <= k:
                    continue
                df[c1 +' * ' + c2 + ' * ' +c3] = df[c1] * df[c2] * df[c3]
    return df

#交差項のみの関数
def addCross(df: pd.DataFrame):
    df_elements = df.loc[:, 'Ba':]    
    columns = df_elements.columns

    for i, c1 in enumerate(columns):
        #df[c1 + '^2'] = df[c1] ** 2
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = df[c1] * df[c2]
    return df

#加減乗除の項を全て追加、但し自乗項等はなし
def addArithmetic(df: pd.DataFrame):
    df_elements = df.loc[:, 'Ba':]
    columns = df_elements.columns
    for i, c1 in enumerate(columns):
        #df[c1 + '^2'] = df[c1] ** 2
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = df[c1] * df[c2]    
            df[c1 +' + ' + c2] = df[c1] + df[c2]
            df[c1 +' - ' + c2] = np.abs(df[c1] - df[c2])
            df[f'min({c1},{c2})/max({c1},{c2})'] = df[[c1, c2]].min(axis=1) / df[[c1, c2]].max(axis=1).replace(0, 1)
    return df

#2元と3元の加減乗除項を作成して追加する 但し3元は加乗のみ
def addArithmetic2(df: pd.DataFrame):
    df_elements = df.loc[:, 'Ba':]
    columns = df_elements.columns
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = df[c1] * df[c2]    
            df[c1 +' + ' + c2] = df[c1] + df[c2]
            df[c1 +' - ' + c2] = np.abs(df[c1] - df[c2])
            df[f'min({c1},{c2})/max({c1},{c2})'] = df[[c1, c2]].min(axis=1) / df[[c1, c2]].max(axis=1).replace(0, 1)
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):            
            if i <= j:
                continue
            for k, c3 in enumerate(columns):
                if j <= k:
                    continue
                df[c1 +' * ' + c2 + ' * ' +c3] = df[c1] * df[c2] * df[c3]            
                df[c1 +' + ' + c2 + ' + ' +c3] = df[c1] + df[c2] + df[c3]
    return df

def Boruta_Apply(df, perc):
    y= df['C2 yield']
    x= df.drop('C2 yield', axis= 1).copy()

    # RandomForestRegressorでBorutaを実行
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, perc= perc)
    feat_selector.fit(x.values, y.values)

    # 選択された特徴量を確認
    selected = feat_selector.support_
    print('選択された特徴量の数: %d' % np.sum(selected))
    print(x.columns[selected])

    #上で選択した説明変数のみを残す。
    boruta_descriptors = x.columns
    selected_features= boruta_descriptors[selected]
    df_b= df.loc[:, selected_features]
    df_b = pd.concat([y, df_b], axis= 1)
    
    return df_b

def BorutaShap_Apply(df):
    y= df['C2 yield']
    x= df.drop('C2 yield', axis= 1).copy()
        
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

#matminerの特徴量を出力する関数
def Matminer_Desc(metal_x):
    metal_x.replace(0, np.nan, inplace= True)
    metal_x.replace('na', np.nan, inplace= True)
    metal_x.replace(np.nan, ' ', inplace= True)
    for index, row in metal_x.iterrows():
        metal1 = row[0]
        metal2 = row[1]    
        metal3 = row[2]    
        ratio1 = row[3]
        ratio2 = row[4]
        ratio3 = row[5]
        metal_x.loc[index, 'chemicalFormula'] = metal1+str(ratio1)+metal2+str(ratio2)+metal3+str(ratio3)
    metal_x['chemicalFormula'] = metal_x['chemicalFormula'].str.rstrip()
    stc = StrToComposition(target_col_id='composition')
    stc.set_n_jobs(1)
    metal_x = stc.featurize_dataframe(metal_x, "chemicalFormula", ignore_errors= True)
    f_Miedema = Miedema()
    f_Miedema.set_n_jobs(1)
    f_Meredig = Meredig()
    f_Meredig.set_n_jobs(1)
    f_WenAlloys = WenAlloys()
    f_WenAlloys.set_n_jobs(1)
    f_AtomicOrbitals = AtomicOrbitals()
    f_AtomicOrbitals.set_n_jobs(1)
    f_BandCenter = BandCenter()
    f_BandCenter.set_n_jobs(1)
    features_Miedema = f_Miedema.featurize_dataframe(metal_x, col_id='composition', ignore_errors= True)
    features_Meredig = f_Meredig.featurize_dataframe(metal_x, col_id='composition', ignore_errors= True)
    features_WenAlloys = f_WenAlloys.featurize_dataframe(metal_x, col_id='composition', ignore_errors= True)
    features_AtomicOrbitals= f_AtomicOrbitals.featurize_dataframe(metal_x, col_id='composition', ignore_errors= True)
    features_BandCenter = f_BandCenter.featurize_dataframe(metal_x, col_id= 'composition', ignore_errors= True)

    df_a = features_Miedema.drop(['chemicalFormula','composition'], axis= 1)
    AtomicOrbitals_list = ['HOMO_energy','LUMO_energy','gap_AO']
    Meredig_list = ['mean AtomicWeight','mean Column','mean Row','range Number','mean Number','range AtomicRadius','mean AtomicRadius','range Electronegativity',\
                    'mean Electronegativity','avg s valence electrons','avg p valence electrons','avg d valence electrons','avg f valence electrons']
    WenAlloys_list = ['Yang delta','Yang omega','APE mean','Radii local mismatch','Radii gamma','Configuration entropy','Atomic weight mean','Total weight','Lambda entropy',\
                    'Electronegativity delta','Electronegativity local mismatch','VEC mean','Mixing enthalpy','Mean cohesive energy','Interant electrons','Interant s electrons',\
                    'Interant p electrons','Interant d electrons','Interant f electrons','Shear modulus mean','Shear modulus delta','Shear modulus local mismatch','Shear modulus strength model']
    df_b = features_Meredig.loc[:, Meredig_list]
    df_d = features_WenAlloys.loc[:, WenAlloys_list]
    df_e = features_AtomicOrbitals.loc[:, AtomicOrbitals_list]
    df_f = features_BandCenter.loc[:, 'band center']
    df = pd.concat([df_a, df_b, df_d, df_e, df_f], axis= 1)
    df.drop(['Atomic weight mean','Total weight'], axis= 1, inplace= True)
   
    return df

#x1生成関数
def CalcX1(metal_x):
    #x1の作成
    metal_x = pd.read_csv('datasets/v361/metal_x.csv', index_col= 0, header= 0)
    xenonpy_original = pd.read_csv('results/xenonpy_element_data.csv', index_col= 0)
    elements = metal_x[['metal1','metal2','metal3','metal4','metal5']].values.flatten().tolist()
    xenonpy_original = xenonpy_original.query('index == @elements')
    xenonpy_original = xenonpy_original.dropna(how= 'any', axis= 1)
    xenonpy_original = xenonpy_original.drop('period', axis= 1)
    xenonpy_merge = xenonpy_original
    #xenonpy_original = xenonpy_original.loc[:, xenonpy_original.isnull().mean() < 0.5]
    # IterativeImputerの初期化
    #iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    # IterativeImputerで補完
    #df_iterative_imputed = pd.DataFrame(iterative_imputer.fit_transform(xenonpy_original), columns=xenonpy_original.columns)
    #xenonpy_merge = df_iterative_imputed
    #xenonpy_merge.index = xenonpy_original.index
    #x1の作成本番
    weighted_average_name = list() # 加重平均の index 名
    weighted_variance_name = list() # 加重分散の index 名
    geometric_mean_name = list() # 幾何平均の index 名
    harmonic_mean_name = list() # 調和平均の index 名
    max_pooling_name = list() # 最大値の index 名
    min_pooling_name = list() # 最小値の index 名
    for j in xenonpy_merge.columns:
        weighted_average_name.append(f'ave_{j}')
        weighted_variance_name.append(f'var_{j}')
        geometric_mean_name.append(f'gmean_{j}')
        harmonic_mean_name.append(f'hmean_{j}')
        max_pooling_name.append(f'max_{j}')
        min_pooling_name.append(f'min_{j}')

    x1_metaldesc = pd.DataFrame(
        index=metal_x.index,
        columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
        )
    for i in range(metal_x.shape[0]):
        metal1 = metal_x.loc[metal_x.index[i], 'metal1']
        metal2 = metal_x.loc[metal_x.index[i], 'metal2']
        metal3 = metal_x.loc[metal_x.index[i], 'metal3']
        metal4 = metal_x.loc[metal_x.index[i], 'metal4']
        metal5 = metal_x.loc[metal_x.index[i], 'metal5']
        metal_rate1 = metal_x.loc[metal_x.index[i], 'ratio1']
        metal_rate2 = metal_x.loc[metal_x.index[i], 'ratio2']
        metal_rate3 = metal_x.loc[metal_x.index[i], 'ratio3']
        metal_rate4 = metal_x.loc[metal_x.index[i], 'ratio4']
        metal_rate5 = metal_x.loc[metal_x.index[i], 'ratio5']   
        if metal5 is not np.nan:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            metal_desc3 = xenonpy_merge.loc[metal3, :].values
            metal_desc4 = xenonpy_merge.loc[metal4, :].values        
            metal_desc5 = xenonpy_merge.loc[metal5, :].values        
            mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])
        elif metal4 is not np.nan:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            metal_desc3 = xenonpy_merge.loc[metal3, :].values
            metal_desc4 = xenonpy_merge.loc[metal4, :].values        
            mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])             
        elif metal3 != 'na':
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            metal_desc3 = xenonpy_merge.loc[metal3, :].values
            mt = np.array([metal_desc1, metal_desc2, metal_desc3])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3])
        elif metal2 != 'na':
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            mt = np.array([metal_desc1, metal_desc2])
            mr = np.array([metal_rate1, metal_rate2])                
        else:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            mt = np.array([metal_desc1])
            mr = np.array([metal_rate1])
        for desc in range(xenonpy_merge.shape[1]):
            d_name = xenonpy_merge.columns[desc]
            if np.isnan(mt[:, desc]).any():
                x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'var_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'hmean_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'max_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'min_{d_name}'].iloc[i] = np.nan
                continue
            #metal_x.to_csv('datasets/metal_x.csv', encoding= 'utf-8-sig')
            x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
            #x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr)
            #x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)
            x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)/np.sum(mr)
            x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
            x1_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
            x1_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
            x1_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])

    x1_metaldesc = x1_metaldesc.replace([np.inf, -np.inf], np.nan)
    #ここでnanが生まれる可能性あり！！
    x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]   #全てnanの列を削除

    return x1_metaldesc

#x1に相互作用項を付与したようなx3を作成する
def Calc_x3(y, x2, metal_x, xenonpy_original, xenonpy_additional, compo, perc):
    #metal_xの列名の変更
    if metal_x.columns.to_list()[0] != 'metal1':
        metal_x = metal_x_columns_name(metal_x)
    #xenonpy_baseの作成
    metal_columns = [col for col in metal_x.columns if 'metal' in col]
    element_list = pd.unique(metal_x[metal_columns].values.ravel('K'))
    element_list = list(set(element_list))
    element_list = sorted([s for s in element_list if s != 0 and not pd.isna(s)])
    xenonpy_base = xenonpy_original.loc[element_list]
    xenonpy_additional = xenonpy_additional.loc[element_list]
    #xenonpy_additional = xenonpy_additional.drop('Te')
    xenonpy_base = pd.merge(xenonpy_base, xenonpy_additional, left_index= True, right_index= True)
    xenonpy_base = xenonpy_base.dropna(how= 'any', axis= 1)
    xenonpy_base = xenonpy_base.drop(['hhi_p', 'hhi_r', 'period'], axis= 1)
    xenonpy_base = delete_high_corr(xenonpy_base, threshold= 0.98)
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
        index=metal_x.index,
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
    perc_x2 = 80
    x2_cross_boruta = Boruta_Apply(target, x2_cross, perc_x2)
    cross_elements = [s for s in x2_cross_boruta.columns.to_list() if '*' in s]   
    print('Number of cross_elements:'+str(len(cross_elements))) 
    while len(cross_elements) >= 10 and perc_x2 < 100:
        if perc_x2 < 90:
            perc_x2 += 10
        else:
            perc_x2 += 5
        x2_cross_boruta = Boruta_Apply(target, x2_cross, perc_x2)
        cross_elements = [s for s in x2_cross_boruta.columns.to_list() if '*' in s] 
        print('Number of cross_elements:'+str(len(cross_elements)))
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

    # 相互作用項の計算
    for index, row in metal_x.iterrows():
        metal_list = [row[f'metal{i}'] for i in range(1, len(row)//2+1) if pd.notna(row[f'metal{i}'])]
        ratio_list = [row[f'ratio{i}'] for i in range(1, len(row)//2+1) if pd.notna(row[f'ratio{i}'])]
        existing_elements = [(metal_list[i], ratio_list[i]) for i in range(len(metal_list)) if metal_list[i] in xenonpy_base.index]

        for element_pair in cross_elements:
            elements = element_pair.split(' * ')
            for col in xenonpy_base.columns:
                col_p = f"{element_pair.replace(' * ', '*')}_{col}_p"
                col_n = f"{element_pair.replace(' * ', '*')}_{col}_n"
                
                # 一致する要素があるか確認
                matched_elements = [el for el in elements if el in [el[0] for el in existing_elements]]
                matched_list = [el for el in existing_elements if el[0] in matched_elements]
                # 全ての要素が一致するときはスケーリングしたうえで加重平均と相互作用を計算する
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
                else:  # 完全一致でない場合
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
    
    desc_for_boruta = pd.concat([x2.iloc[:, : first_element_loc], final_desc], axis= 1)
    desc_boruta = Boruta_Apply(target, desc_for_boruta, perc= perc)
    first_desc = next((s for s in desc_boruta.columns.to_list() if s in final_desc), None)    
    final_df = pd.concat([x2.iloc[:, :first_element_loc], desc_boruta.loc[:, first_desc:]], axis= 1)
    
    return final_df

#吸着しやすさに基づくgroupingの関数
def Group_by_Adsorption(metal_x):
    #元素Grouping
    group_a = ['Ca','Sr','Ba','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Fe','Ru','Os']  #15
    group_b = ['Ni','Co','Rh','Pd','Pt','Ir','Mn','Cu','Tc','Re']  #10
    group_c = ['Al','Au']  #2
    group_d = ['Li','Na','K','Rb','Cs']  #5
    group_e = ['Mg','Ag','Zn','Cd','In','Si','Ge','Sn','Pb','As','Sb','Bi','Se','Te','Ga']  #15
    group_f = ['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']  #17
    
    metals = metal_x.copy()
    metals = metals.replace(np.nan, 0)
    metals = metals.replace('na', 0)
    metals = metals[~metals.index.duplicated(keep= 'first')]
    metals= metals.assign(groupa= 0, groupb= 0, groupc= 0, groupd= 0, groupe= 0, groupf= 0)
    metals= metals.assign(compoa= 0, compob= 0, compoc= 0, compod= 0, compoe= 0, compof= 0)
            
    for index, row in metals.iterrows():
        for i in range(0, 3):
            if row[i] in group_a:
                metals.loc[index, 'groupa']+= 1
                metals.loc[index, 'compoa']+= row[i+3]
            elif row[i] in group_b:
                metals.loc[index, 'groupb']+= 1
                metals.loc[index, 'compob']+= row[i+3]
            elif row[i] in group_c:
                metals.loc[index, 'groupc']+= 1
                metals.loc[index, 'compoc']+= row[i+3]
            elif row[i] in group_d:
                metals.loc[index, 'groupd']+= 1
                metals.loc[index, 'compod']+= row[i+3]
            elif row[i] in group_e:
                metals.loc[index, 'groupe']+= 1
                metals.loc[index, 'compoe']+= row[i+3]
            elif row[i] in group_f:
                metals.loc[index, 'groupf']+= 1
                metals.loc[index, 'compof']+= row[i+3]
                
    return metals

#結晶構造に基づくGrouping
def Group_by_Structure(metal_x):
   #結晶構造Grouping
    group_B = ['Li','Na','K','V','Cr','Fe','Rb','Mo','Cs','Ba','Eu','W']  #13
    group_F = ['Al','Ca','Ni','Cu','Sr','Rh','Pd','Ce','Yb','Ir','Pt','Au','Pb']  #13
    group_H = ['Mg','Sc','Ti','Co','Zn','Y','Zr','Ru','Te','La','Pr','Nd','Gd','Tb','Dy','Ho','Er','Lu','Hf','Re']  #20
    group_O = ['Ga','In','Sn','Sm','Mn']  #4 
    metals = metal_x.copy()
    metals = metals.replace(np.nan, 0)
    metals = metals.replace(' ', '')
    metals= metals.assign(group_B= 0, group_F= 0, group_H= 0, group_O= 0)
    metals= metals.assign(compo_B= 0, compo_F= 0, compo_H= 0, compo_O= 0)
    #触媒ロットsc000372はMnが2つ出てくるため除外、sc000368、369も1成分が0なので除外
    metals = metals.query('index != "sc000372"')
    metals = metals.query('index != "sc000368"')
    metals = metals.query('index != "sc000369"')
    metals = metals[~metals.index.duplicated(keep= 'first')]
            
    for index, row in metals.iterrows():
        for i in range(0, 5):
            if row[i] in group_B:
                metals.loc[index, 'group_B'] += 1
                metals.loc[index, 'compo_B'] += row[i+5]
            if row[i] in group_F:
                metals.loc[index, 'group_F'] += 1
                metals.loc[index, 'compo_F'] += row[i+5]            
            if row[i] in group_H:
                metals.loc[index, 'group_H'] += 1
                metals.loc[index, 'compo_H'] += row[i+5]
            if row[i] in group_O:
                metals.loc[index, 'group_O'] += 1
                metals.loc[index, 'compo_O'] += row[i+5]                
    
    return metals
    
#元素群の組合せによるGrouping
def Group_by_Combi(metal_x):
    #2元系の組合せに着目したGrouping
    group_ep = [['Ru','Rh','Ir'], ['Fe','Co','Cr','Mn']]
    group_la = [['Y','K','La','Ce'], ['Rh','Cu','Ni','Re','Pt']]
    #2元素組合せでグルーピングする
    metals = metal_x.copy()
    metals = metals.replace(np.nan, 0)
    metals = metals.replace(' ', '')
    metals= metals.assign(group_ep= 0, group_la= 0)
    metals= metals.assign(compo_ep= 0, compo_la= 0)
    #触媒ロットsc000372はMnが2つ出てくるため除外、sc000368、369も1成分が0なので除外
    metals = metals.query('index != "sc000372"')
    metals = metals.query('index != "sc000368"')
    metals = metals.query('index != "sc000369"')
    metals = metals[~metals.index.duplicated(keep= 'first')]
    #metals = metals.query('index == @x2_4["触媒ロット"].to_list()')    

    for index, row in metals.iterrows():
        if row[5] == 1:
            metals.loc[index, 'group_ep':'compo_la'] = 0
        else:
            for i in range(0, 5):
                if row[i] in group_ep[0]:
                    if row[i+1] in group_ep[1]:
                        metals.at[index, 'group_ep'] = 1
                        metals.loc[index, 'compo_ep'] = row[i+5]+row[i+6]
                    elif row[i+2] in group_ep[1]:
                        metals.at[index, 'group_ep'] = 1
                        metals.loc[index, 'compo_ep'] = row[i+5]+row[i+7]                    
                    elif row[i+3] in group_ep[1]:
                        metals.at[index, 'group_ep'] = 1
                        metals.loc[index, 'compo_ep'] = row[i+5]+row[i+8] 
                    elif row[i+4] in group_ep[1]:
                        metals.at[index, 'group_ep'] = 1
                        metals.loc[index, 'compo_ep'] = row[i+5]+row[i+9]                     
                elif row[i] in group_ep[1]:
                    if row[i+1] in group_ep[0]:
                        metals.at[index, 'group_ep'] = 1
                        metals.loc[index, 'compo_ep'] = row[i+5]+row[i+6]
                    elif row[i+2] in group_ep[0]:
                        metals.at[index, 'group_ep'] = 1
                        metals.loc[index, 'compo_ep'] = row[i+5]+row[i+7]                    
                    elif row[i+3] in group_ep[0]:
                        metals.at[index, 'group_ep'] = 1
                        metals.loc[index, 'compo_ep'] = row[i+5]+row[i+8] 
                    elif row[i+4] in group_ep[0]:
                        metals.at[index, 'group_ep'] = 1
                        metals.loc[index, 'compo_ep'] = row[i+5]+row[i+9]                        
                elif row[i] in group_la[0]:
                    if row[i+1] in group_la[1]:
                        metals.at[index, 'group_la'] = 1
                        metals.loc[index, 'compo_la'] = row[i+5]+row[i+6]
                    elif row[i+2] in group_la[1]:
                        metals.at[index, 'group_la'] = 1
                        metals.loc[index, 'compo_la'] = row[i+5]+row[i+7]                    
                    elif row[i+3] in group_la[1]:
                        metals.at[index, 'group_la'] = 1
                        metals.loc[index, 'compo_la'] = row[i+5]+row[i+8] 
                    elif row[i+4] in group_la[1]:
                        metals.at[index, 'group_la'] = 1
                        metals.loc[index, 'compo_la'] = row[i+5]+row[i+9]                     
                elif row[i] in group_la[1]:
                    if row[i+1] in group_la[0]:
                        metals.at[index, 'group_la'] = 1
                        metals.loc[index, 'compo_la'] = row[i+5]+row[i+6]
                    elif row[i+2] in group_la[0]:
                        metals.at[index, 'group_la'] = 1
                        metals.loc[index, 'compo_la'] = row[i+5]+row[i+7]                    
                    elif row[i+3] in group_la[0]:
                        metals.at[index, 'group_la'] = 1
                        metals.loc[index, 'compo_la'] = row[i+5]+row[i+8] 
                    elif row[i+4] in group_la[0]:
                        metals.at[index, 'group_la'] = 1
                        metals.loc[index, 'compo_la'] = row[i+5]+row[i+9]   
    return metals 

#データ読込み
x1 = pd.read_csv('Mywork/datasets/x1.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x2 = pd.read_csv('Mywork/datasets/x2.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x3d_9r = pd.read_csv('Mywork/datasets/x3d_9r.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
metal_x = pd.read_csv('Mywork/datasets/metal_x.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
xenonpy_original = pd.read_csv('results/xenonpy_element_data.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
xenonpy_additional = pd.read_csv('results/xenonpy_additional.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
webelements_data = pd.read_csv('results/webelements_data.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
matminer = pd.read_csv('datasets/v445/elements_matminer.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
metal_x = pd.read_csv('MyWork/datasets/metal_x.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
target= ['C2 yield'] 
#使用する元素、プロセス条件、matminerリストなど
elements = ['Cr','Mn','Fe','Co','Ni','Cu','Zn','Ba','Cs','La','Na','Rb','Sn','Sr','W','V','Y']
x0 = x2.loc[:, : 'Support_SiO5']

#X3を利用する　元素Aのないものから元素A以外の元素で交差項を作り、Aを予測するとのロジックとする
x1_Mn = x1[x2['Mn'] == 0]
x1_Mn0 = x1[x2['Mn'] != 0]


x2_V2_2 = x2[(x2['V'] == 0) & (x2['Fe'] == 0)]
x2_V2_3 = x2[x2['V'] != 0]
x1_V2_2 = x1.loc[x2_V2_2.index]
x1_V2_2 = Boruta_Apply(x1_V2_2, perc= 90)
x1_V2_3 = x1.loc[x2_V2_3.index]
x1_V2_3 = x1_V2_3.loc[:, x1_V2_2.columns.to_list()]
x1_V2_2 = x1_V2_2[x1_V2_2['C2 yield'] != 0]
x1_V2_3 = x1_V2_3[x1_V2_3['C2 yield'] != 0]
x1_V2_2.to_csv('MyWork/datasets/predict_wo_sp_el/x1_V2_2.csv', encoding= 'utf-8-sig')
x1_V2_3.to_csv('MyWork/datasets/predict_wo_sp_el/x1_V2_3.csv', encoding= 'utf-8-sig')

#x2を作成する。
x2_Mn_0 = x2[x2['Mn'] == 0]
x2_Mn = x2[x2['Mn'] != 0]
x2_Na_0 = x2[x2['Na'] == 0]
x2_Na = x2[x2['Na'] != 0]
x2_V_0 = x2[x2['V'] == 0]
x2_V = x2[x2['V'] != 0]




print('hello')