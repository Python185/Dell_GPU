import pandas as pd
import numpy as np
import os
import itertools
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from itertools import combinations
from boruta import BorutaPy
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
    xenonpy_original = pd.read_csv('results/xenonpy_element_data.csv', index_col= 0)    
    elements = metal_x[['metal1','metal2','metal3','metal4','metal5', 'metal6']].values.flatten().tolist()
    elements = [s for s in elements if s != '0.0']
    elements = list(set(elements))
    elements = sorted(elements)
    xenonpy_original = xenonpy_original.loc[elements, :]
    xenonpy_original = xenonpy_original.dropna(how= 'any', axis= 1)
    metal_x = metal_x.replace('0.0', np.nan)
    #x1の作成本番
    weighted_average_name = list() # 加重平均の index 名
    weighted_variance_name = list() # 加重分散の index 名
    geometric_mean_name = list() # 幾何平均の index 名
    harmonic_mean_name = list() # 調和平均の index 名
    max_pooling_name = list() # 最大値の index 名
    min_pooling_name = list() # 最小値の index 名
    for j in xenonpy_original.columns:
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
        metal6 = metal_x.loc[metal_x.index[i], 'metal6']
        metal_rate1 = metal_x.loc[metal_x.index[i], 'ratio1']
        metal_rate2 = metal_x.loc[metal_x.index[i], 'ratio2']
        metal_rate3 = metal_x.loc[metal_x.index[i], 'ratio3']
        metal_rate4 = metal_x.loc[metal_x.index[i], 'ratio4']
        metal_rate5 = metal_x.loc[metal_x.index[i], 'ratio5']   
        metal_rate6 = metal_x.loc[metal_x.index[i], 'ratio6']
        if metal6 is not np.nan:
            metal_desc1 = xenonpy_original.loc[metal1, :].values
            metal_desc2 = xenonpy_original.loc[metal2, :].values
            metal_desc3 = xenonpy_original.loc[metal3, :].values
            metal_desc4 = xenonpy_original.loc[metal4, :].values        
            metal_desc5 = xenonpy_original.loc[metal5, :].values  
            metal_desc6 = xenonpy_original.loc[metal6, :].values     
            mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5, metal_desc6])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5, metal_rate6])
        elif metal5 is not np.nan:
            metal_desc1 = xenonpy_original.loc[metal1, :].values
            metal_desc2 = xenonpy_original.loc[metal2, :].values
            metal_desc3 = xenonpy_original.loc[metal3, :].values
            metal_desc4 = xenonpy_original.loc[metal4, :].values     
            metal_desc5 = xenonpy_original.loc[metal5, :].values   
            mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])             
        elif metal4 is not np.nan:
            metal_desc1 = xenonpy_original.loc[metal1, :].values
            metal_desc2 = xenonpy_original.loc[metal2, :].values
            metal_desc3 = xenonpy_original.loc[metal3, :].values
            metal_desc4 = xenonpy_original.loc[metal4, :].values
            mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])
        elif metal3 is not np.nan:
            metal_desc1 = xenonpy_original.loc[metal1, :].values
            metal_desc2 = xenonpy_original.loc[metal2, :].values
            metal_desc3 = xenonpy_original.loc[metal3, :].values
            mt = np.array([metal_desc1, metal_desc2, metal_desc3])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3])
        elif metal2 is not np.nan:
            metal_desc1 = xenonpy_original.loc[metal1, :].values
            metal_desc2 = xenonpy_original.loc[metal2, :].values
            mt = np.array([metal_desc1, metal_desc2])
            mr = np.array([metal_rate1, metal_rate2])            
        else:
            metal_desc1 = xenonpy_original.loc[metal1, :].values
            mt = np.array([metal_desc1])
            mr = np.array([metal_rate1])
        for desc in range(xenonpy_original.shape[1]):
            d_name = xenonpy_original.columns[desc]
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

#C吸着エネルギー付与
#1)組成に応じて吸着エネルギーも按分、2)最大の吸着エネルギーを持つ元素の値で代表, 3)最大組成元素の吸着エネルギーで代表
def Ads_Carbon(metal_x):
    for index, row in metal_x.iterrows():
        if 'metal2' in locals():
            del metal2
        if 'metal3' in locals():
            del metal3
        metal1 = row[0]
        ratio1 = row[3]
        ads_metal1_posi = C_adsorption.index.get_loc(metal1)
        ads_matal1_energy = C_adsorption.iloc[ads_metal1_posi, 0]
        if row[1] != 'na':
            metal2 = row[1]
            ratio2 = row[4]
            ads_metal2_posi = C_adsorption.index.get_loc(metal2)
            ads_matal2_energy = C_adsorption.iloc[ads_metal2_posi, 0]
        if row[2] != 'na':
            metal3 = row[2]
            ratio3 = row[5]
            ads_metal3_posi = C_adsorption.index.get_loc(metal3)
            ads_matal3_energy = C_adsorption.iloc[ads_metal3_posi, 0]
        if 'metal3' in locals():
            metal_x.loc[index, 'C_ads1'] = ads_matal1_energy*ratio1 + ads_matal2_energy*ratio2 + ads_matal3_energy*ratio3
            metal_ratio_dict = {metal1: ratio1, metal2: ratio2, metal3: ratio3} #最大の値を持つ元素名を特性するために辞書にする
            metal_energy = [ads_matal1_energy, ads_matal2_energy, ads_matal3_energy]
            max_ratio_metal = max(metal_ratio_dict, key= metal_ratio_dict.get)
            max_energy = min(metal_energy)
            metal_x.loc[index, 'C_ads2'] = max_energy
            metal_x.loc[index, 'C_ads3'] = C_adsorption.iloc[C_adsorption.index.get_loc(max_ratio_metal), 0]
        elif 'metal2' in locals():
            metal_x.loc[index, 'C_ads1'] = ads_matal1_energy*ratio1 + ads_matal2_energy*ratio2
            metal_ratio_dict = {metal1: ratio1, metal2: ratio2}
            metal_energy = [ads_matal1_energy, ads_matal2_energy]
            max_ratio_metal = max(metal_ratio_dict, key= metal_ratio_dict.get)
            max_energy = min(metal_energy)
            metal_x.loc[index, 'C_ads2'] = max_energy
            metal_x.loc[index, 'C_ads3'] = C_adsorption.iloc[C_adsorption.index.get_loc(max_ratio_metal), 0]        
        else:
            metal_x.loc[index, 'C_ads1'] = ads_matal1_energy
            metal_x.loc[index, 'C_ads2'] = ads_matal1_energy
            metal_x.loc[index, 'C_ads3'] = ads_matal1_energy       
    return metal_x

#データ読込み
original_data = pd.read_excel('MyWork/datasets/original_data/OCM.xlsx', sheet_name= 'OCM_Dataset_-2019', header= 0)
C_adsorption = pd.read_excel('MyWork/datasets/C_adsorption.xlsx', sheet_name= 'data', index_col= 0)
x2 = pd.read_csv('MyWork/datasets/predict_wo_sp_el2/x2.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
xenonpy_merge = pd.read_csv('results/xenonpy_merge.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
xenonpy_merge = xenonpy_merge.drop('oxide', axis= 1)
metal_x = pd.read_csv('MyWork/datasets/predict_wo_sp_el2/metal_x.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
#使用する元素、プロセス条件、matminerリストなど
metal_list = ['Cation 1','Cation 2','Cation 3','Cation 4','Cation 5','Cation 6']
"""
#data概況を確認する。
data = original_data.dropna(subset= ['Support 1'])
data = data.replace(np.nan, 0.0)
#data['sum'] = data['Cation 1 mol%'] + data['Cation 2 mol%'] + data['Cation 3 mol%'] + data['Cation 4 mol%'] + data['Cation 5 mol%'] + data['Cation 6 mol%']
#不要な列を削除して目的変数の位置を先頭に移動
data_columns = data.columns.to_list()
data = data.drop(['Nr of publication','Preparation','X(O2), %','X(CH4), %','S(COx), %','S(C2=), %','S(C2-), %','S(C2), %'], axis= 1)
y = data['Y(C2), %']
data = data.drop('Y(C2), %', axis= 1)
data.insert(0, 'Y(C2), %', y)
non_zero_ratios = data.apply(lambda x: (x != 0).mean())
data = data.drop('Promotor', axis= 1)
#使用元素のリスト作成
metals = pd.concat([data[col].replace(0, '') for col in metal_list]).tolist()
metals = list(sorted(set(metals)))
del metals[0]
anions = pd.concat([data[col].replace(0, '') for col in ['Anion 1', 'Anion 2']]).tolist()
anions = list(sorted(set(anions)))
del anions[0]
supports = pd.concat([data[col].replace(0, '') for col in ['Support 1','Support 2','Support 3']]).tolist()
supports = list(sorted(set(supports)))
del supports[0]

#元素と組成の形式を修正
drop_list = [s for s in data.columns.to_list() if 'Cation' in s or 'Anion' in s or 'Support' in s]
#data = data.assign(**{col: 0 for col in metals}) #metals列を挿入して値は全て0
#data = data.assign(**{f'anion_{anion}': 0 for anion in anions})
#data = data.assign(**{f'support_{support}': 0 for support in supports})
#data = data.iloc[: 50, : 29]
#metal_xの作成
metal_x = data[data['Y(C2), %'] >= 0.5]
metal_x = metal_x.loc[:, : 'Cation 6 mol%']
metal_x = metal_x.drop('Y(C2), %', axis= 1)
metal_percent_list = [s+' mol%' for s in metal_list]
new_columns = metal_list + metal_percent_list
metal_x = metal_x[new_columns]
columns_list = metal_x.columns.to_list()
columns_list_a = columns_list[:6]
columns_list_b = columns_list[6:]
columns_list_a = [s.replace('Cation ', 'metal') for s in columns_list_a]
columns_list_b = [s.replace(' mol%', '') for s in columns_list_b]
columns_list_b = [s.replace('Cation ', 'ratio') for s in columns_list_b]
new_columns_list = columns_list_a + columns_list_b
columns_dict = dict(zip(columns_list, new_columns_list))
metal_x = metal_x.rename(columns= columns_dict)
for column in ['ratio1','ratio2','ratio3','ratio4','ratio5','ratio6']:
    metal_x[column] = metal_x[column] / 100
metal_x.to_csv('MyWork/datasets/predict_wo_sp_el2/metal_x.csv', encoding= 'utf-8-sig')

# Cation列とmol%列を処理
for i in range(1, 7):
    cation_col = f'Cation {i}'
    mol_cation = f'{cation_col} mol%'
    # 各行に対して処理
    for index, row in data.iterrows():
        cation = row[cation_col]
        cation_mol = float(row[mol_cation])    
        # Cationが0またはNaNでない場合に処理
        if cation not in [0, '0.0', '', np.nan]:
            # 新しい列名を生成
            new_cation_name = f'Cation_{cation}'
            # 新しい列にmol%を加算
            if new_cation_name not in data.columns:
                data[new_cation_name] = 0.0
            data.at[index, new_cation_name] += cation_mol    
for i in range(1, 3):    
    anion_col = f'Anion {i}'
    mol_anion = f'{anion_col} mol%'  
    # 各行に対して処理
    for index, row in data.iterrows():  
        anion = row[anion_col]
        anion_mol = float(row[mol_anion])
        if anion not in [0, '0.0', '', np.nan]:
            # 新しい列名を生成
            new_anion_name = f'Anion_{anion}'
            # 新しい列にmol%を加算
            if new_anion_name not in data.columns:
                data[new_anion_name] = 0.0
            data.at[index, new_anion_name] += anion_mol    
for i in range(1, 4):    
    support_col = f'Support {i}'
    mol_support = f'{support_col} mol%'
    # 各行に対して処理
    for index, row in data.iterrows():
        support = row[support_col]
        support_mol = float(row[mol_support])
        if support not in [0, '0.0', '', np.nan]:
            # 新しい列名を生成
            new_support_name = f'Support_{support}'
            # 新しい列にmol%を加算
            if new_support_name not in data.columns:
                data[new_support_name] = 0.0
            data.at[index, new_support_name] += support_mol            

#不要な列を削除して、nanを0埋めする
data = data.drop(drop_list, axis= 1)
data.to_csv('MyWork/datasets/predict_wo_sp_el2/x2.csv', encoding= 'utf-8-sig')
"""
#収率0.5%以上に限定し、1つだけある収率90％以上も削除する
x2_0 = x2[(x2['Y(C2), %'] >= 0.5) & (x2['Y(C2), %'] < 90)]
#x2_0.to_csv('MyWork/datasets/predict_wo_sp_el2/x2_0.csv', encoding= 'utf-8-sig')
x0_0 = pd.concat([x2_0.loc[:, : 'Contact time, s'], x2_0.loc[:, 'Anion_S': ]], axis= 1)
x0_1 = x2_0.loc[:, : 'Anion_N']
x0_1.to_csv('MyWork/datasets/predict_wo_sp_el2/x0_1.csv', encoding= 'utf-8-sig')
x0_0.to_csv('MyWork/datasets/predict_wo_sp_el2/x0_0.csv', encoding= 'utf-8-sig')


#xenonpy_desc = CalcX1(metal_x)

print('hello')