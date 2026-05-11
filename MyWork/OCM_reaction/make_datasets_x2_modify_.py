import pandas as pd
import numpy as np
import os
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

#交差項のルートの関数
def addCrossRoot(df: pd.DataFrame):
    columns = df.columns
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = np.sqrt(df[c1] * df[c2])
    return df

#交差項の2乗の関数
def addCrossCross(df: pd.DataFrame):
    columns = df.columns
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = (df[c1] * df[c2]) ** 2
    return df

#交差項(a-0.5)(b-0.5)@a>=b 凹関数
def addConcave(df: pd.DataFrame):
    columns = df.columns
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i <= j:
                continue
            df[c1 +' * ' + c2] = np.abs((df[c1]- 0.5) * (df[c2]- 0.5))
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

#データ読込み
x2_6_all = pd.read_csv('Mywork/datasets/x2_6_all.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x2_7_all = pd.read_csv('Mywork/datasets/x2_7_all.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x2_38_all = pd.read_csv('Mywork/datasets/x2_38_all.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x2 = pd.read_csv('Mywork/datasets/x2.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
metal_x = pd.read_csv('Mywork/datasets/metal_x.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
C_adsorption = pd.read_excel('MyWork/datasets/C_adsorption.xlsx', sheet_name= 'data', index_col= 0)
xenonpy_merge = pd.read_csv('results/xenonpy_merge.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
xenonpy_merge = xenonpy_merge.drop('oxide', axis= 1)

#元素Grouping
group_a = ['Ca','Sr','Ba','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Fe','Ru','Os']  #15
group_b = ['Ni','Co','Rh','Pd','Pt','Ir','Mn','Cu','Tc','Re']  #10
group_c = ['Al','Au']  #2
group_d = ['Li','Na','K','Rb','Cs']  #5
group_e = ['Mg','Ag','Zn','Cd','In','Si','Ge','Sn','Pb','As','Sb','Bi','Se','Te','Ga']  #15
group_f = ['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']  #17

#結晶構造Grouping
group_B = ['Li','Na','K','V','Cr','Fe','Rb','Mo','Cs','Ba','Eu','W']  #13
group_F = ['Al','Ca','Ni','Cu','Sr','Rh','Pd','Ce','Yb','Ir','Pt','Au','Pb']  #13
group_H = ['Mg','Sc','Ti','Co','Zn','Y','Zr','Ru','Te','La','Pr','Nd','Gd','Tb','Dy','Ho','Er','Lu','Hf','Re']  #20
group_O = ['Ga','In','Sn','Sm','Mn']  #4

#使用する元素、プロセス条件、matminerリストなど
elements = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt', 'Au', 'In', 'Sn']
"""
#吸着エネルギーを特徴量として使用する
#1)組成に応じて吸着エネルギーも按分、2)最大の吸着エネルギーを持つ元素の値で代表, 3)最大組成元素の吸着エネルギーで代表
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

x2_47 = pd.merge(x2_6_all, metal_x, left_index= True, right_index= True)
x2_47 = x2_47.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x2_47.to_csv('MyWork/datasets/x2_47_all.csv', encoding= 'utf-8-sig')

x2_48 = pd.merge(x2_7_all, metal_x, left_index= True, right_index= True)
x2_48 = x2_48.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x2_48.to_csv('MyWork/datasets/x2_48_all.csv', encoding= 'utf-8-sig')

x2_49 = pd.merge(x2_38_all, metal_x, left_index= True, right_index= True)
x2_49 = x2_49.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x2_49.to_csv('MyWork/datasets/x2_49_all.csv', encoding= 'utf-8-sig')

x2_50 = x2_47.drop(['C_ads2', 'C_ads3'], axis= 1)
x2_51 = x2_47.drop(['C_ads1', 'C_ads3'], axis= 1)
x2_52 = x2_47.drop(['C_ads1', 'C_ads2'], axis= 1)
x2_50.to_csv('MyWork/datasets/x2_50_all.csv', encoding= 'utf-8-sig')
x2_51.to_csv('MyWork/datasets/x2_51_all.csv', encoding= 'utf-8-sig')
x2_52.to_csv('MyWork/datasets/x2_52_all.csv', encoding= 'utf-8-sig')

x2_53 = x2_48.drop(['C_ads2', 'C_ads3'], axis= 1)
x2_54 = x2_48.drop(['C_ads1', 'C_ads3'], axis= 1)
x2_55 = x2_48.drop(['C_ads1', 'C_ads2'], axis= 1)
x2_53.to_csv('MyWork/datasets/x2_53_all.csv', encoding= 'utf-8-sig')
x2_54.to_csv('MyWork/datasets/x2_54_all.csv', encoding= 'utf-8-sig')
x2_55.to_csv('MyWork/datasets/x2_55_all.csv', encoding= 'utf-8-sig')

x2_56 = x2_49.drop(['C_ads2', 'C_ads3'], axis= 1)
x2_57 = x2_49.drop(['C_ads1', 'C_ads3'], axis= 1)
x2_58 = x2_49.drop(['C_ads1', 'C_ads2'], axis= 1)
x2_56.to_csv('MyWork/datasets/x2_56_all.csv', encoding= 'utf-8-sig')
x2_57.to_csv('MyWork/datasets/x2_57_all.csv', encoding= 'utf-8-sig')
x2_58.to_csv('MyWork/datasets/x2_58_all.csv', encoding= 'utf-8-sig')

#C吸着エネルギーを含めて交差項作成してBoruta絞込み
x2_28 = pd.merge(x2, metal_x, left_index= True, right_index= True)
x2_28 = x2_28.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x2_28_desc = x2_28.drop('C2 yield', axis= 1)
addCross(x2_28_desc)
x2_28 = pd.concat([x2['C2 yield'], x2_28_desc], axis= 1)
x2_29 = x2_28.copy()
x2_30 = x2_28.copy()
x2_28 = Boruta_Apply(x2_28, perc= 90)
x2_28.to_csv('MyWork/datasets/x2_28.csv', encoding= 'utf-8-sig')
x2_29 = Boruta_Apply(x2_29, perc= 80)
x2_29.to_csv('MyWork/datasets/x2_29.csv', encoding= 'utf-8-sig')
x2_30 = Boruta_Apply(x2_30, perc= 60)
x2_30.to_csv('MyWork/datasets/x2_30.csv', encoding= 'utf-8-sig')

#matminer10を追加
metal_xx = metal_x.copy()
metal_xx = metal_xx.drop(['C_ads1','C_ads2','C_ads3'], axis= 1)
metal_xx.replace(0, np.nan, inplace= True)
metal_xx.replace('na', np.nan, inplace= True)
metal_xx.replace(np.nan, ' ', inplace= True)
for index, row in metal_xx.iterrows():
    metal1 = row[0]
    metal2 = row[1]    
    metal3 = row[2]    
    ratio1 = row[3]
    ratio2 = row[4]
    ratio3 = row[5]
    metal_xx.loc[index, 'chemicalFormula'] = metal1+str(ratio1)+metal2+str(ratio2)+metal3+str(ratio3)
        
metal_xx['chemicalFormula'] = metal_xx['chemicalFormula'].str.rstrip()
stc = StrToComposition(target_col_id='composition')
stc.set_n_jobs(1)
metal_xx = stc.featurize_dataframe(metal_xx, "chemicalFormula", ignore_errors= True)
#metal_xx = StrToComposition(target_col_id='composition').featurize_dataframe(metal_xx, "chemicalFormula", ignore_errors=True)

#featurizerの指定　１つであればfeaturize_dataframeを使用
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
features_Miedema = f_Miedema.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
features_Meredig = f_Meredig.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
features_WenAlloys = f_WenAlloys.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
features_AtomicOrbitals= f_AtomicOrbitals.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
features_BandCenter = f_BandCenter.featurize_dataframe(metal_xx, col_id= 'composition', ignore_errors= True)
#features_Miedema.to_csv('MyWork/datasets/f_Miedema.csv', encoding= 'cp932')

x22_a = features_Miedema.drop(['chemicalFormula','composition'], axis= 1)
AtomicOrbitals_list = ['HOMO_energy','LUMO_energy','gap_AO']
Meredig_list = ['mean AtomicWeight','mean Column','mean Row','range Number','mean Number','range AtomicRadius','mean AtomicRadius','range Electronegativity',\
                'mean Electronegativity','avg s valence electrons','avg p valence electrons','avg d valence electrons','avg f valence electrons']
WenAlloys_list = ['Yang delta','Yang omega','APE mean','Radii local mismatch','Radii gamma','Configuration entropy','Atomic weight mean','Total weight','Lambda entropy',\
                'Electronegativity delta','Electronegativity local mismatch','VEC mean','Mixing enthalpy','Mean cohesive energy','Interant electrons','Interant s electrons',\
                'Interant p electrons','Interant d electrons','Interant f electrons','Shear modulus mean','Shear modulus delta','Shear modulus local mismatch','Shear modulus strength model']
x22_b = features_Meredig.loc[:, Meredig_list]
x22_d = features_WenAlloys.loc[:, WenAlloys_list]
x22_e = features_AtomicOrbitals.loc[:, AtomicOrbitals_list]
x22_f = features_BandCenter.loc[:, 'band center']
x22_1 = pd.concat([x22_a, x22_b, x22_d, x22_e, x22_f], axis= 1)
x22_1.drop(['Atomic weight mean','Total weight'], axis= 1, inplace= True)
x22_1.to_csv('MyWork/datasets/matminer_41.csv', encoding= 'utf-8-sig')

matminer = pd.read_csv('MyWork/datasets/matminer_41.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
matminer_list = ['APE mean','Configuration entropy','Mixing enthalpy','Shear modulus mean','Shear modulus delta','mean AtomicRadius','mean Electronegativity','HOMO_energy','LUMO_energy','band center']
matminer_10 = matminer.loc[:, matminer_list]
x2_31 = pd.concat([x2, matminer_10], axis= 1)
x2_32 = pd.merge(x2_31, metal_x, left_index= True, right_index= True)
x2_32 = x2_32.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x2_31.to_csv('MyWork/datasets/x2_31.csv', encoding= 'utf-8-sig')
x2_32.to_csv('MyWork/datasets/x2_32.csv', encoding= 'utf-8-sig')

#Borutaで緩く絞込み
x2_32 = pd.read_csv('MyWork/datasets/x2_32.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x2_33 = Boruta_Apply(x2_32, perc= 60)
x2_33.to_csv('MyWork/datasets/x2_33.csv', encoding= 'utf-8-sig')

#matminerの特徴量を使用
matminer = pd.read_csv('MyWork/datasets/matminer_41.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
matminer = matminer.drop('C2 yield', axis= 1)
x2_59 = pd.merge(x2, metal_x, left_index= True, right_index= True)
x2_59 = x2_59.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x2_59 = pd.concat([x2_59, matminer], axis= 1)
x2_60 = x2_59.copy()
x2_61 = x2_59.copy()
x2_59 = Boruta_Apply(x2_59, perc= 90)
x2_60 = Boruta_Apply(x2_60, perc= 80)
x2_61 = Boruta_Apply(x2_61, perc= 60)
x2_59.to_csv('MyWork/datasets/x2_59_all.csv', encoding= 'utf-8-sig')
x2_60.to_csv('MyWork/datasets/x2_60_all.csv', encoding= 'utf-8-sig')
x2_61.to_csv('MyWork/datasets/x2_61_all.csv', encoding= 'utf-8-sig')
"""
# OCM反応機構に従い、酸化物生成エンタルピーを追加(241022)
x2_56_all = pd.read_csv('MyWork/datasets/x2_56_all.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
oxide_data = pd.read_csv('results/element_data_normal_string240527.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
metal_list = x2_56_all.loc[:, 'Cr':'W'].columns.tolist()
oxide_data = oxide_data[oxide_data.index.isin(metal_list)]
oxide_data = oxide_data[['formation_energy_of_oxide','oxide']]
x2_62 = x2_56_all.copy()
x2_62['formation_energy_of_oxide'] = 

"""
#xenonpy_mergeの情報を利用
x1 = pd.read_csv('MyWork/datasets/x1.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x1_8 = Boruta_Apply(x1, perc= 90)
x1_9 = Boruta_Apply(x1, perc= 80)
x1_10 = Boruta_Apply(x1, perc= 70)
x1_11 = Boruta_Apply(x1, perc= 60)
x1_8.to_csv('MyWork/datasets/x1_8.csv', encoding= 'utf-8-sig')
x1_9.to_csv('MyWork/datasets/x1_9.csv', encoding= 'utf-8-sig')
x1_10.to_csv('MyWork/datasets/x1_10.csv', encoding= 'utf-8-sig')
x1_11.to_csv('MyWork/datasets/x1_11.csv', encoding= 'utf-8-sig')

xo1 = pd.read_csv('MyWork/datasets/xo1.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x1_12 = Boruta_Apply(xo1, perc= 70)
x1_12.to_csv('MyWork/datasets/x1_12.csv', encoding= 'utf-8-sig')
x1_13 = Boruta_Apply(xo1, perc= 50)
x1_13.to_csv('MyWork/datasets/x1_13.csv', encoding= 'utf-8-sig')

#x1_b60, x1_9にC吸着エネルギーを追加
x1_b60_all = pd.read_csv('Mywork/datasets/x1_b60_all.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x1_9 = pd.read_csv('Mywork/datasets/x1_9.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x1_14 = pd.merge(x1_b60_all, metal_x, left_index= True, right_index= True)
x1_14 = x1_14.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x1_14.to_csv('MyWork/datasets/x1_14.csv', encoding= 'utf-8-sig')
x1_15 = pd.merge(x1_9, metal_x, left_index= True, right_index= True)
x1_15 = x1_15.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x1_15.to_csv('MyWork/datasets/x1_15.csv', encoding= 'utf-8-sig')

#x1, xo1にC吸着エネルギー追加
x1 = pd.read_csv('MyWork/datasets/x1.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
xo1 = pd.read_csv('MyWork/datasets/xo1.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x1_16 = pd.merge(x1, metal_x, left_index= True, right_index= True)
x1_16 = x1_16.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x1_16.to_csv('MyWork/datasets/x1_16.csv', encoding= 'utf-8-sig')
x1_17 = pd.merge(xo1, metal_x, left_index= True, right_index= True)
x1_17 = x1_17.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)
x1_17.to_csv('MyWork/datasets/x1_17.csv', encoding= 'utf-8-sig')

#x1, xo1にC吸着エネルギー追加したものをBorutaで選択
#train_test_splitで分割したtrainに対してBoruta処理
x1_16 = pd.read_csv('MyWork/datasets/x1_16.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x1_17 = pd.read_csv('MyWork/datasets/x1_17.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x1_16tr, x1_16_te = train_test_split(x1_16, test_size= 0.25, random_state= 42, shuffle= True)
x1_17tr, x1_17_te = train_test_split(x1_17, test_size= 0.25, random_state= 42, shuffle= True)
x1_18 = Boruta_Apply(x1_16tr, perc= 90)
x1_19 = Boruta_Apply(x1_16tr, perc= 80)
x1_20 = Boruta_Apply(x1_16tr, perc= 60)
x1_21 = Boruta_Apply(x1_17tr, perc= 90)
x1_22 = Boruta_Apply(x1_17tr, perc= 80)
x1_23 = Boruta_Apply(x1_17tr, perc= 60)
x1_18 = x1_16.loc[:, x1_18.columns.to_list()]
x1_19 = x1_16.loc[:, x1_19.columns.to_list()]
x1_20 = x1_16.loc[:, x1_20.columns.to_list()]
x1_21 = x1_17.loc[:, x1_21.columns.to_list()]
x1_22 = x1_17.loc[:, x1_22.columns.to_list()]
x1_23 = x1_17.loc[:, x1_23.columns.to_list()]
x1_18.to_csv('MyWork/datasets/x1_18.csv', encoding= 'utf-8-sig')
x1_19.to_csv('MyWork/datasets/x1_19.csv', encoding= 'utf-8-sig')
x1_20.to_csv('MyWork/datasets/x1_20.csv', encoding= 'utf-8-sig')
x1_21.to_csv('MyWork/datasets/x1_21.csv', encoding= 'utf-8-sig')
x1_22.to_csv('MyWork/datasets/x1_22.csv', encoding= 'utf-8-sig')
x1_23.to_csv('MyWork/datasets/x1_23.csv', encoding= 'utf-8-sig')
"""
print('hello')