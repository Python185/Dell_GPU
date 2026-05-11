import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./MyWork')
import pandas as pd
import numpy as np
from os.path import dirname, abspath
import sys
import itertools
import warnings
# warning の非表示
warnings.simplefilter('ignore')
from Calculate_descriptors import MetalFeaturizers

xenonpy_element_data = pd.read_csv('results/xenonpy_merge.csv', index_col=0)
#xenonpy_element_data1 = pd.read_csv('results/xenonpy_wo_nan1.csv', encoding= 'utf-8-sig', index_col= 0)
xenonpy_element_data.drop('oxide', axis= 1, inplace= True)

shap_threshold = 0.3
boruta_p = 90

dataset = 'OCM'
#dataset = 'OCM_Mn'
#dataset = 'Random_Catalyst'
#dataset = 'FTS'

def make_dataset1():
    #dataの読込み
    data = pd.read_csv('MyWOrk/original_data/Oxidative_coupling_of_methane_at_CADS.csv', header=0)
    
    target = data['C2 yield']
    #target = data['CH4 conv']
    
    df = pd.concat([data.iloc[:, :8], target], axis= 1)

    #担体のOneHot化
    df = pd.get_dummies(df, columns= ['Support'])
    #df = df.replace('na', 0)
    
    #元素名と組成をmetal1-3, ratio1-3に変換する
    metals = df[['Cation1','Cation2','Cation3']].values.tolist()
    metals = list(itertools.chain.from_iterable(metals))
    metals = list(set(metals))
    metals = sorted(metals)
    del metals[-1]

    metal_df = pd.DataFrame(columns= metals)
    df = pd.concat([df, metal_df], axis= 1)
    df.dropna(how= 'all', axis= 0, inplace= True)
    df['Cation1Amount'] = df['Cation1Amount']/100
    df['Cation2Amount'] = df['Cation2Amount']/100
    df['Cation3Amount'] = df['Cation3Amount']/100        
    x26 = df.iloc[:, :16].copy()
    
    for index, row in df.iterrows():
        metal1 = row[0]
        ratio1 = row[3]
        df.at[index, metal1] = ratio1
        metal2 = row[1]
        ratio2 = row[4]
        df.at[index, metal2] = ratio2        
        metal3 = row[2]
        ratio3 = row[5]
        if metal3 == 'na':
            pass
        else:
            df.at[index, metal3] = ratio3
            
    drop_list = ['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount']
    df.drop(drop_list, axis= 1, inplace= True)
    df.replace(np.nan, 0, inplace= True)
    first_column = df.pop('C2 yield')
    df.insert(0, 'C2 yield', first_column)
    df = df.dropna(how= 'all', axis= 1)        

    #df.to_csv('MyWork/datasets/xx2.csv', encoding= 'utf-8-sig')
        
    #x_baseの作成
    metal_x = x26[drop_list]
    x_base = df.iloc[:, :10]
    
    #一部に元素被りがあるため、そこを修正する
    for index, row in metal_x.iterrows():
        if row[0] == row[1]:
            metal_x.at[index, 'Cation2'] = 'na'
            metal_x.at[index, 'Cation1Amount'] = row[3] + row[4]
            metal_x.at[index, 'Cation2Amount'] = 0
    
    #xx26の作成
    featurizers = MetalFeaturizers()
    #x26 = featurizers.getDesc_X26('Type1', x_base, metal_x, xenonpy_element_data, shap_threshold, boruta_p)
    x2_17 = featurizers.getDesc_X2_17('Type1', x_base, metal_x)
    x2_17.to_csv('MyWork/datasets/xx2_17.csv', encoding='utf-8-sig')

def make_dataset2():
    #dataの読込み
    data = pd.read_csv('MyWOrk/original_data/CatalystModificationinOCMviaMnPromoter.csv', header=0)
    
    target = data['C2 yield (%)']
    #target = data['CH4 conv']
    
    df = pd.concat([data.iloc[:, :10], target], axis= 1)

    #担体のOneHot化
    df = pd.get_dummies(df, columns= ['Support'])
    #df = df.replace('na', 0)
    
    #元素名と組成をmetal1-4, ratio1-4に変換する
    metals = df[['M1','M2','M3','M4']].values.tolist()
    metals = list(itertools.chain.from_iterable(metals))   #多次元リストの1次元化
    metals = list(set(metals))
    metals.remove('none')
    metals.remove(np.nan)
    metals = sorted(metals)
  
    metal_df = pd.DataFrame(columns= metals)
    df = pd.concat([df, metal_df], axis= 1)
    df.dropna(how= 'all', axis= 0, inplace= True)
    df['M1 (wt%)'] = df['M1 (wt%)']/100
    df['M2 (wt%)'] = df['M2 (wt%)']/100
    df['M3 (wt%)'] = df['M3 (wt%)']/100
    df['M4 (wt%)'] = df['M4 (wt%)']/100
    df = df[df['C2 yield (%)'] >= 0.01]
    df = df[df['M1'] != 'none']
    
    x26 = df.iloc[:, :16].copy()
    #x26 = x26[x26.nunique() != 1]
    #組成をdfに入力
    """
    for index, row in df.iterrows():
        metal1 = row[0]
        ratio1 = row[1]
        df.at[index, metal1] = ratio1
        metal2 = row[2]
        ratio2 = row[3]
        if metal2 == 'none':
            pass
        else:
            df.at[index, metal2] = ratio2        
        metal3 = row[4]
        ratio3 = row[5]
        if metal3 == 'none':
            pass
        else:
            df.at[index, metal3] = ratio3
        metal4 = row[6]
        ratio4 = row[7]
        if metal4 == 'none':
            pass
        else:
            df.at[index, metal4] = ratio4
     """
    #組成は均等として入力     
    for index, row in df.iterrows():
        metal1 = row[0]
        metal2 = row[2]
        metal3 = row[4]
        metal4 = row[6]
        if metal2 == 'none':
            df.at[index, metal1] = 1
        elif metal3 == 'none':
            df.at[index, metal1] = 1/2
            df.at[index, metal2] = 1/2
        elif metal4 == 'none':
            df.at[index, metal1] = 1/3
            df.at[index, metal2] = 1/3
            df.at[index, metal3] = 1/3
        else:
            df.at[index, metal1] = 1/4
            df.at[index, metal2] = 1/4
            df.at[index, metal3] = 1/4
            df.at[index, metal4] = 1/4
    check_df = df[metals]
    check = check_df.sum(axis = 1)  
                 
    drop_list = ['M1','M2','M3','M4','M1 (wt%)','M2 (wt%)', 'M3 (wt%)', 'M4 (wt%)']
    df.drop(drop_list, axis= 1, inplace= True)
    df.replace(np.nan, 0, inplace= True)
    first_column = df.pop('C2 yield (%)')
    df.insert(0, 'C2 yield (%)', first_column)
    df = df.dropna(how= 'all', axis= 1)        

    df.to_csv('MyWork/datasets/xx2_1m.csv', encoding= 'utf-8-sig')
    
def make_dataset3():
    #dataの読込み
    data = pd.read_csv('MyWOrk/original_data/Random_Catalyst_OCM.csv', index_col= 0, header=0)
    target_list =['C2 yield (%)','CH4 conversion (%)', 'C2 selectivity (%)']
    target_list1 =['CH4 conversion (%)', 'C2 selectivity (%)']
    target = data.loc[:, target_list]
    df = data

    #担体のOneHot化
    df = pd.get_dummies(df, columns= ['Support'])
    
    #元素名をmetal1-3に変換する
    metals = df[['M1','M2','M3']].values.tolist()
    metals = list(itertools.chain.from_iterable(metals))   #多次元リストの1次元化
    metals = list(set(metals))
    metals.remove('none')
    #metals.remove(np.nan)
    metals = sorted(metals)

    #一部に元素被りがあるため、そこを修正する
    for index, row in df.iterrows():
        if row[0] == row[1]:
            if row[2] == 'none':
                df.at[index, 'M2'] = 'none'
            else:
                df.at[index, 'M2'] = row[2]
                df.at[index, 'M3'] = 'none'
        elif row[1] == row[2]:
            df.at[index, 'M3'] = 'none'
       
    #等分であるという前提で組成を埋める。        
    metal_df = pd.DataFrame(columns= metals)
    df = pd.concat([df, metal_df], axis= 1)
    df.dropna(how= 'all', axis= 0, inplace= True)
    
    for index, row in df.iterrows():
        metal1 = row[0]
        df.at[index, metal1] = 1/3
        metal2 = row[1]
        metal3 = row[2]
        if metal2 == 'none':
            df.at[index, metal1] = 1
        elif metal2 == metal1:
            if metal3 == 'none':
                df.at[index, metal1] = 1
            else:
                df.at[index, metal1] = 1/2
                df.at[index, metal3] = 1/2
        else:
            df.at[index, metal2] = 1/3
        if metal3 == 'none':
            df.at[index, metal1] = 1/2
            if metal2 == 'none':
                df.at[index, metal1] = 1
            else:
                df.at[index, metal2] = 1/2
            if metal1 == metal2:
                df.at[index, metal1] = 1
        elif metal3 == metal2:
            df.at[index, metal1] = 1/2
            df.at[index, metal2] = 1/2
        else:
            if metal1 == metal2:
                df.at[index, metal3] = 1/2
            else:
                df.at[index, metal3] = 1/3
                
    df = df.dropna(subset = 'PAr (atm)')             
    drop_list = ['M1','M2','M3']
    metal_x = df.copy()
    df.drop(drop_list, axis= 1, inplace= True)
    df.replace(np.nan, 0, inplace= True)
    df = df.reset_index(drop= True)
    check_df = df[metals]
    check = check_df.sum(axis = 1)
    #df.to_csv('MyWork/datasets/xx2_2.csv', encoding= 'utf-8-sig')
    
    #metal_x, x_baseの作成
    metal_x = metal_x.reset_index(drop= True)
    for index, row in metal_x.iterrows():
        metal1 = row[0]
        metal2 = row[1]
        metal3 = row[2]
        metal1_posi = metal_x.columns.get_loc(metal1)
        metal_x.at[index, 'ratio1'] = metal_x.iat[index, metal1_posi]
        if metal2 == 'none':
            pass
        else:
            metal2_posi = metal_x.columns.get_loc(metal2)
            metal_x.at[index, 'ratio2'] = metal_x.iat[index, metal2_posi]
        if metal3 == 'none':
            pass
        else:
            metal3_posi = metal_x.columns.get_loc(metal3)
            metal_x.at[index, 'ratio3'] = metal_x.iat[index, metal3_posi]
        
    metal_x = metal_x[['M1','M2','M3','ratio1','ratio2','ratio3']]
    x_base = df.drop(metals+target_list1, axis= 1)
    metal_x = metal_x.replace(['none', None, ''], np.nan)
    x_base = x_base.replace(['none', None, ''], np.nan)
    
    #xx26の作成
    featurizers = MetalFeaturizers()
    x26 = featurizers.getDesc_X('Type1', x_base, metal_x, xenonpy_element_data, shap_threshold, boruta_p)
    x26.to_csv('MyWork/datasets/xx26_4.csv', encoding='utf-8-sig')    
    
def make_dataset4():
    #dataの読込み
    data = pd.read_excel('Mywork/datasets/data_from_article.xlsx', index_col=0, header=0, engine='openpyxl', sheet_name= 'modified')
    #不要な列の削除、目的変数の設定等
    data = data.iloc[:40, :]
    target = data.loc[:, 'Sc2']
    data = data.loc[:, : 'H2/CO']
    data.insert(0, 'S_c2', target)
    #GHSVに1つnanがあるため、そこは平均値で埋める
    data['GHSV'] = data['GHSV'].replace('na', np.nan)
    data['GHSV'] = data['GHSV'].fillna(data['GHSV'].mean())
    data = data.replace([np.nan, 'na'], np.nan)
    data = data.drop('Support', axis= 1)
    df = data
    
    #元素名をx2の要素に変換する
    metals = df[['metal1','metal2','metal3','metal4']].values.tolist()
    metals = list(itertools.chain.from_iterable(metals))   #多次元リストの1次元化
    metals = list(set(metals))
    metals = [s for s in metals if pd.isnull(s) is False]
    metals = sorted(metals)    
    
    #metal_xの作成        
    metal_df = pd.DataFrame(columns= metals)
    metal_x = pd.concat([df, metal_df], axis= 1)
    
    #metal_x, x_baseの作成
    metal_x = metal_x.reset_index(drop= True)
    for index, row in metal_x.iterrows():
        metal1 = row[1]
        metal2 = row[2]
        metal3 = row[3]
        metal4 = row[4]
        metal1_posi = metal_x.columns.get_loc(metal1)
        metal_x.iat[index, metal1_posi] = metal_x.at[index, 'ratio1']
        if pd.isnull(metal2):
            pass
        else:
            metal2_posi = metal_x.columns.get_loc(metal2)
            metal_x.iat[index, metal2_posi] = metal_x.at[index, 'ratio2']
        if pd.isnull(metal3):
            pass
        else:
            metal3_posi = metal_x.columns.get_loc(metal3)
            metal_x.iat[index, metal3_posi] = metal_x.at[index, 'ratio3']
        if pd.isnull(metal4):
            pass
        else:
            metal4_posi = metal_x.columns.get_loc(metal4)
            metal_x.iat[index, metal4_posi] = metal_x.at[index, 'ratio4']        
    #metal_x = metal_x.replace(['none', None, ''], 0)
    
    metal_list = ['metal1','metal2','metal3','metal4','ratio1','ratio2','ratio3','ratio4']
    FT_x2 = metal_x.drop(metal_list, axis= 1).copy()
    #FT_x2.to_csv('MyWork/datasets/FT_x2.csv')        
    
    metal_xx = metal_x[metal_list]
    x_base = metal_x.loc[:, :'H2/CO']
    x_base = x_base.drop(metal_list, axis= 1)
    #x_base = x_base.drop('S_c2', axis= 1)
    x_base = x_base.replace(['none', None, ''], np.nan)
    
    #xx26の作成
    featurizers = MetalFeaturizers()
    x26 = featurizers.getDesc_X26_1('Type1', x_base, metal_xx, xenonpy_element_data, shap_threshold, boruta_p)
    x26.to_csv('MyWork/datasets/FT_x26.csv', encoding='utf-8-sig')        
    
    
if __name__ == '__main__':
    if dataset == 'OCM':
        make_dataset1()
    elif dataset == 'OCM_Mn':
        make_dataset2()
    elif dataset == 'Random_Catalyst':
        make_dataset3()
    elif dataset == 'FTS':
        make_dataset4()

print('hello')
