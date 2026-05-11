import pandas as pd
import numpy as np
import itertools
import random
import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from libs.boruta import BorutaPy
#from BorutaShap import BorutaShap
from itertools import combinations_with_replacement

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
    last_support = [c for c in df.columns if 'support' in c]
    last_support = last_support[-1]
    df_b = pd.concat([df.loc[:, : last_support], df_b], axis= 1)
    df_b = df_b.loc[:, ~df_b.columns.duplicated()]
    importance_df = pd.DataFrame(index= selected_features, data= selected_importance, columns= ['Importance'])
    importance_df = importance_df.sort_values(by= 'Importance', ascending= False)  
    
    return df_b, importance_df
"""
def BorutaShap_Apply(df):
    y= df[['選択率NPA', '収率NPA']].copy()
    y['sum']= y['選択率NPA']+ y['収率NPA']
    y= y['sum']

    x= df.iloc[:, 2:].copy()
    if '触媒ロット' in x.columns:
        x = x.drop('触媒ロット', axis= 1)
        
    # BorutaShpaの実行 defaultではXGB
    Feature_Selector = BorutaShap(importance_measure= 'shap', classification= False)        
    Feature_Selector.fit(X= x, y= y, n_trials= 100) 
    Feature_Selector.plot(which_features= 'all')
    
    #上で選択した説明変数のみを残す。
    important_df = Feature_Selector.Subset()
    selected_features= important_df.columns.to_list()
    df_a = df.loc[:, :'support_ZrO2_RC100']
    df_b= df.loc[:, selected_features]
    df_c = pd.concat([df_a, df_b], axis= 1)    

    return df_c
"""

# データ読込み
x2_41 = pd.read_csv('datasets/v554/x2_41.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
#x1_41 = pd.read_csv('datasets/v554/x1_41.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
interaction = pd.read_csv('results/diatomic_df.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
# 担体の限定
"""
target = ['選択率NPA','収率NPA']
x1_ce = x1_41[x1_41['support_CeO2_HS'] == 1]     #N=293
x1_ti = x1_41[x1_41['support_TiO2_SSP-M'] == 1]  #N=201
x1_zr = x1_41[x1_41['support_ZrO2_RC100'] == 1]  #N=210
x1_ce_m = x1_ce[x1_ce['前担持'] == 1]  #N=35
x1_ti_m = x1_ti[x1_ti['前担持'] == 1]  #N=110
x1_zr_m = x1_zr[x1_zr['前担持'] == 1]  #N=31
x1_ce_a = x1_ce[x1_ce['前担持'] == 0]
x1_ti_a = x1_ti[x1_ti['前担持'] == 0]
x1_zr_a = x1_zr[x1_zr['前担持'] == 0]

x1_ce_m, importance_cem = Boruta_Apply2(target, x1_ce_m, perc=70)
x1_ce_m.to_csv('datasets/v554/x1_ce_m.csv', encoding='utf-8-sig')
x1_ce_a, importance_cea = Boruta_Apply2(target, x1_ce_a, perc=90)
x1_ce_a.to_csv('datasets/v554/x1_ce_a.csv', encoding='utf-8-sig')

x1_ti_m, importance_cem = Boruta_Apply2(target, x1_ti_m, perc=60)
x1_ti_m.to_csv('datasets/v554/x1_ti_m.csv', encoding='utf-8-sig')
x1_ti_a, importance_cea = Boruta_Apply2(target, x1_ti_a, perc=60)
x1_ti_a.to_csv('datasets/v554/x1_ti_a.csv', encoding='utf-8-sig')

x1_zr_m, importance_cem = Boruta_Apply2(target, x1_zr_m, perc=50)
x1_zr_m.to_csv('datasets/v554/x1_zr_m.csv', encoding='utf-8-sig')
x1_zr_a, importance_cea = Boruta_Apply2(target, x1_zr_a, perc=90)
x1_zr_a.to_csv('datasets/v554/x1_zr_a.csv', encoding='utf-8-sig')
"""
x2_ce = x2_41[x2_41['support_CeO2_HS'] == 1]     #N=293
x2_ti = x2_41[x2_41['support_TiO2_SSP-M'] == 1]  #N=201
x2_zr = x2_41[x2_41['support_ZrO2_RC100'] == 1]  #N=210
x2_ce_m = x2_ce[x2_ce['前担持'] == 1]  #N=35
x2_ti_m = x2_ti[x2_ti['前担持'] == 1]  #N=110
x2_zr_m = x2_zr[x2_zr['前担持'] == 1]  #N=31
x2_ce_a = x2_ce[x2_ce['前担持'] == 0]
x2_ti_a = x2_ti[x2_ti['前担持'] == 0]
x2_zr_a = x2_zr[x2_zr['前担持'] == 0]
"""
x2_ce_m.to_csv('datasets/v554/x2_ce_m.csv', encoding='utf-8-sig')
x2_ce_a.to_csv('datasets/v554/x2_ce_a.csv', encoding='utf-8-sig')
x2_ti_m.to_csv('datasets/v554/x2_ti_m.csv', encoding='utf-8-sig')
x2_ti_a.to_csv('datasets/v554/x2_ti_a.csv', encoding='utf-8-sig')
x2_zr_m.to_csv('datasets/v554/x2_zr_m.csv', encoding='utf-8-sig')
x2_zr_a.to_csv('datasets/v554/x2_zr_a.csv', encoding='utf-8-sig')
"""

# 相互作用項作成
elements = x2_ce.loc[:, 'Al':'Zr']
elements = elements.loc[:, elements.sum(axis=0) != 0]
elements = elements.columns.tolist()
combinations = list(itertools.combinations_with_replacement(elements, 2))
cross_elements = [f"{x}_{y}" for x, y in combinations]

# interactionの不要な列を削除
drop_list = ['data_dir_path','vasprun','outcar','no_error','converged','converged_electronic','converged_ionic','stabilized',
             'atomic_symbol_1','atomic_symbol_2','potcar_symbol_1','potcar_symbol_2','calc_stat']
interaction = interaction.drop(drop_list, axis=1)
interaction = interaction.set_index('system_name')

# interactionのsystem_nameは原子番号順になっているので、alphabet順に更新
interaction.index = pd.Index(["_".join(sorted(idx.split("_"))) for idx in interaction.index])
# interactionの必要な行だけ抽出
interaction = interaction[interaction.index.isin(cross_elements)]
interaction = interaction.sort_index()
#cross_elements = [f'{x}_{y}' for x, y in combinations]
# nanがある列はdrop
interaction = interaction.dropna(how='any', axis=1)
# 必要元素の列名を作成
index_list = interaction.index.tolist()
col_list = interaction.columns.tolist()
new_col_list = [f"{idx}_{col}" for idx in index_list for col in col_list]

# x2に元素の組合せを追加
cross_df = pd.DataFrame(index=x2_ce.index, columns=cross_elements)
x2_ce = pd.concat([x2_ce, cross_df], axis=1)
x2_ce_elements = x2_ce.loc[:, 'Al':'Zr']
x2_ce_elements = x2_ce_elements.loc[:, (x2_ce_elements != 0).any(axis=0)]   #全て0の列は削除
x2_ce_elements = pd.concat([x2_ce_elements, pd.DataFrame(index=x2_ce_elements.index, columns=new_col_list)], axis=1)
x2_ce_elements = x2_ce_elements.replace(np.nan, 0)
    
# x2_elementsの値が0でないところにinteraction内容を代入していく
for index, row in x2_ce_elements.iterrows():
    print('Now_calculating',index,x2_ce_elements.shape[0])
    nonzero = row[row != 0].index.tolist()
    nonzero_cross = list(combinations_with_replacement(nonzero, 2))
    nonzero_cross = ['_'.join(l) for l in nonzero_cross]    
    for i,item in enumerate(nonzero_cross):
        col_name = [c for c in new_col_list if c.startswith(item + "_")]
        x2_ce_elements.loc[index, col_name] = interaction.loc[item, :].values

x2_ce_elements = x2_ce_elements.loc[:, (x2_ce_elements != 0).any(axis=0)]          
x2_ce_ita = pd.concat([x2_ce.loc[:, : 'support_ZrO2_RC100'], x2_ce_elements], axis=1)

# 前担持と後担持に分離
x2_ce_ita_m = x2_ce_ita[x2_ce_ita['前担持'] == 1]
x2_ce_ita_a = x2_ce_ita[x2_ce_ita['前担持'] == 0]

x2_ce_ita_m.to_csv('datasets/v554/x2_ce_ita_m.csv', encoding='utf-8-sig')
x2_ce_ita_a.to_csv('datasets/v554/x2_ce_ita_a.csv', encoding='utf-8-sig')
"""
# 全て0の列を削除
x2_ce_ita_a = pd.read_csv('datasets/v554/x2_ce_ita_a.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_ce_ita_m = pd.read_csv('datasets/v554/x2_ce_ita_m.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_ce_ita = pd.concat([x2_ce_ita_a, x2_ce_ita_m], axis=0)
x2_ce_ita = x2_ce_ita.loc[:, (x2_ce_ita != 0).any(axis=0)]
x2_ce_ita_m = x2_ce_ita[x2_ce_ita['前担持'] == 1]
x2_ce_ita_a = x2_ce_ita[x2_ce_ita['前担持'] == 0]

x2_ce_ita_a.to_csv('datasets/v554/x2_ce_ita_a.csv', encoding='utf-8-sig')
x2_ce_ita_m.to_csv('datasets/v554/x2_ce_ita_m.csv', encoding='utf-8-sig')

# Boruta絞込み
x2_ce_ita_a = pd.read_csv('datasets/v554/x2_ce_ita_a.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_ce_ita_m = pd.read_csv('datasets/v554/x2_ce_ita_m.csv', encoding='utf-8-sig', index_col=0, header=0)
#x2_ce_ita = pd.concat([x2_ce_ita_a, x2_ce_ita_m], axis=0)
target = ['選択率NPA','収率NPA']
#x2_ce_ita_mb = Boruta_Apply2(target, x2_ce_ita_m, perc=40)
#x2_ce_ita_mb.to_csv('datasets/v554/x2_ce_ita_mb.csv', encoding='utf-8-sig')
x2_ce_ita_ab, importance = Boruta_Apply2(target, x2_ce_ita_a, perc=90)
x2_ce_ita_ab.to_csv('datasets/v554/x2_ce_ita_ab.csv', encoding='utf-8-sig')
importance.to_csv('datasets/v554/importance_x2_ce_ita_ab.csv', encoding='utf-8-sig')
"""
  
"""
# 上とは異なるパターンの計算を行う。金属/酸化物に分離して相互作用追加@250120→一旦pending250122
# データ読込み
x2_41 = pd.read_csv('datasets/v554/x2_41.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
interaction = pd.read_csv('results/diatomic_df.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
lanthanoid = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
#reactive_elements_list = ['Ca','Y','Sc','Mg','Sr','Li','Ba','Hf','Al','Zr','Ti','Si','Na','Mn','Cs','K','Cr','Rb','Ga','Zn','In'] + lanthanoid
reactive_elements_list = ['Ca','Y','Sc','Mg','Sr','Li','Ba','Hf','Al','Zr','Ti','Si','Na','Mn','Cs','K','Cr','Rb','Ga','Zn','In','V','Mo','Sn'] + lanthanoid
#reactive_elements_list = ['Ca','Y','Sc','Mg','Sr','Li','Ba','Hf','Al','Zr','Ti','Si','Na','Mn','Cs','K','Cr','Rb','Ga','Zn','In','V','Mo','Sn','Fe','W','Co','Ni','Pb','Cu'] + lanthanoid

# 担体の限定
x2_ce = x2_41[x2_41['support_CeO2_HS'] == 1]     #N=277
x2_ti = x2_41[x2_41['support_TiO2_SSP-M'] == 1]  #N=133
x2_zr = x2_41[x2_41['support_ZrO2_RC100'] == 1]  #N=192
x2_ce_m = x2_ce[x2_ce['前担持'] == 1]  #N=19
x2_ti_m = x2_ti[x2_ti['前担持'] == 1]  #N=42
x2_zr_m = x2_zr[x2_zr['前担持'] == 1]  #N=13
x2_ce_a = x2_ce[x2_ce['前担持'] == 0]
x2_ti_a = x2_ti[x2_ti['前担持'] == 0]
x2_zr_a = x2_zr[x2_zr['前担持'] == 0]

# 相互作用項作成
elements = x2_ti.loc[:, 'Al':'Zr']
elements = elements.loc[:, elements.sum(axis=0) != 0]
elements = elements.columns.tolist()
combinations = list(itertools.combinations_with_replacement(elements, 2))
cross_elements = [f"{x}_{y}" for x, y in combinations]

# interactionの不要な列を削除
drop_list = ['data_dir_path','vasprun','outcar','no_error','converged','converged_electronic','converged_ionic','stabilized',
             'atomic_symbol_1','atomic_symbol_2','potcar_symbol_1','potcar_symbol_2','calc_stat']
interaction = interaction.drop(drop_list, axis=1)
interaction = interaction.set_index('system_name')

# interactionのsystem_nameは原子番号順になっているので、alphabet順に更新
interaction.index = pd.Index(["_".join(sorted(idx.split("_"))) for idx in interaction.index])
# interactionの必要な行だけ抽出
interaction = interaction[interaction.index.isin(cross_elements)]
interaction = interaction.sort_index()
cross_elements = [f'{x}_{y}' for x, y in combinations]
# nanがある列はdrop
interaction = interaction.dropna(how='any', axis=1)
# 必要元素の列名を作成
index_list = interaction.index.tolist()
col_list = interaction.columns.tolist()
new_col_list = [f"{idx}_{col}" for idx in index_list for col in col_list]

# x2に元素の組合せを追加
cross_df = pd.DataFrame(index=x2_ce.index, columns=cross_elements)
x2_ce = pd.concat([x2_ce, cross_df], axis=1)
x2_ce_elements = x2_ce.loc[:, 'Al':'Zr']
x2_ce_elements = x2_ce_elements.loc[:, (x2_ce_elements != 0).any(axis=0)]   #全て0の列は削除
x2_ce_elements = pd.concat([x2_ce_elements, pd.DataFrame(index=x2_ce_elements.index, columns=new_col_list)], axis=1)
x2_ce_elements = x2_ce_elements.replace(np.nan, 0)
    
# x2_elementsの値が0でないところにinteraction内容を代入していく
for index, row in x2_ce_elements.iterrows():
    nonzero = row[row != 0].index.tolist()
    nonzero_cross = list(combinations_with_replacement(nonzero, 2))
    nonzero_cross = ['_'.join(l) for l in nonzero_cross]    
    for i,item in enumerate(nonzero_cross):
        col_name = [c for c in new_col_list if c.startswith(item + "_")]
        x2_ce_elements.loc[index, col_name] = interaction.loc[item, :].values

x2_ce_elements = x2_ce_elements.loc[:, (x2_ce_elements != 0).any(axis=0)]          
x2_ce_ita = pd.concat([x2_ce.loc[:, : 'support_ZrO2_RC100'], x2_ce_elements], axis=1)
#x2_ce_ita.to_csv('datasets/v509/x2_ce_ita.csv', encoding='utf-8-sig')

# 前担持と後担持に分離
x2_ce_ita_m = x2_ce_ita[x2_ce_ita['前担持'] == 1]
x2_ce_ita_a = x2_ce_ita[x2_ce_ita['前担持'] == 0]

x2_ce_ita_m.to_csv('datasets/v509/x2_ce_ita_m.csv', encoding='utf-8-sig')
x2_ce_ita_a.to_csv('datasets/v509/x2_ce_ita_a.csv', encoding='utf-8-sig')

# Boruta絞込み

x2_ti_ita = pd.read_csv('datasets/v509/x2_ti_ita.csv', encoding='utf-8-sig', index_col=0, header=0)
target = ['選択率NPA','収率NPA']
x2_ti_ita_b = Boruta_Apply2(target, x2_ti_ita, perc=60)

x2_ti_ita_b.to_csv('datasets/v509/x2_ti_ita_b.csv', encoding='utf-8-sig')


x2_ti_ita_a = pd.read_csv('datasets/v509/x2_ti_ita_a.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_ti_ita_m = pd.read_csv('datasets/v509/x2_ti_ita_m.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_ti_ita = pd.concat([x2_ti_ita_a, x2_ti_ita_m], axis=0)
target = ['選択率NPA','収率NPA']
x2_ti_ita_b = Boruta_Apply2(target, x2_ti_ita, perc=70)

x2_ti_ita_mb = x2_ti_ita_b[x2_ti_ita_b['前担持'] == 1]
x2_ti_ita_ab = x2_ti_ita_b[x2_ti_ita_b['前担持'] == 0]

x2_ti_ita_ab.to_csv('datasets/v509/x2_ti_ita_ab.csv', encoding='utf-8-sig')
x2_ti_ita_mb.to_csv('datasets/v509/x2_ti_ita_mb.csv', encoding='utf-8-sig')


x2_ce_ita_a = pd.read_csv('datasets/v509/x2_ce_ita_a.csv', encoding='utf-8-sig', index_col=0, header=0)
x2_ce_a = x2_ce_ita_a.loc[:, :'Zr']

x2_ce_a.to_csv('datasets/v509/x2_ce_a.csv', encoding='utf-8-sig')
"""

print('End')