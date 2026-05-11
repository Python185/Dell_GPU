import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.linear_model import Ridge,Lasso,ElasticNet,ElasticNetCV
from dcekit.variable_selection import cvpfi
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import ast


def Boruta_Apply(df, perc):
    #y= df[['選択率NPA', '収率NPA']].copy()
    #y['sum']= y['選択率NPA']+ y['収率NPA']
    #y= y['sum']
    y= df['選択率NPA'].copy()
    #y= df['収率NPA'].copy()
    x = df.drop(['選択率NPA', '収率NPA'], axis= 1)
    if '触媒ロット' in x.columns:
        x = x.drop('触媒ロット', axis= 1)

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
    df_a = df.loc[:, :'触媒ロット']
    df_b= df.loc[:, selected_features]
    df_c = pd.concat([df_a, df_b], axis= 1)
    
    return df_c

#FT-IRを描画させる index, columnsの数値が文字となっていることに気づかず1日近く苦悶・・
def plot_with_interval_ticks(df, interval, catalyst):
    df.index = pd.to_numeric(df.index, errors='coerce')
    catalyst.columns = pd.to_numeric(catalyst.columns, errors='coerce')
    x_values = df.index.values
    # Set up the plot
    plt.figure(figsize=(10, 6))
    # Plot each column against the first column
    for col in df.columns[1:]:
        plt.plot(x_values, df[col], label=col)
    # x軸を反転
    plt.gca().invert_xaxis()
 
    # Determine x-axis ticks based on the data range and interval
    xmin, xmax = df.index.values.min(), df.index.values.max()
    x_ticks = np.arange(4200, 900, -300)
    #x_ticks = [4200,3900,3600,3300,3000,2700,2400,2100,1800,1500,1200,900]
    #plt.xlim(4200, 900)
    plt.ylim(df.min().min(), df.max().max()+0.1)
 
    # Set legend, labels, and title
    plt.legend(fontsize= 10, loc= 'upper right')
    plt.xlabel('Wave number/cm-1', fontsize= 20)
    plt.ylabel('Value', fontsize= 20)
    plt.xticks(fontsize= 16)  
    plt.yticks(fontsize= 16)

    peaks = catalyst.columns.to_list()
    #plt.vlines(x= peaks, ymin= df.min().min(), ymax= df.max().max()+0.1, color= 'g', linestyles= 'solid', linewidth= 0.5) 
    plt.vlines(x= peaks, ymin= -0.05, ymax= 0.25, color= 'g', linestyles= 'solid', linewidth= 0.5) 
    
    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# データ読込み
x2 = pd.read_csv('datasets/v592/x2_41.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
catalyst_lot = sorted(x2['触媒ロット'].unique().tolist())
original_data = pd.read_excel('file/★データセット_250428_v592.xlsx', sheet_name= 'データセット', header= 0, index_col= 0)

# x2にエタノールの結果を追加
original_data = original_data[original_data.index.isin(x2.index)]
x2['選択率ETA'] = original_data['選択率エタノール※※']
x2['収率ETA'] = original_data['選択率エタノール※※'] * original_data['転化率※※'] / 100

x2_columns_list = ['選択率NPA','収率NPA','選択率ETA', '収率ETA'] + x2.loc[:, '前処理還元炉温℃':'Zr'].columns.tolist()
x2 = x2[x2_columns_list]

# 仕込み組成でなくXRFデータを使用
original_data = original_data.loc[:, 'Li':'Bi']
x2 = pd.concat([x2.loc[:, :'support_ZrO2_RC100'], original_data], axis= 1)

# x2の評価条件を固定
#x2 = x2[x2['評価反応炉温℃'] == 280]   #N=823
x2 = x2[x2['support_Al2O3_A-11'] == 1]   #N=2236
x2_catalyst_lot = sorted(x2['触媒ロット'].unique().tolist())   #N=820
# 複数のプロセス条件のある触媒は削除
x2_duplicated_catalyst = x2[x2.duplicated(subset= ['触媒ロット'], keep= False)]
x2 = x2.drop(x2_duplicated_catalyst.index)   #N=817
x2_catalyst_lot = sorted(x2['触媒ロット'].unique().tolist())   #N=817
x2 = x2[x2['Rh'] != 0]   #N=761, 1135

"""
# x2の評価条件を固定
x2 = x2[x2['評価反応炉温℃'] == 260]   #N=276
x2_catalyst_lot = sorted(x2['触媒ロット'].unique().tolist())   #N=232
# 複数のプロセス条件のある触媒は削除
x2_duplicated_catalyst = x2[x2.duplicated(subset= ['触媒ロット'], keep= False)]
x2 = x2.drop(x2_duplicated_catalyst.index)   #N=188
x2_catalyst_lot = sorted(x2['触媒ロット'].unique().tolist())   #N=188
#x2 = x2[x2['Rh'] != 0]   #N=175
"""

# x2_41の触媒に合わせてFT-IRデータの読込み
file_lists = [s + '_2.csv' for s in x2_catalyst_lot]
path= './FTIR_data/NIT_HTdata/'
#DataFrame形式で読込み
dic = {}
for file in file_lists:
    file_path = path + file
    if os.path.exists(file_path):
        dic[file] = pd.read_csv(file_path, index_col=0, header=None, skiprows=0)
    else:
        print(f"File not found: {file_path}, skipping.")
#dictのkey名修正
dict = {key[:-6]: value for key, value in dic.items()}

# dictのdata数確認
# 最初のデータを基準としてデータ数を確認
reference_length = len(next(iter(dict.values())))

# データ数が一致しないものを別のdict_2に移動
dict_2 = {}
for key, value in list(dict.items()):
    if len(value) != reference_length:
        dict_2[key] = dict.pop(key)

# dict_2は3つであり、少ないため無視することとする。

# 目的変数、組成のユークリッド距離、FT-IRデータの相関係数の計算
# FT-IRについては、相関係数の計算のみであるため今回はベースライン処理は行わない。
# x2にFT-IRデータを追加する
FTIR_col = dict[x2_catalyst_lot[0]].T.columns
for index, row in x2.iterrows():
    catalyst_lot = row['触媒ロット']
    if catalyst_lot in dict.keys():
        ftir_data = dict[catalyst_lot].T
        x2.loc[index, FTIR_col] = ftir_data.values[0]
    else:
        x2.loc[index, FTIR_col] = np.nan

x2 = x2.dropna(subset= FTIR_col)  #N=123
x2 = x2.loc[:,(x2 != 0).any(axis=0)]
x2.to_csv('FTIR_data/datasets/x2_all_ir_al.csv', encoding= 'utf-8-sig')

"""
# データ読込み
x2 = pd.read_csv('FTIR_data/datasets/x2_280_ir.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
# x2はRh必須
x2 = x2[x2['Rh'] != 0]   #N=110@260C, N=625@280C
x2 = x2.sort_index(axis= 0)

# nan_check
nan_check = x2.isnull().sum()
x2 = x2.dropna(how= 'any')

x2_a = x2[x2['support_Al2O3_A-11'] == 1]
x2_c = x2[x2['support_CeO2_HS'] == 1]
x2_z = x2[x2['support_ZrO2_RC100'] == 1]
x2_t = x2[x2['support_TiO2_SSP-M'] == 1]

x2 = x2_a

# distance_corr_dfの作成
index_list = list(combinations(x2.index, 2))

distance_corr_df = pd.DataFrame(index=index_list, columns=[
    'NPA_distance', 'ETA_distance', 'compo_distance', 'ir_corr', 'NPA_s_dis', 'NPA_y_dis', 'ETA_s_dis', 'ETA_y_dis'
])

compo = x2.loc[:, 'Al':'Zr']
NPA = x2[['選択率NPA', '収率NPA']]
ETA = x2[['選択率ETA', '収率ETA']]
#ftir = x2.loc[:, '998.9622':'3999.706']
ftir = x2.loc[:, '1700.936':'2300.699']

scaler = StandardScaler()
NPA = pd.DataFrame(scaler.fit_transform(NPA), columns=NPA.columns, index=NPA.index)
ETA = pd.DataFrame(scaler.fit_transform(ETA), columns=ETA.columns, index=ETA.index)
#compo = pd.DataFrame(scaler.fit_transform(compo), columns=compo.columns, index=compo.index)

# ユークリッド距離の計算
for idx1, idx2 in index_list:
    distance_corr_df.loc[[(idx1, idx2)], 'compo_distance'] = np.linalg.norm(compo.loc[idx1] - compo.loc[idx2])
    distance_corr_df.loc[[(idx1, idx2)], 'NPA_distance'] = np.linalg.norm(NPA.loc[idx1] - NPA.loc[idx2])
    distance_corr_df.loc[[(idx1, idx2)], 'ETA_distance'] = np.linalg.norm(ETA.loc[idx1] - ETA.loc[idx2])
    distance_corr_df.loc[[(idx1, idx2)], 'NPA_s_dis'] = np.linalg.norm(NPA.loc[idx1, '選択率NPA'] - NPA.loc[idx2, '選択率NPA'])
    distance_corr_df.loc[[(idx1, idx2)], 'NPA_y_dis'] = np.linalg.norm(NPA.loc[idx1, '収率NPA'] - NPA.loc[idx2, '収率NPA'])
    distance_corr_df.loc[[(idx1, idx2)], 'ETA_s_dis'] = np.linalg.norm(ETA.loc[idx1, '選択率ETA'] - ETA.loc[idx2, '選択率ETA'])
    distance_corr_df.loc[[(idx1, idx2)], 'ETA_y_dis'] = np.linalg.norm(ETA.loc[idx1, '収率ETA'] - ETA.loc[idx2, '収率ETA'])
    distance_corr_df.loc[[(idx1, idx2)], 'ir_corr'] = ftir.loc[idx1].corr(ftir.loc[idx2])

distance_corr_df = distance_corr_df.sort_values(by= 'compo_distance', ascending= True)
distance_corr_df.to_csv('FTIR_data/datasets/distance_corr_df_280_al.csv', encoding= 'utf-8-sig')

# plot
# 1つ目のプロット
plt.subplot(2, 3, 1)
sc = plt.scatter(distance_corr_df['compo_distance'], distance_corr_df['ir_corr'], 
                 c=distance_corr_df['NPA_distance'], cmap='viridis')
plt.xlabel('Composition distance', fontsize=18)
plt.ylabel('Correlation FT-IR', fontsize=18)
plt.colorbar(sc, label='NPA_distance')

# 2つ目のプロット
plt.subplot(2, 3, 4)
sc = plt.scatter(distance_corr_df['compo_distance'], distance_corr_df['ir_corr'], 
                 c=distance_corr_df['ETA_distance'], cmap='viridis')
plt.xlabel('Composition_distance', fontsize=18)
plt.ylabel('Correlation_FTIR', fontsize=18)
plt.colorbar(sc, label='ETA_distance')

# 3つ目のプロット
plt.subplot(2, 3, 2)
sc = plt.scatter(distance_corr_df['compo_distance'], distance_corr_df['ir_corr'], 
                 c=distance_corr_df['NPA_s_dis'], cmap='viridis')
plt.xlabel('Composition_distance', fontsize=18)
plt.ylabel('Correlation_FTIR', fontsize=18)
plt.colorbar(sc, label='NPA_selectivity_distance')

# 4つ目のプロット
plt.subplot(2, 3, 3)
sc = plt.scatter(distance_corr_df['compo_distance'], distance_corr_df['ir_corr'], 
                 c=distance_corr_df['NPA_y_dis'], cmap='viridis')
plt.xlabel('Composition_distance', fontsize=18)
plt.ylabel('Correlation_FTIR', fontsize=18)
plt.colorbar(sc, label='NPA_yield_distance')

# 5つ目のプロット
plt.subplot(2, 3, 5)
sc = plt.scatter(distance_corr_df['compo_distance'], distance_corr_df['ir_corr'], 
                 c=distance_corr_df['ETA_s_dis'], cmap='viridis')
plt.xlabel('Composition_distance', fontsize=18)
plt.ylabel('Correlation_FTIR', fontsize=18)
plt.colorbar(sc, label='ETA_selectivity_distance')

# 6つ目のプロット
plt.subplot(2, 3, 6)
sc = plt.scatter(distance_corr_df['compo_distance'], distance_corr_df['ir_corr'], 
                 c=distance_corr_df['ETA_y_dis'], cmap='viridis')
plt.xlabel('Composition_distance', fontsize=18)
plt.ylabel('Correlation_FTIR', fontsize=18)
plt.colorbar(sc, label='ETA_yield_distance')

# 図の間隔を調整
plt.subplots_adjust(wspace=0.2, hspace=0.2)  # wspaceとhspaceを適切な値に設定

#plt.tight_layout()
plt.show()
"""
"""
# データ読込み
x2_260_ir = pd.read_csv('FTIR_data/datasets/x2_260_ir.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
distance_corr_df_al = pd.read_csv('FTIR_data/datasets/distance_corr_df_260_al.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
distance_corr_df_ce = pd.read_csv('FTIR_data/datasets/distance_corr_df_260_ce.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
distance_corr_df_ti = pd.read_csv('FTIR_data/datasets/distance_corr_df_260_ti.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
distance_corr_df_zr = pd.read_csv('FTIR_data/datasets/distance_corr_df_260_zr.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)

# 距離の近いデータの取出し
x2_260_al = distance_corr_df_al.loc[distance_corr_df_al['compo_distance'] < 5.0, :].index.tolist()
x2_260_ce = distance_corr_df_ce.loc[distance_corr_df_ce['compo_distance'] < 5.0, :].index.tolist()
x2_260_ti = distance_corr_df_ti.loc[distance_corr_df_ti['compo_distance'] < 5.0, :].index.tolist()
x2_260_zr = distance_corr_df_zr.loc[distance_corr_df_zr['compo_distance'] < 5.0, :].index.tolist()
# まとめてsupport列を挿入
x2_al = [ast.literal_eval(t) for t in x2_260_al]
x2_ce = [ast.literal_eval(t) for t in x2_260_ce]
x2_ti = [ast.literal_eval(t) for t in x2_260_ti]
x2_zr = [ast.literal_eval(t) for t in x2_260_zr]
x2_al = np.array(x2_al).flatten().tolist()
x2_ce = np.array(x2_ce).flatten().tolist()
x2_ti = np.array(x2_ti).flatten().tolist()
x2_zr = np.array(x2_zr).flatten().tolist()
x2_260 = x2_al + x2_ce + x2_ti + x2_zr
x2_260_df = x2_260_ir.loc[x2_260, :]
# support列を追加して必要な列を取出し保存
support_list_al = ['Al2O3'] * len(x2_al)
support_list_ce = ['CeO2'] * len(x2_ce)
support_list_ti = ['TiO2'] * len(x2_ti)
support_list_zr = ['ZrO2'] * len(x2_zr)
support_list = support_list_al + support_list_ce + support_list_ti + support_list_zr
x2_260_df.insert(0, 'support', support_list)
x2_260_df = pd.concat([x2_260_df.loc[:, :'収率ETA'], x2_260_df.loc[:, 'Al':'Pb']], axis= 1)
x2_260_df = x2_260_df.loc[:, (x2_260_df != 0).any(axis=0)]
#x2_260_df.to_excel('FTIR_data/datasets/x2_260_data.xlsx')

ir_260 = x2_260_ir.loc[x2_260, :]
ir_260 = ir_260.loc[:, '1700.936':'2300.699']
ir_260_al = ir_260.iloc[:172, :]
ir_260_ce = ir_260.iloc[172:218, :]
ir_260_ti = ir_260.iloc[218:230, :]
ir_260_zr = ir_260.iloc[230:236, :]

# FT-IRの描画 al:86pair, ce:23pair, ti:6pair, zr:3pair
plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 18

for i in range(0, 86*2, 2):
    plt.figure()  # Create a new figure for each pair of plots
    plt.plot(ir_260_al.columns.astype(float), ir_260_al.iloc[i, :].to_numpy(), label=ir_260_al.index[i])
    plt.plot(ir_260_al.columns.astype(float), ir_260_al.iloc[i+1, :].to_numpy(), label=ir_260_al.index[i+1])
    plt.legend()
    plt.title(f'Al2O3_{int(i/2)}')
    plt.xticks(np.arange(1700, 2301, 100))
    plt.xlabel('Wavenumber, cm-1')
    plt.ylabel('Instensity')
    plt.gca().invert_xaxis()
    plt.savefig(f'FTIR_data/figures/Al2O3_260_{int(i/2)}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to avoid overlapping plots

for i in range(0, 23*2, 2):
    plt.figure()  # Create a new figure for each pair of plots
    plt.plot(ir_260_ce.columns.astype(float), ir_260_ce.iloc[i, :].to_numpy(), label=ir_260_ce.index[i])
    plt.plot(ir_260_ce.columns.astype(float), ir_260_ce.iloc[i+1, :].to_numpy(), label=ir_260_ce.index[i+1])
    plt.legend()
    plt.title(f'CeO2_{int(i/2)}')
    plt.xticks(np.arange(1700, 2301, 100))
    plt.xlabel('Wavenumber, cm-1')
    plt.ylabel('Instensity')
    plt.gca().invert_xaxis()
    plt.savefig(f'FTIR_data/figures/CeO2_260_{int(i/2)}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to avoid overlapping plots

for i in range(0, 6*2, 2):
    plt.figure()  # Create a new figure for each pair of plots
    plt.plot(ir_260_ti.columns.astype(float), ir_260_ti.iloc[i, :].to_numpy(), label=ir_260_ti.index[i])
    plt.plot(ir_260_ti.columns.astype(float), ir_260_ti.iloc[i+1, :].to_numpy(), label=ir_260_ti.index[i+1])
    plt.legend()
    plt.title(f'TiO2_{int(i/2)}')
    plt.xticks(np.arange(1700, 2301, 100))
    plt.xlabel('Wavenumber, cm-1')
    plt.ylabel('Instensity')
    plt.gca().invert_xaxis()
    plt.savefig(f'FTIR_data/figures/TiO2_260_{int(i/2)}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to avoid overlapping plots

for i in range(0, 3*2, 2):
    plt.figure()  # Create a new figure for each pair of plots
    plt.plot(ir_260_zr.columns.astype(float), ir_260_zr.iloc[i, :].to_numpy(), label=ir_260_zr.index[i])
    plt.plot(ir_260_zr.columns.astype(float), ir_260_zr.iloc[i+1, :].to_numpy(), label=ir_260_zr.index[i+1])
    plt.legend()
    plt.title(f'ZrO2_{int(i/2)}')
    plt.xticks(np.arange(1700, 2301, 100))
    plt.xlabel('Wavenumber, cm-1')
    plt.ylabel('Instensity')
    plt.gca().invert_xaxis()
    plt.savefig(f'FTIR_data/figures/ZrO2_260_{int(i/2)}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to avoid overlapping plots
"""
print('hello')
