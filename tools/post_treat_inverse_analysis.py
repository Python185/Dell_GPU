# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.figure as figure
plt.rcParams['font.family'] = 'Meiryo' 

# 逆解析フォルダの指定等
saved_dir = 'result/inverse_analysis/True/'
save_dir = 'result/inverse_analysis/'
metal_col = ['metal1', 'metal2', 'metal3', 'metal4', 'metal5']

variable_number_1 = '選択率NPA_estimated_inverse_y'
variable_number_2 = '選択率NPA_estimated_inverse_y_std'
variable_number_3 = '収率NPA_estimated_inverse_y'
variable_number_4 = '収率NPA_estimated_inverse_y_std'

# データ内容確認
CeO2_pareto = pd.read_csv(save_dir+'CeO2_pareto_opt.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
CeO2_pareto = CeO2_pareto[~CeO2_pareto[metal_col].eq('Te').any(axis=1)]
dataset = pd.read_csv(saved_dir+'Y_ALL_PTR_True_ZrO2_RC100_BO_results.csv', encoding= 'cp932', index_col= 0, header= 0)

# データの縮小
ZrO2_pareto = pd.read_csv(save_dir+'ZrO2_pareto_opt.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
ZrO2_pareto = ZrO2_pareto.sort_values('one_metrics', ascending=False)
ZrO2_pareto_1000 = ZrO2_pareto.sample(n=100, random_state=11)
ZrO2_pareto_1000.to_csv(save_dir+'ZrO2_pareto_opt_100.csv', encoding='utf-8-sig')
dataset_selected = ZrO2_pareto_1000

# パレート最適解が赤点で表示されます (0, 1列)
plt.rcParams['font.size'] = 18
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 図1: 0, 1列
axes[0].scatter(dataset.loc[:, variable_number_2], dataset.loc[:, variable_number_1], c='blue')
axes[0].scatter(dataset_selected.loc[:, variable_number_2], dataset_selected.loc[:, variable_number_1], c='red')
axes[0].set_title('選択率NPA', fontsize=18)
axes[0].set_xlabel('y_std')
axes[0].set_ylabel('predicted_y')

# 図2: 2, 3列
axes[1].scatter(dataset.loc[:, variable_number_4], dataset.loc[:, variable_number_3], c='blue')
axes[1].scatter(dataset_selected.loc[:, variable_number_4], dataset_selected.loc[:, variable_number_3], c='red')
axes[1].set_title('収率NPA', fontsize=18)
axes[1].set_xlabel('y_std')
axes[1].set_ylabel('predicted_y')

plt.tight_layout()
plt.savefig(save_dir+'ZrO2_pareto_opt_100.png', bbox_inches='tight')  # 図の保存
#plt.show()
print('End')

"""
# パレート最適解(選択率、収率最大化)
# ZrO2
dataset = pd.read_csv(saved_dir+'Y_ALL_PTR_True_ZrO2_RC100_BO_results.csv', encoding= 'cp932', index_col= 0, header= 0)
dataset = dataset.sort_values('one_metrics', ascending=False)
#dataset = dataset.sample(n=1000, random_state=11)

# パレート最適解の探索 (4変数に拡張)
selected_columns = [variable_number_1, variable_number_2, variable_number_3, variable_number_4]  # 変数番号を指定
dataset_selected = dataset.loc[:, selected_columns]
pareto_optimal_indexes = []
for sample_number in range(dataset_selected.shape[0]):
    flag = dataset_selected <= dataset_selected.iloc[sample_number, :]
    if flag.any(axis=1).all():
        pareto_optimal_indexes.append(sample_number)

# パレート最適解が赤点で表示されます (0, 1列)
plt.rcParams['font.size'] = 18
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 図1: 0, 1列
axes[0].scatter(dataset_selected.iloc[:, 1], dataset_selected.iloc[:, 0], c='blue')
axes[0].scatter(dataset_selected.iloc[pareto_optimal_indexes, 1], dataset_selected.iloc[pareto_optimal_indexes, 0], c='red')
axes[0].set_title('選択率NPA', fontsize=18)
axes[0].set_xlabel('y_std')
axes[0].set_ylabel('predicted_y')

# 図2: 2, 3列
axes[1].scatter(dataset_selected.iloc[:, 3], dataset_selected.iloc[:, 2], c='blue')
axes[1].scatter(dataset_selected.iloc[pareto_optimal_indexes, 3], dataset_selected.iloc[pareto_optimal_indexes, 2], c='red')
axes[1].set_title('収率NPA', fontsize=18)
axes[1].set_xlabel('y_std')
axes[1].set_ylabel('predicted_y')

plt.tight_layout()
plt.savefig(save_dir+'ZrO2_pareto_opt.png', bbox_inches='tight')  # 図の保存
#plt.show()

# 保存
pareto_optimal_dataset = dataset.iloc[pareto_optimal_indexes, :]
pareto_optimal_dataset = pareto_optimal_dataset.sort_values('one_metrics', ascending=False)
pareto_optimal_dataset.to_csv(save_dir+'ZrO2_pareto_opt.csv', encoding='utf-8-sig')


# CeO2
dataset = pd.read_csv(saved_dir+'Y_ALL_PTR_True_CeO2_HS_BO_results.csv', encoding= 'cp932', index_col= 0, header= 0)
dataset = dataset.sort_values('one_metrics', ascending=False)
#dataset = dataset.sample(n=1000, random_state=11)

# パレート最適解の探索 (4変数に拡張)
selected_columns = [variable_number_1, variable_number_2, variable_number_3, variable_number_4]  # 変数番号を指定
dataset_selected = dataset.loc[:, selected_columns]
pareto_optimal_indexes = []
for sample_number in range(dataset_selected.shape[0]):
    flag = dataset_selected <= dataset_selected.iloc[sample_number, :]
    if flag.any(axis=1).all():
        pareto_optimal_indexes.append(sample_number)

# パレート最適解が赤点で表示されます (0, 1列)
plt.rcParams['font.size'] = 18
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 図1: 0, 1列
axes[0].scatter(dataset_selected.iloc[:, 1], dataset_selected.iloc[:, 0], c='blue')
axes[0].scatter(dataset_selected.iloc[pareto_optimal_indexes, 1], dataset_selected.iloc[pareto_optimal_indexes, 0], c='red')
axes[0].set_title('選択率NPA', fontsize=18)
axes[0].set_xlabel('y_std')
axes[0].set_ylabel('predicted_y')

# 図2: 2, 3列
axes[1].scatter(dataset_selected.iloc[:, 3], dataset_selected.iloc[:, 2], c='blue')
axes[1].scatter(dataset_selected.iloc[pareto_optimal_indexes, 3], dataset_selected.iloc[pareto_optimal_indexes, 2], c='red')
axes[1].set_title('収率NPA', fontsize=18)
axes[1].set_xlabel('y_std')
axes[1].set_ylabel('predicted_y')

plt.tight_layout()
plt.savefig(save_dir+'CeO2_pareto_opt.png', bbox_inches='tight')  # 図の保存
#plt.show()

# 保存
pareto_optimal_dataset = dataset.iloc[pareto_optimal_indexes, :]
pareto_optimal_dataset = pareto_optimal_dataset.sort_values('one_metrics', ascending=False)
pareto_optimal_dataset.to_csv(save_dir+'CeO2_pareto_opt.csv', encoding='utf-8-sig')


print('End')
"""

"""
#逆解析の図を作成する

#ZrO2
plot_zr = pd.read_csv(saved_dir+'Y_ALL_PTR_True_ZrO2_RC100_BO_results.csv', encoding= 'cp932', index_col= 0, header= 0)
plot_zr = plot_zr.sort_values('one_metrics', ascending=False)

plt.rcParams['font.size'] = 22
y_std = plot_zr['選択率NPA_estimated_inverse_y_std']
predicted_y = plot_zr['選択率NPA_estimated_inverse_y']
y_std_top500 = plot_zr.iloc[:500, :]['選択率NPA_estimated_inverse_y_std']
predicted_y_top500 = plot_zr.iloc[:500, :]['選択率NPA_estimated_inverse_y']

plt.rcParams['font.size'] = 22
plt.figure(figsize=figure.figaspect(1)) # 正方形
plt.scatter(y_std, predicted_y, c='blue', alpha=0.7, edgecolors='blue', s= 60) # プロット
plt.scatter(y_std_top500, predicted_y_top500, c='red', alpha=0.7, edgecolors='blue', s= 60) # プロット

plt.title('選択率NPA_ZrO2', fontname= 'Meiryo')
plt.legend(['all', 'top500'], loc='upper right', fontsize=12)
#plt.ylim(-0.1, 6.0) # Y のサイズ
#plt.xlim(-0.1, 6.0) # X のサイズ
#plt.xticks([0, 2.0, 4.0, 6.0])

plt.xlabel('y_std') #　縦軸ラベル
plt.ylabel('predicted_y') #　横軸ラベル
plt.savefig(saved_dir+'ZrO2_s.png',bbox_inches = 'tight') # 図の保存
#plt.show() # 図の描画

y_std_y = plot_zr['収率NPA_estimated_inverse_y_std']
predicted_y_y = plot_zr['収率NPA_estimated_inverse_y']
y_std_top500_y = plot_zr.iloc[:500, :]['収率NPA_estimated_inverse_y_std']
predicted_y_top500_y = plot_zr.iloc[:500, :]['収率NPA_estimated_inverse_y']

plt.rcParams['font.size'] = 22
plt.figure(figsize=figure.figaspect(1)) # 正方形
plt.scatter(y_std_y, predicted_y_y, c='blue', alpha=0.7, edgecolors='blue', s= 60) # プロット
plt.scatter(y_std_top500_y, predicted_y_top500_y, c='red', alpha=0.7, edgecolors='blue', s= 60) # プロット

plt.title('収率NPA_ZrO2', fontname= 'Meiryo')
plt.legend(['all', 'top500'], loc='upper right', fontsize=12)
#plt.ylim(-0.1, 1.0) # Y のサイズ
#plt.xlim(-0.1, 1.0) # X のサイズ
#plt.xticks([0, 2.0, 4.0, 6.0])

plt.xlabel('y_std') #　縦軸ラベル
plt.ylabel('predicted_y') #　横軸ラベル
plt.savefig(saved_dir+'ZrO2_y.png',bbox_inches = 'tight') # 図の保存
#plt.show() # 図の描画

# CeO2
plot_ce = pd.read_csv(saved_dir+'Y_ALL_PTR_True_CeO2_HS_BO_results.csv', encoding= 'cp932', index_col= 0, header= 0)
plot_ce = plot_ce.sort_values('one_metrics', ascending=False)

plt.rcParams['font.size'] = 22
y_std = plot_ce['選択率NPA_estimated_inverse_y_std']
predicted_y = plot_ce['選択率NPA_estimated_inverse_y']
y_std_top500 = plot_ce.iloc[:500, :]['選択率NPA_estimated_inverse_y_std']
predicted_y_top500 = plot_ce.iloc[:500, :]['選択率NPA_estimated_inverse_y']
noTe_100_df = plot_ce[~plot_ce[metal_col].eq('Te').any(axis=1)]
y_std_noTe_100 = noTe_100_df.iloc[:100, :]['選択率NPA_estimated_inverse_y_std']
predicted_y_noTe_100 = noTe_100_df.iloc[:100, :]['選択率NPA_estimated_inverse_y']

plt.rcParams['font.size'] = 22
plt.figure(figsize=figure.figaspect(1)) # 正方形
plt.scatter(y_std, predicted_y, c='blue', alpha=0.7, edgecolors='blue', s= 60) # プロット
plt.scatter(y_std_top500, predicted_y_top500, c='red', alpha=0.7, edgecolors='blue', s= 60) # プロット
plt.scatter(y_std_noTe_100, predicted_y_noTe_100, c='green', alpha=0.7, edgecolors='blue', s= 60) # プロット

plt.title('選択率NPA_CeO2', fontname= 'Meiryo')
plt.legend(['all', 'top500', 'noTe_100'], loc='upper right', fontsize=12)
#plt.ylim(-0.1, 6.0) # Y のサイズ
#plt.xlim(-0.1, 6.0) # X のサイズ
#plt.xticks([0, 2.0, 4.0, 6.0])

plt.xlabel('y_std') #　縦軸ラベル
plt.ylabel('predicted_y') #　横軸ラベル
plt.savefig(saved_dir+'CeO2_s.png',bbox_inches = 'tight') # 図の保存
#plt.show() # 図の描画

y_std_y = plot_ce['収率NPA_estimated_inverse_y_std']
predicted_y_y = plot_ce['収率NPA_estimated_inverse_y']
y_std_top500_y = plot_ce.iloc[:500, :]['収率NPA_estimated_inverse_y_std']
predicted_y_top500_y = plot_ce.iloc[:500, :]['収率NPA_estimated_inverse_y']
noTe_100_df_y = plot_ce[~plot_ce[metal_col].eq('Te').any(axis=1)]
y_std_noTe_100_y = noTe_100_df_y.iloc[:100, :]['収率NPA_estimated_inverse_y_std']
predicted_y_noTe_100_y = noTe_100_df_y.iloc[:100, :]['収率NPA_estimated_inverse_y']

plt.rcParams['font.size'] = 22
plt.figure(figsize=figure.figaspect(1)) # 正方形
plt.scatter(y_std_y, predicted_y_y, c='blue', alpha=0.7, edgecolors='blue', s= 60) # プロット
plt.scatter(y_std_top500_y, predicted_y_top500_y, c='red', alpha=0.7, edgecolors='blue', s= 60) # プロット
plt.scatter(y_std_noTe_100_y, predicted_y_noTe_100_y, c='green', alpha=0.7, edgecolors='blue', s= 60) # プロット

plt.title('収率NPA_CeO2', fontname= 'Meiryo')
plt.legend(['all', 'top500', 'noTe_100'], loc='upper right', fontsize=12)
#plt.ylim(-0.1, 1.0) # Y のサイズ
#plt.xlim(-0.1, 1.0) # X のサイズ
#plt.xticks([0, 2.0, 4.0, 6.0])

plt.xlabel('y_std') #　縦軸ラベル
plt.ylabel('predicted_y') #　横軸ラベル
plt.savefig(saved_dir+'CeO2_y.png',bbox_inches = 'tight') # 図の保存
#plt.show() # 図の描画

#TiO2
plot_ti = pd.read_csv(saved_dir+'Y_ALL_PTR_True_TiO2_SSP-M_BO_results.csv', encoding= 'cp932', index_col= 0, header= 0)
plot_ti = plot_ti.sort_values('one_metrics', ascending=False)

plt.rcParams['font.size'] = 22
y_std = plot_ti['選択率NPA_estimated_inverse_y_std']
predicted_y = plot_ti['選択率NPA_estimated_inverse_y']
y_std_top30 = plot_ti.iloc[:30, :]['選択率NPA_estimated_inverse_y_std']
predicted_y_top30 = plot_ti.iloc[:30, :]['選択率NPA_estimated_inverse_y']

plt.rcParams['font.size'] = 22
plt.figure(figsize=figure.figaspect(1)) # 正方形
plt.scatter(y_std, predicted_y, c='blue', alpha=0.7, edgecolors='blue', s= 60) # プロット
plt.scatter(y_std_top30, predicted_y_top30, c='red', alpha=0.7, edgecolors='blue', s= 60) # プロット

plt.title('選択率NPA_TiO2', fontname= 'Meiryo')
#plt.ylim(-0.1, 6.0) # Y のサイズ
#plt.xlim(-0.1, 6.0) # X のサイズ
#plt.xticks([0, 2.0, 4.0, 6.0])

plt.xlabel('y_std') #　縦軸ラベル
plt.ylabel('predicted_y') #　横軸ラベル
plt.savefig(saved_dir+'TiO2_s.png',bbox_inches = 'tight') # 図の保存
#plt.show() # 図の描画

y_std_y = plot_ti['収率NPA_estimated_inverse_y_std']
predicted_y_y = plot_ti['収率NPA_estimated_inverse_y']
y_std_top30_y = plot_ti.iloc[:30, :]['収率NPA_estimated_inverse_y_std']
predicted_y_top30_y = plot_ti.iloc[:30, :]['収率NPA_estimated_inverse_y']

plt.rcParams['font.size'] = 22
plt.figure(figsize=figure.figaspect(1)) # 正方形
plt.scatter(y_std_y, predicted_y_y, c='blue', alpha=0.7, edgecolors='blue', s= 60) # プロット
plt.scatter(y_std_top30_y, predicted_y_top30_y, c='red', alpha=0.7, edgecolors='blue', s= 60) # プロット

plt.title('収率NPA_TiO2', fontname= 'Meiryo')
#plt.ylim(-0.1, 1.0) # Y のサイズ
#plt.xlim(-0.1, 1.0) # X のサイズ
#plt.xticks([0, 2.0, 4.0, 6.0])

plt.xlabel('y_std') #　縦軸ラベル
plt.ylabel('predicted_y') #　横軸ラベル
plt.savefig(saved_dir+'TiO2_y.png',bbox_inches = 'tight') # 図の保存
#plt.show() # 図の描画
"""

print('End')