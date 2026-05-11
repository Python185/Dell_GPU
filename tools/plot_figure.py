import pandas as pd
import numpy as np
import os
import itertools
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.font_manager as fm
from sklearn.metrics import r2_score, mean_absolute_error


# 直截的逆解析の結果の図(炉内実温データTemperature)を作成
df_a = pd.read_csv('result/autoencoder_data/estimated_x_cat_15746.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
df_b = pd.read_csv('result/autoencoder_data/estimated_x_cat_14822.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
df_c = pd.read_csv('result/autoencoder_data/estimated_x_cat_13180.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
temp_data = pd.read_csv('datasets/v592_Ti/x2_ce_temp.csv', encoding='utf-8-sig', index_col=0, header=0)

temp_cols = [col for col in temp_data.columns if col.isdigit() and 1 <= int(col) <= 30]
temp_a = temp_data.loc[[15746], temp_cols]
temp_b = temp_data.loc[[14822], temp_cols]
temp_c = temp_data.loc[[13180], temp_cols]

df_a.loc[:, temp_cols] = temp_a.values[0]
df_b.loc[:, temp_cols] = temp_b.values[0]
df_c.loc[:, temp_cols] = temp_c.values[0]

all_temp_cols = [str(i) for i in range(1, 61)]
df_a = df_a[all_temp_cols]
df_b = df_b[all_temp_cols]
df_c = df_c[all_temp_cols]

plot_x = pd.concat([df_a, df_b, df_c], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# df_a のプロット
for i in df_a.index:
    axes[0, 0].plot(df_a.columns.astype(int), df_a.loc[i], alpha=0.7)
axes[0, 0].set_title('df_a (cat_15746)', fontsize=14)
axes[0, 0].set_xlabel('Time, minutes', fontsize=12)
axes[0, 0].set_ylabel('Temperature, ℃', fontsize=12)

# df_b のプロット
for i in df_b.index:
    axes[0, 1].plot(df_b.columns.astype(int), df_b.loc[i], alpha=0.7)
axes[0, 1].set_title('df_b (cat_14822)', fontsize=14)
axes[0, 1].set_xlabel('Time, minutes', fontsize=12)
axes[0, 1].set_ylabel('Temperature, ℃', fontsize=12)

# df_c のプロット
for i in df_c.index:
    axes[1, 0].plot(df_c.columns.astype(int), df_c.loc[i], alpha=0.7)
axes[1, 0].set_title('df_c (cat_13180)', fontsize=14)
axes[1, 0].set_xlabel('Time, minutes', fontsize=12)
axes[1, 0].set_ylabel('Temperature, ℃', fontsize=12)

# 右下は空欄
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 3, figsize=(20, 15))  # 3 rows, 4 columns
#fig, axes = plt.subplots(1, 3, figsize=(20, 15))  # 1 rows, 3 columns
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

for i in range(9):
    ax = axes[i]
    start_idx = i * 10
    end_idx = start_idx + 10
    for sample_name in plot_x.index[start_idx:end_idx]:
        ax.plot(plot_x.columns.astype(int), plot_x.loc[sample_name].iloc[0], label=sample_name)
    ax.set_xlabel('Time, minutes', fontsize=10)
    ax.set_ylabel('Temperature, ℃', fontsize=10)
    #ax_set_ylim = ax.set_ylim(-0.005, 0.08)
    #ax.invert_xaxis()
    ax.set_title(f'Index {start_idx} to {end_idx-1}', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)  # Set font size for x and y axis numbers
    ax.legend(fontsize=10, loc='upper right')  # Add legend with font size 10 and set position to upper right

plt.tight_layout()
plt.show()

print('End')