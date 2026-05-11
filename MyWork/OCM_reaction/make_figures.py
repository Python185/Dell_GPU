import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
import importlib.util  # impの代わりにimportlibを使用
import matplotlib as mpl

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合

# 健康診断データの図作成
data = pd.read_excel('MyWork/MyHealthData/health_data_250613.xlsx', index_col=0, header=0)
data = data.rename(columns={'Unnamed: 1': 'Units', 'Unnamed: 2': 'Normal_range'})
data = data.T
valid_col_list = ['白血球','赤血球','ヘモグロビン','ヘマトクリット','血小板数','総コレステ','中性脂肪','HDL-C','LDL-C','CRP']
data = data[valid_col_list]

# 最初の2行（UnitsとNormal_range）を除外
data = data.iloc[2:]

# インデックスを日時データに変換
def convert_to_datetime(date_str):
    try:
        # 小数点を含む年月形式（例：2015.1）を処理
        if '.' in date_str and len(date_str.split('.')) == 2:
            year, month = date_str.split('.')
            return pd.Timestamp(year=int(year), month=int(float(month)), day=1)
        # 年月日形式（例：2018.1.1）を処理
        elif '.' in date_str and len(date_str.split('.')) == 3:
            year, month, day = date_str.split('.')
            return pd.Timestamp(year=int(year), month=int(month), day=int(day))
        else:
            return pd.NaT
    except:
        return pd.NaT

# インデックスを日時データに変換
data.index = pd.to_datetime([convert_to_datetime(str(idx)) for idx in data.index])

# 日時データでソート
data = data.sort_index()

# 2020年以降のデータのみを選択
data = data[data.index >= '2020-01-01']

# 不要なデータを削除
drop_index_list = ['2018.1.1', '2019.8', '2023.10.1', '2023.10', '2024.10.3']
data = data.drop(drop_index_list, axis=0, errors='ignore')
data.to_csv('MyWork/MyHealthData/health_check_data.csv', encoding='utf-8-sig')

# 数値データに変換（エラーを無視）
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# 不要なデータを削除
data = data.dropna(how='all')

# 欠損値を前の値で補完
data = data.fillna(method='ffill')

# プロットの設定
plt.figure(figsize=(12, 6))

# 各列をプロット
for column in data.columns:
    # 欠損値を含まないデータのみをプロット
    valid_data = data[column].dropna()
    if not valid_data.empty:
        plt.plot(valid_data.index, valid_data.values, label=column, linewidth=2, marker='o', markersize=4, alpha=0.7)

# グラフの設定
plt.title('Health Check Data Over Time', fontsize=14)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Value', fontsize=16)
#plt.yscale('log')  # y軸を対数スケールに設定
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# グラフを保存
plt.savefig('MyWork/MyHealthData/health_check_data.png', dpi=300, bbox_inches='tight')
plt.close()

print('End')

"""
#データの読込み
data = pd.read_csv('MyWork/datasets/data_for_article.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)

# データの再整形（"melt"を使用して長い形式に変換）
# reset_index() で生成されるカラム名を 'index' とする
data_melted = data.reset_index().melt(id_vars='index', var_name='Method', value_name='Value')

# データの再整形（"melt"を使用して長い形式に変換）
data_melted = data.reset_index().melt(id_vars='index', var_name='Method', value_name='Value')

# 特定のメソッド名を2行に分割
data_melted['Method'] = data_melted['Method'].replace({
    'Feature engineering with domain knowledge and libraries': 'Feature engineering with\ndomain knowledge and libraries',
    'Feature engineering based on domain knowledge': 'Feature engineering based on\ndomain knowledge'
})

# バーの順序を逆にするためにカテゴリを指定
data_melted['Method'] = pd.Categorical(data_melted['Method'], 
                                      categories=['Without feature engineering', 
                                                  'Feature engineering based on\ndomain knowledge', 
                                                  'Feature engineering with\ndomain knowledge and libraries'],
                                      ordered=True)

# 棒グラフの描画
plt.figure(figsize=(14, 8))
barplot = sns.barplot(x='Value', y='Method', hue='index', data=data_melted, palette='Set2')

# タイトルとラベルの調整
#plt.title('特徴量エンジニアリング手法別のR²とRMSEの比較', fontsize=20)
plt.xlabel('Values', fontsize=18)
#plt.ylabel('特徴量エンジニアリング手法', fontsize=18)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(0, 1)  # X軸の範囲を0-1に設定
plt.legend(title_fontsize='16', loc='lower right', fontsize='14')

# R2の「2」を上付き文字に設定
for t in barplot.legend_.texts:
    if t.get_text() == 'R2':
        t.set_text(r'$R^2$')

# レイアウトの調整
plt.tight_layout(pad= 5.0)

# グラフの表示
plt.show()
"""

print('hello')