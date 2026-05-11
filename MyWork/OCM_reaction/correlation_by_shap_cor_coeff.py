# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:34:53 2022
@author: maki
"""

#221110説明変数を作成する　方針として、相関係数を参考にしてできるだけ精度を高くする
#対象はx10として、これに元素情報を追加する。
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
number_of_bins = 50  # ヒストグラムのビンの数
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

#rd = 'MyWork/datasets/predict_wo_sp_el/'
rd = 'MyWork/datasets/predict_wo_sp_el/'
target_df = pd.read_csv(rd+'x2_W2_0.csv', index_col=0, header=0, encoding='utf-8-sig')
synthesis_result_x_data_with_dummy_variables = pd.read_excel('results/synthesis_result_x_data_with_dummy_variables.xlsx', index_col=0, header=0, engine='openpyxl')
xenonpy_element_data = pd.read_csv('results/xenonpy_element_data_v2.csv', index_col=0)
synthesis_metal_x_data = pd.read_excel('results/synthesis_metal_x_data.xlsx', index_col=0, header=0, engine='openpyxl')
synthesis_metal_x_data_old = pd.read_excel('results/synthesis_metal_x_data_20221111.xlsx', index_col=0, header=0, engine='openpyxl')
synthesis_condition_x_data_with_dummy_variables = pd.read_excel('results/synthesis_condition_x_data_with_dummy_variables.xlsx', index_col=0, header=0, engine='openpyxl')

#ヒストグラム
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(target_df.iloc[:, 0:1], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel('C2 yield')  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画

# 相関行列
correlation_coefficients = target_df.corr()  # 相関行列の計算
#correlation_coefficients.to_csv('correlation_coefficients.csv')  # 相関行列を csv ファイルとして保存
# 相関行列のヒートマップ (相関係数の値なし) 
plt.rcParams['font.size'] = 12
sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=False)
plt.xlim([0, correlation_coefficients.shape[0]])
plt.ylim([0, correlation_coefficients.shape[0]])
plt.show()

#相関係数のヒストグラム
#選択率と各因子の相関関係の確認
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(correlation_coefficients.iloc[1:, 0], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel('correlation coef. with {0}'.format(target_df.columns[0]))  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画
#factor_PNA = correlation_coefficients.iloc[0]
#factor_EN = correlation_coefficients.iloc[1]
#factor_PNA.sort_values(inplace= True, ascending= False)
#factor_EN.sort_values(inplace= True, ascending= False)
factor = correlation_coefficients.iloc[:, 0:1]
factor['C2 yield'] = factor['C2 yield'].abs()
factor.sort_values(by='C2 yield', inplace= True, ascending= False)

#target_df.drop('触媒ロット', axis= 1, inplace= True)
y1 = target_df.loc[:, 'C2 yield'].copy()
x = target_df.drop('C2 yield', axis= 1).copy()
#y2 = target_df.iloc[:, 1]
model1 = xgb.XGBRegressor().fit(x, y1)
explainer = shap.Explainer(model1)
shap_values = explainer(x)
shap.plots.waterfall(shap_values[0])

model2 = lgb.LGBMRegressor().fit(x, y1)
explainer = shap.Explainer(model2)
shap_values = explainer(x)
shap.plots.waterfall(shap_values[0])

#RandomForestで特徴量の重要度を出力させる
y = target_df.loc[:, 'C2 yield'].copy()
X = target_df.drop('C2 yield', axis= 1).copy()
model3 = RandomForestRegressor(random_state= 42)
model3.fit(X, y)
# 特徴量重要度の取得
feature_importances = model3.feature_importances_
feature_names = target_df.columns.to_list()[1:]
# 特徴量重要度をDataFrameに変換し、重要度でソート
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)
importances_df = importances_df.iloc[:9, :]
# 特徴量重要度のプロット
plt.figure(figsize=(10, 6))
plt.bar(importances_df['Feature'], importances_df['Importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances in Random Forest Model')
plt.xticks(rotation= 90)
plt.show()

print('hello')