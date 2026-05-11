import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from libs.boruta import BorutaPy
from collections import Counter
import shap
from sklearn.model_selection import train_test_split
from imodels import RuleFitRegressor
from itertools import combinations as combination
from tabpfn import TabPFNRegressor
from tabpfn_extensions.interpretability.shap import get_shap_values, plot_shap
import torch
import gpytorch
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# TabPFNのSHAP値出力
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
x4_h = pd.read_csv('MyWork/datasets/larger_data/x4_h.csv', encoding='utf-8-sig', index_col= 0)
#x4_h = x4_h.iloc[:10, :]
# Remove invalid characters from column names
x4_h_columns = x4_h.columns.tolist()
x4_h_columns = [col.replace(",", "_") for col in x4_h.columns]
x4_h.columns = x4_h_columns
special_json_chars = ['"', "'", '[', ']', '{', '}', ':', ',', '\\']
col_with_special_chars = [col for col in x4_h.columns if any(char in col for char in special_json_chars)]
if col_with_special_chars:
    print(f"Columns with special characters: {col_with_special_chars}")
else:
    print("No columns with special characters found.")

X = x4_h.drop('Y(C2)_ %', axis=1)
y = x4_h['Y(C2)_ %']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)
#model = TabPFNRegressor(device='cuda')
model = AutoTabPFNRegressor(device='cuda')
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
print('R2:', r2_score(test_y, y_pred))
print('RMSE:', root_mean_squared_error(test_y, y_pred))

shap_values = get_shap_values(model, train_X)
#plot_shap(shap_values)
# 1枚目：dot plot（SHAP値の分布）
plt.figure()
shap.summary_plot(shap_values, train_X, show=False, plot_type="dot")
plt.savefig("MyWork/datasets/larger_data/shap_bar_x4_h.png", dpi=300)
plt.close()

# 2枚目：bar plot（平均絶対値）
plt.figure()
shap.summary_plot(shap_values, train_X, show=False, plot_type="bar")
plt.savefig("MyWork/datasets/larger_data/shap_summary_x4_h.png", dpi=300)
plt.close()

type(shap_values)
#shap_values = np.array(shap_values)
shap_values_df = pd.DataFrame([row.values for row in shap_values], columns=train_X.columns)
shap_values_df.to_csv('MyWork/datasets/larger_data/shap_x4_h.csv', encoding='utf-8-sig')

print('End')
