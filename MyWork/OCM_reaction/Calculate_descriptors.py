# -*- coding: utf-8 -*-
"""
Created on Mon Feb 6 15:27:31 2023
@author: dcelab
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import itertools
from feature_engine.selection import SmartCorrelatedSelection
from feature_engine.selection import DropHighPSIFeatures
from feature_engine.selection import SelectByInformationValue
from feature_engine.selection import SelectByShuffling
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.selection import SelectByTargetMeanPerformance
from feature_engine.selection import RecursiveFeatureElimination
from feature_engine.selection import RecursiveFeatureAddition
from feature_engine.selection import ProbeFeatureSelection
import featuretools as ft
from tpot.export_utils import set_param_recursive
from tpot import TPOTRegressor
from autofeat import AutoFeatRegressor
from xfeat import GBDTFeatureSelector, GBDTFeatureExplorer
import optuna
import lightgbm as lgb
from optuna import create_study
from functools import partial
from openfe import OpenFE, tree_to_formula, transform, TwoStageFeatureSelector
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
matplotlib.rc('font', family='Meiryo')

#Boruta
def Boruta_Apply(df, perc):
    y= df['C2 yield'].copy()
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
    
    return df_b

def CalcX1(metal_x):
    #x1の作成
    metal_x = pd.read_csv('MyWork/datasets/metal_x.csv', index_col= 0, header= 0)
    xenonpy_original = pd.read_csv('results/xenonpy_element_data.csv', index_col= 0)
    elements = metal_x[['Cation1','Cation2','Cation3']].values.flatten().tolist()
    xenonpy_original = xenonpy_original.query('index == @elements')
    xenonpy_original = xenonpy_original.dropna(how= 'any', axis= 1)
    xenonpy_original = xenonpy_original.drop('period', axis= 1)
    xenonpy_merge = xenonpy_original
    #xenonpy_original = xenonpy_original.loc[:, xenonpy_original.isnull().mean() < 0.5]
    # IterativeImputerの初期化
    #iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    # IterativeImputerで補完
    #df_iterative_imputed = pd.DataFrame(iterative_imputer.fit_transform(xenonpy_original), columns=xenonpy_original.columns)
    #xenonpy_merge = df_iterative_imputed
    #xenonpy_merge.index = xenonpy_original.index
    #x1の作成本番
    weighted_average_name = list() # 加重平均の index 名
    weighted_variance_name = list() # 加重分散の index 名
    geometric_mean_name = list() # 幾何平均の index 名
    harmonic_mean_name = list() # 調和平均の index 名
    max_pooling_name = list() # 最大値の index 名
    min_pooling_name = list() # 最小値の index 名
    for j in xenonpy_merge.columns:
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
        metal1 = metal_x.loc[metal_x.index[i], 'Cation1']
        metal2 = metal_x.loc[metal_x.index[i], 'Cation2']
        metal3 = metal_x.loc[metal_x.index[i], 'Cation3']
        metal_rate1 = metal_x.loc[metal_x.index[i], 'Cation1Amount']
        metal_rate2 = metal_x.loc[metal_x.index[i], 'Cation2Amount']
        metal_rate3 = metal_x.loc[metal_x.index[i], 'Cation3Amount']
            
        if metal3 != 'na':
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            metal_desc3 = xenonpy_merge.loc[metal3, :].values
            mt = np.array([metal_desc1, metal_desc2, metal_desc3])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3])
        elif metal2 != 'na':
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            mt = np.array([metal_desc1, metal_desc2])
            mr = np.array([metal_rate1, metal_rate2])                
        else:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            mt = np.array([metal_desc1])
            mr = np.array([metal_rate1])
        for desc in range(xenonpy_merge.shape[1]):
            d_name = xenonpy_merge.columns[desc]
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

#特定の日付に保存されたファイルを読込む関数
def load_files_as_dict_on_saved_date(directory, date):
    all_files = os.listdir(directory)
    df_dict = {}

    for file in all_files:
        file_path = os.path.join(directory, file)
        # ファイルの最終変更日時を取得
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).date()

        # 指定日付と一致する場合に読み込み
        if str(file_mtime) == date:
            df_dict[file] = pd.read_csv(file_path, index_col= 0, header= 0)

    return df_dict

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
xx2 = df
#df.to_csv('MyWork/datasets/x2.csv', encoding= 'cp932')

#x_baseの作成
metal_x = x26[drop_list]
x_base = df.iloc[:, :10]

#一部に元素被りがあるため、そこを修正する
for index, row in metal_x.iterrows():
    if row[0] == row[1]:
        metal_x.at[index, 'Cation2'] = 'na'
        metal_x.at[index, 'Cation1Amount'] = row[3] + row[4]
        metal_x.at[index, 'Cation2Amount'] = 0
        
#metal_x.to_csv('MyWork/datasets/metal_x.csv', encoding= 'cp932')
"""
#以下Boruta絞込み
#sklearnによる交差項付与
xx2 = pd.read_csv('MyWork/datasets/x2_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
element_term = xx2.loc[:, 'Ba':]
skl_poly = PolynomialFeatures(degree= 2, interaction_only= True, include_bias= False)
cross_array = skl_poly.fit_transform(element_term)
feature_names = skl_poly.get_feature_names_out(input_features=element_term.columns)

cross_term = pd.DataFrame(cross_array, columns=feature_names, index= xx2.index)
xx2_add_cross = pd.concat([xx2.loc[:, :'Support_SiO5'], cross_term], axis= 1)
xx2_add_cross = xx2_add_cross.sort_index()

#Boruta絞込み、保存
Boruta_selected = Boruta_Apply(xx2_add_cross, perc= 90)
xx2 = pd.concat([xx2['C2 yield'], Boruta_selected], axis= 1)
xx2.to_csv('MyWork/datasets/x2_b90.csv', encoding= 'cp932')
#xx2_add_cross.to_csv('MyWork/datasets/X_crs.csv', encoding= 'cp932')

#以下Boruta特徴量の再現
train_df = pd.read_csv('MyWork/datasets/x2_b80.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
columns_list = train_df.columns.to_list()
crs_list = [s for s in columns_list if ' ' in s]
del crs_list[0] #C2 yieldを削除
for item in crs_list:
    metals = item.split(' ')
    test_df[item] = test_df[metals[0]] * test_df[metals[1]]
test_df = test_df.loc[:, train_df.columns.to_list()]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df.to_csv('MyWork/datasets/x2_b80_all.csv', encoding= 'cp932')

#以下は、Feature-engineに関するもの
#Borutaの替わりにFeature-engineで説明変数を絞ってみる
#(1)scs
scs = SmartCorrelatedSelection(threshold=0.9, selection_method="variance")
#Varianceの高い変数を保持して、相関係数0.7を超える変数を削除する
x_scs = scs.fit_transform(xx2_add_cross)
features_scs = x_scs.columns.to_list()
x_scs.to_csv('Mywork/datasets/x2_11.csv', encoding= 'cp932')
#(2)sbs
sbs = SelectByShuffling(threshold= 0.01, estimator= RandomForestRegressor(), scoring= 'r2')
#threshold=0.001で説明変数11, 0.0001で17個
y = xx2['C2 yield'].sort_index()
xx2_add_cross_wo_y = xx2_add_cross.drop('C2 yield', axis= 1)
x_sbs = sbs.fit_transform(xx2_add_cross_wo_y, y)
features_sbs = sbs.get_feature_names_out
x_sbs = pd.concat([y, x_sbs], axis= 1)
x_sbs.to_csv('Mywork/datasets/x2_8.csv', encoding= 'cp932')
#(3)tmp
tmp = SelectByTargetMeanPerformance(scoring="r2", threshold= 0.7, bins=3, strategy="equal_frequency", cv=3, regression= True)
x_tmp = tmp.fit_transform(xx2_add_cross_wo_y, y)
tmp_features = tmp.variables_
x_tmp = xx2_add_cross.loc[:, tmp_features]
x_tmp = pd.concat([y, x_tmp], axis= 1)
x_tmp.to_csv('Mywork/datasets/x2_14.csv', encoding= 'cp932')

#(4)psi
psi = DropHighPSIFeatures(split_col= 'C2 yield', threshold= 0.1)
#thresholdを0.1-0.4まで変化させても殆ど選択数に変化なし
x_psi = psi.fit_transform(xx2_add_cross)
features_psi = psi.get_feature_names_out
#(5)iv: Category変数用のためOmit
#(6)sfp:謎エラーが出て無限ループになってしまうためOmit
#sfp = SelectBySingleFeaturePerformance(RandomForestRegressor(), scoring= 'r2')
#X_sfp = sfp.fit_transform(xx2_add_cross_wo_y, y)
#(7)pfs
pfs = ProbeFeatureSelection(estimator= LinearRegression(), scoring= 'r2', n_probes= 3)
x_pfs =pfs.fit_transform(xx2_add_cross_wo_y, y) #RandomForestではempty dfとなるためLRとした
x_pfs = pd.concat([y, x_pfs], axis= 1)
#(8)rfe
rfe = RecursiveFeatureElimination(estimator= RandomForestRegressor(), scoring= 'r2')
x_rfe = rfe.fit_transform(xx2_add_cross_wo_y, y)
x_rfe = pd.concat([y, x_rfe], axis= 1)
#(9)rfa
rfa = RecursiveFeatureAddition(estimator= RandomForestRegressor(), scoring= 'r2')
x_rfa = rfa.fit_transform(xx2_add_cross_wo_y, y)
x_rfa = pd.concat([y, x_rfa], axis= 1)

#xx2.to_csv('MyWork/datasets/X_Brt.csv', encoding= 'cp932')
x_pfs.to_csv('Mywork/datasets/x_pfs.csv', encoding= 'cp932')
x_psi.to_csv('Mywork/datasets/x_psi.csv', encoding= 'cp932')
x_rfa.to_csv('Mywork/datasets/x_rfa.csv', encoding= 'cp932')
x_rfe.to_csv('Mywork/datasets/x_rfe.csv', encoding= 'cp932')

#以下scsの再現
train_df = pd.read_csv('MyWork/datasets/x2_scs.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
columns_list = train_df.columns.to_list()
crs_list = [s for s in columns_list if ' ' in s]
del crs_list[0] #C2 yieldを削除
for item in crs_list:
    metals = item.split(' ')
    test_df[item] = test_df[metals[0]] * test_df[metals[1]]
test_df = test_df.loc[:, train_df.columns.to_list()]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df.to_csv('MyWork/datasets/x2_scs_all.csv', encoding= 'cp932')

#以下はTPOTに関するもの
#df.to_csv('MyWork/datasets/df_tpot.csv', encoding= 'cp932')
y = df['C2 yield'].copy()
x = df.drop('C2 yield', axis= 1).copy()
features = x
x_train, x_test, y_train, y_test = \
            train_test_split(features, y, random_state=42, train_size= 0.7)
tpot = TPOTRegressor(generations=100, population_size=100, verbosity=2, random_state=42)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))
tpot.export('MyWork/tpot_pipeline.py')
test_df = pd.concat([y_test, y_train], axis= 1)
test_df.to_csv('MyWork/datasets/test_df_tpot.csv', encoding='cp932')
#print(tpot.evaluated_individuals_)
y_predicted = tpot.predict(x_test)
print(r2_score(y_predicted, y_test))
y_predicted = pd.Series(data= y_predicted, name='y_predicted')
test_y = pd.concat([y_test, x_test], axis= 1)
test_y = test_y.reset_index()
test_y = pd.concat([y_predicted, test_y], axis= 1)
test_y.to_csv('MyWork/datasets/predicted_y.csv', encoding= 'cp932')

results = tpot.fitted_pipeline_
print(results)

#以下AutoFeatに関するもの
y = df['C2 yield'].copy()
x = df.drop('C2 yield', axis= 1).copy()
features = x
#x_train, x_test, y_train, y_test = \
#            train_test_split(features, y, random_state=42, train_size= 0.7)
model1= AutoFeatRegressor(verbose=0) #デフォルトのf_steps=2, feasel_runs=5
model2= AutoFeatRegressor(verbose=1)
model3= AutoFeatRegressor(verbose=0, feateng_steps= 2, featsel_runs= 3)
model4= AutoFeatRegressor(verbose=0, feateng_steps= 3, featsel_runs= 5)
x_feature_creation1 = model1.fit_transform(x, y)
x_feature_creation2 = model2.fit_transform(x, y)
x_feature_creation3 = model3.fit_transform(x, y)
x_feature_creation4 = model4.fit_transform(x, y)
x_af1 = pd.concat([y, x_feature_creation1], axis= 1)
x_af2 = pd.concat([y, x_feature_creation2], axis= 1)
x_af3 = pd.concat([y, x_feature_creation3], axis= 1)
x_af4 = pd.concat([y, x_feature_creation4], axis= 1)
x_af1.to_csv('MyWork/datasets/x_af1.csv', encoding= 'cp932')
x_af2.to_csv('MyWork/datasets/x_af2.csv', encoding= 'cp932')
x_af3.to_csv('MyWork/datasets/x_af3.csv', encoding= 'cp932')
x_af4.to_csv('MyWork/datasets/x_af4.csv', encoding= 'cp932')

#以下はXfeatに関するもの
#xfeatに関するコメント:エンコーディングは主としてカテゴリカルデータのためのもの
#数値データに対する処理もあるが、四則演算を自動で行うようなものであったため、特にトライせず。
#特徴量選択方法としてGBDT、Optunaがあったため、これのみ試してみる。
#imput_colは目的変数と説明変数の両方
X_crs = pd.read_csv('MyWork/datasets/X_crs.csv', index_col= 0, header= 0)
feature_col = X_crs.columns.to_list()

params = {
    "objective": "regression",
    "seed": 111,
}
fit_kwargs = {
    "num_boost_round": 10,
}
selector = GBDTFeatureSelector(
    input_cols=feature_col,
    target_col="C2 yield",
    threshold=0.3,
    lgbm_params=params,
    lgbm_fit_kwargs=fit_kwargs,
)
df_selected = selector.fit_transform(X_crs)
#print("Selected columns:", selector._selected_cols)
df_selected.to_csv('MyWork/datasets/Xf_3.csv', encoding= 'cp932')

def main():
    df = X_crs
    df = df.rename(columns={'C2 yield': 'target'})
    print("After applying GBDTFeatureSelector:")
    df = feature_selection(df, df.target)
    evaluate_dataframe(df, df.target)
    df = df.rename(columns={'target': 'C2 yield'})
    df.to_csv('MyWork/datasets/Xf_opt.csv', encoding= 'cp932')

def objective(df, selector, trial):
    selector.set_trial(trial)
    selector.fit(df)
    input_cols = selector.get_selected_cols()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }

    # Evaluate with selected columns
    train_set = lgb.Dataset(df[input_cols], label=df["target"])
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    return rmsle_score

def feature_selection(df, y):
    input_cols = df.columns.tolist()
    n_before_selection = len(input_cols)

    df["target"] = np.log1p(y)
    df_train, _ = train_test_split(df, test_size=0.5, random_state=1)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    fit_params = {
        "num_boost_round": 100,
    }
    selector = GBDTFeatureExplorer(input_cols=input_cols,
                                   target_col="target",
                                   fit_once=True,
                                   threshold_range=(0.6, 1.0),
                                   lgbm_params=params,
                                   lgbm_fit_kwargs=fit_params)

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, df_train, selector), n_trials=20)

    selector.from_trial(study.best_trial)
    selected_cols = selector.get_selected_cols()
    print(f" - {n_before_selection - len(selected_cols)} features are removed.")

    return df[selected_cols]

def evaluate_dataframe(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df.values, y,
                                                        test_size=0.5,
                                                        random_state=1)
    y_train = np.log1p(y_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    print(f" - CV RMSEL: {rmsle_score:.6f}")

    booster = lgb.train(params, train_set, num_boost_round=100)
    y_pred = booster.predict(X_test)
    test_rmsle_score = rmse(np.log1p(y_test), y_pred)
    test_r2_score = r2_score(y_test, y_pred)
    test_mae_score = mean_absolute_error(y_test, y_pred)
    print(f" - test RMSEL: {test_rmsle_score:.6f}")
    print(f" - test R2: {test_r2_score:.6f}")    
    print(f" - test MAE: {test_mae_score:.6f}") 

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    main()

#以下はOpenFEに関するもの
def get_score(train_x, test_x, train_y, test_y):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_absolute_error(test_y, pred)
    score_a = r2_score(test_y, pred)
    return score, score_a

if __name__ == '__main__':
    n_jobs = 5
    data = pd.read_csv('MyWork/datasets/x2.csv', index_col= 0, header= 0)
    label = data[['C2 yield']]
    del data['C2 yield']

    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
    # get baseline score
    score, score_a = get_score(train_x, test_x, train_y, test_y)
    print("The R2 before feature generation is", score_a)
    print("The MAE before feature generation is", score)
    
    # We use the two-stage pruning algorithm of OpenFE to perform Feature Selection
    fs = TwoStageFeatureSelector(n_jobs=n_jobs)
    features = fs.fit(data=train_x, label=train_y)
    # OpenFE gives the ranking of the base features:
    print(features)
    # Select the top 6 features
    new_features = features[:10]
    score, score_a = get_score(train_x[new_features], test_x[new_features], train_y, test_y)
    print("The R2 after feature generation is", score_a)
    print("The MAE after feature generation is", score)
    
    ofe = OpenFE()
    ofe.fit(data= train_x, label= train_y, n_jobs= n_jobs)    
    # OpenFE recommends a list of new features. We include the top 10
    # generated features to see how they influence the model performance
    train_x, test_x = transform(train_x, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
    score, score_a = get_score(train_x, test_x, train_y, test_y)
    print("The R2 after feature generation is", score_a)
    print("The top 10 generated features are")
    for feature in ofe.new_features_list[:10]:
        print(tree_to_formula(feature))    

#x2で計算したTPOTのデータを図にしてMAEを計算する
test_df = pd.read_csv('MyWork/datasets/df_tpot.csv', encoding= 'cp932', index_col= 0, header= 0)
predicted_df = pd.read_csv('MyWork/datasets/predicted_y.csv', encoding= 'cp932', index_col= 0, header= 0)
test_index = predicted_df['index'].to_list()
test_df_a = test_df.loc[test_index, :]
plot_df = pd.merge(test_df_a, predicted_df, left_index= True, right_on = 'index')
plot_df = plot_df.iloc[:96, :]
plot_df = plot_df.loc[:, ['y_predicted', 'C2 yield_y']]
mae = mean_absolute_error(plot_df['C2 yield_y'], plot_df['y_predicted'])
rmse = mean_squared_error(plot_df['C2 yield_y'], plot_df['y_predicted'])
print('MAE=', mae)
print('RMSE=', rmse)

plt.rcParams['font.size'] = 22
raw_y = plot_df['C2 yield_y']
predicted_y = plot_df['y_predicted']
plt.figure(figsize=figure.figaspect(1)) # 正方形
plt.scatter(raw_y, predicted_y, c='blue', alpha=0.7, edgecolors='black', s= 60) # プロット
y_max = np.max(np.array([raw_y,predicted_y.values.flatten()])) # y 値の最大を取得
y_min = np.min(np.array([raw_y, predicted_y.values.flatten()])) # y 値の最小を取得
if y_min <= 0:
    y_min = 0

plt.plot([-0.2, 15], [-0.2, 15], 'k-') # 対角線の描画

plt.ylim(-0.2, 15) # Y のサイズ
plt.xlim(-0.2, 15) # X のサイズ
plt.xticks([0, 5, 10, 15])

plt.xlabel('Actual Y') #　縦軸ラベル
plt.ylabel('Predicted Y') #　横軸ラベル
plt.savefig('MyWork/datasets/TPOT.png',bbox_inches = 'tight') # 図の保存
plt.show() # 図の描画

#x1に相当するものを作成する
metal_x = pd.read_csv('MyWork/datasets/metal_x.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
x1 = CalcX1(metal_x)
x1 = pd.concat([x_base, x1], axis= 1)
x1_tr, x1_te = train_test_split(x1, test_size= 0.25, shuffle= True, random_state= 42)
x1_tr.sort_index().to_csv('MyWork/datasets/x1_tr.csv', encoding= 'cp932')
x1_te.sort_index().to_csv('MyWork/datasets/x1_te.csv', encoding= 'cp932')
x1.to_csv('MyWork/datasets/xo1.csv', encoding= 'cp932')

#x1の読込み
x1 = pd.read_csv('MyWork/datasets/x1_tr.csv', encoding= 'cp932', index_col= 0, header= 0)

#x1に対してBorutaを適用
x1_target = x1.loc[:, 'C2 yield']
Boruta_selected_x1 = Boruta_Apply(x1, perc= 90)
x1_b90 = pd.concat([x1_target, Boruta_selected_x1], axis= 1)
x1_b90.to_csv('MyWork/datasets/x1_b90.csv', encoding= 'cp932')
Boruta_selected_x1 = Boruta_Apply(x1, perc= 80)
x1_b80 = pd.concat([x1_target, Boruta_selected_x1], axis= 1)
x1_b80.to_csv('MyWork/datasets/x1_b80.csv', encoding= 'cp932')
Boruta_selected_x1 = Boruta_Apply(x1, perc= 60)
x1_b60 = pd.concat([x1_target, Boruta_selected_x1], axis= 1)
x1_b60.to_csv('MyWork/datasets/x1_b60.csv', encoding= 'cp932')

#x1_b90-60を再現
x1_b90 = pd.read_csv('MyWork/datasets/x1_b90.csv', encoding= 'cp932', index_col= 0, header= 0)
x1_b80 = pd.read_csv('MyWork/datasets/x1_b80.csv', encoding= 'cp932', index_col= 0, header= 0)
x1_b60 = pd.read_csv('MyWork/datasets/x1_b60.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x1_te.csv', encoding= 'cp932', index_col= 0, header= 0)
x1_b90_test_df = test_df.copy()
x1_b90_test_df = x1_b90_test_df.loc[:, x1_b90.columns.to_list()]
x1_b90_all = pd.concat([x1_b90, x1_b90_test_df], axis= 0)
x1_b90_all = x1_b90_all.sort_index()
x1_b90_all.to_csv('MyWork/datasets/x1_b90_all.csv', encoding= 'cp932')
x1_b80_test_df = test_df.copy()
x1_b80_test_df = x1_b80_test_df.loc[:, x1_b80.columns.to_list()]
x1_b80_all = pd.concat([x1_b80, x1_b80_test_df], axis= 0)
x1_b80_all = x1_b80_all.sort_index()
x1_b80_all.to_csv('MyWork/datasets/x1_b80_all.csv', encoding= 'cp932')
x1_b60_test_df = test_df.copy()
x1_b60_test_df = x1_b60_test_df.loc[:, x1_b60.columns.to_list()]
x1_b60_all = pd.concat([x1_b60, x1_b60_test_df], axis= 0)
x1_b60_all = x1_b60_all.sort_index()
x1_b60_all.to_csv('MyWork/datasets/x1_b60_all.csv', encoding= 'cp932')

#x1に対してfeature-engineを適用
x1_wo_y = x1.drop('C2 yield', axis= 1)
y = x1['C2 yield'].sort_index()
#(1)scs
scs = SmartCorrelatedSelection(threshold=0.9, selection_method="variance")
#Varianceの高い変数を保持して、相関係数0.7を超える変数を削除する
x_scs = scs.fit_transform(x1)
features_scs = x_scs.columns.to_list()
x_scs.to_csv('Mywork/datasets/x1_29.csv', encoding= 'cp932')
#(2)sbs
sbs = SelectByShuffling(threshold= 0.01, estimator= RandomForestRegressor(), scoring= 'r2')
#threshold=0.001で説明変数11, 0.0001で17個
x_sbs = sbs.fit_transform(x1_wo_y, y)
features_sbs = sbs.get_feature_names_out
x_sbs = pd.concat([y, x_sbs], axis= 1)
x_sbs.to_csv('Mywork/datasets/x1_26.csv', encoding= 'cp932')
#(3)tmp
tmp = SelectByTargetMeanPerformance(scoring="r2", threshold= 0.7, bins=3, strategy="equal_frequency", cv=3, regression= True)
x_tmp = tmp.fit_transform(x1_wo_y, y)
tmp_features = tmp.variables_
x_tmp = x1.loc[:, tmp_features]
x_tmp = pd.concat([y, x_tmp], axis= 1)
x_tmp.to_csv('Mywork/datasets/x1_32.csv', encoding= 'cp932')

#(4)psi
#psi = DropHighPSIFeatures(split_col= 'C2 yield', threshold= 0.1)
#thresholdを0.1-0.4まで変化させても殆ど選択数に変化なし
#x_psi = psi.fit_transform(x1)
#features_psi = psi.get_feature_names_out
#x_psi.to_csv('Mywork/datasets/xo1_psi.csv', encoding= 'cp932')
#(5)iv: Category変数用のためOmit
#(6)sfp:謎エラーが出て無限ループになってしまうためOmit
#sfp = SelectBySingleFeaturePerformance(RandomForestRegressor(), scoring= 'r2')
#X_sfp = sfp.fit_transform(xx2_add_cross_wo_y, y)
#(7)pfs
#pfs = ProbeFeatureSelection(estimator= LinearRegression(), scoring= 'r2', n_probes= 3)
#x_pfs =pfs.fit_transform(x1_wo_y, y) #RandomForestではempty dfとなるためLRとした
#x_pfs = pd.concat([y, x_pfs], axis= 1)
#x_pfs.to_csv('Mywork/datasets/xo1_pfs.csv', encoding= 'cp932')
#(8)rfa
#rfa = RecursiveFeatureAddition(estimator= RandomForestRegressor(), scoring= 'r2')
#x_rfa = rfa.fit_transform(x1_wo_y, y)
#x_rfa = pd.concat([y, x_rfa], axis= 1)
#x_rfa.to_csv('Mywork/datasets/xo1_rfa.csv', encoding= 'cp932')
#(9)rfe
#rfe = RecursiveFeatureElimination(estimator= RandomForestRegressor(), scoring= 'r2')
#x_rfe = rfe.fit_transform(x1_wo_y, y)
#x_rfe = pd.concat([y, x_rfe], axis= 1)
#x_rfe.to_csv('Mywork/datasets/xo1_rfe.csv', encoding= 'cp932')

#x1に対してxfeatを適用
#imput_colは目的変数と説明変数の両方
x1 = pd.read_csv('MyWork/datasets/x1_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
feature_col = x1.columns.to_list()

params = {
    "objective": "regression",
    "seed": 111,
}
fit_kwargs = {
    "num_boost_round": 10,
}
selector = GBDTFeatureSelector(
    input_cols=feature_col,
    target_col="C2 yield",
    threshold=0.7,
    lgbm_params=params,
    lgbm_fit_kwargs=fit_kwargs,
)
df_selected = selector.fit_transform(x1)
#print("Selected columns:", selector._selected_cols)
df_selected.to_csv('MyWork/datasets/x1_xf7.csv', encoding= 'cp932')

def main():
    df = x1
    df = df.rename(columns={'C2 yield': 'target'})
    print("After applying GBDTFeatureSelector:")
    df = feature_selection(df, df.target)
    evaluate_dataframe(df, df.target)
    df = df.rename(columns={'target': 'C2 yield'})
    #df.to_csv('MyWork/datasets/x1_6.csv', encoding= 'cp932')

def objective(df, selector, trial):
    selector.set_trial(trial)
    selector.fit(df)
    input_cols = selector.get_selected_cols()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }

    # Evaluate with selected columns
    train_set = lgb.Dataset(df[input_cols], label=df["target"])
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    return rmsle_score

def feature_selection(df, y):
    input_cols = df.columns.tolist()
    n_before_selection = len(input_cols)

    df["target"] = np.log1p(y)
    df_train, _ = train_test_split(df, test_size=0.5, random_state=1)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    fit_params = {
        "num_boost_round": 100,
    }
    selector = GBDTFeatureExplorer(input_cols=input_cols,
                                   target_col="target",
                                   fit_once=True,
                                   threshold_range=(0.6, 1.0),
                                   lgbm_params=params,
                                   lgbm_fit_kwargs=fit_params)

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, df_train, selector), n_trials=20)

    selector.from_trial(study.best_trial)
    selected_cols = selector.get_selected_cols()
    print(f" - {n_before_selection - len(selected_cols)} features are removed.")

    return df[selected_cols]

def evaluate_dataframe(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df.values, y,
                                                        test_size=0.5,
                                                        random_state=1)
    y_train = np.log1p(y_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    print(f" - CV RMSEL: {rmsle_score:.6f}")

    booster = lgb.train(params, train_set, num_boost_round=100)
    y_pred = booster.predict(X_test)
    test_rmsle_score = rmse(np.log1p(y_test), y_pred)
    test_r2_score = r2_score(y_test, y_pred)
    test_mae_score = mean_absolute_error(y_test, y_pred)
    print(f" - test RMSEL: {test_rmsle_score:.6f}")
    print(f" - test R2: {test_r2_score:.6f}")    
    print(f" - test MAE: {test_mae_score:.6f}") 

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    main()

#x1_xf9-xf7を再現
x1_xf9 = pd.read_csv('MyWork/datasets/x1_xf9.csv', encoding= 'cp932', index_col= 0, header= 0)
x1_xf8 = pd.read_csv('MyWork/datasets/x1_xf8.csv', encoding= 'cp932', index_col= 0, header= 0)
x1_xf7 = pd.read_csv('MyWork/datasets/x1_xf7.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x1_te.csv', encoding= 'cp932', index_col= 0, header= 0)
x1_xf9_test_df = test_df.copy()
x1_xf9_test_df = x1_xf9_test_df.loc[:, x1_xf9.columns.to_list()]
x1_xf9_all = pd.concat([x1_xf9, x1_xf9_test_df], axis= 0)
x1_xf9_all = x1_xf9_all.sort_index()
x1_xf9_all.to_csv('MyWork/datasets/x1_xf9_all.csv', encoding= 'cp932')
x1_xf8_test_df = test_df.copy()
x1_xf8_test_df = x1_xf8_test_df.loc[:, x1_xf8.columns.to_list()]
x1_xf8_all = pd.concat([x1_xf8, x1_xf8_test_df], axis= 0)
x1_xf8_all = x1_xf8_all.sort_index()
x1_xf8_all.to_csv('MyWork/datasets/x1_xf8_all.csv', encoding= 'cp932')
x1_xf7_test_df = test_df.copy()
x1_xf7_test_df = x1_xf7_test_df.loc[:, x1_xf7.columns.to_list()]
x1_xf7_all = pd.concat([x1_xf7, x1_xf7_test_df], axis= 0)
x1_xf7_all = x1_xf7_all.sort_index()
x1_xf7_all.to_csv('MyWork/datasets/x1_xf7_all.csv', encoding= 'cp932')

#以下、DCVではなく、train_test_splitを前提として種々のライブラリで特徴量を作成、選択するコードを作成する
#以下AutoFeatに関するもの
x2_tr = pd.read_csv('MyWork/datasets/x2_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
df = x2_tr
y = df['C2 yield'].copy()
x = df.drop('C2 yield', axis= 1).copy()
features = x
#x_train, x_test, y_train, y_test = \
#            train_test_split(features, y, random_state=42, train_size= 0.7)
model1= AutoFeatRegressor(verbose=0) #デフォルトのf_steps=2, feasel_runs=5
x_feature_creation1 = model1.fit_transform(x, y)
x_feature_creation1.index = x.index
x_af1 = pd.concat([y, x_feature_creation1], axis= 1)
x_af1.to_csv('MyWork/datasets/x2_af1.csv', encoding= 'cp932')
model2= AutoFeatRegressor(verbose=1)
x_feature_creation2 = model2.fit_transform(x, y)
x_feature_creation2.index = x.index
x_af2 = pd.concat([y, x_feature_creation2], axis= 1)
x_af2.to_csv('MyWork/datasets/x2_af2.csv', encoding= 'cp932')
model3= AutoFeatRegressor(verbose=0, feateng_steps= 2, featsel_runs= 3)
x_feature_creation3 = model3.fit_transform(x, y)
x_feature_creation3.index = x.index
x_af3 = pd.concat([y, x_feature_creation3], axis= 1)
x_af3.to_csv('MyWork/datasets/x2_af3.csv', encoding= 'cp932')
model4= AutoFeatRegressor(verbose=0, feateng_steps= 3, featsel_runs= 5)
x_feature_creation4 = model4.fit_transform(x, y)
x_feature_creation4.index = x.index
x_af4 = pd.concat([y, x_feature_creation4], axis= 1)
x_af4.to_csv('MyWork/datasets/x2_af4.csv', encoding= 'cp932')

#以下x2_af4_allの作成
x2_tr = pd.read_csv('MyWork/datasets/x2_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_te = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_af4 = pd.read_csv('MyWork/datasets/x2_af4.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_af4_add = x2_te.copy()
x2_af4_add['Na**2*Temp**4'] = x2_af4_add['Na']**2 * x2_af4_add['Temp']**4
x2_af4_add['Temp**3*exp(Mn)'] = x2_af4_add['Temp']**3 * np.exp(x2_af4_add['Mn'])
x2_af4_add['Temp**3*exp(Fe)'] = x2_af4_add['Temp']**3 * np.exp(x2_af4_add['Fe'])
x2_af4_add['Cu*sqrt(Mn)'] = x2_af4_add['Cu'] * np.sqrt(x2_af4_add['Mn'])
x2_af4_add['Na**6*Temp**4'] = x2_af4_add['Na']**6 * x2_af4_add['Temp']**4
x2_af4_add['Support_SiO2*Temp**3'] = x2_af4_add['Support_SiO2'] * x2_af4_add['Temp']**3
x2_af4_add['(-Support_SiO5 + V**3)**3'] = (-x2_af4_add['Support_SiO2'] + x2_af4_add['V']**3)**3
x2_af4_add['Sn*Temp**3'] = x2_af4_add['Sn'] * x2_af4_add['Temp']**3
x2_af4_add['Ba**2*Temp**4'] = x2_af4_add['Ba']**2 * x2_af4_add['Temp']**4
x2_af4_add['(-Na**2 + W)**3'] = (-(x2_af4_add['Na']**2) + x2_af4_add['W'])**3
x2_af4_add['Temp*exp(Ba)'] = x2_af4_add['Temp'] * np.exp(x2_af4_add['Ba'])
x2_af4_add['1/(Ni + 1/Temp)'] = 1/(x2_af4_add['Ni'] + 1/(x2_af4_add['Temp']))
x2_af4_add['sqrt(Na)*Support_SiO2'] = np.sqrt(x2_af4_add['Na']) * x2_af4_add['Support_SiO2']
x2_af4_add['Abs(Cs - Support_SiO5)'] = np.abs(x2_af4_add['Cs'] - x2_af4_add['Support_SiO2'])
x2_af4_add['Abs(sqrt(Fe) - sqrt(Na))'] = np.abs(np.sqrt(x2_af4_add['Fe']) - np.sqrt(x2_af4_add['Na']))
x2_af4_add['Support_SiO2**3*exp(3*Na)'] = x2_af4_add['Support_SiO2']**3 * np.exp(3*(x2_af4_add['Na']))

x2_af4_all = pd.concat([x2_af4, x2_af4_add], axis= 0)
x2_af4_all = x2_af4_all.sort_index()
x2_af4_all.to_csv('MyWork/datasets/x2_af4_all.csv', encoding= 'cp932')

#以下、2元交差項∔Borutaに関するもの
#データ読込み
x2_tr = pd.read_csv('MyWork/datasets/x2_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
#sklearnによる交差項付与
element_term = x2_tr.loc[:, 'Ba':]
skl_poly = PolynomialFeatures(degree= 2, interaction_only= True, include_bias= False)
cross_array = skl_poly.fit_transform(element_term)
feature_names = skl_poly.get_feature_names_out(input_features=element_term.columns)
cross_term = pd.DataFrame(cross_array, columns=feature_names, index= x2_tr.index)
x2_add_crs = pd.concat([x2_tr.loc[:, :'Support_SiO5'], cross_term], axis= 1)
x2_add_crs = x2_add_crs.sort_index()
x2_add_crs.to_csv('MyWork/datasets/x2_add_crs.csv', encoding= 'cp932')
#プロセス変数を含む全特徴量に対してBorutaを適用
Boruta_selected_x2 = Boruta_Apply(x2_add_crs, perc= 80)
x2_b80 = pd.concat([x2_tr['C2 yield'], Boruta_selected_x2], axis= 1)
x2_b80.to_csv('MyWork/datasets/x2_b80.csv', encoding= 'cp932')
Boruta_selected_x2 = Boruta_Apply(x2_add_crs, perc= 70)
x2_b70 = pd.concat([x2_tr['C2 yield'], Boruta_selected_x2], axis= 1)
x2_b70.to_csv('MyWork/datasets/x2_b70.csv', encoding= 'cp932')
Boruta_selected_x2 = Boruta_Apply(x2_add_crs, perc= 60)
x2_b60 = pd.concat([x2_tr['C2 yield'], Boruta_selected_x2], axis= 1)
x2_b60.to_csv('MyWork/datasets/x2_b60.csv', encoding= 'cp932')

#以下x2_add_crs_all, x2_b60の作成
x2 = pd.read_csv('MyWork/datasets/x2.csv', encoding= 'cp932', index_col= 0, header= 0)
#sklearnによる交差項付与
element_term = x2.loc[:, 'Ba':]
skl_poly = PolynomialFeatures(degree= 2, interaction_only= True, include_bias= False)
cross_array = skl_poly.fit_transform(element_term)
feature_names = skl_poly.get_feature_names_out(input_features=element_term.columns)
cross_term = pd.DataFrame(cross_array, columns=feature_names, index= x2.index)
x2_add_crs = pd.concat([x2.loc[:, :'Support_SiO5'], cross_term], axis= 1)
x2_add_crs = x2_add_crs.sort_index()
x2_add_crs.to_csv('MyWork/datasets/x2_add_crs_all.csv', encoding= 'cp932')

x2_tr = pd.read_csv('MyWork/datasets/x2_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_te = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_b60 = pd.read_csv('MyWork/datasets/x2_b60.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_b60_add = x2_te.copy()
x2_b60_add['Ba Fe'] = x2_b60_add['Ba'] * x2_b60_add['Fe']
x2_b60_add['Ba La'] = x2_b60_add['Ba'] * x2_b60_add['La']
x2_b60_add['Ba Mn'] = x2_b60_add['Ba'] * x2_b60_add['Mn']
x2_b60_add['Ba Sn'] = x2_b60_add['Ba'] * x2_b60_add['Sn']
x2_b60_add['Ba V'] = x2_b60_add['Ba'] * x2_b60_add['V']
x2_b60_add['Co Na'] = x2_b60_add['Co'] * x2_b60_add['Na']
x2_b60_add['Cr Cs'] = x2_b60_add['Cr'] * x2_b60_add['Cs']
x2_b60_add['Cr Mn'] = x2_b60_add['Cr'] * x2_b60_add['Mn']
x2_b60_add['Cr Na'] = x2_b60_add['Cr'] * x2_b60_add['Na']
x2_b60_add['Cr Sr'] = x2_b60_add['Cr'] * x2_b60_add['Sr']
x2_b60_add['Cs Mn'] = x2_b60_add['Cs'] * x2_b60_add['Mn']
x2_b60_add['Cu Mn'] = x2_b60_add['Cu'] * x2_b60_add['Mn']
x2_b60_add['Cu V'] = x2_b60_add['Cu'] * x2_b60_add['V']
x2_b60_add['Fe La'] = x2_b60_add['Fe'] * x2_b60_add['La']
x2_b60_add['Fe Mn'] = x2_b60_add['Fe'] * x2_b60_add['Mn']
x2_b60_add['Fe Na'] = x2_b60_add['Fe'] * x2_b60_add['Na']
x2_b60_add['La Mn'] = x2_b60_add['La'] * x2_b60_add['Mn']
x2_b60_add['La Sn'] = x2_b60_add['La'] * x2_b60_add['Sn']
x2_b60_add['Mn Na'] = x2_b60_add['Mn'] * x2_b60_add['Na']
x2_b60_add['Mn Rb'] = x2_b60_add['Mn'] * x2_b60_add['Rb']
x2_b60_add['Mn Sr'] = x2_b60_add['Mn'] * x2_b60_add['Sr']
x2_b60_add['Mn V'] = x2_b60_add['Mn'] * x2_b60_add['V']
x2_b60_add['Mn W'] = x2_b60_add['Mn'] * x2_b60_add['W']
x2_b60_add['Mn Y'] = x2_b60_add['Mn'] * x2_b60_add['Y']
x2_b60_add['Mn Zn'] = x2_b60_add['Mn'] * x2_b60_add['Zn']
x2_b60_add['Na V'] = x2_b60_add['Na'] * x2_b60_add['V']
x2_b60_add['Na W'] = x2_b60_add['Na'] * x2_b60_add['W']
x2_b60_add['Ni V'] = x2_b60_add['Ni'] * x2_b60_add['V']
x2_b60_add['Rb V'] = x2_b60_add['Rb'] * x2_b60_add['V']
x2_b60_add['V Y'] = x2_b60_add['V'] * x2_b60_add['Y']
x2_b60_add['V Zn'] = x2_b60_add['V'] * x2_b60_add['Zn']

x2_b60_add = x2_b60_add.drop(['Support_SiO3', 'Support_SiO4'], axis= 1)
x2_b60_all = pd.concat([x2_b60, x2_b60_add], axis= 0)
x2_b60_all = x2_b60_all.sort_index()
x2_b60_all.to_csv('MyWork/datasets/x2_b60_all.csv', encoding= 'cp932')

#feature-engineを使用して特徴量を選択
x2_add_cross = pd.read_csv('MyWork/datasets/x2_add_crs.csv', encoding= 'cp932', index_col= 0, header= 0)
#(1)scs
scs = SmartCorrelatedSelection(threshold=0.7, selection_method="variance")
#Varianceの高い変数を保持して、相関係数0.7を超える変数を削除する
x_scs = scs.fit_transform(x2_add_cross)
features_scs = x_scs.columns.to_list()
x_scs.to_csv('Mywork/datasets/x2_scs.csv', encoding= 'cp932')
#(2)psi
psi = DropHighPSIFeatures(split_col= 'C2 yield', threshold= 0.1)
#thresholdを0.1-0.4まで変化させても殆ど選択数に変化なし
x_psi = psi.fit_transform(x2_add_cross)
features_psi = psi.get_feature_names_out
x_psi.to_csv('Mywork/datasets/x2_psi.csv', encoding= 'cp932')
#(3)iv: Category変数用のためOmit
#(4)sbs
sbs = SelectByShuffling(threshold= 0.0001, estimator= RandomForestRegressor(), scoring= 'r2')
#threshold=0.001で説明変数11, 0.0001で17個
y = x2_add_cross['C2 yield']
x2_add_cross_wo_y = x2_add_cross.drop('C2 yield', axis= 1)
x_sbs = sbs.fit_transform(x2_add_cross_wo_y, y)
features_sbs = sbs.get_feature_names_out
x_sbs = pd.concat([y, x_sbs], axis= 1)
x_sbs.to_csv('Mywork/datasets/x2_sbs.csv', encoding= 'cp932')
#(5)sfp:謎エラーが出て無限ループになってしまうためOmit
#sfp = SelectBySingleFeaturePerformance(RandomForestRegressor(), scoring= 'r2')
#X_sfp = sfp.fit_transform(xx2_add_cross_wo_y, y)
#(6)tmp
tmp = SelectByTargetMeanPerformance(scoring="r2", threshold= 0.9, bins=3, strategy="equal_frequency", cv=3, regression= True)
x_tmp = tmp.fit_transform(x2_add_cross_wo_y, y)
tmp_features = tmp.variables_
x_tmp = x2_add_cross.loc[:, tmp_features]
x_tmp = pd.concat([y, x_tmp], axis= 1)
x_tmp.to_csv('Mywork/datasets/x2_tmp.csv', encoding= 'cp932')
#(7)pfs
pfs = ProbeFeatureSelection(estimator= LinearRegression(), scoring= 'r2', n_probes= 3)
x_pfs =pfs.fit_transform(x2_add_cross_wo_y, y) #RandomForestではempty dfとなるためLRとした
x_pfs = pd.concat([y, x_pfs], axis= 1)
x_pfs.to_csv('Mywork/datasets/x2_pfs.csv', encoding= 'cp932')
#(8)rfe
rfe = RecursiveFeatureElimination(estimator= RandomForestRegressor(), scoring= 'r2')
x_rfe = rfe.fit_transform(x2_add_cross_wo_y, y)
x_rfe = pd.concat([y, x_rfe], axis= 1)
x_rfe.to_csv('Mywork/datasets/x2_rfe.csv', encoding= 'cp932')
#(9)rfa
rfa = RecursiveFeatureAddition(estimator= RandomForestRegressor(), scoring= 'r2')
x_rfa = rfa.fit_transform(x2_add_cross_wo_y, y)
x_rfa = pd.concat([y, x_rfa], axis= 1)
x_rfa.to_csv('Mywork/datasets/x2_rfa.csv', encoding= 'cp932')

#x2_sbs_allの作成
x2_tr = pd.read_csv('MyWork/datasets/x2_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_te = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_sbs = pd.read_csv('MyWork/datasets/x2_sbs.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_sbs_add = x2_te.copy()
x2_sbs_add['Ba Fe'] = x2_sbs_add['Ba'] * x2_sbs_add['Fe']
x2_sbs_add['Cr Ni'] = x2_sbs_add['Cr'] * x2_sbs_add['Ni']
x2_sbs_add['Cr V'] = x2_sbs_add['Cr'] * x2_sbs_add['V']
x2_sbs_add['Cs Mn'] = x2_sbs_add['Cs'] * x2_sbs_add['Mn']
x2_sbs_add['Cu Mn'] = x2_sbs_add['Cu'] * x2_sbs_add['Mn']
x2_sbs_add['Fe La'] = x2_sbs_add['Fe'] * x2_sbs_add['La']
x2_sbs_add['Fe Mn'] = x2_sbs_add['Fe'] * x2_sbs_add['Mn']
x2_sbs_add['Fe Na'] = x2_sbs_add['Fe'] * x2_sbs_add['Na']
x2_sbs_add['La Sn'] = x2_sbs_add['La'] * x2_sbs_add['Sn']
x2_sbs_add['Mn Na'] = x2_sbs_add['Mn'] * x2_sbs_add['Na']
x2_sbs_add['Mn W'] = x2_sbs_add['Mn'] * x2_sbs_add['W']
x2_sbs_add['Na W'] = x2_sbs_add['Na'] * x2_sbs_add['W']
x2_sbs_add['Sr V'] = x2_sbs_add['Sr'] * x2_sbs_add['V']
x2_sbs_add['V Zn'] = x2_sbs_add['V'] * x2_sbs_add['Zn']

x2_sbs_add = x2_sbs_add.loc[:, x2_sbs.columns.to_list()]
x2_sbs_all = pd.concat([x2_sbs, x2_sbs_add], axis= 0)
x2_sbs_all = x2_sbs_all.sort_index()
x2_sbs_all.to_csv('MyWork/datasets/x2_sbs_all.csv', encoding= 'cp932')

#x2_crsに対してxfeatを適用
#input_colは目的変数と説明変数の両方
x2_crs = pd.read_csv('MyWork/datasets/x2_add_crs.csv', encoding= 'cp932', index_col= 0, header= 0)
feature_col = x2_crs.columns.to_list()

params = {
    "objective": "regression",
    "seed": 111,
}
fit_kwargs = {
    "num_boost_round": 10,
}
selector = GBDTFeatureSelector(
    input_cols=feature_col,
    target_col="C2 yield",
    threshold=0.9,
    lgbm_params=params,
    lgbm_fit_kwargs=fit_kwargs,
)
df_selected = selector.fit_transform(x2_crs)
#print("Selected columns:", selector._selected_cols)
df_selected.to_csv('MyWork/datasets/x2_xf9.csv', encoding= 'cp932')

def main():
    df = x2_crs
    df = df.rename(columns={'C2 yield': 'target'})
    print("After applying GBDTFeatureSelector:")
    df = feature_selection(df, df.target)
    evaluate_dataframe(df, df.target)
    df = df.rename(columns={'target': 'C2 yield'})
    #df.to_csv('MyWork/datasets/x1_6.csv', encoding= 'cp932')

def objective(df, selector, trial):
    selector.set_trial(trial)
    selector.fit(df)
    input_cols = selector.get_selected_cols()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }

    # Evaluate with selected columns
    train_set = lgb.Dataset(df[input_cols], label=df["target"])
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    return rmsle_score

def feature_selection(df, y):
    input_cols = df.columns.tolist()
    n_before_selection = len(input_cols)

    df["target"] = np.log1p(y)
    df_train, _ = train_test_split(df, test_size=0.25, random_state=42)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    fit_params = {
        "num_boost_round": 100,
    }
    selector = GBDTFeatureExplorer(input_cols=input_cols,
                                   target_col="target",
                                   fit_once=True,
                                   threshold_range=(0.6, 1.0),
                                   lgbm_params=params,
                                   lgbm_fit_kwargs=fit_params)

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, df_train, selector), n_trials=20)

    selector.from_trial(study.best_trial)
    selected_cols = selector.get_selected_cols()
    print(f" - {n_before_selection - len(selected_cols)} features are removed.")

    return df[selected_cols]

def evaluate_dataframe(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df.values, y,
                                                        test_size=0.5,
                                                        random_state=1)
    y_train = np.log1p(y_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    print(f" - CV RMSEL: {rmsle_score:.6f}")

    booster = lgb.train(params, train_set, num_boost_round=100)
    y_pred = booster.predict(X_test)
    test_rmsle_score = rmse(np.log1p(y_test), y_pred)
    test_r2_score = r2_score(y_test, y_pred)
    test_mae_score = mean_absolute_error(y_test, y_pred)
    print(f" - test RMSEL: {test_rmsle_score:.6f}")
    print(f" - test R2: {test_r2_score:.6f}")    
    print(f" - test MAE: {test_mae_score:.6f}") 

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    main()

#観念して、trainで作成した特徴量を全データで再現するコードを作る
train_df = pd.read_csv('MyWork/datasets/x2_xf7.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
columns_list = train_df.columns.to_list()
crs_list = [s for s in columns_list if ' ' in s]
del crs_list[0] #C2 yieldを削除
for item in crs_list:
    metals = item.split(' ')
    test_df[item] = test_df[metals[0]] * test_df[metals[1]]
test_df = test_df.loc[:, train_df.columns.to_list()]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df.to_csv('MyWork/datasets/x2_xf7_all.csv', encoding= 'cp932')

#triple_cross項を作成、選択する
x2_add_crs = pd.read_csv('MyWork/datasets/x2_add_crs.csv', encoding= 'cp932', index_col= 0, header= 0)
#sklearnによる交差項付与
element_term = x2_add_crs.loc[:, 'Ba':'Zn']
triple_cross = addTripleCross2(element_term)
x2_trp_crs = pd.concat([x2_add_crs.loc[:, :'Y Zn'], triple_cross.loc[:, 'Cr * Co * Ba':]], axis= 1)
x2_trp_crs = x2_trp_crs.sort_index()
x2_trp_crs.to_csv('MyWork/datasets/x2_trp_crs.csv', encoding= 'cp932')
#x2_add_crsでなく、x2_add_crs_allでも作成
x2_add_crs_all = pd.read_csv('MyWork/datasets/x2_add_crs_all.csv', encoding= 'cp932', index_col= 0, header= 0)
#sklearnによる交差項付与
element_term = x2_add_crs_all.loc[:, 'Ba':'Zn']
triple_cross = addTripleCross2(element_term)
x2_trp_crs = pd.concat([x2_add_crs_all.loc[:, :'Y Zn'], triple_cross.loc[:, 'Cr * Co * Ba': ]], axis= 1)
x2_trp_crs = x2_trp_crs.sort_index()
x2_trp_crs.to_csv('MyWork/datasets/x2_trp_crs_all.csv', encoding= 'cp932')

#triple_cross + Boruta
#プロセス変数を含む全特徴量に対してBorutaを適用
x2_trp_crs = pd.read_csv('MyWork/datasets/x2_trp_crs.csv', encoding= 'cp932', index_col= 0, header= 0)
Boruta_selected_x2 = Boruta_Apply(x2_trp_crs, perc= 60)
x2_t_b60 = pd.concat([x2_trp_crs['C2 yield'].copy(), Boruta_selected_x2], axis= 1)
x2_t_b60.to_csv('MyWork/datasets/x2_t_b60.csv', encoding= 'cp932')
Boruta_selected_x2 = Boruta_Apply(x2_trp_crs, perc= 80)
x2_t_b80 = pd.concat([x2_trp_crs['C2 yield'].copy(), Boruta_selected_x2], axis= 1)
x2_t_b80.to_csv('MyWork/datasets/x2_t_b80.csv', encoding= 'cp932')
Boruta_selected_x2 = Boruta_Apply(x2_trp_crs, perc= 90)
x2_t_b90 = pd.concat([x2_trp_crs['C2 yield'].copy(), Boruta_selected_x2], axis= 1)
x2_t_b90.to_csv('MyWork/datasets/x2_t_b90.csv', encoding= 'cp932')

#x2_t_b40を再現
train_df = pd.read_csv('MyWork/datasets/x2_t_b90.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
columns_list = train_df.columns.to_list()
crs_list = [s for s in columns_list if ' ' in s]
del crs_list[0] #C2 yieldを削除
crs_list2 = [s for s in crs_list if '*' in s]
crs_list1 = set(crs_list) - set(crs_list2)
crs_list1 = list(crs_list1)
for item in crs_list1:
    metals = item.split(' ')
    test_df[item] = test_df[metals[0]] * test_df[metals[1]]
for item in crs_list2:
    metals2 = item.split(' * ')
    test_df[item] = test_df[metals2[0]] * test_df[metals2[1]] * test_df[metals2[2]]
test_df = test_df.loc[:, train_df.columns.to_list()]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df.to_csv('MyWork/datasets/x2_t_b90_all.csv', encoding= 'cp932')

#tempo: x1を保存しそこなっていたため急遽作成したもの
x1_tr = pd.read_csv('MyWork/datasets/x1_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
x1_te = pd.read_csv('MyWork/datasets/x1_te.csv', encoding= 'cp932', index_col= 0, header= 0)
x1 = pd.concat([x1_tr, x1_te], axis= 0)
x1 = x1.sort_index()
x1.to_csv('MyWork/datasets/x1.csv', encoding= 'cp932')

#feature-engineを使用して特徴量を選択
x2_add_cross = pd.read_csv('MyWork/datasets/x2_trp_crs.csv', encoding= 'cp932', index_col= 0, header= 0)
#(1)scs
scs = SmartCorrelatedSelection(threshold=0.9, selection_method="variance")
#Varianceの高い変数を保持して、相関係数0.7を超える変数を削除する
x_scs = scs.fit_transform(x2_add_cross)
features_scs = x_scs.columns.to_list()
x_scs.to_csv('Mywork/datasets/x2_43.csv', encoding= 'cp932')
#(4)sbs
sbs = SelectByShuffling(threshold= 0.01, estimator= RandomForestRegressor(), scoring= 'r2')
#threshold=0.001で説明変数11, 0.0001で17個
y = x2_add_cross['C2 yield'].sort_index()
x2_add_cross_wo_y = x2_add_cross.drop('C2 yield', axis= 1)
x_sbs = sbs.fit_transform(x2_add_cross_wo_y, y)
features_sbs = sbs.get_feature_names_out
x_sbs = pd.concat([y, x_sbs], axis= 1)
x_sbs.to_csv('Mywork/datasets/x2_40.csv', encoding= 'cp932')
#(6)tmp
tmp = SelectByTargetMeanPerformance(scoring="r2", threshold= 0.7, bins=3, strategy="equal_frequency", cv=3, regression= True)
x_tmp = tmp.fit_transform(x2_add_cross_wo_y, y)
tmp_features = tmp.variables_
x_tmp = x2_add_cross.loc[:, tmp_features]
x_tmp = pd.concat([y, x_tmp], axis= 1)
x_tmp.to_csv('Mywork/datasets/x2_46.csv', encoding= 'cp932')

#x2_t_scsを再現
train_df = pd.read_csv('MyWork/datasets/x2_t_scs.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
columns_list = train_df.columns.to_list()
crs_list = [s for s in columns_list if ' ' in s]
del crs_list[0] #C2 yieldを削除
crs_list2 = [s for s in crs_list if '*' in s]
crs_list1 = set(crs_list) - set(crs_list2)
crs_list1 = list(crs_list1)
for item in crs_list1:
    metals = item.split(' ')
    test_df[item] = test_df[metals[0]] * test_df[metals[1]]
for item in crs_list2:
    metals2 = item.split(' * ')
    test_df[item] = test_df[metals2[0]] * test_df[metals2[1]] * test_df[metals2[2]]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df = train_test_df.dropna(how= 'any', axis= 1)
train_test_df.to_csv('MyWork/datasets/x2_t_scs_all.csv', encoding= 'cp932')

#x2_trp_crsに対してxfeatを適用
#input_colは目的変数と説明変数の両方
x2_crs = pd.read_csv('MyWork/datasets/x2_trp_crs.csv', encoding= 'cp932', index_col= 0, header= 0)
feature_col = x2_crs.columns.to_list()

params = {
    "objective": "regression",
    "seed": 111,
}
fit_kwargs = {
    "num_boost_round": 10,
}
selector = GBDTFeatureSelector(
    input_cols=feature_col,
    target_col="C2 yield",
    threshold=0.6,
    lgbm_params=params,
    lgbm_fit_kwargs=fit_kwargs,
)
df_selected = selector.fit_transform(x2_crs)
#print("Selected columns:", selector._selected_cols)
df_selected.to_csv('MyWork/datasets/x2_t_xf6.csv', encoding= 'cp932')

def main():
    df = x2_crs
    df = df.rename(columns={'C2 yield': 'target'})
    print("After applying GBDTFeatureSelector:")
    df = feature_selection(df, df.target)
    evaluate_dataframe(df, df.target)
    df = df.rename(columns={'target': 'C2 yield'})
    #df.to_csv('MyWork/datasets/x1_6.csv', encoding= 'cp932')

def objective(df, selector, trial):
    selector.set_trial(trial)
    selector.fit(df)
    input_cols = selector.get_selected_cols()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }

    # Evaluate with selected columns
    train_set = lgb.Dataset(df[input_cols], label=df["target"])
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    return rmsle_score

def feature_selection(df, y):
    input_cols = df.columns.tolist()
    n_before_selection = len(input_cols)

    df["target"] = np.log1p(y)
    df_train, _ = train_test_split(df, test_size=0.25, random_state=42)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    fit_params = {
        "num_boost_round": 100,
    }
    selector = GBDTFeatureExplorer(input_cols=input_cols,
                                   target_col="target",
                                   fit_once=True,
                                   threshold_range=(0.6, 1.0),
                                   lgbm_params=params,
                                   lgbm_fit_kwargs=fit_params)

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, df_train, selector), n_trials=20)

    selector.from_trial(study.best_trial)
    selected_cols = selector.get_selected_cols()
    print(f" - {n_before_selection - len(selected_cols)} features are removed.")

    return df[selected_cols]

def evaluate_dataframe(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df.values, y,
                                                        test_size=0.5,
                                                        random_state=1)
    y_train = np.log1p(y_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    print(f" - CV RMSEL: {rmsle_score:.6f}")

    booster = lgb.train(params, train_set, num_boost_round=100)
    y_pred = booster.predict(X_test)
    test_rmsle_score = rmse(np.log1p(y_test), y_pred)
    test_r2_score = r2_score(y_test, y_pred)
    test_mae_score = mean_absolute_error(y_test, y_pred)
    print(f" - test RMSEL: {test_rmsle_score:.6f}")
    print(f" - test R2: {test_r2_score:.6f}")    
    print(f" - test MAE: {test_mae_score:.6f}") 

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    main()

#x2_t_xf7を再現
train_df = pd.read_csv('MyWork/datasets/x2_t_xf8.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
columns_list = train_df.columns.to_list()
crs_list = [s for s in columns_list if ' ' in s]
del crs_list[0] #C2 yieldを削除
crs_list2 = [s for s in crs_list if '*' in s]
crs_list1 = set(crs_list) - set(crs_list2)
crs_list1 = list(crs_list1)
for item in crs_list1:
    metals = item.split(' ')
    test_df[item] = test_df[metals[0]] * test_df[metals[1]]
for item in crs_list2:
    metals2 = item.split(' * ')
    test_df[item] = test_df[metals2[0]] * test_df[metals2[1]] * test_df[metals2[2]]
test_df = test_df.loc[:, train_df.columns.to_list()]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df.to_csv('MyWork/datasets/x2_t_xf8_all.csv', encoding= 'cp932')

#x1をベースにする
#プロセス変数を含む全特徴量に対してBorutaを適用
x1_tr = pd.read_csv('MyWork/datasets/x1_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
Boruta_selected_x1 = Boruta_Apply(x1_tr, perc= 60)
x1_b60 = pd.concat([x1_tr['C2 yield'], Boruta_selected_x1], axis= 1)
x1_b60.to_csv('MyWork/datasets/x1_b60.csv', encoding= 'cp932')
Boruta_selected_x1 = Boruta_Apply(x1_tr, perc= 80)
x1_b80 = pd.concat([x1_tr['C2 yield'], Boruta_selected_x1], axis= 1)
x1_b80.to_csv('MyWork/datasets/x1_b80.csv', encoding= 'cp932')
Boruta_selected_x1 = Boruta_Apply(x1_tr, perc= 70)
x1_b70 = pd.concat([x1_tr['C2 yield'], Boruta_selected_x1], axis= 1)
x1_b70.to_csv('MyWork/datasets/x1_b70.csv', encoding= 'cp932')

#x1_b90の再現
train_df = pd.read_csv('MyWork/datasets/x1_b60.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x1_te.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = test_df.loc[:, train_df.columns.to_list()]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df.to_csv('MyWork/datasets/x1_b60_all.csv', encoding= 'cp932')

#feature-engineを使用して特徴量を選択
x2_add_cross = pd.read_csv('MyWork/datasets/x1_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
#(1)scs
scs = SmartCorrelatedSelection(threshold=0.7, selection_method="variance")
#Varianceの高い変数を保持して、相関係数0.7を超える変数を削除する
x_scs = scs.fit_transform(x2_add_cross)
features_scs = x_scs.columns.to_list()
x_scs.to_csv('Mywork/datasets/x1_scs.csv', encoding= 'cp932')
#(4)sbs
sbs = SelectByShuffling(threshold= 0.0001, estimator= RandomForestRegressor(), scoring= 'r2')
#threshold=0.001で説明変数11, 0.0001で17個
y = x2_add_cross['C2 yield']
x2_add_cross_wo_y = x2_add_cross.drop('C2 yield', axis= 1)
x_sbs = sbs.fit_transform(x2_add_cross_wo_y, y)
features_sbs = sbs.get_feature_names_out
x_sbs = pd.concat([y, x_sbs], axis= 1)
x_sbs.to_csv('Mywork/datasets/x1_sbs.csv', encoding= 'cp932')
#(6)tmp
tmp = SelectByTargetMeanPerformance(scoring="r2", threshold= 0.9, bins=3, strategy="equal_frequency", cv=3, regression= True)
x_tmp = tmp.fit_transform(x2_add_cross_wo_y, y)
tmp_features = tmp.variables_
x_tmp = x2_add_cross.loc[:, tmp_features]
x_tmp = pd.concat([y, x_tmp], axis= 1)
x_tmp.to_csv('Mywork/datasets/x1_tmp.csv', encoding= 'cp932')

#x1_scsの再現
train_df = pd.read_csv('MyWork/datasets/x1_sbs.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x1_te.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = test_df.loc[:, train_df.columns.to_list()]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df.to_csv('MyWork/datasets/x1_sbs_all.csv', encoding= 'cp932')

#x2_trp_crsに対してxfeatを適用
#input_colは目的変数と説明変数の両方
x2_crs = pd.read_csv('MyWork/datasets/x1_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
feature_col = x2_crs.columns.to_list()

params = {
    "objective": "regression",
    "seed": 111,
}
fit_kwargs = {
    "num_boost_round": 10,
}
selector = GBDTFeatureSelector(
    input_cols=feature_col,
    target_col="C2 yield",
    threshold=0.8,
    lgbm_params=params,
    lgbm_fit_kwargs=fit_kwargs,
)
df_selected = selector.fit_transform(x2_crs)
#print("Selected columns:", selector._selected_cols)
df_selected.to_csv('MyWork/datasets/x1_xf8.csv', encoding= 'cp932')

def main():
    df = x2_crs
    df = df.rename(columns={'C2 yield': 'target'})
    print("After applying GBDTFeatureSelector:")
    df = feature_selection(df, df.target)
    evaluate_dataframe(df, df.target)
    df = df.rename(columns={'target': 'C2 yield'})
    #df.to_csv('MyWork/datasets/x1_6.csv', encoding= 'cp932')

def objective(df, selector, trial):
    selector.set_trial(trial)
    selector.fit(df)
    input_cols = selector.get_selected_cols()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }

    # Evaluate with selected columns
    train_set = lgb.Dataset(df[input_cols], label=df["target"])
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    return rmsle_score

def feature_selection(df, y):
    input_cols = df.columns.tolist()
    n_before_selection = len(input_cols)

    df["target"] = np.log1p(y)
    df_train, _ = train_test_split(df, test_size=0.25, random_state=42)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    fit_params = {
        "num_boost_round": 100,
    }
    selector = GBDTFeatureExplorer(input_cols=input_cols,
                                   target_col="target",
                                   fit_once=True,
                                   threshold_range=(0.6, 1.0),
                                   lgbm_params=params,
                                   lgbm_fit_kwargs=fit_params)

    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, df_train, selector), n_trials=20)

    selector.from_trial(study.best_trial)
    selected_cols = selector.get_selected_cols()
    print(f" - {n_before_selection - len(selected_cols)} features are removed.")

    return df[selected_cols]

def evaluate_dataframe(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df.values, y,
                                                        test_size=0.5,
                                                        random_state=1)
    y_train = np.log1p(y_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    scores = lgb.cv(params, train_set, num_boost_round=100, stratified=False, seed=1)
    rmsle_score = scores["rmse-mean"][-1]
    print(f" - CV RMSEL: {rmsle_score:.6f}")

    booster = lgb.train(params, train_set, num_boost_round=100)
    y_pred = booster.predict(X_test)
    test_rmsle_score = rmse(np.log1p(y_test), y_pred)
    test_r2_score = r2_score(y_test, y_pred)
    test_mae_score = mean_absolute_error(y_test, y_pred)
    print(f" - test RMSEL: {test_rmsle_score:.6f}")
    print(f" - test R2: {test_r2_score:.6f}")    
    print(f" - test MAE: {test_mae_score:.6f}") 

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    main()

#x1_xf8の再現
train_df = pd.read_csv('MyWork/datasets/x1_xf7.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = pd.read_csv('MyWork/datasets/x1_te.csv', encoding= 'cp932', index_col= 0, header= 0)
test_df = test_df.loc[:, train_df.columns.to_list()]
train_test_df = pd.concat([train_df, test_df], axis= 0)
train_test_df = train_test_df.sort_index()
train_test_df.to_csv('MyWork/datasets/x1_xf7_all.csv', encoding= 'cp932')

#多くのtrデータからteデータを再現する
#保存日付を指定してデータ読込み
file_path = './MyWork/datasets/'
target_date1 = '2023-11-13'
target_date2 = '2023-11-14'

df_dict = load_files_as_dict_on_saved_date(file_path, target_date1)
df_dict2 = load_files_as_dict_on_saved_date(file_path, target_date2)
df_dict.update(df_dict2)
#x1, x2_crs, x2_t_crsに分割する
x1_list = [f"x1_{i}.csv" for i in range(24, 33)]
x2_crs_list = [f"x2_{i}.csv" for i in range(6, 15)]
x2_t_crs_list = [f"x2_{i}.csv" for i in range(38, 47)]

#x1_listの再現
test_df = df_dict['x1_te.csv']
for item in x1_list:
    train_df = df_dict[item]
    test_df_ite = test_df.copy()
    test_df_ite = test_df_ite.loc[:, train_df.columns.to_list()]
    train_test_df = pd.concat([train_df, test_df_ite], axis= 0)
    train_test_df = train_test_df.sort_index()
    train_test_df.to_csv(file_path+item[:-4]+'_all.csv', encoding= 'cp932')

#x2_crs_listの再現
test_df = pd.read_csv(file_path+'x2_te.csv', index_col= 0, header= 0)
for element in x2_crs_list:
    train_df = df_dict[element]
    test_df_ite = test_df.copy()
    columns_list = train_df.columns.to_list()
    crs_list = [s for s in columns_list if ' ' in s]
    del crs_list[0] #C2 yieldを削除
    for item in crs_list:
        metals = item.split(' ')
        test_df_ite[item] = test_df_ite[metals[0]] * test_df_ite[metals[1]]
    test_df_ite = test_df_ite.loc[:, train_df.columns.to_list()]
    train_test_df = pd.concat([train_df, test_df_ite], axis= 0)
    train_test_df = train_test_df.sort_index()
    train_test_df.to_csv(file_path+element[:-4]+'_all.csv', encoding= 'cp932')

#x2_t_crs_listの再現
test_df = pd.read_csv(file_path+'x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
for element in x2_t_crs_list:
    train_df = df_dict[element]
    test_df_ite = test_df.copy()
    columns_list = train_df.columns.to_list()
    crs_list = [s for s in columns_list if ' ' in s]
    del crs_list[0] #C2 yieldを削除
    crs_list2 = [s for s in crs_list if '*' in s]
    crs_list1 = set(crs_list) - set(crs_list2)
    crs_list1 = list(crs_list1)
    for item in crs_list1:
        metals = item.split(' ')
        test_df_ite[item] = test_df_ite[metals[0]] * test_df_ite[metals[1]]
    for item in crs_list2:
        metals2 = item.split(' * ')
        test_df_ite[item] = test_df_ite[metals2[0]] * test_df_ite[metals2[1]] * test_df_ite[metals2[2]]
    test_df_ite = test_df_ite.loc[:, train_df.columns.to_list()]
    train_test_df = pd.concat([train_df, test_df_ite], axis= 0)
    train_test_df = train_test_df.sort_index()
    train_test_df.to_csv(file_path+element[:-4]+'_all.csv', encoding= 'cp932')
"""
#af1-af3を再現する
#自動化しようとしたが思いのほか面倒そうなので、原始的手法とする
#x2_af1とx2_af2は同じだった
x2_tr = pd.read_csv('MyWork/datasets/x2_tr.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_te = pd.read_csv('MyWork/datasets/x2_te.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_af3 = pd.read_csv('MyWork/datasets/x2_af3.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_af2 = pd.read_csv('MyWork/datasets/x2_af2.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_af1 = pd.read_csv('MyWork/datasets/x2_af1.csv', encoding= 'cp932', index_col= 0, header= 0)
x2_af3_add = x2_te.copy()
x2_af2_add = x2_te.copy()
x2_af1_add = x2_te.copy()
x2_af3_add['Sn*Temp**3'] = x2_af3_add['Sn'] * x2_af3_add['Temp']**3
x2_af3_add['Na**3*Temp**3'] = x2_af3_add['Na']**3 * x2_af3_add['Temp']**3
x2_af3_add['Temp**3*exp(Mn)'] = x2_af3_add['Temp']**3 * np.exp(x2_af3_add['Mn'])
x2_af3_add['Support_SiO2*Temp**3'] = x2_af3_add['Support_SiO2'] * x2_af3_add['Temp']**3
x2_af3_add['sqrt(Mn)*Support_SiO5'] = np.sqrt(x2_af3_add['Mn']) * x2_af3_add['Support_SiO5']
x2_af3_add['Cu*Mn'] = x2_af3_add['Cu'] * x2_af3_add['Mn']
x2_af3_add['Na*Support_SiO2'] = x2_af3_add['Na'] * x2_af3_add['Support_SiO2']
x2_af3_add['Temp**3*exp(Fe)'] = x2_af3_add['Temp']**3 * np.exp(x2_af3_add['Fe'])
x2_af3_add['sqrt(Ba)*Temp**3'] = np.sqrt(x2_af3_add['Ba']) * x2_af3_add['Temp']**3
x2_af3_add['Support_SiO2*exp(Mn)'] = x2_af3_add['Support_SiO2'] * np.exp(x2_af3_add['Mn'])
x2_af3_add['Support_SiO2*sqrt(V)'] = x2_af3_add['Support_SiO2'] * np.sqrt(x2_af3_add['V'])

x2_af2_add['Na**3*Temp**3'] = x2_af2_add['Na']**3 * x2_af2_add['Temp']**3
x2_af2_add['Temp**3*exp(Mn)'] = x2_af2_add['Temp']**3 * np.exp(x2_af2_add['Mn'])
x2_af2_add['Support_SiO2*Temp**3'] = x2_af2_add['Support_SiO2'] * x2_af2_add['Temp']**3
x2_af2_add['Na*Support_SiO2'] = x2_af2_add['Na'] * x2_af2_add['Support_SiO2']

x2_af3_all = pd.concat([x2_af3, x2_af3_add], axis= 0)
x2_af3_all = x2_af3_all.sort_index()
x2_af3_all.to_csv('MyWork/datasets/x2_af3_all.csv', encoding= 'cp932')
        
x2_af2_all = pd.concat([x2_af2, x2_af2_add], axis= 0)
x2_af2_all = x2_af2_all.sort_index()
x2_af2_all.to_csv('MyWork/datasets/x2_af2_all.csv', encoding= 'cp932')

print('hello')