# Autoencoderを使わずGRMから実温データを予測する
import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
# 下の行はDell_GPUを使用するときのみ
os.chdir('G:/マイドライブ')
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
from dcekit.generative_model import GMR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# GPUを使用するときは下4行をactiveにすること
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from datetime import datetime
from sklearn.model_selection import KFold

# x2をmetal_xに変換する
def create_metalx(row):
    sorted_elements = row[row > 0].sort_values(ascending= False)
    metals = sorted_elements.index.tolist()
    ratios = sorted_elements.values.tolist()
    metals.extend([0.0] * (5 - len(metals)))
    ratios.extend([0.0] * (5 - len(ratios)))
    return pd.Series(metals + ratios, index= ['metal1','metal2','metal3','metal4','metal5','ratio1','ratio2','ratio3','ratio4','ratio5'])

# Settings
numbers_of_components = np.arange(2, 31, 2)
covariance_types = ['full', 'diag', 'tied', 'spherical']
#covariance_types = ['full', 'diag']
fold_number = 5
save_dir = 'result/autoencoder_data/'
number_of_test_samples = 3
random_state = 100

batch_size = 32
lr = 0.005
#kf = KFold(n_splits = 5)

# data入力
data = pd.read_csv('datasets/v592_Ti/x2_ce_temp.csv', encoding='utf-8-sig', index_col=0, header=0)

# metal_x形式で内容確認
x2_compo = data.loc[:, 'Al':'Zr'].copy()
metal_x = x2_compo.apply(create_metalx, axis= 1)  

# 今回のターゲット触媒の設定(250708)
list_a = [13180, 14822, 15746] #高収率触媒top3
list_b = [15256, 15816, 14874] #低収率、選択率、組成はlist_aに近似、触媒熱暴走タイプ


# list_bに基づいた解析：全時間を予測
x = data.copy().drop(['選択率NPA', '収率NPA', '触媒ロット'], axis=1)
y = data.copy().loc[:, ['選択率NPA', '収率NPA']]

# train_test_split
#x_train, x_test = train_test_split(x, test_size=number_of_test_samples, shuffle=True, random_state=random_state)
x_train = x[~x.index.isin(list_b)].copy()
x_test = x[x.index.isin(list_b)].copy()
print('説明変数の数 : {0}'.format(x_train.shape[1]))

# 同じ値を多く持つ候補を削除
x_train = x_train.loc[:, columns := x_train.columns[x_train.nunique() > 2]]
x_test = x_test.loc[:, columns]
x = x.loc[:, columns]

# dataのサイズを小さくしてdebug
#x = x.iloc[:60, :]

#temp_col = x.loc[:, '1':].columns.tolist()  # 温度の列名を取得
x_col = x.columns.tolist()  # 説明変数の列名を取得

# 触媒1つずつ処理
result_dfs = {}
for cat in x_test.index:
    # targetの設定, time_2_halfを含む
    target_s = list(np.round(np.arange(0.1, 16.1, 0.1), 1))  #選択率 160分割
    target_y = list(np.round(np.arange(0.01, 1.61, 0.01), 2))  #収率
    target_df = pd.DataFrame(columns=['選択率NPA','収率NPA'], index=range(len(target_s)))
    target_df['選択率NPA'] = target_s
    target_df['収率NPA'] = target_y

    y_names = y.columns.tolist()
    x_names = x.columns.tolist()
    y_x = pd.concat([y, x], axis=1)

    numbers_of_X = [y_x.columns.get_loc(c) for c in x_names]
    numbers_of_y = [y_x.columns.get_loc(c) for c in y_names]
    #numbers_of_temp = [y_x.columns.get_loc(c)-y.shape[1] for c in temp_col]

    predicted_y_test = pd.DataFrame(columns=['選択率NPA','収率NPA'], index=target_df.index)
    #estimated_x_test = pd.DataFrame(columns=temp_col, index=target_df.index)
    estimated_x_test = pd.DataFrame(columns=x_col, index=target_df.index)

    target_dfs = {}

    for i in range(16):
        print(i+1, '/16')
        target_dfs[f'target_df_{i}'] = target_df.iloc[i*10:(i+1)*10, :]
        #variables = np.c_[x, y1, y2]
        variables = y_x.values
        # list_bに基づいてtrain/testを分割（64-65行目の分割と一致させる）
        y_x_train = y_x[~y_x.index.isin(list_b)]
        y_x_test = y_x[y_x.index == cat]
        variables_train = y_x_train.values
        variables_test = np.tile(y_x_test.values, (10, 1))

        # variables_testのyをtarget_dfsから取得
        target_y_test = target_dfs[f'target_df_{i}'].values
        variables_test[:, numbers_of_y] = target_y_test

        # Standardize x and y
        autoscaled_variables_train = (variables_train - variables_train.mean(axis=0)) / variables_train.std(axis=0, ddof=1)
        autoscaled_variables_test = (variables_test - variables_train.mean(axis=0)) / variables_train.std(axis=0, ddof=1)

        # GMR
        reg_model = GMR()
        # Grid search with cross-validation
        reg_model.cv_opt(autoscaled_variables_train, numbers_of_X, numbers_of_y, covariance_types, numbers_of_components,
                    fold_number)
        print('max r2cv :', reg_model.r2cv)

        # Modeling
        reg_model.fit(autoscaled_variables_train)

        # Forward analysis (regression)
        predicted_y_test_all = reg_model.predict_rep(autoscaled_variables_test[:, numbers_of_X], numbers_of_X, numbers_of_y)

        # Inverse analysis
        predicted_x_test_all = reg_model.predict_rep(autoscaled_variables_test[:, numbers_of_y], numbers_of_y, numbers_of_X)

        # Check results of forward analysis (regression)
        print('Results of forward analysis (regression)')
        plt.rcParams['font.size'] = 18
        for Y_number in range(len(numbers_of_y)):
            predicted_ytest = np.ndarray.flatten(predicted_y_test_all[:, Y_number])
            predicted_ytest = predicted_ytest * variables_train[:, numbers_of_y[Y_number]].std(ddof=1) + \
                            variables_train[:, numbers_of_y[Y_number]].mean()
            # yy-plot
            plt.figure(figsize=figure.figaspect(1))
            plt.scatter(variables_test[:, numbers_of_y[Y_number]], predicted_ytest)
            y_max = np.max(np.array([np.array(variables_test[:, numbers_of_y[Y_number]]), predicted_ytest]))
            y_min = np.min(np.array([np.array(variables_test[:, numbers_of_y[Y_number]]), predicted_ytest]))
            plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                    [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
            plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
            plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
            plt.xlabel('Actual Y')
            plt.ylabel('Estimated Y')
            #plt.show()
            # r2p, RMSEp, MAEp
            print('r2p: {0}'.format(float(1 - sum((variables_test[:, numbers_of_y[Y_number]] - predicted_ytest) ** 2) / sum(
                (variables_test[:, numbers_of_y[Y_number]] - variables_train[:, numbers_of_y[Y_number]].mean()) ** 2))))
            print('RMSEp: {0}'.format(float((sum((variables_test[:, numbers_of_y[Y_number]] - predicted_ytest) ** 2) / len(
                variables_test[:, numbers_of_y[Y_number]])) ** 0.5)))
            print('MAEp: {0}'.format(float(sum(abs(variables_test[:, numbers_of_y[Y_number]] - predicted_ytest)) / len(
                variables_test[:, numbers_of_y[Y_number]]))))
            input_index = target_dfs[f'target_df_{i}'].index
            predicted_y_test.loc[input_index, y_names[Y_number]] = predicted_ytest

        # Check results of inverse analysis
        print('---------------------------')
        print('Results of inverse analysis')
        estimated_X_test = predicted_x_test_all * np.matlib.repmat(variables_train[:, numbers_of_X].std(ddof=1, axis=0), predicted_x_test_all.shape[0], 1) + \
                        np.matlib.repmat(variables_train[:, numbers_of_X].mean(axis=0), predicted_x_test_all.shape[0], 1)
        input_index = target_dfs[f'target_df_{i}'].index
        #estimated_x_test.loc[input_index, temp_col] = estimated_X_test[:, numbers_of_temp]
        estimated_x_test.loc[input_index, x_col] = estimated_X_test[:, :]

    
    # 各catの結果をresult_dfsに格納
    result_dfs[f'predicted_y_cat_{cat}'] = predicted_y_test
    result_dfs[f'estimated_x_cat_{cat}'] = estimated_x_test

# 全ての結果を保存
for key, df in result_dfs.items():
    df.to_csv(f'result/autoencoder_data/{key}.csv', encoding='utf-8-sig')



# list_aに基づいた解析：1-30分までのデータを渡して残り時間を予測
time_1_half = data.loc[:, '1':'30'].copy().columns.tolist()
time_2_half = data.loc[:, '31':'60'].copy().columns.tolist()
x_a = data.copy().drop(['選択率NPA', '収率NPA', '触媒ロット'] + time_1_half, axis=1)
y_a = data.copy().loc[:, ['選択率NPA', '収率NPA'] + time_1_half]

# train_test_split
#x_train_a, x_test_a = train_test_split(x_a, test_size=number_of_test_samples, shuffle=True, random_state=random_state)
x_train_a = x_a[~x_a.index.isin(list_a)].copy()
x_test_a = x_a[x_a.index.isin(list_a)].copy()
print('説明変数の数 : {0}'.format(x_train_a.shape[1]))

# 同じ値を多く持つ候補を削除
x_train_a = x_train_a.loc[:, columns_a := x_train_a.columns[x_train_a.nunique() > 2]]
x_test_a = x_test_a.loc[:, columns_a]
x_a = x_a.loc[:, columns_a]

# dataのサイズを小さくしてdebug
#x_a = x_a.iloc[:60, :]

#temp_col_a = x_a.loc[:, '1':].columns.tolist()  # 温度の列名を取得
x_col_a = x_a.columns.tolist()  # 説明変数の列名を取得

# 触媒1つずつ処理
result_dfs_a = {}
for cat in x_test_a.index:
    # targetの設定, time_2_halfを含む
    target_s_a = list(np.round(np.arange(0.1, 16.1, 0.1), 1))  #選択率 160分割
    target_y_a = list(np.round(np.arange(0.01, 1.61, 0.01), 2))  #収率
    target_df_a = pd.DataFrame(columns=['選択率NPA','収率NPA'] + time_1_half, index=range(len(target_s_a)))
    target_df_a['選択率NPA'] = target_s_a
    target_df_a['収率NPA'] = target_y_a
    # target_df_aのtime_1_half列に、dataの[cat, time_1_half]の値（1行分）を全行に代入
    for col in time_1_half:
        target_df_a[col] = data.loc[cat, col]

    y_names_a = y_a.columns.tolist()
    x_names_a = x_a.columns.tolist()
    y_x_a = pd.concat([y_a, x_a], axis=1)

    numbers_of_X_a = [y_x_a.columns.get_loc(c) for c in x_names_a]
    numbers_of_y_a = [y_x_a.columns.get_loc(c) for c in y_names_a]
    #numbers_of_temp_a = [y_x_a.columns.get_loc(c)-y_a.shape[1] for c in temp_col_a]

    predicted_y_test_a = pd.DataFrame(columns=['選択率NPA','収率NPA'] + time_1_half, index=target_df_a.index)
    #estimated_x_test_a = pd.DataFrame(columns=temp_col_a, index=target_df_a.index)
    estimated_x_test_a = pd.DataFrame(columns=x_col_a, index=target_df_a.index)

    target_dfs_a = {}

    for i in range(16):
        print(i+1, '/16')
        target_dfs_a[f'target_df_{i}'] = target_df_a.iloc[i*10:(i+1)*10, :]
        #variables_a = np.c_[x_a, y1_a, y2_a]
        variables_a = y_x_a.values
        # list_aに基づいてtrain/testを分割（199-200行目の分割と一致させる）
        y_x_train_a = y_x_a[~y_x_a.index.isin(list_a)]
        y_x_test_a = y_x_a[y_x_a.index == cat]
        variables_train_a = y_x_train_a.values
        variables_test_a = np.tile(y_x_test_a.values, (10, 1))

        # variables_test_aのyをtarget_dfs_aから取得
        target_y_test_a = target_dfs_a[f'target_df_{i}'].values
        variables_test_a[:, numbers_of_y_a] = target_y_test_a

        # Standardize x and y
        autoscaled_variables_train_a = (variables_train_a - variables_train_a.mean(axis=0)) / variables_train_a.std(axis=0, ddof=1)
        autoscaled_variables_test_a = (variables_test_a - variables_train_a.mean(axis=0)) / variables_train_a.std(axis=0, ddof=1)

        # GMR
        reg_model_a = GMR()
        # Grid search with cross-validation
        reg_model_a.cv_opt(autoscaled_variables_train_a, numbers_of_X_a, numbers_of_y_a, covariance_types, numbers_of_components,
                    fold_number)
        print('max r2cv :', reg_model_a.r2cv)

        # Modeling
        reg_model_a.fit(autoscaled_variables_train_a)

        # Forward analysis (regression)
        predicted_y_test_all_a = reg_model_a.predict_rep(autoscaled_variables_test_a[:, numbers_of_X_a], numbers_of_X_a, numbers_of_y_a)

        # Inverse analysis
        predicted_x_test_all_a = reg_model_a.predict_rep(autoscaled_variables_test_a[:, numbers_of_y_a], numbers_of_y_a, numbers_of_X_a)

        # Check results of forward analysis (regression)
        print('Results of forward analysis (regression)')
        plt.rcParams['font.size'] = 18
        for Y_number in range(len(numbers_of_y_a)):
            predicted_ytest_a = np.ndarray.flatten(predicted_y_test_all_a[:, Y_number])
            predicted_ytest_a = predicted_ytest_a * variables_train_a[:, numbers_of_y_a[Y_number]].std(ddof=1) + \
                            variables_train_a[:, numbers_of_y_a[Y_number]].mean()
            # yy-plot
            plt.figure(figsize=figure.figaspect(1))
            plt.scatter(variables_test_a[:, numbers_of_y_a[Y_number]], predicted_ytest_a)
            y_max = np.max(np.array([np.array(variables_test_a[:, numbers_of_y_a[Y_number]]), predicted_ytest_a]))
            y_min = np.min(np.array([np.array(variables_test_a[:, numbers_of_y_a[Y_number]]), predicted_ytest_a]))
            plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                    [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
            plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
            plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
            plt.xlabel('Actual Y')
            plt.ylabel('Estimated Y')
            #plt.show()
            # r2p, RMSEp, MAEp
            print('r2p: {0}'.format(float(1 - sum((variables_test_a[:, numbers_of_y_a[Y_number]] - predicted_ytest_a) ** 2) / sum(
                (variables_test_a[:, numbers_of_y_a[Y_number]] - variables_train_a[:, numbers_of_y_a[Y_number]].mean()) ** 2))))
            print('RMSEp: {0}'.format(float((sum((variables_test_a[:, numbers_of_y_a[Y_number]] - predicted_ytest_a) ** 2) / len(
                variables_test_a[:, numbers_of_y_a[Y_number]])) ** 0.5)))
            print('MAEp: {0}'.format(float(sum(abs(variables_test_a[:, numbers_of_y_a[Y_number]] - predicted_ytest_a)) / len(
                variables_test_a[:, numbers_of_y_a[Y_number]]))))
            input_index = target_dfs_a[f'target_df_{i}'].index
            predicted_y_test_a.loc[input_index, y_names_a[Y_number]] = predicted_ytest_a

        # Check results of inverse analysis
        print('---------------------------')
        print('Results of inverse analysis')
        estimated_X_test_a = predicted_x_test_all_a * np.matlib.repmat(variables_train_a[:, numbers_of_X_a].std(ddof=1, axis=0), predicted_x_test_all_a.shape[0], 1) + \
                        np.matlib.repmat(variables_train_a[:, numbers_of_X_a].mean(axis=0), predicted_x_test_all_a.shape[0], 1)
        input_index = target_dfs_a[f'target_df_{i}'].index
        #estimated_x_test_a.loc[input_index, temp_col_a] = estimated_X_test_a[:, numbers_of_temp_a]
        estimated_x_test_a.loc[input_index, x_col_a] = estimated_X_test_a[:, :]

    
    # 各catの結果をresult_dfs_aに格納
    result_dfs_a[f'predicted_y_cat_{cat}'] = predicted_y_test_a
    result_dfs_a[f'estimated_x_cat_{cat}'] = estimated_x_test_a

# 全ての結果を保存（list_a解析結果）
for key, df in result_dfs_a.items():
    df.to_csv(f'result/autoencoder_data/{key}.csv', encoding='utf-8-sig')


print('End')