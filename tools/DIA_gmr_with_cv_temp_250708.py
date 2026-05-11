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
number_of_test_samples = 10
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

# list_aに基づいた解析：1-30分までのデータを渡して残り時間を予測
time_1_half = data.loc[:, '1':'30'].copy().columns.tolist()
time_2_half = data.loc[:, '31':'60'].copy().columns.tolist()
x = data.copy().drop(['選択率NPA', '収率NPA', '触媒ロット'] + time_1_half, axis=1)
y = data.copy().loc[:, ['選択率NPA', '収率NPA'] + time_1_half]

# train_test_split
#x_train, x_test = train_test_split(x, test_size=number_of_test_samples, shuffle=True, random_state=random_state)
x_train = x[~x.index.isin(list_a)].copy()
x_test = x[x.index.isin(list_a)].copy()
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
for cat in x_test.index:
    # targetの設定, time_2_halfを含む
    target_s = list(np.round(np.arange(0.1, 10.1, 0.1), 1))  #選択率 90分割
    target_y = list(np.round(np.arange(0.01, 1.01, 0.01), 2))  #収率
    target_df = pd.DataFrame(columns=['選択率NPA','収率NPA'] + time_1_half, index=range(len(target_s)))
    target_df['選択率NPA'] = target_s
    target_df['収率NPA'] = target_y
    # target_dfのtime_1_half列に、dataの[cat, time_1_half]の値（1行分）を全行に代入
    for col in time_1_half:
        target_df[col] = data.loc[cat, col]

    y_names = y.columns.tolist()
    x_names = x.columns.tolist()
    y_x = pd.concat([y, x], axis=1)

    numbers_of_X = [y_x.columns.get_loc(c) for c in x_names]
    numbers_of_y = [y_x.columns.get_loc(c) for c in y_names]
    #numbers_of_temp = [y_x.columns.get_loc(c)-y.shape[1] for c in temp_col]

    predicted_y_test = pd.DataFrame(columns=['選択率NPA','収率NPA'] + time_1_half, index=target_df.index)
    #estimated_x_test = pd.DataFrame(columns=temp_col, index=target_df.index)
    estimated_x_test = pd.DataFrame(columns=x_col, index=target_df.index)

    target_dfs = {}
    result_dfs = {}

    for i in range(2):
        print(i+1, '/10')
        target_dfs[f'target_df_{i}'] = target_df.iloc[i*10:(i+1)*10, :]
        #variables = np.c_[x, y1, y2]
        variables = y_x.values
        variables_train, variables_test = train_test_split(variables, test_size=number_of_test_samples, shuffle=True, random_state=random_state)

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

    
    #predicted_y_test.to_csv('result/autoencoder_data/predicted_sel_yld.csv', encoding='utf-8-sig')
    #estimated_x_test.to_csv('result/autoencoder_data/predicted_temperature.csv', encoding='utf-8-sig')
    #estimated_x_test.to_csv('result/autoencoder_data/predicted_X_test.csv', encoding='utf-8-sig')

print('End')