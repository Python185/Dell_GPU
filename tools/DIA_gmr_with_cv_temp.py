# -*- coding: utf-8 -*- %reset -f
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

# Settings
numbers_of_components = np.arange(2, 31, 2)
covariance_types = ['full', 'diag', 'tied', 'spherical']
#covariance_types = ['full', 'diag']
fold_number = 5
save_dir = 'result/autoencoder_data/'
number_of_test_samples = 30
random_state = 100

#number_of_all_samples = 100
#number_of_test_samples = 30
#numbers_of_X = [0, 1, 2]
#numbers_of_y = [3, 4]

# data入力
# 選択率NPA、収率NPAを帰るときは、44-45, 98-101, 416-417を修正する
x2_ce = pd.read_csv('datasets/v592/x2_ce.csv', encoding='utf-8-sig', index_col=0, header=0)
data = pd.concat([x2_ce.loc[:, :'収率NPA'], x2_ce.loc[:, '1':]], axis=1)

# 収率の目標は選択率の1/10とする
#data = data.drop('収率NPA', axis=1)
data = data.drop('選択率NPA', axis=1)

# AutoEncoderを用いてFT-IRデータからデータを抽出する
# GPUを使用するための設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")

preprocess = 'autoscaling' #original, diff, scaling, scaling_diff, autoscaling, autoscaling_diff

# 簡易的な実行時間の計測用
if device == 'cuda:0':
    torch.cuda.synchronize()
t1 = datetime.now()

number_of_epochs = 2000

batch_size = 32
lr = 0.005
kf = KFold(n_splits = 5)

"""
dae_structures = [[16]]

dae_structures = [[16],
                  [32],
                  [32, 16, 32],
                  [64, 32, 64],
                  [64, 32, 16, 32, 64],
                  [128, 64, 32, 64, 128],
                  [128, 64, 32, 16, 32, 64, 128],
                  [256, 128, 64, 32, 64, 128, 256]]
"""

dae_structures = [[8],
                  [16],
                  [32],
                  [16, 8, 16],
                  [32, 16, 32],
                  [64, 32, 64],
                  [32, 16, 8, 16, 32],
                  [64, 32, 16, 32, 64],
                  [32, 16, 8, 4, 8, 16, 32],
                  [64, 32, 16, 8, 4, 2, 4, 8, 16, 32, 64],
                  [128, 64, 32, 16, 8, 4, 2, 4, 8, 16, 32, 64, 128],
                  [64, 32, 16, 8, 16, 32, 64],
                  [128, 64, 32, 16, 32, 64, 128],
                  [256, 128, 64, 32, 64, 128, 256],
                  [512, 256, 128, 64, 32, 64, 128, 256, 512],
                  [1024, 512, 256, 128, 64, 32, 64, 128, 256, 512, 1024]]


#best_dae_structures = [32, 16, 32]

#x = data.copy().drop(['選択率NPA', '収率NPA'], axis=1)
#y = data.copy().loc[:, ['選択率NPA', '収率NPA']]
x = data.copy().drop('収率NPA', axis=1)
y = data.copy().loc[:, '収率NPA']

x_train, x_test = train_test_split(x, test_size=number_of_test_samples, shuffle=True, random_state=random_state)

autoscaled_x_train = (x_train - x_train.mean(axis = 0)) / x_train.std(axis = 0, ddof = 1)
autoscaled_x_test = (x_test - x_train.mean(axis = 0)) / x_train.std(axis = 0, ddof = 1)

print('説明変数の数 : {0}'.format(x_train.shape[1]))

x_train_arr = torch.tensor(autoscaled_x_train.values, dtype=torch.float32).to(device)
x_test_arr = torch.tensor(autoscaled_x_test.values, dtype=torch.float32).to(device)

mae_all = []

class Autoencoder(nn.Module):
    def __init__(self, input_dim, structure):
        super(Autoencoder, self).__init__()
        layers = []
        for i in range(len(structure)):
            if i == 0:
                layers.append(nn.Linear(input_dim, structure[i]))
            else:
                layers.append(nn.Linear(structure[i-1], structure[i]))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers[:len(layers)//2])
        self.decoder = nn.Sequential(*layers[len(layers)//2:])
        self.output_layer = nn.Linear(structure[-1], input_dim)
        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Tanh()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # return self.sigmoid(self.output_layer(decoded))
        return self.output_layer(decoded)

for dae_structure in dae_structures:
    print(dae_structure)
    mae_cv = []
    for train_index, val_index in kf.split(x_train_arr):
        x_train_fold, x_val_fold = x_train_arr[train_index], x_train_arr[val_index]

        train_dataset = TensorDataset(x_train_fold, x_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        
        model = Autoencoder(x_train_fold.shape[1], dae_structure).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        for epoch in range(number_of_epochs):
            model.train()
            for data, _ in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            # print(f'Epoch {epoch}, Loss: {loss.item()}')                

        model.eval()
        with torch.no_grad():
            decoded_data_val = model(x_val_fold)
            mae = mean_absolute_error(x_val_fold.cpu().numpy().flatten(), decoded_data_val.cpu().numpy().flatten())
            mae_cv.append(mae)
    mae_all.append(sum(mae_cv)/len(mae_cv))
    print(mae_all)

mae_all = pd.DataFrame(mae_all)
mae_all.to_csv(save_dir+'dae_results.csv', encoding='utf-8-sig')
best_dae_structures = dae_structures[mae_all.idxmin()[0]]

print(mae_all)
print("best_dae_structures")
print(best_dae_structures)

# 簡易的な実行時間の計測用
if device == 'cuda:0':
    torch.cuda.synchronize()
t2 = datetime.now()
print('time:', t2 - t1)


#最適化された構造でAutoencoderを構築
train_dataset = TensorDataset(x_train_arr, x_train_arr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
model = Autoencoder(x_train_arr.shape[1], best_dae_structures).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(number_of_epochs):
    model.train()
    for data, _ in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    losses.append(loss.item())
    
# 図の作成と保存
plt.figure()
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 24
plt.plot(range(1, number_of_epochs+1), losses)
plt.yscale('log')
plt.xlabel('number of epoch')
plt.ylabel('Loss (log scale)')
plt.savefig(save_dir+'loss.png', bbox_inches='tight')
# plt.show()
plt.close()


# Encode the data using the trained Autoencoder
model.eval()
with torch.no_grad():
    encoded_data_train = model.encoder(x_train_arr)
    encoded_data_test = model.encoder(x_test_arr)

# Convert encoded data to DataFrame
encoded_data_train_df = pd.DataFrame(encoded_data_train.cpu().numpy())
encoded_data_train_df.index = x_train.index
encoded_data_test_df = pd.DataFrame(encoded_data_test.cpu().numpy())
encoded_data_test_df.index = x_test.index

# Save encoded data
encoded_data_train_df.to_csv(save_dir+'encoded_data_train.csv', encoding='utf-8-sig')
encoded_data_test_df.to_csv(save_dir+'encoded_data_test.csv', encoding='utf-8-sig')


# train, testデータのdecode
model.eval()
with torch.no_grad():
    decoded_data_train = model.decoder(encoded_data_train)
    train_output = model.output_layer(decoded_data_train) 
    #decoded_data_test = model.decoder(encoded_data_test)

decoded_data_train = pd.DataFrame(train_output.cpu().numpy())
decoded_data_train.index = x_train.index
decoded_data_train.columns = x_train.columns
decoded_data_train = decoded_data_train * x_train.std(axis = 0, ddof = 1) + x_train.mean(axis = 0)

x_train.to_csv(save_dir+'x_train.csv')
decoded_data_train.to_csv(save_dir+'decoded_data_train.csv')

x_train.columns = x_train.columns.astype(float)
x_test.columns = x_test.columns.astype(float)
decoded_data_train.columns = decoded_data_train.columns.astype(float)

plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('original train data')
for i in range(x_train.shape[0]):
    plt.plot(x_train.columns, x_train.iloc[i, :].values)
#plt.xticks(np.arange(1700, 2301, 100))
plt.xlabel('Time, minutes')
plt.ylabel('Temperature, °C')
plt.savefig(save_dir+'original train data.png', bbox_inches='tight')
# plt.show()
plt.close()

plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('decoded train data')
for i in range(decoded_data_train.shape[0]):
    plt.plot(decoded_data_train.columns, decoded_data_train.iloc[i, :].values)
#plt.xticks(np.arange(1700, 2301, 100))
plt.xlabel('time, minutes')
plt.ylabel('Temperature, °C')
plt.savefig(save_dir+'decoded train data.png', bbox_inches='tight')
# plt.show()
plt.close()

x_train_stacked = x_train.stack().reset_index(drop=True)
decoded_x_train_stacked = decoded_data_train.stack().reset_index(drop=True)

y_max = max(np.max(x_train_stacked), np.max(decoded_x_train_stacked))
y_min = min(np.min(x_train_stacked), np.min(decoded_x_train_stacked))

plt.figure()
plt.rcParams['font.size'] = 20  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.figure(figsize=figure.figaspect(1))
plt.title('Train')
plt.scatter(x_train_stacked, decoded_x_train_stacked, c='blue', alpha=0.7, edgecolors='black')
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
          [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.savefig(save_dir+'yyplot_train.png', bbox_inches='tight')
# plt.show()
plt.close()
"""
# decoded_test_dataの内容確認
decoded_data_test = pd.DataFrame(decoded_data_test.cpu().numpy())
decoded_data_test.index = x_test.index
decoded_data_test.columns = x_test.columns
decoded_data_test = decoded_data_test * x_train.std(axis = 0, ddof = 1) + x_train.mean(axis = 0)

x_test.to_csv(save_dir+'x_test.csv')
decoded_data_test.to_csv(save_dir+'decoded_data_test.csv')

plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('original test data')
for i in range(x_test.shape[0]):
    plt.plot(x_test.columns, x_test.iloc[i, :].values)
plt.xticks(np.arange(1700, 2301, 100))
plt.xlabel('Wavenumber, cm-1')
plt.ylabel('Instensity')
plt.savefig(save_dir+'original test data.png', bbox_inches='tight')
# plt.show()
plt.close()

plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('decoded test data')
for i in range(decoded_data_test.shape[0]):
    plt.plot(decoded_data_test.columns, decoded_data_test.iloc[i, :].values)
plt.xticks(np.arange(1700, 2301, 100))
plt.xlabel('Wavenumber, cm-1')
plt.ylabel('Instensity')
plt.savefig(save_dir+'decoded test data.png', bbox_inches='tight')
# plt.show()
plt.close()

x_test_stacked = x_test.stack().reset_index(drop=True)
decoded_x_test_stacked = decoded_data_test.stack().reset_index(drop=True)

y_max = max(np.max(x_test_stacked), np.max(decoded_x_test_stacked))
y_min = min(np.min(x_test_stacked), np.min(decoded_x_test_stacked))

plt.figure()
plt.rcParams['font.size'] = 20  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.figure(figsize=figure.figaspect(1))
plt.title('Test')
plt.scatter(x_test_stacked, decoded_x_test_stacked, c='blue', alpha=0.7, edgecolors='black')
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
          [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.savefig(save_dir+'yyplot_test.png', bbox_inches='tight')
# plt.show()
plt.close()
"""


# Autoencoder結果の読取り
#encoded_temp_train = pd.read_csv(save_dir+'encoded_data_train.csv', encoding='utf-8-sig', index_col=0, header=0)
#encoded_temp_test = pd.read_csv(save_dir+'encoded_data_test.csv', encoding='utf-8-sig', index_col=0, header=0)
encoded_temp_train = encoded_data_train_df.copy()
encoded_temp_test = encoded_data_test_df.copy()
encoded_temp = pd.concat([encoded_temp_train, encoded_temp_test], axis=0)
encoded_temp.columns = encoded_temp.columns.astype(str)

# 結果を新しいdataとする
encoded_data = pd.concat([y, encoded_temp], axis=1)
#encoded_data.to_csv('SEM_data/20240709/datasets/ftir_sy4.csv', encoding='utf-8-sig')
#encoded_data = encoded_data.drop('触媒ロット', axis=1)


# 同じ値を多く持つ候補を削除
threshold_of_rate_of_same_value = 0.95
rate_of_same_value = list()
for X_variable_name in encoded_data.columns:
    same_value_number = encoded_data[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / encoded_data.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
encoded_data = encoded_data.drop(encoded_data.columns[deleting_variable_numbers], axis=1)

# dataのサイズを小さくしてdebug
#encoded_data = encoded_data.iloc[:60, :]

temp_col = [c for c in encoded_data.columns if 'NPA' not in c]
#ir_col = [c for c in irxrf_col if isinstance(c, (int,)) or (isinstance(c, str) and c.isdigit())]

# targetの設定
target_s = list(np.round(np.arange(0.1, 9.1, 0.1), 1))  #選択率 90分割
target_y = list(np.round(np.arange(0.01, 0.91, 0.01), 2))  #収率
target_df = pd.DataFrame(columns=['選択率NPA','収率NPA'])
target_df['選択率NPA'] = target_s
target_df['収率NPA'] = target_y
#target_df = target_df.drop('収率NPA', axis=1)
#target_df = target_df.drop('選択率NPA', axis=1)

x_names = [c for c in encoded_data.columns if 'NPA' not in c]
y_names = [c for c in encoded_data.columns if 'NPA' in c]

numbers_of_X = [encoded_data.columns.get_loc(c) for c in x_names]
numbers_of_y = [encoded_data.columns.get_loc(c) for c in y_names]
numbers_of_all_samples = encoded_data.shape[0]

predicted_y_test = pd.DataFrame(columns=['選択率NPA','収率NPA'], index=target_df.index)
estimated_x_test = pd.DataFrame(columns=temp_col, index=target_df.index)
target_dfs = {}

for i in range(3):
    print(i, '/3')
    target_dfs[f'target_df_{i}'] = target_df.iloc[i*30:(i+1)*30, :]
    #variables = np.c_[x, y1, y2]
    variables = encoded_data.values
    variables_train, variables_test = train_test_split(variables, test_size=number_of_test_samples, shuffle=True, random_state=random_state)

    # testにtarget挿入
    #variables_test[:len(target_dfs[f'target_df_{i}']), numbers_of_y] = target_dfs[f'target_df_{i}'].iloc[:, :2].values
    variables_test[:len(target_dfs[f'target_df_{i}']), numbers_of_y] = target_dfs[f'target_df_{i}'].iloc[:, :1].values

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
        plt.show()
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
    estimated_x_test.loc[input_index, temp_col] = estimated_X_test

#predicted_y_test.to_csv('SEM_data/20240709/datasets/y_sy_AE_ir.csv', encoding='utf-8-sig')
#estimated_x_test.to_csv('SEM_data/20240709/datasets/x_sy_AE_ir.csv', encoding='utf-8-sig')


# GMRで予測したtestデータのdecode
variables_train_df = pd.DataFrame(variables_train, columns=encoded_data.columns)
temp_train = variables_train_df.loc[:, temp_col]
temp_test = estimated_x_test.loc[:, temp_col]
autoscaled_temp_test = (temp_test - temp_train.mean(axis = 0)) / temp_train.std(axis = 0, ddof = 1)
array_data = autoscaled_temp_test.values.astype(np.float32)
temp_test_tensor = torch.tensor(array_data, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    decoded_data_test = model.decoder(temp_test_tensor)
    output = model.output_layer(decoded_data_test)

decoded_data_test = pd.DataFrame(output.cpu().numpy())
#decoded_data_test = pd.DataFrame(decoded_data_test.cpu().numpy())
#decoded_data_test.index = x_test.index
decoded_data_test.columns = x_test.columns
decoded_data_test = decoded_data_test * x_train.std(axis=0, ddof=1) + x_train.mean(axis=0)
#decoded_data_test = decoded_data_test * ftir_train.std(axis = 0, ddof = 1) + ftir_train.mean(axis = 0)

x_test.to_csv(save_dir+'x_test.csv')
decoded_data_test.to_csv(save_dir+'decoded_data_test.csv')

plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('original test data')
for name in x_test.index:
    plt.plot(x_test.columns.astype(float), x_test.loc[name], label=name)
#plt.xticks(np.arange(1700, 2301, 100))
plt.xlabel('Time, minutes')
plt.ylabel('Temperature, °C')
plt.savefig(save_dir+'original test data.png', bbox_inches='tight')
# plt.show()
plt.close()

plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('decoded test data')
for name in decoded_data_test.index:
    plt.plot(decoded_data_test.columns.astype(float), decoded_data_test.loc[name], label=name)
#plt.xticks(np.arange(1700, 2301, 100))
plt.xlabel('Time, minutes')
plt.ylabel('Temperature, °C')
plt.savefig(save_dir+'decoded test data.png', bbox_inches='tight')
# plt.show()
plt.close()




"""
x_test_stacked = x_test.stack().reset_index(drop=True)
decoded_x_test_stacked = decoded_data_test.stack().reset_index(drop=True)

y_max = max(np.max(x_test_stacked), np.max(decoded_x_test_stacked))
y_min = min(np.min(x_test_stacked), np.min(decoded_x_test_stacked))

plt.figure()
plt.rcParams['font.size'] = 20  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.figure(figsize=figure.figaspect(1))
plt.title('Test')
plt.scatter(x_test_stacked, decoded_x_test_stacked, c='blue', alpha=0.7, edgecolors='black')
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
          [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.savefig(save_dir+'yyplot_test.png', bbox_inches='tight')
# plt.show()
plt.close()
"""







print('End')