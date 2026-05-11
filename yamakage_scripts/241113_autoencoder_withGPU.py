import matplotlib.pyplot as plt
import matplotlib.figure as figure
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from sklearn.model_selection import KFold

# GPUを使用するための設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")

preprocess = 'autoscaling' #original, diff, scaling, scaling_diff, autoscaling, autoscaling_diff

# 簡易的な実行時間の計測用
if device == 'cuda:0':
    torch.cuda.synchronize()
t1 = datetime.now()

number_of_test_samples = 0.3
number_of_epochs = 2000
batch_size = 32
lr = 0.005
kf = KFold(n_splits = 5)
dae_structures = [[16],
                  [32],
                  [32, 16, 32],
                  [64, 32, 64],
                  [64, 32, 16, 32, 64],
                  [128, 64, 32, 64, 128],
                  [128, 64, 32, 16, 32, 64, 128],
                  [256, 128, 64, 32, 64, 128, 256]]

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


best_dae_structures = [128, 64, 32, 64, 128]

dataset = pd.read_csv('plus_xrd_0925_v2.csv', index_col=0)
xrd = dataset.loc[:, str(11.0):str(89.0)]
x_raw = xrd.dropna(how = 'all')

#XRDデータの前処理
if preprocess == 'original':
    x_preprocess = x_raw.copy()
elif preprocess == 'diff':
    x_preprocess = x_raw.diff(axis=1)
elif preprocess == 'scaling':
    x_raw_T = x_raw.T
    x_preprocess_T = x_raw_T / x_raw_T.std(axis = 0, ddof =1)
    x_preprocess = x_preprocess_T.T
elif preprocess == 'scaling_diff':
    x_raw_T = x_raw.T
    x_preprocess_T = x_raw_T / x_raw_T.std(axis = 0, ddof =1)
    x_preprocess = x_preprocess_T.T
    x_preprocess = x_preprocess.diff(axis=1)
elif preprocess == 'autoscaling':
    x_raw_T = x_raw.T
    x_preprocess_T = (x_raw_T - x_raw_T.mean(axis = 0)) / x_raw_T.std(axis = 0, ddof =1)
    x_preprocess = x_preprocess_T.T
elif preprocess == 'autoscaling_diff':
    x_raw_T = x_raw.T
    x_preprocess_T = (x_raw_T - x_raw_T.mean(axis = 0)) / x_raw_T.std(axis = 0, ddof =1)
    x_preprocess = x_preprocess.diff(axis=1)

x = xrd.copy()

x_train, x_test = train_test_split(x, test_size=number_of_test_samples, shuffle=True, random_state=100)

autoscaled_x_train = (x_train - x_train.mean(axis = 0)) / x_train.std(axis = 0, ddof = 1)
autoscaled_x_test = (x_test - x_train.mean(axis = 0)) / x_train.std(axis = 0, ddof = 1)

print('説明変数の数 : {0}'.format(x_train.shape[1]))

x_train_arr = torch.tensor(autoscaled_x_train.values, dtype=torch.float32).to(device)
x_test_arr = torch.tensor(autoscaled_x_test.values, dtype=torch.float32).to(device)

mae_all = []

# Autoencoder関数
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

# 最適daeの選定
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
best_dae_structures = dae_structures[mae_all.idxmin()[0]]

print(mae_all)
print("best_dae_structures")
print(best_dae_structures)

# 簡易的な実行時間の計測用
if device == 'cuda:0':
    torch.cuda.synchronize()
t2 = datetime.now()
print('time:', t2 - t1)


#最適化された構造でAEを構築
train_dataset = TensorDataset(x_train_arr, x_train_arr)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
model = Autoencoder(x_train_arr.shape[1], best_dae_structures).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# lossの変化を描画
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

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"Layer: {name} | Weights: {param.data}")
#         print(f"Layer: {name} | Gradients: {param.grad}")


plt.figure()
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 24
plt.plot(range(1, number_of_epochs+1), losses)
plt.yscale('log')
plt.xlabel('number of epoch')
plt.ylabel('Loss (log scale)')
plt.savefig('loss.png', bbox_inches='tight')
# plt.show()
plt.close()


#トレーニングデータ
model.eval()
with torch.no_grad():
    decoded_data_train = model(x_train_arr)
    decoded_data_test = model(x_test_arr)

decoded_data_train = pd.DataFrame(decoded_data_train.cpu().numpy())
decoded_data_train.index = x_train.index
decoded_data_train.columns = x_train.columns
decoded_data_train = decoded_data_train * x_train.std(axis = 0, ddof = 1) + x_train.mean(axis = 0)

x_train.to_csv('x_train.csv')
decoded_data_train.to_csv('decoded_data_train.csv')

# original_train_data描画
plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('original train data')
for i in range(x_train.shape[0]):
    plt.plot(range(x_train.shape[1]), x_train.iloc[i, :])
plt.xlabel('2theta')
plt.ylabel('Instensity')
plt.savefig('original train data.png', bbox_inches='tight')
# plt.show()
plt.close()

# decodeしたtrain_dataの描画
plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('decoded train data')
for i in range(decoded_data_train.shape[0]):
    plt.plot(range(decoded_data_train.shape[1]), decoded_data_train.iloc[i, :])
plt.xlabel('2theta')
plt.ylabel('Instensity')
plt.savefig('decoded train data.png', bbox_inches='tight')
# plt.show()
plt.close()

# train_dataのy-yプロット
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
plt.savefig('yyplot_train.png', bbox_inches='tight')
# plt.show()
plt.close()

# test_dataのy-yプロット
decoded_data_test = pd.DataFrame(decoded_data_test.cpu().numpy())
decoded_data_test.index = x_test.index
decoded_data_test.columns = x_test.columns
decoded_data_test = decoded_data_test * x_train.std(axis = 0, ddof = 1) + x_train.mean(axis = 0)

x_test.to_csv('x_test.csv')
decoded_data_test.to_csv('decoded_data_test.csv')

plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('original test data')
for i in range(x_test.shape[0]):
    plt.plot(range(x_test.shape[1]), x_test.iloc[i, :])
plt.xlabel('2theta')
plt.ylabel('Instensity')
plt.savefig('original test data.png', bbox_inches='tight')
# plt.show()
plt.close()

plt.figure()
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 24
plt.title('decoded test data')
for i in range(decoded_data_test.shape[0]):
    plt.plot(range(decoded_data_test.shape[1]), decoded_data_test.iloc[i, :])
plt.xlabel('2theta')
plt.ylabel('Instensity')
plt.savefig('decoded test data.png', bbox_inches='tight')
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
plt.savefig('yyplot_test.png', bbox_inches='tight')
# plt.show()
plt.close()

