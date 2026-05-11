import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime


# GPUを使用するための設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")

# 簡易的な実行時間の計測用
if device == 'cuda:0':
    torch.cuda.synchronize()
t1 = datetime.now()

number_of_test_samples = 0.3
number_of_epochs = 1000
batch_size = 16

dae_structures = [[16],
                  [32],
                  [32, 16, 32],
                  [64, 32, 64],
                  [64, 32, 16, 32, 64],
                  [128, 64, 32, 64, 128],
                  [128, 64, 32, 16, 32, 64, 128],
                  [256, 128, 64, 32, 64, 128, 256]]

dataset = pd.read_csv('datasets/x2_Ce.csv', encoding='utf-8-sig', index_col=0, header=0)
xrd = dataset.loc[:, str(998.96):]

x = xrd.copy()
x_scaled = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

x_train, x_test = train_test_split(x_scaled, test_size=number_of_test_samples, shuffle=True, random_state=100)

print('説明変数の数 : {0}'.format(x_train.shape[1]))

x_train_arr = torch.tensor(x_train.values, dtype=torch.float32).to(device)
x_test_arr = torch.tensor(x_test.values, dtype=torch.float32).to(device)

train_dataset = TensorDataset(x_train_arr, x_train_arr)
test_dataset = TensorDataset(x_test_arr, x_test_arr)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.sigmoid(self.output_layer(decoded))

for dae_structure in dae_structures:
    model = Autoencoder(x_train_arr.shape[1], dae_structure).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(number_of_epochs):
        model.train()
        for data, _ in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        decoded_data_test = model(x_test_arr)
        mae = mean_absolute_error(x_test_arr.cpu().numpy().flatten(), decoded_data_test.cpu().numpy().flatten())
        print(mae)
        mae_all.append(mae)

# TensorをDataFrameに変換
decoded_df = pd.DataFrame(decoded_data_test.cpu().numpy())
decoded_df.index = x_test.index
decoded_df.columns = x_test.columns
decoded_df = decoded_df * x_train.std(ddof=1) + x_train.mean()
x_test = x_test * x_train.std(ddof=1) + x_train.mean()
decoded_df.to_csv("datasets/decoded_df_test.csv", encoding='utf-8-sig')
x_test.to_csv('datasets/x_test.csv', encoding='utf-8-sig')
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

