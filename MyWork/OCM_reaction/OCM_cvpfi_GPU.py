# -*- coding: utf-8 -*-
"""
@author: dcelab
GPU accelerated version of CVPFI (Cross-Validated Permutation Feature Importance considering correlation between features)
in GPR (Gaussian Process Regression) using PyTorch and GPyTorch
"""

import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import torch
import gpytorch
from tabpfn import TabPFNRegressor
from dcekit.variable_selection import cvpfi
from dcekit.variable_selection import cvpfi_gmr
from libs.create_model import createModel, createGMMModel
from sklearn.model_selection import train_test_split
from libs.create_model import StdScaler

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPyTorchGPR:
    """GPU accelerated GPR wrapper for CVPFI compatibility"""
    
    def __init__(self, kernel_type='rbf_ard', nu=1.5, random_state=None):
        self.kernel_type = kernel_type
        self.nu = nu
        self.random_state = random_state
        self.model = None
        self.likelihood = None
        self.is_fitted = False
        self.X_train = None
        self.y_train = None
        
    def _create_kernel(self, n_features):
        """Create kernel based on type"""
        if self.kernel_type == 'rbf_ard':
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_features)
            )
        elif self.kernel_type == 'matern_ard':
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=self.nu, ard_num_dims=n_features)
            )
        elif self.kernel_type == 'rbf_linear_ard':
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_features) +
                gpytorch.kernels.LinearKernel()
            )
        elif self.kernel_type == 'matern_linear_ard':
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=self.nu, ard_num_dims=n_features) +
                gpytorch.kernels.LinearKernel()
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return kernel
    
    def fit(self, X, y):
        """Fit the GPR model"""
        # Store original data for compatibility
        self.X_train = X
        self.y_train = y
        
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, 
                               dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, 
                               dtype=torch.float32, device=device).squeeze()
        
        # Create kernel
        kernel = self._create_kernel(X_tensor.shape[1])
        
        # Create likelihood and model
        # GPyTorchでは、WhiteKernelに相当するものはlikelihood.noiseで表現される
        # CPU版のWhiteKernelと同様に、likelihood.noiseを最適化可能にする
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        # WhiteKernelのデフォルトに対応する初期値を設定（最適化可能）
        self.likelihood.noise = 0.01  # WhiteKernelのデフォルトに対応
        self.model = ExactGPModel(X_tensor, y_tensor, self.likelihood, kernel).to(device)
        
        # Training
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Training loop with early stopping
        training_iterations = 500
        patience = 50
        min_delta = 1e-4
        best_loss = np.inf
        epochs_no_improve = 0
        
        for i in range(training_iterations):
            try:
                with gpytorch.settings.cholesky_jitter(1e-4):
                    optimizer.zero_grad()
                    output = self.model(X_tensor)
                    loss = -mll(output, y_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    if loss.item() < best_loss - min_delta:
                        best_loss = loss.item()
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    
                    if i % 50 == 0:  # 50回ごとに進捗を表示
                        print(f"Training iteration {i}, Loss: {loss.item():.6f}, Best: {best_loss:.6f}")
                    
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at iteration {i}")
                        break
                        
            except Exception as e:
                print(f"Training error at iteration {i}: {e}")
                if i > 50:  # ある程度学習が進んだ後にエラーが発生した場合
                    print("Training failed, using current model state")
                    break
                continue
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions - returns only mean for CVPFI compatibility"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, 
                               dtype=torch.float32, device=device)
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
            observed_pred = self.likelihood(self.model(X_tensor))
            mean = observed_pred.mean.cpu().numpy()
        
        return mean.reshape(-1, 1)
    
    def score(self, X, y):
        """Calculate R2 score for CVPFI compatibility"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        y_true = y.values if hasattr(y, 'values') else y
        return r2_score(y_true, y_pred)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'kernel_type': self.kernel_type,
            'nu': self.nu,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

# settings
n_repeats = 5  # the number of repetition J
alpha_r = 0.999  # alpha (the significance level) in the r (correlation) consideration, 1 means the correlations between features are not considered 
              
def main():
        
    # outout preparation
    dircsv = 'csv'
    dirplot = 'plot'
    createSaveDir(dircsv, dirplot)
    dfs = {}
    
    #モデル選択     目的変数: [説明変数ベクトル名、　モデル名]
    selection = {'Y(C2), %':     ['x2_train', 'TPFN']}
    
    #  CVPFI calc and data output
    for y_name, (x_name, model_name)  in selection.items():
        print(f'y:{y_name} x:{x_name} model:{model_name} start ')
        
        # dataset check
        x_train, y_train, autoscaled_x_train, autoscaled_y_train, fold_number, number_of_important_x, number_of_x = \
            checkDataset(y_name, x_name)
        
        # Model fit and CVPFI calculation
        importances_mean, importances_std, importances = \
            calcCVPFI(x_name, model_name, y_name, x_train, y_train, autoscaled_x_train, autoscaled_y_train, fold_number)    
    
        # plot
        plotImportances(dirplot, x_name, y_name, model_name, number_of_important_x, number_of_x, importances_mean)
        
        # csv
        dfs = outpotCsv(dircsv, dfs, x_name, y_name, model_name, x_train, importances_mean, importances_std)
        print(f'y:{y_name} x:{x_name} model:{model_name} end ')
    
    # summary
    excel_path = os.path.join("result", 'cvpfi_importance_summary_GPU.xlsx')
    try:
        with pd.ExcelWriter(excel_path) as writer:
            for name, df in dfs.items():
                df.to_excel(writer, sheet_name=name[:31])
    except Exception as e:
        print(f'Warning: Failed to save Excel to result/, saving to current directory instead. Error: {e}')
        excel_path = 'cvpfi_importance_summary_GPU.xlsx'
        with pd.ExcelWriter(excel_path) as writer:
            for name, df in dfs.items():
                df.to_excel(writer, sheet_name=name[:31])
    print(' summary_csv output end ')


def checkDataset(y_name, x_name):
    print('   checkDataset start ')    
    # read dataset
    data = pd.read_csv(f'MyWork/datasets/larger_data/data_for_article/{x_name}.csv', index_col= 0, header= 0, encoding='utf-8-sig') # データの読み込み
    #data = data.iloc[:200, :]  # for test
    if np.inf in data.values or np.nan in data.values:
        data = data.replace(np.inf, np.nan)    # infをnanに置換
        dropped_columns = data.columns[data.isnull().any()].tolist()   # 欠損値を含む列のリストを取得        
        data = data.dropna(axis=1, how='any')  # 欠損値を含む列を削除 

        print('inf or nan found and dropped ', dropped_columns)  
    
    # x
    x_train = data.iloc[:, 1:] # 特徴量を説明変数x_trainとする
    # y
    y_train_series = data[y_name] # 目的変数y_trainとする
    y_train = pd.DataFrame(y_train_series, columns=[y_name])
    
    # var()==0の場合はcolumnをdropする
    print('　　　 var()==0のcolumns ', x_train.columns[np.where(x_train.var()==0)])
    x_train = x_train.drop(x_train.columns[np.where(x_train.var()==0)],axis=1)
    number_of_x = x_train.shape[1]
    number_of_important_x = number_of_x
    
    # autoscale
    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
    
    # fold
    fold_number = x_train.shape[0]
    
    print('   checkDataset end ')   
    return x_train, y_train, autoscaled_x_train, autoscaled_y_train, fold_number, number_of_important_x, number_of_x 
    
def calcCVPFI(x_name, model_name, y_name, x_train, y_train, autoscaled_x_train, autoscaled_y_train, fold_number):
    print('   calcCVPFI start ')                       
    # Model fit and CVPFI calc
    if (model_name == 'GMR' or model_name == 'VBGMR') == True:
        
        data = np.concatenate([y_train, x_train], axis=1) # data: 説明変数 + 目的変数 
        x_dim = x_train.shape[1]   #説明変数の次元        
        y_dim = y_train.shape[1]   #目的変数の数        
        y_index = list(range(0, y_dim))             # y_index: 目的変数のインデックスリスト
        x_index = list(range(y_dim, x_dim + y_dim)) # x_index: 説明変数のインデックスリスト
        
        # Trained model  
        model = createGMMModel(model_name, data, x_index, y_index, fold_number)        
        # Model fit
        model.fit(data)
                    
        # CVPFI calculation
        importances_mean, importances_std, importances = cvpfi_gmr(
            model,
            data,
            x_index,
            y_index,
            fold_number=fold_number,
            scoring='r2',
            n_repeats=n_repeats,
            alpha_r=alpha_r,
            random_state=9,
        )

    else:      
        # GPU accelerated GPR for specific models
        if model_name in ['GPR_3', 'GPR_4', 'GPR_11', 'GPR_12']:
            print(f"   Using GPU accelerated GPR for {model_name}")
            # Create GPU accelerated GPR model with appropriate kernel
            if model_name == 'GPR_3':
                # RBF kernel with ARD
                regression_model = GPyTorchGPR(kernel_type='rbf_ard', nu=1.5, random_state=9)
            elif model_name == 'GPR_4':
                # RBF + Linear kernel with ARD
                regression_model = GPyTorchGPR(kernel_type='rbf_linear_ard', nu=1.5, random_state=9)
            elif model_name == 'GPR_11':
                # Matern kernel with ARD (nu=1.5)
                regression_model = GPyTorchGPR(kernel_type='matern_ard', nu=1.5, random_state=9)
            elif model_name == 'GPR_12':
                # Matern + Linear kernel with ARD (nu=1.5)
                regression_model = GPyTorchGPR(kernel_type='matern_linear_ard', nu=1.5, random_state=9)
        elif model_name == 'TPFN':
            # GPU accelerated TabPFN
            print(f"   Using GPU accelerated TabPFN for {model_name}")
            device_str = 'cuda' if device.type == 'cuda' else 'cpu'
            regression_model = TabPFNRegressor(device=device_str)
        else:
            # Fallback to original method for other models
            print(f"   Using original method for {model_name}")
            standarization=True
            y_scaler = StdScaler(y_train)
            regression_model = createModel(model_name, x_train, y_train, y_scaler, fold_number, standarization)
        
        # Model fit
        if model_name in ['GPR_3', 'GPR_4', 'GPR_11', 'GPR_12']:
            regression_model.fit(autoscaled_x_train, autoscaled_y_train)
        elif model_name == 'TPFN':
            regression_model.fit(autoscaled_x_train, autoscaled_y_train)
        else:
            # Ensure feature names are ASCII and do not contain special characters
            autoscaled_x_train.columns = [f"feature_{i}" for i in range(autoscaled_x_train.shape[1])]
            regression_model.fit(autoscaled_x_train, autoscaled_y_train)
            
        # CVPFI calculation
        importances_mean, importances_std, importances = cvpfi(
            regression_model,
            autoscaled_x_train,
            autoscaled_y_train,
            fold_number=fold_number,
            scoring='r2',
            n_repeats=n_repeats,
            alpha_r=alpha_r,
            random_state=9,
        )
    print('   calcCVPFI end ')
    return importances_mean, importances_std, importances
    
def plotImportances(dirplot, x_name, y_name, model_name, number_of_important_x, number_of_x, importances_mean):
    print('   plot output start ')
    plt.rcParams['font.size'] = 16 
    plt.title(f'{y_name}  {model_name}  {x_name}', fontsize = 18, fontname = 'MS Gothic') # タイトル
    plt.bar(range(1, number_of_important_x + 1), importances_mean[range(0, number_of_important_x)], color='b', width=1)
    plt.bar(range(number_of_important_x + 1, number_of_x + 1), importances_mean[range(number_of_important_x, number_of_x)], color='k', width=1)
    plt.xlabel('x')
    plt.xlabel('feature number')
    plt.ylabel('importance')
    try:
        plt.savefig(f'result/cvpfi/{dirplot}/graph_{y_name}_{x_name}_{model_name}.png', bbox_inches = 'tight') # 図の保存
    except Exception as e:
        print(f'   Warning: Failed to save plot to result/cvpfi/{dirplot}/, saving to current directory instead. Error: {e}')
        plt.savefig(f'graph_{y_name}_{x_name}_{model_name}.png', bbox_inches = 'tight')
    #plt.show()
    print('   plot output end ')
    
def outpotCsv(dircsv, dfs, x_name, y_name, model_name, x_train, importances_mean, importances_std):
    print('   csv output start ')
    name = f'({y_name})_{x_name}_{model_name}'
    report_path = os.path.join('result/cvpfi/'+dircsv, f'importance_of{name}.csv')
    try:
        df = save(report_path, x_train.columns, importances_mean, importances_std)
    except Exception as e:
        print(f'   Warning: Failed to save CSV to result/cvpfi/{dircsv}/, saving to current directory instead. Error: {e}')
        report_path = f'importance_of{name}.csv'
        df = save(report_path, x_train.columns, importances_mean, importances_std)
    name = f'{y_name} {x_name} {model_name}'
    dfs[name] = df
    print('   csv output end ')    
    return dfs
    
def save(filepath:str, features_name, importance_mean, importance_std):
    data = []
    for mean, std in zip(importance_mean, importance_std):
        data.append([mean, std])
    df = pd.DataFrame(data=data, index=features_name, columns=['mean', 'std'])
    df.to_csv(filepath, encoding='cp932')
    return df

def createSaveDir(dircsv, dirplot):
    savedir = 'result/cvpfi'
    # delete
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    # create
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    if not os.path.exists(savedir+'/'+dircsv):
        os.mkdir(savedir+'/'+dircsv)
    if not os.path.exists(savedir+'/'+dirplot):
        os.mkdir(savedir+'/'+dirplot)
            

if __name__ == '__main__' :
    main()
