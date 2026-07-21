# インストール
import warnings
# warning の非表示
warnings.simplefilter('ignore')
import time
import os
import math
import pandas as pd
import numpy as np
from scipy.special import logit, expit
from sklearn import model_selection, svm, metrics, tree
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from sklearn.linear_model import Ridge,Lasso,ElasticNet,ElasticNetCV
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern
import optuna
import xgboost as xgb
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
matplotlib.rc('font', family='Meiryo')
from dcekit.generative_model import GMR
from dcekit.generative_model import VBGMR
import shutil
import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def sanitize_filename(name):
    """ファイル名として安全な文字列に変換する関数"""
    return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

def dcv_analysis():
    
    #ロジット変換、回帰条件の設定True
    log_transform = False    # False:対数変換なし　True:対数変換あり
    #y_names = ['選択率NPA', '収率NPA']
    y_names = ['STY_ETANPA','NPAbyETA']
    #y_names = ['NPAbyETA']
    #収率予測時、そのまま予測するか、転化率を予測して収率に換算するか
    yield_predict_method = 'direct'  #direct, indeirect より選択
    #転化率を予測するとき、転化率から収率への変換時にNPA選択率としてrawと predictedのどちらを使用するか
    NPA_selectivity = 'predicted'  #raw, predicted より選択
    x_names = ['x1_mg_s'] # x の設定
    #regression_model_method_normals = ['OLS','PLS','RR','LASSO','EN','NLSVR','DT','RF','GPR_0','GPR_1','GPR_2','GPR_3','GPR_4',
    #                                    'GPR_5','GPR_6','GPR_7','GPR_8','GPR_9','GPR_10', 'GBDT','XGB','LGB', 'DNN_opt']
    #regression_model_method_normals = ['OLS','PLS','RR','LASSO','NLSVR','DT','RF','GPR_0','GPR_1','GPR_2','GPR_3','GPR_4',
    #                                   'GPR_5','GPR_6','GPR_7','GPR_8','GPR_9','GPR_10','GPR_11','GPR_12','GBDT','XGB','LGB']
    regression_model_method_normals = ['GPR_0']
    regression_model_method_GMM = ['GMR','VBGMR']
    
    outer_fold_number = 5  # 外側の分割数。触媒が同じで評価方法が異なるサンプルをひとまとまりとして分割
    random_state = 24  # 分割する際の乱数のシード。固定すれば再現性あり
    
    #各種定数、保存場所の設定
    # 保存ファイルの設定
    savedir = 'result/dcv_analysis_results'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for _y_name in y_names:
        _y_name_safe = sanitize_filename(_y_name)
        if not os.path.exists(f'{savedir}/{_y_name_safe}'):
            os.mkdir(f'{savedir}/{_y_name_safe}')
            os.mkdir(f'{savedir}/{_y_name_safe}/predicted_y_in_DCV')
            os.mkdir(f'{savedir}/{_y_name_safe}/prediction_accuracy')
            os.mkdir(f'{savedir}/{_y_name_safe}/yyplot')
        if not os.path.exists(f'{savedir}/{_y_name_safe}/yyplot'):
            os.mkdir(f'{savedir}/{_y_name_safe}/yyplot')
            os.mkdir(f'{savedir}/{_y_name_safe}/yyplot/tempo')    
        if not os.path.exists(f'{savedir}/{_y_name_safe}/yyplot/tempo'):
            os.mkdir(f'{savedir}/{_y_name_safe}/yyplot/tempo')        
    # 相関係数の設定
    threshold_of_r = 0.99
    
    max_pls_component_number = 5 # pls の主成分
    RR_lambdas = np.arange(0.01,0.71,0.01,dtype = float) # Ridge の λ
    LASSO_lambdas = np.arange(0.01,0.71,0.01,dtype = float) # LASSO の λ
    EN_alpha = np.arange(0,1.00,0.01,dtype = float) # Elasticnet の α
    EN_lambdas = np.arange(0.01,0.71,0.01,dtype = float) # Elasticnet の λ
    linear_svr_c = 2**np.arange(-5,7,dtype = float) # SVR （線形） の c
    linear_svr_epsilon = 2 ** np.arange(-7,0,dtype = float) # SVR （線形） の ε
    nonlinear_svr_cs = 2 ** np.arange(-5, 7, dtype=float)  # C for nonlinear svr
    nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for nonlinear svr
    nonlinear_svr_gammas = 2 ** np.arange(-10, 5, dtype=float)  # Gamma for nonlinear svr
    max_max_depth = 10  # 木の深さの上、の最大値
    min_samples_leaf = 2  # 葉ごとのサンプル数の最小値
    random_forest_number_of_trees = 300 # Random forest の木の数
    random_forest_x_variables_rates = np.arange(1, 10,dtype=float) / 10 # Random forest に使用する説明変数の割合
    covariance_types = ['full', 'diag', 'tied', 'spherical']
    numbers_of_components = np.arange(2,22,1)
    weight_concentration_prior_types = ['dirichlet_process', 'dirichlet_distribution']
    weight_concentration_priors = 10 ** np.arange(-4, 2, 2, dtype=float)
    
    # 時間の計測
    start_time = time.time()
    
    for x_name in x_names:
        print(f'select x:{x_name}')
        if 'x29' in x_name:
            data = pd.read_parquet(f'datasets/{x_name}.csv')
        else:
            data = pd.read_csv(f'datasets/v749/{x_name}.csv', index_col=0) # データの読み込み
        data = data.iloc[:20, :]
        #担体ごとにDCV解析する場合は下の行を使用し、allはコメントアウトする
        #tantai_name = [tan.replace('support_', '') for tan in data.columns if 'support' in tan and '*' not in tan]
        tantai_name = ['all']
        for tantai in tantai_name:
            drop_tantai_col = [tan for tan in data.columns if 'support' in tan and '*' not in tan]
            if yield_predict_method == 'direct':
                drop_list = y_names.copy()  # 目的変数のみを削除
                cat_name = data['触媒ロット'].copy()
                drop_list.append('触媒ロット')
            elif yield_predict_method == 'indirect':
                data['転化率'] = 100 * data['収率NPA'] / data['選択率NPA'] 
                drop_list = y_names.copy()  # 目的変数のみを削除
                if '収率NPA' in y_names:
                    # 収率NPAが目的変数の場合、転化率も追加
                    index_y_names = y_names.index('収率NPA')
                    drop_list[index_y_names] = '転化率'
                cat_name = data['触媒ロット'].copy()
                drop_list.append('触媒ロット')                

            if tantai == 'all':
                x_raw = data.drop(drop_list, axis= 1).copy()
                #cat_name = data['触媒ロット'].copy() # 触媒名の設定    
                if yield_predict_method == 'direct':
                    y_all = data[y_names].copy() # yの選択
                elif yield_predict_method == 'indirect':
                    index_y_names = y_names.index('収率NPA')
                    y_names[index_y_names] = '転化率'
                    y_all = data[y_names].copy()
            else:
                data_a = data[data['support_'+tantai]==1].copy()
                #data = data[data[tantai]==1].copy()
                x_raw = data_a.drop(drop_list, axis= 1).copy()
                #drop_tantai_colを足していないので担体列は含まれる しかしその後の同じ値を持つ列削除で落とされるので足す必要がない。
    #            x_raw = data.drop(columns=y_names+['触媒ロット']+drop_tantai_col).copy() # xの選択
                y_all = data_a[y_names].copy() # yの選択
                #cat_name = data_a['触媒ロット'].copy() # 触媒名の設定
  
            # 同じ値を多く持つ列(分散の小さい列)を削除
            threshold_of_rate_of_same_value = 0.95
            rate_of_same_value = list()
            for X_variable_name in x_raw.columns:
                same_value_number = x_raw[X_variable_name].value_counts()
                rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_raw.shape[0]))
            deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
    
            x_data = x_raw.drop(x_raw.columns[deleting_variable_numbers], axis=1)
    
            for method_type in [regression_model_method_normals, regression_model_method_GMM]:
                if method_type == regression_model_method_normals:
                    for y_num, y_name in enumerate(y_names):
                        y_name_safe = sanitize_filename(y_name)
                        if not os.path.exists(f'{savedir}/{y_name_safe}/yyplot/tempo'):        
                            os.mkdir(f'{savedir}/{y_name_safe}/yyplot/tempo')                          
                        raw_y = y_all[y_name].copy()
                        log_y = raw_y.copy()
                        if log_transform:
                            log_y[log_y == 0] = (log_y[log_y != 0].nsmallest(1)/2).iloc[0]
                            log_y = logit((log_y/100).values)
                            log_y = pd.Series(log_y, index=y_all.index, name=y_name)
    
    
                        # r2, MAE, RMSE を保存
                        DCV_values = np.zeros([len(regression_model_method_normals), 4])
                        DCV_values = pd.DataFrame(DCV_values,index=regression_model_method_normals,columns=['r2_DCV', 'MAE_DCV', 'RMSE_DCV', 'W-RMSE_DCV'])

                        # 予測値を保存
                        predicted_y_values = np.zeros([len(log_y), len(regression_model_method_normals)])
                        predicted_y_values = pd.DataFrame(predicted_y_values, index=log_y.index, columns=regression_model_method_normals)
    
                        # DCV の fold 数を設定
                        inner_fold_number = 5
    #                    outer_fold_number = len(set(cat_name))
                        #真木修正＠2212101
                        groups = np.array(cat_name)
                        cat_list = list(set(cat_name))   #このリストが毎回変わっているため結果が変わる
                        cat_list.sort()
    #                    unique_groups2 = np.array(list(set(cat_name)))
                        unique_groups2 = np.array(cat_list)                    
                        unique_groups= unique_groups2.copy()
                        #np.random.seed(random_state)
                        #np.random.shuffle(unique_groups)
                        #真木修正終わり
                        
                        kf = KFold(n_splits=outer_fold_number, shuffle=True, random_state=random_state)
                        #kf = KFold(n_splits=outer_fold_number, shuffle=False)  #前でshuffleしているためここではshuffleしていない
                        for regression_model_method in method_type:
                            print(f'\r\t担体:{tantai}  y:{y_name}  model:{regression_model_method}', end='')
                            estimated_y_in_outer_cv = pd.DataFrame(np.zeros(len(log_y)), index=log_y.index, columns=[y_name])
    
    #                        for fold_cat_outer_cv in set(cat_name):
                            for train_group_idx, test_group_idx in kf.split(unique_groups):
                                train_group_numbers, test_group_numbers = unique_groups[train_group_idx], unique_groups[test_group_idx]
                                train_sample_numbers = np.array([], dtype=np.int64)
                                for i in train_group_numbers:
                                    numbers = np.where(groups == i)[0]
                                    if len(numbers):
                                        train_sample_numbers = np.r_[train_sample_numbers, numbers]
                                test_sample_numbers = np.array([], dtype=np.int64)
                                for i in test_group_numbers:
                                    numbers = np.where(groups == i)[0]
                                    if len(numbers):
                                        test_sample_numbers = np.r_[test_sample_numbers, numbers]
                    
                                x_outer = x_data.iloc[test_sample_numbers, :]
                                log_y_outer = log_y.iloc[test_sample_numbers]
                                x_inner = x_data.iloc[train_sample_numbers, :]
                                log_y_inner = log_y.iloc[train_sample_numbers]
                                
    #                            x_outer = x_data.loc[cat_name == fold_cat_outer_cv, :]
    #                            log_y_outer = log_y[cat_name == fold_cat_outer_cv]
    #                            x_inner = x_data[cat_name != fold_cat_outer_cv]
    #                            log_y_inner = log_y[cat_name != fold_cat_outer_cv]
    
                                x_outer = x_outer.drop(x_outer.columns[np.where(x_inner.var()==0)],axis=1)
                                x_inner = x_inner.drop(x_inner.columns[np.where(x_inner.var()==0)],axis=1)
    
                                autoscaled_x_inner = (x_inner - x_inner.mean(axis=0)) / x_inner.std(axis=0, ddof=1)
                                autoscaled_log_y_inner = (log_y_inner - log_y_inner.mean()) / log_y_inner.std(ddof=1)
                                autoscaled_x_outer = (x_outer - x_inner.mean(axis=0)) / x_inner.std(axis=0, ddof=1)
    
                                # Light-GBM 用に array 型に変換
                                autoscaled_x_inner = autoscaled_x_inner.values
                                autoscaled_log_y_inner = autoscaled_log_y_inner.values
                                autoscaled_x_outer = autoscaled_x_outer.values
                                
                                autoscaled_x_inner = torch.tensor(autoscaled_x_inner, dtype=torch.float32)
                                autoscaled_log_y_inner = torch.tensor(autoscaled_log_y_inner, dtype=torch.float32)
                                autoscaled_x_outer = torch.tensor(autoscaled_x_outer, dtype=torch.float32)
    
                                # GPR_0
                                if regression_model_method == 'GPR_0':
                                    #kernel = ConstantKernel() * DotProduct() + WhiteKernel()
                                    #regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    kernel = gpytorch.kernels.ScaleKernel(
                                                gpytorch.kernels.LinearKernel()
                                            )                                    
    
                                # GRP_1
                                elif regression_model_method == 'GPR_1':
                                    # kernel = ConstantKernel() * RBF() + WhiteKernel()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                                gpytorch.kernels.RBFKernel()
                                            )
        
                                # GPR_2
                                elif regression_model_method == 'GPR_2':
                                    # kernel = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                                gpytorch.kernels.RBFKernel() +
                                                gpytorch.kernels.LinearKernel()
                                            )
        
                                # GPR_3
                                elif regression_model_method == 'GPR_3':
                                    # kernel = ConstantKernel() * RBF(np.ones(x_inner.shape[1])) + WhiteKernel()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                                gpytorch.kernels.RBFKernel(ard_num_dims=x_inner.shape[1])
                                            )
    
                                # GPR_4
                                elif regression_model_method == 'GPR_4':
                                    # kernel = ConstantKernel() * RBF(np.ones(x_inner.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                                gpytorch.kernels.RBFKernel(ard_num_dims=x_inner.shape[1]) +
                                                gpytorch.kernels.LinearKernel()
                                            )
                                    kernel.outputscale = 1.0
                                    kernel.base_kernel.kernels[0].lengthscale = torch.ones(x_inner.shape[1]) * 0.5
        
                                # GPR_5
                                elif regression_model_method == 'GPR_5':
                                    # kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.MaternKernel(nu=1.5)
                                    )
        
                                # GPR_6
                                elif regression_model_method == 'GPR_6':
                                    # kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.MaternKernel(nu=1.5) +
                                        gpytorch.kernels.LinearKernel()
                                    )
        
                                # GPR_7
                                elif regression_model_method == 'GPR_7':
                                    # kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.MaternKernel(nu=0.5)
                                    )
        
                                # GPR_8
                                elif regression_model_method == 'GPR_8':
                                    # kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.MaternKernel(nu=0.5) +
                                        gpytorch.kernels.LinearKernel()
                                    )
        
                                # GPR_9
                                elif regression_model_method == 'GPR_9':
                                    # kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GRP モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.MaternKernel(nu=2.5)
                                    )
        
                                # GPR_10
                                elif regression_model_method == 'GPR_10':
                                    # kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    # regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    # kernelの定義
                                    kernel = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.MaternKernel(nu=2.5) +
                                        gpytorch.kernels.LinearKernel()
                                    )

                                # GPR_11
                                elif regression_model_method == 'GPR_11':
                                    #kernel = ConstantKernel() * Matern(np.ones(x_inner.shape[1]), nu=1.5) + WhiteKernel()
                                    #regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    kernel = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=x_inner.shape[1])
                                    )
    
                                # GPR_12
                                elif regression_model_method == 'GPR_12':
                                    #kernel = ConstantKernel() * Matern(np.ones(x_inner.shape[1]), nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    #regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                    kernel = gpytorch.kernels.ScaleKernel(
                                        gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=x_inner.shape[1]) +
                                        gpytorch.kernels.LinearKernel()
                                    )
                                    
                                # モデル構築
                                try:
                                    # regression_model.fit(autoscaled_x_inner, autoscaled_log_y_inner)
                                    # GPyTorchモデルの定義
                                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                                    regression_model = ExactGPModel(autoscaled_x_inner, autoscaled_log_y_inner, likelihood,
                                                                    kernel)
                                    if regression_model == 'GPR_13':
                                        # カーネルのパラメータに制約を設ける
                                        regression_model.covar_module.base_kernel.kernels[0].raw_lengthscale.constraint = gpytorch.constraints.Interval(1e-4, 10)
                                        regression_model.covar_module.base_kernel.kernels[1].raw_variance.constraint = gpytorch.constraints.Interval(1e-4, 10)
                                        # カーネルのパラメータを手動で初期化
                                        regression_model.covar_module.base_kernel.kernels[0].lengthscale = torch.ones(autoscaled_x_inner.shape[1]) * 0.5
                                        regression_model.covar_module.base_kernel.kernels[1].variance = 1.0                                    
                                    
                                    # モデルを訓練モードに設定
                                    regression_model.train()
                                    likelihood.train()
                                    # 最適化の初期値設定
                                    initial_lr = 0.1
                                    min_lr = 0.001
                                    lr = initial_lr
                                    optimizer = torch.optim.Adam(regression_model.parameters(), lr=lr)
                                    # マージナル対数尤度を最大化するための損失関数
                                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, regression_model)
                                    # 訓練
                                    training_iterations = 500
                                    patience = 50 #早期停止のための許容エポック数
                                    min_delta = 1e-1 #改善とみなされる最小の変化
                                    best_loss = np.inf
                                    epochs_no_improve = 0

                                    for i in range(training_iterations):
                                        try:
                                            with gpytorch.settings.cholesky_jitter(1e-1):
                                                optimizer.zero_grad()
                                                output = regression_model(autoscaled_x_inner)
                                                loss = -mll(output, autoscaled_log_y_inner)
                                                loss.backward()
                                                optimizer.step()
                                                
                                                if loss.item() < best_loss - min_delta:
                                                    best_loss = loss.item()
                                                    epochs_no_improve = 0
                                                else:
                                                    epochs_no_improve += 1
                                                # デバッグメッセージの追加
                                                print(f"Best Loss: {best_loss}, Epoch_no_improve: {epochs_no_improve}")                                                   
    
                                            if epochs_no_improve >= patience:
                                                print(f"Early stopping at epoch {i}")
                                                break
                                                #raise Exception('Early stopping triggtered')
                                                
                                        except Exception as e:
                                            print(f"Error at epoch {i} with lr={lr}: {e}")
                                            # 学習率を小さくする
                                            if lr > min_lr:
                                                lr /= 10
                                                print(f"Reducing learning rate to {lr}")
                                                optimizer = torch.optim.Adam(regression_model.parameters(), lr=lr)
                                            else:
                                                raise e  # 学習率が最小値以下になった場合は例外を再発生させる        
                                            
                                    # 訓練後のモデルパラメータ確認
                                    #print("Trained Model Parameters:")
                                    #for name, param in regression_model.named_parameters():
                                    #    if param.requires_grad:
                                    #        print(f"{name}: {param.data}")
                           
                                except Exception as e:
                                    print('fit error in ', regression_model_method)
                                    print(e, i)
                                    #break
                                    #continue
                                        
                                # 検証データでの評価
                                regression_model.eval()
                                likelihood.eval()
                                # データの推定 推定側にもjitterを入れたらうまく動くようになった                                        
                                with torch.no_grad(), gpytorch.settings.fast_pred_var(),gpytorch.settings.cholesky_jitter(1e-1):
                                    observed_pred = likelihood(regression_model(autoscaled_x_outer))
                                    predicted_log_y_test = observed_pred.mean.numpy()
                                    
                                print('predicted_log_y_test', predicted_log_y_test)
                                
                                predicted_log_y_test = predicted_log_y_test * log_y_inner.std(ddof=1) + log_y_inner.mean() #スケールを戻す
                                
                                print('Scaled predicted_log_y_test', predicted_log_y_test)
                                
                                # estimated_y_in_outer_cv.loc[outer_fold_idx] = predicted_log_y_test.reshape([-1, 1]) #　データを格納　For LOO
                                test_sample_numbers_label = estimated_y_in_outer_cv.index[test_sample_numbers]
                                estimated_y_in_outer_cv.loc[test_sample_numbers_label, y_name] = predicted_log_y_test #　データを格納 For kFold 
                            
                            #　予測値の格納
                            if log_transform:
                                estimated_y_in_outer_cv = expit(estimated_y_in_outer_cv)*100
                            predicted_y_values[regression_model_method] = estimated_y_in_outer_cv.values
                            
                            # 転化率については、収率に換算する
                            if (y_name == '転化率' or y_name =='収率NPA') and yield_predict_method == 'indirect':
                                if NPA_selectivity == 'raw':
                                    predicted_y_values[regression_model_method] = predicted_y_values[regression_model_method] * data['選択率NPA'] / 100
                                elif NPA_selectivity == 'predicted':
                                    s_NPA_r2 = pd.read_csv(f'{savedir}/選択率NPA/prediction_accuracy/{x_name}_{tantai}_prediction_accuracy_1.csv',encoding= 'utf-8-sig' ,index_col= 0, header= 0) 
                                    s_NPA_predicted = pd.read_csv(f'{savedir}/選択率NPA/predicted_y_in_DCV/{x_name}_{tantai}_y_values_1.csv',encoding= 'utf-8-sig' ,index_col= 0, header= 0)
                                    best_GPR_model = s_NPA_r2['r2_DCV'].idxmax()
                                    predicted_NPA = s_NPA_predicted[best_GPR_model]                                    
                                    predicted_y_values[regression_model_method] = predicted_y_values[regression_model_method] * predicted_NPA / 100
                                if y_name == '転化率':
                                    raw_y = raw_y * data['選択率NPA'] / 100
                                    y_name = '収率NPA'
                                if y_name == '収率NPA':
                                    estimated_y_in_outer_cv = predicted_y_values[regression_model_method]

                            #　yyplot
                            plt.rcParams['font.size'] = 18
                            plt.figure(figsize=figure.figaspect(1)) # 正方形
                            plt.title(f'{y_name} {regression_model_method} {x_name}') # タイトル
                            plt.scatter(raw_y, estimated_y_in_outer_cv, c='blue', alpha=0.7, edgecolors='black') # プロット
                            y_max = np.max(np.array([raw_y, estimated_y_in_outer_cv.values.flatten()])) # y 値の最大を取得
                            y_min = np.min(np.array([raw_y, estimated_y_in_outer_cv.values.flatten()])) # y 値の最小を取得
                            if y_min <= 0:
                                y_min = 0
                            if y_name == '選択率NPA':
                                plt.plot([-0.2, 10], [-0.2, 10], 'k-') # 対角線の描画
                                plt.ylim(-0.2, 10) # Y のサイズ
                                plt.xlim(-0.2, 10) # X のサイズ    
                            elif y_name == '収率NPA':
                                plt.plot([-0.1, 1.5], [-0.1, 1.5], 'k-') # 対角線の描画
                                plt.ylim(-0.1, 1.5) # Y のサイズ
                                plt.xlim(-0.1, 1.5) # X のサイズ
                            elif y_name == 'STY_ETANPA':
                                plt.plot([-0.01, 0.2], [-0.01, 0.2], 'k-') # 対角線の描画
                                plt.ylim(-0.01, 0.2) # Y のサイズ
                                plt.xlim(-0.01, 0.2) # X のサイズ
                            elif y_name == 'NPAbyETA':
                                plt.plot([-0.05, 1.2], [-0.05, 1.2], 'k-') # 対角線の描画
                                plt.ylim(-0.05, 1.2) # Y のサイズ
                                plt.xlim(-0.05, 1.2) # X のサイズ                                                                

                            plt.xlabel('Actual Y') #　縦軸ラベル
                            plt.ylabel('Predicted Y in DCV') #　横軸ラベル
                            #plt.savefig(f'{savedir}/{y_name_safe}/yyplot/tempo/fig_{x_name}_{tantai}_{regression_model_method}.png',bbox_inches = 'tight') # 図の保存
                            #plt.show() # 図の描画
    
                            # 評価指標の計算
                            evaluation = np.zeros((4, 1))
                            evaluation[0,0] = float(1 - sum((raw_y - estimated_y_in_outer_cv.values.flatten()) ** 2) / sum((raw_y - raw_y.mean()) ** 2)) # R2 の計算
                            evaluation[1,0] = float(sum(abs(raw_y - estimated_y_in_outer_cv.values.flatten())) / len(raw_y)) # MAE の計算
                            evaluation[2,0] = float((sum((raw_y - estimated_y_in_outer_cv.values.flatten()) ** 2) / len(raw_y)) ** 0.5) # RMSE の計算
                            evaluation[3,0] = float(np.sqrt(np.sum(raw_y * np.square(raw_y - estimated_y_in_outer_cv.values.flatten()))/ np.sum(raw_y)))  #WRMSE の計算
                            DCV_values.loc[regression_model_method,:] = evaluation[:,0] # 評価指標を格納
                            
                        print()
                        #図の出力
                        save_fig = DCV_values.loc[:, 'r2_DCV'].idxmax()
                        GP_rows = DCV_values.index[DCV_values.index.str.contains('GPR')]
                        save_GPR_fig = DCV_values.loc[GP_rows, 'r2_DCV'].idxmax()
                        #shutil.move(f'{savedir}/{y_name_safe}/yyplot/tempo/fig_{x_name}_{tantai}_{save_fig}.png', f'{savedir}/{y_name_safe}/yyplot/fig_{x_name}_{tantai}_{save_fig}_gpu.png')
                        #if 'GPR' not in save_fig:
                        #    shutil.move(f'{savedir}/{y_name_safe}/yyplot/tempo/fig_{x_name}_{tantai}_{save_GPR_fig}.png', f'{savedir}/{y_name_safe}/yyplot/fig_{x_name}_{tantai}_{save_GPR_fig}_gpu.png')
                        #shutil.rmtree(f'{savedir}/{y_name_safe}/yyplot/tempo/')
                        # 結果の保存
                        predicted_y_values = pd.concat([raw_y,predicted_y_values],axis=1) # Y の推定値を格納
                        #predicted_y_values.to_csv(f'{savedir}/{y_name_safe}/predicted_y_in_DCV/{x_name}_{tantai}_y_values_gup.csv', encoding= 'utf-8-sig') # y の推定値を保存
                        #DCV_values.to_csv(f'{savedir}/{y_name_safe}/prediction_accuracy/{x_name}_{tantai}_prediction_accuracy_gpu.csv') # 評価指標を保存
    
                elif method_type == regression_model_method_GMM:
                    #break
                    pass
                    y_raw = y_all.copy()
                    y_GMM = y_all.copy()
                    if log_transform:
                        for i in range(y_all.shape[1]):
                            y_t = y_all.iloc[:, i].copy()
                            y_t[y_t == 0] = y_t[y_t != 0].nsmallest().iloc[0]/2
                            y_t = logit((y_t/100).values)
                            y_GMM.iloc[:, i] = y_t
    
                    predicted_y_values_dict = {}
                    DCV_values_dict = {}
                    for _y_name in y_names:
                        _y_name_safe = sanitize_filename(_y_name)
                        predicted_y_values_df = pd.DataFrame(np.empty([y_GMM.shape[0], len(method_type)]))
                        predicted_y_values_dict[_y_name_safe] = predicted_y_values_df
                        DCV_values_df = pd.DataFrame(np.empty([len(method_type), 3]), index=method_type,columns=['r2_DCV','MAE_DCV','RMSE_DCV'])
                        DCV_values_dict[_y_name_safe] = DCV_values_df
    
                    # データの整理
                    variables = pd.concat([y_GMM,x_data],axis=1)
    
                    # DCV 設定
                    numbers_of_X = list(range(len(y_GMM.columns),len(y_GMM.columns)+len(x_data.columns),1))
                    numbers_of_y = list(range(0,len(y_GMM.columns),1))
                    fold_number = 5
                                                                
                    # オートスケーリング
                    autoscaled_variables = (variables - variables.mean(axis=0)) / variables.std(axis=0, ddof=1)
                    autoscaled_variables = autoscaled_variables.values # Light-GBM 用に numpy に変更
    
                    for regression_model_method in method_type:
                        print(f'\r\tmodel:{regression_model_method}', end='')
                        predicted_y_outer_all = np.zeros([y_GMM.shape[0], len(y_names)])
    
                        for fold_cat_outer_cv in set(cat_name):
                            x_outer = x_data[cat_name == fold_cat_outer_cv]
                            y_outer = y_GMM[cat_name == fold_cat_outer_cv]
                            x_inner = x_data[cat_name != fold_cat_outer_cv]
                            y_inner = y_GMM[cat_name != fold_cat_outer_cv]
    
                            x_outer = x_outer.drop(x_outer.columns[np.where(x_inner.var()==0)],axis=1)
                            x_inner = x_inner.drop(x_inner.columns[np.where(x_inner.var()==0)],axis=1)
    
    #                       # bug fix（工藤さん修正）
                            num_y = len(numbers_of_y)
                            num_x = x_outer.shape[1]
                            numbers_of_X = range(num_y, num_x + num_y)
    
                            autoscaled_x_inner = (x_inner - x_inner.mean(axis=0)) / x_inner.std(axis=0, ddof=1)
                            autoscaled_y_inner = (y_inner - y_inner.mean()) / y_inner.std(ddof=1)
                            autoscaled_x_outer = (x_outer - x_inner.mean(axis=0)) / x_inner.std(axis=0, ddof=1)
    
                            # Light-GBM 用に array 型に変換
                            autoscaled_x_inner = autoscaled_x_inner.values
                            autoscaled_y_inner = autoscaled_y_inner.values
                            autoscaled_x_outer = autoscaled_x_outer.values
    
                            autoscaled_inner = np.concatenate([autoscaled_y_inner, autoscaled_x_inner], axis=1)
    
                            # GMR のハイパーパラメータを最適化
                            if regression_model_method == 'GMR':
                                model = GMR() # GMR モデルの宣言
                                try:
                                    model.cv_bo(autoscaled_inner, numbers_of_X, numbers_of_y, covariance_types, numbers_of_components, fold_number) # CV モデルを構築・最適化
                                except ValueError:
                                    model.cv_opt(autoscaled_inner, numbers_of_X, numbers_of_y, covariance_types, numbers_of_components, fold_number)
                                # print('max r2cv :', model.r2cv)
    
                            # VBGMR のハイパーパラメータを最適化
                            elif regression_model_method == 'VBGMR':
                                model = VBGMR() # VBGR モデルの宣言
                                try:
                                    model.cv_bo(autoscaled_inner, numbers_of_X, numbers_of_y, covariance_types, numbers_of_components,
                                        weight_concentration_prior_types, weight_concentration_priors, fold_number) # CV モデルを構築・最適化
                                except ValueError:
                                    model.cv_opt(autoscaled_inner, numbers_of_X, numbers_of_y, covariance_types, numbers_of_components,
                                        weight_concentration_prior_types, weight_concentration_priors, fold_number)
                                    # print('max r2cv :', model.r2cv)
    
                            model.fit(autoscaled_inner)
    
                            predicted_y_outer_cv = model.predict_rep(autoscaled_x_outer, numbers_of_X, numbers_of_y)
                            for i in range(len(y_names)):
                                predicted_y_outer_cv[:, i] = predicted_y_outer_cv[:, i] * y_GMM.iloc[:, i].std(axis=0, ddof=1) + y_GMM.iloc[:, i].mean(axis=0)
                            predicted_y_outer_all[cat_name == fold_cat_outer_cv] = predicted_y_outer_cv
                        if log_transform:
                            predicted_y_outer_all =  expit(predicted_y_outer_all)*100
    
    
                        plt.rcParams['font.size'] = 18
                        for y_num, y_name in enumerate(y_names):
                            predicted_y_test = np.ndarray.flatten(predicted_y_outer_all[:, y_num]) # y ごとに結果の確認
    
                            # yy-plot
                            plt.figure(figsize=figure.figaspect(1)) # 図の形を正方形
                            plt.title(f'{y_name} {regression_model_method} {x_name}') # タイトル
                            plt.scatter(y_raw[y_name], predicted_y_test, c='blue', alpha=0.7, edgecolors='black') # 図の描画
                            # y_max = np.max(np.array([np.array(Y_data.iloc[:, numbers_of_y[Y_number]]), predicted_y_test])) # y の最大値を取得
                            # y_min = np.min(np.array([np.array(Y_data.iloc[:, numbers_of_y[Y_number]]), predicted_y_test])) # y の最小値を取得
                            # y_max = y_all[y_name].max() # 全ての目的変数でスケールが揃うように設定
                            # y_min = y_all[y_name].min()
                            y_max = max(y_raw[y_name].max(), predicted_y_test.max())
                            y_min = max(y_raw[y_name].min(), predicted_y_test.min())
                            #plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                            #        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') # 対角線の描画
                            #plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) # 縦軸の範囲
                            #plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) # 横軸の範囲
                            #plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                            #        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') # 対角線の描画
                            if y_name == '選択率NPA':
                                plt.plot([-0.2, 6.2], [-0.2, 6.2], 'k-')
                            else:
                                plt.plot([-0.1, 1.2], [-0.1, 1.2], 'k-')
                            #plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) # Y のサイズ
                            #plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) # X のサイズ
                            if y_name == '選択率NPA':
                                plt.ylim(-0.2, 6.2) # Y のサイズ
                                plt.xlim(-0.2, 6.2) # X のサイズ
                            elif y_name == 'STY_ETANPA':
                                plt.ylim(-0.01, 0.2) # Y のサイズ
                                plt.xlim(-0.01, 0.2) # X のサイズ
                            elif y_name == 'NPAbyETA':
                                plt.ylim(-0.05, 1.2)                               
                                plt.xlim(-0.05, 1.2)
                            plt.xlabel('Actual Y') # 縦軸のラベル
                            plt.ylabel('Predicted Y in DCV') # 横軸のラベル
                            y_name_safe = sanitize_filename(y_name)
                            plt.savefig(f'{savedir}/{y_name_safe}/yyplot/fig_{x_name}_{y_name_safe}_{tantai}_{regression_model_method}.png', bbox_inches = 'tight') # 図の保存
                            #plt.show() # 図の描画
    
                            #r2, RMSE ,MAE
                            evaluation = np.zeros((3, 1))
                            evaluation[0 ,0] = float(1 - sum((y_raw[y_name] - predicted_y_test.flatten()) ** 2) / sum((y_raw[y_name] - y_raw[y_name].mean()) ** 2)) # r2
                            evaluation[1, 0] = float(sum(abs(y_raw[y_name] - predicted_y_test.flatten())) / len(y_raw[y_name])) # MAE
                            evaluation[2, 0] = float((sum((y_raw[y_name] - predicted_y_test.flatten()) ** 2) / len(y_raw[y_name])) ** 0.5) # RMSE
    
                            predicted_y_values_dict[y_name_safe].loc[:, regression_model_method] = predicted_y_test
                            DCV_values_dict[y_name_safe].loc[regression_model_method, :] = evaluation[:, 0]
                    else:
                        print()
                    for y_name in y_names:
                        y_name_safe = sanitize_filename(y_name)
                        predicted_y_values_dict[y_name_safe].to_csv(f'{savedir}/{y_name_safe}/predicted_y_in_DCV/{x_name}_{y_name_safe}_{tantai}_y_values_2.csv')
                        DCV_values_dict[y_name_safe].to_csv(f'{savedir}/{y_name_safe}/prediction_accuracy/{x_name}_{y_name_safe}_{tantai}_prediction_acuracy_2.csv')
    
    
    # 計算時間の表示
    elapsed_time = time.time() - start_time
    print("Elapsed_time : {0}[sec]".format(elapsed_time))

if __name__ == '__main__' :
    dcv_analysis()

