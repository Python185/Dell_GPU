# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:10:35 2022

@author: dcela
"""
import pandas as pd
import numpy as np
import math

from sklearn import model_selection, svm, metrics, tree
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.linear_model import Ridge,Lasso,ElasticNet,ElasticNetCV
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern

import optuna
import xgboost as xgb
import lightgbm as lgb
from tabpfn import TabPFNRegressor

from dcekit.generative_model import GMR
from dcekit.generative_model import VBGMR


# -- 定数 --

# 学習モデルパラメータ
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


# GMM学習モデルパラメータ
covariance_types = ['full', 'diag', 'tied', 'spherical']
numbers_of_components = np.arange(1, 31, 1) # for cvpfi 2023/1/16
#numbers_of_components = np.arange(2,22,1)
weight_concentration_prior_types = ['dirichlet_process', 'dirichlet_distribution']
weight_concentration_priors = 10 ** np.arange(-4, 2, 2, dtype=float)


# -----


#決定係数の計算
# actual: 実測値
# estimated: 予測値
def calcR2(actual: np.array, estimated: np.array):
    if actual.ndim != 1 or estimated.ndim != 1:
        raise Exception('No 1d input in calcR2')
    r2 = 1.0 - sum((actual - estimated) ** 2) / sum((actual - actual.mean()) ** 2)
    return r2

# actual: 実測値
# estimated: 予測値
def calcRMSE(actual: np.array, estimated: np.array):
    if actual.ndim != 1 or estimated.ndim != 1:
        raise Exception('No 1d input in calcRMSE')
    rmse = (sum((actual - estimated) ** 2) / len(actual)) ** 0.5
    return rmse

# actual: 実測値
# estimated: 予測値
def calcMAE(actual: np.array, estimated: np.array):
    if actual.ndim != 1 or estimated.ndim != 1:
        raise Exception('No 1d input in calcMAE')
    mae = sum(abs(actual - estimated) / len(actual))
    return mae

def maxNumber(values, metrics):
    if len(values) != len(metrics):
        raise Exception('different length')
    imax = np.argmax(metrics)
    return values[imax]
    

def combine(x: np.array, y: np.array):
    assert(y.ndim==1)
    y_new = y.reshape((len(y), 1))
    data = np.hstack((y_new, x))
    y_index = [0]
    x_index = list(range(1, x.shape[1] + 1))
    return data, x_index, y_index    

# -----

class StdScaler:
    def __init__(self, matrix):
        self.mean = float(matrix.mean())
        self.std = float(matrix.std(ddof=1))
            
    def scale(self, matrix):
        if isinstance(matrix, pd.DataFrame):
            matrix=matrix.to_numpy()
        out = (matrix - self.mean) / self.std
        return out

    def descale(self, matrix):
        if isinstance(matrix, pd.DataFrame):
            matrix=matrix.to_numpy()        
        out = matrix * self.std + self.mean
        return out
    
# ------

#モデル生成
# model_name: モデル名
# x: 説明変数
# y: 目的変数
# y_scaler: 目的変数のスケール, standarization=Falseの場合は未使用
# fold_num: クロスバリデーションの分割数（たとえば5の場合は分割した4つを学習用、1つを評価に使用する）
# standarization: trueのときx, yが標準化処理されている
def createModel(model_name: str, x: np.array, y: np.array, y_scaler: StdScaler, fold_num, standarization):
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy().flatten()
    
    if standarization:
        #標準化の逆変換を行うときはy_train_std, y_train_meanに値が入っていること
        #230830maki修正中
        scaler = y_scaler
        if y_scaler is None:
            raise Exception('Error  y_scaler is None')
    
    # OLS
    if model_name == 'OLS':
        return LinearRegression()

    # PLS
    elif model_name == 'PLS':
        pls_components = np.arange(1, min(np.linalg.matrix_rank(x) + 1, max_pls_component_number + 1), 1)
        r2cvall = []
        for pls_component in pls_components:
            pls_model_in_cv = PLSRegression(n_components=pls_component)
            pls_model_in_cv.fit(x, y)
            estimated_y = np.ndarray.flatten(model_selection.cross_val_predict(pls_model_in_cv, x, y, cv=fold_num)) # CV予測値を計算
            if standarization:
                estimated_y = scaler.descale(estimated_y)
            if y.ndim != 1:
                y = y.ravel()
            #print(y.shape, estimated_y.shape)
            r2cvall.append(calcR2(y, estimated_y))
        optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall)) # 最適な成分数を取得
        optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
        model = PLSRegression(n_components=optimal_pls_component_number) # 最適なPLSモデルを構築
        return model
    
    # RR
    elif model_name == 'RR':
        r2cvall = []
        for ridge_lambda in RR_lambdas:
            rr_model_in_cv = Ridge(alpha=ridge_lambda)
            estimated_y = model_selection.cross_val_predict(rr_model_in_cv, x, y, cv=fold_num)
            if standarization:
                estimated_y = scaler.descale(estimated_y)
            r2cvall.append(calcR2(y, estimated_y))
        optimal_ridge_lambda = RR_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]] # 最適な λ を取得
        model = Ridge(alpha=optimal_ridge_lambda) # 最適な RR モデルを構築
        return model
    
    # LASSO
    elif model_name == 'LASSO':
        r2cvall = []
        for lasso_lambda in LASSO_lambdas:
            lasso_model_in_cv = Lasso(alpha=lasso_lambda) #CV モデルを構築
            estimated_y = model_selection.cross_val_predict(lasso_model_in_cv, x, y, cv=fold_num)
            if standarization:
                estimated_y = scaler.descale(estimated_y)
                if y.ndim != 1:
                    y = y.ravel()
            #print(y.shape, estimated_y.shape)
            r2cvall.append(calcR2(y, estimated_y))    
        optimal_lasso_lambda = LASSO_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]] # 最適な λ を取得
        model = Lasso(alpha=optimal_lasso_lambda) #最適な LASSO モデルを構築
        return model
    
    # EN
    elif model_name == 'EN':
        elastic_net_in_cv = ElasticNetCV(cv=fold_num, l1_ratio=EN_lambdas, alphas=EN_alpha)
        elastic_net_in_cv.fit(x, y) # CV モデルを構築
        optimal_elastic_net_alpha = elastic_net_in_cv.alpha_ # 最適な α を取得
        optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_ # 最適な λ を取得
        model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
        return model
    
    # LSVR
    elif model_name == 'LSVR':
        linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_c, 'epsilon': linear_svr_epsilon}, cv=fold_num)
        linear_svr_in_cv.fit(x, y) # CV モデルを構築
        optimal_linear_svr_c = linear_svr_in_cv.best_params_['C'] # 最適な C を取得
        optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon'] #最適なεを取得
        model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon) # 最適な LSVR モデルを構築
        return model
    
    # NLSVR
    elif model_name == 'NLSVR':
        variance_of_gram_matrix = list()
        numpy_autoscaled_Xtrain = np.array(x)
        for nonlinear_svr_gamma in nonlinear_svr_gammas:
            gram_matrix = np.exp(-nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(axis=2))
            variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
        optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
        # CV による ε の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma), {'epsilon': nonlinear_svr_epsilons},
                                    cv=fold_num, verbose=0)
        model_in_cv.fit(x, y)
        optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon'] #最適な ε を取得
        # CV による C の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf',
                                           epsilon=optimal_nonlinear_epsilon,
                                           gamma=optimal_nonlinear_gamma),
                                           {'C': nonlinear_svr_cs},
                                           cv=fold_num,
                                           verbose=0)
        model_in_cv.fit(x, y)
        optimal_nonlinear_c = model_in_cv.best_params_['C'] #最適な C を取得
        # CV による γ の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf',
                                           epsilon=optimal_nonlinear_epsilon,
                                           C=optimal_nonlinear_c),
                                           {'gamma': nonlinear_svr_gammas},
                                           cv=fold_num,
                                           verbose=0)
        model_in_cv.fit(x, y)
        optimal_nonlinear_gamma = model_in_cv.best_params_['gamma'] #最適な γ を取得
        model = svm.SVR(kernel='rbf',
                        C=optimal_nonlinear_c,
                        epsilon=optimal_nonlinear_epsilon,
                        gamma=optimal_nonlinear_gamma) #最適な NLSVR モデルを構築

        return model
    
    # DT
    elif model_name == 'DT':
        # クロスバリデーションによる木の深さの最適化
        rmse_cv = []
        max_depthes = []
        for max_depth in range(1, max_max_depth):
            model_in_cv = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf) # CV モデルの構築
            estimated_y = model_selection.cross_val_predict(model_in_cv, x, y, cv=fold_num)
            if standarization:
                estimated_y = scaler.descale(estimated_y)
            rmse_cv.append(calcRMSE(y, estimated_y))
            max_depthes.append(max_depth)
        optimal_max_depth = max_depthes[rmse_cv.index(max(rmse_cv))] # 最適な木の深さを取得
        model = tree.DecisionTreeRegressor(max_depth=optimal_max_depth, min_samples_leaf=min_samples_leaf)  #最適なDTモデルの宣言
        return model

    # RF
    elif model_name == 'RF':
        # oob を用いてハイパーパラメータを最適化
        rmse_oob_all = []
        for random_forest_x_variables_rate in random_forest_x_variables_rates:
            RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                max(math.ceil(x.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
            RandomForestResult.fit(x, y)
            estimated_y = RandomForestResult.oob_prediction_
            if standarization:
                estimated_y = scaler.descale(estimated_y)
            if isinstance(y, pd.Series):
                y = y.values.flatten()
            if isinstance(y, pd.DataFrame):
                y = y.values.flatten()
            rmse_oob_all.append(calcRMSE(y, estimated_y))
        optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
            np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]] # 最適なハイパーパラメータを取得
        model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
            max(math.ceil(x.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True) # 最適な RF モデルを構築
        return model
    
    # GPR_0
    elif model_name == 'GPR_0':
        kernel = ConstantKernel() * DotProduct() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GRP_1
    elif model_name == 'GPR_1':
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GPR_2
    elif model_name == 'GPR_2':
        kernel = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GPR_3
    elif model_name == 'GPR_3':
        kernel = ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GPR_4
    elif model_name == 'GPR_4':
        kernel = ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GPR_5
    elif model_name == 'GPR_5':
        kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GPR_6
    elif model_name == 'GPR_6':
        kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model

    # GPR_7
    elif model_name == 'GPR_7':
        kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GPR_8
    elif model_name == 'GPR_8':
        kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GPR_9
    elif model_name == 'GPR_9':
        kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GRP モデルの宣言
        return model
    
    # GPR_10
    elif model_name == 'GPR_10':
        kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
        return model
    
    # GPR_11
    elif model_name == 'GPR_11':
        kernel = kernel = ConstantKernel() * Matern(np.ones(x.shape[1]), nu=1.5) + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0)
        return model
    
    # GPR_12    
    elif model_name == 'GPR_12':
        kernel = kernel = ConstantKernel() * Matern(np.ones(x.shape[1]), nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
        model = GaussianProcessRegressor(kernel=kernel, alpha=0)
        return model    
    
    # GBDT
    elif model_name == 'GBDT':
        model = GradientBoostingRegressor() # GBDT モデルの宣言
        return model
    
    # XGB
    elif model_name == 'XGB':
        model = xgb.XGBRegressor() # XGB モデルの宣言
        return model
    
    # LGB
    elif model_name == 'LGB':
        model = lgb.LGBMRegressor() # LGB モデルの宣言
        return model
    
    # TabPFN
    elif model_name == 'TPFN':
        model = TabPFNRegressor() # TabPFN モデルの宣言
        return model
    
    elif model_name == 'DNN_opt':
        def objective(trial):
            param = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(100,), (50, 10), (100, 10), (50, 50, 10), (100, 100, 10), (50, 50, 50, 10), (100, 100, 100, 10)]),
                'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-6, 1e-1),
                'alpha': trial.suggest_loguniform('alpha', 1e-7, 1e-2),
                'activation': trial.suggest_categorical('activation', ['tanh', 'relu'])
            }
            model = MLPRegressor(**param)
            estimated_y = cross_val_predict(model, x, y, cv=fold_num)
            if standarization:
                estimated_y = scaler.descale(estimated_y)
            rmse = metrics.mean_squared_error(y, estimated_y, squared=False)
            return rmse

        study = optuna.create_study()
        study.optimize(objective, n_trials=30)
        model = MLPRegressor(**study.best_params)
        return model

#モデル生成
# model_name: モデル名
# data: 説明変数 + 目的変数
# x_index: 説明変数のインデックスリスト
# y_index: 目的変数のインデックスリスト
# fold_num: クロスバリデーションの分割数（たとえば5の場合は分割した4つを学習用、1つを評価に使用する）
def createGMMModel(model_name: str, data: np.array, x_index: list, y_index: list, fold_num: int):
            
    # GMR のハイパーパラメータを最適化
    if model_name == 'GMR':
        model = GMR() # GMR モデルの宣言
        try:
            model.cv_bo(data, x_index, y_index, covariance_types, numbers_of_components, fold_num) # CV モデルを構築・最適化
        except ValueError:
            model.cv_opt(data, x_index, y_index, covariance_types, numbers_of_components, fold_num)
        return model

    # VBGMR のハイパーパラメータを最適化
    elif model_name == 'VBGMR':
        model = VBGMR() # VBGR モデルの宣言
        try:
            model.cv_bo(data, x_index, y_index, covariance_types, numbers_of_components,
                weight_concentration_prior_types, weight_concentration_priors, fold_num) # CV モデルを構築・最適化
        except ValueError:
            model.cv_opt(data, x_index, y_index, covariance_types, numbers_of_components,
                weight_concentration_prior_types, weight_concentration_priors, fold_num)
        return model
    
