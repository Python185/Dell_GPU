# インストール
import warnings
# warning の非表示
warnings.simplefilter('ignore')
import math
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna

from sklearn import model_selection, svm, metrics, tree
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.linear_model import Ridge,Lasso,ElasticNet,ElasticNetCV
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern

np.random.seed(1)
random.seed(1)

class Model:
    def __init__(self):
        pass

    def get_regression_model(self, best_model_name,autoscaled_x_train,autoscaled_log_y_train,inner_fold_number,log_y_train,x_train):
        
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

        # OLS
        if best_model_name == 'OLS':
            regression_model = LinearRegression()  # OLSモデルを宣言

        # PLS
        elif best_model_name == 'PLS':
            pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_x_train) + 1, max_pls_component_number + 1),
                                       1)
            r2all = list()
            r2cvall = list()
            for pls_component in pls_components:
                pls_model_in_cv = PLSRegression(n_components=pls_component)
                pls_model_in_cv.fit(autoscaled_x_train, autoscaled_log_y_train)  # CVモデルを構築
                # calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_x_train))
                estimated_y_in_cv = np.ndarray.flatten(
                    model_selection.cross_val_predict(pls_model_in_cv, autoscaled_x_train, autoscaled_log_y_train,
                                                      cv=inner_fold_number))  # CV予測値を計算
                #  calculated_y_in_cv = calculated_y_in_cv * log_y_train.iloc[:,0].std(ddof=1) + log_y_train.iloc[:,0].mean() # スケールを戻す
                estimated_y_in_cv = estimated_y_in_cv * log_y_train.std(ddof=1) + log_y_train.mean()
    
                # r2all.append(float(1 - sum((log_y_train.iloc[:,0] - calculated_y_in_cv) ** 2) / sum((log_y_train.iloc[:,0] - log_y_train.iloc[:,0].mean()) ** 2))) # 2cvを計算
                r2cvall.append(
                    float(1 - sum((log_y_train - estimated_y_in_cv) ** 2) / sum((log_y_train - log_y_train.mean()) ** 2)))
    
            optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))  # 最適な成分数を取得
            optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
            regression_model = PLSRegression(n_components=optimal_pls_component_number)  # 最適なPLSモデルを構築

        # RR
        elif best_model_name == 'RR':
            r2cvall = list()
            for ridge_lambda in RR_lambdas:
                rr_model_in_cv = Ridge(alpha=ridge_lambda)
                estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_x_train,
                                                                      autoscaled_log_y_train,
                                                                      cv=inner_fold_number)  # CV モデルを構築
                estimated_y_in_cv = estimated_y_in_cv * log_y_train.std(ddof=1) + log_y_train.mean()  # スケールを戻す
                r2cvall.append(float(1 - sum((log_y_train - estimated_y_in_cv) ** 2) / sum(
                    (log_y_train - log_y_train.mean()) ** 2)))  # r2cvを計算
            optimal_ridge_lambda = RR_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]  # 最適な λ を取得
            regression_model = Ridge(alpha=optimal_ridge_lambda)  # 最適な RR モデルを構築
    
        # LASSO
        elif best_model_name == 'LASSO':
            r2cvall = list()
            for lasso_lambda in LASSO_lambdas:
                lasso_model_in_cv = Lasso(alpha=lasso_lambda)  # CV モデルを構築
                estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_x_train,
                                                                      autoscaled_log_y_train,
                                                                      cv=inner_fold_number)
                estimated_y_in_cv = estimated_y_in_cv * log_y_train.std(ddof=1) + log_y_train.mean()
                r2cvall.append(float(1 - sum((log_y_train - estimated_y_in_cv) ** 2) / sum(
                    (log_y_train - log_y_train.mean()) ** 2)))  # r2cvを計算
            optimal_lasso_lambda = LASSO_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]  # 最適な λ を取得
            regression_model = Lasso(alpha=optimal_lasso_lambda)  # 最適な LASSO モデルを構築
    
        # EN
        elif best_model_name == 'EN':
            elastic_net_in_cv = ElasticNetCV(cv=inner_fold_number, l1_ratio=EN_lambdas, alphas=EN_alpha)
            elastic_net_in_cv.fit(autoscaled_x_train, autoscaled_log_y_train)  # CV モデルを構築
            optimal_elastic_net_alpha = elastic_net_in_cv.alpha_  # 最適な α を取得
            optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_  # 最適な λ を取得
            regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
    
        # LSVR
        elif best_model_name == 'LSVR':
            linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_c, 'epsilon': linear_svr_epsilon},
                                            cv=inner_fold_number)
            linear_svr_in_cv.fit(autoscaled_x_train, autoscaled_log_y_train)  # CV モデルを構築
            optimal_linear_svr_c = linear_svr_in_cv.best_params_['C']  # 最適な C を取得
            optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon']  # 最適なεを取得
            regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c,
                                       epsilon=optimal_linear_svr_epsilon)  # 最適な LSVR モデルを構築
    
        # NLSVR
        elif best_model_name == 'NLSVR':
            variance_of_gram_matrix = list()
            numpy_autoscaled_Xtrain = np.array(autoscaled_x_train)
            for nonlinear_svr_gamma in nonlinear_svr_gammas:
                gram_matrix = np.exp(
                    -nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(
                        axis=2))
                variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
            optimal_nonlinear_gamma = nonlinear_svr_gammas[
                np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
            # CV による ε の最適化
            model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma),
                                       {'epsilon': nonlinear_svr_epsilons},
                                       cv=inner_fold_number, verbose=0)
            model_in_cv.fit(autoscaled_x_train, autoscaled_log_y_train)
            optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']  # 最適な ε を取得
            # CV による C の最適化
            model_in_cv = GridSearchCV(
                svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                {'C': nonlinear_svr_cs}, cv=inner_fold_number, verbose=0)
            model_in_cv.fit(autoscaled_x_train, autoscaled_log_y_train)
            optimal_nonlinear_c = model_in_cv.best_params_['C']  # 最適な C を取得
            # CV による γ の最適化
            model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                                       {'gamma': nonlinear_svr_gammas}, cv=inner_fold_number, verbose=0)
            model_in_cv.fit(autoscaled_x_train, autoscaled_log_y_train)
            optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']  # 最適な γ を取得
            regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                                       gamma=optimal_nonlinear_gamma)  # 最適な NLSVR モデルを構築
    
        # DT
        elif best_model_name == 'DT':
            # クロスバリデーションによる木の深さの最適化
            rmse_cv = []
            max_depthes = []
            for max_depth in range(1, max_max_depth):
                model_in_cv = tree.DecisionTreeRegressor(max_depth=max_depth,
                                                         min_samples_leaf=min_samples_leaf)  # CV モデルの構築
                estimated_y_in_cv = model_selection.cross_val_predict(model_in_cv, x_train, log_y_train,
                                                                      cv=inner_fold_number)
                rmse_cv.append((sum((log_y_train - estimated_y_in_cv) ** 2) / len(log_y_train)) ** 0.5)  # r2cv を計算
                max_depthes.append(max_depth)
            optimal_max_depth = max_depthes[rmse_cv.index(max(rmse_cv))]  # 最適な木の深さを取得
            regression_model = tree.DecisionTreeRegressor(max_depth=optimal_max_depth,
                                                          min_samples_leaf=min_samples_leaf)  # 最適なDTモデルの宣言
    
        # RF
        elif best_model_name == 'RF':
            # oob を用いてハイパーパラメータを最適化
            rmse_oob_all = list()
            for random_forest_x_variables_rate in random_forest_x_variables_rates:
                RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                    max(math.ceil(x_train.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
                RandomForestResult.fit(autoscaled_x_train, autoscaled_log_y_train)
                estimated_y_in_cv = RandomForestResult.oob_prediction_
                estimated_y_in_cv = estimated_y_in_cv * log_y_train.std(ddof=1) + log_y_train.mean()  # スケールを戻す
                rmse_oob_all.append(
                    (sum((log_y_train - estimated_y_in_cv) ** 2) / len(log_y_train)) ** 0.5)  # ハイパーパラメータの組み合わせごとに r2CV を計算
            optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
                np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]  # 最適なハイパーパラメータを取得
            regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                max(math.ceil(x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)),
                                                     oob_score=True)  # 最適な RF モデルを構築
    
        # GPR_0
        elif best_model_name == 'GPR_0':
            kernel = ConstantKernel() * DotProduct() + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GRP_1
        elif best_model_name == 'GPR_1':
            kernel = ConstantKernel() * RBF() + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GPR_2
        elif best_model_name == 'GPR_2':
            kernel = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GPR_3
        elif best_model_name == 'GPR_3':
            kernel = ConstantKernel() * RBF(np.ones(x_train.shape[1])) + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GPR_4
        elif best_model_name == 'GPR_4':
            kernel = ConstantKernel() * RBF(np.ones(x_train.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GPR_5
        elif best_model_name == 'GPR_5':
            kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GPR_6
        elif best_model_name == 'GPR_6':
            kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GPR_7
        elif best_model_name == 'GPR_7':
            kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GPR_8
        elif best_model_name == 'GPR_8':
            kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GPR_9
        elif best_model_name == 'GPR_9':
            kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GRP モデルの宣言
    
        # GPR_10
        elif best_model_name == 'GPR_10':
            kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
    
        # GBDT
        elif best_model_name == 'GBDT':
            regression_model = GradientBoostingRegressor()  # GBDT モデルの宣言
    
        # XGB
        elif best_model_name == 'XGB':
            regression_model = xgb.XGBRegressor()  # XGB モデルの宣言
    
        # LGB
        elif best_model_name == 'LGB':
            regression_model = lgb.LGBMRegressor()  # LGB モデルの宣言
        
        elif best_model_name == 'DNN_opt':
            def objective(trial):
                param = {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes',
                                                                    [(100,), (50, 10), (100, 10), (50, 50, 10),
                                                                     (100, 100, 10), (50, 50, 50, 10),
                                                                     (100, 100, 100, 10)]),
                    #                        'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
                    'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-6, 1e-1),
                    'alpha': trial.suggest_loguniform('alpha', 1e-7, 1e-2),
                    'activation': trial.suggest_categorical('activation', ['tanh', 'relu'])
                }
    
                model = MLPRegressor(**param)
                estimated_y_in_cv = cross_val_predict(model, autoscaled_x_train, autoscaled_log_y_train,
                                                      cv=inner_fold_number)
                estimated_y_in_cv = estimated_y_in_cv * log_y_train.std() + log_y_train.mean()
                #                    r2 = metrics.r2_score(log_y_train, estimated_y_in_cv)
                #                    return 1.0 - r2
                rmse = metrics.mean_squared_error(log_y_train, estimated_y_in_cv, squared=False)
                return rmse
    
            study = optuna.create_study()
            study.optimize(objective, n_trials=30)
    
            regression_model = MLPRegressor(**study.best_params)
        return regression_model
