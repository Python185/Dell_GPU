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
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.linear_model import Ridge,Lasso,ElasticNet,ElasticNetCV
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
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# GPR_0–GPR_12 を GPyTorch（CUDA 利用可）で学習する
GPR_GPYTORCH_METHODS = frozenset(f'GPR_{i}' for i in range(0, 13))


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _gpr_gpytorch_train_predict(method, x_train_np, y_train_np, x_test_np, device):
    """オートスケール空間の訓練・テストで学習し、テスト予測（オートスケール）を numpy 1d で返す。"""
    n_dim = x_train_np.shape[1]
    train_x = torch.tensor(x_train_np, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train_np, dtype=torch.float32, device=device).reshape(-1)

    if method == 'GPR_0':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )
    elif method == 'GPR_1':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    elif method == 'GPR_2':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() +
            gpytorch.kernels.LinearKernel()
        )
    elif method == 'GPR_3':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_dim)
        )
    elif method == 'GPR_4':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_dim) +
            gpytorch.kernels.LinearKernel()
        )
        kernel.outputscale = 1.0
        kernel.base_kernel.kernels[0].lengthscale = torch.ones(n_dim, device=device) * 0.5
    elif method == 'GPR_5':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
    elif method == 'GPR_6':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5) +
            gpytorch.kernels.LinearKernel()
        )
    elif method == 'GPR_7':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5)
        )
    elif method == 'GPR_8':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5) +
            gpytorch.kernels.LinearKernel()
        )
    elif method == 'GPR_9':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
    elif method == 'GPR_10':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5) +
            gpytorch.kernels.LinearKernel()
        )
    elif method == 'GPR_11':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n_dim)
        )
    elif method == 'GPR_12':
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n_dim) +
            gpytorch.kernels.LinearKernel()
        )
    else:
        raise ValueError(f'Unsupported GPR method for GPyTorch: {method}')

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x, train_y, likelihood, kernel).to(device)
    model.train()
    likelihood.train()

    initial_lr = 0.1
    min_lr = 0.001
    lr = initial_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iterations = 500
    patience = 50
    min_delta = 1e-1
    best_loss = float('inf')
    epochs_no_improve = 0

    for i in range(training_iterations):
        try:
            with gpytorch.settings.cholesky_jitter(1e-1):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
                if loss.item() < best_loss - min_delta:
                    best_loss = loss.item()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
        except Exception as e:
            if lr > min_lr:
                lr /= 10
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                raise e

    test_x = torch.tensor(x_test_np, dtype=torch.float32, device=device)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-1):
        observed_pred = likelihood(model(test_x))
        pred = observed_pred.mean.detach().cpu().numpy().reshape(-1)
    return pred


def dcv_analysis():
    
    #ロジット変換、回帰条件の設定True
    log_transform = False    # False:対数変換なし　True:対数変換あり
    y_names = ['C3-OH selec (%)', 'C3-OH yield (%)']
    x_names = ['x2_as0']
    #regression_model_method_normals = ['OLS','PLS','RR','LASSO','NLSVR','DT','RF','GPR_0','GPR_1','GPR_2',
    #                                    'GPR_5','GPR_6','GPR_7','GPR_8','GPR_9','GPR_10','GBDT','XGB','LGB']
    regression_model_method_normals = ['GPR_3','GPR_4','GPR_11','GPR_12']
    regression_model_method_GMM = ['GMR','VBGMR']
    
    outer_fold_number = 2  # 外側の分割数
    random_state = 99  # 分割する際の乱数のシード。固定すれば再現性あり
    
    #各種定数、保存場所の設定
    # 保存ファイルの設定
    savedir = 'result/dcv_analysis_results'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for _y_name in y_names:
        if not os.path.exists(f'{savedir}/{_y_name}'):
            os.mkdir(f'{savedir}/{_y_name}')
            os.mkdir(f'{savedir}/{_y_name}/predicted_y_in_DCV')
            os.mkdir(f'{savedir}/{_y_name}/prediction_accuracy')
            os.mkdir(f'{savedir}/{_y_name}/yyplot')
        if not os.path.exists(f'{savedir}/{_y_name}/yyplot'):
            os.mkdir(f'{savedir}/{_y_name}/yyplot')
            os.mkdir(f'{savedir}/{_y_name}/yyplot/tempo')    
        if not os.path.exists(f'{savedir}/{_y_name}/yyplot/tempo'):
            os.mkdir(f'{savedir}/{_y_name}/yyplot/tempo')    
                
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
        data =  pd.read_csv(f'MyWork/datasets_CO2_CxOH/{x_name}.csv', index_col=0) # データの読み込み
        #data =  pd.read_csv(f'MyWork/datasets/predict_wo_sp_el2/{x_name}.csv', index_col=0) # データの読み込み
        x_raw = data.drop(y_names, axis= 1).copy()
        y_all = data[y_names].copy() # yの選択

        # 同じ値を多く持つ候補を削除
        threshold_of_rate_of_same_value = 0.9
        rate_of_same_value = list()
        for X_variable_name in x_raw.columns:
            same_value_number = x_raw[X_variable_name].value_counts()
            rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_raw.shape[0]))
        deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
        x_data = x_raw.drop(x_raw.columns[deleting_variable_numbers], axis=1)

        for method_type in [regression_model_method_normals, regression_model_method_GMM]:
            if method_type == regression_model_method_normals:
                for y_num, y_name in enumerate(y_names):
                    raw_y = y_all[y_name].copy()
                    log_y = raw_y.copy()
                    if log_transform:
                        log_y[log_y == 0] = (log_y[log_y != 0].nsmallest(1)/2).iloc[0]
                        log_y = logit((log_y/100).values)
                        log_y = pd.Series(log_y, index=y_all.index, name=y_name)

                    # r2, MAE, RMSE を保存
                    DCV_values = np.zeros([len(regression_model_method_normals), 4])
                    DCV_values = pd.DataFrame(DCV_values,index=regression_model_method_normals,columns=['r2_DCV', 'MAE_DCV', 'RMSE_DCV', 'WRMSE_DCV'])

                    # 予測値を保存
                    scc_mask = x_data['AIST'] == 0
                    aist_mask = x_data['AIST'] == 1
                    scc_xdata = x_data.loc[scc_mask, :].copy()
                    aist_xdata = x_data.loc[aist_mask, :].copy()
                    # インデックス重複があっても X/y の件数がずれないように行位置ベースで切り分ける
                    log_y_scc = log_y.loc[scc_mask].copy()
                    log_y_aist = log_y.loc[aist_mask].copy()
                    raw_y = raw_y.loc[scc_mask].copy()
                    predicted_y_values = np.zeros([len(log_y_scc), len(regression_model_method_normals)])
                    predicted_y_values = pd.DataFrame(predicted_y_values, index=log_y_scc.index, columns=regression_model_method_normals)

                    inner_fold_number = 5
                    for regression_model_method in method_type:
                        print(f'\r\t y:{y_name}  model:{regression_model_method}')
                        estimated_y_in_outer_cv = pd.DataFrame(np.zeros(len(log_y_scc)), index=log_y_scc.index, columns=[y_name])
                        # 住友化学データをN分割し、それにAISTデータを足してモデル作成し、住友化学データを予測する
                        min_number = math.floor(scc_xdata.shape[0] / outer_fold_number)
                        mod_number = scc_xdata.shape[0] - min_number * outer_fold_number
                        index = np.matlib.repmat(np.arange(1, outer_fold_number + 1, 1), 1, min_number).ravel()
                        if mod_number != 0:
                            index = np.r_[index, np.arange(1, mod_number + 1, 1)]
                        if random_state != None:
                            np.random.seed(random_state)
                        fold_index_in_outer_cv = np.random.permutation(index)
                        np.random.seed()

                        for fold_number_in_outer_cv in np.arange(1, outer_fold_number + 1, 1):
                            print(fold_number_in_outer_cv, '/', outer_fold_number)
                            # divide training data and test data
                            train_mask = fold_index_in_outer_cv != fold_number_in_outer_cv
                            test_mask = fold_index_in_outer_cv == fold_number_in_outer_cv
                            x_inner = scc_xdata.iloc[train_mask, :].copy()
                            x_inner = pd.concat([x_inner, aist_xdata], axis=0)
                            log_y_inner = log_y_scc.iloc[train_mask].copy()
                            log_y_inner = pd.concat([log_y_inner, log_y_aist], axis=0)
                            x_outer = scc_xdata.iloc[test_mask, :].copy()
                            
                            x_outer = x_outer.drop(x_outer.columns[np.where(x_inner.var()==0)],axis=1)
                            x_inner = x_inner.drop(x_inner.columns[np.where(x_inner.var()==0)],axis=1)

                            autoscaled_x_inner = (x_inner - x_inner.mean(axis=0)) / x_inner.std(axis=0, ddof=1)
                            autoscaled_log_y_inner = (log_y_inner - log_y_inner.mean()) / log_y_inner.std(ddof=1)
                            autoscaled_x_outer = (x_outer - x_inner.mean(axis=0)) / x_inner.std(axis=0, ddof=1)

                            # Light-GBM 用に array 型に変換
                            autoscaled_x_inner = autoscaled_x_inner.values
                            autoscaled_log_y_inner = autoscaled_log_y_inner.values
                            autoscaled_x_outer = autoscaled_x_outer.values

                            gp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                            if regression_model_method in GPR_GPYTORCH_METHODS:
                                try:
                                    pred_autoscale = _gpr_gpytorch_train_predict(
                                        regression_model_method,
                                        autoscaled_x_inner,
                                        np.asarray(autoscaled_log_y_inner, dtype=np.float64).ravel(),
                                        autoscaled_x_outer,
                                        gp_device,
                                    )
                                except Exception as e:
                                    print('GPR GPyTorch error:', regression_model_method, e)
                                    ytr = np.asarray(autoscaled_log_y_inner, dtype=np.float64).ravel()
                                    pred_autoscale = np.full(
                                        autoscaled_x_outer.shape[0],
                                        float(np.nanmean(ytr)),
                                    )
                            # OLS
                            elif regression_model_method == 'OLS':
                                regression_model = LinearRegression() # OLSモデルを宣言

                            # PLS
                            elif regression_model_method == 'PLS':
                                pls_components = np.arange(1,min(np.linalg.matrix_rank(autoscaled_x_inner)+1,max_pls_component_number+1),1)
                                r2all = list()
                                r2cvall = list()
                                for pls_component in pls_components:
                                    pls_model_in_cv = PLSRegression(n_components=pls_component)
                                    pls_model_in_cv.fit(autoscaled_x_inner, autoscaled_log_y_inner) # CVモデルを構築
                                    # calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_x_inner))
                                    estimated_y_in_cv = np.ndarray.flatten(
                                        model_selection.cross_val_predict(pls_model_in_cv, autoscaled_x_inner, autoscaled_log_y_inner, cv=inner_fold_number)) # CV予測値を計算
                                    #  calculated_y_in_cv = calculated_y_in_cv * y_train.iloc[:,0].std(ddof=1) + y_train.iloc[:,0].mean() # スケールを戻す
                                    estimated_y_in_cv = estimated_y_in_cv * log_y_inner.std(ddof=1) + log_y_inner.mean()

                                    # r2all.append(float(1 - sum((log_y_inner.iloc[:,0] - calculated_y_in_cv) ** 2) / sum((log_y_inner.iloc[:,0] - log_y_inner.iloc[:,0].mean()) ** 2))) # 2cvを計算
                                    r2cvall.append(float(1 - sum((log_y_inner - estimated_y_in_cv) ** 2) / sum((log_y_inner - log_y_inner.mean()) ** 2)))

                                optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall)) # 最適な成分数を取得
                                optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
                                regression_model = PLSRegression(n_components=optimal_pls_component_number) # 最適なPLSモデルを構築

                            # RR
                            elif regression_model_method == 'RR':
                                r2cvall = list()
                                for ridge_lambda in RR_lambdas:
                                    rr_model_in_cv = Ridge(alpha=ridge_lambda)
                                    estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_x_inner, autoscaled_log_y_inner,
                                                                                        cv=inner_fold_number) # CV モデルを構築
                                    estimated_y_in_cv = estimated_y_in_cv * log_y_inner.std(ddof=1) + log_y_inner.mean() # スケールを戻す
                                    r2cvall.append(float(1 - sum((log_y_inner - estimated_y_in_cv) ** 2) / sum((log_y_inner - log_y_inner.mean()) ** 2))) #r2cvを計算
                                optimal_ridge_lambda = RR_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]] # 最適な λ を取得
                                regression_model = Ridge(alpha=optimal_ridge_lambda) # 最適な RR モデルを構築

                            # LASSO
                            elif regression_model_method == 'LASSO':
                                r2cvall = list()
                                for lasso_lambda in LASSO_lambdas:
                                    lasso_model_in_cv = Lasso(alpha=lasso_lambda) #CV モデルを構築
                                    estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_x_inner, autoscaled_log_y_inner,
                                                                                        cv=inner_fold_number)
                                    estimated_y_in_cv = estimated_y_in_cv * log_y_inner.std(ddof=1) + log_y_inner.mean()
                                    r2cvall.append(float(1 - sum((log_y_inner - estimated_y_in_cv) ** 2) / sum((log_y_inner - log_y_inner.mean()) ** 2))) #r2cvを計算
                                optimal_lasso_lambda = LASSO_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]] # 最適な λ を取得
                                regression_model = Lasso(alpha=optimal_lasso_lambda) #最適な LASSO モデルを構築

                            # EN
                            elif regression_model_method == 'EN':
                                elastic_net_in_cv = ElasticNetCV(cv=inner_fold_number, l1_ratio=EN_lambdas, alphas=EN_alpha)
                                elastic_net_in_cv.fit(autoscaled_x_inner, autoscaled_log_y_inner) # CV モデルを構築
                                optimal_elastic_net_alpha = elastic_net_in_cv.alpha_ # 最適な α を取得
                                optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_ # 最適な λ を取得
                                regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)

                            # LSVR
                            elif regression_model_method == 'LSVR':
                                linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_c, 'epsilon': linear_svr_epsilon},
                                                                cv=inner_fold_number)
                                linear_svr_in_cv.fit(autoscaled_x_inner, autoscaled_log_y_inner) # CV モデルを構築
                                optimal_linear_svr_c = linear_svr_in_cv.best_params_['C'] # 最適な C を取得
                                optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon'] #最適なεを取得
                                regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon) # 最適な LSVR モデルを構築

                            # NLSVR
                            elif regression_model_method == 'NLSVR':
                                variance_of_gram_matrix = list()
                                numpy_autoscaled_Xtrain = np.array(autoscaled_x_inner)
                                for nonlinear_svr_gamma in nonlinear_svr_gammas:
                                    gram_matrix = np.exp(
                                        -nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(axis=2))
                                    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
                                optimal_nonlinear_gamma = nonlinear_svr_gammas[
                                    np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
                                # CV による ε の最適化
                                model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma), {'epsilon': nonlinear_svr_epsilons},
                                                            cv=inner_fold_number, verbose=0)
                                model_in_cv.fit(autoscaled_x_inner, autoscaled_log_y_inner)
                                optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon'] #最適な ε を取得
                                # CV による C の最適化
                                model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                                                            {'C': nonlinear_svr_cs}, cv=inner_fold_number, verbose=0)
                                model_in_cv.fit(autoscaled_x_inner, autoscaled_log_y_inner)
                                optimal_nonlinear_c = model_in_cv.best_params_['C'] #最適な C を取得
                                # CV による γ の最適化
                                model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                                                            {'gamma': nonlinear_svr_gammas}, cv=inner_fold_number, verbose=0)
                                model_in_cv.fit(autoscaled_x_inner, autoscaled_log_y_inner)
                                optimal_nonlinear_gamma = model_in_cv.best_params_['gamma'] #最適な γ を取得
                                regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                                                            gamma=optimal_nonlinear_gamma) #最適な NLSVR モデルを構築

                            # DT
                            elif regression_model_method == 'DT':
                                # クロスバリデーションによる木の深さの最適化
                                rmse_cv = []
                                max_depthes = []
                                for max_depth in range(1, max_max_depth):
                                    model_in_cv = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf) # CV モデルの構築
                                    estimated_y_in_cv = model_selection.cross_val_predict(model_in_cv,x_inner,log_y_inner,cv=inner_fold_number)
                                    rmse_cv.append((sum((log_y_inner - estimated_y_in_cv) ** 2) / len(log_y_inner)) ** 0.5) # r2cv を計算
                                    max_depthes.append(max_depth)
                                optimal_max_depth = max_depthes[rmse_cv.index(min(rmse_cv))] # 最適な木の深さを取得
                                regression_model = tree.DecisionTreeRegressor(max_depth=optimal_max_depth, min_samples_leaf=min_samples_leaf)  #最適なDTモデルの宣言

                            # RF
                            elif regression_model_method == 'RF':
                                # oob を用いてハイパーパラメータを最適化
                                rmse_oob_all = list()
                                for random_forest_x_variables_rate in random_forest_x_variables_rates:
                                    RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                                        max(math.ceil(x_inner.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
                                    RandomForestResult.fit(autoscaled_x_inner, autoscaled_log_y_inner)
                                    estimated_y_in_cv = RandomForestResult.oob_prediction_
                                    estimated_y_in_cv = estimated_y_in_cv * log_y_inner.std(ddof=1) + log_y_inner.mean() # スケールを戻す
                                    rmse_oob_all.append((sum((log_y_inner - estimated_y_in_cv) ** 2) / len(log_y_inner)) ** 0.5) # ハイパーパラメータの組み合わせごとに r2CV を計算
                                optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
                                    np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]] # 最適なハイパーパラメータを取得
                                regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                                    max(math.ceil(x_inner.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True) # 最適な RF モデルを構築

                            # GPR_0–GPR_12 は GPyTorch（GPU 利用可）で学習（GPR_GPYTORCH_METHODS）

                            # GBDT
                            elif regression_model_method == 'GBDT':
                                regression_model = GradientBoostingRegressor() # GBDT モデルの宣言

                            # XGB
                            elif regression_model_method == 'XGB':
                                regression_model = xgb.XGBRegressor() # XGB モデルの宣言

                            # LGB
                            elif regression_model_method == 'LGB':
                                regression_model = lgb.LGBMRegressor() # LGB モデルの宣言

                            elif regression_model_method == 'DNN_opt':
                                def objective(trial):
                                    param = {
                                        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(100,), (50, 10), (100, 10), (50, 50, 10), (100, 100, 10), (50, 50, 50, 10), (100, 100, 100, 10)]),
                #                        'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
                                        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-6, 1e-1),
                                        'alpha': trial.suggest_loguniform('alpha', 1e-7, 1e-2),
                                        'activation': trial.suggest_categorical('activation', ['tanh', 'relu'])
                                    }

                                    model = MLPRegressor(**param)
                                    estimated_y_in_cv = cross_val_predict(model, autoscaled_x_inner, autoscaled_log_y_inner, cv=inner_fold_number)
                                    estimated_y_in_cv = estimated_y_in_cv * log_y_inner.std() + log_y_inner.mean()
                #                    r2 = metrics.r2_score(log_y_inner, estimated_y_in_cv)
                #                    return 1.0 - r2
                                    rmse = metrics.mean_squared_error(log_y_inner, estimated_y_in_cv,squared=False)
                                    return rmse

                                study = optuna.create_study()
                                study.optimize(objective, n_trials=30)

                                regression_model = MLPRegressor(**study.best_params)
                                
                            # TPFN（CUDA 無し環境では CPU）
                            elif regression_model_method == 'TPFN':
                                regression_model = AutoTabPFNRegressor(
                                    device='cuda' if torch.cuda.is_available() else 'cpu'
                                )

                            if regression_model_method not in GPR_GPYTORCH_METHODS:
                                # モデル構築
                                regression_model.fit(autoscaled_x_inner, autoscaled_log_y_inner)
                                #　外側のデータの予測（オートスケール空間）
                                pred_autoscale = np.ndarray.flatten(regression_model.predict(autoscaled_x_outer))

                            predicted_log_y_test = pred_autoscale * log_y_inner.std(ddof=1) + log_y_inner.mean() #スケールを戻す
                            #estimated_y_in_outer_cv.iloc[test_sample_numbers] = predicted_log_y_test.reshape([-1, 1]) #　データを格納
                            estimated_y_in_outer_cv.iloc[test_mask, 0] = predicted_log_y_test  # 格納

                        #　予測値の格納
                        if log_transform:
                            estimated_y_in_outer_cv = expit(estimated_y_in_outer_cv)*100
                        predicted_y_values[regression_model_method] = estimated_y_in_outer_cv.values

                        #　yyplot
                        plt.rcParams['font.size'] = 18
                        plt.figure(figsize=figure.figaspect(1)) # 正方形
                        plt.title(f'{y_name} {regression_model_method} {x_name}') # タイトル
                        plt.scatter(raw_y, estimated_y_in_outer_cv, c='blue', alpha=0.7, edgecolors='black') # プロット
                        y_max = np.max(np.array([raw_y, estimated_y_in_outer_cv.values.flatten()])) # y 値の最大を取得
                        y_min = np.min(np.array([raw_y, estimated_y_in_outer_cv.values.flatten()])) # y 値の最小を取得
                        if y_min <= 0:
                            y_min = 0
                        #plt.plot([0, 50], [0, 50], 'k-')
                        #plt.ylim(0, 50) # 縦軸の範囲
                        #plt.xlim(0, 50)  
                        #(-0.2, 20.2)が元々のデフォルト値
                        plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                                [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') # 対角線の描画

                        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) # Y のサイズ
                        plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) # X のサイズ
                        
                        plt.xlabel('Actual Y') #　縦軸ラベル
                        plt.ylabel('Predicted Y in DCV') #　横軸ラベル
                        plt.savefig(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{regression_model_method}.png',bbox_inches = 'tight') # 図の保存
                        #plt.show() # 図の描画

                        # 評価指標の計算
                        evaluation = np.zeros((4, 1))
                        evaluation[0,0] = float(1 - sum((raw_y - estimated_y_in_outer_cv.values.flatten()) ** 2) / sum((raw_y - raw_y.mean()) ** 2)) # R2 の計算
                        evaluation[1,0] = float(sum(abs(raw_y - estimated_y_in_outer_cv.values.flatten())) / len(raw_y)) # MAE の計算
                        evaluation[2,0] = float((sum((raw_y - estimated_y_in_outer_cv.values.flatten()) ** 2) / len(raw_y)) ** 0.5) # RMSE の計算
                        evaluation[3,0] = float(np.sqrt(np.sum(raw_y * np.square(raw_y - estimated_y_in_outer_cv.values.flatten()))/ np.sum(raw_y)))  #WRMSE の計算
                        #x2_wrmse = np.sqrt(np.sum(x2_ce_a['選択率NPA'] * np.square((x2_ce_a['選択率NPA'] - x2_ce_a['RF']))) / np.sum(x2_ce_a['選択率NPA']))
                        DCV_values.loc[regression_model_method,:] = evaluation[:,0] # 評価指標を格納
                    print()
                    
                    #図の出力
                    save_fig = DCV_values.loc[:, 'r2_DCV'].idxmax()
                    save_fig2 = DCV_values.loc[:, 'WRMSE_DCV'].idxmin()
                    if save_fig == save_fig2:
                        shutil.move(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{save_fig}.png', f'{savedir}/{y_name}/yyplot/fig_{x_name}_{save_fig}_.png')
                    else:
                        shutil.move(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{save_fig}.png', f'{savedir}/{y_name}/yyplot/fig_{x_name}_{save_fig}_.png')
                        shutil.move(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{save_fig2}.png', f'{savedir}/{y_name}/yyplot/fig_{x_name}_{save_fig2}_.png')
                    # 結果の保存
                    predicted_y_values = pd.concat([raw_y,predicted_y_values],axis=1) # Y の推定値を格納
                    predicted_y_values.to_csv(f'{savedir}/{y_name}/predicted_y_in_DCV/{x_name}_y_values_0.csv') # y の推定値を保存
                    DCV_values.to_csv(f'{savedir}/{y_name}/prediction_accuracy/{x_name}_prediction_accuracy_0.csv') # 評価指標を保存
                    shutil.rmtree(f'{savedir}/{y_name}/yyplot/tempo/')
                    
            elif method_type == regression_model_method_GMM:
                break
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
                    predicted_y_values_df = pd.DataFrame(np.empty([y_GMM.shape[0], len(method_type)]))
                    predicted_y_values_dict[_y_name] = predicted_y_values_df
                    DCV_values_df = pd.DataFrame(np.empty([len(method_type), 3]), index=method_type,columns=['r2_DCV','MAE_DCV','RMSE_DCV'])
                    DCV_values_dict[_y_name] = DCV_values_df

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
                    #for fold_cat_outer_cv in set(cat_name):
                    #    x_outer = x_data[cat_name == fold_cat_outer_cv]
                    #    y_outer = y_GMM[cat_name == fold_cat_outer_cv]
                    #    x_inner = x_data[cat_name != fold_cat_outer_cv]
                    #    y_inner = y_GMM[cat_name != fold_cat_outer_cv]
                    for fold_number_in_outer_cv in np.arange(1, outer_fold_number + 1, 1):
                        x_outer = x_data.loc[fold_index_in_outer_cv == fold_number_in_outer_cv, :].copy() 
                        x_inner = x_data.loc[fold_index_in_outer_cv != fold_number_in_outer_cv, :].copy()
                        y_inner = y_GMM.loc[fold_index_in_outer_cv != fold_number_in_outer_cv].copy()
                       

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
                        #predicted_y_outer_all[cat_name == fold_cat_outer_cv] = predicted_y_outer_cv
                        predicted_y_outer_all[fold_index_in_outer_cv == fold_number_in_outer_cv] = predicted_y_outer_cv
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
                        plt.plot([-0.2, 20.2], [-0.2, 20.2], 'k-')
                        plt.ylim(-0.2, 20.2) # 縦軸の範囲
                        plt.xlim(-0.2, 20.2)                        
                        plt.xlabel('Actual Y') # 縦軸のラベル
                        plt.ylabel('Predicted Y in DCV') # 横軸のラベル
                        plt.savefig(f'{savedir}/{y_name}/yyplot/fig_{x_name}_{regression_model_method}.png', bbox_inches = 'tight') # 図の保存
                        #plt.show() # 図の描画

                        #r2, RMSE ,MAE
                        evaluation = np.zeros((3, 1))
                        evaluation[0 ,0] = float(1 - sum((y_raw[y_name] - predicted_y_test.flatten()) ** 2) / sum((y_raw[y_name] - y_raw[y_name].mean()) ** 2)) # r2
                        evaluation[1, 0] = float(sum(abs(y_raw[y_name] - predicted_y_test.flatten())) / len(y_raw[y_name])) # MAE
                        evaluation[2, 0] = float((sum((y_raw[y_name] - predicted_y_test.flatten()) ** 2) / len(y_raw[y_name])) ** 0.5) # RMSE

                        predicted_y_values_dict[y_name].loc[:, regression_model_method] = predicted_y_test
                        DCV_values_dict[y_name].loc[regression_model_method, :] = evaluation[:, 0]
                else:
                    print()
                for y_name in y_names:
                    predicted_y_values_dict[y_name].to_csv(f'{savedir}/{y_name}/predicted_y_in_DCV/{x_name}_{y_name}_y_values_2.csv')
                    DCV_values_dict[y_name].to_csv(f'{savedir}/{y_name}/prediction_accuracy/{x_name}_{y_name}_prediction_acuracy_2.csv')

    
    # 計算時間の表示
    elapsed_time = time.time() - start_time
    print("Elapsed_time : {0}[sec]".format(elapsed_time))

if __name__ == '__main__' :
    dcv_analysis()

