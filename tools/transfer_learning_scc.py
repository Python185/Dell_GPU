
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
import shutil
warnings.filterwarnings('ignore')

# 0は転移学習, 1はtargetデータのみ、2はsupport, targetデータ使用
# LOOのDCVとする。テストサンプル数=1なのでfor文で繰り返すだけ。
transfer_learning_flag = 0  # 0: transfer learning, 1: using only target data, 2: using both supporting data and target data
regression_methods = ['GPR_0','GPR_1','GPR_2','GPR_3','GPR_4','GPR_5',
                      'GPR_6','GPR_7','GPR_8','GPR_9','GPR_10','GPR_11','GPR_12']
#regression_methods = ['LGB','GPR_1']
#number_of_test_samples = 2  #target中のtest_sample量となる
outer_fold_number = 10 #外側Fold数

# x, yの設定
#y_names = ['選択率NPA', '収率NPA']
y_names = ['収率NPA']
x_names = ['x1_zr_am'] # x の設定

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

noise_ratio_in_simulation = 0.1
do_autoscaling = True  # True or False
threshold_of_rate_of_same_value = 0.99
fold_number = 5
max_pls_component_number = 30
ridge_lambdas = 2 ** np.arange(-5, 10, dtype=float)  # L2 weight in ridge regression
lasso_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # L1 weight in LASSO
elastic_net_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # Lambda in elastic net
elastic_net_alphas = np.arange(0.01, 1.00, 0.01, dtype=float)  # Alpha in elastic net
linear_svr_cs = 2 ** np.arange(-5, 5, dtype=float)  # C for linear svr
linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for linear svr
nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # C for nonlinear svr
nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for nonlinear svr
nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float)  # Gamma for nonlinear svr
random_forest_number_of_trees = 300  # Number of decision trees for random forest
random_forest_x_variables_rates = np.arange(1, 10, dtype=float) / 10  # Ratio of the number of X-variables for random forest

# 時間の計測
start_time = time.time()

for x_name in x_names:
    # load data set target: small_data, support: large_data
    drop_col = ['触媒ロット']
    #support_dict = {'x2_ti':'TiO2_SSP-M','x2_ce':'CeO2_HS','x2_zr':'ZrO2_RC100'}
    support_dict = {'x1_ti_am':'TiO2_SSP-M','x1_ce_am':'CeO2_HS','x1_zr_am':'ZrO2_RC100'}
    raw_data_with_y_supporting_1 = pd.read_csv('datasets/v572/x1_41.csv', encoding='utf-8-sig', index_col=0, header=0).drop(drop_col, axis=1)
    raw_data_with_y_target = pd.read_csv(f'datasets/v572/{x_name}.csv', encoding='utf-8-sig', index_col=0, header=0).drop(drop_col, axis=1)
    raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[raw_data_with_y_supporting_1['support_'+support_dict[x_name]] == 1.0]
    raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[raw_data_with_y_target.columns]  # x1はaveしか使用していないため
    
    # supportデータからtargetデータを除いておく
    raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[~raw_data_with_y_supporting_1.index.isin(raw_data_with_y_target.index)]

    # targetでデータ数-2以上同じ値の列はdrop LOOできないため
    threshold = raw_data_with_y_target.shape[0] - 2
    raw_data_with_y_target = raw_data_with_y_target.loc[:, raw_data_with_y_target.apply(lambda col:col.value_counts().max() <= threshold)]

    # supportは分散の小さい列を削除
    threshold_of_rate_of_same_value = 0.95
    rate_of_same_value = list()
    for X_variable_name in raw_data_with_y_supporting_1.columns:
        same_value_number = raw_data_with_y_supporting_1[X_variable_name].value_counts()
        rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / raw_data_with_y_supporting_1.shape[0]))
    deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
    raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1.drop(raw_data_with_y_supporting_1.columns[deleting_variable_numbers], axis=1)

    # 列の数が合わなくなるため合わせる
    if raw_data_with_y_target.shape[1] != raw_data_with_y_supporting_1.shape[1]:
        common_columns = [s for s in raw_data_with_y_supporting_1.columns if s in raw_data_with_y_target.columns]
        raw_data_with_y_target = raw_data_with_y_target[common_columns]    
        raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[common_columns]   
    
    for y_num, y_name in enumerate(y_names):
        if not os.path.exists(f'{savedir}/{y_name}/yyplot/tempo'):        
            os.mkdir(f'{savedir}/{y_name}/yyplot/tempo')                          

        #x, yの分離
        y_supporting_1 = raw_data_with_y_supporting_1.loc[:, y_name].copy()
        x_supporting_1 = raw_data_with_y_supporting_1.drop(['選択率NPA','収率NPA'], axis=1).copy()
        y_target = raw_data_with_y_target.loc[:, y_name].copy()
        x_target = raw_data_with_y_target.drop(['選択率NPA','収率NPA'], axis=1).copy()

        # r2, MAE, RMSE を保存
        DCV_values = np.zeros([len(regression_methods), 4])
        DCV_values = pd.DataFrame(DCV_values,index=regression_methods,columns=['r2_DCV', 'MAE_DCV', 'RMSE_DCV','W-RMSE_DCV'])

        # 予測値を保存
        predicted_y_values = np.zeros([len(y_target), len(regression_methods)])
        predicted_y_values = pd.DataFrame(predicted_y_values, index=raw_data_with_y_target.index, columns=regression_methods)
        
        # kFold設定
        #kf = KFold(n_splits=len(x_target), shuffle=True, random_state=48)
        kf = KFold(n_splits=outer_fold_number, shuffle=True, random_state=48)
        #x_train_target, x_test_target, y_train_target, y_test = train_test_split(x_target, y_target, test_size=number_of_test_samples, random_state=0)

        for method in regression_methods:
            print(method)
            estimated_y_in_outer_cv = pd.DataFrame(np.zeros(len(y_target)), index=raw_data_with_y_target.index, columns=[y_name])

            for train_idx, test_idx in kf.split(x_target):
                x_inner, x_outer = x_target.iloc[train_idx], x_target.iloc[test_idx]
                y_inner, y_outer = y_target.iloc[train_idx], y_target.iloc[test_idx]

                x_outer = x_outer.drop(x_outer.columns[np.where(x_inner.var()==0)],axis=1)
                x_inner = x_inner.drop(x_inner.columns[np.where(x_inner.var()==0)],axis=1)

                # autoscaling targetとsupportを合体させたものは、させた後にautoscaleすることとする(241126)
                if do_autoscaling:
                    autoscaled_x_train_target = (x_inner - x_inner.mean(axis=0)) / x_inner.std(axis=0, ddof=1)
                    autoscaled_x_supporting_1 = (x_supporting_1 - x_supporting_1.mean(axis=0)) / x_supporting_1.std(axis=0, ddof=1)
                    autoscaled_x_test_target = (x_outer - x_inner.mean(axis=0)) / x_inner.std(axis=0, ddof=1)
                    autoscaled_y_supporting_1 = (y_supporting_1 - y_supporting_1.mean()) / y_supporting_1.std(ddof=1)
                    autoscaled_y_train_target = (y_inner - y_inner.mean()) / y_inner.std(ddof=1)
                    x_train_support = pd.concat([x_supporting_1, x_inner], axis=0)
                    y_train_support = pd.concat([y_supporting_1, y_inner], axis=0)
                    autoscaled_x_train_support = (x_train_support - x_train_support.mean(axis=0)) / x_train_support.std(axis=0, ddof=1) 
                    autoscaled_y_train_support = (y_train_support - y_train_support.mean(axis=0)) / y_train_support.std(axis=0, ddof=1)                    
                else:
                    autoscaled_x_train_target = x_inner.copy()
                    autoscaled_x_supporting_1 = x_supporting_1.copy()
                    autoscaled_x_test_target = x_outer.copy()
                    autoscaled_y_supporting_1 = y_supporting_1.copy()
                    autoscaled_y_train_target = y_inner.copy()
        
                # testデータが小さく分割してautoscaleするとnanが発生するため、nan処理(241115)
                if autoscaled_x_train_target.isna().any(axis=0).any():
                    columns_to_keep = ~autoscaled_x_train_target.isna().any(axis=0)
                    autoscaled_x_train_target = autoscaled_x_train_target[:, columns_to_keep]
                    autoscaled_x_supporting_1 = autoscaled_x_supporting_1[:,columns_to_keep]
                    autoscaled_x_test_target = autoscaled_x_test_target[:,columns_to_keep]
                    
                # np.arrayに変換
                autoscaled_x_train_target = autoscaled_x_train_target.values
                autoscaled_y_train_target = autoscaled_y_train_target.values
                autoscaled_x_test_target = autoscaled_x_test_target.values
                autoscaled_x_supporting_1 = autoscaled_x_supporting_1.values
                autoscaled_y_supporting_1 = autoscaled_y_supporting_1.values
                autoscaled_x_train_support = autoscaled_x_train_support.values
                autoscaled_y_train_support = autoscaled_y_train_support.values

                if transfer_learning_flag == 1:
                    autoscaled_x_train = autoscaled_x_train_target.copy()
                    autoscaled_x_test = autoscaled_x_test_target.copy()
                    autoscaled_y_train = autoscaled_y_train_target.copy()
                elif transfer_learning_flag == 2:
                    autoscaled_x_train = np.r_[autoscaled_x_supporting_1, autoscaled_x_train_target]
                    autoscaled_x_test = autoscaled_x_test_target.copy()
                    autoscaled_y_train = np.r_[autoscaled_y_supporting_1, autoscaled_y_train_target]
                elif transfer_learning_flag == 0:
                    x_supporting_1_arranged = np.c_[autoscaled_x_supporting_1, autoscaled_x_supporting_1, np.zeros(autoscaled_x_supporting_1.shape), np.zeros(autoscaled_x_supporting_1.shape)]
                    x_train_target_arranged = np.c_[autoscaled_x_train_target, np.zeros(autoscaled_x_train_target.shape), np.zeros(autoscaled_x_train_target.shape), autoscaled_x_train_target]
                    autoscaled_x_train = np.r_[x_supporting_1_arranged, x_train_target_arranged]
                    autoscaled_x_train[:, :autoscaled_x_train_support.shape[1]] = autoscaled_x_train_support # 最左の部分だけautoscaled_x_train_supportに置換え
                    autoscaled_x_test = np.c_[autoscaled_x_test_target, np.zeros(autoscaled_x_test_target.shape), np.zeros(autoscaled_x_test_target.shape), autoscaled_x_test_target]
                    #autoscaled_y_train = np.r_[autoscaled_y_supporting_1, autoscaled_y_train_target]
                    autoscaled_y_train = autoscaled_y_train_support

                fold_number = min(fold_number, len(autoscaled_y_train))

                autoscaled_y_train = pd.Series(autoscaled_y_train)
                y_outer = pd.Series(y_outer)
                autoscaled_x_train = pd.DataFrame(autoscaled_x_train)
                autoscaled_x_test = pd.DataFrame(autoscaled_x_test)

                if method == 'PLS':  # Partial Least Squares
                    pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_x_train) + 1, max_pls_component_number + 1), 1)
                    r2all = list()
                    r2cvall = list()
                    for pls_component in pls_components:
                        pls_model_in_cv = PLSRegression(n_components=pls_component)
                        pls_model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
                        calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_x_train))
                        estimated_y_in_cv = np.ndarray.flatten(
                            model_selection.cross_val_predict(pls_model_in_cv, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
                
                        """
                        plt.figure(figsize=figure.figaspect(1))
                        plt.scatter( y, estimated_y_in_cv)
                        plt.xlabel("Actual Y")
                        plt.ylabel("Calculated Y")
                        plt.show()
                        """
                        r2all.append(float(1 - sum((autoscaled_y_train - calculated_y_in_cv) ** 2) / sum(autoscaled_y_train ** 2)))
                        r2cvall.append(float(1 - sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / sum(autoscaled_y_train ** 2)))
                    #plt.plot(pls_components, r2all, 'bo-')
                    #plt.plot(pls_components, r2cvall, 'ro-')
                    #plt.ylim(0, 1)
                    #plt.xlabel('Number of PLS components')
                    #plt.ylabel('r2(blue), r2cv(red)')
                    #plt.show()
                    optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
                    optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
                    regression_model = PLSRegression(n_components=optimal_pls_component_number)
                elif method == 'OLS':
                    regression_model = LinearRegression()
                elif method == 'RR':  # ridge regression
                    r2cvall = list()
                    for ridge_lambda in ridge_lambdas:
                        rr_model_in_cv = Ridge(alpha=ridge_lambda)
                        estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                            cv=fold_number)
                        r2cvall.append(float(1 - sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / sum(autoscaled_y_train ** 2)))
                    #plt.figure()
                    #plt.plot(ridge_lambdas, r2cvall, 'k', linewidth=2)
                    #plt.xscale('log')
                    #plt.xlabel('Weight for ridge regression')
                    #plt.ylabel('r2cv for ridge regression')
                    #plt.show()
                    optimal_ridge_lambda = ridge_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
                    regression_model = Ridge(alpha=optimal_ridge_lambda)
                elif method == 'LASSO':  # LASSO
                    r2cvall = list()
                    for lasso_lambda in lasso_lambdas:
                        lasso_model_in_cv = Lasso(alpha=lasso_lambda)
                        estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                            cv=fold_number)
                        r2cvall.append(float(1 - sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / sum(autoscaled_y_train ** 2)))
                    #plt.figure()
                    #plt.plot(lasso_lambdas, r2cvall, 'k', linewidth=2)
                    #plt.xlabel('Weight for LASSO')
                    #plt.ylabel('r2cv for LASSO')
                    #plt.show()
                    optimal_lasso_lambda = lasso_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
                    regression_model = Lasso(alpha=optimal_lasso_lambda)
                elif method == 'EN':  # Elastic net
                    elastic_net_in_cv = ElasticNetCV(cv=fold_number, l1_ratio=elastic_net_lambdas, alphas=elastic_net_alphas)
                    elastic_net_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
                    optimal_elastic_net_alpha = elastic_net_in_cv.alpha_
                    optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_
                    regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
                elif method == 'LSVR':  # Linear SVR
                    linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                                                    cv=fold_number)
                    linear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
                    optimal_linear_svr_c = linear_svr_in_cv.best_params_['C']
                    optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon']
                    regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)
                elif method == 'NLSVR':  # Nonlinear SVR
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
                    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma), {'epsilon': nonlinear_svr_epsilons},
                                            cv=fold_number, verbose=0)
                    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
                    optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']
                    # CV による C の最適化
                    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                                            {'C': nonlinear_svr_cs}, cv=fold_number, verbose=0)
                    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
                    optimal_nonlinear_c = model_in_cv.best_params_['C']
                    # CV による γ の最適化
                    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                                            {'gamma': nonlinear_svr_gammas}, cv=fold_number, verbose=0)
                    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
                    optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']
            #        nonlinear_svr_in_cv = GridSearchCV(svm.SVR(kernel='rbf', gamma=optimal_nonlinear_gamma),
            #                                           {'C': nonlinear_svr_cs, 'epsilon': nonlinear_svr_epsilons}, cv=fold_number)
            #        nonlinear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
            #        optimal_nonlinear_c = nonlinear_svr_in_cv.best_params_['C']
            #        optimal_nonlinear_epsilon = nonlinear_svr_in_cv.best_params_['epsilon']
                    regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                                            gamma=optimal_nonlinear_gamma)
                elif method == 'RF':  # Random forest
                    rmse_oob_all = list()
                    for random_forest_x_variables_rate in random_forest_x_variables_rates:
                        RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                            max(math.ceil(autoscaled_x_train.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
                        RandomForestResult.fit(autoscaled_x_train, autoscaled_y_train)
                        estimated_y_in_cv = RandomForestResult.oob_prediction_
                        if do_autoscaling:
                            estimated_y_in_cv = estimated_y_in_cv * autoscaled_y_train.std(ddof=1) + autoscaled_y_train.mean()
                        rmse_oob_all.append((sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / len(autoscaled_y_train)) ** 0.5)
                    #plt.figure()
                    #plt.plot(random_forest_x_variables_rates, rmse_oob_all, 'k', linewidth=2)
                    #plt.xlabel('Ratio of the number of X-variables')
                    #plt.ylabel('RMSE of OOB')
                    #plt.show()
                    optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
                        np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
                    regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                        max(math.ceil(autoscaled_x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True)
                elif method == 'GPR_0':
                    kernel = ConstantKernel() * DotProduct() + WhiteKernel()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_1':
                    kernel = ConstantKernel() * RBF() + WhiteKernel()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_2':
                    kernel = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_3':
                    kernel = ConstantKernel() * RBF(np.ones(autoscaled_x_train.shape[1])) + WhiteKernel()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                    #autoscaled_x_train = autoscaled_x_train.astype(np.float32)
                    #autoscaled_y_train = autoscaled_y_train.astype(np.float32)
                elif method == 'GPR_4':
                    kernel = ConstantKernel() * RBF(np.ones(autoscaled_x_train.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_5':
                    kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_6':
                    kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_7':
                    kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_8':
                    kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_9':
                    kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GRP モデルの宣言
                elif method == 'GPR_10':
                    kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_11':
                    kernel = ConstantKernel() * Matern(np.ones(autoscaled_x_train.shape[1]), nu=1.5) + WhiteKernel()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'GPR_12':
                    kernel = ConstantKernel() * Matern(np.ones(autoscaled_x_train.shape[1]), nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                elif method == 'LGB':  # LightGBM
                    regression_model = lgb.LGBMRegressor()
                elif method == 'XGB':  # XGBoost
                    regression_model = xgb.XGBRegressor()
                elif method == 'GBDT':  # scikit-learn
                    from sklearn.ensemble import GradientBoostingRegressor
                
                    regression_model = GradientBoostingRegressor()
                regression_model.fit(autoscaled_x_train, autoscaled_y_train)
                
                # 外側データの予測
                predicted_y_test = np.ndarray.flatten(regression_model.predict(autoscaled_x_test)) #　予測
                #predicted_y_test = predicted_y_test * y_inner.std(ddof=1) + y_inner.mean() #スケールを戻す
                predicted_y_test = predicted_y_test * y_train_support.std(ddof=1) + y_train_support.mean() #スケールを戻す
                estimated_y_in_outer_cv.iloc[test_idx] = predicted_y_test.reshape([-1, 1]) #　データを格納
    
            # 予測値の格納
            predicted_y_values[method] = estimated_y_in_outer_cv.values
    
            #　yyplot
            plt.rcParams['font.size'] = 18
            plt.figure(figsize=figure.figaspect(1)) # 正方形
            plt.title(f'{y_name} {method}_{x_name}') # タイトル
            plt.scatter(y_target, estimated_y_in_outer_cv, c='blue', alpha=0.7, edgecolors='black') # プロット
            y_max = np.max(np.array([y_target, estimated_y_in_outer_cv.values.flatten()])) # y 値の最大を取得
            y_min = np.min(np.array([y_target, estimated_y_in_outer_cv.values.flatten()])) # y 値の最小を取得
            if y_min <= 0:
                y_min = 0

            if y_name == '選択率NPA':
                plt.plot([-0.2, 10.0], [-0.2, 10.0], 'k-')
            else:
                plt.plot([-0.1, 1.0], [-0.1, 1.0], 'k-')
                plt.xticks([0,0.2,0.4,0.6,0.8,1.0])
            if y_name == '選択率NPA':
                plt.ylim(-0.2, 10.0) # Y のサイズ
                plt.xlim(-0.2, 10.0) # X のサイズ
            else:
                plt.ylim(-0.1, 1.0) # Y のサイズ
                plt.xlim(-0.1, 1.0) # X のサイズ
            plt.xlabel('Actual Y') #　縦軸ラベル
            plt.ylabel('Predicted Y in DCV') #　横軸ラベル
            plt.savefig(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{method}_TL.png',bbox_inches = 'tight') # 図の保存
            plt.show() # 図の描画

            # 評価指標の計算
            evaluation = np.zeros((4, 1))
            evaluation[0,0] = float(1 - sum((y_target - estimated_y_in_outer_cv.values.flatten()) ** 2) / sum((y_target - y_target.mean()) ** 2)) # R2 の計算
            evaluation[1,0] = float(sum(abs(y_target - estimated_y_in_outer_cv.values.flatten())) / len(y_target)) # MAE の計算
            evaluation[2,0] = float((sum((y_target - estimated_y_in_outer_cv.values.flatten()) ** 2) / len(y_target)) ** 0.5) # RMSE の計算
            evaluation[3,0] = float(np.sqrt(np.sum(y_target * np.square(y_target - estimated_y_in_outer_cv.values.flatten()))/ np.sum(y_target)))  #WRMSE の計算 # WRMSE の計算
            DCV_values.loc[method,:] = evaluation[:,0] # 評価指標を格納
            
        print()
        #図の出力
        save_fig = DCV_values.loc[:, 'r2_DCV'].idxmax()
        GP_rows = DCV_values.index[DCV_values.index.str.contains('GPR')]
        save_GPR_fig = DCV_values.loc[GP_rows, 'r2_DCV'].idxmax()
        shutil.move(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{save_fig}_TL.png', f'{savedir}/{y_name}/yyplot/fig_{x_name}_{save_fig}_TL.png')
        if 'GPR' not in save_fig:
            shutil.move(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{save_GPR_fig}_TL.png', f'{savedir}/{y_name}/yyplot/fig_{x_name}_{save_GPR_fig}_TL.png')
        shutil.rmtree(f'{savedir}/{y_name}/yyplot/tempo/')
        # 結果の保存
        y_target = y_target[y_target > 0.001]                        
        predicted_y_values = predicted_y_values.loc[predicted_y_values.index.isin(y_target.index.to_list())] #選択率、収率0のデータ削除
        predicted_y_values = pd.concat([y_target,predicted_y_values],axis=1) # Y の推定値を格納
        predicted_y_values.to_csv(f'{savedir}/{y_name}/predicted_y_in_DCV/{x_name}_y_values_TL.csv') # y の推定値を保存
        DCV_values.to_csv(f'{savedir}/{y_name}/prediction_accuracy/{x_name}_prediction_accuracy_TL.csv') # 評価指標を保存
        
# 計算時間の表示
elapsed_time = time.time() - start_time
print("Elapsed_time : {0}[sec]".format(elapsed_time))

print('end')
    
    
    
    
  