# インストール
import warnings
# warning の非表示
warnings.simplefilter('ignore')
import time
import os
# 外側 joblib 並列時の BLAS/OpenMP ネスト並列を抑える（numpy/sklearn より前に設定）
for _thr_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_thr_var, '1')
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import shutil
#from sklearnex import patch_sklearn
#patch_sklearn


def _metal_element_columns(columns):
    """Al〜Zr の元素列名リスト（columns 内に存在するもの）"""
    col_list = list(columns)
    if 'Al' not in col_list or 'Zr' not in col_list:
        return []
    i_al, i_zr = col_list.index('Al'), col_list.index('Zr')
    if i_al > i_zr:
        return []
    return col_list[i_al:i_zr + 1]


def _scale_metal_block_x2(df_metal, block_mean, block_std, upper, rng):
    """
    Al-Zr ブロック: 非零のみで求めた mean/std を使用。
    非零は (x - mean) / std、0 は [0, upper) の乱数代入後に同じ mean/std でスケール。
    """
    arr = df_metal.values.astype(float).copy()
    scaled = np.empty_like(arr)
    nonzero_mask = arr != 0
    zero_mask = ~nonzero_mask

    scaled[nonzero_mask] = (arr[nonzero_mask] - block_mean) / block_std
    if upper > 0 and zero_mask.any():
        imputed = rng.uniform(0, upper, size=int(zero_mask.sum()))
        scaled[zero_mask] = (imputed - block_mean) / block_std
    else:
        scaled[zero_mask] = (arr[zero_mask] - block_mean) / block_std

    return pd.DataFrame(scaled, index=df_metal.index, columns=df_metal.columns)


def autoscale_x_for_dcv(x_inner, x_outer, use_x2_block, rng):
    """
    x のオートスケール。
    use_x2_block=True: Al-Zr 元素列は非零のみでブロック mean/std を算出し、
    非零をスケール、0 は [0, 非零最小値/2) の乱数代入後に同じ mean/std でスケール。
    非元素列は列ごとの通常オートスケール。
    """
    if not use_x2_block:
        x_mean = x_inner.mean(axis=0)
        x_std = x_inner.std(axis=0, ddof=1)
        autoscaled_x_inner = (x_inner - x_mean) / x_std
        autoscaled_x_outer = (x_outer - x_mean) / x_std
        return autoscaled_x_inner, autoscaled_x_outer, {
            'mode': 'column', 'mean': x_mean, 'std': x_std,
        }

    metal_cols = _metal_element_columns(x_inner.columns)
    other_cols = [c for c in x_inner.columns if c not in metal_cols]
    inner_work = x_inner.copy()
    outer_work = x_outer.copy()
    scale_info = {
        'mode': 'x2_block',
        'metal_cols': metal_cols,
        'other_cols': other_cols,
    }

    if metal_cols:
        flat_inner = inner_work[metal_cols].values.flatten()
        nonzero = flat_inner[flat_inner != 0]
        if len(nonzero) > 0:
            block_mean = float(np.mean(nonzero))
            block_std = float(np.std(nonzero, ddof=1))
            if block_std == 0:
                block_std = 1.0
            upper = float(np.min(nonzero) / 2)
        else:
            block_mean = 0.0
            block_std = 1.0
            upper = 0.0
        scale_info['zero_replace_upper'] = upper
        scale_info['block_mean'] = block_mean
        scale_info['block_std'] = block_std

        inner_work[metal_cols] = _scale_metal_block_x2(
            inner_work[metal_cols], block_mean, block_std, upper, rng)
        outer_work[metal_cols] = _scale_metal_block_x2(
            outer_work[metal_cols], block_mean, block_std, upper, rng)

    if other_cols:
        mean_other = x_inner[other_cols].mean(axis=0)
        std_other = x_inner[other_cols].std(axis=0, ddof=1)
        inner_work[other_cols] = (x_inner[other_cols] - mean_other) / std_other
        outer_work[other_cols] = (x_outer[other_cols] - mean_other) / std_other
        scale_info['other_mean'] = mean_other
        scale_info['other_std'] = std_other

    return inner_work, outer_work, scale_info


def _resolve_gmm_n_jobs(n_tasks):
    """GMM 外側 CV の並列数。DCV_GMM_N_JOBS 未設定時は CPU コア数（タスク数で上限）。"""
    env = os.environ.get('DCV_GMM_N_JOBS', '').strip()
    if env:
        try:
            n_jobs = int(env)
        except ValueError:
            print(f'警告: DCV_GMM_N_JOBS="{env}" は整数ではないため cpu_count を使用')
            n_jobs = os.cpu_count() or 1
    else:
        n_jobs = os.cpu_count() or 1
    return max(1, min(n_jobs, n_tasks))


def _gmm_single_outer_fold(
    regression_model_method,
    fold_idx,
    n_folds,
    x_name,
    train_idx,
    test_idx,
    x_data_np,
    y_GMM_np,
    num_y,
    numbers_of_y,
    covariance_types,
    numbers_of_components,
    weight_concentration_prior_types,
    weight_concentration_priors,
    fold_number,
):
    """GMM 外側 CV の 1 fold（cv_bo/cv_opt → fit → predict）。並列ワーカー用。"""
    task_label = f'{x_name} {regression_model_method} fold {fold_idx + 1}/{n_folds}'
    t0 = time.time()
    print(f'[GMM] {task_label} 開始', flush=True)

    x_outer = x_data_np[test_idx]
    x_inner = x_data_np[train_idx]
    y_inner = y_GMM_np[train_idx]

    var_mask = np.var(x_inner, axis=0) > 0
    x_outer = x_outer[:, var_mask]
    x_inner = x_inner[:, var_mask]
    numbers_of_X_filtered = list(range(num_y, num_y + x_outer.shape[1]))

    x_inner_mean = np.mean(x_inner, axis=0)
    x_inner_std = np.std(x_inner, axis=0, ddof=1)
    y_inner_mean = np.mean(y_inner, axis=0)
    y_inner_std = np.std(y_inner, axis=0, ddof=1)

    autoscaled_x_inner = (x_inner - x_inner_mean) / x_inner_std
    autoscaled_y_inner = (y_inner - y_inner_mean) / y_inner_std
    autoscaled_x_outer = (x_outer - x_inner_mean) / x_inner_std
    autoscaled_inner = np.concatenate([autoscaled_y_inner, autoscaled_x_inner], axis=1)

    if regression_model_method == 'GMR':
        model = GMR()
        try:
            model.cv_bo(
                autoscaled_inner, numbers_of_X_filtered, numbers_of_y,
                covariance_types, numbers_of_components, fold_number,
            )
        except ValueError:
            model.cv_opt(
                autoscaled_inner, numbers_of_X_filtered, numbers_of_y,
                covariance_types, numbers_of_components, fold_number,
            )
    elif regression_model_method == 'VBGMR':
        model = VBGMR()
        try:
            model.cv_bo(
                autoscaled_inner, numbers_of_X_filtered, numbers_of_y,
                covariance_types, numbers_of_components,
                weight_concentration_prior_types, weight_concentration_priors,
                fold_number,
            )
        except ValueError:
            model.cv_opt(
                autoscaled_inner, numbers_of_X_filtered, numbers_of_y,
                covariance_types, numbers_of_components,
                weight_concentration_prior_types, weight_concentration_priors,
                fold_number,
            )
    else:
        raise ValueError(f'Unknown GMM method: {regression_model_method}')

    model.fit(autoscaled_inner)
    predicted_y_outer_cv = model.predict_rep(
        autoscaled_x_outer, numbers_of_X_filtered, numbers_of_y,
    )
    predicted_y_outer_cv = predicted_y_outer_cv * y_inner_std + y_inner_mean
    elapsed = time.time() - t0
    print(f'[GMM] {task_label} 完了 ({elapsed:.0f}s)', flush=True)
    return regression_model_method, test_idx, predicted_y_outer_cv


def _run_gmm_outer_cv_parallel(
    gmm_tasks,
    gmm_n_jobs,
    x_name,
    tantai,
    n_folds,
    x_data_np,
    y_GMM_np,
    num_y,
    numbers_of_y,
    covariance_types,
    numbers_of_components,
    weight_concentration_prior_types,
    weight_concentration_priors,
    fold_number,
):
    """GMM 外側 CV を並列実行し、tqdm でタスク完了数を表示する。"""
    gmm_fold_results = []
    desc = f'GMM {x_name} ({tantai})'
    with ProcessPoolExecutor(max_workers=gmm_n_jobs) as executor:
        futures = {
            executor.submit(
                _gmm_single_outer_fold,
                regression_model_method,
                fold_idx,
                n_folds,
                x_name,
                train_idx,
                test_idx,
                x_data_np,
                y_GMM_np,
                num_y,
                numbers_of_y,
                covariance_types,
                numbers_of_components,
                weight_concentration_prior_types,
                weight_concentration_priors,
                fold_number,
            ): (regression_model_method, fold_idx)
            for regression_model_method, fold_idx, train_idx, test_idx in gmm_tasks
        }
        with tqdm(total=len(futures), desc=desc, unit='task') as pbar:
            for future in as_completed(futures):
                method, fold_idx = futures[future]
                gmm_fold_results.append(future.result())
                pbar.set_postfix(method=method, fold=f'{fold_idx + 1}/{n_folds}', refresh=False)
                pbar.update(1)
    return gmm_fold_results


def dcv_analysis():
    
    #ロジット変換、回帰条件の設定True
    log_transform = False    # False:対数変換なし　True:対数変換あり
    #y_names = ['選択率NPA', '収率NPA'] # y の設定
    y_names = ['STY_ETANPA', 'NPAbyETA']
    x_names = ['x1_mg_s'] # x の設定
    #regression_model_method_normals = ['OLS','PLS','RR','LASSO','NLSVR','DT','RF','GPR_0','GPR_1','GPR_2','GPR_3','GPR_4',
    #                                    'GPR_5','GPR_6','GPR_7','GPR_8','GPR_9','GPR_10','GPR_11','GPR_12','GBDT','XGB','LGB']
    #regression_model_method_normals = ['GPR_0','GPR_1','GPR_2',
    #                                    'GPR_5','GPR_6','GPR_7','GPR_8','GPR_9','GPR_10']
    regression_model_method_normals = ['GPR_0']
    regression_model_method_GMM = ['GMR', 'VBGMR']
    
    outer_fold_number = 5  # 外側の分割数。
    random_state = 24  # 分割する際の乱数シード
    autoscale_x2 = False # x2のautoscaleを全元素の平均と標準偏差とするときには、Trueとする
    
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
        data = pd.read_csv(f'datasets/v749/{x_name}.csv', index_col=0) # データの読み込み
        data = data.iloc[: 50, :] #debugのため
        # 使われていない担体は排除
        data = data.loc[:, (data != 0).any(axis=0)]
        #担体ごとにDCV解析する場合は下の行を使用し、allはコメントアウトする
        #tantai_name = [tan.replace('support_', '') for tan in data.columns if 'support' in tan and '*' not in tan]
        #tantai_name = ['Al2O3']
        #tantai_name = ['SiO2_CARIACTg-10']
        #tantai_name = ['TiO2']
        #tantai_name = ['CeO2_HS']
        #tantai_name = ['ZrO2_RC100']
        tantai_name = ['all']
                
        for tantai in tantai_name:
            drop_tantai_col = [tan for tan in data.columns if 'support' in tan and '*' not in tan]
            drop_list= y_names.copy()
            cat_name = data['触媒ロット'].copy()
            drop_list.append('触媒ロット')

            if tantai == 'all':
    #            x_raw = data.drop(columns=y_names+['触媒ロット']).copy() # xの選択
                x_raw = data.drop(drop_list, axis= 1).copy()
                y_all = data[y_names].copy() # yの選択
                #cat_name = data['触媒ロット'].copy() # 触媒名の設定      
            else:
                data_a = data[data['support_'+tantai]==1].copy()
                tantai = tantai.split('_')[0]
                x_raw = data_a.drop(drop_list, axis= 1).copy()
                #drop_tantai_colを足していないので担体列は含まれる しかしその後の同じ値を持つ列削除で落とされるので足す必要がない。
    #            x_raw = data.drop(columns=y_names+['触媒ロット']+drop_tantai_col).copy() # xの選択
                y_all = data_a[y_names].copy() # yの選択
                #cat_name = data_a['触媒ロット'].copy() # 触媒名の設定
  
            # 同じ値を多く持つ候補を削除
            threshold_of_rate_of_same_value = 0.95
            rate_of_same_value = list()
            for X_variable_name in x_raw.columns:
                same_value_number = x_raw[X_variable_name].value_counts()
                rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_raw.shape[0]))
            deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
    
            x_data = x_raw.drop(x_raw.columns[deleting_variable_numbers], axis=1)
    
            for method_type in [regression_model_method_normals, regression_model_method_GMM]:
                # データが大きいときは、Al2O3担体のGPR_3,4を除外して計算を早くする。
                if method_type == regression_model_method_normals:
                    if tantai == 'Al2O3':
                        method_type = method_type
                        #omit_models = ['GPR_3','GPR_4','GPR_11','GPR_12']
                        #method_type = [s for s in regression_model_method_normals if s not in omit_models]
                        #deleted_models = list(set(regression_model_method_normals) - set(method_type))
                    else:
                        method_type = method_type
                    for y_num, y_name in enumerate(y_names):
                        if not os.path.exists(f'{savedir}/{y_name}/yyplot/tempo'):        
                            os.mkdir(f'{savedir}/{y_name}/yyplot/tempo')                          
                        raw_y = y_all[y_name].copy()
                        log_y = raw_y.copy()
                        if log_transform:
                            log_y[log_y == 0] = (log_y[log_y != 0].nsmallest(1)/2).iloc[0]
                            log_y = logit((log_y/100).values)
                            log_y = pd.Series(log_y, index=y_all.index, name=y_name)
    
    
                        # r2, MAE, RMSE を保存
                        DCV_values = np.zeros([len(regression_model_method_normals), 4])
                        DCV_values = pd.DataFrame(DCV_values,index=regression_model_method_normals,columns=['r2_DCV', 'MAE_DCV', 'RMSE_DCV','W-RMSE_DCV'])
    
                        # 予測値を保存
                        predicted_y_values = np.zeros([len(log_y), len(regression_model_method_normals)])
                        predicted_y_values = pd.DataFrame(predicted_y_values, index=log_y.index, columns=regression_model_method_normals)
    
                        # DCV の fold 数を設定
                        inner_fold_number = 5
    #                    outer_fold_number = len(set(cat_name))
    
                        #真木修正＠2212101
    #                    groups = np.array(cat_name)
    #                    cat_list = list(set(cat_name))   #このリストが毎回変わっているため結果が変わる
    #                    cat_list.sort()
    #                    unique_groups2 = np.array(list(set(cat_name)))
    #                    unique_groups2 = np.array(cat_list)                    
    #                    unique_groups= unique_groups2.copy()
                        #np.random.seed(random_state)
                        #np.random.shuffle(unique_groups)
                        #真木修正終わり
                        
                        kf = KFold(n_splits=outer_fold_number, shuffle=True, random_state=random_state)
                        #kf = KFold(n_splits=outer_fold_number, shuffle=False)  #前でshuffleしているためここではshuffleしていない
                        for regression_model_method in method_type:
                            print(f'\r\t担体:{tantai}  y:{y_name}  model:{regression_model_method}', end='')
                            estimated_y_in_outer_cv = pd.DataFrame(np.zeros(len(log_y)), index=log_y.index, columns=[y_name])
    
    #                        for fold_cat_outer_cv in set(cat_name):
    #                        for train_group_idx, test_group_idx in kf.split(unique_groups):
    #                            train_group_numbers, test_group_numbers = unique_groups[train_group_idx], unique_groups[test_group_idx]
    #                            train_sample_numbers = np.array([], dtype=np.int64)
    #                            for i in train_group_numbers:
    #                                numbers = np.where(groups == i)[0]
    #                                if len(numbers):
    #                                    train_sample_numbers = np.r_[train_sample_numbers, numbers]
    #                            test_sample_numbers = np.array([], dtype=np.int64)
    #                            for i in test_group_numbers:
    #                                numbers = np.where(groups == i)[0]
    #                                if len(numbers):
    #                                    test_sample_numbers = np.r_[test_sample_numbers, numbers]
                    
    #                            x_outer = x_data.iloc[test_sample_numbers, :]
    #                            log_y_outer = log_y.iloc[test_sample_numbers]
    #                            x_inner = x_data.iloc[train_sample_numbers, :]
    #                            log_y_inner = log_y.iloc[train_sample_numbers]
    
                            for train_idx, test_idx in kf.split(x_data):
                                x_inner, x_outer = x_data.iloc[train_idx], x_data.iloc[test_idx]
                                log_y_inner, log_y_outer = log_y.iloc[train_idx], log_y.iloc[test_idx]
    
                                x_outer = x_outer.drop(x_outer.columns[np.where(x_inner.var()==0)],axis=1)
                                x_inner = x_inner.drop(x_inner.columns[np.where(x_inner.var()==0)],axis=1)
                                # autoscale_x2: x2 データセットの Al-Zr 元素列のみブロック共有スケール
                                use_x2_block_autoscale = autoscale_x2 and str(x_name).startswith('x2')
                                fold_rng = np.random.RandomState(random_state + int(train_idx[0]))
                                autoscaled_x_inner, autoscaled_x_outer, _ = autoscale_x_for_dcv(
                                    x_inner, x_outer, use_x2_block_autoscale, fold_rng)
                                autoscaled_log_y_inner = (log_y_inner - log_y_inner.mean()) / log_y_inner.std(ddof=1)

                                # Light-GBM 用に array 型に変換
                                autoscaled_x_inner = autoscaled_x_inner.values
                                autoscaled_log_y_inner = autoscaled_log_y_inner.values
                                autoscaled_x_outer = autoscaled_x_outer.values
    
    
                                # OLS
                                if regression_model_method == 'OLS':
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
    
                                # GPR_0
                                elif regression_model_method == 'GPR_0':
                                    kernel = ConstantKernel() * DotProduct() + WhiteKernel()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GRP_1
                                elif regression_model_method == 'GPR_1':
                                    kernel = ConstantKernel() * RBF() + WhiteKernel()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_2
                                elif regression_model_method == 'GPR_2':
                                    kernel = ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_3
                                elif regression_model_method == 'GPR_3':
                                    kernel = ConstantKernel() * RBF(np.ones(x_inner.shape[1])) + WhiteKernel()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_4
                                elif regression_model_method == 'GPR_4':
                                    kernel = ConstantKernel() * RBF(np.ones(x_inner.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_5
                                elif regression_model_method == 'GPR_5':
                                    kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_6
                                elif regression_model_method == 'GPR_6':
                                    kernel = ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_7
                                elif regression_model_method == 'GPR_7':
                                    kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_8
                                elif regression_model_method == 'GPR_8':
                                    kernel = ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_9
                                elif regression_model_method == 'GPR_9':
                                    kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GRP モデルの宣言
    
                                # GPR_10
                                elif regression_model_method == 'GPR_10':
                                    kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言

                                # GPR_11
                                elif regression_model_method == 'GPR_11':
                                    kernel = ConstantKernel() * Matern(np.ones(x_inner.shape[1]), nu=1.5) + WhiteKernel()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
    
                                # GPR_12
                                elif regression_model_method == 'GPR_12':
                                    kernel = ConstantKernel() * Matern(np.ones(x_inner.shape[1]), nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct()
                                    regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
                                        
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
                                # モデル構築
                                regression_model.fit(autoscaled_x_inner, autoscaled_log_y_inner)
    
                                #　外側のデータの予測
                                predicted_log_y_test = np.ndarray.flatten(regression_model.predict(autoscaled_x_outer)) #　予測
                                predicted_log_y_test = predicted_log_y_test * log_y_inner.std(ddof=1) + log_y_inner.mean() #スケールを戻す
                                estimated_y_in_outer_cv.iloc[test_idx] = predicted_log_y_test.reshape([-1, 1]) #　データを格納
    
                            #　予測値の格納
                            if log_transform:
                                estimated_y_in_outer_cv = expit(estimated_y_in_outer_cv)*100
                            predicted_y_values[regression_model_method] = estimated_y_in_outer_cv.values

                            #選択率、収率が0のデータを削除
                            raw_y = raw_y[raw_y > 0.001]
                            estimated_y_in_outer_cv = estimated_y_in_outer_cv.loc[estimated_y_in_outer_cv.index.isin(raw_y.index.to_list())]
    
                            #　yyplot
                            plt.rcParams['font.size'] = 18
                            plt.figure(figsize=figure.figaspect(1)) # 正方形
                            plt.title(f'{y_name} {regression_model_method}_{tantai}_{x_name}') # タイトル
                            plt.scatter(raw_y, estimated_y_in_outer_cv, c='blue', alpha=0.7, edgecolors='black') # プロット
                            y_max = np.max(np.array([raw_y, estimated_y_in_outer_cv.values.flatten()])) # y 値の最大を取得
                            y_min = np.min(np.array([raw_y, estimated_y_in_outer_cv.values.flatten()])) # y 値の最小を取得
                            if y_min <= 0:
                                y_min = 0

                            if y_name == '選択率NPA':
                                plt.plot([-0.2, 10.0], [-0.2, 10.0], 'k-')
                            elif y_name == '収率NPA':
                                plt.plot([-0.1, 1.5], [-0.1, 1.5], 'k-')
                                plt.xticks([0,0.5,1.0,1.5])
                            elif y_name == 'STY_ETANPA':
                                plt.plot([-0.01, 0.2], [-0.01, 0.2], 'k-')
                            elif y_name == 'NPAbyETA':
                                plt.plot([-0.05, 1.2], [-0.05, 1.2], 'k-')
                            else:
                                plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                                        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') # 対角線の描画
                            if y_name == '選択率NPA':
                                plt.ylim(-0.2, 10.0) # Y のサイズ
                                plt.xlim(-0.2, 10.0) # X のサイズ
                            elif y_name == '収率NPA':
                                plt.ylim(-0.1, 1.5) # Y のサイズ
                                plt.xlim(-0.1, 1.5) # X のサイズ
                            elif y_name == 'STY_ETANPA':
                                plt.ylim(-0.01, 0.2)
                                plt.xlim(-0.01, 0.2)
                            elif y_name == 'NPAbyETA':
                                plt.ylim(-0.05, 1.2)
                                plt.xlim(-0.05, 1.2)
                            else:
                                plt.ylim(y_min, y_max) # Y のサイズ
                                plt.xlim(y_min, y_max) # X のサイズ
                            plt.xlabel('Actual Y') #　縦軸ラベル
                            plt.ylabel('Predicted Y in DCV') #　横軸ラベル
                            #plt.savefig(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{tantai}_{regression_model_method}.png',bbox_inches = 'tight') # 図の保存
                            #plt.show() # 図の描画
    
                            # 評価指標の計算
                            evaluation = np.zeros((4, 1))
                            evaluation[0,0] = float(1 - sum((raw_y - estimated_y_in_outer_cv.values.flatten()) ** 2) / sum((raw_y - raw_y.mean()) ** 2)) # R2 の計算
                            evaluation[1,0] = float(sum(abs(raw_y - estimated_y_in_outer_cv.values.flatten())) / len(raw_y)) # MAE の計算
                            evaluation[2,0] = float((sum((raw_y - estimated_y_in_outer_cv.values.flatten()) ** 2) / len(raw_y)) ** 0.5) # RMSE の計算
                            evaluation[3,0] = float(np.sqrt(np.sum(raw_y * np.square(raw_y - estimated_y_in_outer_cv.values.flatten()))/ np.sum(raw_y)))  #WRMSE の計算
                            DCV_values.loc[regression_model_method,:] = evaluation[:,0] # 評価指標を格納
                            raw_y = y_all[y_name].copy()
                            
                        print()
                        #図の出力
                        save_fig = DCV_values.loc[:, 'r2_DCV'].idxmax()
                        GP_rows = DCV_values.index[DCV_values.index.str.contains('GPR')]
                        save_GPR_fig = DCV_values.loc[GP_rows, 'r2_DCV'].idxmax()
                        #shutil.move(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{tantai}_{save_fig}.png', f'{savedir}/{y_name}/yyplot/fig_{x_name}_{tantai}_{save_fig}.png')
                        #if 'GPR' not in save_fig:
                        #    shutil.move(f'{savedir}/{y_name}/yyplot/tempo/fig_{x_name}_{tantai}_{save_GPR_fig}.png', f'{savedir}/{y_name}/yyplot/fig_{x_name}_{tantai}_{save_GPR_fig}.png')
                        #shutil.rmtree(f'{savedir}/{y_name}/yyplot/tempo/')
                        # 結果の保存
                        raw_y = raw_y[raw_y > 0.001]                        
                        predicted_y_values = predicted_y_values.loc[predicted_y_values.index.isin(raw_y.index.to_list())] #選択率、収率0のデータ削除
                        predicted_y_values = pd.concat([raw_y,predicted_y_values],axis=1) # Y の推定値を格納
                        #predicted_y_values.to_csv(f'{savedir}/{y_name}/predicted_y_in_DCV/{x_name}_{tantai}_y_values_1.csv') # y の推定値を保存
                        #DCV_values.to_csv(f'{savedir}/{y_name}/prediction_accuracy/{x_name}_{tantai}_prediction_accuracy_1.csv') # 評価指標を保存
    
                elif method_type == regression_model_method_GMM:
                    #break
                    pass
                    # numpy配列に事前変換（高速化）
                    y_raw_np = y_all.values  # pandas -> numpy
                    y_GMM_np = y_all.values.copy()
                    
                    if log_transform:
                        for i in range(y_GMM_np.shape[1]):
                            y_col = y_GMM_np[:, i]
                            # 0の値を最小非ゼロ値の半分に置換
                            min_nonzero = np.min(y_col[y_col > 0]) / 2
                            y_col[y_col == 0] = min_nonzero
                            y_GMM_np[:, i] = logit(y_col / 100)

                    x_data_np = x_data.values
                    
                    # DCV設定（外側は通常モデルと同じ K-Fold）
                    num_y = y_GMM_np.shape[1]
                    num_x = x_data_np.shape[1]
                    numbers_of_X = list(range(num_y, num_y + num_x))
                    numbers_of_y = list(range(num_y))
                    fold_number = 5
                    kf_gmm = KFold(n_splits=outer_fold_number, shuffle=True, random_state=random_state)
                    
                    # 結果保存用辞書の初期化
                    predicted_y_values_dict = {}
                    DCV_values_dict = {}
                    for _y_name in y_names:
                        predicted_y_values_dict[_y_name] = np.full((len(y_raw_np), len(method_type)), np.nan)
                        DCV_values_dict[_y_name] = np.zeros((len(method_type), 3))

                    gmm_fold_splits = list(kf_gmm.split(x_data_np))
                    n_gmm_folds = len(gmm_fold_splits)
                    gmm_tasks = [
                        (regression_model_method, fold_idx, train_idx, test_idx)
                        for regression_model_method in method_type
                        for fold_idx, (train_idx, test_idx) in enumerate(gmm_fold_splits)
                    ]
                    gmm_n_jobs = _resolve_gmm_n_jobs(len(gmm_tasks))
                    print(
                        f'\n\t担体:{tantai}  GMM parallel: n_jobs={gmm_n_jobs} '
                        f'(tasks={len(gmm_tasks)}, cpu={os.cpu_count()})'
                    )
                    gmm_fold_results = _run_gmm_outer_cv_parallel(
                        gmm_tasks,
                        gmm_n_jobs,
                        x_name,
                        tantai,
                        n_gmm_folds,
                        x_data_np,
                        y_GMM_np,
                        num_y,
                        numbers_of_y,
                        covariance_types,
                        numbers_of_components,
                        weight_concentration_prior_types,
                        weight_concentration_priors,
                        fold_number,
                    )

                    for regression_model_method in method_type:
                        print(f'\r\t担体:{tantai}  model:{regression_model_method}', end='')
                        predicted_y_outer_all = np.zeros([y_GMM_np.shape[0], len(y_names)])
                        for method, test_idx, predicted_y_outer_cv in gmm_fold_results:
                            if method == regression_model_method:
                                predicted_y_outer_all[test_idx] = predicted_y_outer_cv
                        if not np.all(np.isfinite(predicted_y_outer_all)):
                            raise ValueError(
                                f'GMM 予測に非有限値があります: {regression_model_method} '
                                f'({x_name}, {tantai})'
                            )
                        filled_rows = np.zeros(y_GMM_np.shape[0], dtype=bool)
                        for method, test_idx, _ in gmm_fold_results:
                            if method == regression_model_method:
                                filled_rows[test_idx] = True
                        if not filled_rows.all():
                            raise ValueError(
                                f'GMM 外側 CV の予測行が不足しています: {regression_model_method} '
                                f'({x_name}, {tantai})'
                            )

                        # log transform復元
                        if log_transform:
                            predicted_y_outer_all = expit(predicted_y_outer_all) * 100
                        
                        # 各目的変数について評価・保存
                        for y_num, y_name in enumerate(y_names):
                            predicted_y_test = predicted_y_outer_all[:, y_num]
                            actual_y = y_raw_np[:, y_num] if not log_transform else y_all[y_name].values
                            
                            # 評価指標計算 (numpy - 高速)
                            r2 = 1 - np.sum((actual_y - predicted_y_test) ** 2) / np.sum((actual_y - np.mean(actual_y)) ** 2)
                            mae = np.mean(np.abs(actual_y - predicted_y_test))
                            rmse = np.sqrt(np.mean((actual_y - predicted_y_test) ** 2))
                            
                            # yy-plot
                            plt.rcParams['font.size'] = 18
                            plt.figure(figsize=figure.figaspect(1))
                            plt.title(f'{y_name} {regression_model_method}_{tantai}_{x_name}')
                            plt.scatter(actual_y, predicted_y_test, c='blue', alpha=0.7, edgecolors='black')
                            
                            if y_name == '選択率NPA':
                                plt.plot([-0.2, 8.0], [-0.2, 8.0], 'k-')
                            elif y_name == '収率NPA':
                                plt.plot([-0.1, 1.5], [-0.1, 1.5], 'k-')
                                plt.xticks([0,0.5,1.0,1.5])
                            elif y_name == 'STY_ETANPA':
                                plt.plot([-0.01, 0.2], [-0.01, 0.2], 'k-')
                            elif y_name == 'NPAbyETA':
                                plt.plot([-0.05, 1.2], [-0.05, 1.2], 'k-')
                            else:
                                plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                                        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') # 対角線の描画
                            if y_name == '選択率NPA':
                                plt.ylim(-0.2, 8.0) # Y のサイズ
                                plt.xlim(-0.2, 8.0) # X のサイズ
                            elif y_name == '収率NPA':
                                plt.ylim(-0.1, 1.5) # Y のサイズ
                                plt.xlim(-0.1, 1.5) # X のサイズ
                            elif y_name == 'STY_ETANPA':
                                plt.ylim(-0.01, 0.2)
                                plt.xlim(-0.01, 0.2)
                            elif y_name == 'NPAbyETA':
                                plt.ylim(-0.05, 1.2)
                                plt.xlim(-0.05, 1.2)
                            else:
                                plt.ylim(y_min, y_max) # Y のサイズ
                                plt.xlim(y_min, y_max) # X のサイズ
                                
                            plt.xlabel('Actual Y')
                            plt.ylabel('Predicted Y in DCV')
                            plt.savefig(f'{savedir}/{y_name}/yyplot/fig_{x_name}_{tantai}_{regression_model_method}.png', bbox_inches='tight')
                            # plt.show()
                            
                            # 結果保存
                            method_idx = method_type.index(regression_model_method)
                            predicted_y_values_dict[y_name][:, method_idx] = predicted_y_test
                            DCV_values_dict[y_name][method_idx, :] = [r2, mae, rmse]
                    
                    print()
                    # CSV保存
                    for y_name in y_names:
                        # 予測値保存
                        pred_df = pd.DataFrame(predicted_y_values_dict[y_name], 
                                             columns=method_type, 
                                             index=y_all.index)
                        pred_df = pd.concat([y_all[y_name], pred_df], axis=1)
                        pred_df.to_csv(f'{savedir}/{y_name}/predicted_y_in_DCV/{x_name}_{tantai}_y_values_2.csv')
                        
                        # 評価指標保存
                        eval_df = pd.DataFrame(DCV_values_dict[y_name], 
                                             columns=['r2_DCV', 'MAE_DCV', 'RMSE_DCV'],
                                             index=method_type)
                        eval_df.to_csv(f'{savedir}/{y_name}/prediction_accuracy/{x_name}_{tantai}_prediction_accuracy_2.csv')
    
    
    # 計算時間の表示
    elapsed_time = time.time() - start_time
    print("Elapsed_time : {0}[sec]".format(elapsed_time))

if __name__ == '__main__' :
    dcv_analysis()

