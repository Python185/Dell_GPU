#Y_ALL と Y_EACHを選択する。一番下の部分で。
#Y_ALLはプロパノールとエタノール両方考慮したときのtop1000、EACHのときは片方ずつ考慮したto1000(2つ)を出力する。

import warnings
warnings.simplefilter('ignore') # warning の非表示
import sys, os
sys.path.append(os.pardir)
import shutil
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
matplotlib.use('module://matplotlib_inline.backend_inline')
matplotlib.rc('font', family='Meiryo')
#import configparser
from scipy.special import logit, expit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern
from dcekit.validation import ApplicabilityDomain
from libs.libs_for_inverse.GP_model_inverse import GPinverseprocess
#from pandasgui import show
#from const import  Y_COLUMNS_ALL, RELAXATION_VALUE, ALPHA

covariance_types = ['full', 'diag', 'tied', 'spherical'] 

numbers_of_components = np.arange(2,22,1)
weight_concentration_prior_types = ['dirichlet_process', 'dirichlet_distribution']
weight_concentration_priors = 10 ** np.arange(-4, 2, 2, dtype=float)

# 相関係数の設定
threshold_of_r = 0.99
inner_fold_number = 5

def plotCorrelation(filepath, title, actual, estimated):
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=figure.figaspect(1)) # 正方形
    plt.title(title)
    plt.scatter(actual, estimated, c='blue') # プロット
    y_max = np.max(np.array([actual, estimated.flatten()])) # y 値の最大を取得
    y_min = np.min(np.array([actual, estimated.flatten()])) 
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
            [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-') # 対角線の描画
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) # Y のサイズ
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)) # X のサイズ
    plt.xlabel('Actual Y') #　縦軸ラベル
    plt.ylabel('Predicted Y') #　横軸ラベル
    plt.savefig(filepath, bbox_inches = 'tight') # 図の保存
                
def saveGaussianProcessResult(y_names, y_name, x_name, datadir, tantai_name, acquisition_function,\
                              best_model, log_transform, target_y_dict, show_flag, \
                              x_data_for_inverse_analysis_exp_vals, x_data_for_inverse_analysis_feature_vals):
                            
    dataset = pd.read_csv(f'{datadir}/v669/{x_name}.csv', index_col=0)
    # 目的変数y_namesがnanのものがあるので、除いておく
    #drop_row = dataset[y_name].isnull()
    drop_row = dataset[y_names].isnull().any(axis=1)
    drop_row = ~drop_row
    dataset = dataset[drop_row]
    # プロセス変数のみを使用するため、metalとdescに関する処理は不要
    
    drop_tantai_col = [tan for tan in dataset.columns if 'support_' in tan]                
    # x_dataset = dataset.drop(columns=y_names+['触媒ロット']+drop_tantai_col)
            
    # プロセス変数のみを使用するため、feature_valsは不要
    if x_data_for_inverse_analysis_feature_vals is not None:
        x_data_for_inverse_analysis = pd.concat([x_data_for_inverse_analysis_exp_vals, x_data_for_inverse_analysis_feature_vals], axis=1)
    else:
        x_data_for_inverse_analysis = x_data_for_inverse_analysis_exp_vals
    
    for tantai in tantai_name:
        print(f'{y_name} {tantai}')

        # 保存ディレクトリの設定
        inversesavedir = f'result/inverse_analysis/{y_name}_{x_name}_{acquisition_function}'
        # delete
        if os.path.exists(inversesavedir):
            shutil.rmtree(inversesavedir)
        # create
        if not os.path.exists(inversesavedir):
            os.mkdir(inversesavedir)

        best_model_name = best_model[y_name]

        drop_tantai_col = [tan for tan in dataset.columns if 'support_' in tan]

        if tantai == 'all':
            raw_x_data = dataset.drop(columns=y_names+['触媒ロット']).copy() # xの選択
            # raw_x_data = dataset.drop(columns=y_names + ['触媒ロット'] + drop_tantai_col)
            tantai_data = dataset.copy()

        else:
            # プロセス変数のみのため、data_typeによる場合分けは不要
            tantai_data = dataset[dataset['support_'+tantai]==1].copy()
            raw_x_data = tantai_data.drop(columns=y_names+['触媒ロット']+drop_tantai_col).copy()
        raw_y_data = tantai_data[y_name].copy() # yの選択
        log_y_data = raw_y_data.copy()
        if log_transform[y_name]:
            log_y_data[log_y_data == 0] = (log_y_data[log_y_data != 0].nsmallest(1)/2).iloc[0]
            log_y_data = logit((log_y_data/100).values)
            log_y_data = pd.Series(log_y_data, index=raw_y_data.index, name=y_name)

        # cat_name = tantai_data['触媒ロット'].copy() # 触媒名の設定

        # 同じ値を多く持つ候補を削除
        threshold_of_rate_of_same_value = 0.99
        rate_of_same_value = list()
        for X_variable_name in raw_x_data.columns:
            same_value_number = raw_x_data[X_variable_name].value_counts()
            rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / raw_x_data.shape[0]))
        deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
        x_data = raw_x_data.drop(raw_x_data.columns[deleting_variable_numbers], axis=1)
        
        # support_から始まる列（カテゴリ変数）を除外してスケーリング
        support_cols = [col for col in x_data.columns if col.startswith('support_')]
        numeric_cols = [col for col in x_data.columns if not col.startswith('support_')]
        
        # 数値列のみでスケーリング
        x_data_numeric = x_data[numeric_cols]
        x_data_for_inverse_analysis_numeric = x_data_for_inverse_analysis[numeric_cols]
        
        autoscaled_x_data_for_inverse_analysis_numeric = (x_data_for_inverse_analysis_numeric - x_data_numeric.mean(axis=0)) / x_data_numeric.std(axis=0, ddof=1)
        autoscaled_x_data_numeric = (x_data_numeric - x_data_numeric.mean(axis=0)) / x_data_numeric.std(axis=0, ddof=1)
        autoscaled_log_y_data = (log_y_data - log_y_data.mean(axis=0)) / log_y_data.std(axis=0, ddof=1)
        
        # スケーリングされた数値列とカテゴリ変数を結合
        autoscaled_x_data_for_inverse_analysis = pd.concat([
            autoscaled_x_data_for_inverse_analysis_numeric,
            x_data_for_inverse_analysis[support_cols]
        ], axis=1)
        autoscaled_x_data = pd.concat([
            autoscaled_x_data_numeric,
            x_data[support_cols]
        ], axis=1)
        
        #謎エラー(inputにnan含む)出現のため、内容確認
        if autoscaled_x_data_for_inverse_analysis.isnull().any().sum() != 0:
            print(autoscaled_x_data_for_inverse_analysis[autoscaled_x_data_for_inverse_analysis.isnull().any(axis= 1)])
 
        # # ADの設定（数値列のみを使用）
        ad = ApplicabilityDomain(method_name='ocsvm', rate_of_outliers=0.003)
        ad.fit(autoscaled_x_data_numeric)
        ad_data = ad.predict(autoscaled_x_data_for_inverse_analysis_numeric)
 
        # GPの逆解析用インスタンスの作成
        gp_innverse = GPinverseprocess(log_transform, target_y_dict)
        
        # 指定されたGPRカーネルのみを定義
        def get_kernel(kernel_index, n_features):
            """指定されたインデックスに対応するカーネルを返す"""
            kernels = [
                ConstantKernel() * DotProduct() + WhiteKernel(),
                ConstantKernel() * RBF() + WhiteKernel(),
                ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * RBF(np.ones(n_features)) + WhiteKernel(),
                ConstantKernel() * RBF(np.ones(n_features)) + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
                ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
                ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * Matern(np.ones(n_features), nu=1.5) + WhiteKernel(),
                ConstantKernel() * Matern(np.ones(n_features), nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ]
            return kernels[kernel_index]
        
        # 指定されたGPRモデルを使用
        best_model_name = best_model[y_name]
        best_model_index = int(best_model_name.split('_')[1])
        
        # 指定されたカーネルのみを取得
        kernel = get_kernel(best_model_index, autoscaled_x_data.shape[1])
        

        # GPの逆解析用インスタンスの作成
        gp_innverse = GPinverseprocess(log_transform, target_y_dict)
        if acquisition_function == 'MI':
            cumulative_variance = np.zeros(x_data_for_inverse_analysis.shape[0])
        else:
            cumulative_variance = None
            
        # 獲得関数の設定
        RELAXATION_VALUE = 0.01
        DELTA = 10 ** -6
        ALPHA = np.log(2 / DELTA)

        gp_innverse.setting_acquisition_function(acquisition_function, cumulative_variance, RELAXATION_VALUE, ALPHA)
        
        for tantai in tantai_name:
            print(f"Processing tantai: {tantai}, Using {best_model_name} for {y_name}")
            regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0) # GPR モデルの宣言
            regression_model.fit(autoscaled_x_data_numeric, autoscaled_log_y_data)
            
            #実験データをモデルに入力して推定値との相関グラフを出力する。これを挿入(230222真木)
            estimated_evidence_y, _ = regression_model.predict(autoscaled_x_data_numeric, return_std=True)
            estimated_evidence_y = estimated_evidence_y * log_y_data.std(axis=0, ddof=1) + log_y_data.mean(axis=0)
            path = f'{inversesavedir}/gp_correlation_{x_name}_{best_model_name}.png'
            plotCorrelation(path, f'{y_name}: {x_name} {best_model_name}', raw_y_data, estimated_evidence_y)
            #ここまで(真木)
            
            # モデルに入力（数値列のみを使用）
            autoscaled_estimated_inverse_y, autoscaled_estimated_inverse_y_std = regression_model.predict(autoscaled_x_data_for_inverse_analysis_numeric, return_std=True)
            estimated_inverse_log_y = autoscaled_estimated_inverse_y * log_y_data.std(axis=0, ddof=1) + log_y_data.mean(axis=0)
            estimated_inverse_log_y_std = autoscaled_estimated_inverse_y_std * log_y_data.std(axis=0, ddof=1)
            #estimated_inverse_log_y, estimated_inverse_log_y_std = regression_model.predict(autoscaled_x_data_for_inverse_analysis, return_std=True)
            #estimated_inverse_log_y = estimated_inverse_log_y * log_y_data.std(axis=0, ddof=1) + log_y_data.mean(axis=0)
            #estimated_inverse_log_y_std *= log_y_data.std(axis=0, ddof=1)
            #estimated_inverse_y = estimated_inverse_log_y.copy()
            
            if log_transform[y_name]:
                estimated_inverse_y = expit(estimated_inverse_log_y)*100
                estimated_inverse_y_std = expit(estimated_inverse_log_y_std)
                #estimated_inverse_y = expit(estimated_inverse_y)*100
                #estimated_inverse_log_y_std = expit(estimated_inverse_log_y_std)# 標準偏差のスケールも元に戻す
            else:
                estimated_inverse_y = estimated_inverse_log_y
                estimated_inverse_y_std = estimated_inverse_log_y_std
                
            # 確率・推定値・推定値の分散を出力 
            # 獲得関数(PTR, PI, MI, EI)により分岐
            probability, estimated_inverse_y, estimated_inverse_log_y_std = gp_innverse.calc_probability_acquisition(dataset[y_name], y_name, best_model_index, estimated_inverse_y, estimated_inverse_y_std)
            #probability, estimated_inverse_y, estimated_inverse_log_y_std = gp_innverse.calc_probability_acquisition(dataset[y_name], y_name, best_model_index, estimated_inverse_y, estimated_inverse_y_std)
            
            log_probability = np.log(probability)
            estimated_result = pd.DataFrame(np.concatenate([estimated_inverse_y.reshape((-1, 1)), estimated_inverse_log_y_std.reshape((-1, 1)), ad_data.reshape((-1, 1)), probability.reshape((-1, 1)), log_probability.reshape((-1, 1))], axis=1), index=x_data_for_inverse_analysis.index, columns=['estimated_inverse_y', 'estimated_inverse_y_std', 'ad', 'probability', 'log_probability'])
            estimated_result.to_csv(f'{inversesavedir}/{x_name}_{acquisition_function}_gp_estimated_result_{tantai}_{best_model_name}.csv')
            # if best_model[y_name] == best_model_name: # best modelの結果を{inversesavedir}に出力する
            #     estimated_result.to_csv(f'{inversesavedir}/best_gp_estimated_result_{y_name}_{tantai}_{best_model_name}.csv')
            # sns.set(style='whitegrid')
            x_new = np.concatenate([raw_y_data.values, estimated_inverse_y])
            bins = np.linspace(x_new.min(), x_new.max(), num=26)
            plt.rcParams['font.size'] = 24
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(111)
            ln1 = ax1.hist(estimated_inverse_y, color='blue', alpha=0.7, label='inverse y', ec='black', bins=bins)
            ax2 = ax1.twinx()
            ln2 = ax2.hist(raw_y_data, color='red', label='actual y', ec='black', alpha=0.8, bins=bins)
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.set_xlabel(f'{y_name} {best_model_name}')
            ax1.set_ylabel('Count log scale (blue)')
            ax1.set_yscale('log')
            ax1.grid(True)
            hist, bin_edges = np.histogram(raw_y_data, bins=bins)
            ax2.plot([target_y_dict[y_name]['score'],target_y_dict[y_name]['score']], [0,hist.max()*1.2], c='black')
            ax2.text(target_y_dict[y_name]['score']*0.7, hist.max()*0.85, f'Target\ny = {target_y_dict[y_name]["score"]}')
            ax2.set_ylim([0,hist.max()*1.2])
            ax2.set_ylabel('Count (red)')
            ax2.legend(h1+h2, l1+l2, loc='upper right')
            plt.savefig(f'{inversesavedir}/{x_name}_{acquisition_function}_gp_hist_{tantai}_{y_name}_{best_model_name}.png',bbox_inches = 'tight')
            #plt.show(block=show_flag)
               
def str_reverse(item):
    a = item.split(' * ')
    a.reverse()
    b = ' * '.join(a)
    return b
      
def inverse_analysis(y_type, one_metrics, tantai_name):

    # 保存ファイルの設定
    savedir1 = 'result'
    if not os.path.exists(savedir1):
        os.mkdir(savedir1)
    savedir2 = 'result/inverse_analysis'
    if not os.path.exists(savedir2):
        os.mkdir(savedir2)
    savedir4 = 'result/inverse_analysis/inverse_gpr_optimize'
    if not os.path.exists(savedir4):
        os.mkdir(savedir4)
    
    #各種設定 NPAとETAを別のモデルで逆解析するときは、x_names=['x26_NPA','x26_ETA']のように
    #設定する。実際のdatasetsフォルダのファイル名がx26_NPA1, x26_ETA2のような場合には
    #逆解析のためにRenameした方が良い　フォルダ名を変えるとかして。
    # 獲得関数
    #acquisition_functions = ['PTR','PI','MI','EI']
    acquisition_functions = ['PTR']  
        
    #y_names = Y_COLUMNS_ALL
    y_names = ['選択率NPA', '収率NPA']
    x_names = ['x0']
    # 今回のデータでは、下記のbest modelで逆解析する
    log_transform = {'選択率NPA':False, '収率NPA':False}
    best_model = {'選択率NPA':'GPR_10', '収率NPA':'GPR_11'}
    
    # best_modelからGPRカーネルのインデックスを自動取得
    def get_gpr_kernel_index(best_model_dict, y_name):
        """best_modelからGPRカーネルのインデックスを取得"""
        model_name = best_model_dict[y_name]
        return int(model_name.split('_')[1])
    
    # 選択率NPA用のGPRカーネルインデックスを取得
    gpr_kernel_index = get_gpr_kernel_index(best_model, '選択率NPA')
    #240410 選択率と収率の目標値が変更された     
    target_y_dict = {
        '選択率NPA':{'score': 20, 'objective':'maximum'},
        '収率NPA': {'score': 5, 'objective':'maximum'},
    }
    
    top_size=500
    
    np.random.seed(1)
    random.seed(1)
    
    show_flag = False
    # データセットフォルダの設定
    datadir = 'datasets'
    #datadir = 'matlantis_descriptors/datasets/Akashi#1'    
    # virtual_dataフォルダの設定
    #virtualdatadir = 'result/virtual_data/virtual_data_for_inverse_analysis/炉温300℃/x2_18&x2_17/'
    virtualdatadir = 'result/virtual_data/'
    
    # プロセス変数のみの読込み
    x_data_for_inverse_analysis_exp_vals = pd.read_csv(f'{virtualdatadir}process_data_all_combinations.csv', index_col=0, encoding='cp932')
    x_data_for_inverse_analysis_exp_vals = x_data_for_inverse_analysis_exp_vals.dropna(how= 'all', axis= 1)
    process = x_data_for_inverse_analysis_exp_vals
    process_nan_checker = process[process.isna().any(axis= 1)]
        
    # 逆解析
    print('inverse analysis start')
    savedir3 = 'result/inverse_analysis'
    if not os.path.exists(savedir3):
        os.mkdir(savedir3)
                
    dfs = {}
    for acquisition_function in acquisition_functions:
        print('--- acquisition_function = ', acquisition_function ,' start ---')
        
        # Yごとに逆解析を行い、獲得関数を算出し、ファイルに保存する。
        for y_name in y_names:
            print('y_name : ', y_name,' start')
            #tantai_name = ['all']
            
            #x_namesで場合分け
            if x_names[0] in ['x2_41','x1_av']:
                x_name = x_names[0]
            else:  #NPA, ETAで異なるデータを使うとき
                if y_name == '選択率NPA':
                    x_name = x_names[0]
                else:
                    x_name = x_names[1]
                print('x_name = ', x_name)                    
        
            # プロセス変数のみを使用
                    
            saveGaussianProcessResult(y_names, y_name, x_name, datadir, tantai_name, acquisition_function,\
                                      best_model, log_transform, target_y_dict, show_flag, \
                                      x_data_for_inverse_analysis_exp_vals, None)
            
            print('y_name : ', y_name,' end ')    
                
        
        # 各評価スロットについて各y_nameの同時確率を求めてその対数をとって出力する
        # 確率が0のときは、-infになるので、-999999と処理した
        for tantai in tantai_name:

            # support_Alによる場合分けは不要（プロセス変数のみのため）

            # 保存ファイルの設定
            inversesavedir2 = f'result/inverse_analysis'
                    
            df_best = []
            for y_name in y_names:
                if x_names[0] in ['x2_41','x1_av']:
                    x_name = x_names[0]
                else:
                    if(y_name=='選択率NPA'):
                        x_name = x_names[0]
                    else:
                        x_name = x_names[1]                        
                    
                # 既存ファイルがあるフォルダ
                inversesavedir = f'result/inverse_analysis/{y_name}_{x_name}_{acquisition_function}'
                #df_best = []
                # best_modelで指定されたGPRモデルのみ使用
                model_name = best_model[y_name]
                idx_bo_kernels = int(model_name.split('_')[1])
                csv_name = f'{inversesavedir}/{x_name}_{acquisition_function}_gp_estimated_result_{tantai}_GPR_{idx_bo_kernels}.csv'
                df_best.append(pd.read_csv(csv_name, index_col=0, encoding='cp932').replace([-np.inf], -999999))
                            
            if (y_type == 'Y_EACH'):
                col_list = []
                for y_name_sub, df_tmp in zip(y_name, df_best):
                #for y_name_2, df_tmp in zip(y_names, df_best):
                    col_list = col_list + (f'{y_name}_' + df_tmp.columns).to_list()
                new_df = pd.concat(df_best, axis=1)
                new_df.columns = col_list
                if (one_metrics is True):
                    print('Y_EACH : one_metrics is True. ----- This is not expected at this stage.' )
                    pass                         
                # プロセス変数とのマージ（metalは不要）
                new3_df = pd.merge(x_data_for_inverse_analysis_exp_vals, new_df, left_index=True, right_index=True)                    
                # BO_resultsを出力※100万件の時はコメントアウトしたほうが速い    
                new3_df.to_csv(f'{inversesavedir2}/{y_type}_{y_name}_{acquisition_function}_BO_results.csv', encoding='cp932')
                if (one_metrics is True):
                    print('Y_EACH : one_metrics is True. ----- This is not expected at this stage.' )
                else:
                    # 上位1000を出力
                    if (acquisition_function == 'PTR' or acquisition_function == 'PI'):
                        df_sorted = new3_df.sort_values(ascending=False, by=y_name+'_log_probability')
                    else:
                        df_sorted = new3_df.sort_values(ascending=False, by=y_name+'_probability')
                df_selected = df_sorted.iloc[:top_size, :]  
                df_selected.to_csv(os.path.join(savedir4, acquisition_function+'_'+y_name+'_inverse_gpr_optimize_top'+str(top_size)+'.csv'), encoding='sjis')
                # summary
                dfs[acquisition_function+'_'+y_type+'_'+y_name] = df_selected
                # 一回空にする       
                df_best = []
            else:
                #print('y_type is Y_ALL' )
                pass
                                
            if (y_type == 'Y_ALL'):
                col_list = []
                for y_name, df_tmp in zip(y_names, df_best):
                    col_list = col_list + (f'{y_name}_' + df_tmp.columns).to_list()
                new_df = pd.concat(df_best, axis=1)
                new_df.columns = col_list
                df_log_probability_list = [i_df['log_probability'].replace([-np.inf], -999999) for i_df in df_best]
                df_log_probability = sum(df_log_probability_list)
                df_probability_list = [i_df['probability'] for i_df in df_best]
                df_probability = sum(df_probability_list)
                if (one_metrics is True):
                    if (acquisition_function == 'PTR' or acquisition_function == 'PI'):
                        # 獲得関数：PTR,PIの場合：対数和
                        new_df['one_metrics'] = df_log_probability
                    else:
                        # 獲得関数：MI,EIの場合：和
                        new_df['one_metrics'] = df_probability
                else:
                    print('Y_ALL : one_metrics is False. ----- This is not expected at this stage.')   
                    
                # プロセス変数とのマージ（metalは不要）
                new3_df = pd.merge(x_data_for_inverse_analysis_exp_vals, new_df, left_index=True, right_index=True)                    
                # BO_resultsを出力※100万件の時はコメントアウトしたほうが速い    
                new3_df.to_csv(f'{inversesavedir2}/{y_type}_{acquisition_function}_{tantai}_BO_results.csv', encoding='cp932')
                if (one_metrics is True):
                    # 上位1000を出力
                    df_sorted = new3_df.sort_values(ascending=False, by='one_metrics')
                else:
                    print('one_metrics is False. ----- This is not expected at this stage.')                        
                df_selected = df_sorted.iloc[:top_size, :]  
                df_selected.to_csv(os.path.join(savedir4, acquisition_function+'_'+tantai+'_inverse_gpr_optimize_top'+str(top_size)+'.csv'), encoding='sjis')
                # summary
                dfs[acquisition_function+'_'+y_type] = df_selected
            else:
                #print('y_type is Y_EACH.' )
                pass
              
            print('--- acquisition_function = ', acquisition_function ,' end ---')
        
        #まとめてExcelに書き出す
        excel_path = os.path.join(savedir2, x_name+tantai+'_inverse_gpr_optimize_'+str(top_size)+'.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            for name, df in dfs.items():
                df.to_excel(writer, sheet_name=name)   
        print('inverse analysis end')         

if __name__ == '__main__' :
    
    #解析方法
    #y_type = 'Y_EACH'      #Y_EACHのときはone_metricsはFalseとなる
    #one_metrics = False
    y_type = 'Y_ALL'        #Y_ALLのときはone_metricsをTrueとする
    tantai_name = ['all']   # 元データに複数の担体種があるときに使用
    #tantai_name = ['Al2O3_A-11']
    one_metrics = True
    
    inverse_analysis(y_type, one_metrics, tantai_name)
