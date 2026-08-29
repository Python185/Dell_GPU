# -*- coding: utf-8 -*-
"""
@author: dcelab
"""

# Demonstration of CVPFI (Cross-Validated Permutation Feature Importance considering correlation between features)
# in GPR (Gaussian Process Regression)

import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
from dcekit.variable_selection import cvpfi
from dcekit.variable_selection import cvpfi_gmr
from libs.create_model import createModel, createGMMModel
from sklearn.model_selection import train_test_split
from libs.create_model import StdScaler

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
    #selection = {'C3-OH selec (%)':     ['x1_as3', 'LGB'],
    #             'C3-OH yield (%)':     ['x1_as3', 'LGB'],}
    selection = {'mae':     ['x2_asf', 'RF'],}

    
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
    excel_path = os.path.join("result", 'cvpfi_importance_summary.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31])   
    print(' summary_csv output end ')


def checkDataset(y_name, x_name):
    print('   checkDataset start ')    
    # read dataset
    data = pd.read_csv(f'MyWork/datasets_CO2_CxOH/{x_name}.csv', index_col= 0, header= 0, encoding='utf-8-sig') # データの読み込み
    if np.inf in data.values or np.nan in data.values:
        data = data.replace(np.inf, np.nan)    # infをnanに置換
        dropped_columns = data.columns[data.isnull().any()].tolist()   # 欠損値を含む列のリストを取得        
        data = data.dropna(axis=1, how='any')  # 欠損値を含む列を削除 
  
        print('inf or nan found and dropped ', dropped_columns)  
    #data = pd.read_csv(f'MyWork/datasets/larger_data/{x_name}.csv', index_col=0, header=0, encoding='utf-8-sig') # データの読み込み
    #x_train, x_test = train_test_split(data, test_size=0.25, shuffle=True, random_state=43)
    #data = x_test
    #data.replace(np.nan, 0, inplace= True)
    
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
        # GMM以外の場合
        standarization=True
        # StdScalerオブジェクトを作成
        y_scaler = StdScaler(y_train)
        regression_model = createModel(model_name, x_train, y_train, y_scaler, fold_number, standarization)
        # Model fit
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
    plt.savefig(f'result/cvpfi/{dirplot}/graph_{y_name}_{x_name}_{model_name}.png', bbox_inches = 'tight') # 図の保存
    #plt.show()
    print('   plot output end ')
    
def outpotCsv(dircsv, dfs, x_name, y_name, model_name, x_train, importances_mean, importances_std):
    print('   csv output start ')
    name = f'({y_name})_{x_name}_{model_name}'
    report_path = os.path.join('result/cvpfi/'+dircsv, f'importance_of{name}.csv')
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
