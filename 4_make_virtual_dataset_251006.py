#逆解析用のデータセット作成 プロセス変数のみ

import sys, os
sys.path.append(os.pardir)
import random
import pandas as pd
import numpy as np
import warnings
# warning の非表示
warnings.simplefilter('ignore')
import sys, os
sys.path.append('./libs')
from libs.Calculator_x import Calc_desc, CalcX1
import shutil
import itertools

np.random.seed(8)
random.seed(8)
selector = 'make_data_without_database'

def make_inverse_analysis_data():
    # 全組み合わせを生成するため、サンプル数制限を削除
    print('Generating all possible combinations...')
            
    # データセットフォルダの設定
    datadir = 'datasets/v669'
    # 出力フォルダの設定
    outputdir = 'result/virtual_data'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    
    y_names = ['選択率NPA', '収率NPA']
    x_names = ['x0'] 
    
    x_name = x_names[0]
    print('x_name = ', x_name)
    data = pd.read_csv(f'{datadir}/{x_name}.csv', index_col=0) 
      
    # 目的変数y_namesがnanのものがあるので、除いておく
    drop_row = data[y_names].isnull().any(axis=1)
    drop_row = ~drop_row
    data = data[drop_row]
    x_data = data.drop(columns=y_names+['触媒ロット'])
        
    # サンプル作成
    print('create samples start')    
    
    #リストの定義
    # 住化dataでは、base種とbase濃度となっているが、説明変数としてはbase種として数値が濃度となる(250224)
    # base_Na2CO3は削除, support_TiO2_SSP-Mも削除(250407)
    process_list = ['前処理還元炉温℃', '評価反応炉温℃', 'temp', 'flow_base','flow_slurry', 'flow_red', 'wash','前担持','pressure','conc_ETA','ターゲット担持量',
                    'support_Al2O3_KC501','support_SiO2_CARIACTg-10','base_NaOH','base_LiOH','base_KOH','base_urea'] 
    
    # 条件設定  250224改訂
    inverse_analysis_condition_dict = {
        '前処理還元炉温℃':['list', [350, 400, 450]],
        '評価反応炉温℃':['list', [260]],    
        'temp':['list', [250, 350, 380]],
        'flow_base':['list', [0.1, 1, 5, 10]],
        'flow_slurry':['list', [10, 20, 30]],
        'flow_red':['list', [50, 75, 100]],
        'conc_base':['list', [2.5]],
        'pressure':['list', [25]],   # 20250522 今回は25に固定
        'conc_ETA':['list', [25]],
        'base_NaOH':['list', [0, 2.5]],
        'base_LiOH':['list', [0, 2.5]],
        'base_KOH':['list', [0, 2.5]],
        #'base_Na2CO3':['list', [0, 2.5]],  # baseの合計=2.5の制約必要
        'base_urea':['list', [0, 2.5]],
        'ターゲット担持量wt%':['list', [1, 3, 5, 10]],
        '前担持':['list', [1]],
        'support':['list', ['support_al', 'support_SiO2_CARIACTg-10']],
        }    
    
    # 全組み合わせを生成
    print('Generating all combinations...')
    
    # base以外のパラメータの値のリストを作成
    non_base_param_values = []
    non_base_param_names = []
    base_list = ['base_NaOH', 'base_LiOH', 'base_KOH', 'base_urea']
    
    for key in inverse_analysis_condition_dict.keys():
        if key not in base_list:
            condition_list = inverse_analysis_condition_dict[key]
            if condition_list[0] == 'list':
                non_base_param_values.append(condition_list[1])
                non_base_param_names.append(key)
    
    # supportの種類に応じたbaseの組み合わせを生成
    def generate_base_combinations(support_type):
        """supportの種類に応じてbaseの組み合わせを生成"""
        base_list = ['base_NaOH', 'base_LiOH', 'base_KOH', 'base_urea']
        base_values = [0, 2.5]
        
        if support_type == 'support_SiO2_CARIACTg-10':
            # SiO2の場合はbase_ureaのみ
            return [[0, 0, 0, 2.5]]
        elif support_type == 'support_al':
            # Al2O3の場合は全てのbaseの組み合わせ（合計2.5）
            # 各baseが0または2.5の値を取り、合計が2.5になる組み合わせを生成
            combinations = []
            for combo in itertools.product(base_values, repeat=4):
                if sum(combo) == 2.5:
                    combinations.append(list(combo))
            return combinations
        else:
            return []
    
    # 各supportのbase組み合わせを生成
    support_base_combinations = {}
    for support_type in ['support_SiO2_CARIACTg-10', 'support_al']:
        support_base_combinations[support_type] = generate_base_combinations(support_type)
        print(f'Valid base combinations for {support_type}: {len(support_base_combinations[support_type])}')
    
    # 非baseパラメータの組み合わせを生成
    non_base_combinations = list(itertools.product(*non_base_param_values))
    print(f'Non-base combinations: {len(non_base_combinations)}')
    
    # supportとbaseの組み合わせを結合
    all_combinations = []
    for non_base_combo in non_base_combinations:
        # non_base_comboからsupportの値を取得
        support_value = non_base_combo[non_base_param_names.index('support')]
        
        # supportに対応するbaseの組み合わせを取得
        base_combinations = support_base_combinations.get(support_value, [])
        
        # 各baseの組み合わせと結合
        for base_combo in base_combinations:
            full_combo = list(non_base_combo) + base_combo
            all_combinations.append(full_combo)
    
    print(f'Total combinations generated: {len(all_combinations)}')
    
    # 全パラメータ名のリストを作成
    all_param_names = non_base_param_names + base_list
    
    # DataFrameを作成
    x_data_for_inverse_analysis = pd.DataFrame(all_combinations, columns=all_param_names)
        
    # 担体列をdummy変数に変換する
    tantai_col = inverse_analysis_condition_dict['support'][1]
    for each in tantai_col:
        x_data_for_inverse_analysis[each] = (x_data_for_inverse_analysis['support'] == each)*1
    x_data_for_inverse_analysis = x_data_for_inverse_analysis.drop(columns=['support'])
    x_data_for_inverse_analysis['support_al'].replace(np.nan, 0, inplace= True)
    x_data_for_inverse_analysis['support_SiO2_CARIACTg-10'].replace(np.nan, 0, inplace= True)       
    
    #(flow_red + flow_base) / flow_slurry >= 8.09 のサンプルを削除する
    threshold_of_flow = 8.09 
    result_value = (x_data_for_inverse_analysis['flow_red']+x_data_for_inverse_analysis['flow_base']) / x_data_for_inverse_analysis['flow_slurry']
    # 条件に合わない行を削除
    mask = result_value < threshold_of_flow
    x_data_for_inverse_analysis = x_data_for_inverse_analysis[mask]
    x_data_for_inverse_analysis.reset_index(drop=True, inplace=True)
    print(f'After flow constraint filtering: {len(x_data_for_inverse_analysis)} combinations')
    
    # baseの制約条件は既に組み合わせ生成時に適用済み（合計が2.5になる組み合わせのみ）
    print(f'Base constraint already applied: all combinations have base sum = 2.5')
    
    # プロセス変数のみのため、特徴量計算は不要
    # プロセス変数データは既にx_data_for_inverse_analysisに格納済み                    
    
    # プロセス変数のみのため、金属組成データの処理は不要
    
    fname_process = 'process_data_all_combinations.csv'
    # ファイル出力（プロセス変数のみ）
    x_data_for_inverse_analysis['support_CeO2_HS'] = 0
    x_data_for_inverse_analysis['support_ti'] = 0
    x_data_for_inverse_analysis['support_ZrO2_RC100'] = 0
    x_data_for_inverse_analysis.to_csv(f'{outputdir}/'+fname_process, encoding='cp932')
    print(f'Output file: {outputdir}/{fname_process}')
    print(f'Total combinations saved: {len(x_data_for_inverse_analysis)}')
        
    # ファイル出力
    #x_data_for_inverse_analysis[['前処理還元炉温℃', '評価反応炉温℃', 'temp', 'flow_NaOH','flow_slurry', 'flow_red', 'wash', 'support_Al2O3_A-11','support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100']].to_csv(f'{outputdir}/{constraint}_x_data_for_inverse_analysis_exp_vals.csv', encoding='cp932')
    #inverse_metal_x_data.to_csv(f'{outputdir}/{constraint}_metal_x_data_for_inverse_analysis.csv', encoding='cp932')
    #print('case:'+constraint+' end')
        
    print('create samples end')
    
# merge_allFiles関数は不要になったため削除   

def createSaveDirSub(subdir):
    # delete
    if os.path.exists('result/virtual_data/'+subdir):
        shutil.rmtree('result/virtual_data/'+subdir)
    # create
    if not os.path.exists('result/virtual_data/'+subdir):
        os.mkdir('result/virtual_data/'+subdir)


if __name__ == '__main__' :
    if selector == 'make_data_without_database':
        make_inverse_analysis_data()       