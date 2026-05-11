#逆解析用のデータセット(x26)を作成する。全ての説明変数を計算するため100万個のデータ生成には
#5日程度掛かる。高速で行いたいときはx26データベースを用いて生成する方が良い。
#原則、上記の方法で100万個データセットは作るべき。100万データセットを作るコードをゴチャゴチャいじると
#その後でないとデータ生成させられないため、時間のロスになりやすい。

import sys, os
sys.path.append(os.pardir)
import random
import pandas as pd
import numpy as np
import warnings
# warning の非表示
import sys, os
sys.path.append('./libs')
warnings.simplefilter('ignore')
from libs.InverseAnalysisUtility import MIutility
#from libs_for_inverse.util import MIutility
#from libs.Calculator_x26 import MetalFeaturizers
from libs.Calculator_x import Calc_desc, CalcX1
import shutil
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern
from decimal import Decimal, getcontext
from scipy.stats import zscore
from scipy.linalg import det

np.random.seed(5)
random.seed(5)
#x26用のデータベースを用いてデータを生成するか、用いずに生成するかを選択する。
#データベースを使用する場合は、下のmake_virtual_dataの束縛条件、dataset名等を入力する必要がある。
selector = 'make_data_without_database'
#selector = "make_data_using_database"

#生成する触媒系の選択　単純系触媒(3種、0.1刻み)か通常触媒(5種、0.01刻み)かを選択する
#catalyst = 'simple'
catalyst = 'normal'

def make_inverse_analysis_data():
    # 束縛条件の指定(250224)
    # SCC1:Rhが0.1-0.7の範囲でランダムに生成
    # True:新束縛条件 Rh:0.05-0.7,Mn+Fe:0.05-0.7,FeMn:0-0.5,元素数45
    constraint = ['True']
    #constraint = ['SCC1']
    #constraint = ['matlantis_dft']
        
    #サンプル数
    number_of_generating_samples = 1000000 # 100万
    iteration_number = 100    # check [サンプル数 ÷ iteration数]：割り切れるようにする
    if (number_of_generating_samples % iteration_number) != 0 :
        raise Exception('[サンプル数 ÷ iteration数]：割り切れるように設定してください。')        
    #一つのcsvの件数
    csv_output_count = int(number_of_generating_samples / iteration_number)
    print('number_of_generating_samples = ', number_of_generating_samples)
            
    # データセットフォルダの設定
    #datadir = 'matlantis_descriptors/datasets/Akashi#1'
    #datadir = 'matlantis_descriptors/datasets/#2'
    datadir = 'datasets/v564'
    # 出力フォルダの設定
    outputdir = 'result/virtual_data'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    
    y_names = ['選択率NPA', '収率NPA']
    #x_names = ['xx2_mat1_b9']  #選択率NPA,収率NPAともこれ
    #x_names = ['x2_dft1_b9']
    #x_names = ['x2_17_and_x2_20']  #NPAはx2_20, ETAはx2_17で逆解析
    x_names = ['x1_av']  #選択率、収率ともに同じもの

    
    x_name = x_names[0]
    print('x_name = ', x_name)
    if x_name == 'x2_17_and_x2_20':
        data = pd.read_csv(f'{datadir}/x2_20.csv', index_col=0)
        data_2 = pd.read_csv(f'{datadir}/x2_17.csv', index_col=0)
        drop_row_2 = data_2[y_names].isnull().any(axis=1)
        drop_row_2 = ~drop_row_2
        data_2 = data_2[drop_row_2]
        x_data_2 = data_2.drop(columns=y_names+['触媒ロット'])       
    elif x_name == 'x2_18_and_x2_17':
        data = pd.read_csv(f'{datadir}/x2_18.csv', index_col=0)
        data_2 = pd.read_csv(f'{datadir}/x2_17.csv', index_col=0)
        drop_row_2 = data_2[y_names].isnull().any(axis=1)
        drop_row_2 = ~drop_row_2
        data_2 = data_2[drop_row_2]
        x_data_2 = data_2.drop(columns=y_names+['触媒ロット'])    
    else:
        data = pd.read_csv(f'{datadir}/{x_name}.csv', index_col=0) 
      
    # 目的変数y_namesがnanのものがあるので、除いておく
    drop_row = data[y_names].isnull().any(axis=1)
    drop_row = ~drop_row
    data = data[drop_row]
    x_data = data.drop(columns=y_names+['触媒ロット'])
        
    # サンプル作成
    print('create samples start')    
    for constraints in constraint:
        print('case:'+constraints+' start')
        
        for iter_n in range(iteration_number):
            print(f'--- iteration:{iter_n+1} ---')
            
            #リストの定義
            # 住化dataでは、base種とbase濃度となっているが、説明変数としてはbase種として数値が濃度となる(250224)
            process_list = ['前処理還元炉温℃', '評価反応炉温℃', 'temp', 'flow_base','flow_slurry', 'flow_red', 'wash','前担持','pressure','conc_ETA','ターゲット担持量',
                            'support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100','base_NaOH','base_LiOH','base_KOH','base_Na2CO3','base_urea']
            metal_ratio = ['metal1','metal2','metal3','metal4','metal5','ratio1','ratio2','ratio3','ratio4','ratio5']      
            # 逆解析データdf
            x_data_for_inverse_analysis = pd.DataFrame(index=[f'{i}' for i in range(csv_output_count)], columns=x_data.columns)
            x1_metaldesc = pd.DataFrame(index=[f'{i}' for i in range(csv_output_count)], columns=x_data.columns)
            if (x_name == 'x26_NPA_and_x26_ETA') or (x_name == 'x2_17_and_x2_20') or (x_name == 'x2_18_and_x2_17'):
                x_data_for_inverse_analysis_part2 = pd.DataFrame(index=[f'{i}' for i in range(csv_output_count)], columns=x_data_2.columns) 
        
            # 逆解析金属組成生成
            synthesis_metal_x_data = pd.read_excel('results/synthesis_metal_x_data.xlsx', index_col=0, header=0, engine='openpyxl') # 金属組成データ
            xenonpy_merge = pd.read_csv('results/xenonpy_element_data240515.csv', index_col=0)          
            #xenonpy_merge.drop('oxide', axis= 1, inplace= True)
            # df作成(逆解析用金属組成データ)
            inverse_metal_x_data = pd.DataFrame(index=range(csv_output_count), columns=synthesis_metal_x_data.columns) 
            # 組成金属の割合
            sosei_list = ['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5'] 
            
            # 逆解析用の金属組成データ作成
            mi_util = MIutility() #データ作成クラス生成
            #束縛条件で分岐
            if constraints == 'False':
                if catalyst == 'normal':
                    #束縛条件:なしの場合
                    #  使用元素リスト最大5つ
                    #  組成比が合計100%となるように選出（表示は0~0.99)
                    atom_list = list(pd.read_csv('results/20230316_使用元素リスト.csv').columns) # 元素リスト# 組成金属の割合
                    # 元素数の選択
                    #atom_nums = np.random.randint(2, 5+1, (csv_output_count,))
                    atom_nums = np.full((csv_output_count,), 5)
                    # 元素数に応じた元素の選択
                    inverse_metal_x_data.iloc[:, :5] = pd.DataFrame([random.sample(atom_list, x) for x in atom_nums]).values 
                    # 各組成の計算 : 組成は、下限0.01、上限0.99、間隔0.01　で作成する
                    inverse_metal_x_data = mi_util.get_calcInverMetalData(inverse_metal_x_data,atom_nums,sosei_list)
                elif catalyst == 'simple':
                    atom_list = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt', 'Au', 'In', 'Sn'] # 元素リスト
                    # 元素数の選択
                    atom_nums = np.full((csv_output_count,), 3)
                    # 元素数に応じた元素の選択
                    inverse_metal_x_data.iloc[:, :3] = pd.DataFrame([random.sample(atom_list, x) for x in atom_nums]).values 
                    # 各組成の計算 : 組成は、下限0.1、上限0.9、間隔0.1　で作成する
                    inverse_metal_x_data = mi_util.get_calcInverMetalData_Simple(inverse_metal_x_data,atom_nums,sosei_list)                    
                
            elif constraints == 'True':
                #束縛条件:ありの場合
                # Li, K, Rb, Cs, Tiが消えていることに注意
                atom_list = ['Rh','Fe','Mn','Mg','Al','Ca','Sc','V','Cr','Co','Ni','Cu','Zn','Ga','Se','Sr','Y','Zr','Mo','Ru','Pd','In',\
                    'Sn','Te','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Yb','Lu','Hf','W','Ir','Pt','Au','Pb','Re']
                    
                def generate_sample(atom_list):
                    # 1) 必須元素
                    required_metals = ['Rh', 'Fe', 'Mn']
                    # まず、必須元素 'Rh','Fe','Mn' とその他からランダムに2種を選択（初期候補5種）
                    other_candidates = list(set(atom_list) - set(required_metals))
                    selected_others = np.random.choice(other_candidates, size=2, replace=False).tolist()
                    metals = ['Rh', 'Fe', 'Mn'] + selected_others  # 例：['Rh', 'Fe', 'Mn', 'Pt', 'Pd']
                    # 0.01刻みの組成を、5元素で生成（np.random.multinomial で100点を分配）
                    p = np.random.dirichlet(np.ones(5))
                    counts = np.random.multinomial(100, p)  # 合計100の整数
                    ratios = counts / 100.0  # 0.01刻みの比率
                    # 条件チェック（必須条件）
                    # 1. Rh (index0) の割合が 0.05～0.7
                    if not (0.05 <= ratios[0] <= 0.7):
                        return None
                    # 2. Fe (index1) と Mn (index2) はそれぞれ 0～0.5
                    if not (ratios[1] <= 0.5 and ratios[2] <= 0.5):
                        return None
                    # 3. Fe+Mn が 0.05～0.7（両方とも 0 でなければチェック）
                    if not (0.05 <= (ratios[1] + ratios[2]) <= 0.7):
                        return None                    

                    # --- 追加処理: FeまたはMnが 0 の場合、補充する ---
                    # Fe の index = 1, Mn の index = 2
                    missing_indices = []
                    if ratios[1] == 0:
                        missing_indices.append(1)
                    if ratios[2] == 0:
                        missing_indices.append(2)
                    
                    if missing_indices:
                        # 現在の候補から、該当インデックスの元素と比率を除外
                        metals_keep = [m for i, m in enumerate(metals) if i not in missing_indices]
                        ratios_keep = [r for i, r in enumerate(ratios) if i not in missing_indices]
                        current_sum = sum(ratios_keep)
                        # 補充すべき比率の合計
                        #remaining_ratio = 1 - current_sum
                        # 補充する元素の数
                        num_missing = len(missing_indices)
                        # 補充する各元素は最低 0.01 以上でなければならない
                        #if remaining_ratio < 0.01 * num_missing:
                        #    return None  # 補充不可能ならサンプル無効

                        # remaining_ratio を 0.01刻みの整数値に変換
                        #total_missing_counts = int(round(remaining_ratio * 100))
                        # 各新元素に最低 1 を割り当て、残りをランダムに分割する
                        base = [1] * num_missing
                        #remaining_counts = total_missing_counts - num_missing
                        #if remaining_counts > 0:
                        #    # ランダムに分割（多項分布を利用）
                        #    extra = np.random.multinomial(remaining_counts, [1/num_missing]*num_missing)
                        #    added_counts = [base[i] + extra[i] for i in range(num_missing)]
                        #else:
                        added_counts = base
                        added_ratios = [c / 100.0 for c in added_counts]
                        # 追加する元素は、既に採用している元素以外から選ぶ
                        available = list(set(atom_list) - set(metals_keep))
                        if len(available) < num_missing:
                            return None
                        new_metals = np.random.choice(available, size=num_missing, replace=False).tolist()
                        # 追加した元素の比率が 0.01 以上になる（すでに added_ratios は最低1/100 なのでOK）
                        # 最終的な候補リストは、元々の metals_keep に新たな元素を追加して合計5元素にする
                        metals = metals_keep + new_metals
                        ratios = ratios_keep + added_ratios
                        # ※順序は特に問わなければこのままでよいが、必要に応じて並び替え可能です

                    # ここで、最終的な metals, ratios の長さは必ず 5 になっているはず
                    if len(metals) != 5 or len(ratios) != 5:
                        return None
                    # また、各比率は 0.01 以上（補充した元素は確実に >=0.01）
                    if any(r < 0.01 for r in ratios):
                        return None

                    # DataFrame用の辞書を作成（列名：metal1～metal5, ratio1～ratio5）
                    data = {}
                    for i, (m, r) in enumerate(zip(metals, ratios), start=1):
                        data[f'metal{i}'] = m
                        data[f'ratio{i}'] = r
                    return data

                # 5) N個のデータを生成
                N = inverse_metal_x_data.shape[0]  
                samples = []
                while len(samples) < N:
                    sample = generate_sample(atom_list)
                    if sample is not None:
                        samples.append(sample)

                # DataFrame化
                inverse_metal_x_data = pd.DataFrame(samples)
                # 列名の修正
                inverse_metal_x_data = inverse_metal_x_data[[f'metal{i}' for i in range(1, 6)] + [f'ratio{i}' for i in range(1, 6)]]


            elif constraints == 'SCC1':  #randomと、Rhが0.1-0.7を作成する
                #metal_array1と2の数を決める
                metal_array1_number = int(csv_output_count * 0.8) #random部分の比率
                metal_array2_number = csv_output_count - metal_array1_number
                metal_array1 = inverse_metal_x_data.iloc[: metal_array1_number, :]
                metal_array2 = inverse_metal_x_data.iloc[metal_array1_number: , :]
                #束縛条件:なしの場合
                atom_list = list(pd.read_csv('results/20230316_使用元素リスト.csv').columns) # 元素リスト# 組成金属の割合
                # 元素数の選択
                atom_nums1 = np.full((metal_array1_number,), 5)
                # 元素数に応じた元素の選択
                metal_array1.iloc[:, :5] = pd.DataFrame([random.sample(atom_list, x) for x in atom_nums1]).values 
                # 各組成の計算 : 組成は、下限0.01、上限0.99、間隔0.01　で作成する
                metal_array1 = mi_util.calc_MetalRatio(metal_array1, atom_nums1,sosei_list)
                #cols = ['ratio1','ratio2','ratio3','ratio4','ratio5']
                #metal_array1['sum'] = metal_array1[cols].sum(axis= 1)
                
                #Rh>=10%のデータ作成
                atom_nums2 = np.full((metal_array2_number,), 4)
                atom_list2 = atom_list
                atom_list2.remove('Rh')
                metal_array2.iloc[:, :4] = pd.DataFrame([random.sample(atom_list2, x) for x in atom_nums2]).values 
                # 各組成の計算 : 組成は、下限0.01、上限0.99、間隔0.01　で作成する
                metal_array2 = mi_util.calc_MetalRatio(metal_array2, atom_nums2,sosei_list)
                metal_array2['metal5'] = 'Rh'
                metal_array2['ratio5'] = np.random.choice(np.arange(0.1, 0.7, 0.01), size= len(metal_array2))
                #Rhの値はそのままで、全体が1になるように調整
                cols = ['ratio1','ratio2','ratio3','ratio4']
                data_to_scale = metal_array2[cols].values
                target_sums = 1 - metal_array2['ratio5'].values 
                current_sums = data_to_scale.sum(axis= 1)
                scaling_factors = target_sums / current_sums
                scaled_data = data_to_scale * scaling_factors[:, np.newaxis]
                rounded_data = np.round(scaled_data / 0.01) * 0.01
                for i in range(len(metal_array2)):
                    if np.sum(rounded_data[i]) + metal_array2.at[metal_array1_number + i, 'ratio5'] != 1:
                        # 合計が小さい場合、最小の値を0.01増やす
                        if np.sum(rounded_data[i]) + metal_array2.at[metal_array1_number + i, 'ratio5'] < 1:
                            min_index = np.argmin(rounded_data[i])
                            rounded_data[i][min_index] += 0.01
                        # 合計が大きい場合、最大の値を0.01減らす
                        else:
                            max_index = np.argmax(rounded_data[i])
                            rounded_data[i][max_index] -= 0.01
                        rounded_data[i] = np.round(rounded_data[i] / 0.01) * 0.01                
                metal_array2[cols] = rounded_data
                #metal_array1と2の合体
                inverse_metal_x_data = pd.concat([metal_array1, metal_array2])
                
            elif constraints == 'matlantis_dft':
                # 浮動小数点の精度を設定
                getcontext().prec = 28

                n = csv_output_count
                ratio_condition = 0.2  # 条件1の割合
                atom_list_mat = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ir', 'Pt', 'Au', 'V', 'Mo', 'W']
                atom_list_dft = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Ru', 'Pd', 'Ir', 'Pt', 'Au', 'V']
                # atom_listの切替え
                if x_name == 'xx2_mat1_b9':
                    atom_list = atom_list_mat
                elif x_name == 'x2_dft1_b9':
                    atom_list = atom_list_dft

                # 条件ごとの行数を計算
                num_condition1 = int(n * ratio_condition)
                num_condition2 = n - num_condition1

                # 結果を格納するリストを初期化
                metal_data = {
                    'metal1': [], 'metal2': [], 'metal3': [], 'metal4': [], 'metal5': [],
                    'ratio1': [], 'ratio2': [], 'ratio3': [], 'ratio4': [], 'ratio5': []
                }

                # 'Rh'を除いた元素リスト
                element_list_no_Rh = [elem for elem in atom_list if elem != 'Rh']

                # 条件1のデータ生成（Rhが0.2以上0.7以下）
                for _ in range(num_condition1):
                    # 他の4つの元素をランダムに選択
                    metals = random.sample(element_list_no_Rh, 4)
                    # 'Rh'を追加
                    metals.append('Rh')
                    
                    # 各元素に最低1ユニットを割り当てるため、最初に1ユニットずつ割り当てる
                    min_units_per_element = 1
                    total_min_units = min_units_per_element * 5  # 5元素分の最低ユニット数

                    # 'Rh'の比率を0.2以上0.7以下に設定
                    Rh_min_units = max(int(0.2 * 100), min_units_per_element)  # 最低ユニット数を考慮
                    Rh_max_units = min(int(0.7 * 100), 100 - (total_min_units - min_units_per_element))
                    Rh_units = random.randint(Rh_min_units, Rh_max_units)
                    remaining_units = 100 - Rh_units - (total_min_units - min_units_per_element)

                    # 他の元素の比率をランダムに割り当て
                    other_elements_count = 4
                    other_units = np.random.multinomial(remaining_units, [1/other_elements_count]*other_elements_count)
                    other_units = [int(unit) + min_units_per_element for unit in other_units]
                    # 'Rh'のユニット数を追加
                    units = other_units + [Rh_units]

                    # 金属とユニット数を一緒に組み合わせ
                    combined = list(zip(metals, units))
                    # 元素名でアルファベット順にソート
                    combined.sort(key=lambda x: x[0])
                    metals, units = zip(*combined)
                    
                    # Decimalを使用して比率を計算
                    ratios = [Decimal(unit) / Decimal('100') for unit in units]
                    # 合計が1になるように最後の比率を調整
                    total_ratio = sum(ratios)
                    ratios = list(ratios)
                    ratios[-1] += Decimal('1.0') - total_ratio

                    # 各比率が0.01以上であることを確認
                    for i in range(len(ratios)):
                        if ratios[i] < Decimal('0.01'):
                            diff = Decimal('0.01') - ratios[i]
                            ratios[i] = Decimal('0.01')
                            # 他の比率から差分を引く
                            ratios_to_adjust = [j for j in range(len(ratios)) if j != i]
                            adjustment = diff / len(ratios_to_adjust)
                            for j in ratios_to_adjust:
                                ratios[j] -= adjustment
                                if ratios[j] < Decimal('0.01'):
                                    ratios[j] = Decimal('0.01')
                            # 再度合計が1になるように調整
                            total_ratio = sum(ratios)
                            ratios[-1] += Decimal('1.0') - total_ratio
                            break  # 調整が終わったらループを抜ける

                    # データを追加
                    for i in range(5):
                        metal_data[f'metal{i+1}'].append(metals[i])
                        metal_data[f'ratio{i+1}'].append(float(ratios[i]))

                # 条件2のデータ生成（完全ランダム）
                for _ in range(num_condition2):
                    metals = random.sample(atom_list, 5)

                    # 各元素に最低1ユニットを割り当てる
                    min_units_per_element = 1
                    total_min_units = min_units_per_element * 5

                    remaining_units = 100 - total_min_units

                    other_units = np.random.multinomial(remaining_units, [1/5]*5)
                    units = [int(unit) + min_units_per_element for unit in other_units]

                    # 金属とユニット数を一緒に組み合わせ
                    combined = list(zip(metals, units))
                    # 元素名でアルファベット順にソート
                    combined.sort(key=lambda x: x[0])
                    metals, units = zip(*combined)

                    # Decimalを使用して比率を計算
                    ratios = [Decimal(unit) / Decimal('100') for unit in units]
                    # 合計が1になるように最後の比率を調整
                    total_ratio = sum(ratios)
                    ratios = list(ratios)
                    ratios[-1] += Decimal('1.0') - total_ratio

                    # 各比率が0.01以上であることを確認
                    for i in range(len(ratios)):
                        if ratios[i] < Decimal('0.01'):
                            diff = Decimal('0.01') - ratios[i]
                            ratios[i] = Decimal('0.01')
                            # 他の比率から差分を引く
                            ratios_to_adjust = [j for j in range(len(ratios)) if j != i]
                            adjustment = diff / len(ratios_to_adjust)
                            for j in ratios_to_adjust:
                                ratios[j] -= adjustment
                                if ratios[j] < Decimal('0.01'):
                                    ratios[j] = Decimal('0.01')
                            # 再度合計が1になるように調整
                            total_ratio = sum(ratios)
                            ratios[-1] += Decimal('1.0') - total_ratio
                            break  # 調整が終わったらループを抜ける

                    # データを追加
                    for i in range(5):
                        metal_data[f'metal{i+1}'].append(metals[i])
                        metal_data[f'ratio{i+1}'].append(float(ratios[i]))

                # データフレームを作成
                inverse_metal_x_data = pd.DataFrame(metal_data)
                
            # inverse_metal_x_data整形(float型に変換、Noneをnp.nanに変換)
            #inverse_metal_x_data[sosei_list] = inverse_metal_x_data[sosei_list].astype(float)
            #inverse_metal_x_data = inverse_metal_x_data.replace(['', None], np.nan)
            
            #ratio1～ratio5の合計が「1.00」になっていない行を修正する
            if catalyst == 'normal':
                inverse_metal_x_data = mi_util.checkMetalRatioTotal(inverse_metal_x_data)
            
            #alphabet順に入れ替え
            inverse_metal_x_data = mi_util.checkMetalAlphabetOrder(inverse_metal_x_data)
            inverse_metal_x_data = inverse_metal_x_data.replace(['', None, 'nan'], np.nan)
                    
            # 条件設定  250224改訂
            inverse_analysis_condition_dict = {
                '前処理還元炉温℃':['list', [350, 400, 450]],
                '評価反応炉温℃':['list', [260]],    
                'temp':['list', [250, 350, 380]],
                'flow_base':['list', [0.1, 1, 5, 10]],
                'flow_slurry':['list', [10, 20, 30]],
                'flow_red':['list', [50, 75, 100]],
                'wash':['list', [1]],
                'pressure':['list', [25, 30, 35]],
                'conc_ETA':['list', [25]],
                'base_NaOH':['list', [0, 2.5]],
                'base_LiOH':['list', [0, 2.5]],
                'base_KOH':['list', [0, 2.5]],
                'base_Na2CO3':['list', [0, 2.5]],  # baseの合計=2.5の制約必要
                'base_urea':['list', [0, 2.5]],
                'ターゲット担持量':['list', [1, 3, 5, 10]],
                '前担持':['list', [1]],
                #'support':['list', ['support_Al2O3_A-11']],
                'support':['list', ['support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100']],
                }    
            for key in inverse_analysis_condition_dict.keys():
                condition_list = inverse_analysis_condition_dict[key]
                if condition_list[0] == 'list':
                    x_list = condition_list[1]
                    x_data_for_inverse_analysis[key] = random.choices(x_list, k=csv_output_count)
                elif condition_list[0] == 'one':
                    x_data_for_inverse_analysis[condition_list[1]] = condition_list[2]
            
            # 担体列をdummy変数に変換する
            tantai_col = x_data_for_inverse_analysis['support'].unique().tolist()
            for each in tantai_col:
                x_data_for_inverse_analysis[each] = (x_data_for_inverse_analysis['support'] == each)*1
            x_data_for_inverse_analysis = x_data_for_inverse_analysis.drop(columns=['support'])
            x_data_for_inverse_analysis['support_CeO2_HS'].replace(np.nan, 0, inplace= True)
            x_data_for_inverse_analysis['support_TiO2_SSP-M'].replace(np.nan, 0, inplace= True)
            x_data_for_inverse_analysis['support_ZrO2_RC100'].replace(np.nan, 0, inplace= True)       
                 
            #(flow_red + flow_NaOH) / flow_slurry >= 8.09 のサンプルを削除
            threshold_of_flow = 8.09 
            result_value = (x_data_for_inverse_analysis['flow_red']+x_data_for_inverse_analysis['flow_base']) / x_data_for_inverse_analysis['flow_slurry']
            del_sample_numbers = np.where(result_value >= threshold_of_flow)[0]
            x_data_for_inverse_analysis = x_data_for_inverse_analysis.drop(x_data_for_inverse_analysis.index[del_sample_numbers], axis=0)
            inverse_metal_x_data = inverse_metal_x_data.drop(inverse_metal_x_data.index[del_sample_numbers], axis=0)        
            x_data_for_inverse_analysis.reset_index(drop=True, inplace=True)
            inverse_metal_x_data.reset_index(drop=True, inplace=True)
            
            #featurizers = MetalFeaturizers()
            featurizers2 = Calc_desc()
            def str_reverse(item):
                a = item.split(' * ')
                a.reverse()
                b = ' * '.join(a)
                return b
            
            if x_name == 'xx2_mat1_b9':
                x_data_for_inverse_analysis = featurizers2.getDesc_X2_calc_mtls('Type2', x_data_for_inverse_analysis, inverse_metal_x_data)
                
            elif x_name == 'x2_dft1_b9':
                x_data_for_inverse_analysis = featurizers2.getDesc_X2_calc_dft('Type2', x_data_for_inverse_analysis, inverse_metal_x_data)                
                                
            elif x_name == 'x2_41':
                if x_data_for_inverse_analysis.shape[0] != inverse_metal_x_data.shape[0]:
                    x_data_for_inverse_analysis = x_data_for_inverse_analysis.iloc[:inverse_metal_x_data.shape[0], :]
                
                x_data_for_inverse_analysis = featurizers2.getDesc_X2_18('Type2', x_data_for_inverse_analysis, inverse_metal_x_data)
                metal_data = inverse_metal_x_data    
                
            elif x_name == 'x1_av':
                if x_data_for_inverse_analysis.shape[0] != inverse_metal_x_data.shape[0]:
                    x_data_for_inverse_analysis = x_data_for_inverse_analysis.iloc[:inverse_metal_x_data.shape[0], :]
                
                x_data_for_inverse_analysis = featurizers2.CalcX1_av('Type2',x_data_for_inverse_analysis,inverse_metal_x_data)
                metal_data = inverse_metal_x_data                    
                
            elif x_name == 'x2_17_and_x2_20':
                x_data_for_inverse_analysis_part2 = pd.concat([x_data_for_inverse_analysis.loc[:, process_list], x_data_for_inverse_analysis_part2.drop(process_list, axis= 1).reindex(x_data_for_inverse_analysis.index)], axis= 1, join= 'inner')
                if x_data_for_inverse_analysis.shape[0] != inverse_metal_x_data.shape[0]:
                    x_data_for_inverse_analysis = x_data_for_inverse_analysis.iloc[:inverse_metal_x_data.shape[0], :]
                
                x_data_for_inverse_analysis = featurizers2.getDesc_X2_20('Type2', x_data_for_inverse_analysis, inverse_metal_x_data)   #NPA用
                x_data_for_inverse_analysis_part2 = featurizers2.getDesc_X2_17('Type2', x_data_for_inverse_analysis_part2, inverse_metal_x_data)   #ETA用
                metal_data = inverse_metal_x_data
                x_data_for_inverse_analysis.dropna(how= 'all', inplace= True, axis= 1)  #Te列を削除
                x_data_for_inverse_analysis_part2.dropna(how= 'all', inplace= True, axis= 1)  #Te列を削除
                
            elif x_name == 'x2_18_and_x2_17':
                x_data_for_inverse_analysis_part2 = pd.concat([x_data_for_inverse_analysis.loc[:, process_list], x_data_for_inverse_analysis_part2.drop(process_list, axis= 1).reindex(x_data_for_inverse_analysis.index)], axis= 1, join= 'inner')
                if x_data_for_inverse_analysis.shape[0] != inverse_metal_x_data.shape[0]:
                    x_data_for_inverse_analysis = x_data_for_inverse_analysis.iloc[:inverse_metal_x_data.shape[0], :]
                
                x_data_for_inverse_analysis = featurizers2.getDesc_X2_18('Type2', x_data_for_inverse_analysis, inverse_metal_x_data)   #NPA用
                x_data_for_inverse_analysis_part2 = featurizers2.getDesc_X2_17('Type2', x_data_for_inverse_analysis_part2, inverse_metal_x_data)   #ETA用
                metal_data = inverse_metal_x_data
                #x_data_for_inverse_analysis.dropna(how= 'all', inplace= True, axis= 1)  #Te列を削除
                #x_data_for_inverse_analysis_part2.dropna(how= 'all', inplace= True, axis= 1)  #Te列を削除                
            
            drop_columns_list = ['ChemicalFormula', 'row_id']    
            if len(list(set(drop_columns_list) & set(inverse_metal_x_data.columns.tolist()))) != 0:
                drop_columns_list = list(set(drop_columns_list) & set(inverse_metal_x_data.columns.tolist()))
                inverse_metal_x_data = inverse_metal_x_data.drop(drop_columns_list, axis= 1)

            fname_process,fname_desc,fname_matminer,fname_metal_x = mi_util.createFileName(x_name , constraints, iter_n+1)
            # ファイル出力
            x_data_for_inverse_analysis[process_list].to_csv(f'{outputdir}/subdir/'+fname_process, encoding='cp932')
            if (x_name=='x26_NPA_and_x26_ETA') or (x_name=='x2_17_and_x2_20') or (x_name== 'x2_18_and_x2_17') is True:
                x_data_for_inverse_analysis.drop(columns=process_list).to_csv(f'{outputdir}/subdir/'+fname_desc, encoding='cp932')
                x_data_for_inverse_analysis_part2.drop(columns=process_list).to_csv(f'{outputdir}/subdir/'+fname_matminer, encoding='cp932')
            else:
                x_data_for_inverse_analysis.drop(columns=process_list).to_csv(f'{outputdir}/subdir/'+fname_desc, encoding='cp932')
            inverse_metal_x_data.to_csv(f'{outputdir}/subdir/'+fname_metal_x, encoding='cp932')
        
        #csvファイルのマージ
        merge_allFiles(x_name, constraints, mi_util, outputdir, iteration_number)
            
        # ファイル出力
        #x_data_for_inverse_analysis[['前処理還元炉温℃', '評価反応炉温℃', 'temp', 'flow_NaOH','flow_slurry', 'flow_red', 'wash', 'support_Al2O3_A-11','support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100']].to_csv(f'{outputdir}/{constraint}_x_data_for_inverse_analysis_exp_vals.csv', encoding='cp932')
        #inverse_metal_x_data.to_csv(f'{outputdir}/{constraint}_metal_x_data_for_inverse_analysis.csv', encoding='cp932')
        #print('case:'+constraint+' end')
        
    print('create samples end')
    
def merge_allFiles(x_name, constraints, mi_util, datadir, iteration_number):
    #マージ後のファイル名
    fname_process_0,fname_desc_0,fname_matminer_0,fname_metal_x_0 = mi_util.createFileName(x_name , constraints, 0)
    dataset0_x_data_exp = pd.DataFrame()
    dataset0_x_data_xenon = pd.DataFrame()
    dataset0_x_data_matminer = pd.DataFrame()
    dataset0_metal_x_data = pd.DataFrame()
    
    print('--- merge -------')
    for iter_n in range(iteration_number):
        print(f'iteration:{iter_n+1}')
    
        #ファイル名
        fname_x_data_exp,fname_x_data_xenon,fname_matminer,fname_metal_x_data = mi_util.createFileName(x_name , constraints, iter_n+1)
        
        #データ取得
        dataset_x_data_exp = pd.read_csv(f'{datadir}/subdir/'+fname_x_data_exp, encoding='cp932', index_col=0, header=0)  # データセットの読み込み
        if (x_name=='x26_NPA_and_x26_ETA') or (x_name=='x2_17_and_x2_20') or (x_name=='x2_18_and_x2_17') is True:
            dataset_x_data_xenon = pd.read_csv(f'{datadir}/subdir/'+fname_x_data_xenon, encoding='cp932', index_col=0, header=0)  # データセットの読み込み
            dataset_x_data_matminer = pd.read_csv(f'{datadir}/subdir/'+fname_matminer, encoding='cp932', index_col=0, header=0)  # データセットの読み込み     
        else:
            dataset_x_data_xenon = pd.read_csv(f'{datadir}/subdir/'+fname_x_data_xenon, encoding='cp932', index_col=0, header=0)  # データセットの読み込み
        dataset_metal_x_data = pd.read_csv(f'{datadir}/subdir/'+fname_metal_x_data, encoding='cp932', index_col=0, header=0)  # データセットの読み込み
    
        #マージ
        dataset0_x_data_exp = pd.concat([dataset0_x_data_exp, dataset_x_data_exp])  
        if (x_name=='x26_NPA_and_x26_ETA') or (x_name=='x2_17_and_x2_20') or (x_name=='x2_18_and_x2_17') is True:    
            dataset0_x_data_xenon = pd.concat([dataset0_x_data_xenon, dataset_x_data_xenon])
            dataset0_x_data_matminer = pd.concat([dataset0_x_data_matminer, dataset_x_data_matminer])
        else:
            dataset0_x_data_xenon = pd.concat([dataset0_x_data_xenon, dataset_x_data_xenon])
        dataset0_x_data_xenon = dataset0_x_data_xenon.dropna(how='any', axis=1)
        dataset0_metal_x_data = pd.concat([dataset0_metal_x_data, dataset_metal_x_data])
    
    #indexをリセット、ファイル出力
    dataset0_x_data_exp.reset_index(drop=True, inplace=True)
    dataset0_x_data_exp.to_csv(f'{datadir}/'+fname_process_0, encoding='utf-8-sig')   
    if (x_name=='x26_NPA_and_x26_ETA') or (x_name=='x2_17_and_x2_20') or (x_name=='x2_18_and_x2_17') is True:
        dataset0_x_data_xenon.reset_index(drop=True, inplace=True)
        dataset0_x_data_xenon.to_csv(f'{datadir}/'+fname_desc_0, encoding='cp932') 
        dataset0_x_data_matminer.reset_index(drop=True, inplace=True)
        dataset0_x_data_matminer.to_csv(f'{datadir}/'+fname_matminer_0, encoding='cp932')
    else:
        dataset0_x_data_xenon.reset_index(drop=True, inplace=True)
        dataset0_x_data_xenon.to_csv(f'{datadir}/'+fname_desc_0, encoding='utf-8-sig') 
    dataset0_metal_x_data.reset_index(drop=True, inplace=True) 
    dataset0_metal_x_data.to_csv(f'{datadir}/'+fname_metal_x_0, encoding='cp932')  

def createSaveDirSub(subdir):
    # delete
    if os.path.exists('result/virtual_data/'+subdir):
        shutil.rmtree('result/virtual_data/'+subdir)
    # create
    if not os.path.exists('result/virtual_data/'+subdir):
        os.mkdir('result/virtual_data/'+subdir)


def make_virtual_data():
    #やるべきこと：x_baseを元にしてdesc_NPA, desc_ETAにある記述子をx1_desc～x25_descを使って組み立てる
    #この際、NPAとETAは分けて出力する必要あり
    #230623 Teのmatminerデータについては、x1で予測するように変更する
    #matminerで計算可能なものもあるが、手順の簡素化のため全てx1で予測することとする Teの影響は小さいこともあり
    
    # 束縛条件
    #constraints = ['False','True']
    constraints = ['True']

    #データの種類
    dataset = ['x26_NPA2', 'x26_ETA3']
    
    #リストの定義
    process_cond = ['前処理還元炉温℃', '評価反応炉温℃', 'temp', 'flow_NaOH','flow_slurry', 'flow_red', 'wash', 'support_Al2O3_A-11','support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100']
    metal_ratio = ['metal1','metal2','metal3','metal4','metal5','ratio1','ratio2','ratio3','ratio4','ratio5']
    xmat_list = ['Miedema_deltaH_inter','Miedema_deltaH_amor','Miedema_deltaH_ss_min','Yang delta','Yang omega','APE mean','Radii local mismatch','Radii gamma','Configuration entropy',\
        'Lambda entropy','Electronegativity delta','Electronegativity local mismatch','VEC mean','Mixing enthalpy','Mean cohesive energy','Interant electrons','Interant s electrons',\
        'Interant p electrons','Interant d electrons','Interant f electrons','Shear modulus mean','Shear modulus delta','Shear modulus local mismatch','Shear modulus strength model']
    xmat2_list = ['mean AtomicWeight','mean Column','mean Row','range Number','mean Number','range AtomicRadius','mean AtomicRadius','range Electronegativity','mean Electronegativity',\
        'avg s valence electrons','avg p valence electrons','avg d valence electrons','avg f valence electrons','HOMO_energy','LUMO_energy','gap_AO','band center']
    
    #必要な関数の記載
    def desc_list(desc):
        x1_list = desc['0'][desc['0'].str.startswith(('hmean','gmean'))].values.tolist()
        x1_list_add = desc['0'][desc['0'].str.contains(('en_pauling|hhi_r|hht_p'))].values.tolist()
        x1_list = x1_list + x1_list_add            
        x10_list = desc['0'][desc['0'].str.startswith(('ave','var','min','max','div','subtr'))].values.tolist()
        x10_list = list(set(x10_list) - set(x1_list_add))
        
        x25_list = desc['0'][desc['0'].str.startswith(('group','compo'))].values.tolist()
        x14_list = desc['0'][desc['0'].str.startswith(('atomic','boiling','brinell','bulk','c6','covalent','density','dipole','electron',\
            'en','first','fusion','gs','hhi','heat','icsd','evaporation','gas','lattice','linear','mendeleev','melting','metallic','molar','num','period',\
                'poissons','proton','specific','thermal','vdw','sound','vickers','Polarizability','youngs','work'))].values.tolist()
        x22_list = desc['0'][desc['0'].str.startswith(('Miedema','mean','range','avg','band','Yang','APE','Radii','Configuration','Lambda',\
            'Electronegativity','VEC','Mixing','Mean','Interant','Shear','HOMO','LUMO','gap'))].values.tolist()
        return x1_list, x10_list, x14_list, x22_list, x25_list
    
    def split_descriptors(input_list):
        a = []
        for item in input_list:
            if '*' in item:
                b = item.split(' * ')
                a += b
            elif '^2' in item:
                c = item.rsplit('^2')
                a += c
            else:
                d = item
                a.append(d)
        a = set(a)
        a = sorted(list(a))
        if '' in a:
            a.remove('')
            
        return a

    #データの読込み 逆解析するためのデータとそのときのdescriptorsが必要。
    data_NPA = pd.read_csv('datasets/'+dataset[0]+'.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    data_ETA = pd.read_csv('datasets/'+dataset[1]+'.csv', encoding= 'utf-8-sig', index_col=0, header=0)  
    desc_NPA = pd.read_csv('datasets/descriptors_NPA.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    desc_ETA = pd.read_csv('datasets/descriptors_ETA.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    
    x_base = pd.read_csv('result/virtual_data/million_data/炉温300℃/'+constraints[0]+'_x_base.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    x1_desc = pd.read_csv('result/virtual_data/million_data/炉温300℃/'+constraints[0]+'_x1_desc.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    x10_desc = pd.read_csv('result/virtual_data/million_data/炉温300℃/'+constraints[0]+'_x10_desc.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    x14_desc = pd.read_csv('result/virtual_data/million_data/炉温300℃/'+constraints[0]+'_x14_desc.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    x22_desc = pd.read_csv('result/virtual_data/million_data/炉温300℃/'+constraints[0]+'_x22_desc.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    x25_desc = pd.read_csv('result/virtual_data/million_data/炉温300℃/'+constraints[0]+'_x25_desc.csv', encoding= 'utf-8-sig', index_col=0, header=0)
    
    #desc_listの取得
    x1_list_NPA, x10_list_NPA, x14_list_NPA, x22_list_NPA, x25_list_NPA = desc_list(desc_NPA)
    x1_list_ETA, x10_list_ETA, x14_list_ETA, x22_list_ETA, x25_list_ETA = desc_list(desc_ETA)
    matminer_list = x22_list_NPA + x22_list_ETA
    matminer_list = set(matminer_list)
    
    #データの縮小(最終的にはコメントアウト要)
    #x_base = x_base.iloc[:1000, :]
    #x1_desc = x1_desc.iloc[:1000, :]
    #x10_desc = x10_desc.iloc[:1000, :]
    #x14_desc = x14_desc.iloc[:1000, :]    
    #x22_desc = x22_desc.iloc[:1000, :]
    #x25_desc = x25_desc.iloc[:1000, :]    
    
    #Teを含むdfとそうでないdfに分割する
    Te_x_base = pd.concat([x_base[x_base['metal1']=='Te'], x_base[x_base['metal2']=='Te'], x_base[x_base['metal3']=='Te'],\
        x_base[x_base['metal4']=='Te'], x_base[x_base['metal5']=='Te']], axis= 0)  
    nonTe_x_base = x_base.query('index != @Te_x_base.index.to_list()')
    Te_x22_desc = x22_desc.query('index == @Te_x_base.index.to_list()')
    Te_x22_desc = Te_x22_desc[matminer_list]
    Te_x_base = Te_x_base.sort_index()
    
    #Teを含むmatminer_dfの要素をx1で予測する
    #下のmatminer_dictは41個全て。始めの24個(ShearModulusStrengthModelまで)がTeでnanを出力するもの。
    matminer_dict = {'Miedema_deltaH_inter':'GPR_10', 'Miedema_deltaH_amor':'GPR_10', 'Miedema_deltaH_ss_min':'GPR_6',
        'Yang delta':'GPR_6', 'Yang omega':'GPR_1', 'APE mean':'GPR_7', 'Radii local mismatch':'GPR_0',
        'Radii gamma':'GPR_0', 'Configuration entropy':'GPR_7', 'Lambda entropy':'GPR_7','Electronegativity delta':'GPR_2',
        'Electronegativity local mismatch':'GPR_8','VEC mean':'GPR_0', 'Mixing enthalpy':'GPR_6', 'Mean cohesive energy':'GPR_0',
        'Interant electrons':'GPR_5', 'Interant s electrons':'GPR_0', 'Interant p electrons':'GPR_0','Interant d electrons':'GPR_6',
        'Interant f electrons':'GPR_0', 'Shear modulus mean':'GPR_0', 'Shear modulus delta':'GPR_6', 'Shear modulus local mismatch':'GPR_6',
        'Shear modulus strength model':'GPR_6', 'mean AtomicWeight':'GPR_0', 'mean Column':'GPR_0', 'mean Row':'GPR_0', 'range Number':'GPR_0',
        'mean Number':'GPR_0', 'range AtomicRadius':'GPR_0', 'mean AtomicRadius':'GPR_0', 'range Electronegativity':'GPR_0',
        'mean Electronegativity':'GPR_0', 'avg s valence electrons':'GPR_0', 'avg p valence electrons':'GPR_0', 'avg d valence electrons':'GPR_0',
        'avg f valence electrons':'GPR_0', 'HOMO_energy':'GPR_0', 'LUMO_energy':'GPR_8', 'gap_AO':'GPR_0', 'band center':'GPR_6'}  
    matminer_current_dict = {key: value for key, value in matminer_dict.items() if key in matminer_list}   

    #xmatの読込み xmatはTeで予測不可能な24個のみ、x_mat2は予測可能な17個
    try:
        xmat = pd.read_csv('datasets/xmat.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
        xmat2 = pd.read_csv('datasets/xmat2.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
    except FileNotFoundError:
        print('datasetsフォルダにxmat.csvがありません。Teを含むデータで必要です。')    

    xmat_all = pd.concat([xmat[xmat_list], xmat2], axis= 1)
    metal_x = Te_x_base[metal_ratio]
    x1_desc_Te = CalcX1(metal_x)  #x1_descのみが戻る
    
    #x_train, y_trainの決定
    matminer_keys =list(matminer_dict.keys())
    matminer_current_keys = list(matminer_current_dict.keys())
    y_train = xmat_all.loc[:, matminer_current_keys].copy()
    x_train = xmat_all.drop(matminer_keys, axis= 1).copy()
    x_test = x1_desc_Te.copy()
    """
    # 同じ値を多く持つ候補を削除 逆解析時にautoscaleでnan出るとGPRでエラー出るため
    threshold_of_rate_of_same_value = 0.95
    rate_of_same_value = list()
    for X_variable_name in xmat_all.columns:
        same_value_number = xmat_all[X_variable_name].value_counts()
        rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / xmat_all.shape[0]))
    deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
    xmat_all = xmat_all.drop(xmat_all.columns[deleting_variable_numbers], axis=1)
    """
    # 同じ値を多く持つ候補を削除 逆解析時にautoscaleでnan出るとGPRでエラー出るため
    threshold_of_rate_of_same_value = 0.95
    rate_of_same_value = list()
    for X_variable_name in x_train.columns:
        same_value_number = x_train[X_variable_name].value_counts()
        rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_train.shape[0]))
    deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
    x_train = x_train.drop(x_train.columns[deleting_variable_numbers], axis=1)    
    
    #同じ値を多く持つ候補を削除 逆解析時にautoscaleでnan出るとGPRでエラー出るため
    threshold_of_rate_of_same_value = 0.95
    rate_of_same_value = list()
    for X_variable_name in x_test.columns:
        same_value_number = x_test[X_variable_name].value_counts()
        rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_test.shape[0]))
    deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
    x_test = x_test.drop(x_test.columns[deleting_variable_numbers], axis=1)        
    
    #説明変数がtrainとtestで同じとなっているか確認し、合わせる
    x_train_list = x_train.columns.to_list()
    x_test_list = x_test.columns.to_list()
    diff = set(x_train_list) - set(x_test_list)
    #xmat_all = xmat_all.drop(list(diff), axis= 1)
    x_train = x_train.drop(list(diff), axis= 1)
    print('x_trainの列を'+str(len(list(diff)))+'個削除しました。')
    drop_list = [s for s in list(diff) if s in matminer_keys]
    if len(drop_list) != 0:
        print('同じデータが多いため、'+",".join(drop_list)+'を削除しました 問題あれば閾値を変更するなど対応下さい。')
    #y_train = xmat_all.loc[:, list(matminer_dict.keys())]
    #x_train = xmat_all.drop(list(matminer_dict.keys()), axis= 1)
    diff1 = set(x_test_list) - set(x_train_list)
    x_test = x_test.drop(list(diff1), axis= 1)
    print('x_test(x1_desc)の列を'+str(len(list(diff1)))+'個削除しました。') 
    
    #autoscaling
    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0, ddof=1)
    autoscaled_x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0, ddof=1)
        
    #yをmatminer_dictから選択して予測する 用いるGPRもmatminer_dictに記載
    estimated_y_df = pd.DataFrame()
    for key in matminer_current_dict.keys():
        y_train_key = y_train[key]
        autoscaled_y_train_key = autoscaled_y_train[key]
        GPR_number = matminer_current_dict[key]
        GPR_number = GPR_number.lstrip('GPR_')

        #matminerのdescriptorsを予測する
        GPR_kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
                ConstantKernel() * RBF() + WhiteKernel(),
                ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * RBF(np.ones(autoscaled_x_train.shape[1])) + WhiteKernel(),
                ConstantKernel() * RBF(np.ones(autoscaled_x_train.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
                ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
                ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]

        regression_model = GaussianProcessRegressor(kernel=GPR_kernels[int(GPR_number)], alpha=0) # GPR モデルの宣言
        regression_model.fit(autoscaled_x_train, autoscaled_y_train_key)
        autoscaled_estimated_y, autoscaled_estimated_y_std = regression_model.predict(autoscaled_x_test, return_std=True)
        estimated_y = autoscaled_estimated_y * y_train_key.std(axis=0, ddof=1) + y_train_key.mean(axis=0)    
        estimated_y_df[y_train_key.name] = estimated_y
        print(key+'_end')    
        
    estimated_y_df.index = x_test.index
    Te_x22_desc_calculated = estimated_y_df
    x22_desc_excluded_Te = x22_desc.query('index != @Te_x22_desc_calculated.index.to_list()')
    x22_desc = pd.concat([x22_desc_excluded_Te, Te_x22_desc_calculated], axis= 0)
    x22_desc = x22_desc.sort_index()
    x22_desc = x22_desc[matminer_current_keys]
    
    #交差項が主体のところの処理
    #1)自乗項、交差項を分解して各descriptorsとする
    process = x_base.iloc[:, :11].columns.to_list()
    
    desc_list_NPA = data_NPA.iloc[:, 14:].columns.to_list()
    desc_list_ETA = data_ETA.iloc[:, 14:].columns.to_list()
    
    desc_list_NPA = split_descriptors(desc_list_NPA)
    desc_list_NPA = set(desc_list_NPA) - set(process)
    desc_list_NPA = sorted(list(desc_list_NPA))
    
    desc_list_ETA = split_descriptors(desc_list_ETA)
    desc_list_ETA = set(desc_list_ETA) - set(process)
    desc_list_ETA = sorted(list(desc_list_ETA))
    
    x_base_NPA = x_base.copy()
    x_base_ETA = x_base.copy()
    
    #2)x_baseの列に各descriptorsを入れる 
    for element in desc_list_NPA:
        if element in x1_list_NPA:
            x_base_NPA[element] = x1_desc[element]
        elif element in x10_list_NPA:
            x_base_NPA[element] = x10_desc[element]
        elif element in x14_list_NPA:
            x_base_NPA[element] = x14_desc[element]
        elif element in x22_list_NPA:
            x_base_NPA[element] = x22_desc[element]
        elif element in x25_list_NPA:
            x_base_NPA[element] = x25_desc[element]
        else:
            print(element+ 'の要素が一致しません。')
            
    for element in desc_list_ETA:
        if element in x1_list_ETA:
            x_base_ETA[element] = x1_desc[element]
        elif element in x10_list_ETA:
            x_base_ETA[element] = x10_desc[element]
        elif element in x14_list_ETA:
            x_base_ETA[element] = x14_desc[element]
        elif element in x22_list_ETA:
            x_base_ETA[element] = x22_desc[element]
        elif element in x25_list_ETA:
            x_base_ETA[element] = x25_desc[element]
        else:
            print(element+ 'の要素が一致しません。')
            
    #3)自乗項、交差項を再現する。
    NPA_column_list = data_NPA.iloc[:, 14:].columns.to_list()
    ETA_column_list = data_ETA.iloc[:, 14:].columns.to_list()
    process_list = x_base.iloc[:, :11].columns.to_list()
    
    for desc in NPA_column_list:
        if '*' in desc:
            a = desc.split(' * ')
            x_base_NPA[a[0]+' * '+a[1]] = x_base_NPA[a[0]] * x_base_NPA[a[1]]
        elif '^2' in desc:
            b = desc.rsplit('^2')
            x_base_NPA[b[0]+'^2'] = x_base_NPA[b[0]] **2
            
    for desc in ETA_column_list:
        if '*' in desc:
            a = desc.split(' * ')
            x_base_ETA[a[0]+' * '+a[1]] = x_base_ETA[a[0]] * x_base_ETA[a[1]]
        elif '^2' in desc:
            b = desc.rsplit('^2')
            x_base_ETA[b[0]+'^2'] = x_base_ETA[b[0]] ^2

    #4)余分な列を削除する。
    current_columns_NPA = x_base_NPA.iloc[:, 21:].columns.to_list()
    drop_list_NPA = set(current_columns_NPA) - set(NPA_column_list)
    x_base_NPA.drop(drop_list_NPA, axis= 1, inplace= True)
    
    current_columns_ETA = x_base_ETA.iloc[:, 21:].columns.to_list()
    drop_list_ETA = set(current_columns_ETA) - set(ETA_column_list)
    x_base_ETA.drop(drop_list_ETA, axis= 1, inplace= True)
      
    #生成したvirtual datasetの保存
    x26_x_data_exp = x_base_NPA.loc[:, process_cond]
    x26_metal_x_data = x_base_NPA.loc[:, metal_ratio]
    x26_NPA_desc = x_base_NPA.loc[:, NPA_column_list]
    x26_ETA_desc = x_base_ETA.loc[:, ETA_column_list]

    #x26_ETA_descにnanが混じることがあったため、それを削除する
    if x26_ETA_desc.isnull().any().sum() != 0:
        x26_ETA_desc = x26_ETA_desc[~x26_ETA_desc.isnull().any(axis= 1)]
        x26_NPA_desc = x26_NPA_desc.query('index == @x26_ETA_desc.index.to_list()')
        x26_metal_x_data = x26_metal_x_data.query('index == @x26_ETA_desc.index.to_list()')
        x26_x_data_exp = x26_x_data_exp.query('index == @x26_ETA_desc.index.to_list()')
    
    x26_x_data_exp.to_csv('result/virtual_data/virtual_data_for_inverse_analysis/炉温300℃/x26_'+constraints[0]+'_x_data_exp.csv', encoding= 'utf-8-sig')
    x26_metal_x_data.to_csv('result/virtual_data/virtual_data_for_inverse_analysis/炉温300℃/x26_'+constraints[0]+'_metal_x_data.csv', encoding= 'utf-8-sig')
    x26_NPA_desc.to_csv('result/virtual_data/virtual_data_for_inverse_analysis/炉温300℃/x26_'+constraints[0]+'_NPA_desc.csv', encoding= 'utf-8-sig')
    x26_ETA_desc.to_csv('result/virtual_data/virtual_data_for_inverse_analysis/炉温300℃/x26_'+constraints[0]+'_ETA_desc.csv', encoding= 'utf-8-sig')
       
if __name__ == '__main__' :
    if selector == 'make_data_without_database':
        make_inverse_analysis_data()       
    elif selector == 'make_data_using_database':
        make_virtual_data()