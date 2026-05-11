
import math
import sys
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#metal_xの列名を修正する
def metal_x_columns_name(df):
    # 列数が偶数かどうか確認
    if len(df.columns) % 2 != 0:
        raise ValueError("metal_xの列数は偶数です。")
    # 列数の半分
    n = len(df.columns) // 2
    # 新しい列名の生成
    new_columns = [f'metal{i+1}' if i < n else f'ratio{i-n+1}' for i in range(2*n)]
    # DataFrameの列名の更新
    df.columns = new_columns
    return df   

#metal_xの列順を組成順に入替える
def sort_metals_ratios(row):
    # メタルと比率の列名を動的に特定
    metal_cols = [col for col in row.index if 'metal' in col]
    ratio_cols = [col for col in row.index if 'ratio' in col]
    
    # メタルと比率の列をペアで取得
    pairs = list(zip(row[metal_cols], row[ratio_cols]))
    # 比率に従って降順にソート
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    # ソートされた値を分解して新しい行に割り当て
    sorted_metals, sorted_ratios = zip(*sorted_pairs)
    # 新しい行をSeriesとして返す
    return pd.Series(sorted_metals + sorted_ratios)

#metal_x形式をx2形式へ変換する関数
def transform_x2(df):
    # Cation1, Cation2, Cation3 およびそれらの比率の列を抽出
    cation_columns = [s for s in df.columns if s.startswith('Cation')]
    cation_name = [s for s in cation_columns if not 'Amount' in s]
    cation_df = df[cation_columns]
    # 元素名の集合を取得
    elements = set(list(cation_df[cation_name].values.flatten()))
    elements.discard('na')  # 'na' を取り除く
    elements = sorted(list(elements))
    # 新しいデータフレームを初期化
    new_df = pd.DataFrame(0, index=cation_df.index, columns=elements)

    # 各行について、元素比率を新しいデータフレームに代入
    for i, row in cation_df.iterrows():
        if row['Cation1'] != 'na':
            new_df.at[i, row['Cation1']] = row['Cation1Amount']
        if row['Cation2'] != 'na':
            new_df.at[i, row['Cation2']] = row['Cation2Amount']
        if row['Cation2'] == row['Cation1']:
            new_df.at[i, row['Cation1']] = row['Cation1Amount'] + row['Cation2Amount']   
        if row['Cation3'] != 'na':
            new_df.at[i, row['Cation3']] = row['Cation3Amount']
    # 元のデータフレームから Cation1-Cation3 と Cation1Amount-Cation3Amount の列を削除
    df.drop(columns=cation_columns, inplace=True)
    # 新しいデータフレームを元のデータフレームに統合
    result_df = pd.concat([df, new_df], axis=1)

    return result_df

#上の逆関数、x2形式をmetal_x形式に変換する関数
def transform_metal_x(df):
    # 元素名の列のみを抽出
    elements = [col for col in df.columns if not col.startswith('Cation') and not col.endswith('Amount')]
    # 各行で非ゼロの元素の数を計算し、その最大値を取得
    max_elements = df[elements].astype(bool).sum(axis=1).max()
    # 動的にカラム名を生成
    cation_columns = [f'Cation{i+1}' for i in range(max_elements)]
    amount_columns = [f'Cation{i+1}Amount' for i in range(max_elements)]
    original_columns = cation_columns + amount_columns
    # 新しいデータフレームを初期化
    original_df = pd.DataFrame(index=df.index, columns=original_columns)
    
    # 元素名の列からデータを再構成
    for i, row in df.iterrows():
        non_zero_elements = row[elements][row[elements] > 0].sort_values(ascending=False)
        for j, (element, amount) in enumerate(non_zero_elements.items()):
            if j < max_elements:
                original_df.at[i, f'Cation{j+1}'] = element
                original_df.at[i, f'Cation{j+1}Amount'] = amount
    # 元のデータフレームに結合
    remaining_df = df.drop(columns=elements)
    result_df = pd.concat([remaining_df, original_df], axis=1)
    
    return result_df

def transform_x2modily(df):
    #列名の修正
    df_columns = df.columns.to_list()
    df.reset_index(drop= True, inplace= True)
    new_df_columns = [s.replace('mol%', 'Amount') if 'mol%' in s else s for s in df_columns]
    new_df_columns = [s.replace(' ', '') if('Cation' in s or 'Anion' in s or 'Support' in s) and ' ' in s else s for s in new_df_columns]
    df.columns = new_df_columns
    # Cation1, Cation2, Cation3 およびそれらの比率の列を抽出
    cation_columns = [s for s in df.columns if s.startswith('Cation')]
    cation_name = [s for s in cation_columns if not 'Amount' in s]
    anion_columns = [s for s in df.columns if s.startswith('Anion')]
    anion_name = [s for s in anion_columns if not 'Amount' in s]
    support_columns = [s for s in df.columns if s.startswith('Support')]
    support_name = [s for s in support_columns if not 'Amount' in s]    
    cation_df = df[cation_columns]
    anion_df = df[anion_columns]
    support_df = df[support_columns]
    # 元素名の集合を取得
    cation_elements = set(list(cation_df[cation_name].values.flatten()))
    cation_elements = {x for x in cation_elements if x != 'na' and not (isinstance(x, float) and np.isnan(x))}
    cation_elements = sorted(list(cation_elements))
    anion_elements = set(list(anion_df[anion_name].values.flatten()))
    anion_elements = {x for x in anion_elements if x != 'na' and not (isinstance(x, float) and np.isnan(x))}
    anion_elements = sorted(list(anion_elements))
    large_elements = set(list(support_df[support_name].values.flatten()))
    large_elements = {x for x in large_elements if x != 'na' and not (isinstance(x, float) and np.isnan(x))}
    large_elements = sorted(list(large_elements))    
    cation_name_columns = ['cation_' + s for s in cation_elements]
    anion_name_columns = ['anion_' + s for s in anion_elements]
    support_name_columns = ['support_' + s for s in large_elements]
    # 新しいデータフレームを初期化
    new_df = pd.DataFrame(0, index=cation_df.index, columns=cation_name_columns+anion_name_columns+support_name_columns)
    # 各行について、元素比率を新しいデータフレームに代入
    def add_cation_amount(row):
        for i in range(1, 7):
            cation = row[f'Cation{i}']
            if cation in cation_elements:
                amount = row[f'Cation{i}Amount']
                col_name = 'cation_' + cation
                if col_name in new_df.columns:
                    if new_df.at[row.name, col_name] != 0:
                        new_df.at[row.name, col_name] += amount
                    else:
                        new_df.at[row.name, col_name] = amount
    def add_anion_amount(row):
        for j in range(1, 3):
            anion = row[f'Anion{j}']
            if anion in anion_elements:
                amount = row[f'Anion{j}Amount']
                col_name = 'anion_' + anion
                if col_name in new_df.columns:
                    if new_df.at[row.name, col_name] != 0:
                        new_df.at[row.name, col_name] += amount
                    else:
                        new_df.at[row.name, col_name] = amount
    def add_support_amount(row):
        for k in range(1, 4):
            support = row[f'Support{k}']
            if support in large_elements:
                amount = row[f'Support{k}Amount']
                col_name = 'support_' + support
                if col_name in new_df.columns:
                    if new_df.at[row.name, col_name] != 0:
                        new_df.at[row.name, col_name] += amount
                    else:
                        new_df.at[row.name, col_name] = amount                       
    # applyを用いて各行に更新関数を適用
    cation_df.apply(add_cation_amount, axis= 1)
    anion_df.apply(add_anion_amount, axis= 1)
    support_df.apply(add_support_amount, axis= 1)
    # 元のデータフレームから Cation1-Cation3 と Cation1Amount-Cation3Amount の列を削除
    df = df.drop(cation_columns+anion_columns+support_columns, axis= 1)
    result_df = pd.concat([df, new_df], axis=1)

    return result_df

def transform_x2_for_target(df, small_elements):
  #列名の修正
    df_columns = df.columns.to_list()
    df.reset_index(drop= True, inplace= True)
    new_df_columns = [s.replace('mol%', 'Amount') if 'mol%' in s else s for s in df_columns]
    new_df_columns = [s.replace(' ', '') if('Cation' in s or 'Anion' in s or 'Support' in s) and ' ' in s else s for s in new_df_columns]
    df.columns = new_df_columns
    # Cation1, Cation2, Cation3 およびそれらの比率の列を抽出
    cation_columns = [s for s in df.columns if s.startswith('Cation')]
    cation_name = [s for s in cation_columns if not 'Amount' in s]
    cation_df = df[cation_columns]
    # 元素名の集合を取得
    cation_elements = set(list(cation_df[cation_name].values.flatten()))
    cation_elements = {x for x in cation_elements if x != 'na' and not (isinstance(x, float) and np.isnan(x))}
    cation_elements = sorted(list(cation_elements))
    cation_name_columns = [s for s in cation_elements]
    # 新しいデータフレームを初期化
    new_df = pd.DataFrame(0, index=cation_df.index, columns=cation_name_columns)
    # 各行について、元素比率を新しいデータフレームに代入
    def add_cation_amount(row):
        for i in range(1, 7):
            cation = row[f'Cation{i}']
            if cation in cation_elements:
                amount = row[f'Cation{i}Amount']
                col_name = cation
                if col_name in new_df.columns:
                    if new_df.at[row.name, col_name] != 0:
                        new_df.at[row.name, col_name] += amount
                    else:
                        new_df.at[row.name, col_name] = amount
    # applyを用いて各行に更新関数を適用
    cation_df.apply(add_cation_amount, axis= 1)
    #Cationの種類が4以上のものは削除
    #new_df = new_df[(new_df != 0).sum(axis= 1) <= 3]
    #Cationの種類がtargetにないものも削除
    def check_row(row, small_elements):
        non_zero_col = row[row != 0].index
        return all(col in small_elements for col in non_zero_col)    
    new_df = new_df[new_df.apply(lambda row: check_row(row, small_elements), axis= 1)]
    new_df = new_df[small_elements]
    def normalize_row(row):
        non_zero_values = row[row != 0]
        sum_non_zero = non_zero_values.sum()
        normalized_values = non_zero_values / sum_non_zero
        row[non_zero_values.index] = normalized_values
        return row
    new_df = new_df.apply(normalize_row, axis= 1)    
    # 元のデータフレームから 不要列を削除し名前を更新する
    drop_list = cation_columns + [s for s in df.columns if 'Amount' in s]
    df = df.drop(drop_list, axis= 1)
    #df = df[['Y(C2), %', 'Anion1','Anion2','Promotor','Support1','Support2','Support3', 'Temperature, K']]
    df['Temperature, K'] = df['Temperature, K'] -273
    df = df.rename(columns= {'Y(C2), %': 'C2 yield', 'Temperature, K': 'Temp'})
    result_df = pd.merge(df, new_df, right_index= True, left_index= True)
    result_df = pd.get_dummies(result_df, columns= ['Anion1','Anion2','Promotor','Support1','Support2','Support3','Preparation'])
    nan_checker = result_df.isna().sum()
    inf_checker = result_df.applymap(np.isinf).sum()
    dummy_columns = [s for s in result_df.columns if 'Anion' in s or 'Promotor' in s or 'Support' in s or 'Preparation' in s]
    result_df[dummy_columns] = result_df[dummy_columns].astype(int)
    
    return result_df

def transform_x1(df, element_data, thermo_data, flag):
    if flag == 'support':
        df = df.reset_index(drop= True)
        df = df.dropna(subset= 'Cation 1')
    xenonpy_base = pd.merge(element_data, thermo_data, left_index= True, right_index= True)
    xenonpy_base = xenonpy_base.dropna(how= 'any', axis= 1)
    #相関の高い変数を削除する
    xenonpy_base = delete_high_corr(xenonpy_base, threshold= 0.95)
    #metal_xを組成順に並べる
    if flag == 'target':
        metal_x = df.loc[:, 'Cation1': 'Cation3Amount']
        metal_x = metal_x_columns_name(metal_x)
    elif flag == 'support':
        metal_x = df.loc[:, 'Cation 1': 'Cation 6 mol%']
        df_columns = [s.replace(' mol%', 'Amount') if 'mol%' in s else s for s in metal_x.columns]
        df_columns = [s.replace('Cation ', 'Cation') for s in df_columns if 'Cation ' in s]
        metal_x.columns = df_columns
        new_order = [f'Cation{i}' for i in range(1, 7)] + [f'Cation{i}Amount' for i in range(1, 7)] 
        metal_x = metal_x[new_order]
        metal_x = metal_x_columns_name(metal_x)
    new_columns = metal_x.columns.to_list()
    # applyを使って各行に対してソート関数を適用し、列名を更新
    metal_x[new_columns] = metal_x.apply(sort_metals_ratios, axis=1)
    metal_x.columns = new_columns    
    if xenonpy_base.isna().any().any():
        print('dataにnanが含まれています')
        sys.exit()
    #各レイヤーの作成
    weighted_average_name = list() # 加重平均の index 名
    weighted_variance_name = list() # 加重分散の index 名
    geometric_mean_name = list() # 幾何平均の index 名
    harmonic_mean_name = list() # 調和平均の index 名
    max_pooling_name = list() # 最大値の index 名
    min_pooling_name = list() # 最小値の index 名
    max_composition_name = list() #最大組成値の index 名
    for j in xenonpy_base.columns:
        weighted_average_name.append(f'ave_{j}')
        weighted_variance_name.append(f'var_{j}')
        geometric_mean_name.append(f'gmean_{j}')
        harmonic_mean_name.append(f'hmean_{j}')
        max_pooling_name.append(f'max_{j}')
        min_pooling_name.append(f'min_{j}')
        max_composition_name.append(f'maxcompo_{j}')

    x3_metaldesc = pd.DataFrame(
        index=metal_x.index, dtype= float,
        columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name+max_composition_name
        )
    for i in range(metal_x.shape[0]):
        # メタルと比率の列名を動的に特定
        metal_cols = [col for col in metal_x.columns if 'metal' in col]
        ratio_cols = [col for col in metal_x.columns if 'ratio' in col]
        # 各行からメタルと比率の値を取得
        metals = metal_x.loc[metal_x.index[i], metal_cols].values
        ratios = metal_x.loc[metal_x.index[i], ratio_cols].values
        # 無効なメタル（値が0または'na'）とそれに対応する比率を除外
        def is_invalid(value):
            try:
                return value in (0, 'na', None) or (isinstance(value, float) and (math.isnan(value) or np.isnan(value)))
            except ValueError:
                return False
        valid_indices = [index for index, metal in enumerate(metals) if not is_invalid(metal)]
        valid_metals = [metals[index] for index in valid_indices]
        valid_ratios = [ratios[index] for index in valid_indices]
        # xenonpy_baseから対応するメタルの記述子を取得し、NumPy配列に格納
        metal_descs = [xenonpy_base.loc[metal, :].values for metal in valid_metals]
        mt = np.array(metal_descs)
        mr = np.array(valid_ratios)

        for desc in range(xenonpy_base.shape[1]):
            d_name = xenonpy_base.columns[desc]
            x3_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
            x3_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)/np.sum(mr)
            x3_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
            x3_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
            x3_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
            x3_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])
            if len(mr) == 1:
                x3_metaldesc[f'maxcompo_{d_name}'].iloc[i] = max(mt[:, desc])
            elif mr[0] != mr[1]:
                x3_metaldesc[f'maxcompo_{d_name}'].iloc[i] = mt[0, desc]
            else:
                x3_metaldesc[f'maxcompo_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)      
    #descの値が負であるものはgmeanでnanとなるため、その列はdropする(負の値のものは計算しないようにするのはかなり面倒)
    x3_metaldesc = x3_metaldesc.dropna(how= 'any', axis= 1)
    if flag == 'target':      
        return_df = pd.concat([df[['C2 yield','Temp', 'Support']], x3_metaldesc], axis= 1)
    elif flag == 'support':
        return_df = pd.concat([df[['Y(C2), %','Temperature, K','Support 1','Support 2','Support 3']], x3_metaldesc], axis= 1)
        return_df = return_df.rename(columns= {'Y(C2), %': 'C2 yield', 'Temperature, K': 'Temp', 'Support 1': 'Support','Support 2':'Support2','Support 3':'Support3'})
    return return_df

#相関係数の大きい特徴量を削除する
def delete_high_corr(df, threshold):
    corr_matrix = df.corr().abs()
    upper_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k= 1).astype(np.bool_))
    drop_col = [col for col in upper_matrix.columns if any(upper_matrix[col] > threshold)]
    reduced_df = df.drop(drop_col, axis= 1)
    return reduced_df  

noise_ratio_in_simulation = 0.1
do_autoscaling = True  # True or False
threshold_of_rate_of_same_value = 0.99
fold_number = 5
max_pls_component_number = 30
                                            
# load data set target: small_data, support: large_data
large_data = pd.read_excel('MyWork/datasets/original_data/OCM.xlsx', sheet_name= 'OCM_Dataset_-2019', index_col= 0)
small_data = pd.read_csv('MyWork/original_data/Oxidative_coupling_of_methane_at_CADS.csv')
element_data = pd.read_csv('results/element_data_normal_string240527.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
thermo_data = pd.read_csv('results/thermo_data.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)

# small_dataとlarge_dataの被りの有無を確認→1つ1つ確認して重複なかった
large_data = large_data[large_data['Y(C2), %'].notna()]
#large_data = large_data[large_data['Cation 4'].isna() ]
#large_data = large_data[large_data['Temperature, K'].isin([873, 973, 1073, 1173])]
small_elements = small_data.loc[:, 'Cation1': 'Cation3'].values.flatten()
small_elements = list(set(small_elements))
small_elements = [s for s in small_elements if s != 'na']
small_elements.sort()
#small_data = small_data[small_data['Temp'].isin([700, 800])]
small_data = small_data.sort_values(by= 'Cation1')
small_data_columns = ['C2 yield'] + [s for s in small_data.columns if s != 'C2 yield']
small_data = small_data[small_data_columns]
#small_data = small_data.iloc[88: 116, :]
#list_a = ['Na','Mn','W','na']
#small_data = small_data[small_data['Cation1'].isin(list_a) & small_data['Cation2'].isin(list_a) & small_data['Cation3'].isin(list_a)]
#large_data = large_data[large_data['Cation 1'].isin(small_elements)]
#large_data = large_data[large_data['Cation 2'].isin(small_elements) | large_data['Cation 2'].isna()]
#large_data = large_data[(large_data['Support 1'] == 'Al') | (large_data['Support 1'] == 'Si') ]
large_data = large_data.dropna(how= 'all', axis= 1)
#large_data = pd.concat([large_data.loc[:, : 'Cation 3 mol%'], large_data.loc[:, 'Anion 1': ]], axis= 1)
large_data = large_data[large_data['Cation 2'].notna()]
large_data = large_data.sort_values(by= 'Cation 1')
large_data_columns = ['Y(C2), %'] + [s for s in large_data.columns if s != 'Y(C2), %']
large_data = large_data[large_data_columns]
#large_data = large_data[large_data['Y(C2), %'] < 12]
#large_data = large_data.iloc[8:, :]
large_data['Y(C2), %'] = large_data['Y(C2), %'].round(1)

# element_dataとthermo_dataの処理
large_elements = large_data.loc[:, ['Cation 1','Cation 2','Cation 3','Cation 4','Cation 5','Cation 6']].values.flatten()
large_elements = list(set(large_elements))
large_elements = [s for s in large_elements if isinstance(s, str)]
large_elements.sort()
thermo_data = thermo_data.loc[thermo_data.index.isin(large_elements)]
thermo_data = thermo_data.loc[:, : 'Stockmayer parameter, K']
thermo_data = thermo_data.drop(['CAS', 'Heat of formation, J/mol'], axis= 1)
thermo_data = thermo_data.loc[:, thermo_data.isnull().mean() < 0.2]
check_nan_b = thermo_data.isna().sum()
thermo_data = thermo_data.dropna(axis= 1)
element_data = element_data.loc[element_data.index.isin(large_elements)]
element_data = element_data.loc[:, element_data.isnull().mean() < 0.2]  #nanが20%未満を抽出
element_data = element_data.dropna(axis= 1)
#element_data = element_data.drop(['space group', 'structure','oxide','hhi_p','hhi_r'], axis= 1)
element_data = delete_high_corr(element_data, threshold= 0.95)
thermo_data = delete_high_corr(thermo_data, threshold= 0.95)

# target_dataを展開 列が同一である必要あるため、targetをベースにしてsupportを変形させる
large_data = large_data.loc[:, : 'Contact time, s']
large_data = large_data.dropna(how= 'all', axis= 1)
small_data = small_data.reset_index(drop= True)
small_data_columns = ['C2 yield'] + [s for s in small_data.columns if s != 'C2 yield']
small_data = small_data[small_data_columns]
small_data = small_data.loc[:, : 'Temp']
# x2相当へ変形
#small_data = transform_x2(small_data)
# x1相当へ変形
#small_data = small_data.iloc[150: , :]
#small_data = transform_x1(small_data, element_data, thermo_data, flag= 'target')
#print(small_data.dtypes)
small_data = small_data.replace([np.inf, -np.inf], np.nan)
small_data = small_data.dropna(axis= 1)
small_data = pd.get_dummies(small_data, columns= ['Support'])
small_data[small_data.columns] = small_data[small_data.columns].apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
# supportは、AlとSiに集約する
support_columns = [s for s in small_data.columns if s.startswith('Support')]
small_data['Support_Al'] = small_data[['Support_Al2O3','Support_Al2O4','Support_Al2O5','Support_Al2O6']].max(axis= 1)
small_data['Support_Si'] = small_data[['Support_SiO2','Support_SiO3','Support_SiO4','Support_SiO5']].max(axis= 1)
small_data = small_data.drop(support_columns, axis= 1)
# support_dataを展開
#large_data = large_data.iloc[3000: 3100, :]
# x2への変換
large_data = transform_x2_for_target(large_data, large_elements)
# 元素2種のみを含むデータの抽出
nonzero_col = (large_data != 0).sum()
NaMn_data = large_data[large_data['Na'] != 0]
NaMn_data = NaMn_data[NaMn_data['Mn'] != 0]
NaMn_elements = [el for el in large_elements if el not in ['Mn','Na']]
NaMn_data = NaMn_data[NaMn_data[NaMn_elements].sum(axis= 1) == 0]
NaMn_data = NaMn_data.dropna()

NaW_data = large_data[large_data['Na'] != 0]
NaW_data = NaW_data[NaW_data['W'] != 0]
NaW_elements = [el for el in large_elements if el not in ['W','Na']]
NaW_data = NaW_data[NaW_data[NaW_elements].sum(axis= 1) == 0]
NaW_data = NaW_data.dropna()

MnW_data = large_data[large_data['Mn'] != 0]
MnW_data = MnW_data[MnW_data['W'] != 0]
MnW_elements = [el for el in large_elements if el not in ['Mn','W']]
MnW_data = MnW_data[MnW_data[MnW_elements].sum(axis= 1) == 0]
MnW_data = MnW_data.dropna()

NaW_data = NaW_data.loc[:, (NaW_data != 0).any(axis= 0)]
NaMn_data = NaMn_data.loc[:, (NaMn_data != 0).any(axis= 0)]
NaW_data.to_csv('MyWork/datasets/NaW_data.csv', encoding= 'utf-8-sig')
NaMn_data.to_csv('MyWork/datasets/NaMn_data.csv', encoding= 'utf-8-sig')

NaW_data['NaW'] = NaW_data['Na'] * NaW_data['W']
NaMn_data['NaMn'] = NaMn_data['Na'] * NaMn_data['Mn']
NaW_data.to_csv('MyWork/datasets/NaW_x.csv', encoding= 'utf-8-sig')
NaMn_data.to_csv('MyWork/datasets/NaMn_x.csv', encoding= 'utf-8-sig')

# データセットA
X_A = small_data.drop('C2 yield', axis= 1).copy()
Y_A = small_data['C2 yield'].copy()

# データセットB
X_B = large_data.drop('C2 yield', axis= 1).copy()
Y_B = large_data['C2 yield']

# データのオートスケーリング
scaler_X = StandardScaler()
scaler_Y_A = StandardScaler()
scaler_Y_B = StandardScaler()

X_A_scaled = scaler_X.fit_transform(X_A)
X_B_scaled = scaler_X.transform(X_B)  # 同じスケーラーを使用してスケール
Y_A_scaled = scaler_Y_A.fit_transform(Y_A.values.reshape(-1, 1))
Y_B_scaled = scaler_Y_B.fit_transform(Y_B.values.reshape(-1, 1))

# PLS回帰の適用（データセットA）
pls_A = PLSRegression(n_components= 5)
X_A_scores, Y_A_scores = pls_A.fit_transform(X_A_scaled, Y_A_scaled)

# 主成分の重み（loadings）
loadings_A = pls_A.x_loadings_

# 各主成分の重みの表示
components_A = pd.DataFrame(loadings_A, columns=[f'PLS Component {i+1}' for i in range(loadings_A.shape[1])], index=X_A.columns)
print("主成分の重み：")
print(components_A)

# 主成分の線形結合の様式を表示
for i in range(5):
    component_weights = loadings_A[:, i]
    linear_combination = " + ".join([f"{weight:.4f} * {var}" for weight, var in zip(component_weights, X_A.columns)])
    print(f"PLS Component {i+1} = {linear_combination}")

# 結果の可視化（データセットA）
plt.figure(figsize=(8, 6))
plt.scatter(X_A_scores[:, 0], X_A_scores[:, 1], c=Y_A, cmap='viridis')
plt.xlabel('PLS Component 1')
plt.ylabel('PLS Component 2')
plt.colorbar(label='Target Variable A')
plt.title('PLS Regression of Dataset A')
plt.show()

# PLS回帰の適用（データセットB）
pls_B = PLSRegression(n_components=2)
X_B_scores, Y_B_scores = pls_B.fit_transform(X_B_scaled, Y_B_scaled)

# 主成分の重み（loadings）
loadings_B = pls_B.x_loadings_

# 各主成分の重みの表示
components = pd.DataFrame(loadings_B, columns=[f'PLS Component {i+1}' for i in range(loadings_B.shape[1])], index=X_B.columns)
print("主成分の重み：")
print(components)

# 主成分の線形結合の様式を表示
number_of_componetts = loadings_B.shape[1]
for i in range(number_of_componetts):
    component_weights = loadings_B[:, i]
    linear_combination = " + ".join([f"{weight:.4f} * {var}" for weight, var in zip(component_weights, X_B.columns)])
    print(f"PLS Component {i+1} = {linear_combination}")

# 結果の可視化（データセットB）
plt.figure(figsize=(8, 6))
plt.scatter(X_B_scores[:, 0], X_B_scores[:, 1], c=Y_B, cmap='viridis')
plt.xlabel('PLS Component 1')
plt.ylabel('PLS Component 2')
plt.colorbar(label='Target Variable B')
plt.title('PLS Regression of Dataset B')
plt.show()

print('hello')