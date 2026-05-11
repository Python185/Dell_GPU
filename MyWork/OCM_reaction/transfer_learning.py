
import math
import sys
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, DotProduct
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
    support_elements = set(list(support_df[support_name].values.flatten()))
    support_elements = {x for x in support_elements if x != 'na' and not (isinstance(x, float) and np.isnan(x))}
    support_elements = sorted(list(support_elements))    
    cation_name_columns = ['cation_' + s for s in cation_elements]
    anion_name_columns = ['anion_' + s for s in anion_elements]
    support_name_columns = ['support_' + s for s in support_elements]
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
            if support in support_elements:
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

def transform_x2_for_target(df, target_elements):
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
    def check_row(row, target_elements):
        non_zero_col = row[row != 0].index
        return all(col in target_elements for col in non_zero_col)    
    new_df = new_df[new_df.apply(lambda row: check_row(row, target_elements), axis= 1)]
    new_df = new_df[target_elements]
    def normalize_row(row):
        non_zero_values = row[row != 0]
        sum_non_zero = non_zero_values.sum()
        normalized_values = non_zero_values / sum_non_zero
        row[non_zero_values.index] = normalized_values
        return row
    new_df = new_df.apply(normalize_row, axis= 1)    
    # 元のデータフレームから 不要列を削除し名前を更新する
    drop_list = cation_columns + [s for s in df if 'Anion' in s]
    df = df.drop(drop_list, axis= 1)
    df = df[['Y(C2), %', 'Support1','Support2','Support3', 'Temperature, K']]
    df['Temperature, K'] = df['Temperature, K'] -273
    df = df.rename(columns= {'Y(C2), %': 'C2 yield', 'Temperature, K': 'Temp', 'Support1': 'Support'})
    result_df = pd.merge(df, new_df, right_index= True, left_index= True)

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

# spectra dataset 
transfer_learning_flag = 0  # 0: transfer learning, 1: using only target data, 2: using both supporting data and target data
regression_methods = ['GPR_3']
#regression_methods = ['GPR_1']
#regression_methods = ['pls', 'rr', 'lasso', 'en', 'lsvr', 'nsvr', 'dt', 'rf', 'gp', 'lgb', 'xgb', 'gbdt']
#number_of_test_samples = 50
number_of_test_samples = 80  #target中のtest_sample量となる

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
                                            
# load data set target: small_data, support: large_data
raw_data_with_y_supporting_1 = pd.read_excel('MyWork/datasets/original_data/OCM.xlsx', sheet_name= 'OCM_Dataset_-2019', index_col= 0)
raw_data_with_y_target = pd.read_csv('MyWork/original_data/Oxidative_coupling_of_methane_at_CADS.csv')
element_data = pd.read_csv('results/element_data_normal_string240527.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
thermo_data = pd.read_csv('results/thermo_data.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)

# target_dataとsupport_dataの被りの有無を確認→1つ1つ確認して重複なかった
raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[raw_data_with_y_supporting_1['Y(C2), %'].notna()]
support_data = raw_data_with_y_supporting_1[raw_data_with_y_supporting_1['Cation 4'].isna() ]
support_data = support_data[support_data['Temperature, K'].isin([873, 973, 1073, 1173])]
target_elements = raw_data_with_y_target.loc[:, 'Cation1': 'Cation3'].values.flatten()
target_elements = list(set(target_elements))
target_elements = [s for s in target_elements if s != 'na']
target_elements.sort()
target_data = raw_data_with_y_target[raw_data_with_y_target['Temp'].isin([700, 800])]
target_data = target_data.sort_values(by= 'Cation1')
target_data_columns = ['C2 yield'] + [s for s in target_data.columns if s != 'C2 yield']
target_data = target_data[target_data_columns]
target_data = target_data.iloc[88: 116, :]
list_a = ['Na','Mn','W','na']
target_data = target_data[target_data['Cation1'].isin(list_a) & target_data['Cation2'].isin(list_a) & target_data['Cation3'].isin(list_a)]
support_data = support_data[support_data['Cation 1'].isin(target_elements)]
support_data = support_data[support_data['Cation 2'].isin(target_elements) | support_data['Cation 2'].isna()]
support_data = support_data[(support_data['Support 1'] == 'Al') | (support_data['Support 1'] == 'Si') ]
support_data = pd.concat([support_data.loc[:, : 'Cation 3 mol%'], support_data.loc[:, 'Anion 1': ]], axis= 1)
support_data = support_data[support_data['Cation 2'].notna()]
support_data = support_data.sort_values(by= 'Cation 1')
support_data_columns = ['Y(C2), %'] + [s for s in support_data.columns if s != 'Y(C2), %']
support_data = support_data[support_data_columns]
support_data = support_data[support_data['Y(C2), %'] < 12]
support_data = support_data.iloc[8:, :]
support_data['Y(C2), %'] = support_data['Y(C2), %'].round(1)

# element_dataとthermo_dataの処理
support_elements = raw_data_with_y_supporting_1.loc[:, ['Cation 1','Cation 2','Cation 3','Cation 4','Cation 5','Cation 6']].values.flatten()
support_elements = list(set(support_elements))
support_elements = [s for s in support_elements if isinstance(s, str)]
support_elements.sort()
thermo_data = thermo_data.loc[thermo_data.index.isin(support_elements)]
thermo_data = thermo_data.loc[:, : 'Stockmayer parameter, K']
thermo_data = thermo_data.drop(['CAS', 'Heat of formation, J/mol'], axis= 1)
thermo_data = thermo_data.loc[:, thermo_data.isnull().mean() < 0.2]
check_nan_b = thermo_data.isna().sum()
thermo_data = thermo_data.dropna(axis= 1)
element_data = element_data.loc[element_data.index.isin(support_elements)]
element_data = element_data.loc[:, element_data.isnull().mean() < 0.2]  #nanが20%未満を抽出
element_data = element_data.dropna(axis= 1)
#element_data = element_data.drop(['space group', 'structure','oxide','hhi_p','hhi_r'], axis= 1)
element_data = delete_high_corr(element_data, threshold= 0.95)
thermo_data = delete_high_corr(thermo_data, threshold= 0.95)

# target_dataを展開 列が同一である必要あるため、targetをベースにしてsupportを変形させる
raw_data_with_y_supporting_1 = pd.concat([raw_data_with_y_supporting_1['Y(C2), %'], raw_data_with_y_supporting_1.loc[:, : 'Contact time, s']], axis= 1)
raw_data_with_y_target = raw_data_with_y_target.reset_index(drop= True)
raw_data_with_y_target_columns = ['C2 yield'] + [s for s in raw_data_with_y_target.columns if s != 'C2 yield']
raw_data_with_y_target = raw_data_with_y_target[raw_data_with_y_target_columns]
raw_data_with_y_target = raw_data_with_y_target.loc[:, : 'Temp']
# x2相当へ変形
#raw_data_with_y_target = transform_x2(raw_data_with_y_target)
# x1相当へ変形
#raw_data_with_y_target = raw_data_with_y_target.iloc[150: , :]
raw_data_with_y_target = transform_x1(raw_data_with_y_target, element_data, thermo_data, flag= 'target')
#print(raw_data_with_y_target.dtypes)
raw_data_with_y_target = raw_data_with_y_target.replace([np.inf, -np.inf], np.nan)
raw_data_with_y_target = raw_data_with_y_target.dropna(axis= 1)
raw_data_with_y_target = pd.get_dummies(raw_data_with_y_target, columns= ['Support'])
raw_data_with_y_target[raw_data_with_y_target.columns] = raw_data_with_y_target[raw_data_with_y_target.columns].apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
# supportは、AlとSiに集約する
support_columns = [s for s in raw_data_with_y_target.columns if s.startswith('Support')]
raw_data_with_y_target['Support_Al'] = raw_data_with_y_target[['Support_Al2O3','Support_Al2O4','Support_Al2O5','Support_Al2O6']].max(axis= 1)
raw_data_with_y_target['Support_Si'] = raw_data_with_y_target[['Support_SiO2','Support_SiO3','Support_SiO4','Support_SiO5']].max(axis= 1)
raw_data_with_y_target = raw_data_with_y_target.drop(support_columns, axis= 1)
# support_dataを展開
#raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1.iloc[3000: 3100, :]
# x2への変換
#raw_data_with_y_supporting_1 = transform_x2_for_target(raw_data_with_y_supporting_1, target_elements)
# x1への変換
raw_data_with_y_supporting_1 = transform_x1(raw_data_with_y_supporting_1, element_data, thermo_data, flag= 'support')
# support2, 3があるものは削除し、support1がNaN, AlSi以外も削除
raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[raw_data_with_y_supporting_1['Support3'].isna()]
raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[raw_data_with_y_supporting_1['Support2'].isna()]
raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[(raw_data_with_y_supporting_1['Support']=='Al')|(raw_data_with_y_supporting_1['Support']=='Si')]
raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1.drop(['Support2', 'Support3'], axis= 1)
raw_data_with_y_supporting_1 = pd.get_dummies(raw_data_with_y_supporting_1, columns= ['Support'])
raw_data_with_y_supporting_1[raw_data_with_y_supporting_1.columns] = raw_data_with_y_supporting_1[raw_data_with_y_supporting_1.columns].apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)

# 分散の小さい列を削除
threshold_of_rate_of_same_value = 0.95
rate_of_same_value = list()
for X_variable_name in raw_data_with_y_target.columns:
    same_value_number = raw_data_with_y_target[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / raw_data_with_y_target.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
raw_data_with_y_target = raw_data_with_y_target.drop(raw_data_with_y_target.columns[deleting_variable_numbers], axis=1)
rate_of_same_value = list()
for X_variable_name in raw_data_with_y_supporting_1.columns:
    same_value_number = raw_data_with_y_supporting_1[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / raw_data_with_y_supporting_1.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1.drop(raw_data_with_y_supporting_1.columns[deleting_variable_numbers], axis=1)

# x1の場合support_dataの列が少なくなるためtarget_dataに合わせる
if raw_data_with_y_target.shape[1] != raw_data_with_y_supporting_1.shape[1]:
    print('support_dataの列が少ないためsupportに合わせてtarget_dataの列を削除します')
    raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1.replace([np.inf, -np.inf], np.nan)
    raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1.dropna(how= 'any', axis= 0)
    common_columns = list(set(raw_data_with_y_target.columns) & set(raw_data_with_y_supporting_1.columns))
    common_columns.remove('C2 yield')
    raw_data_with_y_target = raw_data_with_y_target[['C2 yield'] + common_columns]    
    raw_data_with_y_supporting_1 = raw_data_with_y_supporting_1[['C2 yield'] + common_columns]   
    print('support_dataの行と列:', raw_data_with_y_supporting_1.shape)
    nan_checker = raw_data_with_y_supporting_1.isna().sum()

# np.arrayに変換
raw_data_with_y_supporting_1_arr = np.array(raw_data_with_y_supporting_1)
raw_data_with_y_target_arr = np.array(raw_data_with_y_target)
# x, y に分離
y_supporting_1 = raw_data_with_y_supporting_1_arr[:, 0]
x_supporting_1 = raw_data_with_y_supporting_1_arr[:, 1:]
y_target = raw_data_with_y_target_arr[:, 0]
x_target = raw_data_with_y_target_arr[:, 1:]

x_train_target, x_test_target, y_train_target, y_test = train_test_split(x_target, y_target, test_size=number_of_test_samples, random_state=0)

# autoscaling
if do_autoscaling:
    autoscaled_x_train_target = (x_train_target - x_train_target.mean(axis=0)) / x_train_target.std(axis=0, ddof=1)
    autoscaled_x_supporting_1 = (x_supporting_1 - x_supporting_1.mean(axis=0)) / x_supporting_1.std(axis=0, ddof=1)
    #autoscaled_x_supporting_2 = (x_supporting_2 - x_supporting_2.mean(axis=0)) / x_supporting_2.std(axis=0, ddof=1)
    autoscaled_x_test_target = (x_test_target - x_train_target.mean(axis=0)) / x_train_target.std(axis=0, ddof=1)
    autoscaled_y_supporting_1 = (y_supporting_1 - y_supporting_1.mean()) / y_supporting_1.std(ddof=1)
    #autoscaled_y_supporting_2 = (y_supporting_2 - y_supporting_2.mean()) / y_supporting_2.std(ddof=1)
    autoscaled_y_train_target = (y_train_target - y_train_target.mean()) / y_train_target.std(ddof=1)
else:
    autoscaled_x_train_target = x_train_target.copy()
    autoscaled_x_supporting_1 = x_supporting_1.copy()
    #autoscaled_x_supporting_2 = x_supporting_2.copy()
    autoscaled_x_test_target = x_test_target.copy()
    autoscaled_y_supporting_1 = y_supporting_1.copy()
    #autoscaled_y_supporting_2 = y_supporting_2.copy()
    autoscaled_y_train_target = y_train_target.copy()
    
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
    #x_supporting_2_arranged = np.c_[autoscaled_x_supporting_2, np.zeros(autoscaled_x_supporting_2.shape), autoscaled_x_supporting_2, np.zeros(autoscaled_x_supporting_2.shape)]
    x_train_target_arranged = np.c_[autoscaled_x_train_target, np.zeros(autoscaled_x_train_target.shape), np.zeros(autoscaled_x_train_target.shape), autoscaled_x_train_target]
    autoscaled_x_train = np.r_[x_supporting_1_arranged, x_train_target_arranged]
    autoscaled_x_test = np.c_[autoscaled_x_test_target, np.zeros(autoscaled_x_test_target.shape), np.zeros(autoscaled_x_test_target.shape), autoscaled_x_test_target]
    autoscaled_y_train = np.r_[autoscaled_y_supporting_1, autoscaled_y_train_target]

fold_number = min(fold_number, len(autoscaled_y_train))

autoscaled_y_train = pd.Series(autoscaled_y_train)
y_test = pd.Series(y_test)
autoscaled_x_train = pd.DataFrame(autoscaled_x_train)
autoscaled_x_test = pd.DataFrame(autoscaled_x_test)

#z_scores = np.abs((autoscaled_x_train - autoscaled_x_train.mean()) / autoscaled_x_train.std())
#autoscaled_x_train = autoscaled_x_train[(z_scores < 4).all(axis=1)]
#autoscaled_y_train = autoscaled_y_train.loc[autoscaled_x_train.index]

plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
for method in regression_methods:
    print(method)
    if method == 'pls':  # Partial Least Squares
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
        plt.plot(pls_components, r2all, 'bo-')
        plt.plot(pls_components, r2cvall, 'ro-')
        plt.ylim(0, 1)
        plt.xlabel('Number of PLS components')
        plt.ylabel('r2(blue), r2cv(red)')
        plt.show()
        optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
        optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
        regression_model = PLSRegression(n_components=optimal_pls_component_number)
    elif method == 'rr':  # ridge regression
        r2cvall = list()
        for ridge_lambda in ridge_lambdas:
            rr_model_in_cv = Ridge(alpha=ridge_lambda)
            estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                  cv=fold_number)
            r2cvall.append(float(1 - sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / sum(autoscaled_y_train ** 2)))
        plt.figure()
        plt.plot(ridge_lambdas, r2cvall, 'k', linewidth=2)
        plt.xscale('log')
        plt.xlabel('Weight for ridge regression')
        plt.ylabel('r2cv for ridge regression')
        plt.show()
        optimal_ridge_lambda = ridge_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
        regression_model = Ridge(alpha=optimal_ridge_lambda)
    elif method == 'lasso':  # LASSO
        r2cvall = list()
        for lasso_lambda in lasso_lambdas:
            lasso_model_in_cv = Lasso(alpha=lasso_lambda)
            estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                  cv=fold_number)
            r2cvall.append(float(1 - sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / sum(autoscaled_y_train ** 2)))
        plt.figure()
        plt.plot(lasso_lambdas, r2cvall, 'k', linewidth=2)
        plt.xlabel('Weight for LASSO')
        plt.ylabel('r2cv for LASSO')
        plt.show()
        optimal_lasso_lambda = lasso_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
        regression_model = Lasso(alpha=optimal_lasso_lambda)
    elif method == 'en':  # Elastic net
        elastic_net_in_cv = ElasticNetCV(cv=fold_number, l1_ratio=elastic_net_lambdas, alphas=elastic_net_alphas)
        elastic_net_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_elastic_net_alpha = elastic_net_in_cv.alpha_
        optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_
        regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
    elif method == 'lsvr':  # Linear SVR
        linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                                        cv=fold_number)
        linear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_linear_svr_c = linear_svr_in_cv.best_params_['C']
        optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon']
        regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)
    elif method == 'nsvr':  # Nonlinear SVR
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
                                   cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']
        # CV による C の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                                   {'C': nonlinear_svr_cs}, cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_c = model_in_cv.best_params_['C']
        # CV による γ の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                                   {'gamma': nonlinear_svr_gammas}, cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']
#        nonlinear_svr_in_cv = GridSearchCV(svm.SVR(kernel='rbf', gamma=optimal_nonlinear_gamma),
#                                           {'C': nonlinear_svr_cs, 'epsilon': nonlinear_svr_epsilons}, cv=fold_number)
#        nonlinear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
#        optimal_nonlinear_c = nonlinear_svr_in_cv.best_params_['C']
#        optimal_nonlinear_epsilon = nonlinear_svr_in_cv.best_params_['epsilon']
        regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                                   gamma=optimal_nonlinear_gamma)
    elif method == 'rf':  # Random forest
        rmse_oob_all = list()
        for random_forest_x_variables_rate in random_forest_x_variables_rates:
            RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                max(math.ceil(autoscaled_x_train.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
            RandomForestResult.fit(autoscaled_x_train, autoscaled_y_train)
            estimated_y_in_cv = RandomForestResult.oob_prediction_
            if do_autoscaling:
                estimated_y_in_cv = estimated_y_in_cv * autoscaled_y_train.std(ddof=1) + autoscaled_y_train.mean()
            rmse_oob_all.append((sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / len(autoscaled_y_train)) ** 0.5)
        plt.figure()
        plt.plot(random_forest_x_variables_rates, rmse_oob_all, 'k', linewidth=2)
        plt.xlabel('Ratio of the number of X-variables')
        plt.ylabel('RMSE of OOB')
        plt.show()
        optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
            np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
        regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
            max(math.ceil(autoscaled_x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True)
    elif method == 'GPR_1':  # Gaussian process
        regression_model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
    elif method == 'GPR_3':
        regression_model = GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(autoscaled_x_train.shape[1])) + WhiteKernel(), alpha=0)
        autoscaled_x_train = autoscaled_x_train.astype(np.float32)
        autoscaled_y_train = autoscaled_y_train.astype(np.float32)
    elif method == 'GPR_4':
        regression_model = GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(autoscaled_x_train.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(), alpha=0)
    elif method == 'lgb':  # LightGBM
        import lightgbm as lgb
    
        regression_model = lgb.LGBMRegressor()
    elif method == 'xgb':  # XGBoost
        import xgboost as xgb
    
        regression_model = xgb.XGBRegressor()
    elif method == 'gbdt':  # scikit-learn
        from sklearn.ensemble import GradientBoostingRegressor
    
        regression_model = GradientBoostingRegressor()
    regression_model.fit(autoscaled_x_train, autoscaled_y_train)
    """
    # calculate y
    calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_x_train))
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(autoscaled_y_train, calculated_ytrain, c='blue')
    y_max = np.max(np.array([np.array(autoscaled_y_train), calculated_ytrain]))
    y_min = np.min(np.array([np.array(autoscaled_y_train), calculated_ytrain]))
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('Actual Y (autoscaled)')
    plt.ylabel('Calculated Y (autoscaled)')
    plt.show()
    # r2, RMSE, MAE
    print('r2: {0}'.format(float(1 - sum((autoscaled_y_train - calculated_ytrain) ** 2) / sum(autoscaled_y_train ** 2))))
    print('RMSE (autoscaled): {0}'.format(float((sum((autoscaled_y_train - calculated_ytrain) ** 2) / len(autoscaled_y_train)) ** 0.5)))
    print('MAE (autoscaled): {0}'.format(float(sum(abs(autoscaled_y_train - calculated_ytrain)) / len(autoscaled_y_train))))
    
    # estimated_y in cross-validation
    estimated_y_in_cv = np.ndarray.flatten(
        model_selection.cross_val_predict(regression_model, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(autoscaled_y_train, estimated_y_in_cv, c='blue')
    y_max = np.max(np.array([np.array(autoscaled_y_train), estimated_y_in_cv]))
    y_min = np.min(np.array([np.array(autoscaled_y_train), estimated_y_in_cv]))
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('Actual Y (autoscaled)')
    plt.ylabel('Estimated Y in CV (autoscaled)')
    plt.show()
    # r2cv, RMSEcv, MAEcv
    print('r2cv: {0}'.format(float(1 - sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / sum(autoscaled_y_train ** 2))))
    print('RMSEcv (autoscaled): {0}'.format(float((sum((autoscaled_y_train - estimated_y_in_cv) ** 2) / len(autoscaled_y_train)) ** 0.5)))
    print('MAEcv (autoscaled): {0}'.format(float(sum(abs(autoscaled_y_train - estimated_y_in_cv)) / len(autoscaled_y_train))))
    
    # standard regression coefficients
    # standard_regression_coefficients = regression_model.coef_
    # standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients)
    # standard_regression_coefficients.index = Xtrain.columns
    # standard_regression_coefficients.columns = ['standard regression coefficient']
    # standard_regression_coefficients.to_csv( 'standard_regression_coefficients.csv' )
    """
    # prediction
    if autoscaled_x_test.shape[0]:
        predicted_ytest = np.ndarray.flatten(regression_model.predict(autoscaled_x_test))
        if do_autoscaling:
            predicted_ytest = predicted_ytest * y_train_target.std(ddof=1) + y_train_target.mean()
        # yy-plot
        plt.figure(figsize=figure.figaspect(1))
        plt.scatter(y_test, predicted_ytest, c='blue')
        y_max = np.max(np.array([np.array(y_test), predicted_ytest]))
        y_min = np.min(np.array([np.array(y_test), predicted_ytest]))
        plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                 [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlabel('Actual Y')
        plt.ylabel('Predicted Y')
        plt.savefig(f'result/dcv_analysis_results/C2 yield/yyplot/tempo/fig_{method}_{transfer_learning_flag}_x1.png',bbox_inches = 'tight') # 図の保存        
        plt.show()
        # r2p, RMSEp, MAEp
        print('r2p: {0}'.format(float(1 - sum((y_test - predicted_ytest) ** 2) / sum((y_test - y_test.mean()) ** 2))))
        print('RMSEp: {0}'.format(float((sum((y_test - predicted_ytest) ** 2) / len(y_test)) ** 0.5)))
        print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_ytest)) / len(y_test))))

print('hello')