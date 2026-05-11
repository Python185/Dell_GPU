import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

np.random.seed(1)
random.seed(1)

class MIutility:
    def __init__(self):
        pass

    def get_calcInverMetalData(self, inverse_metal_x_data,atom_nums,sosei_list):
        
        # 組成金属が２つのもの
        general_x = np.random.rand(len(np.where(atom_nums==2)[0]), 2)*(99-1)+1 # 乱数を生成
        # inverse_metal_x_data.loc[atom_nums==2, sosei_list[:2]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)) * 100, decimals=1) # 割って1-99にする
        inverse_metal_x_data.loc[atom_nums==2, sosei_list[:2]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=2) # 割って0.01～0.99にする
        # 組成金属が３つのもの
        general_x = np.random.rand(len(np.where(atom_nums==3)[0]), 3)*(99-1)+1
        # inverse_metal_x_data.loc[atom_nums==3, sosei_list[:3]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)) * 100, decimals=1)
        inverse_metal_x_data.loc[atom_nums==3, sosei_list[:3]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=2)
        # 組成金属が４つのもの
        general_x = np.random.rand(len(np.where(atom_nums==4)[0]), 4)*(99-1)+1
        #inverse_metal_x_data.loc[atom_nums==4, sosei_list[:4]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)) * 100, decimals=1)
        inverse_metal_x_data.loc[atom_nums==4, sosei_list[:4]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=2)
        # 組成金属が５つのもの
        general_x = np.random.rand(len(np.where(atom_nums==5)[0]), 5)*(99-1)+1
        # inverse_metal_x_data.loc[atom_nums==5, sosei_list[:]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)) * 100, decimals=1)
        inverse_metal_x_data.loc[atom_nums==5, sosei_list[:]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=2)
 
        return inverse_metal_x_data

    def calc_MetalRatio(self, inverse_metal_x_data,atom_nums,sosei_list):
        
        # 組成金属が２つのもの
        general_x = np.random.rand(len(np.where(atom_nums==2)[0]), 2)*(99-1)+1 # 乱数を生成
        # inverse_metal_x_data.loc[atom_nums==2, sosei_list[:2]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)) * 100, decimals=1) # 割って1-99にする
        inverse_metal_x_data.loc[atom_nums==2, sosei_list[:2]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=2) # 割って0.01～0.99にする
        # 組成金属が３つのもの
        general_x = np.random.rand(len(np.where(atom_nums==3)[0]), 3)*(99-1)+1
        # inverse_metal_x_data.loc[atom_nums==3, sosei_list[:3]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)) * 100, decimals=1)
        inverse_metal_x_data.loc[atom_nums==3, sosei_list[:3]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=2)
        # 組成金属が４つのもの
        general_x = np.random.rand(len(np.where(atom_nums==4)[0]), 4)*(99-1)+1
        #inverse_metal_x_data.loc[atom_nums==4, sosei_list[:4]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)) * 100, decimals=1)
        inverse_metal_x_data.loc[atom_nums==4, sosei_list[:4]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=2)
        # 組成金属が５つのもの
        general_x = np.random.rand(len(np.where(atom_nums==5)[0]), 5)*(99-1)+1
        # inverse_metal_x_data.loc[atom_nums==5, sosei_list[:]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)) * 100, decimals=1)
        inverse_metal_x_data.loc[atom_nums==5, sosei_list[:]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=2)
        
        inverse_metal_x_data = inverse_metal_x_data.fillna(0)
        cols = inverse_metal_x_data.columns.to_list()
        cols_ratio = [s for s in cols if 'ratio' in s]
        normalized_x = inverse_metal_x_data[cols_ratio].values
        
        for i in range(normalized_x.shape[0]):
            if np.sum(normalized_x[i]) != 1.0:
                total = np.sum(normalized_x[i])
                error = 1.0 - total
                # 最小または最大の要素を調整
                if error > 0:  # 合計が1より小さい場合
                    #idx = np.argmin(normalized_x[i])  # 最小値のインデックス
                    #normalized_x[i][idx] += round(error, 2)  # 最小値に誤差を加える
                    non_zero_values = normalized_x[i][normalized_x[i] != 0]
                    sorted_values = np.sort(non_zero_values)
                    next_min_value = sorted_values[0]
                    next_min_index = np.where(normalized_x[i] == next_min_value)[0][0]
                    normalized_x[i][next_min_index] += round(error, 2)                    
                else:  # 合計が1より大きい場合
                    idx = np.argmax(normalized_x[i])  # 最大値のインデックス
                    normalized_x[i][idx] += round(error, 2)  # 最大値から誤差を引く
                #normalized_x[i] = np.round(normalized_x[i], 2)  # 再度丸める

        # DataFrameに正規化されたデータを設定
        inverse_metal_x_data[cols_ratio] = normalized_x    
 
        return inverse_metal_x_data
         
    def get_calcInverMetalData_Simple(self, inverse_metal_x_data,atom_nums,sosei_list):
        # 組成金属が３つのもの
        general_x = np.random.rand(len(np.where(atom_nums==3)[0]), 3)*7+1
        inverse_metal_x_data.loc[atom_nums==3, sosei_list[:3]] = np.round(general_x / general_x.sum(axis=1).reshape((-1, 1)), decimals=1)
        inverse_metal_x_data.loc[atom_nums==3, sosei_list[2]] = 1- (inverse_metal_x_data.loc[atom_nums==3, sosei_list[0]] + inverse_metal_x_data.loc[atom_nums==3, sosei_list[1]])
        inverse_metal_x_data = inverse_metal_x_data[inverse_metal_x_data['ratio3'] != 0]
        inverse_metal_x_data.reset_index(inplace= True, drop= True)
                
        return inverse_metal_x_data

    def get_atomListN(self, metal_salt_constraints_data):
        
        # nanをNに変換
        metal_salt_constraints_data = metal_salt_constraints_data.fillna({"Omit":'N'}) 
        
        #金属塩リスト（omit列が空欄のもの）
        metal_salt_constraints_data_N = metal_salt_constraints_data
        metal_salt_constraints_data_N = metal_salt_constraints_data_N[metal_salt_constraints_data_N['Omit'] == 'N']
        atom_list_N = list(metal_salt_constraints_data_N['元素'])
        atom_list_N = list(set(atom_list_N))

        return atom_list_N
    

    def get_atomListY(self, metal_salt_constraints_data):
        
        #金属塩リスト（omit列がYのもの）
        metal_salt_constraints_data_Y = metal_salt_constraints_data
        metal_salt_constraints_data_Y = metal_salt_constraints_data_Y[metal_salt_constraints_data_Y['Omit'] == 'Y']
        atom_list_Y = list(metal_salt_constraints_data_Y['元素'])
        atom_list_Y = list(set(atom_list_Y))

        return atom_list_Y

    def generateConstrainedRandomArray(self, N, Compo_ratio_N):
        #chatGPTに教えてもらった方法 時間がかかり過ぎるためNG
        while True:
            # 最初の1列を0.01から0.68の範囲でランダムに生成
            arr = np.round(np.random.rand(N, 2) * (Compo_ratio_N-0.03) + 0.01, decimals=2)
            # 3列目を計算（0.7から他の2列の合計を引いたもの）
            third_column = Compo_ratio_N - arr.sum(axis=1)
            # 3列目を配列に追加
            arr = np.column_stack((arr, third_column))
            # 3列目の値が0.01から0.69の範囲内にあるかを確認
            if np.all((arr >= 0.01) & (arr <= (Compo_ratio_N-0.01))):
                return arr    
            
    def five_down_six_up_three_decimals(self, arr):
        # 100倍して0.5を足し、床関数で整数に丸める
        rounded = np.floor(arr * 100 + 0.5) / 100
        # 元の値との差を計算
        diff = arr - rounded
        # 差が-0.005（元の値がx.005）の場合、0.01を足して切り上げる
        rounded[diff == -0.005] += 0.01
        return rounded
    
    def get_calcInverMetalData_N(self, inverse_metal_x_data_N,Compo_ratio_N,atom_list_N,atom_nums_N,sosei_list_N):
        
        # 元素数に応じた元素の選択
        inverse_metal_x_data_N.iloc[:, :3] = pd.DataFrame([random.sample(atom_list_N, x) for x in atom_nums_N]).values 

        # 組成金属が１つのもの
        general_x_N = np.random.rand(len(np.where(atom_nums_N==1)[0]), 1)*(99-1)+1 # 乱数を生成
        inverse_metal_x_data_N.loc[atom_nums_N==1, sosei_list_N[:1]] = np.round(general_x_N / general_x_N.sum(axis=1).reshape((-1, 1)) * Compo_ratio_N, decimals=2) # 0.01～0.69にする
        # 組成金属が２つのもの
        general_x_N = np.random.rand(len(np.where(atom_nums_N==2)[0]), 2)*(99-1)+1 # 乱数を生成
        inverse_metal_x_data_N.loc[atom_nums_N==2, sosei_list_N[:2]] = np.round(general_x_N / general_x_N.sum(axis=1).reshape((-1, 1)) * Compo_ratio_N, decimals=2) # 0.01～0.69にする
        #組成金属が３つのもの
        #組成範囲が0-0.3となっていたため修正して0.01-0.29となるようにした(231116)        
        #general_x_N = np.random.rand(len(np.where(atom_nums_N==3)[0]), 3)*(99-1)+1
        #inverse_metal_x_data_N.loc[atom_nums_N==3, sosei_list_N[:3]] = np.round(general_x_N / general_x_N.sum(axis=1).reshape((-1, 1)) * Compo_ratio_N, decimals=2) # 0.01～0.69にする
        #random_array = self.generateConstrainedRandomArray(len(atom_nums_N), Compo_ratio_N)
        #inverse_metal_x_data_N.loc[atom_nums_N==3, sosei_list_N[:3]] = random_array
        general_x_N = np.random.rand(len(np.where(atom_nums_N==3)[0]), 3)
        array_N = np.round(general_x_N / general_x_N.sum(axis=1).reshape((-1, 1)), decimals= 3)
        array_N = np.round(array_N * (Compo_ratio_N-0.03) + 0.01, decimals=2)  # 0.01～0.69にする  
        #array_N = self.five_down_six_up_three_decimals(array_N * (Compo_ratio_N-0.03) + 0.01) #0.03を引かないと足して0.7以上となる 原因不明・・
        inverse_metal_x_data_N.loc[atom_nums_N==3, sosei_list_N[:3]] = array_N

        return inverse_metal_x_data_N

    def get_calcInverMetalData_Y(self, inverse_metal_x_data_Y,Compo_ratio_Y,atom_list_Y,atom_nums_Y,sosei_list_Y):
        
        # 元素数に応じた元素の選択
        inverse_metal_x_data_Y.iloc[:, 3:5] = pd.DataFrame([random.sample(atom_list_Y, x) for x in atom_nums_Y]).values 
    
        # 組成金属が１つのもの
        general_x_Y = np.random.rand(len(np.where(atom_nums_Y==1)[0]), 1)*(99-1)+1 # 乱数を生成
        inverse_metal_x_data_Y.loc[atom_nums_Y==1, sosei_list_Y[:1]] = np.round(general_x_Y / general_x_Y.sum(axis=1).reshape((-1, 1)) * Compo_ratio_Y, decimals=2) # 割って0.01～0.29にする

        #組成金属が２つのもの
        #組成範囲が0-0.3となっていたため修正して0.01-0.29となるようにした(231116)
        #general_x_Y = np.random.rand(len(np.where(atom_nums_Y==2)[0]), 2)*(99-1)+1 # 乱数を生成
        #inverse_metal_x_data_Y.loc[atom_nums_Y==2, sosei_list_Y[:2]] = np.round(general_x_Y / general_x_Y.sum(axis=1).reshape((-1, 1)) * Compo_ratio_Y, decimals=2) # 割って0.01～0.29にする
        inverse_metal_x_data_Y.loc[atom_nums_Y==2, sosei_list_Y[0]] = np.round(np.random.rand(len(np.where(atom_nums_Y==2)[0]))*(Compo_ratio_Y-0.02)+0.01, decimals=2)
        inverse_metal_x_data_Y.loc[atom_nums_Y==2, sosei_list_Y[1]] = Compo_ratio_Y - inverse_metal_x_data_Y.loc[atom_nums_Y==2, sosei_list_Y[0]]
        
        return inverse_metal_x_data_Y
    
    
    def get_LinkingData(self, inverse_metal_x_data,inverse_metal_x_data_N,inverse_metal_x_data_Y):
    
        ##### Omit列：空欄のもの（N）、Yのものを連結 #####
        inverse_metal_x_data = inverse_metal_x_data_N.loc[:, ["metal1", "metal2", "metal3"]]
        inverse_metal_x_data = pd.concat([inverse_metal_x_data,inverse_metal_x_data_Y.loc[:, ["metal4", "metal5"]]],axis=1)
        inverse_metal_x_data = pd.concat([inverse_metal_x_data,inverse_metal_x_data_N.loc[:, ["ratio1", "ratio2", "ratio3"]]],axis=1)
        inverse_metal_x_data = pd.concat([inverse_metal_x_data,inverse_metal_x_data_Y.loc[:, ["ratio4", "ratio5"]]],axis=1)
        # metal ratio それそれ空欄を左につめる
        # 'metal3'がNoneの場合
        for i in inverse_metal_x_data.index:
            if (inverse_metal_x_data.loc[i,'metal3'] is None):
                inverse_metal_x_data.loc[i,'metal3'] = inverse_metal_x_data.loc[i,'metal4']
                inverse_metal_x_data.loc[i,'metal4'] = inverse_metal_x_data.loc[i,'metal5']
                if (inverse_metal_x_data.loc[i,'metal5'] is not None):
                    inverse_metal_x_data.loc[i,'metal5'] = None
        # 'metal2'がNoneの場合
        for i in inverse_metal_x_data.index:
            if (inverse_metal_x_data.loc[i,'metal2'] is None):
                inverse_metal_x_data.loc[i,'metal2'] = inverse_metal_x_data.loc[i,'metal3']
                inverse_metal_x_data.loc[i,'metal3'] = inverse_metal_x_data.loc[i,'metal4']
                if (inverse_metal_x_data.loc[i,'metal4'] is not None):
                    inverse_metal_x_data.loc[i,'metal4'] = None
        # 'raito3'がnp.nanの場合
        for i in inverse_metal_x_data.index:
            if (inverse_metal_x_data.loc[i,'ratio3'] is np.nan):
                inverse_metal_x_data.loc[i,'ratio3'] = inverse_metal_x_data.loc[i,'ratio4']
                inverse_metal_x_data.loc[i,'ratio4'] = inverse_metal_x_data.loc[i,'ratio5']
                if (inverse_metal_x_data.loc[i,'ratio5'] is not np.nan):
                    inverse_metal_x_data.loc[i,'ratio5'] = np.nan
        # 'raito2'がnp.nanの場合
        for i in inverse_metal_x_data.index:
            if (inverse_metal_x_data.loc[i,'ratio2'] is np.nan):
                inverse_metal_x_data.loc[i,'ratio2'] = inverse_metal_x_data.loc[i,'ratio3']
                inverse_metal_x_data.loc[i,'ratio3'] = inverse_metal_x_data.loc[i,'ratio4']
                if (inverse_metal_x_data.loc[i,'ratio4'] is not np.nan):
                    inverse_metal_x_data.loc[i,'ratio4'] = np.nan

        return inverse_metal_x_data
    
    
    def createFileName(self, x_name, constraints, iter_n):
        if (iter_n == 0):
            fname_exp = x_name + '_' + constraints + '_x_data_for_inverse_analysis_exp_vals.csv'  
            fname_xenon = x_name + '_' + constraints + '_x_data_for_inverse_analysis_xenon_vals.csv'
            fname_matminer = x_name + '_' + constraints + '_x_data_for_inverse_analysis_matminer_vals.csv'
            fname_metal = x_name + '_' + constraints + '_metal_x_data_for_inverse_analysis.csv'
        else:    
            fname_exp = x_name + '_' + constraints + '_iter_' + str(iter_n) + '_x_data_for_inverse_analysis_exp_vals.csv'
            fname_xenon = x_name + '_' + constraints + '_iter_' + str(iter_n) + '_x_data_for_inverse_analysis_xenon_vals.csv'
            fname_matminer = x_name + '_' + constraints + '_iter_' + str(iter_n) + '_x_data_for_inverse_analysis_matminer_vals.csv'
            fname_metal = x_name + '_' + constraints + '_iter_' + str(iter_n) + '_metal_x_data_for_inverse_analysis.csv'

        return fname_exp,fname_xenon,fname_matminer,fname_metal
    
    def createFileName_2(self, x_name, constraints, iter_n):
        if (iter_n == 0):
            fname_desc = x_name + '-' + constraints + '_descriptors.csv'
        else:
            fname_desc = x_name + '-' + constraints + '_iter_' + str(iter_n) + '_descriptors.csv'
            print(fname_desc)
        return fname_desc
    
    def checkMetalRatioTotal(self, inverse_metal_x_data):
        #metalとratioのdfを定義
        metal_ratio_df = pd.DataFrame()
        
        #input情報からmetal情報を抜取り
        metal_ratio_df = inverse_metal_x_data.loc[:,'metal1':'ratio5']
        #metal_ratio_df.replace(np.nan, 0, inplace= True)
        #zero_checker = metal_ratio_df[metal_ratio_df.eq(0).any(axis= 1)]   
                   
        #「1.00」にならない項目を検知、差分を保持
        ratio_df_1 = pd.DataFrame()
        ratio_df_1 = inverse_metal_x_data.loc[:,'ratio1':'ratio5']
        ratio_df_1 = ratio_df_1.astype('float')
        ratio_df_2 = pd.DataFrame()
        ratio_df_2['sum'] = ratio_df_1.sum(axis=1)
        ratio_df_2['sum_diff'] = np.round(1.00 - ratio_df_2['sum'], decimals=2)    
    
        #最大値のcolumnに差分を加算する。→「1.00」になる
        df_min_col = ratio_df_1.idxmax(axis=1) ##ratio1～ratio5で最大値のcolumnを取得
        metal_ratio_df[df_min_col[0]] = np.round(metal_ratio_df[df_min_col[0]].astype('float') + ratio_df_2['sum_diff'], decimals=2)
        #metal_ratio_df[df_min_col[0]] = np.round(metal_ratio_df[df_min_col[0]] + ratio_df_2['sum_diff'], decimals=2)
        #metal_ratio_df.replace(0, np.nan, inplace= True)
        
        zero_checker = metal_ratio_df[metal_ratio_df.eq(0).any(axis= 1)]
        return metal_ratio_df
    
    def checkMetalAlphabetOrder(self, inverse_metal_x_data):
                
        #内部処理用DF        
        metal_ratio_df = pd.DataFrame()
        metal_ratio_df = inverse_metal_x_data.loc[:,'metal1':'ratio5']
        
        metal_ratio_back_df = pd.DataFrame()
        metal_ratio_back_df_sub = pd.DataFrame()
        dset_indexs = inverse_metal_x_data.index
                
        # datasetをindex1分回す
        for i, x in enumerate(range(inverse_metal_x_data.shape[0])):
        
            # Dataframe　→ listに変換
            metal_list_1 = []
            metal_list_1 = inverse_metal_x_data.loc[i,'metal1':'metal5'].to_list()
            ratio_list_1 = []
            ratio_list_1 = inverse_metal_x_data.loc[i,'ratio1':'ratio5'].to_list()
     

            #print("metal_list_1 : ", metal_list_1)
            #print("ratio_list_1 : ", ratio_list_1)
            # list → array 2次元にする(ratioも同時にsortするため)
            array_3 = np.column_stack((metal_list_1,ratio_list_1))
            #print("array_3 : ", array_3)
            # list → array 2次元にする(ratioも同時にsortす
            """
            for i, val in enumerate(array_3):
                #v138で空文字が入っていたので対応した
                if(array_3[i][0]==' ') is True:
                    array_3[i][0] = 'zzzzz'
                if(array_3[i][1]==' ') is True:
                    array_3[i][1] = 'zzzzz'
            """
            # アルファベット順(昇順)
            new_arr = sorted(array_3, reverse=False, key=lambda x:x[0])
            #print("new_arr : ", new_arr)
            
            # array →　Dataframeに戻す
            # 1.array → list
            list_back_1_A = []
            list_back_1_B = []
            for idx, x in enumerate(range(5)):
                list_back_1_A.append(new_arr[idx][0])
                list_back_1_B.append(new_arr[idx][1].astype('float64'))
                #list_back_1_B.append(new_arr[idx][1])
            # 2.list → Series
            series_back_1_A = pd.Series(list_back_1_A)
            series_back_1_B = pd.Series(list_back_1_B)
            # 3.reshape
            chang_style_back_1_A = series_back_1_A.values.reshape((1,-1))
            chang_style_back_1_B = series_back_1_B.values.reshape((1,-1))
            # 4.Series → DataFrame
            metal_df = pd.DataFrame(chang_style_back_1_A)
            ratio_df = pd.DataFrame(chang_style_back_1_B)
            metal_ratio_back_df_sub = pd.concat([metal_df,ratio_df], axis=1)
            
            # 戻し用dfに保持
            metal_ratio_back_df = pd.concat([metal_ratio_back_df,metal_ratio_back_df_sub])
            
        # 元のcolumnsに戻す
        metal_ratio_back_df.index = dset_indexs
        metal_ratio_back_df.columns = metal_ratio_df.columns
        #v138で空文字が入っていたので対応した
        #metal_ratio_back_df.replace('zzzzz', np.nan, inplace= True)
        
        inverse_metal_x_data[inverse_metal_x_data.columns[0:10]] = metal_ratio_back_df[inverse_metal_x_data.columns[0:10]].values
        inverse_metal_x_data.index = dset_indexs
        
        return inverse_metal_x_data
        
    
    """
    def createSample_X1(self, x_data_for_inverse_analysis, inverse_metal_x_data
                            , number_of_generating_samples, xenonpy_element_data):
        # 'x1'の時
        
        # 組成の記述子を計算
        weighted_average_name = list() # 加重平均の index 名
        weighted_variance_name = list() # 加重分散の index 名
        geometric_mean_name = list() # 幾何平均の index 名
        harmonic_mean_name = list() # 調和平均の index 名
        max_pooling_name = list() # 最大値の index 名
        min_pooling_name = list() # 最小値の index 名

        # 名前のリスト
        for j in xenonpy_element_data.columns:
            weighted_average_name.append(f'ave_{j}')
            weighted_variance_name.append(f'var_{j}')
            geometric_mean_name.append(f'gmean_{j}')
            harmonic_mean_name.append(f'hmean_{j}')
            max_pooling_name.append(f'max_{j}')
            min_pooling_name.append(f'min_{j}')

        # 保存用dfの生成
        x_metaldesc = pd.DataFrame(
            # index=range(number_of_generating_samples),
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )
        x_metaldesc_temp = pd.DataFrame(
            # index=range(number_of_generating_samples),
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )

        for i in range(number_of_generating_samples):
            print(f'\r{i+1} / {number_of_generating_samples}', end='')
            # 金属種の取得
            # metal1 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種1']
            # metal2 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種2']
            # metal3 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種3']
            # metal4 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種4']
            # metal5 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種5']
            metal_list = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], ['metal1', 'metal2', 'metal3', 'metal4', 'metal5']]
            # 金属組成の取得
            metal_rate1 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio1']
            metal_rate2 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio2']
            metal_rate3 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio3']
            metal_rate4 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio4']
            metal_rate5 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio5']

            # 金属の数に応じた処理
            if metal_list[4] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                metal_desc4 = xenonpy_element_data.loc[metal_list[3], :].values
                metal_desc5 = xenonpy_element_data.loc[metal_list[4], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])
            elif metal_list[3] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                metal_desc4 = xenonpy_element_data.loc[metal_list[3], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])
            elif metal_list[2] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3])
            else:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                mt = np.array([metal_desc1, metal_desc2])
                mr = np.array([metal_rate1, metal_rate2])

            for desc in range(xenonpy_element_data.shape[1]):
                d_name = xenonpy_element_data.columns[desc]
                if np.isnan(mt[:, desc]).any():
                    x_metaldesc_temp.loc[i, f'ave_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'var_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'gmean_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'hmean_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'max_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'min_{d_name}'] = np.nan
                    continue
                x_metaldesc_array = np.array([
                    np.dot(mt[:, desc],mr) / np.sum(mr),
                    np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr),
                    np.prod(mt[:, desc]**mr)**(1/sum(mr)),
                    sum(mr)/sum((1/mt[:, desc])*mr),
                    max(mt[:, desc]),
                    min(mt[:, desc])
                    ])
                x_metaldesc_array = pd.Series(x_metaldesc_array, index=[f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}', f'min_{d_name}'])
                # x_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
                # x_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr)
                # x_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
                # x_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
                # x_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
                # x_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])
                # x_metaldesc[[f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}', f'min_{d_name}']].append(x_metaldesc_array)
                x_metaldesc_temp.loc[i, [f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}', f'min_{d_name}']] = x_metaldesc_array
            if x_metaldesc_temp.shape[0] >= 50:
                x_metaldesc = x_metaldesc.append(x_metaldesc_temp)
                x_metaldesc_temp = pd.DataFrame(
                    # index=range(number_of_generating_samples),
                    columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
                    )

        print()
        x_metaldesc = x_metaldesc.replace([np.inf, -np.inf], np.nan)
        # x_metaldesc = x_metaldesc.iloc[:,x_metaldesc.notna().all(axis=0).values]
        x_metaldesc = x_metaldesc.fillna(0)
    
        # x_metaldescから設定するもの以外は、
        # '前処理還元炉温℃', '評価反応炉温℃', '評価内温℃', '還元温度内温℃', 'temp', 'flow_NaOH',　'flow_slurry', 'flow_red', 'wash',
        # 'support_Al2O3_A-11','support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100'
        #idx_of_first_metaldescはxenonpy記述子の最初のcolumns名を検出するもの
        idx_of_first_metaldesc = x_data_for_inverse_analysis.columns.tolist().index(x_metaldesc.columns[0])
        x_data_for_inverse_analysis[x_data_for_inverse_analysis.columns[idx_of_first_metaldesc:]] = \
            (x_metaldesc[x_data_for_inverse_analysis.columns[idx_of_first_metaldesc:]].values).astype(float)

        return x_data_for_inverse_analysis
    """

    def createSample_X10(self, x_data_for_inverse_analysis, inverse_metal_x_data
                            , number_of_generating_samples, xenonpy_element_data):
        # 'x10'の時
    
        # 組成の記述子を計算
        weighted_average_name = list() # 加重平均の index 名
        weighted_variance_name = list() # 加重分散の index 名
        max_pooling_name = list() # 最大値の index 名
        min_pooling_name = list() # 最小値の index 名
        subtraction_name = list() #差のindex名
        division_name = list()    #商のindex名

        # 名前のリスト
        for j in xenonpy_element_data.columns:
            weighted_average_name.append(f'ave_{j}')
            weighted_variance_name.append(f'var_{j}')
            max_pooling_name.append(f'max_{j}')
            min_pooling_name.append(f'min_{j}')
            subtraction_name.append(f'subtr_{j}')
            division_name.append(f'div_{j}')

        # 保存用dfの生成
        x_metaldesc = pd.DataFrame(
            # index=range(number_of_generating_samples),
            columns=weighted_average_name+weighted_variance_name+max_pooling_name+min_pooling_name+subtraction_name+division_name
            )
        x_metaldesc_temp = pd.DataFrame(
            # index=range(number_of_generating_samples),
            columns=weighted_average_name+weighted_variance_name+max_pooling_name+min_pooling_name+subtraction_name+division_name
            )

        for i in range(number_of_generating_samples):
            print(f'\r{i+1} / {number_of_generating_samples}', end='')
            # 金属種の取得
            # metal1 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種1']
            # metal2 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種2']
            # metal3 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種3']
            # metal4 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種4']
            # metal5 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], '金属種5']
            metal_list = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], ['metal1', 'metal2', 'metal3', 'metal4', 'metal5']]
            # 金属組成の取得
            metal_rate1 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio1']
            metal_rate2 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio2']
            metal_rate3 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio3']
            metal_rate4 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio4']
            metal_rate5 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio5']
        
            # 金属の数に応じた処理
            if metal_list[4] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                metal_desc4 = xenonpy_element_data.loc[metal_list[3], :].values
                metal_desc5 = xenonpy_element_data.loc[metal_list[4], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5], dtype= float)
            elif metal_list[3] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                metal_desc4 = xenonpy_element_data.loc[metal_list[3], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4], dtype= float)
            elif metal_list[2] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3], dtype= float)
            else:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                mt = np.array([metal_desc1, metal_desc2], dtype= float)
                mr = np.array([metal_rate1, metal_rate2], dtype= float)
            for desc in range(xenonpy_element_data.shape[1]):
                d_name = xenonpy_element_data.columns[desc]
                if np.isnan(mt[:, desc]).any():
                    x_metaldesc_temp.loc[i, f'ave_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'var_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'max_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'min_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'subtr_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'div_{d_name}'] = np.nan            
                    continue
                x_metaldesc_array = np.array([
                    np.dot(mt[:, desc],mr) / np.sum(mr),
                    np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr),
                    max(mt[:, desc]),
                    min(mt[:, desc]),
                    max(mt[:, desc]) - min(mt[:, desc]),
                    max(mt[:, desc]) / min(mt[:, desc])
                    ])
                x_metaldesc_array = pd.Series(x_metaldesc_array, index=[f'ave_{d_name}', f'var_{d_name}', f'max_{d_name}', f'min_{d_name}',f'subtr_{d_name}',f'div_{d_name}'])
                # x_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
                # x_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr)
                # x_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
                # x_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
                # x_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
                # x_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])
                # x_metaldesc[[f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}', f'min_{d_name}']].append(x_metaldesc_array)
                x_metaldesc_temp.loc[i, [f'ave_{d_name}', f'var_{d_name}', f'max_{d_name}', f'min_{d_name}',f'subtr_{d_name}',f'div_{d_name}']] = x_metaldesc_array
            if x_metaldesc_temp.shape[0] >= 50:
                x_metaldesc = x_metaldesc.append(x_metaldesc_temp)
                x_metaldesc_temp = pd.DataFrame(
                    # index=range(number_of_generating_samples),
                    columns=weighted_average_name+weighted_variance_name+max_pooling_name+min_pooling_name+subtraction_name+division_name
                    )
      
        print()
        x_metaldesc = x_metaldesc.replace([np.inf, -np.inf], np.nan)
        # x_metaldesc = x_metaldesc.iloc[:,x_metaldesc.notna().all(axis=0).values]
        x_metaldesc = x_metaldesc.fillna(0)
        
        # x_metaldescから設定するもの以外は、
        # '前処理還元炉温℃', '評価反応炉温℃', '評価内温℃', '還元温度内温℃', 'temp', 'flow_NaOH',　'flow_slurry', 'flow_red', 'wash',
        # 'support_Al2O3_A-11','support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100'
        idx_of_first_metaldesc = x_data_for_inverse_analysis.columns.tolist().index(x_metaldesc.columns[0])
        x_data_for_inverse_analysis[x_data_for_inverse_analysis.columns[idx_of_first_metaldesc:]] = \
            (x_metaldesc[x_data_for_inverse_analysis.columns[idx_of_first_metaldesc:]].values).astype(float)
    
        return x_data_for_inverse_analysis
    
#    def CreateSample_x22(self, x_data_for_inverse_analysis, inverse_metal_x_data
#                         , number_of_generating_samples, xenonpy_element_data):
    
    def createSample_X1_3(self, x_data_for_inverse_analysis, inverse_metal_x_data
                            , number_of_generating_samples, xenonpy_element_data):
        # 'x1_3'の時
        
        # 組成の記述子を計算
        weighted_average_name = list() # 加重平均の index 名
        weighted_variance_name = list() # 加重分散の index 名
        geometric_mean_name = list() # 幾何平均の index 名
        harmonic_mean_name = list() # 調和平均の index 名
        max_pooling_name = list() # 最大値の index 名
        min_pooling_name = list() # 最小値の index 名

        # 名前のリスト
        for j in xenonpy_element_data.columns:
            weighted_average_name.append(f'ave_{j}')
            weighted_variance_name.append(f'var_{j}')
            geometric_mean_name.append(f'gmean_{j}')
            harmonic_mean_name.append(f'hmean_{j}')
            max_pooling_name.append(f'max_{j}')
            min_pooling_name.append(f'min_{j}')

        # 保存用dfの生成
        x_metaldesc = pd.DataFrame(
            # index=range(number_of_generating_samples),
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )
        x_metaldesc_temp = pd.DataFrame(
            # index=range(number_of_generating_samples),
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )
        for i in range(number_of_generating_samples):
            print(f'\r{i+1} / {number_of_generating_samples}', end='')
            # 金属種の取得
            metal_list = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], ['metal1', 'metal2', 'metal3', 'metal4', 'metal5']]
            # 金属組成の取得
            metal_rate1 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio1']
            metal_rate2 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio2']
            metal_rate3 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio3']
            metal_rate4 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio4']
            metal_rate5 = inverse_metal_x_data.loc[inverse_metal_x_data.index[i], 'ratio5']

            # 金属の数に応じた処理
            if metal_list[4] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                metal_desc4 = xenonpy_element_data.loc[metal_list[3], :].values
                metal_desc5 = xenonpy_element_data.loc[metal_list[4], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])
            elif metal_list[3] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                metal_desc4 = xenonpy_element_data.loc[metal_list[3], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])
            elif metal_list[2] is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                metal_desc3 = xenonpy_element_data.loc[metal_list[2], :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3])
            else:
                metal_desc1 = xenonpy_element_data.loc[metal_list[0], :].values
                metal_desc2 = xenonpy_element_data.loc[metal_list[1], :].values
                mt = np.array([metal_desc1, metal_desc2])
                mr = np.array([metal_rate1, metal_rate2])

            for desc in range(xenonpy_element_data.shape[1]):
                d_name = xenonpy_element_data.columns[desc]
                if np.isnan(mt[:, desc]).any():
                    x_metaldesc_temp.loc[i, f'ave_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'var_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'gmean_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'hmean_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'max_{d_name}'] = np.nan
                    x_metaldesc_temp.loc[i, f'min_{d_name}'] = np.nan
                    continue
                x_metaldesc_array = np.array([
                    np.dot(mt[:, desc],mr) / np.sum(mr),
                    np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr),
                    np.prod(mt[:, desc]**mr)**(1/sum(mr)),
                    sum(mr)/sum((1/mt[:, desc])*mr),
                    max(mt[:, desc]),
                    min(mt[:, desc])
                    ])
                x_metaldesc_array = pd.Series(x_metaldesc_array, index=[f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}', f'min_{d_name}'])
                x_metaldesc_temp.loc[i, [f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}', f'min_{d_name}']] = x_metaldesc_array

        x_metaldesc = x_metaldesc.append(x_metaldesc_temp)
            
            #if x_metaldesc_temp.shape[0] >= 50:
            #    print("x_metaldesc_temp.shape[0] >= 50")
            #    x_metaldesc = x_metaldesc.append(x_metaldesc_temp)
            #    x_metaldesc_temp = pd.DataFrame(
            #        # index=range(number_of_generating_samples),
            #        columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            #        )

            
        print()
        x_metaldesc = x_metaldesc.replace([np.inf, -np.inf], np.nan)
        # x_metaldesc = x_metaldesc.iloc[:,x_metaldesc.notna().all(axis=0).values]
        x_metaldesc = x_metaldesc.fillna(0)   #x1も0埋めしていたことに気がついた・・
    
        # x_metaldescから設定するもの以外は、
        # '前処理還元炉温℃', '評価反応炉温℃', '評価内温℃', '還元温度内温℃', 'temp', 'flow_NaOH',　'flow_slurry', 'flow_red', 'wash',
        # 'support_Al2O3_A-11','support_CeO2_HS', 'support_TiO2_SSP-M', 'support_ZrO2_RC100'
        boruta_descriptors = x_data_for_inverse_analysis.columns[11:]
        x_data_for_inverse_analysis[x_data_for_inverse_analysis.columns[11:]]=\
            (x_metaldesc[boruta_descriptors].values).astype(float)

        return x_data_for_inverse_analysis
    
    
    def createSample_X22(self, x_data_for_inverse_analysis, inverse_metal_x_data):
        from matminer.featurizers.conversions import StrToComposition
        from matminer.featurizers.base import MultipleFeaturizer, BaseFeaturizer
        from matminer.featurizers.composition.alloy import Miedema, YangSolidSolution, WenAlloys
        from matminer.featurizers.composition.element import BandCenter, TMetalFraction
        from matminer.featurizers.composition.ion import OxidationStates, IonProperty, ElectronAffinity, ElectronegativityDiff
        from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital
        from matminer.featurizers.composition.composite import ElementProperty, Meredig
        from matminer.featurizers.composition.packing import AtomicPackingEfficiency
        from matminer.featurizers.composition.thermo import CohesiveEnergy, CohesiveEnergyMP

        inverse_metal_x_data.replace(0, np.nan, inplace= True)
        for index, row in inverse_metal_x_data.iterrows():
            metal1 = inverse_metal_x_data.loc[index, 'metal1']
            metal2 = inverse_metal_x_data.loc[index, 'metal2']
            metal3 = inverse_metal_x_data.loc[index, 'metal3']
            metal4 = inverse_metal_x_data.loc[index, 'metal4']
            metal5 = inverse_metal_x_data.loc[index, 'metal5']
            ratio1 = inverse_metal_x_data.loc[index, 'ratio1']
            ratio2 = inverse_metal_x_data.loc[index, 'ratio2']
            ratio3 = inverse_metal_x_data.loc[index, 'ratio3']
            ratio4 = inverse_metal_x_data.loc[index, 'ratio4']
            ratio5 = inverse_metal_x_data.loc[index, 'ratio5']
            inverse_metal_x_data.loc[index, 'chemicalFormula'] = metal1+str(ratio1)+metal2+str(ratio2)+str(metal3)+str(ratio3)+\
                str(metal4)+str(ratio4)+str(metal5)+str(ratio5)
        inverse_metal_x_data['chemicalFormula'] = inverse_metal_x_data['chemicalFormula'].str.rstrip('nan')

        #このセルは範囲指定で動かさないと動かない
        inverse_metal_x_data = StrToComposition(target_col_id='composition').featurize_dataframe(inverse_metal_x_data, "chemicalFormula", ignore_errors=True)
        
        #featurizerの指定　１つであればfeaturize_dataframeを使用
        f_Miedema = Miedema()
        f_Meredig = Meredig()
        f_BandCenter = BandCenter()
        f_WenAlloys = WenAlloys()
        f_AtomicOrbitals = AtomicOrbitals()
        #WenAlloysを使用すると原因不明エラー発生 ignore_errors=Trueが必要
        features_Miedema = f_Miedema.featurize_dataframe(inverse_metal_x_data, col_id='composition', ignore_errors= True)
        features_Meredig = f_Meredig.featurize_dataframe(inverse_metal_x_data, col_id='composition', ignore_errors= True)
        features_BandCenter = f_BandCenter.featurize_dataframe(inverse_metal_x_data, col_id='composition', ignore_errors= True)
        features_WenAlloys = f_WenAlloys.featurize_dataframe(inverse_metal_x_data, col_id='composition', ignore_errors= True)
        features_AtomicOrbitals= f_AtomicOrbitals.featurize_dataframe(inverse_metal_x_data, col_id='composition', ignore_errors= True)
        
        inverse_metal_x_data_2 = features_Miedema.drop(['chemicalFormula','composition'], axis= 1)
        inverse_metal_x_data_4 = features_Meredig.iloc[:, -17:]
        inverse_metal_x_data_5 = features_BandCenter.iloc[:, -1:]
        inverse_metal_x_data_7 = features_WenAlloys.iloc[:, -23:]
        inverse_metal_x_data_8 = features_AtomicOrbitals.iloc[:, -5:]
        inverse_metal_x_data_8.drop(['LUMO_character','LUMO_element'], axis= 1, inplace= True) #AtomicOrbitals用
        inverse_metal_x_data_1 = pd.concat([inverse_metal_x_data_2, inverse_metal_x_data_4, inverse_metal_x_data_5, inverse_metal_x_data_7, inverse_metal_x_data_8], axis= 1)
        inverse_metal_x_data_1.drop(['frac s valence electrons','frac p valence electrons','frac d valence electrons','frac f valence electrons'], axis=1, inplace=True)

        #iGMR, sklearnの種々のImputerを試してみる。
        df_for_imputer= inverse_metal_x_data_1.iloc[:, 10:]
        outlier= (df_for_imputer>2e2)|(df_for_imputer<-1e6)
        outlier_columns= outlier.sum()
        df_for_imputer.mask(outlier, np.nan, inplace= True)

        #iterativeimputerを試したが、まともな値を返してこない
        #異常値があることが原因であるため、異常値を排除する必要がある
        #デフォルトのestimatorはBaysianRidge

        imputer = IterativeImputer(max_iter= 10, random_state= 10)
        imputed_df= imputer.fit_transform(df_for_imputer)

        imputed_df= pd.DataFrame(imputed_df, index= inverse_metal_x_data_1.index)
        outlier_after_impute= (imputed_df>2e2)|(imputed_df<-1e6)
        outlier_columns_after_impute= outlier_after_impute.sum()

        #impute後の処理
        imputed_df= pd.concat([inverse_metal_x_data_1.iloc[:,0:10], imputed_df], axis= 1)
        #imputed_df.insert(loc= 4, column= '触媒ロット', value= inverse_metal_x_data_1['触媒ロット'])
        imputed_df.columns= inverse_metal_x_data_1.columns
        inverse_metal_x_data= imputed_df
        #x_data_for_inverse_analysis= pd.concat([x_data_for_inverse_analysis.iloc[:, 0:11], inverse_metal_x_data.iloc[:, 10:]])
        descriptors = inverse_metal_x_data.columns[10:]
        x_data_for_inverse_analysis[x_data_for_inverse_analysis.columns[11:]]=\
            (inverse_metal_x_data[descriptors].values).astype(float)
                
        return x_data_for_inverse_analysis
        
        
        
        
        
        
        