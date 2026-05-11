import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import time
from libs.InverseAnalysisUtility import MIutility

class Calc_desc():
    def __init__(self):
        pass

    def getDesc_X1X2(self, original_data):
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
        rd = 'results\\'
        evaluation_lot_data = pd.read_excel(rd+'evaluation_lot_data.xlsx', index_col=0, header=0, engine='openpyxl')
        evaluation_y_x_data_with_dummy_variables = pd.read_excel(rd+'evaluation_y_x_data_with_dummy_variables.xlsx', index_col=0, header=0, engine='openpyxl')
        synthesis_condition_x_data_with_dummy_variables = pd.read_excel(rd+'synthesis_condition_x_data_with_dummy_variables.xlsx', index_col=0, header=0, engine='openpyxl')
        synthesis_metal_x_data = pd.read_excel(rd+'synthesis_metal_x_data.xlsx', index_col=0, header=0, engine='openpyxl')
        xenonpy_element_data = pd.read_csv(rd+'xenonpy_element_data240515.csv', index_col=0)
        xenonpy_element_data.drop(['hhi_p','hhi_r'], axis= 1, inplace= True)
        extract_column = ['転化率※※','評価反応炉温℃']
        evaluation_data = original_data
        evaluation_data = evaluation_data.dropna(how='all').dropna(how='all', axis=1)    
        
        # ダミー変数の処理
        if synthesis_condition_x_data_with_dummy_variables.isin([True, False]).any().any():
            bool_col = [c for c in synthesis_condition_x_data_with_dummy_variables.columns if 'support' in c]
            synthesis_condition_x_data_with_dummy_variables[bool_col] = synthesis_condition_x_data_with_dummy_variables[bool_col].astype(int)

        #コラム名作成
        weighted_average_name = list() # 加重平均の index 名
        weighted_variance_name = list() # 加重分散の index 名
        geometric_mean_name = list() # 幾何平均の index 名
        harmonic_mean_name = list() # 調和平均の index 名
        max_pooling_name = list() # 最大値の index 名
        min_pooling_name = list() # 最小値の index 名
        for j in xenonpy_element_data.columns:
            weighted_average_name.append(f'ave_{j}')
            weighted_variance_name.append(f'var_{j}')
            geometric_mean_name.append(f'gmean_{j}')
            harmonic_mean_name.append(f'hmean_{j}')
            max_pooling_name.append(f'max_{j}')
            min_pooling_name.append(f'min_{j}')

        #x1の作成
        metal_x = synthesis_metal_x_data.loc[:, ['metal1', 'metal2', 'metal3', 'metal4', 'metal5', 'ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5']]
        #metal_x = metal_x.iloc[:100, :]
        evaluation_y_x_data_with_dummy_variables = evaluation_y_x_data_with_dummy_variables.drop_duplicates()
        evaluation_lot_data = evaluation_lot_data.drop_duplicates()
        x1 = evaluation_y_x_data_with_dummy_variables.copy()
        #x1_error = x1[x1.index.duplicated()]
        #evaluation_error = evaluation_lot_data[evaluation_lot_data.index.duplicated()]
        x1['触媒ロット'] = evaluation_lot_data.loc[evaluation_y_x_data_with_dummy_variables.index, '触媒ロット']
        x1['評価ロット数字'] = x1.index
        #目的変数の内行は削除
        x1.dropna(subset=['選択率NPA※※'], axis= 0, inplace= True)
        #tantai_col = [i for i in synthesis_condition_x_data_with_dummy_variables.columns if '担体種類' in i]
        x1 = pd.merge(x1, synthesis_condition_x_data_with_dummy_variables, right_index= True, left_on= '触媒ロット')
        x1.drop_duplicates(subset= '評価ロット数字', inplace= True)
        #Seを含む触媒は削除しておく xenonpy_mergeにSeがないためErrorとなるため
        x1 = x1.query('触媒ロット != "sc000033"')
        x1 = x1.query('触媒ロット != "sc000255"')
        x1 = x1.query('触媒ロット != "sc000303"')
        x1 = x1.query('触媒ロット != "sc000329"')
        x1 = x1.query('触媒ロット != "sc000372"')
        x1 = x1.query('触媒ロット != "sc000368"')
        x1 = x1.query('触媒ロット != "sc000369"')
        x1 = x1.query('触媒ロット != "sc001418"')                    
        x2 = x1.copy()

        # Dubug用処理　実行時はコメントアウトすること
        #original_data = original_data.iloc[: 100, :]
        #x1 = x1.iloc[: 100, :]
        #x2 = x2.iloc[: 100, :]
        #metal_x = metal_x.iloc[: 600, :]

        x1_metaldesc = pd.DataFrame(
            index=x1.index,
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )
        for i in range(x1.shape[0]):
            c_lot = x1['触媒ロット'].iloc[i]
            metal1 = metal_x.loc[c_lot, 'metal1']
            metal2 = metal_x.loc[c_lot, 'metal2']
            metal3 = metal_x.loc[c_lot, 'metal3']
            metal4 = metal_x.loc[c_lot, 'metal4']
            metal5 = metal_x.loc[c_lot, 'metal5']
            metal_rate1 = metal_x.loc[c_lot, 'ratio1']
            metal_rate2 = metal_x.loc[c_lot, 'ratio2']
            metal_rate3 = metal_x.loc[c_lot, 'ratio3']
            metal_rate4 = metal_x.loc[c_lot, 'ratio4']
            metal_rate5 = metal_x.loc[c_lot, 'ratio5']
            if isinstance(metal1, str) == False:
                metal1 = metal1.iloc[0]
                metal2 = metal2.iloc[0]
                metal3 = metal3.iloc[0]
                metal4 = metal4.iloc[0]
                metal5 = metal5.iloc[0]
                metal_rate1 = metal_rate1.iloc[0]                
                metal_rate2 = metal_rate2.iloc[0]                 
                metal_rate3 = metal_rate3.iloc[0]
                metal_rate4 = metal_rate4.iloc[0]                 
                metal_rate5 = metal_rate5.iloc[0]
            if metal5 is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data.loc[metal3, :].values
                metal_desc4 = xenonpy_element_data.loc[metal4, :].values        
                metal_desc5 = xenonpy_element_data.loc[metal5, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])
            elif metal4 is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data.loc[metal3, :].values
                metal_desc4 = xenonpy_element_data.loc[metal4, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])    
            elif metal3 is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3])
            elif metal2 is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2])
                mr = np.array([metal_rate1, metal_rate2])                         
            else:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                mt = np.array([metal_desc1])
                mr = np.array([metal_rate1])
            for desc in range(xenonpy_element_data.shape[1]):
                d_name = xenonpy_element_data.columns[desc]
                if np.isnan(mt[:, desc]).any():
                    x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.nan
                    x1_metaldesc[f'var_{d_name}'].iloc[i] = np.nan
                    x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.nan
                    x1_metaldesc[f'hmean_{d_name}'].iloc[i] = np.nan
                    x1_metaldesc[f'max_{d_name}'].iloc[i] = np.nan
                    x1_metaldesc[f'min_{d_name}'].iloc[i] = np.nan
                    continue
                x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
                #x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr)
                x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)
                x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
                x1_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
                x1_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
                x1_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])

        x1_metaldesc = x1_metaldesc.replace([np.inf, -np.inf], np.nan)
        x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]
        x1 = pd.concat([x1, x1_metaldesc], axis=1)
        x1 = x1.sort_index()
        x1.columns = [i.replace('※', '') if '※' in i else i for i in x1.columns]
        x1.drop('評価ロット数字', axis= 1, inplace= True)
        drop_category = ['8_1','8_2','8_3','11_1','11_2','11_3','11_3_1','11_4','11_5','11_6','12_1','12_2','12_3','12_4','12_5','12_6','14']
        #drop_category = ['13_1']
        category_not_8 = original_data.query('category not in @drop_category')
        category_not_8_cata = category_not_8.loc[:, '触媒ロット'].to_list()
        category_not_8_cata = set(category_not_8_cata)
        category_not_8_cata = list(category_not_8_cata)
        x1 = x1.query('触媒ロット in @category_not_8_cata')
        x1 = x1.sort_index()
        x1 = x1[~x1.index.duplicated()]
        # 収率NPAを追加
        conversion_rate = original_data[original_data.index.isin(x1.index)]
        conversion_rate = conversion_rate[~conversion_rate.index.duplicated()]
        x1['収率NPA'] = conversion_rate['転化率※※'] * x1['選択率NPA'] / 100
        x1_columns = ['選択率NPA','収率NPA'] + [s for s in x1.columns if 'NPA' not in s]
        x1 = x1.loc[:, x1_columns]
        x2['収率NPA'] = conversion_rate['転化率※※'] * x2['選択率NPA※※'] / 100
        x2_columns = ['選択率NPA※※','収率NPA'] + [s for s in x2.columns if 'NPA' not in s]
        x2 = x2.loc[:, x2_columns]       
        if '評価反応ガス流量NmL/min' in x1.columns:
            x1 = x1.drop('評価反応ガス流量NmL/min', axis= 1)
            x2 = x2.drop('評価反応ガス流量NmL/min', axis= 1)

        #X2の作成
        #縦軸に触媒ロット、横軸に元素(metal_list)を入れたdfの作成
        #synthesis_metal_x_data.drop_duplicates(inplace= True)
        #小部さん修正 上では触媒ロット異なり狙い組成が同じものがdropしてしまうため
        synthesis_metal_x_data_tmp = synthesis_metal_x_data.reset_index()
        synthesis_metal_x_data_tmp.drop_duplicates(inplace=True)
        synthesis_metal_x_data = synthesis_metal_x_data_tmp.set_index('触媒ロット', drop=True)

        synthesis_metal_x_data.sort_index(inplace= True)
        met1 = synthesis_metal_x_data['metal1'].unique()
        met2 = synthesis_metal_x_data['metal2'].unique()
        met3 = synthesis_metal_x_data['metal3'].unique()
        met4 = synthesis_metal_x_data['metal4'].unique()
        met5 = synthesis_metal_x_data['metal5'].unique()
        metal_list = np.concatenate([met1, met2, met3, met4, met5])
        metal_list = pd.Series(metal_list)
        metal_list = metal_list.dropna()
        metal_list = metal_list.unique()
        metal_list = np.sort(metal_list)
        synthesis_metal_x_data_modified = pd.DataFrame(index= synthesis_metal_x_data.index, columns= metal_list) 

        #触媒組成のデータの元素、組成を上のdf形式へ変換
        synthesis_metal_x_data_modified['dummy'] = np.nan
        for index, row in synthesis_metal_x_data.iterrows():
            content1 = row[0], row[5]
            content2 = row[1], row[6]
            content3 = row[2], row[7]
            content4 = row[3], row[8]
            content5 = row[4], row[9]
            if content2[0] is np.nan:
                content2 = list(content2)
                content2[0] = 'dummy'            
            if content3[0] is np.nan:
                content3 = list(content3)
                content3[0] = 'dummy'
            if content4[0] is np.nan:
                content4 = list(content4)
                content4[0] = 'dummy'
            if content5[0] is np.nan:
                content5 = list(content5)
                content5[0] = 'dummy'
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content1[0])] = content1[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content2[0])] = content2[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content3[0])] = content3[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content4[0])] = content4[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content5[0])] = content5[1]
        synthesis_metal_x_data_modified.drop('dummy', axis= 1, inplace= True)
        synthesis_metal_x_data_modified.replace(np.nan, 0, inplace= True)
            
        x2 = pd.merge(x2, synthesis_metal_x_data_modified, right_index= True, left_on= '触媒ロット')
        x2['評価ロット数字'] = x2.index
        x2.drop_duplicates(subset= '評価ロット数字', inplace= True)
        x2.replace(np.nan, 0, inplace= True)
        x2 = x2.loc[:, (x2 != 0).any(axis= 0)]
        x2.drop('評価ロット数字', axis= 1, inplace= True)

        #check = x2.iloc[:,14:]
        x2 = x2[x2.iloc[:,14:].sum(axis= 1)> 0]  #分析値が0のデータを削除
        x2.sort_index(inplace= True)
        x2.columns = [i.replace('※', '') if '※' in i else i for i in x2.columns]
        #x2.to_csv('datasets\\x2.csv', encoding='utf-8-sig')
        x2 = x2.query('触媒ロット in @category_not_8_cata')
        x2 = x2.sort_index()
        
        #230329_SCC条件を付加する。
        extract_df = evaluation_data[extract_column]
        extract_df_1 = extract_df[extract_df[extract_column[0]] >= 1]
        extract_df_2 = extract_df[(extract_df[extract_column[0]] >= 1) & (extract_df[extract_column[1]] == 300)]
        extract_df_1_error = extract_df_1[extract_df_1.index.duplicated()]
        extract_df_2_error = extract_df_2[extract_df_2.index.duplicated()]
        x1_error = x1[x1.index.duplicated()]
        extract_df_1 = extract_df_1.drop_duplicates()
        extract_df_2 = extract_df_2.drop_duplicates()
        x1_2 = pd.concat([x1, extract_df_1], axis= 1)
        x1_2.dropna(axis= 0, how= 'any', inplace= True)
        x1_2 = x1_2.iloc[:, :-2]
        x1_3 = pd.concat([x1, extract_df_2], axis= 1)
        x1_3.dropna(axis= 0, how= 'any', inplace= True)
        x1_3 = x1_3.iloc[:, :-2]        
        
        x2_2 = pd.concat([x2, extract_df_1], axis= 1)
        x2_2.dropna(axis= 0, how= 'any', inplace= True)
        x2_2 = x2_2.iloc[:, :-2]
        x2_3 = pd.concat([x2, extract_df_2], axis= 1)
        x2_3.dropna(axis= 0, how= 'any', inplace= True)
        x2_3 = x2_3.iloc[:, :-2]
        x1_4 = x1_2.copy()
        x1_5 = x1_3.copy()
        x2_4 = x2_2.copy()
        x2_5 = x2_3.copy()
        x1_41 = x1_2.copy()
        x1_51 = x1_3.copy()
        x2_41 = x2_2.copy()
        x2_51 = x2_3.copy()
        x1_4e = x1_2.copy()
        x2_4e = x2_2.copy()
        
        #category=5 に限定
        category_5 = original_data.query('category=="5" or category=="3_4_1" or category=="3_4_2" or category=="3_5_1" or category=="3_5_2" or category=="3_5_3"')
        category_5_cata = category_5.loc[:, '触媒ロット'].to_list()
        category_5_cata = set(category_5_cata)
        category_5_cata = list(category_5_cata)
        x1_6 = x1.query('触媒ロット in @category_5_cata')
        x2_6 = x2.query('触媒ロット in @category_5_cata')
        x1_7 = x1_6.copy()
        x2_7 = x2_6.copy()       

        #選択率がどちらも0を除外する
        x1['yield'] = x1['選択率NPA'] + x1['選択率エタノール']
        x1_a = x1[x1['yield']!= 0]
        x1_a= x1_a.drop('yield', axis= 1)
        x1_4['yield'] = x1_4['選択率NPA'] + x1_4['選択率エタノール']
        x1_4 = x1_4[x1_4['yield']!= 0]
        x1_4= x1_4.drop('yield', axis= 1)
        x1_5['yield'] = x1_5['選択率NPA'] + x1_5['選択率エタノール']
        x1_5 = x1_5[x1_5['yield']!= 0]
        x1_5= x1_5.drop('yield', axis= 1)        
        x1_7['yield'] = x1_7['選択率NPA'] + x1_7['選択率エタノール']
        x1_7 = x1_7[x1_7['yield']!= 0]
        x1_7= x1_7.drop('yield', axis= 1) 
        
        x2['yield'] = x2['選択率NPA'] + x2['選択率エタノール']
        x2_a = x2[x2['yield']!= 0]
        x2_a= x2_a.drop('yield', axis= 1)
        x2_4['yield'] = x2_4['選択率NPA'] + x2_4['選択率エタノール']
        x2_4 = x2_4[x2_4['yield']!= 0]
        x2_4= x2_4.drop('yield', axis= 1)
        x2_5['yield'] = x2_5['選択率NPA'] + x2_5['選択率エタノール']
        x2_5 = x2_5[x2_5['yield']!= 0]
        x2_5= x2_5.drop('yield', axis= 1)
        x2 = x2.drop('yield', axis= 1)
        x2_7['yield'] = x2_7['選択率NPA'] + x2_7['選択率エタノール']
        x2_7 = x2_7[x2_7['yield']!= 0]
        x2_7= x2_7.drop('yield', axis= 1)  
        
        #選択率がどちらか0を除外する
        x1_41 = x1_41[x1_41['選択率NPA'] != 0]
        x1_41 = x1_41[x1_41['選択率エタノール'] != 0]
        x1_51 = x1_51[x1_51['選択率NPA'] != 0]
        x1_51 = x1_51[x1_51['選択率エタノール'] != 0]        
        x2_41 = x2_41[x2_41['選択率NPA'] != 0]
        x2_41 = x2_41[x2_41['選択率エタノール'] != 0]        
        x2_51 = x2_51[x2_51['選択率NPA'] != 0]
        x2_51 = x2_51[x2_51['選択率エタノール'] != 0]    
        x1_4eN = x1_4e[x1_4e['選択率NPA'] == 0]
        x1_4eE = x1_4e[x1_4e['選択率エタノール'] == 0]
        x1_4e = pd.concat([x1_4eN, x1_4eE], axis= 0)    
        x1_4e = x1_4e.drop_duplicates()
        x2_4eN = x2_4e[x2_4e['選択率NPA'] == 0]
        x2_4eE = x2_4e[x2_4e['選択率エタノール'] == 0]
        x2_4e = pd.concat([x2_4eN, x2_4eE], axis= 0)    
        x2_4e = x2_4e.drop_duplicates()
        
        return_list = [x1, x1_a, x1_2, x1_3, x1_4, x1_5, x1_6, x1_7, x1_41, x1_51, x1_4e, x2, x2_a, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, x2_41, x2_51, x2_4e]
        for df in return_list:
            df.drop(columns = ['選択率エタノール'], inplace= True)

        return return_list
        
    def getDesc_X2_cross(self):
        def addSquareCross(df: pd.DataFrame):
            columns = df.columns
            for i, c1 in enumerate(columns):
                df[c1 + '^2'] = df[c1] ** 2
                for j, c2 in enumerate(columns):
                    if i <= j:
                        continue
                    df[c1 +' * ' + c2] = df[c1] * df[c2]
            return df
        
        #Borutaで特徴量を削減する
        def Boruta_Apply(df):
            y= df[['選択率NPA', '選択率エタノール']].copy()
            y['sum']= y['選択率NPA']+ y['選択率エタノール']
            y= y['sum']

            x= df.iloc[:, 2:].copy()
            if '触媒ロット' in x.columns:
                x = x.drop('触媒ロット', axis= 1)

            # RandomForestRegressorでBorutaを実行
            rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, perc= 90)
            feat_selector.fit(x.values, y.values)

            # 選択された特徴量を確認
            selected = feat_selector.support_
            print('選択された特徴量の数: %d' % np.sum(selected))
            print(x.columns[selected])

            #上で選択した説明変数のみを残す。
            boruta_descriptors = x.columns[11:]
            selected_features= boruta_descriptors[selected[11:]]
            df_a = df.iloc[:, 0:14]
            df_b= df.loc[:, selected_features]
            df_c = pd.concat([df_a, df_b], axis= 1)
            
            return df_c
                    
        x2_4 = pd.read_csv('datasets/x2_4.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
        x2_desc = x2_4.drop(['選択率NPA','選択率エタノール','触媒ロット'], axis= 1)
        x2_desc_a = addSquareCross(x2_desc)
        x2_cross = pd.concat([x2_4.loc[:, ['選択率NPA','選択率エタノール']], x2_desc_a], axis= 1)
        
        x2_cross = Boruta_Apply(x2_cross)
        x2_cross.insert(4, '触媒ロット', x2_4['触媒ロット'])
        
        return x2_cross
    
    def getDesc_X2_calc_mtls(self, ope_type, x_base, metal_x):
        # dfの行名を分解して列名の方に加える
        def rename_index_col(df):
            # 行名（インデックス）を分割して新しいデータフレームを作成
            df.index = df.index.str.split('_', n=1, expand=True)
            # 列名を新しい形式に変換
            new_columns = []
            for col in df.columns:
                for idx in df.index.levels[1]:
                    new_columns.append(f'{col}_{idx}')
            # 新しいデータフレームを作成
            df_new = pd.DataFrame(index=df.index.levels[0], columns=new_columns)
            # データを新しいデータフレームに移動
            for element, group in df.groupby(level=0):
                if element == 'Rh':
                    # 元素がRhの場合の特別処理
                    for col in df.columns:
                        for idx in group.index.get_level_values(1):
                            new_col_name = f'{col}_{idx}'
                            df_new.at[element, new_col_name] = group.loc[(element, idx), col]
                else:
                    # 通常の処理
                    for col in df.columns:
                        for idx in df.index.levels[1]:
                            new_col_name = f'{col}_{idx}'
                            df_new.at[element, new_col_name] = group.loc[element, idx][col]
            return df_new        
        E_CO = pd.read_csv('matlantis_descriptors/#2/20240830_E_ads_CO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        E_ETA = pd.read_csv('matlantis_descriptors/#2/20240830_E_ads_ETA_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        E_HCO = pd.read_csv('matlantis_descriptors/#2/20240830_E_ads_HCO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        E_EtCO = pd.read_csv('matlantis_descriptors/#2/20240830_E_ads_EtCO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        NEB_HCO = pd.read_csv('matlantis_descriptors/#2/20240830_NEB_HCO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)
        NEB_CO = pd.read_csv('matlantis_descriptors/#2/20240830_NEB_EtCO_pivot.csv', encoding= 'cp932', index_col= 0, header= 0)

        # Matlantisにて計算した対象金属
        matlantis_metals = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ir', 'Pt', 'Au', 'V', 'Mo', 'W']
        process_list = ['前処理還元炉温℃','評価反応炉温℃','触媒ロット','temp','flow_NaOH','flow_slurry','flow_red	wash','support_Al2O3_A-11','support_CeO2_HS','support_TiO2_SSP-M','support_ZrO2_RC100','SiO2_CARIACTg-10']

        # 行を採用1、2に合わせる
        E_CO = E_CO.drop('Doped_metal', axis= 1)
        E_ETA = E_ETA.drop('Doped_metal', axis= 1)
        E_HCO = E_HCO.drop('Doped_metal', axis= 1)
        E_EtCO = E_EtCO.drop('Doped_metal', axis= 1)
        NEB_CO = NEB_CO.drop('Doped_metal', axis= 1)
        NEB_HCO = NEB_HCO.drop('Doped_metal', axis= 1)

        # 各ファイル内容の展開
        E_ads_CO = rename_index_col(E_CO)
        E_ads_ETA = rename_index_col(E_ETA)
        E_ads_HCO = rename_index_col(E_HCO)
        E_ads_EtCO = rename_index_col(E_EtCO)
        NEB_CO_ = rename_index_col(NEB_CO)
        NEB_HCO_ = rename_index_col(NEB_HCO)
        mtlnts_desc = pd.concat([E_ads_CO, E_ads_ETA, E_ads_HCO, E_ads_EtCO, NEB_CO_, NEB_HCO_], axis= 1)

        # metal_xをx2へ変換
        metal_x['row_id'] = metal_x.index
        # 'metal1-5'と'ratio1-5'をそれぞれ縦長の形式に変換
        metals_long = metal_x.melt(id_vars='row_id', value_vars=['metal1', 'metal2', 'metal3', 'metal4', 'metal5'],
                                var_name='metal_num', value_name='metal')
        ratios_long = metal_x.melt(id_vars='row_id', value_vars=['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5'],
                                var_name='ratio_num', value_name='ratio')
        # 'metal_num'と'ratio_num'から番号を抽出し、結合のためのキーとして使用
        metals_long['num'] = metals_long['metal_num'].str.extract('(\d+)').astype(int)
        ratios_long['num'] = ratios_long['ratio_num'].str.extract('(\d+)').astype(int)
        # 'row_id'と'num'でデータを結合
        long_df = pd.merge(metals_long[['row_id', 'num', 'metal']],
                        ratios_long[['row_id', 'num', 'ratio']],
                        on=['row_id', 'num'])
        # ピボットテーブルを作成し、元素名を列として使用
        x2 = long_df.pivot(index='row_id', columns='metal', values='ratio')
        x2 = x2.replace(np.nan, 0)
        
        for index, row in x2.iterrows():
            el_ratio_list = [[el, row[el]] for el in x2.loc[:, : 'Zn'].columns if row[el] != 0]
            el_ratio_list = [el for el in el_ratio_list if el[0] in matlantis_metals]
            el_ratio_dic = {el[0]: el[1] for el in el_ratio_list}
            if len(el_ratio_dic) >= 1:
                for prop in mtlnts_desc.columns:
                    prop_value = mtlnts_desc.loc[el_ratio_dic.keys(), prop].values
                    weight_values = np.array(list(el_ratio_dic.values()))
                    weighted_average = np.average(prop_value,weights=weight_values)
                    x2.at[index, f'{prop}_weighted'] = weighted_average
            elif row['Rh'] != 0:
                for prop in mtlnts_desc.columns:
                    x2.at[index, f'{prop}_weighted'] = mtlnts_desc.loc['Rh', prop]
            else:
                print('Error! '+index+'にはRh, ドープ元素ともありません')
                break

        common_columns = [c for c in x_base.columns if c in x2.columns]
        x_base[common_columns] = x2[common_columns]
        x_base = x_base.dropna(how= 'all', axis= 1)
        
        return x_base
    
    def getDesc_X2_calc_dft(self, ope_type, x_base, metal_x):

        dft_data = pd.read_csv('matlantis_descriptors/DFT_DATA_single_impurity_onRh(Akashi20240910) .csv', encoding= 'utf-8-sig', index_col= 1, header= 0)
        # Matlantisにて計算した対象金属
        dft_metals = dft_data.index.tolist()
        dft_metals = [s for s in dft_metals if s not in ['Ag','Os']]  #Rhを含むことに注意
        
        dft_data = dft_data.dropna(how= 'all', axis= 0)
        dft_data = dft_data.drop('Unnamed: 0', axis= 1)
        dft_data = dft_data[dft_data.index.isin(dft_metals)]

        process_list = ['前処理還元炉温℃','評価反応炉温℃','触媒ロット','temp','flow_NaOH','flow_slurry','flow_red	wash','support_Al2O3_A-11','support_CeO2_HS','support_TiO2_SSP-M','support_ZrO2_RC100','SiO2_CARIACTg-10']

        # metal_xをx2へ変換
        metal_x['row_id'] = metal_x.index
        # 'metal1-5'と'ratio1-5'をそれぞれ縦長の形式に変換
        metals_long = metal_x.melt(id_vars='row_id', value_vars=['metal1', 'metal2', 'metal3', 'metal4', 'metal5'],
                                var_name='metal_num', value_name='metal')
        ratios_long = metal_x.melt(id_vars='row_id', value_vars=['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5'],
                                var_name='ratio_num', value_name='ratio')
        # 'metal_num'と'ratio_num'から番号を抽出し、結合のためのキーとして使用
        metals_long['num'] = metals_long['metal_num'].str.extract('(\d+)').astype(int)
        ratios_long['num'] = ratios_long['ratio_num'].str.extract('(\d+)').astype(int)
        # 'row_id'と'num'でデータを結合
        long_df = pd.merge(metals_long[['row_id', 'num', 'metal']],
                        ratios_long[['row_id', 'num', 'ratio']],
                        on=['row_id', 'num'])
        # ピボットテーブルを作成し、元素名を列として使用
        x2 = long_df.pivot(index='row_id', columns='metal', values='ratio')
        x2 = x2.replace(np.nan, 0)
        
        for index, row in x2.iterrows():
            el_ratio_list = [[el, row[el]] for el in x2.loc[:, : 'V'].columns if row[el] != 0]
            el_ratio_list = [el for el in el_ratio_list if el[0] in dft_metals]
            el_ratio_dic = {el[0]: el[1] for el in el_ratio_list}
            if len(el_ratio_dic) >= 1:
                for prop in dft_data.columns:
                    prop_value = dft_data.loc[el_ratio_dic.keys(), prop].values
                    weight_values = np.array(list(el_ratio_dic.values()))
                    weighted_average = np.average(prop_value,weights=weight_values)
                    x2.at[index, f'{prop}_weighted'] = weighted_average
            elif row['Rh'] != 0:
                for prop in dft_data.columns:
                    x2.at[index, f'{prop}_weighted'] = dft_data.loc['Rh', prop]
            else:
                print('Error! '+index+'にはRh, ドープ元素ともありません')
                break

        common_columns_dft = [c for c in x_base.columns if c in x2.columns]
        x_base[common_columns_dft] = x2[common_columns_dft]
        x_base = x_base.dropna(how= 'all', axis= 1)      
        
        return x_base
    
    def getDesc_X27(self):
        rd = 'results\\'
        evaluation_lot_data = pd.read_excel(rd+'evaluation_lot_data.xlsx', index_col=0, header=0, engine='openpyxl')
        evaluation_y_x_data_with_dummy_variables = pd.read_excel(rd+'evaluation_y_x_data_with_dummy_variables.xlsx', index_col=0, header=0, engine='openpyxl')
        synthesis_condition_x_data_with_dummy_variables = pd.read_excel(rd+'synthesis_condition_x_data_with_dummy_variables.xlsx', index_col=0, header=0, engine='openpyxl')
        synthesis_metal_x_data = pd.read_excel(rd+'synthesis_metal_x_data.xlsx', index_col=0, header=0, engine='openpyxl')

        #関数の導入(自乗項、交差項とBoruta)
        def addSquareCross(df: pd.DataFrame):
            columns = df.columns
            for i, c1 in enumerate(columns):
                df[c1 + '^2'] = df[c1] ** 2
                for j, c2 in enumerate(columns):
                    if i <= j:
                        continue
                    df[c1 +' * ' + c2] = df[c1] * df[c2]
            return df
        
        #Borutaで特徴量を削減する
        def Boruta_Apply(df, perc):
            y= df[['選択率NPA', '選択率エタノール']].copy()
            y['sum']= y['選択率NPA']+ y['選択率エタノール']
            y= y['sum']

            x= df.iloc[:, 2:].copy()
            if '触媒ロット' in x.columns:
                x = x.drop('触媒ロット', axis= 1)

            # RandomForestRegressorでBorutaを実行
            rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, perc=perc)
            feat_selector.fit(x.values, y.values)

            # 選択された特徴量を確認
            selected = feat_selector.support_
            print('選択された特徴量の数: %d' % np.sum(selected))
            print(x.columns[selected])

            #上で選択した説明変数のみを残す。
            boruta_descriptors = x.columns[11:]
            selected_features= boruta_descriptors[selected[11:]]
            df_a = df.iloc[:, 0:14]
            df_b= df.loc[:, selected_features]
            df_c = pd.concat([df_a, df_b], axis= 1)
            
            return df_c        
          
        #x27を作る前にx25を作成する。
        metals = synthesis_metal_x_data.loc[:, ['metal1', 'metal2', 'metal3', 'metal4', 'metal5', 'ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5']]
        x27 = evaluation_y_x_data_with_dummy_variables.copy()
        x27['触媒ロット'] = evaluation_lot_data.loc[evaluation_y_x_data_with_dummy_variables.index, '触媒ロット']
        x27['評価ロット数字'] = x27.index
        
        #目的変数のない行は削除する
        x27.dropna(subset=['選択率NPA※※'], axis= 0, inplace= True)
        x27 = pd.merge(x27, synthesis_condition_x_data_with_dummy_variables, right_index= True, left_on= '触媒ロット')
        x27.drop_duplicates(subset= '評価ロット数字', inplace= True)
        
        x27 = x27.sort_index()
        x27.columns = [i.replace('※', '') if '※' in i else i for i in x27.columns]
        x27.drop('評価ロット数字', axis= 1, inplace= True)
        
        #プロパノール、エタノールの選択率が共に0のデータは削除する。
        x27['yield'] = x27['選択率NPA'] + x27['選択率エタノール']
        x27_a = x27[x27['yield']!= 0]
        x27_a= x27_a.drop('yield', axis= 1)
                
        #元素をグルーピングする(x25)
        group_a = ['Ca','Sr','Ba','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Fe','Ru','Os']  #15
        group_b = ['Ni','Co','Rh','Pd','Pt','Ir','Mn','Cu','Tc','Re']  #10
        group_c = ['Al','Au']  #2
        group_d = ['Li','Na','K','Rb','Cs']  #5
        group_e = ['Mg','Ag','Zn','Cd','In','Si','Ge','Sn','Pb','As','Sb','Bi','Se','Te','Ga']  #15
        group_f = ['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']  #17
        
        #触媒リストから、同じ元素が2つ以上入っている行を削除する(sc000372)
        metals.assign(check = 0)
        for index, row in metals.iterrows():
            row = row[0:5].dropna()
            metals.at[index, 'check'] = row.duplicated().sum()
        metals = metals[metals['check'] == 0]
        metals.drop('check', axis= 1, inplace= True)        
        
        metals = metals.replace(np.nan, 0)
        metals = metals.replace(' ', '')
        #metals = metals.drop(index= ['sc000368','sc000369'], axis= 0)
        metals= metals.assign(groupa= 0, groupb= 0, groupc= 0, groupd= 0, groupe= 0, groupf= 0)
        metals= metals.assign(compoa= 0, compob= 0, compoc= 0, compod= 0, compoe= 0, compof= 0)
                
        for row in metals.itertuples():
            for i in range(1, 7):
                if row[i] in group_a:
                    metals.loc[row[0], 'groupa']+= 1
                    loc_a= pd.Index(row).get_loc(row[i])+ 4 
                    compo_a= metals.columns[loc_a]
                    metals.at[row[0], 'compoa']+= metals.at[row[0], compo_a]
                if row[i] in group_b:
                    metals.loc[row[0], 'groupb']+= 1
                    loc_b= pd.Index(row).get_loc(row[i])+ 4 
                    compo_b= metals.columns[loc_b]
                    #metals.to_csv('../datasets/metals.csv', encoding= 'utf-8-sig')
                    metals.at[row[0], 'compob']+= metals.at[row[0], compo_b]
                if row[i] in group_c:
                    metals.loc[row[0], 'groupc']+= 1
                    loc_c= pd.Index(row).get_loc(row[i])+ 4 
                    compo_c= metals.columns[loc_c]
                    metals.at[row[0], 'compoc']+= metals.at[row[0], compo_c]
                if row[i] in group_d:
                    metals.loc[row[0], 'groupd']+= 1
                    loc_d= pd.Index(row).get_loc(row[i])+ 4 
                    compo_d= metals.columns[loc_d]
                    metals.at[row[0], 'compod']+= metals.at[row[0], compo_d]
                if row[i] in group_e:
                    metals.loc[row[0], 'groupe']+= 1
                    loc_e= pd.Index(row).get_loc(row[i])+ 4 
                    compo_e= metals.columns[loc_e]
                    metals.at[row[0], 'compoe']+= metals.at[row[0], compo_e]
                if row[i] in group_f:
                    metals.loc[row[0], 'groupf']+= 1
                    loc_f= pd.Index(row).get_loc(row[i])+ 4
                    compo_f= metals.columns[loc_f]
                    metals.at[row[0], 'compof']+= metals.at[row[0], compo_f]
                    
        x27 = pd.merge(x27_a, metals, right_index= True, left_on= '触媒ロット')
        x27 = x27[~x27.duplicated()]
        x27.sort_index(inplace= True)
        
        #x27とするために、1)Fe＊Ir, 2)epsilon-former, 3)epsilon安定化パラメータの観点から説明変数を追加する。
        x27 = x27.assign(FexIr = 0)
        metal_list = ['metal1', 'metal2', 'metal3', 'metal4', 'metal5']
        for index, row in x27.iterrows():
            if 'Fe' in row[metal_list].values and 'Ir' in row[metal_list].values:
                Fe = row[row == 'Fe'].index.to_list()
                Fe_loc = row.index.get_loc(Fe[0])
                Fe_ratio = row.iat[Fe_loc+5]                
                Ir = row[row == 'Ir'].index.to_list()
                Ir_loc = row.index.get_loc(Ir[0])
                Ir_ratio = row.iat[Ir_loc+5]   
                x27.loc[index, 'FexIr'] = Fe_ratio * Ir_ratio
         
        epsilon_stability = {'Fe': 1000, 'Ir': -4500, 'Ru':-6100, 'Mn': -7000, 'Cr': -3500}
        epsilon_former = pd.Series(data= epsilon_stability, name= 'epsilon_stability')
        
        #Feがmetal1-5にあれば、ε-Feを安定化する元素数、組成、ε安定化エネルギーを計算する。
        x27 = x27.assign(epsilon_former= 0, epsilon_compo= 0, epsilon_stability= 0)
        for index, row in x27.iterrows():
            if 'Fe' in row[metal_list].values:
                if row['metal1'] in epsilon_former.index.to_list():
                    x27.loc[index, 'epsilon_former'] += 1
                    x27.loc[index, 'epsilon_compo'] += x27.loc[index, 'ratio1']
                    epsilon_stab_value = epsilon_former[row['metal1']]
                    x27.loc[index, 'epsilon_stability'] += x27.loc[index, 'ratio1'] * epsilon_stab_value
                if row['metal2'] in epsilon_former.index.to_list():
                    x27.loc[index, 'epsilon_former'] += 1
                    x27.loc[index, 'epsilon_compo'] += x27.loc[index, 'ratio2']
                    epsilon_stab_value = epsilon_former[row['metal2']]
                    x27.loc[index, 'epsilon_stability'] += x27.loc[index, 'ratio2'] * epsilon_stab_value
                if row['metal3'] in epsilon_former.index.to_list():
                    x27.loc[index, 'epsilon_former'] += 1
                    x27.loc[index, 'epsilon_compo'] += x27.loc[index, 'ratio3']
                    epsilon_stab_value = epsilon_former[row['metal3']]
                    x27.loc[index, 'epsilon_stability'] += x27.loc[index, 'ratio3'] * epsilon_stab_value
                if row['metal4'] in epsilon_former.index.to_list():
                    x27.loc[index, 'epsilon_former'] += 1
                    x27.loc[index, 'epsilon_compo'] += x27.loc[index, 'ratio4']
                    epsilon_stab_value = epsilon_former[row['metal4']]
                    x27.loc[index, 'epsilon_stability'] += x27.loc[index, 'ratio4'] * epsilon_stab_value
                if row['metal5'] in epsilon_former.index.to_list():
                    x27.loc[index, 'epsilon_former'] += 1
                    x27.loc[index, 'epsilon_compo'] += x27.loc[index, 'ratio5']
                    epsilon_stab_value = epsilon_former[row['metal5']]
                    x27.loc[index, 'epsilon_stability'] += x27.loc[index, 'ratio5'] * epsilon_stab_value                

        drop_list = metal_list+['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5']
        x27.drop(drop_list, axis= 1, inplace= True)
        
        x27_b = x27.copy()
        x27_desc = x27_b.drop(['選択率NPA','選択率エタノール','触媒ロット'], axis= 1)
        x27_desc_a = addSquareCross(x27_desc)
        x27_1 = pd.concat([x27.iloc[:, :5], x27_desc_a], axis= 1)
        x27_1 = x27_1.loc[:, ~x27_1.columns.duplicated()]        
        x27_1 = Boruta_Apply(x27_1, perc= 80)        

        return x27, x27_1
        
    def getDesc_X2_17(self, ope_type, x_base, metal_x):

        synthesis_metal_x_data = metal_x
        met1 = synthesis_metal_x_data['metal1'].unique()
        met2 = synthesis_metal_x_data['metal2'].unique()
        met3 = synthesis_metal_x_data['metal3'].unique()
        met4 = synthesis_metal_x_data['metal4'].unique()
        met5 = synthesis_metal_x_data['metal5'].unique()
        metal_list = np.concatenate([met1, met2, met3, met4, met5])
        metal_list = pd.Series(metal_list)
        metal_list = metal_list.dropna()
        metal_list = metal_list.unique()
        metal_list = np.sort(metal_list)
        synthesis_metal_x_data_modified = pd.DataFrame(index= synthesis_metal_x_data.index, columns= metal_list) 
        #elements = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt', 'Au', 'In', 'Sn']
        grouping = ['groupa','groupb','groupc','groupd','groupe','groupf','compoa','compob','compoc','compod','compoe','compof']

        #触媒組成のデータの元素、組成を上のdf形式へ変換
        synthesis_metal_x_data_modified['dummy'] = np.nan
        for index, row in synthesis_metal_x_data.iterrows():
            content1 = row[0], row[5]
            content2 = row[1], row[6]
            content3 = row[2], row[7]
            content4 = row[3], row[8]
            content5 = row[4], row[9]
            if pd.isna(content2[0]):
                content2 = list(content2)
                content2[0] = 'dummy'            
            if pd.isna(content3[0]):
                content3 = list(content3)
                content3[0] = 'dummy'
            if pd.isna(content4[0]):
                content4 = list(content4)
                content4[0] = 'dummy'
            if pd.isna(content5[0]):
                content5 = list(content5)
                content5[0] = 'dummy'
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content1[0])] = content1[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content2[0])] = content2[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content3[0])] = content3[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content4[0])] = content4[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content5[0])] = content5[1]
        synthesis_metal_x_data_modified.drop('dummy', axis= 1, inplace= True)
        synthesis_metal_x_data_modified.replace(np.nan, 0, inplace= True)
        elements = synthesis_metal_x_data_modified.columns.to_list()
        
        if ope_type == 'Type1':
            x2 = pd.merge(x_base, synthesis_metal_x_data_modified, right_index= True, left_on= '触媒ロット')
            x2['評価ロット数字'] = x2.index
            x2.drop_duplicates(subset= '評価ロット数字', inplace= True)
            x2.replace(np.nan, 0, inplace= True)
            x2.drop('評価ロット数字', axis= 1, inplace= True)
            
            x2 = x2[x2.iloc[:,14:].sum(axis= 1)> 0]  #分析値が0のデータを削除
            x2.sort_index(inplace= True)
            x2.columns = [i.replace('※', '') if '※' in i else i for i in x2.columns]            
            
        elif ope_type == 'Type2':
            x2 = x_base
            # elements　に無い列は追加(値は0)
            synthesis_metal_x_data_modified = checkElements(synthesis_metal_x_data_modified, elements)
            x2.loc[:, elements] = synthesis_metal_x_data_modified.loc[:, elements]

        #x2_7に情報を追加する。まずはGrouping
        metals = metal_x
        metals = metals.replace(np.nan, 0)
        metals = metals.replace(' ', '')
        metals= metals.assign(groupa= 0, groupb= 0, groupc= 0, groupd= 0, groupe= 0, groupf= 0)
        metals= metals.assign(compoa= 0, compob= 0, compoc= 0, compod= 0, compoe= 0, compof= 0)
        
        #元素Grouping
        group_a = ['Ca','Sr','Ba','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Fe','Ru','Os']  #15
        group_b = ['Ni','Co','Rh','Pd','Pt','Ir','Mn','Cu','Tc','Re']  #10
        group_c = ['Al','Au']  #2
        group_d = ['Li','Na','K','Rb','Cs']  #5
        group_e = ['Mg','Ag','Zn','Cd','In','Si','Ge','Sn','Pb','As','Sb','Bi','Se','Te','Ga']  #15
        group_f = ['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']  #17        
                
        for row in metals.itertuples():
            for i in range(1, 7):
                if row[i] in group_a:
                    metals.loc[row[0], 'groupa']+= 1
                    loc_a= pd.Index(row).get_loc(row[i])+ 4 
                    compo_a= metals.columns[loc_a]
                    metals.at[row[0], 'compoa']+= metals.at[row[0], compo_a]
                if row[i] in group_b:
                    metals.loc[row[0], 'groupb']+= 1
                    loc_b= pd.Index(row).get_loc(row[i])+ 4 
                    compo_b= metals.columns[loc_b]
                    #metals.to_csv('../datasets/metals.csv', encoding= 'utf-8-sig')
                    metals.at[row[0], 'compob']+= metals.at[row[0], compo_b]
                if row[i] in group_c:
                    metals.loc[row[0], 'groupc']+= 1
                    loc_c= pd.Index(row).get_loc(row[i])+ 4 
                    compo_c= metals.columns[loc_c]
                    metals.at[row[0], 'compoc']+= metals.at[row[0], compo_c]
                if row[i] in group_d:
                    metals.loc[row[0], 'groupd']+= 1
                    loc_d= pd.Index(row).get_loc(row[i])+ 4 
                    compo_d= metals.columns[loc_d]
                    metals.at[row[0], 'compod']+= metals.at[row[0], compo_d]
                if row[i] in group_e:
                    metals.loc[row[0], 'groupe']+= 1
                    loc_e= pd.Index(row).get_loc(row[i])+ 4 
                    compo_e= metals.columns[loc_e]
                    metals.at[row[0], 'compoe']+= metals.at[row[0], compo_e]
                if row[i] in group_f:
                    metals.loc[row[0], 'groupf']+= 1
                    loc_f= pd.Index(row).get_loc(row[i])+ 4
                    compo_f= metals.columns[loc_f]
                    metals.at[row[0], 'compof']+= metals.at[row[0], compo_f]
                    
        if ope_type == 'Type1':
            x2_17 = pd.merge(x2, metals, left_on= '触媒ロット', right_index= True)
            x2_17 = x2_17[~x2_17.index.duplicated()]
            x2_17 = x2_17.drop(['metal1','metal2','metal3','metal4','metal5','ratio1','ratio2','ratio3','ratio4','ratio5'], axis= 1)               
        elif ope_type == 'Type2':
            x2_17 = x2
            x2_17.loc[:, grouping] = metals.loc[:, grouping]
      
        return x2_17
    
    def getDesc_X2_18(self, ope_type, x_base, metal_x):
        synthesis_metal_x_data = metal_x
        met1 = synthesis_metal_x_data['metal1'].unique()
        met2 = synthesis_metal_x_data['metal2'].unique()
        met3 = synthesis_metal_x_data['metal3'].unique()
        met4 = synthesis_metal_x_data['metal4'].unique()
        met5 = synthesis_metal_x_data['metal5'].unique()
        metal_list = np.concatenate([met1, met2, met3, met4, met5])
        metal_list = pd.Series(metal_list)
        metal_list = metal_list.dropna()
        metal_list = metal_list.unique()
        metal_list = np.sort(metal_list)
        synthesis_metal_x_data_modified = pd.DataFrame(index= synthesis_metal_x_data.index, columns= metal_list) 

        #触媒組成のデータの元素、組成を上のdf形式へ変換
        synthesis_metal_x_data_modified['dummy'] = np.nan
        for index, row in synthesis_metal_x_data.iterrows():
            content1 = row[0], row[5]
            content2 = row[1], row[6]
            content3 = row[2], row[7]
            content4 = row[3], row[8]
            content5 = row[4], row[9]
            if pd.isna(content2[0]):
                content2 = list(content2)
                content2[0] = 'dummy'            
            if pd.isna(content3[0]):
                content3 = list(content3)
                content3[0] = 'dummy'
            if pd.isna(content4[0]):
                content4 = list(content4)
                content4[0] = 'dummy'
            if pd.isna(content5[0]):
                content5 = list(content5)
                content5[0] = 'dummy'
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content1[0])] = content1[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content2[0])] = content2[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content3[0])] = content3[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content4[0])] = content4[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content5[0])] = content5[1]
        synthesis_metal_x_data_modified.drop('dummy', axis= 1, inplace= True)
        synthesis_metal_x_data_modified.replace(np.nan, 0, inplace= True)
        elements = synthesis_metal_x_data_modified.columns.to_list()
        
        if ope_type == 'Type1':
            x2 = pd.merge(x_base, synthesis_metal_x_data_modified, right_index= True, left_on= '触媒ロット')
            x2['評価ロット数字'] = x2.index
            x2.drop_duplicates(subset= '評価ロット数字', inplace= True)
            x2.replace(np.nan, 0, inplace= True)
            x2.drop('評価ロット数字', axis= 1, inplace= True)
            
            x2 = x2[x2.iloc[:,14:].sum(axis= 1)> 0]  #分析値が0のデータを削除
            x2.sort_index(inplace= True)
            x2.columns = [i.replace('※', '') if '※' in i else i for i in x2.columns]            
            
        elif ope_type == 'Type2':
            x2 = x_base
            # elements　に無い列は追加(値は0)
            synthesis_metal_x_data_modified = checkElements(synthesis_metal_x_data_modified, elements)
            x2.loc[:, elements] = synthesis_metal_x_data_modified.loc[:, elements]

        #自乗項、交差項を埋める
        square_list = [s for s in x2.columns.to_list() if '^2' in s]
        decomposed_square = [s.rstrip('^2') for s in square_list ]
        x2[square_list] = x2[decomposed_square] **2
        cross_list = [s for s in x2.columns.to_list() if '*' in s]
        decomposed_cross = [s.split(' * ') for s in cross_list]
        for item, dec_item in zip(cross_list, decomposed_cross):
            x2[item] = x2[dec_item[0]] * x2[dec_item[1]]
                    
        x2_18 = x2
        return x2_18  
    
    def getDesc_X2_20(self, ope_type, x_base, metal_x):
        #matminerの計算をするための準備 Miedemaは計算不要、残りの4つは計算必要
        #この関数はGA専用かつType2専用　従ってTeは含まない前提とする
        matminer_list = ['APE mean','Configuration entropy','Mixing enthalpy','Shear modulus mean','Shear modulus delta','mean AtomicRadius','mean Electronegativity','HOMO_energy','LUMO_energy','band center']
        
        synthesis_metal_x_data = metal_x
        met1 = synthesis_metal_x_data['metal1'].unique()
        met2 = synthesis_metal_x_data['metal2'].unique()
        met3 = synthesis_metal_x_data['metal3'].unique()
        met4 = synthesis_metal_x_data['metal4'].unique()
        met5 = synthesis_metal_x_data['metal5'].unique()
        metal_list = np.concatenate([met1, met2, met3, met4, met5])
        metal_list = pd.Series(metal_list)
        metal_list = metal_list.dropna()
        metal_list = metal_list.unique()
        metal_list = np.sort(metal_list)
        synthesis_metal_x_data_modified = pd.DataFrame(index= synthesis_metal_x_data.index, columns= metal_list) 

        #触媒組成のデータの元素、組成を上のdf形式へ変換
        synthesis_metal_x_data_modified['dummy'] = np.nan
        for index, row in synthesis_metal_x_data.iterrows():
            content1 = row[0], row[5]
            content2 = row[1], row[6]
            content3 = row[2], row[7]
            content4 = row[3], row[8]
            content5 = row[4], row[9]
            if pd.isna(content2[0]):
                content2 = list(content2)
                content2[0] = 'dummy'            
            if pd.isna(content3[0]):
                content3 = list(content3)
                content3[0] = 'dummy'
            if pd.isna(content4[0]):
                content4 = list(content4)
                content4[0] = 'dummy'
            if pd.isna(content5[0]):
                content5 = list(content5)
                content5[0] = 'dummy'
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content1[0])] = content1[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content2[0])] = content2[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content3[0])] = content3[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content4[0])] = content4[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content5[0])] = content5[1]
        synthesis_metal_x_data_modified.drop('dummy', axis= 1, inplace= True)
        synthesis_metal_x_data_modified.replace(np.nan, 0, inplace= True)
        elements = synthesis_metal_x_data_modified.columns.to_list()
        
        x2_20 = x_base
        #x2 = x_base
        # elements　に無い列は追加(値は0)
        synthesis_metal_x_data_modified = checkElements(synthesis_metal_x_data_modified, elements)
        x2_20.loc[:, elements] = synthesis_metal_x_data_modified.loc[:, elements]
        #x2.loc[:, elements] = synthesis_metal_x_data_modified.loc[:, elements]

        #matminerの記述子を計算する
        matminer_df = CalcMatminer(metal_x)
        x2_20_mat = matminer_df[matminer_list]
        #x2_20 = matminer_df[matminer_list]
        x2_20.loc[:,matminer_list] = x2_20_mat.loc[:,matminer_list]
        #x2_20 = pd.concat([x2, x2_20], axis= 1)
        
        return x2_20    
    
    # Type1には対応していないです。
    def getDesc_X2_22(self, ope_type, x_base, metal_x):

        # Type1には対応していないです。
        
        synthesis_metal_x_data = metal_x
        met1 = synthesis_metal_x_data['metal1'].unique()
        met2 = synthesis_metal_x_data['metal2'].unique()
        met3 = synthesis_metal_x_data['metal3'].unique()
        met4 = synthesis_metal_x_data['metal4'].unique()
        met5 = synthesis_metal_x_data['metal5'].unique()
        metal_list = np.concatenate([met1, met2, met3, met4, met5])
        metal_list = pd.Series(metal_list)
        metal_list = metal_list.dropna()
        metal_list = metal_list.unique()
        metal_list = np.sort(metal_list)
        synthesis_metal_x_data_modified = pd.DataFrame(index= synthesis_metal_x_data.index, columns= metal_list) 
        #elements = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt', 'Au', 'In', 'Sn']
        grouping = ['group_ep','group_la','compo_ep','compo_la']
        #grouping = ['groupa','groupb','groupc','groupd','groupe','groupf','compoa','compob','compoc','compod','compoe','compof']

        #触媒組成のデータの元素、組成を上のdf形式へ変換
        synthesis_metal_x_data_modified['dummy'] = np.nan
        for index, row in synthesis_metal_x_data.iterrows():
            content1 = row[0], row[5]
            content2 = row[1], row[6]
            content3 = row[2], row[7]
            content4 = row[3], row[8]
            content5 = row[4], row[9]
            if pd.isna(content2[0]):
                content2 = list(content2)
                content2[0] = 'dummy'            
            if pd.isna(content3[0]):
                content3 = list(content3)
                content3[0] = 'dummy'
            if pd.isna(content4[0]):
                content4 = list(content4)
                content4[0] = 'dummy'
            if pd.isna(content5[0]):
                content5 = list(content5)
                content5[0] = 'dummy'
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content1[0])] = content1[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content2[0])] = content2[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content3[0])] = content3[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content4[0])] = content4[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content5[0])] = content5[1]
        synthesis_metal_x_data_modified.drop('dummy', axis= 1, inplace= True)
        synthesis_metal_x_data_modified.replace(np.nan, 0, inplace= True)
        elements = synthesis_metal_x_data_modified.columns.to_list()
        
        if ope_type == 'Type1':
            
            """
            x2 = pd.merge(x_base, synthesis_metal_x_data_modified, right_index= True, left_on= '触媒ロット')
            x2['評価ロット数字'] = x2.index
            x2.drop_duplicates(subset= '評価ロット数字', inplace= True)
            x2.replace(np.nan, 0, inplace= True)
            x2.drop('評価ロット数字', axis= 1, inplace= True)
            
            x2 = x2[x2.iloc[:,14:].sum(axis= 1)> 0]  #分析値が0のデータを削除
            x2.sort_index(inplace= True)
            x2.columns = [i.replace('※', '') if '※' in i else i for i in x2.columns]
            """            
            
        elif ope_type == 'Type2':
            x2 = x_base
            # elements　に無い列は追加(値は0)
            synthesis_metal_x_data_modified = checkElements(synthesis_metal_x_data_modified, elements)
            x2.loc[:, elements] = synthesis_metal_x_data_modified.loc[:, elements]

        #x2_7に情報を追加する。まずはGrouping
        metals = metal_x
        metals = metals.replace(np.nan, 0)
        metals = metals.replace(' ', '')
        metals= metals.assign(group_ep= 0, group_la= 0)
        metals= metals.assign(compo_ep= 0, compo_la= 0)
        metals = metals[~metals.index.duplicated(keep= 'first')]  # 追加 2023/8/24
        #metals= metals.assign(groupa= 0, groupb= 0, groupc= 0, groupd= 0, groupe= 0, groupf= 0)
        #metals= metals.assign(compoa= 0, compob= 0, compoc= 0, compod= 0, compoe= 0, compof= 0)
        
        #元素Grouping
        #2元系の組合せに着目したGrouping
        group_ep = [['Ru','Rh','Ir'], ['Fe','Co','Cr','Mn']]
        group_la = [['Y','K','La','Ce'], ['Rh','Cu','Ni','Re','Pt']]
        """
        group_a = ['Ca','Sr','Ba','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Fe','Ru','Os']  #15
        group_b = ['Ni','Co','Rh','Pd','Pt','Ir','Mn','Cu','Tc','Re']  #10
        group_c = ['Al','Au']  #2
        group_d = ['Li','Na','K','Rb','Cs']  #5
        group_e = ['Mg','Ag','Zn','Cd','In','Si','Ge','Sn','Pb','As','Sb','Bi','Se','Te','Ga']  #15
        group_f = ['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']  #17   
        """
           
        for index, row in metals.iterrows():
            if row[5] == 1:
                metals.loc[index, 'group_ep':'compo_la'] = 0
            else:
                for i in range(0, 5):
                    if row[i] in group_ep[0]:
                        if row[i+1] in group_ep[1]:
                            metals.at[index, 'group_ep'] = 1
                            metals.loc[index, 'compo_ep'] = row[i+5]+row[i+6]
                        elif row[i+2] in group_ep[1]:
                            metals.at[index, 'group_ep'] = 1
                            metals.loc[index, 'compo_ep'] = row[i+5]+row[i+7]                    
                        elif row[i+3] in group_ep[1]:
                            metals.at[index, 'group_ep'] = 1
                            metals.loc[index, 'compo_ep'] = row[i+5]+row[i+8] 
                        elif row[i+4] in group_ep[1]:
                            metals.at[index, 'group_ep'] = 1
                            metals.loc[index, 'compo_ep'] = row[i+5]+row[i+9]                     
                    elif row[i] in group_ep[1]:
                        if row[i+1] in group_ep[0]:
                            metals.at[index, 'group_ep'] = 1
                            metals.loc[index, 'compo_ep'] = row[i+5]+row[i+6]
                        elif row[i+2] in group_ep[0]:
                            metals.at[index, 'group_ep'] = 1
                            metals.loc[index, 'compo_ep'] = row[i+5]+row[i+7]                    
                        elif row[i+3] in group_ep[0]:
                            metals.at[index, 'group_ep'] = 1
                            metals.loc[index, 'compo_ep'] = row[i+5]+row[i+8] 
                        elif row[i+4] in group_ep[0]:
                            metals.at[index, 'group_ep'] = 1
                            metals.loc[index, 'compo_ep'] = row[i+5]+row[i+9]                        
                    elif row[i] in group_la[0]:
                        if row[i+1] in group_la[1]:
                            metals.at[index, 'group_la'] = 1
                            metals.loc[index, 'compo_la'] = row[i+5]+row[i+6]
                        elif row[i+2] in group_la[1]:
                            metals.at[index, 'group_la'] = 1
                            metals.loc[index, 'compo_la'] = row[i+5]+row[i+7]                    
                        elif row[i+3] in group_la[1]:
                            metals.at[index, 'group_la'] = 1
                            metals.loc[index, 'compo_la'] = row[i+5]+row[i+8] 
                        elif row[i+4] in group_la[1]:
                            metals.at[index, 'group_la'] = 1
                            metals.loc[index, 'compo_la'] = row[i+5]+row[i+9]                     
                    elif row[i] in group_la[1]:
                        if row[i+1] in group_la[0]:
                            metals.at[index, 'group_la'] = 1
                            metals.loc[index, 'compo_la'] = row[i+5]+row[i+6]
                        elif row[i+2] in group_la[0]:
                            metals.at[index, 'group_la'] = 1
                            metals.loc[index, 'compo_la'] = row[i+5]+row[i+7]                    
                        elif row[i+3] in group_la[0]:
                            metals.at[index, 'group_la'] = 1
                            metals.loc[index, 'compo_la'] = row[i+5]+row[i+8] 
                        elif row[i+4] in group_la[0]:
                            metals.at[index, 'group_la'] = 1
                            metals.loc[index, 'compo_la'] = row[i+5]+row[i+9]    
                    
        if ope_type == 'Type1':
            """
            x2_17 = pd.merge(x2, metals, left_on= '触媒ロット', right_index= True)
            x2_17 = x2_17[~x2_17.index.duplicated()]
            x2_17 = x2_17.drop(['metal1','metal2','metal3','metal4','metal5','ratio1','ratio2','ratio3','ratio4','ratio5'], axis= 1)    
            """
        elif ope_type == 'Type2':
            x2_22 = x2
            x2_22.loc[:, grouping] = metals.loc[:, grouping]
      
        return x2_22    
    
    def CalcX1_av(self, ope_type, x_base, metal_x):
        #x1の作成
        xenonpy_merge = pd.read_csv('results/xenonpy_element_data240515.csv', index_col=0)
        #xenonpy_merge.drop('oxide', axis= 1, inplace= True)    
        
        weighted_average_name = list() # 加重平均の index 名
        for j in xenonpy_merge.columns:
            weighted_average_name.append(f'ave_{j}')

        x1_metaldesc = pd.DataFrame(
            index=metal_x.index,
            columns=weighted_average_name
            )
        for i in range(metal_x.shape[0]):
            metal1 = metal_x.loc[metal_x.index[i], 'metal1']
            metal2 = metal_x.loc[metal_x.index[i], 'metal2']
            metal3 = metal_x.loc[metal_x.index[i], 'metal3']
            metal4 = metal_x.loc[metal_x.index[i], 'metal4']
            metal5 = metal_x.loc[metal_x.index[i], 'metal5']
            metal_rate1 = metal_x.loc[metal_x.index[i], 'ratio1']
            metal_rate2 = metal_x.loc[metal_x.index[i], 'ratio2']
            metal_rate3 = metal_x.loc[metal_x.index[i], 'ratio3']
            metal_rate4 = metal_x.loc[metal_x.index[i], 'ratio4']
            metal_rate5 = metal_x.loc[metal_x.index[i], 'ratio5']
                
            if pd.isna(metal5) is False:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                metal_desc5 = xenonpy_merge.loc[metal5, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])
            elif pd.isna(metal4) is False:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])    
            elif pd.isna(metal3) is False:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3])
            elif pd.isna(metal2) is False:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2])
                mr = np.array([metal_rate1, metal_rate2])                
            else:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                mt = np.array([metal_desc1])
                mr = np.array([metal_rate1])
            for desc in range(xenonpy_merge.shape[1]):
                d_name = xenonpy_merge.columns[desc]
                if np.isnan(mt[:, desc]).any():
                    x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.nan
                    continue
                #metal_x.to_csv('datasets/metal_x.csv', encoding= 'utf-8-sig')
                x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)

        x1_metaldesc = x1_metaldesc.replace([np.inf, -np.inf], np.nan)
        #ここでnanが生まれる可能性あり！！
        x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]   #全てnanの列を削除
        x_base = x_base.dropna(how='all', axis=1)
        x_base = pd.concat([x_base, x1_metaldesc], axis=1)


        return x_base
    
    
    
    def get_x2_forGA(self, individual_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition):
        featurizers2 = Calc_desc()
            
        inverse_metal_x_data = pd.DataFrame(index=individual_df.index, columns=[f'metal{i}' for i in range(1, 6)] +
                                                                               [f'ratio{i}' for i in range(1, 6)])
        range_val = range(1, 1 + len(metal_list[0]))
        inverse_metal_x_data.loc[:, [f'metal{i}' for i in range_val]] = metal_list
        inverse_metal_x_data.loc[:, [f'ratio{i}' for i in range_val]] = metal_component_list
        mi_util = MIutility() #データ作成クラス生成
        #inverse_metal_x_data = mi_util.checkMetalRatioTotalForRatio3(inverse_metal_x_data) 
        inverse_metal_x_data = mi_util.checkMetalRatioTotal(inverse_metal_x_data)  
         
        # 時間計測
        start_time = time.time()
        print(f'start_time:{start_time}')
        for index, row in inverse_metal_x_data.iterrows():
            for i in range(1, 6):
                metal_col = f'metal{i}'
                ratio_col = f'ratio{i}'
                metal = row[metal_col]
                ratio = row[ratio_col]
                inverse_df.at[index, metal] = ratio
        end_time = time.time()
        print(f'end_time:{end_time}')
        print("Elapsed_time : {0:.2f}[sec]".format(end_time - start_time))
                
        return inverse_df, metal_list, metal_component_list
        
    
    def get_x2_18_forGA(self, individual_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition):
        featurizers2 = Calc_desc()
            
        inverse_metal_x_data = pd.DataFrame(index=individual_df.index, columns=[f'metal{i}' for i in range(1, 6)] +
                                                                               [f'ratio{i}' for i in range(1, 6)])
        range_val = range(1, 1 + len(metal_list[0]))
        inverse_metal_x_data.loc[:, [f'metal{i}' for i in range_val]] = metal_list
        inverse_metal_x_data.loc[:, [f'ratio{i}' for i in range_val]] = metal_component_list
        mi_util = MIutility() #データ作成クラス生成
        #inverse_metal_x_data = mi_util.checkMetalRatioTotalForRatio3(inverse_metal_x_data) 
        inverse_metal_x_data = mi_util.checkMetalRatioTotal(inverse_metal_x_data)  
         
        # 時間計測
        start_time = time.time()
        print(f'start_time:{start_time}')
        inverse_df = featurizers2.getDesc_X2_18('Type2', inverse_df, inverse_metal_x_data)
        
        #x26_data, metal_data, descriptors = \
        #    featurizers.getDesc_X('Type2', inverse_df, inverse_metal_x_data, xenonpy_element_data,
        #                          target=target, shap_threshold=shap_threshold_v, boruta_p=boruta_p_v)
        
        end_time = time.time()
        print(f'end_time:{end_time}')
        print("Elapsed_time : {0:.2f}[sec]".format(end_time - start_time))
                
        return inverse_df, metal_list, metal_component_list
    
    def get_x2_20_forGA(self, individual_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition):
        featurizers2 = Calc_desc()
            
        inverse_metal_x_data = pd.DataFrame(index=individual_df.index, columns=[f'metal{i}' for i in range(1, 6)] +
                                                                               [f'ratio{i}' for i in range(1, 6)])
        range_val = range(1, 1 + len(metal_list[0]))
        inverse_metal_x_data.loc[:, [f'metal{i}' for i in range_val]] = metal_list
        inverse_metal_x_data.loc[:, [f'ratio{i}' for i in range_val]] = metal_component_list
        mi_util = MIutility() #データ作成クラス生成
        #inverse_metal_x_data = mi_util.checkMetalRatioTotalForRatio3(inverse_metal_x_data) 
        inverse_metal_x_data = mi_util.checkMetalRatioTotal(inverse_metal_x_data)  
         
        # 時間計測
        start_time = time.time()
        print(f'start_time:{start_time}')
        inverse_df = featurizers2.getDesc_X2_20('Type2', inverse_df, inverse_metal_x_data)
        
        #x26_data, metal_data, descriptors = \
        #    featurizers.getDesc_X('Type2', inverse_df, inverse_metal_x_data, xenonpy_element_data,
        #                          target=target, shap_threshold=shap_threshold_v, boruta_p=boruta_p_v)
        
        end_time = time.time()
        print(f'end_time:{end_time}')
        print("Elapsed_time : {0:.2f}[sec]".format(end_time - start_time))
                
        return inverse_df, metal_list, metal_component_list
  
    def get_x2_22_forGA(self, individual_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition):
        featurizers2 = Calc_desc()
            
        inverse_metal_x_data = pd.DataFrame(index=individual_df.index, columns=[f'metal{i}' for i in range(1, 6)] +
                                                                               [f'ratio{i}' for i in range(1, 6)])
        range_val = range(1, 1 + len(metal_list[0]))
        inverse_metal_x_data.loc[:, [f'metal{i}' for i in range_val]] = metal_list
        inverse_metal_x_data.loc[:, [f'ratio{i}' for i in range_val]] = metal_component_list
        mi_util = MIutility() #データ作成クラス生成
        #inverse_metal_x_data = mi_util.checkMetalRatioTotalForRatio3(inverse_metal_x_data) 
        inverse_metal_x_data = mi_util.checkMetalRatioTotal(inverse_metal_x_data)  
         
        # 時間計測
        start_time = time.time()
        print(f'start_time:{start_time}')
        inverse_df = featurizers2.getDesc_X2_22('Type2', inverse_df, inverse_metal_x_data)
        
        #x26_data, metal_data, descriptors = \
        #    featurizers.getDesc_X('Type2', inverse_df, inverse_metal_x_data, xenonpy_element_data,
        #                          target=target, shap_threshold=shap_threshold_v, boruta_p=boruta_p_v)
        
        end_time = time.time()
        print(f'end_time:{end_time}')
        print("Elapsed_time : {0:.2f}[sec]".format(end_time - start_time))
                
        return inverse_df, metal_list, metal_component_list 
  
  
def checkElements(synthesis_metal_x_data_modified, elements):
    
    # dfのカラム名抽出
    metal_x_columns_list = list(synthesis_metal_x_data_modified.columns.values)
    
    # 共通しない要素を抽出
    #print(set(metal_x_columns_list) ^ set(elements))    
    new_list = list(set(metal_x_columns_list) ^ set(elements))
            
    # 共通しない要素があったら、dfに列を追加(値は0)
    if len(new_list) != 0:
        synthesis_metal_x_data_modified[new_list] = 0
        
    return synthesis_metal_x_data_modified   

def CalcMatminer(metal_x):
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.base import MultipleFeaturizer, BaseFeaturizer
    from matminer.featurizers.composition.alloy import Miedema, YangSolidSolution, WenAlloys
    from matminer.featurizers.composition.ion import OxidationStates, IonProperty, ElectronAffinity, ElectronegativityDiff
    from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital
    from matminer.featurizers.composition.composite import ElementProperty, Meredig
    from matminer.featurizers.composition.element import BandCenter
    
    metal_x.replace(0, np.nan, inplace= True)
    metal_x.replace(np.nan, ' ', inplace= True)
    for index, row in metal_x.iterrows():
        metal1 = metal_x.loc[index, 'metal1']
        metal2 = metal_x.loc[index, 'metal2']        
        metal3 = metal_x.loc[index, 'metal3']        
        metal4 = metal_x.loc[index, 'metal4']
        metal5 = metal_x.loc[index, 'metal5']        
        ratio1 = metal_x.loc[index, 'ratio1']
        ratio2 = metal_x.loc[index, 'ratio2']        
        ratio3 = metal_x.loc[index, 'ratio3']        
        ratio4 = metal_x.loc[index, 'ratio4']
        ratio5 = metal_x.loc[index, 'ratio5']        
        metal_x.loc[index, 'chemicalFormula'] = metal1+str(ratio1)+metal2+str(ratio2)+metal3+str(ratio3)+\
            metal4+str(ratio4)+metal5+str(ratio5)
    
    #Teを含有する系を抽出して、それ以外は通常の方法で計算する
    Te_df = metal_x.query('chemicalFormula.str.contains("Te")', engine= 'python')
    #Teを含まない方の計算        
    metal_x['chemicalFormula'] = metal_x['chemicalFormula'].str.rstrip()
    #x_base_b.to_csv('../datasets/x_base_b.csv', encoding= 'utf-8-sig')

    stc = StrToComposition(target_col_id='composition')
    stc.set_n_jobs(1)
    metal_x = stc.featurize_dataframe(metal_x, "chemicalFormula", ignore_errors= True)

    #featurizerの指定　１つであればfeaturize_dataframeを使用
    f_Miedema = Miedema()
    f_Miedema.set_n_jobs(1)
    f_Meredig = Meredig()
    f_Meredig.set_n_jobs(1)
    f_WenAlloys = WenAlloys()
    f_WenAlloys.set_n_jobs(1)
    f_AtomicOrbitals = AtomicOrbitals()
    f_AtomicOrbitals.set_n_jobs(1)
    f_BandCenter = BandCenter()
    f_BandCenter.set_n_jobs(1)
    features_Miedema = f_Miedema.featurize_dataframe(metal_x, col_id='composition', ignore_errors= True)
    features_Meredig = f_Meredig.featurize_dataframe(metal_x, col_id='composition', ignore_errors= True)
    features_WenAlloys = f_WenAlloys.featurize_dataframe(metal_x, col_id='composition', ignore_errors= True)
    features_AtomicOrbitals= f_AtomicOrbitals.featurize_dataframe(metal_x, col_id='composition', ignore_errors= True)
    features_BandCenter = f_BandCenter.featurize_dataframe(metal_x, col_id= 'composition', ignore_errors= True)

    x22_a = features_Miedema.drop(['chemicalFormula','composition'], axis= 1)
    AtomicOrbitals_list = ['HOMO_energy','LUMO_energy','gap_AO']
    Meredig_list = ['mean AtomicWeight','mean Column','mean Row','range Number','mean Number','range AtomicRadius','mean AtomicRadius','range Electronegativity',\
                    'mean Electronegativity','avg s valence electrons','avg p valence electrons','avg d valence electrons','avg f valence electrons']
    WenAlloys_list = ['Yang delta','Yang omega','APE mean','Radii local mismatch','Radii gamma','Configuration entropy','Atomic weight mean','Total weight','Lambda entropy',\
                    'Electronegativity delta','Electronegativity local mismatch','VEC mean','Mixing enthalpy','Mean cohesive energy','Interant electrons','Interant s electrons',\
                    'Interant p electrons','Interant d electrons','Interant f electrons','Shear modulus mean','Shear modulus delta','Shear modulus local mismatch','Shear modulus strength model']
    x22_b = features_Meredig.loc[:, Meredig_list]
    x22_d = features_WenAlloys.loc[:, WenAlloys_list]
    x22_e = features_AtomicOrbitals.loc[:, AtomicOrbitals_list]
    x22_f = features_BandCenter.loc[:, 'band center']
    x22 = pd.concat([x22_a, x22_b, x22_d, x22_e, x22_f], axis= 1)

    return x22

def CalcX1(metal_x):
    #x1の作成
    xenonpy_merge = pd.read_csv('results/xenonpy_merge.csv', index_col=0)
    xenonpy_merge.drop('oxide', axis= 1, inplace= True)    
    
    weighted_average_name = list() # 加重平均の index 名
    weighted_variance_name = list() # 加重分散の index 名
    geometric_mean_name = list() # 幾何平均の index 名
    harmonic_mean_name = list() # 調和平均の index 名
    max_pooling_name = list() # 最大値の index 名
    min_pooling_name = list() # 最小値の index 名
    for j in xenonpy_merge.columns:
        weighted_average_name.append(f'ave_{j}')
        weighted_variance_name.append(f'var_{j}')
        geometric_mean_name.append(f'gmean_{j}')
        harmonic_mean_name.append(f'hmean_{j}')
        max_pooling_name.append(f'max_{j}')
        min_pooling_name.append(f'min_{j}')

    x1_metaldesc = pd.DataFrame(
        index=metal_x.index,
        columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
        )
    for i in range(metal_x.shape[0]):
        metal1 = metal_x.loc[metal_x.index[i], 'metal1']
        metal2 = metal_x.loc[metal_x.index[i], 'metal2']
        metal3 = metal_x.loc[metal_x.index[i], 'metal3']
        metal4 = metal_x.loc[metal_x.index[i], 'metal4']
        metal5 = metal_x.loc[metal_x.index[i], 'metal5']
        metal_rate1 = metal_x.loc[metal_x.index[i], 'ratio1']
        metal_rate2 = metal_x.loc[metal_x.index[i], 'ratio2']
        metal_rate3 = metal_x.loc[metal_x.index[i], 'ratio3']
        metal_rate4 = metal_x.loc[metal_x.index[i], 'ratio4']
        metal_rate5 = metal_x.loc[metal_x.index[i], 'ratio5']
            
        if pd.isna(metal5) is False:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            metal_desc3 = xenonpy_merge.loc[metal3, :].values
            metal_desc4 = xenonpy_merge.loc[metal4, :].values        
            metal_desc5 = xenonpy_merge.loc[metal5, :].values        
            mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])
        elif pd.isna(metal4) is False:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            metal_desc3 = xenonpy_merge.loc[metal3, :].values
            metal_desc4 = xenonpy_merge.loc[metal4, :].values        
            mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])    
        elif pd.isna(metal3) is False:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            metal_desc3 = xenonpy_merge.loc[metal3, :].values
            mt = np.array([metal_desc1, metal_desc2, metal_desc3])
            mr = np.array([metal_rate1, metal_rate2, metal_rate3])
        elif pd.isna(metal2) is False:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            metal_desc2 = xenonpy_merge.loc[metal2, :].values
            mt = np.array([metal_desc1, metal_desc2])
            mr = np.array([metal_rate1, metal_rate2])                
        else:
            metal_desc1 = xenonpy_merge.loc[metal1, :].values
            mt = np.array([metal_desc1])
            mr = np.array([metal_rate1])
        for desc in range(xenonpy_merge.shape[1]):
            d_name = xenonpy_merge.columns[desc]
            if np.isnan(mt[:, desc]).any():
                x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'var_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'hmean_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'max_{d_name}'].iloc[i] = np.nan
                x1_metaldesc[f'min_{d_name}'].iloc[i] = np.nan
                continue
            metal_x.to_csv('datasets/metal_x.csv', encoding= 'utf-8-sig')
            x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
            #x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr)
            x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)
            x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
            x1_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
            x1_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
            x1_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])

    x1_metaldesc = x1_metaldesc.replace([np.inf, -np.inf], np.nan)
    #ここでnanが生まれる可能性あり！！
    x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]   #全てnanの列を削除

    return x1_metaldesc



       
        