# -*- coding: utf-8 -*-
"""
Created on Mon Feb 6 15:27:31 2023
@author: dcelab
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer, BaseFeaturizer
from matminer.featurizers.composition.alloy import Miedema, YangSolidSolution, WenAlloys
from matminer.featurizers.composition.ion import OxidationStates, IonProperty, ElectronAffinity, ElectronegativityDiff
from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital
from matminer.featurizers.composition.composite import ElementProperty, Meredig
from matminer.featurizers.composition.element import BandCenter
import shap
import xgboost as xgb
import lightgbm as lgb

class MetalFeaturizers:
    def __init__(self):
        pass
         
    def getDesc_X(self, ope_type, x_base, metal_x, xenonpy_element_data, xenonpy_element_data1, target, shap_threshold, boruta_p):
        print("MetalFeaturizers")
        print("getMetalDesc_x26() start")

        def SHAP_Desc_Select(x_base, x_desc, target, shap_threshold):
            if target == 'NPA':
                y1 = x_base.loc[:, '選択率NPA']
                model1 = xgb.XGBRegressor().fit(x_desc, y1)
                explainer = shap.Explainer(model1)
                shap_values = explainer(x_desc)
                shap.plots.waterfall(shap_values[0])
            
                df_shap_values = pd.DataFrame(data=shap_values.values,columns=x_desc.columns)
                df_feature_importance = pd.DataFrame(columns=['feature','importance'])
                for col in df_shap_values.columns:
                    importance = df_shap_values[col].abs().mean()
                    df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
                df_feature_importance = df_feature_importance.sort_values('importance',ascending=False)
                
                model2 = lgb.LGBMRegressor().fit(x_desc, y1)
                explainer_lgb = shap.Explainer(model2)
                shap_values_lgb = explainer_lgb(x_desc)
                shap.plots.waterfall(shap_values_lgb[0])
                
                df_shap_values_lgb = pd.DataFrame(data=shap_values_lgb.values,columns=x_desc.columns)
                df_feature_importance_lgb = pd.DataFrame(columns=['feature','importance'])
                for col in df_shap_values_lgb.columns:
                    importance = df_shap_values_lgb[col].abs().mean()
                    df_feature_importance_lgb.loc[len(df_feature_importance_lgb)] = [col,importance]
                df_feature_importance_lgb = df_feature_importance_lgb.sort_values('importance',ascending=False)

                desc_list = df_feature_importance.loc[df_feature_importance['importance'] > shap_threshold]
                desc_list_lgb = df_feature_importance_lgb.loc[df_feature_importance_lgb['importance'] > shap_threshold]
                
            elif target == 'ETA':
                y2 = x_base.loc[:, '選択率エタノール']
                model1 = xgb.XGBRegressor().fit(x_desc, y2)
                explainer = shap.Explainer(model1)
                shap_values = explainer(x_desc)
                shap.plots.waterfall(shap_values[0])
            
                df_shap_values = pd.DataFrame(data=shap_values.values,columns=x_desc.columns)
                df_feature_importance = pd.DataFrame(columns=['feature','importance'])
                for col in df_shap_values.columns:
                    importance = df_shap_values[col].abs().mean()
                    df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
                df_feature_importance = df_feature_importance.sort_values('importance',ascending=False)

                model2 = lgb.LGBMRegressor().fit(x_desc, y2)
                explainer_lgb = shap.Explainer(model2)
                shap_values_lgb = explainer_lgb(x_desc)
                shap.plots.waterfall(shap_values_lgb[0])
                
                df_shap_values_lgb = pd.DataFrame(data=shap_values_lgb.values,columns=x_desc.columns)
                df_feature_importance_lgb = pd.DataFrame(columns=['feature','importance'])
                for col in df_shap_values_lgb.columns:
                    importance = df_shap_values_lgb[col].abs().mean()
                    df_feature_importance_lgb.loc[len(df_feature_importance)] = [col,importance]
                df_feature_importance_lgb = df_feature_importance_lgb.sort_values('importance',ascending=False)           
        
                desc_list = df_feature_importance.loc[df_feature_importance['importance'] > shap_threshold]
                desc_list_lgb = df_feature_importance_lgb.loc[df_feature_importance_lgb['importance'] > shap_threshold]

            desc_list = pd.concat([desc_list, desc_list_lgb])
            desc_list.drop_duplicates(subset= ['feature'], inplace= True)         
            x_desc_col = desc_list['feature'].values.tolist()

            return x_desc_col
        
        #自乗項と交差項を追加
        def addSquareCross(df: pd.DataFrame):
            columns = df.columns
            for i, c1 in enumerate(columns):
                df[c1 + '^2'] = df[c1] ** 2
                for j, c2 in enumerate(columns):
                    if i <= j:
                        continue
                    df[c1 +' * ' + c2] = df[c1] * df[c2]

        #ここから説明変数の計算開始
        if ope_type == 'Type2':
            metal_x.assign(check = 0)
            for index, row in metal_x.iterrows():
                row = row.dropna()
                metal_x.at[index, 'check'] = row.duplicated().sum()
            metal_x = metal_x[metal_x['check'] == 0]
            metal_x.drop('check', axis= 1, inplace= True)
            
            x_base_process = x_base.iloc[:, :11]
            data_num = min(x_base_process.shape[0], metal_x.shape[0])
            base_condition = pd.concat([x_base_process.iloc[:, :data_num], metal_x.iloc[:, :data_num]], axis= 1, join= 'inner')
            base_condition.drop('index', axis= 1, inplace= True, errors= 'ignore')
            x_base_a = base_condition.iloc[:, :11]
            metal_xx = base_condition.iloc[:, 11:]
            base_condition.to_csv('../datasets/base_condition.csv', encoding= 'utf-8-sig')
            
            #descriptorsの処理:今は使用しないこととする。
            descriptors = pd.read_csv('../datasets/descriptors.csv', encoding= 'utf-8-sig', index_col= 0, header=0)
            x1_list = descriptors['0'][descriptors['0'].str.startswith(('hmean','gmean'))].values.tolist()
            x1_list_add = descriptors['0'][descriptors['0'].str.contains(('en_pauling|hhi_r'))].values.tolist()
            x1_list = x1_list + x1_list_add            
            x10_list = descriptors['0'][descriptors['0'].str.startswith(('ave','var','min','max','div','subtr'))].values.tolist()
            x10_list = list(set(x10_list) - set(x1_list_add))
            
            x25_list = descriptors['0'][descriptors['0'].str.startswith(('group','compo'))].values.tolist()
            x14_list = descriptors['0'][descriptors['0'].str.startswith(('atomic','boiling','brinell','bulk','c6','covalent','density','dipole','electron',\
                'en','first','fusion','gs','hhi','heat','icsd','evaporation','gas','lattice','linear','mendeleev','melting','metallic','molar','num','period',\
                    'poissons','proton','specific','thermal','vdw','sound','vickers','Polarizability','youngs'))].values.tolist()
            x22_list = descriptors['0'][descriptors['0'].str.startswith(('Miedema','mean','range','avg','band','Yang','APE','Radii','Configuration','Lambda',\
                'Electronegativity','VEC','Mixing','Mean','Interant','Shear','HOMO','LUMO','gap'))].values.tolist()
            
            check_list = x1_list + x10_list + x14_list + x22_list + x25_list
            check_list = set(check_list)   #setして重複を排除
            check_list = list(check_list)
            if descriptors.shape[0] != len(check_list):
                raise Exception('descriptorsの分割エラーです。MetalFeaturizers内容を確認して下さい。')
            
        else:
            x_base_a = x_base.copy()
            metal_xx = metal_x.copy()
            
        metals = metal_x.copy()
    
        #x1の作成
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

        x1_metaldesc = pd.DataFrame(
            index=x_base_a.index,
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )
        for i in range(x_base_a.shape[0]):
            if ope_type == 'Type1':
                c_lot = x_base_a['触媒ロット'].iloc[i]
                metal1 = metal_xx.loc[c_lot, 'metal1']
                metal2 = metal_xx.loc[c_lot, 'metal2']
                metal3 = metal_xx.loc[c_lot, 'metal3']
                metal4 = metal_xx.loc[c_lot, 'metal4']    
                metal5 = metal_xx.loc[c_lot, 'metal5']    
                metal_rate1 = metal_xx.loc[c_lot, 'ratio1']
                metal_rate2 = metal_xx.loc[c_lot, 'ratio2']
                metal_rate3 = metal_xx.loc[c_lot, 'ratio3']
                metal_rate4 = metal_xx.loc[c_lot, 'ratio4']
                metal_rate5 = metal_xx.loc[c_lot, 'ratio5']
            else:
                metal1 = metal_xx.loc[metal_xx.index[i], 'metal1']
                metal2 = metal_xx.loc[metal_xx.index[i], 'metal2']
                metal3 = metal_xx.loc[metal_xx.index[i], 'metal3']
                metal4 = metal_xx.loc[metal_xx.index[i], 'metal4']
                metal5 = metal_xx.loc[metal_xx.index[i], 'metal5']
                metal_rate1 = metal_xx.loc[metal_xx.index[i], 'ratio1']
                metal_rate2 = metal_xx.loc[metal_xx.index[i], 'ratio2']
                metal_rate3 = metal_xx.loc[metal_xx.index[i], 'ratio3']
                metal_rate4 = metal_xx.loc[metal_xx.index[i], 'ratio4']
                metal_rate5 = metal_xx.loc[metal_xx.index[i], 'ratio5']
                
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
            else:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2])
                mr = np.array([metal_rate1, metal_rate2])
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
        #x1_metaldesc.to_csv('../datasets/x1_metaldesc.csv', encoding= 'utf-8-sig')
        
        #x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]   #make_datasetでは必要かも
        if ope_type == 'Type1':
            x1_desc_col = SHAP_Desc_Select(x_base_a, x1_metaldesc, target, shap_threshold)
            #x1_desc_col = ['gmean_boiling_point','min_boiling_point','hmean_hhi_r','max_vdw_radius_uff','hmean_covalent_radius_pyykko']
        else:
            x1_desc_col = x1_list
            
        x1_desc = x1_metaldesc.loc[:, x1_desc_col]
               
        #Simpleimputerを使ってみる
        imputer= SimpleImputer(strategy= 'mean')
        imputed_df = imputer.fit_transform(x1_desc)
        x1_desc = pd.DataFrame(imputed_df, index= x1_desc.index, columns=x1_desc.columns)
        #x1_desc.to_csv('../datasets/x1_desc.csv', encoding= 'utf-8-sig')

        #x10の作成
        #xenonpy_element_data1.drop('oxide', axis= 1, inplace= True)
        weighted_average_name = list() # 加重平均の index 名
        weighted_variance_name = list() # 加重分散の index 名
        max_pooling_name = list() # 最大値の index 名
        min_pooling_name = list() # 最小値の index 名
        subtraction_name = list() #差のindex名
        division_name = list()    #商のindex名
        for j in xenonpy_element_data1.columns:
            weighted_average_name.append(f'ave_{j}')
            weighted_variance_name.append(f'var_{j}')
            max_pooling_name.append(f'max_{j}')
            min_pooling_name.append(f'min_{j}')
            subtraction_name.append(f'subtr_{j}')
            division_name.append(f'div_{j}')
            
        #!!!　metal_xxのインデックスを触媒ロットとして、インデックスで置換えをする。
        x10_metaldesc = pd.DataFrame(
            index=x_base_a.index,
            columns=weighted_average_name+weighted_variance_name+max_pooling_name+min_pooling_name+subtraction_name+division_name
            )
        for i in range(x_base_a.shape[0]):
            if ope_type == 'Type1':
                c_lot = x_base_a['触媒ロット'].iloc[i]   #iはx10の行に対応する
                metal1 = metal_xx.loc[c_lot, 'metal1']
                metal2 = metal_xx.loc[c_lot, 'metal2']
                metal3 = metal_xx.loc[c_lot, 'metal3']
                metal4 = metal_xx.loc[c_lot, 'metal4']    
                metal5 = metal_xx.loc[c_lot, 'metal5']    
                metal_rate1 = metal_xx.loc[c_lot, 'ratio1']
                metal_rate2 = metal_xx.loc[c_lot, 'ratio2']
                metal_rate3 = metal_xx.loc[c_lot, 'ratio3']
                metal_rate4 = metal_xx.loc[c_lot, 'ratio4']
                metal_rate5 = metal_xx.loc[c_lot, 'ratio5']
            else:
                metal1 = metal_xx.loc[metal_xx.index[i], 'metal1']
                metal2 = metal_xx.loc[metal_xx.index[i], 'metal2']
                metal3 = metal_xx.loc[metal_xx.index[i], 'metal3']
                metal4 = metal_xx.loc[metal_xx.index[i], 'metal4']
                metal5 = metal_xx.loc[metal_xx.index[i], 'metal5']
                metal_rate1 = metal_xx.loc[metal_xx.index[i], 'ratio1']
                metal_rate2 = metal_xx.loc[metal_xx.index[i], 'ratio2']
                metal_rate3 = metal_xx.loc[metal_xx.index[i], 'ratio3']
                metal_rate4 = metal_xx.loc[metal_xx.index[i], 'ratio4']
                metal_rate5 = metal_xx.loc[metal_xx.index[i], 'ratio5']
                
            if metal5 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_element_data1.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data1.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data1.loc[metal3, :].values
                metal_desc4 = xenonpy_element_data1.loc[metal4, :].values        
                metal_desc5 = xenonpy_element_data1.loc[metal5, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5], dtype= float)
            elif metal4 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_element_data1.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data1.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data1.loc[metal3, :].values
                metal_desc4 = xenonpy_element_data1.loc[metal4, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4], dtype= float)    
            elif metal3 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_element_data1.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data1.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data1.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3], dtype= float)
            else:
                metal_desc1 = xenonpy_element_data1.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data1.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2], dtype= float)
                mr = np.array([metal_rate1, metal_rate2], dtype= float)
            for n in range(xenonpy_element_data1.shape[1]):    #nは触媒中元素に対応する
                d_name = xenonpy_element_data1.columns[n]     #d_nameはxenonpyで定義するdescriptorに対応
                if np.isnan(mt[:, n]).any():     
                    x10_metaldesc[f'ave_{d_name}'].iloc[i] = np.nan
                    x10_metaldesc[f'var_{d_name}'].iloc[i] = np.nan
                    x10_metaldesc[f'max_{d_name}'].iloc[i] = np.nan
                    x10_metaldesc[f'min_{d_name}'].iloc[i] = np.nan
                    x10_metaldesc[f'subtr_{d_name}'].iloc[i] = np.nan
                    x10_metaldesc[f'div_{d_name}'].iloc[i] = np.nan
                    continue
                x10_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, n],mr) / np.sum(mr)
                x10_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, n] - (np.dot(mt[:, n],mr)/np.sum(mr)))**2 , mr)
                x10_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, n])
                x10_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, n])
                x10_metaldesc[f'subtr_{d_name}'].iloc[i] = max(mt[:, n]) - min(mt[:, n])
                x10_metaldesc[f'div_{d_name}'].iloc[i] = max(mt[:, n]) / min(mt[:, n])
        x10_metaldesc = x10_metaldesc.replace([np.inf, -np.inf], np.nan)
        #x10_metaldesc = x10_metaldesc.iloc[:,x10_metaldesc.notna().all(axis=0).values]
        #x10_metaldesc.to_csv('../datasets/x10_metaldesc.csv', encoding= 'utf-8-sig')

        #nanが8割以上で情報が少ない列を削除
        threshold_nan= 0.8
        ratio_of_nan= x10_metaldesc.isnull().sum()/x10_metaldesc.shape[0]
        drop_columns= ratio_of_nan.loc[lambda x: x> threshold_nan].index
        x10_metaldesc.drop(drop_columns, axis= 1, inplace= True)

        #df_for_imputer= x10_metaldesc.copy()
        #outlier= (df_for_imputer>1e7)|(df_for_imputer<-1e6)
        #outlier_columns= outlier.sum()

        #Simpleimputerを使ってみる
        imputer= SimpleImputer(strategy= 'mean')
        imputed_df = imputer.fit_transform(x10_metaldesc)
        imputed_df = pd.DataFrame(imputed_df, index= x10_metaldesc.index, columns=x10_metaldesc.columns)
   
        if ope_type == 'Type1':
            #x10_desc_col =  ['max_specific_heat','var_c6_gb','subtr_electron_affinity','max_linear_expansion_coefficient',\
            #                 'subtr_vdw_radius_uff','subtr_bulk_modulus','max_bulk_modulus','ave_boiling_point']
            x10_desc_col = SHAP_Desc_Select(x_base_a, x10_metaldesc, target, shap_threshold)
        else:
            x10_desc_col = x10_list
        
        x10_desc = imputed_df.loc[:, x10_desc_col]
        #x10_desc.to_csv('../datasets/x10_desc.csv', encoding= 'utf-8-sig')

        #x14の作成
        xenonpy_desc = xenonpy_element_data1.columns[4:]
        for index, row in metal_xx.iterrows():
            max_metal = pd.to_numeric(row[5:10]).idxmax()
            major_metal_position = 'metal'+ max_metal[-1]
            major_metal = metal_xx.at[index, major_metal_position]
            metal_xx.at[index, 'major_metal'] = major_metal
        x_base_c = x_base_a.copy()
        
        if ope_type == 'Type1':
            for index, row in x_base_c.iterrows():
                support_lot = row[4]
                major_metal1 = metal_xx.iat[metal_xx.index.get_loc(support_lot), 10]
                major_xenonpy_desc = xenonpy_element_data1.iloc[xenonpy_element_data1.index.get_loc(major_metal1), 4:]
                x_base_c.loc[index, xenonpy_desc] = major_xenonpy_desc 
        else:
            for index, row in x_base_c.iterrows():
                major_metal1 = metal_xx.loc[index, 'major_metal']
                major_xenonpy_desc = xenonpy_element_data1.iloc[xenonpy_element_data1.index.get_loc(major_metal1), 4:]
                x_base_c.loc[index, xenonpy_desc] = major_xenonpy_desc 
        
        #x_base_c.to_csv('../datasets/x_base_c.csv', encoding= 'utf-8-sig')
        #x14_desc_col = ['electron_affinity','Polarizability']
        x14_metaldesc = x_base_c.iloc[:, 15:]
        #x14_metaldesc.to_csv('../datasets/x14_metaldesc.csv', encoding= 'utf-8-sig')
        
        if ope_type == 'Type1':
            x14_desc_col = SHAP_Desc_Select(x_base_a, x14_metaldesc, target, shap_threshold)
        else:
            x14_desc_col = x14_list
            
        x14_desc = x_base_c.loc[:, x14_desc_col]
        #x14_desc.to_csv('../datasets/x14_desc.csv', encoding= 'utf-8-sig')

        #x25の作成
        #元素をグルーピングする
        group_a = ['Ca','Sr','Ba','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Fe','Ru','Os']  #15
        group_b = ['Ni','Co','Rh','Pd','Pt','Ir','Mn','Cu','Tc','Re']  #10
        group_c = ['Al','Au']  #2
        group_d = ['Li','Na','K','Rb','Cs']  #5
        group_e = ['Mg','Ag','Zn','Cd','In','Si','Ge','Sn','Pb','As','Sb','Bi','Se','Te','Ga']  #15
        group_f = ['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']  #17
        
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
                    
        if ope_type == 'Type1':
            #触媒ロットをメタル情報に置換
            x_base_d = x_base_a.copy()
            x25_metaldesc = metals.columns[10:].to_list()
            for index, row in x_base_d.iterrows():
                cat = row[4]
                x_base_d.loc[index, x25_metaldesc] = metals.iloc[metals.index.get_loc(cat), 10:]
                
            x25 = x_base_d.loc[:, x25_metaldesc]
            x25_desc_col = SHAP_Desc_Select(x_base_a, x25, target, shap_threshold)
        else:
            x25 = metals
            x25_metaldesc = x25.iloc[:, 10:]
            #x25_metaldesc.to_csv('../datasets/x25_metaldesc.csv', encoding= 'utf-8-sig')
            x25_desc_col = x25_list
            
        x25_desc = x25.loc[:, x25_desc_col]

        #x22の作成
        metal_xx.replace(0, np.nan, inplace= True)
        metal_xx.replace(np.nan, ' ', inplace= True)
        metal_xx.drop('major_metal', axis= 1, inplace= True)
        
        x_base_b = x_base_a.copy()
        if ope_type == 'Type1':
            for index, row in x_base_b.iterrows():
                catalyst_lot = row[4]
                metal1 = metal_xx.loc[catalyst_lot, 'metal1']
                metal2 = metal_xx.loc[catalyst_lot, 'metal2']    
                metal3 = metal_xx.loc[catalyst_lot, 'metal3']    
                metal4 = metal_xx.loc[catalyst_lot, 'metal4']
                metal5 = metal_xx.loc[catalyst_lot, 'metal5']    
                ratio1 = metal_xx.loc[catalyst_lot, 'ratio1']
                ratio2 = metal_xx.loc[catalyst_lot, 'ratio2']
                ratio3 = metal_xx.loc[catalyst_lot, 'ratio3']
                ratio4 = metal_xx.loc[catalyst_lot, 'ratio4']
                ratio5 = metal_xx.loc[catalyst_lot, 'ratio5']
                x_base_b.loc[index, 'chemicalFormula'] = metal1+str(ratio1)+metal2+str(ratio2)+metal3+str(ratio3)+\
                    metal4+str(ratio4)+metal5+str(ratio5)
        else:
            for index, row in x_base_b.iterrows():
                metal1 = metal_xx.loc[index, 'metal1']
                metal2 = metal_xx.loc[index, 'metal2']
                metal3 = metal_xx.loc[index, 'metal3']
                metal4 = metal_xx.loc[index, 'metal4']
                metal5 = metal_xx.loc[index, 'metal5']
                ratio1 = metal_xx.loc[index, 'ratio1']
                ratio2 = metal_xx.loc[index, 'ratio2']
                ratio3 = metal_xx.loc[index, 'ratio3']
                ratio4 = metal_xx.loc[index, 'ratio4']
                ratio5 = metal_xx.loc[index, 'ratio5']
                x_base_b.loc[index, 'chemicalFormula'] = metal1+str(ratio1)+metal2+str(ratio2)+metal3+str(ratio3)+\
                    metal4+str(ratio4)+metal5+str(ratio5)
                
        x_base_b['chemicalFormula'] = x_base_b['chemicalFormula'].str.rstrip()
        #x_base_b.to_csv('../datasets/x_base_b.csv', encoding= 'utf-8-sig')

        x_base_b = StrToComposition(target_col_id='composition').featurize_dataframe(x_base_b, "chemicalFormula", ignore_errors=True)

        #featurizerの指定　１つであればfeaturize_dataframeを使用
       
        f_Miedema = Miedema()
        f_Meredig = Meredig()
        f_WenAlloys = WenAlloys()
        f_AtomicOrbitals = AtomicOrbitals()
        f_BandCenter = BandCenter()
        features_Miedema = f_Miedema.featurize_dataframe(x_base_b, col_id='composition', ignore_errors= True)
        features_Meredig = f_Meredig.featurize_dataframe(x_base_b, col_id='composition', ignore_errors= True)
        features_WenAlloys = f_WenAlloys.featurize_dataframe(x_base_b, col_id='composition', ignore_errors= True)
        features_AtomicOrbitals= f_AtomicOrbitals.featurize_dataframe(x_base_b, col_id='composition', ignore_errors= True)
        features_BandCenter = f_BandCenter.featurize_dataframe(x_base_b, col_id= 'composition', ignore_errors= True)

        x22_aa = features_Miedema.drop(['chemicalFormula','composition'], axis= 1)
        x22_ab = x22_aa.iloc[:, :15]
        x22_ac = x22_aa.iloc[:, -3:]
        x22_a = pd.concat([x22_ab, x22_ac], axis= 1)
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
        #x22_e.drop(['LUMO_character','LUMO_element'], axis= 1, inplace= True) #AtomicOrbitals用
        x22_1 = pd.concat([x22_a, x22_b, x22_d, x22_e, x22_f], axis= 1)
        x22_1.drop(['Atomic weight mean','Total weight'], axis= 1, inplace= True)
        #x22_1.to_csv('../datasets/x22_1.csv', encoding= 'utf-8-sig')
        
        #x14もnanを含むため、ここでx22と結合して一緒にImpute処理することにする。
        df_for_imputer_a= pd.concat([x22_1.iloc[:, 14:], x14_desc], axis= 1)
        #outlier= (df_for_imputer_a>2e2)|(df_for_imputer__a<-1e6)
        #outlier_columns= outlier.sum()
        #df_for_imputer_a.mask(outlier, np.nan, inplace= True)

        #デフォルトのestimatorはBaysianRidge
        imputer_a = IterativeImputer(max_iter= 10, random_state= 10)
        imputed_df_a= imputer_a.fit_transform(df_for_imputer_a)

        imputed_df_a= pd.DataFrame(imputed_df_a, index= x22_1.index)
        #outlier_after_impute= (imputed_df>2e2)|(imputed_df<-1e6)
        #outlier_columns_after_impute= outlier_after_impute.sum()

        #impute後の処理
        #imputed_df= pd.concat([x22_1.iloc[:,0:14], imputed_df], axis= 1)
        imputed_df_a.columns= df_for_imputer_a.columns
        x22= imputed_df_a
        x22= x22.sort_index()
        #x22_desc_col = ['Mean cohesive energy','Miedema_deltaH_ss_min','Mixing enthalpy','Interant electrons',\
        #                'range AtomicRadius','APE mean','HOMO_energy','range Number']
        x22_metaldesc = x22.iloc[:, 15:]
        x22_metaldesc.to_csv('../datasets/x22_metaldesc.csv', encoding= 'utf-8-sig')
        
        if ope_type == 'Type1':
            x22_desc_col = SHAP_Desc_Select(x_base_a, x22_metaldesc, target, shap_threshold)
        else:
            x22_desc_col = x22_list
        
        x22_desc = x22.loc[:, x22_desc_col]
        x14_desc = x22.loc[:, x14_desc_col]

        x26_a = pd.concat([x_base_a, x1_desc, x10_desc, x14_desc, x22_desc, x25_desc], axis= 1)
        x26_ab = x26_a.loc[:, ~x26_a.columns.duplicated()]
        #x26_ab.to_csv('../datasets/x26_ab.csv', encoding= 'utf-8-sig')
        descriptors = x1_desc_col + x10_desc_col + x14_desc_col + x22_desc_col + x25_desc_col
        descriptors = set(descriptors)        
        
        if ope_type == 'Type1':
            x26_c = x26_ab.drop('評価ロット数字', axis= 1)
    
            #エタノール選択率の説明変数(SHAP)
            #features_list = ['gmean_boiling_point','min_boiling_point','hmean_hhi_r','max_vdw_radius_uff','hmean_covalent_radius_pyykko',\
            #                 'max_specific_heat','var_c6_gb','subtr_electron_affinity','max_linear_expansion_coefficient','electron_affinity',\
            #                 'subtr_vdw_radius_uff','subtr_bulk_modulus','Polarizability','max_bulk_modulus','ave_boiling_point',\
            #                 'Mean cohesive energy','Miedema_deltaH_ss_min','Mixing enthalpy','Interant electrons','range AtomicRadius',\
            #                 'APE mean','HOMO_energy','range Number','compoa','compob','compod','compof','groupa']
    
            #選択率がどちらも0を除外する
            x26_c.columns = [i.replace('※', '') if '※' in i else i for i in x26_c.columns]
            x26_c['yield'] = x26_c['選択率NPA'] + x26_c['選択率エタノール']
            x26_c = x26_c[x26_c['yield']!= 0]
            x26_c= x26_c.drop('yield', axis= 1)
            x26_c.sort_index(inplace= True)
            #x26_c.to_csv('../datasets/x26_c.csv', encoding= 'utf-8-sig')
          
        if ope_type == 'Type1':
            x26_e = x26_c.drop(['選択率NPA','選択率エタノール','触媒ロット'], axis= 1)
            addSquareCross(x26_e)
    
            x26_f = pd.concat([x26_c.iloc[:, :2], x26_e], axis= 1)
            x26_f.insert(4, '触媒ロット', x26_c['触媒ロット'])
            #check1 = x25_desc.isnull().sum()
            #x26_f.to_csv('../datasets/x26_f.csv', encoding= 'utf-8-sig')
    
            #Borutaで特徴量を削減する
            y= x26_f[['選択率NPA', '選択率エタノール']].copy()
            y['sum']= y['選択率NPA']+ y['選択率エタノール']
            y= y['sum']
    
            x= x26_f.iloc[:, 2:]
            #y= y.iloc[0:100]
            #x= x1.iloc[0:100, 2:20]
            x = x.drop('触媒ロット', axis= 1)
    
            # RandomForestRegressorでBorutaを実行
            rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, perc= boruta_p)
            feat_selector.fit(x.values, y.values)
    
            # 選択された特徴量を確認
            selected = feat_selector.support_
            print('選択された特徴量の数: %d' % np.sum(selected))
            print(x.columns[selected])
    
            #上で選択した説明変数のみを残す。
            boruta_descriptors = x.columns[11:]
            selected_features= boruta_descriptors[selected[11:]]
            x26_g = x26_f.iloc[:, 0:14]
            x26_h= x26_f.loc[:, selected_features]
            x26_2 = pd.concat([x26_g, x26_h], axis= 1)
            
        else:
            x26_2 = x26_ab
            addSquareCross(x26_2)
        
        print("\ngetMetalDescXenon1() end")
        return x26_2, metal_xx, descriptors

