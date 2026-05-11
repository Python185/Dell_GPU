# -*- coding: utf-8 -*-
"""
Created on Mon Feb 6 15:27:31 2023
@author: dcelab
"""
import sys, os
sys.path.append(os.pardir)
sys.path.append('./')
sys.path.append('./libs')
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
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern
from libs.InverseAnalysisUtility import MIutility

class common_function():
   #自乗項と交差項を追加
    def addSquareCross(df: pd.DataFrame):
        columns = df.columns
        for i, c1 in enumerate(columns):
            df[c1 + '^2'] = df[c1] ** 2
            for j, c2 in enumerate(columns):
                if i <= j:
                    continue
                df[c1 +' * ' + c2] = df[c1] * df[c2]

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

    def str_reverse(item):
        a = item.split(' * ')
        a.reverse()
        b = ' * '.join(a)
        return b

class MetalFeaturizers():
    def __init__(self):
        pass

    def getDesc_X(self, ope_type, x_base, metal_x, xenonpy_merge, target, shap_threshold, boruta_p):
        print("MetalFeaturizers")
        print("getMetalDesc_x26() start")

        #ここから説明変数の計算開始
        if ope_type == 'Type2':
            #metal1-5に被りがあるため、これを削除する
            metal_x.assign(check = 0)
            for index, row in metal_x.iterrows():
                row = row[1:6].dropna()
                metal_x.at[index, 'check'] = row.duplicated().sum()
            metal_x = metal_x[metal_x['check'] == 0]
            metal_x.drop('check', axis= 1, inplace= True)
            
            x_base_process = x_base.loc[:, :'support_ZrO2_RC100']
            data_num = min(x_base_process.shape[0], metal_x.shape[0])
            base_condition = pd.concat([x_base_process.iloc[:data_num, :], metal_x.iloc[:data_num, :]], axis= 1, join= 'inner')
            base_condition.drop('index', axis= 1, inplace= True, errors= 'ignore')
            x_base_a = base_condition.loc[:, :'support_ZrO2_RC100']
            metal_xx = base_condition.loc[:, 'metal1':]
            #base_condition.to_csv('../datasets/base_condition.csv', encoding= 'utf-8-sig')
            
            #descriptorsの処理
            if target == 'NPA':
                descriptors = pd.read_csv('datasets/descriptors_NPA.csv', encoding= 'utf-8-sig', index_col= 0, header=0)
            elif  target == 'ETA':
                descriptors = pd.read_csv('datasets/descriptors_ETA.csv', encoding= 'utf-8-sig', index_col= 0, header=0)
            elif  target == 'NPA_ETA_both':
                descriptors = pd.read_csv('datasets/descriptors_NPA_ETA_both.csv', encoding= 'utf-8-sig', index_col= 0, header=0)
                
            x1_list = descriptors['0'][descriptors['0'].str.startswith(('hmean','gmean'))].values.tolist()
            x10_list = descriptors['0'][descriptors['0'].str.startswith(('ave','var','min','max','div','subtr'))].values.tolist()
            x25_list = descriptors['0'][descriptors['0'].str.startswith(('group','compo'))].values.tolist()
            x14_list = descriptors['0'][descriptors['0'].str.startswith(('bandgap','work','atomic','boiling','brinell','bulk','c6','covalent','density','dipole','electron',\
                'en','first','fusion','gs','hhi','heat','icsd','evaporation','gas','lattice','linear','mendeleev','melting','metallic','molar','num','period',\
                    'poissons','proton','specific','thermal','vdw','sound','vickers','Polarizability','youngs'))].values.tolist()
            x22_list = descriptors['0'][descriptors['0'].str.startswith(('Miedema','mean','range','avg','band ','Yang','APE','Radii','Configuration','Lambda',\
                'Electronegativity','VEC','Mixing','Mean','Interant','Shear','HOMO','LUMO','gap', 'band '))].values.tolist()
            x1_list = [s.strip() for s in x1_list]
            x10_list = [s.strip() for s in x10_list]            
            x14_list = [s.strip() for s in x14_list]            
            x22_list = [s.strip() for s in x22_list]            
            x25_list = [s.strip() for s in x25_list]            
            
            check_list = x1_list + x10_list + x14_list + x22_list + x25_list
            check_list = set(check_list)   #setして重複を排除
            check_list = list(check_list)
            if descriptors.shape[0] != len(check_list):
                raise Exception('descriptorsの分割エラーです。MetalFeaturizers内容を確認して下さい。')
            metals = metal_xx.copy()
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
        for j in xenonpy_merge.columns:
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
                x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
                #x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr)
                x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)
                x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
                x1_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
                x1_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
                x1_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])

        x1_metaldesc = x1_metaldesc.replace([np.inf, -np.inf], np.nan)
        #ここでnanが生まれる可能性あり！！
        #x1_metaldesc.to_csv('datasets/x1_metaldesc.csv', encoding= 'utf-8-sig')
        #x1_list_df = pd.DataFrame(x1_list)
        #x1_list_df.to_csv('datasets/x1_list.csv', encoding= 'utf-8-sig')
        x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]   #全てnanの列を削除
        if ope_type == 'Type1':
            x1_desc_col = common_function.SHAP_Desc_Select(x_base_a, x1_metaldesc, target, shap_threshold)
            x1_desc = x1_metaldesc[x1_desc_col]
        else:  #x1_desc=x1_metaldesc.loc[:, x1_desc_col]で謎エラー発生
            x1_desc_col = x1_list
            x1_metaldesc_columns = x1_metaldesc.columns.to_list()
            x1_metaldesc_index_list = [i for i, x in enumerate(x1_metaldesc_columns) if x in x1_desc_col]
            x1_desc = x1_metaldesc.iloc[:, x1_metaldesc_index_list]
            
        x1_desc_columns = x1_desc.columns.to_list()
        x1_desc_columns = [s.strip() for s in x1_desc_columns]
        x1_desc.columns = x1_desc_columns

        #Simpleimputerを使ってみる→nan除去に伴いimputer使用停止
        imputer= SimpleImputer(strategy= 'mean')
        try:
            if x1_desc.empty == True:
                imputed_df = pd.DataFrame()
            else:
                #imputed_df = imputer.fit_transform(x1_desc)
                imputed_df = x1_desc                
        except ValueError:
            print("SHAP_thresholdを超えるDescriptorがありません。")
        
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
        for j in xenonpy_merge.columns:
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
                
            if pd.isna(metal5) is False:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                metal_desc5 = xenonpy_merge.loc[metal5, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5], dtype= float)
            elif pd.isna(metal4) is False:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4], dtype= float)    
            elif pd.isna(metal3) is False:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3], dtype= float)
            elif pd.isna(metal2) is False:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2], dtype= float)
                mr = np.array([metal_rate1, metal_rate2], dtype= float)                
            else:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                mt = np.array([metal_desc1], dtype= float)
                mr = np.array([metal_rate1], dtype= float)
            for n in range(xenonpy_merge.shape[1]):    #nは触媒中元素に対応する
                d_name = xenonpy_merge.columns[n]     #d_nameはxenonpyで定義するdescriptorに対応
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
        x10_metaldesc = x10_metaldesc.iloc[:,x10_metaldesc.notna().all(axis=0).values]
        #x10_metaldesc.to_csv('../datasets/x10_metaldesc.csv', encoding= 'utf-8-sig')

        #nanが8割以上で情報が少ない列を削除
        #threshold_nan= 0.8
        #ratio_of_nan= x10_metaldesc.isnull().sum()/x10_metaldesc.shape[0]
        #drop_columns= ratio_of_nan.loc[lambda x: x> threshold_nan].index
        #x10_metaldesc.drop(drop_columns, axis= 1, inplace= True)

        #df_for_imputer= x10_metaldesc.copy()
        #outlier= (df_for_imputer>1e7)|(df_for_imputer<-1e6)
        #outlier_columns= outlier.sum()

        #Simpleimputerを使ってみる→nan除去に伴いimputer使用停止
        imputer= SimpleImputer(strategy= 'mean')
        #imputed_df = imputer.fit_transform(x10_metaldesc)
        imputed_df = x10_metaldesc        
        imputed_df = pd.DataFrame(imputed_df, index= x10_metaldesc.index, columns=x10_metaldesc.columns)
   
        if ope_type == 'Type1':
            #x10_desc_col =  ['max_specific_heat','var_c6_gb','subtr_electron_affinity','max_linear_expansion_coefficient',\
            #                 'subtr_vdw_radius_uff','subtr_bulk_modulus','max_bulk_modulus','ave_boiling_point']
            x10_desc_col = common_function.SHAP_Desc_Select(x_base_a, x10_metaldesc, target, shap_threshold)
        else:
            x10_desc_col = x10_list
        
        x10_desc = imputed_df.loc[:, x10_desc_col]
        #x10_desc.to_csv('../datasets/x10_desc.csv', encoding= 'utf-8-sig')

        #x14の作成
        xenonpy_desc_list = xenonpy_merge.columns.to_list()
        for index, row in metal_xx.iterrows():
            metal_max = pd.to_numeric(row[5:10]).max(skipna= True)
            metal_max_group = row[row == metal_max]
            max_list = metal_max_group.index.to_list()
            metal_max_group_list = [s.replace('ratio', 'metal') for s in max_list]
            metal_max_group = row.loc[metal_max_group_list]
            metal_max_group = metal_max_group.sort_values()
            
            if metal_max_group.shape[0] == 1:
                max_metal = metal_max_group.index[0]
                metal_xx.at[index, 'major_metal'] = metal_xx.at[index, max_metal]
            elif metal_max_group.shape[0] == 2:
                max_metal = metal_max_group.index[0]
                max_metal1 = metal_max_group.index[1]
                metal_xx.at[index, 'major_metal'] = metal_xx.at[index, max_metal]
                metal_xx.at[index, 'major_metal1'] = metal_xx.at[index, max_metal1]
            elif metal_max_group.shape[0] == 3:
                max_metal = metal_max_group.index[0]
                max_metal1 = metal_max_group.index[1]
                max_metal2 = metal_max_group.index[2]
                metal_xx.at[index, 'major_metal'] = metal_xx.at[index, max_metal]
                metal_xx.at[index, 'major_metal1'] = metal_xx.at[index, max_metal1]
                metal_xx.at[index, 'major_metal2'] = metal_xx.at[index, max_metal2]                
            elif metal_max_group.shape[0] == 4:
                max_metal = metal_max_group.index[0]
                max_metal1 = metal_max_group.index[1]
                max_metal2 = metal_max_group.index[2]
                max_metal3 = metal_max_group.index[3]
                metal_xx.at[index, 'major_metal'] = metal_xx.at[index, max_metal]
                metal_xx.at[index, 'major_metal1'] = metal_xx.at[index, max_metal1]
                metal_xx.at[index, 'major_metal2'] = metal_xx.at[index, max_metal2]
                metal_xx.at[index, 'major_metal3'] = metal_xx.at[index, max_metal3]                                
            elif metal_max_group.shape[0] == 5:
                max_metal = metal_max_group.index[0]
                max_metal1 = metal_max_group.index[1]
                max_metal2 = metal_max_group.index[2]                
                max_metal3 = metal_max_group.index[3]                  
                max_metal4 = metal_max_group.index[4]                   
                metal_xx.at[index, 'major_metal'] = metal_xx.at[index, max_metal]
                metal_xx.at[index, 'major_metal1'] = metal_xx.at[index, max_metal1]
                metal_xx.at[index, 'major_metal2'] = metal_xx.at[index, max_metal2]   
                metal_xx.at[index, 'major_metal3'] = metal_xx.at[index, max_metal3]
                metal_xx.at[index, 'major_metal4'] = metal_xx.at[index, max_metal4]            
            
        x_base_c = x_base_a.copy()
        
        if ope_type == 'Type1':
            for index, row in x_base_c.iterrows():
                support_lot = row[4]
                major_metal_list = [s for s in metal_xx.columns.to_list() if 'major' in s]
                major_metals = metal_xx.loc[support_lot, major_metal_list]
                major_metals = [s for s in major_metals.values.tolist() if str(s) != 'nan']
                major_xenonpy_desc = xenonpy_merge.loc[major_metals, xenonpy_desc_list]
                major_xenonpy_desc_mean = major_xenonpy_desc.mean(axis= 0)
                x_base_c.loc[index, xenonpy_desc_list] = major_xenonpy_desc_mean
        else:
            for index, row in x_base_c.iterrows():
                major_metal_list = [s for s in metal_xx.columns.to_list() if 'major' in s]
                major_metals = metal_xx.loc[index, major_metal_list]
                major_metals = [s for s in major_metals.values.tolist() if str(s) != 'nan']
                major_xenonpy_desc = xenonpy_merge.loc[major_metals, xenonpy_desc_list]
                major_xenonpy_desc_mean = major_xenonpy_desc.mean(axis= 0)
                x_base_c.loc[index, xenonpy_desc_list] = major_xenonpy_desc_mean
        
        #x_base_c.to_csv('../datasets/x_base_c.csv', encoding= 'utf-8-sig')
        x14_metaldesc = x_base_c.loc[:, xenonpy_desc_list]
        #x14_metaldesc.to_csv('../datasets/x14_metaldesc.csv', encoding= 'utf-8-sig')
        
        if ope_type == 'Type1':
            x14_desc_col = common_function.SHAP_Desc_Select(x_base_a, x14_metaldesc, target, shap_threshold)
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
            x25_desc_col = common_function.SHAP_Desc_Select(x_base_a, x25, target, shap_threshold)
        else:
            x25 = metals
            x25_metaldesc = x25.iloc[:, 10:]
            #x25_metaldesc.to_csv('../datasets/x25_metaldesc.csv', encoding= 'utf-8-sig')
            x25_desc_col = x25_list
            
        x25_desc = x25.loc[:, x25_desc_col]

        #x22の作成
        metal_xx.replace(0, np.nan, inplace= True)
        metal_xx.replace(np.nan, ' ', inplace= True)
        metal_ratio = ['metal1','metal2','metal3','metal4','metal5','ratio1','ratio2','ratio3','ratio4','ratio5']
        metal_xx = metal_xx[metal_ratio]
        
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
        
        #Teを含有する系を抽出して、それ以外は通常の方法で計算する
        if ope_type == 'Type1':
            x_base_Te = x_base_b.copy()
            Te_df1 = metal_xx[metal_xx['metal1'] == 'Te']
            Te_df2 = metal_xx[metal_xx['metal2'] == 'Te']
            Te_df3 = metal_xx[metal_xx['metal3'] == 'Te']
            Te_df4 = metal_xx[metal_xx['metal4'] == 'Te']
            Te_df5 = metal_xx[metal_xx['metal5'] == 'Te']        
            Te_df = pd.concat([Te_df1, Te_df2, Te_df3, Te_df4, Te_df5], axis= 0)
            if Te_df.shape[0] != 0:
                Te_data = pd.merge(Te_df, x_base_Te, left_index= True, right_on= '触媒ロット')
                Te_cat_list = Te_data['触媒ロット'].to_list()
                x_base_Te = x_base_Te.query('触媒ロット == @Te_cat_list')
                #x_base_b = x_base_b.query('触媒ロット != @Te_cat_list')
        else:
            x_base_Te = x_base_b.query('chemicalFormula.str.contains("Te")', engine= 'python')
            #x_base_b = x_base_b.query('index != @x_base_Te.index.to_list()') 
        #Teを含まない方の計算        
        x_base_b['chemicalFormula'] = x_base_b['chemicalFormula'].str.rstrip()
        #x_base_b.to_csv('../datasets/x_base_b.csv', encoding= 'utf-8-sig')

        stc = StrToComposition(target_col_id='composition')
        stc.set_n_jobs(1)
        x_base_b = stc.featurize_dataframe(x_base_b, "chemicalFormula", ignore_errors= True)

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
        #x_base_columns = x_base_a.copy().columns.to_list()
        #x22_b = pd.concat([features_Meredig.loc[:, x_base_columns], features_Meredig.loc[:, Meredig_list]], axis= 1)
        x22_b = features_Meredig.loc[:, Meredig_list]
        x22_d = features_WenAlloys.loc[:, WenAlloys_list]
        x22_e = features_AtomicOrbitals.loc[:, AtomicOrbitals_list]
        x22_f = features_BandCenter.loc[:, 'band center']
        #x22_e.drop(['LUMO_character','LUMO_element'], axis= 1, inplace= True) #AtomicOrbitals用
        x22_1 = pd.concat([x22_a, x22_b, x22_d, x22_e, x22_f], axis= 1)
        x22_1.drop(['Atomic weight mean','Total weight'], axis= 1, inplace= True)
        #x22_1.drop(matminer_nan_list, axis= 1, inplace= True)
        #x22_1.to_csv('datasets/x22_1.csv', encoding= 'utf-8-sig')
        
        x22 = x22_1
        x22 = x22.sort_index()
        #x22_desc_col = ['Mean cohesive energy','Miedema_deltaH_ss_min','Mixing enthalpy','Interant electrons',\
        #                'range AtomicRadius','APE mean','HOMO_energy','range Number']
        x22_columns = set(x22.columns.to_list()) - set(x_base.columns.to_list())
        x22_columns = list(x22_columns)
        x22_columns.sort()
        x22_metaldesc = x22.loc[:, x22_columns]
        #Miedema_deltaH_ssにnanが大量発生　理由は不明　Miedema_deltaH_amorと相関高いためdropすることとする
        x22_metaldesc = x22_metaldesc.drop('Miedema_deltaH_ss_min', axis= 1)        
        x22_metaldesc.to_csv('datasets/x22_metaldesc.csv', encoding= 'utf-8-sig')
        
        #場合分け Teを含む触媒のデータが1つしかないときはautoscaling不可であるため、データそのものをdropする    
        if x22_metaldesc.isnull().any(axis= 1).sum() >= 2:
            #Teを含むデータをx1のデータを用いて予測する
            matminer_dict = {'Miedema_deltaH_inter':'GPR_10', 'Miedema_deltaH_amor':'GPR_10', 'Miedema_deltaH_ss_min':'GPR_6',
                'Yang delta':'GPR_6', 'Yang omega':'GPR_1', 'APE mean':'GPR_7', 'Radii local mismatch':'GPR_0',
                'Radii gamma':'GPR_0', 'Configuration entropy':'GPR_7', 'Lambda entropy':'GPR_7',
                'Electronegativity delta':'GPR_2', 'Electronegativity local mismatch':'GPR_8',
                'VEC mean':'GPR_0', 'Mixing enthalpy':'GPR_6', 'Mean cohesive energy':'GPR_0',
                'Interant electrons':'GPR_5', 'Interant s electrons':'GPR_0', 'Interant p electrons':'GPR_0',
                'Interant d electrons':'GPR_6', 'Interant f electrons':'GPR_0', 'Shear modulus mean':'GPR_0',
                'Shear modulus delta':'GPR_6', 'Shear modulus local mismatch':'GPR_6', 'Shear modulus strength model':'GPR_6'}
            #xmatの読込み
            try:
                xmat = pd.read_csv('datasets/xmat.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
            except FileNotFoundError:
                print('datasetsフォルダにxmat.csvがありません。Teを含むデータで必要です。')
                
            #逆解析のときは、matminer記述子を全て計算する必要ないため、matminer_dict1を必要な項目だけに縮小する
            if ope_type == 'Type2':
                matminer_current_dict = {key: value for key, value in matminer_dict.items() if key in x22_list}
            else:
                matminer_current_dict = matminer_dict
            
            #x_train, y_trainの決定
            matminer_keys =list(matminer_current_dict.keys())
            y_train = xmat.loc[:, matminer_keys]
            x_train = xmat.drop(matminer_keys, axis= 1)
            x_test = x1_metaldesc.copy()
            if ope_type == 'Type1':
                x_test = x_test.query('index in @x_base_Te["評価ロット数字"].to_list()')
            else:
                x_test = x_test.query('index == @x_base_Te.index.to_list()')            
                
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
            #xmat = xmat.drop(list(diff), axis= 1)
            x_train = x_train.drop(list(diff), axis= 1)
            print('x_trainの列を'+str(len(list(diff)))+'個削除しました。')
            #y_train = xmat.loc[:, list(matminer_dict.keys())]
            #x_train = xmat.drop(list(matminer_dict.keys()), axis= 1)
            diff1 = set(x_test_list) - set(x_train_list)
            x_test = x_test.drop(list(diff1), axis= 1)
            print('x_testの列を'+str(len(list(diff1)))+'個削除しました。')
                
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
             
            x22_metaldesc_add = estimated_y_df
            x22_metaldesc_add.index = x_test.index
        
            #Teを含まないdfとTeを含むdfの合成
            if ope_type == 'Type1':
                x22_metaldesc_add = x22_metaldesc_add.drop('Miedema_deltaH_ss_min', axis= 1)
            matminer_nan_list = list(matminer_dict.keys())
            Te_matminer_list = set(x22_metaldesc.columns.to_list()) - set(matminer_nan_list)
            Te_matminer_list = list(Te_matminer_list)
            x22_metaldesc_nan = x22_metaldesc[x22_metaldesc.isnull().any(axis= 1)]
            if x22_metaldesc_add.shape[0] != x22_metaldesc_nan.shape[0]:
                x22_metaldesc_nan = x22_metaldesc_nan.loc[x22_metaldesc_nan.index.isin(x22_metaldesc_add.index.to_list())]
            
            x22_metaldesc_add.loc[x22_metaldesc_nan.index, Te_matminer_list] = x22_metaldesc.loc[x22_metaldesc_nan.index, Te_matminer_list]
            if ope_type == 'Type1':
                x22_metaldesc = pd.concat([x22_metaldesc, x22_metaldesc_add], axis= 0)
            else:
                x22_metaldesc = x22_metaldesc.loc[:, ~x22_metaldesc.columns.duplicated()]
                x22_metaldesc = pd.concat([x22_metaldesc, x22_metaldesc_add], axis= 0)
            x22_metaldesc = x22_metaldesc[~x22_metaldesc.index.duplicated(keep= 'last')]
            x22_metaldesc = x22_metaldesc.sort_index()
            x22 = x22.sort_index()
            if ope_type == 'Type1':
                x22 = pd.concat([x22.loc[:, '選択率NPA':'support_ZrO2_RC100'], x22_metaldesc], axis= 1)
            else:
                x22 = pd.concat([x22.loc[:, '前処理還元炉温℃':'support_ZrO2_RC100'], x22_metaldesc], axis= 1)
            
        else:
            if ope_type == 'Type1':
                x22_metaldesc = x22_metaldesc.dropna(how= 'any', axis= 0)
                x22 = x22.sort_index()
                x22 = pd.concat([x22.loc[:, '選択率NPA':'support_ZrO2_RC100'], x22_metaldesc], axis= 1)
                x22 = x22.query('評価ロット数字 in @x22_metaldesc.index.to_list()')
                x_base_a =  x_base_a.query('評価ロット数字 in @x22_metaldesc.index.to_list()')
                x1_desc =  x1_desc.query('評価ロット数字 in @x22_metaldesc.index.to_list()')            
                x10_desc =  x10_desc.query('評価ロット数字 in @x22_metaldesc.index.to_list()')                
                x14_desc =  x14_desc.query('評価ロット数字 in @x22_metaldesc.index.to_list()')                
                x25_desc =  x25_desc.query('評価ロット数字 in @x22_metaldesc.index.to_list()')
            else:
                x22_metaldesc = x22_metaldesc.dropna(how= 'any', axis= 0)
                x22 = pd.concat([x22.loc[:, '前処理還元炉温℃':'support_ZrO2_RC100'], x22_metaldesc], axis= 1)
                x22 = x22.query('index in @x22_metaldesc.index.to_list()') 
                x_base_a =  x_base_a.query('index in @x22_metaldesc.index.to_list()')
                x1_desc =  x1_desc.query('index in @x22_metaldesc.index.to_list()')            
                x10_desc =  x10_desc.query('index in @x22_metaldesc.index.to_list()')                
                x14_desc =  x14_desc.query('index in @x22_metaldesc.index.to_list()')                
                x25_desc =  x25_desc.query('index in @x22_metaldesc.index.to_list()')                           
                  
        #x22_metaldesc.to_csv('datasets/x22_metaldesc.csv', encoding= 'utf-8-sig')            
                    
        if ope_type == 'Type1':
            x22_desc_col = common_function.SHAP_Desc_Select(x_base_a, x22_metaldesc, target, shap_threshold)
        else:
            x22_desc_col = x22_list
        
        x22_desc = x22.loc[:, x22_desc_col]
        #x14_desc = x22.loc[:, x14_desc_col]

        x26_a = pd.concat([x_base_a, x1_desc, x10_desc, x14_desc, x22_desc, x25_desc], axis= 1)
        x26_a = x26_a.dropna(how= 'any', axis= 0)
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
            #x26_c.columns = [i.replace('※', '') if '※' in i else i for i in x26_c.columns]
            #x26_c['yield'] = x26_c['選択率NPA'] + x26_c['選択率エタノール']
            #x26_c = x26_c[x26_c['yield']!= 0]
            #x26_c= x26_c.drop('yield', axis= 1)
            #x26_c.sort_index(inplace= True)
            #x26_c.to_csv('../datasets/x26_c.csv', encoding= 'utf-8-sig')
          
        if ope_type == 'Type1':
            x26_e = x26_c.drop(['選択率NPA','選択率エタノール','触媒ロット'], axis= 1)
            common_function.addSquareCross(x26_e)
    
            x26_f = pd.concat([x26_c.iloc[:, :2], x26_e], axis= 1)
            x26_f.insert(4, '触媒ロット', x26_c['触媒ロット'])
            #check1 = x25_desc.isnull().sum()
            #x26_f.to_csv('../datasets/x26_f.csv', encoding= 'utf-8-sig')
    
            #Borutaで特徴量を削減する
            y= x26_f[['選択率NPA', '選択率エタノール']].copy()
            if target == 'NPA':
                y = y['選択率NPA']
            elif target == 'ETA':
                y = y['選択率エタノール']
            
            #y['sum']= y['選択率NPA']+ y['選択率エタノール']
            #y= y['sum']
    
            x = x26_f.iloc[:, 2:]
            x_nan = x[x.isnull().any(axis= 1)]
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
            common_function.addSquareCross(x26_2)
            #addSquareCross(x26_2)
        
        print("\ngetMetalDescXenon1() end")
        return x26_2, metal_xx, descriptors


    def getMillionDesc(self, x_base, metal_x, xenonpy_merge):
        print("Make Million Data Start")
        
        matminer_nan_list = ['Miedema_deltaH_inter', 'Miedema_deltaH_amor', 'Miedema_deltaH_ss_min',
       'Yang delta', 'Yang omega', 'APE mean', 'Radii local mismatch',
       'Radii gamma', 'Configuration entropy', 'Lambda entropy',
       'Electronegativity delta', 'Electronegativity local mismatch',
       'VEC mean', 'Mixing enthalpy', 'Mean cohesive energy',
       'Interant electrons', 'Interant s electrons', 'Interant p electrons',
       'Interant d electrons', 'Interant f electrons', 'Shear modulus mean',
       'Shear modulus delta', 'Shear modulus local mismatch',
       'Shear modulus strength model']

        #ここから説明変数の計算開始
        #metal1-5に被りがあるため、これを削除する ratio2がnanのデータもあるのでこれも削除
        metal_x = metal_x.assign(check = 0)
        for index, row in metal_x.iterrows():
            row = row[0:5].dropna()
            metal_x.at[index, 'check'] = row.duplicated().sum()
        metal_x = metal_x[metal_x['check'] == 0]
        metal_x.drop('check', axis= 1, inplace= True)
        metal_x = metal_x.dropna(subset=['ratio1', 'ratio2'])
        
        x_base_process = x_base.iloc[:, :11]
        data_num = min(x_base_process.shape[0], metal_x.shape[0])
        base_condition = pd.concat([x_base_process.iloc[:data_num, :], metal_x.iloc[:data_num, :]], axis= 1, join= 'inner')
        base_condition.drop('index', axis= 1, inplace= True, errors= 'ignore')
        x_base_a = base_condition.iloc[:, :11]
        metal_xx = base_condition.iloc[:, 11:]
        #base_condition.to_csv('../datasets/base_condition.csv', encoding= 'utf-8-sig')
        metals = metal_xx.copy()

        #x1の作成
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
            index=x_base_a.index,
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )
        for i in range(x_base_a.shape[0]):
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
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                metal_desc5 = xenonpy_merge.loc[metal5, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])
            elif metal4 is not np.nan:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])    
            elif metal3 is not np.nan:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3])         
            else:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2])
                mr = np.array([metal_rate1, metal_rate2])
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
                x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
                #x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - np.average(mt[:, desc]))**2 , mr)
                x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)
                x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
                x1_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
                x1_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
                x1_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])

        x1_metaldesc = x1_metaldesc.replace([np.inf, -np.inf], np.nan)
        x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]   #全てnanの列を削除
        #x1_metaldesc.to_csv('datasets/x1_metaldesc.csv', encoding= 'utf-8-sig')

        #欠損値の補完
        #threshold_nan= 0.8
        #x1_metaldesc = x1_metaldesc.fillna(x1_metaldesc.mean())
        #ratio_of_nan= x1_metaldesc.isnull().sum()/x1_metaldesc.shape[0]
        #drop_columns= ratio_of_nan.loc[lambda x: x> threshold_nan].index
        #x1_metaldesc.drop(drop_columns, axis= 1, inplace= True)
 
        #x10の作成
        #xenonpy_merge.drop('oxide', axis= 1, inplace= True)
        weighted_average_name = list() # 加重平均の index 名
        weighted_variance_name = list() # 加重分散の index 名
        max_pooling_name = list() # 最大値の index 名
        min_pooling_name = list() # 最小値の index 名
        subtraction_name = list() #差のindex名
        division_name = list()    #商のindex名
        for j in xenonpy_merge.columns:
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
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                metal_desc5 = xenonpy_merge.loc[metal5, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5], dtype= float)
            elif metal4 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4], dtype= float)    
            elif metal3 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3], dtype= float)
            else:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2], dtype= float)
                mr = np.array([metal_rate1, metal_rate2], dtype= float)
            for n in range(xenonpy_merge.shape[1]):    #nは触媒中元素に対応する
                d_name = xenonpy_merge.columns[n]     #d_nameはxenonpyで定義するdescriptorに対応
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
        x10_metaldesc = x10_metaldesc.iloc[:,x10_metaldesc.notna().all(axis=0).values]

        #nanが8割以上で情報が少ない列を削除
        #threshold_nan= 0.8
        #ratio_of_nan= x10_metaldesc.isnull().sum()/x10_metaldesc.shape[0]
        #drop_columns= ratio_of_nan.loc[lambda x: x> threshold_nan].index
        #x10_metaldesc.drop(drop_columns, axis= 1, inplace= True)

        #df_for_imputer= x10_metaldesc.copy()
        #outlier= (df_for_imputer>1e7)|(df_for_imputer<-1e6)
        #outlier_columns= outlier.sum()

        #Simpleimputerを使ってみる
        #imputer= SimpleImputer(strategy= 'mean')
        #imputed_df = imputer.fit_transform(x10_metaldesc)
        #imputed_df = pd.DataFrame(imputed_df, index= x10_metaldesc.index, columns=x10_metaldesc.columns)
        #x10_metaldesc = imputed_df
        
        #x1とx10の重複を削除する
        x1_list = x1_metaldesc.columns[x1_metaldesc.columns.str.startswith(('hmean','gmean'))].values.tolist()
        #x1_list_add = x1_metaldesc.columns[x1_metaldesc.columns.str.contains(('en_pauling|hhi_r'))].values.tolist()
        #x1_list = x1_list + x1_list_add            
        x10_list = x10_metaldesc.columns[x10_metaldesc.columns.str.startswith(('ave','var','min','max','div','subtr'))].values.tolist()
        #x10_list = list(set(x10_list) - set(x1_list_add))
        x1_metaldesc = x1_metaldesc[x1_list]
        x10_metaldesc = x10_metaldesc[x10_list]

        #x14の作成
        xenonpy_desc = xenonpy_merge.columns[:]
        for index, row in metal_xx.iterrows():
            max_metal = pd.to_numeric(row[5:10]).idxmax()
            major_metal_position = 'metal'+ max_metal[-1]
            major_metal = metal_xx.at[index, major_metal_position]
            metal_xx.at[index, 'major_metal'] = major_metal
        x_base_c = x_base_a.copy()
        
        for index, row in x_base_c.iterrows():
            major_metal1 = metal_xx.loc[index, 'major_metal']
            major_xenonpy_desc = xenonpy_merge.iloc[xenonpy_merge.index.get_loc(major_metal1), :]
            x_base_c.loc[index, xenonpy_desc] = major_xenonpy_desc 
        
        #x_base_c.to_csv('../datasets/x_base_c.csv', encoding= 'utf-8-sig')
        #x14_desc_col = ['electron_affinity','Polarizability']
        x14_metaldesc = x_base_c.loc[:, xenonpy_desc.to_list()]
        #x14_metaldesc.to_csv('../datasets/x14_metaldesc.csv', encoding= 'utf-8-sig')
        
        #欠損値の補完
        #threshold_nan= 0.8
        #x14_metaldesc = x14_metaldesc.fillna(x14_metaldesc.mean())
        #ratio_of_nan= x14_metaldesc.isnull().sum()/x14_metaldesc.shape[0]
        #drop_columns= ratio_of_nan.loc[lambda x: x> threshold_nan].index
        #x14_metaldesc.drop(drop_columns, axis= 1, inplace= True)
        
        #x25の作成
        #元素をグルーピングする
        group_a = ['Ca','Sr','Ba','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Fe','Ru','Os']  #15
        group_b = ['Ni','Co','Rh','Pd','Pt','Ir','Mn','Cu','Tc','Re']  #10
        group_c = ['Al','Au']  #2
        group_d = ['Li','Na','K','Rb','Cs']  #5
        group_e = ['Mg','Ag','Zn','Cd','In','Si','Ge','Sn','Pb','As','Sb','Bi','Se','Te','Ga']  #15
        group_f = ['Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']  #17
        
        #metals = metals.replace(np.nan, 0)
        #metals.replace(0, np.nan, inplace= True)
        #metals.replace(np.nan, ' ', inplace= True)
        metals.replace([np.nan,' '], 0, inplace= True)    
        metals= metals.assign(groupa= 0, groupb= 0, groupc= 0, groupd= 0, groupe= 0, groupf= 0)
        metals= metals.assign(compoa= 0, compob= 0, compoc= 0, compod= 0, compoe= 0, compof= 0)
        
        try:        
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
        except:
            metals.to_csv('datasets/metals.csv', encoding= 'utf-8-sig')
            print('Excption!!')
        #unhashable typeエラーが出たときはmetal1-5の元素被りが原因

        x25 = metals
        x25_metaldesc = x25.iloc[:, 10:]
        #x25_metaldesc.to_csv('../datasets/x25_metaldesc.csv', encoding= 'utf-8-sig')

        #x22の作成
        metal_xx.replace(0, np.nan, inplace= True)
        metal_xx.replace(np.nan, ' ', inplace= True)
        metal_xx.drop('major_metal', axis= 1, inplace= True)
        
        x_base_b = x_base_a.copy()
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
        #f_Miedema = Miedema()
        f_Meredig = Meredig()
        f_Meredig.set_n_jobs(1)
        #f_WenAlloys = WenAlloys()
        f_AtomicOrbitals = AtomicOrbitals()
        f_AtomicOrbitals.set_n_jobs(1)
        f_BandCenter = BandCenter()
        f_BandCenter.set_n_jobs(1)
        #features_Miedema = f_Miedema.featurize_dataframe(x_base_b, col_id='composition', ignore_errors= True)
        features_Meredig = f_Meredig.featurize_dataframe(x_base_b, col_id='composition', ignore_errors= True)
        #features_WenAlloys = f_WenAlloys.featurize_dataframe(x_base_b, col_id='composition', ignore_errors= True)
        features_AtomicOrbitals= f_AtomicOrbitals.featurize_dataframe(x_base_b, col_id='composition', ignore_errors= True)
        features_BandCenter = f_BandCenter.featurize_dataframe(x_base_b, col_id= 'composition', ignore_errors= True)

        #x22_aa = features_Miedema.drop(['chemicalFormula','composition'], axis= 1)
        #x22_ab = x22_aa.iloc[:, :15]
        #x22_ac = x22_aa.iloc[:, -3:]
        #x22_a = pd.concat([x22_ab, x22_ac], axis= 1)
        AtomicOrbitals_list = ['HOMO_energy','LUMO_energy','gap_AO']
        Meredig_list = ['mean AtomicWeight','mean Column','mean Row','range Number','mean Number','range AtomicRadius','mean AtomicRadius','range Electronegativity',\
                        'mean Electronegativity','avg s valence electrons','avg p valence electrons','avg d valence electrons','avg f valence electrons']
        #WenAlloys_list = ['Yang delta','Yang omega','APE mean','Radii local mismatch','Radii gamma','Configuration entropy','Atomic weight mean','Total weight','Lambda entropy',\
        #                'Electronegativity delta','Electronegativity local mismatch','VEC mean','Mixing enthalpy','Mean cohesive energy','Interant electrons','Interant s electrons',\
        #                'Interant p electrons','Interant d electrons','Interant f electrons','Shear modulus mean','Shear modulus delta','Shear modulus local mismatch','Shear modulus strength model']
        x_base_columns = x_base_a.copy().columns.to_list()
        x22_b = pd.concat([features_Meredig.loc[:, x_base_columns], features_Meredig.loc[:, Meredig_list]], axis= 1)
        #x22_d = features_WenAlloys.loc[:, WenAlloys_list]
        x22_e = features_AtomicOrbitals.loc[:, AtomicOrbitals_list]
        x22_f = features_BandCenter.loc[:, 'band center']
        #x22_e.drop(['LUMO_character','LUMO_element'], axis= 1, inplace= True) #AtomicOrbitals用
        x22_1 = pd.concat([x22_b, x22_e, x22_f], axis= 1)
        #x22_1.drop(['Atomic weight mean','Total weight'], axis= 1, inplace= True)
        #x22_1.drop(matminer_nan_list, axis= 1, inplace= True)
        #x22_1.to_csv('../datasets/x22_1.csv', encoding= 'utf-8-sig')
        
        #Impute処理
        #df_for_imputer_a= x22_1.iloc[:, 14:]   #Miedemaに重複列あるため14列目からスライスする
        #outlier= (df_for_imputer_a>2e2)|(df_for_imputer__a<-1e6)
        #outlier_columns= outlier.sum()
        #df_for_imputer_a.mask(outlier, np.nan, inplace= True)

        #デフォルトのestimatorはBaysianRidge
        #imputer_a = IterativeImputer(max_iter= 10, random_state= 10)
        #imputed_df_a= imputer_a.fit_transform(df_for_imputer_a)

        #imputed_df_a= pd.DataFrame(imputed_df_a, columns= df_for_imputer_a.columns, index= df_for_imputer_a.index)
        #x22_metaldesc = imputed_df_a
        x22_columns = set(x22_1.columns.to_list()) - set(x_base_columns)
        x22_columns = list(x22_columns)
        x22_columns.sort()
        x22_metaldesc = x22_1.loc[:, x22_columns]
        #x22_metaldesc.to_csv('../datasets/x22_metaldesc.csv', encoding= 'utf-8-sig')

        print("Make Million Data End")
        return x1_metaldesc, x10_metaldesc, x14_metaldesc, x22_metaldesc, x25_metaldesc, base_condition

    def getAllDesc(self, x_base, metal_x, xenonpy_merge):
        print("Make All Descriptors Start")
        
        matminer_nan_list = ['Miedema_deltaH_inter', 'Miedema_deltaH_amor', 'Miedema_deltaH_ss_min',
       'Yang delta', 'Yang omega', 'APE mean', 'Radii local mismatch',
       'Radii gamma', 'Configuration entropy', 'Lambda entropy',
       'Electronegativity delta', 'Electronegativity local mismatch',
       'VEC mean', 'Mixing enthalpy', 'Mean cohesive energy',
       'Interant electrons', 'Interant s electrons', 'Interant p electrons',
       'Interant d electrons', 'Interant f electrons', 'Shear modulus mean',
       'Shear modulus delta', 'Shear modulus local mismatch',
       'Shear modulus strength model']

        #ここから説明変数の計算開始
        #metal1-5に被りがあるため、これを削除する ratio2がnanのデータもあるのでこれも削除
        metal_x = metal_x.assign(check = 0)
        for index, row in metal_x.iterrows():
            row = row[0:5].dropna()
            metal_x.at[index, 'check'] = row.duplicated().sum()
        metal_x = metal_x[metal_x['check'] == 0]
        metal_x.drop('check', axis= 1, inplace= True)
        metal_x = metal_x.dropna(subset=['ratio1', 'ratio2'])

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
        for j in xenonpy_merge.columns:
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
                
            if metal5 is not np.nan:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                metal_desc5 = xenonpy_merge.loc[metal5, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5])
            elif metal4 is not np.nan:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])    
            elif metal3 is not np.nan:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3])         
            else:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2])
                mr = np.array([metal_rate1, metal_rate2])
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
                x1_metaldesc[f'ave_{d_name}'].iloc[i] = np.dot(mt[:, desc],mr) / np.sum(mr)
                x1_metaldesc[f'var_{d_name}'].iloc[i] = np.dot((mt[:, desc] - (np.dot(mt[:, desc],mr)/np.sum(mr)))**2 , mr)
                x1_metaldesc[f'gmean_{d_name}'].iloc[i] = np.prod(mt[:, desc]**mr)**(1/sum(mr))
                x1_metaldesc[f'hmean_{d_name}'].iloc[i] = sum(mr)/sum((1/mt[:, desc])*mr)
                x1_metaldesc[f'max_{d_name}'].iloc[i] = max(mt[:, desc])
                x1_metaldesc[f'min_{d_name}'].iloc[i] = min(mt[:, desc])

        x1_metaldesc = x1_metaldesc.replace([np.inf, -np.inf], np.nan)
        x1_metaldesc = x1_metaldesc.iloc[:,x1_metaldesc.notna().all(axis=0).values]   #全てnanの列を削除
               
        #Simpleimputerを使ってみる Simpleimputerは、情報の少ない列を勝手に削除するため、下のように列名が失われないようにする
        #imputer= SimpleImputer(strategy= 'mean')
        #x1_metaldesc = pd.DataFrame(imputer.fit_transform(x1_metaldesc), columns= imputer.get_feature_names_out(), index=x1_metaldesc.index)

        #x10の作成
        #xenonpy_element_data1.drop('oxide', axis= 1, inplace= True)
        weighted_average_name = list() # 加重平均の index 名
        weighted_variance_name = list() # 加重分散の index 名
        max_pooling_name = list() # 最大値の index 名
        min_pooling_name = list() # 最小値の index 名
        subtraction_name = list() #差のindex名
        division_name = list()    #商のindex名
        for j in xenonpy_merge.columns:
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
                
            if metal5 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                metal_desc5 = xenonpy_merge.loc[metal5, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4, metal_desc5], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4, metal_rate5], dtype= float)
            elif metal4 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                metal_desc4 = xenonpy_merge.loc[metal4, :].values        
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4], dtype= float)    
            elif metal3 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                metal_desc3 = xenonpy_merge.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3], dtype= float)
            else:
                metal_desc1 = xenonpy_merge.loc[metal1, :].values
                metal_desc2 = xenonpy_merge.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2], dtype= float)
                mr = np.array([metal_rate1, metal_rate2], dtype= float)
            for n in range(xenonpy_merge.shape[1]):    #nは触媒中元素に対応する
                d_name = xenonpy_merge.columns[n]     #d_nameはxenonpyで定義するdescriptorに対応
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
        x10_metaldesc = x10_metaldesc.iloc[:,x10_metaldesc.notna().all(axis=0).values]
        #x10_metaldesc.to_csv('../datasets/x10_metaldesc.csv', encoding= 'utf-8-sig')

        #nanが8割以上で情報が少ない列を削除
        #threshold_nan= 0.8
        #ratio_of_nan= x10_metaldesc.isnull().sum()/x10_metaldesc.shape[0]
        #drop_columns= ratio_of_nan.loc[lambda x: x> threshold_nan].index
        #x10_metaldesc.drop(drop_columns, axis= 1, inplace= True)

        #Simpleimputerを使ってみる
        #imputer= SimpleImputer(strategy= 'mean')
        #imputed_df = imputer.fit_transform(x10_metaldesc)
        #imputed_df = pd.DataFrame(imputed_df, index= x10_metaldesc.index, columns=x10_metaldesc.columns)
        
        #x10_metaldesc = imputed_df
        #x10_desc.to_csv('../datasets/x10_desc.csv', encoding= 'utf-8-sig')

        #x14の作成
        xenonpy_desc = xenonpy_merge.columns[:]
        for index, row in metal_xx.iterrows():
            max_metal = pd.to_numeric(row[5:10]).idxmax()
            major_metal_position = 'metal'+ max_metal[-1]
            major_metal = metal_xx.at[index, major_metal_position]
            metal_xx.at[index, 'major_metal'] = major_metal
        x_base_c = x_base_a.copy()
        
        for index, row in x_base_c.iterrows():
            support_lot = row[4]
            major_metal1 = metal_xx.iat[metal_xx.index.get_loc(support_lot), 10]
            major_xenonpy_desc = xenonpy_merge.iloc[xenonpy_merge.index.get_loc(major_metal1), :]
            x_base_c.loc[index, xenonpy_desc] = major_xenonpy_desc 

        x14_metaldesc = x_base_c.loc[:, xenonpy_desc.to_list()]
        #x14_metaldesc.to_csv('../datasets/x14_metaldesc.csv', encoding= 'utf-8-sig')

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
                    
        #触媒ロットをメタル情報に置換
        x_base_d = x_base_a.copy()
        x25_metaldesc = metals.columns[10:].to_list()
        for index, row in x_base_d.iterrows():
            cat = row[4]
            x_base_d.loc[index, x25_metaldesc] = metals.iloc[metals.index.get_loc(cat), 10:]
            
        x25 = x_base_d.loc[:, x25_metaldesc]
        x25_metaldesc = x25

        #x22の作成
        metal_xx.replace(0, np.nan, inplace= True)
        metal_xx.replace(np.nan, ' ', inplace= True)
        metal_xx.drop('major_metal', axis= 1, inplace= True)
        
        x_base_b = x_base_a.copy()
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
                
        x_base_b['chemicalFormula'] = x_base_b['chemicalFormula'].str.rstrip()
        #x_base_b.to_csv('../datasets/x_base_b.csv', encoding= 'utf-8-sig')

        x_base_b = StrToComposition(target_col_id='composition').featurize_dataframe(x_base_b, "chemicalFormula", ignore_errors=True)

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
        x22_1.drop(matminer_nan_list, axis= 1, inplace= True)        
        #x22_1.to_csv('datasets/x22_1.csv', encoding= 'utf-8-sig')
        x22_metaldesc = x22_1.iloc[:, 15:]
        x22_metaldesc_list = x22_metaldesc.columns.to_list()
        
        #x14もnanを含むため、ここでx22と結合して一緒にImpute処理することにする。
        #df_for_imputer_a= pd.concat([x22_metaldesc, x14_metaldesc], axis= 1)
        #outlier= (df_for_imputer_a>1e12)|(df_for_imputer_a<-1e12)
        #outlier_columns= outlier.sum()
        #df_for_imputer_a.mask(outlier, np.nan, inplace= True)        
        
        #デフォルトのestimatorはBaysianRidge
        #imputer_a = IterativeImputer(max_iter= 10, random_state= 10)
        #imputed_df_a= imputer_a.fit_transform(df_for_imputer_a)

        #imputed_df_a= pd.DataFrame(imputed_df_a, index= x22_1.index)

        #impute後の処理
        #imputed_df= pd.concat([x22_1.iloc[:,0:14], imputed_df], axis= 1)
        #imputed_df_a.columns= df_for_imputer_a.columns
        #x22= imputed_df_a
        #x22 = x22.sort_index()
        #x22_desc_col = ['Mean cohesive energy','Miedema_deltaH_ss_min','Mixing enthalpy','Interant electrons',\
        #                'range AtomicRadius','APE mean','HOMO_energy','range Number']
        #x22_metaldesc = x22.loc[:, x22_metaldesc_list]
        #x22_metaldesc.to_csv('../datasets/x22_metaldesc.csv', encoding= 'utf-8-sig')
        #x14_metaldesc = x22.loc[:, x14_metaldesc.columns.to_list()]

        print("Make Million Data End")
        return x1_metaldesc, x10_metaldesc, x14_metaldesc, x22_metaldesc, x25_metaldesc, x_base

    def get_x26_forGA(self, individual_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition):
        xenonpy_element_data = pd.read_csv('results/xenonpy_merge.csv', index_col=0)
        #xenonpy_element_data = pd.read_csv('results/xenonpy_wo_nan.csv', index_col=0)
        #xenonpy_element_data1 = pd.read_csv('results/xenonpy_wo_nan1.csv', encoding= 'utf-8-sig', index_col= 0)  
        xenonpy_element_data.drop('oxide', axis=1, inplace=True)
        #xenonpy_element_data1.drop('oxide', axis=1, inplace=True)
        featurizers = MetalFeaturizers()
        if x_name == 'x26_NPA':
            target = 'NPA'
            shap_threshold_v =  0.03
            boruta_p_v = 90
        elif x_name == 'x26_ETA' :
            target = 'ETA'
            shap_threshold_v= 0.1
            boruta_p_v = 90
        elif x_name == 'x26_NPA_ETA_both':
            target = 'NPA_ETA_both'
            shap_threshold_v= 0.1
            boruta_p_v = 90
            
        inverse_metal_x_data = pd.DataFrame(index=individual_df.index, columns=[f'metal{i}' for i in range(1, 6)] +
                                                                               [f'ratio{i}' for i in range(1, 6)])
        range_val = range(1, 1 + len(metal_list[0]))
        inverse_metal_x_data.loc[:, [f'metal{i}' for i in range_val]] = metal_list
        inverse_metal_x_data.loc[:, [f'ratio{i}' for i in range_val]] = metal_component_list
        mi_util = MIutility() #データ作成クラス生成
        #inverse_metal_x_data = mi_util.checkMetalRatioTotalForRatio3(inverse_metal_x_data) 
        inverse_metal_x_data = mi_util.checkMetalRatioTotal(inverse_metal_x_data)  
        
        """
        #Te(テルル)が含まれた行は、除外する
        te_idx_list = []  
        inverse_metal_x_data = inverse_metal_x_data[~inverse_metal_x_data['metal1'].str.contains('Te')]
        inverse_metal_x_data = inverse_metal_x_data[~inverse_metal_x_data['metal2'].str.contains('Te')]
        inverse_metal_x_data = inverse_metal_x_data[~inverse_metal_x_data['metal3'].str.contains('Te')]
        inverse_metal_x_data = inverse_metal_x_data[~inverse_metal_x_data['metal4'].str.contains('Te')]
        inverse_metal_x_data = inverse_metal_x_data[~inverse_metal_x_data['metal5'].str.contains('Te')]   
        print("inverse_metal_x_data len  = ", len(inverse_metal_x_data))       
        te_idx_list = inverse_metal_x_data.index.values.tolist()
        
        # 元素「Te(テルル)」の行が削除された場合 : inverse_dfも同じindexくを削除
        non_idx_list = []      
        if len(inverse_df) != len(inverse_metal_x_data): 
            for idx, row in inverse_df.iterrows():
                if idx not in te_idx_list:   
                    non_idx_list.append(idx)
            # inverse_dfから drop 
            inverse_df = inverse_df.drop(non_idx_list, axis=0)                   
            #indexの降り直し
            inverse_metal_x_data = inverse_metal_x_data.reset_index(drop=True)
            inverse_df = inverse_df.reset_index(drop=True)
        print("non_idx_list : ", non_idx_list)        
        print("inverse_df_len = ", len(inverse_df))
        """              
        
        # 時間計測
        start_time = time.time()
        print(f'start_time:{start_time}')
        x26_data, metal_data, descriptors = \
            featurizers.getDesc_X('Type2', inverse_df, inverse_metal_x_data, xenonpy_element_data,
                                  target=target, shap_threshold=shap_threshold_v, boruta_p=boruta_p_v)
        
        end_time = time.time()
        print(f'end_time:{end_time}')
        print("Elapsed_time : {0:.2f}[sec]".format(end_time - start_time))
        
        # inverse_dfの交差項でx26_dataにないものを調べる。その交差項は、A * B ではなくB * Aならば、x26_dataにある可能性あり
        # inverse_dfの交差項,二乗項で一番最初のもののindexを求める
        inverse_df_cross_square = [item for item in inverse_df.columns if ' * ' in item or '^2' in item]
        if inverse_df_cross_square:
            idx_of_first_descdata = inverse_df.columns.to_list().index(inverse_df_cross_square[0])
            diff = set(inverse_df.columns[idx_of_first_descdata:].tolist()) - set(x26_data.columns.tolist())
            diff = [item for item in diff if ' * ' in item or '^2' in item]  # supportが入る可能性があるので、* ^2がついていないものを除く
        else:
            diff = []
        #x26_NPA_ETA_bothの時
        if target == 'NPA_ETA_both':
            x26_data['temp * 評価反応炉温℃']=0
            x26_data['min_icsd_volume * 評価反応炉温℃']=0
            x26_data['min_lattice_constant * 評価反応炉温℃']=0
            x26_data['Configuration entropy * 評価反応炉温℃']=0
        for idx in range(len(inverse_df)):
            if len(diff) != 0:
                diff = [s for s in diff if '*' in s]
                rev_diff = []
                for i in diff:
                    rev_diff.append(common_function.str_reverse(i))
                rename_dict = dict(zip(rev_diff, diff))
                x26_data.rename(columns=rename_dict, inplace=True)
            # 代入する列は、交差項と二乗項に限るために、*と^2がついていないものを除く
            clm = [item for item in inverse_df.columns[idx_of_first_descdata:] if ' * ' in item or '^2' in item]            
            inverse_df.loc[idx, clm] = x26_data.loc[idx, clm].astype(float)  
        
        return inverse_df, metal_list, metal_component_list
    
    """    
    def get_x26_forGA(self, individual_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition):
        xenonpy_element_data = pd.read_csv('results/xenonpy_merge.csv', index_col=0)
        #xenonpy_element_data = pd.read_csv('results/xenonpy_wo_nan.csv', index_col=0)
        #xenonpy_element_data1 = pd.read_csv('results/xenonpy_wo_nan1.csv', encoding= 'utf-8-sig', index_col= 0)  
        xenonpy_element_data.drop('oxide', axis=1, inplace=True)
        #xenonpy_element_data1.drop('oxide', axis=1, inplace=True)
        featurizers = MetalFeaturizers()
        if x_name == 'x26_NPA':
            target = 'NPA'
            shap_threshold_v =  0.03
            boruta_p_v = 90
        elif x_name == 'x26_ETA' :
            target = 'ETA'
            shap_threshold_v= 0.1
            boruta_p_v = 90
        elif x_name == 'x26_NPA_ETA_both':
            target = 'NPA_ETA_both'
            shap_threshold_v= 0.1
            boruta_p_v = 90
            
        inverse_metal_x_data = pd.DataFrame(index=individual_df.index, columns=[f'metal{i}' for i in range(1, 6)] +
                                                                               [f'ratio{i}' for i in range(1, 6)])
        inverse_metal_x_data.loc[:, [f'metal{i}' for i in range(1, 6)]] = metal_list
        inverse_metal_x_data.loc[:, [f'ratio{i}' for i in range(1, 6)]] = metal_component_list
        # 時間計測
        start_time = time.time()
        print(f'start_time:{start_time}')
        x26_data, metal_data, descriptors = \
            featurizers.getDesc_X('Type2', inverse_df, inverse_metal_x_data, xenonpy_element_data,
                                  target=target, shap_threshold=shap_threshold_v, boruta_p=boruta_p_v)
    
        #if(ssc_condition=='scc_1'):
        #    x26_data.insert(3, '評価反応炉温℃', 300)
        #    first_column = inverse_df.pop('評価反応炉温℃')
        #    inverse_df.insert(3,'評価反応炉温℃',first_column)
    
        end_time = time.time()
        print(f'end_time:{end_time}')
        print("Elapsed_time : {0}[sec]".format(end_time - start_time))
        for idx in range(len(inverse_df)):
            last_support = 'support_ZrO2_RC100'
            idx_of_first_descdata = inverse_df.iloc[idx, :].index.to_list().index(last_support) + 1
            diff = set(inverse_df.columns[idx_of_first_descdata:].tolist()) - set(x26_data.columns.tolist())
            if len(diff) != 0:
                diff = [s for s in diff if '*' in s]
                rev_diff = []
                for i in diff:
                    rev_diff.append(common_function.str_reverse(i))
                rename_dict = dict(zip(rev_diff, diff))
                x26_data.rename(columns=rename_dict, inplace=True)
            inverse_df.loc[idx, inverse_df.columns[idx_of_first_descdata:]] = \
                x26_data.loc[idx, inverse_df.columns[idx_of_first_descdata:]].astype(float)
                
        return inverse_df, metal_list, metal_component_list
    """    