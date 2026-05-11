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

class MetalFeaturizers():
    def __init__(self):
        pass
         
    def getDesc_X26(self, ope_type, x_base, metal_x, xenonpy_element_data, shap_threshold, boruta_p):
        print("MetalFeaturizers")
        print("getMetalDesc_x26() start")

        def SHAP_Desc_Select(x_base, x_desc, shap_threshold):

            y1 = x_base.loc[:, 'C2 yield (%)']
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
        x_base_a = x_base.copy()
        metal_xx = metal_x.copy()
        metals = metal_x.copy()   
        xx_base = pd.concat([x_base_a, metal_xx], axis= 1)
        xx_base.replace('na', np.nan, inplace= True)         
        
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
            index=xx_base.index,
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )
        for i in range(xx_base.shape[0]):
            if ope_type == 'Type1':
                metal1 = xx_base['M1'].iloc[i]
                metal2 = xx_base['M2'].iloc[i]
                metal3 = xx_base['M3'].iloc[i]
                metal_rate1 = xx_base['ratio1'].iloc[i]
                metal_rate2 = xx_base['ratio2'].iloc[i]
                metal_rate3 = xx_base['ratio3'].iloc[i]
            if pd.isna(metal3) is False:
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
        
        if ope_type == 'Type1':
            x1_desc_col = SHAP_Desc_Select(xx_base, x1_metaldesc, shap_threshold)
            
        x1_desc = x1_metaldesc.loc[:, x1_desc_col]
               
        #Simpleimputerを使ってみる
        imputer= SimpleImputer(strategy= 'mean')
        try:
            imputed_df = imputer.fit_transform(x1_desc)
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
        for j in xenonpy_element_data.columns:
            weighted_average_name.append(f'ave_{j}')
            weighted_variance_name.append(f'var_{j}')
            max_pooling_name.append(f'max_{j}')
            min_pooling_name.append(f'min_{j}')
            subtraction_name.append(f'subtr_{j}')
            division_name.append(f'div_{j}')
            
        #!!!　metal_xxのインデックスを触媒ロットとして、インデックスで置換えをする。
        x10_metaldesc = pd.DataFrame(
            index=xx_base.index,
            columns=weighted_average_name+weighted_variance_name+max_pooling_name+min_pooling_name+subtraction_name+division_name
            )
        for i in range(xx_base.shape[0]):
            if ope_type == 'Type1':
                metal1 = xx_base['M1'].iloc[i]
                metal2 = xx_base['M2'].iloc[i]
                metal3 = xx_base['M3'].iloc[i]
                metal_rate1 = xx_base['ratio1'].iloc[i]
                metal_rate2 = xx_base['ratio2'].iloc[i]
                metal_rate3 = xx_base['ratio3'].iloc[i]
            if metal3 is not np.nan:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3], dtype= float)
            elif metal2 is not np.nan:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2], dtype= float)
                mr = np.array([metal_rate1, metal_rate2], dtype= float)
            else:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                mt = np.array([metal_desc1], dtype= float)
                mr = np.array([metal_rate1], dtype= float)         
            for n in range(xenonpy_element_data.shape[1]):    #nは触媒中元素に対応する
                d_name = xenonpy_element_data.columns[n]     #d_nameはxenonpyで定義するdescriptorに対応
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
            x10_desc_col = SHAP_Desc_Select(xx_base, x10_metaldesc, shap_threshold)
        
        x10_desc = imputed_df.loc[:, x10_desc_col]
        #x10_desc.to_csv('../datasets/x10_desc.csv', encoding= 'utf-8-sig')

        #x14の作成
        xenonpy_desc = xenonpy_element_data.columns
        x_base_c = xx_base.copy()        
        for index, row in x_base_c.iterrows():
            max_metal = pd.to_numeric(row[-3:]).idxmax()
            major_metal_position = 'M'+ max_metal[-1]  #Cation1Amountの後ろから番号を数える
            major_metal = x_base_c.at[index, major_metal_position]
            major_xenonpy_desc = xenonpy_element_data.iloc[xenonpy_element_data.index.get_loc(major_metal), :]
            x_base_c.loc[index, xenonpy_desc] = major_xenonpy_desc 
        
        x14_metaldesc = x_base_c.loc[:, xenonpy_desc]
        x14_desc_col = SHAP_Desc_Select(x_base_a, x14_metaldesc, shap_threshold)
        x14_desc = x14_metaldesc.loc[:, x14_desc_col]

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
        metals= metals.assign(groupa= 0, groupb= 0, groupc= 0, groupd= 0, groupe= 0, groupf= 0)
        metals= metals.assign(compoa= 0, compob= 0, compoc= 0, compod= 0, compoe= 0, compof= 0)
                
        for row in metals.itertuples():
            for i in range(1, 4):
                if row[i] in group_a:
                    metals.loc[row[0], 'groupa']+= 1
                    loc_a= pd.Index(row).get_loc(row[i])+ 2 
                    compo_a= metals.columns[loc_a]
                    metals.at[row[0], 'compoa']+= metals.at[row[0], compo_a]
                elif row[i] in group_b:
                    metals.loc[row[0], 'groupb']+= 1
                    loc_b= pd.Index(row).get_loc(row[i])+ 2 
                    compo_b= metals.columns[loc_b]
                    metals.at[row[0], 'compob']+= metals.at[row[0], compo_b]
                elif row[i] in group_c:
                    metals.loc[row[0], 'groupc']+= 1
                    loc_c= pd.Index(row).get_loc(row[i])+ 2 
                    compo_c= metals.columns[loc_c]
                    metals.at[row[0], 'compoc']+= metals.at[row[0], compo_c]
                elif row[i] in group_d:
                    metals.loc[row[0], 'groupd']+= 1
                    loc_d= pd.Index(row).get_loc(row[i])+ 2 
                    compo_d= metals.columns[loc_d]
                    metals.at[row[0], 'compod']+= metals.at[row[0], compo_d]
                elif row[i] in group_e:
                    metals.loc[row[0], 'groupe']+= 1
                    loc_e= pd.Index(row).get_loc(row[i])+ 2 
                    compo_e= metals.columns[loc_e]
                    metals.at[row[0], 'compoe']+= metals.at[row[0], compo_e]
                elif row[i] in group_f:
                    metals.loc[row[0], 'groupf']+= 1
                    loc_f= pd.Index(row).get_loc(row[i])+ 2
                    compo_f= metals.columns[loc_f]
                    metals.at[row[0], 'compof']+= metals.at[row[0], compo_f]
        
        x_base_d = xx_base.copy()
        x25_metaldesc = metals.columns[6:].to_list()
        x25_metaldesc = metals.loc[:, x25_metaldesc]
  
        x25_desc_col = SHAP_Desc_Select(x_base_d, x25_metaldesc, shap_threshold)
        x25_desc = x25_metaldesc.loc[:, x25_desc_col]

        #x22の作成
        metal_xx.replace(0, np.nan, inplace= True)
        metal_xx.replace('na', np.nan, inplace= True)
        metal_xx.replace(np.nan, ' ', inplace= True)

        for index, row in metal_xx.iterrows():
            metal1 = row[0]
            metal2 = row[1]    
            metal3 = row[2]    
            ratio1 = row[3]
            ratio2 = row[4]
            ratio3 = row[5]
            metal_xx.loc[index, 'chemicalFormula'] = metal1+str(ratio1)+metal2+str(ratio2)+metal3+str(ratio3)
                
        metal_xx['chemicalFormula'] = metal_xx['chemicalFormula'].str.rstrip()
        stc = StrToComposition(target_col_id='composition')
        stc.set_n_jobs(1)
        metal_xx = stc.featurize_dataframe(metal_xx, "chemicalFormula", ignore_errors= True)
        #metal_xx = StrToComposition(target_col_id='composition').featurize_dataframe(metal_xx, "chemicalFormula", ignore_errors=True)

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
        features_Miedema = f_Miedema.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
        features_Meredig = f_Meredig.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
        features_WenAlloys = f_WenAlloys.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
        features_AtomicOrbitals= f_AtomicOrbitals.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
        features_BandCenter = f_BandCenter.featurize_dataframe(metal_xx, col_id= 'composition', ignore_errors= True)
        #features_Miedema.to_csv('MyWork/datasets/f_Miedema.csv', encoding= 'cp932')

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
        x22_1 = pd.concat([x22_a, x22_b, x22_d, x22_e, x22_f], axis= 1)
        x22_1.drop(['Atomic weight mean','Total weight'], axis= 1, inplace= True)
        #x22_1.to_csv('MyWork/datasets/x22_1.csv', encoding= 'utf-8-sig')
        
        #x14もnanを含むため、ここでx22と結合して一緒にImpute処理することにする。
        x22_1.drop(['M1','M2','M3','ratio1','ratio2','ratio3'], axis= 1, inplace= True)
        df_for_imputer_a= pd.concat([x22_1, x14_desc], axis= 1)

        #デフォルトのestimatorはBaysianRidge
        imputer_a = IterativeImputer(max_iter= 10, random_state= 10)
        imputed_df_a= imputer_a.fit_transform(df_for_imputer_a)
        imputed_df_a= pd.DataFrame(imputed_df_a, index= x22_1.index)

        #impute後の処理
        imputed_df_a.columns= df_for_imputer_a.columns
        x22= imputed_df_a
        #x22= x22.sort_index()
        #x22.to_csv('MyWork/datasets/x22.csv', encoding= 'cp932')

        x22_metaldesc = x22.loc[:, x22_1.columns.to_list()]
        #x22_metaldesc.to_csv('../datasets/x22_metaldesc.csv', encoding= 'utf-8-sig')
        
        x22_desc_col = SHAP_Desc_Select(xx_base, x22_metaldesc, shap_threshold)
        
        x22_desc = x22.loc[:, x22_desc_col]
        x14_desc = x22.loc[:, x14_desc_col]

        x26_a = pd.concat([xx_base, x1_desc, x10_desc, x14_desc, x22_desc, x25_desc], axis= 1)
        x26_ab = x26_a.loc[:, ~x26_a.columns.duplicated()]
        x26_ab.drop(['M1','M2','M3','ratio1','ratio2','ratio3'], axis= 1, inplace= True)
        #x26_ab.to_csv('MyWork/datasets/x26_ab.csv', encoding= 'cp932')
        
        x26_e = x26_ab.drop(['C2 yield (%)'], axis= 1)
        addSquareCross(x26_e)

        x26_f = pd.concat([x26_ab['C2 yield (%)'], x26_e], axis= 1)
        #x26_f.to_csv('MyWOrk/datasets/x26_f.csv', encoding= 'utf-8-sig')

        #Borutaで特徴量を削減する
        y= x26_f['C2 yield (%)'].copy()
        x= x26_f.drop(['C2 yield (%)'], axis= 1).copy()

        # RandomForestRegressorでBorutaを実行
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, perc= boruta_p)
        feat_selector.fit(x.values, y.values)

        # 選択された特徴量を確認
        selected = feat_selector.support_
        print('選択された特徴量の数: %d' % np.sum(selected))
        selected = x.columns[selected]
        print(selected)

        #上で選択した説明変数のみを残す。
        x26_2 = x26_f.loc[:, selected]
        x26_2 = pd.concat([x26_f['C2 yield (%)'], x26_2], axis= 1)
            
        print("\ngetMetalDescXenon1() end")
        return x26_2

    def getDesc_X2_17(self, ope_type, x_base, metal_x):

        synthesis_metal_x_data = metal_x
        synthesis_metal_x_data = synthesis_metal_x_data.replace('na', np.nan)
        met1 = synthesis_metal_x_data['Cation1'].unique()
        met2 = synthesis_metal_x_data['Cation2'].unique()
        met3 = synthesis_metal_x_data['Cation3'].unique()
        metal_list = np.concatenate([met1, met2, met3])
        metal_list = pd.Series(metal_list)
        metal_list = metal_list.dropna()
        metal_list = metal_list.unique()
        metal_list = np.sort(metal_list)
        synthesis_metal_x_data_modified = pd.DataFrame(index= synthesis_metal_x_data.index, columns= metal_list) 
        elements = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt', 'Au', 'In', 'Sn']
        grouping = ['groupa','groupb','groupc','groupd','groupe','groupf','compoa','compob','compoc','compod','compoe','compof']

        #触媒組成のデータの元素、組成を上のdf形式へ変換
        synthesis_metal_x_data_modified['dummy'] = np.nan
        for index, row in synthesis_metal_x_data.iterrows():
            content1 = row[0], row[3]
            content2 = row[1], row[4]
            content3 = row[2], row[5]
            if pd.isna(content2[0]):
                content2 = list(content2)
                content2[0] = 'dummy'            
            if pd.isna(content3[0]):
                content3 = list(content3)
                content3[0] = 'dummy'
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content1[0])] = content1[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content2[0])] = content2[1]
            synthesis_metal_x_data_modified.iat[synthesis_metal_x_data_modified.index.get_loc(index), synthesis_metal_x_data_modified.columns.get_loc(content3[0])] = content3[1]
        synthesis_metal_x_data_modified.drop('dummy', axis= 1, inplace= True)
        synthesis_metal_x_data_modified.replace(np.nan, 0, inplace= True)
        
        if ope_type == 'Type1':
            x2 = pd.concat([x_base, synthesis_metal_x_data_modified], axis= 1)
            x2.replace(np.nan, 0, inplace= True)

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
            x2_17 = pd.concat([x2, metals], axis= 1)
            x2_17 = x2_17[~x2_17.index.duplicated()]
            x2_17 = x2_17.drop(['Cation1','Cation2','Cation3','Cation1Amount','Cation2Amount','Cation3Amount'], axis= 1)               
      
        return x2_17

    def getDesc_X26_1(self, ope_type, x_base, metal_x, xenonpy_element_data, shap_threshold, boruta_p):
        print("MetalFeaturizers")
        print("getMetalDesc_x26() start")

        def SHAP_Desc_Select(x_base, x_desc, shap_threshold):

            y1 = x_base.loc[:, 'S_c2']
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
        x_base_a = x_base.copy()
        metal_xx = metal_x.copy()
        metals = metal_x.copy()   
        xx_base = pd.concat([x_base_a, metal_xx], axis= 1)
        xx_base.replace('na', np.nan, inplace= True)         
        
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
            index=xx_base.index,
            columns=weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
            )
        for i in range(xx_base.shape[0]):
            if ope_type == 'Type1':
                metal1 = xx_base['metal1'].iloc[i]
                metal2 = xx_base['metal2'].iloc[i]
                metal3 = xx_base['metal3'].iloc[i]
                metal4 = xx_base['metal4'].iloc[i]
                metal_rate1 = xx_base['ratio1'].iloc[i]
                metal_rate2 = xx_base['ratio2'].iloc[i]
                metal_rate3 = xx_base['ratio3'].iloc[i]
                metal_rate4 = xx_base['ratio4'].iloc[i]
            if pd.isna(metal4) is False:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data.loc[metal3, :].values
                metal_desc4 = xenonpy_element_data.loc[metal4, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4])         
            elif pd.isna(metal3) is False:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3])
                mr = np.array([metal_rate1, metal_rate2, metal_rate3])
            elif pd.isna(metal2) is False:
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
        
        if ope_type == 'Type1':
            x1_desc_col = SHAP_Desc_Select(xx_base, x1_metaldesc, shap_threshold)
            
        x1_desc = x1_metaldesc.loc[:, x1_desc_col]
               
        #Simpleimputerを使ってみる
        imputer= SimpleImputer(strategy= 'mean')
        try:
            imputed_df = imputer.fit_transform(x1_desc)
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
        for j in xenonpy_element_data.columns:
            weighted_average_name.append(f'ave_{j}')
            weighted_variance_name.append(f'var_{j}')
            max_pooling_name.append(f'max_{j}')
            min_pooling_name.append(f'min_{j}')
            subtraction_name.append(f'subtr_{j}')
            division_name.append(f'div_{j}')
            
        #!!!　metal_xxのインデックスを触媒ロットとして、インデックスで置換えをする。
        x10_metaldesc = pd.DataFrame(
            index=xx_base.index,
            columns=weighted_average_name+weighted_variance_name+max_pooling_name+min_pooling_name+subtraction_name+division_name
            )
        for i in range(xx_base.shape[0]):
            if ope_type == 'Type1':
                metal1 = xx_base['metal1'].iloc[i]
                metal2 = xx_base['metal2'].iloc[i]
                metal3 = xx_base['metal3'].iloc[i]
                metal4 = xx_base['metal4'].iloc[i]
                metal_rate1 = xx_base['ratio1'].iloc[i]
                metal_rate2 = xx_base['ratio2'].iloc[i]
                metal_rate3 = xx_base['ratio3'].iloc[i]
                metal_rate4 = xx_base['ratio4'].iloc[i]
            if pd.isna(metal4) is False:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data.loc[metal3, :].values
                metal_desc4 = xenonpy_element_data.loc[metal4, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3, metal_desc4], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3, metal_rate4], dtype= float)
            if pd.isna(metal3) is False:     #if metal3.isna().sum() == 0:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                metal_desc3 = xenonpy_element_data.loc[metal3, :].values
                mt = np.array([metal_desc1, metal_desc2, metal_desc3], dtype= float)
                mr = np.array([metal_rate1, metal_rate2, metal_rate3], dtype= float)
            elif pd.isna(metal2) is False:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                metal_desc2 = xenonpy_element_data.loc[metal2, :].values
                mt = np.array([metal_desc1, metal_desc2], dtype= float)
                mr = np.array([metal_rate1, metal_rate2], dtype= float)
            else:
                metal_desc1 = xenonpy_element_data.loc[metal1, :].values
                mt = np.array([metal_desc1], dtype= float)
                mr = np.array([metal_rate1], dtype= float)         
            for n in range(xenonpy_element_data.shape[1]):    #nは触媒中元素に対応する
                d_name = xenonpy_element_data.columns[n]     #d_nameはxenonpyで定義するdescriptorに対応
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
            x10_desc_col = SHAP_Desc_Select(xx_base, x10_metaldesc, shap_threshold)
        
        x10_desc = imputed_df.loc[:, x10_desc_col]
        #x10_desc.to_csv('../datasets/x10_desc.csv', encoding= 'utf-8-sig')

        #x14の作成
        xenonpy_desc = xenonpy_element_data.columns
        x_base_c = xx_base.copy()        
        for index, row in x_base_c.iterrows():
            max_metal = pd.to_numeric(row[-4:]).idxmax()
            major_metal_position = 'metal'+ max_metal[-1]  #Cation1Amountの後ろから番号を数える
            major_metal = x_base_c.at[index, major_metal_position]
            major_xenonpy_desc = xenonpy_element_data.iloc[xenonpy_element_data.index.get_loc(major_metal), :]
            x_base_c.loc[index, xenonpy_desc] = major_xenonpy_desc 
        
        x14_metaldesc = x_base_c.loc[:, xenonpy_desc]
        x14_desc_col = SHAP_Desc_Select(x_base_a, x14_metaldesc, shap_threshold)
        x14_desc = x14_metaldesc.loc[:, x14_desc_col]

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
        metals= metals.assign(groupa= 0, groupb= 0, groupc= 0, groupd= 0, groupe= 0, groupf= 0)
        metals= metals.assign(compoa= 0, compob= 0, compoc= 0, compod= 0, compoe= 0, compof= 0)
                
        for row in metals.itertuples():
            for i in range(1, 4):
                if row[i] in group_a:
                    metals.loc[row[0], 'groupa']+= 1
                    loc_a= pd.Index(row).get_loc(row[i])+ 4 
                    compo_a= metals.columns[loc_a]
                    metals.at[row[0], 'compoa']+= metals.at[row[0], compo_a]
                elif row[i] in group_b:
                    metals.loc[row[0], 'groupb']+= 1
                    loc_b= pd.Index(row).get_loc(row[i])+ 4 
                    compo_b= metals.columns[loc_b]
                    metals.at[row[0], 'compob']+= metals.at[row[0], compo_b]
                elif row[i] in group_c:
                    metals.loc[row[0], 'groupc']+= 1
                    loc_c= pd.Index(row).get_loc(row[i])+ 4 
                    compo_c= metals.columns[loc_c]
                    metals.at[row[0], 'compoc']+= metals.at[row[0], compo_c]
                elif row[i] in group_d:
                    metals.loc[row[0], 'groupd']+= 1
                    loc_d= pd.Index(row).get_loc(row[i])+ 4 
                    compo_d= metals.columns[loc_d]
                    metals.at[row[0], 'compod']+= metals.at[row[0], compo_d]
                elif row[i] in group_e:
                    metals.loc[row[0], 'groupe']+= 1
                    loc_e= pd.Index(row).get_loc(row[i])+ 4 
                    compo_e= metals.columns[loc_e]
                    metals.at[row[0], 'compoe']+= metals.at[row[0], compo_e]
                elif row[i] in group_f:
                    metals.loc[row[0], 'groupf']+= 1
                    loc_f= pd.Index(row).get_loc(row[i])+ 4
                    compo_f= metals.columns[loc_f]
                    metals.at[row[0], 'compof']+= metals.at[row[0], compo_f]
        
        x_base_d = xx_base.copy()
        x25_metaldesc = metals.columns[6:].to_list()
        x25_metaldesc = metals.loc[:, x25_metaldesc]
  
        x25_desc_col = SHAP_Desc_Select(x_base_d, x25_metaldesc, shap_threshold)
        x25_desc = x25_metaldesc.loc[:, x25_desc_col]

        #x22の作成
        metal_xx.replace(0, np.nan, inplace= True)
        metal_xx.replace('na', np.nan, inplace= True)
        metal_xx.replace(np.nan, ' ', inplace= True)

        for index, row in metal_xx.iterrows():
            metal1 = row[0]
            metal2 = row[1]    
            metal3 = row[2]    
            ratio1 = row[3]
            ratio2 = row[4]
            ratio3 = row[5]
            metal_xx.loc[index, 'chemicalFormula'] = metal1+str(ratio1)+metal2+str(ratio2)+metal3+str(ratio3)
                
        metal_xx['chemicalFormula'] = metal_xx['chemicalFormula'].str.rstrip()
        stc = StrToComposition(target_col_id='composition')
        stc.set_n_jobs(1)
        metal_xx = stc.featurize_dataframe(metal_xx, "chemicalFormula", ignore_errors= True)
        #metal_xx = StrToComposition(target_col_id='composition').featurize_dataframe(metal_xx, "chemicalFormula", ignore_errors=True)

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
        features_Miedema = f_Miedema.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
        features_Meredig = f_Meredig.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
        features_WenAlloys = f_WenAlloys.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
        features_AtomicOrbitals= f_AtomicOrbitals.featurize_dataframe(metal_xx, col_id='composition', ignore_errors= True)
        features_BandCenter = f_BandCenter.featurize_dataframe(metal_xx, col_id= 'composition', ignore_errors= True)
        #features_Miedema.to_csv('MyWork/datasets/f_Miedema.csv', encoding= 'cp932')

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
        x22_1 = pd.concat([x22_a, x22_b, x22_d, x22_e, x22_f], axis= 1)
        x22_1.drop(['Atomic weight mean','Total weight'], axis= 1, inplace= True)
        #x22_1.to_csv('MyWork/datasets/x22_1.csv', encoding= 'utf-8-sig')
        
        #x14もnanを含むため、ここでx22と結合して一緒にImpute処理することにする。
        x22_1.drop(['metal1','metal2','metal3','metal4','ratio1','ratio2','ratio3','ratio4'], axis= 1, inplace= True)
        df_for_imputer_a= pd.concat([x22_1, x14_desc], axis= 1)

        #デフォルトのestimatorはBaysianRidge
        imputer_a = IterativeImputer(max_iter= 10, random_state= 10)
        imputed_df_a= imputer_a.fit_transform(df_for_imputer_a)
        imputed_df_a= pd.DataFrame(imputed_df_a, index= x22_1.index)

        #impute後の処理
        imputed_df_a.columns= df_for_imputer_a.columns
        x22= imputed_df_a
        #x22= x22.sort_index()
        #x22.to_csv('MyWork/datasets/x22.csv', encoding= 'cp932')

        x22_metaldesc = x22.loc[:, x22_1.columns.to_list()]
        #x22_metaldesc.to_csv('../datasets/x22_metaldesc.csv', encoding= 'utf-8-sig')
        
        x22_desc_col = SHAP_Desc_Select(xx_base, x22_metaldesc, shap_threshold)
        
        x22_desc = x22.loc[:, x22_desc_col]
        x14_desc = x22.loc[:, x14_desc_col]

        x26_a = pd.concat([xx_base, x1_desc, x10_desc, x14_desc, x22_desc, x25_desc], axis= 1)
        x26_ab = x26_a.loc[:, ~x26_a.columns.duplicated()]
        x26_ab.drop(['metal1','metal2','metal3','metal4','ratio1','ratio2','ratio3','ratio4'], axis= 1, inplace= True)
        #x26_ab.to_csv('MyWork/datasets/x26_ab.csv', encoding= 'cp932')
        
        x26_e = x26_ab.drop(['S_c2'], axis= 1)
        addSquareCross(x26_e)

        x26_f = pd.concat([x26_ab['S_c2'], x26_e], axis= 1)
        #x26_f.to_csv('MyWOrk/datasets/x26_f.csv', encoding= 'utf-8-sig')

        #Borutaで特徴量を削減する
        y= x26_f['S_c2'].copy()
        x= x26_f.drop(['S_c2'], axis= 1).copy()

        # RandomForestRegressorでBorutaを実行
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, perc= boruta_p)
        feat_selector.fit(x.values, y.values)

        # 選択された特徴量を確認
        selected = feat_selector.support_
        print('選択された特徴量の数: %d' % np.sum(selected))
        selected = x.columns[selected]
        print(selected)

        #上で選択した説明変数のみを残す。
        x26_2 = x26_f.loc[:, selected]
        x26_2 = pd.concat([x26_f['S_c2'], x26_2], axis= 1)
            
        print("\ngetMetalDescXenon1() end")
        return x26_2