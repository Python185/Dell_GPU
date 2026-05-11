import random
random.seed(1)

import time
import copy
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # import IterativeImputerのために必要
from sklearn.impute import IterativeImputer
# matminerのfeaturizersを使ってみる。
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer, BaseFeaturizer
from matminer.featurizers.composition.alloy import Miedema, YangSolidSolution, WenAlloys
from matminer.featurizers.composition.element import BandCenter, TMetalFraction
from matminer.featurizers.composition.ion import OxidationStates, IonProperty, ElectronAffinity, ElectronegativityDiff
from matminer.featurizers.composition.orbital import AtomicOrbitals, ValenceOrbital
from matminer.featurizers.composition.composite import ElementProperty, Meredig
from matminer.featurizers.composition.packing import AtomicPackingEfficiency
from matminer.featurizers.composition.thermo import CohesiveEnergy, CohesiveEnergyMP

from .GP_model import GP_model_process
from .util_ga import GAutility, checkMetalRatioTotal
from libs.Calculator_x26 import MetalFeaturizers
from libs.Calculator_x import Calc_desc

ga_util = GAutility()


def str_reverse(item):
    a = item.split(' * ')
    a.reverse()
    b = ' * '.join(a)
    return b


class GApreprocess(GP_model_process):
    def __init__(self, x_name):
        self.condition_dict = None
        self.x_name = x_name
        if x_name == 'x1' or 'x1_' in x_name or x_name == 'x10':
            xenonpy_element_data = pd.read_csv('results/xenonpy_element_data.csv', index_col=0)
            xenonpy_col_name = ga_util.get_xenonpyList(xenonpy_element_data)
            self.xenonpy_element_data = xenonpy_element_data
            self.xenonpy_col_name = xenonpy_col_name

    def add_metal_to_dict(self, condition_dict, metal_num_N, atom_list, step):
        rtn_condition_dict = copy.deepcopy(condition_dict)
        if step == 0.1:
            range_list = [0.1, 0.9]
        else:
            range_list = [0.01, 0.99]
        # 金属の追加
        for m in range(metal_num_N):
            rtn_condition_dict[f'metal{m+1}'] = ['metal_list', atom_list]
            rtn_condition_dict[f'ratio{m+1}'] = ['metal_float', range_list, 2]
        return rtn_condition_dict

    def add_metal_to_dict_constraints(self, condition_dict, metal_num_N, atom_list_N, atom_list_Y, step):
        rtn_condition_dict = copy.deepcopy(condition_dict)
        if step == 0.1:
            range_list = [0.1, 0.9]
        else:
            range_list = [0.01, 0.99]
        # 金属の数を選択:"Omit":'N'の場合
        # 金属種の追加
        for m in range(metal_num_N):
            rtn_condition_dict[f'metal{m+1}'] = ['metal_list',  atom_list_N, 'N']
            rtn_condition_dict[f'ratio{m+1}'] = ['metal_float', range_list, 2]
        # 金属の数を選択"Omit":'Y'
        metal_num_Y = 2
        #metal_num_Y = random.choice([1, 2])
        # 金属種の追加
        for m in range(metal_num_Y):
            rtn_condition_dict[f'metal{metal_num_N+m+1}'] = ['metal_list',  atom_list_Y, 'Y']
            rtn_condition_dict[f'ratio{metal_num_N+m+1}'] = ['metal_float', range_list, 2]
        return rtn_condition_dict

    # 遺伝子をxに変換する関数
    def ind_to_xdata(self, x_name, gene_metal_df, inverse_df, condition_dict, step, constraints, return_metal=False):
        self.condition_dict = condition_dict
        rtn_message = ['' for i in range(len(gene_metal_df))]
        metal_list = [[] for i in range(len(gene_metal_df))]
        metal_component_list = [[] for i in range(len(gene_metal_df))]
        for idx in range(len(gene_metal_df)):
            rtn_message[idx] = 'success'
            for i, col in enumerate(self.condition_dict.keys()):
                if col in ['前処理還元炉温℃', '評価反応炉温℃', 'temp', 'flow_NaOH', 'flow_slurry', 'flow_red', 'wash'] \
                        and col not in inverse_df.columns:
                    # この項目は本来ならばinverse_df.columnsに入っているはずだが、
                    # drop_same_values(x_data, threshold_of_rate_of_same_value=0.95)でその列が同じ値のもののために削除された
                    # 例えば、'評価反応炉温℃'は、300度ですべて生成されるので、同じ値となりdrop_same_valuesで消えている
                    continue
                if self.condition_dict[col][0] == 'list':
                    # list の場合は項目リストのインデックスを指定
                    idx_list = int(int(gene_metal_df.loc[idx, col])/(100/len(self.condition_dict[col][1])))
                    ind_list_selection = self.condition_dict[col][1][idx_list]
                    if col == 'support':
                        inverse_df.loc[idx, ind_list_selection] = 1
                    else:
                        inverse_df.loc[idx, col] = ind_list_selection
                elif self.condition_dict[col][0] == 'metal_list':  # 金属種リスト
                    metal_list[idx].append(gene_metal_df.loc[idx, col])
                elif self.condition_dict[col][0] == 'metal_float':  # 金属組成リスト
                    metal_component_list[idx].append(float(gene_metal_df.loc[idx, col]))
                else:
                    raise Exception(f'{self.condition_dict[col][0]}はまだ、実装していません')

            if len(metal_list[idx]) != len(set(metal_list[idx])):
                # metalが重複しているときは、エラーを返す
                rtn_message[idx] = 'metal duplicated'
            if 'flow_red' in inverse_df.columns and 'flow_NaOH' in inverse_df.columns and 'flow_slurry' in inverse_df.columns and \
                    (inverse_df.loc[idx, 'flow_red'] + inverse_df.loc[idx, 'flow_NaOH']) / inverse_df.loc[idx, 'flow_slurry'] >= 8.09:
                    # (flow_red + flow_NaOH) / flow_slurry >= 8.09　は除外
                    rtn_message[idx] = 'invalid condition'

            # 束縛条件で分岐
            if not constraints:
                metal_component_list[idx] = [round(x/sum(metal_component_list[idx]), 2) for x in metal_component_list[idx]]
            else:
                metal_component_list[idx] = ga_util.get_ratioListConstraints_v2(metal_list[idx], metal_component_list[idx])
            metal_component_list[idx] = checkMetalRatioTotal(metal_component_list[idx], step)

        # 統計量の計算
        if x_name == 'x1' or 'x1_' in x_name:
            self.get_x1_(inverse_df, metal_list, metal_component_list)

        elif x_name == 'x10':
            self.get_x10_(inverse_df, metal_list, metal_component_list)

        elif x_name == 'x22_4' or x_name == 'x22':
            self.get_x22_(inverse_df, metal_list, metal_component_list)

        elif 'x26_ETA' in x_name or 'x26_NPA' in x_name:
            if x_name == 'x26_ETA3':
                x_name_new = 'x26_ETA'
            elif x_name == 'x26_NPA2':
                x_name_new = 'x26_NPA'
            featurizers = MetalFeaturizers()
            ssc_condition = 'scc_0'
            _, _, _ = featurizers.get_x26_forGA(
                gene_metal_df, inverse_df, metal_list, metal_component_list, x_name_new, ssc_condition)

        elif 'x2_17' in x_name:
            featurizers2 = Calc_desc()
            ssc_condition = 'scc_0'
            featurizers2.get_x2_forGA(gene_metal_df, inverse_df, metal_list, metal_component_list, x_name,ssc_condition)

        elif 'x2_18' in x_name:
            featurizers2 = Calc_desc()
            ssc_condition = 'scc_0'
            featurizers2.get_x2_18_forGA(gene_metal_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition)

        elif 'x2_20' in x_name:
            featurizers2 = Calc_desc()
            ssc_condition = 'scc_0'
            featurizers2.get_x2_20_forGA(gene_metal_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition)

        elif 'x2_22' in x_name:
            featurizers2 = Calc_desc()
            ssc_condition = 'scc_0'
            featurizers2.get_x2_22_forGA(gene_metal_df, inverse_df, metal_list, metal_component_list, x_name, ssc_condition)
        else:
            raise Exception(f"{x_name}に対するGA逆解析が未実装です")

        if return_metal:
            return rtn_message, inverse_df, metal_list, metal_component_list
        else:
            return rtn_message, inverse_df

    def get_x1_(self, inverse_df, metal_list, metal_component_list):
        for idx in range(len(inverse_df)):
            # xenonpy記述子の計算
            metal_to_xenonpy_array = np.zeros((len(metal_list[idx]), self.xenonpy_element_data.shape[1]))
            for i, metal in enumerate(metal_list[idx]):
                # 金属種データの取得
                metal_to_xenonpy_array[i, :] = self.xenonpy_element_data.loc[metal, :].values
            metal_component_array = np.array(metal_component_list[idx])
            x_metaldesc = pd.DataFrame(columns=self.xenonpy_col_name)
            for desc in range(self.xenonpy_element_data.shape[1]):
                d_name = self.xenonpy_element_data.columns[desc]

                if np.isnan(metal_to_xenonpy_array[:, desc]).any():
                    x_metaldesc.loc[i, f'ave_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'var_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'gmean_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'hmean_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'max_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'min_{d_name}'] = np.nan
                    continue
                x_metaldesc_array = np.array([
                    np.dot(metal_to_xenonpy_array[:, desc], metal_component_array) / np.sum(metal_component_array),
                    np.dot((metal_to_xenonpy_array[:, desc] - np.average(metal_to_xenonpy_array[:, desc])) ** 2,
                           metal_component_array),
                    np.prod(metal_to_xenonpy_array[:, desc] ** metal_component_array) ** (
                                1 / sum(metal_component_array)),
                    sum(metal_component_array) / sum((1 / metal_to_xenonpy_array[:, desc]) * metal_component_array),
                    max(metal_to_xenonpy_array[:, desc]),
                    min(metal_to_xenonpy_array[:, desc])
                ])
                x_metaldesc_array = pd.Series(x_metaldesc_array,
                                              index=[f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}',
                                                     f'hmean_{d_name}',
                                                     f'max_{d_name}', f'min_{d_name}'])
                x_metaldesc.loc[
                    i, [f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'max_{d_name}',
                        f'min_{d_name}']] = x_metaldesc_array
            x_metaldesc = x_metaldesc.replace([np.inf, -np.inf], np.nan)
            last_support = [i for i in inverse_df.loc[idx:idx + 1, :].columns.tolist() if 'support_' in i][-1]
            idx_of_first_metaldesc = inverse_df.loc[idx:idx + 1, :].columns.tolist().index(last_support) + 1
            inverse_df.loc[idx, inverse_df.loc[idx:idx + 1, :].columns[idx_of_first_metaldesc:]] = \
                x_metaldesc[inverse_df.loc[idx:idx + 1, :].columns[idx_of_first_metaldesc:]].values.astype(float)
        return

    def get_x10_(self, inverse_df, metal_list, metal_component_list):
        for idx in range(len(inverse_df)):
            # xenonpy記述子の計算
            metal_to_xenonpy_array = np.zeros((len(metal_list[idx]), self.xenonpy_element_data.shape[1]))
            for i, metal in enumerate(metal_list[idx]):
                # 金属種データの取得
                metal_to_xenonpy_array[i, :] = self.xenonpy_element_data.loc[metal, :].values
            metal_component_array = np.array(metal_component_list[idx])
            x_metaldesc = pd.DataFrame(columns=self.xenonpy_col_name)
            for desc in range(self.xenonpy_element_data.shape[1]):
                d_name = self.xenonpy_element_data.columns[desc]

                if np.isnan(metal_to_xenonpy_array[:, desc]).any():
                    x_metaldesc.loc[i, f'ave_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'var_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'max_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'min_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'subtr_{d_name}'] = np.nan
                    x_metaldesc.loc[i, f'div_{d_name}'] = np.nan
                    continue
                x_metaldesc_array = np.array([
                    np.dot(metal_to_xenonpy_array[:, desc], metal_component_array) / np.sum(metal_component_array),
                    np.dot((metal_to_xenonpy_array[:, desc] - np.average(metal_to_xenonpy_array[:, desc])) ** 2,
                           metal_component_array),
                    max(metal_to_xenonpy_array[:, desc]),
                    min(metal_to_xenonpy_array[:, desc]),
                    max(metal_to_xenonpy_array[:, desc]) - min(metal_to_xenonpy_array[:, desc]),
                    max(metal_to_xenonpy_array[:, desc]) / min(metal_to_xenonpy_array[:, desc])
                ])
                x_metaldesc_array = pd.Series(x_metaldesc_array,
                                              index=[f'ave_{d_name}', f'var_{d_name}', f'max_{d_name}', f'min_{d_name}',
                                                     f'subtr_{d_name}', f'div_{d_name}'])
                x_metaldesc.loc[
                    i, [f'ave_{d_name}', f'var_{d_name}', f'gmean_{d_name}', f'hmean_{d_name}', f'subtr_{d_name}',
                        f'div_{d_name}']] = x_metaldesc_array
            x_metaldesc = x_metaldesc.replace([np.inf, -np.inf], np.nan)
            # inverse_df[x_metaldesc.columns] = x_metaldesc.values
            idx_of_first_metaldesc = inverse_df.loc[idx:idx + 1, :].columns.tolist().index(x_metaldesc.columns[0])
            inverse_df.loc[idx, inverse_df.loc[idx:idx + 1, :].columns[idx_of_first_metaldesc:]] = \
                x_metaldesc[inverse_df.loc[idx:idx + 1, :].columns[idx_of_first_metaldesc:]].values.astype(float)

    def get_x22_(self, inverse_df, metal_list, metal_component_list):
        inverse_metal_x_data = pd.DataFrame(columns=['chemicalFormula'], index=range(len(inverse_df)))
        for idx in range(len(inverse_df)):
            # chemicalFormulaの作成
            chemicalFormula = ''
            for metal, ratio in zip(metal_list[idx], metal_component_list[idx]):
                chemicalFormula += metal + str(ratio)
            inverse_metal_x_data.loc[idx, 'chemicalFormula'] = chemicalFormula.rstrip('nan')
        inverse_metal_x_data = StrToComposition(target_col_id='composition'). \
            featurize_dataframe(inverse_metal_x_data, "chemicalFormula", ignore_errors=True)
        # featurizerの指定　１つであればfeaturize_dataframeを使用
        n_jobs = 1
        f_Miedema = Miedema()
        f_Miedema.set_n_jobs(n_jobs)
        f_Meredig = Meredig()
        f_Meredig.set_n_jobs(n_jobs)
        f_BandCenter = BandCenter()
        f_BandCenter.set_n_jobs(n_jobs)
        f_WenAlloys = WenAlloys()
        f_WenAlloys.set_n_jobs(n_jobs)
        f_AtomicOrbitals = AtomicOrbitals()
        f_AtomicOrbitals.set_n_jobs(n_jobs)
        # f_Miedema = Miedema()
        # f_Meredig = Meredig()
        # f_BandCenter = BandCenter()
        # f_WenAlloys = WenAlloys()
        # f_AtomicOrbitals = AtomicOrbitals()
        # WenAlloysを使用すると原因不明エラー発生 ignore_errors=Trueが必要
        features_Miedema = f_Miedema.featurize_dataframe(inverse_metal_x_data, col_id='composition')
        features_Meredig = f_Meredig.featurize_dataframe(inverse_metal_x_data, col_id='composition')
        features_BandCenter = f_BandCenter.featurize_dataframe(inverse_metal_x_data, col_id='composition')
        features_WenAlloys = f_WenAlloys.featurize_dataframe(inverse_metal_x_data, col_id='composition',
                                                             ignore_errors=True)
        features_AtomicOrbitals = f_AtomicOrbitals.featurize_dataframe(inverse_metal_x_data, col_id='composition',
                                                                       ignore_errors=True)

        inverse_metal_x_data_2 = features_Miedema.drop(['chemicalFormula', 'composition'], axis=1)
        inverse_metal_x_data_4 = features_Meredig.iloc[:, -17:]
        inverse_metal_x_data_5 = features_BandCenter.iloc[:, -1:]
        inverse_metal_x_data_7 = features_WenAlloys.iloc[:, -23:]
        inverse_metal_x_data_8 = features_AtomicOrbitals.iloc[:, -5:]
        inverse_metal_x_data_8.drop(['LUMO_character', 'LUMO_element'], axis=1, inplace=True)  # AtomicOrbitals用
        inverse_metal_x_data_1 = pd.concat(
            [inverse_metal_x_data_2, inverse_metal_x_data_4, inverse_metal_x_data_5, inverse_metal_x_data_7,
             inverse_metal_x_data_8], axis=1)
        inverse_metal_x_data_1.drop(
            ['frac s valence electrons', 'frac p valence electrons', 'frac d valence electrons',
             'frac f valence electrons'], axis=1, inplace=True)

        if inverse_metal_x_data_1.shape[0] > 1:
            # iGMR, sklearnの種々のImputerを試してみる。
            df_for_imputer = inverse_metal_x_data_1.iloc[:, 10:]
            outlier = (df_for_imputer > 2e2) | (df_for_imputer < -1e6)
            outlier_columns = outlier.sum()
            df_for_imputer.mask(outlier, np.nan, inplace=True)

            # iterativeimputerを試したが、まともな値を返してこない
            # 異常値があることが原因であるため、異常値を排除する必要がある
            # デフォルトのestimatorはBaysianRidge

            imputer = IterativeImputer(max_iter=10, random_state=10)
            try:
                imputed_df = imputer.fit_transform(df_for_imputer)
            except ValueError:
                # 異常値(NaN)があることが原因、元に戻す
                imputed_df = df_for_imputer.fillna(value=df_for_imputer.mean())
            if imputed_df.shape[1] != df_for_imputer.shape[1]:
                # shapeの形が変わってしまっているので、元に戻す
                imputed_df = df_for_imputer.fillna(0)

            imputed_df = pd.DataFrame(imputed_df, index=inverse_metal_x_data_1.index)
            outlier_after_impute = (imputed_df > 2e2) | (imputed_df < -1e6)
            outlier_columns_after_impute = outlier_after_impute.sum()
            # impute後の処理
            imputed_df = pd.concat([inverse_metal_x_data_1.iloc[:, 0:10], imputed_df], axis=1)
            # imputed_df.insert(loc= 4, column= '触媒ロット', value= inverse_metal_x_data_1['触媒ロット'])
            imputed_df.columns = inverse_metal_x_data_1.columns
            inverse_metal_x_data = imputed_df
            inverse_metal_x_data = inverse_metal_x_data_1
            inverse_df[inverse_metal_x_data.columns] = inverse_metal_x_data
        else:
            inverse_metal_x_data = inverse_metal_x_data_1.fillna(0)
            inverse_df[inverse_metal_x_data.columns] = inverse_metal_x_data_1
        return

