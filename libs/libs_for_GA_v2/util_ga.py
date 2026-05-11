import random
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import norm

np.random.seed(1)
random.seed(1)

# 元素(金属塩)束縛条件データ取得
METAL_SALT_CONSTRAINTS_DATA = pd.read_csv('results/20230316受領_住友化学用金属塩リスト.csv', encoding="shift-jis")


class GAutility:
    def __init__(self):
        pass

    @staticmethod
    def get_atomListN(metal_salt_constraints_data):

        # nanをNに変換
        metal_salt_constraints_data = metal_salt_constraints_data.fillna({"Omit": 'N'})
        # 金属塩リスト（omit列が空欄のもの）
        metal_salt_constraints_data_N = metal_salt_constraints_data
        metal_salt_constraints_data_N = metal_salt_constraints_data_N[metal_salt_constraints_data_N['Omit'] == 'N']
        atom_list_N = list(metal_salt_constraints_data_N['元素'])

        return atom_list_N

    @staticmethod
    def get_atomListY(metal_salt_constraints_data):
        # 金属塩リスト（omit列がYのもの）
        metal_salt_constraints_data_Y = metal_salt_constraints_data
        metal_salt_constraints_data_Y = metal_salt_constraints_data_Y[metal_salt_constraints_data_Y['Omit'] == 'Y']
        atom_list_Y = list(metal_salt_constraints_data_Y['元素'])

        return atom_list_Y

    def get_ratioListConstraints(self, metal_list, metal_component_list):
        now_ratio_list_N = []  # ratioリスト：N用
        now_ratio_list_Y = []  # ratioリスト：Y用
        Compo_ratio_N = 0.7
        Compo_ratio_Y = 0.3
        # 元素(金属塩)束縛条件データ取得
        org_atom_list_N = self.get_atomListN(METAL_SALT_CONSTRAINTS_DATA)

        # Omit'N'と'Y'に分解
        for i, metal in enumerate(metal_list):
            if metal in org_atom_list_N:
                now_ratio_list_N.append(metal_component_list[i])
            else:
                now_ratio_list_Y.append(metal_component_list[i])
                # 　それぞれで100%に換算
        now_ratio_list_N = [round((x / sum(now_ratio_list_N)) * Compo_ratio_N, 2) for x in now_ratio_list_N]
        now_ratio_list_Y = [round((x / sum(now_ratio_list_Y)) * Compo_ratio_Y, 2) for x in now_ratio_list_Y]

        # 分解したものを戻す
        now_ratio_list = now_ratio_list_N + now_ratio_list_Y

        return now_ratio_list

    def get_ratioListConstraints_v2(self, metal_list, metal_component_list, step=0.01):
        now_ratio_list_N = []  # ratioリスト：N用
        now_ratio_list_Y = []  # ratioリスト：Y用
        Compo_ratio_N = 0.7
        Compo_ratio_Y = 0.3
        # 元素(金属塩)束縛条件データ取得
        org_atom_list_N = self.get_atomListN(METAL_SALT_CONSTRAINTS_DATA)
        # Omit'N'と'Y'に分解
        n_y_list = []
        for i, metal in enumerate(metal_list):
            if metal in org_atom_list_N:
                n_y_list.append('N')
                now_ratio_list_N.append(metal_component_list[i])
            else:
                n_y_list.append('Y')
                now_ratio_list_Y.append(metal_component_list[i])
        # 　それぞれで100%に換算
        if step == 0.1:
            now_ratio_list_N = roundNumbers(now_ratio_list_N, 1, Compo_ratio_N)
            now_ratio_list_Y = roundNumbers(now_ratio_list_Y, 1, Compo_ratio_Y)
        else:
            now_ratio_list_N = roundNumbers(now_ratio_list_N, 2, Compo_ratio_N)
            now_ratio_list_Y = roundNumbers(now_ratio_list_Y, 2, Compo_ratio_Y)
        # 分解したものを戻す
        now_ratio_list = []
        idx_N = 0
        idx_Y = 0
        for y_n in n_y_list:
            if y_n == 'N':
                now_ratio_list.append(now_ratio_list_N[idx_N])
                idx_N += 1
            else:
                now_ratio_list.append(now_ratio_list_Y[idx_Y])
                idx_Y += 1
        return now_ratio_list

    @staticmethod
    def convertNaN(metal_df_all):
        metal_df_all = metal_df_all.reset_index(drop=True)
        for i in metal_df_all.index:
            if (metal_df_all.loc[i, 'metal5'] == 'nan'):    metal_df_all.loc[i, 'metal5'] = ''
            if (metal_df_all.loc[i, 'metal4'] == 'nan'):    metal_df_all.loc[i, 'metal4'] = ''
            if (metal_df_all.loc[i, 'metal3'] == 'nan'):    metal_df_all.loc[i, 'metal3'] = ''
            if (metal_df_all.loc[i, 'ratio5'] == 'nan'):    metal_df_all.loc[i, 'ratio5'] = ''
            if (metal_df_all.loc[i, 'ratio4'] == 'nan'):    metal_df_all.loc[i, 'ratio4'] = ''
            if (metal_df_all.loc[i, 'ratio3'] == 'nan'):    metal_df_all.loc[i, 'ratio3'] = ''
        return metal_df_all

    def get_xenonpyList(self, xenonpy_element_data):
        weighted_average_name = list()  # 加重平均の index 名
        weighted_variance_name = list()  # 加重分散の index 名
        geometric_mean_name = list()  # 幾何平均の index 名
        harmonic_mean_name = list()  # 調和平均の index 名
        max_pooling_name = list()  # 最大値の index 名
        min_pooling_name = list()  # 最小値の index 名
        # 名前のリスト
        for j in xenonpy_element_data.columns:
            weighted_average_name.append(f'ave_{j}')
            weighted_variance_name.append(f'var_{j}')
            geometric_mean_name.append(f'gmean_{j}')
            harmonic_mean_name.append(f'hmean_{j}')
            max_pooling_name.append(f'max_{j}')
            min_pooling_name.append(f'min_{j}')
        xenonpy_col_name = weighted_average_name + weighted_variance_name + geometric_mean_name + harmonic_mean_name + max_pooling_name + min_pooling_name
        return xenonpy_col_name


def randomNumbers(size: int, round_num: float, sumvalue: float) -> list:
    u = int(100 * pow(10, round_num))
    nums = []
    for _ in range(size):
        n = np.random.randint(1, high=u + 1, size=1)
        nums.append(n[0])
    out = []
    for num in nums:
        out.append(round(num * sumvalue / sum(nums), round_num))
    maxval = max(out)
    index = out.index(maxval)
    out[index] = round(sumvalue - (sum(out) - maxval), round_num)
    if min(out) <= 0:
        # print('recalc')
        out = randomNumbers(size, round_num, sumvalue)
    return out


def roundNumbers(nums: list, round_num: float, sumvalue: float) -> list:
    u = int(100 * pow(10, round_num))
    out = []
    for num in nums:
        out.append(round(num * sumvalue / sum(nums), round_num))
    maxval = max(out)
    index = out.index(maxval)
    out[index] = round(sumvalue - (sum(out) - maxval), round_num)
    if min(out) <= 0:
        minval = min(out)
        minindex = out.index(minval)
        out[minindex] = 1 / 10 ** round_num
        out[index] = round(sumvalue - (sum(out) - maxval), round_num)
    return out


def checkMetalRatioTotal(ratio_list, step):
    new_ratio_list = ratio_list.copy()
    if 0.0 in new_ratio_list:
        new_ratio_list = replace_zero(new_ratio_list, step)
    if step == 0.1:
        decimals_val = 1
    else:
        decimals_val = 2
    # 「1.00」にならない項目を検知、差分を保持
    sum_val = sum(new_ratio_list)
    sum_diff = np.round(1.00 - sum_val, decimals=decimals_val)
    # 最大値のcolumnに差分を加算する。→「1.00」になる
    max_col = np.argmax(new_ratio_list)
    new_ratio_list[max_col] = np.round(new_ratio_list[max_col] + sum_diff, decimals=decimals_val)
    return new_ratio_list


def drop_same_values(x_data, threshold_of_rate_of_same_value=0.95):
    rate_of_same_value = list()
    for X_variable_name in x_data.columns:
        same_value_number = x_data[X_variable_name].value_counts()
        rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_data.shape[0]))
    deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
    x_data = x_data.drop(x_data.columns[deleting_variable_numbers], axis=1)

    return x_data


def cat_split(x_data, y_data, cat_name, fold_cat_outer_cv):
    x_outer = x_data.loc[cat_name == fold_cat_outer_cv, :]
    y_outer = y_data[cat_name == fold_cat_outer_cv]
    x_inner = x_data[cat_name != fold_cat_outer_cv]
    y_inner = y_data[cat_name != fold_cat_outer_cv]

    return x_inner, x_outer, y_inner, y_outer


def cat_split_for_fold(x_data, y_data, cat_name, fold_cat_outer_cv_list):
    x_outer = y_outer = pd.DataFrame()
    for each in fold_cat_outer_cv_list:
        x_outer = pd.concat([x_outer, x_data[cat_name == each]], axis=0)
        y_outer = pd.concat([y_outer, y_data[cat_name == each]], axis=0)
    x_inner = x_data[[False if i in x_outer.index else True for i in x_data.index]]
    y_inner = y_data[[False if i in y_outer.index else True for i in y_data.index]]
    return x_inner, x_outer, y_inner, y_outer


def autoscaling(data, base_data):
    autoscaled_data = (data - base_data.mean(axis=0)) / base_data.std(axis=0, ddof=1)
    return autoscaled_data


def rescaling(data, base_data):
    rescaled_data = data * base_data.std(axis=0, ddof=1) + base_data.mean(axis=0)
    return rescaled_data


def log_transform(arg_data):
    data = arg_data.copy()
    data[data == 0] = (data[data != 0].nsmallest(1) / 2).iloc[0]
    data = logit((data / 100).values)
    return data


def log_inverse_transform(data):
    data = expit(data) * 100
    return data


# target: (ターゲットタイプ, ターゲット値) 　'maximum': 最大化...リミット値以上, 'minimum':　最小化... リミット値以下　
# estimated: 目的変数推定値
# estimated_std: 目的変数標準偏差
def calcPTR(target: tuple, estimated: np.array, estimated_std: np.array):
    target_type = target[0]
    target_score = target[1]
    if target_type == 'minimum':
        probability = norm.cdf(x=target_score, loc=estimated, scale=estimated_std)
    elif target_type == 'maximum':
        probability = 1 - norm.cdf(x=target_score, loc=estimated, scale=estimated_std)
    for i, prob in enumerate(probability):
        if prob == 0:
            probability[i] = pow(10, -100)
    return probability


# target: (ターゲットタイプ, ターゲット値)   'maximum': 最大化...リミット値以上, 'minimum':　最小化... リミット値以下
# actual: 目的変数実験値
# estimated: 目的変数推定値
# estimated_std: 目的変数標準偏差
# relaxation
def calcPI(target: tuple, actual: np.array, estimated: np.array, estimated_std: np.array, relaxation: float = 0.01):
    target_type = target[0]
    if target_type == 'minimum':
        min_value = actual.min() - relaxation
        probability = norm.cdf(x=min_value, loc=estimated, scale=estimated_std)
    elif target_type == 'maximum':
        max_value = actual.max() + relaxation
        probability = 1 - norm.cdf(x=max_value, loc=estimated, scale=estimated_std)
    for i, prob in enumerate(probability):
        if prob == 0:
            probability[i] = pow(10, -100)
    return probability


# target: (ターゲット値, ターゲットタイプ)　 'maximum': 最大化...リミット値以上, 'minimum':　最小化... リミット値以下
# estimated: 目的変数推定値
# estimated_std: 目的変数標準偏差
# cumulative_variance
# alpha
def calcMI(target: tuple, estimated: np.array, estimated_std: np.array, cumulative_variance: np.array,
           alpha: float = np.log(2 / (10 ** -6))):
    target_type = target[0]
    if target_type == 'minimum':
        estimated *= -1
    acquisition_function_values = estimated + \
                                  alpha ** 0.5 * ((estimated_std ** 2 + cumulative_variance) ** 0.5 -
                                                  cumulative_variance ** 0.5)
    new_cumulative_variance = cumulative_variance + np.mean(estimated_std ** 2)
    return acquisition_function_values, new_cumulative_variance


# target: (ターゲット値, ターゲットタイプ)  'maximum': 最大化...リミット値以上, 'minimum':　最小化... リミット値以下
# actual: 目的変数実験値
# estimated: 目的変数推定値
# estimated_std: 目的変数標準偏差
# relaxation
def calcEI(target: tuple, actual: np.array, estimated: np.array, estimated_std: np.array, relaxation: float = 0.01):
    target_type = target[0]
    if target_type == 'minimum':
        imp = ((actual.min() - relaxation) - estimated)
    elif target_type == 'maximum':
        imp = (estimated - (actual.max() + relaxation))
    z = imp / estimated_std
    acquisition_function_values = imp * norm.cdf(z) + estimated_std * norm.pdf(z)
    return acquisition_function_values


# リストに0.0が含まれていると0.1、または、0.01に置き換える
def replace_zero(list_a, step):
    if 0.0 not in list_a:
        return list_a
    list_b = list_a.copy()
    zero_index = list_b.index(0.0)
    if step == 0.1:
        list_b[zero_index] = 0.1
    else:
        list_b[zero_index] = 0.01
    return list_b