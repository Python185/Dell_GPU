import numpy as np
import pandas as pd

from scipy.special import logit, expit

class GPpreprocess:
    def __init__(self):
        pass

    def get_xenonpyList(self, xenonpy_element_data):
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
        
        xenonpy_col_name = weighted_average_name+weighted_variance_name+geometric_mean_name+harmonic_mean_name+max_pooling_name+min_pooling_name
        return xenonpy_col_name
    
    def drop_same_values(self, x_data, threshold_of_rate_of_same_value=0.95):
        rate_of_same_value = list()
        for X_variable_name in x_data.columns:
            same_value_number = x_data[X_variable_name].value_counts()
            rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_data.shape[0]))
        deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
        x_data = x_data.drop(x_data.columns[deleting_variable_numbers], axis=1)

        return x_data

    def cat_split(self, x_data, y_data, cat_name, fold_cat_outer_cv):
        x_outer = x_data.loc[cat_name == fold_cat_outer_cv, :]
        y_outer = y_data[cat_name == fold_cat_outer_cv]
        x_inner = x_data[cat_name != fold_cat_outer_cv]
        y_inner = y_data[cat_name != fold_cat_outer_cv]

        return x_inner, x_outer, y_inner, y_outer

    def cat_split_for_fold(self, x_data, y_data, cat_name, fold_cat_outer_cv_list):
        x_outer = y_outer = pd.DataFrame()
        for each in fold_cat_outer_cv_list:
            x_outer = pd.concat([x_outer, x_data[cat_name == each]], axis=0)
            y_outer = pd.concat([y_outer, y_data[cat_name == each]], axis=0)
        x_inner = x_data[[False if i in x_outer.index else True for i in x_data.index]]
        y_inner = y_data[[False if i in y_outer.index else True for i in y_data.index]]
        return x_inner, x_outer, y_inner, y_outer
    
    def autoscaling(self, data, base_data):
        autoscaled_data = (data - base_data.mean(axis=0)) / base_data.std(axis=0, ddof=1)
        return autoscaled_data
    
    def rescaling(self, data, base_data):
        rescaled_data = data * base_data.std(axis=0, ddof=1) + base_data.mean(axis=0)
        return rescaled_data
    
    def log_transform(self, arg_data):
        data = arg_data.copy()
        data[data == 0] = (data[data != 0].nsmallest(1)/2).iloc[0]
        data = logit((data/100).values)
        return data
    
    def log_inverse_transform(self, data):
        data = expit(data) * 100
        return data