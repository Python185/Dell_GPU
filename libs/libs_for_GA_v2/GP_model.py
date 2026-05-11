from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, DotProduct, Matern
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import os

from scipy.stats import norm
from scipy.special import logit, expit


import matplotlib
import matplotlib.pyplot as plt

from .util_ga import log_inverse_transform, log_transform, rescaling, autoscaling, cat_split_for_fold, \
    drop_same_values, calcPTR, calcPI, calcMI, calcEI

matplotlib.rc('font', family='Meiryo')


class GP_model_process():
    def __init__(self, model_method, y_name, frag_log_transform):

        self.model_method = model_method
        self.y_name = y_name
        self.frag_log_transform = frag_log_transform
        self.raw_x_data = None
        self.raw_y_data = None
        self.cat_name = None
        self.log_y_data = None
        self.x_data = None
        self.predicted_y_values = None
        self.r2_score_list = []

    def data_settings(self, raw_x_data, raw_y_data, cat_name):
        self.raw_x_data = raw_x_data
        self.raw_y_data = raw_y_data
        self.cat_name = cat_name

    def data_preprocess(self, threshold_of_rate_of_same_value):
        if self.frag_log_transform:
            self.log_y_data = log_transform(self.raw_y_data)
            self.log_y_data = pd.Series(self.log_y_data, index=self.raw_y_data.index, name=self.y_name)
        else:
            self.log_y_data = self.raw_y_data
        self.x_data = drop_same_values(self.raw_x_data, threshold_of_rate_of_same_value)
        self.x_data = self.x_data.drop(columns=[i for i in self.x_data.columns if '担体種類' in i])
        return self.x_data

    def set_kernels(self, x_data):
        bo_kernels = [
            ConstantKernel() * DotProduct() + WhiteKernel(),
            ConstantKernel() * RBF() + WhiteKernel(),
            ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * RBF(np.ones(x_data.shape[1])) + WhiteKernel(),
            ConstantKernel() * RBF(np.ones(x_data.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
            ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
            ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()
        ]
        return bo_kernels

    def model_selection(self, fixed_best_model=None):
        outer_fold_number = 10  # 外側の分割数。触媒が同じで評価方法が異なるサンプルをひとまとまりとして分割
        random_state = 99  # 分割する際の乱数のシード。固定すれば再現性あり
        predicted_y_values = np.zeros([self.log_y_data.shape[0], len(self.model_method)])
        predicted_y_values = pd.DataFrame(predicted_y_values, index=self.log_y_data.index,
                                               columns=self.model_method)
        unique_groups = np.array(list(set(self.cat_name)))
        unique_groups.sort()
        kf = KFold(n_splits=outer_fold_number, shuffle=True, random_state=random_state)
        for fold_i, (train_group_idx, test_group_idx) in enumerate(kf.split(unique_groups)):
            test_group_cat = [unique_groups[i] for i in test_group_idx]
            x_inner, x_outer, log_y_inner, log_y_outer = \
                cat_split_for_fold(self.x_data, self.log_y_data, self.cat_name, test_group_cat)
            # x_inner, x_outer, log_y_inner, log_y_outer = self.cat_split(self.x_data, self.log_y_data, self.cat_name, fold_cat_outer_cv)
            x_outer = x_outer.drop(x_outer.columns[np.where(x_inner.var() == 0)], axis=1)
            x_inner = x_inner.drop(x_inner.columns[np.where(x_inner.var() == 0)], axis=1)

            autoscaled_x_inner = autoscaling(x_inner, x_inner)
            autoscaled_log_y_inner = autoscaling(log_y_inner, log_y_inner)
            autoscaled_x_outer = autoscaling(x_outer, x_inner)

            # GPによる解析
            bo_kernels = self.set_kernels(autoscaled_x_inner)

            for i, kernel in enumerate(bo_kernels):
                model_name = f'GPR_{i}'
                if fixed_best_model and model_name != fixed_best_model:
                    continue
                print(f'\r\tfold:{fold_i + 1}/{outer_fold_number}, {model_name} ', end='')
                regression_model = GaussianProcessRegressor(kernel=kernel, alpha=0)  # GPR モデルの宣言
                regression_model.fit(autoscaled_x_inner, autoscaled_log_y_inner)
                estimated_log_y_test = regression_model.predict(autoscaled_x_outer)
                estimated_log_y_test = rescaling(estimated_log_y_test, log_y_inner)

                if self.frag_log_transform:
                    estimated_y_test = log_inverse_transform(estimated_log_y_test)
                else:
                    estimated_y_test = estimated_log_y_test.copy()
                predicted_y_values.loc[log_y_outer.index, model_name] = estimated_y_test
        print()
        self.predicted_y_values = predicted_y_values
        return self.predicted_y_values

    def calc_r2score(self):
        for i in range(self.predicted_y_values.shape[1]):

            self.r2_score_list.append(metrics.r2_score(self.raw_y_data, self.predicted_y_values.iloc[:, i]))
        return self.r2_score_list

    def best_model_construction(self, savefig_path=None, output_plt=True):
        self.autoscaled_x_data = autoscaling(self.x_data, self.x_data)
        autoscaled_y_data = autoscaling(self.log_y_data, self.log_y_data)

        bo_kernels = self.set_kernels(self.autoscaled_x_data)

        best_kernel = bo_kernels[self.r2_score_list.index(max(self.r2_score_list))]
        best_kernel_df = pd.DataFrame(self.r2_score_list, columns=['r2'],
                                      index=[f'GPR_{i}' for i in range(len(self.r2_score_list))])
        save_r2_path = os.path.join(os.path.dirname(savefig_path),
                                    os.path.splitext(os.path.basename(savefig_path))[0] + '.xlsx')
        best_kernel_df.to_excel(save_r2_path)
        best_model_name = f'GPR_{self.r2_score_list.index(max(self.r2_score_list))}'
        regression_model = GaussianProcessRegressor(kernel=best_kernel, alpha=0)  # GPR モデルの宣言
        regression_model.fit(self.autoscaled_x_data, autoscaled_y_data)

        estimated_log_y = regression_model.predict(self.autoscaled_x_data)
        estimated_log_y = rescaling(estimated_log_y, self.log_y_data)
        if self.frag_log_transform:
            estimated_y = log_inverse_transform(estimated_log_y)
        else:
            estimated_y = estimated_log_y.copy()

        if output_plt:
            plt.rcParams['font.size'] = 24
            plt.figure(figsize=(6, 6))
            plt.scatter(self.raw_y_data, estimated_y, c='blue', alpha=0.7, edgecolors='black')  # 実測値 vs. 推定値プロット
            y_max = max(self.raw_y_data.max(), estimated_y.max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
            y_min = min(self.raw_y_data.min(), estimated_y.min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
            plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                     [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                     'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
            plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
            plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
            plt.xlabel('actual y')  # x 軸の名前
            plt.ylabel('estimated y')  # y 軸の名前
            plt.savefig(savefig_path, bbox_inches='tight')
            # plt.show()  # 以上の設定で描画

        return regression_model, best_model_name


class GpInverseProcess():
    def __init__(self, frag_log_transform, target_y_dict):
        self.frag_log_transform = frag_log_transform
        self.target_y_dict = target_y_dict

    def setting_data(self, inverse_x, base_x):
        self.base_x = base_x
        self.autoscaled_inverse_x = autoscaling(inverse_x, base_x)

    def setting_acquisition_function(self, acquisition_function, cumulative_variance, relaxation_value, alpha):
        self.acquisition_function = acquisition_function
        self.cumulative_variance = cumulative_variance
        self.relaxation_value = relaxation_value
        self.alpha = alpha

    def calc_probability(self, regression_model, base_y, y_name, y_name_idx):
        if self.frag_log_transform:
            log_y_data = log_transform(base_y)
            log_y_data = pd.Series(log_y_data, index=base_y.index, name=y_name)
        else:
            log_y_data = base_y.copy()

        autoscaled_estimated_inverse_y, autoscaled_estimated_inverse_y_std = regression_model.predict(
            self.autoscaled_inverse_x, return_std=True)
        estimated_inverse_log_y = rescaling(autoscaled_estimated_inverse_y, log_y_data)
        estimated_inverse_log_y_std = autoscaled_estimated_inverse_y_std * log_y_data.std(axis=0, ddof=1)
        if self.frag_log_transform:
            estimated_inverse_y = log_inverse_transform(estimated_inverse_log_y)
            estimated_inverse_y_std = expit(estimated_inverse_log_y_std)
        else:
            estimated_inverse_y = estimated_inverse_log_y
            estimated_inverse_y_std = estimated_inverse_log_y_std

        # 評価関数（目標達成の確率を計算）
        target_objective = self.target_y_dict[y_name]['objective']
        target_y_score = self.target_y_dict[y_name]['score']
        target = (target_objective, target_y_score)

        if self.acquisition_function == 'PTR':
            probability = calcPTR(target, estimated_inverse_y, estimated_inverse_y_std)
            probability = probability.reshape((-1, 1))

        elif self.acquisition_function == 'PI':
            probability = calcPI(target, base_y, estimated_inverse_y,
                                 estimated_inverse_y_std, relaxation=self.relaxation_value)
            probability = probability.reshape((-1, 1))

        elif self.acquisition_function == 'MI':
            acquisition_function_values, cumulative_variance = \
                calcMI(target, estimated_inverse_y, estimated_inverse_y_std,
                       self.cumulative_variance[:, y_name_idx],  self.alpha)
            probability = acquisition_function_values
            self.cumulative_variance[:, y_name_idx] = cumulative_variance

        elif self.acquisition_function == 'EI':
            acquisition_function_values = calcEI(target, base_y, estimated_inverse_y, estimated_inverse_y_std, self.relaxation_value)
            probability = acquisition_function_values

        return probability, estimated_inverse_y, estimated_inverse_y_std
