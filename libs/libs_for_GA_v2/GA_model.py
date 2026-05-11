import numpy as np
import pandas as pd
import random

random.seed(1)
import csv
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from operator import attrgetter

from deap import base
from deap import creator
from deap import tools

from .preprocess_for_ga import GApreprocess
from .util_ga import GAutility, checkMetalRatioTotal, log_inverse_transform, log_transform, rescaling, autoscaling, \
    calcPTR, calcPI, calcMI, calcEI, replace_zero

ga_util = GAutility()

Tournament_size = 4

class GA_model_process(GApreprocess):
    def __init__(self, y_names, datasets, regression_model_dict, target_y_dict, condition_dict, num_metal, step,
                 constraints, x_names, debugDir):
        self.y_names = y_names
        self.datasets = datasets
        self.regression_model_dict = regression_model_dict
        self.target_y_dict = target_y_dict
        self.condition_dict = condition_dict
        self.num_metal = num_metal
        self.step = step
        self.omit_constraints = constraints['Omit']
        self.x_names = x_names
        self.debugDir = debugDir
        # 出力するxの列名を作成
        self.output_clm = {}
        for each in datasets.keys():
            self.output_clm[each] = datasets[each].columns.to_list()
            if '触媒ロット' in self.output_clm[each]:
                self.output_clm[each].remove('触媒ロット')
        # 組成の記述子を計算
        xenonpy_element_data = pd.read_csv('results/xenonpy_element_data.csv', index_col=0)
        xenonpy_col_name = ga_util.get_xenonpyList(xenonpy_element_data)
        self.xenonpy_element_data = xenonpy_element_data
        self.xenonpy_col_name = xenonpy_col_name
        # 獲得関数関連
        self.acquisition_function = None
        self.cumulative_variance = None
        self.relaxation_value = None
        self.alpha = None
        # toolbox
        self.toolbox = None
        # x_datas
        self.x_datas = None

    def setting_acquisition_function(self, acquisition_function, cumulative_variance, relaxation_value, alpha):
        self.acquisition_function = acquisition_function
        self.cumulative_variance = cumulative_variance
        self.relaxation_value = relaxation_value
        self.alpha = alpha

    def setting_ga_toolbox(self, boundary_dict):
        # 適応度クラスの作成
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # 個体クラスの作成
        creator.create("Individual", list, fitness=creator.FitnessMax)
        # Toolboxの作成
        self.toolbox = base.Toolbox()
        # 遺伝子を生成する関数"attr_gene"を登録
        self.toolbox.register("attr_gene", random.randint, 0, 99)
        len_ind = list(boundary_dict.keys()).index('metal1')
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_gene, len_ind)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        # 遺伝的操作の設定
        self.toolbox.register('evaluate', self.evaluate)  # 評価関数の設定
        self.toolbox.register('mate', tools.cxTwoPoint)  # 交叉の設定
        self.toolbox.register('mutate', tools.mutUniformInt, low=0, up=99, indpb=0.3)  # 突然変異の設定
        # 個体選択法"select"を登録
        self.toolbox.register("select", self.Tournament, tournsize=Tournament_size)

    def generation_process(self, x_datas, frag_log_transform, number_of_population, number_of_generation,
                           probability_of_crossover, probability_of_mutation, iter_n):
        self.x_datas = x_datas
        self.frag_log_transform = frag_log_transform
        # 個体集団の生成
        pop = self.toolbox.population(n=number_of_population)
        pop_metal = self.create_pop_metal(number_of_population)
        print("Start of evolution")

        # 個体集団の適応度の評価
        fitnesses = self.toolbox.evaluate(pop, pop_metal)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        print("  Evaluated %i individuals" % len(pop))

        # 適応度の抽出
        fits = [ind.fitness.values[0] for ind in pop]

        # 進化ループ開始
        df_fitness = pd.DataFrame(np.zeros((number_of_population, number_of_generation + 1)),
                                  columns=[f'{i}-gen' for i in range(number_of_generation + 1)])
        df_metal = pd.DataFrame([[[]] * (number_of_generation + 1)] * number_of_population,
                                columns=[f'{i}-gen' for i in range(number_of_generation + 1)])
        best_fit_list = []
        best_fit_list.append(sorted(fits, reverse=True)[0])
        df_fitness.iloc[:, 0] = sorted(fits, reverse=True)
        for i in range(number_of_population):
            df_metal.iat[i, 0] = pop_metal[i]
        for generation in range(number_of_generation):
            print('\n-- Generation {0} --'.format(generation + 1))

            # 次世代個体の選択・複製
            offspring, pop_metal_new = self.toolbox.select(pop, len(pop), pop_metal)
            offspring = list(map(self.toolbox.clone, offspring))
            pop_metal_new = [item.copy() for item in pop_metal_new]
            changed_individulas = set()
            # 交叉
            num_of_mate_offspring = 0
            for idx1, idx2 in zip(range(len(offspring))[::2], range(len(offspring))[1::2]):
                # 交叉させる個体を選択
                if random.random() < probability_of_crossover:
                    changed_individulas.add(idx1)
                    changed_individulas.add(idx2)
                    self.toolbox.mate(offspring[idx1], offspring[idx2])
                    num_of_mate_offspring += 2
                    # 交叉させた個体は適応度を削除する
                    del offspring[idx1].fitness.values
                    del offspring[idx2].fitness.values
            # 変異_offspring
            num_of_mutate_offspring = 0
            for idx in range(len(offspring)):
                # 変異させる個体を選択
                if random.random() < probability_of_mutation:
                    changed_individulas.add(idx)
                    self.toolbox.mutate(offspring[idx])
                    num_of_mutate_offspring += 1
                    # 変異させた個体は適応度を削除する
                    del offspring[idx].fitness.values
            # 変異_metal
            num_of_mutate_metal = 0
            for idx in range(len(pop_metal_new)):
                # 変異させる個体を選択
                if random.random() < probability_of_mutation:
                    changed_individulas.add(idx)
                    new_mutant = self.mutant_metal(pop_metal_new[idx], indpb=0.3)
                    pop_metal_new[idx] = new_mutant
                    num_of_mutate_metal += 1
            # 適応度を再評価する
            fitnesses = self.toolbox.evaluate(offspring, pop_metal_new)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            print(f"\r\t-- Generation {generation + 1} --  Evaluated {len(changed_individulas)} ,  "
                  f"Changed(mate):{num_of_mate_offspring}, Changed(mutate):{num_of_mutate_offspring}, "
                  f"Changed(metal):{num_of_mutate_metal}", end=' ')
            # 個体集団を新世代個体集団で更新
            pop[:] = offspring
            pop_metal[:] = pop_metal_new
            # 新世代の全個体の適応度の抽出
            fits = [ind.fitness.values[0] for ind in pop]
            best_fit_list.append(sorted(fits, reverse=True)[0])
            df_fitness.iloc[:, generation + 1] = sorted(fits, reverse=True)
            for i in range(number_of_population):
                df_metal.iat[i, generation + 1] = pop_metal[i]
            # n世代にわたりbest_fitの値(小数点以下5桁)が変化していないならば、ほほ収束したと判断し、ループから抜け出す
            convergence_generation = 500
            if len(best_fit_list) >= convergence_generation:
                if len(set(best_fit_list[-convergence_generation:])) == 1:
                    break
        print()
        best_fit_df = pd.DataFrame(best_fit_list, columns=['fitness'],
                                   index=[f"{i}世代" for i in range(len(best_fit_list))])
        best_fit_df.plot()
        savefig_path = f"{self.debugDir}fitness_progress_({iter_n}).png"
        plt.savefig(savefig_path, bbox_inches='tight')
        # plt.show()
        best_fit_df.to_excel(f'{self.debugDir}best_fitness_progress_({iter_n}).xlsx')
        df_fitness.to_excel(f'{self.debugDir}df_fitness_({iter_n}).xlsx')
        df_metal.to_excel(f'{self.debugDir}df_metal_({iter_n}).xlsx')
        return pop, pop_metal

    def select_best_ind(self, pop, pop_metal):
        # ベストの遺伝子を選択
        fitness = [each.fitness.values[0] for each in pop]
        best_ind_index = fitness.index(max(fitness))
        best_pop = np.array(pop[best_ind_index]).tolist()
        best_pop_metal = pop_metal[best_ind_index]
        # 個体の取得
        best_gene_metal_df = pd.DataFrame([best_pop + best_pop_metal], columns=self.condition_dict.keys())
        # 遺伝子をxに変換
        rtn_message = {}
        best_individual_x_df = {}
        for x_name in self.x_names:
            best_individual_x_df[x_name] = pd.DataFrame(np.zeros((1, len(self.output_clm[x_name]))),
                                                                         columns=self.output_clm[x_name])
            rtn_message[x_name], best_individual_x_df[x_name] = \
                self.ind_to_xdata(x_name, best_gene_metal_df, best_individual_x_df[x_name],
                                  self.condition_dict, self.step, self.omit_constraints, return_metal=False)
        # ベスト遺伝子の金属の組み合わせと組成を保存
        inverse_metal_list = best_pop_metal[::2]
        inverse_metal_component_list = best_pop_metal[1::2]
        inverse_metal_list = inverse_metal_list + [np.nan] * (5 - len(inverse_metal_list))
        inverse_metal_component_list = inverse_metal_component_list + [np.nan] * (5 - len(inverse_metal_component_list))
        best_individual_inverse_analysis_metal_df = pd.DataFrame(
            np.array(inverse_metal_list + inverse_metal_component_list).reshape((1, -1)))
        return best_individual_x_df, best_individual_inverse_analysis_metal_df

    # 遺伝子の設定
    def create_ind_uniform(self, boundary_dict):
        index = []
        metal_idx = []
        for key in boundary_dict.keys():
            val_type = boundary_dict[key][0]
            # 変数の種類ごとで処理
            if val_type == 'int':  # 整数値
                index.append(round(random.randint(boundary_dict[key]['limit'][0], boundary_dict[key]['limit'][1]),
                                   boundary_dict[key]['round']))
            elif val_type == 'float':  # 小数値
                index.append(round(random.uniform(boundary_dict[key]['limit'][0], boundary_dict[key]['limit'][1]),
                                   boundary_dict[key]['round']))
            elif val_type == 'list':  # リストのインデックス番号
                index.append(random.randint(0, len(boundary_dict[key][1]) - 1))
            elif val_type == 'bi':  # 2値変数
                index.append(random.randint(0, 1))
            elif val_type == 'metal_float':
                index.append(round(random.uniform(0, 1), 4))
            elif val_type == 'metal_list':
                tmp_idx = random.randint(0, len(boundary_dict[key][1]) - 1)
                while tmp_idx in metal_idx:
                    tmp_idx = random.randint(0, len(boundary_dict[key][1]) - 1)
                index.append(tmp_idx)
                metal_idx.append(tmp_idx)
        return index

    # metal_name, metal_ratioの作成
    def create_pop_metal(self, number_of_population):
        #  使用元素リストnum_of_metal個
        #  組成比が合計100%となるように選出（表示は0~0.99)
        if not self.omit_constraints:  # 束縛条件なし
            atom_list = self.condition_dict['metal1'][1]
            inverse_metal_x_data = []
            while len(inverse_metal_x_data) < number_of_population:
                metal_candidate = random.sample(atom_list, self.num_metal)  # 元素数に応じた元素の選択
                if len(metal_candidate) == len(set(metal_candidate)):
                    # 各組成の計算 : 組成は、0～1の間でself.step刻み
                    if self.step == 0.1:
                        ratio_x = [random.randint(1, 9) for _ in range(self.num_metal)]
                        metal_ratio = [round(i / sum(ratio_x), 1) for i in ratio_x]
                    else:
                        ratio_x = [random.randint(1, 99) for _ in range(self.num_metal)]
                        metal_ratio = [round(i / sum(ratio_x), 2) for i in ratio_x]
                    while 0.0 in metal_ratio:
                        metal_ratio = replace_zero(metal_ratio, self.step)
                    metal_ratio = checkMetalRatioTotal(metal_ratio, self.step)
                    metal_ratio_pair = []
                    for a, b in zip(metal_candidate, metal_ratio):
                        metal_ratio_pair = metal_ratio_pair + [a, b]
                    inverse_metal_x_data.append(metal_ratio_pair)
            return inverse_metal_x_data
        else:  # 束縛あり
            ga_util = GAutility()
            #  omit列が空欄のもの（N）から最大3つ、Yのものから最大2つ
            #  Omit列が空欄のもの（N）については組成比が合計70%、Yのものは合計30%となるように選出
            inverse_metal_x_data = []
            while len(inverse_metal_x_data) < number_of_population:
                idx_metal = 1
                metal_name_list = []
                metal_name_omit = []
                while len(metal_name_list) < self.num_metal:
                    atom_list = self.condition_dict[f'metal{idx_metal}'][1]
                    metal_name_omit.append(self.condition_dict[f'metal{idx_metal}'][2])
                    metal_candidate = random.choice(atom_list)
                    while metal_candidate in metal_name_list:
                        metal_candidate = random.choice(atom_list)
                    metal_name_list.append(metal_candidate)
                    idx_metal += 1

                # 各組成の計算 : 組成は、下限0.01、上限0.99、間隔0.01　で作成する
                #  Omit列が空欄のもの（N）については組成比が合計70%、Yのものは合計30%となるように選出
                if self.step == 0.1:
                    metal_component_list = [random.randint(1, 9) / 10 for _ in range(self.num_metal)]
                else:
                    metal_component_list = [random.randint(1, 99) / 100 for _ in range(self.num_metal)]
                metal_component_list = ga_util.get_ratioListConstraints_v2(metal_name_list, metal_component_list,
                                                                           step=self.step)
                while 0.0 in metal_component_list:
                    metal_component_list = replace_zero(metal_component_list, self.step)
                metal_component_list = checkMetalRatioTotal(metal_component_list, self.step)
                metal_ratio_pair = []
                for a, b in zip(metal_name_list, metal_component_list):
                    metal_ratio_pair = metal_ratio_pair + [a, b]
                inverse_metal_x_data.append(metal_ratio_pair)
            return inverse_metal_x_data

    # 評価関数の計算
    def evaluate(self, individual_gene, individual_metal):
        # gene_metal_df: 遺伝子 + 元素とその構成比
        gene_metal_df = pd.DataFrame(
            np.concatenate([np.array(individual_gene), np.array(individual_metal)], axis=1),
            columns=list(self.condition_dict.keys()))

        # individual_x_df:作成すべき仮想実験データX
        individual_x_df = {}
        for x_name in self.x_names:
            individual_x_df[x_name] = \
                pd.DataFrame(np.zeros((gene_metal_df.shape[0], len(self.output_clm[x_name]))),
                             columns=self.output_clm[x_name])
            # clm = self.x_datas[x_name].columns.to_list()
            # clm[-1] = 'groupa * compob'
            # individual_x_df[x_name] = \
            #     pd.DataFrame(np.zeros((gene_metal_df.shape[0], self.x_datas[x_name].shape[1])),
            #                  columns=clm)

        # 遺伝子をxに変換
        rtn_message = {}
        for x_name in self.x_names:
            rtn_message[x_name], individual_x_df[x_name] = self.ind_to_xdata(x_name, gene_metal_df,
                    individual_x_df[x_name], self.condition_dict, self.step,self.omit_constraints, return_metal=False)
        # rtn_message=='success'以外のときは、金属が重複している、実験条件に合わないなどのエラー

        # GPでの予測
        x_df_for_gp = {}
        autoscaled_x_df_for_gp = {}
        for x_name in self.x_names:
            x_df_for_gp[x_name] = \
                individual_x_df[x_name].loc[:, self.x_datas[x_name].columns]
            x_df_for_gp[x_name] = x_df_for_gp[x_name].fillna(0)
            autoscaled_x_df_for_gp[x_name] = \
                autoscaling(x_df_for_gp[x_name], self.x_datas[x_name])
        probability_list = []
        for y_name_idx, (x_name, y_name) in enumerate(zip(self.x_names, self.y_names)):
            log_y_data = self.datasets[x_name][y_name].copy()

            if self.frag_log_transform:
                log_y_data = log_transform(log_y_data)
                log_y_data = pd.Series(log_y_data, index=self.datasets[x_name].index, name=y_name)

            # yの推定値と標準偏差を出力
            autoscaled_estimated_inverse_log_y, autoscaled_estimated_inverse_log_y_std = \
                self.regression_model_dict[y_name].predict(autoscaled_x_df_for_gp[x_name], return_std=True)
            estimated_inverse_log_y = rescaling(autoscaled_estimated_inverse_log_y, log_y_data)
            estimated_inverse_log_y_std = autoscaled_estimated_inverse_log_y_std * log_y_data.std(axis=0, ddof=1)
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
                probability = np.array([np.log(prob) if prob > 0 else -999999 for prob in probability])

            elif self.acquisition_function == 'PI':
                probability = calcPI(target, self.datasets[x_name][y_name], estimated_inverse_y,
                                     estimated_inverse_y_std, relaxation=self.relaxation_value)
                probability = np.array([np.log(prob) if prob > 0 else -999999 for prob in probability])

            elif self.acquisition_function == 'MI':
                acquisition_function_values, new_cumulative_variance = \
                    calcMI(target, estimated_inverse_y, estimated_inverse_y_std,
                           self.cumulative_variance[x_name][:, y_name_idx], self.alpha)
                # cumulative_varianceは、GAの中では更新しない
                probability = acquisition_function_values

            elif self.acquisition_function == 'EI':
                acquisition_function_values = calcEI(target, self.datasets[x_name][y_name], estimated_inverse_y,
                                                     estimated_inverse_y_std, self.relaxation_value)
                probability = acquisition_function_values

            probability_list.append(probability)

        # return probability_list
        probability_array = np.array(probability_list).T
        probabiity_sum = probability_array.sum(axis=1).astype(np.float)
        # 適合度の計算のところで、除外条件や金属元素が重複していたら適合度を強制的に非常に悪い値 (-10 ** 100 など) にする
        for i in range(len(probabiity_sum)):
            for x_name in self.x_names:
                if rtn_message[x_name][i] != 'success':
                    probabiity_sum[i] = -1.0e100
        return [(i,) for i in probabiity_sum]

    def Tournament(self, individuals, k, pop_metal, tournsize, fit_attr="fitness"):
        chosen = []
        chosen_metal = []
        for i in range(k):
            choice = [random.randint(0, k - 1) for _ in range(tournsize)]
            aspirants = [individuals[i] for i in choice]
            aspirant_metals = [pop_metal[i] for i in choice]
            fitness_list = [item.fitness for item in aspirants]
            max_idx = fitness_list.index(max(fitness_list))
            chosen.append(aspirants[max_idx])
            chosen_metal.append(aspirant_metals[max_idx])
        return chosen, chosen_metal

    def mutant_metal(self, individual_metal, indpb):
        if not self.omit_constraints:
            metal_list = self.condition_dict['metal1'][1]
            new_metal = individual_metal.copy()
            for idx in range(len(new_metal)):
                if random.random() < indpb:
                    if idx % 2 == 0:  # metal_name
                        new_metal_name = metal_list[random.randint(0, len(metal_list) - 1)]
                        while len(set(new_metal[::2] + [new_metal_name])) == len(set(new_metal[::2])):
                            # new_metal_name　は、重複するmetal_name
                            new_metal_name = metal_list[random.randint(0, len(metal_list) - 1)]
                        new_metal[idx] = new_metal_name
                    else:  # ratio
                        if self.step == 0.1:
                            new_ratio = random.randint(1, 9) / 10
                        else:
                            new_ratio = random.randint(1, 99) / 100
                        new_metal[idx] = new_ratio
            metal_ratio_list = new_metal[1::2]
            if self.step == 0.1:
                metal_ratio_list = [round(i / sum(metal_ratio_list), 1) for i in metal_ratio_list]
            else:
                metal_ratio_list = [round(i / sum(metal_ratio_list), 2) for i in metal_ratio_list]
            while 0.0 in metal_ratio_list:
                metal_ratio_list = replace_zero(metal_ratio_list, self.step)
            metal_ratio_list = checkMetalRatioTotal(metal_ratio_list, self.step)
            new_metal[1::2] = metal_ratio_list
            return new_metal
        else:  # 束縛あり
            ga_util = GAutility()  # データ作成クラス生成
            new_metal = individual_metal.copy()
            for idx in range(len(new_metal)):
                if random.random() < indpb:
                    if idx % 2 == 0:  # metal_name
                        metal_list = self.condition_dict[f'metal{int(idx / 2) + 1}'][1]
                        new_metal_name = random.choice(metal_list)
                        while len(set(new_metal[::2] + [new_metal_name])) == len(set(new_metal[::2])):
                            # new_metal_name　は、重複するmetal_name
                            new_metal_name = random.choice(metal_list)
                        new_metal[idx] = new_metal_name
                    else:  # ratio
                        if self.step == 0.1:
                            new_ratio = random.randint(1, 9) / 10
                        else:
                            new_ratio = random.randint(1, 99) / 100
                        new_metal[idx] = new_ratio
            metal_name_list = new_metal[::2]
            metal_ratio_list = new_metal[1::2]
            metal_ratio_list = ga_util.get_ratioListConstraints_v2(metal_name_list, metal_ratio_list, step=self.step)
            new_metal[1::2] = metal_ratio_list
            return new_metal

