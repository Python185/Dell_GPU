
from scipy.stats import norm
#from .preprocess_for_gp import GPpreprocess

class GPinverseprocess():
    def __init__(self, frag_log_transform, target_y_dict):
        self.frag_log_transform = frag_log_transform
        self.target_y_dict = target_y_dict

    def setting_acquisition_function(self, acquisition_function, cumulative_variance=None, relaxation_value=None, alpha=None):
        self.acquisition_function = acquisition_function
        self.cumulative_variance = cumulative_variance
        self.relaxation_value = relaxation_value
        self.alpha = alpha
        
    #def calc_probability_acquisition(self, base_y, y_name, cumulative_i,  estimated_inverse_log_y, estimated_inverse_log_y_std):
    def calc_probability_acquisition(self, base_y, y_name, cumulative_i,  estimated_inverse_y, estimated_inverse_y_std):

        # 評価関数（目標達成の確率を計算）
        target_objective = self.target_y_dict[y_name]['objective']
        target_y_score = self.target_y_dict[y_name]['score']
        
        if self.acquisition_function == 'PTR':
            if target_objective == 'minimum':
                probability = norm.cdf(x=target_y_score, loc=estimated_inverse_y.reshape((-1, 1)), scale=estimated_inverse_y_std.reshape((-1, 1)))
            elif target_objective == 'maximum':
                probability = 1 - norm.cdf(x=target_y_score, loc=estimated_inverse_y.reshape((-1, 1)), scale=estimated_inverse_y_std.reshape((-1, 1)))
            else:
                raise Exception('目標値を確認してください')
            for i, prob in enumerate(probability):
                if prob == 0: probability[i] = pow(10, -10)

        elif self.acquisition_function == 'PI':
            if target_objective == 'minimum':
                min_value = base_y.min()
                # UPDATE by mase on 2022/11/29
                probability = norm.cdf(x=min_value-self.relaxation_value, loc=estimated_inverse_y.reshape((-1, 1)), scale=estimated_inverse_y_std.reshape((-1, 1)))
                #probability = norm.cdf(x=min_value+self.relaxation_value, loc=estimated_inverse_log_y.reshape((-1, 1)), scale=estimated_inverse_log_y_std.reshape((-1, 1)))
            elif target_objective == 'maximum':
                max_value = base_y.max()
                probability = 1-norm.cdf(x=max_value+self.relaxation_value, loc=estimated_inverse_y.reshape((-1, 1)), scale=estimated_inverse_y_std.reshape((-1, 1)))
            for i, prob in enumerate(probability):
                if prob == 0: probability[i] = pow(10, -10)

        elif self.acquisition_function == 'MI':
            # raise Exception('Under Construction')
            if target_objective == 'minimum':
                estimated_inverse_y *= -1
            elif target_objective == 'maximum':
                estimated_inverse_y *= 1   
            acquisition_function_values = estimated_inverse_y.flatten() + self.alpha ** 0.5 * ((estimated_inverse_y_std.flatten() ** 2 + self.cumulative_variance) ** 0.5 - self.cumulative_variance ** 0.5)
            #acquisition_function_values = estimated_inverse_log_y.flatten() + self.alpha ** 0.5 * ((estimated_inverse_log_y_std.flatten() ** 2 + self.cumulative_variance[:, cumulative_i].flatten()) ** 0.5 - self.cumulative_variance[:, cumulative_i].flatten() ** 0.5)
            probability = acquisition_function_values

        elif self.acquisition_function == 'EI':
            imp = (estimated_inverse_y - base_y.max() - self.relaxation_value)
            z = imp / estimated_inverse_y_std
            acquisition_function_values = imp * norm.cdf(z) + estimated_inverse_y_std * norm.pdf(z)
            probability = acquisition_function_values

        return probability, estimated_inverse_y, estimated_inverse_y_std
