import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dcekit.validation import ApplicabilityDomain

#データのADを計算する
#trainデータの読込み
train_df = pd.read_csv('MyWork/datasets/predict_wo_sp_el/x1_Cu2_1.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
#testデータ読込み
test_df = pd.read_csv('MyWork/datasets/predict_wo_sp_el/x1_Cu2_5.csv', encoding= 'utf-8-sig', index_col= 0, header= 0)
y_data = train_df['C2 yield'].copy()
x_data = train_df.drop('C2 yield', axis= 1).copy()
x_base = test_df.drop('C2 yield', axis= 1).copy()
y_ad = test_df['C2 yield'].copy()

#スケーリング
autoscaled_x = (x_data - x_data.mean(axis=0)) / x_data.std(axis=0, ddof=1)
autoscaled_x_base = (x_base - x_data.mean(axis=0)) / x_data.std(axis=0, ddof=1)
check_nan = autoscaled_x_base.isnull().any()

#ADの計算
#ad_dataはk-NNでは平均距離の逆数、この値がマイナスのときはAD外であることを意味する
#ad = ApplicabilityDomain(method_name='ocsvm', rate_of_outliers=0.01)
#ad = ApplicabilityDomain(method_name='knn', rate_of_outliers=0.01)
ad = ApplicabilityDomain(method_name='lof', rate_of_outliers=0.01)
ad.fit(autoscaled_x)
ad_data = ad.predict(autoscaled_x_base)
ad_data = pd.DataFrame(data= ad_data, index= x_base.index)
ad_data = ad_data.rename(columns= {0: 'ad_data'})
y_ad = pd.concat([y_ad, ad_data], axis= 1)
y_ad['ad_data'] = np.where(y_ad['ad_data'] <= 0, 'outofAD', 'OK')
y_ad.to_csv('MyWork/datasets/predict_wo_sp_el/x1_Cu2_ad.csv', encoding= 'utf-8-sig')

print('hello')
