# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:08:22 2022

@author: dcelab
"""

import numpy as np

# 目的変数
Y_COLUMNS_ALL = ['選択率NPA', '収率NPA']

#説明変数名
X_COLUMNS_ALL = []

# 評価
R2 = 'R2'
MAE = 'MAE'
RMSE = 'RMSE'

# 獲得関数関連
AcqusitFunction = str
PTR: AcqusitFunction = 'PTR'
PI: AcqusitFunction = 'PI'
EI: AcqusitFunction = 'EI'
MI: AcqusitFunction = 'MI'

# 獲得関数の設定
RELAXATION_VALUE = 0.01
DELTA = 10 ** -6
ALPHA = np.log(2 / DELTA)

#ターゲット
KIND = 'kind'
TARGET = 'target'
#ターゲットタイプ
TargetType = int
# ターゲット値以上
MAXIMIZE: TargetType = 1
# ターゲット範囲
RANGE: TargetType = 0
# ターゲット値以下
MINIMIZE: TargetType = -1