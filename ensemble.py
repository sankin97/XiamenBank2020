# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime

import random
import time
import re
import requests
import sys
import xlrd
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

#忽略警告提示
import warnings
warnings.filterwarnings('ignore')

train_base_path = "./data/x_train/"
test_base_path = "./data/x_test/"
base_save_path = "./data/tmp/"


y_train_q3 = pd.read_csv('./data/y_train_3/y_Q3_3.csv')
y_train_q4 = pd.read_csv('./data/y_train_3/y_Q4_3.csv')
print("*" * 10)
print(np.sum(y_train_q3['label']==1)/len(y_train_q3))
print(np.sum(y_train_q3['label']==0)/len(y_train_q3))
print(np.sum(y_train_q3['label']==-1)/len(y_train_q3))
print("*"*10)
print(np.sum(y_train_q4['label']==1)/len(y_train_q4))
print(np.sum(y_train_q4['label']==0)/len(y_train_q4))
print(np.sum(y_train_q4['label']==-1)/len(y_train_q4))


print("*"*10)
y_test = pd.read_csv('./result_prob_concat_all_500_1203_nolambda_491544.csv')
y_test2 = pd.read_csv('./result_prob_concat_all_500_1203_nolambda_49261.csv')
y_test3 = pd.read_csv('./result_prob_lgb_stacking_491.csv')
###49.465
# y_test["1"] = y_test["1"]*0.35 + y_test2["1"]*0.35+y_test3["1"]*0.3
# y_test["0"] = y_test["0"]*0.35 + y_test2["0"]*0.35+y_test3["0"]*0.3
# y_test["-1"] = y_test["-1"]*0.35 + y_test2["-1"]*0.35+y_test3["-1"]*0.3

# y_test["1"] *= 0.832
# # y_test["0"] = 
# y_test["-1"] *=1.055


###49.475
# **********
# 0.6310836526680743
# 0.2143844008237533
# 0.15453194650817237
# y_test["1"] = y_test["1"]*0.35 + y_test2["1"]*0.35+y_test3["1"]*0.3
# y_test["0"] = y_test["0"]*0.35 + y_test2["0"]*0.35+y_test3["0"]*0.3
# y_test["-1"] = y_test["-1"]*0.35 + y_test2["-1"]*0.35+y_test3["-1"]*0.3



# y_test["1"] *= 0.822
# # y_test["0"] = 
# y_test["-1"] *=1.05
####49.53
# 0.630783869033654 48395
# 0.21370662912854202 16396
# 0.15550950183780402 11931

###49.552
y_test["1"] = y_test["1"]*0.32 + y_test2["1"]*0.38+y_test3["1"]*0.3
y_test["0"] = y_test["0"]*0.32 + y_test2["0"]*0.38+y_test3["0"]*0.3
y_test["-1"] = y_test["-1"]*0.32 + y_test2["-1"]*0.38+y_test3["-1"]*0.3



y_test["1"] *= 0.822
# y_test["0"] = 
y_test["-1"] *=1.05

y_test["label"] = np.argmax(y_test[['-1', '0', '1']].values, axis=1) - 1
print(np.sum(y_test['label'] == 1) / len(y_test))
print(np.sum(y_test['label'] == 0) / len(y_test))
print(np.sum(y_test['label'] == -1) / len(y_test))
y_test[["cust_no", "label"]].to_csv("result_prob_lgb_concat_all_ensemble_3m_1201.csv", index=False)




