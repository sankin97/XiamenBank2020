#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# warnings.filterwarnings('ignore')

train_base_path = "./data/x_train/"
test_base_path = "./data/x_test/"
base_save_path = "./data/tmp/"

train_cust_avli_q4=pd.read_csv(train_base_path + 'cust_avli_Q4.csv')
train_cust_avli_q3=pd.read_csv(train_base_path + 'cust_avli_Q3.csv')
test_cust_avli_q1=pd.read_csv(test_base_path + 'cust_avli_Q1.csv')

cust_avli_all = pd.concat([train_cust_avli_q4, train_cust_avli_q3],axis=0,ignore_index=True)
cust_avli_all = pd.concat([cust_avli_all, test_cust_avli_q1],axis=0,ignore_index=True)
cust_avli_all = cust_avli_all.drop_duplicates(subset="cust_no", keep='first', inplace=False)
cust_avli_all.to_csv(base_save_path + 'cust_avli_all.csv',index=0)
