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
warnings.filterwarnings('ignore')

train_base_path = "./data/x_train/"
test_base_path = "./data/x_test/"
base_save_path = "./data/tmp/"

cust_no_all = pd.read_csv(base_save_path + 'cust_avli_all.csv')

def agg_numeric(df, group_var, df_name):
    for col in df:
        if col != group_var and 'cust_no' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    agg = numeric_df.groupby(group_var).agg(['max', 'min', 'std','median','mean']).reset_index()

    columns = [group_var]

    for var in agg.columns.levels[0]:
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:

                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

def filter_users(df):
    df = pd.merge(df, cust_no_all,on='cust_no',how='right')
    return df


train_cust_avli_q4=pd.read_csv(train_base_path + 'cust_avli_Q4.csv')
train_cust_avli_q3=pd.read_csv(train_base_path + 'cust_avli_Q3.csv')
train_cust_info_q4=pd.read_csv(train_base_path + 'cust_info_q4.csv')
train_cust_info_q4 = filter_users(train_cust_info_q4)
train_cust_info_q3=pd.read_csv(train_base_path + 'cust_info_q3.csv')
train_cust_info_q3 = filter_users(train_cust_info_q3)

train_big_event_q4 = pd.read_csv(train_base_path + 'big_event_train/big_event_Q4.csv')
train_big_event_q3 = pd.read_csv(train_base_path + 'big_event_train/big_event_Q3.csv')
train_big_event_q4 = filter_users(train_big_event_q4)
train_big_event_q3 = filter_users(train_big_event_q3)

test_cust_avli_q1=pd.read_csv(test_base_path + 'cust_avli_Q1.csv')
test_cust_info_q1=pd.read_csv(test_base_path + 'cust_info_q1.csv')
test_cust_info_q1 = filter_users(test_cust_info_q1)

test_big_event_q1 = pd.read_csv(test_base_path + 'big_event_test/big_event_Q1.csv')
test_big_event_q1 = filter_users(test_big_event_q1)

train_aum_m7=pd.read_csv(train_base_path + 'aum_train/aum_m7.csv')
train_aum_m8=pd.read_csv(train_base_path + 'aum_train/aum_m8.csv')
train_aum_m9=pd.read_csv(train_base_path + 'aum_train/aum_m9.csv')
train_aum_m7 = filter_users(train_aum_m7)
train_aum_m8 = filter_users(train_aum_m8)
train_aum_m9 = filter_users(train_aum_m9)

train_aum_m10=pd.read_csv(train_base_path + 'aum_train/aum_m10.csv')
train_aum_m11=pd.read_csv(train_base_path + 'aum_train/aum_m11.csv')
train_aum_m12=pd.read_csv(train_base_path + 'aum_train/aum_m12.csv')
train_aum_m10 = filter_users(train_aum_m10)
train_aum_m11 = filter_users(train_aum_m11)
train_aum_m12 = filter_users(train_aum_m12)

test_aum_m1=pd.read_csv(test_base_path + 'aum_test/aum_m1.csv')
test_aum_m2=pd.read_csv(test_base_path + 'aum_test/aum_m2.csv')
test_aum_m3=pd.read_csv(test_base_path + 'aum_test/aum_m3.csv')
test_aum_m1 = filter_users(test_aum_m1)
test_aum_m2 = filter_users(test_aum_m2)
test_aum_m3 = filter_users(test_aum_m3)

train_behavior_m7=pd.read_csv(train_base_path + 'behavior_train/behavior_m7.csv')
train_behavior_m8=pd.read_csv(train_base_path + 'behavior_train/behavior_m8.csv')
train_behavior_m9=pd.read_csv(train_base_path + 'behavior_train/behavior_m9.csv')
train_behavior_m7 = filter_users(train_behavior_m7)
train_behavior_m8 = filter_users(train_behavior_m8)
train_behavior_m9 = filter_users(train_behavior_m9)

train_behavior_m10=pd.read_csv(train_base_path + 'behavior_train/behavior_m10.csv')
train_behavior_m11=pd.read_csv(train_base_path + 'behavior_train/behavior_m11.csv')
train_behavior_m12=pd.read_csv(train_base_path + 'behavior_train/behavior_m12.csv')
train_behavior_m10 = filter_users(train_behavior_m10)
train_behavior_m11 = filter_users(train_behavior_m11)
train_behavior_m12 = filter_users(train_behavior_m12)

test_behavior_m1=pd.read_csv(test_base_path + 'behavior_test/behavior_m1.csv')
test_behavior_m2=pd.read_csv(test_base_path + 'behavior_test/behavior_m2.csv')
test_behavior_m3=pd.read_csv(test_base_path + 'behavior_test/behavior_m3.csv')
test_behavior_m1 = filter_users(test_behavior_m1)
test_behavior_m2 = filter_users(test_behavior_m2)
test_behavior_m3 = filter_users(test_behavior_m3)

train_cunkuan_m7=pd.read_csv(train_base_path + 'cunkuan_train/cunkuan_m7.csv')
train_cunkuan_m8=pd.read_csv(train_base_path + 'cunkuan_train/cunkuan_m8.csv')
train_cunkuan_m9=pd.read_csv(train_base_path + 'cunkuan_train/cunkuan_m9.csv')
train_cunkuan_m7 = filter_users(train_cunkuan_m7)
train_cunkuan_m8 = filter_users(train_cunkuan_m8)
train_cunkuan_m9 = filter_users(train_cunkuan_m9)

train_cunkuan_m10=pd.read_csv(train_base_path + 'cunkuan_train/cunkuan_m10.csv')
train_cunkuan_m11=pd.read_csv(train_base_path + 'cunkuan_train/cunkuan_m11.csv')
train_cunkuan_m12=pd.read_csv(train_base_path + 'cunkuan_train/cunkuan_m12.csv')
train_cunkuan_m10 = filter_users(train_cunkuan_m10)
train_cunkuan_m11 = filter_users(train_cunkuan_m11)
train_cunkuan_m12 = filter_users(train_cunkuan_m12)

test_cunkuan_m1=pd.read_csv(test_base_path + 'cunkuan_test/cunkuan_m1.csv')
test_cunkuan_m2=pd.read_csv(test_base_path + 'cunkuan_test/cunkuan_m2.csv')
test_cunkuan_m3=pd.read_csv(test_base_path + 'cunkuan_test/cunkuan_m3.csv')
test_cunkuan_m1 = filter_users(test_cunkuan_m1)
test_cunkuan_m2 = filter_users(test_cunkuan_m2)
test_cunkuan_m3 = filter_users(test_cunkuan_m3)

y_train_q3 = pd.read_csv('./data/y_train_3/y_Q3_3.csv')
y_train_q4 = pd.read_csv('./data/y_train_3/y_Q4_3.csv')


def extract_behavior_features(tmp):
    tmp['B3-B5'] = tmp['B3'] - tmp["B5"]
    tmp['B3+B5'] = tmp['B3'] + tmp["B5"]
    tmp['B4-B2'] = tmp['B4'] - tmp["B2"]
    tmp['B4+B2'] = tmp['B4'] + tmp["B2"]
    tmp['B3/B2'] = tmp['B3'] / (tmp["B2"] + 0.000001)
    tmp['B5/B4'] = tmp['B5'] / (tmp["B4"] + 0.000001)
    tmp['B3/B2-B5/B4'] = tmp['B3/B2'] - tmp['B5/B4']
    tmp['B3/B2+B5/B4'] = tmp['B3/B2'] + tmp['B5/B4']
    tmp['B1-B2-B4'] = tmp['B1'] - tmp['B4+B2']
    tmp['B1/B2'] = tmp['B1'] / (tmp["B2"] + 0.000001)
    tmp['B1/B4'] = tmp['B1'] / (tmp["B4"] + 0.000001)
    tmp['B4/B2'] = tmp['B4'] / (tmp["B2"] + 0.000001)
    tmp['B5/B3'] = tmp['B5'] / (tmp["B3"] + 0.000001)
    tmp['B3/B2/(B5/B4)'] = tmp['B3/B2'] / (tmp['B5/B4'] + 0.000001)
    return tmp


def extract_aum_feature(tmp):
    tmp["X_sum"] = tmp["X1"] + tmp["X2"] + tmp["X3"] + tmp["X4"] + tmp["X5"] + tmp["X6"] + tmp["X8"]
    tmp["X1+X2+X3"] = tmp["X1"] + tmp["X2"] + tmp["X3"]
    tmp["X2+X3"] = tmp["X2"] + tmp["X3"]
    tmp["X4+X5"] = tmp["X4"] + tmp["X5"]
    tmp["X4+X5+X6"] = tmp["X4"] + tmp["X5"] + tmp["X6"]
    tmp["X2/X3"] = tmp["X2"] / (tmp["X3"] + 0.000001)
    tmp["X5/X4"] = tmp["X5"] / (tmp["X4"] + 0.000001)
    tmp["(X4+X5+X6)/(X1+X2+X3)"] = tmp["X4+X5+X6"] / (tmp["X1+X2+X3"] + 0.000001)
    tmp["(X4+X5)/(X2+X3)"] = tmp["X4+X5"] / (tmp["X2+X3"] + 0.000001)
    tmp["X_sum/X7"] = tmp["X_sum"] / (tmp["X7"] + 0.000001)
    tmp["(X4+X5+X6)/X7"] = tmp["X4+X5+X6"] / (tmp["X7"] + 0.000001)
    tmp["(X4+X5)/X7"] = tmp["X4+X5"] / (tmp["X7"] + 0.000001)
    return tmp


def extract_cunkuan_feature(tmp):
    tmp["C1/C2"] = tmp["C1"] / (tmp["C2"] + 0.000001)
    return tmp


def extract_behavior_merge_feature(behavior_merge):
    for name in ["B1", "B2", "B3", "B4", "B5", 'B3-B5', 'B3+B5', 'B3/B2', 'B5/B4', 'B3/B2-B5/B4', 'B3/B2+B5/B4', 'B4-B2', 'B4+B2', 'B1-B2-B4', 'B1/B2', 'B4/B2', 'B5/B3', 'B3/B2/(B5/B4)']:
        behavior_merge["{}_m1_m2_m3_max".format(name)] = behavior_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].max(axis=1)
        behavior_merge["{}_m1_m2_m3_min".format(name)] = behavior_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].min(axis=1)
        behavior_merge["{}_m1_m2_m3_max/min".format(name)] = behavior_merge["{}_m1_m2_m3_max".format(name)] / (behavior_merge["{}_m1_m2_m3_min".format(name)]+0.000001)
        behavior_merge["{}_m1_m2_m3_max-min".format(name)] = behavior_merge["{}_m1_m2_m3_max".format(name)] - behavior_merge["{}_m1_m2_m3_min".format(name)]
        behavior_merge["{}_m1_m2_m3_mean".format(name)] = behavior_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].mean(axis=1)
        behavior_merge["{}_m1_m2_m3_sum".format(name)] = behavior_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].sum(axis=1)
        behavior_merge["{}_m1_m2_m3_std".format(name)] = behavior_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].std(axis=1)
        # behavior_merge["{}_m1_m2_m3_last".format(name)] = behavior_merge[
        #     ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].last(axis=1)
        behavior_merge["{}_m3-m2".format(name)] = behavior_merge["{}".format(name)] - behavior_merge[
            "{}_m2".format(name)]
        behavior_merge["{}_m2-m1".format(name)] = behavior_merge["{}_m2".format(name)] - behavior_merge[
            "{}_m1".format(name)]
        behavior_merge["{}_m3-m1".format(name)] = behavior_merge["{}".format(name)] - behavior_merge[
            "{}_m1".format(name)]
        behavior_merge["{}_m3/m2".format(name)] = behavior_merge["{}".format(name)] / (behavior_merge[
            "{}_m2".format(name)] + 0.000001)
        behavior_merge["{}_m2/m1".format(name)] = behavior_merge["{}_m2".format(name)] / (behavior_merge[
            "{}_m1".format(name)] + 0.000001)
        behavior_merge["{}_m3/m1".format(name)] = behavior_merge["{}".format(name)] / (behavior_merge[
            "{}_m1".format(name)] + 0.000001)
    return behavior_merge


def extract_aum_merge_feature(aum_merge):
    for i in range(1, 9):
        aum_merge["X{}_m1_m2_m3_max".format(i)] = aum_merge[
            ["X{}_m1".format(i), "X{}_m2".format(i), "X{}".format(i)]].max(axis=1)
        aum_merge["X{}_m1_m2_m3_min".format(i)] = aum_merge[
            ["X{}_m1".format(i), "X{}_m2".format(i), "X{}".format(i)]].min(axis=1)
        aum_merge["X{}_m1_m2_m3_max/min".format(i)] = aum_merge["X{}_m1_m2_m3_max".format(i)] / (
                    aum_merge["X{}_m1_m2_m3_min".format(i)] + 0.000001)
        aum_merge["X{}_m1_m2_m3_max-min".format(i)] = aum_merge["X{}_m1_m2_m3_max".format(i)] - aum_merge[
            "X{}_m1_m2_m3_min".format(i)]
        aum_merge["X{}_m1_m2_m3_mean".format(i)] = aum_merge[
            ["X{}_m1".format(i), "X{}_m2".format(i), "X{}".format(i)]].mean(axis=1)
        aum_merge["X{}_m1_m2_m3_sum".format(i)] = aum_merge[
            ["X{}_m1".format(i), "X{}_m2".format(i), "X{}".format(i)]].sum(axis=1)
        aum_merge["X{}_m1_m2_m3_std".format(i)] = aum_merge[
            ["X{}_m1".format(i), "X{}_m2".format(i), "X{}".format(i)]].std(axis=1)
        # aum_merge["X{}_m1_m2_m3_var".format(i)] = aum_merge[
        #     ["X{}_m1".format(i), "X{}_m2".format(i), "X{}".format(i)]].var(axis=1)
        aum_merge["X{}_m3-m2".format(i)] = train_aum_merge["X{}".format(i)] - train_aum_merge["X{}_m2".format(i)]
        aum_merge["X{}_m2-m1".format(i)] = train_aum_merge["X{}_m2".format(i)] - train_aum_merge["X{}_m1".format(i)]
        aum_merge["X{}_m3-m1".format(i)] = train_aum_merge["X{}".format(i)] - train_aum_merge["X{}_m1".format(i)]
        aum_merge["X{}_m3/m2".format(i)] = train_aum_merge["X{}".format(i)] / (
                    train_aum_merge["X{}_m2".format(i)] + 0.000001)
        aum_merge["X{}_m2/m1".format(i)] = train_aum_merge["X{}_m2".format(i)] / (
                    train_aum_merge["X{}_m1".format(i)] + 0.000001)
        aum_merge["X{}_m3/m1".format(i)] = train_aum_merge["X{}".format(i)] / (
                    train_aum_merge["X{}_m1".format(i)] + 0.000001)

    for name in ["X_sum", "X1+X2+X3", "X4+X5+X6", "X2+X3", "X4+X5", "X2/X3", "X5/X4", "(X4+X5+X6)/(X1+X2+X3)",
                 "(X4+X5)/(X2+X3)", "X_sum/X7", "(X4+X5+X6)/X7", "(X4+X5)/X7"]:
        aum_merge["{}_m1_m2_m3_max".format(name)] = aum_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].max(axis=1)
        aum_merge["{}_m1_m2_m3_min".format(name)] = aum_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].min(axis=1)
        aum_merge["{}_m1_m2_m3_max/min".format(name)] = aum_merge["{}_m1_m2_m3_max".format(name)] / (
                aum_merge["{}_m1_m2_m3_min".format(name)] + 0.000001)
        aum_merge["{}_m1_m2_m3_max-min".format(name)] = aum_merge["{}_m1_m2_m3_max".format(name)] - aum_merge[
            "{}_m1_m2_m3_min".format(name)]
        aum_merge["{}_m1_m2_m3_mean".format(name)] = aum_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].mean(axis=1)
        aum_merge["{}_m1_m2_m3_sum".format(name)] = aum_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].sum(axis=1)
        aum_merge["{}_m1_m2_m3_std".format(name)] = aum_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].std(axis=1)
        # aum_merge["{}_m1_m2_m3_var".format(name)] = aum_merge[
        #     ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].var(axis=1)
        aum_merge["{}_m3-m2".format(name)] = aum_merge["{}".format(name)] - aum_merge["{}_m2".format(name)]
        aum_merge["{}_m2-m1".format(name)] = aum_merge["{}_m2".format(name)] - aum_merge["{}_m1".format(name)]
        aum_merge["{}_m3-m1".format(name)] = aum_merge["{}".format(name)] - aum_merge["{}_m1".format(name)]
        aum_merge["{}_m3/m2".format(name)] = aum_merge["{}".format(name)] / (
                train_aum_merge["{}_m2".format(name)] + 0.000001)
        aum_merge["{}_m2/m1".format(name)] = aum_merge["{}_m2".format(name)] / (
                aum_merge["{}_m1".format(name)] + 0.000001)
        aum_merge["{}_m3/m1".format(name)] = aum_merge["{}".format(name)] / (
                aum_merge["{}_m1".format(name)] + 0.000001)
    return aum_merge


def extract_cunkuan_merge_feature(cunkuan_merge):
    for name in ["C1", 'C2', "C1/C2"]:
        cunkuan_merge["{}_m1_m2_m3_max".format(name)] = cunkuan_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].max(axis=1)
        cunkuan_merge["{}_m1_m2_m3_min".format(name)] = cunkuan_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].min(axis=1)
        cunkuan_merge["{}_m1_m2_m3_max/min".format(name)] = cunkuan_merge["{}_m1_m2_m3_max".format(name)] / (
                    cunkuan_merge["{}_m1_m2_m3_min".format(name)] + 0.000001)
        cunkuan_merge["{}_m1_m2_m3_max-min".format(name)] = cunkuan_merge["{}_m1_m2_m3_max".format(name)] - \
                                                            cunkuan_merge["{}_m1_m2_m3_min".format(name)]
        cunkuan_merge["{}_m1_m2_m3_mean".format(name)] = cunkuan_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].mean(axis=1)
        cunkuan_merge["{}_m1_m2_m3_sum".format(name)] = cunkuan_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].sum(axis=1)
        cunkuan_merge["{}_m1_m2_m3_std".format(name)] = cunkuan_merge[
            ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].std(axis=1)
        # cunkuan_merge["{}_m1_m2_m3_var".format(name)] = cunkuan_merge[
        #     ["{}_m1".format(name), "{}_m2".format(name), "{}".format(name)]].var(axis=1)
        cunkuan_merge["{}_m3-m2".format(name)] = cunkuan_merge["{}".format(name)] - cunkuan_merge["{}_m2".format(name)]
        cunkuan_merge["{}_m2-m1".format(name)] = cunkuan_merge["{}_m2".format(name)] - cunkuan_merge[
            "{}_m1".format(name)]
        cunkuan_merge["{}_m3-m1".format(name)] = cunkuan_merge["{}".format(name)] - cunkuan_merge["{}_m1".format(name)]
        cunkuan_merge["{}_m3/m2".format(name)] = cunkuan_merge["{}".format(name)] / (cunkuan_merge[
                                                                                         "{}_m2".format(
                                                                                             name)] + 0.000001)
        cunkuan_merge["{}_m2/m1".format(name)] = cunkuan_merge["{}_m2".format(name)] / (cunkuan_merge[
                                                                                            "{}_m1".format(
                                                                                                name)] + 0.000001)
        cunkuan_merge["{}_m3/m1".format(name)] = cunkuan_merge["{}".format(name)] / (cunkuan_merge[
                                                                                         "{}_m1".format(
                                                                                             name)] + 0.000001)
    return cunkuan_merge


def extract_big_event(df):
    for i in range(1, 19):
        if i != 15 and i != 17:
            df['E{}'.format(i)] = pd.to_datetime(df['E{}'.format(i)])
            df['E{}_year'.format(i)] = df['E{}'.format(i)].map(lambda x: x.year)
            df['E{}_now_day_diff'.format(i)] = (datetime.strptime('20200331', '%Y%m%d') - pd.to_datetime(df["E{}".format(i)])).apply(lambda x: x.days)
            # df['E{}_now_year_diff'.format(i)] = (
            #             datetime.strptime('20200331', '%Y%m%d') - pd.to_datetime(df["E{}".format(i)])).apply(lambda x: x.days/365.0)
            # df['E{}_now_year_diff_2'.format(i)] = 2020 - pd.to_datetime(df["E{}".format(i)]).apply(lambda x: x.year)

    for i in range(1, 19):
        if i != 15 and i != 17:
            for j in range(i+1, 19):
                if j != 15 and j != 17:
                    df['E{}_E{}_day_diff'.format(i, j)] = (pd.to_datetime(df["E{}".format(i)]) - pd.to_datetime(df["E{}".format(j)])).apply(lambda x: x.days)
                    # df['E{}_E{}_year_diff'.format(i, j)] = (
                    #             pd.to_datetime(df["E{}".format(i)]) - pd.to_datetime(df["E{}".format(j)])).apply(
                    #     lambda x: x.days/365.0)
                    # df['E{}_E{}_year_diff_2'.format(i, j)] = pd.to_datetime(df["E{}".format(i)]).apply(lambda x: x.year) - pd.to_datetime(df["E{}".format(j)]).apply(lambda x: x.year)
    df["E15+E17"] = df["E15"] + df["E17"]
    df["E15-E17"] = df["E15"] - df["E17"]
    df["E15/E17"] = df["E15"] / df["E17"]
    return df


train_behavior_m9['B6'] = pd.to_datetime(train_behavior_m9['B6'])
train_behavior_m9['B6_day'] = train_behavior_m9['B6'].map(lambda x: x.day)
train_behavior_m9['B6_day_diff'] = (datetime.strptime('20200331', '%Y%m%d') - pd.to_datetime(train_behavior_m9["B6"])).apply(lambda x: x.days)
train_behavior_m7 = extract_behavior_features(train_behavior_m7)
train_behavior_m8 = extract_behavior_features(train_behavior_m8)
train_behavior_m9 = extract_behavior_features(train_behavior_m9)

train_aum_m7 = extract_aum_feature(train_aum_m7)
train_aum_m8 = extract_aum_feature(train_aum_m8)
train_aum_m9 = extract_aum_feature(train_aum_m9)

train_cunkuan_m7 = extract_cunkuan_feature(train_cunkuan_m7)
train_cunkuan_m8 = extract_cunkuan_feature(train_cunkuan_m8)
train_cunkuan_m9 = extract_cunkuan_feature(train_cunkuan_m9)

train_behavior_merge = pd.merge(train_behavior_m7,train_behavior_m8,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
train_behavior_merge = pd.merge(train_behavior_merge,train_behavior_m9,on='cust_no',how='left', suffixes=(None, '_m3'))
train_behavior_concat = pd.concat([train_behavior_m7,train_behavior_m8],axis=0,ignore_index=True)
train_behavior_concat = pd.concat([train_behavior_concat,train_behavior_m9],axis=0,ignore_index=True)

train_aum_merge = pd.merge(train_aum_m7,train_aum_m8,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
train_aum_merge = pd.merge(train_aum_merge,train_aum_m9,on='cust_no',how = 'left', suffixes=(None, '_m3'))
train_aum_concat = pd.concat([train_aum_m7,train_aum_m8],axis=0,ignore_index=True)
train_aum_concat = pd.concat([train_aum_concat,train_aum_m9],axis=0,ignore_index=True)

train_cunkuan_merge = pd.merge(train_cunkuan_m7,train_cunkuan_m8,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
train_cunkuan_merge = pd.merge(train_cunkuan_merge,train_cunkuan_m9,on='cust_no',how = 'left', suffixes=(None, '_m3'))
train_cunkuan_concat = pd.concat([train_cunkuan_m7,train_cunkuan_m8],axis=0,ignore_index=True)
train_cunkuan_concat = pd.concat([train_cunkuan_concat,train_cunkuan_m9],axis=0,ignore_index=True)

train_behavior_concat_agg = agg_numeric(train_behavior_concat,group_var='cust_no',df_name='concat_3m_')
train_aum_concat_agg = agg_numeric(train_aum_concat,group_var='cust_no',df_name='concat_3m_')
train_cunkuan_concat_agg = agg_numeric(train_cunkuan_concat,group_var='cust_no',df_name='concat_3m_')


# print(list(train_behavior_merge.columns))
train_behavior_merge = extract_behavior_merge_feature(train_behavior_merge)
train_aum_merge = extract_aum_merge_feature(train_aum_merge)
train_cunkuan_merge = extract_cunkuan_merge_feature(train_cunkuan_merge)
train_big_event_q3 = extract_big_event(train_big_event_q3)
train_big_event_q3_agg = agg_numeric(train_big_event_q3,group_var='cust_no',df_name='concat_3m_')
# train_cust_info_q3["age_class"] = train_cust_info_q3['I2'].map(lambda x: int(x/5))
train_1 = train_cust_info_q3
print(train_1.shape)
train_y = pd.merge(train_1,y_train_q3,on='cust_no',how = 'left')
print(train_y.shape, y_train_q3.shape)
train_y = pd.merge(train_y,train_aum_merge,on='cust_no',how = 'left')
print(train_aum_merge.shape)
train_y = pd.merge(train_y,train_aum_concat_agg)
print(train_aum_concat_agg.shape)
train_y = pd.merge(train_y,train_behavior_merge,on='cust_no',how = 'left')
print(train_behavior_merge.shape)
# train_y = pd.merge(train_y,train_behavior_concat_agg,on='cust_no',how='left')
# print(train_behavior_concat_agg.shape)
train_y = pd.merge(train_y,train_cunkuan_merge,on='cust_no',how = 'left')
print(train_cunkuan_merge.shape)
train_y = pd.merge(train_y,train_cunkuan_concat_agg,on='cust_no',how='left')
print(train_cunkuan_concat_agg.shape)
train_y = pd.merge(train_y,train_big_event_q3,on='cust_no',how = 'left')
print(train_big_event_q3.shape)
# train_y = pd.merge(train_y,train_big_event_q3_agg,on='cust_no',how='left')
# print(train_big_event_q3_agg.shape)
print(train_y.shape,y_train_q3.shape)
train_y.to_csv(base_save_path + 'Q3_orgmerge_new_1204.csv',index=0)

train_behavior_m12['B6'] = pd.to_datetime(train_behavior_m12['B6'])
train_behavior_m12['B6_day'] = train_behavior_m12['B6'].map(lambda x: x.day)
train_behavior_m12['B6_day_diff'] = (datetime.strptime('20200331', '%Y%m%d') - pd.to_datetime(train_behavior_m12["B6"])).apply(lambda x: x.days)
train_behavior_m10 = extract_behavior_features(train_behavior_m10)
train_behavior_m11 = extract_behavior_features(train_behavior_m11)
train_behavior_m12 = extract_behavior_features(train_behavior_m12)

train_aum_m10 = extract_aum_feature(train_aum_m10)
train_aum_m11 = extract_aum_feature(train_aum_m11)
train_aum_m12 = extract_aum_feature(train_aum_m12)

train_cunkuan_m10 = extract_cunkuan_feature(train_cunkuan_m10)
train_cunkuan_m11 = extract_cunkuan_feature(train_cunkuan_m11)
train_cunkuan_m12 = extract_cunkuan_feature(train_cunkuan_m12)

train_behavior_merge = pd.merge(train_behavior_m10,train_behavior_m11,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
train_behavior_merge = pd.merge(train_behavior_merge,train_behavior_m12,on='cust_no',how = 'left', suffixes=(None, '_m3'))
train_behavior_concat = pd.concat([train_behavior_m10,train_behavior_m11],axis=0,ignore_index=True)
train_behavior_concat = pd.concat([train_behavior_concat,train_behavior_m12],axis=0,ignore_index=True)

train_aum_merge = pd.merge(train_aum_m10,train_aum_m11,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
train_aum_merge = pd.merge(train_aum_merge,train_aum_m12,on='cust_no',how = 'left', suffixes=(None, '_m3'))
train_aum_concat = pd.concat([train_aum_m10,train_aum_m11],axis=0,ignore_index=True)
train_aum_concat = pd.concat([train_aum_concat,train_aum_m12],axis=0,ignore_index=True)


train_cunkuan_merge = pd.merge(train_cunkuan_m10,train_cunkuan_m11,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
train_cunkuan_merge = pd.merge(train_cunkuan_merge,train_cunkuan_m12,on='cust_no',how = 'left', suffixes=(None, '_m3'))
train_cunkuan_concat = pd.concat([train_cunkuan_m10,train_cunkuan_m11],axis=0,ignore_index=True)
train_cunkuan_concat = pd.concat([train_cunkuan_concat,train_cunkuan_m12],axis=0,ignore_index=True)

train_behavior_concat_agg = agg_numeric(train_behavior_concat,group_var='cust_no',df_name='concat_3m_')
train_aum_concat_agg = agg_numeric(train_aum_concat,group_var='cust_no',df_name='concat_3m_')
train_cunkuan_concat_agg = agg_numeric(train_cunkuan_concat,group_var='cust_no',df_name='concat_3m_')


train_behavior_merge = extract_behavior_merge_feature(train_behavior_merge)
train_aum_merge = extract_aum_merge_feature(train_aum_merge)
train_cunkuan_merge = extract_cunkuan_merge_feature(train_cunkuan_merge)
train_big_event_q4 = extract_big_event(train_big_event_q4)
train_big_event_q4_agg = agg_numeric(train_big_event_q4,group_var='cust_no',df_name='concat_3m_')

train_1 = train_cust_info_q4
print(train_1.shape)
train_y = pd.merge(train_1, y_train_q4,on='cust_no',how='left')
print(train_y.shape,y_train_q4.shape)
train_y = pd.merge(train_y,train_aum_merge,on='cust_no',how = 'left')
print(train_aum_merge.shape)
train_y = pd.merge(train_y,train_aum_concat_agg)
print(train_aum_concat_agg.shape)
train_y = pd.merge(train_y,train_behavior_merge,on='cust_no',how = 'left')
print(train_behavior_merge.shape)
# train_y = pd.merge(train_y,train_behavior_concat_agg,on='cust_no',how='left')
# print(train_behavior_concat_agg.shape)
train_y = pd.merge(train_y,train_cunkuan_merge,on='cust_no',how = 'left')
print(train_cunkuan_merge.shape)
train_y = pd.merge(train_y,train_cunkuan_concat_agg,on='cust_no',how='left')
print(train_cunkuan_concat_agg.shape)
train_y = pd.merge(train_y, train_big_event_q4,on='cust_no',how = 'left')
print(train_big_event_q4.shape)
# train_y = pd.merge(train_y,train_big_event_q4_agg,on='cust_no',how = 'left')
# print(train_big_event_q4_agg.shape)
print(train_y.shape,y_train_q4.shape)
train_y.to_csv(base_save_path + 'Q4_orgmerge_new_1204.csv',index=0)


test_behavior_m3['B6'] = pd.to_datetime(test_behavior_m3['B6'])
test_behavior_m3['B6_day'] = test_behavior_m3['B6'].map(lambda x: x.day)
test_behavior_m3['B6_day_diff'] = (datetime.strptime('20200331', '%Y%m%d') - pd.to_datetime(test_behavior_m3["B6"])).apply(lambda x: x.days)
test_behavior_m1 = extract_behavior_features(test_behavior_m1)
test_behavior_m2 = extract_behavior_features(test_behavior_m2)
test_behavior_m3 = extract_behavior_features(test_behavior_m3)

test_aum_m1 = extract_aum_feature(test_aum_m1)
test_aum_m2 = extract_aum_feature(test_aum_m2)
test_aum_m3 = extract_aum_feature(test_aum_m3)

test_cunkuan_m1 = extract_cunkuan_feature(test_cunkuan_m1)
test_cunkuan_m2 = extract_cunkuan_feature(test_cunkuan_m2)
test_cunkuan_m3 = extract_cunkuan_feature(test_cunkuan_m3)

test_cunkuan_merge = pd.merge(test_cunkuan_m1,test_cunkuan_m2,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
test_cunkuan_merge = pd.merge(test_cunkuan_merge,test_cunkuan_m3,on='cust_no',how = 'left', suffixes=(None, '_m3'))
test_behavior_concat = pd.concat([test_behavior_m1,test_behavior_m2],axis=0,ignore_index=True)
test_behavior_concat = pd.concat([test_behavior_concat,test_behavior_m3],axis=0,ignore_index=True)

test_aum_merge = pd.merge(test_aum_m1,test_aum_m2,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
test_aum_merge = pd.merge(test_aum_merge,test_aum_m3,on='cust_no',how = 'left', suffixes=(None, '_m3'))
test_aum_concat = pd.concat([test_aum_m1,test_aum_m2],axis=0,ignore_index=True)
test_aum_concat = pd.concat([test_aum_concat,test_aum_m3],axis=0,ignore_index=True)

test_behavior_merge = pd.merge(test_behavior_m1,test_behavior_m2,on='cust_no',how = 'left', suffixes=('_m1', '_m2'))
test_behavior_merge = pd.merge(test_behavior_merge,test_behavior_m3,on='cust_no',how = 'left', suffixes=(None, '_m3'))
test_cunkuan_concat = pd.concat([test_cunkuan_m1,test_cunkuan_m2],axis=0,ignore_index=True)
test_cunkuan_concat = pd.concat([test_cunkuan_concat,test_cunkuan_m3],axis=0,ignore_index=True)

test_behavior_concat_agg = agg_numeric(test_behavior_concat,group_var='cust_no',df_name='concat_3m_')
test_aum_concat_agg = agg_numeric(test_aum_concat,group_var='cust_no',df_name='concat_3m_')
test_cunkuan_concat_agg = agg_numeric(test_cunkuan_concat,group_var='cust_no',df_name='concat_3m_')

test_behavior_merge = extract_behavior_merge_feature(test_behavior_merge)
test_aum_merge = extract_aum_merge_feature(test_aum_merge)
test_cunkuan_merge = extract_cunkuan_merge_feature(test_cunkuan_merge)
test_big_event_q1 = extract_big_event(test_big_event_q1)
test_big_event_q1_agg = agg_numeric(test_big_event_q1,group_var='cust_no',df_name='concat_3m_')
# test_cust_info_q1["age_class"] = test_cust_info_q1['I2'].map(lambda x: int(x/5))
test_1 = test_cust_info_q1
print(test_1.shape)
test_1 = pd.merge(test_1,test_cust_avli_q1,on='cust_no',how = 'right')
print(test_1.shape)
test_1 = pd.merge(test_1,test_aum_merge,on='cust_no',how = 'left')
print(test_aum_merge.shape)
test_1 = pd.merge(test_1,test_aum_concat_agg,on='cust_no',how='left')
print(test_aum_concat_agg.shape)
test_1 = pd.merge(test_1,test_behavior_merge,on='cust_no',how = 'left')
print(test_behavior_merge.shape)
# test_1 = pd.merge(test_1,test_behavior_concat_agg,on='cust_no',how='left')
# print(test_behavior_concat_agg.shape)
test_1 = pd.merge(test_1,test_cunkuan_merge,on='cust_no',how = 'left')
print(test_cunkuan_merge.shape)
test_1 = pd.merge(test_1,test_cunkuan_concat_agg,on='cust_no',how = 'left')
print(test_cunkuan_concat_agg.shape)
test_1 = pd.merge(test_1,test_big_event_q1,on='cust_no',how = 'left')
print(test_big_event_q1.shape)
# test_1 = pd.merge(test_1,test_big_event_q1_agg,on='cust_no',how = 'left')
# print(test_big_event_q1_agg.shape)
print(test_1.shape)
test_1.to_csv(base_save_path + 'Q1_orgmerge_1204.csv',index=0)

