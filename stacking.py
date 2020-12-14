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
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import gc
#忽略警告提示
import warnings
warnings.filterwarnings('ignore')

train_base_path = "./data/x_train/"
test_base_path = "./data/x_test/"
base_save_path = "./data/tmp/"


y_train_q3 = pd.read_csv('./data/y_train_3/y_Q3_3.csv')
y_train_q4 = pd.read_csv('./data/y_train_3/y_Q4_3.csv')
test_feature = None
train_feature = None

for feature_name in ['100', '200', '400', "500", "600", "800", "1000", "1300", "1500", "2000"]: #
    for model_name in ["lgb"]:
        test_pro_tmp = pd.read_csv('./results/CV/result_prob_{}_{}.csv'.format(model_name, feature_name))
        # y_test.reset_index(drop=True, inplace=True)
        # y_test.set_index(['cust_no'], inplace=True)
        train_pro_tmp = pd.read_csv('./results/CV/train_prob_{}_{}.csv'.format(model_name, feature_name))
        if test_feature is None:
            test_feature = test_pro_tmp
        else:
            test_feature = pd.merge(test_feature, test_pro_tmp, on='cust_no', how='left')

        if train_feature is None:
            train_feature = train_pro_tmp
        else:
            train_feature = pd.merge(train_feature, train_pro_tmp, on='cust_no', how='left')


def model_lgb(features, test_features, n_folds=5):
    # 提取ID
    train_ids = features['cust_no']
    test_ids = test_features['cust_no']

    # 提取训练集的结果
    labels = features['label']
    # 移除ID和target
    features = features.drop(columns=['cust_no', 'label'])
    test_features = test_features.drop(columns=['cust_no'])

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    feature_names = list(features.columns)

    features = np.array(features)
    test_features = np.array(test_features)

    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=50)

    feature_importance_values = np.zeros(len(feature_names))

    test_predictions = np.zeros((test_features.shape[0], 3))

    out_of_fold = np.zeros((features.shape[0], 3))
    # print(out_of_fold)

    valid_scores = []
    train_scores = []

    for train_indices, valid_indices in k_fold.split(features, labels):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        train_labels += 1

        valid_labels += 1
        # 建模
        model = lgb.LGBMClassifier(n_estimators=500, objective='multiclass', num_class=3,
                                    learning_rate=0.01,
                                   boosting_type='gbdt', n_jobs=44,
                                   random_state=50)  # , device='gpu'

        # 训练模型 eval_metric=lambda y_true, y_pred: [custom_kappa_eval(y_true, y_pred)]
        model.fit(train_features, train_labels, eval_metric="multi_logloss",
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], early_stopping_rounds=50, verbose=200)

        best_iteration = model.best_iteration_
        # 特征重要性
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        # 做预测
        test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, :] / k_fold.n_splits

        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, :]

        valid_score = accuracy_score(valid_labels, np.argmax(out_of_fold[valid_indices], axis=1))
        train_score = accuracy_score(train_labels, np.argmax(model.predict_proba(train_features)[:, :], axis=1))

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    y_pred = np.argmax(test_predictions, axis=1) - 1
    print(y_pred)
    submission = pd.DataFrame({'cust_no': test_ids, 'label': y_pred})
    test_predictions = pd.DataFrame({'cust_no': test_ids, '-1': test_predictions[:, 0], '0': test_predictions[:, 1],
                                     '1': test_predictions[:, 2]})
    train_predictions = pd.DataFrame({'cust_no': train_ids, '-1': out_of_fold[:, 0], '0': out_of_fold[:, 1],
                                      '1': out_of_fold[:, 2]})
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    valid_auc = accuracy_score(labels + 1, np.argmax(out_of_fold, axis=1))
    valid_scores.append(valid_auc)

    train_scores.append(np.mean(train_scores))

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics, test_predictions, train_predictions


submission, fi, metrics, test_predictions, train_predictions = model_lgb(train_feature, test_feature)
print(metrics)
print(submission)
test_predictions.to_csv('./result_prob_lgb_stacking_2.csv',index=0)


def stacking(features, test_features):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import StackingClassifier

    train_ids = features['cust_no']
    test_ids = test_features['cust_no']

    # 提取训练集的结果
    labels = features['label']
    labels += 1
    # 移除ID和target
    features = features.drop(columns=['cust_no', 'label'])
    test_features = test_features.drop(columns=['cust_no'])

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    feature_names = list(features.columns)

    features = np.array(features)
    test_features = np.array(test_features)
    clf = LogisticRegression(max_iter=500)
    clf.fit(features, labels)
    print(clf.score(features, labels))
    test_predictions = clf.predict_proba(test_features)
    test_predictions = pd.DataFrame({'cust_no': test_ids, '-1': test_predictions[:, 0], '0': test_predictions[:, 1],
                                     '1': test_predictions[:, 2]})
    return test_predictions


# test_predictions = stacking(train_feature, test_feature)
# test_predictions.to_csv('./result_prob_lgb_stacking_3.csv',index=0)


print("*" * 10)
print(np.sum(y_train_q3['label']==1)/len(y_train_q3))
print(np.sum(y_train_q3['label']==0)/len(y_train_q3))
print(np.sum(y_train_q3['label']==-1)/len(y_train_q3))
print("*"*10)
print(np.sum(y_train_q4['label']==1)/len(y_train_q4))
print(np.sum(y_train_q4['label']==0)/len(y_train_q4))
print(np.sum(y_train_q4['label']==-1)/len(y_train_q4))


print("*"*10)
y_test = pd.read_csv('./result_prob_lgb_stacking.csv')
y_test["1"] *= 0.60
y_test["0"] *= 1.12
y_test["-1"] *= 1.35
y_test["label"] = np.argmax(y_test[['-1', '0', '1']].values, axis=1) - 1
print(np.sum(y_test['label'] == 1) / len(y_test))
print(np.sum(y_test['label'] == 0) / len(y_test))
print(np.sum(y_test['label'] == -1) / len(y_test))
y_test[["cust_no", "label"]].to_csv("result_prob_stacking_after_processed.csv", index=False)
