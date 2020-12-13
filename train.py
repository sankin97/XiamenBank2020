#!/usr/bin/env python
# coding: utf-8

# In[30]:


# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
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
import lightgbm as lgb
import catboost as cbt
import pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score, cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import gc
from scipy.misc import derivative
def agg_numeric(df, group_var, df_name):
    for col in df:
        if col != group_var and 'cust_no' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    agg = numeric_df.groupby(group_var).agg(['median', 'mean', 'max', 'min', 'std']).reset_index()

    columns = [group_var]

    for var in agg.columns.levels[0]:
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:

                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

train_base_path = "./data/x_train/"
test_base_path = "./data/x_test/"
base_save_path = "./data/tmp/"

train_3 = pd.read_csv(base_save_path + 'Q3_orgmerge_new_1127.csv')
train_4 = pd.read_csv(base_save_path + 'Q4_orgmerge_new_1127.csv')
test_1 = pd.read_csv(base_save_path + 'Q1_orgmerge_new_1127.csv')
train_cust_avli_q4=pd.read_csv(train_base_path + 'cust_avli_Q4.csv')
train_cust_avli_q3=pd.read_csv(train_base_path + 'cust_avli_Q3.csv')
test_cust_avli_q1=pd.read_csv(test_base_path + 'cust_avli_Q1.csv')

print(train_3.shape)
print(train_4.shape)
print(test_1.shape)

train_concat = pd.concat((train_4,train_3),axis=0,ignore_index=True)
train_concat_new = agg_numeric(train_concat.drop(columns = ['label']),group_var='cust_no',df_name='all_concat')

train_3["label_pre"] = train_3['label']
train_3 = train_3.drop(columns = ['label'])
# train_concat = pd.concat((train_4,train_3),axis=0,ignore_index=True)
# train_concat_new = agg_numeric(train_concat.drop(columns = ['label']),group_var='cust_no',df_name='all_concat')
train = pd.merge(train_4, train_3,on='cust_no', how='left')
print('train:',train.shape)
train = pd.merge(train,train_concat_new,on='cust_no', how='left')
print('train_concat_new:',train_concat_new.shape)
train = pd.merge(train, train_cust_avli_q4, on='cust_no', how='right')
print('train:',train.shape)

train_4["label_pre"] = train_4['label']
train_4 = train_4.drop(columns=['label'])
test_concat = pd.concat((test_1,train_4),axis=0,ignore_index=True)
test_concat_new = agg_numeric(test_concat,group_var='cust_no',df_name='all_concat')
print('test_concat_new:',test_concat_new.shape)
test = pd.merge(test_1, train_4, on='cust_no', how='left')
print('test:',test.shape)
test = pd.merge(test,test_concat_new, on='cust_no', how='left')
print('test+concat:',test.shape)
test = pd.merge(test, test_cust_avli_q1, on='cust_no', how='right')
# train = pd.concat([train_3,train_4],axis=0,ignore_index=True)
print(train.shape, test.shape)
del train_3
del train_4
del test_1
def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
	a,g = alpha, gamma
	y_true = dtrain.label
	def fl(x,t):
		p = 1/(1+np.exp(-x))
		return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
	partial_fl = lambda x: fl(x, y_true)
	grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
	hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
	return grad, hess
def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
	a,g = alpha, gamma
	y_true = dtrain.label
	p = 1/(1+np.exp(-y_pred))
	loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
	return 'focal_loss', np.mean(loss), False
def missing_values_table(df):
        # 总的缺失值
        mis_val = df.isnull().sum()
        
        # 缺失值占比
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # 将上述值合并成表
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # 重命名列名
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # 按缺失值占比降序排列
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # 显示结果
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        return mis_val_table_ren_columns


def custom_kappa_eval(y_true, y_pred):
    y_pred = np.argmax(y_pred.reshape(3, -1), axis=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    return "Kappa", kappa, True


# important_features = list(pd.read_csv("./data/tmp/feature_importance_new_concat_1128_zj.csv")['feature'].values)
# important_features = important_features[:500]

important_features = list(pd.read_csv("./data/tmp/feature_importance_new_1204.csv")['feature'].values)
important_features = important_features[:300]
def model_new_lgb(features, test_features, encoding='ohe', n_folds = 5):
    #提取ID
    train_ids = features['cust_no']
    test_ids = test_features['cust_no']
    
    # 提取训练集的结果
    labels = features['label']
    # 移除ID和label
    features = features.drop(columns=['cust_no', 'label'])
    test_features = test_features.drop(columns=['cust_no'])
    one_hot = OneHotEncoder(3,sparse=False)
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        label_encoder = LabelEncoder()
        
        cat_indices = []
        
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                cat_indices.append(i)
    
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    features = features[important_features]
    test_features = test_features[important_features]
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    feature_names = list(features.columns)
    
    features = np.array(features)
    test_features = np.array(test_features)

    # features[np.isnan(features)] = -1000000
    # features[np.where(features >= np.finfo(np.float64).max)] = -1000000
    # test_features[np.isnan(test_features)] = -1000000
    # test_features[np.where(test_features >= np.finfo(np.float64).max)] = -1000000

    k_fold = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
    feature_importance_values = np.zeros(len(feature_names))
    
    test_predictions = np.zeros((test_features.shape[0],3))
    
    out_of_fold = np.zeros((features.shape[0],3))
    
    valid_scores = []
    train_scores = []
    cat_feats = 'auto'
    for train_indices, valid_indices in k_fold.split(features, labels):
        train_x, train_y = features[train_indices], labels[train_indices]
        valid_x, valid_y = features[valid_indices], labels[valid_indices]
        train_y += 1
        # focal_loss = lambda x,y: focal_loss_lgb(x, y, alpha=0.25,  gamma=1., num_class= 3)
        # eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, alpha=0.25, gamma=1., num_class=3)
        valid_y += 1
        
        # train_x, train_y = SMOTE().fit_sample(train_x, train_y)
        lgtrain = lgb.Dataset(train_x, train_y,
                        categorical_feature = cat_feats)
        lgvalid = lgb.Dataset(valid_x, valid_y,
                        categorical_feature = cat_feats)
 
        print('get lgb train valid dataset end')
        lgb_params =  {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class':3,
            'metric': 'multi_logloss',
            "learning_rate": 0.01,
          
            'min_data_in_leaf':20,
            "feature_fraction":0.8, 
            "bagging_fraction":0.8, 
            'num_leaves': 64,
            'min_child_samples': 20,
            'nthread': 40,
            'random_state':50,
            # 'is_unbalance':True
        }
        clf = lgb.train(
            lgb_params,
            lgtrain,
            num_boost_round=2000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train','valid'],
            early_stopping_rounds=100,
            verbose_eval=200
        )
        feature_importance_values +=  clf.feature_importance() / k_fold.n_splits
        test_predictions += clf.predict(test_features, num_iteration = clf.best_iteration)[:, :] / k_fold.n_splits
        out_of_fold[valid_indices] = clf.predict(valid_x, num_iteration=clf.best_iteration)[:, :] 
        valid_score = accuracy_score(valid_y, np.argmax(out_of_fold[valid_indices], axis=1))
        train_score = accuracy_score(train_y, np.argmax(clf.predict(train_x,num_iteration=clf.best_iteration)[:, :], axis=1))
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        gc.enable()
        del train_x, valid_x
        gc.collect()
    y_pred = np.argmax(test_predictions, axis=1)-1
    print(y_pred)
    submission = pd.DataFrame({'cust_no': test_ids, 'label': y_pred})
    test_predictions = pd.DataFrame({'cust_no': test_ids, '-1': test_predictions[:, 0], '0': test_predictions[:, 1],
                                     '1': test_predictions[:, 2]})
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    valid_auc = accuracy_score(labels+1, np.argmax(out_of_fold, axis=1))
    valid_scores.append(valid_auc)

    train_scores.append(np.mean(train_scores))

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics, test_predictions
   

def model_lgb(features, test_features, encoding='ohe', n_folds = 5):
    #提取ID
    train_ids = features['cust_no']
    test_ids = test_features['cust_no']
    
    # 提取训练集的结果
    labels = features['label']
    # 移除ID和target
    features = features.drop(columns=['cust_no', 'label'])
    test_features = test_features.drop(columns=['cust_no'])
    one_hot = OneHotEncoder(3,sparse=False)
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        label_encoder = LabelEncoder()
        
        cat_indices = []
        
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                cat_indices.append(i)
    
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    features = features[important_features]
    test_features = test_features[important_features]
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    feature_names = list(features.columns)
    
    features = np.array(features)
    test_features = np.array(test_features)

    features[np.isnan(features)] = -1000000
    features[np.where(features >= np.finfo(np.float64).max)] = -1000000
    test_features[np.isnan(test_features)] = -1000000
    test_features[np.where(test_features >= np.finfo(np.float64).max)] = -1000000

    k_fold = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
    feature_importance_values = np.zeros(len(feature_names))
    
    test_predictions = np.zeros((test_features.shape[0],3))
    
    out_of_fold = np.zeros((features.shape[0],3))
    # print(out_of_fold)
    
    valid_scores = []
    train_scores = []
    
    for train_indices, valid_indices in k_fold.split(features, labels):
        
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        train_labels += 1
        # focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 1., 3)
        # eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 1., 3)

        train_features, train_labels = SMOTE().fit_sample(train_features, train_labels)
        train_features, train_labels = SMOTE().fit_sample(train_features, train_labels)

        valid_labels += 1
        #建模
        model = lgb.LGBMClassifier(n_estimators=2000, objective = 'multiclass',num_class=3,

                                   class_weight='balanced',
                                   learning_rate=0.05,
                                   boosting_type='gbdt',num_leaves=63, min_data_in_leaf=20,
                                   feature_fraction=0.8, bagging_fraction=0.8, n_jobs=44, random_state=50) #, device='gpu'
        
        # 训练模型 eval_metric=lambda y_true, y_pred: [custom_kappa_eval(y_true, y_pred)]
        model.fit(train_features, train_labels, 
                fobj=focal_loss,
                  feval = eval_error,
                #   eval_metric="multi_error",
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds=100, verbose = 200)
        
        best_iteration = model.best_iteration_
        
        # 特征重要性
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # 做预测
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, :] / k_fold.n_splits
        
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, :]
        
        valid_score = accuracy_score(valid_labels, np.argmax(out_of_fold[valid_indices], axis=1))
        train_score = accuracy_score(train_labels, np.argmax(model.predict_proba(train_features)[:, :], axis=1))
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    y_pred = np.argmax(test_predictions, axis=1)-1
    print(y_pred)
    submission = pd.DataFrame({'cust_no': test_ids, 'label': y_pred})
    test_predictions = pd.DataFrame({'cust_no': test_ids, '-1': test_predictions[:, 0], '0': test_predictions[:, 1],
                                     '1': test_predictions[:, 2]})
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    valid_auc = accuracy_score(labels+1, np.argmax(out_of_fold, axis=1))
    valid_scores.append(valid_auc)

    train_scores.append(np.mean(train_scores))

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics, test_predictions


def model_xgb(features, test_features, encoding='ohe', n_folds=5):
    # 提取ID
    train_ids = features['cust_no']
    test_ids = test_features['cust_no']

    # 提取训练集的结果
    labels = features['label']
    # 移除ID和target
    features = features.drop(columns=['cust_no', 'label'])
    test_features = test_features.drop(columns=['cust_no'])
    one_hot = OneHotEncoder(3, sparse=False)

    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        features, test_features = features.align(test_features, join='inner', axis=1)

        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        label_encoder = LabelEncoder()

        cat_indices = []

        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                cat_indices.append(i)

    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    # features = features[important_features]
    # test_features = test_features[important_features]
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    feature_names = list(features.columns)

    features = np.array(features)
    test_features = np.array(test_features)

    features[np.isnan(features)] = -1000000
    features[np.where(features >= np.finfo(np.float64).max)] = -1000000
    test_features[np.isnan(test_features)] = -1000000
    test_features[np.where(test_features >= np.finfo(np.float64).max)] = -1000000

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
        train_features, train_labels = SMOTE().fit_sample(train_features, train_labels)
        # 建模
        model = xgb.XGBClassifier(random_seed=2020, objective='multiclass',num_class=3, nthread=24, n_estimators=1000,
                                  eval_metric='auc', learning_rate=0.1, max_depth=6,early_stopping_rounds=1500,
                                  feval=custom_kappa_eval, verbose=0, verbose_eval=0)

        # 训练模型 eval_metric=lambda y_true, y_pred: [custom_kappa_eval(y_true, y_pred)]
        model.fit(train_features, train_labels, eval_metric="merror",
                  eval_set=[(train_features, train_labels), (valid_features, valid_labels)],
                  early_stopping_rounds=20, verbose=True)

        # 特征重要性
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # 做预测
        test_predictions += model.predict_proba(test_features)[:, :] / k_fold.n_splits

        out_of_fold[valid_indices] = model.predict_proba(valid_features)[:, :]

        valid_score = accuracy_score(valid_labels, np.argmax(out_of_fold[valid_indices], axis=1))
        train_score = accuracy_score(train_labels, np.argmax(model.predict_proba(train_features)[:, :], axis=1))

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    y_pred = np.argmax(test_predictions, axis=1) - 1
    test_predictions = pd.DataFrame({'cust_no': test_ids, '-1': test_predictions[:, 0], '0': test_predictions[:, 1],
                                     '1': test_predictions[:, 2]})
    print(y_pred)
    submission = pd.DataFrame({'cust_no': test_ids, 'label': y_pred})

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    valid_auc = accuracy_score(labels + 1, np.argmax(out_of_fold, axis=1))
    valid_scores.append(valid_auc)

    train_scores.append(np.mean(train_scores))

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics, test_predictions


def model_cbt(features, test_features, encoding='ohe', n_folds=5):
    # 提取ID
    train_ids = features['cust_no']
    test_ids = test_features['cust_no']

    # 提取训练集的结果
    labels = features['label']
    # 移除ID和target
    features = features.drop(columns=['cust_no', 'label'])
    test_features = test_features.drop(columns=['cust_no'])
    one_hot = OneHotEncoder(3, sparse=False)

    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        features, test_features = features.align(test_features, join='inner', axis=1)

        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        label_encoder = LabelEncoder()

        cat_indices = []

        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                cat_indices.append(i)

    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    features = features[important_features]
    test_features = test_features[important_features]
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    feature_names = list(features.columns)

    features = np.array(features)
    test_features = np.array(test_features)

    features[np.isnan(features)] = -1000000
    features[np.where(features >= np.finfo(np.float64).max)] = -1000000
    test_features[np.isnan(test_features)] = -1000000
    test_features[np.where(test_features >= np.finfo(np.float64).max)] = -1000000

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
        train_features, train_labels = SMOTE().fit_sample(train_features, train_labels)
        # 建模
        model = cbt.CatBoostClassifier(random_seed=2020, iterations=1500, learning_rate=0.1, max_depth=11, l2_leaf_reg=1,
                                   verbose=1, early_stopping_rounds=20, task_type='CPU', eval_metric='Kappa')
        # 训练模型 eval_metric=lambda y_true, y_pred: [custom_kappa_eval(y_true, y_pred)]
        model.fit(train_features, train_labels,
                  eval_set=[(valid_features, valid_labels)],
                  early_stopping_rounds=20, verbose=True)

        # 特征重要性
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # 做预测
        test_predictions += model.predict_proba(test_features)[:, :] / k_fold.n_splits

        out_of_fold[valid_indices] = model.predict_proba(valid_features)[:, :]

        valid_score = accuracy_score(valid_labels, np.argmax(out_of_fold[valid_indices], axis=1))
        train_score = accuracy_score(train_labels, np.argmax(model.predict_proba(train_features)[:, :], axis=1))

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    y_pred = np.argmax(test_predictions, axis=1) - 1
    test_predictions = pd.DataFrame({'cust_no': test_ids, '-1': test_predictions[:, 0], '0': test_predictions[:, 1],
                                     '1': test_predictions[:, 2]})
    print(y_pred)
    submission = pd.DataFrame({'cust_no': test_ids, 'label': y_pred})

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    fold_names = list(range(n_folds))
    fold_names.append('overall')

    valid_auc = accuracy_score(labels + 1, np.argmax(out_of_fold, axis=1))
    valid_scores.append(valid_auc)

    train_scores.append(np.mean(train_scores))

    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics, test_predictions


def plot_feature_importances(df):
    df = df.sort_values('importance', ascending=False).reset_index()

    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df.to_csv(base_save_path + 'feature_importance_new_1204.csv',index=0)
    return df


def extract_feature(df):
    def sub_extract_1(df, name1, name2):
        df[name1+"+"+name2] = df[name1] + df[name2]
        df[name1+"-"+name2] = df[name1] - df[name2]
        df[name1+"/"+name2] = df[name1] / (df[name2] + 0.000001)
        return df

    def sub_extract_2(df, name):
        df["{}_all_sum".format(name)] = df[
            ["{}_m1_y".format(name), "{}_m2_y".format(name), "{}_y".format(name), "{}_m1_x".format(name),
             "{}_m2_x".format(name), "{}_x".format(name)]].sum(axis=1)
        df["{}_all_min".format(name)] = df[
            ["{}_m1_y".format(name), "{}_m2_y".format(name), "{}_y".format(name), "{}_m1_x".format(name),
             "{}_m2_x".format(name), "{}_x".format(name)]].min(axis=1)
        df["{}_all_std".format(name)] = df[
            ["{}_m1_y".format(name), "{}_m2_y".format(name), "{}_y".format(name), "{}_m1_x".format(name),
             "{}_m2_x".format(name), "{}_x".format(name)]].std(axis=1)
        df["{}_all_mean".format(name)] = df[
            ["{}_m1_y".format(name), "{}_m2_y".format(name), "{}_y".format(name), "{}_m1_x".format(name),
             "{}_m2_x".format(name), "{}_x".format(name)]].mean(axis=1)
        df["{}_m4-m3".format(name)] = df["{}_m1_x".format(name)]-df["{}_y".format(name)]
        df["{}_m4/m3".format(name)] = df["{}_m1_x".format(name)] / (df["{}_y".format(name)]+0.000001)
        return df

    df = sub_extract_1(df, "B7_x", "B7_y")
    df = sub_extract_1(df, "X_sum_m1_m2_m3_sum_x", "X_sum_m1_m2_m3_sum_y")
    df = sub_extract_1(df, "X_sum_m1_m2_m3_min_x", "X_sum_m1_m2_m3_min_y")
    df = sub_extract_1(df, "X_sum_m1_m2_m3_max_x", "X_sum_m1_m2_m3_max_y")
    df = sub_extract_1(df, "B1/B2_m1_m2_m3_max/min_x", "B1/B2_m1_m2_m3_max/min_y")
    df = sub_extract_1(df, "X_sum/X7_m3-m1_x", "X_sum/X7_m3-m1_y")
    df = sub_extract_1(df, "X_sum/X7_m3-m2_x", "X_sum/X7_m3-m2_y")
    df = sub_extract_1(df, "B1-B2-B4_m3/m2_x", "B1-B2-B4_m3/m2_y")
    df = sub_extract_1(df, "B1-B2-B4_m2/m1_x", "B1-B2-B4_m2/m1_y")
    df = sub_extract_1(df, "X3_m1_m2_m3_max/min_x", "X3_m1_m2_m3_max/min_y")
    df = sub_extract_1(df, "B3/B2/(B5/B4)_m1_m2_m3_max/min_x", "B3/B2/(B5/B4)_m1_m2_m3_max/min_y")

    df = sub_extract_2(df, "X3")
    df = sub_extract_2(df, "X_sum")
    df = sub_extract_2(df, "C1/C2")
    df = sub_extract_2(df, "B1/B2")
    df = sub_extract_2(df, "C1")
    df = sub_extract_2(df, "C2")
    df = sub_extract_2(df, "X_sum/X7")
    df = sub_extract_2(df, "B1-B2-B4")
    df = sub_extract_2(df, "X1+X2+X3")
    df = sub_extract_2(df, "B3/B2/(B5/B4)")

    for i in range(1, 19):
        if i != 15 and i != 17:
            df["B6_x_E{}_x_day_diff".format(i)] = (pd.to_datetime(df["B6_x"]) - pd.to_datetime(df["E{}_x".format(i)])).apply(lambda x: x.days)
    df = df.drop(columns=["B6_x", "B6_y"])
    df = df.drop(columns=['E{}_x'.format(i) for i in range(1, 19) if i != 15 and i != 17])
    df = df.drop(columns=['E{}_y'.format(i) for i in range(1, 19) if i != 15 and i != 17])
    return df



train = extract_feature(train)
test = extract_feature(test)
submission, fi, metrics, test_predictions = model_new_lgb(train, test)
print('Baseline metrics')
print(metrics)
gc.collect
print(submission)
submission.to_csv('./result_lgb_concat_all_1207_new_300.csv',index=0)
test_predictions.to_csv('./result_prob_concat_all_1207_new_300.csv',index=0)