# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

test_1 = pd.read_csv('/home/sankin/result_prob_lgb_concat_all_ensemble_3m_1201_49475.csv')
test_4 = pd.read_csv('/home/sankin/result_prob_lgb_concat_all_ensemble_3m_1204_49.552.csv')
test_2 = pd.read_csv('/home/sankin/result_prob_concat_all_300_1205_nolambda_2_493387_single.csv')
test_3 = pd.read_csv('/home/sankin/result_prob_lgb_concat_all_ensemble_2m_49.35.csv')
test_5 = pd.read_csv('/home/sankin/result_prob_lgb_stacking_491_1208_2.csv')

print(np.sum(test_1['label'] == 1) / len(test_1))
print(np.sum(test_1['label'] == 0) / len(test_1))
print(np.sum(test_1['label'] == -1) / len(test_1))
print('*'*10)
print('*'*10)
print(np.sum(test_5['label'] == 1) / len(test_3))
print(np.sum(test_5['label'] == 0) / len(test_3))
print(np.sum(test_5['label'] == -1) / len(test_3))
all_test  = pd.merge(test_1,test_2,how='left',on='cust_no')
all_test  = pd.merge(all_test,test_3,how='left',on='cust_no')
all_test  = pd.merge(all_test,test_4,how='left',on='cust_no')
all_test  = pd.merge(all_test,test_5,how='left',on='cust_no')

temp  = all_test.mode(axis=1)
temp  = temp.fillna(-3)

temp2 = temp.rename(columns={0:'target',1:'org'})

all_test['label'] =temp2['target']
all_test['label_org'] =temp2['org']
# all_test['label'] = all_test.apply(lambda x : int(max(x['label'],0)) if x['label']==-1 and x['label_org']==1 else x['label'] ,axis=1)
# all_test['label'] = all_test.apply(lambda x : int(max(x['label'],x['label_org'])) if x['label']==-1 and x['label_org']==1 else x['label'],axis=1)
y_test=all_test.drop(columns=['label_x', 'label_y','label_org'])
print('*'*10)
print(np.sum(y_test['label'] == 1) / len(y_test))
print(np.sum(y_test['label'] == 0) / len(y_test))
print(np.sum(y_test['label'] == -1) / len(y_test))
y_test.to_csv('./all_test_submit_1210_5m_2.csv',index=0)
