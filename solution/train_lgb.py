# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:33:52 2018

@author: NURBS
"""

import os
import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics


print('开始处理特征......')

train_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# =============================================================================
# used_feature = ['create_count', 'create_day_diff_mean', 'create_day_diff_std', 'create_day_diff_max',
#                 'create_day_diff_min', 'create_mode', 'last_day_cut_max_day',
#                 'register_type', 'device_type', 'register_day_cut_max_day',
#                 'launch_count', 'launch_day_diff_mean', 'launch_day_diff_std',
#                 'launch_day_diff_max', 'launch_day_diff_min', 'launch_day_diff_kur',
#                 'launch_day_diff_ske', 'launch_day_diff_last', 'launch_day_cut_max_day',
#                 'activity_count',
#                 'activity_day_diff_mean',
#                 'activity_day_diff_std',
#                 'activity_day_diff_max', 'activity_day_diff_min', 'activity_day_diff_kur',
#                 'activity_day_diff_ske',
#                 'activity_day_diff_last',
#                 '0_page_count', '1_page_count', '2_page_count', '3_page_count', '4_page_count',
#                 '0_page_count_div_sum', '1_page_count_div_sum', '2_page_count_div_sum',
#                 '3_page_count_div_sum', '4_page_count_div_sum',
#                 '0_action_count',
#                 '1_action_count', '2_action_count', '3_action_count', '4_action_count',
#                 '5_action_count', '0_action_count_div_sum', '1_action_count_div_sum',
#                 '2_action_count_div_sum', '3_action_count_div_sum',
#                 '4_action_count_div_sum', '5_action_count_div_sum',
#                 'video_id_mode', 'author_id_mode', 'activity_count_mean',
#                 'activity_count_std', 'activity_count_max', 'activity_count_min',
#                 'activity_count_kur', 'activity_count_ske', 'activity_count_last',
#                 'activity_diff_count_mean', 'activity_diff_count_std', 'activity_diff_count_max',
#                 'activity_diff_count_min', 'activity_diff_count_kur', 'activity_diff_count_ske',
#                 'activity_diff_count_last', 'activity_page0_mean', 'activity_page0_std',
#                 'activity_page0_max', 'activity_page0_min', 'activity_page0_kur', 'activity_page0_ske',
#                 'activity_page0_last', 'activity_page1_mean', 'activity_page1_std', 'activity_page1_max',
#                 'activity_page1_min', 'activity_page1_kur', 'activity_page1_ske', 'activity_page1_last',
#                 'activity_page2_mean', 'activity_page2_std', 'activity_page2_max', 'activity_page2_min',
#                 'activity_page2_kur', 'activity_page2_ske', 'activity_page2_last', 'activity_page3_mean',
#                 'activity_page3_std', 'activity_page3_max', 'activity_page3_min', 'activity_page3_kur',
#                 'activity_page3_ske', 'activity_page3_last', 'activity_page4_mean', 'activity_page4_std',
#                 'activity_page4_max', 'activity_page4_min', 'activity_page4_kur', 'activity_page4_ske',
#                 'activity_page4_last', 'activity_type0_mean', 'activity_type0_std', 'activity_type0_max',
#                 'activity_type0_min', 'activity_type0_kur', 'activity_type0_ske', 'activity_type0_last',
#                 'activity_type1_mean', 'activity_type1_std', 'activity_type1_max', 'activity_type1_min',
#                 'activity_type1_kur', 'activity_type1_ske', 'activity_type1_last', 'activity_type2_mean',
#                 'activity_type2_std', 'activity_type2_max', 'activity_type2_min', 'activity_type2_kur',
#                 'activity_type2_ske', 'activity_type2_last', 'activity_type3_mean', 'activity_type3_std',
#                 'activity_type3_max', 'activity_type3_min', 'activity_type3_kur', 'activity_type3_ske',
#                 'activity_type3_last', 'activity_type4_mean', 'activity_type4_std', 'activity_type4_max',
#                 'activity_type4_min', 'activity_type4_kur', 'activity_type4_ske', 'activity_type4_last',
#                 'activity_type5_mean', 'activity_type5_std', 'activity_type5_max', 'activity_type5_min',
#                 'activity_type5_kur', 'activity_type5_ske', 'activity_type5_last', 'activity_day_cut_max_day',
#                 'max_activity_day',
#                  'create_sub_register', 'activity_sub_register', 'launch_sub_register',
#                 ]
# used_feature = np.array(used_feature)
# print(used_feature)
# importance_feature = [21,60,54,71,106,44,50,27,33,58,43,11,19,35,64,32,45,9,37,143,142,10,7,18,8]
# used_feature = used_feature[np.array(importance_feature)]
# print(used_feature)
# train_feature = train[used_feature]
# test_feature = test[used_feature]
# label = train['label']
# =============================================================================

train_feature = train.drop(['user_id','label'],axis=1)
train_label = train['label']
test_feature = test.drop(['user_id'],axis=1)

# 切分训练
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2,random_state=1017)

print('特征处理完毕......')


###################### lgb ##########################
import lightgbm as lgb

print('载入数据......')
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)


print('开始训练......')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc', 'binary_logloss'}
}



gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval
                )
gbm.save_model('model/lgb_model.txt')

temp = gbm.predict(X_test)
temp[temp>0.5]=1
temp[temp<0.5]=0
print('结果：' + str(sklearn.metrics.f1_score(Y_test, temp)))
print('特征重要性：'+ str(list(gbm.feature_importance())))


########################## 保存结果 ############################
pre = gbm.predict(test_feature)
df_result = pd.DataFrame()
df_result['user_id'] = test['user_id']
df_result['result'] = pre
df_result.to_csv('result/lgb_result.csv', index=False)
pre[pre >= 0.5]=1
pre[pre < 0.5]=0
pre = list(map(int,pre))
print('为1的个数：' + str(len(np.where(np.array(pre)==1)[0])))
print('为0的个数：' + str(len(np.where(np.array(pre)==0)[0])))







