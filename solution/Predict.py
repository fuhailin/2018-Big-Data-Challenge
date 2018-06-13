# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics

print('开始处理特征......')

train_path = 'train.csv'
test_path = 'test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# used_feature = ['create_count', 'create_day_diff_mean', 'create_day_diff_std', 'create_day_diff_max',]
# used_feature = np.array(used_feature)
# print(used_feature)
# importance_feature = [21, 60, 54, 71, 106, 44, 50, 27, 33, 58, 43, 11, 19, 35, 64, 32, 45, 9, 37, 143, 142, 10, 7, 18, 8]
# used_feature = used_feature[np.array(importance_feature)]
# print(used_feature)
train_feature = train
test_feature = test
label = train['label']

# 切分训练
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, label, test_size=0.2, random_state=1017)
# train_feature = X_train
# label = Y_train

print('特征处理完毕......')

###################### lgb ##########################
import lightgbm as lgb

print('载入数据......')
lgb_train = lgb.Dataset(train_feature, label)
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
                num_boost_round=58,
                valid_sets=lgb_eval
                )
gbm.save_model('lgb_model.txt')

temp = gbm.predict(X_test)
temp[temp > 0.5] = 1
temp[temp < 0.5] = 0
print('结果：' + str(sklearn.metrics.f1_score(Y_test, temp)))
print('特征重要性：' + str(list(gbm.feature_importance())))

########################## 保存结果 ############################
pre = gbm.predict(test_feature)
df_result = pd.DataFrame()
df_result['user_id'] = test['user_id']
df_result['result'] = pre
df_result.to_csv('lgb_result.csv', index=False)
pre[pre >= 0.5] = 1
pre[pre < 0.5] = 0
pre = map(int, pre)
print('为1的个数：' + str(len(np.where(np.array(pre) == 1)[0])))
print('为0的个数：' + str(len(np.where(np.array(pre) == 0)[0])))
