import os
import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics
from datetime import datetime

time_start = datetime.now()
print('Start time:', time_start.strftime('%Y-%m-%d %H:%M:%S'))

print('Loading data...')
train_path = os.path.join(os.pardir, 'input\\train_and_test\\')
test_path = os.path.join(os.pardir, 'input\\train_and_test\\test.csv')
train1 = pd.read_csv(train_path + 'data_1.csv')
train2 = pd.read_csv(train_path + 'data_2.csv')
train3 = pd.read_csv(train_path + 'data_3.csv')
train4 = pd.read_csv(train_path + 'data_4.csv')
train5 = pd.read_csv(train_path + 'data_5.csv')
all_train = pd.concat([train1, train2, train3, train4, train5])
# fore4_train = pd.concat([train1, train2, train3, train4])
test = pd.read_csv(test_path)
print('loading all the features and label...')
# train5_feature = train5.drop(['user_id', 'label'], axis=1)
# train5_label = train5['label']
# fore4_feature = fore4_train.drop(['user_id', 'label'], axis=1)
# fore4_label = fore4_train['label']
train_feature = all_train.drop(['user_id', 'label'], axis=1)
train_label = all_train['label']
test_feature = test.drop(['user_id'], axis=1)

# 切分训练
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2, random_state=2018)

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
                num_boost_round=345,
                valid_sets=lgb_eval
                )
gbm.save_model('../model/lgb_model.txt')

temp = gbm.predict(X_test)
threshold = 0.42
temp[temp >= threshold] = 1
temp[temp < threshold] = 0
# valid_result = pd.DataFrame()
# valid_result['user_id'] = X_test.reset_index()['index']
# valid_result['result'] = temp
# threshold = valid_result.sort_values(by='result', axis=0, ascending=False).iloc[np.sum(Y_test) - 1, 1]
print('特征重要性：' + str(list(gbm.feature_importance())))
print('f1_score：' + str(sklearn.metrics.f1_score(Y_test, temp)))

########################## 保存结果 ############################
print('Save result...')
prediction = gbm.predict(test_feature)
df_result = pd.DataFrame()
df_result['user_id'] = test['user_id']
df_result['result'] = prediction
df_result.to_csv('../result/lgb_result.csv', index=False)
prediction[prediction >= threshold] = 1
prediction[prediction < threshold] = 0
prediction = list(map(int, prediction))
print('为1的个数：' + str(len(np.where(np.array(prediction) == 1)[0])))
print('为0的个数：' + str(len(np.where(np.array(prediction) == 0)[0])))

time_end = datetime.now()
print('End time:', time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:', "%.2f" % ((time_end - time_start).seconds / 60), 'minutes')
