# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:09:01 2018

@author: ASUS
"""

import os
import warnings

warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

print('start_time:', datetime.now().time())

input_dir = os.path.join(os.pardir, 'Kesci-data')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')
register_df = pickle.load(open(os.path.join(input_dir, 'user_register.pkl'), "rb"))
launch_df = pickle.load(open(os.path.join(input_dir, 'app_launch.pkl'), "rb"))
video_df = pickle.load(open(os.path.join(input_dir, 'video_create.pkl'), "rb"))
activity_df = pickle.load(open(os.path.join(input_dir, 'user_activity.pkl'), "rb"))


def data_during(start_day, end_day):
    launch_df_selected = launch_df.loc[(launch_df['day'] >= start_day) & (launch_df['day'] <= end_day)]
    video_df_selected = video_df.loc[(video_df['day'] >= start_day) & (video_df['day'] <= end_day)]
    activity_df_selected = activity_df.loc[(activity_df['day'] >= start_day) & (activity_df['day'] <= end_day)]
    launch_freq = launch_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    launch_freq.columns = ['user_id', 'launch_count']
    video_freq = video_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    video_freq.columns = ['user_id', 'video_count']
    activity_freq = activity_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    activity_freq.columns = ['user_id', 'activity_count']
    merged_df = launch_freq.merge(video_freq, how='outer', on='user_id')
    merged_df = merged_df.merge(activity_freq, how='outer', on='user_id')
    merged_df = merged_df.fillna(0)
    merged_df['total_count'] = np.sum(merged_df[['launch_count', 'video_count', 'activity_count']], axis=1)
    return merged_df


def prepare_set(start_day, end_day):
    user_info = register_df.loc[(register_df['register_day'] >= start_day) & (register_df['register_day'] <= end_day)]
    x_raw = user_info.merge(data_during(end_day - 6, end_day - 0), how='left', on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day - 13, end_day - 7), how='left', on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day - 22, end_day - 14), how='left', on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day - 0, end_day - 0), how='left', on='user_id').fillna(0)

    # =============================================================================
    #     activity_df_selected = activity_df.loc[(activity_df['day'] >= start_day) & (activity_df['day'] <= end_day)]#训练集和测试集时间区间是否应保持一致？
    #     author_count = pd.DataFrame(activity_df_selected['author_id'].value_counts())
    #     author_count['index'] = author_count.index
    #     author_count.columns = ['author_count','author_id']
    #     x_raw = x_raw.merge(author_count,how='left',left_on='user_id',right_on='author_id').fillna(0)
    #     x_raw = x_raw.drop(['author_id'],axis=1)#改列名
    #
    #     for i in range(6):
    #         action_freq = activity_df_selected[['user_id','action_type']].loc[activity_df_selected['action_type']==i]
    #         action_freq = action_freq.groupby('user_id').agg({'user_id': 'mean', 'action_type': 'count'})
    #         x_raw = x_raw.merge(action_freq,how='left',on='user_id').fillna(0)#改列名
    # =============================================================================

    label_data = data_during(end_day + 1, end_day + 7)
    label_data['total_count'] = np.sum(label_data[['launch_count', 'video_count', 'activity_count']], axis=1)
    label_data.loc[label_data['total_count'] > 1, 'total_count'] = 1
    label_data = label_data[['user_id', 'total_count']]
    xy_set = x_raw.merge(label_data, how='left', on='user_id').fillna(0)
    x = xy_set.drop(['user_id', 'total_count'], axis=1).values
    y = xy_set['total_count'].values
    return x, y


train_x, train_y = prepare_set(1, 23)
test_x, test_y = prepare_set(1, 30)


## 待修改参数：xgb.cv(metrics='auc')
# clf.fit(eval_metric='auc')      xgb1 = XGBClassifier(scale_pos_weight=1)   gsearch1 = GridSearchCV(scoring='roc_auc',n_jobs=4)

# 这里原blog指定了xy_set和predictors(特征名)
def modelFit(clf, train_x, train_y, isCv=True, cv_folds=5, early_stopping_rounds=50):
    if isCv:
        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(train_x, label=train_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)  # 是否显示目前几颗树
        clf.set_params(n_estimators=cvresult.shape[0])
        print('n_est:', cvresult.shape[0])

    # 训练
    clf.fit(train_x, train_y, eval_metric='auc')

    # 预测
    train_predictions = clf.predict(train_x)
    train_predprob = clf.predict_proba(train_x)[:, 1]  # 1的概率

    # 打印
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob))

    feat_imp = pd.Series(clf.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature importance')
    plt.ylabel('Feature Importance Score')
    plt.show()


# =============================================================================
# print('model_1...')
# # 类别平衡的问题：min_child_weight=2,scale_pos_weight=sum(negative cases) / sum(positive cases),
# xgb1 = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,  
#                      colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)  
# modelFit(xgb1,train_x,train_y)
# 
# print('tuning max_depth and min_child_weight...')
# param_test1 = {
#         'max_depth':[i for i in range(3,10,2)],
#         'min_child_weight':[i for i in range(1,6,2)]
# }  
# gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate =0.1, n_estimators=61, max_depth=5, 
#                                                 min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,  
#                                                 objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
#                         param_grid=param_test1,scoring='roc_auc',iid=False,cv=5)
# gsearch1.fit(train_x,train_y)  
# print('best_params:',gsearch1.best_params_,'\nbest_score:',gsearch1.best_score_)
# 
# print('fine-tuning max_depth and min_child_weight...')
# param_test2 = {
#         'max_depth':[4,5,6],
#         'min_child_weight':[1,2,3]
# }
# gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=61, max_depth=5,
#                                                   min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                                                   objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27), 
#                         param_grid=param_test2,scoring='roc_auc',iid=False,cv=5)
# gsearch2.fit(train_x,train_y)
# print('best_params:',gsearch2.best_params_,'\nbest_score:',gsearch2.best_score_)
# =============================================================================

# =============================================================================
# print('model2...')
# xgb2 = XGBClassifier(
#         learning_rate =0.1,
#         n_estimators=1000,
#         max_depth=5,
#         min_child_weight=2,
#         gamma=0,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective= 'binary:logistic',
#         nthread=4,
#         scale_pos_weight=1,
#         seed=27)
# modelFit(xgb2,train_x,train_y)
# =============================================================================

# =============================================================================
# print('tuning gamma...')
# # max_depth and min_child_weight根据上面结果改
# param_test3 = {
#         'gamma':[i/10.0 for i in range(0,5)]
# }
# gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=67, max_depth=5,
#                                                   min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                                                   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                         param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(train_x,train_y)
# print('best_params:',gsearch3.best_params_,'\nbest_score:',gsearch3.best_score_)
# 
# print('model3...')
# xgb3 = XGBClassifier(
#         learning_rate =0.1,
#         n_estimators=1000,
#         max_depth=5,
#         min_child_weight=2,
#         gamma=0,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective= 'binary:logistic',
#         nthread=4,
#         scale_pos_weight=1,
#         seed=27)
# modelFit(xgb3,train_x,train_y)
# =============================================================================

# =============================================================================
# print('tuning subsample and colsample_bytree...')
# param_test4 = {
#         'subsample':[i/10.0 for i in range(6,11)],
#         'colsample_bytree':[i/10.0 for i in range(6,11)]
# }
# # n_estimators根据xgb3改
# gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=67, max_depth=5,
#                                                   min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                                                   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                         param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(train_x,train_y)
# print('best_params:',gsearch4.best_params_,'\nbest_score:',gsearch4.best_score_)
# =============================================================================

# =============================================================================
# print('fine-tuning subsample and colsample_bytree...')
# param_test5 = {
#         'subsample':[i/100.0 for i in range(85,100,5)],
#         'colsample_bytree':[i/100.0 for i in range(85,100,5)]
# }
# gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=67, max_depth=5,
#                                                   min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                                                   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#                         param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
# gsearch5.fit(train_x,train_y)
# print('best_params:',gsearch5.best_params_,'\nbest_score:',gsearch5.best_score_)
# =============================================================================

# =============================================================================
# print('tuning Regularization Parameters...')
# param_test6 = {
#         'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
# }
# gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=4,
#                                                   min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
#                                                   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
#                         param_grid = param_test6, scoring='roc_auc',iid=False, cv=5)
# gsearch6.fit(train_x,train_y)
# print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)
# 
# param_test7 = {
#         'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
# }
# gsearch7 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=4,
#                                                   min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
#                                                   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#                         param_grid = param_test7, scoring='roc_auc',iid=False, cv=5)
# gsearch7.fit(train_x,train_y)
# print(gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_)
# =============================================================================

print('model4...')
xgb4 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=2,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.95,
    #        reg_alpha=0.005,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
modelFit(xgb4, train_x, train_y)

print('model5...')
xgb5 = XGBClassifier(
    learning_rate=0.01,
    n_estimators=5000,
    max_depth=4,
    min_child_weight=6,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.005,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
modelFit(xgb5, train_x, train_y)

gbm = xgb5.fit(train_x, train_y)
predictions = gbm.predict(test_x)
user_all = register_df.loc[(register_df['register_day'] >= 1) & (register_df['register_day'] <= 30)]
user_all['isactive'] = predictions
result = user_all.loc[user_all['isactive'] > 0]['user_id']
result.to_csv('result.txt', index=False)


def result_score(y, predictions):
    accuracy = np.sum((y == predictions), axis=0) / len(predictions)
    # F1 score
    M = np.sum(predictions)
    N = np.sum(y)
    MandN = np.sum((predictions + y) > 1)
    precision = MandN / M
    recall = MandN / N
    F1_score = 2 * precision * recall / (precision + recall)
    return accuracy, F1_score


predictions_valid = gbm.predict(train_x)
print(result_score(train_y, predictions_valid))
