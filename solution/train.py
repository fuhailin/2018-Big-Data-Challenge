# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:40:11 2018

@author: wxk06
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

input_dir = os.path.join(os.pardir, 'Kesci-data')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')
register_df = pickle.load(open(os.path.join(input_dir, 'user_register.pkl'),"rb"))
launch_df = pickle.load(open(os.path.join(input_dir, 'app_launch.pkl'),"rb"))
video_df = pickle.load(open(os.path.join(input_dir, 'video_create.pkl'),"rb"))
activity_df = pickle.load(open(os.path.join(input_dir, 'user_activity.pkl'),"rb"))


def data_during(start_day,end_day):
    launch_df_selected = launch_df.loc[(launch_df['day'] >= start_day) & (launch_df['day'] <= end_day)]
    video_df_selected = video_df.loc[(video_df['day'] >= start_day) & (video_df['day'] <= end_day)]
    activity_df_selected = activity_df.loc[(activity_df['day'] >= start_day) & (activity_df['day'] <= end_day)]
    launch_freq = launch_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    launch_freq.columns = ['user_id', 'launch_count']
    video_freq = video_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    video_freq.columns = ['user_id', 'video_count']
    activity_freq = activity_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    activity_freq.columns = ['user_id', 'activity_count']
    merged_df = launch_freq.merge(video_freq,how='outer',on='user_id')
    merged_df = merged_df.merge(activity_freq,how='outer',on='user_id')
    merged_df = merged_df.fillna(0)
    merged_df['total_count'] = np.sum(merged_df[['launch_count','video_count','activity_count']],axis = 1)
    return merged_df


def prepare_set(start_day,end_day):
    user_info = register_df.loc[(register_df['register_day'] >= start_day) & (register_df['register_day'] <= end_day)]
    x_raw = user_info.merge(data_during(end_day-6,end_day-0),how='left',on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day-13,end_day-7),how='left',on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day-22,end_day-14),how='left',on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day-0,end_day-0),how='left',on='user_id').fillna(0)
    
    activity_df_selected = activity_df.loc[(activity_df['day'] >= start_day) & (activity_df['day'] <= end_day)]#训练集和测试集时间区间是否应保持一致？
    author_count = pd.DataFrame(activity_df_selected['author_id'].value_counts())
    author_count['index'] = author_count.index
    author_count.columns = ['author_count','author_id']
    x_raw = x_raw.merge(author_count,how='left',left_on='user_id',right_on='author_id').fillna(0)
    x_raw = x_raw.drop(['author_id'],axis=1)#改列名
    
    for i in range(6):
        action_freq = activity_df_selected[['user_id','action_type']].loc[activity_df_selected['action_type']==i]
        action_freq = action_freq.groupby('user_id').agg({'user_id': 'mean', 'action_type': 'count'})
        x_raw = x_raw.merge(action_freq,how='left',on='user_id').fillna(0)#改列名
    
    label_data = data_during(end_day+1,end_day+7)
    label_data['total_count'] = np.sum(label_data[['launch_count','video_count','activity_count']],axis = 1)
    label_data.loc[label_data['total_count'] > 1, 'total_count'] = 1
    label_data = label_data[['user_id','total_count']]
    xy_set = x_raw.merge(label_data,how='left',on='user_id').fillna(0)
    x = xy_set.drop(['user_id','total_count'],axis=1).values
    y = xy_set['total_count'].values
    return x,y

train_x,train_y = prepare_set(1,23)
train_x = np.delete(train_x,[4,8,12,16],axis=1)
test_x,test_y = prepare_set(1,30)
test_x = np.delete(test_x,[4,8,12,16],axis=1)

print('training...')
# =============================================================================
# gbm = xgb.XGBClassifier(
#         #learning_rate = 0.02,
#         n_estimators= 2000,
#         max_depth= 4,
#         min_child_weight= 2,
#         #gamma=1,
#         gamma=0.9,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective= 'binary:logistic',
#         nthread= -1,
#         scale_pos_weight=1).fit(train_x, train_y)
# =============================================================================
gbm = xgb.XGBClassifier(
        #learning_rate = 0.02,
        n_estimators= 2000,
        max_depth= 5,
        min_child_weight= 2,
        #gamma=1,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.95,
        objective= 'binary:logistic',
        nthread= -1,
        scale_pos_weight=1).fit(train_x, train_y)


predictions = gbm.predict(test_x)
user_all = register_df.loc[(register_df['register_day'] >= 1) & (register_df['register_day'] <= 30)]
user_all['isactive']=predictions
result = user_all.loc[user_all['isactive'] > 0]['user_id']
result.to_csv('result.txt', index=False)

def result_score(y,predictions):
    accuracy = np.sum((y == predictions),axis = 0)/len(predictions)
    # F1 score
    M = np.sum(predictions)
    N = np.sum(y)
    MandN = np.sum((predictions + y)>1)
    precision = MandN/M
    recall = MandN/N
    F1_score = 2*precision*recall/(precision + recall)
    return accuracy,F1_score

predictions_valid = gbm.predict(train_x)
print(result_score(train_y,predictions_valid))











