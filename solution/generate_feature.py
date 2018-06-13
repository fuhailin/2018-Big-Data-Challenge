# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:00:17 2018

@author: ASUS
"""

import os
import pandas as pd
import numpy as np

dataset_1_feat_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_1_feat')
dataset_1_label_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_1_label')
dataset_2_feat_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_2_feat')
dataset_2_label_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_2_label')
dataset_3_feat_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_3_feat')

train_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')

def get_train_label(feat_path,label_path):
    feat_register = pd.read_csv(feat_path + '/register.csv', usecols=['user_id'])
    feat_launch = pd.read_csv(feat_path + '/launch.csv', usecols=['user_id'])
    feat_video = pd.read_csv(feat_path + '/video.csv', usecols=['user_id'])
    feat_activity = pd.read_csv(feat_path + '/activity.csv', usecols=['user_id'])
    feat_data_id = np.unique(pd.concat([feat_register,feat_launch,feat_video,feat_activity]))
    
    label_register = pd.read_csv(label_path + '/register.csv', usecols=['user_id'])
    label_launch = pd.read_csv(label_path + '/launch.csv', usecols=['user_id'])
    label_video = pd.read_csv(label_path + '/video.csv', usecols=['user_id'])
    label_activity = pd.read_csv(label_path + '/activity.csv', usecols=['user_id'])
    label_data_id = np.unique(pd.concat([label_register,label_launch,label_video,label_activity]))
    
    train_label = []
    for i in feat_data_id:
        if i in label_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = feat_data_id
    train_data['label'] = train_label
    return train_data

def get_test_id(feat_path):
    feat_register = pd.read_csv(feat_path + '/register.csv', usecols=['user_id'])
    feat_launch = pd.read_csv(feat_path + '/launch.csv', usecols=['user_id'])
    feat_video = pd.read_csv(feat_path + '/video.csv', usecols=['user_id'])
    feat_activity = pd.read_csv(feat_path + '/activity.csv', usecols=['user_id'])
    feat_data_id = np.unique(pd.concat([feat_register,feat_launch,feat_video,feat_activity]))
    test_data = pd.DataFrame()
    test_data['user_id'] = feat_data_id
    return test_data

def get_register_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    return feature

def get_launch_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    return feature

def get_video_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    return feature

def get_activity_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    return feature
    
def deal_feature(path, user_id):
    register = pd.read_csv(path + '/register.csv')
    launch = pd.read_csv(path + '/launch.csv')
    video = pd.read_csv(path + '/video.csv')
    activity = pd.read_csv(path + '/activity.csv')
    feature = pd.DataFrame()
    feature['user_id'] = user_id
    
    register['max_day'] = np.max(register['register_day']) #?
    register_feature = register.groupby('user_id', sort=True).apply(get_register_feature)
    feature = pd.merge(feature, pd.DataFrame(register_feature), on='user_id', how='left')
    
    launch['max_day'] = np.max(register['register_day'])
    launch_feature = launch.groupby('user_id', sort=True).apply(get_launch_feature)
    feature = pd.merge(feature, pd.DataFrame(launch_feature), on='user_id', how='left')
    
    video['max_day'] = np.max(register['register_day'])
    video_feature = video.groupby('user_id', sort=True).apply(get_video_feature)
    feature = pd.merge(feature, pd.DataFrame(video_feature), on='user_id', how='left')
    
    activity['max_day'] = np.max(register['register_day'])
    activity_feature = activity.groupby('user_id', sort=True).apply(get_activity_feature)
    feature = pd.merge(feature, pd.DataFrame(activity_feature), on='user_id', how='left')
    return feature

def get_data_feature():
    train_label_1 = get_train_label(dataset_1_feat_dir,dataset_1_label_dir)
    data_1 = deal_feature(dataset_1_feat_dir,train_label_1['user_id'])
    data_1['label'] = train_label_1['label']
    
    train_label_2 = get_train_label(dataset_2_feat_dir,dataset_2_label_dir)
    data_2 = deal_feature(dataset_2_feat_dir,train_label_2['user_id'])
    data_2['label'] = train_label_2['label']
    
    train_data = pd.concat([data_1,data_2])
    train_data.to_csv(train_path,index=False)
    
    test_id = get_test_id(dataset_3_feat_dir)
    test_data = deal_feature(dataset_3_feat_dir,test_id['user_id'])
    test_data.to_csv(test_path,index=False)


get_data_feature()














