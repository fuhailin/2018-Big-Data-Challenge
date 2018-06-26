# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from scipy import stats
from itertools import groupby
from operator import itemgetter
import stats as sts
import warnings

warnings.filterwarnings("ignore")
from datetime import datetime

time_start = datetime.now()
print('Start time:', time_start.strftime('%Y-%m-%d %H:%M:%S'))

# train_1_user_dir = os.path.join(os.pardir, 'input\\train_1_user')
train_1_feat_dir = os.path.join(os.pardir, 'input\\train_1_feat')
train_1_label_dir = os.path.join(os.pardir, 'input\\train_1_label')
# train_2_user_dir = os.path.join(os.pardir, 'input\\train_2_user')
train_2_feat_dir = os.path.join(os.pardir, 'input\\train_2_feat')
train_2_label_dir = os.path.join(os.pardir, 'input\\train_2_label')
# train_3_user_dir = os.path.join(os.pardir, 'input\\train_3_user')
train_3_feat_dir = os.path.join(os.pardir, 'input\\train_3_feat')
train_3_label_dir = os.path.join(os.pardir, 'input\\train_3_label')
# train_4_user_dir = os.path.join(os.pardir, 'input\\train_4_user')
train_4_feat_dir = os.path.join(os.pardir, 'input\\train_4_feat')
train_4_label_dir = os.path.join(os.pardir, 'input\\train_4_label')
# train_5_user_dir = os.path.join(os.pardir, 'input\\train_5_user')
train_5_feat_dir = os.path.join(os.pardir, 'input\\train_5_feat')
train_5_label_dir = os.path.join(os.pardir, 'input\\train_5_label')
# test_user_dir = os.path.join(os.pardir, 'input\\test_user')
test_feat_dir = os.path.join(os.pardir, 'input\\test_feat')

train_path = os.path.join(os.pardir, 'input\\train_and_test\\')
test_path = os.path.join(os.pardir, 'input\\train_and_test\\test.csv')


# 极差
def ptp(column):
    return max(column) - min(column)


# 方差除以频率
def var_divide_count(column):
    return np.var(column) / len(column)


def last_one(column):
    if len(column) < 1:
        return column
    return column.iloc[-1]


def downquantile(column):
    return column.quantile(.25)


def upquantile(column):
    return column.quantile(.75)


# 偏度Skewness
def skew(column):
    return stats.skew(column)


# 峰度kurtosis
def kurt(column):
    return stats.kurtosis(column)


def discrete(column):
    if np.mean(column) != 0:
        return np.std(column) / np.mean(column)
    else:
        return np.NaN


# 连续多少天启动
def get_continuous(data):
    ranges = []
    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        group = (map(itemgetter(1), g))
        group = list(map(int, group))
        ranges.append(group[-1] - group[0] + 1)
    return max(ranges)


# 统计 page 和 type | 比例
def other_count(df, others, feature_name, other_name):
    others = pd.DataFrame(others.groupby(['user_id', feature_name])[feature_name].count()).reset_index('user_id')
    others.rename(columns={feature_name: feature_name + '_count'}, inplace=True)
    others = others.reset_index(feature_name)
    others = others.pivot('user_id', feature_name, feature_name + '_count').reset_index('user_id')
    columns = others.columns.tolist()
    percent_lists = []
    for c in columns:
        if c is not 'user_id':
            others.rename(columns={c: feature_name + '_' + str(c) + '_' + other_name}, inplace=True)
            percent_lists.append(feature_name + '_' + str(c) + '_' + other_name)
    df = pd.merge(df, others, how='left', on='user_id')
    # extract percentage feature
    for c in percent_lists:
        df[c + '_percent'] = df[c] / df[other_name + '_action_count']
    return df


# 统计 video_id , author_id
def video_count(df, action, feature_name, other_name):
    def getVideo(l):
        return len(np.unique(l))

    temp = action.groupby(['user_id', feature_name])[feature_name].count().reset_index('user_id')
    temp.rename(columns={feature_name: feature_name + '_count' + '_' + other_name}, inplace=True)
    temp = temp.reset_index(feature_name)
    temp_num = temp.copy()
    temp = temp.groupby(['user_id']).agg('max').reset_index('user_id')[['user_id', feature_name + '_count' + '_' + other_name]]
    temp_num = temp_num.groupby('user_id').agg(getVideo).reset_index('user_id')[['user_id', feature_name]]
    temp_num.rename(columns={feature_name: feature_name + '_' + other_name}, inplace=True)
    # 最多的那个video收看了多少次
    df = pd.merge(df, temp, how='left', on='user_id')
    # 一共收看了多少个video
    df = pd.merge(df, temp_num, how='left', on='user_id')
    df[feature_name + '_count' + '_' + other_name + '_percent'] = df[feature_name + '_count' + '_' + other_name] / df[other_name + '_action_count']
    df[feature_name + '_' + other_name + '_percent'] = df[feature_name + '_' + other_name] / df[other_name + '_action_count']
    return df


# action , Page 0 1 2 3 4 , action_type 0 1 2 3 4 5 最后一次相对日期
def last_day(df, action, start, other_name):
    # all action
    all_action = action.groupby(['user_id'])['day'].agg('max')
    all_action = all_action.reset_index('user_id')
    all_action.rename(columns={'day': other_name + '_all_action_last_day'}, inplace=True)
    all_action[other_name + '_all_action_last_day'] = all_action[other_name + '_all_action_last_day'] - start
    df = pd.merge(df, all_action, how='left', on='user_id')
    # Page 0 1 2 3 4 | action_type 0 1 2 3 4 5
    all_feature = ['page', 'action_type']
    for i in range(len(all_feature)):
        temp = action.groupby(['user_id', all_feature[i]])['day'].agg('max')
        temp = temp.reset_index(['user_id', all_feature[i]])
        temp['day'] = temp['day'] - start
        temp = temp.pivot('user_id', all_feature[i], 'day').reset_index('user_id')
        for c in temp.columns:
            if c is not 'user_id':
                temp.rename(columns={c: other_name + '_' + all_feature[i] + '_' + str(c) + '_last_day'}, inplace=True)
        df = pd.merge(df, temp, how='left', on='user_id')
    return df


# 最长连续登陆天数，最长登陆天数最后一天相对日期，action最多那一天的相对日期,最多单天action数
def continous_day(df, action, other_name):
    def longest_day(seq):
        count = 1
        longest = 1
        last_one = seq[0]
        for i in range(len(seq)):
            if i + 1 < len(seq):
                if seq[i] + 1 == seq[i + 1]:
                    count += 1
                    if count > longest:
                        longest = count
                        last_one = seq[i + 1]
                else:
                    count = 1
        return [longest, last_one]

    def most_action(df):
        seq_day = df.day
        seq_day_count = df.day_count
        max_day_count = 0
        max_day = 0
        for x, y in zip(seq_day, seq_day_count):
            if y > max_day_count:
                max_day = x
                max_day_count = y
        return max_day

    page = action.groupby(['user_id', 'day'])['day'].count()
    page = page.reset_index(['user_id'])
    page.rename(columns={'day': 'day_count'}, inplace=True)
    page = page.reset_index('day')
    temp_df = pd.DataFrame({
        'user_id': np.unique(page.user_id).tolist()
    })
    longest = []
    longest_last = []
    most_action_date = []
    for user in temp_df.user_id:
        piece_df = page[page.user_id == user]
        longest.append(longest_day(piece_df.day.tolist())[0])
        longest_last.append(longest_day(piece_df.day.tolist())[1])
        most_action_date.append(most_action(piece_df))
    temp_df[other_name + '_longest_action_day'] = longest
    temp_df[other_name + '_longest_action_day_last'] = longest_last
    temp_df[other_name + '_max_action_day'] = most_action_date
    df = pd.merge(df, temp_df, how='left', on='user_id')
    page = page.groupby('user_id')['day_count'].agg('max')
    page = page.reset_index('user_id')
    page.rename(columns={'day_count': other_name + '_max_action_one_day'}, inplace=True)
    df = pd.merge(df, page, how='left', on='user_id')
    return df


# user_id 和 author_id 相同的 action
def same_author(df, action, other_name):
    temp = action[action.user_id == action.author_id].groupby('user_id').count().reset_index('user_id')[
        ['user_id', 'day']]
    temp.rename(columns={'day': other_name + '_same_author_count'}, inplace=True)
    df = pd.merge(df, temp, how='left', on='user_id')
    df[other_name + '_same_author_count'] = df[other_name + '_same_author_count'].fillna(0).astype(int)
    df[other_name + '_same_author_count_percent'] = df[other_name + '_same_author_count'] / df[
        other_name + '_action_count']
    return df


# 最后七天，三天，两天，一天是否有action
def whether_action(df, action, other_name):
    temp = action.groupby('user_id').count().reset_index('user_id')[['user_id', 'day']]
    temp.rename(columns={'day': other_name + '_whether_active'}, inplace=True)
    temp.loc[temp[other_name + '_whether_active'] > 0, other_name + '_whether_active'] = 1
    df = pd.merge(df, temp, how='left', on='user_id')
    df[other_name + '_whether_active'] = df[other_name + '_whether_active'].fillna(0).astype(int)
    return df


# 最后一天是否有拍摄video的行为
def whether_video(df, video, end):
    video = video[video.create_day == end]
    video = video.drop_duplicates()
    video.loc[video.create_day > 0, 'create_day'] = 1
    video.rename(columns={'create_day': 'whether_last_video'}, inplace=True)
    df = pd.merge(df, video, how='left', on='user_id')
    df['whether_last_video'] = df['whether_last_video'].fillna(0).astype(int)
    return df


def get_sequence_feature(df, start, agg_feature, feature_name):
    func = ['mean', 'median', 'max', 'min', 'var', 'std', skew, kurt, discrete, upquantile, downquantile]
    sequence_feature_diff = pd.concat([df['user_id'], df.groupby(['user_id']).diff().rename({agg_feature: agg_feature + '_diff'}, axis=1)], axis=1)
    df[agg_feature] = df[agg_feature] - start
    sequence_feature = df.groupby('user_id')[agg_feature].agg(func).reset_index()
    sequence_feature = sequence_feature.rename(
        columns={'mean': feature_name + '_mean', 'median': feature_name + '_median', 'max': feature_name + '_max', 'min': feature_name + '_min',
                 'skew': feature_name + '_skew', 'std': feature_name + '_std', 'var': feature_name + '_var', 'kurt': feature_name + '_kurt',
                 'discrete': feature_name + '_discrete', 'upquantile': feature_name + '_upquantile', 'downquantile': feature_name + '_downquantile'})

    sequence_diff_feature = sequence_feature_diff.groupby('user_id')[agg_feature + '_diff'].agg(func).reset_index()
    sequence_diff_feature = sequence_diff_feature.rename(
        columns={'mean': feature_name + '_gap_mean', 'median': feature_name + '_gap_median', 'max': feature_name + '_gap_max', 'min': feature_name + '_gap_min',
                 'skew': feature_name + '_gap_skew', 'std': feature_name + '_gap_std', 'var': feature_name + '_gap_var', 'kurt': feature_name + '_gap_kurt',
                 'discrete': feature_name + '_gap_discrete', 'upquantile': feature_name + '_gap_upquantile', 'downquantile': feature_name + '_gap_downquantile'})
    sequence_feature = pd.merge(sequence_feature, sequence_diff_feature, how='left', on='user_id')
    return sequence_feature


# APP 启动次数，平均次数，方差，启动天数，最大值，最小值，连续几天启动总次数，平均次数，某一天启动次数
# agg要使用的函数
# func = ['min', 'mean', 'median', 'skew', 'std', 'mad', ptp, np.var, var_divide_count, last_one, q25, q75, kurt]


def get_train_label(feat_path, label_path):
    feat_register = pd.read_csv(feat_path + '_register.csv', usecols=['user_id'])
    feat_launch = pd.read_csv(feat_path + '_launch.csv', usecols=['user_id'])
    feat_video = pd.read_csv(feat_path + '_create.csv', usecols=['user_id'])
    feat_activity = pd.read_csv(feat_path + '_activity.csv', usecols=['user_id'])
    feat_data_id = np.unique(pd.concat([feat_register, feat_launch, feat_video, feat_activity]))
    # feat_register = pd.read_csv(feat_path + '_register.csv', usecols=['user_id'])
    # feat_data_id = np.unique(feat_register)

    #    label_register = pd.read_csv(label_path + '/register.csv', usecols=['user_id'])
    label_launch = pd.read_csv(label_path + '_launch.csv', usecols=['user_id'])
    label_create = pd.read_csv(label_path + '_create.csv', usecols=['user_id'])
    label_activity = pd.read_csv(label_path + '_activity.csv', usecols=['user_id'])
    label_data_id = np.unique(pd.concat([label_launch, label_create, label_activity]))

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
    feat_register = pd.read_csv(feat_path + '_register.csv', usecols=['user_id'])
    feat_launch = pd.read_csv(feat_path + '_launch.csv', usecols=['user_id'])
    feat_video = pd.read_csv(feat_path + '_create.csv', usecols=['user_id'])
    feat_activity = pd.read_csv(feat_path + '_activity.csv', usecols=['user_id'])
    feat_data_id = np.unique(pd.concat([feat_register, feat_launch, feat_video, feat_activity]))
    test_data = pd.DataFrame()
    test_data['user_id'] = feat_data_id
    return test_data


'''
    - 提取注册特征(register)
    1.相对注册日期，在时间窗之前注册的为-1
    2.注册类型
    3.设备类型

'''


def get_register_feature(register):
    feature = pd.DataFrame()
    feature['user_id'] = register['user_id'].drop_duplicates()
    # end_day = np.max(register['register_day'])
    start_day = np.min(register['register_day'])
    feature['register_day'] = register['register_day'] - start_day + 1
    feature['register_day'] = feature['register_day'].fillna(-1)
    feature['register_day'] = feature['register_day'].astype(int)
    feature['register_type'] = register['register_type']
    feature['device_type'] = register['device_type']
    # register_rest_day = (end_day - register.groupby(['user_id'])['register_day'].max()).rename('register_rest_day').reset_index()
    # feature = pd.merge(feature, register_rest_day, how='left', on='user_id')
    return feature


"""
    - 提取启动特征(launch)
    1.launch天数序列特征 : 所有序列特征
    2.launch_diff序列(加入end day取diff) : 所有序列特征 ， 最后一个和倒数第二个值(没有则为0)
    3.launch 总天数,最后 7 5 3 1天的次数
    5.launch 第一次启动到时间窗末尾
    4.launch percent ： 启动总天数 / 第一次启动到时间窗末尾
    6.launch 连续天数序列： 所有序列特征
    7.周末的launch总次数(unique)
    8.周末的Launch总数占总launch数比
"""


def get_launch_feature(launch):
    launch.sort_values(by=['user_id', 'launch_day'], inplace=True)
    feature = pd.DataFrame()
    feature['user_id'] = launch['user_id'].drop_duplicates()
    func = ['mean', 'median', 'max', 'min', 'var', 'std', skew, kurt, discrete]
    '''
    launch_fun = launch.groupby('user_id')['launch_day'].agg(func).reset_index()
    launch_fun = launch_fun.rename(columns={'min': 'launch_gap_day_min', 'mean': 'launch_gap_day_mean', 'median': 'launch_gap_day_median', 'skew': 'launch_gap_day_skew',
                                            'std': 'launch_gap_day_std', 'var': 'launch_gap_day_var', 'kurt': 'launch_gap_day_kurt'})
                                            '''
    end_day = np.max(launch['launch_day'])
    start_day = np.min(launch['launch_day'])

    launch_total_count = launch.groupby(['user_id']).size().rename('launch_total_count').reset_index()  # 启动总次数
    launch_day_diff = pd.concat([launch['user_id'], launch.groupby(['user_id']).diff().rename({'launch_day': 'launch_day_diff'}, axis=1)],
                                axis=1)
    launch_fun = launch_day_diff.groupby('user_id')['launch_day_diff'].agg(func).reset_index()
    launch_fun = launch_fun.rename(
        columns={'mean': 'launch_gap_day_mean', 'median': 'launch_gap_day_median', 'max': 'launch_gap_day_max', 'min': 'launch_gap_day_min',
                 'skew': 'launch_gap_day_skew', 'std': 'launch_gap_day_std', 'var': 'launch_gap_day_var', 'kurt': 'launch_gap_day_kurt',
                 'discrete': 'launch_gap_day_discrete'})
    # duration / percent
    launch_day_duration = launch.groupby(['user_id'])['launch_day'].agg(
        {'launch_day_duration': lambda x: (end_day - min(x) + 1)}).reset_index()
    launch_day_duration['launch_day_percent'] = (launch_total_count['launch_total_count'] - 1) / launch_day_duration[
        'launch_day_duration']  # -1是为了区分只有一次启动但percent却为100%, #平均每天启动次数
    weekend_day = launch.groupby(['user_id'])['launch_day'].agg(
        {'launch_weekend_day_count': lambda x: (
            len([day for day in np.unique(x[:-1]) if day % 7 == 0 or (day + 1) % 7 == 0]))}).reset_index()
    weekend_day['launch_weekend_day_count_percent'] = weekend_day['launch_weekend_day_count'] / launch_total_count['launch_total_count']
    continuous_launch_day = launch.groupby(['user_id'])['launch_day'].agg([get_continuous]).reset_index().rename(
        columns={'get_continuous': 'continuous_launch_day'})
    last_seven_launch_count = launch[(launch['launch_day'] >= end_day - 6) & (launch['launch_day'] <= end_day)].groupby(['user_id'])[
        'launch_day'].size().rename('last_seven_launch_count').reset_index()
    last_five_launch_count = launch[(launch['launch_day'] >= end_day - 4) & (launch['launch_day'] <= end_day)].groupby(['user_id'])[
        'launch_day'].size().rename('last_five_launch_count').reset_index()
    last_three_launch_count = launch[(launch['launch_day'] >= end_day - 2) & (launch['launch_day'] <= end_day)].groupby(['user_id'])[
        'launch_day'].size().rename('last_three_launch_count').reset_index()
    last_one_launch_count = launch[(launch['launch_day'] == end_day)].groupby(['user_id'])['launch_day'].size().rename(
        'last_one_create_count').reset_index()
    launch_rest_day = (end_day - launch.groupby(['user_id'])['launch_day'].max()).rename('launch_rest_day').reset_index()  # 已有多少天未启动

    feature = pd.merge(feature, launch_total_count, how='left', on='user_id')
    feature = pd.merge(feature, launch_fun, how='left', on='user_id')
    feature = pd.merge(feature, launch_day_duration, how='left', on='user_id')
    feature = pd.merge(feature, weekend_day, how='left', on='user_id')
    feature = pd.merge(feature, continuous_launch_day, how='left', on='user_id')
    feature = pd.merge(feature, last_seven_launch_count, how='left', on='user_id')
    feature = pd.merge(feature, last_five_launch_count, how='left', on='user_id')
    feature = pd.merge(feature, last_three_launch_count, how='left', on='user_id')
    feature = pd.merge(feature, last_one_launch_count, how='left', on='user_id')
    feature = pd.merge(feature, launch_rest_day, how='left', on='user_id')
    return feature


"""
    - 提取拍摄视频特征(create)
    1.create天数序列特征 : 所有序列特征
    2.create_diff序列(取unique,加入end day取diff) : 所有序列特征 ， 最后一个和倒数第二个值(没有则为0)
    3.create 总天数,最后 7 5 3 1天的次数
    4.第一次拍摄到时间窗末尾 
    5.create percent ： 拍摄总天数(unique) / 第一次拍摄到时间窗末尾
    6.create day连续序列: 所有序列特征
    7.create count序列 : 所有序列特征
    8.create count diff序列：所有序列特征 最后一个和倒数第一个值(没有则为0) 加和
    9.周末的 create 总数
    10.发生create的周末天数(unique)
    10.周末create总次数 占比 周末create天数(unique)
"""


def get_create_feature(create):
    create.sort_values(by=['user_id', 'create_day'], inplace=True)
    feature = pd.DataFrame()
    feature['user_id'] = create['user_id'].drop_duplicates()
    '''
    video_fun = video.groupby('user_id')['create_day'].agg(func).reset_index()
    video_fun = video_fun.rename(
        columns={'min': 'video_min', 'mean': 'video_mean', 'median': 'video_median',
                 'skew': 'video_skew', 'std': 'video_std', 'mad': 'video_mad',
                 'ptp': 'video_ptp', 'var': 'video_var', 'var_divide_count': 'video_var_divide_count',
                 'last_one': 'video_last_one', 'q25': 'video_q25', 'q75': 'video_q75', 'kurt': 'video_kurt'})
                 '''
    end_day = np.max(create['create_day'])

    create_total_count = create.groupby(['user_id']).size().rename('create_total_count').reset_index()  # 总创建个数
    create_day_diff = pd.concat([create['user_id'], create.groupby(['user_id']).diff().rename({'create_day': 'video_day_diff'}, axis=1)],
                                axis=1)
    func = ['mean', 'median', 'max', 'min', 'var', 'std', skew, kurt, discrete]
    create_fun = create_day_diff.groupby('user_id')['create_day_diff'].agg(func).reset_index()
    create_fun = create_fun.rename(
        columns={'mean': 'create_gap_day_mean', 'median': 'create_gap_day_median', 'max': 'create_gap_day_max', 'min': 'create_gap_day_min',
                 'skew': 'create_gap_day_skew', 'std': 'create_gap_day_std', 'var': 'create_gap_day_var', 'kurt': 'create_gap_day_kurt',
                 'discrete': 'create_gap_day_discrete'})
    create_day_duration = create.groupby(['user_id'])['create_day'].agg(
        {'create_day_duration': lambda x: (end_day - min(x) + 1)}).reset_index()  # 第一次拍摄到时间窗末尾的时间间隔
    create_day_duration['create_day_count'] = create.groupby(['user_id'])['create_day'].agg(
        {'launch_day_duration': lambda x: len(np.unique(x))})  # 统计创建过视频的天数
    create_day_duration['create_day_percent'] = (create_day_duration['create_day_count'] - 1) / create_day_duration[
        'create_day_duration']  # -1是为了区分只有一次启动但percent却为100%, #时间窗内有创建的天数时间比
    create_total_count['create_count_percent'] = (create_total_count['create_total_count'] - 1) / create_day_duration[
        'create_day_duration']  # 在时间窗内平均每天创建几个视频
    weekend_day = create.groupby(['user_id'])['create_day'].agg(
        {'create_weekend_day_count': lambda x: len([day for day in np.unique(x[:-1]) if day % 7 == 0 or (day + 1) % 7 == 0])})  # 有多少天在周末创建
    weekend_day['create_weekend_day_all_count'] = create.groupby(['user_id'])['create_day'].agg(
        {'create_weekend_day_count': lambda x: len([day for day in x[:-1] if day % 7 == 0 or (day + 1) % 7 == 0])})  # 在周末总共创建了多少个视频
    weekend_day['create_weekend_day_count_percent'] = weekend_day['create_weekend_day_count'] / weekend_day['create_weekend_day_all_count']
    continuous_cteate_day = create.groupby(['user_id'])['create_day'].agg([get_continuous]).reset_index().rename(
        columns={'get_continuous': 'continuous_cteate_day'})
    last_seven_create_count = create[(create['create_day'] >= end_day - 6) & (create['create_day'] <= end_day)].groupby(['user_id'])[
        'create_day'].size().rename('last_seven_create_count').reset_index()
    last_five_create_count = create[(create['create_day'] >= end_day - 4) & (create['create_day'] <= end_day)].groupby(['user_id'])[
        'create_day'].size().rename('last_five_create_count').reset_index()
    last_three_create_count = create[(create['create_day'] >= end_day - 2) & (create['create_day'] <= end_day)].groupby(['user_id'])[
        'create_day'].size().rename('last_three_create_count').reset_index()
    last_one_create_count = create[(create['create_day'] == end_day)].groupby(['user_id'])['create_day'].size().rename(
        'last_one_create_count').reset_index()

    create_rest_day = (end_day - create.groupby(['user_id'])['create_day'].max()).rename('create_rest_day').reset_index()  # 已有多少天未创建视频
    video_everyday_count = create.groupby(['user_id', 'create_day']).agg({'user_id': 'mean', 'create_day': 'count'})
    video_day_most = video_everyday_count.groupby(['user_id'])['create_day'].max().rename('video_day_most').reset_index()
    video_day_mode = video_everyday_count.groupby(['user_id'])['create_day'].agg(lambda x: np.mean(pd.Series.mode(x))).rename(
        'video_day_mode').reset_index()

    feature = whether_video(feature, create, end_day)
    feature = pd.merge(feature, create_total_count, how='left', on='user_id')
    feature = pd.merge(feature, create_day_duration, how='left', on='user_id')
    feature = pd.merge(feature, weekend_day, how='left', on='user_id')
    feature = pd.merge(feature, continuous_cteate_day, how='left', on='user_id')
    feature = pd.merge(feature, last_seven_create_count, how='left', on='user_id')
    feature = pd.merge(feature, last_five_create_count, how='left', on='user_id')
    feature = pd.merge(feature, last_three_create_count, how='left', on='user_id')
    feature = pd.merge(feature, last_one_create_count, how='left', on='user_id')
    feature = pd.merge(feature, create_rest_day, how='left', on='user_id')
    feature = pd.merge(feature, create_fun, how='left', on='user_id')
    feature = pd.merge(feature, video_day_most, how='left', on='user_id')
    feature = pd.merge(feature, video_day_mode, how='left', on='user_id')
    # feature = pd.merge(feature, video_fun, how='left', on=['user_id'])
    return feature


def get_activity_feature(activity):
    activity = activity.rename({'action_type': 'action'}, axis=1)
    feature = pd.DataFrame()
    feature['user_id'] = activity['user_id'].drop_duplicates()
    activity.sort_values(by=['user_id', 'activity_day'], inplace=True)
    end_day = np.max(activity['activity_day'])
    start_day = np.min(activity['activity_day'])

    # action 总次数, 最后 7 5 3 1
    activity_total_count = activity.groupby(['user_id'])['activity_day'].agg({'all_action_count': 'count'})  # 总活动次数
    activity_total_count['last_seven_action_count'] = activity[(activity['activity_day'] >= end_day - 6) & (activity['activity_day'] <= end_day)].groupby(['user_id'])['activity_day'].size()  #
    activity_total_count['last_five_action_count'] = activity[(activity['activity_day'] >= end_day - 4) & (activity['activity_day'] <= end_day)].groupby(['user_id'])['activity_day'].size()
    activity_total_count['last_three_action_count'] = activity[(activity['activity_day'] >= end_day - 2) & (activity['activity_day'] <= end_day)].groupby(['user_id'])['activity_day'].size()
    activity_total_count['last_one_action_count'] = activity[(activity['activity_day'] == end_day)].groupby(['user_id'])['activity_day'].size()

    # 周末总次数 , 周末的 Page 0 1 2 3 4 个数 , action_type 0 1 2 3 4 5 个数;周末的 Page 0 1 2 3 4 个数 , action_type 0 1 2 3 4 5 个数 分别占 周末总次数的比例
    act_weekend_count = activity.groupby(['user_id'])['activity_day'].agg({'activity_weekend_count': lambda x: len([day for day in x if day % 7 == 0 or (day + 1) % 7 == 0])})  # 在周末的总共活动次数
    for i in range(5):  # 周末的 Page 0 1 2 3 4 个数
        act_weekend_count['act_weekend_count_page_%d' % i] = activity[activity['page'] == i].groupby(['user_id'])['activity_day'].agg({lambda x: len([day for day in x if day % 7 == 0 or (day + 1) % 7 == 0])})
        act_weekend_count['act_weekend_count_page_%d_percent' % i] = act_weekend_count['act_weekend_count_page_%d' % i] / act_weekend_count['activity_weekend_count']
    for i in range(6):  # 周末的 action_type 0 1 2 3 4 5 个数
        act_weekend_count['act_weekend_count_type_%d' % i] = activity[activity['action_type'] == i].groupby(['user_id'])['activity_day'].agg({lambda x: len([day for day in x if day % 7 == 0 or (day + 1) % 7 == 0])})
        act_weekend_count['act_weekend_count_type_%d_percent' % i] = act_weekend_count['act_weekend_count_type_%d' % i] / act_weekend_count['activity_weekend_count']

    # 总的Page 0 1 2 3 4 个数,最后7,5,3,1 | 总的action_type 0 1 2 3 4 5 个数，最后7,5,3,1
    all_type = ['page', 'action_type']
    for i in range(2):
        activity_total_count = other_count(activity_total_count, activity, all_type[i], 'all')
        activity_total_count = other_count(activity_total_count, activity[(activity['activity_day'] >= end_day - 6) & (activity['activity_day'] <= end_day)], all_type[i], 'last_seven')
        activity_total_count = other_count(activity_total_count, activity[(activity['activity_day'] >= end_day - 4) & (activity['activity_day'] <= end_day)], all_type[i], 'last_five')
        activity_total_count = other_count(activity_total_count, activity[(activity['activity_day'] >= end_day - 2) & (activity['activity_day'] <= end_day)], all_type[i], 'last_three')
        activity_total_count = other_count(activity_total_count, activity[activity['activity_day'] == end_day], all_type[i], 'last_one')

    # video 看了几个，看的最多的那个video有多少次action ，author 同 video
    all_type = ['author_id', 'video_id']
    for i in range(2):
        activity_total_count = video_count(activity_total_count, activity, all_type[i], 'all')
        activity_total_count = video_count(activity_total_count, activity[(activity['activity_day'] >= end_day - 6) & (activity['activity_day'] <= end_day)], all_type[i], 'last_seven')
        activity_total_count = video_count(activity_total_count, activity[(activity['activity_day'] >= end_day - 4) & (activity['activity_day'] <= end_day)], all_type[i], 'last_five')
        activity_total_count = video_count(activity_total_count, activity[(activity['activity_day'] >= end_day - 2) & (activity['activity_day'] <= end_day)], all_type[i], 'last_three')
        activity_total_count = video_count(activity_total_count, activity[activity['activity_day'] == end_day], all_type[i], 'last_one')

    # action , Page 0 1 2 3 4 , action_type 0 1 2 3 4 5 最后一次相对日期
    activity_total_count = last_day(activity_total_count, activity, start_day, 'all')

    # 总的，最后7,5,3,1 ： 最长连续登陆天数，最长登陆天数最后一天相对日期，action最多那一天的相对日期, 最多单天action数
    activity_total_count = continous_day(activity_total_count, activity, 'all')
    activity_total_count = continous_day(activity_total_count, activity[(activity['activity_day'] >= end_day - 6) & (activity['activity_day'] <= end_day)], 'last_seven')
    activity_total_count = continous_day(activity_total_count, activity[(activity['activity_day'] >= end_day - 4) & (activity['activity_day'] <= end_day)], 'last_five')
    activity_total_count = continous_day(activity_total_count, activity[(activity['activity_day'] >= end_day - 2) & (activity['activity_day'] <= end_day)], 'last_three')
    activity_total_count = continous_day(activity_total_count, activity[activity['activity_day'] == end_day], 'last_one')

    # user_id 和 author_id 相同的 count | percentage
    activity_total_count = same_author(activity_total_count, activity, 'all')
    activity_total_count = same_author(activity_total_count, activity[(activity.activity_day >= end_day - 6) & (activity.day <= end_day)], 'last_seven')
    activity_total_count = same_author(activity_total_count, activity[(activity.activity_day >= end_day - 4) & (activity.day <= end_day)], 'last_five')
    activity_total_count = same_author(activity_total_count, activity[(activity.activity_day >= end_day - 2) & (activity.day <= end_day)], 'last_three')
    activity_total_count = same_author(activity_total_count, activity[activity.activity_day == end_day], 'last_one')

    # 最后七天，三天，两天，一天是否有action
    activity_total_count = whether_action(activity_total_count, activity, 'all')
    activity_total_count = whether_action(activity_total_count, activity[(activity.activity_day >= end_day - 6) & (activity.activity_day <= end_day)], 'last_seven')
    activity_total_count = whether_action(activity_total_count, activity[(activity.activity_day >= end_day - 4) & (activity.activity_day <= end_day)], 'last_five')
    activity_total_count = whether_action(activity_total_count, activity[(activity.activity_day >= end_day - 2) & (activity.activity_day <= end_day)], 'last_three')
    activity_total_count = whether_action(activity_total_count, activity[activity.activity_day == end_day], 'last_one')

    # 下面都是各个行为发生的那天构成的序列(以及该序列的gap序列)特征
    # activity['activity_day'] = activity['activity_day'] - start_day
    action_seq_fea = get_sequence_feature(activity, start_day, 'activity_day', 'all_action')
    # action_seq_fea = action_seq(feature, activity, start_day, end_day)
    page_seq_fea = page_seq(feature, activity, start_day, end_day)
    action_type_seq_fea = action_type_seq(feature, activity, start_day, end_day)
    user_author_seq_fea = user_author_seq(feature, activity, start_day, end_day)

    '''
    activity_total_count = activity[['user_id']].groupby(['user_id']).size().rename('activity_total_count').reset_index()  # 总活动次数
    activity_day_unique = activity.groupby(['user_id', 'activity_day']).agg({'user_id': 'mean', 'activity_day': 'mean'})
    activity_day_diff = activity_day_unique.groupby(['user_id']).diff().rename({'activity_day': 'activity_day_diff'},
                                                                               axis=1).reset_index().drop('activity_day', axis=1)
    activity_day_diff_max = activity_day_diff.groupby(['user_id'])['activity_day_diff'].max().rename('activity_day_diff_max').reset_index()
    activity_day_diff_min = activity_day_diff.groupby(['user_id'])['activity_day_diff'].min().rename('activity_day_diff_min').reset_index()
    activity_day_diff_mean = activity_day_diff.groupby(['user_id'])['activity_day_diff'].mean().rename(
        'activity_day_diff_mean').reset_index()
    activity_day_diff_std = activity_day_diff.groupby(['user_id'])['activity_day_diff'].std().rename('activity_day_diff_std').reset_index()
    activity_day_diff_kurt = activity_day_diff.groupby(['user_id'])['activity_day_diff'].agg(lambda x: pd.Series.kurt(x)).rename(
        'activity_day_diff_kurt').reset_index()
    activity_day_diff_skew = activity_day_diff.groupby(['user_id'])['activity_day_diff'].skew().rename(
        'activity_day_diff_skew').reset_index()
    activity_day_diff_last = activity_day_diff.groupby(['user_id'])['activity_day_diff'].last().rename(
        'activity_day_diff_last').reset_index()
    activity_last_day = activity.groupby(['user_id'])['activity_day'].max().rename('activity_last_day').reset_index()  # 最后一次活动的日期
    activity_rest_day = (end_day - activity_day_unique.groupby(['user_id'])['activity_day'].max()).rename(
        'activity_rest_day').reset_index()  # 已有多少天未活动
    page_count = activity.groupby(['user_id', 'page']).agg({'page': 'count'}).rename({'page': 'page_count'}, axis=1).reset_index()
    page0_count = page_count[page_count.page == 0].drop('page', axis=1).rename({'page_count': 'page0_count'}, axis=1)
    page1_count = page_count[page_count.page == 1].drop('page', axis=1).rename({'page_count': 'page1_count'}, axis=1)
    page2_count = page_count[page_count.page == 2].drop('page', axis=1).rename({'page_count': 'page2_count'}, axis=1)
    page3_count = page_count[page_count.page == 3].drop('page', axis=1).rename({'page_count': 'page3_count'}, axis=1)
    page4_count = page_count[page_count.page == 4].drop('page', axis=1).rename({'page_count': 'page4_count'}, axis=1)
    page_percent = pd.merge(activity_total_count, page0_count, how='left', on='user_id')
    page_percent = pd.merge(page_percent, page1_count, how='left', on='user_id')
    page_percent = pd.merge(page_percent, page2_count, how='left', on='user_id')
    page_percent = pd.merge(page_percent, page3_count, how='left', on='user_id')
    page_percent = pd.merge(page_percent, page4_count, how='left', on='user_id')
    page_percent['page0_pct'] = page_percent['page0_count'] / page_percent['activity_total_count']
    page_percent['page1_pct'] = page_percent['page1_count'] / page_percent['activity_total_count']
    page_percent['page2_pct'] = page_percent['page2_count'] / page_percent['activity_total_count']
    page_percent['page3_pct'] = page_percent['page3_count'] / page_percent['activity_total_count']
    page_percent['page4_pct'] = page_percent['page4_count'] / page_percent['activity_total_count']
    page_percent = page_percent.drop('activity_total_count', axis=1)
    action_count = activity.groupby(['user_id', 'action']).agg({'action': 'count'}).rename({'action': 'action_count'}, axis=1).reset_index()
    action0_count = action_count[action_count.action == 0].drop('action', axis=1).rename({'action_count': 'action0_count'}, axis=1)
    action1_count = action_count[action_count.action == 1].drop('action', axis=1).rename({'action_count': 'action1_count'}, axis=1)
    action2_count = action_count[action_count.action == 2].drop('action', axis=1).rename({'action_count': 'action2_count'}, axis=1)
    action3_count = action_count[action_count.action == 3].drop('action', axis=1).rename({'action_count': 'action3_count'}, axis=1)
    action4_count = action_count[action_count.action == 4].drop('action', axis=1).rename({'action_count': 'action4_count'}, axis=1)
    action5_count = action_count[action_count.action == 5].drop('action', axis=1).rename({'action_count': 'action5_count'}, axis=1)
    action_percent = pd.merge(activity_total_count, action0_count, how='left', on='user_id')
    action_percent = pd.merge(action_percent, action1_count, how='left', on='user_id')
    action_percent = pd.merge(action_percent, action2_count, how='left', on='user_id')
    action_percent = pd.merge(action_percent, action3_count, how='left', on='user_id')
    action_percent = pd.merge(action_percent, action4_count, how='left', on='user_id')
    action_percent = pd.merge(action_percent, action5_count, how='left', on='user_id')
    action_percent['action0_pct'] = action_percent['action0_count'] / action_percent['activity_total_count']
    action_percent['action1_pct'] = action_percent['action1_count'] / action_percent['activity_total_count']
    action_percent['action2_pct'] = action_percent['action2_count'] / action_percent['activity_total_count']
    action_percent['action3_pct'] = action_percent['action3_count'] / action_percent['activity_total_count']
    action_percent['action4_pct'] = action_percent['action4_count'] / action_percent['activity_total_count']
    action_percent['action5_pct'] = action_percent['action5_count'] / action_percent['activity_total_count']
    action_percent = action_percent.drop('activity_total_count', axis=1)
    video_id_count = activity.groupby(['user_id', 'video_id']).agg({'user_id': 'mean', 'video_id': 'count'})
    video_id_most = video_id_count.groupby(['user_id'])['video_id'].max().rename('video_id_most').reset_index()
    author_id_count = activity.groupby(['user_id', 'author_id']).agg({'user_id': 'mean', 'author_id': 'count'})
    author_id_most = author_id_count.groupby(['user_id'])['author_id'].max().rename('author_id_most').reset_index()
    activity_count = activity.groupby(['user_id', 'activity_day']).agg({'activity_day': 'count'}).rename({'activity_day': 'activity_count'},
                                                                                                         axis=1).reset_index()
    activity_count_max = activity_count.groupby(['user_id'])['activity_count'].max().rename('activity_count_max').reset_index()
    activity_count_min = activity_count.groupby(['user_id'])['activity_count'].min().rename('activity_count_min').reset_index()
    activity_count_mean = activity_count.groupby(['user_id'])['activity_count'].mean().rename('activity_count_mean').reset_index()
    activity_count_std = activity_count.groupby(['user_id'])['activity_count'].std().rename('activity_count_std').reset_index()
    activity_count_kurt = activity_count.groupby(['user_id'])['activity_count'].agg(lambda x: pd.Series.kurt(x)).rename(
        'activity_count_kurt').reset_index()
    activity_count_skew = activity_count.groupby(['user_id'])['activity_count'].skew().rename('activity_count_skew').reset_index()
    activity_count_last = activity_count.groupby(['user_id'])['activity_count'].last().rename('activity_count_last').reset_index()
    activity_count_diff = pd.concat(
        [activity_count['user_id'], activity_count.groupby(['user_id']).diff().rename({'activity_count': 'activity_count_diff'}, axis=1)],
        axis=1).drop(
        'activity_day', axis=1)
    activity_count_diff_max = activity_count_diff.groupby(['user_id'])['activity_count_diff'].max().rename(
        'activity_count_diff_max').reset_index()
    activity_count_diff_min = activity_count_diff.groupby(['user_id'])['activity_count_diff'].min().rename(
        'activity_count_diff_min').reset_index()
    activity_count_diff_mean = activity_count_diff.groupby(['user_id'])['activity_count_diff'].mean().rename(
        'activity_count_diff_mean').reset_index()
    activity_count_diff_std = activity_count_diff.groupby(['user_id'])['activity_count_diff'].std().rename(
        'activity_count_diff_std').reset_index()
    activity_count_diff_kurt = activity_count_diff.groupby(['user_id'])['activity_count_diff'].agg(lambda x: pd.Series.kurt(x)).rename(
        'activity_count_diff_kurt').reset_index()
    activity_count_diff_skew = activity_count_diff.groupby(['user_id'])['activity_count_diff'].skew().rename(
        'activity_count_diff_skew').reset_index()
    activity_count_diff_last = activity_count_diff.groupby(['user_id'])['activity_count_diff'].last().rename(
        'activity_count_diff_last').reset_index()
    page_everyday_count = activity.groupby(['user_id', 'activity_day', 'page']).agg({'page': 'count'}).rename(
        {'page': 'page_everyday_count'}, axis=1).reset_index()
    page0_max = page_everyday_count[page_everyday_count.page == 0].groupby(['user_id'])['page_everyday_count'].max().rename(
        'page0_max').reset_index()
    page0_min = page_everyday_count[page_everyday_count.page == 0].groupby(['user_id'])['page_everyday_count'].min().rename(
        'page0_min').reset_index()
    page0_mean = page_everyday_count[page_everyday_count.page == 0].groupby(['user_id'])['page_everyday_count'].mean().rename(
        'page0_mean').reset_index()
    page0_std = page_everyday_count[page_everyday_count.page == 0].groupby(['user_id'])['page_everyday_count'].std().rename(
        'page0_std').reset_index()
    page0_kurt = page_everyday_count[page_everyday_count.page == 0].groupby(['user_id'])['page_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename('page0_kurt').reset_index()
    page0_skew = page_everyday_count[page_everyday_count.page == 0].groupby(['user_id'])['page_everyday_count'].skew().rename(
        'page0_skew').reset_index()
    page0_last = page_everyday_count[page_everyday_count.page == 0].groupby(['user_id'])['page_everyday_count'].last().rename(
        'page0_last').reset_index()
    page1_max = page_everyday_count[page_everyday_count.page == 1].groupby(['user_id'])['page_everyday_count'].max().rename(
        'page1_max').reset_index()
    page1_min = page_everyday_count[page_everyday_count.page == 1].groupby(['user_id'])['page_everyday_count'].min().rename(
        'page1_min').reset_index()
    page1_mean = page_everyday_count[page_everyday_count.page == 1].groupby(['user_id'])['page_everyday_count'].mean().rename(
        'page1_mean').reset_index()
    page1_std = page_everyday_count[page_everyday_count.page == 1].groupby(['user_id'])['page_everyday_count'].std().rename(
        'page1_std').reset_index()
    page1_kurt = page_everyday_count[page_everyday_count.page == 1].groupby(['user_id'])['page_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename('page1_kurt').reset_index()
    page1_skew = page_everyday_count[page_everyday_count.page == 1].groupby(['user_id'])['page_everyday_count'].skew().rename(
        'page1_skew').reset_index()
    page1_last = page_everyday_count[page_everyday_count.page == 1].groupby(['user_id'])['page_everyday_count'].last().rename(
        'page1_last').reset_index()
    page2_max = page_everyday_count[page_everyday_count.page == 2].groupby(['user_id'])['page_everyday_count'].max().rename(
        'page2_max').reset_index()
    page2_min = page_everyday_count[page_everyday_count.page == 2].groupby(['user_id'])['page_everyday_count'].min().rename(
        'page2_min').reset_index()
    page2_mean = page_everyday_count[page_everyday_count.page == 2].groupby(['user_id'])['page_everyday_count'].mean().rename(
        'page2_mean').reset_index()
    page2_std = page_everyday_count[page_everyday_count.page == 2].groupby(['user_id'])['page_everyday_count'].std().rename(
        'page2_std').reset_index()
    page2_kurt = page_everyday_count[page_everyday_count.page == 2].groupby(['user_id'])['page_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename('page2_kurt').reset_index()
    page2_skew = page_everyday_count[page_everyday_count.page == 2].groupby(['user_id'])['page_everyday_count'].skew().rename(
        'page2_skew').reset_index()
    page2_last = page_everyday_count[page_everyday_count.page == 2].groupby(['user_id'])['page_everyday_count'].last().rename(
        'page2_last').reset_index()
    page3_max = page_everyday_count[page_everyday_count.page == 3].groupby(['user_id'])['page_everyday_count'].max().rename(
        'page3_max').reset_index()
    page3_min = page_everyday_count[page_everyday_count.page == 3].groupby(['user_id'])['page_everyday_count'].min().rename(
        'page3_min').reset_index()
    page3_mean = page_everyday_count[page_everyday_count.page == 3].groupby(['user_id'])['page_everyday_count'].mean().rename(
        'page3_mean').reset_index()
    page3_std = page_everyday_count[page_everyday_count.page == 3].groupby(['user_id'])['page_everyday_count'].std().rename(
        'page3_std').reset_index()
    page3_kurt = page_everyday_count[page_everyday_count.page == 3].groupby(['user_id'])['page_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename('page3_kurt').reset_index()
    page3_skew = page_everyday_count[page_everyday_count.page == 3].groupby(['user_id'])['page_everyday_count'].skew().rename(
        'page3_skew').reset_index()
    page3_last = page_everyday_count[page_everyday_count.page == 3].groupby(['user_id'])['page_everyday_count'].last().rename(
        'page3_last').reset_index()
    page4_max = page_everyday_count[page_everyday_count.page == 4].groupby(['user_id'])['page_everyday_count'].max().rename(
        'page4_max').reset_index()
    page4_min = page_everyday_count[page_everyday_count.page == 4].groupby(['user_id'])['page_everyday_count'].min().rename(
        'page4_min').reset_index()
    page4_mean = page_everyday_count[page_everyday_count.page == 4].groupby(['user_id'])['page_everyday_count'].mean().rename(
        'page4_mean').reset_index()
    page4_std = page_everyday_count[page_everyday_count.page == 4].groupby(['user_id'])['page_everyday_count'].std().rename(
        'page4_std').reset_index()
    page4_kurt = page_everyday_count[page_everyday_count.page == 4].groupby(['user_id'])['page_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename('page4_kurt').reset_index()
    page4_skew = page_everyday_count[page_everyday_count.page == 4].groupby(['user_id'])['page_everyday_count'].skew().rename(
        'page4_skew').reset_index()
    page4_last = page_everyday_count[page_everyday_count.page == 4].groupby(['user_id'])['page_everyday_count'].last().rename(
        'page4_last').reset_index()
    action_everyday_count = activity.groupby(['user_id', 'activity_day', 'action']).agg({'action': 'count'}).rename(
        {'action': 'action_everyday_count'}, axis=1).reset_index()
    action0_max = action_everyday_count[action_everyday_count.action == 0].groupby(['user_id'])['action_everyday_count'].max().rename(
        'action0_max').reset_index()
    action0_min = action_everyday_count[action_everyday_count.action == 0].groupby(['user_id'])['action_everyday_count'].min().rename(
        'action0_min').reset_index()
    action0_mean = action_everyday_count[action_everyday_count.action == 0].groupby(['user_id'])['action_everyday_count'].mean().rename(
        'action0_mean').reset_index()
    action0_std = action_everyday_count[action_everyday_count.action == 0].groupby(['user_id'])['action_everyday_count'].std().rename(
        'action0_std').reset_index()
    action0_kurt = action_everyday_count[action_everyday_count.action == 0].groupby(['user_id'])['action_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename(
        'action0_kurt').reset_index()
    action0_skew = action_everyday_count[action_everyday_count.action == 0].groupby(['user_id'])['action_everyday_count'].skew().rename(
        'action0_skew').reset_index()
    action0_last = action_everyday_count[action_everyday_count.action == 0].groupby(['user_id'])['action_everyday_count'].last().rename(
        'action0_last').reset_index()
    action1_max = action_everyday_count[action_everyday_count.action == 1].groupby(['user_id'])['action_everyday_count'].max().rename(
        'action1_max').reset_index()
    action1_min = action_everyday_count[action_everyday_count.action == 1].groupby(['user_id'])['action_everyday_count'].min().rename(
        'action1_min').reset_index()
    action1_mean = action_everyday_count[action_everyday_count.action == 1].groupby(['user_id'])['action_everyday_count'].mean().rename(
        'action1_mean').reset_index()
    action1_std = action_everyday_count[action_everyday_count.action == 1].groupby(['user_id'])['action_everyday_count'].std().rename(
        'action1_std').reset_index()
    action1_kurt = action_everyday_count[action_everyday_count.action == 1].groupby(['user_id'])['action_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename(
        'action1_kurt').reset_index()
    action1_skew = action_everyday_count[action_everyday_count.action == 1].groupby(['user_id'])['action_everyday_count'].skew().rename(
        'action1_skew').reset_index()
    action1_last = action_everyday_count[action_everyday_count.action == 1].groupby(['user_id'])['action_everyday_count'].last().rename(
        'action1_last').reset_index()
    action2_max = action_everyday_count[action_everyday_count.action == 2].groupby(['user_id'])['action_everyday_count'].max().rename(
        'action2_max').reset_index()
    action2_min = action_everyday_count[action_everyday_count.action == 2].groupby(['user_id'])['action_everyday_count'].min().rename(
        'action2_min').reset_index()
    action2_mean = action_everyday_count[action_everyday_count.action == 2].groupby(['user_id'])['action_everyday_count'].mean().rename(
        'action2_mean').reset_index()
    action2_std = action_everyday_count[action_everyday_count.action == 2].groupby(['user_id'])['action_everyday_count'].std().rename(
        'action2_std').reset_index()
    action2_kurt = action_everyday_count[action_everyday_count.action == 2].groupby(['user_id'])['action_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename(
        'action2_kurt').reset_index()
    action2_skew = action_everyday_count[action_everyday_count.action == 2].groupby(['user_id'])['action_everyday_count'].skew().rename(
        'action2_skew').reset_index()
    action2_last = action_everyday_count[action_everyday_count.action == 2].groupby(['user_id'])['action_everyday_count'].last().rename(
        'action2_last').reset_index()
    action3_max = action_everyday_count[action_everyday_count.action == 3].groupby(['user_id'])['action_everyday_count'].max().rename(
        'action3_max').reset_index()
    action3_min = action_everyday_count[action_everyday_count.action == 3].groupby(['user_id'])['action_everyday_count'].min().rename(
        'action3_min').reset_index()
    action3_mean = action_everyday_count[action_everyday_count.action == 3].groupby(['user_id'])['action_everyday_count'].mean().rename(
        'action3_mean').reset_index()
    action3_std = action_everyday_count[action_everyday_count.action == 3].groupby(['user_id'])['action_everyday_count'].std().rename(
        'action3_std').reset_index()
    action3_kurt = action_everyday_count[action_everyday_count.action == 3].groupby(['user_id'])['action_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename(
        'action3_kurt').reset_index()
    action3_skew = action_everyday_count[action_everyday_count.action == 3].groupby(['user_id'])['action_everyday_count'].skew().rename(
        'action3_skew').reset_index()
    action3_last = action_everyday_count[action_everyday_count.action == 3].groupby(['user_id'])['action_everyday_count'].last().rename(
        'action3_last').reset_index()
    action4_max = action_everyday_count[action_everyday_count.action == 4].groupby(['user_id'])['action_everyday_count'].max().rename(
        'action4_max').reset_index()
    action4_min = action_everyday_count[action_everyday_count.action == 4].groupby(['user_id'])['action_everyday_count'].min().rename(
        'action4_min').reset_index()
    action4_mean = action_everyday_count[action_everyday_count.action == 4].groupby(['user_id'])['action_everyday_count'].mean().rename(
        'action4_mean').reset_index()
    action4_std = action_everyday_count[action_everyday_count.action == 4].groupby(['user_id'])['action_everyday_count'].std().rename(
        'action4_std').reset_index()
    action4_kurt = action_everyday_count[action_everyday_count.action == 4].groupby(['user_id'])['action_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename(
        'action4_kurt').reset_index()
    action4_skew = action_everyday_count[action_everyday_count.action == 4].groupby(['user_id'])['action_everyday_count'].skew().rename(
        'action4_skew').reset_index()
    action4_last = action_everyday_count[action_everyday_count.action == 4].groupby(['user_id'])['action_everyday_count'].last().rename(
        'action4_last').reset_index()
    action5_max = action_everyday_count[action_everyday_count.action == 5].groupby(['user_id'])['action_everyday_count'].max().rename(
        'action5_max').reset_index()
    action5_min = action_everyday_count[action_everyday_count.action == 5].groupby(['user_id'])['action_everyday_count'].min().rename(
        'action5_min').reset_index()
    action5_mean = action_everyday_count[action_everyday_count.action == 5].groupby(['user_id'])['action_everyday_count'].mean().rename(
        'action5_mean').reset_index()
    action5_std = action_everyday_count[action_everyday_count.action == 5].groupby(['user_id'])['action_everyday_count'].std().rename(
        'action5_std').reset_index()
    action5_kurt = action_everyday_count[action_everyday_count.action == 5].groupby(['user_id'])['action_everyday_count'].agg(
        lambda x: pd.Series.kurt(x)).rename(
        'action5_kurt').reset_index()
    action5_skew = action_everyday_count[action_everyday_count.action == 5].groupby(['user_id'])['action_everyday_count'].skew().rename(
        'action5_skew').reset_index()
    action5_last = action_everyday_count[action_everyday_count.action == 5].groupby(['user_id'])['action_everyday_count'].last().rename(
        'action5_last').reset_index()
    most_activity_day = \
        activity_count.groupby('user_id').apply(lambda x: x[x.activity_count == x.activity_count.max()]).rename(
            {'activity_day': 'most_activity_day'}, axis=1).drop(
            'activity_count',
            axis=1).groupby(
            'user_id')['most_activity_day'].max().reset_index()
    author_count = activity.groupby('author_id').agg({'author_id': 'mean', 'activity_day': 'count'}).rename(
        {'author_id': 'user_id', 'activity_day': 'author_count'}, axis=1)

    feature = pd.merge(feature, activity_total_count, how='left', on='user_id')
    feature = pd.merge(feature, activity_day_diff_max, how='left', on='user_id')
    feature = pd.merge(feature, activity_day_diff_min, how='left', on='user_id')
    feature = pd.merge(feature, activity_day_diff_mean, how='left', on='user_id')
    feature = pd.merge(feature, activity_day_diff_std, how='left', on='user_id')
    feature = pd.merge(feature, activity_day_diff_kurt, how='left', on='user_id')
    feature = pd.merge(feature, activity_day_diff_skew, how='left', on='user_id')
    feature = pd.merge(feature, activity_day_diff_last, how='left', on='user_id')
    feature = pd.merge(feature, activity_last_day, how='left', on='user_id')
    feature = pd.merge(feature, activity_rest_day, how='left', on='user_id')
    feature = pd.merge(feature, page_percent, how='left', on='user_id')
    feature = pd.merge(feature, action_percent, how='left', on='user_id')
    feature = pd.merge(feature, video_id_most, how='left', on='user_id')
    feature = pd.merge(feature, author_id_most, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_max, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_min, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_mean, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_std, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_kurt, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_skew, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_last, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_diff_max, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_diff_min, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_diff_mean, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_diff_std, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_diff_kurt, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_diff_skew, how='left', on='user_id')
    feature = pd.merge(feature, activity_count_diff_last, how='left', on='user_id')
    feature = pd.merge(feature, page0_max, how='left', on='user_id')
    feature = pd.merge(feature, page0_min, how='left', on='user_id')
    feature = pd.merge(feature, page0_mean, how='left', on='user_id')
    feature = pd.merge(feature, page0_std, how='left', on='user_id')
    feature = pd.merge(feature, page0_kurt, how='left', on='user_id')
    feature = pd.merge(feature, page0_skew, how='left', on='user_id')
    feature = pd.merge(feature, page0_last, how='left', on='user_id')
    feature = pd.merge(feature, page1_max, how='left', on='user_id')
    feature = pd.merge(feature, page1_min, how='left', on='user_id')
    feature = pd.merge(feature, page1_mean, how='left', on='user_id')
    feature = pd.merge(feature, page1_std, how='left', on='user_id')
    feature = pd.merge(feature, page1_kurt, how='left', on='user_id')
    feature = pd.merge(feature, page1_skew, how='left', on='user_id')
    feature = pd.merge(feature, page1_last, how='left', on='user_id')
    feature = pd.merge(feature, page2_max, how='left', on='user_id')
    feature = pd.merge(feature, page2_min, how='left', on='user_id')
    feature = pd.merge(feature, page2_mean, how='left', on='user_id')
    feature = pd.merge(feature, page2_std, how='left', on='user_id')
    feature = pd.merge(feature, page2_kurt, how='left', on='user_id')
    feature = pd.merge(feature, page2_skew, how='left', on='user_id')
    feature = pd.merge(feature, page2_last, how='left', on='user_id')
    feature = pd.merge(feature, page3_max, how='left', on='user_id')
    feature = pd.merge(feature, page3_min, how='left', on='user_id')
    feature = pd.merge(feature, page3_mean, how='left', on='user_id')
    feature = pd.merge(feature, page3_std, how='left', on='user_id')
    feature = pd.merge(feature, page3_kurt, how='left', on='user_id')
    feature = pd.merge(feature, page3_skew, how='left', on='user_id')
    feature = pd.merge(feature, page3_last, how='left', on='user_id')
    feature = pd.merge(feature, page4_max, how='left', on='user_id')
    feature = pd.merge(feature, page4_min, how='left', on='user_id')
    feature = pd.merge(feature, page4_mean, how='left', on='user_id')
    feature = pd.merge(feature, page4_std, how='left', on='user_id')
    feature = pd.merge(feature, page4_kurt, how='left', on='user_id')
    feature = pd.merge(feature, page4_skew, how='left', on='user_id')
    feature = pd.merge(feature, page4_last, how='left', on='user_id')
    feature = pd.merge(feature, action0_max, how='left', on='user_id')
    feature = pd.merge(feature, action0_min, how='left', on='user_id')
    feature = pd.merge(feature, action0_mean, how='left', on='user_id')
    feature = pd.merge(feature, action0_std, how='left', on='user_id')
    feature = pd.merge(feature, action0_kurt, how='left', on='user_id')
    feature = pd.merge(feature, action0_skew, how='left', on='user_id')
    feature = pd.merge(feature, action0_last, how='left', on='user_id')
    feature = pd.merge(feature, action1_max, how='left', on='user_id')
    feature = pd.merge(feature, action1_min, how='left', on='user_id')
    feature = pd.merge(feature, action1_mean, how='left', on='user_id')
    feature = pd.merge(feature, action1_std, how='left', on='user_id')
    feature = pd.merge(feature, action1_kurt, how='left', on='user_id')
    feature = pd.merge(feature, action1_skew, how='left', on='user_id')
    feature = pd.merge(feature, action1_last, how='left', on='user_id')
    feature = pd.merge(feature, action2_max, how='left', on='user_id')
    feature = pd.merge(feature, action2_min, how='left', on='user_id')
    feature = pd.merge(feature, action2_mean, how='left', on='user_id')
    feature = pd.merge(feature, action2_std, how='left', on='user_id')
    feature = pd.merge(feature, action2_kurt, how='left', on='user_id')
    feature = pd.merge(feature, action2_skew, how='left', on='user_id')
    feature = pd.merge(feature, action2_last, how='left', on='user_id')
    feature = pd.merge(feature, action3_max, how='left', on='user_id')
    feature = pd.merge(feature, action3_min, how='left', on='user_id')
    feature = pd.merge(feature, action3_mean, how='left', on='user_id')
    feature = pd.merge(feature, action3_std, how='left', on='user_id')
    feature = pd.merge(feature, action3_kurt, how='left', on='user_id')
    feature = pd.merge(feature, action3_skew, how='left', on='user_id')
    feature = pd.merge(feature, action3_last, how='left', on='user_id')
    feature = pd.merge(feature, action4_max, how='left', on='user_id')
    feature = pd.merge(feature, action4_min, how='left', on='user_id')
    feature = pd.merge(feature, action4_mean, how='left', on='user_id')
    feature = pd.merge(feature, action4_std, how='left', on='user_id')
    feature = pd.merge(feature, action4_kurt, how='left', on='user_id')
    feature = pd.merge(feature, action4_skew, how='left', on='user_id')
    feature = pd.merge(feature, action4_last, how='left', on='user_id')
    feature = pd.merge(feature, action5_max, how='left', on='user_id')
    feature = pd.merge(feature, action5_min, how='left', on='user_id')
    feature = pd.merge(feature, action5_mean, how='left', on='user_id')
    feature = pd.merge(feature, action5_std, how='left', on='user_id')
    feature = pd.merge(feature, action5_kurt, how='left', on='user_id')
    feature = pd.merge(feature, action5_skew, how='left', on='user_id')
    feature = pd.merge(feature, action5_last, how='left', on='user_id')
    feature = pd.merge(feature, most_activity_day, how='left', on='user_id')
    feature = pd.merge(feature, author_count, how='left', on='user_id')
    '''
    feature = pd.merge(feature, act_weekend_count, how='left', on=['user_id'])
    feature = pd.merge(feature, activity_total_count, how='left', on=['user_id'])
    feature = pd.merge(feature, action_seq_fea, how='left', on=['user_id'])
    feature = pd.merge(feature, page_seq_fea, how='left', on=['user_id'])
    feature = pd.merge(feature, action_type_seq_fea, how='left', on=['user_id'])
    feature = pd.merge(feature, user_author_seq_fea, how='left', on=['user_id'])
    return feature


def deal_feature(path, user_id):
    # register = pd.read_csv(path + '_register.csv')
    # launch = pd.read_csv(path + '_launch.csv')
    # create = pd.read_csv(path + '_create.csv')
    activity = pd.read_csv(path + '_activity.csv')
    feature = pd.DataFrame()
    feature['user_id'] = user_id

    # print('getting register feature...')
    # register_feature = get_register_feature(register)
    # feature = pd.merge(feature, register_feature, on='user_id', how='left')

    # print('getting launch feature...')
    # launch_feature = get_launch_feature(launch)
    # feature = pd.merge(feature, launch_feature, on='user_id', how='left')

    # print('getting create feature...')
    # create_feature = get_create_feature(create)
    # feature = pd.merge(feature, create_feature, on='user_id', how='left')

    print('getting activity feature...')
    activity_feature = get_activity_feature(activity)
    feature = pd.merge(feature, activity_feature, on='user_id', how='left')

    feature['last_launch_sub_register'] = feature['launch_last_day'] - feature['register_day']
    feature['last_video_sub_register'] = feature['video_last_day'] - feature['register_day']
    feature['last_activity_sub_register'] = feature['activity_last_day'] - feature['register_day']

    feature = feature.fillna(0)
    return feature


def get_data_feature():
    print('Feature engineering...')
    print('Getting train data 1 ...')
    train_label_1 = get_train_label(train_1_feat_dir, train_1_label_dir)
    data_1 = deal_feature(train_1_feat_dir, train_label_1['user_id'])
    data_1['label'] = train_label_1['label']
    data_1.to_csv(train_path + 'data_1.csv', index=False)

    print('Getting train data 2 ...')
    train_label_2 = get_train_label(train_2_feat_dir, train_2_label_dir)
    data_2 = deal_feature(train_2_feat_dir, train_label_2['user_id'])
    data_2['label'] = train_label_2['label']
    data_2.to_csv(train_path + 'data_2.csv', index=False)

    print('Getting train data 3 ...')
    train_label_3 = get_train_label(train_3_feat_dir, train_3_label_dir)
    data_3 = deal_feature(train_3_feat_dir, train_label_3['user_id'])
    data_3['label'] = train_label_3['label']
    data_3.to_csv(train_path + 'data_3.csv', index=False)

    print('Getting train data 4 ...')
    train_label_4 = get_train_label(train_4_feat_dir, train_4_label_dir)
    data_4 = deal_feature(train_4_feat_dir, train_label_4['user_id'])
    data_4['label'] = train_label_4['label']
    data_4.to_csv(train_path + 'data_4.csv', index=False)

    print('Getting train data 5 ...')
    train_label_5 = get_train_label(train_5_feat_dir, train_5_label_dir)
    data_5 = deal_feature(train_5_feat_dir, train_label_5['user_id'])
    data_5['label'] = train_label_5['label']
    data_5.to_csv(train_path + 'data_5.csv', index=False)

    # train_data = pd.concat([data_1, data_2])
    # train_data.to_csv(train_path, index=False)

    print('Getting test data...')
    test_id = get_test_id(test_feat_dir)
    test_data = deal_feature(test_feat_dir, test_id['user_id'])
    test_data.to_csv(test_path, index=False)


get_data_feature()
time_end = datetime.now()
print('End time:', time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:', "%.2f" % ((time_end - time_start).seconds / 60), 'minutes')
