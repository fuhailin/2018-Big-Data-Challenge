"""
    author: yang yiqing
    comment: 数据预处理
    date: 2018年06月16日16:32:00

"""


import pandas as pd
import numpy as np


def split_data(action, launch, register, video):
    print('split data...')
    for i in range(Num_dataSet):
        start_date = start_date_list[i]
        end_date = end_date_list[i]
        temp_action = action[(action.day >= start_date) & (action.day <= end_date)]
        temp_launch = launch[(launch.day >= start_date) & (launch.day <= end_date)]
        temp_register = register[(register.day >= start_date) & (register.day <= end_date)]
        temp_video = video[(video.day >= start_date) & (video.day <= end_date)]
        temp_all_user = np.unique(
            temp_action['user_id'].tolist() + temp_register['user_id'].tolist() + temp_launch['user_id'].tolist() +
            temp_video['user_id'].tolist())
        temp_label_user = np.unique(
            action[(action.day > end_date) & (action.day <= end_date + 7)]['user_id'].tolist() +
            launch[(launch.day > end_date) & (launch.day <= end_date + 7)]['user_id'].tolist() +
            register[(register.day > end_date) & (register.day <= end_date + 7)]['user_id'].tolist() +
            video[(video.day > end_date) & (video.day <= end_date + 7)]['user_id'].tolist())
        # get label
        temp_DF = get_label(temp_all_user, temp_label_user)
        # save file
        # df中是user_id和label,其他日志文件通过user_id来left merge到df中即可
        temp_DF.to_csv('splited_date/df_%d_%d.csv' % (start_date, end_date))
        temp_action.to_csv('splited_date/action_%d_%d.csv' % (start_date, end_date))
        temp_launch.to_csv('splited_date/launch_%d_%d.csv' % (start_date, end_date))
        temp_register.to_csv('splited_date/register_%d_%d.csv' % (start_date, end_date))
        temp_video.to_csv('splited_date/video_%d_%d.csv' % (start_date, end_date))


def get_label(all_user, label_user):
    print('get label...')
    print(len(all_user))
    # 测试集的label全为0
    print(len(label_user))
    df = pd.DataFrame()
    df['user_id'] = all_user
    label = np.zeros(len(all_user))
    for i in range(len(all_user)):
        label[i] = 1 if all_user[i] in label_user else 0
    df['label'] = label
    df['label'] = df['label'].astype(int)
    return df


if __name__ == '__main__':
    # 修改时间窗只需要修改下面的参数即可
    Num_dataSet = 3
    start_date_list = [1, 8, 15]
    end_date_list = [16, 23, 30]
    # read data
    action = pd.read_csv('data/action.csv', index_col=0)
    launch = pd.read_csv('data/launcher.csv', index_col=0)
    register = pd.read_csv('data/register.csv', index_col=0)
    video = pd.read_csv('data/video.csv', index_col=0)
    split_data(action, launch, register, video)
