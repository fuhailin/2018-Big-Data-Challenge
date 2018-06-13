# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:47:58 2018

@author: NURBS
"""

import os
import pandas as pd

print('Generating datasets...')
input_dir = os.path.join(os.pardir, 'input\\sorted_data')
print('Input files:\n{}'.format(os.listdir(input_dir)))
dataset_1_feat_dir = os.path.join(os.pardir, 'input\\dataset_1_feat')
dataset_1_label_dir = os.path.join(os.pardir, 'input\\dataset_1_label')
dataset_2_feat_dir = os.path.join(os.pardir, 'input\\dataset_2_feat')
dataset_2_label_dir = os.path.join(os.pardir, 'input\\dataset_2_label')
dataset_3_feat_dir = os.path.join(os.pardir, 'input\\dataset_3_feat')

print('Loading data sets...')
register = pd.read_csv(input_dir + '/user_register_log.csv')
launch = pd.read_csv(input_dir + '/app_launch_log.csv')
create = pd.read_csv(input_dir + '/video_create_log.csv')
activity = pd.read_csv(input_dir + '/user_activity_log.csv')


def cut_data_on_time(output_path, begin_day, end_day):
    temp_register = register[(register['register_day'] >= begin_day) & (register['register_day'] <= end_day)]
    temp_launch = launch[(launch['launch_day'] >= begin_day) & (launch['launch_day'] <= end_day)]
    temp_create = create[(create['create_day'] >= begin_day) & (create['create_day'] <= end_day)]
    temp_activity = activity[(activity['activity_day'] >= begin_day) & (activity['activity_day'] <= end_day)]

    temp_register.to_csv(output_path + '/register.csv', index=False)
    temp_launch.to_csv(output_path + '/launch.csv', index=False)
    temp_create.to_csv(output_path + '/create.csv', index=False)
    temp_activity.to_csv(output_path + '/activity.csv', index=False)


def generate_dataset():
    print('Cutting train data set 1 ...')
    begin_day = 1
    end_day = 16
    cut_data_on_time(dataset_1_feat_dir, begin_day, end_day)
    begin_day = 17
    end_day = 23
    print('Cutting train data set 2 ...')
    cut_data_on_time(dataset_1_label_dir, begin_day, end_day)
    begin_day = 8
    end_day = 23
    cut_data_on_time(dataset_2_feat_dir, begin_day, end_day)
    begin_day = 24
    end_day = 30
    print('Cutting test data set...')
    cut_data_on_time(dataset_2_label_dir, begin_day, end_day)
    begin_day = 15
    end_day = 30
    cut_data_on_time(dataset_3_feat_dir, begin_day, end_day)


generate_dataset()
print('Dataset generated.')
