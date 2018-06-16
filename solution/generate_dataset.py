# -*- coding: utf-8 -*-
import os
import pandas as pd

print('Generating datasets...')
input_dir = os.path.join(os.pardir, 'input\\sorted_data')
print('Input files:\n{}'.format(os.listdir(input_dir)))
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

    temp_register.to_csv(output_path + '_register.csv', index=False)
    temp_launch.to_csv(output_path + '_launch.csv', index=False)
    temp_create.to_csv(output_path + '_create.csv', index=False)
    temp_activity.to_csv(output_path + '_activity.csv', index=False)

'''
def extra_user_on_time(user_dir, begin_day, end_day):
    temp_register = register[(register['register_day'] >= begin_day) & (register['register_day'] <= end_day)]
    temp_register.to_csv(user_dir + '_register.csv', index=False)
'''

def generate_dataset():
    print('Cutting train data set 1 ...')
    # extra_user_on_time(train_1_user_dir, begin_day=1, end_day=19)
    cut_data_on_time(train_1_feat_dir, begin_day=1, end_day=19)
    cut_data_on_time(train_1_label_dir, begin_day=20, end_day=26)

    print('Cutting train data set 2 ...')
    # extra_user_on_time(train_2_user_dir, begin_day=1, end_day=20)
    cut_data_on_time(train_2_feat_dir, begin_day=2, end_day=20)
    cut_data_on_time(train_2_label_dir, begin_day=21, end_day=27)

    print('Cutting train data set 3 ...')
    # extra_user_on_time(train_3_user_dir, begin_day=1, end_day=21)
    cut_data_on_time(train_3_feat_dir, begin_day=3, end_day=21)
    cut_data_on_time(train_3_label_dir, begin_day=22, end_day=28)

    print('Cutting train data set 4 ...')
    # extra_user_on_time(train_4_user_dir, begin_day=1, end_day=22)
    cut_data_on_time(train_4_feat_dir, begin_day=4, end_day=22)
    cut_data_on_time(train_4_label_dir, begin_day=23, end_day=29)

    print('Cutting train data set 5 ...')
    # extra_user_on_time(train_5_user_dir, begin_day=1, end_day=23)
    cut_data_on_time(train_5_feat_dir, begin_day=5, end_day=23)
    cut_data_on_time(train_5_label_dir, begin_day=24, end_day=30)

    print('Cutting test data set...')
    # extra_user_on_time(test_user_dir, begin_day=1, end_day=30)
    cut_data_on_time(test_feat_dir, begin_day=12, end_day=30)


generate_dataset()
print('Dataset generated.')
