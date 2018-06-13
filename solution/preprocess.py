# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:16:49 2018

@author: NURBS
"""

import os
import pandas as pd

input_dir = os.path.join(os.pardir, 'input')
output_dir = os.path.join(os.pardir, 'input\\sorted_data')

app_launch_log = pd.read_csv(input_dir + '/app_launch_log.txt', sep='\t', header=None, names=['user_id', 'launch_day'])
app_launch_log = app_launch_log.sort_values(by=['user_id', 'launch_day'])
app_launch_log.to_csv(output_dir + '/app_launch_log.csv', index=False)

user_activity_log = pd.read_csv(input_dir + '/user_activity_log.txt', sep='\t', header=None, names=['user_id', 'activity_day', 'page', 'video_id', 'author_id', 'action_type'])
user_activity_log = user_activity_log.sort_values(by=['user_id', 'activity_day'])
user_activity_log.to_csv(output_dir + '/user_activity_log.csv', index=False)

user_register_log = pd.read_csv(input_dir + '/user_register_log.txt', sep='\t', header=None, names=['user_id', 'register_day', 'register_type', 'device_type'])
user_register_log = user_register_log.sort_values(by=['user_id', 'register_day'])
user_register_log.to_csv(output_dir + '/user_register_log.csv', index=False)

video_create_log = pd.read_csv(input_dir + '/video_create_log.txt', sep='\t', header=None, names=['user_id', 'create_day'])
video_create_log = video_create_log.sort_values(by=['user_id', 'create_day'])
video_create_log.to_csv(output_dir + '/video_create_log.csv', index=False)
