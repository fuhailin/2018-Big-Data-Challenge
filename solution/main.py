# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:16:06 2018

@author: Administrator
"""

import os

cmd = "python ./preprocess.py"
os.system(cmd)

cmd = "python ./generate_dataset.py"
os.system(cmd)

cmd = "python ./generate_feature_mine.py"
os.system(cmd)

cmd = "python ./train_lgb_tune_params.py"
os.system(cmd)

cmd = "python ./submit.py"
os.system(cmd)

