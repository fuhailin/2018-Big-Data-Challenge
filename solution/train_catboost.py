import os
import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics

print('开始处理特征......')

train_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train_feature = train.drop(['user_id', 'label'], axis=1)
train_label = train['label']
test_feature = test.drop(['user_id'], axis=1)

# 切分训练
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2, random_state=1017)

print('特征处理完毕......')

###################### cat ##########################
import catboost as cat

cat2 = cat.CatBoostClassifier(iterations=2000,
                              od_type='Iter',
                              od_wait=120,
                              max_depth=8,
                              learning_rate=0.02,
                              l2_leaf_reg=9,
                              random_seed=2018,
                              metric_period=50,
                              fold_len_multiplier=1.1,
                              loss_function='Logloss',
                              logging_level='Verbose')
cat2 = cat2.fit(X_tr, y_tr, eval_set=(X_va, y_va))
