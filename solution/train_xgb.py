import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import os

target = 'label_x'
IDcol = 'user_id'

path = os.path.join(os.path.curdir, 'data')

# ------------load data----------
test = pd.read_csv('train_and_test_addauthor0/testauthormrg.csv')
train = pd.read_csv('train_and_test_addauthor0/trainauthormrg.csv')
predictors = [x for x in train.columns if x not in [target, IDcol,'label','label_y']]

# 切分训练
X_train, X_test, Y_train, Y_test = train_test_split(train[predictors], train[target], test_size=0.3, random_state=2018)

eval_set = [(X_test, Y_test)]

params = {
    'gamma': 0,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'iterations': 100,
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 1,
    'learning_rate': 0.01,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,  # 在各类别样本十分不平衡时，把这个参数设定为正值，可以是算法更快收敛
    'booster': 'gbtree',
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.01,  # 如同学习率
    'seed': 1000,
    'nthread': 4,  # cpu 线程数
}

grid_params = {
    'learning_rate': [0.01, 0.05, 0.1],  # 得到最佳参数0.01，
    'max_depth': [5, 6, 7, 8, 12],
    'min_child_weight': [0.1, 0.5, 1, 2],
    'n_estimators': [100, 200, 500, ],
}
#
# GridSearch code

clf = xgb.XGBClassifier(**params)
grid = GridSearchCV(clf, grid_params)
grid.fit(X_train, Y_train)
print(grid.best_params_)
print("Accuracy:{0:.1f}%".format(100 * grid.best_score_))

# model code
model1 = xgb.XGBClassifier(**params)
model1.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
y_pred1 = model1.predict(X_test)

print(f'AUC: {metrics.roc_auc_score(Y_test, y_pred1)}')
print(f'F1: {metrics.f1_score(Y_test, y_pred1)}')
print(f'特征重要性：{list(model1.feature_importances_)}')
'''
index = []
for i, f in enumerate(list(model1.feature_importances_)):
    if f >= 0.01:
        index.append(i)
print(index)

plot_importance(model1)
# plt.show()
print(model1.get_booster().get_fscore().keys())

# valid
valid_label = model1.predict(_valid_data)
valid_label2df = pd.DataFrame({'valid_label': valid_label})
valid = pd.concat([valid_data, valid_label2df], axis=1)
pre_ids = set(valid[valid['valid_label'] == 1]['user_id'])
rel_ids = set(valid_data[valid_data['label'] == 1]['user_id'])

Precision = len(pre_ids.intersection(rel_ids)) / len(pre_ids)
Recall = len(pre_ids.intersection(rel_ids)) / len(rel_ids)
F1 = 2 * Precision * Recall / (Precision + Recall)

print(f"Precision: {Precision}\nRecall:{Recall}\nF1:{F1}")
#
# submit
submit_pred = model1.predict(test)
pred_label = pd.DataFrame({'label': submit_pred})
submit = pd.concat([test_data, pred_label], axis=1)
submit = submit[submit['label'] == 1]['user_id']
print(submit.shape)
submit.to_csv(os.path.join(path,'submission12.csv'),encoding='utf-8',index=None,header=None)
'''
