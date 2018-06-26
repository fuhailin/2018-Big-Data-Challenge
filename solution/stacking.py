# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

target = 'label_x'
IDcol = 'user_id'
test = pd.read_csv('train_and_test_addauthor0/testauthormrg.csv')
train = pd.read_csv('train_and_test_addauthor0/trainauthormrg.csv')
predictors = [x for x in train.columns if x not in [target, IDcol, 'label', 'label_y', 'create_weekendcount']]

# Ensembling & Stacking models
# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


# Class to extend XGboost classifer
# Out-of-Fold Predictions
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Generating our Base First-Level Models¶
# Put in our parameters for said classifiers
# Random Forest parameters
lgb_params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['auc'],
              'num_leaves': 40, 'max_depth': 6, 'verbose': 1, 'max_bin': 249, 'min_data_in_leaf': 36,
              'feature_fraction': 0.85, 'bagging_fraction': 0.9, 'bagging_freq': 5,
              'lambda_l1': 1.0, 'lambda_l2': 0.7, 'min_split_gain': 0.0,
              'learning_rate': 0.01}

# Extra Trees Parameters
xgb_params = {'objective': 'binary:logistic', 'booster': 'gbtree',
              'max_depth': 5,
              # 'eval_metric': 'auc',
              'eta': 0.01,
              'silent': 0,
              'num_leaves': 20,
              'gamma': 2,
              'colsample_bytree ': 1
              }

# Create 5 objects that represent our 4 models
xg = SklearnHelper(clf=xgb.XGBClassifier, seed=SEED, params=xgb_params)
lg = SklearnHelper(clf=lgb.LGBMClassifier, seed=SEED, params=lgb_params)
# ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
# gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
# svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train[target].ravel()  # flatten
# train = train.drop(['Survived'], axis=1)
x_train = train[predictors].values  # Creates an array of the train data
x_test = test[predictors].values  # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
lg_oof_train, lg_oof_test = get_oof(lg, x_train, y_train, x_test)  # LightGBM
xg_oof_train, xg_oof_test = get_oof(xg, x_train, y_train, x_test)  # XGBoost
# ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
# gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
# svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")

lg_feature = lg.feature_importances(x_train, y_train)
xg_feature = xg.feature_importances(x_train, y_train)
# ada_feature = ada.feature_importances(x_train, y_train)
# gb_feature = gb.feature_importances(x_train,y_train)

# x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
# x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
x_train = np.concatenate((lg_oof_train, xg_oof_train), axis=1)
x_test = np.concatenate((lg_oof_test, xg_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    # learning_rate = 0.02,
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    # gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(x_train, y_train)
# predictions = gbm.predict(x_test)
y_submission = gbm.predict_proba(x_test)[:, 1]

# Generate Submission File
# StackingSubmission = pd.DataFrame({ 'user_id': PassengerId, 'score': predictions })
# StackingSubmission.to_csv("StackingSubmission.csv", index=False)

########################## 保存结果 ############################
print('Save result...')
df_result = pd.DataFrame()
df_result['user_id'] = test['user_id']
df_result['result'] = y_submission
df_result.to_csv('../result/stack_result.csv', index=False)
threshold = 0.418
y_submission[y_submission >= threshold] = 1
y_submission[y_submission < threshold] = 0
prediction = list(map(int, y_submission))
print('为1的个数：' + str(len(np.where(np.array(prediction) == 1)[0])))
print('为0的个数：' + str(len(np.where(np.array(prediction) == 0)[0])))

active_user_id = df_result.sort_values(by='result', ascending=False)
# active_user_id = active_user_id.head(23800)
active_user_id = df_result[df_result['result'] >= threshold]
# active_user_id = result.sort_values(by='result', axis=0, ascending=False).iloc[0:23760,:]
# print('threshold:',active_user_id.iloc[-1,1])
print(len(active_user_id))

del active_user_id['result']
from datetime import datetime

active_user_id.to_csv('../submission/stack_lgb_xgb_result_{}.txt'.format(datetime.now().strftime('%m%d_%H%M')), index=False, header=False)
