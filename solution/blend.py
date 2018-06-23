from __future__ import division

import sklearn
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

sklearn_version = sklearn.__version__
print('The scikit-learn version is {}.'.format(sklearn.__version__))

sklearn_version = sklearn_version.split('.')
main_sklearn_verison = int(sklearn_version[1])

current_scikit_verison_flag = True

if main_sklearn_verison < 18:
    print('Your version of scikit learn is less than version 18.')
    print('Denson will stop supporting versions less than 18 in March 2017.')
    current_scikit_verison_flag = False

if current_scikit_verison_flag:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
else:
    from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit

if __name__ == '__main__':
    np.random.seed(0)  # seed to shuffle the train set

    target = 'label_x'
    IDcol = 'user_id'
    # Number of folds between 5-20 is usually best...your milage may vary
    n_folds = 5
    verbose = True
    shuffle = False

    # Generate the problem
    test = pd.read_csv('train_and_test_addauthor0/testauthormrg.csv')
    train = pd.read_csv('train_and_test_addauthor0/trainauthormrg.csv')
    df_result = pd.DataFrame()
    df_result['user_id'] = test['user_id']
    predictors = [x for x in train.columns if x not in [target, IDcol, 'label', 'label_y', 'create_weekendcount']]
    X_gen = train[predictors]
    y_gen = train[target]
    X_submission = test[predictors]

    if shuffle:
        idx = np.random.permutation(y_gen.size)
        X_gen = X_gen[idx]
        y_gen = y_gen[idx]

    # We run the classifiers with different parameters to make the predictions
    # less correlated.
    lgb_params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['auc'],
                  'num_leaves': 40, 'max_depth': 6, 'verbose': 1, 'max_bin': 249, 'min_data_in_leaf': 36,
                  'feature_fraction': 0.85, 'bagging_fraction': 0.9, 'bagging_freq': 5,
                  'lambda_l1': 1.0, 'lambda_l2': 0.7, 'min_split_gain': 0.0,
                  'learning_rate': 0.01}
    xgb_params = {'objective': 'binary:logistic', 'booster': 'gbtree',
                  'max_depth': 5,
                  # 'eval_metric': 'auc',
                  'eta': 0.01,
                  'silent': 1,
                  'num_leaves': 20,
                  'gamma': 2,
                  'colsample_bytree ': 1
                  }

    clfs = [lgb.LGBMClassifier(**lgb_params),
            xgb.XGBClassifier(**xgb_params)
            # cb.CatBoostClassifier()
            ]

    print("Creating train and test sets for blending.")

    # These arrays will hold the blended predictions
    dataset_blend_train = np.zeros((X_gen.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    # Use SSS to create training and test sets
    if current_scikit_verison_flag:
        # For each fold train on 80% and test on 20%. If you have a bunch of data
        # you might want to make it a 50/50 split.
        sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.2)
        # We will resuse the same train-test splits for each of the models.
        splits = list(sss.split(X_gen, y_gen))
    else:
        sss = StratifiedShuffleSplit(y_gen, n_folds, test_size=0.2)
        splits = list(sss)

    data_array = X_gen.values
    for jdx, clf in enumerate(clfs):
        print('jdx, clf', jdx, clf)
        # dataset__blend_test_j is for this fold of this model (RDF-gini, fold 1 etc)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(splits)))
        for idx, (train, test) in enumerate(splits):
            print("Fold", idx)
            # Split the training data into train-test sets for this fold
            X_fold_train = data_array[train]
            y_fold_train = y_gen[train]
            X_fold_test = data_array[test]
            y_fold_test = y_gen[test]

            # Fit this model on this fold of data
            clf.fit(X_fold_train, y_fold_train)

            # Predict this test fold
            y_fold_pred = clf.predict_proba(X_fold_test)[:, 1]

            '''
            This is where things get slightly confusing. We are using part of the
            training data to predict the rest of the training data. We store the
            predictions as the transformed training data. A given row of the 
            training data is likely to be predicted more than once and will wind
            up only with the last prediction. There is no absolute guarantee that 
            every row of the training data will be predicted but it is highly
            likely in 10 folds.
            '''
            # Store the predictions as transformed training data
            # jdx is the index of this model (RDF-gini)
            dataset_blend_train[test, jdx] = y_fold_pred

            '''
            Now we use this model and use it to predict the holdout test data.
            We store the predictions of each fold of this model and take the mean
            at the end to create the transformation for this model.
            '''
            dataset_blend_test_j[:, idx] = clf.predict_proba(X_submission.values)[:, 1]

        # Take the mean prediction of each fold and use it as the transformed
        # test data.
        dataset_blend_test[:, jdx] = dataset_blend_test_j.mean(1)

    print("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y_gen)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    # print("Linear stretch of predictions to [0,1]")
    # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    ########################## 保存结果 ############################
    print('Save result...')
    # df_result = pd.DataFrame()
    # df_result['user_id'] = test['user_id']
    df_result['result'] = y_submission
    df_result.to_csv('../result/blend_result.csv', index=False)
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

    active_user_id.to_csv('../submission/blend_lgb_xgb_result_{}.txt'.format(datetime.now().strftime('%m%d_%H%M')), index=False, header=False)
