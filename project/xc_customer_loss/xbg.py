# -*- coding: utf-8 -*-
from project.xc_customer_loss.load_data import train, test
from project.xc_customer_loss.utils import *
import numpy as np
import xgboost as xgb
from sklearn.grid_search import GridSearchCV


x = train.drop(['d', 'arrival', 'label'], axis=1)
x = x.fillna(0)
y = train['label']
sf = DataShuffle(y)
train_x, test_x = sf.get_split_data(x)
train_y, test_y = sf.get_split_data(y)

# dm_train = xgb.DMatrix(x, feature_names=x.columns, feature_types=['int']*len(x.columns))



xgb_clf = xgb.XGBClassifier(silent=1)
xgb_clf.fit(train_x, train_y, verbose=1)

pred_test = xgb_clf.predict_proba(test_x)
pred_train = xgb_clf.predict_proba(train_x)
get_auc(train_y, pred_train[:, 1])
get_auc(test_y, pred_test[:, 1])
get_xc_score(train_y, pred_train[:, 1])


parameters = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.1],
    'max_depth': [3, 5, 7, 10],
    'gamma': [0],
    'subsample': [0.5, 0.7],
    'scale_pos_weight': [0.3, 1, 3]
}
xgb_clf = xgb.XGBClassifier(silent=1)
gs_clf = GridSearchCV(xgb_clf, param_grid=parameters, scoring='roc_auc', verbose=2)
gs_clf.fit(x, y)
print gs_clf.grid_scores_
gs_clf.best_score_



