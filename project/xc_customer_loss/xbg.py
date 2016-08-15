# -*- coding: utf-8 -*-
from project.xc_customer_loss.load_data import train, test
from project.xc_customer_loss.utils import *
import numpy as np
import xgboost as xgb


x = train.drop(['d', 'arrival', 'label'], axis=1)
x = x.fillna(0)
y = train['label']
sf = DataShuffle(y)
train_x, test_x = sf.get_split_data(x)
train_y, test_y = sf.get_split_data(y)

dm_train = xgb.DMatrix(x, feature_names=x.columns, feature_types=['int']*len(x.columns))



xgb_clf = xgb.XGBClassifier(silent=1, )
xgb_clf.fit(train_x, train_y)

pred_test = xgb_clf.predict_proba(test_x)
pred_train = xgb_clf.predict_proba(train_y)
xgb_clf.feature_importances_


get_auc(test_y, pred_test[:, 1])
get_auc(train_y, pred_train[:, 1])


x.dtypes.values
[int]*len(x.columns)

