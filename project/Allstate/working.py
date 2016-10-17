# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from cfg.config import data_path
from util.utils import *
from util.utils_feature import cate_feature_mean
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


train_data = pd.read_csv(data_path('Allstate_Claims_Severity') + 'train.csv')
# train_data.index = train_data['id']
train_data = train_data.drop('id', axis=1)
# y = train_data['loss']
x = train_data.drop('loss', axis=1)
discrete_feature = x.dtypes.index[x.dtypes == 'object']  # 116
continuous_feature = x.dtypes.index[x.dtypes == 'float64']  # 14

#######################################################################################
# data exploration
#######################################################################################

for i in discrete_feature:
    n_value = x[i].unique()
    print i, len(n_value)

train_data['cat91'].value_counts()
cate_feature_mean(train_data['cat91'], train_data['loss'])

#######################################################################################
# data preprocess
#######################################################################################

dm_features = pd.get_dummies(train_data[discrete_feature])
dm_feature_count = dm_features.sum()
dm_features_filtered = dm_features.columns[dm_feature_count > 2000]
dm_data = dm_features[dm_features_filtered]


x_all = pd.concat([train_data[continuous_feature], dm_data], axis=1)
y_all = train_data['loss']

sf = DataShuffle(y_all)
train_x, test_x = sf.get_split_data(x_all)
train_y, test_y = sf.get_split_data(y_all)

from scipy.stats import randint as sp_randint


xgbrg = xgb.XGBRegressor(max_depth=20, learning_rate=0.1, n_estimators=50, silent=False)
xgbrg.fit(train_x, train_y)
print 'done'
pred_y = xgbrg.predict(train_x)

mean_absolute_error(train_y, pred_y)


pred_y = xgbrg.predict(test_x)
mean_absolute_error(test_y, pred_y)



print 3
xgbrg
