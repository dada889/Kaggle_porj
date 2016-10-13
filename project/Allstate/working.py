# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from cfg.config import data_path
from util.utils import *
from util.utils_feature import cate_feature_mean
from sklearn.preprocessing import LabelEncoder


train_data = pd.read_csv(data_path('Allstate_Claims_Severity') + 'train.csv')
y = train_data['loss']
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

pd.get_dummies(train_data['cat1'])



sf = DataShuffle(y)
train_x, test_x = sf.get_split_data(x)
train_y, test_y = sf.get_split_data(y)



