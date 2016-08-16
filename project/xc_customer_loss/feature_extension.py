# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from project.xc_customer_loss.load_data import train, test
from project.xc_customer_loss.utils import DataShuffle, get_xc_score, get_auc, auc_scoring
from project.xc_customer_loss.utils_feature import bin_iv_woe
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV


obj_var = train.columns[train.dtypes == 'object'].tolist()
cat_var = train.columns[train.dtypes == 'int64'].tolist()
cat_var.remove('label')
con_var = train.columns[train.dtypes == 'float64'].tolist()
# train_y = train['label']

# ################# unique value of continuous variables
for i in con_var:
    print i, len(train[i].unique())

missing_ratio = train[con_var].isnull().sum()/float(len(train))
# print len(train['ordernum_oneyear'].unique())

#########################################################################################
# feature extend
# woe_df
#########################################################################################

################ special feature
extend_df = pd.DataFrame(index=train.index)
d_time = pd.to_datetime(train['d'])
arrival_time = pd.to_datetime(train['arrival'])
extend_df['d_arrival'] = (arrival_time - d_time).apply(lambda x: pd.tslib.Timedelta(x).days)
print extend_df.shape


################# woe value
woe_df = pd.DataFrame(index=train.index)
iv_details = {}
for i in con_var:
    if len(train[i].unique()) > 500:
        print i
        iv_details[i], woe_df[i+'_woe'] = bin_iv_woe(train[i], train['label'], n_bins=50)


def update_woe_data(data, woe_df, details, var_name, tgt, n_bins, encoder=None):
    details[var_name], woe_df[var_name+'_woe'] = bin_iv_woe(data[var_name], tgt, n_bins=n_bins, encoder=encoder)
    return details, woe_df

iv_details, woe_df = update_woe_data(train, woe_df, iv_details, 'h', train['label'], 24)
iv_details, woe_df = update_woe_data(train, woe_df, iv_details, 'ordernum_oneyear', train['label'], 25)
iv_details, woe_df = update_woe_data(train, woe_df, iv_details, 'cr', train['label'], 20)
iv_details, woe_df = update_woe_data(train, woe_df, iv_details, 'historyvisit_7ordernum', train['label'], 50)
iv_details, woe_df = update_woe_data(train, woe_df, iv_details, 'd', train['label'], 50, encoder=LabelEncoder())
iv_details, woe_df = update_woe_data(train, woe_df, iv_details, 'arrival', train['label'], 50, encoder=LabelEncoder())
iv_details, woe_df = update_woe_data(extend_df, woe_df, iv_details, 'd_arrival', train['label'], 50, encoder=LabelEncoder())


# _, temp_woe = bin_iv_woe(extend_df['d_arrival'], train['label'], n_bins=50, encoder=LabelEncoder())



iv_value = {}
for i in iv_details.keys():
    iv_value[i] = iv_details[i][0]['iv'].sum()
iv_value = pd.DataFrame(iv_value.items())
iv_value.sort(1)

#########################################################################################
# testing
#
#########################################################################################



################# test ######################
import xgboost as xgb

x = pd.concat([train, woe_df, extend_df], axis=1, join_axes=[train.index])
x = x.drop(['d', 'arrival', 'label'], axis=1)
x = x.fillna(0)
y = train['label']
sf = DataShuffle(y)
train_x, test_x = sf.get_split_data(x)
train_y, test_y = sf.get_split_data(y)

### fitting
xgb_clf = xgb.XGBClassifier(silent=1, n_estimators=500)
xgb_clf.fit(train_x, train_y, verbose=2)

xgb.plot_importance(xgb_clf)

pred_test = xgb_clf.predict_proba(test_x)
pred_train = xgb_clf.predict_proba(train_x)
get_auc(train_y, pred_train[:, 1])
print get_auc(test_y, pred_test[:, 1])
print get_xc_score(test_y, pred_test[:, 1])

### fitting
def get_importance_features(feature_importance, feature_name, sel=0.5):
    importances_sorted = np.argsort(feature_importance)
    if isinstance(sel, float):
        n = int(len(feature_name) * sel)
    else:
        n = sel
    sel_indx = importances_sorted[-n:]
    return feature_name[sel_indx]

sel_feature = get_importance_features(feature_importance=xgb_clf.feature_importances_, feature_name=x.columns, sel=50)
xgb_clf = xgb.XGBClassifier(silent=1, n_estimators=500)
xgb_clf.fit(train_x[sel_feature], train_y, verbose=2)

# xgb.plot_importance(xgb_clf)

pred_test = xgb_clf.predict_proba(test_x[sel_feature])
pred_train = xgb_clf.predict_proba(train_x[sel_feature])
# get_auc(train_y, pred_train[:, 1])
print get_auc(test_y, pred_test[:, 1])
print get_xc_score(test_y, pred_test[:, 1])
# 0.76937347979
# 0.000176000563202

#########################################################################################
# grid search
#
#########################################################################################

x = pd.concat([train, woe_df, extend_df], axis=1, join_axes=[train.index])
x = x.drop(['d', 'arrival', 'label'], axis=1)
x = x.fillna(0)
y = train['label']
sf = DataShuffle(y)
train_x, test_x = sf.get_split_data(x)
train_y, test_y = sf.get_split_data(y)


parameters = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800],
    'learning_rate': [0.1, 0.01, 0.05, 0.2],
    'max_depth': [3, 4, 5, 7, 8, 10, 12, 15, 20],
    'gamma': [0, 0.5],
    'subsample': [0.5, 0.7],
    'min_child_weight': [1, 0.5, 2],
    'scale_pos_weight': [0.3, 1, 3]
}
xgb_clf = xgb.XGBClassifier(silent=1)
gs_clf = GridSearchCV(xgb_clf, param_grid=parameters, scoring='roc_auc', verbose=2)
gs_clf.fit(x, y)
print gs_clf.grid_scores_
gs_clf.best_score_





