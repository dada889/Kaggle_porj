# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from project.xc_customer_loss.load_data import train, test
from project.xc_customer_loss.utils import DataShuffle, get_xc_score, get_auc\
    , auc_scoring, KCrossFold, get_importance_features

from project.xc_customer_loss.utils_feature import bin_iv_woe
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
pd.set_option('display.max_columns', 1000)

void_var = ['ordernum_oneyear', 'sampleid']

personal_feature = ['d', 'arrival', 'decisionhabit_user', 'starprefer', 'consuming_capacity', 'price_sensitive', 'ctrip_profits']

one_day_feature = ['iforderpv_24h', 'landhalfhours', 'delta_price1', 'businessrate_pre', 'cr_pre', 'customereval_pre2',
                   'delta_price2', 'commentnums_pre', 'commentnums_pre2', 'cancelrate_pre', 'novoters_pre2',
                   'novoters_pre', 'deltaprice_pre2_t1', 'lowestprice_pre', 'uv_pre', 'uv_pre2', 'lowestprice_pre2',
                   'businessrate_pre2']

seven_day_feature = ['historyvisit_7ordernum', 'historyvisit_visit_detailpagenum']

one_year_feature = ['historyvisit_totalordernum', 'ordercanceledprecent', 'ordercanncelednum', 'lasthtlordergap',
                    'lastpvgap', 'customer_value_profit', 'visitnum_oneyear', 'historyvisit_avghotelnum']

city_feature = ['cityuvs', 'cityorders']

hotel_var = ['hotelcr', 'commentnums', 'novoters', 'cancelrate', 'hoteluv', 'lowestprice']


other_feature = ['cr', 'sid', 'h', 'firstorder_bu', 'avgprice']

train_ = train.fillna(0)
all_feature = void_var + personal_feature + one_day_feature + seven_day_feature + one_year_feature + city_feature + hotel_var + other_feature
train_id_index = train.set_index(train['sampleid'])



def checkmylist(mylist):
    it = (x for x in mylist)
    first = next(it)
    return all(a == b for a, b in enumerate(it, first + 1))


def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))



###########################################################################
# first level detection
###########################################################################

person_dect = personal_feature + one_year_feature
print len(train_[person_dect].drop_duplicates())  # 160454

gb_obj = train_.groupby(person_dect)
id_list = gb_obj['sampleid'].unique().values
sid_list = gb_obj['sid']
temp = gb_obj[['sid', 'sampleid']]
# temp.reset_index()

a = {'sid': [], 'sampleid': [], 'label': []}
for ix, value in temp:
    a['sid'].append(value['sid'].tolist())
    a['sampleid'].append(value['sampleid'].tolist())
    a['label'].append(value['label'].tolist())

personal_df = pd.DataFrame(a)
print personal_df.shape
personal_df['unique_label'] = personal_df['label'].apply(lambda x: len(set(x)) == 1)
personal_df['uniq_sid'] = personal_df['sid'].apply(lambda x: set(x))
personal_df['continu_sid'] = personal_df['uniq_sid'].apply(lambda x: checkmylist(list(x)))
# personal_df['sid_increm'] = personal_df['sid'].apply(lambda x: non_decreasing(x))
# personal_df['id_increm'] = personal_df['sampleid'].apply(lambda x: non_decreasing(x))
personal_df['valid'] = personal_df['continu_sid']

# personal_df.ix[personal_df['continu_sid'], 'unique_label']
personal_1 = personal_df.ix[personal_df['valid'], :]  # 139880
temp = personal_1['sampleid'].apply(lambda x: len(x))
# personal_df.ix[(personal_df['continu_sid'] == True) & (personal_df['unique_label'] == False), :]
#
# personal_df.ix[~personal_df.ix[personal_df['continu_sid'], 'unique_label'], :]


faul_sampleid = personal_df.ix[~personal_df['valid'], 'sampleid'].values
print len(faul_sampleid)   # 20574
wrong_sampleid = []
for i in faul_sampleid:
    wrong_sampleid += i
print len(wrong_sampleid)  # 241034

###########################################################################
# second level detection
###########################################################################

train_part = train_id_index.ix[wrong_sampleid, :]
train_part = train_part.fillna(0)
person_dect = personal_feature + one_year_feature + seven_day_feature + city_feature + ['h']
print len(train_part[person_dect].drop_duplicates())  # 34719

gb_obj = train_part.groupby(person_dect)
print len(gb_obj)
# id_list = gb_obj['sampleid'].unique().values
# sid_list = gb_obj['sid']
temp = gb_obj[['sid', 'sampleid']]
# temp.reset_index()

a = {'sid': [], 'sampleid': [], 'label': []}
for ix, value in temp:
    a['sid'].append(value['sid'].tolist())
    a['sampleid'].append(value['sampleid'].tolist())
    a['label'].append(value['label'].tolist())

personal_df_2 = pd.DataFrame(a)
print personal_df_2.shape
personal_df_2['unique_label'] = personal_df_2['label'].apply(lambda x: len(set(x)) == 1)
personal_df_2['uniq_sid'] = personal_df_2['sid'].apply(lambda x: set(x))
personal_df_2['continu_sid'] = personal_df_2['uniq_sid'].apply(lambda x: checkmylist(list(x)))
personal_df_2['sid_increm'] = personal_df_2['sid'].apply(lambda x: non_decreasing(x))
personal_df_2['id_increm'] = personal_df_2['sampleid'].apply(lambda x: non_decreasing(x))
personal_df_2['valid'] = personal_df_2['sid_increm'] & personal_df_2['continu_sid']

personal_df_2.ix[personal_df_2['valid'], 'unique_label']
personal_2 = personal_df_2.ix[personal_df_2['valid'], :]  # 13596
# temp = personal_2['sampleid'].apply(lambda x: len(x))

faul_sampleid = personal_df_2.ix[~personal_df_2['valid'], 'sampleid'].values
print len(faul_sampleid)  # 21066
wrong_sampleid = []
for i in faul_sampleid:
    wrong_sampleid += i
print len(wrong_sampleid)  # 201239

###########################################################################
# third level detection
###########################################################################
train_third = train_id_index.ix[wrong_sampleid, :]

person_dect = personal_feature + one_year_feature + seven_day_feature + city_feature + ['h']
print len(train_third[person_dect].drop_duplicates())

gb_obj = train_third.groupby(person_dect)
# id_list = gb_obj['sampleid'].unique().values
# sid_list = gb_obj['sid']
temp = gb_obj[['sid', 'sampleid']]
# temp.reset_index()

a = {'sid': [], 'sampleid': [], 'label': []}
for ix, value in temp:
    a['sid'].append(value['sid'].tolist())
    a['sampleid'].append(value['sampleid'].tolist())
    a['label'].append(value['label'].tolist())

personal_df_3 = pd.DataFrame(a)
print personal_df_3.shape
personal_df_3['unique_label'] = personal_df_3['label'].apply(lambda x: len(set(x)) == 1)
personal_df_3['uniq_sid'] = personal_df_3['sid'].apply(lambda x: set(x))
personal_df_3['continu_sid'] = personal_df_3['uniq_sid'].apply(lambda x: checkmylist(list(x)))

print all(personal_df_3.ix[personal_df_3['continu_sid'], 'unique_label'])
personal_3 = personal_df_3.ix[personal_df_3['continu_sid'], :]
# personal_df_3.ix[personal_df_3['uniq_sid'].apply(lambda x: len(x)) > 1, :]
temp = personal_3['sampleid'].apply(lambda x: len(x))

faul_sampleid = personal_df_3.ix[~personal_df_3['continu_sid'], 'sampleid'].values
print len(faul_sampleid)  # 64

wrong_sampleid = []
for i in faul_sampleid:
    wrong_sampleid += i
print len(wrong_sampleid) #346


###########################################################################
# four level detection
###########################################################################
train_four = train_id_index.ix[wrong_sampleid, :]

person_dect = personal_feature + one_year_feature + seven_day_feature + city_feature + ['h', 'sid']
print len(train_four[person_dect].drop_duplicates())


gb_obj = train_four.groupby(person_dect)
id_list = gb_obj['sampleid'].unique().values
sid_list = gb_obj['sid']
temp = gb_obj[['sid', 'sampleid']]
# temp.reset_index()

a = {'sid': [], 'sampleid': [], 'label': []}
for ix, value in temp:
    a['sid'].append(value['sid'].tolist())
    a['sampleid'].append(value['sampleid'].tolist())
    a['label'].append(value['label'].tolist())

personal_df_4 = pd.DataFrame(a)
print personal_df_4.shape
personal_df_4['unique_label'] = personal_df_4['label'].apply(lambda x: len(set(x)) == 1)
personal_df_4['uniq_sid'] = personal_df_4['sid'].apply(lambda x: set(x))
personal_df_4['continu_sid'] = personal_df_4['uniq_sid'].apply(lambda x: checkmylist(list(x)))

print all(personal_df_4.ix[personal_df_3['continu_sid'], 'unique_label'])
personal_4 = personal_df_4.ix[personal_df_4['continu_sid'], :]
temp = personal_4['sampleid'].apply(lambda x: len(x))



faul_sampleid = personal_df_4.ix[~personal_df_4['continu_sid'], 'sampleid'].values
len(faul_sampleid)

wrong_sampleid = []
for i in faul_sampleid:
    wrong_sampleid += i
print len(wrong_sampleid)  # 0


###########################################################################
# data reshape
###########################################################################










#########################################################################################
# feature extend
# woe_df
#########################################################################################

################ special feature
extend_df = pd.DataFrame(index=train.index)
d_time = pd.to_datetime(train['d'])
# d_weekday = d_time.dt.dayofweek
arrival_time = pd.to_datetime(train['arrival'])
extend_df['arrival_weekday'] = arrival_time.dt.dayofweek
extend_df['d_arrival'] = (arrival_time - d_time).dt.days
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
iv_details, woe_df = update_woe_data(extend_df, woe_df, iv_details, 'arrival_weekday', train['label'], 50, encoder=LabelEncoder())


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
x = x.drop(['d', 'arrival', 'label', 'simpleid'], axis=1)
x = x.fillna(0)
y = train['label']

kf = KCrossFold(len(y))
train_x, test_x, train_y, test_y = kf.get_data(x, y)


sf = DataShuffle(y)
sf.shuffle_split()
train_x, test_x = sf.get_split_data(x)
train_y, test_y = sf.get_split_data(y)

### fitting
xgb_clf = xgb.XGBClassifier(silent=1, n_estimators=500, max_depth=10)
xgb_clf.fit(train_x, train_y, verbose=2)
xgb.plot_importance(xgb_clf)
# get_importance_features(xgb_clf.feature_importances_, train_x.columns)
# fea = pd.DataFrame(xgb_clf.feature_importances_, index=train_x.columns)
# fea.sort(0)
pred_test = xgb_clf.predict_proba(test_x)
pred_train = xgb_clf.predict_proba(train_x)






print 'train auc: %s' % get_auc(train_y, pred_train[:, 1])
print 'test auc: %s' % get_auc(test_y, pred_test[:, 1])
print 'xiecheng score: %s' % get_xc_score(test_y, pred_test[:, 1])




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
    'n_estimators': [800, 1000, 1500],
    # 'learning_rate': [0.05],
    'max_depth': [10],
    # 'gamma': [0, 0.5],
    # 'subsample': [0.5, 0.7],
    # 'min_child_weight': [1, 0.5, 2],
    # 'scale_pos_weight': [0.3, 1, 3]
}
xgb_clf = xgb.XGBClassifier(silent=1)
gs_clf = GridSearchCV(xgb_clf, param_grid=parameters, scoring='roc_auc', verbose=2)
gs_clf.fit(x, y)
print gs_clf.grid_scores_
gs_clf.best_score_

pred_test = xgb_clf.predict_proba(test_x)
pred_train = xgb_clf.predict_proba(train_x)
print 'train auc: %s' % get_auc(train_y, pred_train[:, 1])
print 'test auc: %s' % get_auc(test_y, pred_test[:, 1])
print 'xiecheng score: %s' % get_xc_score(test_y, pred_test[:, 1])





