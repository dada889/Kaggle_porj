# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from project.xc_customer_loss.load_data import train, test
from project.xc_customer_loss.utils import DataShuffle, get_xc_score, get_auc\
    , auc_scoring, KCrossFold, get_importance_features

from project.xc_customer_loss.utils_feature import bin_iv_woe
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
import time
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

hotel_var = ['hotelcr', 'commentnums', 'novoters', 'cancelrate', 'hoteluv', 'lowestprice', 'cr']


other_feature = ['sid', 'h', 'firstorder_bu', 'avgprice']

train_ = train.fillna(0)
all_feature = void_var + personal_feature + one_day_feature + seven_day_feature + one_year_feature + city_feature + hotel_var + other_feature
train_id_index = train.set_index(train['sampleid'])
train_id_index = train_id_index.fillna(0)



def checkmylist(mylist):
    it = (x for x in mylist)
    first = next(it)
    return all(a == b for a, b in enumerate(it, first + 1))


def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))



###########################################################################
# first level detection
###########################################################################

person_dect = personal_feature + one_year_feature + city_feature
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

# personal_df.ix[personal_df['continu_sid'], 'unique_label']  # 177776/177903
personal_1 = personal_df.ix[personal_df['valid'], :]  # 139880
temp = personal_1['sampleid'].apply(lambda x: len(x))
# personal_df.ix[(personal_df['continu_sid'] == True) & (personal_df['unique_label'] == False), :]
temp = personal_df.ix[(personal_df['continu_sid'] == True) & (personal_df['unique_label'] == False), :]
l = []
for i in temp['label']:
    l += i
print len(l)

# personal_df.ix[~personal_df.ix[personal_df['continu_sid'], 'unique_label'], :]


faul_sampleid = personal_df.ix[~personal_df['valid'], 'sampleid'].values
print len(faul_sampleid)   # 10443
wrong_sampleid = []
for i in faul_sampleid:
    wrong_sampleid += i
print len(wrong_sampleid)  # 124057

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
# personal_df_2['sid_increm'] = personal_df_2['sid'].apply(lambda x: non_decreasing(x))
# personal_df_2['id_increm'] = personal_df_2['sampleid'].apply(lambda x: non_decreasing(x))
personal_df_2['valid'] = personal_df_2['continu_sid']

# personal_df_2.ix[personal_df_2['valid'], 'unique_label']  # 31839/32052
personal_2 = personal_df_2.ix[personal_df_2['valid'], :]  # 32052
# personal_df_2.ix[(personal_df_2['continu_sid'] == True) & (personal_df_2['unique_label'] == False), :]
l = []
for i in temp:
    l += i
print l

# temp = personal_2['sampleid'].apply(lambda x: len(x))

faul_sampleid = personal_df_2.ix[~personal_df_2['valid'], 'sampleid'].values
print len(faul_sampleid)  # 2962
wrong_sampleid = []
for i in faul_sampleid:
    wrong_sampleid += i
print len(wrong_sampleid)  # 27373

###########################################################################
# third level detection
###########################################################################
train_third = train_id_index.ix[wrong_sampleid, :]

person_dect = personal_feature + one_year_feature + seven_day_feature + city_feature + ['h', 'sid']
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

print all(personal_df_3.ix[personal_df_3['continu_sid'], 'unique_label'])  # 7877/8169
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
# train_four = train_id_index.ix[wrong_sampleid, :]
#
# person_dect = personal_feature + one_year_feature + seven_day_feature + city_feature + ['h', 'sid']
# print len(train_four[person_dect].drop_duplicates())
#
#
# gb_obj = train_four.groupby(person_dect)
# id_list = gb_obj['sampleid'].unique().values
# sid_list = gb_obj['sid']
# temp = gb_obj[['sid', 'sampleid']]
# # temp.reset_index()
#
# a = {'sid': [], 'sampleid': [], 'label': []}
# for ix, value in temp:
#     a['sid'].append(value['sid'].tolist())
#     a['sampleid'].append(value['sampleid'].tolist())
#     a['label'].append(value['label'].tolist())
#
# personal_df_4 = pd.DataFrame(a)
# print personal_df_4.shape
# personal_df_4['unique_label'] = personal_df_4['label'].apply(lambda x: len(set(x)) == 1)
# personal_df_4['uniq_sid'] = personal_df_4['sid'].apply(lambda x: set(x))
# personal_df_4['continu_sid'] = personal_df_4['uniq_sid'].apply(lambda x: checkmylist(list(x)))
#
# print all(personal_df_4.ix[personal_df_3['continu_sid'], 'unique_label'])
# personal_4 = personal_df_4.ix[personal_df_4['continu_sid'], :]
# temp = personal_4['sampleid'].apply(lambda x: len(x))
#
#
#
# faul_sampleid = personal_df_4.ix[~personal_df_4['continu_sid'], 'sampleid'].values
# len(faul_sampleid)
#
# wrong_sampleid = []
# for i in faul_sampleid:
#     wrong_sampleid += i
# print len(wrong_sampleid)  # 0


###########################################################################
# data reshape
###########################################################################

persons_df = pd.concat([personal_1, personal_2, personal_3])
persons_df['n_sampleid'] = persons_df['sampleid'].apply(lambda x: len(x))
# persons_df['label']
# persons_df['unique_label']
persons_df['p_label'] = persons_df.apply(lambda x: x['label'][0] if x['unique_label'] else 0, axis=1)
persons_df = persons_df.reset_index(drop=True)
non_feature = ['firstorder_bu']
max_feature = ['historyvisit_7ordernum', 'historyvisit_totalordernum',
               'landhalfhours', 'ordercanncelednum', 'historyvisit_visit_detailpagenum',
               'ordernum_oneyear', 'lasthtlordergap', 'arrival']
min_feature = ['lowestprice_pre2', 'd']
personal_feature_ = ['decisionhabit_user', 'starprefer', 'consuming_capacity', 'price_sensitive', 'ctrip_profits']
mean_feature = hotel_var + personal_feature_ + city_feature +\
               ['historyvisit_avghotelnum', 'delta_price1', 'businessrate_pre', 'cr_pre',
                'avgprice', 'customereval_pre2', 'delta_price2', 'commentnums_pre',
                'customer_value_profit', 'commentnums_pre2', 'cancelrate_pre', 'novoters_pre2',
                'novoters_pre', 'deltaprice_pre2_t1', 'lowestprice_pre', 'uv_pre', 'uv_pre2',
                'businessrate_pre2', 'visitnum_oneyear', 'ordercanceledprecent'] + non_feature

print time.asctime()
train_id_index = train.set_index('sampleid')
train_id_index['group'] = -1
sampleid = []
uid = []

for ix, i in persons_df.iterrows():
    for j in i['sampleid']:
        sampleid.append(j)
        uid.append(ix)
train_id_index.ix[sampleid, 'group'] = uid

max_df = train_id_index.groupby('group')[max_feature].max()
min_df = train_id_index.groupby('group')[min_feature].min()
mean_df = train_id_index.groupby('group')[mean_feature].mean()
n_id = train_id_index.groupby('group')['sid'].count()
n_sid = train_id_index.groupby('group')['sid'].nunique()

print 'done'

person_details_df = pd.concat([max_df, min_df, mean_df], axis=1)
# person_details_df['lable'] = -1
person_details_df.ix[persons_df.index, 'label'] = persons_df['p_label']
# person_details_df['n_id'] = -1
person_details_df.ix[n_id.index, 'n_id'] = n_id
# person_details_df['n_sid'] = -1
person_details_df.ix[n_sid.index, 'n_sid'] = n_sid


#########################################################################################
# feature extend
# woe_df
#########################################################################################







################ special feature
extend_df = pd.DataFrame(index=person_details_df.index)
d_time = pd.to_datetime(person_details_df['d'])
# d_weekday = d_time.dt.dayofweek
arrival_time = pd.to_datetime(person_details_df['arrival'])
extend_df['arrival_weekday'] = arrival_time.dt.dayofweek
extend_df['d_arrival'] = (arrival_time - d_time).dt.days
print extend_df.shape


################# woe value
woe_df = pd.DataFrame(index=person_details_df.index)
iv_details = {}
woe_var = person_details_df.columns[person_details_df.isnull().sum() > 10000]
for i in woe_var:
    if len(train[i].unique()) > 500:
        print i
        iv_details[i], woe_df[i+'_woe'] = bin_iv_woe(person_details_df[i], person_details_df['label'], n_bins=40)


def update_woe_data(data, woe_df, details, var_name, tgt, n_bins, encoder=None):
    details[var_name], woe_df[var_name+'_woe'] = bin_iv_woe(data[var_name], tgt, n_bins=n_bins, encoder=encoder)
    return details, woe_df

# iv_details, woe_df = update_woe_data(person_details_df, woe_df, iv_details, 'h', person_details_df['label'], 24)
# iv_details, woe_df = update_woe_data(person_details_df, woe_df, iv_details, 'ordernum_oneyear', person_details_df['label'], 25)
# iv_details, woe_df = update_woe_data(person_details_df, woe_df, iv_details, 'cr', person_details_df['label'], 20)
# iv_details, woe_df = update_woe_data(person_details_df, woe_df, iv_details, 'historyvisit_7ordernum', person_details_df['label'], 50)
iv_details, woe_df = update_woe_data(person_details_df, woe_df, iv_details, 'd', person_details_df['label'], 50, encoder=LabelEncoder())
iv_details, woe_df = update_woe_data(person_details_df, woe_df, iv_details, 'arrival', person_details_df['label'], 50, encoder=LabelEncoder())
iv_details, woe_df = update_woe_data(extend_df, woe_df, iv_details, 'd_arrival', person_details_df['label'], 50, encoder=LabelEncoder())
iv_details, woe_df = update_woe_data(extend_df, woe_df, iv_details, 'arrival_weekday', person_details_df['label'], 50, encoder=LabelEncoder())


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

x = pd.concat([person_details_df, woe_df, extend_df], axis=1, join_axes=[person_details_df.index])
# x = person_details_df
x = x.drop(['d', 'arrival', 'label'], axis=1)
x = x.fillna(0)
y = person_details_df['label']

kf = KCrossFold(len(y))
train_x, test_x, train_y, test_y = kf.get_data(x, y)


# sf = DataShuffle(y)
# sf.shuffle_split()
# train_x, test_x = sf.get_split_data(x)
# train_y, test_y = sf.get_split_data(y)

### fitting
xgb_clf = xgb.XGBClassifier(silent=1, n_estimators=700, max_depth=10, subsample=0.5, min_child_weight=800)
xgb_clf.fit(train_x, train_y, verbose=2)
# xgb.plot_importance(xgb_clf)
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

# x = pd.concat([train, woe_df, extend_df], axis=1, join_axes=[train.index])
# x = x.drop(['d', 'arrival', 'label'], axis=1)
# x = x.fillna(0)
# y = train['label']
# sf = DataShuffle(y)
# train_x, test_x = sf.get_split_data(x)
# train_y, test_y = sf.get_split_data(y)

x = pd.concat([person_details_df, woe_df, extend_df], axis=1, join_axes=[person_details_df.index])
# x = person_details_df
x = x.drop(['d', 'arrival', 'label'], axis=1)
x = x.fillna(0)
y = person_details_df['label']

kf = KCrossFold(len(y))
train_x, test_x, train_y, test_y = kf.get_data(x, y)

parameters = {
    'n_estimators': [800],
    # 'learning_rate': [0.05],
    'max_depth': [4],
    # 'gamma': [0, 0.5],
    'subsample': [0.5, 0.7],
    # 'min_child_weight': [1, 0.5, 2],
    # 'scale_pos_weight': [0.3, 1, 3]
}
xgb_clf = xgb.XGBClassifier(silent=1)
gs_clf = GridSearchCV(xgb_clf, param_grid=parameters, scoring='roc_auc', verbose=2)
gs_clf.fit(x, y)
print gs_clf.best_score_
gs_clf.grid_scores_

pred_test = gs_clf.predict_proba(test_x)
pred_train = gs_clf.predict_proba(train_x)
print 'train auc: %s' % get_auc(train_y, pred_train[:, 1])
print 'test auc: %s' % get_auc(test_y, pred_test[:, 1])
print 'xiecheng score: %s' % get_xc_score(test_y, pred_test[:, 1])





