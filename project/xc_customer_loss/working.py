# -*- coding: utf-8 -*-
from cfg.config import xc_dir
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import random






def get_woe(binids, y, minlen=2, y_0=None, y_1=None):
    '''
    根据数据的分组和对应的目标变量计算每个分组的iv值和woe值
    :param binids: np.array/pd.series, 每个数据的分组id
    :param y: np.array/pd.series, 每个数据对应的y值
    :param minlen: int, 最小分组长度
    :param y_0: int,
    :param y_1: int,
    :return:
    '''
    y_len = len(y)
    y_0 = y_0 if y_0 else sum(y == 0)
    y_1 = y_1 if y_1 else y_len - y_0
    bin_true = np.bincount(binids, weights=y, minlength=minlen)
    bin_total = np.bincount(binids, minlength=minlen)
    bin_false = bin_total - bin_true
    # 分组频率统计
    total_dist = bin_total / float(y_len)
    true_dist = bin_true / y_1
    false_dist = bin_false / y_0

    true_rate = bin_true / bin_total
    # 计算woe和iv值
    woe = np.log(false_dist / true_dist)
    woe[np.where(np.isinf(woe))] = 0
    return woe, false_dist, true_dist, bin_true, bin_false, bin_total, true_rate, total_dist


class DataShuffle(object):
    """
    训练测试数据切分
    """
    def __init__(self, y, seed=random.randint(0, 9999)):
        self.__seed = seed  # 随机数生成种子
        self.__n = len(y)  # 数据集长度
        self.random_state = self._random_state()  # 随机数生成器
        self.train_idx = None  # 训练集的index
        self.test_idx = None  # 测试集的index
        self.shuffle_split()

    def _random_state(self):
        return np.random.RandomState(self.__seed)

    def shuffle_split(self, test_ratio=0.3):
        """
        随机生成训练测试集的index
        :param test_ratio: float, 测试集的比例
        :return:
        """
        random_idx = self.random_state.permutation(self.__n)
        self.train_idx = random_idx[self.__n * test_ratio:]
        self.test_idx = random_idx[:self.__n * test_ratio]

    def get_split_data(self, x):
        # self.shuffle_split(test_ratio)
        if len(x.shape) == 2:
            return x.ix[self.train_idx, :].reset_index(drop=True), x.ix[self.test_idx, :].reset_index(drop=True)
        elif len(x.shape) == 1:
            return x[self.train_idx].reset_index(drop=True), x[self.test_idx].reset_index(drop=True)



train = pd.read_csv(xc_dir+'userlostprob_train.txt', sep='\t')
train = train.drop('sampleid', axis=1)
print train.shape  # (689945, 51)
# test = pd.read_csv(xc_dir+'userlostprob_test.txt', sep='\t')
# test = test.drop('sampleid', axis=1)
# print test.shape  # (435075, 50)


##################################################################################################
# data exploration
##################################################################################################

obj_var = train.columns[train.dtypes == 'object'].tolist()
cat_var = train.columns[train.dtypes == 'int64'].tolist()
cat_var.remove('label')
con_var = train.columns[train.dtypes == 'float64'].tolist()
# train_y = train['label']

# for i in con_var:
    # print i, len(train[i].unique())

missing_ratio = train[con_var].isnull().sum()/float(len(train))


##################################################################################################
# bench mark
##################################################################################################
x = train.drop(['d', 'arrival'], axis=1)
x = x.fillna(0)
y = train['label']
sf = DataShuffle(y)
train_x, test_x = sf.get_split_data(x)
train_y, test_y = sf.get_split_data(y)


cf = RandomForestClassifier()
cf.fit(train_x, train_y)
cf.predict(test_x)






