# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV


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



def get_auc(true_y, pred_y):
    fpr, tpr, _ = roc_curve(true_y, pred_y)
    return auc(fpr, tpr)


def auc_scoring(estimator, x, y):
    pred_y = estimator(x)[:, 1]
    return get_auc(y, pred_y)


def get_xc_score(true_y, pred_y):
    precision, recall, _ = precision_recall_curve(true_y, pred_y)
    gh_th = precision >= 0.97
    return max(recall[gh_th])

