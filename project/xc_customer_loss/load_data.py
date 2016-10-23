# -*- coding: utf-8 -*-
from cfg.config import data_path
import pandas as pd
import bayesopt

train = pd.read_csv(data_path('xiecheg_customers_loss')+'userlostprob_train.txt', sep='\t')
# train = train.drop('sampleid', axis=1)
# print train.shape  # (689945, 51)
test = pd.read_csv(data_path('xiecheg_customers_loss')+'userlostprob_test.txt', sep='\t')
# test = test.drop('sampleid', axis=1)
# print test.shape  # (435075, 50)


