# -*- coding: utf-8 -*-
from cfg.config import xc_dir
import pandas as pd



train = pd.read_csv(xc_dir+'userlostprob_train.txt', sep='\t')
train = train.drop('sampleid', axis=1)
# print train.shape  # (689945, 51)
test = pd.read_csv(xc_dir+'userlostprob_test.txt', sep='\t')
test = test.drop('sampleid', axis=1)
# print test.shape  # (435075, 50)


