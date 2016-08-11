# -*- coding: utf-8 -*-
from cfg.config import xc_dir
import pandas as pd



train = pd.read_csv(xc_dir+'userlostprob_train.txt', sep='\t')
print train.shape  # (689945, 51)
test = pd.read_csv(xc_dir+'userlostprob_test.txt', sep='\t')
print test.shape  # (435075, 50)


obj_var = train.columns[train.dtypes == 'object'].tolist()
cat_var = train.columns[train.dtypes == 'int64'].tolist()
con_var = train.columns[train.dtypes == 'float64'].tolist()

