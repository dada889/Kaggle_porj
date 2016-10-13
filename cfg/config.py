# -*- coding: utf-8 -*-
import os

root_dir = os.getcwd()
data_dir = root_dir + os.sep + 'data'


# xc_dir = data_dir + os.sep + 'xiecheg_customers_loss' + os.sep

def data_path(proj_name):
    data_path = data_dir + os.sep + proj_name + os.sep
    return  data_path