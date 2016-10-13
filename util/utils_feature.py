# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def dict_reduce(dict_bin):
    """
    离散变量分箱转换函数
    :param dict_bin: dict, 离散变量分箱参数
    :return:
    """
    result = {}
    for key in dict_bin.keys():
        for value in dict_bin[key]:
            result[value] = key
    return result

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



def bin_iv_woe(x, y, n_bins=7, decimal=4, manual_bin=None, encoder=LabelEncoder(),
               nan_min=0.05, merge_nan=True):
    '''
    快速分箱计算iv,woe值并返回替换woe值后的数据,兼容极端分布,category数据,bool数据

    对于极端分布的连续变量,将频次最高的变量单独分一组,仍需增加兼容性

    离散变量:bool,将数据转换成数值,每个值分为一箱
             object(str), 暂时不支持自动分箱

    连续变量:
        nan数据将自动分为一组, 但是如果nan数据占比小于nan_min, nan数据将自动并入woe值最接近的一组,从新计算woe值

    使用searchsorted(binary search)进行分箱, 对于部分变量bisect可能速度会更快,待测试

    :param x: pd.Series, 需要分箱的数据
    :param y: pd.Series, 目标变量
    :param n_bins: int, 自动分箱的组数, 设为0的时候每个数一组
    :param decimal: int, 保留小数位数
    :param woe_data: bool, 是否返回替换woe值后的数据
    :param manual_bin: list/dict, 手动分箱的切割点
            list: 处理连续变量, 第一个分箱点应该是数据的最小值,最后一个应该是最大值
            # dict: 处理离散变量, 不在分组中的值和nan用'未知'替换
    :param encoder: LabelEncoder object, 用于将str转成int并转会str
    :param nan_min: float, nan单独分组的最小百分比
    :param merge_nan: bool, True时自动合并数量太小的nan分组,False时不对nan分组进行合并
    :return: woe_data false返回iv统计表格, ture返回woe替换后数据
    '''
    # ========================判断数据类型=========================================
    discrete = False
    if x.dtype == object:
        temp = x.copy()
        x = temp.fillna(u'未知')
        # 为离散变量手动分箱
        bining = None
        if isinstance(manual_bin, dict):  # 如果手动分箱点是字典, 对离散变量执行分箱
            bining = dict_reduce(manual_bin)
            x = x.apply(lambda xx: bining.get(xx, u'未知'))  # 根据输入的字典将离散变量映射到对应的类别上进行分箱
        temp = encoder.fit_transform(x)
        x = pd.Series(temp)
        discrete = True
    elif x.dtype == bool:
        x = x.astype(int).copy()
        discrete = True

    y_1 = sum(y == 1)
    y_0 = sum(y == 0)
    len_y = y_1 + y_0
    has_nan = any(x.isnull())
    # ========================计算分箱点=========================================
    # 计算分箱点,自动按分位数分箱/手动分箱
    if manual_bin:
        bins = manual_bin
    elif discrete:
        pass
    else:
        if has_nan:  # 如果有nan数据,排除nan进行切分
            x_part = x.ix[~x.isnull()]
            bin_v = np.linspace(0, 100, n_bins).tolist()
            bins = [np.percentile(x_part, i) for i in bin_v]
            bins = sorted(list(set(bins)))
            bins[-1] += 0.0001
        else:
            bin_v = np.linspace(0, 100, n_bins + 1).tolist()
            bins = [np.percentile(x, i) for i in bin_v]
            bins = sorted(list(set(bins)))
            bins[-1] += 0.01
        if len(bins) == 2:  # 对极端分布的数据(只分成一组)进一步切分
            # uni, uni_count = np.unique(x, return_counts=True)
            x_part = x.ix[~x.isnull()]
            uni, uni_count = np.unique(np.array(x_part), return_counts=True)
            max_count_id = np.where(uni_count > 0.5 * len_y)[0][0]
            bins = []
            if any(uni_count > 0.95 * len(y)):
                print x.name
                raise ValueError('超过百分95数据为同一值,丢弃')
            elif max_count_id == 0:
                bins = [uni[0], uni[1], uni[-1] + 1]
            elif max_count_id == len(uni):
                bins = [uni[0], uni[-1], uni[-1] + 1]
            else:
                bins += [uni_count[0]]
                if sum(uni_count[0:max_count_id]) > 0.05 * len_y:
                    bins += [uni_count[max_count_id]]
                if sum(uni_count[max_count_id: len(uni_count)]) > 0.05 * len_y:
                    bins += [uni_count[max_count_id + 1]]
                bins += [uni[-1] + 1]
    # ========================分箱并计算woe=========================================
    # 每个值分一箱(针对str和bool类数据)
    if discrete:
        if len(x.unique()) == 1:
            raise ValueError('数据只有一种值,丢弃')
        binids = x
        try:
            label = encoder.inverse_transform(np.unique(x).tolist())
            if bining:
                if u'未知' in bining.keys():
                    nan_label = label.tolist().index(bining.get(u'未知', ''))
                    # print nan_label, label[nan_label]
                    label[nan_label] = label[nan_label] + u'+未知'
        except:
            label = np.unique(x).tolist()
        # 分组频次统计
        binid_ix, binid_count = np.unique(binids, return_counts=True)
        woe, false_dist, true_dist, bin_true, bin_false, bin_total, true_rate, total_dist \
            = get_woe(binids, y, y_0=y_0, y_1=y_1, minlen=len(binid_ix))
    # 根据bins进行分箱,如果nan太少,自动对nan分组进行合并
    else:
        bins_name = [round(bin_point, 4) for bin_point in bins]
        temp_bins = bins[:]
        if temp_bins[0] > x.min():
            temp_bins[0] = x.min()
            print '将第一个分箱点替换成数据最小值', x.name
        if temp_bins[-1] < x.max():
            temp_bins[-1] = x.max() + 0.00001
            print '将最后一个分箱点替换成数据最大值', x.name
        binids = np.digitize(x, temp_bins) - 1
        # binids = np.digitize(np.array(x), bins) - 1

        # 设置分组名称
        label = ['[%s-%s)' % (str(bins_name[i]), str(bins_name[i + 1])) for i in xrange(len(bins) - 1)]
        if has_nan:
            label += ['nan']
        # 分组频次统计
        binid_ix, binid_count = np.unique(binids, return_counts=True)
        woe, false_dist, true_dist, bin_true, bin_false, bin_total, true_rate, total_dist \
            = get_woe(binids, y, y_0=y_0, y_1=y_1, minlen=len(binid_ix))
        # ============================对nan进行合并并重新计算woe====================================
        if merge_nan:
            if has_nan & (binid_count[-1] <= len_y * nan_min):
                woe_sorted_ix = np.argsort(woe)
                # woe[woe_sorted_ix]
                nan_woe = woe[-1]
                nan_order_ix = np.where(woe_sorted_ix == binid_ix[-1])[0][0]
                if nan_order_ix == 0:
                    close_id = woe_sorted_ix[nan_order_ix + 1]
                    # close_id = np.where(woe_sorted_ix == nan_order_ix+1)[0][0]
                    mv_posi = 0
                elif nan_order_ix == binid_ix[-1]:
                    close_id = woe_sorted_ix[nan_order_ix - 1]
                    # close_id = np.where(woe_sorted_ix == nan_order_ix-1)[0][0]
                    mv_posi = 0
                else:
                    nan_woe_up = woe[woe_sorted_ix[nan_order_ix + 1]]
                    nan_woe_down = woe[woe_sorted_ix[nan_order_ix - 1]]
                    close_woe = [nan_woe_up, nan_woe_down][
                        np.argmin([abs(nan_woe - nan_woe_up), abs(nan_woe - nan_woe_down)])]
                    close_id = np.where(woe == close_woe)[0][0]
                    mv_posi = -1 if close_id == nan_order_ix + 1 else 0
                # 将nan组并入woe值最接近的组
                binids[binids == binid_ix[-1]] = close_id
                binid_ix, binid_count = np.unique(binids, return_counts=True)
                # 重新计算woe
                woe, false_dist, true_dist, bin_true, bin_false, bin_total, true_rate, total_dist \
                    = get_woe(binids, y, y_0=y_0, y_1=y_1, minlen=len(binid_ix) - 1)
                # 更新label
                label = ['[%s-%s)' % (str(bins_name[i]), str(bins_name[i + 1])) for i in xrange(len(bins) - 1)]
                label[close_id + mv_posi] += '+nan'
    # ===============================生成表格并返回结果=====================================
    iv = (false_dist - true_dist) * woe
    # 统计表格
    df_data = [bin_total, total_dist, bin_true, true_dist, bin_false, false_dist,
               true_rate, woe, iv]
    iv_table = pd.DataFrame(df_data).transpose().round(decimal)
    iv_table.columns = ['count', 'total_dist', 'good_count', 'good_dist', 'bad_count', 'bad_dist',
                        'bad_rate', 'woe', 'iv']
    # print label
    # print iv_table
    iv_table.index = label
    # print iv_table
    try:
        bins = np.round(bins, 7).tolist()
    except:
        bins = manual_bin
    return [iv_table, bins], woe[binids]


def cate_feature_mean(x, y):
    """
    计算离散变量每个类别的个数和对应的目标变量的平均值
    :param x: pd.Series, 离散特征
    :param y: pd.Series, 连续目标变量
    :return:
    """
    assert (isinstance(x, pd.Series)) & isinstance(y, pd.Series), 'x and y are not both pandas series'
    assert (x.dtype == 'object') & (y.dtype == 'float64'), 'x is not discrete or y is not continuous'
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    num_x = lb.fit_transform(x)
    cate_count = np.bincount(num_x)
    cate_sum = np.bincount(num_x, weights=y)
    cate_mean = cate_sum / cate_count
    return pd.DataFrame(np.column_stack([cate_mean, cate_count]), index=lb.classes_, columns=['mean', 'count'])

