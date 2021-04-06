#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Date   ：2021/4/6 10:24
@Author ：wfh1300
"""

import cupy as cp


def cp_rolling_window(a, window):
    #https: // blog.csdn.net / brucewong0516 / article / details / 84840469
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return cp.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def count_inf(x):
    return cp.count_nonzero(cp.isnan(x))

def count_nan(x):
    return cp.count_nonzero(cp.isnan(x))

def count_nan_inf(x):
    bool_arr = cp.isinf(x) | cp.isnan(x)
    return cp.count_nonzero(bool_arr)


def rank(x):
    # https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
    # 将x的空值,-inf,inf全都变为inf，排序时会被放在数组最末尾
    x = cp.where(cp.isinf(x)|cp.isnan(x), cp.inf, x)

    denominator = len(x) - count_inf(x)

    # 此方式，只一次排序，对于长数组更有优势
    temp = x.argsort()
    ranks = cp.empty_like(temp)
    ranks[temp] = cp.arange(len(x))
    ranks = ranks.astype(float)

    # 将nan和inf的位置恢复为nan,不参与rank
    ranks[cp.isinf(x)] = cp.nan

    return (ranks + 1) / denominator

def delay(x, period):
    # window是>
    if cp.abs(period) >= len(x):
        return cp.full(len(x), cp.nan)

    #period 和 window的区别，这里不再减1
    prefix = cp.full(cp.abs(period), cp.nan)
    # 只有float才能加cp.nan, 否则int就会变成-2147483648
    result = cp.roll(x, period).astype(float)
    if period >= 0:
        result[:period] = prefix
    else:
        result[period:] = prefix
    return result

def delta(x, period):
    if cp.abs(period) >= len(x):
        return cp.full(len(x), cp.nan)

    return x - delay(x, period)


def covariance(x, y):
    # 未使用cp.cov计算 协方差，速度比cp.cov快一倍
    # 1/(n-1)*Σ(xi-xbar)(yi-ybar)
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")

    if x.shape[0] <= 2:
        return cp.nan
    return cp.sum((x - x.mean()) * (y - y.mean())) / (x.shape[0] - 1)


def correlation(x, y):
    # 未使用cp.corrcoef, 以提高计算速度。速度比cp.corrcoef快一倍
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")

    # drop掉x和y中，存在nan和inf的位置，用剩余位置求解。
    # drop_bool = cp.isnan(x) | cp.isinf(x) | cp.isnan(y) | cp.isinf(y)
    # x = x[~drop_bool]
    # y = y[~drop_bool]
    # 若剩余的长度不足以计算相关系数，那返回空值
    if x.shape[0] <= 2:
        return cp.nan

    x_mean = x.mean()
    y_mean = y.mean()
    numerator = cp.sum((x - x_mean) * (y - y_mean))
    denominator = cp.sqrt(cp.sum((x - x_mean) ** 2) * cp.sum((y - y_mean) ** 2))
    # 增加除0的处理
    if cp.abs(denominator) < 0.000001:
        return cp.nan
    return numerator / denominator


def ts_correlation(x, y, window):
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")
        # window是>
    if window > len(x):
        return cp.full(len(x), cp.nan)

    pre_fix = cp.full(window - 1, cp.nan)
    x_array = cp_rolling_window(x, window)
    y_array = cp_rolling_window(y, window)

    # 这里没有使用reuslt = [], 考虑并行计算append可能会导致结果乱序
    result = cp.zeros(x_array.shape[0])
    for i in range(x_array.shape[0]):
        result[i] = correlation(x_array[i], y_array[i])
    return cp.concatenate((pre_fix, result))


def ts_covariance(x, y, window):
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")
    if window > len(x):
        return cp.full(len(x), cp.nan)

    pre_fix = cp.full(window - 1, cp.nan)
    x_rolling_array = cp_rolling_window(x, window)
    y_rolling_array = cp_rolling_window(y, window)

    result = cp.zeros(x_rolling_array.shape[0])
    for i in range(x_rolling_array.shape[0]):
        result[i] = covariance(x_rolling_array[i], y_rolling_array[i])
    return cp.concatenate((pre_fix, result))

def scale(x, a=1):
    bool_x = cp.isnan(x) | cp.isinf(x)

    # 变为原来的 a / sum 倍
    #
    denominator = cp.sum(cp.abs(x[~bool_x]))
    #如果和为0的处理
    if cp.abs(denominator) < 0.000001:
        return cp.full(len(x), cp.nan)
    x[bool_x] = cp.nan

    result = (a / denominator) * x
    return result

def decay_linear(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)
    prefix = cp.full(window - 1, cp.nan)

    x = cp.where(cp.isinf(x), cp.nan, x)

    x_rolling_array = cp_rolling_window(x, window)
    # 等差数列求和公式
    denominator = (window + 1) * window / 2
    # window各元素权重
    weight = (cp.arange(window) + 1) * 1.0 / denominator
    result = cp.zeros(x_rolling_array.shape[0])
    for i in range(x_rolling_array.shape[0]):
        result[i] = x_rolling_array[i].dot(weight)
    return cp.concatenate((prefix, result))


def signedpower(x, a):
    #将inf和-inf处理为nan
    #x[cp.isinf(x)] = cp.nan
    # 经测试where替换更快一些

    x = cp.where(cp.isinf(x), cp.nan, x)
    return cp.sign(x) * cp.abs(x)**a


def ts_sum(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)

    # 增加了一步，对含有nan值的放缩处理，比如[1, cp.nan, 3]的和本来为4， 这里乘以3/2变为6，
    # scale_nan_inf = window / ts_count_not_nan_inf(x, window)
    # 不直接修改x，通过copy的方式完成，目的是保留x原来的取值，否则参数x也会直接更改。
    # 空值，inf填充为0

    x = cp.where(cp.isinf(x), cp.nan, x)
    pre_fix = cp.full(window - 1, cp.nan)

    rolling_array = cp_rolling_window(x, window)
    result = cp.sum(rolling_array, axis=1)
    # 对求和结果进行放缩
    # result = result * scale_nan_inf
    return cp.concatenate((pre_fix, result))


def ts_sma(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)
    pre_fix = cp.full(window - 1, cp.nan)

    x = cp.where(cp.isinf(x), cp.nan, x)
    rolling_array = cp_rolling_window(x, window)

    result = cp.mean(rolling_array, axis=1)
    return cp.concatenate((pre_fix, result))


def ts_stddev(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)
    cov = ts_covariance(x, x, window)
    return cp.sqrt(cov)


def ts_rank(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)

    # 只排名了，没有除总数形成分位
    def _rank(x):

        x = cp.where(cp.isinf(x)|cp.isnan(x), cp.inf, x)

        # 此方式，只一次排序，对于长数组更有优势
        temp = x.argsort()
        ranks = cp.empty_like(temp)
        ranks[temp] = cp.arange(len(x))
        ranks = ranks.astype(float)

        # 将nan和inf的位置恢复为nan,不参与rank

        ranks = cp.where(cp.isinf(x), cp.nan, ranks)
        #ranks[cp.isinf(x)] = cp.nan
        # 返回x的最后一个元素的排名
        return (ranks + 1)[-1]

    x_rolling_array = cp_rolling_window(x, window)
    pre_fix = cp.full(window - 1, cp.nan)

    result = cp.zeros(x_rolling_array.shape[0])
    for i in range(x_rolling_array.shape[0]):
        result[i] = _rank(x_rolling_array[i])

    return cp.concatenate((pre_fix, result))


def ts_product(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)
    # # 增加对溢出的处理
    # max_element = cp.max(cp.abs(x))
    # if max_element**window == cp.inf:
    #     return cp.nan

    # 对于有空值的地方，开方再取幂
    # exp_nan_inf = window / ts_count_not_nan_inf(x, window)

    # 空值，填充为1
    x = cp.where(cp.isinf(x), cp.nan, x)

    pre_fix = cp.full(window - 1, cp.nan)

    x_rolling_array = cp_rolling_window(x, window)
    result = cp.prod(x_rolling_array, axis=1)

    # 指数放缩, 有一个奇怪的现象就是1^nan = 1
    # result = cp.sign(result) * (cp.abs(result)**exp_nan_inf)
    # 当window内均为nan时，处理为nan
    # result[cp.isnan(exp_nan_inf)] = cp.nan
    return cp.concatenate((pre_fix, result))


def ts_min(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)
    # 把空值填充为 +inf
    x = cp.where(cp.isinf(x)|cp.isnan(x), cp.inf, x)

    prefix = cp.full(window - 1, cp.nan)
    x_rolling_array = cp_rolling_window(x, window)
    result = cp.min(x_rolling_array, axis=1)
    # 结果中如果存在inf,说明此window值均为空，我们返回nan
    result = result.astype(float)

    result = cp.where(cp.isinf(result), cp.nan, result)
    return cp.concatenate((prefix, result))


def ts_max(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)
    # 把空值填充为-inf
    x = cp.where(cp.isinf(x)|cp.isnan(x), -cp.inf, x)

    prefix = cp.full(window - 1, cp.nan)
    x_rolling_array = cp_rolling_window(x, window)
    result = cp.max(x_rolling_array, axis=1)
    # 结果中如果存在-inf,说明此window均为空，返回nan
    result = result.astype(float)
    result = cp.where(cp.isinf(result), cp.nan, result)
    return cp.concatenate((prefix, result))


def ts_argmax(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)
    # 把空值和正无穷填充为-inf
    x = cp.where(cp.isinf(x)|cp.isnan(x), -cp.inf, x)

    prefix = cp.full(window - 1, cp.nan)
    x_rolling_array = cp_rolling_window(x, window)
    result = cp.argmax(x_rolling_array, axis=1)

    # 找到window中全为-inf的情况，填充为nan
    result = result.astype(float)

    cp.where(cp.isinf(x_rolling_array).all(axis=1), cp.nan, result)

    result += 1
    return cp.concatenate((prefix, result))


def ts_argmin(x, window):
    if window > len(x):
        return cp.full(len(x), cp.nan)
    # 将nan及-inf填充为inf
    x = cp.where(cp.isinf(x) | cp.isnan(x), cp.inf, x)

    prefix = cp.full(window - 1, cp.nan)
    x_rolling_array = cp_rolling_window(x, window)
    result = cp.argmin(x_rolling_array, axis=1)

    # 找到window中全为-inf的情况，填充为nan
    result = result.astype(float)
    result = cp.where(cp.isinf(x_rolling_array).all(axis=1), cp.nan, result)
    result += 1
    return cp.concatenate((prefix, result))


def indneutralize(y, X):
    # 采用最小二乘法拟合
    # 保证二者同维度
    if X.shape[0] != y.shape[0]:
        X = X.T
    # 如果仍然不相等，那就报错
    if X.shape[0] != y.shape[0]:
        raise ValueError("X和y不同维度")

    result = cp.zeros(len(y))

    # index = cp.arange(len(y))
    bool_y = cp.isnan(y) | cp.isinf(y)
    # X中的每一行矩阵加和!=1 的地方
    bool_X = ~(cp.abs(cp.sum(X, axis=1) - 1) < 0.000001)
    bool_na = bool_y | bool_X

    # used_index = index[~bool_na]
    used_y = y[~bool_na]
    used_X = X[~bool_na]

    b = cp.linalg.inv(used_X.T.dot(used_X)).dot(used_X.T).dot(used_y)

    pred_used_y = used_X.dot(b)

    result[~bool_na] = used_y - pred_used_y
    result[bool_na] = cp.nan

    return result


#以下函数取自gplearn
def protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with cp.errstate(divide='ignore', invalid='ignore'):
        return cp.where(cp.abs(x2) > 0.001, cp.divide(x1, x2), 1.)


def protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return cp.sqrt(cp.abs(x1))

def protected_log(x1):
    """Closure of log for zero arguments."""
    with cp.errstate(divide='ignore', invalid='ignore'):
        return cp.where(cp.abs(x1) > 0.001, cp.log(cp.abs(x1)), 0.)

def protected_inverse(x1):
    """Closure of log for zero arguments."""
    with cp.errstate(divide='ignore', invalid='ignore'):
        return cp.where(cp.abs(x1) > 0.001, 1. / x1, 0.)

def sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with cp.errstate(over='ignore', under='ignore'):
        return 1 / (1 + cp.exp(-x1))









