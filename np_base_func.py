import numpy as np
from numpy.core.numeric import roll

#新版numpy 自带了rolling window
def np_rolling_window(x, window_shape, axis=None):
    #https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    return np.lib.stride_tricks.sliding_window_view(x, window_shape, axis)
def count_nan_inf(x):
    #统计inf 或者 nan的bool
    bool_arr = np.isinf(x) | np.isnan(x)
    return np.count_nonzero(bool_arr)

def count_nan(x):
    return np.count_nonzero(np.isnan(x))
def count_inf(x):
    return np.count_nonzero(np.isinf(x))
#查找window内非(空值/inf)的数量
def ts_count_not_nan_inf(x, window):
    bool_x = np.isinf(x) | np.isnan(x)
    
    bool_x_rolling_array = np_rolling_window(bool_x, window)
    result = np.zeros(bool_x_rolling_array.shape[0])
    for i in range(bool_x_rolling_array.shape[0]):
        result[i] = np.count_nonzero(bool_x_rolling_array[i])
    result = window - result
    # 将0替换为np.nan, 0意味着当前窗口内的值全部都是inf或者nan
    return np.where(np.abs(result) < 0.000001, np.nan, result)

#测试时和rankdata版本有些不一样的地方。
def rank(x):
    #https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
    #将x的空值,-inf,inf全都变为inf，排序时会被放在数组最末尾
    x = np.nan_to_num(x, nan=np.inf, posinf=np.inf, neginf=np.inf)
    denominator = len(x) - count_inf(x)
  
    #此方式，只一次排序，对于长数组更有优势
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    ranks = ranks.astype(float)
    
    #将nan和inf的位置恢复为nan,不参与rank
    ranks[np.isinf(x)] = np.nan

    return (ranks + 1) / denominator

def delay(x, period):
    # window是>
    if np.abs(period) >= len(x):
        return np.full(len(x), np.nan)

    #period 和 window的区别，这里不再减1
    prefix = np.full(np.abs(period), np.nan)
    # 只有float才能加np.nan, 否则int就会变成-2147483648
    result = np.roll(x, period).astype(float)
    if period >= 0:
        result[:period] = prefix
    else:
        result[period:] = prefix
    return result
def delta(x, period):
    if np.abs(period) >= len(x):
        return np.full(len(x), np.nan)

    return x - delay(x, period)

def covariance(x, y):
    # 未使用np.cov计算 协方差，速度比np.cov快一倍
    # 1/(n-1)*Σ(xi-xbar)(yi-ybar)
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")
    
    #drop掉x和y中，存在nan和inf的位置，用剩余位置求解。
    # drop_bool = np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y)
    # x = x[~drop_bool]
    # y = y[~drop_bool]
    #若剩余的长度不足以计算协方差，那返回空值
    if x.shape[0] <= 2:
        return np.nan
    print(x, y)
    return np.sum((x - x.mean()) * (y - y.mean())) / (x.shape[0] - 1)
def correlation(x, y):
    # 未使用np.corrcoef, 以提高计算速度。速度比np.corrcoef快一倍
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")
    
    #drop掉x和y中，存在nan和inf的位置，用剩余位置求解。
    # drop_bool = np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y)
    # x = x[~drop_bool]
    # y = y[~drop_bool]
    #若剩余的长度不足以计算相关系数，那返回空值
    if x.shape[0] <= 2:
        return np.nan
    
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    # 增加除0的处理
    if np.abs(denominator) < 0.000001:
        return np.nan
    return numerator / denominator



def ts_correlation(x, y, window):
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")    
    # window是>
    if window > len(x):
        return np.full(len(x), np.nan)


    pre_fix = np.full(window - 1, np.nan)
    x_array = np_rolling_window(x, window)
    y_array = np_rolling_window(y, window)
    
    # 这里没有使用reuslt = [], 考虑并行计算append可能会导致结果乱序
    result = np.zeros(x_array.shape[0])
    for i in range(x_array.shape[0]):
        result[i] = correlation(x_array[i], y_array[i])
    return np.append(pre_fix, result)
def ts_covariance(x, y, window):
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")
    if window > len(x):
        return np.full(len(x), np.nan)
    
    pre_fix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    y_rolling_array = np_rolling_window(y, window)
    
    result = np.zeros(x_rolling_array.shape[0])
    for i in range(x_rolling_array.shape[0]):
        result[i] = covariance(x_rolling_array[i], y_rolling_array[i])
    return np.append(pre_fix, result)

#测试过程中遇到了问题。
def scale(x, a=1):
    bool_x = np.isnan(x) | np.isinf(x)

    # 变为原来的 a / sum 倍
    # 
    denominator = np.sum(np.abs(x[~bool_x]))
    #如果和为0的处理
    if abs(denominator) < 0.000001:
        return np.full(len(x), np.nan)
    x[bool_x] = np.nan

    result = (a / denominator) * x
    return result


def decay_linear(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    prefix = np.full(window - 1, np.nan)

    x = np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan)
    x_rolling_array = np_rolling_window(x, window)
    # 等差数列求和公式
    denominator = (window + 1) * window / 2
    # window各元素权重
    weight = (np.arange(window) + 1) * 1.0 / denominator
    result = np.zeros(x_rolling_array.shape[0])
    for i in range(x_rolling_array.shape[0]):
        result[i] = x_rolling_array[i].dot(weight)
    return np.append(prefix, result)



# def decay_linear(x, window):
#     if window > len(x):
#         return np.full(len(x), np.nan)

#     prefix = np.full(window - 1, np.nan)
#     x_rolling_array = np_rolling_window(x, window)
    
#     # 等差数列求和公式
#     #denominator = (window + 1) * window / 2
#     # window各元素权重
#     weight = (np.arange(window) + 1) * 1.0
    
#     #内包含空值处理
#     def _weight_sum(window_x, weight):
#         bool_x = np.isnan(window_x) | np.isinf(window_x)
#         #这个window全是空值或inf的时候
#         if bool_x.all():
#             return np.nan
        
#         window_x = window_x[~bool_x]
#         weight = weight[~bool_x]
#         return window_x.dot(weight) / np.sum(weight)
    
#     result = np.zeros(x_rolling_array.shape[0])
#     for i in range(x_rolling_array.shape[0]):
#         result[i] = _weight_sum(x_rolling_array[i], weight)
#     return np.append(prefix, result)

def signedpower(x, a):
    #将inf和-inf处理为nan
    x[np.isinf(x)] = np.nan

    return np.sign(x) * np.abs(x)**a



#numpy 不支持[[1,2],[3,4,5]]这样不等长度的array，所以放弃了dropna的想法。
def ts_sum(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)

    #增加了一步，对含有nan值的放缩处理，比如[1, np.nan, 3]的和本来为4， 这里乘以3/2变为6，
    #scale_nan_inf = window / ts_count_not_nan_inf(x, window)
    #不直接修改x，通过copy的方式完成，目的是保留x原来的取值，否则参数x也会直接更改。
    # 空值，inf填充为0
    x = np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pre_fix = np.full(window - 1, np.nan)
    
    rolling_array = np_rolling_window(x, window)
    result = np.sum(rolling_array, axis=1)
    #对求和结果进行放缩
    #result = result * scale_nan_inf
    
    return np.append(pre_fix, result)

def ts_sma(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    pre_fix = np.full(window - 1, np.nan)
    
    #先查找非空值个数
    #denominator = ts_count_not_nan_inf(x, window)
    #再填充空值为0
    x = np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan)
    rolling_array = np_rolling_window(x, window)
    
    #result = np.mean(rolling_array, axis=1)
    #numerator = np.sum(rolling_array, axis=1)

    #denominator = 0时，已经处理为np.nan，这里无需处理了。
    #result = numerator / denominator
    result = np.mean(rolling_array, axis=1)
    
    return np.append(pre_fix, result)

def ts_stddev(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    cov = ts_covariance(x, x, window)
    return np.sqrt(cov)




def ts_rank(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    # 只排名了，没有除总数形成分位
    def _rank(x):

        x = np.nan_to_num(x, nan=np.inf, posinf=np.inf, neginf=np.inf)

        #此方式，只一次排序，对于长数组更有优势
        temp = x.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(x))
        ranks = ranks.astype(float)

        #将nan和inf的位置恢复为nan,不参与rank
        ranks[np.isinf(x)] = np.nan
        #返回x的最后一个元素的排名
        return (ranks + 1)[-1]
    
    x_rolling_array = np_rolling_window(x, window)
    pre_fix = np.full(window - 1, np.nan)
    
    result = np.zeros(x_rolling_array.shape[0])
    for i in range(x_rolling_array.shape[0]):
        result[i] = _rank(x_rolling_array[i])
    return np.append(pre_fix, result)

def ts_product(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    # # 增加对溢出的处理
    # max_element = np.max(np.abs(x))
    # if max_element**window == np.inf:
    #     return np.nan


    #对于有空值的地方，开方再取幂
    #exp_nan_inf = window / ts_count_not_nan_inf(x, window)
    
    #空值，填充为1
    x = np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan)
    pre_fix = np.full(window - 1, np.nan)
    
    x_rolling_array = np_rolling_window(x, window)
    result = np.prod(x_rolling_array, axis=1)

    #指数放缩, 有一个奇怪的现象就是1^nan = 1
    #result = np.sign(result) * (np.abs(result)**exp_nan_inf)
    # 当window内均为nan时，处理为nan
    #result[np.isnan(exp_nan_inf)] = np.nan
    
    return np.append(pre_fix, result)

def ts_min(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    #把空值填充为 +inf
    x = np.nan_to_num(x, nan=np.inf, posinf=np.inf, neginf=np.inf)
    
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.min(x_rolling_array, axis=1)
    #结果中如果存在inf,说明此window值均为空，我们返回nan
    result = result.astype(float)
    result[np.isinf(result)] = np.nan

    return np.append(prefix, result)
def ts_max(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    #把空值填充为-inf
    x = np.nan_to_num(x, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.max(x_rolling_array, axis=1)
    #结果中如果存在-inf,说明此window均为空，返回nan
    result = result.astype(float)
    result[np.isinf(result)] = np.nan
    
    return np.append(prefix, result)



def ts_argmax(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    #把空值和正无穷填充为-inf
    x = np.nan_to_num(x, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.argmax(x_rolling_array, axis=1)
    
    #找到window中全为-inf的情况，填充为nan
    result = result.astype(float)
    result[np.isinf(x_rolling_array).all(axis=1)] = np.nan
    result += 1

    return np.append(prefix, result)

def ts_argmin(x, window):
    if window > len(x):
        return np.full(len(x), np.nan)
    #将nan及-inf填充为inf
    x = np.nan_to_num(x, nan=np.inf, posinf=np.inf, neginf=np.inf)
    
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.argmin(x_rolling_array, axis=1)
    
    #找到window中全为-inf的情况，填充为nan
    result = result.astype(float)
    result[np.isinf(x_rolling_array).all(axis=1)] = np.nan
    result += 1

    return np.append(prefix, result)


def indneutralize(y, X):
    # 采用最小二乘法拟合
    # 保证二者同维度
    if X.shape[0] != y.shape[0]:
        X = X.T
    #如果仍然不相等，那就报错
    if X.shape[0] != y.shape[0]:
        raise ValueError("X和y不同维度")
    
    result = np.zeros(len(y))

    #index = np.arange(len(y))
    bool_y = np.isnan(y) | np.isinf(y)
    #X中的每一行矩阵加和!=1 的地方
    bool_X = ~(np.abs(np.sum(X, axis=1) - 1) < 0.000001)
    bool_na = bool_y | bool_X

    #used_index = index[~bool_na]
    used_y = y[~bool_na]
    used_X = X[~bool_na]


    b = np.linalg.inv(used_X.T.dot(used_X)).dot(used_X.T).dot(used_y)

    pred_used_y = used_X.dot(b)

    result[~bool_na] = used_y - pred_used_y
    result[bool_na] = np.nan

    return result


#以下函数取自gplearn
def protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

def protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))

def protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def protected_inverse(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)

def sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))



if __name__ == "__main__":
    x = np.array([1,-2,np.nan, np.inf, -np.inf])
    print(scale(x, 0))
    print(np_rolling_window(x, 0))
