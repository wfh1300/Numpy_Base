import numpy as np

#新版numpy 自带了rolling window
def np_rolling_window(x, window_shape, axis=None):
    #https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    return np.lib.stride_tricks.sliding_window_view(x, window_shape, axis)


def rank(x):
    return (np.argsort(x) + 1) / x.shape[0]

def delay(x, period):
    #period 和 window的区别，这里不再减1
    prefix = np.full(period, np.nan)
    # 只有float才能加np.nan, 否则int就会变成-2147483648
    result = np.roll(x, period).astype(float)
    result[:period] = prefix
    return result
def delta(x, period):
    return x - delay(x, period)

def covariance(x, y):
    # 未使用np.cov计算 协方差，速度比np.cov快一倍
    # 1/(n-1)*Σ(xi-xbar)(yi-ybar)
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")
    
    return np.sum((x - x.mean()) * (y - y.mean())) / (x.shape[0] - 1)
def correlation(x, y):
    # 未使用np.corrcoef, 以提高计算速度。速度比np.corrcoef快一倍
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator
def ts_correlation(x, y, window):
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")    

    prefix = np.full(window - 1, np.nan)
    x_array = np_rolling_window(x, window)
    y_array = np_rolling_window(y, window)
    
    # 这里没有使用reuslt = [], 考虑并行计算append可能会导致结果乱序
    result = np.zeros(x_array.shape[0])
    for i in range(x_array.shape[0]):
        result[i] = correlation(x_array[i], y_array[i])
    return np.append(prefix, result)
def ts_covariance(x, y, window):
    if x.shape[0] != y.shape[0]:
        raise ValueError("The length of x and y must be the same")
    
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    y_rolling_array = np_rolling_window(y, window)
    
    result = np.zeros(x_rolling_array.shape[0])
    for i in range(x_rolling_array.shape[0]):
        result[i] = covariance(x_rolling_array[i], y_rolling_array[i])
    return np.append(prefix, result)

def scale(x, a=1):
    # 变为原来的 a / sum 倍
    return a / np.sum(np.abs(x)) * x

def decay_linear(x, window):
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    
    # 等差数列求和公式
    denominator = (window + 1) * window / 2
    # window各元素权重
    weight = (np.arange(window) + 1) * 1.0 / denominator
    
    result = np.zeros(x_rolling_array.shape[0])
    for i in range(x_rolling_array.shape[0]):
        result[i] = x_rolling_array[i].dot(weight.T)
    return np.append(prefix, result)

def signedpower(x, a):
    return np.sign(x) * np.abs(x)**a



def ts_sum(x, window):
    # 只处理(10, ) 类型的一维数组，不包括(10,1)类型， 后续若开放限制，可更改
    assert len(x.shape) == 1
    
    prefix = np.full(window - 1, np.nan)
    
    rolling_array = np_rolling_window(x, window)
    result = np.sum(rolling_array, axis=1)
    return np.append(prefix, result)

def ts_sma(x, window):
    prefix = np.full(window - 1, np.nan)
    rolling_array = np_rolling_window(x, window)
    result = np.mean(rolling_array, axis=1)
    return np.append(prefix, result)

def ts_stddev(x, window):
    prefix = np.full(window - 1, np.nan)
    #numpy 的样本方差是 ./n 不是 ./n-1
    rolling_array = np_rolling_window(x, window)
    result = np.std(rolling_array, axis=1)
    return np.append(prefix, result)




def ts_rank(x, window):
    # 出于完全用numpy实现考虑，此处用np.argsort实现了
    # 此函数可以用 from scipy.stats import rankdata 替换
    # 区别：rankdata函数 [1, 3, 2, 1] 会排序为array([1.5, 4. , 3. , 1.5])
    #       np.argsort 会排序为array([0, 3, 2, 1]
    x_rolling_array = np_rolling_window(x, window)
    prefix = np.full(window - 1, np.nan)
    # +1 的作用是将排序从1开始，而不是从0开始。
    #axis=-1，取最后一个axis。我们不知道每一个子排序内部的长度，所以为了取结果的最后一列，我们转置之后取最后一个元素。
    result = (np.argsort(x_rolling_array, axis=-1).T[-1] + 1) / window
    return np.append(prefix, result)

def ts_product(x, window):
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.prod(x_rolling_array, axis=1)
    return np.append(prefix, result)

def ts_min(x, window):
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.min(x_rolling_array, axis=1)
    return np.append(prefix, result)
def ts_max(x, window):
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.max(x_rolling_array, axis=1)
    return np.append(prefix, result)



def ts_argmax(x, window):
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.argmax(x_rolling_array, axis=1)
    return np.append(prefix, result)

def ts_argmin(x, window):
    prefix = np.full(window - 1, np.nan)
    x_rolling_array = np_rolling_window(x, window)
    result = np.argmin(x_rolling_array, axis=1)
    return np.append(prefix, result)


def indneutralize(y, X):
    # 保证二者同维度
    if X.shape[0] != y.shape[0]:
        X = X.T
    
    # 采用最小二乘法拟合
    # b = (X'X)^-1X'y
    b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    pred_y = X.dot(b)
    return y - pred_y


if __name__ == "__main__":
    x = np.arange(10)
    y = np.array([1, 3, 5, 9, -2, 4, 0, 1, 2, 3])
    print("rank", rank(x))
    print("delay", delay(x, 1))
    print("scale", scale(x, 3))
    print("delta", delta(x, 1))
    print("signedpower", signedpower(y, 2))

# 以上适用于(n, 1)
# 以下仅适用于(n, )形式
    print("corr", ts_correlation(x, y, 3))
    print("cov", ts_covariance(x, y, 3))
    print("decay_linear", decay_linear(x, 3))
    print("ts_min", ts_min(x, 3))
    print("ts_max", ts_max(x, 3))
    print("ts_argmin", ts_argmin(x, 3))
    print("ts_argmax", ts_argmax(x, 3))
    print("ts_rank", ts_rank(x, 3))
    print("ts_sum", ts_sum(x, 3))
    print("ts_product", ts_product(x, 3))
    print("ts_stddev", ts_stddev(x, 3))
