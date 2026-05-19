from numba import cuda
import math

@cuda.jit(device=True)
def cost(real, estimated):
    max_i, max_j = real.shape
    mse = 0
    for i in range(max_i):
        for j in range(max_j):
            r = real[i,j]
            e = estimated[i,j]
            # do not take into account gain measurements with numerical errors (saved as nan in real)
            # do not take into account gain calc with numerical errors (saved as nan in estimated matrix), but it is recommended to add penalization to cost later on
            err = r-e
            mse += err*err if not (math.isnan(r) or math.isnan(e) or math.isinf(r) or math.isinf(e)) else 0
    mse /= (max_i*max_j)
    return mse

@cuda.jit(device=True)
def count_nan(estimated):
    max_i, max_j = estimated.shape
    nan_count = 0
    for i in range(max_i):
        for j in range(max_j):
            nan_count += 1 if math.isnan(estimated[i,j]) else 0  # do not take into account gain measurements with numerical errors (saved as nan)
    return nan_count

