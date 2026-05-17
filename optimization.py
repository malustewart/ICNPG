import numpy as np

def cost(real, estimated):
    real = np.where(np.isnan(real), estimated, real)
    return np.mean((real - estimated) ** 2)