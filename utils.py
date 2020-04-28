import numpy as np

def svd_diag(sigma, m, n):
    sigma = np.diag(sigma)
    if m < n:
        sigma = np.c_[sigma, np.zeros((m, n - m))]
    elif m > n:
        sigma = np.r_[sigma, np.zeros((m - n, n))]
    return sigma


def is_equal_array_over_eps(a, b, ep=1e-8):
    # Relative Comparision
    diff = abs(a - b)
    maximal = np.max(np.c_[abs(a), abs(b)])
    rel_error = diff / maximal
    return np.all(rel_error < ep)