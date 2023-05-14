import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

import cv2 as cv
import torch

def list_to_array(*lst):
    r""" Convert a list if in numpy format """
    if len(lst) > 1:
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]

def mySinkhorn(a, b, M, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, warn=True):
    a, b, M = list_to_array(a, b, M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    # we assume that no distances are null except those of the diagonal of distances
    if n_hists:
        u = np.ones((dim_a, n_hists), dtype=np.float64) / dim_a
        v = np.ones((dim_b, n_hists), dtype=np.float64) / dim_b
    else:
        u = np.ones(dim_a, dtype=np.float64) / dim_a
        v = np.ones(dim_b, dtype=np.float64) / dim_b

    K = np.exp(M / (-reg))

    Kp = (1 / a).reshape(-1, 1) * K

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = b / KtransposeU
        u = 1. / np.dot(Kp, v)
    Gs = u.reshape((-1, 1)) * K * v.reshape((1, -1))
    if n_hists:  # return only loss
        res = np.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))



n = 200  # nb bins
m = 150
x = np.arange(n, dtype=np.float64)
y = np.arange(m, dtype=np.float64)
# Gaussian distributions
a = gauss(n, m=20, s=5)  # m= mean, s= std
b = gauss(m, m=60, s=10)
# loss matrix
M = ot.dist(x.reshape((n, 1)), y.reshape((m, 1)))
M /= M.max()

fs = cv.FileStorage('./sinkhorn_debug.yaml',cv.FILE_STORAGE_WRITE)
fs.write("a",a)
fs.write("b",b)
fs.write("M",M)
fs.release()


lambd = 1e-3
Gs = ot.sinkhorn(a, b, M, reg=lambd, verbose=False)
Gx = mySinkhorn(a, b, M, reg=lambd, numItermax=20, verbose=False)
pl.figure(4, figsize=(5, 5))
# ot.plot.plot1D_mat(a, b, Gs, 'OT matrix Sinkhorn')
# pl.show()
# ot.plot.plot1D_mat(a, b, Gx, 'OT matrix Sinkhorn')
pl.imshow(Gx)
pl.show()