from mypca import *
from load_data import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import cross_validation as cv



"""
experiment on random data
"""

n = 10
m = 6

n = 5000
m = 150
klist = xrange(2, n)
n_iters = 100
obj = np.zeros((n_iters, len(klist)))

for j in range(n_iters):
    print "iter# %d" % j
    for i, k in enumerate(klist):
        x = np.random.normal(0, 1.0/m, (m, n))
        C = np.cov(x.T)
        pca = PCA(n_components=1)
        pca.fit(x)

        pca2 = BatchPCA(k=k, T=100)
        pca2.fit(x)
        obj[j, i] = pca2.explained_variance / pca.explained_variance_


plt.figure()
plt.errorbar(klist, obj.mean(axis=0), yerr=obj.std(axis=0), fmt='x-')






