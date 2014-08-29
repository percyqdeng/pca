from mypca import *
from load_data import *
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import cross_validation as cv

data = load_mnist()

x, y = convert_one_vs_all(data, 1)
x = preprocessing.scale(x)


n, p = x.shape

n_iter = 1
T = 1000
ss = cv.ShuffleSplit(n, n_iter=1, test_size=0.4, train_size=0.6, random_state=22)

for i, (train_ind, test_ind) in enumerate(ss):
    xtrain = x[train_ind, :]
    ntrain = xtrain.shape[0]
    print "run online PCA"
    l0opca = OnlinePCA(problem='l0l2-c', k=30, T=T)
    l0opca.fit(xtrain)
    print "run offline PCA"
    pca2 = BatchPCA(problem='l0l2-c', T=T, k=30)
    pca2.fit(xtrain)

plt.figure()
plt.plot(l0opca.iters, l0opca.trainobj, label='onlinepca')
plt.plot(pca2.iters, pca2.trainobj, label='batchpca')
plt.legend(loc='best')
plt.show()





