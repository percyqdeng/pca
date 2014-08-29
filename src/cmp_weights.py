import numpy as np
import matplotlib.pyplot as plt


T = 10000
weight = np.zeros(T+1)
seq = np.asarray(range(1, T+1))
gm = 1 / ((seq + 2) * np.log(seq+2) * np.log(seq+2))
# gm = 1/ (seq+1.0)
# gm =
# gm = np.fabs(np.sin(seq)/seq)
lmda = np.cumprod((1-gm)[::-1])[::-1]
lmda = np.hstack((lmda, 1))
weight[0] = 1
weight[1::] = gm
weight = weight * lmda
plt.plot(weight)
plt.yscale('log')

ss = (weight**2).sum()


