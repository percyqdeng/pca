import numpy as np


class MyPCA(object):

    def __init__(self, k, problem='l0l2-c', centered=True):
        self.k = k
        self.problem = problem
        self.explained_variance = None
        self.w = None
        self.centered = centered

    @staticmethod
    def proj_l0(g, k):
        p = len(g)
        w = np.zeros(p)
        ind = np.argsort(np.fabs(g))[-k::]
        wk = g[ind]
        w[ind] = wk
        w /= np.linalg.norm(wk)
        return w

class BruteForcePCA(MyPCA):
    def __init__(self, k=10, T=1000, problem='l0l2-c', centered=True):
        super(BruteForcePCA, self).__init__(k, problem, centered)
        self.T = T

    def fit(self, x):
        pass


class BatchPCA(MyPCA):

    def __init__(self, k=10, T=1000, problem='l0l2-c', centered=True):
        super(BatchPCA, self).__init__(k, problem, centered)
        self.T = T

    def fit(self, x):
        x = np.copy(x)
        if self.centered:
            x -= x.mean(axis=0)
        n, p = x.shape
        C = x.T.dot(x) / n
        w = np.random.normal(0, 1, p)
        self.trainobj = []
        self.iters = []
        num_iters = 0
        interval = np.maximum(1, self.T / 20)
        for t in xrange(self.T):
            g = C.dot(w)
            w = MyPCA.proj_l0(g, self.k)
            if t == num_iters:
                # print "iter %d" % t
                self.trainobj.append(w.dot(C.dot(w)))
                self.iters.append(num_iters)
                num_iters += interval
        self.w = w
        self.explained_variance = w.dot(C.dot(w))


class OnlinePCA(MyPCA):
    """
    conditional gradient online pca
    """
    def __init__(self, problem='l0l2-c', centered=True, k=10, r=10, lmda=0.1, T=1000):
        """
        :param problem: problem type:  'l0l2-c',
        :param k:  ||w||_p <=k
        :param r:  rank constraint in approximated online pca
        :param lmda:  regularizer in penalized pca
        :param T:   maximal number of iteration
        :return:
        """
        super(OnlinePCA, self).__init__(k, problem, centered)
        self.lmda = lmda
        self.T = T
        self.r = r
        self.trainobj = None
        self.iters = None

    def fit(self, x):
        x = np.copy(x)
        if self.centered:
            x -= x.mean(axis=0)
        n, p = x.shape
        if self.problem == 'l0l2-c':
            self._l0_l2_const_pca(x)

    def _l2_const_pca(self, x, w0=None):
        n, p = x.shape
        C = np.zeros((p, p))
        if w0 is None:
            w = np.random.normal(0, 1, p)
        else:
            w = w0
        ind = np.random.choice(n, size=self.T, replace=True)
        interval = self.T / 20
        for t in xrange(1, self.T+1):
            i = ind[t]
            cur_x = x[i, :]
            gm = 1.0 / t
            # gm = 1.0 / (t*(np.log(t))**2)
            C = (1-gm)*C + gm*np.dot(cur_x[:, np.newaxis], cur_x[np.newaxis, :])
            grad = np.dot(C, w)
            w = MyPCA.proj_l0(grad, self.k)
        self.w = w

    def _l0_l2_const_pca(self, x, w0=None):
        """
        l0 l2 constraint online pca
         |w||_2 <= 1; ||w||_0 <= k
        :param x: sample data
        :param w0:initialized value
        :param k:
        :return: w
        """
        n, p = x.shape
        C = np.zeros((p, p))
        if w0 is None:
            w = np.random.normal(0, 1, p)
        else:
            w = w0
        ind = np.random.choice(n, size=self.T, replace=True)
        interval = np.maximum(1, self.T / 20)
        num_iters = 0
        self.trainobj = []
        self.iters = []
        Cov = np.cov(x.T)
        for t in xrange(self.T):
            i = ind[t]
            cur_x = x[i, :]
            gm = 1.0 / (1+t)
            # gm = 1.0 / (t*(np.log(t))**2)
            C = (1-gm)*C + gm*np.dot(cur_x[:, np.newaxis], cur_x[np.newaxis, :])
            grad = np.dot(C, w)
            w = MyPCA.proj_l0(grad, self.k)
            if t == num_iters:
                self.iters.append(t)
                self.trainobj.append(w.dot(Cov.dot(w)))
                num_iters += interval
                # print "%d " % num_iters
        self.w = w
        self.explained_variance = w.dot(C.dot(w))

    def _l1_const_pca(self, x):
        pass

    def _l1_reg_pca(self, x):
        pass
