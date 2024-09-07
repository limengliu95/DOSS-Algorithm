import pySOT
import scipy.spatial as scpspatial
import numpy as np

class MultiquadricKernel(pySOT.surrogate.Kernel):
    def __init__(self):
        super().__init__()
        self.order = 1
        self.gamma_sq = 0.1*0.1
    
    def eval(self, dists):
        return (dists*dists + self.gamma_sq)**0.5

    def deriv(self, dists):
        return dists / self.eval(dists)

class RBFInterpolant(pySOT.surrogate.RBFInterpolant):
    def __init__(self, dim, kernel=None, tail=None, eta=1e-6):
        # print("DIM: %d, KERNEL ORDER: %d" % (dim, kernel.order))
        super().__init__(dim=dim, kernel=kernel, tail=tail, eta=eta)

        self.firstdet = True
        self.firstdet1 = True
        self.firstdet2 = True
        self.kernel0 = pySOT.surrogate.LinearKernel()
        self.tail0 = pySOT.surrogate.ConstantTail(self.dim)
        self.kernel1 = MultiquadricKernel()
        self.tail1 = pySOT.surrogate.ConstantTail(self.dim)
        self.kernel2 = pySOT.surrogate.CubicKernel()
        self.tail2 = pySOT.surrogate.LinearTail(self.dim)
        self.kernel3 = pySOT.surrogate.TPSKernel()
        self.tail3 = pySOT.surrogate.LinearTail(self.dim)
        # self.kernel2 = pySOT.surrogate.LinearKernel()
        # self.tail2 = pySOT.surrogate.ConstantTail(self.dim)
        # self.kernel2 = MultiquadricKernel()
        # self.tail2 = pySOT.surrogate.ConstantTail(self.dim)

        self.kernel = self.kernel1
        self.tail = self.tail1
        self.ntail = self.tail.dim_tail
        # print("DIM: %d, KERNEL ORDER: %d" % (self.dim, self.kernel.order))

    def reset(self):
        super().reset()
        self.firstdet = True
        self.firstdet1 = True
        self.firstdet2 = True

        self.kernel = self.kernel1
        self.tail = self.tail1
        self.ntail = self.tail.dim_tail

    def _fit(self):
        if self.fX.shape[0] >= 2*(self.dim+1):
            # print("DET %d" % (self.fX.shape[0]))
            if self.firstdet2:
                self.c = None
                self.firstdet2 = False
                self.kernel = self.kernel2
                self.tail = self.tail2
            # self.kernel = self.kernel3
            # self.tail = self.tail3
                self.ntail = self.tail.dim_tail
            super()._fit()
        elif self.fX.shape[0] >= self.dim+1:
            if self.firstdet1:
                self.c = None
                self.firstdet1 = False
            # self.kernel = self.kernel1
            # self.tail = self.tail1
            # self.ntail = self.tail.dim_tail
            super()._fit()
        else:
            # print("UDET %d" % (self.fX.shape[0]))
            self._fit_underdet()

    def _fit_underdet(self):
        n = self.num_pts
        ntail = self.ntail
        nact = ntail + n

        assert self.num_pts >= ntail

        X = self.X[0:n, :]
        D = scpspatial.distance.cdist(X, X)
        Phi = self.kernel.eval(D) + self.eta * np.eye(n)
        P = self.tail.eval(X)

        # Set up the systems matrix
        A1 = np.hstack((np.zeros((ntail, ntail)), P.T))
        A2 = np.hstack((P, Phi))
        A = np.vstack((A1, A2))

        # Construct the usual pivoting vector so that we can increment
        # self.piv = np.arange(0, nact)
        # for i in range(nact):
        #     self.piv[i], self.piv[piv[i]] = \
        #         self.piv[piv[i]], self.piv[i]

        rhs = np.vstack((np.zeros((ntail, 1)), self.fX))

        self.c, res, rank, svd = np.linalg.lstsq(A, rhs, rcond=-1)

        # print(self.c)

        self.updated = True
