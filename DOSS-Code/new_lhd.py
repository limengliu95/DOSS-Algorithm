import pySOT

from rbfopt.rbfopt_utils import get_lhd_maximin_points, get_min_distance

import numpy as np
import scipy.linalg as la

class LatinHypercube(pySOT.experimental_design.LatinHypercube):
    def __init__(self, dim, num_pts, criterion=None, iterations=1000, use_midpoint=False):
        super().__init__(dim=dim, num_pts=num_pts, criterion=criterion, iterations=iterations)
        self.use_midpoint = use_midpoint
        # self.use_midpoint = True

    def generate_points(self, lb=None, ub=None, int_var=None):
        midpoint = (lb + ub)/2
        dependent = True
        maxit = 50
        it = 0
        while(dependent and it < maxit):
            it += 1
            nodes = get_lhd_maximin_points(lb, ub, int_var, self.num_pts, num_trials=50)

            if(self.use_midpoint and get_min_distance(midpoint, nodes) > 1.0e-5):
                nodes = np.vstack((midpoint, nodes))[0:self.num_pts, :]

            norms = la.norm(nodes, axis=1)
            U, s, V = np.linalg.svd(nodes[norms > 1.0e-15])
            if (min(s) > 1.0e-6):
                dependent = False

        # print("GENERATE %d INITIAL PTS" % nodes.shape[0])
        return nodes