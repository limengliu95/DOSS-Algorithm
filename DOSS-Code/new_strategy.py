# import pySOT.strategy
# from pySOT.strategy import DYCORSStrategy
import pySOT.strategy
from pySOT.auxiliary_problems import round_vars, weighted_distance_merit, unit_rescale, candidate_dycors
# from pySOT.surrogate import SurrogateUnitBox, RBFInterpolant, LinearTail, CubicKernel
from pySOT.surrogate import SurrogateUnitBox, LinearTail, CubicKernel
from new_auxfunc import getCdnKnowledge, checkSurrogate, setSurrogate, truncated_get_pareto, generate_centers, candidate_dycors_ls, candidate_mcdycors, candidate_tvpdycors, candidate_tvpwdycors, candidate_trdycors, getNeighbors, localopt, candidate_lodycors, candidate_dsdycors, candidate_ckdycors, candidate_ddsdycors, candidate_gadycors, candidate_cddycors, candidate_sdsgdycors, candidate_sdsgckdycors, candidate_sdsgckdycors_std

import scipy.spatial as scpspatial
import scipy.linalg as scplinalg
import numpy as np
import logging, math

logger = logging.getLogger(__name__)

class Accuracy:
    def __init__(self):
        self.global_ave = []
        self.global_min = []
        self.global_max = []

        self.local_ave = []
        self.local_min = []
        self.local_max = []

        self.mixed_ave = []
        self.mixed_min = []
        self.mixed_max = []

    def addAcc(self, global_ave, global_min, global_max,
        local_ave, local_min, local_max, 
        mixed_ave, mixed_min, mixed_max):
        self.global_ave.append(global_ave)
        self.global_min.append(global_min)
        self.global_max.append(global_max)

        self.local_ave.append(local_ave)
        self.local_min.append(local_min)
        self.local_max.append(local_max)

        self.mixed_ave.append(mixed_ave)
        self.mixed_min.append(mixed_min)
        self.mixed_max.append(mixed_max)

    def reset(self):
        self.global_ave = []
        self.global_min = []
        self.global_max = []

        self.local_ave = []
        self.local_min = []
        self.local_max = []

        self.mixed_ave = []
        self.mixed_min = []
        self.mixed_max = []

class RBFInterpolant(pySOT.surrogate.RBFInterpolant):
    def _fit(self):
        """Compute new coefficients if the RBF is not updated.

        We try to update an existing LU factorization by computing a Cholesky
        factorization of the Schur complemented system. This may fail if the
        system is ill-conditioned, in which case we compute a new LU
        factorization.
        """
        if not self.updated:
            n = self.num_pts
            ntail = self.ntail
            nact = ntail + n

            if self.c is None:  # Initial fit
                assert self.num_pts >= ntail

                X = self.X[0:n, :]
                D = scpspatial.distance.cdist(X, X)
                Phi = self.kernel.eval(D) + self.eta * np.eye(n)
                P = self.tail.eval(X)

                # Set up the systems matrix
                A1 = np.hstack((np.zeros((ntail, ntail)), P.T))
                A2 = np.hstack((P, Phi))
                A = np.vstack((A1, A2))

                # print(A)
                [LU, piv] = scplinalg.lu_factor(A)
                self.L = np.tril(LU, -1) + np.eye(nact)
                self.U = np.triu(LU)

                # Construct the usual pivoting vector so that we can increment
                self.piv = np.arange(0, nact)
                for i in range(nact):
                    self.piv[i], self.piv[piv[i]] = \
                        self.piv[piv[i]], self.piv[i]

            else:  # Extend LU factorization
                k = self.c.shape[0] - ntail
                numnew = n - k
                kact = ntail + k

                X = self.X[:n, :]
                XX = self.X[k:n, :]
                D = scpspatial.distance.cdist(X, XX)
                Pnew = np.vstack((self.tail.eval(XX).T,
                                  self.kernel.eval(D[:k, :])))
                Phinew = self.kernel.eval(D[k:, :]) + self.eta * np.eye(numnew)

                L21 = np.zeros((kact, numnew))
                U12 = np.zeros((kact, numnew))
                for i in range(numnew):  # TODO: Can we use level-3 BLAS?
                    L21[:, i] = scplinalg.solve_triangular(
                        a=self.U, b=Pnew[:kact, i], lower=False, trans='T')
                    U12[:, i] = scplinalg.solve_triangular(
                        a=self.L, b=Pnew[self.piv[:kact], i],
                        lower=True, trans='N')
                L21 = L21.T
                try:  # Compute Cholesky factorization of the Schur complement
                    C = scplinalg.cholesky(
                        a=Phinew - np.dot(L21, U12), lower=True)
                finally:  # Compute a new LU factorization if Cholesky fails
                    self.c = None
                    return self._fit()

                self.piv = np.hstack((self.piv, np.arange(kact, nact)))
                self.L = np.vstack((self.L, L21))
                L2 = np.vstack((np.zeros((kact, numnew)), C))
                self.L = np.hstack((self.L, L2))
                self.U = np.hstack((self.U, U12))
                U2 = np.hstack((np.zeros((numnew, kact)), C.T))
                self.U = np.vstack((self.U, U2))

            # Update coefficients
            rhs = np.vstack((np.zeros((ntail, 1)), self.fX))
            self.c = scplinalg.solve_triangular(
                a=self.L, b=rhs[self.piv], lower=True)
            self.c = scplinalg.solve_triangular(
                a=self.U, b=self.c, lower=False)
            self.updated = True

class DYCORSStrategy(pySOT.strategy.DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None):

        # self.stat_fbest = []
        # self.stat_neval = 0
        self.stat_numpts_gs = []

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        # self.stat_numpts_gs.append(self.surrogate.num_pts)

        # xbest = np.copy(self._X[np.argmin(self._fX), :]).ravel()
        # getCdnKnowledge(self._X, self._fX, xbest)

        # print(self.sampling_radius)
        new_points = candidate_dycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb)

        # print(new_points)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

class SOPStrategy(pySOT.strategy.SOPStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate, ncenters=4,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, use_restarts=True, num_cand=None, lsg=False):

        self.lsg = lsg

        super().__init__(max_evals=max_evals, opt_prob=opt_prob, exp_design=exp_design,
                         surrogate=surrogate, ncenters=ncenters, asynchronous=asynchronous, 
                         batch_size=batch_size, extra_points=extra_points,extra_vals=extra_vals, use_restarts=use_restarts, num_cand=num_cand)
    
    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""

        # Update the list of center points
        if self.F_ranked is None:  # If this is the start of adaptive phase
            self.update_ranks()
        self.update_center_list()

        # Compute dycors perturbation probability
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        # Perturb each center to propose one new eval per center
        new_points = np.zeros((num_pts, self.opt_prob.dim))
        weights = [1.0]
        for i in range(num_pts):
            # Deduce index of next available center
            center_index = 0
            for center in self.centers:
                if center.new_point is None:
                    break
                center_index += 1
            # Select new point by candidate search around center
            X_c = self.centers[center_index].xc
            sampling_radius =\
                self.evals[self.centers[center_index].index].sigma

            if self.lsg:
                # Generate local surrogate
                # xbest = np.copy(self._X[np.argmin(self._fX), :]).ravel()
                xbest = X_c
                local_surrogate = SurrogateUnitBox(
                    RBFInterpolant(dim=self.opt_prob.dim, kernel=CubicKernel(),
                    tail=LinearTail(self.opt_prob.dim), eta=1e-6), lb=self.opt_prob.lb, ub=self.opt_prob.ub)
                radius = 0.1*np.average(self.opt_prob.ub-self.opt_prob.lb)
                xx, fx = getNeighbors(self._X, self._fX, radius, xbest)
                local_surrogate.add_points(xx, fx)
            else:
                local_surrogate = None
                radius = None
                # self.stat_numpts_gs.append(surrogate.num_pts)

            new_points[i, :] =\
                candidate_dycors_ls(num_pts=1, opt_prob=self.opt_prob,
                                 surrogate=self.surrogate, X=self._X,
                                 fX=self._fX, weights=weights,
                                 sampling_radius=sampling_radius,
                                 num_cand=self.num_cand, Xpend=self.Xpend,
                                 prob_perturb=prob_perturb, xbest=X_c, 
                                 localsurrogate=local_surrogate, radius=radius)

            self.centers[center_index].new_point = new_points[i, :]

        # submit the new points
        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

class TVDYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, use_restarts=True, weights=None, num_cand=None, 
                 Pstrgy=True, Wstrgy=True):

        self.Pstrgy = Pstrgy
        self.Wstrgy = Wstrgy

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts,  weights=weights,
                         num_cand=num_cand)
    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        ''' P-strategy '''
        if self.Pstrgy == True:
            theta = 2 * (1.0 - (np.log(num_evals)/np.log(budget)))
            # theta = 2 * (1.0 - (np.log(num_evals)/np.log(self.max_evals)))
            # print(theta)
            if theta > 1:
                C = 1
            elif theta >= 0.5:
                C = theta
            else:
                C = 0.5
        else:
            C = 1

        ''' W-strategy '''
        if self.Wstrgy == True:
            if num_evals % self.opt_prob.dim == 0:
                weights = [0.95]
            else:
                weights = [0.95-(0.95-0.5)*(1-np.log(num_evals % self.opt_prob.dim)/np.log(self.opt_prob.dim))]
        else:
            weights = self.get_weights(num_pts=num_pts)
        new_points = candidate_dycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius*C,
            prob_perturb=prob_perturb)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

class MCDYCORSStrategy(TVDYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, ncenters=4, batch_size=None, extra_points=None,
                 extra_vals=None, use_restarts=True, weights=None, num_cand=None, 
                 Pstrgy=False, Wstrgy=False, lsg=False):

        self.lsg = lsg
        self.ncenters = ncenters
        self.isexploration = True
        self.explore_counter = 0
        self.exploit_radius = 0.2
        # self.Pstrgy = Pstrgy
        # self.Wstrgy = Wstrgy

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts,  weights=weights,
                         num_cand=num_cand, Pstrgy=Pstrgy, Wstrgy=Wstrgy)
    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        # prob_good = 0.25 * (1.0 - (np.log(num_evals)/np.log(budget)))
        # prob_good = max(prob_good, 1e-6)
        # meritweight = 0.7 * (1.0 - (np.log(num_evals)/np.log(budget)))
        # num_center = min(max(num_pts, self.ncenters), self.num_cand)
        num_center = min(self.ncenters, self.num_cand)
        # num_center = 1
        # self.isexploration = False
        # self.isexploration = True
        if self.isexploration:
            num_center = min(self.ncenters, self.num_cand)
        else:
            num_center = 1
        # frequency = 3
        # frequency = 1 + np.ceil(2*(1.0 - (1.0 - (np.log(num_evals)/np.log(budget)))))
        # print(frequency)
        center, ind_center = generate_centers(num_center=num_center, isexploration=self.isexploration, opt_prob=self.opt_prob, X=self._X, fX=self._fX, Xpend=self.Xpend)
        num_center = len(center)
        # print(num_center)

        num_new_points_from_center = max(num_center,num_pts)
        if self.isexploration:
            pts_ratio = 1.0/num_center
            num_cand_per_center = int(pts_ratio*self.num_cand)*np.ones((num_center,))
            num_cand_per_center[num_cand_per_center.shape[0]-1] = self.num_cand-np.sum(num_cand_per_center[0:num_cand_per_center.shape[0]-1])
            num_new_points_per_center = int(pts_ratio*num_new_points_from_center)*np.ones((num_center,))
            num_new_points_per_center[num_new_points_per_center.shape[0]-1] = num_new_points_from_center - np.sum(num_new_points_per_center[0:num_new_points_per_center.shape[0]-1])
        else:
            if num_center == 2:
                pts_ratio = [0.7, 0.3]
            elif num_center == 1:
                pts_ratio = [1.0]
            num_cand_per_center = np.array(list(int(r*self.num_cand) for r in pts_ratio))
            num_new_points_per_center = np.array(list(int(r*num_new_points_from_center) for r in pts_ratio))
            num_cand_per_center[num_cand_per_center.shape[0]-1] = self.num_cand-np.sum(num_cand_per_center[0:num_cand_per_center.shape[0]-1])
            num_new_points_per_center[num_new_points_per_center.shape[0]-1] = num_new_points_from_center - np.sum(num_new_points_per_center[0:num_new_points_per_center.shape[0]-1])
        
        # num_cand_per_center = int(np.floor(self.num_cand/num_center))
        
        # print(num_center)
        
        # num_new_points_per_center = int(np.floor(num_new_points_from_center/num_center))
        
        cand = np.zeros((num_new_points_from_center, self.opt_prob.dim))
        # pts = 0
        # weights = np.ones((num_new_points_from_center,))
        tmp_npis = []
        current_cand = 0
        for ind_center in range(num_center):
            # Each center generates one point
            # if ind_center < num_center - 1:
            #     num_cand = num_cand_per_center
            #     num_new_points = num_new_points_per_center
            # else:
            #     num_cand = self.num_cand-(num_center-1)*num_cand_per_center
            #     num_new_points = num_new_points_from_center-(num_center-1)*num_new_points_per_center
            num_cand = int(num_cand_per_center[ind_center])
            num_new_points = int(num_new_points_per_center[ind_center])

            # print((num_cand, num_new_points))

            if self.lsg == True:
                # Generate local surrogate
                xbest = center[ind_center]
                local_surrogate = SurrogateUnitBox(
                    RBFInterpolant(dim=self.opt_prob.dim, kernel=CubicKernel(),
                    tail=LinearTail(self.opt_prob.dim)), lb=self.opt_prob.lb, ub=self.opt_prob.ub)
                radius = 0.2*np.average(self.opt_prob.ub-self.opt_prob.lb)
                xx, fx = getNeighbors(self._X, self._fX, radius, xbest)
                local_surrogate.add_points(xx, fx)
                surrogate = local_surrogate
                
            else:
                surrogate = self.surrogate

            tmp_npis.append(surrogate.num_pts)

            if self.isexploration:
                weights = 0.4*np.ones((num_new_points,))
                sampling_radius = 0.8

                cand[current_cand:current_cand+num_new_points] = candidate_dycors(
                opt_prob=self.opt_prob, num_pts=num_new_points, surrogate=surrogate,
                X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
                num_cand=num_cand, sampling_radius=sampling_radius,
                prob_perturb=prob_perturb, xbest=center[ind_center])
            else:
                weights = self.get_weights(num_pts=num_new_points)
                sampling_radius = self.sampling_radius
                # if self.explore_counter % 2 == 0:
                #     weights = 0.8*np.ones((num_new_points,))
                #     # sampling_radius = 0.3
                #     sampling_radius = self.sampling_radius
                # else:
                #     weights = 0.95*np.ones((num_new_points,))
                #     # sampling_radius = 0.2
                #     sampling_radius = self.sampling_radius
                #     # print(self.sampling_radius)
                cand[current_cand:current_cand+num_new_points] = candidate_trdycors(
                opt_prob=self.opt_prob, num_pts=num_new_points, surrogate=surrogate,
                X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
                num_cand=num_cand, sampling_radius=sampling_radius,
                prob_perturb=prob_perturb, xbest=center[ind_center])
                    

            # self.sampling_radius = 0.2
            # cand[current_cand:current_cand+num_new_points] = candidate_dycors(
            #     opt_prob=self.opt_prob, num_pts=num_new_points, surrogate=surrogate,
            #     X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            #     num_cand=num_cand, sampling_radius=sampling_radius,
            #     prob_perturb=prob_perturb, xbest=center[ind_center])
            
            current_cand += num_new_points
            # cand[ind_center, :] = candidate_dycors(
            #     opt_prob=self.opt_prob, num_pts=1, surrogate=surrogate,
            #     X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            #     num_cand=num_cand, sampling_radius=self.sampling_radius,
            #     prob_perturb=prob_perturb, xbest=center[ind_center])
            

        if num_center > num_pts:
            self.stat_numpts_gs.append(np.average(tmp_npis))
            # weights = self.get_weights(num_pts=num_pts)
            if self.isexploration:
                weights = 0.4*np.ones((num_pts,))
            else:
                weights = self.get_weights(num_pts=num_new_points)
                # if self.explore_counter % 2 == 0:
                #     weights = 0.8*np.ones((num_new_points,))
                # else:
                #     weights = 0.95*np.ones((num_new_points,))
                # weights = 0.85*np.ones((num_pts,))
            new_points = weighted_distance_merit(
                num_pts=num_pts, surrogate=self.surrogate, X=self._X, fX=self._fX,
                Xpend=self.Xpend, cand=cand, dtol=1e-3, weights=weights)
        else:
            new_points = cand
        # new_points = candidate_mcdycors(
        #     opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
        #     X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
        #     num_cand=self.num_cand, sampling_radius=self.sampling_radius,
        #     prob_perturb=prob_perturb, prob_good=prob_good, meritweight=meritweight)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))


    def adjust_step(self):
        """Adjust the sampling radius sigma.
        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.
        """
        # Check if we succeeded at significant improvement
        fbest_new = min([record.value for record in self.record_queue])
        if fbest_new < self._fbest - 1e-3*math.fabs(self._fbest) or np.isinf(self._fbest):  # Improvement
            self._fbest = fbest_new
            if not self.isexploration:
                self.status = max(1, self.status + 1)
                self.failcount = 0
        else:
            if not self.isexploration:
                self.status = min(-1, self.status - 1)  # No improvement
                self.failcount += 1

        if self.isexploration:
            self.isexploration = False
            self.sampling_radius = self.exploit_radius

            self.record_queue = []
            return

        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            logger.info("Reducing sampling radius")
            # print("Reducing sampling radius: %.3f -> %.3f" % (self.sampling_radius, self.sampling_radius/2))
            self.sampling_radius /= 2
            
        if self.status >= self.succtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            logger.info("Increasing sampling radius")
            # print("Increasing sampling radius: %.3f -> %.3f" % (self.sampling_radius, min([2.0 * self.sampling_radius, self.sampling_radius_max])))
            self.sampling_radius = min([2.0 * self.sampling_radius,
                                        self.sampling_radius_max])
            

        # Check if we have converged
        if self.failcount >= self.maxfailtol or \
                self.sampling_radius <= self.sampling_radius_min:
            self.converged = True

        # Empty the queue
        self.record_queue = []

        frequency = 100
        self.explore_counter += 1
        if self.explore_counter >= frequency:# and self._fX.shape[0] < 0.8*self.max_evals:
            self.isexploration = True
            self.exploit_radius = self.sampling_radius
            self.explore_counter = 0

class TRDYCORSStrategy(pySOT.strategy.DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None):

        # self.stat_fbest = []
        # self.stat_neval = 0
        self.stat_numpts_gs = []

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)

        self.sampling_radius = 0.4
        self.sampling_radius_min = 2**(-8)
        self.sampling_radius_max = 0.8
        # self.failtol = math.ceil(opt_prob.dim/batch_size)
        # self.succtol = 3
        # self.weights = [0.95, 0.8, 0.5, 0.3]
        # self.weights = [0.9, 0.7, 0.3, 0.1]
        # self.weights = [0.3, 0.5, 0.8, 0.95]
        # self.weights = [0.6, 0.75, 0.9, 0.975]
        # self.weights = [1.0]
        self.weights_queue = []
        self.highweightwin = 0
    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)
        self.weights_queue = weights

        self.stat_numpts_gs.append(self.surrogate.num_pts)

        new_points = candidate_trdycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

    def adjust_step(self):
        """Adjust the sampling radius sigma.
        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.
        """
        # Check if we succeeded at significant improvement
        # print(len(self.record_queue))
        fbest_new = min([record.value for record in self.record_queue])
        fbest_new_ind = np.argmin([record.value for record in self.record_queue])
        if fbest_new < self._fbest - 1e-3*math.fabs(self._fbest) or np.isinf(self._fbest):  # Improvement
            # print("Improvement: %.3f -> %.3f (weight=%.3f, %d)" % (self._fbest, fbest_new, self.weights_queue[fbest_new_ind], fbest_new_ind))
            self._fbest = fbest_new
            self.status = max(1, self.status + 1)
            self.failcount = 0
        else:
            self.status = min(-1, self.status - 1)  # No improvement
            self.failcount += 1

        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            logger.info("Reducing sampling radius")
            # print("Reducing sampling radius: %.3f -> %.3f" % (self.sampling_radius, self.sampling_radius/2))
            self.sampling_radius /= 2
            
        if self.status >= self.succtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            logger.info("Increasing sampling radius")
            # print("Increasing sampling radius: %.3f -> %.3f" % (self.sampling_radius, min([2.0 * self.sampling_radius, self.sampling_radius_max])))
            self.sampling_radius = min([2.0 * self.sampling_radius,
                                        self.sampling_radius_max])
            

        # Check if we have converged
        if self.failcount >= self.maxfailtol or \
                self.sampling_radius <= self.sampling_radius_min:
            self.converged = True

        # Empty the queue
        self.record_queue = []

    def sample_initial(self):
        super().sample_initial()
        self.status = 0          # Status counter
        self.failcount = 0       # Failure counter
        self.sampling_radius = 0.4
        self._fbest = np.inf  # Current best function value

class LSDYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, use_restarts=True, weights=None, num_cand=None):

        self.localsurrogate = None
        self.stat_accuracy = Accuracy()
        self.stat_numpts_ls = []
        self.stat_numpts_gs = []
        self.subspace_dim = []

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts,  weights=weights,
                         num_cand=num_cand)
    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        # Generate local surrogate
        xbest_ind = np.argmin(self._fX)
        xbest = np.copy(self._X[xbest_ind, :]).ravel()
        radius = 0.3*np.average(self.opt_prob.ub-self.opt_prob.lb)
        # print(self._fX.shape[0])
        xx, fx, lb, ub, self.subspace_dim, radius = getNeighbors(self._X, self._fX, radius, xbest, xbest_ind)
        lb = self.opt_prob.lb[self.subspace_dim]
        ub = self.opt_prob.ub[self.subspace_dim]
        # if self.localsurrogate == None:
            # self.localsurrogate = SurrogateUnitBox(
            #     RBFInterpolant(dim=self.opt_prob.dim, kernel=CubicKernel(),
            #     tail=LinearTail(self.opt_prob.dim), eta=1e-6), lb=self.opt_prob.lb, ub=self.opt_prob.ub)
        self.localsurrogate_dim = xx.shape[1]
        self.localsurrogate = SurrogateUnitBox(
            RBFInterpolant(dim=self.localsurrogate_dim, kernel=CubicKernel(),
            tail=LinearTail(self.localsurrogate_dim), eta=1e-6), lb=lb, ub=ub)
            # self.localsurrogate = RBFInterpolant(dim=self.opt_prob.dim, kernel=CubicKernel(), tail=LinearTail(self.opt_prob.dim), eta=1e-3)
        # else:
            # self.localsurrogate.reset()
        self.localsurrogate.add_points(xx, fx)
        self.stat_numpts_ls.append(self.localsurrogate.num_pts)
        self.stat_numpts_gs.append(self.surrogate.num_pts)
        # print("%d, %d, %.3f" % (self.localsurrogate.num_pts, self.surrogate.num_pts, radius))

        new_points = candidate_dycors_ls(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb, localsurrogate=self.localsurrogate, radius=radius, 
            subspace_dim=self.subspace_dim, accuracy=self.stat_accuracy)
        # new_points = candidate_dycors_ls(
        #     opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
        #     X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
        #     num_cand=self.num_cand, sampling_radius=self.sampling_radius,
        #     prob_perturb=prob_perturb)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

class LODYCORSStrategy(pySOT.strategy.DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate, lo_density_metric=True, 
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None):

        # self.stat_fbest = []
        # self.stat_neval = 0
        self.stat_numpts_gs = []
        self.stat_density = []
        # self.lo_density_metric = lo_density_metric
        self.lo_density_metric = False
        self.lo_max_density = 0.05*max_evals

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)
    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        self.stat_numpts_gs.append(self.surrogate.num_pts)

        new_points = candidate_lodycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb, stat_density=self.stat_density, 
            lo_max_density=self.lo_max_density, 
            lo_density_metric=self.lo_density_metric)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))


class DSDYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None):

        self.cand_scheme = 0 # 0: DYCORS, 1: DS
        self.stat_evalfrom_dycors = []
        self.stat_evalfrom_ds = []
        self._fbestx = None
        self.first_best = None
        self.ave_dist = []
        self.lowratio = 0
        self.highratio = 0
        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)
        # self.sampling_radius_max = 0.4
        # self.weights = [0.95, 0.95, 0.95, 0.95]
        # self.weights = [0.7, 0.95, 0.95]
        # self.weights = [0.7, 0.9]
        self.weights = [0.3, 0.5, 0.8, 0.95, 0.95, 0.95]
        # self.weights = [0.94, 0.94, 0.95, 0.95]
    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)
        # print(weights)
        # weights = [0.95, 0.95, 0.95, 0.95]

        self.stat_numpts_gs.append(self.surrogate.num_pts)

        new_points, evalfrom_dycors = candidate_dsdycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb, cand_scheme=1, 
            batch_size = self.batch_size)

        # self.cand_scheme = 1
        # # print("Scheme 1:")
        # new_points1, evalfrom_dycors = candidate_dsdycors(
        #     opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
        #     X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
        #     num_cand=self.num_cand, sampling_radius=self.sampling_radius,
        #     prob_perturb=prob_perturb, cand_scheme=0, 
        #     batch_size = self.batch_size)

        # # print("Scheme 2:")
        # new_points2, evalfrom_dycors = candidate_dsdycors(
        #     opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
        #     X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
        #     num_cand=self.num_cand, sampling_radius=self.sampling_radius,
        #     prob_perturb=prob_perturb, cand_scheme=1, 
        #     batch_size = self.batch_size)

        # # print("Scheme 3:")
        # new_points3, evalfrom_dycors = candidate_dsdycors(
        #     opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
        #     X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
        #     num_cand=self.num_cand, sampling_radius=self.sampling_radius,
        #     prob_perturb=prob_perturb, cand_scheme=2, 
        #     batch_size = self.batch_size)

        # if self.surrogate.predict(new_points1[0, :]) < self.surrogate.predict(new_points2[0, :]):
        #     if self.surrogate.predict(new_points1[0, :]) < self.surrogate.predict(new_points3[0, :]):
        #         new_points = new_points1
        #     else:
        #         new_points = new_points3
        # else:
        #     if self.surrogate.predict(new_points2[0, :]) < self.surrogate.predict(new_points3[0, :]):
        #         new_points = new_points2
        #     else:
        #         new_points = new_points3

        # getCdnKnowledge(self._X, self._fX, self._X[np.argmin(self._fX)])
        # print("Weight: %f, True value: %.2f, %.2f, %.2f" % (weights[0], self.opt_prob.eval(new_points1[0]), self.opt_prob.eval(new_points2[0]), self.opt_prob.eval(new_points3[0])))
        evalfrom_ds = num_pts - evalfrom_dycors
        self.stat_evalfrom_dycors.append(evalfrom_dycors)
        self.stat_evalfrom_ds.append(evalfrom_ds)

        self.cand_scheme = 1 - self.cand_scheme

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

    def adjust_step_new_backup(self):
        """Adjust the sampling radius sigma.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.
        """
        # Check if we succeeded at significant improvement
        # fbest_new = min([record.value for record in self.record_queue])
        fbest_new = self.record_queue[0].value
        best_record = self.record_queue[0]
        for record in self.record_queue:
            if record.value < fbest_new:
                fbest_new = record.value
                best_record = record
        if fbest_new < self._fbest - 1e-3*math.fabs(self._fbest) or np.isinf(self._fbest):  # Improvement
            if np.isinf(self._fbest):
                self.first_best = fbest_new
                ratio = 1.0
            else:
                ratio = (self._fbest - fbest_new) / (self.first_best - fbest_new)
                # print("%.3f" % ratio)

                

            self._fbest = fbest_new
            if self._fbestx is not None:
                dist = np.linalg.norm(self._fbestx - best_record.params[0])
                # ave_dist = np.mean(self.ave_dist)
                # print("%.3f, %.3f" % (dist, ave_dist))

                self.ave_dist.append(dist)

                
            self._fbestx = np.copy(best_record.params[0])
            self.status = max(1, self.status + 1)
            self.failcount = 0
          
        else:
            self.status = min(-1, self.status - 1)  # No improvement
            self.failcount += 1
            
            ratio = 0

        if ratio < 0.01:
            if ratio == 0:
                if self.failcount >= self.failtol:
                    self.lowratio += 1
                    self.highratio = 0
            else:
                self.lowratio += 1
            self.highratio = 0
            if self.lowratio >= 3:
                # self.converged = True
                # print("Reduce weights")
                self.lowratio = 0
                # if self.weights[0] > 0.3 and self.weights[1] > 0.3:
                #     if self.weights[0] < self.weights[1]:
                #         self.weights[0] -= 0.1
                #     else:
                #         self.weights[1] -= 0.1
        else:
            self.lowratio = 0
            self.highratio += 1
            if self.highratio == 1:
                self.highratio = 0

                # print("Increase weights")

                # if self.weights[0] < 0.9:
                #     self.weights[0] = 0.7
                # else:
                #     self.weights[1] = 0.7


        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            self.sampling_radius /= 2
            logger.info("Reducing sampling radius")
            # print("Reducing sampling radius")

            # print("Reduce weights")
            # if 0.8 not in self.weights:
            #     self.weights = [0.8, 0.95, 0.8, 0.95]
            # elif 0.5 not in self.weights: 
            #     self.weights = [0.5, 0.8, 0.95, 0.8]
            # elif 0.3 not in self.weights: 
            #     self.weights = [0.3, 0.5, 0.8, 0.95]


            # if 0.95 in self.weights: 
            #     self.weights = [0.8, 0.5, 0.8, 0.8]
            # elif 0.8 in self.weights: 
            #     self.weights = [0.8, 0.5, 0.3, 0.5]


            # else:
                # self.converged = True
                

        if self.status >= self.succtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            self.sampling_radius = min([2.0 * self.sampling_radius,
                                        self.sampling_radius_max])
            logger.info("Increasing sampling radius")
            # print("Increasing sampling radius")

            # if 0.3 in self.weights: 
            #     print("Increase weights")
            #     self.weights = [0.5, 0.8, 0.95, 0.8]
            # elif 0.5 in self.weights: 
            #     print("Increase weights")
            #     self.weights = [0.95, 0.8, 0.95, 0.8]
            # elif 0.8 in self.weights:
            #     print("Increase weights")
            #     self.weights = [0.95, 0.95, 0.95, 0.95]


        # Check if we have converged
        if self.failcount >= self.maxfailtol or \
                self.sampling_radius <= self.sampling_radius_min:
            self.converged = True

        # Empty the queue
        self.record_queue = []

        if self.converged:
            print("Restart!")


class DDSDYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None):
        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)

        # self.weights = [0.3, 0.5, 0.8, 0.95, 0.95, 0.95]

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        # self.stat_numpts_gs.append(self.surrogate.num_pts)

        # xbest = np.copy(self._X[np.argmin(self._fX), :]).ravel()
        # getCdnKnowledge(self._X, self._fX, xbest)

        # print(self.sampling_radius)
        new_points = candidate_ddsdycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb)

        # print(new_points)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

class GADYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None):
        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        new_points = candidate_gadycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

class CDDYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None):
        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)

        # self.weights = [0.3, 0.5, 0.8, 0.95]
        self.sampling_radius_max = 2
        self.lastbest = None
        self.lastweight = 0.3
        self.bestlocal = None
        self.localsurrogate = None

    def sample_initial(self):
        super().sample_initial()
        self.lastbest = None
        self.lastweight = 0.3

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        xbest_ind = np.argmin(self._fX)
        xbest = self._X[xbest_ind, :]
        if self.lastbest == None:
            self.lastbest = xbest_ind

        # if self._fX.shape[0] > 2*self.opt_prob.dim + 2 and self._fX.shape[0]%10==0:
            # self.bestlocal = checkSurrogate(self.opt_prob, self._X, self._fX, xbest)

        # print('xbest=%.3f, predict=%.3f' % (self.opt_prob.eval(xbest), self.surrogate.predict(xbest)[0]))

        # '''
        # print(weights[0])
        # print(self.lastweight)
        # print(self.sampling_radius)
        surrogate = self.surrogate
        if self.lastweight == 0.3:
            weight = 0.5
            self.lastweight = 0.5
        elif self.lastweight == 0.5:
        # if self.lastweight == 0.5:
            weight = 0.8
            self.lastweight = 0.8
            # if self._fX.shape[0] > 2*self.opt_prob.dim+2:
                # surrogate = setSurrogate(self.bestlocal, self.opt_prob, self._X, self._fX)
        elif self.lastweight == 0.8:
            weight = 0.95
            self.lastweight = 0.95
            # if self._fX.shape[0] > 2*self.opt_prob.dim+2:
            #     self.bestlocal = checkSurrogate(self.opt_prob, self._X, self._fX, xbest)
            #     surrogate = setSurrogate(self.bestlocal, self.opt_prob, self._X, self._fX)
        elif self.lastweight == 0.95:
            # weight = 1.0
            # self.lastweight = 1.0
        # elif self.lastweight == 1.0:
            weight = 0.3
            self.lastweight = 0.3
            # weight = 0.5
            # self.lastweight = 0.5
            if self.lastbest != xbest_ind:
                # self.sampling_radius = min([2.0 * self.sampling_radius, self.sampling_radius_max])
                dist1 = np.linalg.norm(self._X[self.lastbest, :] - xbest)
                dist2 = np.mean(scpspatial.distance.cdist(np.multiply(np.ones((1,self.opt_prob.dim)),xbest), self._X)[0])
                
                self.lastbest = xbest_ind
                # print('%.3f, %.3f, %.3f' % (dist1, dist2, dist1/dist2*100))
                # if dist1/dist2*100 > 0.1:
                    # self.sampling_radius = 0.2
                    # if self._fX.shape[0] > 2*self.opt_prob.dim+2:
                    #     self.bestlocal = checkSurrogate(self.opt_prob, self._X, self._fX, xbest)
                    #     surrogate = setSurrogate(self.bestlocal, self.opt_prob, self._X, self._fX)
                    # weight = 0.95
                    # self.lastweight = 0.95
                # else:
                #     weight = 1
                #     self.lastweight = 1
                    # self.sampling_radius = min([2.0 * self.sampling_radius, self.sampling_radius_max])
            # elif self.lastweight == 0.95:
            #     weight = 1
            #     self.lastweight = 1
            # else:
            #     weight = 0.3
            #     self.lastweight = 0.3
        # '''

        # print(weight)

        # if self.lastbest != xbest_ind:
        #     dist1 = np.linalg.norm(self._X[self.lastbest, :] - xbest)
        #     dist2 = np.mean(scpspatial.distance.cdist(np.multiply(np.ones((1,self.opt_prob.dim)),xbest), self._X)[0])
            
        #     self.lastbest = xbest_ind
        #     if dist1/dist2*100 > 0.1:
        #         # print('%.3f, %.3f, %.3f' % (dist1, dist2, dist1/dist2*100))
        #         weight = 0.98
        #         self.lastweight = 0.98

        # weight = self.get_weights(num_pts=num_pts)[0]
        new_point = candidate_cddycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weight=weight,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb)

        
        dist = np.min(scpspatial.distance.cdist(np.multiply(np.ones((1,self.opt_prob.dim)),new_point), np.vstack(self._X))[0])
        fval = self.surrogate.predict(new_point)[0]
        trueval = self.opt_prob.eval(new_point)
        # print('%.2f, %.3f, %.3f, %.3f, %.3f' % (weight, dist, fval, trueval, self._fX[xbest_ind]))

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_point)))

    def adjust_step(self):
        """Adjust the sampling radius sigma.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.
        """
        # Check if we succeeded at significant improvement
        fbest_new = min([record.value for record in self.record_queue])
        if fbest_new < self._fbest - 1e-3*math.fabs(self._fbest) or np.isinf(self._fbest):  # Improvement
            self._fbest = fbest_new
            self.status = max(1, self.status + 1)
            self.failcount = 0
        else:
            self.status = min(-1, self.status - 1)  # No improvement
            self.failcount += 1

        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            self.sampling_radius /= 2
            logger.info("Reducing sampling radius")
        if self.status >= self.succtol:
            self.ev_last = self.get_ev()  # Update the event id
            self.status = 0
            self.sampling_radius = min([2.0 * self.sampling_radius,
                                        self.sampling_radius_max])
            logger.info("Increasing sampling radius")

        # Check if we have converged
        if self.failcount >= self.maxfailtol or \
                self.sampling_radius <= self.sampling_radius_min:
            self.converged = True

        # Empty the queue
        self.record_queue = []

class SDSGDYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None,
                    sdsg_hybrid=False):
        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)
        self.sdsg_hybrid = sdsg_hybrid

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        # self.stat_numpts_gs.append(self.surrogate.num_pts)

        new_point = candidate_sdsgdycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb, sdsg_hybrid=self.sdsg_hybrid)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_point)))

class CKDYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None):

        # self.stat_fbest = []
        # self.stat_neval = 0
        self.stat_numpts_gs = []

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        # self.stat_numpts_gs.append(self.surrogate.num_pts)

        new_point = candidate_ckdycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_point)))


class SDSGCKDYCORSStrategy(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                 asynchronous=True, batch_size=None, extra_points=None,
                 extra_vals=None, use_restarts=True, weights=None, num_cand=None,
                 sdsg_hybrid=False):
        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                         exp_design=exp_design, surrogate=surrogate,
                         asynchronous=asynchronous, batch_size=batch_size,
                         extra_points=extra_points, extra_vals=extra_vals,
                         use_restarts=use_restarts, weights=weights,
                         num_cand=num_cand)

        self.sdsg_hybrid = sdsg_hybrid

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0 / self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0 / self.opt_prob.dim, 1.0]) * (
                               1.0 - (np.log(num_evals) / np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        # self.stat_numpts_gs.append(self.surrogate.num_pts)

        new_point = candidate_sdsgckdycors(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb, sdsg_hybrid=self.sdsg_hybrid)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_point)))

class SDSGCKDYCORSStrategy_std(DYCORSStrategy):
    def __init__(self, max_evals, opt_prob, exp_design, surrogate,
                    asynchronous=True, batch_size=None, extra_points=None,
                    extra_vals=None, use_restarts=True, weights=None, num_cand=None,
                    sdsg_hybrid=False):

        super().__init__(max_evals=max_evals, opt_prob=opt_prob,
                            exp_design=exp_design, surrogate=surrogate,
                            asynchronous=asynchronous, batch_size=batch_size,
                            extra_points=extra_points, extra_vals=extra_vals,
                            use_restarts=use_restarts,  weights=weights,
                            num_cand=num_cand)
        
        self.sdsg_hybrid = sdsg_hybrid

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0/self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min(
            [20.0/self.opt_prob.dim, 1.0]) * (
                1.0 - (np.log(num_evals)/np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        weights = self.get_weights(num_pts=num_pts)

        # self.stat_numpts_gs.append(self.surrogate.num_pts)

        new_point = candidate_sdsgckdycors_std(
            opt_prob=self.opt_prob, num_pts=num_pts, surrogate=self.surrogate,
            X=self._X, fX=self._fX, Xpend=self.Xpend, weights=weights,
            num_cand=self.num_cand, sampling_radius=self.sampling_radius,
            prob_perturb=prob_perturb, sdsg_hybrid=self.sdsg_hybrid)

        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_point)))