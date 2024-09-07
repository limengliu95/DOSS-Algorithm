from pySOT.auxiliary_problems import round_vars, weighted_distance_merit, unit_rescale, candidate_dycors
from pySOT.experimental_design import LatinHypercube, SymmetricLatinHypercube
from pySOT.surrogate import CubicKernel, TPSKernel, LinearKernel, \
    ConstantKernel, LinearTail, ConstantTail, SurrogateUnitBox
from new_surrogate import MultiquadricKernel
# from new_surrogate import RBFInterpolant
from pySOT.surrogate import RBFInterpolant
from rbfopt.rbfopt_utils import get_lhd_maximin_points, get_min_distance
from rbfopt.rbfopt_aux_problems import MetricSRSMObj, ga_mate, ga_mutate
from rbfopt.rbfopt_settings import RbfoptSettings
# from new_strategy import Accuracy

import scipy.stats as stats
import scipy.spatial as scpspatial
import math
import numpy as np
import random

class WeightedSum:
    def __init__(self, surrogate):
        self.surrogate = surrogate

    def updateFunction(self, X, w):
        self.X = X
        self.w = w

    def evaluate(self, x):
        # if x.shape[0] == 1:
            # x = np.multiply(np.ones((1,x.shape[1])),x)
        dists = scpspatial.distance.cdist(x, self.X)
        dmerit = np.amin(dists, axis=1, keepdims=True)
        return self.w*self.surrogate.predict(x) + (1-self.w)*dmerit

def getCdnKnowledge(X, fX, center):
    dim = X.shape[1]
    dists = scpspatial.distance.cdist(np.multiply(np.ones((1,dim)),center), np.vstack(X))[0]

    # print("Average distance: %.2f" % np.mean(np.sort(dists)[0:dim]))

    num_pts_in_region = 2*dim+2
    ind = np.argsort(dists)[0:min(num_pts_in_region,len(dists))]

    xx = np.copy(X[ind, :])
    knowledge = np.empty((dim,))
    for i in range(dim):
        knowledge[i] = np.unique(xx[:,i]).shape[0]
    # print(knowledge)

    # ave = np.average(knowledge)
    ave = np.median(knowledge)
    # ave = dim/2
    # ave = min(dim/2, np.max(knowledge))
    cdn = np.zeros((dim,))
    cdn[np.where(knowledge >= ave)[0]] = 1
    subset = np.arange(0, dim)
    subset = subset[np.where(knowledge <= ave)[0]]

    # cdn = np.ones((dim,))
    # print(cdn)
    return cdn, subset

def checkSurrogate(prob, X, fX, point):
    num_pts = fX.shape[0]
    dim = fX.shape[1]
    dists = scpspatial.distance.cdist(np.multiply(np.ones((1,dim)),point), np.vstack(X))[0]

    num_pts_in_region = 2*dim+2
    ind_in_region = np.argsort(dists)[0:min(num_pts_in_region,len(dists))]

    rbf = None
    errors = []
    for k in range(4):
        if k == 0:
            # print("LinearKernel ConstantTail")
            rbf = SurrogateUnitBox(
                RBFInterpolant(dim=prob.dim, kernel=LinearKernel(),
                tail=ConstantTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        elif k == 1:
            # print("MultiquadricKernel ConstantTail")
            rbf = SurrogateUnitBox(
                RBFInterpolant(dim=prob.dim, kernel=MultiquadricKernel(),
                tail=ConstantTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        elif k == 2:
            # print("CubicKernel LinearTail")
            rbf = SurrogateUnitBox(
                RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
                tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        else:
            # print("TPSKernel LinearTail")
            rbf = SurrogateUnitBox(
                RBFInterpolant(dim=prob.dim, kernel=TPSKernel(),
                tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        error = 0
        for i in range(num_pts_in_region):
            ind = list(range(num_pts))
            ind.remove(ind_in_region[i])
            rbf.add_points(X[ind, :], fX[ind])
            prediction = rbf.predict(X[ind_in_region[i], :])[0, 0]
            # print(prediction)
            # print(fX[ind_in_region[i]])
            error += abs(fX[ind_in_region[i], 0]-prediction)
            # print('%d, %.3f' % (i, abs(trueval[i]-prediction[i])))
            rbf.reset()

        errors.append(error/(num_pts_in_region))

    # for num in errors:
    # print(" ".join(("%.3f" % (num*100)) for num in errors))
    return np.argmin(errors)

def setSurrogate(bestlocal, prob, X, fX):
    if bestlocal == 0:
        # print("LinearKernel ConstantTail")
        rbf = SurrogateUnitBox(
            RBFInterpolant(dim=prob.dim, kernel=LinearKernel(),
            tail=ConstantTail(prob.dim)), lb=prob.lb, ub=prob.ub)
    elif bestlocal == 1:
        # print("MultiquadricKernel ConstantTail")
        rbf = SurrogateUnitBox(
            RBFInterpolant(dim=prob.dim, kernel=MultiquadricKernel(),
            tail=ConstantTail(prob.dim)), lb=prob.lb, ub=prob.ub)
    elif bestlocal == 2:
        # print("CubicKernel LinearTail")
        rbf = SurrogateUnitBox(
            RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
            tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
    else:
        # print("TPSKernel LinearTail")
        rbf = SurrogateUnitBox(
            RBFInterpolant(dim=prob.dim, kernel=TPSKernel(),
            tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)

    rbf.add_points(X, fX)
    return rbf


def ds_getMaxStepsize(x, p, lb, ub):
    # Lower bound
    a = None
    b = None
    ind = np.where(p > 0)[0]
    if ind.shape[0] != 0:
        a = min(np.divide(ub[ind]-x[ind], p[ind]))
    ind = np.where(p < 0)[0]
    if ind.shape[0] != 0:
        b = min(np.divide(lb[ind]-x[ind], p[ind]))
    # print(a, b)
    if a == None:
        return b
    elif b == None:
        return a
    else:
        return max(0, min(a, b))

def candidate_dsdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None, 
                     cand_scheme=None, batch_size=None):
    # Find best solution
    if xbest is None:
        ind_xbest = np.argmin(fX)
        xbest = np.copy(X[ind_xbest, :]).ravel()

    # getCdnKnowledge(X, fX, xbest)

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    # num_cand = 200*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # if cand_scheme == 0:
    # if weights[0] < 0.8 or fX.shape[0] < 350:
    # if weights[0] < 0.8 or fX.shape[0] > 200:
    # if weights[0] < 0.8:
    if weights[0] < 0.8:
    # if weights[0] < 0.95:
    # if cand_scheme == 0:
        num_cand1 = num_cand
        num_cand2 = 0
        # num_cand1 = int(num_cand*0.2)
        # num_cand2 = num_cand-num_cand1
        # num_cand1 = 0
        # num_cand2 = num_cand
    else:
        # num_cand = 1000*opt_prob.dim
        # num_cand1 = 0
        # num_cand2 = num_cand
        num_cand1 = int(num_cand*0.2)
        num_cand2 = num_cand-num_cand1
        # num_cand1 = num_cand
        # num_cand2 = 0

    # num_cand1 = int(num_cand*0.8)
    # num_cand2 = num_cand-num_cand1

    # DYCORS Scheme
    if num_cand1 != 0:
        # num_cand1_part1 = int(0.5*num_cand1)
        # num_cand1_part2 = int(0.5*num_cand1)

        if cand_scheme == 1:
            num_cand1_part1 = int(num_cand1)
        #     num_cand1_part2 = int(0)
        # elif cand_scheme == 2:
        #     num_cand1_part1 = int(0)
        #     num_cand1_part2 = int(num_cand1)
        # '''
            if weights[0] == 0.03:
                cand1 = opt_prob.lb + (opt_prob.ub-opt_prob.lb)*np.random.rand(num_cand1_part1, opt_prob.dim)
            else:
                # Compute scale factors for each dimension and make sure they
                # are correct for integer variables (at least 1)
                scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
                ind = np.intersect1d(opt_prob.int_var, subset)
                if len(ind) > 0:
                    scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

                if weights[0] < 0.8:
                    cdn, subset = getCdnKnowledge(X, fX, xbest)
                    # print(subset)

                # Generate candidate points
                if len(subset) == 1:  # Fix when nlen is 1
                    ar = np.ones((num_cand1_part1, 1))
                else:
                    ar = (np.random.rand(num_cand1_part1, len(subset)) < prob_perturb)
                    ind = np.where(np.sum(ar, axis=1) == 0)[0]
                    ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

                # cand1 = np.multiply(np.ones((num_cand1, opt_prob.dim)), xbest)
                cand1 = np.multiply(np.ones((num_cand1_part1, opt_prob.dim)), xbest)
                for j in range(len(subset)):
                    i = subset[j]
                    lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
                    ind = np.where(ar[:, j] == 1)[0]
                    if weights[0] < 0.05:
                        cand1[ind, i] = lower + (upper-lower)*np.random.rand(len(ind))
                    else:
                        cand1[ind, i] = stats.truncnorm.rvs(
                            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
                            loc=xbest[i], scale=sigma, size=len(ind))

        # '''

        # '''
        elif cand_scheme == 2:
            start_radius = 0.1
            start_lb = np.maximum(xbest-start_radius*(opt_prob.ub-opt_prob.lb), opt_prob.lb)
            start_ub = np.minimum(xbest+start_radius*(opt_prob.ub-opt_prob.lb), opt_prob.ub)
            # cand1 = np.vstack((cand1, np.random.rand(num_cand1_part2, opt_prob.dim)*(start_ub-start_lb)+start_lb))
            # cand1 = np.vstack((cand1, get_lhd_maximin_points(start_lb, start_ub, opt_prob.int_var, num_cand1_part2, num_trials=1)))
            # cand1 = np.random.rand(int(num_cand1/2), opt_prob.dim)*(opt_prob.ub-opt_prob.lb)+opt_prob.lb
            cand1 = np.random.rand(num_cand1, opt_prob.dim)*(start_ub-start_lb)+start_lb
        # '''
        

    # DS Scheme
    if num_cand2 != 0:
        # max_step = max(0,1/max(max(np.divide(dfx,xbest-opt_prob.lb+1e-6)), max(np.divide(dfx,xbest-opt_prob.ub+1e-6)))-1e-6)
        # print(max_step)
        # np.random.rand(num_cand)*max_step
        '''
        # 1st Method: use steepest descent direction only
        step_size = np.random.rand(num_cand2)*max_step
        # scale = 0.2*max_step
        # step_size = abs(stats.truncnorm.rvs(
        #         a=-max_step/scale, b=max_step/scale,
        #         loc=0, scale=scale, size=num_cand2))
        cand2 = np.multiply(np.ones((num_cand2, opt_prob.dim)), xbest)
        step = np.multiply(np.ones((num_cand2, opt_prob.dim)), dfx)
        step = np.transpose(np.multiply(np.transpose(step), step_size))
        cand2 = cand2 - step
        # # Sparsity
        # if len(subset) == 1:  # Fix when nlen is 1
        #     ar = np.ones((num_cand2, 1))
        # else:
        #     ar = (np.random.rand(num_cand2, len(subset)) < prob_perturb)
        #     ind = np.where(np.sum(ar, axis=1) == 0)[0]
        #     ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1
        # for i in subset:
        #     lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        #     ind = np.where(ar[:, i] == 1)[0]
        #     cand2[ind, subset[i]] = stats.truncnorm.rvs(
        #         a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
        #         loc=xbest[i], scale=sigma, size=len(ind))
        '''
        # 2nd Method: use sparse descent direction

        cand2 = np.empty((num_cand2, opt_prob.dim))
        num_start_point = 1
        if num_start_point > 1:
            # start_radius = 0.2
            # start_lb = np.maximum(xbest-start_radius*(opt_prob.ub-opt_prob.lb), opt_prob.lb)
            # start_ub = np.minimum(xbest+start_radius*(opt_prob.ub-opt_prob.lb), opt_prob.ub)
            # start_points = get_lhd_maximin_points(start_lb, start_ub, opt_prob.int_var, num_start_point-1, num_trials=10)
            start_points = X[fX.shape[0]-num_start_point+1:fX.shape[0], :]
        num_cand_per_start = int(num_cand2/num_start_point)

        # Sparsity
        # if weights[0] >= 0.8:
        # cdn, subset = getCdnKnowledge(X, fX, xbest)
        # else:
        # subset = np.arange(0, opt_prob.dim)
        # ar = np.zeros((num_cand_per_start, opt_prob.dim))
        # ar[:, subset] = (np.random.rand(num_cand_per_start, len(subset)) < prob_perturb)
        # ind = np.where(np.sum(ar, axis=1) == 0)[0]
        # ar[ind, subset[np.random.randint(0, len(subset)-1, size=len(ind))]] = 1

        # print(sampling_radius)
        for j in range(num_start_point):
            if j == 0:
                # ind_point = ind_xbest
                point = xbest
            else:
                # ind_point = fX.shape[0] - 1 - ind_start_point[j-1]
                # ind_point = ind_start_point[j-1]
                point = start_points[j-1]

            cand2[j*num_cand_per_start:(j+1)*num_cand_per_start] = dsCand(num_cand_per_start, prob_perturb, opt_prob, surrogate, point)
            '''
            # point = X[ind_point, :]

            # getCdnKnowledge(X, fX, point)

            dfx = surrogate.predict_deriv(point)[0]
            direction = np.random.random(size=(10*num_cand_per_start, opt_prob.dim))-0.5
            # direction = np.multiply(np.ones((10*num_cand_per_start, opt_prob.dim)), -dfx)
            # Sparsity
            direction = np.multiply(direction, ar)

            ndirect = np.linalg.norm(direction, axis=-1)
            direction = direction / ndirect[..., np.newaxis]

            quality = np.dot(direction, dfx)
            quality_ind = np.argsort(quality) # Ascend
            cand_for_point = np.multiply(np.ones((num_cand_per_start, opt_prob.dim)), point)
            num_direction = int(min(opt_prob.dim, num_cand_per_start))
            # num_direction = 40
            num_cand_per_direction = int(num_cand_per_start/num_direction)
            for i in range(num_direction):
                # Descend Direction
                if i != 0:
                    p = direction[quality_ind[i-1], :]
                else:
                    # p = np.multiply(-dfx, ar[0, :])
                    # p = np.multiply(-dfx, ar)
                    p = -dfx
                # p = np.multiply(-dfx, ar[i, :])
                # p = np.multiply(direction[quality_ind[i], :], ar[i, :])
                # p = direction[quality_ind[i], :]
                # p = -dfx
                
                if(np.linalg.norm(p) == 0):
                    max_step = 0
                else:
                    max_step = ds_getMaxStepsize(point, p, opt_prob.lb, opt_prob.ub)

                # print("%.8f" % max_step)
                # Uniform Step 
                # step_size = np.random.rand(int(num_cand_per_direction/2))*max_step
                # Normal Step
                if max_step < 1e-6:
                    # scale = 1
                    step_size = np.zeros((int(num_cand_per_direction/2),))
                else:
                    scale = max(sampling_radius*max_step, 1.0)
                    # scale = 1
                    step_size = abs(stats.truncnorm.rvs(
                        a=-max_step/scale, b=max_step/scale,
                        loc=0, scale=scale, size=int(num_cand_per_direction/2)))
                # Mixed Step 
                # scale = 0.2*max_step
                # step_size = np.hstack((np.random.rand(int(num_cand_per_direction/4))*max_step, abs(stats.truncnorm.rvs(
                #     a=-max_step/scale, b=max_step/scale,
                #     loc=0, scale=scale, size=int(num_cand_per_direction/4)))))

                step = np.multiply(np.ones((int(num_cand_per_direction/2), opt_prob.dim)), p)
                step = np.transpose(np.multiply(np.transpose(step), step_size))
                cand_for_point[i*num_cand_per_direction:int((i+0.5)*num_cand_per_direction), :] += step

                # Ascend Direction
                if i != 0:
                    p = direction[quality_ind[num_direction-i], :]
                else:
                    # p = np.multiply(dfx, ar[0, :])
                    # p = np.multiply(dfx, ar)
                    p = dfx
                # p = np.multiply(dfx, ar[i, :])
                # p = np.multiply(direction[quality_ind[num_direction-i], :], ar[i, :])
                # p = -direction[quality_ind[num_direction-i], :]
                # p = dfx

                if(np.linalg.norm(p) == 0):
                    max_step = 0
                else:
                    max_step = ds_getMaxStepsize(point, p, opt_prob.lb, opt_prob.ub)

                # print("%.8f" % max_step)
                # Uniform Step
                # step_size = np.random.rand(int(num_cand_per_direction/2))*max_step
                # Normal Step
                if max_step < 1e-6:
                    # scale = 1
                    step_size = np.zeros((int(num_cand_per_direction/2),))
                else:
                    scale = max(sampling_radius*max_step, 1.0)
                    # scale = 1
                    step_size = abs(stats.truncnorm.rvs(
                        a=-max_step/scale, b=max_step/scale,
                        loc=0, scale=scale, size=int(num_cand_per_direction/2)))
                # Mixed Step
                # scale = 0.2*max_step
                # step_size = np.hstack((np.random.rand(int(num_cand_per_direction/4))*max_step, abs(stats.truncnorm.rvs(
                #     a=-max_step/scale, b=max_step/scale,
                #     loc=0, scale=scale, size=int(num_cand_per_direction/4)))))

                step = np.multiply(np.ones((int(num_cand_per_direction/2), opt_prob.dim)), p)
                step = np.transpose(np.multiply(np.transpose(step), step_size))
                cand_for_point[int((i+0.5)*num_cand_per_direction):(i+1)*num_cand_per_direction, :] += step
            cand2[j*num_cand_per_start:(j+1)*num_cand_per_start] = cand_for_point
            '''

    if num_cand1 != 0 and num_cand2 != 0:
        cand = np.vstack((cand1, cand2))
    elif num_cand1 != 0:
        cand = cand1
    else:
        cand = cand2
    candsepa = num_cand1

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # Make selections
    new_points, evalfrom_dycors = ds_weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX,
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights, candsepa=candsepa)

    # print("True value of selected: %.3f" % opt_prob.eval(new_points[0]))
    # val_cand = np.empty((cand.shape[0],))
    # sur_cand = surrogate.predict(cand)
    # ind_cand = np.argsort(sur_cand[:,0])
    # for i in range(cand.shape[0]):
    #     val_cand[i] = opt_prob.eval(cand[i, :])
    # print("First five sur: ")
    # print(sur_cand[ind_cand[0:10], 0])
    # print("Their true value: ")
    # print(val_cand[ind_cand[0:10]])

    # ws = WeightedSum(surrogate)
    # ws.updateFunction(np.vstack((X, Xpend)), weights[0])
    # print(ws.evaluate(np.multiply(np.ones((1,opt_prob.dim)),np.random.randn(opt_prob.dim))))
    # metric = MetricSRSMObj(RbfoptSettings(), opt_prob.dim, X.shape[0], X, surrogate.model.c[surrogate.model.ntail:surrogate.model.ntail+surrogate.model.num_pts], surrogate.model.c[:surrogate.model.ntail], 1-weights[0])
    # new_points = ga_optimize(RbfoptSettings(), opt_prob.dim, opt_prob.lb, opt_prob.ub, np.array([]), None, metric.evaluate)

    return new_points, evalfrom_dycors

def uniformCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point):
    # '''
    # radius = 0.1
    # start_lb = np.maximum(point-radius*(opt_prob.ub-opt_prob.lb), opt_prob.lb)
    # start_ub = np.minimum(point+radius*(opt_prob.ub-opt_prob.lb), opt_prob.ub)

    # print(len(subset))
    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), point)
    cand[:,subset] = np.random.rand(num_cand, len(subset))*(opt_prob.ub[subset]-opt_prob.lb[subset])+opt_prob.lb[subset]
    # cand = np.random.rand(num_cand, opt_prob.dim)*(start_ub-start_lb)+start_lb
    # cand = get_lhd_maximin_points(start_lb, start_ub, opt_prob.int_var, num_cand, num_trials=1)
    # '''
    '''
    # Generate candidate points
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), point)

    for j in range(len(subset)):
        i = subset[j]
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, j] == 1)[0]

        # cand[ind, i] = stats.truncnorm.rvs(
        #     a=(lower - point[i]) / sigma, b=(upper - point[i]) / sigma,
        #     loc=point[i], scale=sigma, size=len(ind))
        
        cand[ind, i] = lower + (upper-lower)*np.random.rand(len(ind))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)
    '''

    return cand

def normalCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point):
    # Generate candidate points
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), point)

    for j in range(len(subset)):
        i = subset[j]
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, j] == 1)[0]

        cand[ind, i] = stats.truncnorm.rvs(
            a=(lower - point[i]) / sigma, b=(upper - point[i]) / sigma,
            loc=point[i], scale=sigma, size=len(ind))

        # cand[ind, i] = lower + (upper-lower)*np.random.rand(len(ind))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    return cand

def dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset, sampling_radius, point):
    # sampling_radius = 0.02
    # sampling_radius = 0.2

    dfx = surrogate.predict_deriv(point)[0]
    
    ar = np.zeros((num_cand, opt_prob.dim))
    ar[:, subset] = (np.random.rand(num_cand, len(subset)) < prob_perturb)
    ind = np.where(np.sum(ar, axis=1) == 0)[0]
    if len(subset) > 1:
        ar[ind, subset[np.random.randint(0, len(subset)-1, size=len(ind))]] = 1
    else:
        ar[ind, subset[0]] = 1

    # ar[:,subset] = 1

    # direction = np.random.random(size=(num_cand, opt_prob.dim))-0.5
    direction = np.multiply(np.ones((num_cand, opt_prob.dim)), -dfx)
    # Sparsity
    direction = np.multiply(direction, ar)

    ndirect = np.linalg.norm(direction, axis=-1)
    direction = direction / ndirect[..., np.newaxis]

    # quality = np.dot(direction, dfx)
    # quality_ind = np.argsort(quality) # Ascend

    # quality = np.dot(direction, dfx)
    # quality_ind = np.argsort(-np.abs(quality))
    quality_ind = np.arange(0, num_cand)
    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), point)
    num_direction = int(min(opt_prob.dim, num_cand))
    # num_direction = num_cand
    # num_direction = 40
    num_cand_per_direction = int(num_cand/num_direction)
    for i in range(num_direction):
        ''' Descend Direction '''
        p = direction[quality_ind[i-1], :]
        # if(quality[quality_ind[i-1]] > 0):
            # p *= -1
        
        if(np.linalg.norm(p) == 0):
            step_lb = 0
            step_ub = 0
            # max_step = 0
        else:
            # max_step = ds_getMaxStepsize(point, p, opt_prob.lb, opt_prob.ub)
            step_lb = -ds_getMaxStepsize(point, -p, opt_prob.lb, opt_prob.ub)
            step_ub = ds_getMaxStepsize(point, p, opt_prob.lb, opt_prob.ub)

        # print("%.8f" % max_step)
        ''' Uniform Step '''
        # step_size = np.random.rand(int(num_cand_per_direction/2))*max_step
        # step_size = step_lb + np.random.rand(int(num_cand_per_direction))*(step_ub-step_lb)
        ''' Normal Step '''
        
        # if max_step < 1e-6:
        if step_ub - step_lb < 1e-6:
            # scale = 1
            step_size = np.zeros((int(num_cand_per_direction),))
        else:
            # scale = max(sampling_radius*max_step, 1.0)
            scale = max(sampling_radius*(step_ub-step_lb), 1.0)

            # step_size = abs(stats.truncnorm.rvs(
            #     a=-max_step/scale, b=max_step/scale,
            #     loc=0, scale=scale, size=int(num_cand_per_direction)))
            step_size = stats.truncnorm.rvs(
                a=step_lb/scale, b=step_ub/scale,
                loc=0, scale=scale, size=int(num_cand_per_direction))

        step = np.multiply(np.ones((int(num_cand_per_direction), opt_prob.dim)), p)
        step = np.transpose(np.multiply(np.transpose(step), step_size))
        cand[i*num_cand_per_direction:int((i+1)*num_cand_per_direction), :] += step

    return cand

def cdCand(num_cand, subset, surrogate, opt_prob, X, Xpend, fX, weight, scalefactors, point):
    num_cand_per_dim = int(num_cand / opt_prob.dim)
    # num_cand_per_dim = int(num_cand / opt_prob.dim * 25)
    cand_budget = num_cand
    # xbest_val = surrogate.predict(point)[0]
    # print('xbest=%.3f, predict=%.3f' % (opt_prob.eval(xbest), surrogate.predict(xbest)[0]))
    for i in range(opt_prob.dim):
        if fX.shape[0] < 2*(opt_prob.dim+1):
            dim_perturb = np.random.permutation(opt_prob.dim).tolist()
            dim_perturb = np.array([x for x in dim_perturb if x in subset])
        else:
            dfx = surrogate.predict_deriv(point)[0]
            # dim_perturb = np.random.choice(range(opt_prob.dim), p=abs(dfx)/np.sum(abs(dfx)), size=num_perturb, replace=False)
            # print(abs(dfx)/np.sum(abs(dfx)))
            # num_perturb = np.where(abs(dfx) >= np.mean(abs(dfx)))[0].shape[0]
            # dim_perturb = np.where(abs(dfx) >= np.mean(abs(dfx)))[0]
            # dim_perturb = np.random.choice(range(opt_prob.dim), p=abs(dfx)/np.sum(abs(dfx)), size=num_perturb, replace=False)
            dim_perturb = np.argsort(-abs(dfx)).tolist()
            # print(len(dim_perturb))
            dim_perturb = np.array([x for x in dim_perturb if x in subset])
            # print(dim_perturb.shape[0])

            # dim_perturb = np.random.choice(range(opt_prob.dim), size=num_perturb, replace=False)
            # dim_perturb = [i]
            # dim_perturb = np.random.permutation(opt_prob.dim)
            # cdn, subset = getCdnKnowledge(X, fX, xbest)
            # dim_perturb = subset
            # num_perturb = dim_perturb.shape[0]
            # num_perturb = int(opt_prob.dim/5)

        num_perturb = dim_perturb.shape[0]
        for j in range(num_perturb):
            if cand_budget <= 0:
                return point
            else:
                cand_budget -= num_cand_per_dim

            cand = np.multiply(np.ones((num_cand_per_dim, opt_prob.dim)), point)
            lower, upper, sigma = opt_prob.lb[dim_perturb[j]], opt_prob.ub[dim_perturb[j]], scalefactors[dim_perturb[j]]
            # if weight >= 0.08:
            cand[:, dim_perturb[j]] = stats.truncnorm.rvs(
                a=(lower - point[dim_perturb[j]]) / sigma, b=(upper - point[dim_perturb[j]]) / sigma,
                loc=point[dim_perturb[j]], scale=sigma, size=num_cand_per_dim)
            # else:
                # cand[:, dim_perturb[j]] = lower + (upper-lower)*np.random.rand(num_cand_per_dim)

            # if weight >= 0.95:
            cand = np.vstack((cand, point))

            # print(cand.shape[0])
            # fvals = surrogate.predict(cand)[:,0]
            # ind = np.where(fvals/(xbest_val+1e-6) < 1.1)[0]
            # if ind.shape[0] == 0:
                # print('error')
                # return point
            # cand = cand[ind, :]
            # print(cand.shape[0])

            fvals = surrogate.predict(cand)[:,0]
            ufvals = unit_rescale(fvals)
            dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
            dmerit = np.amin(dists, axis=1, keepdims=True)[:,0]
            rank = weight*ufvals + (1.0-weight)*(1.0 - unit_rescale(np.copy(dmerit)))

            # print(np.argmin(rank))
            # print('fval: %.2f' % fvals[np.argmin(rank)])
            cand_best = np.argmin(rank)
            # cand_best = np.argmin(fvals)
            # if cand_best == cand.shape[0]-1:
                # print(num_perturb, cand.shape[0])
                # num_perturb = min(opt_prob.dim, num_perturb+1)
            # else:
            if cand_best != cand.shape[0]-1:
                # num_perturb = 1
                # num_perturb = max(1, num_perturb-1)
                # print('Success! fval=%.3f dist=%.3f' % (fvals[cand_best], dmerit[cand_best]))
                point = cand[cand_best, :]
                break
            # print('Failed! fval=%.3f dist=%.3f' % (fvals[cand_best], dmerit[cand_best]))

    # cand = np.vstack((point1, point))
    # dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    # dmerit = np.amin(dists, axis=1, keepdims=True)[:,0]
    # fvals = surrogate.predict(cand)[:,0]
    # ufvals = unit_rescale(fvals)
    # rank = weight*ufvals + (1.0-weight)*(1.0 - unit_rescale(np.copy(dmerit)))
    # point = cand[np.argmin(rank), :]
    # if np.argmin(rank) == 0:
    #     print('dycors')
    # else:
    #     print('cd')

    # print('True=%.3f' % opt_prob.eval(point))
    # print('\n')
    return point

def ga_newpoint(opt_prob, X, Xpend, surrogate, weight, point=None, prob_perturb=None, scalefactors=None):
    population_size = 400 + 20 * opt_prob.dim//5
    mutation_rate = 0.1
    num_surviving = population_size//4
    num_new = population_size - 2*num_surviving - 1

    is_integer = np.zeros(opt_prob.dim, dtype=bool)
    if (len(opt_prob.int_var)):
        is_integer[opt_prob.int_var] = True

    if point is not None:
        radius = 0.1
        start_lb = np.maximum(point-radius*(opt_prob.ub-opt_prob.lb), opt_prob.lb)
        start_ub = np.minimum(point+radius*(opt_prob.ub-opt_prob.lb), opt_prob.ub)

    # Compute initial population
    if point is not None:
        # population = np.random.rand(population_size, opt_prob.dim)*(start_ub-start_lb)+start_lb
        population = normalCand(population_size, np.arange(0, opt_prob.dim), prob_perturb, opt_prob, scalefactors, point)
        # population = dsCand(population_size, prob_perturb, opt_prob, surrogate, point)
        # population = uniformCand(population_size, opt_prob, radius, point)
        # cand = get_lhd_maximin_points(start_lb, start_ub, opt_prob.int_var, num_cand, num_trials=1)
    else:
        population = (np.random.rand(population_size, opt_prob.dim) * (opt_prob.ub - opt_prob.lb) + opt_prob.lb)

    # assert(isinstance(opt_prob.lb, np.ndarray))
    # assert(isinstance(population, np.ndarray))

    # Round integer vars
    if (len(opt_prob.int_var)):
        population[:, opt_prob.int_var] = np.around(population[:, opt_prob.int_var])

    for gen in range(20):
    # for gen in range(5):
        # Mutation rate and maximum perturbed coordinates for this
        # generation of individuals
        curr_mutation_rate = (mutation_rate * (20 - gen) / 20)
        # Compute fitness score to determine remaining individuals
        dists = scpspatial.distance.cdist(population, np.vstack((X, Xpend)))
        dmerit = np.amin(dists, axis=1, keepdims=True)[:,0]

        fvals = surrogate.predict(population)[:,0]
        fvals = unit_rescale(fvals)
        fitness_val = weight*fvals + (1.0-weight)*(1.0 - unit_rescale(np.copy(dmerit)))

        rank = np.argsort(fitness_val)
        best_individuals = population[rank[:num_surviving]]
        # Crossover: select how mating is done, then create offspring
        father = np.random.permutation(best_individuals)
        mother = np.random.permutation(best_individuals)
        offspring = ga_mate(father, mother)
        # New individuals
        if point is not None:
            # new_individuals = np.random.rand(num_new, opt_prob.dim)*(start_ub-start_lb)+start_lb
            new_individuals = normalCand(num_new, np.arange(0, opt_prob.dim), prob_perturb, opt_prob, scalefactors, best_individuals[0, :])
            # new_individuals = dsCand(num_new, prob_perturb, opt_prob, surrogate, best_individuals[0, :])
            # new_individuals = uniformCand(num_new, opt_prob, radius, point)
        else:
            new_individuals = (np.random.rand(num_new, opt_prob.dim) * (opt_prob.ub - opt_prob.lb) + opt_prob.lb)
        # new_individuals = (np.random.rand(num_new, opt_prob.dim) * (opt_prob.ub - opt_prob.lb) + opt_prob.lb)
        # Round integer vars
        if (len(opt_prob.int_var)):
            new_individuals[:, opt_prob.int_var] = np.around(new_individuals[:, opt_prob.int_var])

        # Compute perturbation.
        max_size_pert = min(opt_prob.dim, max(2, int(opt_prob.dim * curr_mutation_rate)))
        # Make a copy of best individual, and mutate it
        best_mutated = best_individuals[0, :].copy()

        # Randomly mutate some of the coordinates. First determine how
        # many are mutated, then pick them randomly.
        size_pert = np.random.randint(max_size_pert)
        perturbed = np.random.choice(np.arange(opt_prob.dim), size_pert, replace=False)
        new = (opt_prob.lb[perturbed] + np.random.rand(size_pert) * 
            (opt_prob.ub[perturbed] - opt_prob.lb[perturbed]))
        new[is_integer[perturbed]] = np.around(new[is_integer[perturbed]])
        best_mutated[perturbed] = new

        # Mutate surviving (except best) if necessary
        for point in best_individuals[1:]:
            if (np.random.uniform() < curr_mutation_rate):
                ga_mutate(opt_prob.dim, opt_prob.lb, opt_prob.ub, is_integer, point, max_size_pert)
        # Generate new population
        population = np.vstack((best_individuals, offspring,
                                new_individuals, best_mutated))
    # dump.close()
    # Determine ranking of last generation.
    # Compute fitness score to determine remaining individuals
    dists = scpspatial.distance.cdist(population, np.vstack((X, Xpend)))
    dmerit = np.amin(dists, axis=1, keepdims=True)

    fvals = surrogate.predict(population)
    fvals = unit_rescale(fvals)
    fitness_val = weight*fvals + (1.0-weight)*(1.0 - unit_rescale(np.copy(dmerit)))
    rank = np.argsort(fitness_val)
    # Return best individual
    return population[rank[0]]


def candidate_ddsdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    weight = weights[0]

    # return ga_newpoint(opt_prob, X, Xpend, surrogate, weight, point=xbest, prob_perturb=prob_perturb, scalefactors=scalefactors)

    # dists = scpspatial.distance.cdist(np.multiply(np.ones((1,opt_prob.dim)),xbest), np.vstack(X))[0]
    # print("Average distance: %.2f, %.2f" % (np.mean(np.sort(dists)[0:opt_prob.dim]), np.mean(opt_prob.ub-opt_prob.lb)))

    # 1: normal, 2: uniform, 3: direction search
    # if weight < 0.8:
    #     cand_routine = [1]
    #     num_newcand = [100*opt_prob.dim]
    #     num_newcenter = [1]
    # else:
    # cand_routine = [1]*100
    # num_newcand = [1]*100
    # num_newcenter = [1]*100

    # if fX.shape[0] < 3*opt_prob.dim:
    # if weight >= 0.95:
    #     cand_routine = [3]*10
    #     num_newcand = [10*opt_prob.dim]*10
    #     num_newcenter = [1]*10
    # else:
    #     cand_routine = [1]*10
    #     num_newcand = [10*opt_prob.dim]*10
    #     num_newcenter = [1]*10
    #     cdn, subset = getCdnKnowledge(X, fX, xbest)

    # cand_routine = [3, 1]
    # num_newcand = [50*opt_prob.dim]*2
    # num_newcenter = [1]*2

    # cand_routine = [2, 3, 1]
    # num_newcand = [30*opt_prob.dim]*3
    # num_newcenter = [1]*3

    if weight == 0.3:
        cand_routine = [2]*3
        num_newcand = [30*opt_prob.dim]*3
        num_newcenter = [1]*3
        cdn, subset = getCdnKnowledge(X, fX, xbest)
    elif weight == 0.5:
        cand_routine = [1]*3
        num_newcand = [30*opt_prob.dim]*3
        num_newcenter = [1]*3
        cdn, subset = getCdnKnowledge(X, fX, xbest)
    else:
        cand_routine = [3]
        num_newcand = [100*opt_prob.dim]
        num_newcenter = [1]
    


    # cand_routine = [2, 3, 1]
    # num_newcand = [20*opt_prob.dim, 10*opt_prob.dim, 40*opt_prob.dim]
    # num_newcenter = [4, 1, 1]
    center = np.empty((1, opt_prob.dim))
    center[0, :] = xbest

    for ind_newcenter in range(len(num_newcenter)):
        routine = cand_routine[ind_newcenter]
        num_point = num_newcenter[ind_newcenter]
        num_cand = num_newcand[ind_newcenter]
        cands = np.empty((0, opt_prob.dim))
        for ind_point in range(center.shape[0]):
            # print(ind_point)
            point = center[ind_point, :]

            # dists = scpspatial.distance.cdist(np.multiply(np.ones((1,opt_prob.dim)), point), np.vstack(X))[0]
            # avedist = np.mean(np.sort(dists)[0:opt_prob.dim])
            # print("Average distance: %.2f, %.2f" % (avedist, np.mean(opt_prob.ub-opt_prob.lb)))
            # if avedist > 0.5*np.mean(opt_prob.ub-opt_prob.lb):
            #     routine = 3
            # elif avedist > 0.1*np.mean(opt_prob.ub-opt_prob.lb):
            #     routine = 2
            # else:
            #     routine = 1
            
            if routine == 1:
                cand = normalCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point)
            elif routine == 2:
                cand = uniformCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point)
            elif routine == 3:
                cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, point)

            cands = np.vstack((cands, cand))

        if ind_newcenter != 0:
            cands = np.vstack((cands, center))
        # Make selections
        center = weighted_distance_merit(
            num_pts=num_point, surrogate=surrogate, X=X, fX=fX,
            Xpend=Xpend, cand=cands, dtol=dtol, weights=weight*np.ones((num_point,)))

    return center

def candidate_gadycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    weight = weights[0]

    return ga_newpoint(opt_prob, X, Xpend, surrogate, weight, point=xbest, prob_perturb=prob_perturb, scalefactors=scalefactors)
    # return ga_newpoint(opt_prob, X, Xpend, surrogate, weight)

def candidate_cddycors(num_pts, opt_prob, surrogate, X, fX, weight,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
        # num_cand = 1000*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    
    point = xbest
        
    # subset = np.arange(0, opt_prob.dim)
    # dfx = surrogate.predict_deriv(point)[0]
    # subset = np.where(abs(dfx) >= np.mean(abs(dfx)))[0].tolist()

    dim = X.shape[1]
    dists = scpspatial.distance.cdist(np.multiply(np.ones((1,dim)),point), np.vstack(X))[0]

    num_pts_in_region = 2*dim+2
    # num_pts_in_region = dim+1
    # num_pts_in_region = len(dists)
    ind = np.argsort(dists)[0:min(num_pts_in_region,len(dists))]

    xx = np.copy(X[ind, :])
    knowledge = np.empty((dim,))
    for i in range(dim):
        knowledge[i] = np.unique(xx[:,i]).shape[0]
    ave = np.median(knowledge)
    subset = np.arange(0, dim)
    subset1 = subset[np.where(knowledge >= ave)[0]]
    subset2 = subset[np.where(knowledge <= ave)[0]]

    # num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
    # budget = self.max_evals - self.num_exp
    # prob_perturb = min([20.0/self.opt_prob.dim, 1.0]) * (1.0 - (np.log(num_evals)/np.log(budget)))
    # prob_perturb = max(prob_perturb, min_prob)
    # print(prob_perturb)

    # if weight == 0.8 or weight == 0.95:
    if weight == 0.8:
        # cand = normalCand(int(0.2*num_cand), subset2, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.vstack((cand, dsCand(int(0.8*num_cand), prob_perturb, opt_prob, surrogate, subset2, point)))

        # cand = normalCand(num_cand, subset2, prob_perturb, opt_prob, scalefactors, point)
        cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset2, sampling_radius, point)
        # cand = normalCand(int(0.2*num_cand), subset2, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.vstack((cand, dsCand(int(0.8*num_cand), prob_perturb, opt_prob, surrogate, subset2, point)))
        # cand = np.reshape(cdCand(num_cand, subset2, surrogate, opt_prob, X, Xpend, fX, weight, scalefactors, point), (1, opt_prob.dim))

        # cand = normalCand(int((1.0-prob_perturb)*num_cand), subset, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.vstack((cand, dsCand(int(prob_perturb*num_cand), prob_perturb, opt_prob, surrogate, subset1, point)))
    elif weight == 0.95:
        # print(cand.shape)
        # print(cdCand(int(0.4*num_cand), surrogate, opt_prob, X, Xpend, fX, weight, scalefactors, point).shape)

        # cand = normalCand(int(0.2*num_cand), subset, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.vstack((cand, dsCand(int(0.8*num_cand), prob_perturb, opt_prob, surrogate, subset, point)))

        # cand = uniformCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point)
        # cand = normalCand(num_cand, subset1, prob_perturb, opt_prob, scalefactors, point)
        # cand = normalCand(int(0.5*num_cand), subset1, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.reshape(cdCand(num_cand, subset1, surrogate, opt_prob, X, Xpend, fX, weight, scalefactors, point), (1, opt_prob.dim))
        cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset1, sampling_radius, point)

        # cand = normalCand(int((1.0-prob_perturb)*num_cand), subset, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.vstack((cand, dsCand(int(prob_perturb*num_cand), prob_perturb, opt_prob, surrogate, subset1, point)))

        # cand = np.vstack((cand, np.reshape(cdCand(int(0.5*num_cand), subset, surrogate, opt_prob, X, Xpend, fX, weight, scalefactors, point), (1, opt_prob.dim))))
    # elif weight == 1.0:
    #     cand = uniformCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.reshape(cdCand(num_cand, subset2, surrogate, opt_prob, X, Xpend, fX, weight, scalefactors, point), (1, opt_prob.dim))
        # cand = normalCand(num_cand, subset1, prob_perturb, opt_prob, scalefactors, point)
        # cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset2, point)
    else:
        # cdn, subset = getCdnKnowledge(X, fX, xbest)
        # cand = normalCand(num_cand, subset2, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.reshape(cdCand(num_cand, subset2, surrogate, opt_prob, X, Xpend, fX, weight, scalefactors, point), (1, opt_prob.dim))
        # cand = uniformCand(num_cand, subset2, prob_perturb, opt_prob, scalefactors, point)
        # cand = normalCand(int(0.2*num_cand), subset2, prob_perturb, opt_prob, scalefactors, point)
        # cand = np.vstack((cand, dsCand(int(0.8*num_cand), prob_perturb, opt_prob, surrogate, subset2, point)))
        cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset2, sampling_radius, point)
    # else:
    #     subset = np.arange(0, opt_prob.dim)
    #     cand = uniformCand(int(num_cand), subset, prob_perturb, opt_prob, scalefactors, point)

    dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    dmerit = np.amin(dists, axis=1, keepdims=True)[:,0]

    fvals = surrogate.predict(cand)[:,0]
    ufvals = unit_rescale(fvals)
    rank = weight*ufvals + (1.0-weight)*(1.0 - unit_rescale(np.copy(dmerit)))

    bestcand = np.argmin(rank)
    # if weight >= 0.8:
    #     if weight == 0.95 and bestcand/opt_prob.dim == 60:
    #         print("CD")
    #     elif bestcand / opt_prob.dim <= 19:
    #         print("DYCORS")
    #     else:
    #         print("DS")
    point = cand[bestcand, :]
    # if weight <= 0.5:
    #     return point
    # else:
        # point1 = point
        # print(weight)
        

    return point

def candidate_sdsgdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None, 
                     sdsg_hybrid=False):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    point = xbest

    dim = X.shape[1]
    dists = scpspatial.distance.cdist(np.multiply(np.ones((1,dim)),point), np.vstack(X))[0]

    num_pts_in_region = 2*dim+2
    # num_pts_in_region = dim+1
    # num_pts_in_region = len(dists)
    ind = np.argsort(dists)[0:min(num_pts_in_region,len(dists))]

    xx = np.copy(X[ind, :])
    knowledge = np.empty((dim,))
    for i in range(dim):
        knowledge[i] = np.unique(xx[:,i]).shape[0]
    ave = np.median(knowledge)
    subset = np.arange(0, dim)
    subset1 = subset[np.where(knowledge >= ave)[0]]
    subset2 = subset[np.where(knowledge <= ave)[0]]

    weight = weights[0]
    if weight == 0.95:
        num_iter = 1
        # subset = subset1
    else:
        num_iter = 1
        # subset = subset2
    num_cand_per_iter = int(num_cand/num_iter)
    
    for i in range(num_iter):
        # weight = weights[i]
        if sdsg_hybrid:
            if weight < 0.95:
                cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset, sampling_radius, point)
            else:
                cand = normalCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point)
        else:
            cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset, sampling_radius, point)
        # Round integer variables
        cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

        # Make selections
        dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
        dmerit = np.amin(dists, axis=1, keepdims=True)[:,0]

        fvals = surrogate.predict(cand)[:,0]
        ufvals = unit_rescale(fvals)
        rank = weight*ufvals + (1.0-weight)*(1.0 - unit_rescale(np.copy(dmerit)))

        if i < num_iter - 1:
            bestcand = np.argmin(fvals)
        else:
            bestcand = np.argmin(rank)
        # bestcand = np.argmin(rank)
        point = cand[bestcand, :]

    return point

def candidate_sdsgckdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None,
                     sdsg_hybrid=False):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    point = xbest

    dim = X.shape[1]
    dists = scpspatial.distance.cdist(np.multiply(np.ones((1,dim)),point), np.vstack(X))[0]

    num_pts_in_region = 2*dim+2
    # num_pts_in_region = dim+1
    # num_pts_in_region = len(dists)
    ind = np.argsort(dists)[0:min(num_pts_in_region,len(dists))]

    xx = np.copy(X[ind, :])
    knowledge = np.empty((dim,))
    for i in range(dim):
        knowledge[i] = np.unique(xx[:,i]).shape[0]
    ave = np.median(knowledge)
    subset = np.arange(0, dim)
    subset1 = subset[np.where(knowledge >= ave)[0]]
    subset2 = subset[np.where(knowledge <= ave)[0]]

    weight = weights[0]
    if weight == 0.95:
        num_iter = 1
        subset = subset1
    else:
        num_iter = 1
        subset = subset2
    num_cand_per_iter = int(num_cand/num_iter)
    
    for i in range(num_iter):
        # weight = weights[i]
        if sdsg_hybrid:
            if weight < 0.95:
                cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset, sampling_radius, point)
            else:
                cand = normalCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point)
        else:
            cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset, sampling_radius, point)

        # Round integer variables
        cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

        # Make selections
        dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
        dmerit = np.amin(dists, axis=1, keepdims=True)[:,0]

        fvals = surrogate.predict(cand)[:,0]
        ufvals = unit_rescale(fvals)
        rank = weight*ufvals + (1.0-weight)*(1.0 - unit_rescale(np.copy(dmerit)))

        if i < num_iter - 1:
            bestcand = np.argmin(fvals)
        else:
            bestcand = np.argmin(rank)
        # bestcand = np.argmin(rank)
        point = cand[bestcand, :]

    return point

def ds_weighted_distance_merit(num_pts, surrogate, X, fX, cand,
                            weights, Xpend=None, dtol=1e-3, candsepa=None):
    # Distance
    dim = X.shape[1]
    if Xpend is None:  # cdist can't handle None arguments
        Xpend = np.empty([0, dim])
    dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    dmerit = np.amin(dists, axis=1, keepdims=True)

    # Values
    fvals = surrogate.predict(cand)
    fvals = unit_rescale(fvals)

    evalfrom_dycors = 0
    # evalfrom_ds = 0

    # Pick candidate points
    new_points = np.ones((num_pts,  dim))
    for i in range(num_pts):
        w = weights[i]
        merit = w*fvals + (1.0-w)*(1.0 - unit_rescale(np.copy(dmerit)))
        # merit = fvals + (1.0-w)*(1.0 - unit_rescale(np.copy(dmerit)))

        merit[dmerit < dtol] = np.inf
        jj = np.argmin(merit)
        fvals[jj] = np.inf
        new_points[i, :] = cand[jj, :].copy()

        # print("Predict: %.2f, distance: %.2f" % (surrogate.predict(cand[jj, :])[0], dmerit[jj]))

        if jj < candsepa:
            evalfrom_dycors += 1

        # Update distances and weights
        ds = scpspatial.distance.cdist(
            cand, np.atleast_2d(new_points[i, :]))
        dmerit = np.minimum(dmerit, ds)

    return new_points, evalfrom_dycors


def candidate_ckdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    point = xbest

    dim = X.shape[1]
    dists = scpspatial.distance.cdist(np.multiply(np.ones((1,dim)),point), np.vstack(X))[0]

    num_pts_in_region = 2*dim+2
    # num_pts_in_region = dim+1
    # num_pts_in_region = len(dists)
    ind = np.argsort(dists)[0:min(num_pts_in_region,len(dists))]

    xx = np.copy(X[ind, :])
    knowledge = np.empty((dim,))
    for i in range(dim):
        knowledge[i] = np.unique(xx[:,i]).shape[0]
    ave = np.median(knowledge)
    subset = np.arange(0, dim)
    subset1 = subset[np.where(knowledge >= ave)[0]]
    subset2 = subset[np.where(knowledge <= ave)[0]]

    weight = weights[0]
    if weight == 0.95:
        subset = subset1
    else:
        subset = subset2
    
    cand = normalCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point)

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # Make selections
    dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    dmerit = np.amin(dists, axis=1, keepdims=True)[:,0]

    fvals = surrogate.predict(cand)[:,0]
    ufvals = unit_rescale(fvals)
    rank = weight*ufvals + (1.0-weight)*(1.0 - unit_rescale(np.copy(dmerit)))

    bestcand = np.argmin(rank)
    point = cand[bestcand, :]

    return point
    

def candidate_trdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    for i in subset:
        # lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        # ind = np.where(ar[:, i] == 1)[0]
        # lower, upper, sigma = xbest[i]-scalefactors[i], xbest[i]+scalefactors[i], scalefactors[i]
        lower, upper, sigma = xbest[i]-scalefactors[i], xbest[i]+scalefactors[i], sampling_radius*(opt_prob.ub[i]-opt_prob.lb[i])
        lower = max(opt_prob.lb[i], lower)
        upper = min(opt_prob.ub[i], upper)
        ind = np.where(ar[:, i] == 1)[0]
        cand[ind, subset[i]] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
            loc=xbest[i], scale=sigma, size=len(ind))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX,
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)

def get_pareto(P, isexploration, max_num_center):
    # print(P)
    # reverse=True: minimal/best fx first
    # reverse=False: maximal/worst fx first
    if isexploration:
        def takeSecond(elem):
            return elem[1]
        P.sort(reverse=True, key=takeSecond)
    else:
        P.sort(reverse=True)
    # P.sort(reverse=False)
    # print(P)
    F = []
    for ind in range(len(P)):
        p = P[ind]
        # if p[0] < -0.7:
        #     continue
        # F1 = []
        # Sp = []
        isdominate = True
        for q in P:
            if p[0] < q[0] and p[1] < q[1]:
                isdominate = False
                break

        if isdominate:
            F.append(ind)
            if len(F) == max_num_center:
                break
    return F

def truncated_get_pareto(P):
    P.sort(reverse=True)
    F = []
    for ind in range(len(P)):
        p = P[ind]
        if p[0] < -0.7:
            continue
        # F1 = []
        # Sp = []
        isdominate = True
        for q in P:
            if p[0] < q[0] and p[1] < q[1]:
                isdominate = False
                break

        if isdominate:
            F.append(ind)
    return F

# def sort_evaluated_points(pts, max_num_center):
#     indices = []
#     pts.sort(reverse=False)

#     return indices
    

def generate_centers(num_center, isexploration, opt_prob, X, fX, Xpend):
    max_num_center = 10
    nevaled = fX.shape[0]
    fvals = np.copy(fX)
    fvals = unit_rescale(fvals)
    dists = scpspatial.distance.cdist(X, np.vstack((X, Xpend)))
    dmerit = np.zeros((nevaled, 1))
    for i in range(nevaled):
        a = dists[i, :]
        dmerit[i] = np.min(a[np.nonzero(a)])
    dmerit = unit_rescale(dmerit)
    
    if isexploration:
        def sortkey(x):
            return 0.5*(1.0-dmerit[x]) + 0.5*fvals[x]
    else:
        def sortkey(x):
            return fvals[x]
    indices = list(range(nevaled))
    # print(indices)
    indices.sort(reverse=False, key=sortkey)
    # print(indices)
    # indices = truncated_get_pareto(list(zip([x*(-1) for x in fvals], dmerit)))
    # indices = get_pareto(list(zip([x*(-1) for x in fvals], dmerit)), isexploration, max_num_center)
    # indices = sort_evaluated_points(list(zip([x for x in fvals], range(fvals.shape[0]))), max_num_center)

    sel_centers = []
    conflict = 0.3*np.average(opt_prob.ub-opt_prob.lb)
    while len(indices) > 0 and len(sel_centers) < max_num_center:
        isselected = True
        for c in sel_centers:
            if np.linalg.norm(X[c,:]-X[indices[0],:]) < conflict:
                isselected = False
                # print("conflict")
                break
        if isselected == False:
            indices.pop(0)
        else:
            sel_centers.append(indices[0])
            indices.pop(0)

    indices = sel_centers
    if isexploration == False:
        # ind = np.argmin(fX)
        # if ind in indices:
            # indices.pop(indices.index(ind))
        num_center = np.min((num_center, len(indices)))
        center = np.empty((num_center, opt_prob.dim))
        ind_center = []
        # print("Center 1: %.3f, Center 2: %.3f" % (np.min(fX), fX[indices[0]]))
        for i in range(num_center):
            # if i == 0:
                # ind = np.argmin(fX)
            # else:
            ind = indices[0]
            indices.pop(0)
            center[i,:] = np.copy(X[ind, :]).ravel()
            ind_center.append(ind)

    else:
        num_center = np.min((num_center, len(indices)))
        center = np.empty((num_center, opt_prob.dim))
        ind_center = []
        for i in range(num_center):
            # if i == 0:
            #     ind = np.argmin(fX)
            # else:
            ind = indices[0]
            indices.pop(0)
            center[i,:] = np.copy(X[ind, :]).ravel()
            ind_center.append(ind)
    
    # print(np.min(fX))
    # for c in range(center.shape[0]):
    #     print("Center %d: %.3f" % (c, fX[ind_center[c]]))
    return center, ind_center

def getNeighbors(X, fX, radius, center, center_ind):
    dim = X.shape[1]
    # if Xpend is None:  # cdist can't handle None arguments
    #     Xpend = np.empty([0, dim])
    dists = scpspatial.distance.cdist(np.multiply(np.ones((1,dim)),center), np.vstack(X))[0]
    # '''
    ind = np.array(range(min(2*dim+2, X.shape[0])))
    # ind = np.array(range(min(0, X.shape[0])))
    # print(ind)
    # print(dists.shape[0])
    # print(dists[2*dim+2:dists.shape[0]])
    if 2*dim+2 < dists.shape[0]:
        tmp_ind = np.where(dists <= radius)[0]
        # print('tmp_ind: {}'.format(tmp_ind))
        neighbors = tmp_ind[np.where(tmp_ind >= 2*dim+2)]
        # print(neighbors)
        if len(neighbors) > 0:
            # print(ind)
            ind = np.hstack((ind, np.array(neighbors[0:min(len(neighbors),2*dim+2)])))
            # print(ind)
    # '''
    # print("Length of ind: %d" % len(ind))
    # test_ind = np.argsort(dists)[0:2*dim+2]
    num_pts_in_ls = 2*dim+2
    test_ind = np.argsort(dists)[0:min(num_pts_in_ls,len(dists))]
    radius = dists[test_ind[min(num_pts_in_ls,len(dists))-1]]
    # test_xx = np.copy(X)
    test_xx = np.copy(X[test_ind, :])
    # test_fx = np.copy(fX[test_ind]).ravel()
    # test_lb = np.min(test_xx, axis=0)
    # test_ub = np.max(test_xx, axis=0)
    test_knowledge = np.empty((dim,))
    for i in range(dim):
        test_knowledge[i] = np.unique(test_xx[:,i]).shape[0]
    # if int(min(test_knowledge)) == 1:
    #     print("Yeah!")
    # print(test_knowledge)
    subspace_dim = list(range(dim))
    # subspace_dim = np.sort(np.argsort(-test_knowledge)[0:int(min(dim/2+len(fX)/400*dim,dim))])
    # subspace_dim = np.sort(np.argsort(-test_knowledge)[0:int(min(dim/2,dim))]).tolist()
    # for i in range(len(subspace_dim)):
    #     if test_knowledge[subspace_dim[i]] <= 2:
    #         subspace_dim.pop(i)
    subspace_dim = np.where(test_knowledge[subspace_dim] >= 3)[0]
    # subspace_dim = np.where(test_knowledge >= 3)[0]
    # print(subspace_dim)


    test_xx = np.copy(test_xx[:, subspace_dim])
    test_fx = np.copy(fX[test_ind]).ravel()
    # test_xx = np.copy(X[:, subspace_dim])
    # test_fx = np.copy(fX).ravel()
    test_lb = np.min(test_xx, axis=0)
    test_ub = np.max(test_xx, axis=0)


    # print(test_knowledge)
    # print(subspace_dim)
    # print(np.sort(dists)[0:2*dim+2])
    # dists = unit_rescale(dists[0])
    # ratio = 0.5
    # ind = np.where(dists <= 0.4)[0]
    # if ind.shape[0] < 2*dim + 2:
    #     ind = np.argsort(dists)
    #     ind = ind[0:2*dim+2]
    # ind = np.where(dists <= ratio)[0]
    # while ind.shape[0] < 2*(dim + 0):
    #     ratio *= 1.5
    #     ind = np.where(dists <= ratio)[0]
    # print(dists)
    # print(np.average(dists))
    # print(ind)
    # print(X)
    # print(fX)
    # npts = dists[ind].shape[0]
    # xx = np.empty((npts, dim))
    xx = np.copy(X[ind, :])
    fx = np.copy(fX[ind]).ravel()
    lb = np.min(xx, axis=0)
    ub = np.max(xx, axis=0)
    # print(len(ind), len(fX))
    # print(lb, ub)
    # print(ind)
    # print(dists[ind])
    # print(xx)
    # print(fx)

    # return xx, fx, lb, ub
    return test_xx, test_fx, test_lb, test_ub, subspace_dim, radius

def weighted_distance_merit_ls(num_pts, surrogate, X, fX, cand,
                            weights, Xpend=None, dtol=1e-3, 
                            localsurrogate=None, center=None, radius=None, prob=None,
                            subspace_dim=None, accuracy=None):
    # Distance
    dim = X.shape[1]
    if Xpend is None:  # cdist can't handle None arguments
        Xpend = np.empty([0, dim])
    dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    dmerit = np.amin(dists, axis=1, keepdims=True)

    # Values
    if localsurrogate is None:
        fvals = surrogate.predict(cand)
    else:
        subspace_cand = cand[:, subspace_dim]
        # print(subspace_cand)
        fvals1 = surrogate.predict(cand)
        fvals2 = localsurrogate.predict(subspace_cand)
        num_cand = np.shape(dmerit)[0]
        dists = scpspatial.distance.cdist(cand, np.reshape(center, (1,dim)))
        # ind_best = np.argmin(fX)
        fvals = np.zeros((num_cand, 1))
        for i in range(num_cand):
            if dists[i, 0] < radius:
                # print("Use local model!")
                fvals[i] = fvals2[i]
            else:
                fvals[i] = fvals1[i]

        # if len(fX) > 50:
        #     fvals = fvals1
        
        # Accuracy
        if accuracy is not None:
            tmp_acc_global = []
            tmp_acc_local = []
            tmp_acc_mixed = []
            for i in range(num_cand):
                # global
                tmp_acc_global.append(abs(prob.eval(cand[i, :])-fvals1[i]))
                # local
                tmp_acc_local.append(abs(prob.eval(cand[i, :])-fvals2[i]))
                tmp_acc_mixed.append(abs(prob.eval(cand[i, :])-fvals[i]))
            accuracy.addAcc(float(np.average(tmp_acc_global)), float(min(tmp_acc_global)), float(max(tmp_acc_global)), float(np.average(tmp_acc_local)), float(min(tmp_acc_local)), float(max(tmp_acc_local)), 
            float(np.average(tmp_acc_mixed)), float(min(tmp_acc_mixed)), float(max(tmp_acc_mixed)))
            # accuracy_global.append(abs(float(tmp_acc_global/num_cand)))
            # accuracy_local.append(abs(float(tmp_acc_local/num_cand)))
        
    fvals = unit_rescale(fvals)

    # Pick candidate points
    new_points = np.ones((num_pts,  dim))
    for i in range(num_pts):
        w = weights[i]
        merit = w*fvals + (1.0-w)*(1.0 - unit_rescale(np.copy(dmerit)))

        merit[dmerit < dtol] = np.inf
        jj = np.argmin(merit)
        fvals[jj] = np.inf
        new_points[i, :] = cand[jj, :].copy()

        # Update distances and weights
        ds = scpspatial.distance.cdist(
            cand, np.atleast_2d(new_points[i, :]))
        dmerit = np.minimum(dmerit, ds)

    return new_points

def localopt(opt_prob, surrogate, newpts):
    x0 = np.copy(newpts)
    # print(np.shape(x0))

    ''' Steepest Descent '''
    '''
    step = 1
    rho = 0.5
    c = 0.3
    num_iter = 0
    maxiter = 10
    fx0 = surrogate.predict(x0)
    dfx0 = surrogate.predict_deriv(x0).reshape(opt_prob.dim,)
    # print(np.shape(dfx0))
    x = np.copy(x0)
    fx = fx0
    dfx = dfx0
    fxbar = 0
    while step > 1e-6:
        fxbar = surrogate.predict(x + step*(-dfx))
        if fxbar > fx - c*step*np.linalg.norm(dfx):
            step *= rho
        else:
            if np.all(x+step*(-dfx) <= opt_prob.ub) and \
                np.all(x+step*(-dfx) >= opt_prob.lb):
                x += step*(-dfx)
                num_iter += 1
            else:
                break
            fx = surrogate.predict(x)
            if num_iter == maxiter:
                break
            else:
                dfx = surrogate.predict_deriv(x).reshape(opt_prob.dim,)
                step = 1
                continue
    '''

    # '''
    # ss = 0.001
    ss = 0.3
    # ss = 0.01*np.average(opt_prob.ub-opt_prob.lb)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    num_iter = 0
    maxiter = 100
    fx0 = surrogate.predict(x0)
    # dfx0 = surrogate.predict_deriv(x0).reshape(opt_prob.dim,)
    x = np.copy(x0)
    # fx = fx0
    # fxbar = 0
    m = np.zeros((opt_prob.dim,))
    v = np.zeros((opt_prob.dim,))
    while num_iter < maxiter:
        dfx = surrogate.predict_deriv(x).reshape(opt_prob.dim,)
        if np.linalg.norm(dfx) < 1e-6:
            break
        m = beta1*m + (1-beta1)*dfx
        v = beta2*v + (1-beta2)*(dfx**2)
        # mhat = m/(1-beta1**(num_iter+1))
        # vhat = v/(1-beta2**(num_iter+1))
        step = - ss*m/(np.sqrt(v) + eps)
        # step = - ss*mhat/(np.sqrt(vhat) + eps)
        if np.all(x+step <= opt_prob.ub) and \
            np.all(x+step >= opt_prob.lb):
            x += step
            num_iter += 1
        else:
            break
    # '''
    # print(step, x0, x)
    
    # if float((opt_prob.eval(x0)-opt_prob.eval(x))/opt_prob.eval(x0)*100) > 0:
        # fx = surrogate.predict(x)
        # print("%.2f, %.2f" % (float((fx0-fx)/fx0*100), float((opt_prob.eval(x0)-opt_prob.eval(x))/opt_prob.eval(x0)*100)))
    return x
    # else:
        # return x0
    
    # print(opt_prob.eval(x0[0]), opt_prob.eval(x[0]))
    # return x

def weighted_distance_merit_lo(num_pts, surrogate, X, fX, cand,
                            weights, opt_prob, 
                            stat_density, lo_max_density, lo_density_metric, 
                            Xpend=None, dtol=1e-3):
    # Distance
    dim = X.shape[1]
    if Xpend is None:  # cdist can't handle None arguments
        Xpend = np.empty([0, dim])
    dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    dmerit = np.amin(dists, axis=1, keepdims=True)

    # Values
    fvals = surrogate.predict(cand)
    fvals = unit_rescale(fvals)

    # Pick candidate points
    new_points = np.ones((num_pts,  dim))
    for i in range(num_pts):
        w = weights[i]
        merit = w*fvals + (1.0-w)*(1.0 - unit_rescale(np.copy(dmerit)))

        merit[dmerit < dtol] = np.inf
        jj = np.argmin(merit)
        fvals[jj] = np.inf
        new_points[i, :] = cand[jj, :].copy()

        density = np.where(dists[jj,:] < 0.1*np.average(opt_prob.ub-opt_prob.lb))[0].shape[0]
        # print(density)
        stat_density.append(density)
        if lo_density_metric:
            if density > lo_max_density:
            # if density < lo_max_density:
                new_points[i, :] = localopt(opt_prob=opt_prob, surrogate=surrogate, newpts=new_points[i, :])
        else: 
            if w >= 0.8:
                new_points[i, :] = localopt(opt_prob=opt_prob, surrogate=surrogate, newpts=new_points[i, :])

        # Update distances and weights
        ds = scpspatial.distance.cdist(
            cand, np.atleast_2d(new_points[i, :]))
        dmerit = np.minimum(dmerit, ds)

    return new_points

def candidate_dycors_ls(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None, 
                     localsurrogate=None, radius=None, subspace_dim=None, accuracy=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    # subset = subspace_dim.tolist()
    # print(subset)
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        # print(ar)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        # print(ind)
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    for j in range(len(subset)):
        i = subset[j]
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, j] == 1)[0]
        cand[ind, i] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
            loc=xbest[i], scale=sigma, size=len(ind))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)
    # print(cand)

    # Make selections
    return weighted_distance_merit_ls(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX,
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights, 
        localsurrogate=localsurrogate, center=xbest, radius=radius, prob=opt_prob, 
        subspace_dim=subspace_dim, accuracy=accuracy)

def candidate_lodycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, stat_density, lo_max_density, lo_density_metric, 
                     Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    for i in subset:
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, i] == 1)[0]
        cand[ind, subset[i]] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
            loc=xbest[i], scale=sigma, size=len(ind))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # for i in range(num_cand):
    #     if random.randint(0,9) < 1:
    #         cand[i, :] = localopt(opt_prob=opt_prob, surrogate=surrogate, newpts=cand[i, :])

    # Make selections
    return weighted_distance_merit_lo(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX,
        stat_density=stat_density, 
        lo_max_density=lo_max_density, lo_density_metric=lo_density_metric, 
        opt_prob=opt_prob, Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)
    

def candidate_mcdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, meritweight, prob_good=None, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    nevaled = fX.shape[0]
    # num_center = math.ceil(prob_good*nevaled)
    num_center = 10
    fvals = np.copy(fX)
    fvals = unit_rescale(fvals)
    dists = scpspatial.distance.cdist(X, np.vstack((X, Xpend)))
    dmerit = np.zeros((nevaled, 1))
    for i in range(nevaled):
        a = dists[i, :]
        dmerit[i] = np.min(a[np.nonzero(a)])

    ''' Merit version '''
    # # w = weights[0]
    # w = 1 - meritweight
    # merit = w*fvals + (1.0-w)*(1.0 - unit_rescale(np.copy(dmerit)))
    # merit[dmerit < dtol] = np.inf

    # xbetter = np.empty((maxgood, opt_prob.dim))
    # # print('Number of good points: {}'.format(maxgood))
    # for i in range(maxgood):
    #     if i == 0:
    #         ind = np.argmin(fX)
    #     else:
    #         ind = np.argpartition(merit, kth=i-1, axis=0)[i-1]
    #     xbetter[i,:] = np.copy(X[ind, :]).ravel()
    #     # print('{}: merit: {}, fval: {}'.format(i, merit[ind], fvals[ind]))

    ''' Non-dominated sorting version '''
    # print('{} {}'.format(len(get_pareto(list(zip([x*(-1) for x in fvals], dmerit)))), \
    #     len(truncated_get_pareto(list(zip([x*(-1) for x in fvals], dmerit))))))

    indices = truncated_get_pareto(list(zip([x*(-1) for x in fvals], dmerit)))
    # indices.
    center = np.empty((num_center, opt_prob.dim))
    # print('Number of good points: {}'.format(maxgood))
    for i in range(num_center):
        if i == 0:
            ind = np.argmin(fX)
            if ind in indices:
                indices.pop(indices.index(ind))
            center[i,:] = np.copy(X[ind, :]).ravel()
        else:
            if len(indices) > 0:
                ind = indices[0]
                indices.pop(0)
                center[i,:] = np.copy(X[ind, :]).ravel()
            else:
                break
    num_center = np.shape(center)[0]

    cand = np.empty((num_cand, opt_prob.dim))
    num_cand_per_center = int(np.ceil(num_cand/num_center))
    for i in range(num_cand):
        ind_center = i // num_cand_per_center
        cand[i, :] = np.copy(center[ind_center, :]).ravel()

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    # scalefactors = 0.1 * (opt_prob.ub - opt_prob.lb)
    # scalefactors = 0.2 * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    # cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    # cand = xgood
    for i in subset:
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, i] == 1)[0]

        # cand[ind, subset[i]] += stats.norm.rvs(loc=0, scale=sigma, size=len(ind))
        # cand[ind, subset[i]] = np.clip(cand[ind, subset[i]], lower, upper)
        # if cand[ind, subset[i]].any() < upper == False or cand[ind, subset[i]].any() > lower == False:
        #     print('Error! {}'.format(nevaled))
        
        for c in range(num_center):
            ind_subset = ind[np.where((ind >= c*num_cand_per_center) \
                & (ind <= (c+1)*num_cand_per_center-1))[0]]
            cand[ind_subset, subset[i]] = stats.truncnorm.rvs(
                a=(lower - center[c,i]) / sigma, b=(upper - center[c,i]) / sigma,
                loc=center[c,i], scale=sigma, size=len(ind_subset))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # print(weights)
    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX,
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)


def candidate_tvpdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, C, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * C * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    for i in subset:
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, i] == 1)[0]
        cand[ind, subset[i]] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
            loc=xbest[i], scale=sigma, size=len(ind))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX,
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)

def candidate_tvpwdycors(num_pts, opt_prob, surrogate, X, fX, weights,
                     prob_perturb, C, Xpend=None, sampling_radius=0.2,
                     subset=None, dtol=1e-3, num_cand=None, xbest=None):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100*opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * C * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = (np.random.rand(num_cand, len(subset)) < prob_perturb)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    for i in subset:
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, i] == 1)[0]
        cand[ind, subset[i]] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma,
            loc=xbest[i], scale=sigma, size=len(ind))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX,
        Xpend=Xpend, cand=cand, dtol=dtol, weights=weights)


def candidate_sdsgckdycors_std(num_pts, opt_prob, surrogate, X, fX, weights,
                           prob_perturb, Xpend=None, sampling_radius=0.2,
                           subset=None, dtol=1e-3, num_cand=None, xbest=None,
                           sdsg_hybrid=False):
    # Find best solution
    if xbest is None:
        xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100 * opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    point = xbest

    dim = X.shape[1]
    dists = scpspatial.distance.cdist(np.multiply(np.ones((1, dim)), point), np.vstack(X))[0]

    num_pts_in_region = 2 * dim + 2
    # num_pts_in_region = dim+1
    # num_pts_in_region = len(dists)
    ind = np.argsort(dists)[0:min(num_pts_in_region, len(dists))]

    xx = np.copy(X[ind, :])
    xx_scaled = (xx - opt_prob.lb) / (opt_prob.ub - opt_prob.lb)
    knowledge = np.empty((dim,))
    for i in range(dim):
        knowledge[i] = np.std(xx_scaled[:, i])

    # knowledge = np.empty((dim,))

    # for i in range(dim):
    #     knowledge[i] = np.unique(xx[:,i]).shape[0]
    ave = np.median(knowledge)
    subset = np.arange(0, dim)
    subset1 = subset[np.where(knowledge >= ave)[0]]
    subset2 = subset[np.where(knowledge <= ave)[0]]

    weight = weights[0]
    if weight == 0.95:
        num_iter = 1
        subset = subset1
    else:
        num_iter = 1
        subset = subset2
    num_cand_per_iter = int(num_cand / num_iter)

    for i in range(num_iter):
        # weight = weights[i]
        if sdsg_hybrid:
            if weight < 0.95:
                cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset, sampling_radius, point)
            else:
                cand = normalCand(num_cand, subset, prob_perturb, opt_prob, scalefactors, point)
        else:
            cand = dsCand(num_cand, prob_perturb, opt_prob, surrogate, subset, sampling_radius, point)

        # Round integer variables
        cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

        # Make selections
        dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
        dmerit = np.amin(dists, axis=1, keepdims=True)[:, 0]

        fvals = surrogate.predict(cand)[:, 0]
        ufvals = unit_rescale(fvals)
        rank = weight * ufvals + (1.0 - weight) * (1.0 - unit_rescale(np.copy(dmerit)))

        if i < num_iter - 1:
            bestcand = np.argmin(fvals)
        else:
            bestcand = np.argmin(rank)
        # bestcand = np.argmin(rank)
        point = cand[bestcand, :]

    return point