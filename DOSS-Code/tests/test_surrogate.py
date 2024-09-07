#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.surrogate import CubicKernel, TPSKernel, LinearKernel, \
    ConstantKernel, LinearTail, ConstantTail, SurrogateUnitBox
from new_surrogate import MultiquadricKernel
from poap.controller import ThreadController, BasicWorkerThread, SerialController

from setting import parse_arg, set_problem, set_strategy
# from new_surrogate import RBFInterpolant
from pySOT.surrogate import RBFInterpolant
from new_lhd import LatinHypercube
# from pySOT.experimental_design import LatinHypercube

import numpy as np
import time, psutil, sys
# import matplotlib.pyplot as plt

def test_surrogate(prob, prob_dim, max_evals, batch_size, num_trial, num_init,  strgy):
    num_threads = psutil.cpu_count()

    str_prob = prob
    prob = set_problem(prob, prob_dim)

    num_init_default = (2*prob_dim + 2)*1
    # num_init_default = round(0.5*(prob_dim+1)) if prob_dim < 20 else round(0.4*(prob_dim+1))
    if num_init == 0:
        num_init = num_init_default

    print("INIT: %d" % num_init)
    # slhd = LatinHypercube(dim=prob.dim, num_pts=num_init)
    slhd = SymmetricLatinHypercube(dim=prob.dim, num_pts=num_init)

    points = slhd.generate_points(lb=prob.lb, ub=prob.ub, int_var=prob.int_var)
    trueval = np.empty((num_init,))
    for i in range(num_init):
        trueval[i] = prob.eval(points[i, :])
    
    for k in range(4):
        if k == 0:
            print("LinearKernel ConstantTail")
            rbf = SurrogateUnitBox(
                RBFInterpolant(dim=prob.dim, kernel=LinearKernel(),
                tail=ConstantTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        elif k == 1:
            print("MultiquadricKernel ConstantTail")
            rbf = SurrogateUnitBox(
                RBFInterpolant(dim=prob.dim, kernel=MultiquadricKernel(),
                tail=ConstantTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        elif k == 2:
            print("CubicKernel LinearTail")
            rbf = SurrogateUnitBox(
                RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
                tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        else:
            print("TPSKernel LinearTail")
            rbf = SurrogateUnitBox(
                RBFInterpolant(dim=prob.dim, kernel=TPSKernel(),
                tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        error = 0
        for i in range(num_init):
            ind = list(range(num_init))
            ind.remove(i)
            rbf.add_points(points[ind, :], trueval[ind])
            prediction = rbf.predict(points)[:, 0]
            error += abs(trueval[i]-prediction[i])
            # print('%d, %.3f' % (i, abs(trueval[i]-prediction[i])))
            rbf.reset()
        print('%.3f' % (error/num_init))

if __name__ == '__main__':
    prob, prob_dim, max_evals, batch_size, num_trial, num_init, \
        num_tr, dir_result, strgy = \
        parse_arg(argv=sys.argv[1:])

    test_surrogate(prob, prob_dim, max_evals, batch_size, num_trial, num_init, strgy)