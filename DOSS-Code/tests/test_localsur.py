#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.surrogate import RBFInterpolant, CubicKernel, \
    LinearTail, SurrogateUnitBox
from poap.controller import ThreadController, BasicWorkerThread, SerialController

from setting import parse_arg, set_problem, set_strategy

import numpy as np
import scipy.stats as stats
import time, psutil
import matplotlib.pyplot as plt

from setting import parse_arg
from test import test

import os.path
import logging

import os, re, psutil, sys, getopt
from datetime import datetime

def test_localsur(prob, prob_dim, max_evals, batch_size, num_trial, num_init,  strgy):
    # num_threads = 4
    num_threads = psutil.cpu_count()
    # max_evals = 100

    prob_str = prob
    prob = set_problem(prob_str, prob_dim)
    # print(prob.__class__.__name__)

    rbf = SurrogateUnitBox(
        RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
        tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
    num_init_default = 2*prob_dim + 2
    if num_init == 0:
        num_init = num_init_default
    elif num_init < num_init_default:
        print("Error: number of initial points must > 2*prob_dim + 2!")
        exit()
    slhd = SymmetricLatinHypercube(
        dim=prob.dim, num_pts=num_init)

    X_init = slhd.generate_points(prob.lb, prob.ub, prob.int_var)
    fX_init = np.empty((X_init.shape[0], 1))
    for i in range(X_init.shape[0]):
        fX_init[i] = prob.eval(X_init[i,:])

    radius = 0.1
    center_region = 1 - radius*2
    num_cand = 100*prob.dim
    prob_perturb = min([20.0/prob.dim, 1.0])

    trial = 50
    acc_global = []
    acc_local = []
    for i in range(trial):
    
        print("Trial %d start!" % i)
        # Select a center
        center = prob.lb+(prob.ub-prob.lb)*radius+np.multiply(np.random.rand(prob.dim)*center_region, prob.ub-prob.lb)
        local_lb = center-(prob.ub-prob.lb)*radius
        local_ub = center+(prob.ub-prob.lb)*radius
        # print(prob.lb, prob.ub)
        # print(center)

        # Generate symmetric latin hypercube design around the center
        X_local = slhd.generate_points(local_lb, local_ub, prob.int_var)
        fX_local = np.empty((X_local.shape[0], 1))
        for i in range(X_local.shape[0]):
            fX_local[i] = prob.eval(X_local[i,:])

        # Generate global and local surrogates
        #   Global: contains global slhd and local slhd
        #   Local: contains local slhd only
        globalsur = SurrogateUnitBox(
            RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
            tail=LinearTail(prob.dim), eta=1e-6), lb=prob.lb, ub=prob.ub)
        globalsur.add_points(np.vstack((X_init,X_local)), np.vstack((fX_init, fX_local)))

        localsur = SurrogateUnitBox(
            RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
            tail=LinearTail(prob.dim), eta=1e-6), lb=prob.lb, ub=prob.ub)
        localsur.add_points(X_local, fX_local)

        # Generate candidates
        cand = np.multiply(np.ones((num_cand, prob.dim)), center)
        scalefactors = 0.2*(prob.ub - prob.lb)
        # subset = subspace_dim.tolist()
        # print(subset)
        ar = (np.random.rand(num_cand, prob.dim) < prob_perturb)
        # print(ar)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        # print(ind)
        ar[ind, np.random.randint(0, prob.dim - 1, size=len(ind))] = 1

        
        for j in range(prob.dim):
            i = j
            # lower, upper, sigma = prob.lb[i], prob.ub[i], scalefactors[i]
            lower, upper, sigma = local_lb[i], local_ub[i], scalefactors[i]
            ind = np.where(ar[:, j] == 1)[0]
            cand[ind, i] = stats.truncnorm.rvs(
                a=(lower - center[i]) / sigma, b=(upper - center[i]) / sigma,
                loc=center[i], scale=sigma, size=len(ind))

        # Predict candidates using global and local surrogates
        s1 = globalsur.predict(cand)
        s2 = localsur.predict(cand)
        tmp_acc_global = []
        tmp_acc_local = []
        # tmp_acc_mixed = []
        for i in range(num_cand):
            tmp_acc_global.append(abs(prob.eval(cand[i, :])-s1[i]))
            tmp_acc_local.append(abs(prob.eval(cand[i, :])-s2[i]))
            # tmp_acc_mixed.append(abs(prob.eval(cand[i, :])-fvals[i]))

        acc_global.append(np.average(tmp_acc_global))
        acc_local.append(np.average(tmp_acc_local))

        print("Trial %d finished!" % i)

    plt.grid()
    plt.plot(list(range(trial)), acc_global, label="Global")
    plt.plot(list(range(trial)), acc_local, label="Local")
    plt.xlabel('Trial')
    plt.ylabel('Average Model Deviation')
    plt.title(prob.__class__.__name__+' '+str(prob_dim)+'-D')
    plt.legend()

    plt.show()
    # print('Global: %.3f' % np.average(tmp_acc_global))
    # print('Local: %.3f' % np.average(tmp_acc_local))

    
    return

    # print(globalsur.predict(np.array([0])))
    # print(localsur.predict(np.array([0])))

    # f1: true function, f2: global surrogate, f3: localsurrogate
    if prob.dim == 1:
        resolution = 300
        xx = np.linspace(prob.lb, prob.ub, resolution)
        # print(type(xx))
        f1 = np.array(list(prob.eval(np.array([x])) for x in xx.tolist()))
        f2 = np.array(list(float(globalsur.predict(np.array([x]))) for x in xx.tolist()))
        f3 = np.array(list(float(localsur.predict(np.array([x]))) for x in xx.tolist()))
        plt.grid()
        plt.plot(xx, f1, label='True function', zorder=2)
        plt.plot(xx, f2, label='Global Surrogate', zorder=2)
        plt.plot(xx, f3, label='Local Surrogate', zorder=2)
        plt.legend()

        print("X_init:\n", X_init)
        print("fX_init:\n", fX_init)
        print("X_local:\n", X_local)
        print("fX_local:\n", fX_local)

        plt.show()

    return


if __name__ == '__main__':
    prob, prob_dim, max_evals, batch_size, num_trial, num_init, \
        num_tr, dir_result, strgy = \
        parse_arg(argv=sys.argv[1:])
    test_localsur(prob, prob_dim, max_evals, batch_size, num_trial, num_init, strgy)