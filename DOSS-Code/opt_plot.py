#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pySOT.experimental_design import SymmetricLatinHypercube
# from pySOT.strategy import SRBFStrategy, DYCORSStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, \
    LinearTail, SurrogateUnitBox
import pySOT.optimization_problems
from pySOT.optimization_problems import OptimizationProblem
# from pySOT.auxiliary_problems import candidate_dycors
# from pySOT.auxiliary_problems import round_vars, weighted_distance_merit, unit_rescale, candidate_dycors

from poap.controller import ThreadController, BasicWorkerThread, SerialController
import numpy as np
import os.path
import logging
import time
import os, re, psutil, sys
from datetime import datetime
# from new_strategy import DYCORSStrategy, MCDYCORSStrategy, TVPDYCORSStrategy, TVWDYCORSStrategy, TVPWDYCORSStrategy
# import new_optprob
from setting import parse_arg, set_problem, set_strategy

import matplotlib.pyplot as plt

import cocoex

def test(prob, prob_dim, max_evals, num_trial, strgy):
    num_threads = 4
    # max_evals = 100
    prob = set_problem(prob, prob_dim)
    print(prob.info)
    # print(prob.prob.best_observed_fvalue1)

    rbf = SurrogateUnitBox(
        RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
        tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
    slhd = SymmetricLatinHypercube(
        dim=prob.dim, num_pts=2*(prob.dim+1))

    print(strgy)
    start = time.time()

    result = []
    for i in range(num_trial):
        # Create a strategy and a controller
        controller = ThreadController()
        # controller = SerialController(prob.eval)
        controller.strategy = set_strategy(strgy, prob, max_evals, slhd, rbf)

        # Launch the threads and give them access to the objective function
        for _ in range(num_threads):
            worker = BasicWorkerThread(controller, prob.eval)
            controller.launch_worker(worker)

        # Run the optimization strategy
        final_obj = controller.run()

        num_evals = len(controller.strategy.fevals)
        if hasattr(controller.strategy, 'fevals'):
            fevals = list(controller.strategy.fevals[i].value for i in range(num_evals))
            # for i in range(1,len(fevals)):
                # if fevals[i] > fevals[i-1]:
                    # fevals[i] = fevals[i-1]
            # if result == []:
            #     result = fevals
            # else:
            #     result = np.sum([result, fevals], axis=0).tolist()
            result.append(np.minimum.accumulate(fevals).tolist())
        else:
            result = list(final_obj.value for i in range(num_evals))

    
    
    if hasattr(controller.strategy, 'stat_accuracy_global'):
        acc_global = controller.strategy.stat_accuracy_global
        acc_local = controller.strategy.stat_accuracy_local
        # fig = plt.figure()
        # len_acc = controller.strategy.stat_accuracy_global
        # plt.plot(range(len(len_acc)), controller.strategy.stat_accuracy_global, label='global surrogate')
        # plt.plot(range(len(len_acc)), controller.strategy.stat_accuracy_local, label='local surrogate')

        # plt.xlabel('Number of Evaluations')
        # plt.ylabel('Candidates\' Average Distance Between Surrogate and True Function')
        # plt.title(prob.__class__.__name__+' '+str(prob_dim)+'-D')
        # plt.legend()
        # plt.show()
        # print(len(controller.strategy.stat_accuracy_global))
        # print(controller.strategy.stat_accuracy_global)
    # if hasattr(controller.strategy, 'stat_accuracy_local'):
        # print(len(controller.strategy.stat_accuracy_local))
        # print(controller.strategy.stat_accuracy_local)
    else:
        acc_global = []
        acc_local = []
    # result = np.multiply(result, 1/num_trial)

    end = time.time()

    npgs = []
    npls = []
    if hasattr(controller.strategy, 'stat_numpts_ls'):
        npgs = controller.strategy.stat_numpts_gs
        npls = controller.strategy.stat_numpts_ls
    elif hasattr(controller.strategy, 'stat_numpts_gs'):
        npgs = controller.strategy.stat_numpts_gs
    # if hasattr(controller.strategy, 'stat_density'):
    #     fig = plt.figure()
    #     density = controller.strategy.stat_density
    #     iter = range(len(density))
    #     plt.plot(iter, density)
    #     plt.xlabel("Number of Evaluations")
    #     plt.ylabel("Number of Neighbors")
    #     plt.show()
    print('Time: {:.2f}'.format(end-start))
    # print('{} {}'.format(len(npis), npis))
    # print('Time: {:.2f}, value: {:.2f}'.format(end-start, result.value))
    # print('{} \n {}'.format(len(controller.strategy.stat_fbest),np.array_str(np.array(controller.strategy.stat_fbest), precision=5)))
    # print('{} {}'.format(len(controller.strategy.stat_fbest), controller.strategy.stat_neval))
    # print('{}({}): {:.2f} {:.2f}'.format(os.getpid(), max_evals, result.value, end-start))
    # print('Best solution found: {0}\n'.format(
        # np.array_str(result.params[0], max_line_width=np.inf,
                    #  precision=5, suppress_small=True)))
    # print(result[len(result)-1])
    # print(prob.prob.best_observed_fvalue1)
    return result, npgs, npls, acc_global, acc_local
    # return result, npgs

if __name__ == '__main__':
    prob, prob_dim, max_evals, num_trial, dir_result, strgy = \
        parse_arg(argv=sys.argv[1:])

    if strgy == []:
        exit()
    if not os.path.exists(dir_result):
        print('No such dir!')
        exit(0)

    result = []
    npis = []
    for strategy in strgy:
        rst, ns, npls, acc_global, acc_local = test(prob, prob_dim, max_evals, num_trial, strategy)
        result.append(rst)
        npis.append(ns)

    np.savez(dir_result+'/'+str(int(time.time()))+'.npz', prob=prob, prob_dim=prob_dim, max_evals=max_evals, num_trial=num_trial, strgy=strgy, result=result)

    # colormarker = ['k-', 'r:^', 'b:o', 'g--v', 'y-*']

    
    
    fig = plt.figure()
    # fig.add_subplot(1,2,1)
    for i in range(len(result)):
        result_ave = np.average(result[i], axis=0)
        result_wst = np.max(result[i], axis=0)
        result_bst = np.min(result[i], axis=0)
        # print(average)
        iter = np.array(list(range(1,result_ave.shape[0]+1)))
        if strgy[i] == 'SOP32':
            iter = iter/32
        elif strgy[i] == 'SOP8':
            iter = iter/8
        elif strgy[i] == 'SOPLS32':
            iter = iter/32
        elif strgy[i] == 'SOPLS8':
            iter = iter/8
        plt.plot(iter[50:max_evals], result_ave[50:max_evals], label=strgy[i]+'_ave', markevery=max_evals//10)
        plt.fill_between(iter[50:max_evals], result_wst[50:max_evals], result_bst[50:max_evals], alpha=0.2)
        # plt.plot(iter, result_wst, label=strgy[i]+'_wst', markevery=max_evals//10)
        # plt.plot(iter, result_bst, label=strgy[i]+'_bst', markevery=max_evals//10)
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Function Value in '+str(num_trial)+' Trials')
    plt.title(prob+' '+str(prob_dim)+'-D')
    plt.legend()

    # fig.add_subplot(1,2,2)
    # for i in range(len(npis)):
    #     plt.plot(range(len(npis[i])), npis[i], label=strgy[i])
    # plt.xlabel('Number of Evaluations')
    # plt.ylabel('Number of Points in the Current Surrogate')
    # plt.title(prob+' '+str(prob_dim)+'-D')
    # plt.legend()
    
    plt.show()
