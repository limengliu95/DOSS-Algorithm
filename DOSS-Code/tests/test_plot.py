#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test_rbf import test_rbf
from test_turbo import test_turbo
from setting import parse_arg
from test import test
# from turbotest import test

import numpy as np
import os.path
import logging
import time

import os, re, psutil, sys, getopt
from datetime import datetime
import matplotlib.pyplot as plt

colormarker = ['-v', '-^', '-o', '-*', '->', '-<', '-s']
EASY = ['Ackley', 'Keane', 'Levy', 'Michalewicz', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Weierstrass', 'Zakharov', 'Eggholder']
EASYDIM = [10, 30, 50, 100, 200]

def plotResults(dir_result, prob, max_evals, num_method):
    # Read files and save results
    for f in np.sort(os.listdir(dir_result)):
        if f.endswith('.npz') == False:
            continue
        if prob not in f:
            continue
        # print(f)
        data = np.load(dir_result+'/'+f, allow_pickle=True)
        r_prob = str(data['prob'])
        r_strgy = data['strgy']
        r_prob_dim = data['prob_dim']
        r_max_evals = data['max_evals']
        r_num_trial = data['num_trial']
        if 'result' in data.files:
            r_fX = data['result']
        # r_num_tr = data['num_tr']
        elif 'result_fX' in data.files:
            r_fX = data['result_fX']

        if 'result_X' in data.files:
            r_X = data['result_X']
        else:
            r_X = []

        trunc = 0
        r_max_evals = min(r_max_evals, max_evals)
        r_fX = np.minimum.accumulate(r_fX[0], axis=1)[0:r_max_evals]
        result_ave = np.mean(r_fX, axis=0)[0:r_max_evals]
        if r_num_trial == 1:
            error_bar = np.zeros((r_max_evals,))
        elif r_num_trial <= 30:
            result_std = np.std(r_fX, axis=0)
            error_bar = result_std / np.sqrt(r_num_trial)
            error_bar *= tdist95[r_num_trial-2]
        # elif num_trial == 40:
        else:
            error_bar = np.zeros((r_max_evals,))

        iter = np.array(list(range(1,result_ave.shape[0]+1)))
        plt.plot(iter[trunc:r_max_evals], result_ave[trunc:r_max_evals], colormarker[num_method], label=r_strgy[0], markevery=r_max_evals//10)
        plt.fill_between(iter[trunc:r_max_evals], result_ave[trunc:r_max_evals]+error_bar[trunc:r_max_evals], result_ave[trunc:r_max_evals]-error_bar[trunc:r_max_evals], alpha=0.2)

        print("%s: %.3f" % (r_strgy[0], np.min(result_ave)))

if __name__ == '__main__':
    prob, prob_dim, max_evals, batch_size, num_trial, num_init, \
        num_tr, dir_result, strgy = \
        parse_arg(argv=sys.argv[1:])

    # str_prob = prob

    result_X, result_fX = test(prob, prob_dim, max_evals, batch_size, num_trial, num_init, num_tr, dir_result, strgy)

    # print(result_X)

    tdist95 = [12.69, 4.271, 3.179, 2.776, 2.570, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042, 2.021, 2.009, 2.000, 1.994, 1.990, 1.987, 1.984]

    fig = plt.figure()
    # fig.add_subplot(1,2,1)
    num_method = 0
    if batch_size == 1 and \
        num_init == 0 and \
            "DYCORS" not in strgy and \
                prob in EASY and \
                    prob_dim in EASYDIM:
        # print("Plot DYCORS")
        plotResults('./results/DYCORS/EASY/'+str(prob_dim), prob, max_evals, num_method)
        num_method += 1
    if batch_size == 1 and \
        num_init == 0 and \
            "TuRBO" not in strgy and \
                prob in EASY and \
                    prob_dim in EASYDIM and \
                        os.path.exists('./results/TuRBO/EASY/'+str(prob_dim)):
        # print("Plot DYCORS")
        plotResults('./results/TuRBO/EASY/'+str(prob_dim), prob, max_evals, num_method)
        num_method += 1

    # if batch_size == 1 and \
    #     num_init == 0 and \
    #         "DYCORS" not in strgy and \
    #             prob == "RoverTrajPlan" and \
    #                     os.path.exists('./results/DYCORS/ML/Rover'):
    #     # print("Plot DYCORS")
    #     plotResults('./results/DYCORS/ML/Rover', prob, max_evals, num_method)
    #     num_method += 1
    # if batch_size == 1 and \
    #     num_init == 0 and \
    #         "RBFOpt" not in strgy and \
    #             prob == "RoverTrajPlan" and \
    #                     os.path.exists('./results/RBFOpt/ML/Rover'):
    #     # print("Plot DYCORS")
    #     plotResults('./results/RBFOpt/ML/Rover', prob, max_evals, num_method)
    #     num_method += 1
    # if batch_size == 1 and \
    #     num_init == 0 and \
    #         "TuRBO" not in strgy and \
    #             prob == "RoverTrajPlan" and \
    #                     os.path.exists('./results/TuRBO/ML/Rover'):
    #     # print("Plot DYCORS")
    #     plotResults('./results/TuRBO/ML/Rover', prob, max_evals, num_method)
    #     num_method += 1

    if batch_size == 1 and \
        num_init == 0 and \
            "DYCORS" not in strgy and \
                prob in EASY and \
                    prob_dim == 36:
        # print("Plot DYCORS")
        plotResults('./results/DYCORS/EASY36', prob, max_evals, num_method)
        num_method += 1

    if batch_size == 1 and \
        num_init == 0 and \
            "RBFOpt" not in strgy and \
                prob in EASY and \
                    prob_dim == 36:
        # print("Plot DYCORS")
        plotResults('./results/RBFOpt/EASY36', prob, max_evals, num_method)
        num_method += 1
    trunc = 0
    
    num_strgy = 0
    for i in range(len(result_fX)):
        name_strgy = strgy[num_strgy]
        if name_strgy == 'TuRBO':
            name_strgy = name_strgy + '_' + str(num_tr[i-num_strgy])
            if i-num_strgy == len(num_tr) - 1:
                num_strgy += 1
        else:
            num_strgy += 1
        num_method += 1
        result_fX[i] = np.minimum.accumulate(result_fX[i], axis=1)
        result_ave = np.mean(result_fX[i], axis=0)
        if num_trial == 1:
            error_bar = np.zeros((max_evals,))
        elif num_trial <= 30:
            result_std = np.std(result_fX[i], axis=0)
            error_bar = result_std / np.sqrt(num_trial)
            error_bar *= tdist95[num_trial-2]
        # elif num_trial == 40:
        else:
            error_bar = np.zeros((max_evals,))
        # result_wst = np.max(result_fX[i], axis=0)
        # result_bst = np.min(result_fX[i], axis=0)
        iter = np.array(list(range(1,result_ave.shape[0]+1)))
        # if strgy[i] == 'SOP32':
        #     iter = iter/32
        # elif strgy[i] == 'SOP8':
        #     iter = iter/8
        # elif strgy[i] == 'SOPLS32':
        #     iter = iter/32
        # elif strgy[i] == 'SOPLS8':
        #     iter = iter/8
        plt.plot(iter[trunc:max_evals], result_ave[trunc:max_evals], colormarker[num_method-1], label=name_strgy, markevery=max_evals//10)
        plt.fill_between(iter[trunc:max_evals], result_ave[trunc:max_evals]+error_bar[trunc:max_evals], result_ave[trunc:max_evals]-error_bar[trunc:max_evals], alpha=0.2)
        # plt.fill_between(iter[trunc:max_evals], result_wst[trunc:max_evals], result_bst[trunc:max_evals], alpha=0.2)
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Function Value in '+str(num_trial)+' Trials')
    plt.title(prob+' '+str(prob_dim)+'-D')
    plt.grid()
    plt.legend()

    # fig.add_subplot(1,2,2)
    # for i in range(len(npis)):
    #     plt.plot(range(len(npis[i])), npis[i], label=strgy[i])
    # plt.xlabel('Number of Evaluations')
    # plt.ylabel('Number of Points in the Current Surrogate')
    # plt.title(prob+' '+str(prob_dim)+'-D')
    # plt.legend()
    
    plt.show()