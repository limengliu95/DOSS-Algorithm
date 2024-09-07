#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test_rbf import test_rbf

TURBO_AVAIL = True
try:
	from test_turbo import test_turbo
except ImportError:
    TURBO_AVAIL = False

RBFOPT_AVAIL = True
try:
	from test_rbfopt import test_rbfopt
except ImportError:
    RBFOPT_AVAIL = False

from setting import parse_arg

import numpy as np
import os.path
import logging
import time

import os, re, psutil, sys, getopt
from datetime import datetime

# import matplotlib.pyplot as plt

# import cocoex

def test(prob, prob_dim, max_evals, batch_size, num_trial, num_init, \
    num_tr, dir_result, strgy):
    str_prob = prob
    print("Problem: {}-{}D".format(str_prob, prob_dim))

    if strgy == []:
        exit()
    if not os.path.exists(dir_result):
        print('No such dir!')
        exit(0)
    
    result_X = []
    result_fX = []
    num_method = 0
    for strategy in strgy:
        num_method += 1
        print("\nMethod {}: {} ({} Evaluations)".format(num_method, strategy, max_evals))
        if strategy == "TuRBO":
            if not TURBO_AVAIL:
                print("TuRBO is unavailable!")
                continue
            # '''
            for tr in num_tr:
                start = time.time()
                X, fX = test_turbo(prob, prob_dim, max_evals, batch_size, num_trial, num_init, tr)
                end = time.time()
                print('\n   Time: {:.2f}'.format(end-start))
                print("Average: %.3f" % np.average(np.min(fX, axis=1)))
                print("   Best: %.3f" % np.min(np.min(fX, axis=1)))
                print("  Worst: %.3f" % np.max(np.min(fX, axis=1)))
                result_X.append(X)
                result_fX.append(fX)
            # '''
        elif strategy == "RBFOpt":
            if not RBFOPT_AVAIL:
                print("RBFOpt is unavailable!")
                continue
            start = time.time()
            fX = test_rbfopt(prob, prob_dim, max_evals, batch_size, num_trial, num_init)
            end = time.time()
            print('\n   Time: {:.2f}'.format(end-start))
            print("Average: %.3f" % np.average(np.min(fX, axis=1)))
            print("   Best: %.3f" % np.min(np.min(fX, axis=1)))
            print("  Worst: %.3f" % np.max(np.min(fX, axis=1)))
            # result_X.append(X)
            result_fX.append(fX)
        else:
            start = time.time()
            X, fX = test_rbf(prob, prob_dim, max_evals, batch_size, num_trial, num_init, strategy)
            end = time.time()
            print('\n   Time: {:.2f}'.format(end-start))
            print("Average: %.3f" % np.average(np.min(fX, axis=1)))
            print("   Best: %.3f" % np.min(np.min(fX, axis=1)))
            print("  Worst: %.3f" % np.max(np.min(fX, axis=1)))
            result_X.append(X)
            result_fX.append(fX)
        # print(result)

    command=' '.join(sys.argv)
    # np.savez(dir_result+'/'+str(int(time.time()))+'.npz', command=command, prob=prob, prob_dim=prob_dim, max_evals=max_evals, batch_size=batch_size, num_trial=num_trial, num_init=num_init, num_tr=num_tr, strgy=strgy, result_X=result_X, result_fX=result_fX)
    np.savez(dir_result+'/'+str_prob+str(int(time.time()))+'.npz', command=command, prob=prob, prob_dim=prob_dim, max_evals=max_evals, batch_size=batch_size, num_trial=num_trial, num_init=num_init, num_tr=num_tr, strgy=strgy, result_fX=result_fX)

    return result_X, result_fX

if __name__ == '__main__':
    prob, prob_dim, max_evals, batch_size, num_trial, num_init, \
        num_tr, dir_result, strgy = \
        parse_arg(argv=sys.argv[1:])
    
    test(prob, prob_dim, max_evals, batch_size, num_trial, num_init, \
        num_tr, dir_result, strgy)


    
