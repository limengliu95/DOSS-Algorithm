#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test_rbf import test_rbf
from test_turbo import test_turbo
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
            # print("TuRBO is unavailable!")
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


    return result_X, result_fX




    
