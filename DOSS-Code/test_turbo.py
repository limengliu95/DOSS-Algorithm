#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from turbo import Turbo1, TurboM
import numpy as np
import torch
import math
import getopt, sys, time

from setting import set_problem

def test_turbo(prob, prob_dim, max_evals, batch_size, num_trial, num_init, num_tr):
    prob = set_problem(prob, prob_dim)

    # start = time.time()
    result_X = []
    result_fX = []
    print("Number of TR: {}".format(num_tr))
    if num_tr == 1:
        num_init_default = 2*prob_dim
        # num_init_default = 20
        if num_init == 0:
            num_init = num_init_default
        # elif num_init < num_init_default:
            # print("Error: number of initial points must > 2*prob_dim!")
            # exit()

        for t in range(num_trial):
            turbo1 = Turbo1(
                f=prob,  # Handle to objective function
                lb=prob.lb,  # Numpy array specifying lower bounds
                ub=prob.ub,  # Numpy array specifying upper bounds
                n_init=num_init,  # Number of initial bounds from an Latin hypercube design
                max_evals =max_evals,  # Maximum number of evaluations
                batch_size=batch_size,  # How large batch size TuRBO uses
                # verbose=True,  # Print information from each batch
                verbose=False,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # "cpu" or "cuda"
                dtype="float64"  # float64 or float32
            )
            turbo1.optimize()
            X = turbo1.X
            fX = np.reshape(turbo1.fX[0:max_evals], (max_evals,))  # Observed values
            # print(fX)
            ind_best = np.argmin(fX)
            f_best = fX[ind_best]
            # print("Best value found: %.3f" % (f_best))
            print("Trial %d: %.3f" % (t+1, f_best))

            # fX = np.minimum.accumulate(fX)
            # result_strgy.append(np.reshape(fX,(fX.shape[0]),).tolist())
            # fX = np.minimum.accumulate(fX).tolist()
            # result_strgy.append(fX)
            result_X.append(X)
            result_fX.append(fX.tolist())
    else:
        # num_init_default = 2*prob_dim
        num_init_default = 100
        if num_init == 0:
            num_init = num_init_default
        elif num_init < num_init_default:
            print("Error: number of initial points must > 2*prob_dim!")
            exit()

        for t in range(num_trial):
            turbo_m = TurboM(
                f=prob,  # Handle to objective function
                lb=prob.lb,  # Numpy array specifying lower bounds
                ub=prob.ub,  # Numpy array specifying upper bounds
                n_init=num_init,  # Number of initial bounds from an Symmetric Latin hypercube design
                max_evals=max_evals,  # Maximum number of evaluations
                n_trust_regions=num_tr,  # Number of trust regions
                # n_trust_regions=1,
                batch_size=batch_size,  # How large batch size TuRBO uses
                verbose=False,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # "cpu" or "cuda"
                dtype="float64"  # float64 or float32
            )
            turbo_m.optimize()
            X = turbo_m.X  # Evaluated points
            fX = np.reshape(turbo_m.fX[0:max_evals], (max_evals,))  # Observed values
            ind_best = np.argmin(fX)
            f_best = fX[ind_best]
            # print("Best value found: %.3f" % (f_best))
            print("Trial %d: %.3f" % (t+1, f_best))
            # fX = np.minimum.accumulate(fX).tolist()
            # result_strgy.append(fX)
            result_X.append(X)
            result_fX.append(fX.tolist())

    # end = time.time()
    # print('Time: {:.2f}'.format(end-start))
    return result_X, result_fX
