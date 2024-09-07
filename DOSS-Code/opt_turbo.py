#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from turbo import Turbo1, TurboM
import numpy as np
import torch
import math
# import matplotlib
import matplotlib.pyplot as plt
import getopt, sys, time

class Levy:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val

class Ackley:
    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim)

        # self.lb = -15 * np.ones(dim)
        # self.ub = 20 * np.ones(dim)

        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)

        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Ackley function \n" +\
                               "Global optimum: f(0,0,...,0) = 0"

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        d = float(self.dim)
        return -20.0 * np.exp(-0.2*np.sqrt(np.sum(x**2) / d)) - \
            np.exp(np.sum(np.cos(2.0*np.pi*x)) / d) + 20 + np.exp(1)

class Rastrigin:
    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Rastrigin function \n" + \
                               "Global optimum: f(0,0,...,0) = 0"

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        return 10 * self.dim + sum(x**2 - 10 * np.cos(2 * np.pi * x))

class Schwefel:
    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = 420.968746 * np.ones(dim)
        self.lb = -512 * np.ones(dim)
        self.ub = 512 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Schwefel function \n" +\
                               "Global optimum: f(420.9687,...,420.9687) = 0"

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        return 418.9829 * self.dim - \
            sum([y * np.sin(np.sqrt(abs(y))) for y in x])

class Zakharov:
    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0.0
        self.minimum = np.zeros(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Zakharov function \n" + \
                               "Global optimum: f(0,0,...,0) = 1"

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        return np.sum(x**2) + np.sum(0.5*(1 + np.arange(self.dim))*x)**2 + \
            np.sum(0.5*(1 + np.arange(self.dim))*x)**4

PROBLIST = ['Ackley', 'Zakharov', 'Rastrigin', 'Schwefel', 'Levy']
# STRGYLIST = ['1' '5' '10']

def parse_arg(argv):
    prob = 'Ackley'
    prob_dim = 10
    max_evals = 50
    num_trial = 1
    dir_result = './results/temp'
    strgy = []

    try:
        opts, args = getopt.getopt(argv, 'hp:s:t:d:e:r:')
    except getopt.GetoptError:
        print('usage: '+sys.argv[0]+' [-p prob | -s strategy | -t num_trial | -d prob_dim | -e max_evals | -r dir_result]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: '+sys.argv[0]+' [-p prob | -s strategy | -t num_trial | -d prob_dim | -e max_evals | -r dir_result]')
            sys.exit()
        elif opt == '-p':
            if arg not in PROBLIST:
                print('No such problem!')
                exit()
            else:
                prob = arg
        elif opt == '-s':
            # if arg not in STRGYLIST:
            #     print('No such strategy!')
            #     exit()
            # else:
            if int(arg) < 0 or int(arg) > 100:
                print('Invalid number of trust regions!')
                exit()
            else:
                strgy.append(int(arg))
        elif opt == '-t':
            num_trial = int(arg)
        elif opt == '-d':
            prob_dim = int(arg)
        elif opt == '-e':
            max_evals = int(arg)
        elif opt == '-r':
            dir_result = arg

    return prob, prob_dim, max_evals, num_trial, dir_result, strgy

print(sys.argv[1:])
prob, prob_dim, max_evals, num_trial, dir_result, strgy = \
    parse_arg(argv=sys.argv[1:])

if prob == 'Ackley':
    f = Ackley(prob_dim)
elif prob == 'Rastrigin':
    f = Rastrigin(prob_dim)
elif prob == 'Schwefel':
    f = Schwefel(prob_dim)
elif prob == 'Zakharov':
    f = Zakharov(prob_dim)
elif prob == 'Levy':
    f = Levy(prob_dim)

start = time.time()
result = []
for i in range(len(strgy)):
    num_tr = strgy[i]
    result_strgy = []
    if num_tr == 1:
        for i in range(num_trial):
            turbo1 = Turbo1(
                f=f,  # Handle to objective function
                lb=f.lb,  # Numpy array specifying lower bounds
                ub=f.ub,  # Numpy array specifying upper bounds
                n_init=2*prob_dim,  # Number of initial bounds from an Latin hypercube design
                max_evals =max_evals,  # Maximum number of evaluations
                batch_size=10,  # How large batch size TuRBO uses
                verbose=True,  # Print information from each batch
                # verbose=False,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # "cpu" or "cuda"
                dtype="float64"  # float64 or float32
            )
            turbo1.optimize()
            fX = turbo1.fX  # Observed values
            ind_best = np.argmin(fX)
            f_best = fX[ind_best]
            print("Best value found: %.3f" % (f_best))
            fX = np.minimum.accumulate(fX)
            result_strgy.append(np.reshape(fX,(fX.shape[0]),).tolist())
    else:
        for i in range(num_trial):
            turbo_m = TurboM(
                f=f,  # Handle to objective function
                lb=f.lb,  # Numpy array specifying lower bounds
                ub=f.ub,  # Numpy array specifying upper bounds
                n_init=2*prob_dim,  # Number of initial bounds from an Symmetric Latin hypercube design
                max_evals=max_evals,  # Maximum number of evaluations
                n_trust_regions=num_tr,  # Number of trust regions
                # n_trust_regions=1,
                batch_size=10,  # How large batch size TuRBO uses
                # verbose=False,  # Print information from each batch
                verbose = True,
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # "cpu" or "cuda"
                dtype="float64"  # float64 or float32
            )
            turbo_m.optimize()
            # X = turbo_m.X  # Evaluated points
            fX = turbo_m.fX  # Observed values
            ind_best = np.argmin(fX)
            f_best = fX[ind_best]
            print("Best value found: %.3f" % (f_best))
            fX = np.minimum.accumulate(fX)
            result_strgy.append(np.reshape(fX,(fX.shape[0]),).tolist())
    result.append(result_strgy)

end = time.time()
print('Time: {:.2f}'.format(end-start))

# print(result)

command=' '.join(sys.argv)
np.savez(dir_result+'/'+str(int(time.time()))+'.npz', command=command, prob=prob, prob_dim=prob_dim, max_evals=max_evals, num_trial=num_trial, strgy=strgy, result=result)

# fig = plt.figure()
# for i in range(len(result)):
#     result_ave = np.average(result[i], axis=0)
#     result_wst = np.max(result[i], axis=0)
#     result_bst = np.min(result[i], axis=0)
#     iter = np.array(list(range(1,result_ave.shape[0]+1)))
#     plt.plot(iter, result_ave, label=strgy[i], markevery=max_evals//10)
#     plt.fill_between(iter, result_wst, result_bst, alpha=0.2)
#     # plt.plot(iter, result_wst, label=strgy[i]+'_wst', markevery=max_evals//10)
#     # plt.plot(iter, result_bst, label=strgy[i]+'_bst', markevery=max_evals//10)
# plt.xlabel('Number of Evaluations')
# plt.ylabel('Function Value in '+str(num_trial)+' Trials')
# plt.title(prob+' '+str(prob_dim)+'-D')
# plt.legend()

# plt.show()