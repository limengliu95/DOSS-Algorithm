#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pySOT.experimental_design import SymmetricLatinHypercube
# import pySOT.strategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, \
    LinearTail, SurrogateUnitBox
import pySOT.optimization_problems

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging
import os, re, psutil, sys, getopt, time
from datetime import datetime
# from new_strategy import DYCORSStrategy, MCDYCORSStrategy, TVDYCORSStrategy
# import new_optprob
from setting import set_problem, set_strategy


def test(prob, prob_dim, max_evals, strgy):
                        
    num_threads = 4
    # max_evals = 500
    prob = set_problem(prob, prob_dim)

    rbf = SurrogateUnitBox(
        RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
        tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
    slhd = SymmetricLatinHypercube(
        dim=prob.dim, num_pts=2*(prob.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = set_strategy(strgy, prob, max_evals, slhd, rbf)

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    # print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Strategy: {}".format(strgy))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(rbf.__class__.__name__))

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, prob.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    prob = 'Ackley'
    prob_dim = 10
    max_evals = 50
    dir_log = './logfiles'
    strgy = 'DYCORS'

    problist = ['Ackley', 'Zakharov', 'Rastrigin', 'Schwefel', 'F15']
    strgylist = ['DYCORS', 'MC', 'TVP', 'TVW', 'TVPW', 'MCTVP', 'MCTVW', 'MCTVPW']

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hp:s:d:e:l:')
    except getopt.GetoptError:
        print('usage: '+sys.argv[0]+' [-p prob | -s strategy | -d prob_dim | -e max_evals | -l dir_log]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: '+sys.argv[0]+' [-p prob | -s strategy | -d prob_dim | -e max_evals | -l dir_log]')
            sys.exit()
        elif opt == '-p':
            if arg not in problist:
                print('No such problem!')
                exit()
            else:
                prob = arg
        elif opt == '-s':
            if arg not in strgylist:
                print('No such strategy!')
                exit()
            else:
                strgy = arg
        elif opt == '-d':
            prob_dim = int(arg)
        elif opt == '-e':
            max_evals = int(arg)
        elif opt == '-l':
            dir_log = arg

    if strgy == []:
        exit()
    if not os.path.exists(dir_log):
        print('No such dir!')
        exit(0)
    
    logging.basicConfig(filename=dir_log+'/'+str(int(time.time()))+'.log',
                        level=logging.INFO)
    start = time.time()
    test(prob, prob_dim, max_evals, strgy)
    end = time.time()
    print('Time: {}'.format(end-start))

