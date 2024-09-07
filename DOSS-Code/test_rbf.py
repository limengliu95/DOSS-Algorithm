from pySOT.experimental_design import SymmetricLatinHypercube

from pySOT.surrogate import CubicKernel, TPSKernel, LinearKernel, \
    ConstantKernel, LinearTail, ConstantTail, SurrogateUnitBox
from poap.controller import ThreadController, BasicWorkerThread, SerialController

from setting import parse_arg, set_problem, set_strategy
# from new_surrogate import RBFInterpolant
# from pySOT.surrogate import RBFInterpolant
# from new_lhd import LatinHypercube
# from pySOT.experimental_design import LatinHypercube
import pySOT
import new_lhd
import new_surrogate

import numpy as np
import time, psutil
# import matplotlib.pyplot as plt
import os, re, psutil, sys, getopt

def test_rbf(prob, prob_dim, max_evals, batch_size, num_trial, num_init, strgy):
    # num_threads = 4
    num_threads = psutil.cpu_count()
    # max_evals = 100

    str_prob = prob
    prob = set_problem(prob, prob_dim)
    # print(prob.__class__.__name__)

    if strgy == 'DYCORS':
        num_init_default = 2*prob_dim + 2
    elif strgy == 'SDSGCK_SLHD':
        num_init_default = 2*prob_dim + 2
    elif strgy == 'SDSGDYCK_SLHD':
        num_init_default = 2*prob_dim + 2
    else:
        # num_init_default = 2*prob_dim + 2
        num_init_default = round(0.5*(prob_dim+1)) if prob_dim < 20 else round(0.4*(prob_dim+1))
    if num_init == 0:
        num_init = num_init_default

    print("INIT: %d" % num_init)
    if num_init < 2*prob_dim+2:
        rbf = SurrogateUnitBox(
            new_surrogate.RBFInterpolant(dim=prob.dim, kernel=LinearKernel(),
            tail=ConstantTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        if strgy == 'DSMD':
            slhd = new_lhd.LatinHypercube(dim=prob.dim, num_pts=num_init, use_midpoint=True)
        else:
            slhd = new_lhd.LatinHypercube(dim=prob.dim, num_pts=num_init, use_midpoint=True)
            # slhd = new_lhd.LatinHypercube(dim=prob.dim, num_pts=num_init)
        print("Use Non-symmetric Latin Hypercube Design")
        # print("Error: number of initial points must > 2*prob_dim + 2!")
        # exit()
    else:
        rbf = SurrogateUnitBox(
            pySOT.surrogate.RBFInterpolant(dim=prob.dim, kernel=CubicKernel(),
            tail=LinearTail(prob.dim)), lb=prob.lb, ub=prob.ub)
        slhd = pySOT.experimental_design.SymmetricLatinHypercube(dim=prob.dim, num_pts=num_init)
        # slhd = LatinHypercube(dim=prob.dim, num_pts=num_init)

    # print(strgy)

    result_X = []
    result_fX = []
    for t in range(num_trial):
        # Create a strategy and a controller
        # controller = ThreadController()
        controller = SerialController(prob.eval)
        controller.strategy = set_strategy(strgy, prob, max_evals, batch_size, slhd, rbf)

        # Launch the threads and give them access to the objective function
        # for _ in range(num_threads):
        #     worker = BasicWorkerThread(controller, prob.eval)
        #     controller.launch_worker(worker)

        # Run the optimization strategy
        final_obj = controller.run()
        print("Trial %d: %.3f" % (t+1, final_obj.value))

        # if str_prob == 'LunarLanding':
            # print(prob.show_render(final_obj.params[0]))

        num_evals = len(controller.strategy.fevals)
        if hasattr(controller.strategy, 'fevals'):
            xx = list(controller.strategy.fevals[i].params[0].tolist() for i in range(num_evals))
            fevals = list(controller.strategy.fevals[i].value for i in range(num_evals))
            result_X.append(xx)
            # result_fX.append(np.minimum.accumulate(fevals).tolist())
            result_fX.append(fevals)
        else:
            result_fX = list(final_obj.value for i in range(num_evals))

    '''
    if hasattr(controller.strategy, 'stat_evalfrom_dycors'):
        fig = plt.figure()
        evalfrom_dycors = controller.strategy.stat_evalfrom_dycors
        evalfrom_ds = controller.strategy.stat_evalfrom_ds
        len_evalfrom = len(evalfrom_dycors)
        plt.plot(range(len_evalfrom), evalfrom_dycors, '-v', label='DYCORS')
        plt.plot(range(len_evalfrom), evalfrom_ds, '-^', label='DS')
        plt.xlabel('Number of iterations')
        plt.ylabel('Num_pts from the scheme')
        plt.title(prob.__class__.__name__+' '+str(prob_dim)+'-D')
        plt.legend()
        plt.grid()
        plt.show()

    if hasattr(controller.strategy, 'stat_accuracy'):
        acc = controller.strategy.stat_accuracy
        # print(controller.strategy.stat_accuracy.global_ave)
        # print(controller.strategy.stat_accuracy.global_min)
        # print(controller.strategy.stat_accuracy.global_max)

        np.savez('./results/acc/'+str(int(time.time()))+'.npz', prob=prob.__class__.__name__, dim=prob_dim, global_ave=controller.strategy.stat_accuracy.global_ave, global_min=controller.strategy.stat_accuracy.global_min, global_max=controller.strategy.stat_accuracy.global_max, local_ave=controller.strategy.stat_accuracy.local_ave, local_min=controller.strategy.stat_accuracy.local_min, local_max=controller.strategy.stat_accuracy.local_max, mixed_ave=controller.strategy.stat_accuracy.mixed_ave, mixed_min=controller.strategy.stat_accuracy.mixed_min, mixed_max=controller.strategy.stat_accuracy.mixed_max)

        fig = plt.figure()
        len_acc = len(controller.strategy.stat_accuracy.global_ave)
        plt.plot(range(len_acc), controller.strategy.stat_accuracy.global_ave, label='Global')
        # plt.fill_between(range(len_acc), controller.strategy.stat_accuracy.global_min, controller.strategy.stat_accuracy.global_max, alpha=0.2)
        plt.plot(range(len_acc), controller.strategy.stat_accuracy.local_ave, label='Local')
        # plt.fill_between(range(len_acc), controller.strategy.stat_accuracy.local_min, controller.strategy.stat_accuracy.local_max, alpha=0.2)
        plt.plot(range(len_acc), controller.strategy.stat_accuracy.mixed_ave, label='Mixed')
        # plt.fill_between(range(len_acc), controller.strategy.stat_accuracy.mixed_min, controller.strategy.stat_accuracy.mixed_max, alpha=0.2)

        plt.xlabel('Number of Iterations')
        plt.ylabel('Distance Between Surrogate and True Function')
        # plt.title(prob.__class__.__name__+' '+str(prob_dim)+'-D')
        plt.legend()
        plt.show()
    '''
    
    # end = time.time()

    # print('Time: {:.2f}'.format(end-start))
    return result_X, result_fX

