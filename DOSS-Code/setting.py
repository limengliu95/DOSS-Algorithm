#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import getopt, sys

import pySOT.optimization_problems
import new_optprob
from pySOT.strategy import RandomSampling, SRBFStrategy
from new_strategy import DYCORSStrategy, SOPStrategy, MCDYCORSStrategy, TVDYCORSStrategy, LSDYCORSStrategy, LODYCORSStrategy, TRDYCORSStrategy, DSDYCORSStrategy, CKDYCORSStrategy, DDSDYCORSStrategy, GADYCORSStrategy, CDDYCORSStrategy, SDSGDYCORSStrategy, SDSGCKDYCORSStrategy, SDSGCKDYCORSStrategy_std

PROBLIST = ['Ackley', 'Zakharov', 'Rastrigin', 'Schwefel', 'Levy', 'Griewank', 'Weierstrass', 'Rosenbrock', 'Michalewicz', 'Keane', 'Eggholder', 'StyblinskiTang', 'Schubert', 'Rana', 'Branin', 'Camel', 'Hartman3', 'Hartman6', 'Shekel5', 'Shekel7', 'Shekel10', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'RobotPushing', 'RoverTrajPlan', 'LunarLanding', 'Cosmo', 'Walker', 'MNISTWeight', 'BBOB_F15', 'BBOB_F16', 'BBOB_F17', 'BBOB_F18', 'BBOB_F19', 'BBOB_F20', 'BBOB_F21', 'BBOB_F22', 'BBOB_F23', 'BBOB_F24']
RBFPROBLIST = ['branin', 'camel', 'ex4_1_1', 'ex4_1_2', 'ex8_1_1', 'ex8_1_4', 'goldsteinprice', 'hartman3', 'hartman6', 'least', 'perm_6', 'schaeffer_f7_12_1', 'schaeffer_f7_12_2', 'schoen_6_1', 'schoen_6_2', 'schoen_10_1', 'schoen_10_2', 'shekel10', 'shekel5', 'shekel7']
PROBLIST += RBFPROBLIST
ORIDIMDICT = {'branin':2, 'camel':2, 'ex4_1_1':1, 'ex4_1_2':1, 'ex8_1_1':2, 'ex8_1_4':2, 'goldsteinprice':2, 'hartman3':3, 'hartman6':6, 'least':3, 'perm_6':6, 'schaeffer_f7_12_1':12, 'schaeffer_f7_12_2':12, 'schoen_6_1':6, 'schoen_6_2':6, 'schoen_10_1':10, 'schoen_10_2':10, 'shekel10':4, 'shekel5':4, 'shekel7':4,}
STRGYLIST = ['RAND', 'TuRBO', 'RBFOpt', 'DYCORS', 'SOP8', 'SOP32', 'MC', 'TVP', 'TVW', 'TVPW', 'MCTVP', 'MCTVW', 'MCTVPW', 'LS', 'SOPLS8', 'SOPLS32', 'MCLS', 'LO', 'LO_DEN', 'TR', 'DS', 'CK', 'DDS', 'GA', 'CD', 'SDSG', 'SDSGDY', 'SDSGCK', 'SDSGCK_SLHD', 'SDSGDYCK_SLHD', 'SDSGDYCK', 'SDSGDYCK_std']

def parse_arg(argv):
    prob = 'Ackley'
    prob_dim = 10
    max_evals = 50
    batch_size = 1
    num_trial = 1
    num_init = 0
    num_tr = []
    dir_result = './results/temp'
    strgy = []

    try:
        opts, args = getopt.getopt(argv, 'hp:s:t:d:e:r:b:i:', ["num_tr="])
    except getopt.GetoptError:
        print('usage: '+sys.argv[0]+' [-p prob | -s strategy | -t num_trial | -d prob_dim | -e max_evals | -b batch_size | -i num_init | -r dir_result | --num_tr=num_tr]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: '+sys.argv[0]+' [-p prob | -s strategy | -t num_trial | -d prob_dim | -e max_evals | -b batch_size | -i num_init | -r dir_result | --num_tr=num_tr]')
            sys.exit()
        elif opt == '-p':
            if arg not in PROBLIST:
                print('No such problem!')
                exit()
            else:
                prob = arg
        elif opt == '-s':
            if arg not in STRGYLIST:
                print('No such strategy!')
                exit()
            else:
                strgy.append(arg)
        elif opt == '-t':
            num_trial = int(arg)
        elif opt == '-d':
            prob_dim = int(arg)
        elif opt == '-e':
            max_evals = int(arg)
        elif opt == '-b':
            batch_size = int(arg)
        elif opt == '-i':
            num_init = int(arg)
        elif opt == '-r':
            dir_result = arg
        elif opt == '--num_tr':
            if int(arg) > 0:
                num_tr.append(int(arg))
            else:
                print('Error: number of trust regions must be positive!')
                exit()
    return prob, prob_dim, max_evals, batch_size, num_trial, num_init, num_tr, dir_result, strgy

def set_problem(prob, prob_dim):
    if prob == 'Ackley':
        # return pySOT.optimization_problems.Ackley(dim=prob_dim)
        return new_optprob.Ackley(dim=prob_dim)
    elif prob == 'Zakharov':
        # return pySOT.optimization_problems.Zakharov(dim=prob_dim)
        return new_optprob.Zakharov(dim=prob_dim)
    elif prob == 'Rastrigin':
        # return pySOT.optimization_problems.Rastrigin(dim=prob_dim)
        return new_optprob.Rastrigin(dim=prob_dim)
    elif prob == 'Schwefel':
        # return pySOT.optimization_problems.Schwefel(dim=prob_dim)
        return new_optprob.Schwefel(dim=prob_dim)
    elif prob == 'Levy':
        return new_optprob.Levy(dim=prob_dim)
    elif prob == 'Griewank':
        return new_optprob.Griewank(dim=prob_dim)
    elif prob == 'Weierstrass':
        return new_optprob.Weierstrass(dim=prob_dim)
    elif prob == 'Rosenbrock':
        if prob_dim < 2:
            print("Error: dimension must >= 2!")
            exit()
        else:
            return new_optprob.Rosenbrock(dim=prob_dim)
    elif prob == 'Michalewicz':
        return new_optprob.Michalewicz(dim=prob_dim)
    elif prob == 'Keane':
        return new_optprob.Keane(dim=prob_dim)
    elif prob == 'Eggholder':
        return new_optprob.Eggholder(dim=prob_dim)
    elif prob == 'StyblinskiTang':
        return new_optprob.StyblinskiTang(dim=prob_dim)
    elif prob == 'Schubert':
        return new_optprob.Schubert(dim=prob_dim)
    elif prob == 'Rana':
        return new_optprob.Rana(dim=prob_dim)
    elif prob == 'Hartman3':
        return new_optprob.Hartman3(dim=prob_dim)
    elif prob == 'Hartman6':
        return new_optprob.Hartman6(dim=prob_dim)
    elif prob == 'Branin':
        return new_optprob.Branin(dim=prob_dim)
    elif prob == 'Camel':
        return new_optprob.Camel(dim=prob_dim)
    elif prob == 'Shekel5':
        return new_optprob.Shekel5(dim=prob_dim)
    elif prob == 'Shekel7':
        return new_optprob.Shekel7(dim=prob_dim)
    elif prob == 'Shekel10':
        return new_optprob.Shekel10(dim=prob_dim)
    elif prob in RBFPROBLIST:
        if prob_dim%ORIDIMDICT[prob] != 0:
            print("Error: dimension is not divisible by the original dimension!")
            print("Original dimension: %d" % ORIDIMDICT[prob])
            exit()
        return new_optprob.TestEnlargedBlackBox(prob, int(prob_dim/ORIDIMDICT[prob]))
    elif prob == 'BBOB_F15':
        return new_optprob.BBOB_F15(dim=prob_dim)
    elif prob == 'BBOB_F16':
        return new_optprob.BBOB_F16(dim=prob_dim)
    elif prob == 'BBOB_F17':
        return new_optprob.BBOB_F17(dim=prob_dim)
    elif prob == 'BBOB_F18':
        return new_optprob.BBOB_F18(dim=prob_dim)
    elif prob == 'BBOB_F19':
        return new_optprob.BBOB_F19(dim=prob_dim)
    elif prob == 'BBOB_F20':
        return new_optprob.BBOB_F20(dim=prob_dim)
    elif prob == 'BBOB_F21':
        return new_optprob.BBOB_F21(dim=prob_dim)
    elif prob == 'BBOB_F22':
        return new_optprob.BBOB_F22(dim=prob_dim)
    elif prob == 'BBOB_F23':
        return new_optprob.BBOB_F23(dim=prob_dim)
    elif prob == 'BBOB_F24':
        return new_optprob.BBOB_F24(dim=prob_dim)
    elif prob == 'F15':
        return new_optprob.F15(dim=prob_dim)
    elif prob == 'F16':
        return new_optprob.F16(dim=prob_dim)
    elif prob == 'F17':
        return new_optprob.F17(dim=prob_dim)
    elif prob == 'F18':
        return new_optprob.F18(dim=prob_dim)
    elif prob == 'F19':
        return new_optprob.F19(dim=prob_dim)
    elif prob == 'F20':
        return new_optprob.F20(dim=prob_dim)
    elif prob == 'F21':
        return new_optprob.F21(dim=prob_dim)
    elif prob == 'F22':
        return new_optprob.F22(dim=prob_dim)
    elif prob == 'F23':
        return new_optprob.F23(dim=prob_dim)
    elif prob == 'F24':
        return new_optprob.F24(dim=prob_dim)
    elif prob == 'RobotPushing':
        if not new_optprob.ROBOT_AVAIL:
            print("Error: Problem Unavailable!")
            exit()
        if prob_dim == 14:
            return new_optprob.RobotPushing()
        else:
            print("Error: Dimension of Robot Pushing Problem != 14")
            exit()
    elif prob == 'RoverTrajPlan':
        if not new_optprob.ROVER_AVAIL:
            print("Error: Problem Unavailable!")
            exit()
        if prob_dim == 20 or prob_dim == 60:
            return new_optprob.RoverTrajPlan(prob_dim)
        else:
            print("Error: Dimension of Robot Pushing Problem != 20 or 60")
            exit()
    elif prob == 'LunarLanding':
        if not new_optprob.LUNAR_AVAIL:
            print("Error: Problem Unavailable!")
            exit()
        if prob_dim == 12:
            return new_optprob.LunarLanding()
        else:
            print("Error: Dimension of Lunar Landing Problem != 12")
            exit()
    elif prob == 'Cosmo':
        if not new_optprob.COSMO_AVAIL:
            print("Error: Problem Unavailable!")
            exit()
        if prob_dim == 9:
            return new_optprob.Cosmo()
        else:
            print("Error: Dimension of Cosmological Constant Problem != 9")
            exit()
    elif prob == 'Walker':
        if not new_optprob.WALKER_AVAIL:
            print("Error: Problem Unavailable!")
            exit()
        if prob_dim == 25:
            return new_optprob.Walker()
        else:
            print("Error: Dimension of Walker Problem != 25")
            exit()
    elif prob == 'MNISTWeight':
        if not new_optprob.MNISTWeight_AVAIL:
            print("Error: Problem Unavailable!")
            exit()
        if prob_dim == 100 or prob_dim == 200 or prob_dim == 500:
            return new_optprob.MNISTWeight(dim=prob_dim)
        else:
            print("Error: Dimension of MNISTWeight Problem != 100, 200 or 500")
            exit()
    else:
        print("No such problem!")
        return None

def set_strategy(strgy, prob, max_evals, batch_size, exp_design, surrogate):
    # if batch_size == 1:
        # asynchronous = True
    # else:
    asynchronous = False
    if strgy == 'RAND':
        return RandomSampling(
            max_evals=max_evals, opt_prob=prob)
    elif strgy == 'SOP8':
        return SOPStrategy(
            max_evals=max_evals*8, opt_prob=prob, exp_design=exp_design, surrogate=surrogate, ncenters=8, asynchronous=asynchronous, batch_size=batch_size, extra_points=None, extra_vals=None, use_restarts=True, num_cand=None, lsg=False)
    elif strgy == 'SOP32':
        return SOPStrategy(
            max_evals=max_evals*32, opt_prob=prob, exp_design=exp_design, surrogate=surrogate, ncenters=32, asynchronous=asynchronous, batch_size=batch_size, extra_points=None, extra_vals=None, use_restarts=True, num_cand=None, lsg=False)
    elif strgy == 'MC':
        # print('MCDYCORS')
        return MCDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, ncenters=5,
            Pstrgy=False, Wstrgy=False, lsg=False, batch_size=batch_size)
    elif strgy == 'TVP':
        # print('TVPDYCORS')
        return TVDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, Pstrgy=True, Wstrgy=False, batch_size=batch_size)
    elif strgy == 'TVW':
        # print('TVWDYCORS')
        return TVDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, Pstrgy=False, Wstrgy=True, batch_size=batch_size)
    elif strgy == 'TVPW':
        # print('TVPWDYCORS')
        return TVDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, Pstrgy=True, Wstrgy=True, batch_size=batch_size)
    elif strgy == 'MCTVP':
        return MCDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design, ncenters=8,
            surrogate=surrogate, asynchronous=asynchronous, Pstrgy=True, Wstrgy=False, batch_size=batch_size)
    elif strgy == 'MCTVW':
        return MCDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design, ncenters=8,
            surrogate=surrogate, asynchronous=asynchronous, Pstrgy=False, Wstrgy=True, batch_size=batch_size)
    elif strgy == 'MCTVPW':
        return MCDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design, ncenters=8,
            surrogate=surrogate, asynchronous=asynchronous, Pstrgy=True, Wstrgy=True, batch_size=batch_size)
    elif strgy == 'LS':
        return LSDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'SOPLS8':
        return SOPStrategy(
            max_evals=max_evals*8, opt_prob=prob, exp_design=exp_design, surrogate=surrogate, ncenters=8, asynchronous=asynchronous, batch_size=batch_size, extra_points=None, extra_vals=None, use_restarts=True, num_cand=None, lsg=True)
    elif strgy == 'SOPLS32':
        return SOPStrategy(
            max_evals=max_evals*32, opt_prob=prob, exp_design=exp_design, surrogate=surrogate, ncenters=32, asynchronous=asynchronous, batch_size=batch_size, extra_points=None, extra_vals=None, use_restarts=True, num_cand=None, lsg=True)
    elif strgy == 'MCLS':
        return MCDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, ncenters=8,
            Pstrgy=False, Wstrgy=False, lsg=True, batch_size=batch_size)
    elif strgy == 'LO':
        return LODYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, lo_density_metric=False, batch_size=batch_size)
    elif strgy == 'LO_DEN':
        return LODYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, lo_density_metric=True, batch_size=batch_size)
    elif strgy == 'TR':
        return TRDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'DS':
        return DSDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'CK':
        return CKDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'DDS':
        return DDSDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'GA':
        return GADYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'CD':
        return CDDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'SDSG':
        return SDSGDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'SDSGDY':
        return SDSGDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size,
            sdsg_hybrid=True)
    elif strgy == 'SDSGCK':
        return SDSGCKDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'SDSGCK_SLHD':
        return SDSGCKDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    elif strgy == 'SDSGDYCK_SLHD':
        return SDSGCKDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size,
            sdsg_hybrid=True)
    elif strgy == 'SDSGDYCK':
        return SDSGCKDYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size,
            sdsg_hybrid=True)
    elif strgy == 'SDSGDYCK_std':
        return SDSGCKDYCORSStrategy_std(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size,
            sdsg_hybrid=True)
    elif strgy == 'DYCORS':
        return DYCORSStrategy(
            max_evals=max_evals, opt_prob=prob, exp_design=exp_design,
            surrogate=surrogate, asynchronous=asynchronous, batch_size=batch_size)
    else:
        print('No such strategy!')
        return None