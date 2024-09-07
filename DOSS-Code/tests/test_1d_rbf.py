#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test_rbf import test_rbf
# from test_turbo import test_turbo
from setting import parse_arg, set_problem
from test import test

import numpy as np
import os.path
import logging
import time
import os, re, psutil, sys, getopt
from datetime import datetime
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import matplotlib.animation as animation

colormarker = ['-v', '-^', '-o', '-*', '->', '-<', '-s']

if __name__ == '__main__':


    prob, prob_dim, max_evals, batch_size, num_trial, num_init, \
        num_tr, dir_result, strgy = \
        parse_arg(argv=sys.argv[1:])

    # if prob_dim != 1:
    #     print("Error: dimension must = 1!")
    #     exit()
    # prob_dim = 1
    # num_trial = 1

    result_X, result_fX = test(prob, prob_dim, max_evals, batch_size, num_trial, num_init, num_tr, dir_result, strgy)

    nrow = 1
    ncol = len(strgy)+1
    # plt.style.use('ggplot')
    fig = plt.figure(figsize=(13,3))
    # plt.ion()
    fig.add_subplot(nrow, ncol, 1)
    # trunc = 2*prob_dim+2
    trunc = 0
    num_method = 0
    for i in range(len(result_fX)):
        name_strgy = strgy[num_method]
        if name_strgy == 'TuRBO':
            name_strgy = name_strgy + '_' + str(num_tr[i-num_method])
            if i-num_method == len(num_tr) - 1:
                num_method += 1
        else:
            num_method += 1
        fX = np.minimum.accumulate(result_fX[i], axis=1)
        result_ave = np.average(fX, axis=0)
        result_wst = np.max(fX, axis=0)
        result_bst = np.min(fX, axis=0)
        iter = np.array(list(range(1,result_ave.shape[0]+1)))
        # if strgy[i] == 'SOP32':
        #     iter = iter/32
        # elif strgy[i] == 'SOP8':
        #     iter = iter/8
        # elif strgy[i] == 'SOPLS32':
        #     iter = iter/32
        # elif strgy[i] == 'SOPLS8':
        #     iter = iter/8
        plt.plot(iter[trunc:max_evals], result_ave[trunc:max_evals], colormarker[i], label=name_strgy, markevery=max_evals//10)
        plt.fill_between(iter[trunc:max_evals], result_wst[trunc:max_evals], result_bst[trunc:max_evals], alpha=0.2)
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Function Value in '+str(num_trial)+' Trials')
    plt.title(prob+' '+str(prob_dim)+'-D')
    plt.grid()
    plt.legend()

    num_method = 0
    scat = []
    for i in range(len(result_fX)):
        name_strgy = strgy[num_method]
        if name_strgy == 'TuRBO':
            name_strgy = name_strgy + '_' + str(num_tr[i-num_method])
            if i-num_method == len(num_tr) - 1:
                num_method += 1
        else:
            num_method += 1

        fig.add_subplot(nrow, ncol, i+2)
        if prob_dim == 1:
            plt.subplots_adjust(top=0.9,bottom=0.16,left=0.05,right=0.98,hspace=0,wspace=0.19)
        elif prob_dim == 2:
            plt.subplots_adjust(top=0.9,bottom=0.16,left=0.06,right=0.98,hspace=0,wspace=0.28)
        problem = set_problem(prob, prob_dim)
        resolution = 300
        # ims = []
        if prob_dim == 1:
            xx = np.linspace(problem.lb, problem.ub, resolution)
            # print(type(xx))
            fx = np.array(list(problem.eval(np.array([x])) for x in xx.tolist()))
            plt.grid()
            plt.plot(xx, fx, zorder=2)
            
            # ims.append(im)
        elif prob_dim == 2:
            xx = np.linspace(problem.lb[0], problem.ub[0], resolution)
            yy = np.linspace(problem.lb[1], problem.ub[1], resolution)
            XX, YY = np.meshgrid(xx, yy)
            fx = np.empty((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    fx[i,j] = problem.eval(np.array([XX[i,j],YY[i,j]]))
            # fx = np.array(list(problem.eval(np.array(x)) for x in zip(XX.flat, YY.flat)))
            CS = plt.contourf(XX,YY,fx, levels=10)
            # plt.clabel(CS, inline=1, fontsize=10)
            plt.colorbar()
        
        # xx = np.array(result_X[i][0])
        # fx = np.array(result_fX[i][0])
        # print(x.shape)
        # print(fx.shape)
        # print(x[:,0].shape)
        # scat = []
        if prob_dim == 1:
            # x = xx[:, 0]
            # y = fx
            if hasattr(problem, 'min') and hasattr(problem, 'minimum'):
                plt.plot(problem.minimum, problem.min, 'rx')
            # scat = plt.scatter(x, y, c='k', marker='.')
            scat.append(plt.scatter([], [], c='orange', edgecolors='black', marker='.', s=100, zorder=3))
        
            

        elif prob_dim == 2:
            # x = xx[:,0]
            # y = xx[:,1]
            if hasattr(problem, 'min') and hasattr(problem, 'minimum'):
                plt.plot(problem.minimum[0], problem.minimum[1], 'rx', markersize=10)
            # scat = plt.scatter(x, y, c='k', marker='.')
            scat.append(plt.scatter([], [], c='orange', edgecolors='black', marker='.', s=100))

            # def ani_init():
            #     scat.set_offsets([])
            #     return scat,

            # def ani_update(i):
            #     data = np.hstack((x[:i,np.newaxis], y[:i, np.newaxis]))
            #     # print(data)
            #     scat.set_offsets(data)
            #     return scat,
        
            # anim = animation.FuncAnimation(fig, ani_update, init_func=ani_init, frames=range(0,max_evals,batch_size), interval=10000/max_evals*batch_size, blit=False, repeat=False)

        plt.title(name_strgy)

    def ani_init():
        for i in range(len(result_fX)):
            scat[i].set_offsets([])
        return scat,

    def ani_update(j):
        for i in range(len(result_fX)):
            if prob_dim == 1:
                x = np.array(result_X[i][0])[:, 0]
                y = np.array(result_fX[i][0])
            elif prob_dim == 2:
                x = np.array(result_X[i][0])[:,0]
                y = np.array(result_X[i][0])[:,1]
            data = np.hstack((x[:j,np.newaxis], y[:j, np.newaxis]))
            # print(data)
            scat[i].set_offsets(data)
        return scat,

    anim = animation.FuncAnimation(fig, ani_update, init_func=ani_init, frames=range(0,max_evals,batch_size), interval=5000/max_evals*batch_size, blit=False, repeat=False)

    # anim = animation.FuncAnimation(fig, ani_update, init_func=ani_init, frames=range(0,max_evals,batch_size), interval=500, blit=False, repeat=False)
    

            # anim.save('animation.mp4')
    # anim.save("Schwefel_2d.gif", writer='imagemagick')
    # anim.save("DM_Schwefel_2d_10b_200e_100t.gif", writer='imagemagick')
    anim.save("test.gif", writer='imagemagick')
    
    plt.show()