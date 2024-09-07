#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matlab
import matlab.engine

from pySOT.optimization_problems import OptimizationProblem

class Walker(OptimizationProblem):
    def __init__(self, dim=25):
        curpath = os.path.dirname(os.path.realpath(__file__))
        # print(curpath)
        self.engine = matlab.engine.start_matlab()
        self.engine.cd(curpath+'/WGCCM_three_link_walker_example')

        self.dim = dim
        self.lb = np.hstack((1, -2*np.ones((24,))))
        self.ub = np.hstack((10, 2*np.ones((24,))))
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Walker Problem \n" + \
                               "Global optimum: ??"
    
    # def __del__(self):
        # os.system('rm '+self.inipath)
    
    def eval(self, x):
        self.__check_input__(x)
        return -self.engine.walker_speed(matlab.double(list(x)))

    def __call__(self, x):
        return self.eval(x)


if __name__ == '__main__':
    # curpath = os.path.dirname(os.path.realpath(__file__))
    # engine = matlab.engine.start_matlab()
    # res = engine.cd(curpath+'/WGCCM_three_link_walker_example')
    # res = engine.walker_speed(matlab.double(list(np.ones((25,)))))
    # print(res)

    prob = Walker()
    for _ in range(100):
        print(prob.eval(prob.lb + np.multiply(np.random.rand(25), prob.ub-prob.lb)))