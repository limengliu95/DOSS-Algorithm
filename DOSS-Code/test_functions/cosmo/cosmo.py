#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pySOT.optimization_problems import OptimizationProblem

import os, time
import numpy as np
# from configobj import ConfigObj
import configparser

class Cosmo(OptimizationProblem):
    def __init__(self, dim=9):
        fname = str(int(time.time()))+'.ini'
        self.curpath = os.path.dirname(os.path.realpath(__file__))
        # print(curpath)
        self.inipath = os.path.join(self.curpath, fname)
        os.system('cp '+self.curpath+'/lrgdr7like/CAMBfeb09patch/params.ini '+self.inipath)

        self.cf = configparser.ConfigParser()
        self.cf.read(self.inipath, encoding="utf-8")

        self.dim = dim
        self.lb = np.array([0.01, 0.01, 0.01, 52.5, 2.7, 0.2, 2.9, 1.5e-9, 0.72])
        self.ub = np.array([0.25, 0.25, 0.25, 100, 2.8, 0.3, 3.09, 2.6e-8, 5])
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Cosmological Constant Learning \n" + \
                               "Global optimum: ??"
    
    def __del__(self):
        os.system('rm '+self.inipath)
    
    def eval(self, x):
        self.__check_input__(x)

        # curpath = os.path.dirname(os.path.realpath(__file__))
        # print(curpath)

        self.cf.set("DEFAULT", "ombh2", str(x[0]))
        self.cf.set("DEFAULT", "omch2", str(x[1]))
        self.cf.set("DEFAULT", "omk", str(x[2]))
        self.cf.set("DEFAULT", "hubble", str(x[3]))
        self.cf.set("DEFAULT", "temp_cmb", str(x[4]))
        self.cf.set("DEFAULT", "helium_fraction", str(x[5]))
        self.cf.set("DEFAULT", "massless_neutrinos", str(x[6]))
        self.cf.set("DEFAULT", "scalar_amp(1)", str(x[7]))
        self.cf.set("DEFAULT", "scalar_spectral_index(1)", str(x[8]))

        self.cf.write(open(self.inipath, "r+", encoding="utf-8"))
        fx = float(os.popen('cd '+self.curpath+'/lrgdr7like/CAMBfeb09patch; ./camb '+self.inipath+' > /dev/null; cd ..; ./getlrgdr7like;').read())
        if np.isnan(fx):
            return 241.555
        return fx
        # return float(os.popen('cd lrgdr7like/CAMBfeb09patch; ./camb '+self.inipath+' > /dev/null; cd ..; ./getlrgdr7like;').read())

        # ombh2 = 0.0225740
        # omch2 = 0.116197
        # omnuh2 = 0
        # omk = 0
        # hubble = 69.0167
        # temp_cmb = 2.726
        # helium_fraction = 0.24
        # massless_neutrinos = 3.04
        # scalar_amp(1) = 2.15547e-9
        # scalar_spectral_index(1) = 0.959959

    def __call__(self, x):
        return self.eval(x)

if __name__ == '__main__':
    prob = Cosmo()
    # print(prob.eval([0.0225740, 0.116197, 0, 69.0167, 2.726, 0.24, 3.04, 2.15547e-9, 0.959959]))
    # fx = prob.eval([2, 0.116197, 0, 69.0167, 2.726, 0.24, 3.04, 2.15547e-9, 0.959959])
    # x = [0.0284826222, 0.124951136, 0.137150379, 87.6481015, 2.73672042, 0.222211830, 2.96305730, 4.82164797e-09, 4.90742381]
    x = [0.028, 0.124951136, 0.137150379, 87.6481015, 2.73672042, 0.222211830, 2.96305730, 4.82164797e-09, 4.90742381]
    fx = prob.eval(x)
    if np.isnan(fx):
        print("haha")
    print(fx)
    # print(fx)
    # for _ in range(1000):
    #     x = prob.lb + np.multiply(np.random.rand(9), prob.ub-prob.lb)
    #     print(x)
    #     print(prob.eval(x))