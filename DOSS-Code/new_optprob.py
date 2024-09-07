import math
import pySOT
# import rbfopt

ROBOT_AVAIL = True
try:
	from test_functions.robot_pushing.push_function import PushReward as RobotPushing
except ImportError:
    ROBOT_AVAIL = False

ROVER_AVAIL = True
try:
	from test_functions.rover_trajectory_planning.rover_function import RoverTrajPlan
except ImportError:
    ROVER_AVAIL = False

LUNAR_AVAIL = True
try:
	from test_functions.lunar_landing.lunar_lander import LunarLanding
except ImportError:
	LUNAR_AVAIL = False

COSMO_AVAIL = True
try:
	from test_functions.cosmo.cosmo import Cosmo
except ImportError:
	COSMO_AVAIL = False

WALKER_AVAIL = True
try:
	from test_functions.walker.walker import Walker
except ImportError:
	WALKER_AVAIL = False

MNISTWeight_AVAIL = True
try:
	from test_functions.mnist_weight.mnist_weight import MNISTWeight
except ImportError:
	MNISTWeight_AVAIL = False

from test_functions.BBOB.BBOB import BBOB
RBFOPT_AVAIL = True
try:
    import test_functions.rbfopt.rbfopt_test_functions
except ImportError:
    RBFOPT_AVAIL = False

# from rbfopt.rbfopt_test_functions import TestEnlargedBlackBox
# import cocoex

import numpy as np

BBOB_INSTANCE_INDICES = 0
INSTANCE_INDICES = 14

class OptimizationProblem(pySOT.optimization_problems.OptimizationProblem):
    def __call__(self, x):
        return self.eval(x)

class BBOB_F15(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=15, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F16(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=16, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F17(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=17, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F18(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=18, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F19(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=19, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F20(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=20, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F21(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=21, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F22(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=22, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F23(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=23, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)

class BBOB_F24(OptimizationProblem):
    def __init__(self, dim=10):
        self.prob = BBOB(id=24, instance=BBOB_INSTANCE_INDICES, dim=dim)
        self.dim = dim
        self.lb = self.prob.xlow
        self.ub = self.prob.xup
        self.int_var = self.prob.integer
        self.cont_var = self.prob.continuous
        self.info = self.prob.info

    def eval(self, x):
        return self.prob.objfunction(x)


class F15(OptimizationProblem):
    """ F15: Rastrigin function """
    def __init__(self, dim=10):
        # cocoex.interface.Problem
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(14)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)
        # return 10 * self.dim + sum(x**2 - 10 * np.cos(2 * np.pi * x))

class F16(OptimizationProblem):
    """ F16: Weierstrass function """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(15)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)
        # d = len(x)
        # f0, val = 0.0, 0.0
        # for k in range(12):
        #     f0 += 1.0 / (2 ** k) * np.cos(np.pi * (3 ** k))
        #     for i in range(d):
        #         val += 1.0 / (2**k) * np.cos(2*np.pi * (3**k) * (x[i] + 0.5))
        # return 10 * ((1.0 / float(d) * val - f0) ** 3)

    def __call__(self, x):
        return self.eval(x)

class F17(OptimizationProblem):
    """ F17: Schaffers function """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(16)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)
        # d = len(x)
        # f0, val = 0.0, 0.0
        # for k in range(12):
        #     f0 += 1.0 / (2 ** k) * np.cos(np.pi * (3 ** k))
        #     for i in range(d):
        #         val += 1.0 / (2**k) * np.cos(2*np.pi * (3**k) * (x[i] + 0.5))
        # return 10 * ((1.0 / float(d) * val - f0) ** 3)

    def __call__(self, x):
        return self.eval(x)

class F18(OptimizationProblem):
    """ F18: Schaffers function (moderately ill-conditioned) """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(17)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)

    def __call__(self, x):
        return self.eval(x)

class F19(OptimizationProblem):
    """ F19: Composite Griewank-Rosenbrock Function F8F2 """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(18)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)

    def __call__(self, x):
        return self.eval(x)

class F20(OptimizationProblem):
    """ F20: Schwefel Function """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(19)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)

    def __call__(self, x):
        return self.eval(x)

class F21(OptimizationProblem):
    """ F21: Gallagher’s Gaussian 101-me Peaks Function """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(20)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)

    def __call__(self, x):
        return self.eval(x)

class F22(OptimizationProblem):
    """ F22: Gallagher’s Gaussian 21-hi Peaks Function """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(21)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)

    def __call__(self, x):
        return self.eval(x)

class F23(OptimizationProblem):
    """ F23: Katsuura Function """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(22)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)

    def __call__(self, x):
        return self.eval(x)

class F24(OptimizationProblem):
    """ F24: Lunacek bi-Rastrigin Function """
    def __init__(self, dim=10):
        if dim <= 40:
            # Dimension: 2, 3, 5, 10, 20, 40
            suite = cocoex.Suite("bbob", "year:2009", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        else:
            # Dimension: 40, 80, 160, 320, 640
            suite = cocoex.Suite("bbob-largescale", "", "dimensions:"+str(dim)+" instance_indices:"+str(INSTANCE_INDICES))
        prob = suite.get_problem(23)
        self.prob = prob
        self.dim = dim
        self.lb = prob.lower_bounds
        self.ub = prob.upper_bounds
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = prob.info
        # self.min = 0
        # self.minimum = np.zeros(dim)

    def eval(self, x):
        self.__check_input__(x)
        return self.prob(x)

    def __call__(self, x):
        return self.eval(x)

class Levy(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 15 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(self.dim)+'-D Levy problem'
        self.min = 0
        self.minimum = np.ones(dim)

    def eval(self, x):
        self.__check_input__(x)
        # assert len(x) == self.dim
        # assert x.ndim == 1
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val

    def __call__(self, x):
        return self.eval(x)

class Ackley(pySOT.optimization_problems.Ackley):
    def __init__(self, dim=10):
        super().__init__(dim)
        self.lb = -10 * np.ones(dim)
        self.ub = 20 * np.ones(dim)

    def __call__(self, x):
        return self.eval(x)

    # def eval(self, x):

class Zakharov(pySOT.optimization_problems.Zakharov):
    def __call__(self, x):
        return self.eval(x)

class Rastrigin(pySOT.optimization_problems.Rastrigin):
    def __init__(self, dim=10):
        super().__init__(dim)
        self.lb = -3.12 * np.ones(dim)
        self.ub = 8.12 * np.ones(dim)

    def __call__(self, x):
        return self.eval(x)

class Schwefel(pySOT.optimization_problems.Schwefel):
    def __call__(self, x):
        return self.eval(x)

class Griewank(pySOT.optimization_problems.Griewank):
    def __call__(self, x):
        return self.eval(x)

class Weierstrass(pySOT.optimization_problems.Weierstrass):
    def __init__(self, dim=10):
        super().__init__(dim)
        self.lb = -3 * np.ones(dim)
        self.ub = 8 * np.ones(dim)

    def __call__(self, x):
        return self.eval(x)

class Rosenbrock(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = np.ones(dim)
        self.lb = -2.048 * np.ones(dim)
        self.ub = 2.048 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Rosenbrock function \n" +\
                               "Global optimum: f(1,1,...,1) = 0"

    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.dim - 1):
            total += 100 * (x[i] ** 2 - x[i+1]) ** 2 + (x[i] - 1) ** 2
        return total

    def __call__(self, x):
        return self.eval(x)

class Michalewicz(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = np.zeros(dim)
        self.ub = np.pi * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Michalewicz function \n" + \
                               "Global optimum: ??"

    def eval(self, x):
        self.__check_input__(x)
        return -np.sum(np.sin(x) * (
            np.sin(((1 + np.arange(self.dim)) * x**2)/np.pi)) ** 20)

    def __call__(self, x):
        return self.eval(x)

class Keane(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = 1*np.ones(dim)
        self.ub = 10*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Keane function \n" + \
                               "Global optimum: ??"

    def eval(self, x):
        self.__check_input__(x)
        return -abs((np.sum(np.cos(x)**4) - 2*np.prod(np.cos(x)**2))/
            np.sqrt(np.sum((1+np.arange(self.dim))*x**2)))

    def __call__(self, x):
        return self.eval(x)


class Eggholder(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -512*np.ones(dim)
        self.ub = 512*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Eggholder function \n" + \
                               "Global optimum: ??"

    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.dim-1):
            total += -(x[i+1]+47.0)*np.sin(np.sqrt(abs(x[i+1]+x[i]*0.5+47.0)))+np.sin(np.sqrt(abs(x[i]-x[i+1]-47.0)))*(-x[i])
        return total/(self.dim-1)

    def __call__(self, x):
        return self.eval(x)

class StyblinskiTang(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5*np.ones(dim)
        self.ub = 5*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Styblinski-Tang function \n" + \
                               "Global optimum: ??"

    def eval(self, x):
        self.__check_input__(x)
        return np.sum(x**4 - 16*x**2 + 5*x)*0.5


    def __call__(self, x):
        return self.eval(x)

class Schubert(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -10*np.ones(dim)
        self.ub = 10*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Schubert function \n" + \
                               "Global optimum: ??"

    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.dim):
            for j in range(5):
                total += (j+1)*np.sin((j+2)*x[i]+j+1)
        return total


    def __call__(self, x):
        return self.eval(x)

class Rana(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -500*np.ones(dim)
        self.ub = 500*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Rana function \n" + \
                               "Global optimum: ??"

    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.dim-1):
            total += (x[i+1]+1.0)*np.cos(np.sqrt(np.abs(x[i+1]-x[i]+1.0)))*np.sin(np.sqrt(np.abs(x[i+1]+x[i]+1.0))) + \
                np.cos(np.sqrt(np.abs(x[i+1]+x[i]+1.0)))*np.sin(np.sqrt(np.abs(x[i+1]-x[i]+1.0)))*x[i]
        return total/(self.dim-1)


    def __call__(self, x):
        return self.eval(x)

class Hartman3(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Hartman3 function \n" + \
                               "Global optimum: ??"

        self.A = [ [3.0,  0.1,  3.0,  0.1], 
            [10.0, 10.0, 10.0, 10.0],
            [30.0, 35.0, 30.0, 35.0] ]
        self.p = [ [0.36890, 0.46990, 0.10910, 0.03815],
            [0.11700, 0.43870, 0.87320, 0.57430],
            [0.26730, 0.74700, 0.55470, 0.88280] ]
        self.c = [1.0, 1.2, 3.0, 3.2]

        self.subdim = 3
        self.copies = int(self.dim/self.subdim)

    def eval_one_copy(self, x):
        value = -math.fsum([self.c[i]*np.exp(-math.fsum([self.A[j][i]*(x[j] - self.p[j][i])**2 for j in range(3)]))
            for i in range(4)])
        return value
    
    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.copies):
            total += self.eval_one_copy(x[i*self.subdim:(i+1)*self.subdim])

        return total/self.copies

    def __call__(self, x):
        return self.eval(x)

class Hartman6(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Hartman6 function \n" + \
                               "Global optimum: ??"

        self.A = [ [10.00,  0.05,  3.00, 17.00],
            [3.00, 10.00,  3.50,  8.00],
            [17.00, 17.00,  1.70,  0.05],
            [3.50,  0.10, 10.00, 10.00],
            [1.70,  8.00, 17.00,  0.10],
            [8.00, 14.00,  8.00, 14.00] ]
        self.p = [ [0.1312, 0.2329, 0.2348, 0.4047],
            [0.1696, 0.4135, 0.1451, 0.8828],
            [0.5569, 0.8307, 0.3522, 0.8732],
            [0.0124, 0.3736, 0.2883, 0.5743],
            [0.8283, 0.1004, 0.3047, 0.1091],
            [0.5886, 0.9991, 0.6650, 0.0381] ]
        self.c = [1.0, 1.2, 3.0, 3.2]

        self.subdim = 6
        self.copies = int(self.dim/self.subdim)

    def eval_one_copy(self, x):
        value = -math.fsum([self.c[i]*np.exp(-math.fsum([self.A[j][i]*(x[j] - self.p[j][i])**2 for j in range(6)]))
            for i in range(4)])
        return value
    
    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.copies):
            total += self.eval_one_copy(x[i*self.subdim:(i+1)*self.subdim])

        return total/self.copies

    def __call__(self, x):
        return self.eval(x)

class Branin(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.subdim = 2
        self.copies = int(self.dim/self.subdim)
        self.lb = np.array([-5.0, 0.0]*self.copies)
        self.ub = np.array([10.0, 15.0]*self.copies)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Branin function \n" + \
                               "Global optimum: ??"

    def eval_one_copy(self, x):
        value = ((x[1] - (5.1/(4*math.pi*math.pi))*x[0]*x[0] + 
                  5/math.pi*x[0] - 6)**2 + 10*(1-1/(8*math.pi)) *
                 math.cos(x[0]) +10)
        return value
    
    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.copies):
            total += self.eval_one_copy(x[i*self.subdim:(i+1)*self.subdim])

        return total/self.copies

    def __call__(self, x):
        return self.eval(x)

class Camel(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5*np.ones(dim)
        self.ub = 5*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Camel function \n" + \
                               "Global optimum: ??"

        self.subdim = 2
        self.copies = int(self.dim/self.subdim)

    def eval_one_copy(self, x):
        value = ((4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + 
                 x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2)
        return value
    
    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.copies):
            total += self.eval_one_copy(x[i*self.subdim:(i+1)*self.subdim])

        return total/self.copies

    def __call__(self, x):
        return self.eval(x)

class Shekel5(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = 0*np.ones(dim)
        self.ub = 10*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Shekel5 function \n" + \
                               "Global optimum: ??"

        self.subdim = 4
        self.copies = int(self.dim/self.subdim)

        self.A = [ [4.0, 1.0, 8.0, 6.0, 3.0],
            [4.0, 1.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 3.0],
            [4.0, 1.0, 8.0, 6.0, 7.0] ]
        self.c = [0.1, 0.2, 0.2, 0.4, 0.4]

    def eval_one_copy(self, x):
        value = -math.fsum([1.0/(math.fsum([math.fsum([(x[i] - self.A[i][j])**2 for i in range(4)]), self.c[j]]))
            for j in range(5)])

        return value
    
    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.copies):
            total += self.eval_one_copy(x[i*self.subdim:(i+1)*self.subdim])

        return total/self.copies

    def __call__(self, x):
        return self.eval(x)

class Shekel7(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = 0*np.ones(dim)
        self.ub = 10*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Shekel7 function \n" + \
                               "Global optimum: ??"

        self.subdim = 4
        self.copies = int(self.dim/self.subdim)

        self.A = [ [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 5.0],
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 3.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0] ]
        self.c = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3]

    def eval_one_copy(self, x):
        value = -math.fsum([1.0/(math.fsum([math.fsum([(x[i] - self.A[i][j])**2 for i in range(4)]), self.c[j]]))
            for j in range(7)])

        return value
    
    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.copies):
            total += self.eval_one_copy(x[i*self.subdim:(i+1)*self.subdim])

        return total/self.copies

    def __call__(self, x):
        return self.eval(x)

class Shekel10(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = 0*np.ones(dim)
        self.ub = 10*np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Shekel10 function \n" + \
                               "Global optimum: ??"

        self.subdim = 4
        self.copies = int(self.dim/self.subdim)

        self.A = [ [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 5.0, 1.0, 2.0, 3.6],
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 3.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6] ]
        self.c = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]

    def eval_one_copy(self, x):
        value = -math.fsum([1.0/(math.fsum([math.fsum([(x[i] - self.A[i][j])**2 for i in range(4)]), self.c[j]]))
            for j in range(10)])

        return value
    
    def eval(self, x):
        self.__check_input__(x)
        total = 0
        for i in range(self.copies):
            total += self.eval_one_copy(x[i*self.subdim:(i+1)*self.subdim])

        return total/self.copies

    def __call__(self, x):
        return self.eval(x)

class TestEnlargedBlackBox(OptimizationProblem):
    def __init__(self, name, dimension_multiplier=1):
        self.prob = test_functions.rbfopt.rbfopt_test_functions.TestEnlargedBlackBox(name, dimension_multiplier)

        self.dim = self.prob.get_dimension()
        self.prob.perm = np.arange(0, self.dim)
        self.prob.weight = np.ones((dimension_multiplier,))

        self.lb = self.prob.get_var_lower().astype(float)
        self.ub = self.prob.get_var_upper().astype(float)
        # self.int_var = self.get_var_type()
        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)
        self.info = ""
        # print(self.dim)
        # print(self.lb)
        # print(self.lb.dtype)
        # print(self.eval(np.zeros(self.dim,)))

    def eval(self, x):
        return self.prob.evaluate(x)

    def __call__(self, x):
        return self.eval(x)
