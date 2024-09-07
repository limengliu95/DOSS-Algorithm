import rbfopt
import numpy as np
import pyomo
import pyomo.environ
def obj_funct(x):
  return x[0]*x[1] - x[2]

MINLP_SOLVER_PATH = "D:/Limeng/goss-code/solvers/bonmin.exe"
NLP_SOLVER_PATH = "D:/Limeng/goss-code/solvers/ipopt.exe"
settings = rbfopt.RbfoptSettings(minlp_solver_path=MINLP_SOLVER_PATH, nlp_solver_path=NLP_SOLVER_PATH)

opt = pyomo.opt.SolverFactory(
    'bonmin', executable=settings.minlp_solver_path, solver_io='nl')

print(opt.available())
# bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
#                                np.array(['R', 'I', 'R']), obj_funct)
# settings = rbfopt.RbfoptSettings(max_evaluations=50)
# alg = rbfopt.RbfoptAlgorithm(settings, bb)
# val, x, itercount, evalcount, fast_evalcount = alg.optimize()