import rbfopt
import numpy as np
import os
import random
from setting import set_problem


#settings = rbfopt.RbfoptSettings(minlp_solver_path='full/path/to/bonmin', nlp_solver_path='full/path/to/ipopt')
MINLP_SOLVER_PATH = "./solvers/bonmin"
NLP_SOLVER_PATH = "./solvers/ipopt"

def test_rbfopt(prob, prob_dim, max_evals, batch_size, num_trial, num_init):
    str_prob = prob
    prob = set_problem(prob, prob_dim)

    var_type = []
    cont_var = np.array(prob.cont_var).tolist()
    int_var = np.array(prob.int_var).tolist()
    for i in range(prob_dim):
        if i in cont_var:
            var_type.append('R')
        else:
            var_type.append('I')

    var_type = np.array(var_type)

    result_fX = []
    # num_trial = 1
    
    
    
    for i in range(num_trial):
        seed = random.randint(0, 2**32)
        print("Random Seed: %d" % seed)
        bb = rbfopt.RbfoptUserBlackBox(prob_dim, prob.lb, prob.ub, var_type, prob.eval)

        # settings = rbfopt.RbfoptSettings(minlp_solver_path=MINLP_SOLVER_PATH, nlp_solver_path=NLP_SOLVER_PATH, max_evaluations=max_evals, rand_seed=seed, init_include_midpoint=False, max_consecutive_refinement=0, refinement_frequency=100000, global_search_method='sampling')
        # settings = rbfopt.RbfoptSettings(minlp_solver_path=MINLP_SOLVER_PATH, nlp_solver_path=NLP_SOLVER_PATH, max_evaluations=max_evals, rand_seed=seed, init_include_midpoint=False)
        settings = rbfopt.RbfoptSettings(minlp_solver_path=MINLP_SOLVER_PATH, nlp_solver_path=NLP_SOLVER_PATH, max_evaluations=max_evals, max_cycles=max_evals*2, max_iterations=max_evals*2, rand_seed=seed)

        alg = rbfopt.RbfoptAlgorithm(settings, bb)
        alg.set_output_stream(open(os.devnull, 'w'))
        val, x, itercount, evalcount, fast_evalcount = alg.optimize()
        print("Trial %d: %.3f" % (i+1, val))

        # print(alg.all_node_val)
        # print(alg.all_node_val.shape)
        fX = alg.all_node_val[0:max_evals]
        result_fX.append(fX.tolist())


    return result_fX