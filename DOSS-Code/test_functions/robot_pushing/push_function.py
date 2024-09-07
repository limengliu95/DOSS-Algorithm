from .push_utils import b2WorldInterface, make_base, create_body, end_effector, run_simulation
from pySOT.optimization_problems import OptimizationProblem

import numpy as np
import time


class PushReward(OptimizationProblem):
    def __init__(self):

        # domain of this function
        self.xmin = [-5., -5., -10., -10., 2., 0., -5., -5., -10., -10., 2., 0., -5., -5.]
        self.xmax = [5., 5., 10., 10., 30., 2.*np.pi, 5., 5., 10., 10., 30., 2.*np.pi, 5., 5.]

        self.lb = np.array(self.xmin)
        self.ub = np.array(self.xmax)

        # starting xy locations for the two objects
        self.sxy = (0, 2)
        self.sxy2 = (0, -2)
        # goal xy locations for the two objects
        self.gxy = [4, 3.5]
        self.gxy2 = [-4, 3.5]

        self.dim = 14
        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)
        self.info = "14D Robot Pushing Problem"
        # self.min = -self.f_max()
        # print(self.f_max)

        self.initial_dist = self.f_max
        # self.world = b2WorldInterface(False)
        self.oshape, self.osize, self.ofriction, self.odensity, self.bfriction, self.hand_shape, self.hand_size = \
            'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (1, 0.3)

        # self.base = make_base(500, 500, self.world)
        # self.body = create_body(self.base, self.world, 'rectangle', (0.5, 0.5), self.ofriction, self.odensity, self.sxy)
        # self.body2 = create_body(self.base, self.world, 'circle', 1, self.ofriction, self.odensity, self.sxy2)

    @property
    def f_max(self):
        # maximum value of this function
        return np.linalg.norm(np.array(self.gxy) - np.array(self.sxy)) \
            + np.linalg.norm(np.array(self.gxy2) - np.array(self.sxy2))
    # @property
    # def dx(self):
    #     # dimension of the input
    #     return self._dx
    
    def __call__(self, argv):
        # returns the reward of pushing two objects with two robots
        rx = float(argv[0])
        ry = float(argv[1])
        xvel = float(argv[2])
        yvel = float(argv[3])
        simu_steps = int(float(argv[4]) * 10)
        init_angle = float(argv[5])
        rx2 = float(argv[6])
        ry2 = float(argv[7])
        xvel2 = float(argv[8])
        yvel2 = float(argv[9])
        simu_steps2 = int(float(argv[10]) * 10)
        init_angle2 = float(argv[11])
        rtor = float(argv[12])
        rtor2 = float(argv[13])
        
        initial_dist = self.initial_dist

        # world = b2WorldInterface(True)
        world = b2WorldInterface(False)
        # oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = \
        #     'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (1, 0.3)

        base = make_base(500, 500, world)
        body = create_body(base, world, 'rectangle', (0.5, 0.5), self.ofriction, self.odensity, self.sxy)
        body2 = create_body(base, world, 'circle', 1, self.ofriction, self.odensity, self.sxy2)

        robot = end_effector(world, (rx,ry), base, init_angle, self.hand_shape, self.hand_size)
        robot2 = end_effector(world, (rx2,ry2), base, init_angle2, self.hand_shape, self.hand_size)
        (ret1, ret2) = run_simulation(world, body, body2, robot, robot2, xvel, yvel, xvel2, yvel2, rtor, rtor2, simu_steps, simu_steps2)

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1)
        ret2 = np.linalg.norm(np.array(self.gxy2) - ret2)
        # return initial_dist - ret1 - ret2 
        return ret1 + ret2 - initial_dist
    
    def eval(self, argv):
        return self.__call__(argv)


# def main():
#     start = time.time()

#     f = PushReward()
#     # x = np.random.uniform(f.xmin, f.xmax)
#     x = [-0.08849589, -1.84829932,  5.18430299, -6.14263147, 22.45762459,  0.52372063, 4.01852218, -3.11397385, -7.06055894,  5.81246104, 17.53844962,  4.22889853, 0.50470136,  4.65816131]
#     print('Input = {}'.format(x))
#     print('Output = {}'.format(f(x)))

#     end = time.time()
#     print('Time: {:.2f}'.format(end-start))


# if __name__ == '__main__':
#     main()
