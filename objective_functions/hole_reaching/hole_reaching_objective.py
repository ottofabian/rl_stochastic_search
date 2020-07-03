import tensorflow

from objective_functions.hole_reaching.abstract_env import AbstractEnvironment
from objective_functions.hole_reaching import planar_forward_kinematics as pfk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from objective_functions.hole_reaching.mp_lib import dmps
from objective_functions.hole_reaching.mp_lib.basis import DMPBasisGenerator
from objective_functions.hole_reaching.mp_lib.phase import ExpDecayPhaseGenerator


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class HoleReachingCostFunction:
    def __init__(self, num_links, num_basis, hole_x, hole_width, hole_depth, allow_self_collision=False, pid=0):
        self.pid = pid
        self.hole_x = hole_x  # x-position of center of hole
        self.hole_width = hole_width  # width of hole
        self.hole_depth = hole_depth  # depth of hole
        self.num_links = num_links
        self.bottom_center_of_hole = np.hstack([hole_x, -hole_depth])
        self.top_center_of_hole = np.hstack([hole_x, 0])
        self.left_wall_edge = np.hstack([hole_x - self.hole_width / 2, 0])
        self.right_wall_edge = np.hstack([hole_x + self.hole_width / 2, 0])
        self.allow_self_collision = allow_self_collision
        self.pfk = pfk.PlanarForwardKinematics(num_joints=num_links)

        # use 5 basis functions per dof
        self.num_basis = num_basis
        self.t = np.linspace(0, 2, 200)
        phase_generator = ExpDecayPhaseGenerator(alpha_phase=5, duration=2)
        basis_generator = DMPBasisGenerator(phase_generator, num_basis=self.num_basis, duration=2)

        self.dmp = dmps.DMP(num_dof=num_links,
                            basis_generator=basis_generator,
                            phase_generator=phase_generator,
                            num_time_steps=200
                            )

        # self.dmp.dmp_beta_x = 0
        self.dmp.dmp_start_pos = np.zeros((1, num_links))
        self.dmp.dmp_start_pos[0, 0] = np.pi / 2

        weights = np.random.normal(loc=0.0, scale=0.01, size=(num_basis, num_links))
        # goal = np.random.normal(0.0, 0.01, (num_links, ))
        # specific for 5 links
        # goal[0] = np.pi / 2
        # goal[1:] = -np.pi / 5
        goal = np.array([1.16164463, -0.3358335, -1.54852792, -0.65031571, -0.21446159])

        self.dmp.set_weights(weights, goal)

        self.n_evaluations = 0

    def rollout(self, num_points_per_link=10, render=False, render_full=False):
        # trajectory should be [num_time_steps, num_joints]
        trajectory, vel = self.dmp.reference_trajectory(self.t)

        acc = np.sum(np.diff(trajectory, n=2, axis=0) ** 2)

        total_number_of_points_collided = 0
        end_effector_points = []

        # endeffector = self.pfk  # maybe put an initial position?
        for t, traj in enumerate(trajectory):
            line_points_in_taskspace = self.pfk.get_forward_kinematics(traj[:, None],
                                                                       num_points_per_link=num_points_per_link)

            if render_full:
                if t == 0:
                    fig, ax = plt.subplots()
                    rect_1 = patches.Rectangle((-self.num_links, -1),
                                               self.num_links + self.hole_x - self.hole_width / 2, 1,
                                               fill=True, edgecolor='k', facecolor='k')
                    rect_2 = patches.Rectangle((self.hole_x + self.hole_width / 2, -1),
                                               self.num_links - self.hole_x + self.hole_width / 2, 1,
                                               fill=True, edgecolor='k', facecolor='k')
                    rect_3 = patches.Rectangle((self.hole_x - self.hole_width / 2, -1), self.hole_width,
                                               1 - self.hole_depth,
                                               fill=True, edgecolor='k', facecolor='k')
                ax.clear()
                plt.xlim(-self.num_links, self.num_links), plt.ylim(-1, self.num_links)
                ax.plot(line_points_in_taskspace[:, 0, 0],
                        line_points_in_taskspace[:, 0, 1],
                        line_points_in_taskspace[:, -1, 0],
                        line_points_in_taskspace[:, -1, 1], marker='o')

                # Add the patch to the Axes
                ax.add_patch(rect_1)
                ax.add_patch(rect_2)
                ax.add_patch(rect_3)
                plt.pause(0.01)

            end_effector_points.append(line_points_in_taskspace[-1, -1, :])

            if np.any(np.abs(traj) > np.pi) and not self.allow_self_collision:
                is_collided = 1
                break

            is_collided = self.check_collision(line_points_in_taskspace)

            # add percentage of points collided during a step
            # total_number_of_points_collided += number_of_points_collided  # / (num_points_per_link * self.num_links)

            if is_collided:
                break
                # check distance of endeffector to bottom center of hole
                # endeffector = line_points_in_taskspace[-1, -1, :]
                # distance = np.linalg.norm(endeffector - self.bottom_center_of_hole)
                # return np.abs(distance)**2 + 100 * np.abs(acc) + 100000

        if render:
            fig, ax = plt.subplots()
            rect_1 = patches.Rectangle((-self.num_links, -1),
                                       self.num_links + self.hole_x - self.hole_width / 2, 1,
                                       fill=True, edgecolor='k', facecolor='k')
            rect_2 = patches.Rectangle((self.hole_x + self.hole_width / 2, -1),
                                       self.num_links - self.hole_x + self.hole_width / 2, 1,
                                       fill=True, edgecolor='k', facecolor='k')
            rect_3 = patches.Rectangle((self.hole_x - self.hole_width / 2, -1), self.hole_width,
                                       1 - self.hole_depth,
                                       fill=True, edgecolor='k', facecolor='k')
            plt.xlim(-self.num_links, self.num_links), plt.ylim(-1, self.num_links)
            ax.plot(line_points_in_taskspace[:, 0, 0],
                    line_points_in_taskspace[:, 0, 1],
                    line_points_in_taskspace[:, -1, 0],
                    line_points_in_taskspace[:, -1, 1], marker='o')

            # Add the patch to the Axes
            ax.add_patch(rect_1)
            ax.add_patch(rect_2)
            ax.add_patch(rect_3)
            plt.pause(0.01)
        # check the distance the endeffector travelled to the center of the hole
        # end_effector_travel = np.sum(
        #     np.sqrt(np.sum(np.diff(np.stack(end_effector_points), axis=0)[:, 4, :] ** 2, axis=1, keepdims=True))) ** 2
        # end_effector_travel = np.sum(np.sqrt(np.sum(np.diff(np.stack(end_effector_points), axis=0) ** 2, axis=2)))

        end_eff_traj = np.vstack(end_effector_points)
        t_hole = np.argmin(np.abs(end_eff_traj[:, 1]))
        distance_end_eff_center = end_eff_traj[t_hole, 0] - self.hole_x
        # distance_end_eff_center = end_eff_traj[:, 0] - self.hole_x
        # distance_end_eff_center = np.where(end_eff_traj[:, 1] > 0,
        #                                    np.linalg.norm(end_eff_traj - self.top_center_of_hole, axis=1),
        #                                    end_eff_traj[:, 0] - self.hole_x)
        # check distance of endeffector to bottom center of hole
        endeffector = line_points_in_taskspace[-1, -1, :]

        last_joint = line_points_in_taskspace[-2, -1, :]
        # roughly normalized to be between 0 and 1
        # distance = np.linalg.norm(endeffector - self.bottom_center_of_hole)  # / (self.num_links + np.abs(self.hole_x))

        distance_end_eff_bottom = np.linalg.norm(endeffector - (self.bottom_center_of_hole + np.array([0, 1e-2])))
        # if endeffector[1] > 0:
        #     distance_end_eff_center = np.linalg.norm(endeffector - self.top_center_of_hole)
        #     # distance_end_eff_right_wall = np.linalg.norm(endeffector - self.right_wall_edge)
        #
        #     distance_last_joint_center = np.linalg.norm(last_joint - self.top_center_of_hole)
        #     # distance_last_joint_right_wall = np.linalg.norm(last_joint - self.right_wall_edge)
        # else:
        # distance_end_eff_center = endeffector[0] - self.hole_x
        distance_last_joint_center = last_joint[0] - self.hole_x

        final_vel = np.linalg.norm(vel[-1])

        if acc > 10:
            acc = 10
            is_collided = True

        # TODO: tune factors
        # distance in [0, 1]
        # |acc| in [0, 0.1]
        out = 10 * np.abs(distance_end_eff_bottom) ** 2 \
              + 0.50 * np.abs(distance_end_eff_center ** 2) \
              + 0.10 * np.abs(distance_last_joint_center) ** 2 \
              + 1 * final_vel ** 2 \
              + 1 * np.abs(acc) \
              + is_collided * 1000
        # + 0.1 * total_number_of_points_collided\
        # + 0.01 * end_effector_travel ** 2

        if np.abs(distance_end_eff_bottom) ** 2 < 0.001:
            success = True
        else:
            success = False

        return out, success

    def check_collision(self, line_points):

        if not self.allow_self_collision:
            for i, line1 in enumerate(line_points):
                for line2 in line_points[i + 2:, :, :]:
                    # if line1 != line2:
                    if intersect(line1[0], line1[-1], line2[0], line2[-1]):
                        return True

        # all points that are before the hole in x
        r, c = np.where(line_points[:, :, 0] < (self.hole_x - self.hole_width / 2))

        # check if any of those points are below surface
        nr_line_points_below_surface_before_hole = np.sum(line_points[r, c, 1] < 0)

        # all points that are after the hole in x
        r, c = np.where(line_points[:, :, 0] > (self.hole_x + self.hole_width / 2))

        # check if any of those points are below surface
        nr_line_points_below_surface_after_hole = np.sum(line_points[r, c, 1] < 0)

        # all points that are above the hole
        r, c = np.where((line_points[:, :, 0] > (self.hole_x - self.hole_width / 2)) & (
                line_points[:, :, 0] < (self.hole_x + self.hole_width / 2)))

        # check if any of those points are below surface
        nr_line_points_below_surface_in_hole = np.sum(line_points[r, c, 1] < -self.hole_depth)

        # total_nr_line_points_below_surface = nr_line_points_below_surface_before_hole + nr_line_points_below_surface_after_hole + nr_line_points_below_surface_in_hole

        if nr_line_points_below_surface_before_hole > 0 or nr_line_points_below_surface_after_hole > 0 or nr_line_points_below_surface_in_hole > 0:
            return True

        return False


class HoleReachingCostFunctionMP(HoleReachingCostFunction):
    def __init__(self, pid):
        super().__init__(pid=pid, num_links=5, num_basis=5, hole_x=2, hole_width=0.5, hole_depth=1,
                         allow_self_collision=False)

    def __call__(self, contexts=None, parameters=None, render=False):
        if parameters is not None:
            if len(parameters.shape) > 1:
                assert parameters.shape[0] == 1
                parameters = parameters.flatten()
                parameters = parameters.flatten()
            weight_matrix = np.reshape(parameters[0:-self.num_links], [self.num_basis, self.num_links]) * 100
            goal_attractor = parameters[-self.num_links:]
            self.dmp.set_weights(weight_matrix, goal_attractor)
            # self.dmp.dmp_goal_pos[0] = goal_attractor

        # FIXME: How to ensure goal velocity is reached?
        self.n_evaluations += 1
        rew, succ = self.rollout(num_points_per_link=10, render=render)
        return rew, succ


class HoleReachingEnv(AbstractEnvironment):
    def __init__(self, name, n_cores, learn_goal=True):
        # self.dim = 30  # this should be adaptive
        self.n_evaluations = 0
        self.allow_parallel = True
        self.learn_goal = learn_goal
        if learn_goal:
            self.dim = 30
            self.goal = None
        else:
            self.dim = 25
            self.goal = np.array([1.16164463, -0.3358335, -1.54852792, -0.65031571, -0.21446159])
        self.xopt = np.zeros(self.dim)  # only for the record
        self.f_opt = tensorflow.constant(0)

        super().__init__(name, n_cores, "default", {"default": ()}, HoleReachingCostFunctionMP)

    def __call__(self, xs, context=None, render=False):
        if len(xs.shape) == 1:
            xs = xs.reshape((1, -1))
        n = xs.shape[0]
        if not self.learn_goal:
            xs = np.hstack([xs, np.tile(self.goal, (n, 1))])
        cs = np.ones(shape=(n, 1))
        rs, succs = self.evaluate(cs, xs, render)
        self.n_evaluations += n
        return rs

    @staticmethod
    def getfopt():
        return 0


class HoleReachingObjective:
    def __init__(self, num_links=5, num_basis=5, hole_x=2, hole_width=0.5, allow_self_collision=False, learn_goal=True):
        self.num_links = num_links
        self.num_basis = num_basis
        self.learn_goal = learn_goal
        if learn_goal:
            self.dim = num_links * num_basis + num_links
            self.goal = None
        else:
            self.dim = num_links * num_basis
            self.goal = np.array([1.16164463, -0.3358335, -1.54852792, -0.65031571, -0.21446159])
        self.f_opt = 0
        self.xopt = np.zeros(self.dim)  # only for the record
        self.n_evaluations = 0
        # the center of the hole is placed randomly between +1 and +3 for 5 links
        # TODO: make variable for != 5 links
        # TODO: make random once it works, but keep constant during optimization. only make random in iteration with
        # context variables
        hole_x = hole_x  # 2 * np.random.rand(1) + 1
        # width of the whole is between 0.5 and 1
        hole_width = hole_width  # 0.5 * np.random.rand(1) + 0.5
        # holedepth is between 0.5 and 1
        hole_depth = 1  # 0.5 * np.random.rand(1) + 0.5
        # create task
        self.cost_fn = HoleReachingCostFunction(num_links=num_links,
                                                num_basis=num_basis,
                                                hole_x=hole_x,
                                                hole_width=hole_width,
                                                hole_depth=hole_depth,
                                                allow_self_collision=allow_self_collision)

        self.opt = None
        self.allow_parallel = False

    def getfopt(self):
        return self.f_opt

    def __call__(self, parameters=None, plot=False):
        if parameters is not None:
            if len(parameters.shape) > 1:
                assert parameters.shape[0] == 1
                parameters = parameters.flatten()

            weight_matrix = np.reshape(parameters[0:self.num_basis * self.num_links],
                                       [self.num_basis, self.num_links]) * 100
            if self.learn_goal:
                goal_attractor = parameters[-self.num_links:]
            else:
                goal_attractor = self.goal
            self.cost_fn.dmp.set_weights(weight_matrix, goal_attractor)
            # self.dmp.dmp_goal_pos[0] = goal_attractor
        # ref_pos_learned, ref_vel_learned = self.cost_fn.dmp.reference_trajectory(self.cost_fn.t)
        # FIXME: How to ensure goal velocity is reached?
        self.n_evaluations += 1
        rew, succ = self.cost_fn.rollout(num_points_per_link=10, render=plot)
        return np.atleast_1d(rew)

    def save_result(self, filename):
        np.save(filename + "_dmp_weights", self.cost_fn.dmp._dmp_weights)
        np.save(filename + "_dmp_goal", self.cost_fn.dmp._dmp_goal_pos[0])

    def load_result(self, filename):
        # if not self.opt:
        #     with open(filename, 'rb') as f:
        #         self.opt = dill.load(f)
        #
        # if 'more' in filename:
        #     params = self.opt.q.sample(1).flatten()
        #     self.dmp.dmp_weights = params[0:-self.num_links].reshape([self.num_basis, self.num_links])
        #     self.dmp.dmp_goal_pos[0] = params[-self.num_links:]
        # elif 'cma' in filename:
        #     params = self.opt.ask(1)[0]
        #     self.dmp.dmp_weights = params[0:-self.num_links].reshape([self.num_basis, self.num_links])
        #     self.dmp.dmp_goal_pos[0] = params[-self.num_links:]

        w = np.load(filename + "_dmp_weights.npy")
        g = np.load(filename + "_dmp_goal.npy")
        self.cost_fn.dmp.set_weights(w, g)


if __name__ == '__main__':
    nl = 5
    objective = HoleReachingObjective(num_links=nl, allow_self_collision=True)  # , hole_x=1)
    # objective.load_result("/tmp/cma_optimizer.pkl")

    # x_start = np.hstack([1 * np.random.randn(10, nl*5), 2 * np.pi * np.random.rand(10, nl) - np.pi])
    x_start = np.array([[-0.07059705, 2.1287255, 5.35993763, -2.32228719, -0.50191509,
                         0.36346237, 1.02355225, 0.06599974, -0.49766831, -0.56894609,
                         1.2894393, -0.66217232, 0.43598053, 0.12761387, -2.51911921,
                         -1.84101713, 0.13423536, -1.35846347, 0.32241375, 2.37876331,
                         -0.51732188, -0.05723916, -0.49908854, 0.15258334, 0.40831075,
                         1.94612141, -1.1411417, -0.60913849, -1.02754119, -0.56151009]])

    for i in range(100):
        # objective.load_result("/tmp/cma")
        rew = objective(parameters=x_start, plot=False)  # , parameters=x_start[i])

        print(rew)
