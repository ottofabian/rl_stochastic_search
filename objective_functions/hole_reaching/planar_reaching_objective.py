from objective_functions.hole_reaching.mp_lib import ExpDecayPhaseGenerator
from objective_functions.hole_reaching.mp_lib import DMPBasisGenerator
from objective_functions.hole_reaching.mp_lib import dmps
from experiments.robotics import planar_forward_kinematics as pfk
import numpy as np
import matplotlib.pyplot as plt


def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


class ReachingTask:
    def __init__(self, num_links, via_points=()):
        self.num_links = num_links
        self.via_points = via_points
        self.goal_point = np.array((num_links, 0))
        self.pfk = pfk.PlanarForwardKinematics(num_joints=num_links)

    def rollout(self, trajectory, num_points_per_link, plot=False):
        # trajectory should be [num_time_steps, num_joints]

        acc = np.sum(np.diff(trajectory, n=2, axis=0) ** 2)
        total_number_of_points_collided = 0
        self.end_effector_points = []

        distance = 0

        if plot:
            fig, ax = plt.subplots()
            plt.xlim(-self.num_links, self.num_links), plt.ylim(-self.num_links, self.num_links)

        for t, traj in enumerate(trajectory):
            line_points_in_taskspace = self.pfk.get_forward_kinematics(traj[:, None],
                                                                       num_points_per_link=num_points_per_link)

            endeffector = line_points_in_taskspace[-1, -1, :]

            for vp in self.via_points:
                if t == vp['t']:
                    distance += np.abs(np.linalg.norm(endeffector - np.array(vp["vp"]))) ** 2

            self.end_effector_points.append(line_points_in_taskspace[-1, -1, :])

            is_collided = self.check_collision(line_points_in_taskspace)

            if plot:
                ax.clear()
                plt.xlim(-self.num_links, self.num_links), plt.ylim(-self.num_links, self.num_links)
                ax.plot(line_points_in_taskspace[:, 0, 0],
                        line_points_in_taskspace[:, 0, 1],
                        line_points_in_taskspace[:, -1, 0],
                        line_points_in_taskspace[:, -1, 1], marker='o')

                for vp in self.via_points:
                    ax.scatter(vp["vp"][0], vp["vp"][1], c="r", marker="x")

                plt.pause(0.1)

            if is_collided:
                break

        # check the distance the endeffector travelled to the center of the hole
        # end_effector_travel = np.sum(
        #     np.sqrt(np.sum(np.diff(np.stack(end_effector_points), axis=0)[:, 4, :] ** 2, axis=1, keepdims=True))) ** 2
        # end_effector_travel = np.sum(np.sqrt(np.sum(np.diff(np.stack(end_effector_points), axis=0) ** 2, axis=2)))

        # check distance of endeffector to bottom center of hole
        endeffector = line_points_in_taskspace[-1, -1, :]
        # roughly normalized to be between 0 and 1
        distance += np.abs(np.linalg.norm(endeffector - self.goal_point)) ** 2  # / (self.num_links + np.abs(self.hole_x))


        # TODO: tune factors
        # distance in [0, 1]
        # |acc| in [0, 0.1]
        out = 1 * distance \
            + 100 * np.abs(acc) \
            + is_collided * 100000
            # + 0.1 * total_number_of_points_collided\
            # + 0.01 * end_effector_travel ** 2

        return np.atleast_1d(out)

    def check_collision(self, line_points):

        for i, line1 in enumerate(line_points):
            for line2 in line_points[i+2:, :, :]:
                # if line1 != line2:
                if intersect(line1[0], line1[1], line2[0], line2[1]):
                    return True

        return False

    def plot_trajectory(self, trajectory):

        fig, ax = plt.subplots()
        plt.xlim(-self.num_links, self.num_links), plt.ylim(-1, self.num_links)

        for t in trajectory:
            fk = self.pfk.get_forward_kinematics(t, num_points_per_link=2)

            # print(fk)

            ax.plot(fk[:, 0, 0], fk[:, 0, 1], fk[:, 1, 0], fk[:, 1, 1], marker='o')

            # Add the patch to the Axes
            plt.pause(0.1)

            ax.clear()
            plt.xlim(-self.num_links, self.num_links), plt.ylim(-1, self.num_links)


class ReachingObjective:
    def __init__(self, num_links=5, num_basis=5, via_points=None, dmp_weights=None):
        self.num_links = num_links
        self.d = num_links * num_basis
        self.f_opt = 0
        # create task
        self.task = ReachingTask(num_links=num_links,
                                 via_points=via_points)

        # use 5 basis functions per dof
        self.num_basis = num_basis
        self.t = np.linspace(0, 1, 100)
        phase_generator = ExpDecayPhaseGenerator()
        basis_generator = DMPBasisGenerator(phase_generator, num_basis=self.num_basis)

        self.dmp = dmps.DMP(num_dof=num_links,
                            basis_generator=basis_generator,
                            phase_generator=phase_generator
                            )

        # self.dmp.dmp_beta_x = 0
        self.dmp.dmp_start_pos = np.zeros((1, num_links))
        self.dmp.dmp_start_pos[0, 0] = np.pi / 2
        self.dmp.dmp_goal_pos = np.zeros((1, num_links))
        self.dmp.dmp_weights = dmp_weights if dmp_weights is not None else np.random.normal(0.0, 10.0, (num_basis, num_links))

    def __call__(self, parameters=None, plot=False):
        if parameters is not None:
            if len(parameters.shape) > 1:
                assert parameters.shape[0] == 1
                parameters = parameters.flatten()

            weight_matrix = np.reshape(parameters, [self.num_basis, self.num_links])
            self.dmp.dmp_weights = weight_matrix
        ref_pos_learned, ref_vel_learned = self.dmp.reference_trajectory(self.t)
        # FIXME: How to ensure goal velocity is reached?
        return self.task.rollout(ref_pos_learned, num_points_per_link=2, plot=plot)

    def save_result(self, filename):
        np.save(filename + "_dmp_weights", self.dmp.dmp_weights)

    def load_result(self, filename):
        self.dmp.dmp_weights = np.load(filename + "_dmp_weights.npy")


if __name__ == '__main__':
    nl = 5
    objective = ReachingObjective(num_links=nl, via_points=({"t": 50, "vp": (1, 1)}, ))  # , hole_x=1)
    # objective.load_result("/tmp/sac")

    x_start = 1 * np.random.randn(10, nl*5)

    for i in range(1):
        rew = objective(plot=True)  # , parameters=x_start[i])

        print(rew)
