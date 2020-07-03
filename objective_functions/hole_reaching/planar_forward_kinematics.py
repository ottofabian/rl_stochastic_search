import numpy as np
import matplotlib.pyplot as plt


class PlanarForwardKinematics:

    def __init__(self, num_joints):
        self.lengths = np.ones((num_joints, 1))
        self.num_joints = num_joints
        self.offset = np.asarray([0, 0])

    def get_forward_kinematics(self, theta, num_points_per_link=1, num_link=None):
        theta = np.atleast_2d(theta)
        if theta.shape[0] == 1:
            theta = theta.T
        assert theta.shape[0] == self.num_joints

        if num_link is None:
            num_link = self.num_joints

        if num_points_per_link > 1:
            intermediate_points = np.linspace(0, 1, num_points_per_link)
        else:
            intermediate_points = 1

        accumulated_theta = np.cumsum(theta, axis=0)

        endeffector = np.zeros(shape=(num_link, num_points_per_link, 2))

        x = np.cos(accumulated_theta) * self.lengths * intermediate_points
        y = np.sin(accumulated_theta) * self.lengths * intermediate_points

        endeffector[0, :, 0] = x[0, :]
        endeffector[0, :, 1] = y[0, :]

        for i in range(1, num_link):
            endeffector[i, :, 0] = x[i, :] + endeffector[i - 1, -1, 0]
            endeffector[i, :, 1] = y[i, :] + endeffector[i - 1, -1, 1]

        return np.squeeze(endeffector + self.offset)

    def get_task_space_velocity(self, joint_positions, joint_velocities):
        task_space_velocity = np.zeros(np.shape(joint_velocities)[0], 2)
        for i in range(0, joint_velocities):
            J, _ = self.get_jacobian(joint_positions[i, :])
            task_space_velocity[i, :] = np.matmul(J, np.transpose(joint_velocities[i, :]))

    def get_jacobian(self, theta, num_link=None):
        if num_link == None:
            num_link = self.num_joints

        si = self.get_forward_kinematics(theta, num_link)
        J = np.zeros(2, self.num_joints)

        for j in range(0, num_link - 1):
            pj = [0, 0]
            for i in range(0, j):
                pj += self.angles_to_line(theta, i)
            pj = -(si - pj)
            J[0:1, j + 1] = np.asarray([-pj(1), pj(0)])
        return J, si


if __name__ == '__main__':
    num_links = 10
    pfk = PlanarForwardKinematics(num_links)

    fig, ax = plt.subplots()
    plt.xlim(-num_links, num_links), plt.ylim(-num_links, num_links)

    # t = 2 * np.pi * np.random.rand(num_links, 1)
    t = np.zeros(shape=(num_links, 1))
    t[0] = np.pi/2
    # t = np.pi / 180 * np.array([[45], [-45], [-45]])

    for i in range(100):

        fk = pfk.get_forward_kinematics(t, num_points_per_link=2)

        # print(fk)

        ax.plot(fk[:, 0, 0], fk[:, 0, 1], fk[:, 1, 0], fk[:, 1, 1], marker='o')
        plt.pause(0.2)

        ax.clear()
        plt.xlim(-num_links, num_links), plt.ylim(-num_links, num_links)

        t += 0.1 * np.random.randn(*t.shape)
