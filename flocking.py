import numpy as np
import scipy.integrate as integrate


class Flock:
    """
    This class implements the flocking algorithm according to the olfati method
    """

    def __init__(self,
                 space_dimension,
                 number_agent,
                 position,
                 velocity,
                 target=None):
        self.m = space_dimension
        self.n = number_agent
        self.EPSILON = 0.1
        self.A = 5
        self.B = 5
        self.C = np.abs(self.A - self.B) / np.sqrt(4 * self.A * self.B)
        self.d = 15  # desired distance between two agents
        self.d_prime = 10  # desired distance between agents and obstacles
        self.k = 1.2  # scalar factor for alpha, alpha
        self.k_prime = self.k  # scalar factor for alpha, beta
        self.R = self.k * self.d  # The maximum distance between agents under which two agents are considered neighbors
        self.r_prime = self.k_prime * self.R  # interaction ranges of alpha, beta agents
        self.q = position  # Position configuration of all nodes
        self.p = velocity  # velocity of each node
        self.gamma_agent = target
        self.gamma_agent_vel = np.array([0, 0])
        self.value_d = []
        self.value_g = []
        self.distance_m = []
        # ---- Positive constants for ui command
        self.c1_alpha = 20
        # self.c2_alpha = 1
        self.c2_alpha = 2 * np.sqrt(self.c1_alpha)
        self.c1_gamma = 1.1
        self.c2_gamma = 2 * np.sqrt(self.c1_gamma)
        self.c1_beta = 1500
        self.c2_beta = 2 * np.sqrt(self.c1_beta)
        # ---- Params for bump functions
        self.h_alpha = 0.2
        self.h_beta = 0.2
        self.d_beta = self.sigma_norm(self.d_prime)
        self.q_i_k = []

    def set_adj_matrix(self):
        """
        :return:
        """
        adj_matrix = np.zeros([self.n, self.n])
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    distance = np.linalg.norm(self.q[i] - self.q[j])
                    if distance <= self.R:
                        adj_matrix[i, j] = 1
        return adj_matrix

    def sigma_norm(self, z):
        """
        :param z:
        :return:
        """
        norm_z = np.linalg.norm(z)
        square = np.sqrt(1 + self.EPSILON * (norm_z ** 2))
        z_sigma = (square - 1) / self.EPSILON
        return z_sigma

    def sigma_epsilon(self, z):
        """
        :param z:
        :return:
        """
        sig_eps = z / (1 + self.EPSILON * self.sigma_norm(z))
        return sig_eps

    def sigma_1(self, z):
        """
        :param z:
        :return:
        """
        sig_1 = z / np.sqrt(1 + z ** 2)
        return sig_1

    def bump_function(self, z, h):
        """
        :param h:
        :param z:
        :return:
        """
        if 0 <= z < h:
            bump = 1
        elif h <= z < 1:
            frac = (z - h) / (1 - h)
            bump = (1 + np.cos(np.pi * frac)) / 2
        else:
            bump = 0
        return bump

    def get_a_ij(self, i, j):
        """
        :param i:
        :param j:
        :return:
        """
        sigma_qij = self.sigma_norm(self.q[j] - self.q[i])
        sigma_r = self.sigma_norm(self.R)
        bump_param = sigma_qij / sigma_r
        a_ij = self.bump_function(bump_param, self.h_alpha)
        return a_ij

    def get_phi(self, z):
        """
        """
        sigma_1 = self.sigma_1(z + self.C)
        phi = ((self.A + self.B) * sigma_1 + (self.A - self.B)) / 2
        return phi

    def get_phi_alpha(self, z):
        """
        :param z:
        :return:
        """
        phi_return = self.get_phi(z - self.sigma_norm(self.d))
        bump_return = self.bump_function(z / self.sigma_norm(self.R), self.h_alpha)
        phi_alpha = bump_return * phi_return
        return phi_alpha

    def get_phi_beta(self, z):
        """
        :param z:
        :return:
        """
        bump_return = self.bump_function(z / self.d_beta, self.h_beta)
        phi_beta = (self.sigma_1(z - self.d_beta) - 1) * bump_return
        return phi_beta

    def get_potential(self, z):
        """
        :return:
        """
        d_alpha = self.sigma_norm(self.d)
        potential = integrate.quad(self.get_phi_alpha, d_alpha, z)
        return potential

    def get_n_ij(self, i=None, j=None):
        """
        :return:
        """
        n_ij = self.sigma_epsilon(self.q[j] - self.q[i])
        return n_ij

    def consensus(self, i):
        """
        :param i:
        :return:
        """
        sum_1 = np.zeros(self.m, dtype=float)
        sum_2 = np.zeros(self.m, dtype=float)
        for j in range(self.n):
            if i != j:
                distance = np.linalg.norm(self.q[j] - self.q[i])
                if distance <= self.R:
                    sig_norm = self.sigma_norm(self.q[j] - self.q[i])
                    phi_alpha = self.get_phi_alpha(sig_norm)
                    value_1 = phi_alpha * self.get_n_ij(i, j)
                    sum_1 += value_1
                    velocity = self.p[j] - self.p[i]
                    value_2 = self.get_a_ij(i, j) * velocity
                    # if i == 2 and j == 3:
                    #     self.distance_m.append(sig_norm)
                    #     self.value_d.append(value_1)
                    #     self.value_g.append(value_1)
                    sum_2 += value_2
        u_i = self.c1_alpha * sum_1 + self.c2_alpha * sum_2
        return u_i

    def group_objective(self, i):
        """
        :param i:
        :return:
        """
        u_i = self.consensus(i) - self.c1_gamma * (self.q[i] - self.gamma_agent) - self.c2_gamma * (
                self.p[i] - self.gamma_agent_vel)
        return u_i

    def obstacle_avoiding(self, i, r_k, y_k, type_obs, walls_ak, old_position):
        """
        :return:
        """
        # Calculation of u_alpha_i
        sum_1 = np.zeros(self.m, dtype=float)
        sum_2 = np.zeros(self.m, dtype=float)
        for j in range(self.n):
            if i != j:
                distance = np.linalg.norm(self.q[j] - self.q[i])
                if distance <= self.R:
                    sig_norm = self.sigma_norm(self.q[j] - self.q[i])
                    phi_alpha = self.get_phi_alpha(sig_norm)
                    value_1 = phi_alpha * self.get_n_ij(i, j)
                    sum_1 += value_1
                    velocity = self.p[j] - self.p[i]
                    value_2 = self.get_a_ij(i, j) * velocity
                    sum_2 += value_2
        u_alpha_i = self.c1_alpha * sum_1 + self.c2_alpha * sum_2

        # Calculation of u_beta_i
        sum_3 = np.zeros(self.m, dtype=float)
        sum_4 = np.zeros(self.m, dtype=float)
        q_i_k, p_i_k = None, None
        I = np.identity(2)
        for k in range(len(y_k)):
            if type_obs[k] == 'round':
                mu = r_k[k] / np.linalg.norm(old_position - y_k[k])
                a_k = (old_position - y_k[k]) / np.linalg.norm(old_position - y_k[k])
                # print("old", old_position)
                # print("a_k", a_k)
                p = I - np.dot(a_k, a_k.T)
                # print("p", p)
                p_i_k = np.dot(p, self.p[i])
                q_i_k = mu * old_position + (1 - mu) * y_k[k]

            elif type_obs[k] == 'wall':
                vec = np.array([walls_ak[k-3]]).T
                a_k = vec / np.linalg.norm(vec)
                p = I - np.dot(a_k, a_k.T)
                p_i_k = np.dot(p, np.array(self.p[i]))
                p_i_k = p_i_k.flatten()
                q_i_k = np.dot(p, np.array(old_position)) + np.dot((I - p), np.array(y_k[k]))
                q_i_k = q_i_k.flatten()
            distance_obs = np.linalg.norm(q_i_k - old_position)

            if distance_obs < self.r_prime:
                self.q_i_k.append(q_i_k)
                phi_beta = self.get_phi_beta(self.sigma_norm(q_i_k - old_position))
                n_i_k = self.sigma_epsilon(q_i_k - old_position)
                value_3 = phi_beta * n_i_k
                sum_3 += value_3
                b_i_k = self.bump_function(self.sigma_norm(q_i_k - old_position) / self.d_beta, self.h_beta)
                value_4 = b_i_k * (p_i_k - self.p[i])

        u_beta_i = self.c1_beta * sum_3 + self.c2_beta * sum_4

        # Calculation of u_gamma_i
        u_gamma_i = - self.c1_gamma * (self.q[i] - self.gamma_agent) \
                    - self.c2_gamma * (self.p[i] - self.gamma_agent_vel)

        return u_alpha_i + u_beta_i + u_gamma_i

    def get_q_i_k(self):
        return self.q_i_k
