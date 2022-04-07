import numpy as np
import matplotlib.pyplot as plt
from flocking import Flock

space_dimension = 2
number_agent = 30
scale_factor = 100
number_iteration = 300
snap = 50
dt = 0.009
Pos_X = np.zeros([number_agent, number_iteration])
Pos_Y = np.zeros([number_agent, number_iteration])
POS = np.zeros((number_agent, number_iteration, 2))
position = np.random.rand(number_agent, space_dimension) * 40
velocity = np.zeros([number_agent, space_dimension])
gamma_agent = np.array([150, 150])
flocking = Flock(space_dimension,
                 number_agent,
                 position,
                 velocity,
                 gamma_agent)

potential = []


# for z in z_value:
# potential.append(flocking.get_potential(z))
# plt.plot(z_value, potential)
# plt.show()

def init_figure(xmin, xmax, ymin, ymax):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto', autoscale_on=True)
    ax.xmin = xmin
    ax.xmax = xmax
    ax.ymin = ymin
    ax.ymax = ymax
    clear()
    return ax


def clear():
    plt.pause(0.02)
    plt.cla()
    # ax.set_xlim(ax.xmin,ax.xmax)
    # ax.set_ylim(ax.ymin,ax.ymax)


ax = init_figure(-50, 50, -50, 50)
plt.plot(position[:, 0], position[:, 1], 'ro')

for i in range(number_agent):
    ax.annotate(i, (position[i, 0], position[i, 1]))

for k in range(number_iteration):
    clear()
    print(k)
    if k == 0:
        for i in range(number_agent):
            Pos_X[i, k] = position[i, 0]
            Pos_Y[i, k] = position[i, 1]
            POS[i, k] = position[i, :]
    else:
        for i in range(number_agent):
            u_i = flocking.consensus(i)
            # print("U", i, ':', u_i)
            _velocity = velocity[i, :]
            # print("Velocity", _velocity)
            _position = position[i, :]
            # print("Position", _position)
            new_velocity = _velocity + u_i * dt
            new_position = _position + new_velocity * dt + (dt ** 2 / 2) * u_i
            [Pos_X[i, k], Pos_Y[i, k]] = new_position
            POS[i, k] = new_position
            velocity[i, :] = new_velocity
            position[i, :] = new_position

        # if (k + 1) % snap == 0:
    for i in range(number_agent):
        for j in range(number_agent):
            distance = np.linalg.norm(position[j] - position[i])
            if distance <= flocking.R:
                plt.plot([position[i, 0], position[j, 0]],
                         [position[i, 1], position[j, 1]],
                         'b-', lw=1)
    plt.plot(position[:, 0], position[:, 1], 'ro')
    for i in range(number_agent):
        ax.annotate(i, (position[i, 0], position[i, 1]))
        # plt.show()

# def show_graph():
#     """
#     :return:
#     """
#     phi_alpha = []
#     z_x = []
#     z_value = np.linspace(0, 20, num=500)
#     potential = []
#     a_ij = []
#     for z in z_value:
#         a_ij.append(flocking.bump_function(flocking.sigma_norm(z) / flocking.sigma_norm(flocking.R)))
#     plt.plot(z_value, a_ij, color="black")
#     plt.show()
#     # for z in z_value:
#     #     # z_x.append(flocking.sigma_norm(z))
#     #     phi_alpha.append(flocking.get_phi_alpha(z))
#     #     potential.append(flocking.get_potential(z))
#     #     sigma_r = flocking.sigma_norm(flocking.R)
#     #     bump_param = z / sigma_r
#     #     a_ij.append(flocking.bump_function(bump_param))
#     #
#     # plt.plot(z_value, phi_alpha, color='green')
#     # plt.plot(z_value, potential, color='black')
#     # plt.plot(z_value, a_ij, color='blue')
#     # plt.show()
#     # for j in range(number_agent):
#     #     plt.figure(j+2)
#     #     for i in range(number_iteration):
#     #         z_x.append(flocking.sigma_norm(POS[3, i] - POS[j, i]))
#     #         phi_alpha.append(flocking.get_phi_alpha(flocking.sigma_norm(POS[3, i] - POS[j, i])))
#     #         potential.append(flocking.get_potential(flocking.sigma_norm(POS[3, i] - POS[j, i])))
#     #         a_ij.append(flocking.bump_function(flocking.sigma_norm(POS[3, i] - POS[j, i]) / flocking.R))
#     #
#     #     plt.plot(z_x, phi_alpha, color='green')
#     #     plt.plot(z_x, potential, color='black')
#     #     plt.plot(z_x, a_ij, color='red')
#     #     # plt.plot(flocking.value_d, flocking.distance_m)
#     #     plt.show()
#
#     #print(phi_alpha)
