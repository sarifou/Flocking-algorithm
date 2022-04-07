import numpy as np
import matplotlib.pyplot as plt
from flocking import Flock

# Variables
space_dim = 2
number_agent = 10
scale_factor = 100
number_iteration = 1000
snap = 50
dt = 0.009
position = np.random.rand(number_agent, space_dim) * 40
velocity = np.zeros([number_agent, space_dim])
Pos_X = np.zeros([number_agent, number_iteration])
Pos_Y = np.zeros([number_agent, number_iteration])
gamma_agent = np.array([500, 20])
obstacles = np.array([[125, 25],
                      [300, 10],
                      [125, -20],
                      [-40, 50],
                      [-40, -50],
                      [510, 50]])
                      # [510, 30],
                      # [510, 20],
                      # [510, 10],
                      # [510, 0],
                      # [510, -10],
                      # [510, -20],
                      # [510, -30],
                      # [510, -40]])
type_obs = ['round', 'round', 'round', 'wall', 'wall', 'wall', 'wall', 'wall', 'wall', 'wall', 'wall', 'wall']
R_k = np.array([10, 10, 10])
a_k = np.array([[0, -1],
                [0, 1],
                [1, 0]])
flocking = Flock(space_dim,
                 number_agent,
                 position,
                 velocity,
                 gamma_agent)


def init_figure():
    fig, axs = plt.subplots()
    # ax = fig.add_subplot(111, autoscale_on=True)
    return axs


def clear():
    plt.pause(0.02)
    plt.cla()


ax = init_figure()
ax.plot(gamma_agent[0], gamma_agent[1], 'ro', color='green')
ax.plot(position[:, 0], position[:, 1], 'ro')

# Plot obstacles
for i in range(len(obstacles)):
    if type_obs[i] == 'round':
        ax.add_artist(plt.Circle((obstacles[i, 0], obstacles[i, 1]), R_k[i], color='black'))

# Plot walls
ax.add_patch(plt.Rectangle((-40.0, -50.0), 550, 100, edgecolor="blue", linewidth=5, facecolor='none'))
for i in range(number_agent):
    ax.annotate(i, (position[i, 0], position[i, 1]))
ax.axis('equal')
ax.set_title('Obstacle avoiding', fontsize=10)
ax.set(xlim=(0, 600))

for k in range(number_iteration):
    clear()
    print(k)
    if k == 0:
        for i in range(number_agent):
            Pos_X[i, k] = position[i, 0]
            Pos_Y[i, k] = position[i, 1]
    else:
        for i in range(number_agent):
            _velocity = velocity[i, :]
            _position = np.array([Pos_X[i, k - 1],
                                  Pos_Y[i, k - 1]])
            u_i = flocking.obstacle_avoiding(i, R_k, obstacles, type_obs, a_k, _position)
            new_position = _position + _velocity * dt + (dt ** 2 / 2) * u_i
            new_velocity = (new_position - _position) / dt
            [Pos_X[i, k], Pos_Y[i, k]] = new_position
            velocity[i, :] = new_velocity
            position[i, :] = new_position

    for i in range(number_agent):
        for j in range(number_agent):
            distance = np.linalg.norm(position[j] - position[i])
            if distance <= flocking.R:
                ax.plot([position[i, 0], position[j, 0]],
                         [position[i, 1], position[j, 1]],
                         'b-', lw=1)
    ax.plot(gamma_agent[0], gamma_agent[1], 'ro', color='green')
    ax.plot(position[:, 0], position[:, 1], 'ro')

    for i in range(len(obstacles)):
        if type_obs[i] == "round":
            ax.add_artist(plt.Circle((obstacles[i, 0], obstacles[i, 1]), R_k[i], color='black'))

    ax.add_patch(plt.Rectangle((-40.0, -50.0), 550, 100, edgecolor="black", linewidth=5, facecolor='none'))
    for i in range(number_agent):
        ax.annotate(i, (position[i, 0], position[i, 1]))
    ax.axis('equal')
    ax.set_title('Obstacle avoiding', fontsize=10)
    ax.set(xlim=(-45, 560))

# print(flocking.q_i_k)
for el in flocking.q_i_k:
    ax.plot(el[0], el[1], 'ro')
plt.show()
