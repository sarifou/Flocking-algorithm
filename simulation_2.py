import numpy as np
import matplotlib.pyplot as plt
from flocking import Flock

space_dimension = 2
number_agent = 30
scale_factor = 100
number_iteration = 300
snap = 50
dt = 0.02
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


ax = init_figure(-50, 50, -50, 50)
plt.plot(gamma_agent[0], gamma_agent[1], 'ro', color='green')
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
            _velocity = velocity[i, :]
            _position = np.array([Pos_X[i, k - 1],
                                  Pos_Y[i, k - 1]])
            u_i = flocking.group_objective(i)
            new_position = _position + _velocity * dt + (dt ** 2 / 2) * u_i
            new_velocity = (new_position - _position) / dt
            [Pos_X[i, k], Pos_Y[i, k]] = new_position
            POS[i, k] = new_position
            velocity[i, :] = new_velocity
            position[i, :] = new_position

    for i in range(number_agent):
        for j in range(number_agent):
            distance = np.linalg.norm(position[j] - position[i])
            if distance <= flocking.R:
                plt.plot([position[i, 0], position[j, 0]],
                         [position[i, 1], position[j, 1]],
                         'b-', lw=1)
    plt.plot(gamma_agent[0], gamma_agent[1], 'ro', color='green')
    plt.plot(position[:, 0], position[:, 1], 'ro')
    for i in range(number_agent):
        ax.annotate(i, (position[i, 0], position[i, 1]))
