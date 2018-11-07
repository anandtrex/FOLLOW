import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import george

# from arm_2link_todorov import armAngles, armXY, evolveFns
from arm_2link_todorov_gravity import Arm
from plot_utils import beautify_plot, axes_labels


q0 = np.zeros(2)
dq0 = np.zeros(2)
u0 = np.zeros(2)
dt = .01


arm = Arm(l1=.7)


def roll_out():
    q = q0.copy()
    dq = dq0.copy()
    u = u0.copy()
    u[0] = .1
    for i in range(10):
        q, dq = arm.evolveFns(q, dq, u, dt)
        print(q)


label_font_size = 12
fig = plt.figure(facecolor='w',figsize=(3, 3),dpi=300)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.65, 1.65), ylim=(-1.65, 1.65), clip_on=False)
line_ref, = ax.plot([], [], 'o-r', lw=2, clip_on=False)
line_prediction, = ax.plot([], [], 'o-b', lw=2, clip_on=False)
time_text = ax.text(0.2, 0.78, '', transform=ax.transAxes, fontsize=label_font_size)
beautify_plot(ax, x0min=False, y0min=False, xticks=[], yticks=[], drawxaxis=False, drawyaxis=False)
axes_labels(ax, '', '$\longleftarrow$ gravity', xpad=-20)
ax.text(0.45, 0.86, 'Acrobot', transform=fig.transFigure)


def init():
    line_ref.set_data([], [])
    line_prediction.set_data([], [])
    time_text.set_text('')
    return line_ref, line_prediction, time_text


q = q0.copy()
dq = dq0.copy()
T = 10.
t = np.arange(0, T, dt)
kernel = george.kernels.ExpSquaredKernel(.5)
np.random.seed(3000)
gp = george.GP(kernel)
u = gp.sample(t, 2) * 4.
dt_animation = .01
n_steps_per_dt = int(dt_animation / dt)


def simulate(T, dt, u, q0, dq0):
    n_steps = int(T / dt)
    q = np.zeros((n_steps + 1, *q0.shape))
    dq = np.zeros((n_steps + 1, *dq0.shape))
    q[0] = q0
    dq[0] = dq0
    for i in range(1, int(T / dt) + 1):
        res = arm.evolveFns(q[i - 1], dq[i - 1], u[i - 1], dt=dt)
        q[i] = res[0] * dt + q[i - 1]
        dq[i] = res[1] * dt + dq[i - 1]
    return q[1:], dq[1:]


rq, drq = simulate(T, dt, u.T, q0, dq0)


fig, axes = plt.subplots(4, figsize=(16, 8), sharex=True)

axes[0].plot(u.T)
axes[1].plot(rq)
axes[2].plot(drq)
axes[3].plot(np.array([arm.armXY(a) for a in rq])[:, :2])

fig.tight_layout()
plt.show()


def animate(i):
    for j in range(n_steps_per_dt):
        res = arm.evolveFns(q, dq, u[:, i * n_steps_per_dt + j], dt=dt)
        q[:] += res[0] * dt
        dq[:] += res[1] * dt
    x0, y0, x1, y1 = arm.armXY(q)
    line_ref.set_data([[0, y0, y1], [0, -x0, -x1]])
    time_text.set_text('time%.2f; X,Y=%.2f,%.2f' % ((i+1) * dt, y1, -x1))
    return line_ref, line_prediction, time_text


# bug with blit=True with default tkAgg backend
#  see https://github.com/matplotlib/matplotlib/issues/4901/
# install python-qt4 via apt-get and set QT4Agg backend; as at top of this file
anim = animation.FuncAnimation(fig, animate,
                               init_func=init, frames=int(T / dt_animation), interval=30, blit=True, repeat=False)

plt.show()


# if __name__ == '__main__':
#     roll_out()
