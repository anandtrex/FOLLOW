import numpy as np


def simulate(arm, time_span, dt, control, q0=None, dq0=None):
    n_steps = int(time_span / dt)
    if q0 is None:
        q0 = np.zeros(arm.dim_config_space)
    if dq0 is None:
        dq0 = np.zeros(arm.dim_config_space)
    q = np.zeros((n_steps + 1, *q0.shape))
    dq = np.zeros((n_steps + 1, *dq0.shape))
    q[0] = q0
    dq[0] = dq0
    for i in range(1, int(time_span / dt) + 1):
        res = arm.evolveFns(q[i - 1], dq[i - 1], control[i - 1], dt=dt)
        q[i] = res[0] * dt + q[i - 1]
        dq[i] = res[1] * dt + dq[i - 1]
    return q[1:], dq[1:]


def _check_simulate():
    import george
    from arm_2link_todorov_gravity import Arm
    arm = Arm()
    q0, dq0 = (np.zeros(arm.dim_config_space) for _ in range(2))
    simulation_time_span = 10.
    dt = .001
    t = np.arange(0, simulation_time_span, dt)
    kernel = george.kernels.ExpSquaredKernel(.5)
    np.random.seed(3000)
    gp = george.GP(kernel)
    u = gp.sample(t[::10], arm.dim_config_space) * 4.
    u = np.array([np.interp(t, t[::10], u[i]) for i in range(u.shape[0])])
    # time major
    u = u.T

    import matplotlib.pyplot as plt
    n_arms = 7
    lengths = np.linspace(.3, 1.6, n_arms)
    fig, axes = plt.subplots(n_arms + 1, figsize=(16, 10), sharex=True)
    axes[0].plot(u)
    axes[0].set_ylabel('control torques')
    for i in range(n_arms):
        arm.l1 = lengths[i]
        rq, drq = simulate(arm, simulation_time_span, dt, u, q0, dq0)
        axes[i + 1].plot(rq)
        axes[i + 1].set_ylabel('arm configuration $l_1 = {}$'.format(arm.l1))
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    _check_simulate()
