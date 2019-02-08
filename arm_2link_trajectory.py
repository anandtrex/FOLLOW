import numpy as np
from scipy.interpolate import interp1d


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


def generate_follow_input(Tmax, dt):
    Tperiod = 1.
    inputreduction = 0.3
    reprRadius = 1.0
    N = 4

    heights = np.random.normal(size=(N // 2, int(Tmax / Tperiod) + 1))
    heights = heights / np.linalg.norm(heights, axis=0) / inputreduction
    ## random uniform 'white-noise' with 50 ms steps interpolated
    ##  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    noisedt = 50e-3
    # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
    noiseN = np.random.uniform(-reprRadius / inputreduction, reprRadius / inputreduction,
                               size=(N // 2, int(Tmax / noisedt) + 1))
    noisefunc = interp1d(np.linspace(0, Tmax, int(Tmax / noisedt) + 1), noiseN, kind='linear', \
                         bounds_error=False, fill_value=0., axis=1)
    heightsfunc = interp1d(np.linspace(0, Tmax, int(Tmax / Tperiod) + 1), heights, kind='linear', \
                           bounds_error=False, fill_value=0., axis=1)

    #inpfn = lambda t: noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadius
    inpfn = lambda t: noisefunc(t) + heightsfunc(t) * reprRadius
    ts = np.arange(0, Tmax, dt)

    return inpfn(ts)


def generate_gp_input(Tmax, dt, arm):
    import george
    # q0, dq0 = (np.zeros(arm.dim_config_space) for _ in range(2))
    t = np.arange(0, Tmax, dt)
    kernel = george.kernels.ExpSquaredKernel(.5)
    np.random.seed(3000)
    gp = george.GP(kernel)
    u = gp.sample(t[::10], arm.dim_config_space) * 4.
    u = np.array([np.interp(t, t[::10], u[i]) for i in range(u.shape[0])])
    return u


def _check_simulate():
    from arm_2link_todorov_gravity import Arm
    arm = Arm()
    q0, dq0 = (np.zeros(arm.dim_config_space) for _ in range(2))

    simulation_time_span = 10.  # s
    dt = .001  # s

    u = generate_gp_input(simulation_time_span, dt, arm)
    # u = generate_follow_input(simulation_time_span, dt)
    # time major
    u = u.T

    import matplotlib.pyplot as plt
    n_arms = 7
    # lengths = np.linspace(.3, 1.6, n_arms)
    fig, axes = plt.subplots(n_arms + 1, figsize=(16, 10), sharex=True)
    axes[0].plot(u)
    axes[0].set_ylabel('control torques')
    for i in range(n_arms):
        arm.l1 = np.random.uniform(0.5, 3.)
        arm.l2 = np.random.uniform(0.5, 3.)
        arm.m1 = np.random.uniform(0.5, 3.)
        arm.m2 = np.random.uniform(0.5, 3.)
        rq, drq = simulate(arm, simulation_time_span, dt, u, q0, dq0)
        axes[i + 1].plot(rq)
        axes[i + 1].set_ylabel('$l_1 = {:.2f}$\n$l_2 = {:.2f}$\n$m_1 = {:.2f}$\n$m_2 = {:.2f}$'.format(arm.l1, arm.l2, arm.m1, arm.m2))
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    _check_simulate()
