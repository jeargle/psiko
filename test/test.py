# John Eargle
# 2017


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, quad

import psiko.psiko as pk


def square_comp_test1():
    omega = 2
    x = np.linspace(0,1,500)
    y = np.array([pk.square_comp(x, omega, k) for k in range(1,6)])
    for i in range(5):
        plt.plot(x,[y[:i,j].sum() for j in range(500)])
    plt.show()


def square_test1():
    omega = 2
    x = np.linspace(0,1,500)
    ns = [1, 2, 8, 32, 128, 512]
    y = np.array([pk.square(x, omega, i) for i in ns])
    print('np.shape(y):', np.shape(y))
    y2 = np.array([pk.square2(x, omega, i) for i in ns])
    print('np.shape(y2):', np.shape(y2))
    print('np.abs(y-y2).max():', np.abs(y-y2).max())
    print('np.abs(y-y2).sum():', np.abs(y-y2).sum())
    for i in y:
        plt.plot(x,i)
    plt.show()


def time_plot_test1():
    """
    Simulate motion of a particle in 1D
    """
    v = 1.0
    dt = 0.001
    t = np.arange(0, 1+dt, dt)
    x = np.zeros(len(t))
    x[0] = 0

    for n in range(1, len(t)):
        x[n] = x[n-1] + dt*v

    plt.plot(t, x)
    plt.show()


def boundary_cond_test1():
    """
    Reflective boundary conditions in 1D
    """
    v = 1.75
    dt = 0.001
    t = np.arange(0,1+dt,dt)
    x = np.zeros(len(t))
    x[0] = 0.0
    l = 0.5

    for n in range(1,len(t)):
        x[n] = x[n-1] + dt*v
        # boundary conditions
        if abs(x[n]) >= l:
            v = -v

    plt.plot(t, x)
    plt.show()


def forces_test1():
    """
    External forces on particle in 1D
    """
    v = 2.0
    dt = 0.001
    t = np.arange(0,2+dt,dt)
    x = np.zeros(len(t))
    l = 0.5

    for n in range(1,len(t)):
        # update velocity and position
        v += dt*5.0*np.sin(2*t[n])
        # v += 0.5*np.sin(20*t[n])
        # v -= 0.01
        x[n] = x[n-1] + dt*v
        # boundary conditions
        if abs(x[n]) >= l:
            v = -v

    plt.plot(t, x)
    plt.show()


def forces_test2():
    dt = 0.001
    t = np.arange(0,1+dt,dt)
    v1, v2 = 2.0, -2.0
    m1, m2 = 1, 1
    x1, x2 = np.zeros(len(t)), np.zeros(len(t))
    x1[0], x2[0] = 0.1, -0.1
    t = np.arange(0,1+dt,dt)
    l = 0.5

    for i in range(1,len(t)):
        # particle 1: update velocity then position
        v1 = v1 + dt*pk.force1(t[i-1],m1)
        x1[i]=x1[i-1]+dt*v1
        # particle 1: boundary condition
        v1 = pk.boundary_1d(x1[i], v1, l)

        # particle 2: update velocity than position
        v2 = v2 + dt*pk.force1(t[i-1],m2)
        x2[i]=x2[i-1]+dt*v2
        # particle 2: boundary conditions
        v2 = pk.boundary_1d(x2[i], v2, l)

    plt.plot(t, x1)
    plt.plot(t, x2)
    plt.show()


def forces_test3():
    dt = 0.001
    t = np.arange(0,1+dt,dt)
    v1, v2 = 2.0, -2.0
    m1, m2 = 1, 1
    x1, x2 = np.zeros(len(t)), np.zeros(len(t))
    x1[0], x2[0] = 0.1, -0.1
    t = np.arange(0,1+dt,dt)
    l = 0.5

    # repulsion force array
    repul = np.zeros(len(t))

    #iterate over t array
    for i in range(1,len(t)):
        # particle-particle interaction force
        repul[i] = pk.repulsion(x1[i-1], x2[i-1])

        # particle 1
        v1 = v1 + dt*repul[i]
        x1[i] = x1[i-1] + dt *v1
        v1 = pk.boundary_1d(x1[i], v1, l)

        # particle 2
        v2 = v2 - dt*repul[i]
        x2[i] = x2[i-1] + dt *v2
        v2 = pk.boundary_1d(x2[i], v2, l)

    plt.plot(t, x1)
    plt.plot(t, x2)
    plt.show()

    plt.clf()
    plt.plot(t, repul)
    plt.show()


def pib_test1():
    l = 10
    x = np.linspace(0, l, 1000)
    y = np.zeros(3*len(x)).reshape(3, len(x))

    # calculate 4 harmonics
    for n in range(1,4):
        y[n-1] = pk.pib_ti_1D(x, n, l)

    for i in y:
        plt.plot(x, i)
    plt.legend()
    plt.show()


def pib_test2():
    l = 10.0
    c = 0.1
    t = np.linspace(0, 100, 1000)
    y = np.zeros(4*len(t)).reshape(4, len(t))

    # use a for loop for the multiple harmonics
    for n in range(1,5):
        y[n-1] = pk.pib_td_1D(t, c, n, l)

    for i in y:
        plt.plot(t, i)
    plt.legend()
    plt.show()


def pib_test3():
    c = 0.1
    l = 10
    x = np.arange(0, l, 0.01)
    t = np.arange(0, 30, 0.1)
    y = np.zeros((len(x), len(t)))
    n = 3

    for step, time in enumerate(t):
        # time-dependent and time-independent terms
        # y[:, step] = pk.pib_td_1D(time, c, n, l) * pk.pib_ti_1D(x, n, l)
        y[:, step] = pk.wave_solution(x, time, c, n, l)

    pk.time_plot(x, y, t)
    plt.show()


def pib_interference_test1():
    """
    Plot time trace for a single position in a wavefunction made from
    the first two eigenstates.
    """
    t = np.arange(0, 100, 0.1)
    c = 0.5
    l = 10
    x = l/3.0
    wave = np.zeros(len(t))

    # sum 2 eigenstates
    for step, time in enumerate(t):
        wave[step] = (pk.wave_solution(x, time, c, 1, l) +
                      pk.wave_solution(x, time, c, 2, l))

    plt.plot(t, wave)
    plt.show()


def pib_interference_test2():
    """
    Plot time evolution of full wavefunction made from the first two
    eigenstates.
    """
    c = 0.5
    l = 10
    x = np.linspace(0, l, 100)
    t = np.arange(0, 30, 0.1)
    y = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    for step, time in enumerate(t):
        y[:, step] = (pk.wave_solution(x, time, c, 1, l) +
                      pk.wave_solution(x, time, c, 2, l))

    pk.time_plot(x, y, t, timestep=1)
    plt.show()


def quadrature_test1():
    l = 10.0
    x = np.arange(0, l, 0.01)
    c = np.zeros(10)

    for n in range(0, 10):
        c[n], _ = quad(pk.projection_integrand, 0.0, l, args=(n, l))

    plt.plot(range(0, 10), c)
    plt.show()


def quadrature_test2():
    """
    Wave decomposition
    """
    l = 10.0
    x = np.arange(0, l, 0.01)
    y = np.array([pk.square_function(xi, l) for xi in x])
    square_approx = np.zeros(len(x))

    for n in range(10):
        # project amplitudes and integrate components
        cn, _ = quad(pk.projection_integrand, 0.0, l, args=(n, l))
        square_approx += cn * np.sqrt(2.0/l) * np.sin(n*np.pi*x/l)

    plt.plot(x, square_approx)
    plt.plot(x, [pk.square_function(i, l) for i in x])
    plt.show()


def normalize_test1():
    """
    Set up a mixed state from the ground and first three excited states
    for the particle in a box.  Normalize the wavefunction.
    """
    l = 10.0
    # l = 1.0
    x = np.arange(0, l, 0.01)
    psi_x = np.zeros(len(x))

    # Build wavefunction from 4 eigenfunctions
    for n in range(1,5):
        psi_x += pk.pib_ti_1D(x, n, l)

    # Get PDF and normalize for psi_x
    pdf = pk.prob_density(psi_x)
    psi_normed = pk.normalize_wfn(x, psi_x)

    norm_pre = pk.psi_norm(x, psi_x)
    norm_post = pk.psi_norm(x, psi_normed)
    print('Norm pre:', norm_pre)
    print('Norm post:', norm_post)

    plt.plot(x, psi_x)
    plt.plot(x, pdf)
    plt.show()


def schroedinger_test1():
    """
    Set up a mixed state from the ground and first excited states for
    the particle in a box.  Normalize the wavefunction.
    """
    l = 10
    x = np.arange(0, l, 0.01)

    # First eigenstate
    psi1_x = pk.pib_ti_1D(x, 1, l)
    c1 = 1.0/np.sqrt(2)
    E1 = pk.pib_energy(1, l)

    # Second eigenstate
    psi2_x = pk.pib_ti_1D(x, 2, l)
    c2 = 1.0/np.sqrt(2)
    E2 = pk.pib_energy(2, l)

    # Mixed state
    psi0 = c1*psi1_x + c2*psi2_x
    psi0_norm = pk.psi_norm(x, psi0)
    print('Norm is ', psi0_norm)

    plt.plot(x, psi0)
    plt.show()


def schroedinger_test2():
    """
    Time-dependent Schroedinger equation.
    """
    l = 10
    x = np.arange(0, l, 0.01)
    t = np.linspace(0, 50, 100)
    psi = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    # First eigenstate
    c1_0 = 1/np.sqrt(2)
    psi1_x = pk.pib_ti_1D(x, 1, l)
    E1 = pk.pib_energy(1, l)

    # Second eigenstate
    c2_0 = 1/np.sqrt(2)
    psi2_x = pk.pib_ti_1D(x, 2, l)
    E2 = pk.pib_energy(2, l)

    for step, time in enumerate(t):
        # Get time evolved coefficients
        c1 = pk.cnt_evolve(c1_0, time, E1)
        c2 = pk.cnt_evolve(c2_0, time, E2)
        psi[:, step] = c1*psi1_x + c2*psi2_x

    pk.time_plot(x, psi, t)
    plt.show()



if __name__=='__main__':

    print '*******************'
    print '*** PSIKO TESTS ***'
    print '*******************'

    # ====================
    # Square Wave tests
    # ====================

    # square_comp_test1()
    # square_test1()

    # ====================
    # 1D Classical Particle tests
    # ====================

    # time_plot_test1()
    # boundary_cond_test1()
    # forces_test1()
    # forces_test2()
    # forces_test3()

    # ====================
    # 1D Quantum Particle tests
    # ====================

    # pib_test1()
    # pib_test2()
    # pib_test3()
    # pib_interference_test1()
    # pib_interference_test2()
    # quadrature_test1()
    # quadrature_test2()
    # normalize_test1()
    schroedinger_test1()
    schroedinger_test2()
