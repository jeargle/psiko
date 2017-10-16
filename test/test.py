# John Eargle
# 2017


import numpy as np
import matplotlib.pyplot as plt

import psiko.psiko as pk


def square_comp_test1():
    omega = 2
    x = np.linspace(0,1,500)
    y = np.array([pk.square_comp(x, omega, k) for k in range(1,6)])
    for i in range(5):
        plt.plot(x,[y[:i,j].sum() for j in range(500)])
    plt.show()


def square_test1():
    omega=2
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

    time_plot_test1()
    boundary_cond_test1()
