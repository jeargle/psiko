# John Eargle
# 2017


import numpy as np
import matplotlib.pyplot as plt

__all__ = ["square_comp", "square", "square2", "force1", "repulsion",
           "boundary_1d", "pib_ti_1D", "pib_td_1D", "wave_solution",
           "time_plot"]

def square_comp(x, omega, k):
    """
    Single component of square function.
    """
    return (4.0/np.pi) * np.sin(2*np.pi*(2*k-1)*omega*x)/(2*k-1)

def square(x, omega, n):
    """
    Approximate a square wave with sin() series.
    """
    a = np.array([square_comp(x, omega, k) for k in range(1,n+1)])
    sum = np.zeros(len(x))
    for i in a:
        sum += i
    return sum

def square2(x, omega, n):
    """
    Approximate a square wave with sin() series.
    """
    a = np.array([square_comp(x, omega, k) for k in range(1,n+1)])
    return np.array([a[:n,i].sum() for i in range(len(x))])

def force1(ti, m):
    """
    Time dependent force calculation.
    """
    return 5.0*np.sin(2*ti)/m

def repulsion(r1, r2):
    """
    """
    #return 5.0*np.sin(2*ti)/m
    return 5/(abs(r1-r2))


def boundary_1d(xi, v, l):
    """
    Reflective boundary conditions in 1D.
    """
    if xi >= l:
        v=-v
    if xi < -l:
        v=-v
    return v


def pib_ti_1D(x,n,l):
    """
    Harmonic solutions to time-independent Particle In a Box.
    """
    return np.sin(n*np.pi*x/l)


def pib_td_1D(t,c,n,l):
    """
    Time varying prefactor to time-independent Particle In a Box.
    """
    return np.cos(n*np.pi*c*t/l)


def wave_solution(x, t, c, n, l):
    """
    Harmonic solutions to time-dependent Particle In a Box.
    """
    return pib_td_1D(t, c, n, l) * pib_ti_1D(x, n, l)



# ====================
# Plotting
# ====================

def time_plot(x, y, t):
    for i, time in enumerate(t):
        plt.plot(x, y[:,i])
    return
