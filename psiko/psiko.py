# John Eargle
# 2017


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

__all__ = ["square_comp", "square", "square2", "force1", "repulsion",
           "boundary_1d", "pib_ti_1D", "pib_td_1D", "wave_solution",
           "pib_energy", "square_function", "projection_integrand",
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



# ====================
# Wavefunction Functions
# ====================

def prob_density(psi):
    """
    Probability density for a normalized wavefunction.
    """
    return np.conjugate(psi) * psi


def normalize_wfn(x, psi):
    """
    Normalize a wavefunction.
    """
    return psi / psi_norm(x, psi)


def psi_norm(x, psi):
    """
    Norm of a wavefunction.
    """
    result = simps(prob_density(psi), x)
    return np.sqrt(result)


def finite_diff(y, dx):
    """
    Finite difference method.
    """
    diff = np.zeros(len(y))

    # start at step 0
    diff[0] = (y[1] - y[0])/dx

    for i in range(1, len(y)-1):
        diff[i] = (y[i+1] - y[i-1]) / (2*dx)

    # end at step n-1
    diff[-1] = (y[-1] - y[-2])/dx

    return diff


# ====================
# Particle in a Box
# ====================

def pib_ti_1D(x, n, l):
    """
    Normalized energy eigenfunctions to time-independent Particle In a
    Box.
    """
    return np.sqrt(2.0/l) * np.sin(n*np.pi*x/l)


def pib_td_1D(t, c, n, l):
    """
    Time varying prefactor to time-independent Particle In a Box.
    """
    return np.cos(n*np.pi*c*t/l)


def wave_solution(x, t, c, n, l):
    """
    Harmonic solutions to time-dependent Particle In a Box.
    """
    return pib_td_1D(t, c, n, l) * pib_ti_1D(x, n, l)


def pib_energy(n,l, hbar=1, m=1):
    """
    Energy eigenvalues
    """
    return (n**2 * hbar**2 * np.pi**2) / (2.0 * m * l**2)


def square_function(x, l):
    """
    Square wave 0->1->0 in the middle third of the length l.
    """
    if x < (1.0/3)*l or x > (2.0/3)*l:
        return 0.0
    else:
        return 1.0


def projection_integrand(x, n, l):
    """
    """
    return (np.sqrt(2.0/l) *
            np.sin(n*np.pi*x/l) *
            square_function(x,l))



# ====================
# Plotting
# ====================

def time_plot(x, y, t, timestep=1):
    for i in range(0, len(t), timestep):
        plt.plot(x, y[:,i])
    return
