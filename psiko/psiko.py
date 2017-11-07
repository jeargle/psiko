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


def complex_simps(y, x):
    """
    Complex Simpson's rule.
    """
    if np.all(np.isreal(y)):
        return simps(y, x) + 0j
    else:
        return simps(y.real, x) + 1j * simps(y.imag, x)



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


# def finite_diff(psi):
#     """
#     Finite difference method.
#     """
#     diff = np.zeros(len(psi.x))

#     # start at step 0
#     diff[0] = (psi.x[1] - psi.x[0])/psi.dx

#     for i in range(1, len(psi.x)-1):
#         diff[i] = (psi.x[i+1] - psi.x[i-1]) / (2*psi.dx)

#     # end at step n-1
#     diff[-1] = (psi.x[-1] - psi.x[-2])/psi.dx

#     return diff


def cnt_evolve(cn_0, t, E_n, hbar=1.0):
    """
    Time-evolve a complex, time-dependent coefficient.
    """
    return cn_0 * np.exp(-1j*E_n*t/hbar)


def position_operator(psi, x, dx, hbar=1.0):
    """
    Position operator.
    """
    return x*psi


def momentum_operator(psi, x, dx, hbar=1.0):
    """
    Momentum operator.
    """
    return -1j * hbar * finite_diff(psi, dx)


# def momentum_operator(psi, hbar=1.0):
#     """
#     Momentum operator.
#     """
#     return -1j * hbar * finite_diff(psi)


def kinetic_mat_operator(x, dx, m=1, h_bar=1):
    """
    Kinetic energy matrix operator.
    """
    t = -(h_bar**2) / (2*m*(dx**2))
    T = np.zeros(len(x)**2).reshape(len(x),len(x))

    for i, pos in enumerate(x):
        # consider first diagonal elements
        T[i][i] = -2*t
        # then side diagonal elements, consider edge cases (i=0) or (i=n-1)
        if i==0:
            T[i][i+1] = t
        elif i==len(x)-1:
            T[i][i-1] = t
        else:
            T[i][i-1] = t
            T[i][i+1] = t
    return T


def linear_ramp(x):
    """
    Linear potential.
    """
    b = 2
    return b*x


def eval_expectation(psi, x, dx, operator):
    """
    """
    integrand = np.conjugate(psi) * operator(psi, x, dx)
    exp = complex_simps(integrand, x)
    exp = 0.0 if np.abs(exp) < 1e-7 else exp

    return exp


_wf_type = {
    'position': 1,
    'momentum': 2,
    'energy': 3,
}

_wf_type_name = {val: key for key, val in _wf_type.items()}

class Psi(object):

    def __init__(self, x, dx, psi, wf_type=_wf_type['position']):
        self.x = x
        self.dx = dx
        self.psi = psi
        self.wf_type = wf_type
        self._normalize()

    def prob_density(self):
        """
        Probability density for the wavefunction.
        """
        return np.conjugate(self.psi) * self.psi

    def psi_norm(self):
        """
        Norm of the wavefunction.
        """
        result = simps(prob_density(self.psi), self.x)
        return np.sqrt(result)

    def _normalize(self):
        """
        Normalize the wavefunction.
        """
        self.psi = self.psi / self.psi_norm()
        return

    def expectation(operator):
        """
        Expectation value for an operator on this wavefunction.
        """
        integrand = np.conjugate(self.psi) * operator(self.psi)
        exp = complex_simps(integrand, self.x)
        exp = 0.0 if np.abs(exp) < 1e-7 else exp

        return exp


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


def pib_energy(n, l, hbar=1.0, m=1):
    """
    Energy eigenvalues
    """
    return (n**2 * hbar**2 * np.pi**2) / (2.0 * m * l**2)


def pib_superposition(x, t, l, n1, n2):
    # First eigenstate
    psi1 = pib_ti_1D(x, n1, l)
    c1_0 = 1.0/np.sqrt(2) + 0.0j
    E1 = pib_energy(n1, l)

    # Second eigenstate
    psi2 = pib_ti_1D(x, n2, l)
    c2_0 = 1.0/np.sqrt(2) + 0.0j
    E2 = pib_energy(n2, l)

    psi = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    for step, time in enumerate(t):
        # Get time evolved coefficients
        c1 = cnt_evolve(c1_0, time, E1)
        c2 = cnt_evolve(c2_0, time, E2)
        psi[:, step] = c1*psi1 + c2*psi2

    return psi


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
