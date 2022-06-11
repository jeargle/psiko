# John Eargle
# 2017-2022

import numpy as np
import scipy as sp
from scipy.integrate import simps, quad, nquad
from scipy.special import (
    factorial, eval_genlaguerre, expi, sph_harm
)

__all__ = ["square_comp", "square", "square2", "force1", "repulsion",
           "boundary_1d", "pib_ti_1D", "pib_td_1D", "pib_wave_solution",
           "pib_energy", "square_function", "projection_integrand",
           "time_plot", "traj_plot"]



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
        return simps(y.real, x) + simps(y.imag, x) * 1j

def complex_nquad(func, ranges, **kwargs):
    """
    Complex quadrature.
    """
    real_integral = nquad(
        lambda *args: sp.real(func(*args)), ranges, **kwargs
    )
    imag_integral = nquad(
        lambda *args: sp.imag(func(*args)), ranges, **kwargs
    )
    return (real_integral[0] + imag_integral[0] * 1j,
            real_integral[1] + imag_integral[1] * 1j)


# ====================
# Wavefunction Functions
# ====================

def prob_density(psi):
    """
    Probability density for a normalized wavefunction.
    """
    return (np.conjugate(psi) * psi).real

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
    diff = np.zeros(len(y), dtype=complex)

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

def kinetic_mat_operator(x, dx=None, m=1, h_bar=1):
    """
    Kinetic energy matrix operator.
    """
    if dx is None:
        dx = x[1]-x[0]

    t = -(h_bar**2) / (2*m*(dx**2))
    T = np.zeros(len(x)**2).reshape(len(x),len(x))

    for i, pos in enumerate(x):
        # diagonal elements
        T[i][i] = -2*t
        # side diagonal elements with edge cases i==0 and i==n-1
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

def build_hamiltonian(x, vx, dx=None, m=1.0, h_bar=1.0):
    """
    Get kinetic energy, potential energy, and hamiltonian
    """
    if dx is None:
        dx = x[1]-x[0]
    T = kinetic_mat_operator(x, dx)
    V = np.diag(vx)
    H = T + V

    return H

def coulomb_double_well(x, r):
    """
    Build Coulomb-like double well potential centered around 0.
    """
    x0_1 = -r / 2.0  # first well
    x0_2 = r / 2.0   # second well

    return np.array([
        -1.0 / np.abs(xi-x0_1)
        if xi <= 0
        else -1.0 / np.abs(xi-x0_2)
        for xi in x
    ])

def coulomb_well(x, x0=0.0):
    """
    Build Coulomb-like well potential at x0.
    """
    return -1.0 / np.abs(x-x0)

def square_barrier(x, length=1.0, height=9.0, x0=4.0):
    """
    Build square barrier potential at x0.
    """
    result = np.zeros(len(x))
    for i, xval in enumerate(x):
        if xval >= x0 and xval <= x0 + length:
            result[i] = height

    return result

def transmission_probability(pdf, n_cutoff=300):
    """
    """
    return (2.0 / (1.0 + np.mean(pdf[-n_cutoff:]))).real

def complex_plane_wave(x, energy, mass=1.0, hbar=1.0):
    """
    Complex plane wave.
    """
    k = np.sqrt(2 * mass * energy / hbar**2)
    psi = np.zeros(len(x), dtype=complex)
    psi = np.exp(-1j * k * x)

    return psi

def eval_expectation(psi, x, dx, operator):
    """
    """
    integrand = np.conjugate(psi) * operator(psi, x, dx)
    exp = complex_simps(integrand, x)
    exp = 0.0 if np.abs(exp) < 1e-7 else exp

    return exp

def tunnel_finite_diff(x, psi_x, v_x, E):
    """
    tunnel function

    return time-propagated wavefunction
    """
    psi_new = np.copy(psi_x)
    for i in range(1, len(x)-1):
        dx = x[i+1]-x[i]
        psi_new[i+1] = (2.0 + (2.0*dx**2)*(v_x[i]-E))*psi_new[i] - psi_new[i-1]

    return psi_new


class Eigenstate(object):
    """
    Component of a wavefunction Psi.
    """

    def __init__(self, y, energy, mix_coeff=1.0, hbar=1.0, quantum_numbers=None):
        """
        y:
        energy: energy of this Eigenstate
        mix_coeff: mixture coefficient for Eigenstate's contribution to Psi
        hbar: Plank's constant
        """
        self.y = y
        self.energy = energy
        self.mix_coeff = mix_coeff
        self.hbar = hbar
        self.quantum_numbers = quantum_numbers

    def _cnt_evolve(self, t):
        """
        Time-evolve a complex, time-dependent coefficient.
        """
        return np.exp(-1j * self.energy * t / self.hbar)

    def at_time(self, t):
        return self.mix_coeff * self._cnt_evolve(t) * self.y


_wf_type = {
    'position': 1,
    'momentum': 2,
    'energy': 3,
}

_wf_type_name = {val: key for key, val in _wf_type.items()}

class Psi(object):
    """
    Wavefunction evaluated at discrete points x.
    """

    def __init__(self, x, y, dx=None,
                 wf_type=_wf_type['position'], normalize=True):
        """
        x: wavefunction domain
        y: wavefunction values at time 0
        dx: distance between points of x
        wf_type: wavefunction type
        normalize: whether or not to normalize the wavefunction
        """
        self.x = x
        self.y = y

        self.dx = dx
        if len(x) > 1:
            if self.dx is None:
                self.dx = x[1] - x[0]
            if normalize:  # Do not normalize if there is only one point.
                self._normalize()

        self.wf_type = wf_type

    def prob_density(self):
        """
        Probability density for the wavefunction.
        """
        return (np.conjugate(self.y) * self.y).real

    def psi_norm(self):
        """
        Norm of the wavefunction.
        """
        result = simps(prob_density(self.y), self.x)
        return np.sqrt(result)

    def _normalize(self):
        """
        Normalize the wavefunction.
        """
        self.y = self.y / self.psi_norm()

    def expectation(operator):
        """
        Expectation value for an operator on this wavefunction.

        operator: operator function to take the expectation for
        """
        integrand = np.conjugate(self.y) * operator(self.y)
        exp = complex_simps(integrand, self.x)
        exp = 0.0 if np.abs(exp) < 1e-7 else exp

        return exp

    def eigenfunction(self):
        raise NotImplementedError()

    def energy(self):
        raise NotImplementedError()


class PsiTraj(object):
    """
    Trajectory of a wavefunction evaluated at discrete points x and times t.
    """

    def __init__(self, psi, time, dt=None, wave_soln=None, **values):
        self.psi = psi
        self.time = time
        self.traj = np.zeros((len(self.psi.x), len(self.time)))

        self.dt = dt
        if self.dt is None:
            self.dt = self.time[1] - self.time[0]

        for step, t in enumerate(time):
            self.traj[:, step] = wave_soln(self.psi, t, **values)


# ====================
# Helper Functions
# ====================

def square_function(x, l):
    """
    Square wave 0->1->0 in the middle third of the length l.

    x:
    l:
    """
    if x < (1.0/3)*l or x > (2.0/3)*l:
        return 0.0
    else:
        return 1.0

def projection_integrand(x, n, l):
    """
    x:
    n:
    l:
    """
    return (np.sqrt(2.0/l) *
            np.sin(n*np.pi*x/l) *
            square_function(x,l))

def gaussian_x(x, sigma):
    """
    Position gaussian.

    x:
    sigma:
    """
    return ( np.exp(-(x**2 / (2 * sigma**2))) /
             (np.sqrt(sigma * np.sqrt(np.pi))) )

def gaussian_p(p, sigma):
    """
    Momentum gaussian.

    p:
    sigma:
    """
    return ( (np.sqrt(sigma) * np.exp(-(p**2 * sigma**2)/2)) /
             (np.sqrt(np.sqrt(np.pi))) )

def x_int(x, sigma):
    """
    Gaussian position integrand.
    """
    gx = gaussian_x(x, sigma)
    return gx * x * gx

def x2_int(x, sigma):
    """
    Gaussian position-squared integrand.
    """
    gx = gaussian_x(x, sigma)
    return gx * x**2 * gx

def p_int(p, sigma):
    """
    Gaussian momentum integrand.
    """
    gp = gaussian_p(p, sigma)
    return gp * p * gp

def p2_int(p, sigma):
    """
    Gaussian momentum-squared integrand.
    """
    gp = gaussian_p(p, sigma)
    return gp * p**2 * gp

def mu_operator(phi, theta, mu):
    """
    phi:
    theta:
    mu:
    """
    return mu * (np.sin(theta)*np.cos(phi) +
                 np.sin(theta)*np.sin(phi) +
                 np.cos(theta))

def dipole_moment_integrand(phi, theta, mu, l1, m1, l2, m2, real=False):
    """
    phi:
    theta:
    mu:
    l1:
    m1:
    l2:
    m2:
    real:
    """
    mu_op = mu_operator(phi, theta, mu)
    Y_l1m1 = sph_harm(m1, l1, phi, theta)
    Y_l2m2 = sph_harm(m2, l2, phi, theta)

    if real:
        return (Y_l1m1 * mu_op * Y_l2m2 * np.sin(theta)).real

    return Y_l1m1 * mu_op * Y_l2m2 * np.sin(theta)

def dipole_moment_superposition_integrand(phi, theta, mu, c1, c2, l1, m1, l2, m2, real=False):
    """
    phi:
    theta:
    mu:
    c1:
    c2:
    l1:
    m1:
    l2:
    m2:
    real:
    """
    mu_op = mu_operator(phi, theta, mu)
    Y_l1m1 = sph_harm(m1, l1, phi, theta)
    Y_l2m2 = sph_harm(m2, l2, phi, theta)
    Y_lm = c1*Y_l1m1 + c2*Y_l2m2

    if real:
        return (np.conjugate(Y_lm) * mu_op * Y_lm * np.sin(theta)).real

    return np.conjugate(Y_lm) * mu_op * Y_lm * np.sin(theta)

def hartrees_to_wavelength(energy):
    """
    Get wavelength (nm) corresponding to a given energy (Hartrees).

    energy: energy in Hartrees
    """

    return np.abs(45.56 * 1.0 / energy)
