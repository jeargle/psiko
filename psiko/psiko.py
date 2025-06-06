# John Eargle

import numpy as np
import scipy as sp
from scipy.integrate import simps, quad, nquad
from scipy.special import (
    factorial, eval_genlaguerre, expi, sph_harm
)

__all__ = ["square_comp", "square", "square2", "force1", "repulsion",
           "boundary_1d", "square_function", "projection_integrand"]



def square_comp(x, omega, k):
    """
    Single component of square function.

    x:
    omega:
    k:
    """
    return (4.0/np.pi) * np.sin(2*np.pi*(2*k-1)*omega*x)/(2*k-1)

def square(x, omega, n):
    """
    Approximate a square wave with sin() series.

    x:
    omega:
    n:
    """
    a = np.array([square_comp(x, omega, k) for k in range(1,n+1)])
    sum = np.zeros(len(x))
    for i in a:
        sum += i
    return sum

def square2(x, omega, n):
    """
    Approximate a square wave with sin() series.

    x:
    omega:
    n:
    """
    a = np.array([square_comp(x, omega, k) for k in range(1,n+1)])
    return np.array([a[:n,i].sum() for i in range(len(x))])

def force1(ti, m):
    """
    Time dependent force calculation.

    ti:
    n:
    """
    return 5.0*np.sin(2*ti)/m

def repulsion(r1, r2):
    """
    r1:
    r2:
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

    y:
    x:
    """
    if np.all(np.isreal(y)):
        return simps(y, x) + 0j
    else:
        return simps(y.real, x) + simps(y.imag, x) * 1j

def complex_nquad(func, ranges, **kwargs):
    """
    Complex quadrature.

    func:
    ranges:
    kwargs:
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

def cnt_evolve(cn_0, t, E_n, hbar=1.0):
    """
    Time-evolve a complex, time-dependent coefficient.

    cn_0:
    t: time
    E_n:
    hbar: Planck's constant
    """
    return cn_0 * np.exp(-1j*E_n*t/hbar)

def kinetic_mat_operator(x, dx=None, m=1, h_bar=1):
    """
    Kinetic energy matrix operator.

    hbar: Planck's constant
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

    x:
    """
    b = 2
    return b*x

def build_hamiltonian(x, vx, dx=None, m=1.0, h_bar=1.0):
    """
    Get kinetic energy, potential energy, and hamiltonian

    x:
    vx:
    dx:
    m: mass
    hbar: Planck's constant
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

    x:
    r:
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

    x:
    x0:
    """
    return -1.0 / np.abs(x-x0)

def square_barrier(x, length=1.0, height=9.0, x0=4.0):
    """
    Build square barrier potential at x0.

    x:
    length:
    height:
    x0:
    """
    result = np.zeros(len(x))
    for i, xval in enumerate(x):
        if xval >= x0 and xval <= x0 + length:
            result[i] = height

    return result

def transmission_probability(pdf, n_cutoff=300):
    """
    pdf:
    n_cutoff:
    """
    return (2.0 / (1.0 + np.mean(pdf[-n_cutoff:]))).real

def complex_plane_wave(x, energy, mass=1.0, hbar=1.0):
    """
    Complex plane wave.

    x:
    energy:
    mass: mass
    hbar: Planck's constant
    """
    k = np.sqrt(2 * mass * energy / hbar**2)
    psi = np.zeros(len(x), dtype=complex)
    psi = np.exp(-1j * k * x)

    return psi

def tunnel_finite_diff(x, psi_x, v_x, E):
    """
    tunnel function

    x:
    psi_x:
    v_x:
    E:

    return time-propagated wavefunction
    """
    psi_new = np.copy(psi_x)
    for i in range(1, len(x)-1):
        dx = x[i+1]-x[i]
        psi_new[i+1] = (2.0 + (2.0*dx**2)*(v_x[i]-E))*psi_new[i] - psi_new[i-1]

    return psi_new


# ====================
# Wavefunction Classes
# ====================

class Eigenstate(object):
    """
    Component of a wavefunction Psi.
    """

    def __init__(self, y, energy, mix_coeff=1.0, hbar=1.0, quantum_numbers=None):
        """
        y:
        energy: energy of this Eigenstate
        mix_coeff: mixture coefficient for Eigenstate's contribution to Psi
        hbar: Planck's constant
        quantum_numbers: quantum numbers that define this Eigenstate
        """
        self.y = y
        self.energy = energy
        self.mix_coeff = mix_coeff
        self.hbar = hbar
        self.quantum_numbers = quantum_numbers

        print(f'*** Eigenstate {self.quantum_numbers["n"]} ***')
        print(f'  energy: {self.energy}')
        print(f'  mixture coefficient: {self.mix_coeff}')
        print(f'  hbar: {self.hbar}')

    def _coeff_evolve(self, t):
        """
        Time-evolve a complex, time-dependent coefficient.

        t: time; number or array
        """
        # print(np.exp(-1j * self.energy * t / self.hbar))
        return np.exp(-1j * self.energy * t / self.hbar)

    def at_time(self, t):
        """
        Time-evolve a complex, time-dependent coefficient.

        t: time; number or array
        """
        return self.mix_coeff * self._coeff_evolve(t) * self.y


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

    def __init__(self, length=None, num_points=None, dx=None, x_left=None, wf_type=_wf_type['position'],
                 normalize=True, hbar=1.0, eigenstate_params=None):
        """
        length: length of domain
        num_points: number of points to track in domain (x)
        dx: distance between points of x
        wf_type: wavefunction type
        normalize: whether or not to normalize the wavefunction
        hbar: Planck's constant
        eigenstate_params: list of parameters for eigenstates
        """
        raise NotImplementedError()

    def prob_density(self, t=0.0):
        """
        Probability density for the wavefunction.

        t: time
        """
        raise NotImplementedError()

    def psi_norm(self, t=0.0):
        """
        Norm of the wavefunction.

        t: time
        """
        raise NotImplementedError()

    def _normalize(self):
        """
        Normalize the wavefunction by scaling the mixture
        coefficients of the eigenstates.
        """
        raise NotImplementedError()

    def position(self, y):
        """
        Position operator

        y:
        """
        raise NotImplementedError()

    def momentum(self, y):
        """
        Momentum operator

        y:
        """
        raise NotImplementedError()

    def expectation(self, operator, t=0.0):
        """
        Expectation value for an operator on this wavefunction.

        operator: operator function to take the expectation for
        t: time
        """
        raise NotImplementedError()

    def eigenfunction(self):
        raise NotImplementedError()

    def energy(self):
        raise NotImplementedError()

    def at_time(self, t):
        """
        t: time
        """
        raise NotImplementedError()


class Psi1D(Psi):
    """
    1D wavefunction evaluated at discrete points x.
    """

    def __init__(self, length=None, num_points=None, dx=None, x_left=None, wf_type=_wf_type['position'],
                 normalize=True, hbar=1.0, eigenstate_params=None):
        """
        length: length of domain
        num_points: number of points to track in domain (x)
        dx: distance between points of x
        wf_type: wavefunction type
        normalize: whether or not to normalize the wavefunction
        hbar: Planck's constant
        eigenstate_params: list of parameters for eigenstates
        """
        self.length = length

        if x_left is None:
            self.x_left = 0
        else:
            self.x_left = x_left

        if num_points is not None:
            # self.x = np.linspace(0, self.length, num_points)
            self.x = np.linspace(self.x_left, self.length+self.x_left, num_points)
        elif dx is not None:
            self.dx = dx
            # self.x = np.arange(0, self.length, dx)
            self.x = np.arange(self.x_left, self.length+self.x_left, dx)

        self.wf_type = wf_type
        self.hbar = hbar

        self._init_eigenstates(
            eigenstate_params,
            normalize
        )

    def _init_eigenstates(self, eigenstate_params, normalize):
        self.eigenstates = []

        if eigenstate_params is None or len(eigenstate_params) <= 0:
            raise ValueError('Must provide list of eigenstate parameters: eigenstate_params')

        # Validate mixture coefficients.
        mix_coeff_sum = sum(ep.get('mix_coeff', 1.0) for ep in eigenstate_params)
        print(f'sum(mix_coeff): {mix_coeff_sum}')
        mix_coeff_sum = sum(ep.get('mix_coeff', 1.0)**2 for ep in eigenstate_params)
        print(f'sum(mix_coeff**2): {mix_coeff_sum}')

        # Build eigenstates.
        for ep in eigenstate_params:
            n = ep['quantum_numbers']['n']
            self.eigenstates.append(
                Eigenstate(
                    self.eigenfunction(n),
                    self.energy(n),
                    mix_coeff=ep.get('mix_coeff', 1.0),
                    hbar=self.hbar,
                    quantum_numbers=ep.get('quantum_numbers', None)
                )
            )

        if len(self.x) > 1:
            if normalize:  # Do not normalize if there is only one point.
                self._normalize()

    def prob_density(self, t=0.0):
        """
        Probability density for the wavefunction.

        t: time
        """
        y = self.at_time(t)
        return (np.conjugate(y) * y).real

    def psi_norm(self, t=0.0):
        """
        Norm of the wavefunction.

        t: time
        """
        result = simps(self.prob_density(t), self.x)
        return np.sqrt(result)

    def _normalize(self):
        """
        Normalize the wavefunction by scaling the mixture
        coefficients of the eigenstates.
        """
        psi_norm = self.psi_norm()
        for eigenstate in self.eigenstates:
            eigenstate.mix_coeff /= psi_norm

    def position(self, y):
        """
        Position operator

        y:
        """
        return self.x * y

    def momentum(self, y):
        """
        Momentum operator

        y:
        """
        return -1j * self.hbar * finite_diff(y, self.dx)

    def expectation(self, operator, t=0.0):
        """
        Expectation value for an operator on this wavefunction.

        operator: operator function to take the expectation for
        t: time
        """
        y = self.at_time(t)
        integrand = np.conjugate(y) * operator(y)
        exp = complex_simps(integrand, self.x)
        exp = 0.0 if np.abs(exp) < 1e-7 else exp

        return exp

    def eigenfunction(self):
        raise NotImplementedError()

    def energy(self):
        raise NotImplementedError()

    def at_time(self, t):
        """
        t: time
        """
        return sum(eigenstate.at_time(t)
                   for eigenstate in self.eigenstates)


class Psi2D(Psi):
    """
    2D wavefunction evaluated at discrete points (x, y).
    """

    def __init__(self, x_length=None, x_num_points=None, dx=None, x_left=None,
                 y_length=None, y_num_points=None, dy=None, y_left=None,
                 wf_type=_wf_type['position'],
                 normalize=True, hbar=1.0, eigenstate_params=None):
        """
        x_length: length of domain
        x_num_points: number of points to track in domain (x)
        dx: distance between points of x
        wf_type: wavefunction type
        normalize: whether or not to normalize the wavefunction
        hbar: Planck's constant
        eigenstate_params: list of parameters for eigenstates
        """
        # self.length = length

        # if x_left is None:
        #     self.x_left = 0
        # else:
        #     self.x_left = x_left

        # if num_points is not None:
        #     # self.x = np.linspace(0, self.length, num_points)
        #     self.x = np.linspace(self.x_left, self.length+self.x_left, num_points)
        # elif dx is not None:
        #     self.dx = dx
        #     # self.x = np.arange(0, self.length, dx)
        #     self.x = np.arange(self.x_left, self.length+self.x_left, dx)

        # self.wf_type = wf_type
        # self.hbar = hbar

        # self._init_eigenstates(
        #     eigenstate_params,
        #     normalize
        # )
        pass

    def _init_eigenstates(self, eigenstate_params, normalize):
        # self.eigenstates = []

        # if eigenstate_params is None or len(eigenstate_params) <= 0:
        #     raise ValueError('Must provide list of eigenstate parameters: eigenstate_params')

        # # Validate mixture coefficients.
        # mix_coeff_sum = sum(ep.get('mix_coeff', 1.0) for ep in eigenstate_params)
        # print(f'sum(mix_coeff): {mix_coeff_sum}')
        # mix_coeff_sum = sum(ep.get('mix_coeff', 1.0)**2 for ep in eigenstate_params)
        # print(f'sum(mix_coeff**2): {mix_coeff_sum}')

        # # Build eigenstates.
        # for ep in eigenstate_params:
        #     n = ep['quantum_numbers']['n']
        #     self.eigenstates.append(
        #         Eigenstate(
        #             self.eigenfunction(n),
        #             self.energy(n),
        #             mix_coeff=ep.get('mix_coeff', 1.0),
        #             hbar=self.hbar,
        #             quantum_numbers=ep.get('quantum_numbers', None)
        #         )
        #     )

        # if len(self.x) > 1:
        #     if normalize:  # Do not normalize if there is only one point.
        #         self._normalize()
        pass

    def prob_density(self, t=0.0):
        """
        Probability density for the wavefunction.

        t: time
        """
        # y = self.at_time(t)
        # return (np.conjugate(y) * y).real
        pass

    def psi_norm(self, t=0.0):
        """
        Norm of the wavefunction.

        t: time
        """
        # result = simps(self.prob_density(t), self.x)
        # return np.sqrt(result)
        pass

    def _normalize(self):
        """
        Normalize the wavefunction by scaling the mixture
        coefficients of the eigenstates.
        """
        # psi_norm = self.psi_norm()
        # for eigenstate in self.eigenstates:
        #     eigenstate.mix_coeff /= psi_norm
        pass

    def position(self, y):
        """
        Position operator

        y:
        """
        # return self.x * y
        pass

    def momentum(self, y):
        """
        Momentum operator

        y:
        """
        # return -1j * self.hbar * finite_diff(y, self.dx)
        pass

    def expectation(self, operator, t=0.0):
        """
        Expectation value for an operator on this wavefunction.

        operator: operator function to take the expectation for
        t: time
        """
        # y = self.at_time(t)
        # integrand = np.conjugate(y) * operator(y)
        # exp = complex_simps(integrand, self.x)
        # exp = 0.0 if np.abs(exp) < 1e-7 else exp

        # return exp
        pass

    def eigenfunction(self):
        raise NotImplementedError()

    def energy(self):
        raise NotImplementedError()

    def at_time(self, t):
        """
        t: time
        """
        # return sum(eigenstate.at_time(t)
        #            for eigenstate in self.eigenstates)
        pass


class PsiTraj(object):
    """
    Trajectory of a wavefunction or its pdf evaluated at discrete points.
    """

    def __init__(self, psi, time, dt=None, pdf=False):
        """
        psi: Psi
        time: time
        dt: distance between time points
        pdf: create probability density function instead of wavefunction
        """
        self.psi = psi
        self.time = time
        self.traj = np.zeros((len(self.psi.x), len(self.time)), dtype=complex)

        self.dt = dt
        if self.dt is None:
            self.dt = self.time[1] - self.time[0]

        self.pdf = pdf

        if self.pdf:
            for step, t in enumerate(time):
                self.traj[:, step] = self.psi.prob_density(t)
        else:
            for step, t in enumerate(time):
                self.traj[:, step] = self.psi.at_time(t)


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


# --------------------
# Old
# --------------------

# DEPRECATED
def position_operator(psi, x, dx, hbar=1.0):
    """
    Position operator.
    """
    return x*psi

# DEPRECATED
def momentum_operator(psi, x, dx, hbar=1.0):
    """
    Momentum operator.
    """
    return -1j * hbar * finite_diff(psi, dx)

# DEPRECATED
def eval_expectation(psi, x, dx, operator):
    """
    """
    integrand = np.conjugate(psi) * operator(psi, x, dx)
    exp = complex_simps(integrand, x)
    exp = 0.0 if np.abs(exp) < 1e-7 else exp

    return exp

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

# DEPRECATED
def psi_norm(x, psi):
    """
    Norm of a wavefunction.
    """
    result = simps(prob_density(psi), x)
    return np.sqrt(result)
