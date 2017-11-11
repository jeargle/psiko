# John Eargle
# 2017


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
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



# ====================
# Harmonic Oscillator
# ====================


def harmonic_oscillator_1D(x, n, mass=1.0, omega=1.0, hbar=1.0):
    """
    Harmonic Oscillator
    """
    prefactor = (1.0 / (np.sqrt(2**n * misc.factorial(n))) *
                 (mass * omega / (np.pi * hbar))**(1.0/4.0))
    gaussian = np.exp(-(mass * omega * x * x) / (2.0 * hbar))

    coeff_grid = np.sqrt(mass * omega / hbar)
    coeff = np.zeros(n + 1)
    coeff[n] = 1.0
    hermite = np.polynomial.hermite.hermval(coeff_grid * x, coeff)

    return prefactor * gaussian * hermite


def harmonic_oscillator_1D_in_field(x, t, omega_f, omega_0=1, lam=1, E_0=1.0, mass=1.0, hbar=1.0):
    """
    Time-dependent solution to ground state Harmonic Oscillator in a sinusoidal
    potential solved using perturbation theory.  Used for system bathed in EM
    field (spectroscopy).

    omega_0: frequency of harmonic oscillator
    omega_f: frequency of incoming field
    """
    omega_diff = omega_0 - omega_f
    omega_sum = omega_0 + omega_f
    c1 = ( ((1j*E_0*(2.0*np.pi/lam)) / (2.0*np.sqrt(2.0*mass*hbar*omega_0))) *
           ( ((np.exp(-1j*omega_diff*t) - 1.0) / omega_diff) +
             ((np.exp(1j*omega_sum*t) - 1.0) / omega_sum) ) )
    psi0 = harmonic_oscillator_1D(x, 0)
    psi1 = harmonic_oscillator_1D(x, 1)

    return psi0 + c1*psi1


def harmonic_potential_2D(xx, yy, kx, ky, x0=0, y0=0):
    return 0.5*kx*(xx-x0)**2 + 0.5*ky*(yy-y0)**2


def harmonic_oscillator_2D(xx, yy, l, m, mass=1.0, omega=1.0, hbar=1.0):
    """
    Harmonic Oscillator
    """
    # Prefactor for the HO eigenfunctions
    prefactor = ( ((mass*omega) / (np.pi*hbar))**(1.0/2.0) /
                  (np.sqrt(2**l * 2**m * misc.factorial(l) * misc.factorial(m))) )

    # Gaussian for the HO eigenfunctions
    gaussian = np.exp(-(mass * omega * (xx**2 + yy**2)) / (2.0*hbar))

    # Hermite polynomial setup
    coeff_grid = np.sqrt(mass * omega / hbar)
    # coeff_l = np.zeros((l+1, ))
    coeff_l = np.zeros(l+1)
    coeff_l[l] = 1.0
    # coeff_m = np.zeros((m+1, ))
    coeff_m = np.zeros(m+1)
    coeff_m[m] = 1.0
    # Hermite polynomials for the HO eigenfunctions
    hermite_l = np.polynomial.hermite.hermval(coeff_grid * xx, coeff_l)
    hermite_m = np.polynomial.hermite.hermval(coeff_grid * yy, coeff_m)

    # The eigenfunction is the product of all of the above.
    return prefactor * gaussian * hermite_l * hermite_m



# ====================
# Helper Functions
# ====================


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


def plot_surface(xx, yy, zz):
    """
    Plot a surface.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, rstride=10, cstride=10, alpha=0.25)
    # Set viewing angle
    ax.view_init(30, -60)
    # ax.view_init(80, -60)
    # Add contour lines
    ax.contour(xx, yy, zz, zdir='z')
    plt.xlabel('x')
    plt.ylabel('y')

    return


def plot_contours(xx, yy, zz, vmin=None, vmax=None):
    """
    Plot a heatmap and contours for a 3d surface.
    """
    # Use viridis colormap
    cmap = mpl.cm.get_cmap('viridis')
    plt.contour(xx, yy, zz, linewidths=3.0, cmap=cmap)
    contour = plt.contour(xx, yy, zz, colors='k', linewidths=0.5)
    im = plt.imshow(zz, interpolation='nearest',
                    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    vmin=vmin, vmax=vmax, aspect='auto')
    plt.clabel(contour, fontsize='x-small', inline=1)
    _add_colorbar(im, label='z')
    plt.xlabel('x')
    plt.ylabel('y')

    return


def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0/aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes('right', size=width, pad=pad)
    plt.sca(current_ax)

    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
