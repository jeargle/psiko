# John Eargle
# 2017


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from scipy import misc
from scipy.integrate import simps, quad, nquad
from scipy.special import sph_harm, eval_genlaguerre

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
    return 2.0 / (1.0 + np.mean(pdf[-n_cutoff:]))


def complex_plane_wave(x, energy, mass=1.0, hbar=1.0):
    """
    Complex plane wave.
    """
    k = np.sqrt(2 * mass * energy / hbar**2)
    psi = np.zeros(len(x), dtype=np.dtype(complex))
    psi = np.exp(-1j * k * x)

    return psi


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

    psi = np.zeros(len(x)*len(t), dtype=complex).reshape(len(x), len(t))

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
    c1 = excited_overlap(t, omega_f, omega_0, lam, E_0, mass, hbar)
    psi0 = harmonic_oscillator_1D(x, 0)
    psi1 = harmonic_oscillator_1D(x, 1)

    return psi0 + c1*psi1


def excited_overlap(t, omega_f, omega_0=1, lam=1, E_0=1.0, m=1.0, hbar=1.0):
    """
    Overlap of the ground state and first excited states for 1D
    time-dependent Harmonic Oscillator.
    """
    omega_diff = omega_0 - omega_f
    omega_sum = omega_0 + omega_f
    c1 = ((1j*E_0*(2.0*np.pi/lam))/(2.0*np.sqrt(2.0*m*hbar*omega_0))) * \
        ( ((np.exp(-1j*omega_diff*t) - 1.0) / omega_diff) + \
        ((np.exp(1j*omega_sum*t) - 1.0) / omega_sum))
    return c1


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
    coeff_l = np.zeros(l+1)
    coeff_l[l] = 1.0
    coeff_m = np.zeros(m+1)
    coeff_m[m] = 1.0
    # Hermite polynomials for the HO eigenfunctions
    hermite_l = np.polynomial.hermite.hermval(coeff_grid * xx, coeff_l)
    hermite_m = np.polynomial.hermite.hermval(coeff_grid * yy, coeff_m)

    # The eigenfunction is the product of all of the above.
    return prefactor * gaussian * hermite_l * hermite_m


def harmonic_oscillator_wigner(x, p, omega, mass=1.0, hbar=1.0):
    """
    """
    return ( 1.0 / (np.pi * hbar) *
             np.exp(-mass * omega * x**2 / hbar) *
             np.exp(-p**2/(mass * omega * hbar)) )

def harmonic_oscillator_wigner_01(x, p, t):
    """
    """
    return ( np.exp(-x**2 - p**2) *
             (x**2 + p**2 + np.sqrt(2.0)*x*np.cos(t) - np.sqrt(2)*p*np.sin(t)) )



# ====================
# Hydrogen
# ====================

def radial_psi(r, n, l, a0=1.0, z=1.0):
    """
    Radial wavefunction for hydrogen electron orbital.
    """
    rho = (2.0 * z * r) / (n * a0)
    sub = n - l - 1.0
    sup = 2.0 * l + 1.0
    normFactor = np.sqrt(
        (2.0 * z / (n * a0))**3 * misc.factorial(sub) /
        (2.0 * n * misc.factorial(n+l))
    )

    wf = (rho**l * np.exp(-rho / 2.0) *
          eval_genlaguerre(sub, sup, rho))
    return wf * normFactor


def radial_integrand(r, n, l, a0=1.0, z=1.0):
    """
    Expectation integrand for radial distribution.
    """
    psi = radial_psi(r, n, l, a0=1.0, z=1.0)
    return np.conjugate(psi) * r**3.0 * psi


def hydrogen_energy(n, a0=1.0, Z=1.0, mu=1.0, c=137.0, alpha=1.0/137.0):
    """
    Hydrogen energy value for a given state.
    """
    return (-mu * c**2 * Z**2 * alpha**2) / (2.0 * n**2)


def hydrogen_transition_energy(n1, n2, a0=1.0, Z=1.0, mu=1.0,
                               c=137.0, alpha=1.0/137.0):
    """
    Hydrogen transition energy value between two given states.
    """
    return ( (-mu * c**2 * Z**2 * alpha**2) /
             (2.0)) * ((1.0/n1**2) - (1.0/n2**2) )



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


def gaussian_x(x, sigma):
    """
    Position gaussian.
    """
    return ( np.exp(-(x**2 / (2 * sigma**2))) /
             (np.sqrt(sigma * np.sqrt(np.pi))) )


def gaussian_p(p, sigma):
    """
    Momentum gaussian.
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


def mu_operator(mu, theta, phi):
    """
    """
    return mu * (np.sin(theta)*np.cos(phi) +
                 np.sin(theta)*np.sin(phi) +
                 np.cos(theta))


def dipole_moment_integrand(phi, theta, mu, l1, m1, l2, m2):
    """
    """
    mu_op = mu_operator(mu, theta, phi)
    Y_l1m1 = sph_harm(m1, l1, phi, theta)
    Y_l2m2 = sph_harm(m2, l2, phi, theta)

    return Y_l1m1 * mu_op * Y_l2m2 * np.sin(theta)


def dipole_moment_superposition_integrand(phi, theta, mu, c1, c2, l1, m1, l2, m2):
    """
    """
    mu_op = mu_operator(mu, theta, phi)
    Y_l1m1 = sph_harm(m1, l1, phi, theta)
    Y_l2m2 = sph_harm(m2, l2, phi, theta)
    Y_lm = c1*Y_l1m1 + c2*Y_l2m2

    return np.conjugate(Y_lm) * mu_op * Y_lm * np.sin(theta)


def sph_harm_real(m, l, phi, theta):
    """
    Spherical harmonics in real space.
    """
    Y_lm = sph_harm(m, l, phi, theta)
    if m < 0:
        Y_lm_real = np.sqrt(2.0) * (-1.0)**m * Y_lm.imag
    elif m > 0:
        Y_lm_real = np.sqrt(2.0) * (-1.0)**m * Y_lm.real
    else:
        Y_lm_real = Y_lm
    return Y_lm_real


def sphere_to_cart(theta, phi, r=1.0):
    """
    Converts spherical coordinates to 3D cartesian coordinates.
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return x, y, z


def hartrees_to_wavelength(energy):
    """
    Get wavelength (nm) corresponding to a given energy (Hartrees)
    """

    return np.abs(45.56 * 1.0 / energy)




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
                    vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
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


def plot_sphere(m, l):
    """
    Plot a function onto a sphere.
    """
    theta = np.arange(0, 2 * np.pi, 0.01)
    phi = np.arange(0, np.pi, 0.01)
    theta_mg, phi_mg = np.meshgrid(theta, phi)
    x, y, z = sphere_to_cart(theta_mg, phi_mg, r=1.0)

    Y_real = sph_harm_real(m, l, theta_mg, phi_mg)

    # Color points on sphere.
    cmap = mpl.cm.get_cmap(name='seismic', lut=None)
    cm = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
    mapped_Y = cm.to_rgba(Y_real)
    cm.set_array(mapped_Y)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=mapped_Y)
    fig.colorbar(cm, shrink=0.5)
    ax.view_init(20, 45)
    ax.set_title('l=' + str(l) + ' m=' + str(m))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return


def plot_energy_levels(energies, figsize=(14, 6), fontsize='xx-small'):
    """
    Energy level plot.

    energies: list of dicts
    """
    e = [i['energy'] for i in energies]
    e_max = np.max(e)
    e_min = np.min(e)
    e_diff = e_max - e_min
    linewidth = 200.0
    offset_to_add = 480

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    prev_e_val = 0.0
    offset = 0.0
    i = 0.0
    for energy_dict in energies:
        e_val = energy_dict['energy']
        if prev_e_val == e_val:
            offset += offset_to_add
            i += 1
        else:
            offset = 0.0
            i = 0.0

        yalign = 'top' if i % 2 else 'bottom'
        y_offset = -e_diff * 0.01 if i % 2 else e_diff * 0.01

        xmin = -linewidth / 2.0 + offset
        xmax = linewidth / 2.0 + offset
        plt.hlines(e_val, xmin, xmax, linewidth=2)
        ax.annotate(
            _energy_label(energy_dict), xy=(xmin, e_val + y_offset),
            fontsize=fontsize, horizontalalignment='left',
            verticalalignment=yalign
        )
        prev_e_val = energy_dict['energy']

    plt.xlim([-linewidth / 2.0 - 5.0, 2000.0])
    plt.ylim([e_min - 0.05 * e_diff, e_max + 0.05 * e_diff])
    plt.tick_params(axis='x', which='both', bottom='off',
                    top='off', labelbottom='off')
    plt.ylabel('Energy')

    return


def _energy_label(energy_dict):
    """
    Create string with quantum numbers n, l, and m.
    """
    vals = []

    if 'n' in energy_dict:
        vals.append('n={:d}'.format(energy_dict['n']))
    if 'l' in energy_dict:
        vals.append('l={:d}'.format(energy_dict['l']))
    if 'm' in energy_dict:
        vals.append('m={:d}'.format(energy_dict['m']))

    return ', '.join(vals)


def wavelength_to_colour(wavelength):
    """
    Get RGB values for a given wavelength of light.
    """
    if wavelength >= 380 and wavelength < 440:
        R = -(wavelength - 440.0) / (440.0 - 350.0)
        G = 0.0
        B = 1.0
    elif wavelength >= 440 and wavelength < 490:
        R = 0.0
        G = (wavelength - 440.0) / (490.0 - 440.0)
        B = 1.0
    elif wavelength >= 490 and wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510.0) / (510.0 - 490.0)
    elif wavelength >= 510 and wavelength < 580:
        R = (wavelength - 510.0) / (580.0 - 510.0)
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength < 645:
        R = 1.0
        G = -(wavelength - 645.0) / (645.0 - 580.0)
        B = 0.0
    elif wavelength >= 645 and wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    # intensity correction
    if wavelength >= 380 and wavelength < 420:
        intensity = 0.3 + 0.7 * (wavelength - 350) / (420 - 350)
    elif wavelength >= 420 and wavelength <= 700:
        intensity = 1.0
    elif wavelength > 700 and wavelength <= 780:
        intensity = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        intensity = 0.0

    return (intensity * R, intensity * G, intensity * B)
