# John Eargle
# 2017-2022


import numpy as np
from scipy.special import factorial

__all__ = ["harmonic_oscillator_1D", "harmonic_oscillator_1D_in_field",
           "excited_overlap", "harmonic_potential_2D", "harmonic_oscillator_2D",
           "harmonic_oscillator_wigner", "harmonic_oscillator_wigner_01"]


# ====================
# Harmonic Oscillator
# ====================

def harmonic_oscillator_1D(x, n, mass=1.0, omega=1.0, hbar=1.0):
    """
    Harmonic Oscillator
    """
    prefactor = (1.0 / (np.sqrt(2**n * factorial(n))) *
                 (mass * omega / (np.pi * hbar))**(1.0/4.0))
    gaussian = np.exp(-(mass * omega * x * x) / (2.0 * hbar))

    coeff_grid = np.sqrt(mass * omega / hbar)
    coeff = np.zeros(n + 1)
    coeff[n] = 1.0
    hermite = np.polynomial.hermite.hermval(coeff_grid * x, coeff)

    return prefactor * gaussian * hermite

def harmonic_oscillator_1D_in_field(x, t, omega_f, omega_0=1, lam=1,
                                    E_0=1.0, mass=1.0, hbar=1.0):
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
                  (np.sqrt(2**l * 2**m * factorial(l) * factorial(m))) )

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
