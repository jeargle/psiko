# John Eargle
# 2017-2022


import numpy as np
from scipy.special import (
    factorial, eval_genlaguerre, expi
)

__all__ = []


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
        (2.0 * z / (n * a0))**3 * factorial(sub) /
        (2.0 * n * factorial(n+l))
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


# Integrals and PES (Potential Energy Surface) functions
# H2- one-electron 2-proton Hamiltonian

def J(R):
    return np.exp(-2.0*R) * (1.0 + 1.0/R)

def S(R):
    return np.exp(-R) * (1.0 + R + (R**2)/3)

def K(R):
    return S(R)/R - np.exp(-R) * (1.0 + R)

def E_plus(R):
    return (-0.5 + ((J(R) + 1.0/R) / (1 - S(R))) +
            ((K(R) + S(R)/R) / (1 - S(R))))

def E_minus(R):
    return (-0.5 + ((J(R) + 1.0/R) / (1 - S(R))) -
            ((K(R) + S(R)/R) / (1 - S(R))))


# Basically the formulas above

def T11(R):
    return 0.5

def T12(R):
    return -0.5*(S12(R) - 2.0*(1.0+R)*np.exp(-R))

def V11A(R):
    return -1.0

def V11B(R):
    return -1.0/R + (1.0 + 1.0/R)*np.exp(-2.0*R)

def V12A(R):
    return -(1.0 + R)*np.exp(-R)

def V12B(R):
    return -(1.0 + R)*np.exp(-R)

def H2_H11(R):
    return T11(R) + V11A(R) + V11B(R)

def H2_H12(R):
    return T12(R) + V12A(R) + V12B(R)

def S12(R):
    return (1 + R + R**2/3.0)*np.exp(-R)


# Two electron integrals
def int_1111(R):
    asdf = np.ones_like(R)
    return 5.0/8.0*asdf

def int_1122(R):
    return (1.0 - (1.0 + 11.0/8.0*R + 3.0/4.0*R*R + 1.0/6.0*R**3 )*np.exp(-2.0*R))/R

def int_1112(R):
    return (R + (1.0/8.0 + 5.0/(16.0*R))*(1.0 - np.exp(-2.0*R)) )*np.exp(-R)

def int_1212(R):
    A = (1.0 - R + R*R/3.0) * np.exp(R)
    return (1.0/5.0) * (
        (25.0/8.0 - 23.0/4.0*R - 3.0*R*R - R**3.0/3.0) * np.exp(-2.0*R) +
        6.0/R * ((0.57722 + np.log(R)) * S12(R)**2.0 + A*A*expi(-4.0*R) - 2.0*A*S12(R)*expi(-2.0*R))
    )

def J11(R):
    return 1.0/(1.0+S12(R))**2 * (0.5*int_1111(R) + 0.5*int_1122(R) + int_1212(R) + 2.0*int_1112(R))

def V(R):
    return 1.0/R

def J12(R):
    return 1.0/(1.0-S12(R)**2)*(0.5*int_1111(R) + 0.5*int_1122(R) - int_1212(R))

def J22(R):
    return (0.5*int_1111(R) + 0.5*int_1122(R) + int_1212(R) - 2.0*int_1112(R))/(1.0 - S12(R))**2

def K12(R):
    return 1.0/(2.0*(1.0-S12(R)**2))*(int_1111(R) - int_1122(R))

def H2_E_ground(R):
    return (2.0*((H2_H11(R)+H2_H12(R)) / (1.0+S12(R)))) + J11(R) + V(R)

def H2_E_excited(R):
    return (2.0*((H2_H11(R)-H2_H12(R)) / (1.0-S12(R)))) + J22(R) + V(R)

def H2_energy_CI(R):
    """
    Get CI Hamiltonian for a single atom-atom distance R.
    """
    k12 = K12(R)
    H_ci = np.array(
        [[H2_E_ground(R) - V(R), k12],
         [k12, H2_E_excited(R) - V(R)]]
    )
    evals, evecs = np.linalg.eigh(H_ci)
    return evals
