# John Eargle
# 2017-2022


import numpy as np
from scipy.integrate import tplquad

__all__ = []


# ====================
# Helium
# ====================

def He_H11(Z):
    """
    Ground state for CI matrix
    From components:
        H11a = -0.5 * Z**2
        H11b = -0.5 * Z**2
        H11c = 5.0/4.0
    """
    return 2 * (-0.5 * Z**2) + 5.0/4.0

def He_phi1(s, t, u, zeta, Z=2):
    return np.exp(-zeta * s)

def He_H11_vpm(t, u, s, zeta=1.6875, Z=2):
    return (He_phi1(s, t, u, zeta, Z)**2 *
            (s**2 - t**2 - 4*s*u*Z + (s-t)*(s+t)*u*zeta**2))

def He_S11_vpm(t, u, s, zeta=1.6875, Z=2):
    return (He_phi1(s, t, u, zeta, Z)**2 *
            u*(s**2 - t**2))

def He_expected_phi1(zeta):
    # 3D integrate with tplquad
    H11_int, error = tplquad(
        He_H11_vpm,
        0.0, 50.0,
        lambda x: 0.0, lambda x: x,
        lambda x, y: 0.0, lambda x, y: y,
        args=(zeta, 2.0)
    )

    S11_int, error = tplquad(
        He_S11_vpm,
        0.0, np.inf,
        lambda x: 0.0, lambda x: x,
        lambda x, y: 0.0, lambda x, y: y,
        args=(zeta, 2.0)
    )

    return H11_int/S11_int

J22_pm = -337.0 / 162.0
K22_pm = 32.0 / 729.0

H11_pm =  -11.0 / 4.0
H12_pm = 16384.0 / 64827.0
H22_pm = 2.0 * (J22_pm + K22_pm)

def H0():
    return np.array([[H11_pm-H12_pm, 0.], [0., -5.0]])

def H1(lam):
    return lam * np.array([[H12_pm, H12_pm],
                           [H12_pm, 2.0*(J22_pm+2.5 + K22_pm)]])

def H_lambda(lam):
    return H0() + H1(lam)
