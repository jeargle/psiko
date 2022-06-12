# John Eargle
# 2017-2022


import numpy as np

from psiko.psiko import Eigenstate, Psi, cnt_evolve

__all__ = ["pib_ti_1D", "pib_td_1D", "pib_wave_solution", "pib_energy"]


# ====================
# Particle in a Box
# ====================

def pib_ti_1D(x, n, l):
    """
    Normalized energy eigenfunctions to time-independent Particle In a
    Box.

    x: domain numpy array starting at 0
    n: eigenfunction index
    l: length of box spanned by x
    """
    return np.sqrt(2.0/l) * np.sin(n*np.pi*x/l)

def pib_ti_1D_psi(x, n, l):
    """
    Normalized energy eigenfunctions to time-independent Particle In a
    Box.

    x: domain numpy array starting at 0
    n: eigenfunction index
    l: length of box spanned by x
    """
    y = np.sqrt(2.0/l) * np.sin(n*np.pi*x/l)
    return PibPsi(x, y)

def pib_td_1D(t, c, n, l):
    """
    Time varying prefactor to time-independent Particle In a Box.

    Only the real component???

    Note: could be replaced by cnt_evolve()

    t: time
    c: scaling factor???
    n: eigenfunction index
    l: length of box spanned by x
    """
    return np.cos(n*np.pi*c*t/l)

def pib_wave_solution(x, t, c, n, l):
    """
    Harmonic solutions to time-dependent Particle In a Box.

    x: domain numpy array starting at 0
    t: time array
    c: scaling factor???
    n: eigenfunction index
    l: length of box spanned by psi.x
    """
    # time-dependent and time-independent terms
    return pib_td_1D(t, c, n, l) * pib_ti_1D(x, n, l)

def pib_wave_solution_psi(psi, t, c=None, n=None, l=None):
    """
    Harmonic solutions to time-dependent Particle In a Box.

    psi: Psi for particle in a box
    t: time array
    c: scaling factor???
    n: eigenfunction index
    l: length of box spanned by psi.x
    """
    # time-dependent and time-independent terms
    return pib_td_1D(t, c, n, l) * psi.y

def pib_energy(n, l, hbar=1.0, m=1):
    """
    Energy eigenvalues

    n: eigenfunction index
    l: length of box spanned by psi.x
    hbar: Plank's constant
    m: mass
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


class PibPsi(Psi):

    def eigenfunction(self, n):
        """
        Normalized energy eigenfunctions to time-independent Particle In a
        Box.

        x: domain numpy array starting at 0
        n: eigenfunction index
        l: length of box spanned by x
        """
        return np.sqrt(2.0/self.l) * np.sin(n*np.pi*x/self.l)

    def energy(self, n):
        return (n**2 * self.hbar**2 * np.pi**2) / (2.0 * m * self.l**2)
