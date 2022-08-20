# John Eargle
# 2017-2022


import numpy as np

from psiko.psiko import Psi, cnt_evolve

__all__ = ["PirPsi"]


# ====================
# Particle in a Ring
# ====================

class PirPsi(Psi):

    def eigenfunction(self, n):
        """
        Normalized energy eigenfunctions to time-independent Particle In a
        Ring.

        n: eigenfunction index; 0, ±1, ±2, ...
        """
        # Need translation of theta to self.x
        # return 1.0/sqrt(2.0*np.pi) * np.exp(1j*n*theta)
        return 1.0/sqrt(2.0*np.pi) * np.exp(1j*n*self.x)

    def energy(self, n):
        """
        Energy eigenvalue for given eigenfunction.

        n: eigenfunction index; 0, ±1, ±2, ...
        """
        mass = 1.0
        # Need translation of radius R to self.length
        # return (n**2 * self.hbar**2) / (2.0 * mass * R**2)
        return (n**2 * self.hbar**2) / (2.0 * mass * self.length**2)
