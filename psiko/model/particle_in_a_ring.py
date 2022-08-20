# John Eargle
# 2017-2022


import numpy as np

from psiko.psiko import Psi, cnt_evolve, _wf_type

__all__ = ["PirPsi"]


# ====================
# Particle in a Ring
# ====================

class PirPsi(Psi):
    """
    Wavefunction for a particle in a ring.

    The ring is set up in polar coordinates for a circle with a set
    radius.  The eigenfunctions depend on angle theta for position
    within the ring, and the energy eigenvalues depend on the radius.
    """

    theta = None
    radius = None

    def __init__(self, length=None, num_points=None, dx=None, x_left=None,
                 wf_type=_wf_type['position'], normalize=True, hbar=1.0,
                 eigenstate_params=None):
        """
        length: length of domain
        num_points: number of points to track in domain (x)
        dx: distance between points of x
        wf_type: wavefunction type
        normalize: whether or not to normalize the wavefunction
        hbar: Planck's constant
        eigenstate_params: list of parameters for eigenstates
        """
        # TODO - does x need to have a point removed for the periodic connection?
        self.length = length

        if x_left is None:
            self.x_left = 0
        else:
            self.x_left = x_left

        if num_points is not None:
            self.x = np.linspace(self.x_left, self.length+self.x_left, num_points)
        elif dx is not None:
            self.dx = dx
            self.x = np.arange(self.x_left, self.length+self.x_left, dx)

        self.wf_type = wf_type
        self.hbar = hbar

        # Translation of x to periodic angle theta.
        self.theta = (2.0 * np.pi * self.x) / self.length
        print(f'theta: {self.theta}')

        # Translation of length to radius.
        self.radius = self.length / (2.0 * np.pi)

        self._init_eigenstates(
            eigenstate_params,
            normalize
        )

    def eigenfunction(self, n):
        """
        Normalized energy eigenfunctions to time-independent Particle In a
        Ring.

        n: eigenfunction index; 0, ±1, ±2, ...
        """
        return 1.0/np.sqrt(2.0*np.pi) * np.exp(1j*n*self.theta)

    def energy(self, n):
        """
        Energy eigenvalue for given eigenfunction.

        There is a single lowest energy state, but higher states are
        degenerate with multiplicity 2.

        n: eigenfunction index; 0, ±1, ±2, ...
        """
        mass = 1.0
        return (n**2 * self.hbar**2) / (2.0 * mass * self.radius**2)
