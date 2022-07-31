# psiko

Toolkit for playing around with quantum mechanics.

## General

This project is loosely based on work done for the edX course The Quantum World (CHEM160x).

As a toolkit, this library provides classes and functions for building simple quantum systems and drawing various plots based on them.  Complex-valued wavefunctions can be created from sets of eigenstates and then plotted over space or momentum axes.  These wavefunctions can be sampled at given times or across multiple timepoints to produce trajectories which can then be used for further plotting or animation.

## Psi - Wavefunction

`Psi` is the base class that represents wavefunctions (Ψ).  Its constructor accepts a domain and parameters for setting up the eigenstates that make up the wavefunction.  `Psi` is an abstract class, and its children are responsible for handling setup and calculation for specific Hamiltonians.  Wavefunction classes are provided for a few simple Hamiltonians, such as particle-in-a-box (`PsiPib`) and harmonic oscillator (`PsiHarm`), but you can also subclass `Psi` to build wavefunctions for other systems.

You can construct a wavefunction from one or more eigenfunctions along with corresponding misture coefficients.  Eigenfunctions for a particular system are time-independent solutions to HΨ=EΨ where H is the Hamiltonian (linear operator) and E is the energy (eigenvalue) for the wavefunction Ψ.

There are also methods for calculating the time-dependent wavefunction at a given time, its probability density, and expectation values for quantum operators.

### Particle in a Box

`PibPsi` models a 1D particle in a box.  The setup is that the particle is free to move within a finite space.  The potential energy surface is 0 within that space and infinite outside of it so the particle can never escape.  The eigenfunctions are sinusoidal with the box boundary conditions set to 0, similar to the vibrational modes of a string.

### Harmonic Oscillator

`HarmPsi` models a 1D particle in a harmonic potential.  The eigenfunctions are similar in shape to particle-in-a-box solutions, but based on Gaussians since the boundary conditions tend to 0 as the edges approach infinity.

## PsiTraj - Wavefunction Trajectory

`PsiTraj` is a class for generating a wavefunction trajectory based on a given `Psi` and a timespan.

## Plotting

Plotting is relatively straightforward for single timepoints of `Psi`.  For an instantiated wavefunction `psi`, simply grab the domain `psi.x` and the function you care about (e.g. `psi.at_time(0.0).real`, `psi.at_time(0.0).imag`, `psi.prob_density()`) and plot them with the included matplotlib functions.

There are several other plotting functions available for `PsiTraj`, including animation:

* `time_plot_psi()` - plot multiple timepoints on a static graph
* `plot_trisurf()` - 3D plot a single timepoint as a ribbon with one side on the origin line and the other at the edge of the complex-valued wavefunction taken in cylindrical coordinates around the origin line
* `plot_quiver()` - 3D plot a single timepoint as a set of rods each with one side on the origin line and the other at the edge of the complex-valued wavefunction taken in cylindrical coordinates around the origin line
* `traj_plot_psi()` - animate the `time_plot_psi()` line
* `traj_plot_psi2()` - animate the `plot_trisurf()` or `plot_quiver()` representations

## Dependencies

### Python

* numpy
* scipy
* matplotlib

### System libraries

* ImageMagick - for saving animated gifs
