# psiko

Toolkit for playing around with quantum mechanics.

## General

This project is loosely based on work done for the edX course The Quantum World (CHEM160x).

As a toolkit, this library provides classes and functions for building simple quantum systems and drawing various plots based on them.  Complex-valued wavefunctions can be created from sets of eigenstates and then plotted over space or momentum axes.  These wavefunctions can be sampled at given times or across multiple timepoints to produce trajectories which can then be used for further plotting or animation.

## Psi - Wavefunction

`Psi` is the base class that represents wavefunctions.  Its constructor accepts a domain and parameters for setting up the eigenstates that make up the wavefunction.  `Psi` is an abstract class, and its children are responsible for handling setup and calculation for specific Hamiltonians.  Wavefunction classes are provided for a few simple Hamiltonians, such as particle-in-a-box (`PsiPib`) and harmonic oscillator (`PsiHarm`), but you can also subclass `Psi` to build wavefunctions for other systems.

You can construct a wavefunction from one or more eigenfunctions.  There are methods for calculating the time-dependent wavefunction, its probability density, and expectation values for operators.

## PsiTraj - Wavefunction Trajectory

`PsiTraj` is a class for generating a wavefunction trajectory based on a given `Psi` and a timespan.

## Plotting

## Particle in a Box

`PsiPib`

## Harmonic Oscillator

`PsiHarm`

## Dependencies

### Python

* numpy
* scipy
* matplotlib

### System libraries

* ImageMagick - for saving animated gifs
