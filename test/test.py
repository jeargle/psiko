# John Eargle
# 2017-2022


# from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import simps, quad, nquad
from scipy.optimize import minimize

import psiko.psiko as pk
import psiko.plot as pk_plot
import psiko.model.particle_in_a_box as pib
import psiko.model.harmonic_oscillator as ho
import psiko.model.hydrogen as pk_h
import psiko.model.helium as pk_he


def square_comp_test1():
    omega = 2
    x = np.linspace(0,1,500)
    y = np.array([pk.square_comp(x, omega, k) for k in range(1,6)])
    for i in range(5):
        plt.plot(x,[y[:i,j].sum() for j in range(500)])
    plt.show()


def square_test1():
    omega = 2
    x = np.linspace(0,1,500)
    ns = [1, 2, 8, 32, 128, 512]
    y = np.array([pk.square(x, omega, i) for i in ns])
    print(f'np.shape(y): {np.shape(y)}')
    y2 = np.array([pk.square2(x, omega, i) for i in ns])
    print(f'np.shape(y2): {np.shape(y2)}')
    print(f'np.abs(y-y2).max(): {np.abs(y-y2).max()}')
    print(f'np.abs(y-y2).sum(): {np.abs(y-y2).sum()}')

    for i in y:
        plt.plot(x,i)
    plt.show()


def time_plot_test1():
    """
    Simulate motion of a particle in 1D
    """
    v = 1.0
    dt = 0.001
    t = np.arange(0, 1+dt, dt)
    x = np.zeros(len(t))
    x[0] = 0

    for n in range(1, len(t)):
        x[n] = x[n-1] + dt*v

    plt.plot(t, x)
    plt.show()


def boundary_cond_test1():
    """
    Reflective boundary conditions in 1D
    """
    v = 1.75
    dt = 0.001
    t = np.arange(0,1+dt,dt)
    x = np.zeros(len(t))
    x[0] = 0.0
    l = 0.5

    for n in range(1,len(t)):
        x[n] = x[n-1] + dt*v
        # boundary conditions
        if abs(x[n]) >= l:
            v = -v

    plt.plot(t, x)
    plt.show()


def forces_test1():
    """
    External forces on particle in 1D
    """
    v = 2.0
    dt = 0.001
    t = np.arange(0,2+dt,dt)
    x = np.zeros(len(t))
    l = 0.5

    for n in range(1,len(t)):
        # update velocity and position
        v += dt*5.0*np.sin(2*t[n])
        # v += 0.5*np.sin(20*t[n])
        # v -= 0.01
        x[n] = x[n-1] + dt*v
        # boundary conditions
        if abs(x[n]) >= l:
            v = -v

    plt.plot(t, x)
    plt.show()


def forces_test2():
    dt = 0.001
    t = np.arange(0,1+dt,dt)
    v1, v2 = 2.0, -2.0
    m1, m2 = 1, 1
    x1, x2 = np.zeros(len(t)), np.zeros(len(t))
    x1[0], x2[0] = 0.1, -0.1
    t = np.arange(0,1+dt,dt)
    l = 0.5

    for i in range(1,len(t)):
        # particle 1: update velocity then position
        v1 = v1 + dt*pk.force1(t[i-1],m1)
        x1[i]=x1[i-1]+dt*v1
        # particle 1: boundary condition
        v1 = pk.boundary_1d(x1[i], v1, l)

        # particle 2: update velocity than position
        v2 = v2 + dt*pk.force1(t[i-1],m2)
        x2[i]=x2[i-1]+dt*v2
        # particle 2: boundary conditions
        v2 = pk.boundary_1d(x2[i], v2, l)

    plt.plot(t, x1)
    plt.plot(t, x2)
    plt.show()


def forces_test3():
    dt = 0.001
    t = np.arange(0,1+dt,dt)
    v1, v2 = 2.0, -2.0
    m1, m2 = 1, 1
    x1, x2 = np.zeros(len(t)), np.zeros(len(t))
    x1[0], x2[0] = 0.1, -0.1
    t = np.arange(0,1+dt,dt)
    l = 0.5

    # repulsion force array
    repul = np.zeros(len(t))

    #iterate over t array
    for i in range(1,len(t)):
        # particle-particle interaction force
        repul[i] = pk.repulsion(x1[i-1], x2[i-1])

        # particle 1
        v1 = v1 + dt*repul[i]
        x1[i] = x1[i-1] + dt *v1
        v1 = pk.boundary_1d(x1[i], v1, l)

        # particle 2
        v2 = v2 - dt*repul[i]
        x2[i] = x2[i-1] + dt *v2
        v2 = pk.boundary_1d(x2[i], v2, l)

    plt.plot(t, x1)
    plt.plot(t, x2)
    plt.show()

    plt.clf()
    plt.plot(t, repul)
    plt.show()


def pib_test1():
    """
    Plot first three eigenfunctions for a particle in a box.
    """
    # Calculate 3 harmonics.
    num_harmonics = 3
    mix_coeff = np.sqrt(1/num_harmonics)
    eigenstate_params = [
        {
            'mix_coeff': mix_coeff,
            'quantum_numbers': {
                'n': n
            }
        }
        for n in range(1, num_harmonics+1)
    ]
    l = 10
    psi = pib.PibPsi(
        l,
        num_points=1001,
        eigenstate_params=eigenstate_params
    )

    for eigen_state in psi.eigenstates:
        plt.plot(psi.x, eigen_state.y, label=f'{eigen_state.quantum_numbers["n"]}')
    plt.legend()
    plt.show()


def pib_test1_old():
    """
    Plot first three eigenfunctions for a particle in a box.
    """
    l = 10
    x = np.linspace(0, l, 1001)
    y = []

    # Calculate 3 harmonics.
    for n in range(1,4):
        y.append(pib.pib_ti_1D(x, n, l))

    for i, psi in enumerate(y):
        plt.plot(x, psi, label=f'{i+1}')
    plt.legend()
    plt.show()


def pib_test2():
    """
    Plot time evolution of time-dependent prefactor for first four
    eigenfunctions of a particle in a box.
    """
    l = 10.0
    c = 0.1
    t = np.linspace(0, 100, 1001)
    y = np.zeros(4*len(t)).reshape(4, len(t))

    # Use a for loop for the multiple harmonics.
    for n in range(1,5):
        y[n-1] = pib.pib_td_1D(t, c, n, l)

    for i, psi in enumerate(y):
        plt.plot(t, psi, label=f'{i+1}')
    plt.legend()
    plt.show()


def pib_test3():
    """
    Plot time series for third eigenfunction of particle in a box.
    """
    length = np.pi
    psi = pib.PibPsi(
        length,
        # dx = 0.01,
        num_points=101,
        eigenstate_params=[{
            'mix_coeff': 1,
            'quantum_numbers': {
                'n': 3
            }
        }]
    )

    t_wavelength = 2*np.pi/psi.eigenstates[0].energy
    t = np.linspace(0, t_wavelength, 661)
    psi_traj = pk.PsiTraj(psi, t)

    pk_plot.time_plot_psi(psi_traj)
    plt.show()
    plt.clf()

    pk_plot.traj_plot_psi(
        psi_traj,
        # ylim=(-0.5, 0.5),
        ylim=(-1.0, 1.0),
        # skip=5,
        skip=10,
        show=True
    )


def pib_test3_old():
    """
    Plot time series for third eigenfunction of particle in a box.
    """
    c = 0.1
    n = 3
    l = 10
    x = np.arange(0, l, 0.01)
    dt = 0.1
    t = np.arange(0, 66, dt)

    traj = np.zeros((len(x), len(t)))
    for step, time in enumerate(t):
        traj[:, step] = pib.pib_wave_solution(x, time, c, n, l)

    pk_plot.time_plot(x, traj, t)
    plt.show()
    plt.clf()

    pk_plot.traj_plot(
        x, traj, t, dt,
        xlim=(0, l),
        ylim=(-0.5, 0.5),
        skip=10,
        show=True
    )


def pib_interference_test1():
    """
    Plot time trace for a single position in a wavefunction made from
    the first two eigenstates.
    """
    length = 10
    mix_coeff = 0.5

    # Sum 2 eigenstates.
    psi = pib.PibPsi(
        length,
        num_points=4,
        eigenstate_params=[
            {
                'mix_coeff': mix_coeff,
                'quantum_numbers': {
                    'n': 1
                }
            },
            {
                'mix_coeff': mix_coeff,
                'quantum_numbers': {
                    'n': 2
                }
            }
        ]
    )

    # Plot single point (length/3) from wavefunction.
    t_wavelength = 2*np.pi/psi.eigenstates[0].energy
    t_extent = 2.5
    t = np.linspace(0, t_wavelength*t_extent, 661)
    psi_traj = pk.PsiTraj(psi, t)

    plt.plot(t, psi_traj.traj[1,:])
    plt.show()


def pib_interference_test1_old():
    """
    Plot time trace for a single position in a wavefunction made from
    the first two eigenstates.
    """
    t = np.arange(0, 100, 0.1)
    c = 0.5
    l = 10
    # Track specific point within the space l.
    x = np.array([l/3.0])

    wave = np.zeros(len(t))

    # Sum 2 eigenstates.
    for step, time in enumerate(t):
        wave[step] = (pib.pib_wave_solution(x, time, c, 1, l)[0] +
                      pib.pib_wave_solution(x, time, c, 2, l)[0])

    plt.plot(t, wave)
    plt.show()


def pib_interference_test2():
    """
    Plot time evolution of full wavefunction made from the first two
    eigenstates.
    """
    length = 10
    mix_coeff = 0.5

    # Sum 2 eigenstates.
    psi = pib.PibPsi(
        length,
        num_points=101,
        eigenstate_params=[
            {
                'mix_coeff': mix_coeff,
                'quantum_numbers': {
                    'n': 1
                }
            },
            {
                'mix_coeff': mix_coeff,
                'quantum_numbers': {
                    'n': 2
                }
            }
        ]
    )

    t_wavelength = 2*np.pi/psi.eigenstates[0].energy
    t_extent = 1.0
    t = np.linspace(0, t_wavelength*t_extent, 661)
    psi_traj = pk.PsiTraj(psi, t)

    pk_plot.time_plot_psi(psi_traj)
    plt.show()
    plt.clf()

    pk_plot.traj_plot_psi(
        psi_traj,
        ylim=(-0.5, 0.5),
        # ylim=(-1.0, 1.0),
        skip=5,
        # skip=10,
        show=True
    )


def pib_interference_test2_old():
    """
    Plot time evolution of full wavefunction made from the first two
    eigenstates.
    """
    c = 0.5
    l = 10
    x = np.linspace(0, l, 101)
    dt = 0.1
    t = np.arange(0, 30, dt)
    traj = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    for step, time in enumerate(t):
        traj[:, step] = (pib.pib_wave_solution(x, time, c, 1, l) +
                         pib.pib_wave_solution(x, time, c, 2, l))

    pk_plot.time_plot(x, traj, t, timestep=1)
    plt.show()
    plt.clf()

    pk_plot.traj_plot(
        x, traj, t, dt,
        xlim=(0, l), ylim=(-1.0, 1.0),
        skip=5, show=True
    )


def quadrature_test1():
    l = 10.0
    x = np.arange(0, l, 0.01)
    c = np.zeros(10)

    for n in range(0, 10):
        c[n], _ = quad(pk.projection_integrand, 0.0, l, args=(n, l))

    plt.plot(range(0, 10), c)
    plt.show()


def quadrature_test2():
    """
    Wave decomposition
    """
    l = 10.0
    x = np.arange(0, l, 0.01)
    y = np.array([pk.square_function(xi, l) for xi in x])
    square_approx = np.zeros(len(x))

    for n in range(10):
        # project amplitudes and integrate components
        cn, _ = quad(pk.projection_integrand, 0.0, l, args=(n, l))
        square_approx += cn * np.sqrt(2.0/l) * np.sin(n*np.pi*x/l)

    plt.plot(x, square_approx)
    plt.plot(x, [pk.square_function(i, l) for i in x])
    plt.show()


def normalize_test1():
    """
    Set up a mixed state from the ground and first three excited states
    for the particle in a box.  Normalize the wavefunction.
    """
    # Build wavefunctions from 4 eigenfunctions
    length = 10.0
    num_harmonics = 4
    eigenstate_params = [
        {
            'quantum_numbers': {
                'n': n
            }
        }
        for n in range(1, num_harmonics+1)
    ]

    psi = pib.PibPsi(
        length=length,
        dx=0.01,
        normalize=False,
        eigenstate_params=eigenstate_params
    )

    psi_normed = pib.PibPsi(
        length=length,
        dx=0.01,
        eigenstate_params=eigenstate_params
    )

    norm_pre = psi.psi_norm()
    norm_post = psi_normed.psi_norm()

    print(f'Norm pre: {norm_pre}')
    print(f'Norm post: {norm_post}')

    plt.plot(psi.x, psi.at_time(0.0))
    plt.plot(psi.x, psi.prob_density())
    plt.show()
    plt.clf()

    plt.plot(psi_normed.x, psi_normed.at_time(0.0))
    plt.plot(psi_normed.x, psi_normed.prob_density())
    plt.show()


def normalize_test1_old():
    """
    Set up a mixed state from the ground and first three excited states
    for the particle in a box.  Normalize the wavefunction.
    """
    l = 10.0
    x = np.arange(0, l, 0.01)
    psi_x = np.zeros(len(x))

    # Build wavefunction from 4 eigenfunctions
    for n in range(1,5):
        psi_x += pib.pib_ti_1D(x, n, l)

    # Get PDF and normalize for psi_x
    pdf = pk.prob_density(psi_x)
    psi_normed = pk.normalize_wfn(x, psi_x)

    norm_pre = pk.psi_norm(x, psi_x)
    norm_post = pk.psi_norm(x, psi_normed)

    print(f'Norm pre: {norm_pre}')
    print(f'Norm post: {norm_post}')

    plt.plot(x, psi_x)
    plt.plot(x, pdf)
    plt.show()


def schroedinger_test1():
    """
    Set up a mixed state from the ground and first excited states for
    the particle in a box.  Normalize the wavefunction.
    """
    length = 10
    mix_coeff = 1.0/np.sqrt(2)

    # Sum 2 eigenstates.
    psi = pib.PibPsi(
        length,
        dx=0.01,
        normalize=False,
        eigenstate_params=[
            {
                'mix_coeff': mix_coeff,
                'quantum_numbers': {
                    'n': 1
                }
            },
            {
                'mix_coeff': mix_coeff,
                'quantum_numbers': {
                    'n': 2
                }
            }
        ]
    )

    print(f'Norm is {psi.psi_norm()}')

    plt.plot(psi.x, psi.at_time(0.0))
    plt.show()


def schroedinger_test1_old():
    """
    Set up a mixed state from the ground and first excited states for
    the particle in a box.  Normalize the wavefunction.
    """
    l = 10
    x = np.arange(0, l, 0.01)

    # First eigenstate
    psi1_x = pib.pib_ti_1D(x, 1, l)
    c1 = 1.0/np.sqrt(2)
    E1 = pib.pib_energy(1, l)

    # Second eigenstate
    psi2_x = pib.pib_ti_1D(x, 2, l)
    c2 = 1.0/np.sqrt(2)
    E2 = pib.pib_energy(2, l)

    # Mixed state
    psi0 = c1*psi1_x + c2*psi2_x
    psi0_norm = pk.psi_norm(x, psi0)
    print(f'Norm is {psi0_norm}')

    plt.plot(x, psi0)
    plt.show()


def schroedinger_test2():
    """
    Time-dependent Schroedinger equation.
    """
    l = 10
    x = np.arange(0, l, 0.01)
    t = np.linspace(0, 50, 100)
    psi = np.zeros(len(x)*len(t), dtype=complex).reshape(len(x), len(t))
    pdf = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    # First eigenstate
    c1_0 = 1/np.sqrt(2)
    psi1_x = pib.pib_ti_1D(x, 1, l)
    E1 = pib.pib_energy(1, l)

    # Second eigenstate
    c2_0 = 1/np.sqrt(2)
    psi2_x = pib.pib_ti_1D(x, 2, l)
    E2 = pib.pib_energy(2, l)

    for step, time in enumerate(t):
        # Get time evolved coefficients
        c1 = pk.cnt_evolve(c1_0, time, E1)
        c2 = pk.cnt_evolve(c2_0, time, E2)
        psi[:, step] = c1*psi1_x + c2*psi2_x
        pdf[:, step] = pk.prob_density(psi[:, step])

    pk_plot.time_plot(x, psi.real, t)
    plt.show()
    plt.clf()

    pk_plot.traj_plot(
        x, psi.real, t,
        # xlim=(0, l),
        ylim=(-0.5, 0.75),
        # skip=5,
        show=True
    )

    pk_plot.time_plot(x, psi.imag, t)
    plt.show()
    plt.clf()

    pk_plot.traj_plot(
        x, psi.imag, t,
        # xlim=(0, l),
        ylim=(-0.6, 0.3),
        # skip=5,
        show=True
    )

    pk_plot.time_plot(x, pdf, t)
    plt.show()

    pk_plot.traj_plot(
        x, pdf, t,
        # xlim=(0, l),
        ylim=(-0.1, 0.4),
        # skip=5,
        show=True
    )


def schroedinger_test2_old():
    """
    Time-dependent Schroedinger equation.
    """
    l = 10
    x = np.arange(0, l, 0.01)
    t = np.linspace(0, 50, 100)
    psi = np.zeros(len(x)*len(t), dtype=complex).reshape(len(x), len(t))
    pdf = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    # First eigenstate
    c1_0 = 1/np.sqrt(2)
    psi1_x = pib.pib_ti_1D(x, 1, l)
    E1 = pib.pib_energy(1, l)

    # Second eigenstate
    c2_0 = 1/np.sqrt(2)
    psi2_x = pib.pib_ti_1D(x, 2, l)
    E2 = pib.pib_energy(2, l)

    for step, time in enumerate(t):
        # Get time evolved coefficients
        c1 = pk.cnt_evolve(c1_0, time, E1)
        c2 = pk.cnt_evolve(c2_0, time, E2)
        psi[:, step] = c1*psi1_x + c2*psi2_x
        pdf[:, step] = pk.prob_density(psi[:, step])

    pk_plot.time_plot(x, psi.real, t)
    plt.show()
    plt.clf()

    pk_plot.traj_plot(
        x, psi.real, t,
        # xlim=(0, l),
        ylim=(-0.5, 0.75),
        # skip=5,
        show=True
    )

    pk_plot.time_plot(x, psi.imag, t)
    plt.show()
    plt.clf()

    pk_plot.traj_plot(
        x, psi.imag, t,
        # xlim=(0, l),
        ylim=(-0.6, 0.3),
        # skip=5,
        show=True
    )

    pk_plot.time_plot(x, pdf, t)
    plt.show()

    pk_plot.traj_plot(
        x, pdf, t,
        # xlim=(0, l),
        ylim=(-0.1, 0.4),
        # skip=5,
        show=True
    )


def operator_test1():
    """
    Position and momentum operators.
    """
    l = 10
    dx = 0.01
    x = np.arange(0, l, dx)
    c1 = 1.0/np.sqrt(2)
    c2 = 1.0/np.sqrt(2)
    psi1_x = pib.pib_ti_1D(x, 1, l)
    psi2_x = pib.pib_ti_1D(x, 2, l)

    psi0 = c1*psi1_x + c2*psi2_x

    x_integrand = psi0 * x * psi0
    exp_x = pk.complex_simps(x_integrand, x)
    print(f'Expectation of position: {exp_x}')

    p_integrand = psi0 * pk.momentum_operator(psi0, x, dx)
    exp_p = pk.complex_simps(p_integrand, x)
    print(f'Expectation of momentum: {exp_p}')


def operator_test2():
    """
    Observables and expectation values.
    """
    l = 10
    dx = 0.01
    x = np.arange(0, l, dx)
    t = np.arange(0, 100)
    psi = pib.pib_superposition(x, t, l, 1, 2)
    p_array = np.zeros(len(t), dtype=complex)
    x_array = np.zeros(len(t), dtype=complex)

    for step, time in enumerate(t):
        p_array[step] = pk.eval_expectation(psi[:, step], x, dx, pk.momentum_operator)
        x_array[step] = pk.eval_expectation(psi[:, step], x, dx, pk.position_operator)

    plt.plot(t, p_array)
    plt.show()

    plt.clf()
    plt.plot(t, x_array)
    plt.show()


def operator_test3():
    """
    Build a Hamiltonian matrix.
    """

    # initial variables
    l = 10
    dx = 0.01
    x = np.arange(0, l, dx)
    vx = pk.linear_ramp(x)

    # get kinetic energy, potential energy, and hamiltonian
    T = pk.kinetic_mat_operator(x, dx)
    V = np.diag(vx)
    H = T + V
    print('T:')
    print(T)
    print('V:')
    print(V)


def operator_test4():
    """
    Build a potential surface.
    """

    l = 5.0
    dx = 0.05
    x = np.arange(0, l+dx, dx)

    vx = pk.linear_ramp(x)
    plt.plot(x, vx)
    plt.show()


def operator_test5():
    """
    Solving for states of a wavefunction
    """

    l = 5.0
    dx = 0.02
    x = np.arange(0, l+dx, dx)

    vx = pk.linear_ramp(x)
    H = pk.build_hamiltonian(x, vx, dx=dx)

    evals, evecs = np.linalg.eigh(H)

    for i in range(5):
        psi = evecs[:, i]
        psi = pk.normalize_wfn(x, psi)
        pdf = pk.prob_density(psi)

        plt.plot(x, psi)
        plt.plot(x, pdf)
        plt.show()


def operator_test6():
    """
    Time-independent Schroedinger equation in Coulomb-like double well
    """
    x = np.linspace(-3, 3, 500)
    dx = x[1] - x[0]
    r = 1.21

    vx = pk.coulomb_double_well(x, r)
    H = pk.build_hamiltonian(x, vx, dx=dx)
    evals, evecs = np.linalg.eigh(H)
    plt.plot(x, vx)
    plt.show()
    plt.plot(x, evecs[:,0])
    plt.plot(x, evecs[:,1])
    plt.show()

    vx2 = pk.coulomb_well(x)
    H2 = pk.build_hamiltonian(x, vx2, dx=dx)
    evals2, evecs2 = np.linalg.eigh(H2)
    print(evals[0], evals2[0])
    plt.plot(x, vx2)
    plt.show()
    plt.plot(x, evecs2[:,0])
    plt.plot(x, evecs2[:,1])
    plt.show()


def harmonic_2d_test1():
    """
    The 2D Harmonic potential
    """
    kx = 0.05
    ky = 0.05
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)

    # meshgrid
    xx, yy = np.meshgrid(x, y)
    vxy = ho.harmonic_potential_2D(xx, yy, kx, ky)

    pk_plot.plot_surface(xx, yy, vxy)
    plt.show()

    plt.clf()
    pk_plot.plot_contours(xx, yy, vxy)
    plt.show()


def harmonic_2d_test2():
    """
    The 2D Harmonic Oscillator eigenstates
    """
    l = 1
    m = 2
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    xx, yy = np.meshgrid(x, y)

    ho1 = ho.harmonic_oscillator_2D(xx, yy, l, m)

    pk_plot.plot_surface(xx, yy, ho1)
    plt.show()

    plt.clf()
    pk_plot.plot_contours(xx, yy, ho1)
    plt.show()


def harmonic_2d_test3():
    """
    2D Harmonic Oscillator eigenstate superposition
    """
    l1, m1 = 0, 0
    l2, m2 = 1, 2
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    xx, yy = np.meshgrid(x, y)

    # Two eigenfunctions
    psi1 = ho.harmonic_oscillator_2D(xx, yy, l1, m1)
    E1 = l1 + m1 + 1
    c1 = 1.0/np.sqrt(2)

    psi2 = ho.harmonic_oscillator_2D(xx, yy, l2, m2)
    E2 = l2 + m2 + 1
    c2 = c1

    # superposition
    psi = c1*psi1 + c2*psi2

    pk_plot.plot_contours(xx, yy, psi)
    plt.show()


def harmonic_2d_test4():
    """
    Time-dependent 2D Harmonic Oscillator
    """
    l1, m1 = 0, 0
    l2, m2 = 1, 2
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    xx, yy = np.meshgrid(x, y)
    t = np.linspace(0, 4, 80)
    psi = np.zeros(len(t)*xx.shape[0]*xx.shape[1]).reshape(len(t), xx.shape[0], xx.shape[1])

    # Two eigenfunctions
    psi1 = ho.harmonic_oscillator_2D(xx, yy, l1, m1)
    E1 = l1 + m1 + 1
    c1 = 1.0/np.sqrt(2)

    psi2 = ho.harmonic_oscillator_2D(xx, yy, l2, m2)
    E2 = l2 + m2 + 1
    c2 = c1

    for step, time in enumerate(t):
        c1_t = pk.cnt_evolve(c1, time, E1)
        c2_t = pk.cnt_evolve(c2, time, E2)
        psi[step] = (c1_t*psi1 + c2_t*psi2).real

    vmin = np.min(psi)
    vmax = np.max(psi)

    for i in range(0, 80, 8):
        # pk_plot.plot_surface(xx, yy, psi[i])
        pk_plot.plot_contours(xx, yy, psi[i], vmin=vmin, vmax=vmax)
        plt.show()
        plt.clf()


def field_test1():
    """
    Time-dependent harmonic oscillator under a field.
    """
    dx = 0.01
    x = np.arange(-3, 3+dx, dx)
    ti = 0.0
    omega_f = 3.0

    test = ho.harmonic_oscillator_1D_in_field(x, ti, omega_f).real
    plt.plot(x, test)
    plt.show()


def field_test2():
    """
    Wavepacket dynamics for time-dependent harmonic oscillator under
    a field.
    """
    x = np.arange(-3, 3+0.01, 0.01)
    t = np.linspace(0,4,250)
    omega_f = 3.0
    psit = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    for step, time in enumerate(t):
        psit[:,step] = ho.harmonic_oscillator_1D_in_field(x, time, omega_f).real

    pk_plot.time_plot(x, psit, t, timestep=8)
    plt.show()
    plt.clf()

    pk_plot.traj_plot(
        x, psit, t,
        ylim=(-0.75, 1.5),
        skip=2,
        show=True
    )


def field_test3():
    """
    Excited states overlap.
    """
    t = np.arange(0, 10, 0.01)
    omegas = [3.0, 5.0, 8.0, 10.0]
    prob_excited_state = np.zeros(len(omegas)*len(t)).reshape(len(omegas), len(t))

    # iterate over EM field frequencies
    for omega_idx, omega in enumerate(omegas):
        c1 = ho.excited_overlap(t, omega)
        prob_excited_state[omega_idx] = abs(c1)**2

    for idx in range(len(omegas)):
        plt.plot(t, prob_excited_state[idx])
    plt.show()


def field_test4():
    """
    Calculate infrared spectrum.
    """
    omegas = np.linspace(0, 3, 95)
    t = np.arange(0, 30, 0.1)
    ir_spectrum = np.zeros_like(omegas)

    # iterate over EM field frequencies
    for omega_idx, omega in enumerate(omegas):
        # calculate overlaps and integrate
        c1t = ho.excited_overlap(t, omega)
        A_trans = pk.complex_simps(c1t, t)
        ir_spectrum[omega_idx] = np.absolute(A_trans)**2

    plt.plot(omegas, ir_spectrum)
    plt.show()


def phase_space_test1():
    """
    Gaussians and uncertainty.
    """
    sigma = 0.1

    # calculate position part
    exp_x, _ = quad(pk.x_int, -np.inf, np.inf, args=(sigma))
    exp_x2, _ = quad(pk.x2_int, -np.inf, np.inf, args=(sigma))

    # calculate momentum part
    exp_p, _ = quad(pk.p_int, -np.inf, np.inf, args=(sigma))
    exp_p2, _ = quad(pk.p2_int, -np.inf, np.inf, args=(sigma))

    # calculate uncertainty
    delta_x = np.sqrt(exp_x2 - exp_x**2)
    delta_p = np.sqrt(exp_p2 - exp_p**2)
    uncertainty = delta_x*delta_p
    print(f'delta_x: {delta_x}')
    print(f'delta_p: {delta_p}')
    print(f'uncertainty: {uncertainty}')

    grid = np.linspace(-10.0, 10.0, 1000)
    plt.plot(grid, pk.gaussian_x(grid, sigma))
    plt.plot(grid, pk.gaussian_p(grid, sigma))
    plt.show()


def phase_space_test2():
    """
    Wigner wavepackets
    """
    omega = 1.0
    x = np.linspace(-3.0, 3.0, 50)
    p = np.linspace(-3.0, 3.0, 50)

    xx, pp = np.meshgrid(x, p)
    wxp = ho.harmonic_oscillator_wigner(xx, pp, omega)

    pk_plot.plot_contours(xx, pp, wxp)
    plt.show()


def phase_space_test3():
    """
    Propagating a Wigner density
    """
    x = np.linspace(-3.0, 3.0, 40)
    p = np.linspace(-3.0, 3.0, 40)
    xx, pp = np.meshgrid(x, p)
    omega = 1.5
    t_0 = 0.0
    t_f = 2*np.pi
    t = np.linspace(t_0, t_f, 80)
    wxpt = np.zeros(len(t)*len(x)*len(p)).reshape(len(t),len(x),len(p))

    for i, time in enumerate(t):
        wxpt[i] = ho.harmonic_oscillator_wigner(
            xx - np.cos(omega*time),
            pp - np.sin(omega*time),
            omega
        )

    z_min = np.min(wxpt)
    z_max = np.max(wxpt)
    print(f'z_min: {z_min}')
    print(f'z_max: {z_max}')

    for i in range(0, 80, 20):
        pk_plot.plot_contours(xx, pp, wxpt[i])
        plt.show()
        plt.clf()


def phase_space_test4():
    """
    Wigner density superposition of the HO
    """
    x = np.linspace(-3.0, 3.0, 40)
    p = np.linspace(-3.0, 3.0, 40)
    xx, pp = np.meshgrid(x, p)
    t = np.linspace(0, 5, 80)
    wxpt = np.zeros(len(t)*len(x)*len(p)).reshape(len(t),len(x),len(p))

    for i, time in enumerate(t):
        wxpt[i] = ho.harmonic_oscillator_wigner_01(xx, pp, time)

    z_min = np.min(wxpt)
    z_max = np.max(wxpt)
    print(f'z_min: {z_min}')
    print(f'z_max: {z_max}')

    for i in range(0, 80, 8):
        pk_plot.plot_contours(xx, pp, wxpt[i])
        plt.show()
        plt.clf()


def tunnel_test1():
    """
    """
    dx = 0.01
    x = np.arange(0, 10, dx)
    energy = 10.0
    psi_x = pk.complex_plane_wave(x, energy)
    barrier = pk.square_barrier(x)

    psi_tunnel = pk.tunnel_finite_diff(x, psi_x, barrier, energy)
    pdf = pk.prob_density(psi_tunnel)

    plt.plot(x, barrier)
    plt.plot(x, psi_tunnel.real)
    plt.plot(x, psi_tunnel.imag)
    plt.plot(x, pdf)
    plt.legend()
    plt.show()


def tunnel_test2():
    """
    Transmission probability
    """
    x = np.arange(0, 10, 0.01)
    barrier = pk.square_barrier(x)
    energies = np.arange(0, 25, 0.1)
    transmission = np.zeros_like(energies)

    for idx, energy in enumerate(energies):
        psi_x = pk.complex_plane_wave(x, energy)
        psi_tunnel = pk.tunnel_finite_diff(x, psi_x, barrier, energy)
        pdf = pk.prob_density(psi_tunnel)
        transmission[idx] = pk.transmission_probability(pdf)

    plt.plot(energies, transmission)
    plt.show()


def tunnel_test3():
    """
    Transmission vs mass/energy (2D scan)
    """
    dx = 0.01
    x = np.arange(0, 10, dx)
    barrier = pk.square_barrier(x)
    energies = np.linspace(1,25,20)
    masses = np.linspace(0.5, 5, 20)
    ee, mm = np.meshgrid(energies, masses)
    transmission = np.zeros(len(energies)*len(masses)).reshape(len(energies), len(masses))

    for i, e in enumerate(energies):
        for j, m in enumerate(masses):
            psi_x = pk.complex_plane_wave(x, e, m)
            psi_tunnel = pk.tunnel_finite_diff(x, psi_x, barrier, e)
            pdf = pk.prob_density(psi_tunnel)
            transmission[i, j] = pk.transmission_probability(pdf)

    pk_plot.plot_contours(ee, mm, transmission)
    plt.show()


def rotation_test1():
    """
    Rotational states: Spherical Harmonics
    """
    # spherical harmonics (m=2, l=2)
    pk_plot.plot_sphere(2, 2)
    plt.show()
    plt.clf()

    # spherical harmonics (m=1,l=2)
    pk_plot.plot_sphere(1, 2)
    plt.show()
    plt.clf()


def rotation_test2():
    """
    Dipole moment of rotational wavefunctions
    """
    mu = 0.425

    # Expectation for dipole moment Y_0^0
    dipole_00, _ = nquad(
        pk.dipole_moment_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, 0, 0, 0, 0, True]
    )
    dipole_00_alt, _ = pk.complex_nquad(
        pk.dipole_moment_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, 0, 0, 0, 0]
    )

    print(f'Dipole moment for $Y^0_0$ is (nquad): {dipole_00}')
    print(f'Dipole moment for $Y^0_0$ is (complex_nquad): {dipole_00_alt}')
    print(f'Difference between approaches {np.abs(dipole_00-dipole_00_alt)}')

    # Expectation for dipole moment Y_1^1
    dipole_11, _ = nquad(
        pk.dipole_moment_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, 1, 1, 1, 1, True]
    )
    dipole_11_alt, _ = pk.complex_nquad(
        pk.dipole_moment_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, 1, 1, 1, 1]
    )

    print(f'The dipole moment for $Y^1_1$ is (nquad): {dipole_11}')
    print(f'The dipole moment for $Y^1_1$ is (complex_nquad): {dipole_11_alt}')
    print(f'Difference between approaches {np.abs(dipole_11-dipole_11_alt)}')


def rotation_test3():
    """
    A superposition of 2 rotational eigenstates
    """
    B = 4.82671733e-5
    mu = 0.425

    c1_0 = c2_0 = 1.0/np.sqrt(2)

    dipole_1, _ = nquad(
        pk.dipole_moment_superposition_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, c1_0, c2_0, 0, 0, 1, 0, True]
    )
    print(f'The dipole moment for $Y^0_0$+$Y^1_0$ is (nquad): {dipole_1}')

    dipole_2, _ = nquad(
        pk.dipole_moment_superposition_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, c1_0, c2_0, 0, 0, 1, 1, True]
    )
    print(f'The dipole moment for $Y^0_0$+$Y^1_1$ is (nquad): {dipole_2}')

    dipole_3, _ = nquad(
        pk.dipole_moment_superposition_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, c1_0, c2_0, 0, 0, 1, -1, True]
    )
    print(f'The dipole moment for $Y^0_0$+$Y^1_-1$ is (nquad): {dipole_3}')


def rotation_test4():
    """
    Time-propagate a rotational superposition
    """
    B = 4.82671733e-5
    mu = 0.425
    l1, l2 = 0, 1
    t = np.linspace(0, 70000, 50)
    dipoles_1 = np.zeros(len(t))
    dipoles_2 = np.zeros(len(t))
    dipoles_3 = np.zeros(len(t))

    l1 = m1 = 0
    l2 = 1
    c1_0 = c2_0 = 1.0/np.sqrt(2)

    E1 = 0.0
    E2 = 2.0*B

    for i, ti in enumerate(t):
        c1 = pk.cnt_evolve(c1_0, ti, E1)
        c2 = pk.cnt_evolve(c2_0, ti, E2)

        # compute dipole moments, get values, and store absolute values
        m2 = 0
        dipoles_1[i], _ = nquad(
            pk.dipole_moment_superposition_integrand,
            [[0, 2.0*np.pi], [0, np.pi]],
            args=[mu, c1, c2, l1, m1, l2, m2, True]
        )
        m2 = 1
        dipoles_2[i], _ = nquad(
            pk.dipole_moment_superposition_integrand,
            [[0, 2.0*np.pi], [0, np.pi]],
            args=[mu, c1, c2, l1, m1, l2, m2, True]
        )
        m2 = -1
        dipoles_3[i], _ = nquad(
            pk.dipole_moment_superposition_integrand,
            [[0, 2.0*np.pi], [0, np.pi]],
            args=[mu, c1, c2, l1, m1, l2, m2, True]
        )

    plt.plot(t, dipoles_1)
    plt.plot(t, dipoles_2)
    plt.plot(t, dipoles_3)
    plt.show()


def rotation_test5():
    """
    Selection rules for the rigid rotor.
    """
    mu = 0.425
    m1 = m2 = 0
    l1_arr = np.array(range(0, 6))
    l2_arr = np.array(range(0, 6))
    trans_moment_l = np.zeros(len(l1_arr)*len(l2_arr)).reshape(len(l1_arr), len(l2_arr))

    for l1 in l1_arr:
        for l2 in l2_arr:
            trans_moment_l[l1, l2], _ = nquad(
                pk.dipole_moment_integrand,
                [[0, 2.0*np.pi], [0, np.pi]],
                args=[mu, l1, m1, l2, m2, True]
            )

    plt.imshow(trans_moment_l, interpolation='nearest', extent=[0,5,0,5], origin='lower')
    plt.colorbar()
    plt.show()


def rotation_test6():
    """
    More selection rules for the rigid rotor.
    """
    mu = 0.425
    l1, l2 = 5, 6
    m1_arr = np.array(range(-5,6))
    m2_arr = np.array(range(-5,6))
    trans_moment_m = np.zeros(len(m1_arr)*len(m2_arr)).reshape(len(m1_arr), len(m2_arr))

    for i, m1 in enumerate(m1_arr):
        for j, m2 in enumerate(m2_arr):
            trans_moment_m[i, j], _ = nquad(
                pk.dipole_moment_integrand,
                [[0, 2.0*np.pi], [0, np.pi]],
                args=[mu, l1, m1, l2, m2, True]
            )

    plt.imshow(trans_moment_m, interpolation='nearest', extent=[-5,5,-5,5])
    plt.colorbar()
    plt.show()


def hydrogen_test1():
    """
    Radial wavefunction
    """
    r = np.linspace(0.0, 30.0, 500)
    l_m_list = [(1,0), (2,1), (3,0), (3,1), (5,2)]
    r_psi = np.zeros(len(l_m_list)*len(r)).reshape(len(l_m_list), len(r))

    for i, (l, m) in enumerate(l_m_list):
        # radial wavefunction and PDF
        psi = pk_h.radial_psi(r, l, m) * r
        r_psi[i] = pk.prob_density(psi)
        plt.plot(r, r_psi[i])

    plt.show()
    plt.clf()

    ns = range(1,10)
    r_exp = np.zeros(len(ns))
    l = 0.0
    for i, n in enumerate(ns):
        # radial expectation value
        r_exp[i], _ = quad(
            pk_h.radial_integrand,
            0.0, np.inf, args=(n, l)
        )
        plt.axvline(r_exp[i])

    plt.show()



def hydrogen_test2():
    """
    Wavefuntion energy values
    """
    qdict = {"n": 1, "l": 0, "m": 0}
    energy_list = []

    # quantum number n
    for n in range(1,4):
        # quantum number l fixed
        l = n-1
        # quantum number m
        for m in range(-l, l+1):
            # make dictionary, calculate and store the energy, add to a list
            qdict = {'n': n, 'l': l, 'm': m}
            qdict['energy'] = pk_h.hydrogen_energy(qdict['n'])
            energy_list.append(qdict)

    # sorted energy values
    e_vals = np.unique(np.array([e['energy'] for e in energy_list]))

    pk_plot.plot_energy_levels(energy_list)
    plt.show()


def hydrogen_test3():
    """
    Transition energies (Hydrogen Spectogram)
    """
    spectrum = []

    for n1 in range(1, 11):
        for n2 in range(1, 11):
            # check for valid transition
            if n1 != n2:
                qdict = {'n1': n1, 'n2': n2, 'l': 0, 'm': 0}
                qdict['energy'] = pk_h.hydrogen_transition_energy(
                    qdict['n1'], qdict['n2']
                )
                spectrum.append(qdict)

    # sorted energy values
    e_vals = np.unique([s['energy'] for s in spectrum])

    for e_val in e_vals:
        wavelength = pk.hartrees_to_wavelength(e_val)
        c = pk_plot.wavelength_to_colour(wavelength)
        plt.axvline(wavelength, color=c)

    plt.xlabel('Wavelength of light')
    plt.show()


def helium_test1():
    """
    Building the Helium Hamiltonian: H11
    """
    Z = 2
    h11 = pk_he.He_H11(Z)
    print(f'h11: {h11}')


def helium_test2():
    """
    Hamiltonian of a CI basis set
    """
    H11 = pk_he.H11_pm
    H12 = pk_he.H12_pm
    H22 = pk_he.H22_pm
    H = np.array([[H11, H12], [H12, H22]])


def helium_test3():
    """
    Compare the CI energies
    """

    hartree_to_ev = 27.211399  # conversion factor
    exact_g = -79.0
    verypoor_g = -108.8
    H11, H12, H22 = pk_he.H11_pm, pk_he.H12_pm, pk_he.H22_pm
    H = np.array([[H11, H12], [H12, H22]])
    evals, evecs = np.linalg.eigh(H)
    ci_g = evals[0] * hartree_to_ev
    c0_g = evecs[0][0]
    c1_g = evecs[1][0]
    rel_error = (abs(exact_g - ci_g)/abs(exact_g)) * 100

    print(f'evals: {evals}')
    print(f'H[0][0]: {H[0][0]}')
    print(f'11/4: {11.0/4}')

    print(f'Exact Energy --> {exact_g} eV')
    print(f'Very Poor Mans approx. --> {verypoor_g} eV')
    print(f'Our Configuration interaction estimate --> {ci_g} eV')
    print(f'With a relative error of {rel_error}%')


def hydrogen2_test1():
    """
    Energies and operators
    """

    r = np.linspace(0.2, 6, 100)

    plt.plot(r, pk_h.J(r))
    plt.plot(r, pk_h.S(r))
    plt.plot(r, pk_h.K(r))
    plt.show()
    plt.clf()

    plt.plot(r, pk_h.E_plus(r))
    plt.plot(r, pk_h.E_minus(r))
    plt.show()


def hydrogen2_test2():
    """
    Molecular Hydrogen derived
    """
    r = np.linspace(0.2, 6, 100)

    plt.plot(r, pk_h.J11(r))
    plt.show()
    plt.clf()

    plt.plot(r, pk_h.H2_H11(r))
    plt.plot(r, pk_h.H2_H12(r))
    plt.show()
    plt.clf()

    plt.plot(r, pk_h.int_1111(r))
    plt.plot(r, pk_h.int_1212(r))
    plt.plot(r, pk_h.int_1122(r))
    plt.plot(r, pk_h.int_1112(r))
    plt.show()


def hydrogen2_test3():
    """
    Molecular Hydrogen energies
    """
    r = np.linspace(0.2, 6.0, 100)

    e_ground = pk_h.H2_E_ground(r)
    e_excited = pk_h.H2_E_excited(r)
    e_min = np.min(e_ground)
    i_min = np.argmin(e_ground)
    r_min = r[i_min]
    print(e_min)
    print(i_min)
    print(r_min)

    print(f'Ground minima at {r_min} Bohr with E={e_min} Hartrees')

    plt.plot(r, e_ground)
    plt.plot(r, e_excited)
    plt.show()
    plt.clf()


def ci_test1():
    """
    Build CI matrix
    """
    r = np.linspace(0.2, 10.0, 100)
    ci_ground = np.zeros_like(r)
    ci_excited = np.zeros_like(r)

    for i, rad in enumerate(r):
        ci_g, ci_e = pk_h.H2_energy_CI(rad)
        ci_ground[i] = ci_g + pk_h.V(rad)
        ci_excited[i] = ci_e + pk_h.V(rad)

    e_ground = pk_h.H2_E_ground(r)
    e_excited = pk_h.H2_E_excited(r)

    plt.plot(r, ci_ground)
    plt.plot(r, ci_excited)
    plt.plot(r, e_ground)
    plt.plot(r, e_excited)
    plt.show()


def vpm_test1():
    """
    Optimizing Very Poor Man's Wave-Function
    """

    # zeta = 2
    zeta = 1.6875
    ev_per_hartree = 27.2114
    exact_e = -2.903
    e = pk_he.He_expected_phi1(zeta)
    print(f'expectation value for phi: {e}')

    x_init = np.linspace(0.5, 2.0, 6)
    # x_init = np.linspace(1.68745, 1.6755, 3)
    e_exact = -2.903

    for x in x_init:
        res = minimize(pk_he.He_expected_phi1, x)
        print(f'starting x: {x}')
        print(f'res.x: {res.x}')
        print(f'res.success: {res.success}')
        print(f'res.message: {res.message}')
        e_opt = pk_he.He_expected_phi1(res.x)
        e_diff = np.abs(e_exact - e_opt)
        print(f'e_diff: {e_diff}')


def pert_test1():
    """
    Make H(lambda) for He
    """

    test1 = pk_he.H0()
    test2 = pk_he.H1(0.5)
    test3 = pk_he.H_lambda(0.75)

    print('H0:\n', test1)
    print('H1:\n', test2)
    print('H_lambda:\n', test3)


def pert_test2():
    """
    Exploring lambda
    """

    lambdas = np.linspace(0,1,100)
    c_g = np.zeros(len(lambdas))
    c_e = np.zeros(len(lambdas))
    e_ground = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        H = pk_he.H_lambda(lam)
        evals, evecs = np.linalg.eigh(H)
        e_ground[i] = evals[0]
        c_g[i] = evecs[0][0]
        c_e[i] = evecs[0][1]

    plt.plot(lambdas, c_g**2)
    plt.plot(lambdas, c_e**2)
    plt.plot(lambdas, e_ground)
    plt.show()



if __name__=='__main__':

    print('*******************')
    print('*** PSIKO TESTS ***')
    print('*******************')

    # ====================
    # Square Wave tests
    # ====================

    # square_comp_test1()
    # square_test1()

    # ====================
    # 1D Classical Particle tests
    # ====================

    # time_plot_test1()
    # boundary_cond_test1()
    # forces_test1()
    # forces_test2()
    # forces_test3()

    # ====================
    # 1D Quantum Particle tests
    # ====================

    # pib_test1()
    # pib_test1_old()
    # pib_test2()
    # pib_test3()
    # pib_test3_old()
    # pib_interference_test1()
    # pib_interference_test1_old()
    # pib_interference_test2()
    # pib_interference_test2_old()
    # quadrature_test1()
    # quadrature_test2()

    # ====================
    # QM postulates
    # ====================

    # normalize_test1()
    # normalize_test1_old()
    schroedinger_test1_old()
    schroedinger_test1()
    # schroedinger_test2_old()
    # schroedinger_test2()
    # operator_test1()
    # operator_test2()

    # ====================
    # 1D Time-Independent Schroedinger Equation (TISE)
    # ====================

    # operator_test3()
    # operator_test4()
    # operator_test5()
    # operator_test6()

    # ====================
    # Quantum Mechanics in 2D
    # ====================

    # harmonic_2d_test1()
    # harmonic_2d_test2()
    # harmonic_2d_test3()
    # harmonic_2d_test4()

    # ====================
    # Spectrum via a Time-Dependent field
    # ====================

    # field_test1()
    # field_test2()
    # field_test3()
    # field_test4()

    # ====================
    # Phase and Momentum Space Intro
    # ====================

    # phase_space_test1()
    # phase_space_test2()
    # phase_space_test3()
    # phase_space_test4()

    # ====================
    # Quantum Tunneling and reactions
    # ====================

    # tunnel_test1()
    # tunnel_test2()
    # tunnel_test3()

    # ====================
    # Rotation theory
    # ====================

    # rotation_test1()
    # rotation_test2()
    # rotation_test3()
    # rotation_test4()
    # rotation_test5()
    # rotation_test6()

    # ====================
    # The Hydrogen Atom Intro
    # ====================

    # hydrogen_test1()
    # hydrogen_test2()
    # hydrogen_test3()

    # ====================
    # Helium Atom through Configuration Interaction (CI)
    # ====================

    # helium_test1()
    # helium_test2()
    # helium_test3()

    # ====================
    # The Hydrogen Molecule
    # ====================

    # hydrogen2_test1()
    # hydrogen2_test2()
    # hydrogen2_test3()

    # ====================
    # Configuration Interaction
    # ====================

    # ci_test1()

    # ====================
    # The variational principle
    # ====================

    # Note: long runtime
    # vpm_test1()

    # ====================
    # First Order Perturbation theory
    # ====================

    # pert_test1()
    # pert_test2()
