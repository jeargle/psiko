# John Eargle
# 2017


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, quad, nquad

import psiko.psiko as pk


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
    print('np.shape(y):', np.shape(y))
    y2 = np.array([pk.square2(x, omega, i) for i in ns])
    print('np.shape(y2):', np.shape(y2))
    print('np.abs(y-y2).max():', np.abs(y-y2).max())
    print('np.abs(y-y2).sum():', np.abs(y-y2).sum())
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
    l = 10
    x = np.linspace(0, l, 1000)
    y = np.zeros(3*len(x)).reshape(3, len(x))

    # calculate 4 harmonics
    for n in range(1,4):
        y[n-1] = pk.pib_ti_1D(x, n, l)

    for i in y:
        plt.plot(x, i)
    plt.legend()
    plt.show()


def pib_test2():
    l = 10.0
    c = 0.1
    t = np.linspace(0, 100, 1000)
    y = np.zeros(4*len(t)).reshape(4, len(t))

    # use a for loop for the multiple harmonics
    for n in range(1,5):
        y[n-1] = pk.pib_td_1D(t, c, n, l)

    for i in y:
        plt.plot(t, i)
    plt.legend()
    plt.show()


def pib_test3():
    c = 0.1
    l = 10
    x = np.arange(0, l, 0.01)
    t = np.arange(0, 30, 0.1)
    y = np.zeros((len(x), len(t)))
    n = 3

    for step, time in enumerate(t):
        # time-dependent and time-independent terms
        # y[:, step] = pk.pib_td_1D(time, c, n, l) * pk.pib_ti_1D(x, n, l)
        y[:, step] = pk.wave_solution(x, time, c, n, l)

    pk.time_plot(x, y, t)
    plt.show()


def pib_interference_test1():
    """
    Plot time trace for a single position in a wavefunction made from
    the first two eigenstates.
    """
    t = np.arange(0, 100, 0.1)
    c = 0.5
    l = 10
    x = l/3.0
    wave = np.zeros(len(t))

    # sum 2 eigenstates
    for step, time in enumerate(t):
        wave[step] = (pk.wave_solution(x, time, c, 1, l) +
                      pk.wave_solution(x, time, c, 2, l))

    plt.plot(t, wave)
    plt.show()


def pib_interference_test2():
    """
    Plot time evolution of full wavefunction made from the first two
    eigenstates.
    """
    c = 0.5
    l = 10
    x = np.linspace(0, l, 100)
    t = np.arange(0, 30, 0.1)
    y = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    for step, time in enumerate(t):
        y[:, step] = (pk.wave_solution(x, time, c, 1, l) +
                      pk.wave_solution(x, time, c, 2, l))

    pk.time_plot(x, y, t, timestep=1)
    plt.show()


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
    l = 10.0
    # l = 1.0
    x = np.arange(0, l, 0.01)
    psi_x = np.zeros(len(x))

    # Build wavefunction from 4 eigenfunctions
    for n in range(1,5):
        psi_x += pk.pib_ti_1D(x, n, l)

    # Get PDF and normalize for psi_x
    pdf = pk.prob_density(psi_x)
    psi_normed = pk.normalize_wfn(x, psi_x)

    norm_pre = pk.psi_norm(x, psi_x)
    norm_post = pk.psi_norm(x, psi_normed)
    print('Norm pre:', norm_pre)
    print('Norm post:', norm_post)

    plt.plot(x, psi_x)
    plt.plot(x, pdf)
    plt.show()


def schroedinger_test1():
    """
    Set up a mixed state from the ground and first excited states for
    the particle in a box.  Normalize the wavefunction.
    """
    l = 10
    x = np.arange(0, l, 0.01)

    # First eigenstate
    psi1_x = pk.pib_ti_1D(x, 1, l)
    c1 = 1.0/np.sqrt(2)
    E1 = pk.pib_energy(1, l)

    # Second eigenstate
    psi2_x = pk.pib_ti_1D(x, 2, l)
    c2 = 1.0/np.sqrt(2)
    E2 = pk.pib_energy(2, l)

    # Mixed state
    psi0 = c1*psi1_x + c2*psi2_x
    psi0_norm = pk.psi_norm(x, psi0)
    print('Norm is ', psi0_norm)

    plt.plot(x, psi0)
    plt.show()


def schroedinger_test2():
    """
    Time-dependent Schroedinger equation.
    """
    l = 10
    x = np.arange(0, l, 0.01)
    t = np.linspace(0, 50, 100)
    psi = np.zeros(len(x)*len(t)).reshape(len(x), len(t))

    # First eigenstate
    c1_0 = 1/np.sqrt(2)
    psi1_x = pk.pib_ti_1D(x, 1, l)
    E1 = pk.pib_energy(1, l)

    # Second eigenstate
    c2_0 = 1/np.sqrt(2)
    psi2_x = pk.pib_ti_1D(x, 2, l)
    E2 = pk.pib_energy(2, l)

    for step, time in enumerate(t):
        # Get time evolved coefficients
        c1 = pk.cnt_evolve(c1_0, time, E1)
        c2 = pk.cnt_evolve(c2_0, time, E2)
        psi[:, step] = c1*psi1_x + c2*psi2_x

    pk.time_plot(x, psi, t)
    plt.show()


def operator_test1():
    """
    Position and momentum operators.
    """
    l = 10
    dx = 0.01
    x = np.arange(0, l, dx)
    c1 = 1.0/np.sqrt(2)
    c2 = 1.0/np.sqrt(2)
    psi1_x = pk.pib_ti_1D(x, 1, l)
    psi2_x = pk.pib_ti_1D(x, 2, l)

    psi0 = c1*psi1_x + c2*psi2_x

    x_integrand = psi0 * x * psi0
    exp_x = pk.complex_simps(x_integrand, x)
    print('Expectation of position:', exp_x)

    p_integrand = psi0 * pk.momentum_operator(psi0, x, dx)
    exp_p = pk.complex_simps(p_integrand, x)
    print('Expectation of momentum:', exp_p)


def operator_test2():
    """
    Observables and expectation values.
    """
    l = 10
    dx = 0.01
    x = np.arange(0, l, dx)
    t = np.arange(0, 100)
    psi = pk.pib_superposition(x, t, l, 1, 2)
    p_array = np.zeros(len(t))
    x_array = np.zeros(len(t))

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
    dx =0.02
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
    vxy = pk.harmonic_potential_2D(xx, yy, kx, ky)

    pk.plot_surface(xx, yy, vxy)
    plt.show()

    plt.clf()
    pk.plot_contours(xx, yy, vxy)
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

    ho = pk.harmonic_oscillator_2D(xx, yy, l, m)

    pk.plot_surface(xx, yy, ho)
    plt.show()

    plt.clf()
    pk.plot_contours(xx, yy, ho)
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
    psi1 = pk.harmonic_oscillator_2D(xx, yy, l1, m1)
    E1 = l1 + m1 + 1
    c1 = 1.0/np.sqrt(2)

    psi2 = pk.harmonic_oscillator_2D(xx, yy, l2, m2)
    E2 = l2 + m2 + 1
    c2 = c1

    # superposition
    psi = c1*psi1 + c2*psi2

    pk.plot_contours(xx, yy, psi)
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
    psi1 = pk.harmonic_oscillator_2D(xx, yy, l1, m1)
    E1 = l1 + m1 + 1
    c1 = 1.0/np.sqrt(2)

    psi2 = pk.harmonic_oscillator_2D(xx, yy, l2, m2)
    E2 = l2 + m2 + 1
    c2 = c1

    for step, time in enumerate(t):
        c1_t = pk.cnt_evolve(c1, time, E1)
        c2_t = pk.cnt_evolve(c2, time, E2)
        psi[step] = c1_t*psi1 + c2_t*psi2

    vmin = np.min(psi)
    vmax = np.max(psi)

    for i in range(0, 80, 8):
        # pk.plot_surface(xx, yy, psi[i])
        pk.plot_contours(xx, yy, psi[i], vmin=vmin, vmax=vmax)
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

    test = pk.harmonic_oscillator_1D_in_field(x, ti, omega_f)
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
        psit[:,step] = pk.harmonic_oscillator_1D_in_field(x, time, omega_f)

    pk.time_plot(x, psit, t, timestep=8)
    plt.show()


def field_test3():
    """
    Excited states overlap.
    """
    t = np.arange(0, 10, 0.01)
    omegas = [3.0, 5.0, 8.0, 10.0]
    prob_excited_state = np.zeros(len(omegas)*len(t)).reshape(len(omegas), len(t))

    # iterate over EM field frequencies
    for omega_idx, omega in enumerate(omegas):
        c1 = pk.excited_overlap(t, omega)
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
        c1t = pk.excited_overlap(t, omega)
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
    print('delta_x:', delta_x)
    print('delta_p:', delta_p)
    print('uncertainty:', uncertainty)

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
    wxp = pk.harmonic_oscillator_wigner(xx, pp, omega)

    pk.plot_contours(xx, pp, wxp)
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
        wxpt[i] = pk.harmonic_oscillator_wigner(
            xx - np.cos(omega*time),
            pp - np.sin(omega*time),
            omega
        )

    zmin = np.min(wxpt)
    zmax = np.max(wxpt)
    print('zmin:', zmin)
    print('zmax:', zmax)

    for i in range(0, 80, 20):
        pk.plot_contours(xx, pp, wxpt[i])
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
        wxpt[i] = pk.harmonic_oscillator_wigner_01(xx, pp, time)

    z_min = np.min(wxpt)
    z_max = np.max(wxpt)
    print('z_min:', z_min)
    print('z_max:', z_max)

    for i in range(0, 80, 8):
        pk.plot_contours(xx, pp, wxpt[i])
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
    # m = 1
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
    Transmission vs mass/energy (2-D scan)
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

    pk.plot_contours(ee, mm, transmission)
    plt.show()


def rotation_test1():
    """
    Rotational states: Spherical Harmonics
    """
    # spherical harmonics (m=2, l=2)
    pk.plot_sphere(2, 2)
    plt.show()
    plt.clf()

    # spherical harmonics (m=1,l=2)
    pk.plot_sphere(1, 2)
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
        args=[mu, 0, 0, 0, 0]
    )
    dipole_00_alt, _ = pk.complex_nquad(
        pk.dipole_moment_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, 0, 0, 0, 0]
    )

    print("Dipole moment for $Y^0_0$ is (nquad): ", dipole_00)
    print("Dipole moment for $Y^0_0$ is (complex_nquad): ", dipole_00_alt)
    print("Difference between approaches", np.abs(dipole_00-dipole_00_alt))

    # Expectation for dipole moment Y_1^1
    dipole_11, _ = nquad(
        pk.dipole_moment_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, 1, 1, 1, 1]
    )
    dipole_11_alt, _ = pk.complex_nquad(
        pk.dipole_moment_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, 1, 1, 1, 1]
    )

    print("The dipole moment for $Y^1_1$ is (nquad): ", dipole_11)
    print("The dipole moment for $Y^1_1$ is (complex_nquad): ", dipole_11_alt)
    print("Difference between approaches", np.abs(dipole_11-dipole_11_alt))


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
        args=[mu, c1_0, c2_0, 0, 0, 1, 0]
    )
    print("The dipole moment for $Y^0_0$+$Y^1_0$ is (nquad): ", dipole_1)

    dipole_2, _ = nquad(
        pk.dipole_moment_superposition_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, c1_0, c2_0, 0, 0, 1, 1]
    )
    print("The dipole moment for $Y^0_0$+$Y^1_1$ is (nquad): ", dipole_2)

    dipole_3, _ = nquad(
        pk.dipole_moment_superposition_integrand,
        [[0, 2.0*np.pi], [0, np.pi]],
        args=[mu, c1_0, c2_0, 0, 0, 1, -1]
    )
    print("The dipole moment for $Y^0_0$+$Y^1_-1$ is (nquad): ", dipole_3)


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
        c1 = pk.cnt_evolve(c1_0, ti, E1, hbar=1.0)
        c2 = pk.cnt_evolve(c2_0, ti, E2, hbar=1.0)

        # compute dipole moments, get values, and store absolute values
        m2 = 0
        dipoles_1[i], _ = nquad(
            pk.dipole_moment_superposition_integrand,
            [[0, 2.0*np.pi], [0, np.pi]],
            args=[mu, c1, c2, l1, m1, l2, m2]
        )
        m2 = 1
        dipoles_2[i], _ = nquad(
            pk.dipole_moment_superposition_integrand,
            [[0, 2.0*np.pi], [0, np.pi]],
            args=[mu, c1, c2, l1, m1, l2, m2]
        )
        m2 = -1
        dipoles_3[i], _ = nquad(
            pk.dipole_moment_superposition_integrand,
            [[0, 2.0*np.pi], [0, np.pi]],
            args=[mu, c1, c2, l1, m1, l2, m2]
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
                args=[mu, l1, m1, l2, m2]
            )

    plt.imshow(trans_moment_l, interpolation='nearest')
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
                args=[mu, l1, m1, l2, m2]
            )

    plt.imshow(trans_moment_m, interpolation='nearest', extent=[-5,5,-5,5])
    plt.colorbar()
    plt.show()




if __name__=='__main__':

    print '*******************'
    print '*** PSIKO TESTS ***'
    print '*******************'

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
    # pib_test2()
    # pib_test3()
    # pib_interference_test1()
    # pib_interference_test2()
    # quadrature_test1()
    # quadrature_test2()

    # ====================
    # QM postulates
    # ====================

    # normalize_test1()
    # schroedinger_test1()
    # schroedinger_test2()
    # operator_test1()
    # operator_test2()

    # ====================
    # 1-D Time-Independent Schroedinger Equation (TISE)
    # ====================

    # operator_test3()
    # operator_test4()
    # operator_test5()
    # operator_test6()

    # ====================
    # Quantum Mechanics in 2-D
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
    rotation_test5()
    rotation_test6()
