# John Eargle
# 2017-2022

import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits import axes_grid1
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm



# ====================
# Helper Functions
# ====================

def sphere_harm_real(m, l, phi, theta):
    """
    Spherical harmonics in real space.
    """
    Y_lm = sph_harm(m, l, phi, theta)
    if m < 0:
        Y_lm_real = np.sqrt(2.0) * (-1.0)**m * Y_lm.imag
    elif m > 0:
        Y_lm_real = np.sqrt(2.0) * (-1.0)**m * Y_lm.real
    else:
        Y_lm_real = Y_lm
    return Y_lm_real

def sphere_to_cart(theta, phi, r=1.0):
    """
    Converts spherical coordinates to 3D cartesian coordinates.
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return x, y, z


# ====================
# Plotting
# ====================

def time_plot(x, y, t, timestep=1):
    for i in range(0, len(t), timestep):
        plt.plot(x, y[:,i])

def time_plot_psi(psi_traj, timestep=1, imaginary=False):
    """
    Plot full wavefunction trajectory on a static graph.  Timesteps
    are plotted in order so that later states are drawn on top of
    earlier ones.

    psi: Psi for particle in a box
    timestep: step to take through the trajectory array
    """

    psi = psi_traj.psi
    x = psi.x
    y = psi_traj.traj
    t = psi_traj.time
    # dt = psi_traj.dt

    if imaginary:
        for i in range(0, len(t), timestep):
            plt.plot(x, y[:,i].imag)
    else:
        for i in range(0, len(t), timestep):
            plt.plot(x, y[:,i].real)

def plot_trisurf(x, y):
    """
    Plot complex wavefunction cylindrically around an origin
    axis.  The function is connected to the origin axis by a
    surface.

    x: domain; origin axis
    y: complex-valued wavefunction
    """
    # Order points by those on the domain axis followed by those in
    # the wavefunction.
    len_x = len(x)
    point_x = np.concatenate((x, x))
    point_y = np.concatenate((np.zeros(len_x), y.real))
    point_z = np.concatenate((np.zeros(len_x), y.imag))

    tri1 = [[i, i+len_x+1, i+len_x] for i in range(len_x-1)]
    tri2 = [[i, i+1, i+len_x+1] for i in range(len_x-1)]
    triangles = np.concatenate((tri1, tri2))

    # viridis, plasma, inferno, magma, cividis
    # spring, summer, autumn, winter, cool, hot, copper
    # coolwarm, bwr, seismic
    # twilight, hsv
    # flag, prism, brg, rainbow, jet, turbo
    cmap = mpl.cm.get_cmap('jet')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(point_x, point_y, point_z, triangles=triangles, cmap=cmap)

def plot_quiver(x, y):
    """
    Plot complex wavefunction cylindrically as arrows pointing
    out from an origin axis.  The function lies at the tips of
    the arrows.

    x: domain; origin axis
    y: complex-valued wavefunction
    """
    # Order points by those on the domain axis followed by those in
    # the wavefunction.
    len_x = len(x)

    # Origin points.
    point_x = x
    point_y = np.zeros(len_x)
    point_z = np.zeros(len_x)

    # Arrow directions.
    dir_u = np.zeros(len_x)
    dir_v = y.real
    dir_w = y.imag

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ylim([-1.0, 1.0])
    # ax.quiver(point_x, point_y, point_z, dir_u, dir_v, dir_w, length=0.1)
    # ax.quiver(point_x, point_y, point_z, dir_u, dir_v, dir_w, normalize=True)
    cmap = mpl.cm.get_cmap('jet')
    q = ax.quiver(
        point_x, point_y, point_z,
        dir_u, dir_v, dir_w,
        arrow_length_ratio=0,
        cmap=cmap,
    )
    # q.set_array(np.random.rand(np.prod(x.shape)))
    q.set_array(dir_w)
    ax.set_zlim([-1.0, 1.0])  # No plt.zlim() available.
    ax.view_init(0, -90)

def plot_surface(xx, yy, zz):
    """
    Plot a surface.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, rstride=10, cstride=10, alpha=0.25)
    # Set viewing angle
    ax.view_init(30, -60)
    # ax.view_init(80, -60)
    # Add contour lines
    ax.contour(xx, yy, zz, zdir='z')
    plt.xlabel('x')
    plt.ylabel('y')

def plot_contours(xx, yy, zz, vmin=None, vmax=None):
    """
    Plot a heatmap and contours for a 3d surface.
    """
    # Use viridis colormap
    cmap = mpl.cm.get_cmap('viridis')
    plt.contour(xx, yy, zz, linewidths=3.0, cmap=cmap)
    contour = plt.contour(xx, yy, zz, colors='k', linewidths=0.5)
    im = plt.imshow(zz, interpolation='nearest',
                    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    plt.clabel(contour, fontsize='x-small', inline=1)
    _add_colorbar(im, label='z')
    plt.xlabel('x')
    plt.ylabel('y')

def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0/aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes('right', size=width, pad=pad)
    plt.sca(current_ax)

    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def plot_sphere(m, l):
    """
    Plot a function onto a sphere.
    """
    theta = np.arange(0, 2 * np.pi, 0.01)
    phi = np.arange(0, np.pi, 0.01)
    theta_mg, phi_mg = np.meshgrid(theta, phi)
    x, y, z = sphere_to_cart(theta_mg, phi_mg, r=1.0)

    Y_real = sphere_harm_real(m, l, theta_mg, phi_mg)

    # Color points on sphere.
    cmap = mpl.cm.get_cmap(name='seismic', lut=None)
    cm = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
    mapped_Y = cm.to_rgba(Y_real)
    cm.set_array(mapped_Y)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=mapped_Y)
    fig.colorbar(cm, shrink=0.5)
    ax.view_init(20, 45)
    ax.set_title('l=' + str(l) + ' m=' + str(m))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def plot_energy_levels(energies, figsize=(14, 6), fontsize='xx-small'):
    """
    Energy level plot.

    energies: list of dicts
    """
    e = [i['energy'] for i in energies]
    e_max = np.max(e)
    e_min = np.min(e)
    e_diff = e_max - e_min
    linewidth = 200.0
    offset_to_add = 480

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    prev_e_val = 0.0
    offset = 0.0
    i = 0.0
    for energy_dict in energies:
        e_val = energy_dict['energy']
        if prev_e_val == e_val:
            offset += offset_to_add
            i += 1
        else:
            offset = 0.0
            i = 0.0

        yalign = 'top' if i % 2 else 'bottom'
        y_offset = -e_diff * 0.01 if i % 2 else e_diff * 0.01

        xmin = -linewidth / 2.0 + offset
        xmax = linewidth / 2.0 + offset
        plt.hlines(e_val, xmin, xmax, linewidth=2)
        ax.annotate(
            _energy_label(energy_dict), xy=(xmin, e_val + y_offset),
            fontsize=fontsize, horizontalalignment='left',
            verticalalignment=yalign
        )
        prev_e_val = energy_dict['energy']

    plt.xlim([-linewidth / 2.0 - 5.0, 2000.0])
    plt.ylim([e_min - 0.05 * e_diff, e_max + 0.05 * e_diff])
    plt.tick_params(axis='x', which='both', bottom='off',
                    top='off', labelbottom='off')
    plt.ylabel('Energy')

def _energy_label(energy_dict):
    """
    Create string with quantum numbers n, l, and m.
    """
    vals = []

    if 'n' in energy_dict:
        vals.append('n={:d}'.format(energy_dict['n']))
    if 'l' in energy_dict:
        vals.append('l={:d}'.format(energy_dict['l']))
    if 'm' in energy_dict:
        vals.append('m={:d}'.format(energy_dict['m']))

    return ', '.join(vals)

def wavelength_to_colour(wavelength):
    """
    Get RGB values for a given wavelength of light.
    """
    if wavelength >= 380 and wavelength < 440:
        R = -(wavelength - 440.0) / (440.0 - 350.0)
        G = 0.0
        B = 1.0
    elif wavelength >= 440 and wavelength < 490:
        R = 0.0
        G = (wavelength - 440.0) / (490.0 - 440.0)
        B = 1.0
    elif wavelength >= 490 and wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510.0) / (510.0 - 490.0)
    elif wavelength >= 510 and wavelength < 580:
        R = (wavelength - 510.0) / (580.0 - 510.0)
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength < 645:
        R = 1.0
        G = -(wavelength - 645.0) / (645.0 - 580.0)
        B = 0.0
    elif wavelength >= 645 and wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    # intensity correction
    if wavelength >= 380 and wavelength < 420:
        intensity = 0.3 + 0.7 * (wavelength - 350) / (420 - 350)
    elif wavelength >= 420 and wavelength <= 700:
        intensity = 1.0
    elif wavelength > 700 and wavelength <= 780:
        intensity = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        intensity = 0.0

    return (intensity * R, intensity * G, intensity * B)

def traj_plot(x, traj, t, dt=1, xlim=None, ylim=None, skip=1, gif=None, mp4=None, show=False):
    """
    Create an animated plot for a wavefunction trajectory.

    x: x-axis array
    traj: 2D array of function sampling at different timepoints
    t: time point array
    dt: timestep
    xlim: tuple with x-axis bounds
    ylim: tuple with y-axis bounds
    skip: number of timepoints to skip between each frame
    gif: gif filename
    mp4: mp4 filename
    show: whether or not to show the plot during execution
    """

    if xlim is None:
        xlim = (x[0], x[-1])

    if ylim is None:
        ylim = (-1.0, 1.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=xlim, ylim=ylim)
    ax.grid()

    # line, = ax.plot([], [], 'o-', lw=2)
    line, = ax.plot([], [])
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = x
        thisy = traj[:,i]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(t), skip),
                                  interval=50, blit=True, init_func=init)

    if gif is not None:
        ani.save(gif, dpi=80, fps=15, writer='imagemagick')
    if mp4 is not None:
        ani.save(mp4, fps=15)
    if show:
        plt.show()

def traj_plot_psi(psi_traj, d_type='real', xlim=None, ylim=None, skip=1,
                  gif=None, mp4=None, show=False):
    """
    Create an animated plot for a wavefunction trajectory.

    psi_traj: PsiTraj
    d_type: 'real', 'imaginary', or 'complex' (2 lines)
    xlim: tuple with x-axis bounds
    ylim: tuple with y-axis bounds
    skip: number of timepoints to skip between each frame
    gif: gif filename
    mp4: mp4 filename
    show: whether or not to show the plot during execution
    """

    psi = psi_traj.psi
    x = psi.x
    t = psi_traj.time
    dt = psi_traj.dt

    if xlim is None:
        xlim = (x[0], x[-1])

    if ylim is None:
        ylim = (-1.0, 1.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=xlim, ylim=ylim)
    ax.grid()

    # line, = ax.plot([], [], 'o-', lw=2)
    # line, = ax.plot([], [])
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    plot_num = 1
    if d_type == 'complex':
        plot_num = 2

    colors = ["blue", "green"]
    lines = []
    for index in range(plot_num):
        # lobj = ax.plot([], [], lw=2, color=colors[index])[0]
        lobj = ax.plot([], [], color=colors[index])[0]
        lines.append(lobj)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(i):
        thisx = x

        if d_type == 'real':
            thisy1 = psi_traj.traj[:,i].real
            xlist = [thisx]
            ylist = [thisy1]
        elif d_type == 'imaginary':
            thisy1 = psi_traj.traj[:,i].imag
            xlist = [thisx]
            ylist = [thisy1]
        elif d_type == 'complex':
            thisy1 = psi_traj.traj[:,i].real
            thisy2 = psi_traj.traj[:,i].imag
            xlist = [thisx, thisx]
            ylist = [thisy1, thisy2]

        for i, line in enumerate(lines):
            line.set_data(xlist[i], ylist[i]) # set data for each line separately.

        return lines

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(t), skip),
                                  interval=50, blit=True, init_func=init)

    if gif is not None:
        ani.save(gif, dpi=80, fps=15, writer='imagemagick')
    if mp4 is not None:
        ani.save(mp4, fps=15)
    if show:
        plt.show()
