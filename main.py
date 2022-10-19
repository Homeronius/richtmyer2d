import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gridtools import Grid, primitive_vars, mach_number, compact_vorticity

def main(eps, t_end):
    fig, ax = plt.subplots()

    # Adiabatic exponent
    gamma = 1.4

    # CFL Number
    cfl = 0.45

    # x_range = [5.0, 5.0]
    # y_range = [5.0, 5.0]
    
    x_range = [0.0, 1.0]
    y_range = [0.0, 1.0]

    cells_per_dim = 50
    halo_size = 1
    dx = (x_range[1] - x_range[0]) / cells_per_dim
    dy = (y_range[1] - y_range[0]) / cells_per_dim

    g = Grid(x_range, y_range, gamma, eps, cells_per_dim, halo_size)

    # Satisfy CFL condition
    rho = g.get_rho_inner()
    u, v, _, _, p = primitive_vars(g.get_q_inner(), gamma)
    
    velocity = np.stack([u,v], axis=-1)
    # Sound speed
    c = np.sqrt(gamma * p / rho)
    max_vel = np.max(np.abs(velocity), axis=-1)
    dt = cfl * min(dx, dy) / np.max(max_vel + c)
    print(f'dt = {dt}')

    img_list = []

    # Position
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    t = 0.0
    print_ctr = 0
    tot_steps = int((t_end - t)/dt)
    frame_every = tot_steps//140
    create_movie = True
    while t < t_end:
        # First Step
        # topright (half-step)
        g_topright = copy.deepcopy(g)
        q_topright = 0.25 * (g_topright.get_q_inner(shift='top') + 
                             g_topright.get_q_inner(shift='topright') +
                             g_topright.get_q_inner(shift='right') +
                             g_topright.get_q_inner(shift='center'))

        # topleft (half-step)
        g_topleft = copy.deepcopy(g)
        q_topleft = 0.25 * (g_topleft.get_q_inner(shift='top') +
                             g_topleft.get_q_inner(shift='topleft') +
                             g_topleft.get_q_inner(shift='left') +
                             g_topleft.get_q_inner(shift='center'))

        # bottomleft (half-step)
        g_bottomleft = copy.deepcopy(g)
        q_bottomleft = 0.25 * (g_bottomleft.get_q_inner(shift='bottom') +
                             g_bottomleft.get_q_inner(shift='bottomleft') +
                             g_bottomleft.get_q_inner(shift='left') +
                             g_bottomleft.get_q_inner(shift='center'))

        # bottomright (half-step)
        g_bottomright = copy.deepcopy(g)
        q_bottomright = 0.25 * (g_bottomright.get_q_inner(shift='bottom') +
                             g_bottomright.get_q_inner(shift='bottomright') +
                             g_bottomright.get_q_inner(shift='right') +
                             g_bottomright.get_q_inner(shift='center'))

        # Compute vorticity
        if create_movie and not print_ctr%frame_every:
            vorticity = compact_vorticity(q_topleft, q_topright, q_bottomleft, q_bottomright)

        # Evolve half timestep
        q_topright -= 0.25*dt/dx * (g.get_F('topright') + g.get_F('right') - 
                                    g.get_F('top') - g.get_F('center'))
        q_topright -= 0.25*dt/dy * (g.get_G('right') + g.get_G('center') - 
                                    g.get_G('top') - g.get_G('topright'))
        
        q_topleft -= 0.25*dt/dx * (g.get_F('top') + g.get_F('center') -
                                   g.get_F('topleft') - g.get_F('left'))
        q_topleft -= 0.25*dt/dy * (g.get_G('left') + g.get_G('center') -
                                   g.get_G('topleft') - g.get_G('top'))
        
        q_bottomleft -= 0.25*dt/dx * (g.get_F('bottom') + g.get_F('center') -
                                      g.get_F('bottomleft') - g.get_F('left'))
        q_bottomleft -= 0.25*dt/dy * (g.get_G('bottom') + g.get_G('bottomleft') -
                                      g.get_G('left') - g.get_G('center'))

        q_bottomright -= 0.25*dt/dx * (g.get_F('bottomright') + g.get_F('right') -
                                       g.get_F('center') - g.get_F('bottom'))
        q_bottomright -= 0.25*dt/dy * (g.get_G('bottom') + g.get_G('bottomright') -
                                       g.get_G('right') - g.get_G('center'))

        g_topright.set_q_inner(q_topright)
        g_topleft.set_q_inner(q_topleft)
        g_bottomleft.set_q_inner(q_bottomleft)
        g_bottomright.set_q_inner(q_bottomright)

        # Second Step
        g.set_q_inner(g.get_q_inner('center') -
                        0.5*dt/dx * (g_topright.get_F('center') + g_bottomright.get_F('center') -
                                       g_topleft.get_F('center') - g_bottomleft.get_F('center')) -
                        0.5*dt/dy * (g_bottomleft.get_G('center') + g_bottomright.get_G('center') -
                                       g_topleft.get_G('center') - g_topright.get_G('center')))

        g.enforce_periodic_bc()

        if create_movie and not print_ctr%frame_every:
            # Compute Mach number
            M = mach_number(g.get_q_inner(), gamma)

            # Append currenst state as frame animation list
            img_list.append(M)

            # Log vorticity together with primitive variables
            mean_vorticity = np.mean(vorticity)
            g.log_mean_state(time=t, vorticity=mean_vorticity)

        if not print_ctr%frame_every:
            print(t)
        print_ctr += 1

        t += dt

    # Log statistics
    g.logger.save(f'logging/logged_vars_{eps:.1E}.csv')

    if not create_movie:
        return 0

    # Save frames as movie
    vmax_tot = np.max(np.asarray(img_list))
    vmin_tot = np.min(np.asarray(img_list))

    im = ax.imshow(img_list[0], vmin=vmax_tot, vmax=vmax_tot)
    fig.colorbar(im, cax=cax)
    ax.set_aspect('equal')
    ax.set_title(f'Mach number at t = {0.00}')

    print(f'Created {len(img_list)} frames')

    def animate(i):
        arr = img_list[i]
        im.set_data(arr)
        im.set_clim(vmin_tot, vmax_tot)
        ax.set_title(f'Mach number at t = {frame_every*float(i)*dt:.2f}')
        return im

    ani = animation.FuncAnimation(fig, animate, interval=20, frames=len(img_list))
    writer = animation.FFMpegWriter(
        fps=20, metadata=dict(artist='Me'), bitrate=200)
    ani.save(f'figures/movie_{eps}.mp4', writer=writer, dpi=100)

    plt.show()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read cmdline inputs')
    parser.add_argument('eps', metavar='eps', type=float,
                        help='set epsilon parameter')
    parser.add_argument('t_end', metavar='t_end', type=float,
                        help='set t_end parameter')
    args = parser.parse_args()
    main(eps=args.eps, t_end=args.t_end)
