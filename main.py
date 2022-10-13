import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def isentropic_vortex_ic(xx_spacing, yy_spacing, gamma, epsilon):
    r_grid = np.sqrt((xx_spacing)**2 + (yy_spacing)**2)
    p_0 = 1.0 / (gamma * epsilon * epsilon) - 0.5

    def v_init(r):
        if r < 0.2:
            return 5.0*r
        elif r < 0.4:
            return 2.0 - 5.0*r
        else:
            return 0.0

    def p_init(r):
        if r < 0.2:
            return p_0 + 12.5*r*r
        elif r < 0.4:
            return p_0 + 4.0*np.log(5*r) + 4.0 - 20.0*r + 12.5*r*r
        else:
            return p_0 + 4.0*np.log(2.0) - 2.0

    v_phi = np.vectorize(v_init)(r_grid)
    p = np.vectorize(p_init)(r_grid)

    return np.multiply(v_phi, xx_spacing), np.multiply(v_phi, yy_spacing), p

def total_cell_energy(p, velocity, rho, gamma):
    vel_norm = np.linalg.norm(velocity, axis=-1)
    return p / (gamma - 1.0) + 0.5 * np.multiply(rho, vel_norm**2)

def primitive_vars(q, gamma):
    u = q[..., 1]/q[..., 0]
    v = q[..., 2]/q[..., 0]
    e = q[..., 3]

    vel_norm = np.linalg.norm(np.stack([u,v], axis=-1), axis=-1)

    # Internal Energy
    e_internal = e - 0.5 * q[..., 0] * vel_norm**2
    p = (gamma - 1.0) * e_internal

    return u, v, e, p

# Vector-valued flux in x-direction
def F(q, gamma):
    u, _, e, p = primitive_vars(q, gamma)
    return np.stack([q[..., 1], p+u*q[..., 1], u*q[..., 2], u*(e + p)], axis=-1)

# Vector-valued flux in y-direction
def G(q, gamma):
    _, v, e, p = primitive_vars(q, gamma)
    return np.stack([q[..., 2], v*q[..., 1], p+v*q[..., 2], v*(e + p)], axis=-1)

class Grid:
    def __init__(self, x_range, y_range, gamma, epsilon, cells_per_dim, halo_size):
        # Set up gridpoint coordinates
        x_spacing = np.linspace(*x_range, cells_per_dim)
        y_spacing = np.linspace(*y_range, cells_per_dim)

        # Create centered coordinate system
        xx_spacing, yy_spacing = np.meshgrid(x_spacing, y_spacing)
        xx_spacing -= 0.5
        yy_spacing -= 0.5

        # Index helpers for shifted grids
        self.ds = 2*halo_size
        self.hs = halo_size
        self.center_idx = np.s_[self.hs:-self.hs]
        self.plus_idx = np.s_[self.ds:]
        self.minus_idx = np.s_[:-self.ds]

        self.dim_tot = cells_per_dim + 2*self.hs
        self.gamma = gamma

        # Enforce initial conditions
        u_init, v_init, p_init = isentropic_vortex_ic(xx_spacing,
                                                      yy_spacing,
                                                      gamma,
                                                      epsilon)
        rho_init = np.ones_like(u_init)
        velocity = np.stack([u_init, v_init], axis=-1)
        e_init = total_cell_energy(p_init, velocity, rho_init, gamma)

        self.q = np.empty((self.dim_tot, self.dim_tot, 4))
        self.p = np.empty((self.dim_tot, self.dim_tot))
        self.p[self.center_idx, self.center_idx] = p_init
        
        # System state
        self.q[self.center_idx, 
               self.center_idx] = np.stack([rho_init,
                                            u_init,
                                            v_init,
                                            e_init], axis=-1)

        # Enforce b.c.
        self.enforce_periodic_bc()

    # Getters and Setters
    def get_q(self):
        return self.q

    def get_rho(self):
        return self.q[..., 0]

    def get_rho_u(self):
        return self.q[..., 1]

    def get_rho_v(self):
        return self.q[..., 2]

    def get_e(self):
        return self.q[..., 3]

    def get_p(self):
        return self.p

    # Grid without halo cells
    def get_rho_inner(self):
        return self.q[self.center_idx, self.center_idx, 0]

    def get_rho_u_inner(self):
        return self.q[self.center_idx, self.center_idx, 1]

    def get_rho_v_inner(self):
        return self.q[self.center_idx, self.center_idx, 2]

    def get_e_inner(self):
        return self.q[self.center_idx, self.center_idx, 3]

    def get_p_inner(self):
        return self.p[self.center_idx, self.center_idx]

    # Return view, potentially also boundary in one directionj
    def get_q_inner(self, shift=None):
        if shift == 'top':
            return self.q[self.minus_idx, self.center_idx, :]
        elif shift == 'topright':
            return self.q[self.minus_idx, self.plus_idx, :]
        elif shift == 'topleft':
            return self.q[self.minus_idx, self.minus_idx, :]
        elif shift == 'right':
            return self.q[self.center_idx, self.plus_idx, :]
        elif shift == 'bottomright':
            return self.q[self.plus_idx, self.plus_idx, :]
        elif shift == 'bottom':
            return self.q[self.plus_idx, self.center_idx, :]
        elif shift == 'bottomleft':
            return self.q[self.plus_idx, self.minus_idx, :]
        elif shift == 'left':
            return self.q[self.center_idx, self.minus_idx, :]
        elif shift is None or 'center':
            return self.q[self.center_idx, self.center_idx, :]
        else:
            raise ValueError('Shift type not known.')

    def set_q_inner(self, q_new):
        self.q[self.center_idx, self.center_idx, :] = q_new

    def get_F(self, shift):
        return F(self.get_q_inner(shift), self.gamma)

    def get_G(self, shift):
        return G(self.get_q_inner(shift), self.gamma)

    def enforce_periodic_bc(self):
        # Update Pressure
        _, _, _, p = primitive_vars(self.get_q_inner(), self.gamma)
        if np.array_equal(p, self.p[self.center_idx, self.center_idx]):
            print('Should be equal in the first step!!!!')
        self.p[self.center_idx, self.center_idx] = p

        # x-dim
        bottom_bdry = self.q[:, self.hs:self.ds, :]
        self.q[:, :self.hs, :] = self.q[:, -self.ds:-self.hs, :]
        self.q[:, -self.hs:, :] = bottom_bdry

        bottom_p = self.p[:, self.hs:self.ds]
        self.p[:, :self.hs] = self.p[:, -self.ds:-self.hs]
        self.p[:,-self.hs:] = bottom_p
        
        # y-dim
        left_bdry = self.q[self.hs:self.ds, :, :]
        self.q[:self.hs, :] = self.q[-self.ds:-self.hs, :,:]
        self.q[-self.hs, :] = left_bdry
        
        left_p = self.p[self.hs:self.ds, :]
        self.p[:self.hs, :] = self.p[-self.ds:-self.hs, :]
        self.p[-self.hs:,:] = left_p


def main():
    # Adiabatic exponent
    gamma = 1.4
    epsilon = 1e-2

    # CFL Number
    cfl = 0.95

    x_range = [0.0, 1.0]
    y_range = [0.0, 1.0]

    cells_per_dim = 50
    halo_size = 1
    dx = (x_range[1] - x_range[0]) / cells_per_dim
    dy = (y_range[1] - y_range[0]) / cells_per_dim

    g = Grid(x_range, y_range, gamma, epsilon, cells_per_dim, halo_size)

    # Evolution with 2D Richtmyer scheme
    t = 0.0
    t_end = 0.1

    # Satisfy CFL condition
    rho = g.get_rho_inner()
    u, v, _, p = primitive_vars(g.get_q_inner(), gamma)
    
    velocity = np.stack([u,v], axis=-1)
    # Sound speed
    c = np.sqrt(gamma * p / rho)
    max_vel = np.max(np.abs(velocity), axis=-1)
    dt = cfl * min(dx, dy) / np.max(max_vel + c)
    print(f'dt = {dt}')

    img_list = []
    fig, ax = plt.subplots()

    print_ctr = 0
    while t < t_end:
        if not print_ctr%50:
            print(t)
        print_ctr += 1

        # First Step
        # topright (half-step)
        g_topright = copy.deepcopy(g)
        q_topright = 0.25 * (g_topright.get_q_inner(shift='top') + 
                             g_topright.get_q_inner(shift='topright') +
                             g_topright.get_q_inner(shift='right') +
                             g_topright.get_q_inner(shift='center'))
        
        q_topright -= 0.25*dt/dx * (g.get_F('topright') + g.get_F('right') - 
                                    g.get_F('top') - g.get_F('center'))
        q_topright -= 0.25*dt/dy * (g.get_G('right') + g.get_G('center') - 
                                    g.get_G('top') - g.get_G('topright'))
        
        g_topright.set_q_inner(q_topright)

        # topleft (half-step)
        g_topleft = copy.deepcopy(g)
        q_topleft = 0.25 * (g_topleft.get_q_inner(shift='top') +
                             g_topleft.get_q_inner(shift='topleft') +
                             g_topleft.get_q_inner(shift='left') +
                             g_topleft.get_q_inner(shift='center'))

        q_topleft -= 0.25*dt/dx * (g.get_F('top') + g.get_F('center') -
                                   g.get_F('topleft') - g.get_F('left'))
        q_topleft -= 0.25*dt/dy * (g.get_G('left') + g.get_G('center') -
                                   g.get_G('topleft') - g.get_G('top'))
        
        g_topleft.set_q_inner(q_topleft)

        # bottomleft (half-step)
        g_bottomleft = copy.deepcopy(g)
        q_bottomleft = 0.25 * (g_bottomleft.get_q_inner(shift='bottom') +
                             g_bottomleft.get_q_inner(shift='bottomleft') +
                             g_bottomleft.get_q_inner(shift='left') +
                             g_bottomleft.get_q_inner(shift='center'))

        q_bottomleft -= 0.25*dt/dx * (g.get_F('bottom') + g.get_F('center') -
                                      g.get_F('bottomleft') - g.get_F('left'))
        q_bottomleft -= 0.25*dt/dy * (g.get_G('bottom') + g.get_G('bottomleft') -
                                      g.get_G('left') - g.get_G('center'))

        g_bottomleft.set_q_inner(q_bottomleft)

        # bottomright (half-step)
        g_bottomright = copy.deepcopy(g)
        q_bottomright = 0.25 * (g_bottomright.get_q_inner(shift='bottom') +
                             g_bottomright.get_q_inner(shift='bottomright') +
                             g_bottomright.get_q_inner(shift='right') +
                             g_bottomright.get_q_inner(shift='center'))

        q_bottomright -= 0.25*dt/dx * (g.get_F('bottomright') + g.get_F('right') -
                                       g.get_F('center') - g.get_F('bottom'))
        q_bottomright -= 0.25*dt/dy * (g.get_G('bottom') + g.get_G('bottomright') -
                                       g.get_G('right') - g.get_G('center'))

        g_bottomright.set_q_inner(q_bottomright)

        # Second Step
        g.set_q_inner(g.get_q_inner('center') -
                        0.5*dt/dx * (g_topright.get_F('center') + g_bottomright.get_F('center') -
                                       g_topleft.get_F('center') - g_bottomleft.get_F('center')) -
                        0.5*dt/dy * (g_bottomleft.get_G('center') + g_bottomright.get_G('center') -
                                       g_topleft.get_G('center') - g_topright.get_G('center')))

        g.enforce_periodic_bc()

        mpl_img = ax.pcolormesh(g.get_p())#, animated=True, interpolation='none')
        ax.set_aspect('equal')

        # Append currenst state as frame animation list
        # if not print_ctr%20:
        #     img_list.append([mpl_img])
        img_list.append([mpl_img])
        # if t > 100*dt:
        #     break
        # if np.isclose(t,1*dt):
        #     ax.imshow(g.get_p())
        #     plt.show()

        t += dt

    # ax.imshow(g.get_p_inner())
    # ax.quiver(xx_spacing, yy_spacing, u, v)

    ani = animation.ArtistAnimation(fig, img_list, interval=15, blit=True,
                                repeat_delay=1000)

    # Save frames as movie
    writer = animation.FFMpegWriter(
        fps=25, metadata=dict(artist='Me'), bitrate=120)
    ani.save("figures/movie.mp4", writer=writer, dpi=80)
    # ax.imshow(g.get_rho_u())

    plt.show()

    return 0

if __name__ == "__main__":
    main()























