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

def internal_energy(p, velocity, rho, gamma, epsilon):
    vel_norm = np.linalg.norm(velocity, axis=-1)
    return p / (gamma-1.0) + 0.5 * epsilon * epsilon * np.multiply(rho, vel_norm**2)

# Vector-valued flux in x-direction
def F(q, gamma):
    u = q[..., 1]/q[..., 0]
    v = q[..., 2]/q[..., 0]
    e = q[..., 3]/q[..., 0]

    vel_norm = np.linalg.norm(np.stack([u,v], axis=-1), axis=-1)
    # Internal Energy
    e_internal = q[..., 3] - 0.5 * q[..., 0] * vel_norm**2
    p = (gamma - 1.0) * e_internal
    return np.stack([q[..., 1], p+u*q[..., 1], u*q[..., 2], q[..., 1]*e + p*u], axis=-1)

# Vector-valued flux in x-direction
def G(q, gamma):
    u = q[..., 1]/q[..., 0]
    v = q[..., 2]/q[..., 0]
    e = q[..., 3]/q[..., 0]

    vel_norm = np.linalg.norm(np.stack([u,v], axis=-1), axis=-1)
    # Internal Energy
    e_internal = q[..., 3] - 0.5 * q[..., 0] * vel_norm**2
    p = (gamma - 1.0) * e_internal
    return np.stack([q[..., 1], q[..., 1]*v, p+v*q[..., 2], q[..., 2]*e + p*v], axis=-1)

class Grid:
    def __init__(self, x_range, y_range, gamma, epsilon, cells_per_dim, halo_size):
        # Set up gridpoint coordinates
        x_spacing = np.linspace(*x_range, cells_per_dim)
        y_spacing = np.linspace(*y_range, cells_per_dim)

        # Create centered coordinate system
        xx_spacing, yy_spacing = np.meshgrid(x_spacing, y_spacing)
        xx_spacing -= 0.5
        yy_spacing -= 0.5

        # Enforce initial conditions
        self.center_idx = np.s_[halo_size:-halo_size]
        self.halo_size = halo_size
        self.dim_tot = cells_per_dim + 2*halo_size
        self.gamma = gamma

        u_init, v_init, p_init = isentropic_vortex_ic(xx_spacing,
                                                      yy_spacing,
                                                      gamma,
                                                      epsilon)
        rho_init = np.ones_like(u_init)
        velocity = np.stack([u_init, v_init], axis=-1)
        e_init = internal_energy(p_init, velocity, rho_init, gamma, epsilon)

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
        ds = 2*self.halo_size
        if shift == 'top':
            return self.q[self.center_idx, :-ds, :]
        elif shift == 'right':
            return self.q[ds:, self.center_idx, :]
        elif shift == 'bottom':
            return self.q[self.center_idx, ds:, :]
        elif shift == 'left':
            return self.q[:-ds, self.center_idx, :]
        else:
            return self.q[self.center_idx, self.center_idx, :]

    def set_q_inner(self, q_new):
        self.q[self.center_idx, self.center_idx, :] = q_new

    def get_F(self, shift):
        return F(self.get_q_inner(shift), self.gamma)

    def get_G(self, shift):
        return G(self.get_q_inner(shift), self.gamma)

    def enforce_periodic_bc(self):
        # x-dim
        left_bdry = self.q[1, :, :]
        self.q[0,...] = self.q[-2, :,:]
        self.q[-1,...] = left_bdry
        
        left_p = self.p[1, :]
        self.p[0, :] = self.p[-2, :]
        self.p[-1,:] = left_p

        # y-dim
        bottom_bdry = self.q[:, 1, :]
        self.q[:, 0, :] = self.q[-2, :, :]
        self.q[:, -1, :] = bottom_bdry

        bottom_p = self.p[1, :]
        self.p[:, 0] = self.p[:, -2]
        self.p[:,-1] = bottom_p


def main():
    # Adiabatic exponent
    gamma = 1.4
    epsilon = 1e-2

    # CFL Number
    cfl = 0.45

    x_range = [0.0, 1.0]
    y_range = [0.0, 1.0]

    cells_per_dim = 100
    halo_size = 1
    dx = (x_range[1] - x_range[0]) / cells_per_dim
    dy = (y_range[1] - y_range[0]) / cells_per_dim

    g = Grid(x_range, y_range, gamma, epsilon, cells_per_dim, halo_size)

    # Evolution with 2D Richtmyer scheme
    t = 0.0
    t_end = 1.0

    # Satisfy CFL condition
    rho = g.get_rho_inner()
    u = g.get_rho_u_inner()/rho
    v = g.get_rho_v_inner()/rho
    p = g.get_p_inner()
    
    velocity = np.stack([u,v], axis=-1)
    # Sound speed
    c = np.sqrt(gamma * p / rho)
    max_vel = np.max(np.abs(velocity), axis=-1)
    dt = 0.5 * dx / np.max(max_vel + c)
    print(dt)


    img_list = []
    fig, ax = plt.subplots()

    while t < t_end:
        g_new = copy.copy(g)

        # First Step
        q_new = 0.25*(g_new.get_q_inner(shift='top') + 
                      g_new.get_q_inner(shift='right') +
                      g_new.get_q_inner(shift='bottom') +
                      g_new.get_q_inner(shift='left'))
        q_new += 0.5*dt/dx * (g.get_F('right') - g.get_F('left') + 
                              g.get_G('bottom') - g.get_G('top'))
        g_new.set_q_inner(q_new)

        # Second Step
        g.set_q_inner(g.get_q_inner() - 
                        dt/dx * (g_new.get_F('right') - g_new.get_F('left') + 
                                 g_new.get_G('bottom') - g_new.get_G('top')))

        g.enforce_periodic_bc()

        # mpl_img = ax.imshow(g.get_rho_u() / g.get_rho(), animated=True, interpolation='none')

        # Append currenst state as frame animation list
        # img_list.append([mpl_img])

        t += dt

    # ax.imshow(g.get_p_inner())
    # ax.quiver(xx_spacing, yy_spacing, u, v)

    # ani = animation.ArtistAnimation(fig, img_list, interval=50, blit=True,
    #                             repeat_delay=1000)

    # Save frames as movie
    # writer = animation.FFMpegWriter(
    #     fps=40, metadata=dict(artist='Me'), bitrate=200)
    # ani.save("figures/movie.mp4", writer=writer, dpi=100)
    ax.imshow(g.get_rho_u())

    plt.show()

    return 0

if __name__ == "__main__":
    main()























