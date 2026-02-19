"""
Core solver for Williamson fluid bifurcation analysis.
Contains the DirectNavierStokesSolver class implementing the full governing equations.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
import json
import pickle

class SolutionStore:
    """
    Class to store solutions for different flow rates
    """
    def __init__(self):
        self.solutions = []  # List of dictionaries for each F

    def add_solution(self, F, X, Y, psi, u, v, solver_params=None):
        """
        Add a solution for a specific flow rate
        """
        solution = {
            'F': float(F),
            'X': X.tolist() if X is not None else None,
            'Y': Y.tolist() if Y is not None else None,
            'psi': psi.tolist() if psi is not None else None,
            'u': u.tolist() if u is not None else None,
            'v': v.tolist() if v is not None else None,
            'params': solver_params.copy() if solver_params else None
        }
        self.solutions.append(solution)

    def get_solution(self, F, tolerance=1e-6):
        """
        Retrieve solution for a specific flow rate
        """
        for sol in self.solutions:
            if abs(sol['F'] - F) < tolerance:
                return sol
        return None

    def get_all_F(self):
        """Get all flow rates in the store"""
        return [sol['F'] for sol in self.solutions]

    def clear(self):
        """Clear all stored solutions"""
        self.solutions = []

    def __len__(self):
        return len(self.solutions)

    def __repr__(self):
        return f"SolutionStore with {len(self)} solutions for F = {self.get_all_F()}"

    def save_to_json(self, filename):
        """Save solutions to JSON file"""
        with open(filename, 'w') as f:
            json.dump({'solutions': self.solutions}, f, indent=2)
        print(f"Solutions saved to {filename}")

    def save_to_pickle(self, filename):
        """Save solutions to pickle file (preserves numpy arrays)"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Solutions pickled to {filename}")

    @classmethod
    def load_from_pickle(cls, filename):
        """Load solutions from pickle file"""
        with open(filename, 'rb') as f:
            store = pickle.load(f)
        return store


class DirectNavierStokesSolver:
    """
    Direct Navier-Stokes solver - solves for given parameters without continuation.
    Solves the full governing equations without lubrication approximation.
    """
    def __init__(self, Nx=21, Ny=21, Re=0.0, We=0.0, delta=0.1, F=1.0,
                 a=0.0, b=0.0, d=1.0, phi=0.0, store_solutions=False):

        # Parameters
        self.Nx = Nx
        self.Ny = Ny
        self.Re = Re
        self.We = We
        self.delta = delta
        self.F = F
        self.a = a
        self.b = b
        self.d = d
        self.phi = phi
        self.store_solutions = store_solutions

        # Initialize solution store
        self.solution_store = SolutionStore()

        # Domain
        self.Lx = np.pi  # Periodic in x
        self.dx = self.Lx / (Nx - 1)

        # Transformed coordinate η ∈ [0, 1]
        self.eta = np.linspace(0, 1, Ny)
        self.deta = 1.0 / (Ny - 1)

        # Physical x-grid
        self.x = np.linspace(-self.Lx, self.Lx, Nx)

        # Wall geometries
        self.h1 = 1 + a * np.cos(2 * np.pi * self.x)
        self.h2 = -d - b * np.cos(2 * np.pi * self.x + phi)
        self.H = self.h1 - self.h2

        # Transformation metrics
        self.compute_metrics()

        # Number of unknowns (all grid points)
        self.total_points = Nx * Ny

        # Mapping from (i,j) to vector index and vice versa
        self.create_index_mapping()

        # Solution vector
        self.psi_vec = np.zeros(self.total_points)

        # Velocity fields
        self.u_field = None
        self.v_field = None

        # Newton iteration history
        self.newton_history = {'iter': [], 'residual': []}

    def compute_metrics(self):
        """Compute transformation metrics"""
        Nx, Ny = self.Nx, self.Ny

        # ∂η/∂y = 1/H
        self.deta_dy = np.zeros((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                self.deta_dy[i, j] = 1.0 / self.H[i]

        # Precompute (dη/dy)² and (dη/dy)⁴ for efficiency
        self.deta_dy_sq = self.deta_dy**2
        self.deta_dy_quartic = self.deta_dy**4

    def create_index_mapping(self):
        """Create mapping between (i,j) and vector index"""
        self.idx_map = np.zeros((self.Nx, self.Ny), dtype=int)
        self.idx_inv = []  # List of (i,j) for each index

        idx = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.idx_map[i, j] = idx
                self.idx_inv.append((i, j))
                idx += 1

    def get_psi_ij(self, psi_vec, i, j, handle_boundaries=True):
        """Get ψ(i,j) from solution vector"""
        if not handle_boundaries:
            return psi_vec[self.idx_map[i % self.Nx, j]]

        i_wrapped = i % self.Nx

        if j < 0:
            return psi_vec[self.idx_map[i_wrapped, 0]]
        elif j >= self.Ny:
            return psi_vec[self.idx_map[i_wrapped, self.Ny-1]]
        else:
            return psi_vec[self.idx_map[i_wrapped, j]]

    def initial_guess(self):
        """Create initial guess based on parameters"""
        psi_vec = np.zeros(self.total_points)

        for idx in range(self.total_points):
            i, j = self.idx_inv[idx]
            η = self.eta[j]
            H = self.H[i]

            # For Stokes-like initial guess
            # Ψ(η) = A + Bη + Cη² + Dη³
            # with Ψ(0) = -F/2, Ψ(1) = F/2, Ψ_y(0) = Ψ_y(1) = -1

            A = -self.F/2
            B = -H  # because Ψ_η = Ψ_y * (dη/dy) = -1 * H

            # Solve for C and D:
            C = 3 * (self.F + H)
            D = -2 * (self.F + H)

            psi_vec[idx] = A + B*η + C*η**2 + D*η**3

        return psi_vec

    def compute_derivatives(self, psi_vec, i, j):
        """Compute derivatives needed for full equation at point (i,j)"""
        # Get indices with periodic wrapping
        im1 = (i - 1) % self.Nx
        ip1 = (i + 1) % self.Nx
        im2 = (i - 2) % self.Nx
        ip2 = (i + 2) % self.Nx

        # Get ψ value
        ψ = self.get_psi_ij(psi_vec, i, j, handle_boundaries=False)

        # First derivatives (central where possible)
        ψ_x = (self.get_psi_ij(psi_vec, ip1, j, handle_boundaries=False) -
               self.get_psi_ij(psi_vec, im1, j, handle_boundaries=False)) / (2 * self.dx)

        if j == 0:  # Bottom - forward
            ψ_y_eta = (self.get_psi_ij(psi_vec, i, 1, handle_boundaries=False) - ψ) / self.deta
        elif j == self.Ny-1:  # Top - backward
            ψ_y_eta = (ψ - self.get_psi_ij(psi_vec, i, self.Ny-2, handle_boundaries=False)) / self.deta
        else:  # Interior - central
            ψ_y_eta = (self.get_psi_ij(psi_vec, i, j+1, handle_boundaries=False) -
                       self.get_psi_ij(psi_vec, i, j-1, handle_boundaries=False)) / (2 * self.deta)

        # Second derivatives
        ψ_xx = (self.get_psi_ij(psi_vec, ip1, j, handle_boundaries=False) -
                2*ψ + self.get_psi_ij(psi_vec, im1, j, handle_boundaries=False)) / (self.dx**2)

        if j == 0:  # Bottom
            ψ_yy_eta = (self.get_psi_ij(psi_vec, i, 2, handle_boundaries=False) -
                        2*self.get_psi_ij(psi_vec, i, 1, handle_boundaries=False) + ψ) / (self.deta**2)
        elif j == self.Ny-1:  # Top
            ψ_yy_eta = (ψ - 2*self.get_psi_ij(psi_vec, i, self.Ny-2, handle_boundaries=False) +
                        self.get_psi_ij(psi_vec, i, self.Ny-3, handle_boundaries=False)) / (self.deta**2)
        else:  # Interior
            ψ_yy_eta = (self.get_psi_ij(psi_vec, i, j+1, handle_boundaries=False) -
                        2*ψ + self.get_psi_ij(psi_vec, i, j-1, handle_boundaries=False)) / (self.deta**2)

        # Mixed derivative
        if 0 < j < self.Ny-1:
            ψ_xy_eta = (self.get_psi_ij(psi_vec, ip1, j+1, handle_boundaries=False) -
                        self.get_psi_ij(psi_vec, ip1, j-1, handle_boundaries=False) -
                        self.get_psi_ij(psi_vec, im1, j+1, handle_boundaries=False) +
                        self.get_psi_ij(psi_vec, im1, j-1, handle_boundaries=False)) / (4 * self.dx * self.deta)
        else:
            ψ_xy_eta = 0.0

        # Transform to physical coordinates
        η_y = self.deta_dy[i, j]

        ψ_y = ψ_y_eta * η_y
        ψ_yy = ψ_yy_eta * η_y**2
        ψ_xy = ψ_xy_eta * η_y

        # For interior points, compute higher derivatives
        ψ_xxxx = ψ_xxyy = ψ_yyyy = 0.0

        if 2 <= i <= self.Nx-3 and 2 <= j <= self.Ny-3:
            # Ψ_xxxx
            ψ_xxxx = (self.get_psi_ij(psi_vec, ip2, j, handle_boundaries=False) -
                      4*self.get_psi_ij(psi_vec, ip1, j, handle_boundaries=False) +
                      6*ψ -
                      4*self.get_psi_ij(psi_vec, im1, j, handle_boundaries=False) +
                      self.get_psi_ij(psi_vec, im2, j, handle_boundaries=False)) / (self.dx**4)

            # Ψ_yyyy in transformed coordinates
            ψ_yyyy_eta = (self.get_psi_ij(psi_vec, i, j+2, handle_boundaries=False) -
                          4*self.get_psi_ij(psi_vec, i, j+1, handle_boundaries=False) +
                          6*ψ -
                          4*self.get_psi_ij(psi_vec, i, j-1, handle_boundaries=False) +
                          self.get_psi_ij(psi_vec, i, j-2, handle_boundaries=False)) / (self.deta**4)
            ψ_yyyy = ψ_yyyy_eta * η_y**4

            # Ψ_xxyy (9-point stencil)
            ψ_ip1_jp1 = self.get_psi_ij(psi_vec, ip1, j+1, handle_boundaries=False)
            ψ_ip1_jm1 = self.get_psi_ij(psi_vec, ip1, j-1, handle_boundaries=False)
            ψ_im1_jp1 = self.get_psi_ij(psi_vec, im1, j+1, handle_boundaries=False)
            ψ_im1_jm1 = self.get_psi_ij(psi_vec, im1, j-1, handle_boundaries=False)
            ψ_i_jp1 = self.get_psi_ij(psi_vec, i, j+1, handle_boundaries=False)
            ψ_i_jm1 = self.get_psi_ij(psi_vec, i, j-1, handle_boundaries=False)

            ψ_xxyy_eta = (ψ_ip1_jp1 - 2*ψ_i_jp1 + ψ_im1_jp1 -
                          2*self.get_psi_ij(psi_vec, ip1, j, handle_boundaries=False) + 4*ψ -
                          2*self.get_psi_ij(psi_vec, im1, j, handle_boundaries=False) +
                          ψ_ip1_jm1 - 2*ψ_i_jm1 + ψ_im1_jm1) / (self.dx**2 * self.deta**2)
            ψ_xxyy = ψ_xxyy_eta * η_y**2

        return {
            'ψ': ψ, 'ψ_x': ψ_x, 'ψ_y': ψ_y, 'ψ_xx': ψ_xx, 'ψ_yy': ψ_yy,
            'ψ_xy': ψ_xy, 'ψ_xxxx': ψ_xxxx, 'ψ_xxyy': ψ_xxyy, 'ψ_yyyy': ψ_yyyy,
            'η_y': η_y
        }

    def compute_C1_C2(self, derivs):
        """Compute C1 = ψ_y·ψ_xy - ψ_x·ψ_yy and C2 = ψ_y·ψ_xx - ψ_x·ψ_xy"""
        C1 = derivs['ψ_y'] * derivs['ψ_xy'] - derivs['ψ_x'] * derivs['ψ_yy']
        C2 = derivs['ψ_y'] * derivs['ψ_xx'] - derivs['ψ_x'] * derivs['ψ_xy']
        return C1, C2

    def compute_F_ij(self, psi_vec, i, j):
        """
        Compute F(i,j) for the full equation.
        """
        # Boundary conditions
        if j == 0:  # Bottom wall: Ψ = -F/2
            ψ = self.get_psi_ij(psi_vec, i, j, handle_boundaries=False)
            return ψ + self.F/2.0

        elif j == self.Ny-1:  # Top wall: Ψ = F/2
            ψ = self.get_psi_ij(psi_vec, i, j, handle_boundaries=False)
            return ψ - self.F/2.0

        elif j == 1:  # Near bottom wall: ∂Ψ/∂y = -1
            ψ_0 = self.get_psi_ij(psi_vec, i, 0, handle_boundaries=False)
            ψ_1 = self.get_psi_ij(psi_vec, i, 1, handle_boundaries=False)
            ψ_2 = self.get_psi_ij(psi_vec, i, 2, handle_boundaries=False)

            ψ_y = (-3*ψ_0 + 4*ψ_1 - ψ_2) / (2 * self.deta) * self.deta_dy[i, 0]
            return ψ_y + 1.0

        elif j == self.Ny-2:  # Near top wall: ∂Ψ/∂y = -1
            ψ_N = self.get_psi_ij(psi_vec, i, self.Ny-1, handle_boundaries=False)
            ψ_Nm1 = self.get_psi_ij(psi_vec, i, self.Ny-2, handle_boundaries=False)
            ψ_Nm2 = self.get_psi_ij(psi_vec, i, self.Ny-3, handle_boundaries=False)

            ψ_y = (3*ψ_N - 4*ψ_Nm1 + ψ_Nm2) / (2 * self.deta) * self.deta_dy[i, self.Ny-1]
            return ψ_y + 1.0

        # Interior points: compute full equation
        if j < 2 or j >= self.Ny-2 or i < 2 or i >= self.Nx-2:
            # For near-boundary points, use simple approximation
            return 0.0

        # Get derivatives
        derivs = self.compute_derivatives(psi_vec, i, j)

        # Compute C1 and C2 at this point
        C1, C2 = self.compute_C1_C2(derivs)

        # Get derivatives of C1 and C2 using finite differences
        # ∂C1/∂y
        dC1_dy = 0.0
        if j+1 < self.Ny-1 and j-1 > 0:
            derivs_jp1 = self.compute_derivatives(psi_vec, i, j+1)
            derivs_jm1 = self.compute_derivatives(psi_vec, i, j-1)

            C1_jp1, _ = self.compute_C1_C2(derivs_jp1)
            C1_jm1, _ = self.compute_C1_C2(derivs_jm1)

            dC1_dy_eta = (C1_jp1 - C1_jm1) / (2 * self.deta)
            dC1_dy = dC1_dy_eta * derivs['η_y']

        # ∂C2/∂x
        dC2_dx = 0.0
        if i > 0 and i < self.Nx-1:
            derivs_ip1 = self.compute_derivatives(psi_vec, (i+1) % self.Nx, j)
            derivs_im1 = self.compute_derivatives(psi_vec, (i-1) % self.Nx, j)

            _, C2_ip1 = self.compute_C1_C2(derivs_ip1)
            _, C2_im1 = self.compute_C1_C2(derivs_im1)

            dC2_dx = (C2_ip1 - C2_im1) / (2 * self.dx)

        # Convective terms
        convective = self.delta * self.Re * (dC1_dy + self.delta**2 * dC2_dx)

        # Viscous/viscoelastic terms
        viscous = -(1 + self.We) * (
            self.delta**4 * derivs['ψ_xxxx'] +
            2 * self.delta**2 * derivs['ψ_xxyy'] +
            derivs['ψ_yyyy']
        )

        # Total F
        F_ij = convective + viscous

        return F_ij

    def compute_residual_vector(self, psi_vec):
        """Compute full residual vector F(Ψ)"""
        F = np.zeros(self.total_points)

        for idx in range(self.total_points):
            i, j = self.idx_inv[idx]
            F[idx] = self.compute_F_ij(psi_vec, i, j)

        return F

    def compute_Jacobian_FD(self, psi_vec, eps=1e-6):
        """Compute Jacobian using finite differences"""
        start_time = time.time()

        N = self.total_points
        J = lil_matrix((N, N))

        # Base residual
        F0 = self.compute_residual_vector(psi_vec)

        for l in range(N):
            # Perturb l-th variable
            psi_pert = psi_vec.copy()
            psi_pert[l] += eps

            # Compute perturbed residual
            F_pert = self.compute_residual_vector(psi_pert)

            # Finite difference derivative
            J[:, l] = ((F_pert - F0) / eps)[:, np.newaxis]

        elapsed = time.time() - start_time

        return J.tocsr()

    def solve(self, max_iter=100, tol=1e-4):
        """
        Solve directly for given parameters using Newton's method.
        """
        # Start with initial guess
        self.psi_vec = self.initial_guess()
        psi_vec = self.psi_vec.copy()

        for iter_num in range(max_iter):
            # Compute residual
            F = self.compute_residual_vector(psi_vec)
            norm_F = np.linalg.norm(F)

            self.newton_history['iter'].append(iter_num)
            self.newton_history['residual'].append(norm_F)

            # Check convergence
            if norm_F < tol:
                self.psi_vec = psi_vec
                # Compute velocities for this solution
                self.compute_velocities()

                # Store solution if requested
                if self.store_solutions:
                    self.store_current_solution()
                return True

            # Compute Jacobian
            J = self.compute_Jacobian_FD(psi_vec)

            # Solve linear system
            try:
                # Add small regularization
                I = diags([1e-8], [0], shape=J.shape)
                J_reg = J + I
                delta_psi = spsolve(J_reg, -F)
            except Exception as e:
                delta_psi = np.linalg.lstsq(J.toarray(), -F, rcond=1e-6)[0]

            # Line search
            alpha = 1.0
            norm_F0 = norm_F
            psi_best = psi_vec.copy()

            for ls_iter in range(10):
                psi_new = psi_vec + alpha * delta_psi
                F_new = self.compute_residual_vector(psi_new)
                norm_F_new = np.linalg.norm(F_new)

                if norm_F_new < norm_F0:
                    psi_best = psi_new.copy()
                    norm_F0 = norm_F_new
                    alpha *= 1.5
                else:
                    alpha *= 0.5

                if alpha < 1e-4:
                    break

            psi_vec = psi_best

            # Ensure boundary conditions are satisfied exactly
            for idx in range(self.total_points):
                i, j = self.idx_inv[idx]
                if j == 0:  # Bottom wall
                    psi_vec[idx] = -self.F/2.0
                elif j == self.Ny-1:  # Top wall
                    psi_vec[idx] = self.F/2.0

        self.psi_vec = psi_vec
        # Compute velocities even if not fully converged
        self.compute_velocities()

        # Store solution if requested
        if self.store_solutions:
            self.store_current_solution()
        return False

    def store_current_solution(self):
        """Store current solution in the solution store"""
        solver_params = {
            'Nx': self.Nx, 'Ny': self.Ny,
            'Re': self.Re, 'We': self.We,
            'delta': self.delta,
            'a': self.a, 'b': self.b,
            'd': self.d, 'phi': self.phi
        }

        # Convert to grid format
        psi_grid = self.psi_to_grid()
        u_grid, v_grid = self.compute_velocity(psi_grid)

        # Get physical coordinates
        X_phys, Y_phys = self.get_physical_coordinates()

        # Store in solution store WITH PHYSICAL COORDINATES
        self.solution_store.add_solution(
            self.F,
            X_phys,  # Add X physical coordinates
            Y_phys,  # Add Y physical coordinates
            psi_grid,
            u_grid,
            v_grid,
            solver_params
        )

    def compute_velocities(self):
        """Compute and store velocity fields"""
        psi_grid = self.psi_to_grid()
        self.u_field, self.v_field = self.compute_velocity(psi_grid)

    def psi_to_grid(self, psi_vec=None):
        """Convert solution vector to 2D grid"""
        if psi_vec is None:
            psi_vec = self.psi_vec

        psi_grid = np.zeros((self.Nx, self.Ny))
        for i in range(self.Nx):
            for j in range(self.Ny):
                psi_grid[i, j] = psi_vec[self.idx_map[i, j]]

        return psi_grid

    def compute_velocity(self, psi_grid=None):
        """Compute velocity field u = Ψ_y, v = -δΨ_x"""
        if psi_grid is None:
            psi_grid = self.psi_to_grid()

        u = np.zeros((self.Nx, self.Ny))
        v = np.zeros((self.Nx, self.Ny))

        for i in range(self.Nx):
            for j in range(self.Ny):
                # Ψ_x (periodic)
                ip1 = (i + 1) % self.Nx
                im1 = (i - 1) % self.Nx
                ψ_x = (psi_grid[ip1, j] - psi_grid[im1, j]) / (2 * self.dx)

                # Ψ_y (careful at boundaries)
                if j == 0:  # Bottom - one-sided
                    ψ_y = (-3*psi_grid[i, 0] + 4*psi_grid[i, 1] - psi_grid[i, 2]) / (2 * self.deta) * self.deta_dy[i, 0]
                elif j == self.Ny-1:  # Top - one-sided
                    ψ_y = (3*psi_grid[i, -1] - 4*psi_grid[i, -2] + psi_grid[i, -3]) / (2 * self.deta) * self.deta_dy[i, -1]
                else:  # Interior - central
                    ψ_y = (psi_grid[i, j+1] - psi_grid[i, j-1]) / (2 * self.deta) * self.deta_dy[i, j]

                u[i, j] = ψ_y
                v[i, j] = -self.delta * ψ_x

        return u, v

    def get_physical_coordinates(self):
        """Get physical (x, y) coordinates for the grid"""
        X_phys = np.zeros((self.Nx, self.Ny))
        Y_phys = np.zeros((self.Nx, self.Ny))

        for i in range(self.Nx):
            for j in range(self.Ny):
                X_phys[i, j] = self.x[i]
                Y_phys[i, j] = self.h2[i] + self.eta[j] * self.H[i]

        return X_phys, Y_phys

    def plot_solution(self):
        """Plot the solution"""
        psi_grid = self.psi_to_grid()
        u, v = self.compute_velocity(psi_grid)

        # Get physical coordinates
        X_phys, Y_phys = self.get_physical_coordinates()

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Stream function in PHYSICAL COORDINATES
        contour1 = axes[0, 0].contourf(X_phys, Y_phys, psi_grid,
                                      levels=20, cmap='RdBu')
        axes[0, 0].set_title(f'Stream Function Ψ\nRe={self.Re}, We={self.We}, δ={self.delta}, F={self.F}')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y (physical)')
        plt.colorbar(contour1, ax=axes[0, 0])
        axes[0, 0].plot(self.x, self.h1, 'k-', linewidth=2)
        axes[0, 0].plot(self.x, self.h2, 'k-', linewidth=2)

        # Velocity magnitude
        vel_magnitude = np.sqrt(u**2 + v**2)
        contour2 = axes[0, 1].contourf(X_phys, Y_phys, vel_magnitude,
                                      levels=20, cmap='viridis', extend='both')
        axes[0, 1].set_title('Velocity Magnitude')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y (physical)')
        plt.colorbar(contour2, ax=axes[0, 1])
        axes[0, 1].plot(self.x, self.h1, 'w-', linewidth=1, alpha=0.7)
        axes[0, 1].plot(self.x, self.h2, 'w-', linewidth=1, alpha=0.7)

        # Velocity profile at middle x
        mid_x = self.Nx // 2
        axes[1, 0].plot(u[mid_x, :], Y_phys[mid_x, :], 'b-', linewidth=2)
        axes[1, 0].set_xlabel('u velocity')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_title(f'Velocity profile at x={self.x[mid_x]:.2f}')
        axes[1, 0].grid(True)

        # Channel geometry
        axes[1, 1].plot(self.x, self.h1, 'r-', label='Upper wall', linewidth=2)
        axes[1, 1].plot(self.x, self.h2, 'b-', label='Lower wall', linewidth=2)
        axes[1, 1].fill_between(self.x, self.h2, self.h1, alpha=0.3)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('Channel Geometry')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_aspect('equal', adjustable='box')

        # Check wall velocity BC
        ψ_y_bottom = np.zeros(self.Nx)
        ψ_y_top = np.zeros(self.Nx)

        for i in range(self.Nx):
            ψ_y_bottom[i] = (-3*psi_grid[i, 0] + 4*psi_grid[i, 1] - psi_grid[i, 2]) / (2 * self.deta) * self.deta_dy[i, 0]
            ψ_y_top[i] = (3*psi_grid[i, -1] - 4*psi_grid[i, -2] + psi_grid[i, -3]) / (2 * self.deta) * self.deta_dy[i, -1]

        axes[1, 2].plot(self.x, ψ_y_bottom, 'r-', label='Bottom wall', linewidth=2)
        axes[1, 2].plot(self.x, ψ_y_top, 'b-', label='Top wall', linewidth=2)
        axes[1, 2].axhline(y=-1, color='k', linestyle='--', label='Target: -1', alpha=0.5)
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('∂Ψ/∂y')
        axes[1, 2].set_title('Wall Velocity BC Check')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.show()