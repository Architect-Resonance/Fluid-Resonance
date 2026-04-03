import numpy as np

class SpectralNSSolver:
    def __init__(self, N, L=2*np.pi, nu=0.01, use_dealias=True):
        self.N = N
        self.L = L
        self.nu = nu
        self.use_dealias = use_dealias
        self.dx = L / N
        self.dt = 0.001
        
        # Real-FFT Grid (N, N, N//2 + 1)
        kx = np.fft.fftfreq(N, d=L/(2*np.pi*N))
        ky = np.fft.fftfreq(N, d=L/(2*np.pi*N))
        kz = np.fft.rfftfreq(N, d=L/(2*np.pi*N))
        
        self.kx, self.ky, self.kz = np.meshgrid(kx, ky, kz, indexing='ij')
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2
        self.k_sq[0,0,0] = 1.0 # Avoid div by zero
        
        # Projection P_ij = delta_ij - k_i*k_j / k^2
        self.P11 = 1.0 - self.kx*self.kx/self.k_sq
        self.P12 = -self.kx*self.ky/self.k_sq
        self.P13 = -self.kx*self.kz/self.k_sq
        self.P22 = 1.0 - self.ky*self.ky/self.k_sq
        self.P23 = -self.ky*self.kz/self.k_sq
        self.P33 = 1.0 - self.kz*self.kz/self.k_sq
        
        self.k_sq[0,0,0] = 0.0 # Reset
        self.u_hat = None

    def initialize_random_field(self, energy_scale=1.0):
        # Shape (3, N, N, N//2 + 1)
        u_hat = np.random.randn(3, self.N, self.N, self.N//2 + 1) + 1j * np.random.randn(3, self.N, self.N, self.N//2 + 1)
        u_hat_proj = np.zeros_like(u_hat)
        u0, u1, u2 = u_hat[0], u_hat[1], u_hat[2]
        u_hat_proj[0] = self.P11*u0 + self.P12*u1 + self.P13*u2
        u_hat_proj[1] = self.P12*u0 + self.P22*u1 + self.P23*u2
        u_hat_proj[2] = self.P13*u0 + self.P23*u1 + self.P33*u2
        
        current_energy = 0.5 * np.mean(np.sum(np.abs(np.fft.irfftn(u_hat_proj, s=(self.N,self.N,self.N), axes=(1,2,3)))**2, axis=0))
        self.u_hat = u_hat_proj * np.sqrt(energy_scale / (2 * current_energy + 1e-9))
        return self.u_hat

    def initialize_taylor_green(self, scale=1.0):
        x = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        u = scale * np.sin(X) * np.cos(Y) * np.cos(Z)
        v = -scale * np.cos(X) * np.sin(Y) * np.cos(Z)
        w = np.zeros_like(X)
        self.u_hat = np.fft.rfftn(np.stack([u, v, w]), axes=(1,2,3))
        # Project
        u0, u1, u2 = self.u_hat[0], self.u_hat[1], self.u_hat[2]
        self.u_hat[0] = self.P11*u0 + self.P12*u1 + self.P13*u2
        self.u_hat[1] = self.P12*u0 + self.P22*u1 + self.P23*u2
        self.u_hat[2] = self.P13*u0 + self.P23*u1 + self.P33*u2
        return self.u_hat

    def initialize_pelz_flow(self, scale=1.0):
        x = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        u = np.sin(X) * (np.cos(3*Y)*np.cos(Z) - np.cos(Y)*np.cos(3*Z))
        v = np.sin(Y) * (np.cos(3*Z)*np.cos(X) - np.cos(Z)*np.cos(3*X))
        w = np.sin(Z) * (np.cos(3*X)*np.cos(Y) - np.cos(X)*np.cos(3*Y))
        self.u_hat = np.fft.rfftn(np.stack([u, v, w]), axes=(1,2,3)) * scale
        return self.u_hat

    def zero_helical_mode(self, sign='+'):
        u_p, u_m = self.get_helical_decomposition()
        
        # Reconstruct basis
        xi = np.zeros_like(self.u_hat)
        xi[0] = self.ky; xi[1] = -self.kx; xi[2] = 0.0
        xi_mag = np.sqrt(np.sum(np.abs(xi)**2, axis=0)); xi_mag[xi_mag == 0] = 1.0
        xi /= xi_mag
        k_mag = np.sqrt(self.k_sq); k_mag[0,0,0] = 1.0
        k_x_xi = np.zeros_like(self.u_hat)
        k_x_xi[0] = self.ky*xi[2] - self.kz*xi[1]
        k_x_xi[1] = self.kz*xi[0] - self.kx*xi[2]
        k_x_xi[2] = self.kx*xi[1] - self.ky*xi[0]
        h_plus = (xi + 1j * k_x_xi / k_mag) / np.sqrt(2)
        h_minus = (xi - 1j * k_x_xi / k_mag) / np.sqrt(2)
        
        if sign == '+':
            self.u_hat = u_m * h_minus # Keep only minus
        else:
            self.u_hat = u_p * h_plus  # Keep only plus
        return self.u_hat

    def get_real_fields(self):
        u = np.fft.irfftn(self.u_hat, s=(self.N,self.N,self.N), axes=(1,2,3))
        w_hat = np.zeros_like(self.u_hat)
        w_hat[0] = 1j * (self.ky*self.u_hat[2] - self.kz*self.u_hat[1])
        w_hat[1] = 1j * (self.kz*self.u_hat[0] - self.kx*self.u_hat[2])
        w_hat[2] = 1j * (self.kx*self.u_hat[1] - self.ky*self.u_hat[0])
        omega = np.fft.irfftn(w_hat, s=(self.N,self.N,self.N), axes=(1,2,3))
        return u, omega

    def apply_filter(self, f_hat):
        # Exponential Spectral Filter: f(k) = exp(-alpha * (k/k_max)^s)
        # Prevents aliasing while preserving more high-frequency content than 2/3 rule
        k_mag = np.sqrt(self.k_sq)
        k_max = float(self.N) / 2.0
        alpha = 36.0 # Standard for double precision
        s = 16.0    # Order of filter
        filt = np.exp(-alpha * (k_mag / k_max)**s)
        return f_hat * filt

    def compute_rhs(self, u_hat):
        # 1. Real space velocity and vorticity
        u = np.fft.irfftn(u_hat, s=(self.N,self.N,self.N), axes=(1,2,3))
        w_hat = np.zeros_like(u_hat)
        w_hat[0] = 1j * (self.ky*u_hat[2] - self.kz*u_hat[1])
        w_hat[1] = 1j * (self.kz*u_hat[0] - self.kx*u_hat[2])
        w_hat[2] = 1j * (self.kx*u_hat[1] - self.ky*u_hat[0])
        omega = np.fft.irfftn(w_hat, s=(self.N,self.N,self.N), axes=(1,2,3))
        
        # 2. Nonlinear term: u x omega
        f = np.zeros_like(u)
        f[0] = u[1]*omega[2] - u[2]*omega[1]
        f[1] = u[2]*omega[0] - u[0]*omega[2]
        f[2] = u[0]*omega[1] - u[1]*omega[0]
        
        f_hat = np.fft.rfftn(f, axes=(1,2,3))
        # Apply Filter
        if self.use_dealias:
            f_hat = self.apply_filter(f_hat)
        
        # 3. Project nonlinear term
        rhs = np.zeros_like(u_hat)
        f0, f1, f2 = f_hat[0], f_hat[1], f_hat[2]
        rhs[0] = self.P11*f0 + self.P12*f1 + self.P13*f2
        rhs[1] = self.P12*f0 + self.P22*f1 + self.P23*f2
        rhs[2] = self.P13*f0 + self.P23*f1 + self.P33*f2
        
        # 4. Viscous term: -nu * k^2 * u_hat
        rhs -= self.nu * self.k_sq * u_hat
        
        return rhs

    def step(self):
        # Simple Euler for audit speed (RK4 better for real physics)
        self.u_hat += self.dt * self.compute_rhs(self.u_hat)
        return self.u_hat

    def compute_diagnostics(self):
        # In Fourier space, sum(|u_hat|^2) / N^6 is the mean energy
        # However, for Real-FFT, we must account for the Hermitian symmetry (half the modes)
        # For simplicity and robustness, we just use the real fields
        u, omega = self.get_real_fields()
        energy = 0.5 * np.mean(np.sum(u**2, axis=0))
        enstrophy = 0.5 * np.mean(np.sum(omega**2, axis=0))
        return energy, enstrophy

    def get_helical_decomposition(self):
        # Basis vector xi perpendicular to k
        # We use a simple construction: xi = k x e_z (or e_x if k is on z-axis)
        u_hat = self.u_hat
        xi = np.zeros_like(u_hat)
        
        # Cross product k x (0,0,1)
        xi[0] = self.ky
        xi[1] = -self.kx
        xi[2] = 0.0
        
        # Norm
        xi_mag = np.sqrt(np.sum(np.abs(xi)**2, axis=0))
        xi_mag[xi_mag == 0] = 1.0
        xi /= xi_mag
        
        # Helical basis: h_s = (xi + i*s* (k x xi)/|k|) / sqrt(2)
        k_mag = np.sqrt(self.k_sq)
        k_mag[0,0,0] = 1.0 # Avoid div
        
        k_x_xi = np.zeros_like(u_hat)
        k_x_xi[0] = self.ky*xi[2] - self.kz*xi[1]
        k_x_xi[1] = self.kz*xi[0] - self.kx*xi[2]
        k_x_xi[2] = self.kx*xi[1] - self.ky*xi[0]
        
        h_plus = (xi + 1j * k_x_xi / k_mag) / np.sqrt(2)
        h_minus = (xi - 1j * k_x_xi / k_mag) / np.sqrt(2)
        
        # Projections: u_s = dot(u_hat, conj(h_s))
        # Note: h_s are ortho-normal basis in the plane perp to k
        u_p = np.sum(u_hat * np.conj(h_plus), axis=0)
        u_m = np.sum(u_hat * np.conj(h_minus), axis=0)
        
        return u_p, u_m

    def step_rk4(self):
        # 4th-order Runge-Kutta step
        h = self.dt
        k1 = self.compute_rhs(self.u_hat)
        k2 = self.compute_rhs(self.u_hat + 0.5*h*k1)
        k3 = self.compute_rhs(self.u_hat + 0.5*h*k2)
        k4 = self.compute_rhs(self.u_hat + h*k3)
        self.u_hat += (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self.u_hat

    def get_energy_spectrum(self):
        # E(k) = 0.5 * sum_{|k_vec| in [k, k+1)} |u_hat(k_vec)|^2
        if self.u_hat is None:
            return np.zeros(1)
        k_mag = np.array(np.sqrt(self.k_sq))
        k_max = int(np.max(k_mag))
        spectrum = np.zeros(k_max + 1)
        
        # Energy density e_hat = 0.5 * |u_hat|^2 / N^6 (normalization)
        u_h = np.array(self.u_hat, dtype=complex)
        e_hat = np.array(0.5 * np.sum(np.abs(u_h)**2, axis=0) / (float(self.N)**6))
        
        for k in range(k_max + 1):
            mask = (k_mag >= k) & (k_mag < k + 1)
            # Use np.where to be extra safe for the linter
            spectrum[k] = np.sum(e_hat[np.where(mask)])
            
        return spectrum

    def get_vorticity_smoothness(self):
        u, omega = self.get_real_fields()
        mag = np.sqrt(np.sum(omega**2, axis=0)) + 1e-12
        xi = omega / mag
        # Compute gradient of xi (very rough proxy: max jump)
        dxi = np.max(np.abs(np.diff(xi, axis=1))) + np.max(np.abs(np.diff(xi, axis=2))) + np.max(np.abs(np.diff(xi, axis=3)))
        return dxi

if __name__ == "__main__":
    solver = SpectralNSSolver(N=32)
    solver.initialize_random_field()
    e, z = solver.compute_diagnostics()
    print(f"Spectral Solver Initialized (N=32): Energy={e:.4f}, Enstrophy={z:.4f}")
