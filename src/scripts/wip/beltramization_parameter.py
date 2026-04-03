import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import time as clock

class SpectralSolver:
    """Standard pseudo-spectral NS solver for Taylor-Green flow."""
    def __init__(self, N=64, Re=400):
        self.N = N
        self.nu = 1.0 / Re
        L = 2.0 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')
        
        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0, 0] = 1.0
        
        # Dealiasing mask (2/3 rule)
        kmax = N // 3
        self.dealias_mask = (
            (np.abs(self.kx) <= kmax) &
            (np.abs(self.ky) <= kmax) &
            (np.abs(self.kz) <= kmax)
        )
        
        # Leray projector
        self.P = {}
        K = [self.kx, self.ky, self.kz]
        for i in range(3):
            for j in range(3):
                self.P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / self.k2_safe

    def get_vorticity_hat(self, u_hat):
        return np.array([
            1j * (self.ky * u_hat[2] - self.kz * u_hat[1]),
            1j * (self.kz * u_hat[0] - self.kx * u_hat[2]),
            1j * (self.kx * u_hat[1] - self.ky * u_hat[0]),
        ])

    def compute_rhs(self, u_hat):
        w_hat = self.get_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        w = np.array([np.real(ifftn(w_hat[i])) for i in range(3)])
        
        # Lamb vector u x w
        lamb = np.array([
            u[1]*w[2] - u[2]*w[1],
            u[2]*w[0] - u[0]*w[2],
            u[0]*w[1] - u[1]*w[0]
        ])
        lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])
        for i in range(3):
            lamb_hat[i] *= self.dealias_mask
            
        rhs = np.zeros_like(u_hat)
        for i in range(3):
            for j in range(3):
                rhs[i] += self.P[(i, j)] * lamb_hat[j]
        
        rhs -= self.nu * self.k2[np.newaxis] * u_hat
        rhs[:, 0, 0, 0] = 0.0
        return rhs

    def step_rk4(self, u_hat, dt):
        k1 = self.compute_rhs(u_hat)
        k2 = self.compute_rhs(u_hat + 0.5*dt*k1)
        k3 = self.compute_rhs(u_hat + 0.5*dt*k2)
        k4 = self.compute_rhs(u_hat + dt*k3)
        return u_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def taylor_green_ic(self):
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        u_hat = np.array([fftn(u[i]) for i in range(3)])
        return u_hat

def run_beltramization_analysis():
    N = 64
    Re = 400
    T_max = 7.0
    dt = 0.01
    
    solver = SpectralSolver(N=N, Re=Re)
    u_hat = solver.taylor_green_ic()
    
    t = 0.0
    max_Z = -1.0
    u_hat_peak = None
    
    # Run to find peak enstrophy
    print(f"Running Taylor-Green (N={N}, Re={Re}) to find peak enstrophy...")
    while t <= T_max:
        w_hat = solver.get_vorticity_hat(u_hat)
        # Compute enstrophy in Fourier space (Parseval)
        Z = np.sum(np.abs(w_hat)**2) / (N**6)
        
        if Z > max_Z:
            max_Z = Z
            u_hat_peak = u_hat.copy()
            t_peak = t
            
        u_hat = solver.step_rk4(u_hat, dt)
        t += dt
        
    print(f"Peak enstrophy Z = {max_Z:.4f} found at t = {t_peak:.2f}")
    
    # Calculation at peak
    u_p = np.array([np.real(ifftn(u_hat_peak[i])) for i in range(3)])
    w_p_hat = solver.get_vorticity_hat(u_hat_peak)
    w_p = np.array([np.real(ifftn(w_p_hat[i])) for i in range(3)])
    
    u_mag = np.sqrt(np.sum(u_p**2, axis=0))
    w_mag = np.sqrt(np.sum(w_p**2, axis=0))
    
    # |w x u|
    uxw = np.array([
        w_p[1]*u_p[2] - w_p[2]*u_p[1],
        w_p[2]*u_p[0] - w_p[0]*u_p[2],
        w_p[0]*u_p[1] - w_p[1]*u_p[0]
    ])
    uxw_mag = np.sqrt(np.sum(uxw**2, axis=0))
    
    # beta = |w x u| / (|w| * |u|)
    # Handle zeros: if either w_mag or u_mag is 0, beta is undefined (mask them)
    safe_mask = (w_mag > 1e-10) & (u_mag > 1e-10)
    beta = np.zeros_like(w_mag)
    beta[safe_mask] = uxw_mag[safe_mask] / (w_mag[safe_mask] * u_mag[safe_mask])
    
    w_rms = np.sqrt(np.mean(w_mag**2))
    print(f"Vorticity RMS: {w_rms:.4f}")
    
    # Report conditioned beta
    print("\n| Range | Mean Beta | Count |")
    print("|-------|-----------|-------|")
    
    ranges = [
        (0, 1, "< 1x rms"),
        (1, 2, "1-2x rms"),
        (2, 3, "2-3x rms"),
        (3, 5, "3-5x rms"),
        (5, 7, "5-7x rms"),
        (7, 1000, "> 7x rms")
    ]
    
    for low, high, label in ranges:
        mask = (w_mag >= low * w_rms) & (w_mag < high * w_rms)
        if np.any(mask):
            mean_beta = np.mean(beta[mask])
            count = np.sum(mask)
            print(f"| {label:<10} | {mean_beta:<9.4f} | {count:<5} |")
        else:
            print(f"| {label:<10} | {'N/A':<9} | 0     |")

if __name__ == "__main__":
    run_beltramization_analysis()
