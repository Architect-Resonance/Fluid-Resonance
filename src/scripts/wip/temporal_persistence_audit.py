import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import time as clock

class SpectralNS:
    def __init__(self, N=64, Re=400):
        self.N = N
        self.nu = 1.0 / Re
        L = 2.0 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z_grid = np.meshgrid(x, x, x, indexing='ij')
        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.kmag = np.sqrt(self.k2)
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0, 0] = 1.0
        K = [self.kx, self.ky, self.kz]
        self.P = {}
        for i in range(3):
            for j in range(3):
                self.P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / self.k2_safe
        kmax = N // 3
        self.dealias_mask = (np.abs(self.kx) <= kmax) & (np.abs(self.ky) <= kmax) & (np.abs(self.kz) <= kmax)

    def project_leray(self, f_hat):
        result = np.zeros_like(f_hat)
        for i in range(3):
            for j in range(3):
                result[i] += self.P[(i, j)] * f_hat[j]
        return result

    def compute_vorticity_hat(self, u_hat):
        return np.array([
            1j * (self.ky * u_hat[2] - self.kz * u_hat[1]),
            1j * (self.kz * u_hat[0] - self.kx * u_hat[2]),
            1j * (self.kx * u_hat[1] - self.ky * u_hat[0]),
        ])

    def compute_rhs(self, u_hat):
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        lamb = np.array([u[1]*omega[2]-u[2]*omega[1], u[2]*omega[0]-u[0]*omega[2], u[0]*omega[1]-u[1]*omega[0]])
        lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])
        for i in range(3): lamb_hat[i] *= self.dealias_mask
        rhs = self.project_leray(lamb_hat)
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
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z_grid)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z_grid)
        u_hat = np.array([fftn(u[i]) for i in range(3)])
        return self.project_leray(u_hat)

def compute_shell_enstrophy(solver, u_hat):
    n_shells = int(np.log2(solver.N)) + 1
    omega_hat = solver.compute_vorticity_hat(u_hat)
    Z_shells = np.zeros(n_shells)
    for j in range(n_shells):
        if j == 0: mask = solver.kmag < 1.0
        else:
            k_lo, k_hi = 2.0**(j-1), 2.0**j
            mask = (solver.kmag >= k_lo) & (solver.kmag < k_hi)
        for i in range(3): Z_shells[j] += np.sum(np.abs(omega_hat[i][mask])**2) / solver.N**3
    return Z_shells

def run_temporal_audit():
    print("Starting Temporal Persistence Audit (Re=800)")
    solver = SpectralNS(N=64, Re=800)
    u_hat = solver.taylor_green_ic()
    t = 0.0
    dt = 0.005
    ratios = []
    while t < 10.0:
        u_hat = solver.step_rk4(u_hat, dt)
        t += dt
        if int(t/dt) % 2 == 0:
            Z_s = compute_shell_enstrophy(solver, u_hat)
            pk = np.argmax(Z_s)
            if pk > 0 and Z_s[pk-1] > 1e-4:
                ratio = Z_s[pk] / Z_s[pk-1]
                ratios.append(ratio)
    
    ratios = np.array(ratios)
    bins = np.linspace(1.5, 2.5, 101)
    hist, _ = np.histogram(ratios, bins=bins)
    
    # Target value
    R_TARGET = 1.85731
    itarget = np.argmin(np.abs(bins - R_TARGET))
    
    print("-" * 40)
    print(f"{'Bin Center':<12} | {'Count'}")
    print("-" * 40)
    # Output top 5 peaks
    peak_indices = np.argsort(hist)[-5:][::-1]
    for idx in peak_indices:
        print(f"{0.5*(bins[idx]+bins[idx+1]):<12.5f} | {hist[idx]}")
    
    print("-" * 40)
    print(f"Count at Target (1.857): {hist[itarget]}")
    print(f"Max Count: {np.max(hist)}")
    print(f"Mean Ratio: {np.mean(ratios):.5f}")

if __name__ == '__main__':
    run_temporal_audit()
