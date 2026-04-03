import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import json

class ExtremeVorticitySolver:
    """Pseudo-spectral solver with local S/D budget diagnostics."""
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
        
        self.kmax = N // 3
        self.dealias_mask = (
            (np.abs(self.kx) <= self.kmax) &
            (np.abs(self.ky) <= self.kmax) &
            (np.abs(self.kz) <= self.kmax)
        )
        
        K = [self.kx, self.ky, self.kz]
        self.P = {}
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

    def analyze_local_budget(self, u_hat):
        """Compute local S and D in physical space."""
        N = self.N
        K = [self.kx, self.ky, self.kz]
        w_hat = self.get_vorticity_hat(u_hat)
        w = np.array([np.real(ifftn(w_hat[i])) for i in range(3)])
        w_mag = np.sqrt(np.sum(w**2, axis=0))
        
        # Local Stretching: S = 2 * omega_i * S_ij * omega_j
        du_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                du_hat[i, j] = 1j * K[j] * u_hat[i]
        
        S_tensor = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                # S_ij = 0.5 * (dui/dxj + duj/dxi)
                S_tensor[i, j] = 0.5 * (np.real(ifftn(du_hat[i, j])) + np.real(ifftn(du_hat[j, i])))
        
        S_local = np.zeros(w[0].shape)
        for i in range(3):
            for j in range(3):
                S_local += 2.0 * w[i] * S_tensor[i, j] * w[j]
                
        # Local Dissipation: D = 2 * nu * |grad w|^2
        dw_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                dw_hat[i, j] = 1j * K[j] * w_hat[i]
                
        D_local = np.zeros(w[0].shape)
        for i in range(3):
            for j in range(3):
                D_local += 2.0 * self.nu * np.real(ifftn(dw_hat[i, j]))**2
        
        return w_mag, S_local, D_local

def run_extreme_vorticity_audit():
    N = 64
    Re = 400
    T_max = 10.0 # Increased to ensure peak
    dt = 0.01
    
    solver = ExtremeVorticitySolver(N=N, Re=Re)
    u_hat = np.zeros((3, N, N, N), dtype=complex)
    u = np.zeros((3, N, N, N))
    u[0] = np.sin(solver.X) * np.cos(solver.Y) * np.cos(solver.Z)
    u[1] = -np.cos(solver.X) * np.sin(solver.Y) * np.cos(solver.Z)
    u_hat = np.array([fftn(u[i]) for i in range(3)])
    
    print(f"Auditing Extreme Vorticity (N={N}, Re={Re}) to peak growth...")
    
    t = 0.0
    max_Z = -1.0
    u_hat_peak = None
    t_peak = 0.0
    
    # Store trajectory for context
    trajectory = []
    
    while t <= T_max:
        w_hat = solver.get_vorticity_hat(u_hat)
        Z = np.sum(np.abs(w_hat)**2) / (N**6)
        
        trajectory.append({'t': t, 'Z': float(Z)})
        
        if Z > max_Z:
            max_Z = Z
            u_hat_peak = u_hat.copy()
            t_peak = t
        u_hat = solver.step_rk4(u_hat, dt)
        t += dt
        
    print(f"Peak enstrophy reached at t = {t_peak:.2f}, Z = {max_Z:.4f}")
    
    w_mag, S_local, D_local = solver.analyze_local_budget(u_hat_peak)
    
    # Percentiles
    percentiles = [0, 50, 90, 95, 98, 99]
    results = {
        'metadata': {'N': N, 'Re': Re, 't_peak': t_peak, 'Z_peak': max_Z},
        'table': [],
        'trajectory': trajectory
    }
    
    print("\n| Percentile | Mean S_local | Mean D_local | Gap Ratio (D-S)/D |")
    print("|------------|--------------|--------------|-------------------|")
    
    for p in percentiles:
        threshold = np.percentile(w_mag, p)
        mask = w_mag >= threshold
        
        S_avg = np.mean(S_local[mask])
        D_avg = np.mean(D_local[mask])
        gap = (D_avg - S_avg) / (max(D_avg, 1e-15))
        
        label = f">{p}%" if p > 0 else "Global"
        row = {
            'p': p, 'label': label, 
            'S_avg': float(S_avg), 'D_avg': float(D_avg), 
            'gap_ratio': float(gap), 'count': int(np.sum(mask))
        }
        results['table'].append(row)
        print(f"| {label:<10} | {S_avg:<12.4f} | {D_avg:<12.4f} | {gap:<17.4f} |")

    with open("extreme_vorticity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to extreme_vorticity_results.json")

if __name__ == "__main__":
    run_extreme_vorticity_audit()
