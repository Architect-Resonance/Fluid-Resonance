import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import json
import os
import matplotlib.pyplot as plt

class HelicityRatioAuditor:
    def __init__(self, N=32, nu=0.0025):
        self.N = N
        self.nu = nu
        self.L = 2 * np.pi
        
        # Grid
        x = np.linspace(0, self.L, N, endpoint=False)
        self.X, self.Y, self.Z_grid = np.meshgrid(x, x, x, indexing='ij')
        
        # Wavenumbers
        k = fftfreq(N) * N
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k_abs = np.sqrt(self.k2)
        
        # Dealiasing
        self.dealias_mask = (self.k_abs < (2/3 * N / 2))
        
        # Helical basis
        self.h_plus = np.zeros((3, N, N, N), dtype=complex)
        self.h_minus = np.zeros((3, N, N, N), dtype=complex)
        
        for i in range(N):
            for j in range(N):
                for l in range(N):
                    vk = np.array([self.kx[i,j,l], self.ky[i,j,l], self.kz[i,j,l]])
                    kmag = np.linalg.norm(vk)
                    if kmag == 0: continue
                    uk = vk / kmag
                    if abs(uk[0]) < 0.9: xi = np.cross(uk, [1, 0, 0])
                    else: xi = np.cross(uk, [0, 1, 0])
                    xi /= np.linalg.norm(xi)
                    hp = (xi + 1j * np.cross(uk, xi)) / np.sqrt(2)
                    self.h_plus[:, i, j, l] = hp
                    self.h_minus[:, i, j, l] = np.conj(hp)

        # Solenoidal projector
        self.P = {}
        for i in range(3):
            for j in range(3):
                delta = 1.0 if i == j else 0.0
                mask = (self.k2 > 0)
                proj = np.zeros((N, N, N))
                proj[mask] = delta - ([self.kx, self.ky, self.kz][i][mask] * [self.kx, self.ky, self.kz][j][mask] / self.k2[mask])
                self.P[(i, j)] = proj

    def decompose_helical(self, f_hat):
        fp_amp = np.sum(np.conj(self.h_plus) * f_hat, axis=0)
        fm_amp = np.sum(np.conj(self.h_minus) * f_hat, axis=0)
        return fp_amp[np.newaxis] * self.h_plus, fm_amp[np.newaxis] * self.h_minus

    def apply_solenoidal(self, f_hat):
        res = np.zeros_like(f_hat)
        for i in range(3):
            for j in range(3):
                res[i] += self.P[(i, j)] * f_hat[j]
        return res

    def setup_chiral_ic(self, rho):
        """Setup IC with specified helicity mixrho = ||v-||^2 / (||v+||^2 + ||v-||^2)."""
        # Random initial field
        u_hat = np.random.randn(3, self.N, self.N, self.N) + 1j * np.random.randn(3, self.N, self.N, self.N)
        u_hat *= self.dealias_mask
        u_hat = self.apply_solenoidal(u_hat) # Incompressible
        
        up, um = self.decompose_helical(u_hat)
        
        # Normalize and mix
        ep = np.sum(np.abs(up)**2)
        em = np.sum(np.abs(um)**2)
        
        u_final = np.sqrt(1 - rho) * (up / np.sqrt(ep)) + np.sqrt(rho) * (um / np.sqrt(em))
        return u_final

    def compute_rhs(self, u_hat):
        w_hat = np.zeros_like(u_hat)
        w_hat[0] = 1j * (self.ky * u_hat[2] - self.kz * u_hat[1])
        w_hat[1] = 1j * (self.kz * u_hat[0] - self.kx * u_hat[2])
        w_hat[2] = 1j * (self.kx * u_hat[1] - self.ky * u_hat[0])
        
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        w = np.array([np.real(ifftn(w_hat[i])) for i in range(3)])
        lamb = np.cross(u, w, axis=0)
        lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])
        for i in range(3): lamb_hat[i] *= self.dealias_mask
            
        rhs = self.apply_solenoidal(lamb_hat)
        rhs -= self.nu * self.k2[np.newaxis] * u_hat
        rhs[:, 0, 0, 0] = 0.0
        return rhs, lamb_hat

    def step_rk4(self, u_hat, dt):
        def f(u): return self.compute_rhs(u)[0]
        k1 = f(u_hat)
        k2 = f(u_hat + 0.5*dt*k1)
        k3 = f(u_hat + 0.5*dt*k2)
        k4 = f(u_hat + dt*k3)
        return u_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def audit_rho(self, rho, t_max=5.0, dt=0.02):
        u_hat = self.setup_chiral_ic(rho)
        t = 0.0
        data = []
        
        while t < t_max:
            up_hat, um_hat = self.decompose_helical(u_hat)
            ep = np.sum(np.abs(up_hat)**2)
            em = np.sum(np.abs(um_hat)**2)
            r = em / ep if ep > 0 else 0
            
            _, lamb_hat = self.compute_rhs(u_hat)
            sol_lamb_hat = self.apply_solenoidal(lamb_hat)
            
            # Energy norms
            e_lamb = np.sum(np.abs(lamb_hat)**2)
            e_sol = np.sum(np.abs(sol_lamb_hat)**2)
            s = np.sqrt(e_sol / e_lamb) if e_lamb > 0 else 0
            
            # Vorticity magnitude (normalized by energy)
            w_hat = np.zeros_like(u_hat)
            w_hat[0] = 1j * (self.ky * u_hat[2] - self.kz * u_hat[1])
            w_hat[1] = 1j * (self.kz * u_hat[0] - self.kx * u_hat[2])
            w_hat[2] = 1j * (self.kx * u_hat[1] - self.ky * u_hat[0])
            enstrophy = np.sum(np.abs(w_hat)**2)
            
            data.append({
                "r": float(r),
                "s": float(s),
                "e_sol": float(e_sol),
                "z": float(enstrophy)
            })
            u_hat = self.step_rk4(u_hat, dt)
            t += dt
            
        return data

def main():
    auditor = HelicityRatioAuditor(N=32, nu=1/400)
    rheos = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    all_results = {}
    
    print("Starting Helicity Ratio Audit...")
    for rho in rheos:
        print(f"Running mix rho={rho}...")
        results = auditor.audit_rho(rho)
        all_results[str(rho)] = results

    # Save data
    with open("helicity_ratio_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Simple analysis
    r_list = []
    s_list = []
    for rho_str in all_results:
        for r, s in all_results[rho_str]:
            if r > 1e-6 and s > 1e-6:
                r_list.append(r)
                s_list.append(s)
    
    log_r = np.log(r_list)
    log_s = np.log(s_list)
    slope, intercept = np.polyfit(log_r, log_s, 1)
    
    print(f"\nAudit Complete.")
    print(f"Detected Power Law: s ~ r^{slope:.4f}")
    print(f"Correlation coefficient: {np.corrcoef(log_r, log_s)[0,1]:.4f}")

if __name__ == "__main__":
    main()
