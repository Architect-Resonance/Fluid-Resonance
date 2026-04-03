import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def run_continuum_audit():
    print(f"--- S47: CONTINUUM SCALING AUDIT ---")
    print(f"{'Resolution N':<15} | {'Enstrophy Z':<15} | {'Max Stretching'}")
    print("-" * 55)
    
    resolutions = [16, 32, 64]
    for N in resolutions:
        # We keep nu high enough to prevent aliasing at low resolution
        solver = SpectralNSSolver(N=N, nu=0.01)
        solver.initialize_random_field(energy_scale=10.0)
        
        u, omega = solver.get_real_fields()
        
        # Compute stretching field omega . S . omega
        # S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        # In Fourier: d_xi u_j = i k_i u_hat_j
        grad_u_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i, ki in enumerate([solver.kx, solver.ky, solver.kz]):
            for j in range(3):
                grad_u_hat[i, j] = 1j * ki * solver.u_hat[j]
        
        grad_u = np.real(np.fft.ifftn(grad_u_hat, axes=(2,3,4)))
        
        # Stretching = omega_i * S_ij * omega_j
        # Since S is symm: Stretching = omega_i * (grad_u)_ij * omega_j
        stretching_field = np.zeros((N, N, N))
        for i in range(3):
            for j in range(3):
                stretching_field += omega[i] * grad_u[i, j] * omega[j]
        
        max_stretch = np.max(np.abs(stretching_field))
        avg_enstrophy = np.mean(np.sum(omega**2, axis=0))
        
        # C majorant proxy = Max_Stretch / Avg_Enstrophy
        # (This is more conservative than pointwise Z)
        C_proxy = max_stretch / (avg_enstrophy + 1e-9)
        
        print(f"{N:<15} | {avg_enstrophy:<15.4f} | {max_stretch:.4f} (C={C_proxy:.4f})")

    print("-" * 55)
    print("ANALYSIS: If C stays constant as N increases, point-vortex scaling was an artifact.")
    print("If C vanishes, then the fluid is structurally regular.")

if __name__ == "__main__":
    run_continuum_audit()
