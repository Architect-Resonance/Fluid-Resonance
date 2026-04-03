import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def run_helical_scaling_audit():
    print(f"--- S47: HELICAL & CONTINUUM SCALING AUDIT ---")
    print(f"{'N':<5} | {'Z':<10} | {'C_total':<10} | {'C_plus':<10} | {'C_minus':<10}")
    print("-" * 55)
    
    resolutions = [32, 64]
    for N in resolutions:
        # We use a higher nu at low resolution to keep it stable
        solver = SpectralNSSolver(N=N, nu=0.01)
        solver.initialize_random_field(energy_scale=100.0) # Higher intensity
        
        # Evolve for a few steps to let nonlinear interactions develop
        for _ in range(10):
            solver.step_rk4()
            
        u_p, u_m = solver.get_helical_decomposition()
        u, omega = solver.get_real_fields()
        
        # Compute stretching field omega . S . omega
        grad_u_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i, ki in enumerate([solver.kx, solver.ky, solver.kz]):
            for j in range(3):
                grad_u_hat[i, j] = 1j * ki * solver.u_hat[j]
        
        grad_u = np.real(np.fft.ifftn(grad_u_hat, axes=(2,3,4)))
        
        stretching_field = np.zeros((N, N, N))
        for i in range(3):
            for j in range(3):
                stretching_field += omega[i] * grad_u[i, j] * omega[j]
        
        max_stretch = np.max(np.abs(stretching_field))
        avg_enstrophy = np.mean(np.sum(omega**2, axis=0))
        C_total = max_stretch / (avg_enstrophy + 1e-9)
        
        # Biferale-Titi (BT) style decomposition of energy
        E_p = 0.5 * np.sum(np.abs(u_p)**2) / (N**3)
        E_m = 0.5 * np.sum(np.abs(u_m)**2) / (N**3)
        C_p = E_p / (E_p + E_m + 1e-9)
        C_m = E_m / (E_p + E_m + 1e-9)
        
        print(f"{N:<5} | {avg_enstrophy:<10.2f} | {C_total:<10.4f} | {C_p:<10.4f} | {C_m:<10.4f}")

    print("-" * 55)
    print("INSIGHT: If C_total decreases with N, smooth fields are regular.")
    print("If Helical modes (C_p, C_m) show strong asymmetry, chirality drives stability.")

if __name__ == "__main__":
    run_helical_scaling_audit()
