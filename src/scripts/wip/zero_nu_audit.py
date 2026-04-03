import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_zero_nu():
    print(f"--- S51: ZERO-VISCOSITY AUDIT (nu=0) ---")
    print(f"{'Step':<10} | {'Z':<15} | {'Max S':<12} | {'Spec Ratio'}")
    print("-" * 60)
    
    N = 64
    nu = 0.0
    solver = SpectralNSSolver(N=N, nu=nu, use_dealias=True) # Use filter to keep it "clean"
    solver.dt = 0.0005
    solver.initialize_pelz_flow(scale=5.0)
    
    for i in range(301):
        solver.step_rk4()
        
        if i % 50 == 0:
            e, z = solver.compute_diagnostics()
            spec = solver.get_energy_spectrum()
            k_cutoff = int(0.9 * len(spec))
            ratio = np.sum(spec[k_cutoff-2:k_cutoff]) / (np.sum(spec) + 1e-12)
            u, omega = solver.get_real_fields()
            max_s = np.max(np.abs(omega))
            
            print(f"{i:<10} | {z:<15.4f} | {max_s:<12.4f} | {ratio:.6f}")
            
        if np.isnan(z) or z > 1e20:
            print(f"UNRESTRICTED BLOW-UP at step {i}")
            break

    print("-" * 60)
    print("INSIGHT: If Z explodes while Ratio < 0.05, it may be a physical singularity.")
    print("If Ratio > 0.1, it is a numerical failure.")

if __name__ == "__main__":
    audit_zero_nu()
