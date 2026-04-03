import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_spectral_cascade():
    print(f"--- S51: SPECTRAL CASCADE AUDIT ---")
    print(f"{'Step':<10} | {'Z':<12} | {'E(k_max)':<12} | {'Slope'}")
    print("-" * 60)
    
    N = 64
    nu = 0.001
    solver = SpectralNSSolver(N=N, nu=nu)
    solver.dt = 0.002
    solver.initialize_pelz_flow(scale=3.0)
    
    for i in range(501):
        solver.step_rk4()
        
        if i % 100 == 0:
            e, z = solver.compute_diagnostics()
            spec = solver.get_energy_spectrum()
            
            # Estimate slope in the inertial range (k=5 to k=15)
            k = np.arange(len(spec))
            valid = (k >= 5) & (k <= 15) & (spec > 1e-15)
            if np.sum(valid) > 2:
                slope, _ = np.polyfit(np.log(k[valid]), np.log(spec[valid]), 1)
            else:
                slope = 0
                
            e_high = spec[-1] if len(spec) > 0 else 0
            print(f"{i:<10} | {z:<12.4f} | {e_high:<12.4e} | {slope:.2f}")
            
    print("-" * 60)
    print("INSIGHT: If Slope is near -1.67, you have a Kolmogorov cascade.")
    print("If E(k_max) is high (>1e-6), the simulation is UNRESOLVED.")

if __name__ == "__main__":
    audit_spectral_cascade()
