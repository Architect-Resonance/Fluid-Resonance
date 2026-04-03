import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_rapid_sweep():
    print(f"--- S51: RAPID BLOWUP SWEEP (N=32) ---")
    print(f"{'Step':<10} | {'Z':<15} | {'Max S':<12} | {'Vort Smooth'}")
    print("-" * 60)
    
    N = 32
    nu = 0.0001
    solver = SpectralNSSolver(N=N, nu=nu)
    solver.dt = 0.0002
    solver.initialize_pelz_flow(scale=10.0)
    
    for i in range(1001):
        solver.step_rk4()
        
        if i % 100 == 0:
            e, z = solver.compute_diagnostics()
            spec = solver.get_energy_spectrum()
            k_cutoff = int(0.9 * len(spec))
            spec_ratio = np.sum(spec[k_cutoff-2:k_cutoff]) / (np.sum(spec) + 1e-12)
            v_smooth = solver.get_vorticity_smoothness()
            u, omega = solver.get_real_fields()
            max_s = np.max(np.abs(omega))
            
            print(f"{i:<10} | {z:<15.4f} | {max_s:<12.4f} | {spec_ratio:.6f}")
            
        if np.isnan(z) or z > 1e15:
            print(f"BLOW-UP DETECTED at step {i}")
            break
            
    print("-" * 60)
    print(f"Sweep Complete. Final Z: {z:.4f}")

if __name__ == "__main__":
    audit_rapid_sweep()
