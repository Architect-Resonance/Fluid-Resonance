import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_pelz_verbose():
    print(f"--- S51: HIGH-VERBOSITY PELZ AUDIT ---")
    print(f"{'Step':<10} | {'Z':<12} | {'Max S':<12} | {'Spec Ratio'}")
    print("-" * 60)
    
    N = 64
    nu = 0.001
    solver = SpectralNSSolver(N=N, nu=nu)
    solver.dt = 0.001
    solver.initialize_pelz_flow(scale=2.0)
    
    prev_z = 0
    start_time = time.time()
    
    for i in range(200):
        solver.step_rk4()
        e, z = solver.compute_diagnostics()
        
        if i % 20 == 0:
            u, omega = solver.get_real_fields()
            max_s = np.max(np.abs(omega))
            
            # Check for Spectral Pile-up
            spec = solver.get_energy_spectrum()
            # Ratio of energy in high-k (last 10%) vs total
            k_cutoff = int(0.9 * len(spec))
            high_k_energy = np.sum(spec[k_cutoff-2:k_cutoff]) # Near 2/3 limit
            total_energy = np.sum(spec) + 1e-12
            spec_ratio = high_k_energy / total_energy
            
            print(f"{i:<10} | {z:<12.4f} | {max_s:<12.4f} | {spec_ratio:.6f}")
            
    print("-" * 60)
    print(f"Final Spec Ratio: {spec_ratio:.6f}")
    if spec_ratio > 0.01:
        print("ALERT: SPECTRAL PILE-UP DETECTED. Simulation is UNRESOLVED.")
    else:
        print("INSIGHT: Simulation appears resolved at N=64.")

if __name__ == "__main__":
    audit_pelz_verbose()
