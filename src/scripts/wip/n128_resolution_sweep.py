import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_n80_sweep():
    print(f"--- S52: N=80 RESOLUTION SWEEP ---")
    print(f"{'Step':<10} | {'Z':<15} | {'E(k_max)':<12} | {'Dissipation'}")
    print("-" * 60)
    
    N = 80
    nu = 0.001
    solver = SpectralNSSolver(N=N, nu=nu)
    solver.dt = 0.0005 # Small dt for stability
    solver.initialize_pelz_flow(scale=2.0)
    
    start_time = time.time()
    
    for i in range(51): 
        solver.step_rk4()
        
        if i % 10 == 0:
            e, z = solver.compute_diagnostics()
            spec = solver.get_energy_spectrum()
            # Dissipation = 2 * nu * Z
            diss = 2 * nu * z
            print(f"{i:<10} | {z:<15.4f} | {spec[-1]:<12.4e} | {diss:.6e}")
            
    print("-" * 60)
    print(f"Sweep Complete. Time taken: {time.time() - start_time:.2f}s")
    print("Compare these results with N=64 to check for enstrophy convergence.")

if __name__ == "__main__":
    audit_n80_sweep()
