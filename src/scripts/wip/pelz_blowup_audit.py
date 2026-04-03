import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_pelz_blowup():
    print(f"--- S51: ADVERSARIAL BLOWUP AUDIT (Pelz-Flow) ---")
    print(f"{'Step':<10} | {'Enstrophy Z':<15} | {'Z_dot':<15} | {'Max Stretch'}")
    print("-" * 60)
    
    N = 64
    nu = 0.0005 # Even lower viscosity
    solver = SpectralNSSolver(N=N, nu=nu)
    solver.dt = 0.0005 # Smaller time step for Pelz stability
    solver.initialize_pelz_flow(scale=5.0) # High intensity
    
    prev_z = 0
    max_z = 0
    start_time = time.time()
    
    for i in range(1000):
        solver.step_rk4()
        e, z = solver.compute_diagnostics()
        z_dot = (z - prev_z) / solver.dt if i > 0 else 0
        prev_z = z
        if z > max_z: max_z = z
        
        if i % 100 == 0:
            u, omega = solver.get_real_fields()
            max_s = np.max(np.abs(omega))
            print(f"{i:<10} | {z:<15.4f} | {z_dot:<15.4f} | {max_s:.4f}")
            
        if np.isnan(z) or z > 1e8:
            print(f"SINGULARITY DETECTED at step {i}")
            break
            
    print("-" * 60)
    print(f"Audit Complete. Max Enstrophy reached: {max_z:.4f}")
    if max_z > 1000:
        print("RESULT: MASSIVE ENSTROPHY SPIKE DETECTED. Regularity in jeopardy.")
    else:
        print("RESULT: No singular blow-up seen at N=64.")

if __name__ == "__main__":
    audit_pelz_blowup()
