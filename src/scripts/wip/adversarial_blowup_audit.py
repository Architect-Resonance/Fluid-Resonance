import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_adversarial_blowup():
    print(f"--- S51: ADVERSARIAL BLOWUP AUDIT (Taylor-Green) ---")
    print(f"{'Step':<10} | {'Enstrophy Z':<15} | {'Z_dot':<15} | {'Max Stretch'}")
    print("-" * 60)
    
    N = 64
    nu = 0.001 # Extremely low viscosity
    solver = SpectralNSSolver(N=N, nu=nu)
    solver.dt = 0.001
    solver.initialize_taylor_green(scale=1.0)
    
    prev_z = 0
    max_z = 0
    start_time = time.time()
    
    for i in range(500):
        solver.step_rk4()
        e, z = solver.compute_diagnostics()
        z_dot = (z - prev_z) / solver.dt if i > 0 else 0
        prev_z = z
        if z > max_z: max_z = z
        
        if i % 50 == 0:
            # Compute max stretch field locally
            u, omega = solver.get_real_fields()
            # Simplified max stretch for speed
            max_s = np.max(np.abs(omega)) # Rough proxy
            print(f"{i:<10} | {z:<15.4f} | {z_dot:<15.4f} | {max_s:.4f}")
            
        # Stopping condition if things explode or NaN
        if np.isnan(z) or z > 1e6:
            print(f"SINGULARITY DETECTED at step {i}")
            break
            
    print("-" * 60)
    print(f"Audit Complete. Max Enstrophy reached: {max_z:.4f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
    
    if max_z > 10 * 0.5: # Initial Z is 0.5 for TG
        print("CRITICAL: Significant enstrophy growth detected.")
    else:
        print("INSIGHT: Viscosity or Resolution is damping the singularity.")

if __name__ == "__main__":
    audit_adversarial_blowup()
