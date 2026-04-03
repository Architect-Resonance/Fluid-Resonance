import numpy as np
import time
from spectral_ns_solver import SpectralNSSolver

def audit_extreme_blowup():
    print(f"--- S51: EXTREME ADVERSARIAL BLOWUP AUDIT ---")
    print(f"{'Step':<10} | {'Z':<15} | {'E(k_max)':<12} | {'Spec Ratio'}")
    print("-" * 60)
    
    N = 64
    nu = 0.0001 # Extreme low viscosity
    solver = SpectralNSSolver(N=N, nu=nu)
    solver.dt = 0.0001 # Extremely small dt for high-speed stability
    solver.initialize_pelz_flow(scale=10.0)
    
    start_time = time.time()
    
    for i in range(1001):
        solver.step_rk4()
        
        if i % 100 == 0:
            e, z = solver.compute_diagnostics()
            spec = solver.get_energy_spectrum()
            k_cutoff = int(0.9 * len(spec))
            high_k_energy = np.sum(spec[k_cutoff-2:k_cutoff])
            total_energy = np.sum(spec) + 1e-12
            spec_ratio = high_k_energy / total_energy
            
            print(f"{i:<10} | {z:<15.4f} | {spec[-1]:<12.4e} | {spec_ratio:.6f}")
            
        if np.isnan(z) or z > 1e12:
            print(f"BLOW-UP DETECTED at step {i}")
            break
            
    print("-" * 60)
    print(f"Audit Complete. Final Enstrophy: {z:.4f}")
    if spec_ratio > 0.1:
        print("VERDICT: Singularity is an UNRESOLVED ARTIFACT (Spectral Pile-up).")
    elif z > 10 * 103: # Initial Z for scale 10 is large
        print("VERDICT: POTENTIAL REAL SINGULARITY DETECTED (Resolved Growth).")
    else:
        print("VERDICT: Flow remains regular (Bounded Enstrophy).")

if __name__ == "__main__":
    audit_extreme_blowup()
