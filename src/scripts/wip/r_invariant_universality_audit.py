import numpy as np
from spectral_ns_solver import SpectralNSSolver

def audit_r_universality():
    print(f"--- S52: R-INVARIANT UNIVERSALITY AUDIT ---")
    print(f"{'Flow Type':<15} | {'N':<5} | {'R Ratio'}")
    print("-" * 40)
    
    resolutions = [32, 64]
    flow_types = ['Taylor-Green', 'Pelz', 'Random']
    
    for N in resolutions:
        solver = SpectralNSSolver(N=N)
        
        for ftype in flow_types:
            if ftype == 'Taylor-Green':
                solver.initialize_taylor_green()
            elif ftype == 'Pelz':
                solver.initialize_pelz_flow()
            else:
                solver.initialize_random_field()
                
            e, z = solver.compute_diagnostics()
            # Calculate R = C / sqrt(Z) or similar based on S47 definitions
            # In S47, we looked at C ~ sqrt(N) for discrete, but for spectral C is smooth.
            # Let's look at the ratio C / (sqrt(E)*Z) or similar scale-invariant
            
            u, omega = solver.get_real_fields()
            # C is the max stretching majorant
            # For simplicity, we use max(|omega|) as a proxy for the stretch rate C
            max_omega = np.max(np.abs(omega))
            
            # The hypothesized invariant was related to the topological constant 1.85731
            # R = max(|omega|) / (Z^0.5 * something)
            # Let's check the ratio R = max(|omega|) / sqrt(2*Z)
            # Since Z = 0.5 * mean(|omega|^2), sqrt(2*Z) is the RMS vorticity.
            # So R is essentially the "Vorticity Peak Factor"
            
            r_ratio = max_omega / (np.sqrt(2 * z) + 1e-12)
            
            print(f"{ftype:<15} | {N:<5} | {r_ratio:.6f}")

if __name__ == "__main__":
    audit_r_universality()
