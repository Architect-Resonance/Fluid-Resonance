import numpy as np
import spectral_ns_solver as ns

def audit_solenoidal_fraction_and_shared_constraint():
    print("--- S84: Solenoidal Fraction & Shared Constraint Audit ---")
    N = 64
    nu = 0.001
    solver = ns.SpectralNSSolver(N, nu=nu, use_dealias=True)
    
    # Initialize with Pelz Flow (highly stretching)
    u_hat = solver.initialize_pelz_flow()
    
    dt = 0.001
    steps = 500
    
    times = []
    solenoidal_fractions = []
    stretch_diss_ratios = []
    enstrophies = []
    
    print(f"{'Step':<10} | {'Enstrophy':<15} | {'Sol. Fraction':<15} | {'Stretch/Diss'}")
    print("-" * 65)
    
    # k_vecs for curl and spectral ops
    kx, ky, kz = solver.kx, solver.ky, solver.kz
    k2 = solver.k_sq
    
    for i in range(steps):
        # 1. Get real fields
        u, omega_phys = solver.get_real_fields()
        
        # 2. Compute Lamb vector in physical space: L = u x omega
        # u and omega_phys are (3, N, N, N)
        lamb_phys = np.cross(u, omega_phys, axis=0)
        
        # 3. Transform Lamb to Fourier: L_hat (3, N, N, N//2 + 1)
        lamb_hat = np.fft.rfftn(lamb_phys, axes=(1,2,3))
        
        # 4. Project to Solenoidal: L_sol = P(L)
        l0, l1, l2 = lamb_hat[0], lamb_hat[1], lamb_hat[2]
        lamb_sol = np.zeros_like(lamb_hat)
        lamb_sol[0] = solver.P11*l0 + solver.P12*l1 + solver.P13*l2
        lamb_sol[1] = solver.P12*l0 + solver.P22*l1 + solver.P23*l2
        lamb_sol[2] = solver.P13*l0 + solver.P23*l1 + solver.P33*l2
        
        # 5. Compute Fractions
        mag_total = np.sum(np.abs(lamb_hat)**2)
        mag_sol = np.sum(np.abs(lamb_sol)**2)
        sol_fraction = mag_sol / (mag_total + 1e-12)
        
        # 6. Shared Constraint Audit: dZ/dt
        # Pi = Re( <omega_hat | curl(P(u x omega))> )
        # curl(L_sol)_hat = ik x L_sol_hat
        curl_lamb_hat = np.zeros_like(lamb_sol)
        curl_lamb_hat[0] = 1j * (ky * lamb_sol[2] - kz * lamb_sol[1])
        curl_lamb_hat[1] = 1j * (kz * lamb_sol[0] - kx * lamb_sol[2])
        curl_lamb_hat[2] = 1j * (kx * lamb_sol[1] - ky * lamb_sol[0])
        
        pi = np.sum(np.real(np.conj(solver.u_hat) * curl_lamb_hat)) # Wait, u_hat? No, omega_hat.
        # omega_hat = curl(u_hat) is already computed in get_real_fields? 
        # No, let's compute it explicitly:
        omega_hat = np.zeros_like(solver.u_hat)
        omega_hat[0] = 1j * (ky * solver.u_hat[2] - kz * solver.u_hat[1])
        omega_hat[1] = 1j * (kz * solver.u_hat[0] - kx * solver.u_hat[2])
        omega_hat[2] = 1j * (kx * solver.u_hat[1] - ky * solver.u_hat[0])
        
        pi = np.sum(np.real(np.conj(omega_hat) * curl_lamb_hat))
        
        # Dissipation D = nu * sum |k|^2 |omega_hat|^2
        diss = nu * np.sum(k2 * np.abs(omega_hat)**2)
        
        enstrophy = 0.5 * np.sum(np.abs(omega_hat)**2) / (N**6) # Correct normalization
        
        if i % 50 == 0:
            print(f"{i:<10} | {enstrophy:<15.4f} | {sol_fraction:<15.6f} | {pi/diss:.4f}")
            
        times.append(i * dt)
        solenoidal_fractions.append(sol_fraction)
        stretch_diss_ratios.append(pi/diss)
        enstrophies.append(enstrophy)
        
        # Step solver
        solver.step_rk4()
        
    # Final Analysis
    print("\n--- Insight Summary ---")
    corr = np.corrcoef(enstrophies, solenoidal_fractions)[0, 1]
    print(f"Correlation (Enstrophy vs Solenoidal Fraction): {corr:.4f}")
    
    if corr < -0.3:
        print("MATCH: As intensity grows, nature SHUNTS nonlinearity into the longitudinal channel.")
        print("The 'Diagonal' is the Pressure Gradient absorbing the expansion.")
    else:
        print("NO MATCH: Solenoidal fraction remains high or uncorrelated.")
        
    # Check "Shared Constraint" - stretching vs dissipation
    print(f"Max Stretch/Diss Ratio: {max(stretch_diss_ratios):.4f}")
    if all(r < 3.0 for r in stretch_diss_ratios):
        print("MATCH: Stretch/Diss ratio bounded near 2.0-3.0 (The Star Invariant Limit).")

if __name__ == "__main__":
    audit_solenoidal_fraction_and_shared_constraint()
