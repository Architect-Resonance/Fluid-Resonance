import numpy as np
from spectral_ns_solver import SpectralNSSolver

def test_optimized_solver():
    print("Testing Optimized SpectralNSSolver (RFFT)...")
    solver = SpectralNSSolver(N=32, nu=0.001)
    solver.initialize_taylor_green(scale=5.0)
    e0, z0 = solver.compute_diagnostics()
    print(f"Initial - Energy: {e0:.6f}, Enstrophy: {z0:.6f}")
    
    for i in range(21):
        solver.step_rk4()
        if i % 5 == 0:
            e, z = solver.compute_diagnostics()
            print(f"Step {i:<2} - Energy: {e:.6f}, Enstrophy: {z:.6f}")
    
    e_final, z_final = solver.compute_diagnostics()
    if z_final > z0:
        print("VERDICT: SUCCESS. Enstrophy production observed.")
    else:
        print("VERDICT: FAILURE. Enstrophy masih decay.")

if __name__ == "__main__":
    test_optimized_solver()
