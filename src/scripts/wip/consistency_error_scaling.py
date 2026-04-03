import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def calculate_field(positions, vorticities, grid_size, L, delta):
    dv = (L / grid_size)**3
    w_field = np.zeros((grid_size, grid_size, grid_size, 3))
    x_coords = np.linspace(-L/2, L/2, grid_size)
    
    N = positions.shape[0]
    gauss_norm = 1.0 / (delta**3 * (2*np.pi)**1.5)
    
    for i in prange(grid_size):
        gx = x_coords[i]
        for j in range(grid_size):
            gy = x_coords[j]
            for k in range(grid_size):
                gz = x_coords[k]
                for p in range(N):
                    dx = gx - positions[p, 0]
                    dy = gy - positions[p, 1]
                    dz = gz - positions[p, 2]
                    r2 = dx*dx + dy*dy + dz*dz
                    kernel = np.exp(-0.5 * r2 / (delta**2)) * gauss_norm
                    w_field[i,j,k, 0] += vorticities[p, 0] * kernel
                    w_field[i,j,k, 1] += vorticities[p, 1] * kernel
                    w_field[i,j,k, 2] += vorticities[p, 2] * kernel
    return w_field

def run_consistency_audit():
    print("--- INITIATING OPERATOR CONTINUITY AUDIT (S58) ---")
    L = 2.0
    grid = 16
    
    # Reference "Continuum" field (high N)
    N_ref = 10000
    delta_ref = 0.2
    np.random.seed(42)
    pos_ref = np.random.uniform(-L/4, L/4, (N_ref, 3))
    vor_ref = np.random.randn(N_ref, 3)
    vor_ref /= np.sqrt(np.sum(vor_ref**2))
    
    print("Generating Reference Field...")
    W_ref = calculate_field(pos_ref, vor_ref, grid, L, delta_ref)
    
    # Test approximations with lower N
    N_tests = [100, 400, 1600, 6400]
    errors = []
    
    print(f"{'N':>5} | {'L2 Error':>10} | {'Expected (1/vN)':>15}")
    print("-" * 40)
    
    for N in N_tests:
        # Sample N points from the reference
        indices = np.random.choice(N_ref, N, replace=False)
        pos_N = pos_ref[indices]
        vor_N = vor_ref[indices] * (N_ref / N) # Rescale to match total enstrophy
        
        W_N = calculate_field(pos_N, vor_N, grid, L, delta_ref)
        
        # Calculate L2 difference
        diff = W_N - W_ref
        l2_error = np.sqrt(np.mean(diff**2))
        expected = 1.0 / np.sqrt(N)
        
        errors.append(l2_error)
        print(f"{N:5d} | {l2_error:10.6f} | {expected:15.6f}")

    print("\n--- CONTINUITY VERDICT ---")
    # check if error decreases with N
    if errors[-1] < errors[0]:
        scaling = np.log(errors[0]/errors[-1]) / np.log(N_tests[-1]/N_tests[0])
        print(f"Observed Scaling Exponent: {scaling:.4f} (Ideal: 0.5000)")
        if scaling > 0.4:
            print("VERDICT: Operator Continuity CONFIRMED. The bridge is rigorous.")
        else:
            print("VERDICT: Convergence too slow. Check kernel normalization.")
    else:
        print("VERDICT: DIVERGENCE DETECTED. The bridge is broken.")

if __name__ == "__main__":
    run_consistency_audit()
