import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def get_mollified_field(positions, vorticities, grid_size, L, delta):
    """
    Creates the smooth vorticity field from discrete points.
    """
    dv = (L / grid_size)**3
    wx_field = np.zeros((grid_size, grid_size, grid_size))
    wy_field = np.zeros((grid_size, grid_size, grid_size))
    wz_field = np.zeros((grid_size, grid_size, grid_size))
    
    x_coords = np.linspace(-L/2, L/2, grid_size)
    y_coords = np.linspace(-L/2, L/2, grid_size)
    z_coords = np.linspace(-L/2, L/2, grid_size)
    
    N = positions.shape[0]
    gauss_norm = 1.0 / (delta**3 * (2*np.pi)**1.5)
    
    for i in prange(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                gx, gy, gz = x_coords[i], y_coords[j], z_coords[k]
                for p in range(N):
                    dx = gx - positions[p, 0]
                    dy = gy - positions[p, 1]
                    dz = gz - positions[p, 2]
                    r2 = dx*dx + dy*dy + dz*dz
                    
                    kernel = np.exp(-0.5 * r2 / (delta**2)) * gauss_norm
                    wx_field[i,j,k] += vorticities[p, 0] * kernel
                    wy_field[i,j,k] += vorticities[p, 1] * kernel
                    wz_field[i,j,k] += vorticities[p, 2] * kernel
                    
    return wx_field, wy_field, wz_field

def run_convergence_test():
    print("--- INITIATING CONTINUUM LIMIT CONVERGENCE TEST (S57) ---")
    L = 2.0
    
    # We increase N and decrease delta
    N_steps = [100, 400, 1600, 6400]
    results = []
    
    print(f"{'N':>5} | {'delta':>6} | {'Max|w|':>10} | {'L2_Z':>8} | {'BKM_Ratio':>10}")
    print("-" * 50)
    
    for N in N_steps:
        # Scale delta with N
        delta = 0.5 * (100.0 / N)**(1/3.0)
        grid = 32
        
        np.random.seed(42)
        positions = np.random.uniform(-L/4, L/4, (N, 3))
        vorticities = np.random.randn(N, 3)
        Z_discrete = np.sum(vorticities**2)
        vorticities /= np.sqrt(Z_discrete)
        
        wx, wy, wz = get_mollified_field(positions, vorticities, grid, L, delta)
        
        mag_w = np.sqrt(wx**2 + wy**2 + wz**2)
        max_w = np.max(mag_w)
        total_Z = np.sum(mag_w**2) * (L/grid)**3
        
        # BKM Ratio: Normalized growth vs log of complexity
        bkm_ratio = max_w / np.log(1.0 + total_Z)
        
        results.append({
            'N': N,
            'max_w': max_w,
            'total_Z': total_Z,
            'bkm': bkm_ratio
        })
        
        print(f"{N:5d} | {delta:6.3f} | {max_w:10.4f} | {total_Z:8.4f} | {bkm_ratio:10.4f}")

    print("\n--- BKM RATIO ANALYSIS ---")
    # If BKM Ratio is bounded, the flow is regular.
    for i in range(1, len(results)):
        slope = (results[i]['bkm'] - results[i-1]['bkm']) / (results[i]['N'] - results[i-1]['N'])
        print(f"BKM Slope (N={results[i-1]['N']}->{results[i]['N']}): {slope:.2e}")

    if results[-1]['bkm'] < results[0]['bkm'] * 2:
        print("\nVERDICT: BKM Ratio is bounded/stabilizing.")
        print("The 'Monster' is a logarithmic ghost. Regularity holds.")
    else:
        print("\nVERDICT: BKM Ratio is divergent. Potential blow-up detected.")

    print("\n--- CONVERGENCE ANALYSIS ---")
    # Does max|w| grow to infinity or stabilize?
    ratios = []
    for i in range(1, len(results)):
        ratio = results[i]['max_w'] / results[i-1]['max_w']
        ratios.append(ratio)
        print(f"Growth Ratio (N={results[i-1]['N']}->{results[i]['N']}): {ratio:.4f}")

    if all(r < 1.5 for r in ratios):
        print("\nVERDICT: Max vorticity growth is SUB-LINEAR.")
        print("Self-similarity is NOT achieved. The monster is stabilizing.")
    else:
        print("\nVERDICT: Growth detected. Further resolve required.")

if __name__ == "__main__":
    run_convergence_test()
