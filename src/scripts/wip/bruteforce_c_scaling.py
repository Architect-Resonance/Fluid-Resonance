import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def calculate_strain_and_stretching(positions, vorticities, delta):
    N = positions.shape[0]
    Z = 0.0
    for i in range(N):
        for k in range(3):
            Z += vorticities[i, k]**2
            
    # Local maxes for parallel reduction
    thread_max_S = np.zeros(N)
    thread_max_sigma = np.zeros(N)
    
    for i in prange(N):
        r_ix = positions[i, 0]
        r_iy = positions[i, 1]
        r_iz = positions[i, 2]
        
        o_ix = vorticities[i, 0]
        o_iy = vorticities[i, 1]
        o_iz = vorticities[i, 2]
        
        s11 = 0.0; s12 = 0.0; s13 = 0.0
        s22 = 0.0; s23 = 0.0; s33 = 0.0
        
        delta_sq = delta**2
        
        for j in range(N):
            if i == j: continue
            
            dx = r_ix - positions[j, 0]
            dy = r_iy - positions[j, 1]
            dz = r_iz - positions[j, 2]
            
            dist_sq = dx*dx + dy*dy + dz*dz + delta_sq
            dist = np.sqrt(dist_sq)
            # Regularized kernel: 1/(r^2 + d^2)^(5/2)
            inv_dist5 = 1.0 / (dist_sq * dist_sq * dist)
            
            vx = vorticities[j, 1]*dz - vorticities[j, 2]*dy
            vy = vorticities[j, 2]*dx - vorticities[j, 0]*dz
            vz = vorticities[j, 0]*dy - vorticities[j, 1]*dx
            
            s11 += 3.0 * vx * dx * inv_dist5
            s22 += 3.0 * vy * dy * inv_dist5
            s33 += 3.0 * vz * dz * inv_dist5
            
            s12 += 1.5 * (vx * dy + vy * dx) * inv_dist5
            s13 += 1.5 * (vx * dz + vz * dx) * inv_dist5
            s23 += 1.5 * (vy * dz + vz * dy) * inv_dist5
            
        S_norm = np.sqrt(s11**2 + s22**2 + s33**2 + 2.0*(s12**2 + s13**2 + s23**2))
        thread_max_S[i] = S_norm
            
        sigma = o_ix**2 * s11 + o_iy**2 * s22 + o_iz**2 * s33 + \
                2.0 * (o_ix*o_iy*s12 + o_ix*o_iz*s13 + o_iy*o_iz*s23)
        thread_max_sigma[i] = np.abs(sigma)
            
    return np.max(thread_max_S), np.max(thread_max_sigma), Z

@njit
def initialize_pelz_vortices(N):
    positions = np.zeros((N, 3))
    vorticities = np.zeros((N, 3))
    
    # Simple Pelz-like symmetry: 8 octants, mirrored vorticities
    # Use N/8 points per octant
    n_oct = N // 8
    if n_oct == 0: n_oct = 1
    
    idx = 0
    for octant in range(8):
        # 8 combinations of signs for (x,y,z)
        sx = 1 if (octant & 1) else -1
        sy = 1 if (octant & 2) else -1
        sz = 1 if (octant & 4) else -1
        
        for _ in range(n_oct):
            if idx >= N: break
            x = np.random.uniform(0.1, 0.9) * sx
            y = np.random.uniform(0.1, 0.9) * sy
            z = np.random.uniform(0.1, 0.9) * sz
            
            positions[idx] = np.array([x, y, z])
            # Pelz-like vorticity: sin(x)(cos(3y)cos(z) - cos(y)cos(3z))
            # We keep signs consistent with the octant mirroring
            vx = sx * np.sin(x*np.pi) * (np.cos(3*y*np.pi)*np.cos(z*np.pi) - np.cos(y*np.pi)*np.cos(3*z*np.pi))
            vy = sy * np.sin(y*np.pi) * (np.cos(3*z*np.pi)*np.cos(x*np.pi) - np.cos(z*np.pi)*np.cos(3*x*np.pi))
            vz = sz * np.sin(z*np.pi) * (np.cos(3*x*np.pi)*np.cos(y*np.pi) - np.cos(x*np.pi)*np.cos(3*y*np.pi))
            vorticities[idx] = np.array([vx, vy, vz])
            idx += 1
            
    return positions, vorticities

def run_sweep(N_values, num_trials=5, mode='pelz'):
    print(f"MODE: {mode}")
    print(f"{'N':>10} | {'Mean Sigma':>15} | {'Std Dev':>15} | {'Time/Trial (s)':>15}")
    print("-" * 65)
    
    all_results = []
    
    for N in N_values:
        sigmas = []
        start_N = time.time()
        
        delta = N**(-1/3.0)
        current_trials = num_trials if N < 100000 else 1
        
        for _ in range(current_trials):
            if mode == 'pelz':
                positions, vorticities = initialize_pelz_vortices(N)
            else:
                positions = np.random.uniform(-1, 1, (N, 3))
                vorticities = np.random.randn(N, 3)
                
            vorticities /= np.sqrt(np.sum(vorticities**2)) 
            
            max_S, max_sigma, Z = calculate_strain_and_stretching(positions, vorticities, delta)
            sigmas.append(max_sigma)
            
        end_N = time.time()
        mean_sigma = np.mean(np.array(sigmas))
        std_sigma = np.std(np.array(sigmas))
        
        print(f"{N:10d} | {mean_sigma:15.6e} | {std_sigma:15.6e} | {(end_N-start_N)/current_trials:15.4f}")
        all_results.append((N, mean_sigma, std_sigma))
        
    return all_results

if __name__ == "__main__":
    # The Big Sweep
    Ns = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000]
    run_sweep(Ns, num_trials=3, mode='pelz')
