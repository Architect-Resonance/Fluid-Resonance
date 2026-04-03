import numpy as np
from scipy.linalg import eigh

def get_gap_ratio(n_spokes, hub_weight):
    """
    Computes the ratio of the first non-zero eigenvalues (Stokes gaps)
    for a star graph with variable hub interaction strength.
    """
    # 1. Geometry: Star Graph
    n = n_spokes + 1
    # All spoke-spoke weights = 1.0 (local interaction)
    # Hub-spoke weight = hub_weight (global/Biot-Savart attraction)
    adj = np.zeros((n, n))
    for i in range(n_spokes):
        adj[i, n-1] = adj[n-1, i] = hub_weight
        # Neighbor spokes are connected with unit weight
        for j in range(i+1, n_spokes):
            adj[i, j] = adj[j, i] = 1.0
            
    # 2. Laplacian
    deg = np.diag(adj.sum(axis=1))
    L = deg - adj
    
    # 3. Grounded (Stokes) Gap
    # We remove the hub (sink) to measure the internal dispersive pull
    L_grounded = L[:-1, :-1]
    evals = np.sort(eigh(L_grounded, eigvals_only=True))
    gap_full = evals[0]
    
    # 4. Reduced Gap (Simulating the removal of one Spoke-Hub anchor)
    L_red = L_grounded.copy()
    L_red[0, 0] -= hub_weight # Remove one spoke connection
    evals_red = np.sort(eigh(L_red, eigvals_only=True))
    gap_red = evals_red[0]
    
    return gap_full / gap_red

def solve_for_R(target_ratio=1.518, n_range=range(5, 12)):
    """
    Finds the hub weight R that yields the critical 'Snap' ratio.
    """
    from scipy.optimize import brentq
    
    findings = []
    for n in n_range:
        def f(R):
            return get_gap_ratio(n, R) - target_ratio
        
        try:
            R_found = brentq(f, 0.1, 10.0)
            findings.append(R_found)
        except ValueError:
            continue
            
    return np.mean(findings) if findings else 0

if __name__ == "__main__":
    print("--- DERIVING R FROM SPECTRAL INTERACTION RATIOS ---")
    # We define the goal: what hub weight R is REQUIRED for the 1.518 Snap-Back?
    R_derived = solve_for_R(1.518)
    print(f"Derived hub-weight (R) for 1.518 Snap: {R_derived:.10f}")
    
    # Test for n=7 (the core cluster number)
    ratio_at_1857 = get_gap_ratio(7, 1.85731)
    print(f"Ratio at 1.85731 (N=7): {ratio_at_1857:.10f}")
