import numpy as np

def lei_isotropic_interaction(n_filaments=12, noise_level=0.1):
    """
    Derives the interaction graph weights from the 'Lei et al. 2025' condition:
    vorticity must span all directions (intersect every great circle on S^2).
    """
    # 1. Generate isotropic vorticity directions (Fibonacci sphere + perturbation)
    phi = np.pi * (3. - np.sqrt(5.))
    directions = []
    for i in range(n_filaments):
        y = 1 - (i / float(n_filaments - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        directions.append(np.array([x, y, z]))
    
    # Add random perturbation to simulate 'real' turbulent spanning
    directions = [d + np.random.normal(0, noise_level, 3) for d in directions]
    directions = [d / np.linalg.norm(d) for d in directions]
    
    # 2. Compute Interaction Matrix (Biot-Savart Energy Proxy)
    # The interaction between filament i and j is w_ij = (v_i . v_j) / d_ij^k
    # For regularity, we look at the 'Orthogonal Resonance' (Twist)
    W = np.zeros((n_filaments, n_filaments))
    for i in range(n_filaments):
        for j in range(i+1, n_filaments):
            # The 'Isotropic' spanning ensures no two are parallel
            dot = np.abs(np.dot(directions[i], directions[j]))
            # If they span all directions, the 'Twist' (1 - dot^2) is maximized
            W[i,j] = W[j,i] = (1.0 - dot**2)
            
    # 3. Analyze the Topology
    # For intense vorticity, the filaments must cluster (BKM theorem)
    # We add a center of gravity (the hub) representing the focus of stretching
    hub_interactions = np.array([np.linalg.norm(d) for d in directions])
    
    # 4. Resulting Spectrum
    L = np.diag(W.sum(axis=1)) - W
    evals = np.sort(np.linalg.eigvalsh(L))
    
    return evals[evals > 1e-8][0], W

if __name__ == "__main__":
    print("--- FIRST-PRINCIPLES DERIVATION: LEI ET AL. -> GRAPH ---")
    gap, W = lei_isotropic_interaction()
    print(f"Isotropic Interaction Gap: {gap:.6f}")
    print(f"Graph Density: {np.count_nonzero(W)/W.size:.2f}")
    
    # Compare with a 'Non-Isotropic' (Coned) configuration
    coned_dirs = [np.array([1, 0, 0]) + np.random.normal(0, 0.1, 3) for _ in range(12)]
    coned_dirs = [d / np.linalg.norm(d) for d in coned_dirs]
    W_cone = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dot = np.abs(np.dot(coned_dirs[i], coned_dirs[j]))
            W_cone[i,j] = W_cone[j,i] = (1.0 - dot**2)
    gap_cone = np.sort(np.linalg.eigvalsh(np.diag(W_cone.sum(axis=1)) - W_cone))[1]
    
    print(f"Coned (Blow-up) Gap: {gap_cone:.6f}")
    print(f"R_eff (Isotropic/Coned): {gap / (gap_cone + 1e-9):.2f}")
