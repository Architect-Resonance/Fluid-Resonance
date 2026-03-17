import numpy as np
import matplotlib.pyplot as plt

def magnetic_star_laplacian(phi):
    # Standard 8-node Star Cluster Grounded Laplacian
    # We introduce a magnetic phase exp(i * phi/7) on the Fano 7-cycle edges.
    
    # 7-cycle: (0, 1, 2, 3, 4, 5, 6, 0)
    # Note: These indices must match the actual connectivity.
    # In our L_eff:
    # 0 = Hub/Spoke? No, in L_eff (8x8), these are the spoke-cluster modes.
    # Connectivity from L_eff:
    # 0-1, 0-2, 0-3, 0-4, 0-5
    # 1-0, 1-2, 1-3, 1-4, 1-5
    # etc. It's k5-like.
    
    # Let's use the actual L_eff structure and add phases to a specific cycle.
    L = np.array([
        [ 7, -1, -1, -1, -1, -1, -1, -1], # Node 0
        [-1,  6, -1, -1, -1, -1, -1,  0], # Node 1
        [-1, -1,  6, -1, -1, -1, -1,  0], # Node 2
        [-1, -1, -1,  5, -1, -1,  0,  0], # Node 3
        [-1, -1, -1, -1,  6, -1, -1,  0], # Node 4
        [-1, -1, -1, -1, -1,  6, -1,  0], # Node 5
        [-1, -1, -1,  0, -1, -1,  5, -1], # Node 6
        [-1,  0,  0,  0,  0,  0, -1,  2]  # Node 7 (Anchor)
    ], dtype=complex)
    
    # Cycle of edges to 'magnetize': (0,1,2,3,4,5,6,0)
    cycle = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,0)]
    phase_per_edge = phi / 7.0
    
    for u, v in cycle:
        # L_uv = -exp(i*alpha)
        # L_vu = -exp(-i*alpha)
        L[u, v] = -np.exp(1j * phase_per_edge)
        L[v, u] = -np.exp(-1j * phase_per_edge)
        
    # Grounded Laplacian at Node 7 is already the principal submatrix?
    # No, the previous analysis showed L_eff was the grounded matrix.
    # If L 7,7 = 2 and L 6,7 = -1, node 7 is "inside" the matrix.
    # If we ground Node 7, we remove row/col 7.
    L_g = np.delete(L, 7, axis=0)
    L_g = np.delete(L_g, 7, axis=1)
    
    eigs = np.linalg.eigvalsh(L_g)
    return eigs[0]

def sensitivity_analysis():
    phis = np.linspace(0, 2*np.pi, 100)
    l_mins = [magnetic_star_laplacian(p) for p in phis]
    
    plt.figure(figsize=(10, 6))
    plt.plot(phis/np.pi, l_mins, label='Magnetic Spectral Gap')
    plt.axhline(y=0.4950, color='r', linestyle='--', label='Target 0.4950')
    plt.xlabel('Flux Phi (pi units)')
    plt.ylabel('lambda_min')
    plt.title('Sensitivity of 8-node Star Cluster Gap to Magnetic Flux')
    plt.legend()
    plt.grid(True)
    plt.savefig('h:/Project/Entropy/Fluid-Resonance/scripts/wip/magnetic_sensitivity.png')
    
    # Find flux for 0.4950
    target = 0.4950
    best_phi = 0
    min_diff = 100
    for p, l in zip(phis, l_mins):
        if abs(l - target) < min_diff:
            min_diff = abs(l - target)
            best_phi = p
            
    print(f"Target lambda_min: {target}")
    print(f"Flux producing target: {best_phi/np.pi:.4f} pi")
    print(f"Value at 1.5pi (3pi/2): {magnetic_star_laplacian(1.5*np.pi):.6f}")

if __name__ == "__main__":
    sensitivity_analysis()
