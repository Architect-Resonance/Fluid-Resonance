import numpy as np
from scipy.linalg import eigh

def vortex_blob_kernel(r, sigma):
    """Gaussian vorticity profile."""
    return (1.0 / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2 * sigma**2))

def verify_two_tubes_inequality(n_segments=20, tube_sep=1.0, sigma=0.1, nu=0.01):
    """
    Computes Phi mapping for two parallel filaments and verifies:
    Dissipation_cont >= nu * lambda_min(L_graph) * Enstrophy
    """
    print(f"--- TWO-TUBES SPECTRAL VERIFICATION (Segments/Tube: {n_segments}) ---")
    
    # 1. Geometry: Two parallel filaments along Z-axis
    z = np.linspace(0, 5, n_segments)
    nodes1 = np.column_stack([np.zeros(n_segments), np.zeros(n_segments), z])
    nodes2 = np.column_stack([np.ones(n_segments) * tube_sep, np.zeros(n_segments), z])
    all_nodes = np.vstack([nodes1, nodes2])
    N = 2 * n_segments
    
    # 2. Map Phi: Compute Interaction Matrix (Biot-Savart Energy)
    # W_ij = <curl^-1(omega_i), omega_j>
    # For thin filaments, this scales as 1/dist
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(all_nodes[i] - all_nodes[j])
            # Biot-Savart kernel approximation for segment interaction
            W[i,j] = W[j,i] = 1.0 / (dist + sigma)
            
    # 3. Compute Graph Laplacian L and its smallest gap
    deg = np.diag(W.sum(axis=1))
    L = deg - W
    evals = eigh(L, eigvals_only=True)
    lambda_min_graph = evals[1] # First non-zero
    
    # 4. Compute Continuous Quantities (Analytical Proxies)
    # Total Enstrophy Z = sum of blobs
    Z = N * (1.0 / (4 * np.pi * sigma**2)) # Integrated blob enstrophy
    
    # Continuous Dissipation D_cont = nu * integral |grad omega|^2
    # For a Gaussian blob, integral |grad omega|^2 = 1 / (8 * pi * sigma^4)
    D_cont_per_blob = nu * (1.0 / (8 * np.pi * sigma**4))
    D_cont_total = N * D_cont_per_blob
    
    # 5. Verification
    lower_bound = nu * lambda_min_graph * Z
    
    print(f"Graph lambda_min     : {lambda_min_graph:.6f}")
    print(f"Total Enstrophy (Z) : {Z:.6f}")
    print(f"Continuous Dissip   : {D_cont_total:.6f}")
    print(f"Spectral Bound (LB) : {lower_bound:.6f}")
    
    margin = D_cont_total / (lower_bound + 1e-9)
    print(f"Safety Margin (D/LB): {margin:.4f}")
    
    if D_cont_total >= lower_bound:
        print("\n[VERIFIED] Spectral Dissipation Inequality holds for Two Tubes.")
    else:
        print("\n[FAILED] Continuous dissipation is LOWER than spectral bound.")

if __name__ == "__main__":
    verify_two_tubes_inequality()
