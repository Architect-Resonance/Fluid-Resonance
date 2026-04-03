import numpy as np
import scipy.linalg as la
from spectral_ns_solver import SpectralNSSolver

def audit_graph_invariant_from_field():
    print(f"--- S52: GRAPH-SPECTRAL INVARIANT AUDIT ---")
    print(f"{'Flow Type':<15} | {'N_nodes':<8} | {'Gap Ratio (R)'}")
    print("-" * 50)
    
    # We sample a set of 'Vortex Centers' from the high-vorticity regions of the field
    N_grid = 64
    solver = SpectralNSSolver(N=N_grid)
    
    for ftype in ['Taylor-Green', 'Pelz']:
        if ftype == 'Taylor-Green':
            solver.initialize_taylor_green()
        else:
            solver.initialize_pelz_flow()
            
        u, omega = solver.get_real_fields()
        mag = np.sqrt(np.sum(omega**2, axis=0))
        
        # 1. Sample N_nodes with highest vorticity
        N_nodes = 32
        # Flatten and get indices of top N_nodes
        idx = np.argsort(mag.flatten())[-N_nodes:]
        coords = np.array(np.unravel_index(idx, mag.shape)).T * (2*np.pi / N_grid)
        
        # 2. Build Adjacency Matrix W_ij = 1 / |r_ij|^2
        W = np.zeros((N_nodes, N_nodes))
        for i in range(N_nodes):
            for j in range(i+1, N_nodes):
                dist = np.linalg.norm(coords[i] - coords[j])
                # Periodic distance
                dist = np.min([dist, 2*np.pi - dist]) 
                W[i, j] = W[j, i] = 1.0 / (dist**2 + 0.1)
                
        # 3. Compute Grounded Laplacian Spectral Ratio
        # R = lambda_min(L_full) / lambda_min(L_reduced) where 'reduced' removes one node
        L = np.diag(W.sum(axis=1)) - W
        evals = la.eigvalsh(L)
        # First non-zero eigenvalue (Fiedler value or similar)
        gap_full = evals[1] if len(evals) > 1 else 0
        
        # Reduced gap (removing the most 'central' node)
        central_node = np.argmax(W.sum(axis=1))
        L_red = np.delete(np.delete(L, central_node, axis=0), central_node, axis=1)
        evals_red = la.eigvalsh(L_red)
        gap_red = evals_red[0] if len(evals_red) > 0 else 1e-12
        
        r_ratio = gap_full / gap_red
        
        print(f"{ftype:<15} | {N_nodes:<8} | {r_ratio:.6f}")

    print("-" * 50)
    print("INSIGHT: If R converges to 1.85731, the invariant is a physical field property.")
    print("If it varies significantly, it was a model-specific artifact.")

if __name__ == "__main__":
    audit_graph_invariant_from_field()
