import numpy as np
from scipy.linalg import eigh

def compute_normalized_gap(nodes, sigma=0.1):
    N = len(nodes)
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    W = 1.0 / (dist + sigma)
    np.fill_diagonal(W, 0)
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
    L_norm = D_inv_sqrt @ (np.diag(d) - W) @ D_inv_sqrt
    evals = eigh(L_norm, eigvals_only=True)
    return evals[1]

def run_braiding_audit():
    print(f"--- BRAIDING AUDIT: NORMALIZED TOPOLOGICAL SEARCH ---")
    print(f"{'Structure':<15} | {'Normalized Gap':<20}")
    print("-" * 40)
    
    # 1. Borromean Rings (simplified segments)
    nodes_borro = []
    t = np.linspace(0, 2*np.pi, 100)
    # Ring 1
    for p in t: nodes_borro.append(np.array([np.cos(p), np.sin(p), 0]))
    # Ring 2
    for p in t: nodes_borro.append(np.array([0, np.cos(p), np.sin(p)]))
    # Ring 3
    for p in t: nodes_borro.append(np.array([np.sin(p), 0, np.cos(p)]))
    
    print(f"{'Borromean':<15} | {compute_normalized_gap(np.array(nodes_borro)):<20.6f}")

    # 2. Hopf Link
    nodes_hopf = []
    for p in t: nodes_hopf.append(np.array([np.cos(p), np.sin(p), 0]))
    for p in t: nodes_hopf.append(np.array([np.cos(p)+1, 0, np.sin(p)]))
    print(f"{'Hopf Link':<15} | {compute_normalized_gap(np.array(nodes_hopf)):<20.6f}")

    # 3. Simple Parallel (Control)
    nodes_para = []
    for z in np.linspace(0, 1, 300): 
        nodes_para.append(np.array([0,0,z]))
        nodes_para.append(np.array([1,0,z]))
    print(f"{'Parallel':<15} | {compute_normalized_gap(np.array(nodes_para)):<20.6f}")

if __name__ == "__main__":
    run_braiding_audit()
