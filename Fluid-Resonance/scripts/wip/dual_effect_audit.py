import numpy as np
import networkx as nx

def compute_dual_gaps(G, hub):
    # 1. Vertex Connectivity Gap (Standard Laplacian)
    L_vertex = nx.laplacian_matrix(G).toarray()
    evals_v = np.sort(np.linalg.eigvalsh(L_vertex))
    gap_v = evals_v[1] # Fiedler value
    
    # 2. Flow Dissipation Gap (Grounded Laplacian/Stokes)
    # Removing the hub simulates grounding the flow
    nodes = list(G.nodes())
    nodes.remove(hub)
    L_full = nx.laplacian_matrix(G).toarray()
    # Mask of nodes except hub
    idx = [i for i, n in enumerate(G.nodes()) if n != hub]
    L_stokes = L_full[np.ix_(idx, idx)]
    evals_s = np.sort(np.linalg.eigvalsh(L_stokes))
    gap_s = evals_s[0]
    
    return gap_v, gap_s

def audit_dual_effect():
    print("--- Burst Audit: The Dual Effect ---")
    
    # Wheel Graph W8 (Hub + 7 Rim nodes)
    G = nx.wheel_graph(8)
    hub = 0
    
    # Baseline
    gv1, gs1 = compute_dual_gaps(G, hub)
    print(f"Baseline W8: Vertex Gap = {gv1:.3f}, Stokes Gap = {gs1:.3f}, Sum = {gv1+gs1:.3f}")
    
    # "Surgery": Weaken the hub connection
    # Instead of removing, we'll scale the hub edges down
    G_weak = G.copy()
    for neighbor in list(G_weak.neighbors(hub)):
        G_weak[hub][neighbor]['weight'] = 0.1
        
    # Re-compute with weights
    L_v_weak = nx.laplacian_matrix(G_weak, weight='weight').toarray()
    evals_v_w = np.sort(np.linalg.eigvalsh(L_v_weak))
    gv2 = evals_v_w[1]
    
    L_full_w = nx.laplacian_matrix(G_weak, weight='weight').toarray()
    idx = [i for i, n in enumerate(G.nodes()) if n != hub]
    L_s_weak = L_full_w[np.ix_(idx, idx)]
    evals_s_w = np.sort(np.linalg.eigvalsh(L_s_weak))
    gs2 = evals_s_w[0]
    
    print(f"Weakened W8: Vertex Gap = {gv2:.3f}, Stokes Gap = {gs2:.3f}, Sum = {gv2+gs2:.3f}")
    
    print("\n--- Scrutiny Verdict ---")
    if gv2 < gv1 and gs2 > gs1:
        print("FACT: Valve surgery trades Connectivity for Dissipation.")
    else:
        # Check if gap_s increases if we ground it more effectively?
        # Actually g_s is the smallest eigenvalue of the grounded part.
        # If we reduce hub weight, the diagonal of the grounded matrix decreases.
        # So g_s should DECREASE?
        print(f"Finding: Stokes Gap changed from {gs1:.3f} to {gs2:.3f}")

if __name__ == "__main__":
    audit_dual_effect()
