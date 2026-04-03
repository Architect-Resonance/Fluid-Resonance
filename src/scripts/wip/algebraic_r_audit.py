import numpy as np

def compute_algebraic_r():
    print("--- Scientific Hat: Algebraic Precision Audit ---")
    
    # Adjacency for K5 star-cluster (8 nodes total)
    def c2e(clauses):
        edges = set()
        for c in clauses:
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    edges.add((min(c[i],c[j]), max(c[i],c[j])))
        return edges

    # Topology: K5 core + 2 anchors + bridge
    cluster_clauses = [(0,1,2),(1,2,3),(2,3,4),(3,4,0),(4,0,1),(5,0,3),(6,2,4),(7,5,6)]
    edges8 = c2e(cluster_clauses)
    A8 = np.zeros((8,8))
    for i,j in edges8: A8[i][j] = 1; A8[j][i] = 1
    L8 = np.diag(A8.sum(axis=1)) - A8

    # Grounding (Hub = 0 constraint)
    D_bridge = np.zeros(8)
    D_bridge[0] = 2; D_bridge[1] = 2; D_bridge[2] = 1; D_bridge[4] = 1
    L_eff = L8 + np.diag(D_bridge)
    l8 = np.sort(np.linalg.eigvalsh(L_eff))[0]

    # Reduced topology (6 nodes)
    keep = [0, 1, 3, 5, 6, 7]
    A_red = A8[np.ix_(keep, keep)]
    D_bridge_red = D_bridge[keep]
    L_eff_red = np.diag(A_red.sum(axis=1)) - A_red + np.diag(D_bridge_red)
    l6 = np.sort(np.linalg.eigvalsh(L_eff_red))[0]

    R = l8 / l6
    
    print(f"lambda_min(8x8): {l8:.16f}")
    print(f"lambda_min(6x6): {l6:.16f}")
    print(f"Invariant R: {R:.16f}")
    
    # Reference value from CLAUDE_BRIDGE.md
    REF_R = 1.8573068741389058
    print(f"Reference R: {REF_R:.16f}")
    print(f"Consistency Check: {abs(R - REF_R) < 1e-12}")
    
    return R

if __name__ == "__main__":
    compute_algebraic_r()
