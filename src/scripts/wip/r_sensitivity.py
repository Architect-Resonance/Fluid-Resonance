import numpy as np
import scipy.linalg as la

def get_r_for_weight(w):
    # Base L_eff for K5+2a
    # Nodes: 0,1,2,3,4 (Core), 5,6 (Bridges), 7 (Anchor)
    
    L = np.zeros((8, 8))
    # Core K5 (all connected)
    for i in range(5):
        for j in range(5):
            if i != j:
                L[i, j] = -1
    
    # Bridges: 0 connect to 5, 2 connect to 6, 3 connect to 5, 4 connect to 6
    # (Matches the matrix from analyze_fano_graph.py)
    L[0, 5] = L[5, 0] = -1
    L[2, 6] = L[6, 2] = -1
    L[3, 5] = L[5, 3] = -1
    L[4, 6] = L[6, 4] = -1
    
    # Anchor: 7 connects to 5 and 6 with weight 'w'
    L[5, 7] = L[7, 5] = -w
    L[6, 7] = L[7, 6] = -w
    
    # Diagonals
    for i in range(8):
        L[i, i] = -np.sum(L[i, :])
    
    # Ground at Anchor (node 7): Remove row/col 7
    L_grounded = np.delete(L, 7, axis=0)
    L_grounded = np.delete(L_grounded, 7, axis=1)
    
    eigs = np.sort(np.real(la.eigvals(L_grounded)))
    l_min = eigs[0]
    
    # Reduced: remove 0 and 1 from the grounded 7x7
    L_red = np.delete(L_grounded, [0, 1], axis=0)
    L_red = np.delete(L_red, [0, 1], axis=1)
    eigs_red = np.sort(np.real(la.eigvals(L_red)))
    l_min_red = eigs_red[0]
    
    return l_min / l_min_red  # Ratio R = full / red or red / full?
    # R = 1.857 is P7_root / P5_root. 
    # l_min_7 = 0.495, l_min_5 = 0.266. 
    # Ratio 0.495 / 0.266 = 1.857.
    # So R = l_min(full) / l_min(red).


def sensitivity_test():
    ws = [1.0, 1.6, 2.0]
    print(f"{'Weight':>10} | {'R':>10}")
    print("-" * 23)
    for w in ws:
        R = get_r_for_weight(w)
        print(f"{w:10.2f} | {R:10.5f}")

if __name__ == "__main__":
    sensitivity_test()
