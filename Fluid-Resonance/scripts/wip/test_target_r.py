import numpy as np
import scipy.linalg as la

def get_ratio(L):
    # L is the 8x8 matrix (7 spoke/bridge nodes + 1 anchor)
    # Ground at the anchor (last node)
    L_g = np.delete(L, -1, axis=0)
    L_g = np.delete(L_g, -1, axis=1)
    
    eigs_full = np.sort(np.real(la.eigvals(L_g)))
    l_min_full = eigs_full[0]
    
    # Reduced (remove 2 valves)
    L_red = np.delete(L_g, [0, 1], axis=0)
    L_red = np.delete(L_red, [0, 1], axis=1)
    
    eigs_red = np.sort(np.real(la.eigvals(L_red)))
    l_min_red = eigs_red[0]
    
    return l_min_full / l_min_red

def find_target_r():
    target_r = 1.857306874138865
    
    # Construction: 5 core nodes + 2 bridge nodes + 1 anchor node
    # Let's try to match the degree sequence [7, 6, 6, 5, 6, 4, 4, 2]
    # (Note: Degree 7 on 8 nodes means it's connected to ALL other nodes)
    
    L = np.zeros((8, 8))
    
    # 0 is the Hub/Core connected to all
    for i in range(1, 8):
        L[0, i] = L[i, 0] = -1
        
    # K5 core (nodes 0,1,2,3,4) is fully connected
    for i in range(5):
        for j in range(i+1, 5):
            L[i, j] = L[j, i] = -1
            
    # Bridges (5, 6) and Anchor (7)
    # Let's try a specific connectivity to match the degrees
    # Node 0: deg 7 (connected to 1,2,3,4, 5,6,7)
    # Node 1: deg 6 (connected to 0,2,3,4, 5,6)
    # Node 2: deg 6 (connected to 0,1,3,4, 5,6)
    # Node 3: deg 5 (connected to 0,1,2,4, 5)
    # Node 4: deg 6 (connected to 0,1,2,3, 5,6)
    # Node 5: deg 4 (connected to 0,1,2,3?) - wait, sum must be 20.
    
    # Let's just try the matrix the user provided directly
    L_user = np.array([
        [ 7, -1, -1, -1, -1, -1, -1, -1], # All neighbors
        [-1,  6, -1, -1, -1, -1, -1,  0], # All but 7
        [-1, -1,  6, -1, -1, -1, -1,  0], # All but 7
        [-1, -1, -1,  5, -1, -1,  0,  0], # Core + 0, 5
        [-1, -1, -1, -1,  6, -1, -1,  0], # All but 7
        [-1, -1, -1, -1, -1,  6, -1,  0], # Wait, this is different
        [-1, -1, -1,  0, -1, -1,  5, -1],
        [-1,  0,  0,  0,  0,  0, -1,  2] 
    ])
    
    # Correction: find_ratio uses grounding at node 7.
    r = get_ratio(L_user)
    print(f"User matrix ratio: {r:.15f}")
    print(f"Target ratio:      {target_r:.15f}")
    print(f"Match: {abs(r - target_r) < 1e-10}")

if __name__ == "__main__":
    find_target_r()
