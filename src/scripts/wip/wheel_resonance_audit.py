import numpy as np

def wheel_gap(n_rim):
    # n_rim vertices on the ring, 1 hub
    n = n_rim
    W = np.zeros((n+1, n+1))
    for i in range(n):
        W[i, n] = W[n, i] = 1.0 # hub-rim
        j = (i + 1) % n
        W[i, j] = W[j, i] = 1.0 # rim-rim
    
    L = np.diag(W.sum(axis=1)) - W
    evals = np.linalg.eigvalsh(L)
    # Smallest non-zero eigenvalue
    return evals[1]

if __name__ == "__main__":
    g7 = wheel_gap(7) # Wheel with 7 rim vertices (b1=7? No, b1=n)
    # Wait, a wheel graph W_{n+1} has n rim edges + n hub edges.
    # Total edges = 2n. Vertices = n+1.
    # b1 = E - V + 1 = 2n - (n+1) + 1 = n.
    # So W_8 has b1=7.
    
    g5 = wheel_gap(5) # W_6 has b1=5
    
    print(f"Gap W8 (b1=7): {g7}")
    print(f"Gap W6 (b1=5): {g5}")
    print(f"Ratio R: {g7/g5}")
    
    # Try different combinations
    # Maybe Claude meant n+1 vertices total
    # P_7: 7 vertices. P_5: 5 vertices.
    print("\nTrying n vertices total:")
    for n in range(3, 10):
        print(f"W_{n} Gap: {wheel_gap(n-1)}")

