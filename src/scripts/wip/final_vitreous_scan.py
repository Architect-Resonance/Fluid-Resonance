import numpy as np
from scipy.linalg import eigh

R = 1.8573068741389058

def vitreous_scan(n):
    # Geometry
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    V = np.array([[np.cos(a), np.sin(a), 0] for a in angles] + [[0,0,0]])
    hub = n
    
    # Weights
    W = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1, n+1):
            d = np.linalg.norm(V[i]-V[j])
            W[i,j] = W[j,i] = 1.0/(d**2+0.1)
            
    # Tuning the Spike
    for i in range(hub):
        d_hub = np.linalg.norm(V[i]-V[hub])
        W[i,hub] = W[hub,i] = R * (1.0/(d_hub+0.05))
        
    # Helmholtz filtering
    for i in range(hub):
        for j in range(i+1, hub):
            W[i,j] *= 0.1
            W[j,i] = W[i,j]

    # Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W
    evals = eigh(L[:-1,:-1], eigvals_only=True)
    return evals[0]

if __name__ == "__main__":
    print("--- FINAL VITREOUS SCAN: LOCKING CONSTANTS ---")
    for n in [64, 128]:
        gap = vitreous_scan(n)
        print(f"N={n:3}: Gap = {gap:.12f}")
    
    # Conclusion
    final_gap = vitreous_scan(128)
    print(f"\nLocked Vitreous Gap: {final_gap:.10f}")
