import numpy as np
from scipy.linalg import eigvalsh

R = 1.8573068741389058

def scan_crystallization(n):
    # 1. Geometry (Tornado Core)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    V = np.array([[np.cos(a), np.sin(a), 0] for a in angles] + [[0,0,0]])
    hub = n
    
    # 2. Adjacency (Spiked & Filtered)
    W = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1, n+1):
            d = np.linalg.norm(V[i]-V[j])
            W[i,j] = W[j,i] = 1.0/(d**2+0.1)
            
    # Tuning the Spike
    for i in range(hub):
        d_hub = np.linalg.norm(V[i]-V[hub])
        W[i,hub] = W[hub,i] = R * (1.0/(d_hub+0.05))
        
    # Helmholtz Sifting (Filter)
    for i in range(hub):
        for j in range(i+1, hub):
            W[i,j] *= 0.1
            W[j,i] = W[i,j]
            
    # 3. Grounded Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W
    L_eff = L[:-1, :-1]
    
    evals = np.sort(np.linalg.eigvalsh(L_eff))
    return evals

if __name__ == "__main__":
    for n in [8, 16, 32]:
        spectrum = scan_crystallization(n)
        gap = spectrum[0]
        density = np.mean(spectrum[:4])
        print(f"N={n:2}: Gap={gap:.10f}, LocalDensity={density:.10f}")
