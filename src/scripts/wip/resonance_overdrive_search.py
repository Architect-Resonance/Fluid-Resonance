import numpy as np
from scipy.linalg import eigvalsh

def execute_overdrive_sweep():
    print("--- RESONANCE OVER-DRIVE: THRESHOLD SWEEP ---")
    n = 8
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    V = np.array([[np.cos(a), np.sin(a), 0] for a in angles] + [[0,0,0]])
    hub = n
    
    thresholds = np.linspace(1.85731, 4.0, 20)
    results = []
    
    for R in thresholds:
        W = np.zeros((n+1, n+1))
        # Base connectivity
        for i in range(n+1):
            for j in range(i+1, n+1):
                d = np.linalg.norm(V[i]-V[j])
                W[i,j] = W[j,i] = 1.0/(d**2+0.1)
        
        # Over-Drive Spike
        for i in range(hub):
            d_hub = np.linalg.norm(V[i]-V[hub])
            W[i,hub] = W[hub,i] = R * (1.0/(d_hub+0.05))
            
        # Filter
        for i in range(hub):
            for j in range(i+1, hub):
                W[i,j] *= 0.1
                W[j,i] = W[i,j]
                
        L_eff = (np.diag(W.sum(axis=1)) - W)[:-1, :-1]
        gap = np.sort(eigvalsh(L_eff))[0]
        results.append(gap)
        print(f"R={R:.4f} | Gap={gap:.10f}")

    # Look for discontinuities (The Snap)
    diffs = np.diff(results)
    snap_idx = np.argmax(np.abs(diffs))
    print(f"\nPotential Snap-Back at R ~ {thresholds[snap_idx]:.4f}")

if __name__ == "__main__":
    execute_overdrive_sweep()
