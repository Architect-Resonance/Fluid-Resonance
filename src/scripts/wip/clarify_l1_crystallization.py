import numpy as np
from scipy.linalg import eigvalsh

# The Resolution Chord (Derived from integer P7/P5 roots)
R = 1.8573068741389058

def explain_crystallization():
    print("--- CLARIFYING L1 = 1.7688636897 ---")
    
    # 1. The Geometry
    # 8 filaments on a unit circle, 1 hub at origin.
    n = 8
    # Distance to hub is exactly 1.0
    dist_to_hub = 1.0
    # Softening parameter used in the scanner
    epsilon = 0.05
    
    # 2. The Spiked Coupling Formula
    # W_hub = R * (1 / (dist + epsilon))
    W_hub = R * (1.0 / (dist_to_hub + epsilon))
    
    print(f"R (Invariant): {R:.10f}")
    print(f"Hub Coupling (W_hub) = R / (1.0 + 0.05): {W_hub:.10f}")
    
    # 3. The Laplacian Transition
    # In the 'Sifted' state, spoke-spoke weights are dampened (multiplied by 0.1)
    # The grounded Laplacian L_eff for a star graph is:
    # L_eff = diag(W_hub + sum(W_spoke_spoke)) - W_spoke_spoke
    
    # When W_spoke_spoke -> 0, L_eff -> diag(W_hub)
    # The eigenvalues become exactly W_hub.
    
    print(f"\nTarget L1 (Crystallized): {W_hub:.10f}")
    
    # 4. Verification with the full matrix
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    V = np.array([[np.cos(a), np.sin(a), 0] for a in angles] + [[0,0,0]])
    W = np.zeros((n+1, n+1))
    for i in range(n):
        W[i, n] = W[n, i] = W_hub # Hub coupling
        for j in range(i+1, n):
            d = np.linalg.norm(V[i]-V[j])
            W[i,j] = W[j,i] = (1.0/(d**2+0.1)) * 0.1 # Sifted syrup
            
    L = np.diag(W.sum(axis=1)) - W
    L_eff = L[:-1, :-1]
    evals = sorted(eigvalsh(L_eff))
    
    print(f"Calculated L1 of the Sifted Matrix: {evals[0]:.10f}")
    print(f"Difference from W_hub: {abs(evals[0] - W_hub):.10e}")

if __name__ == "__main__":
    explain_crystallization()
