import numpy as np
from scipy.linalg import eigvalsh

R = 1.8573068741389058

def test_epsilon_scaling(eps_range=[0.001, 0.01, 0.05, 0.1, 0.5]):
    print(f"{'Epsilon':<10} | {'Gap (L1)':<15} | {'Ratio (L1/R)':<15}")
    print("-" * 45)
    
    n = 8
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    V = np.array([[np.cos(a), np.sin(a), 0] for a in angles] + [[0,0,0]])
    hub = n
    
    for eps in eps_range:
        W = np.zeros((n+1, n+1))
        # Spiked hub
        for i in range(hub):
            W[i, hub] = W[hub, i] = R * (1.0 / (1.0 + eps))
            
        # Sifted syrup (set to 0 to see the pure invariant limit)
        # In the 'Crystallized' limit, we assume perfect Helmholtz separation
        L_eff = np.diag(W.sum(axis=1))[:-1, :-1] - W[:-1, :-1]
        evals = sorted(eigvalsh(L_eff))
        l1 = evals[0]
        
        print(f"{eps:<10} | {l1:<15.6f} | {l1/R:<15.6f}")

if __name__ == "__main__":
    test_epsilon_scaling()
