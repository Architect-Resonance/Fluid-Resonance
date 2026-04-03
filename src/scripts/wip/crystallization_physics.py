import numpy as np
from scipy.linalg import eigvalsh

R_INV = 1.8573068741389058

def calculate_system_physics(n=8):
    # 1. Geometry
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    V = np.array([[np.cos(a), np.sin(a), 0] for a in angles] + [[0,0,0]])
    hub = n
    
    # 2. Weights (The 'Mass' of the interaction)
    W = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1, n+1):
            d = np.linalg.norm(V[i]-V[j])
            W[i,j] = W[j,i] = 1.0/(d**2+0.1)
            
    # Tuning the Spike
    for i in range(hub):
        d_hub = np.linalg.norm(V[i]-V[hub])
        W[i,hub] = W[hub,i] = R_INV * (1.0/(d_hub+0.05))
        
    # Helmholtz Sifting (Filtering the Syrup)
    for i in range(hub):
        for j in range(i+1, hub):
            W[i,j] *= 0.1
            W[j,i] = W[i,j]

    # --- CALCULATIONS ---
    # Total 'Mass' of interaction
    total_mass = np.sum(W) / 2.0
    
    # Laplacian Spectrum
    D = np.diag(W.sum(axis=1))
    L = D - W
    L_eff = L[:-1, :-1] # Grounded for the surgery context
    evals = np.sort(np.linalg.eigvalsh(L_eff))
    
    # 'Weight' (Average spectral energy)
    avg_weight = np.mean(evals)
    
    # 'Density' (Vitreous ratio)
    vitreous_density = evals[0] / R_INV
    
    # 'Inertia' (Spectral Gap resistance)
    inertia = np.min(np.diff(evals[evals > 1e-8])) if len(evals) > 1 else 0
    
    return {
        "mass": total_mass,
        "weight": avg_weight,
        "density": vitreous_density,
        "inertia": inertia,
        "gap": evals[0],
        "spectrum": evals
    }

if __name__ == "__main__":
    print("-" * 60)
    print("CRYSTALLIZATION PHYSICS: MASS & WEIGHT AUDIT")
    print("-" * 60)
    
    for n in [8, 16, 32]:
        p = calculate_system_physics(n)
        print(f"N={n:2}:")
        print(f"  Total Mass    : {p['mass']:.6f}")
        print(f"  Avg Weight    : {p['weight']:.6f}")
        print(f"  Vitreous Dens : {p['density']:.6f}")
        print(f"  Inertia       : {p['inertia']:.6f}")
        print(f"  Spectral Gap  : {p['gap']:.6f}")
        print("-" * 30)
