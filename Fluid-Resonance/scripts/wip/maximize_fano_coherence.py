import numpy as np
from scipy.optimize import minimize

def maximize_fano_coherence():
    # Weights: 1 point has 32 modes, 6 points have 20 modes.
    # Total = 152.
    weights = np.array([32, 20, 20, 20, 20, 20, 20])
    
    # Fano Triads (indices 0-6)
    triads = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6),
        (2, 3, 6), (2, 4, 5)
    ]
    
    # We want to maximize S = |\sum_{j=0}^6 T_j e^{i \Phi_j}|
    # Where T_j is the 'Triad Weight'. 
    # How to assign weights to triads? 
    # Let's assume each triad L inherits weights from its points.
    # Triad weight T_j = mean(weights[triads[j]])? Or product?
    # Navier-Stokes is product: a1 * a2. 
    triad_weights = []
    for t in triads:
        w_eff = np.sqrt(weights[t[0]] * weights[t[1]]) # Magnitude u1*u2
        triad_weights.append(w_eff)
    triad_weights = np.array(triad_weights)
    
    # Obstruction: sum(point_phases) + Omega_j = const? No.
    # Every point k has a phase phi_k. 
    # Every triad has total phase psi_j = phi_{k1} + phi_{k2} + phi_{k3} + Omega_j.
    # Total stretching S = sum triad_weights_j * e^{i psi_j}.
    
    # We choose phi_k to maximize |S|.
    # Total Berry holonomy sum Omega_j = Phi_target (e.g., pi).
    # Since Omega_j are constants, we can absorb most into phi_k.
    # EXCEPT for the global obstruction.
    
    def objective(phi, Omega, w):
        # phi: 7 point phases
        # S = sum w_j * exp(i * (phi[t1] + phi[t2] + phi[t3] + Omega_j))
        S = 0
        for j, t in enumerate(triads):
            psi = phi[t[0]] + phi[t[1]] + phi[t[2]] + Omega[j]
            S += w[j] * np.exp(1j * psi)
        return -np.abs(S)
    
    # Try multiple Global Flux levels
    results = []
    for Phi in [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
        # Distribute Phi among Omega_j
        Omega = np.zeros(7)
        Omega[0] = Phi # Put all obstruction in one triad to test worst case
        
        res = minimize(objective, np.zeros(7), args=(Omega, triad_weights), method='L-BFGS-B')
        q = -res.fun / np.sum(triad_weights)
        results.append((Phi, q))
        
    print(f"{'Global Flux (Phi)':<20} | {'Max Coherence q'}")
    print("-" * 45)
    for Phi, q in results:
        print(f"{Phi/np.pi:<20.4f} pi | {q:.6f}")
        
    # Conclusion for Phi = pi
    q_pi = results[3][1]
    print(f"\nAt Phi = pi (Fano Obstruction): q = {q_pi:.6f}")
    print(f"1 - q = {1 - q_pi:.6f}  (cf. alpha_iso = 0.3069)")
    print(f"Is 1-q related to sin^2(theta)? q_pi^2 = {q_pi**2:.6f}")

if __name__ == "__main__":
    maximize_fano_coherence()
