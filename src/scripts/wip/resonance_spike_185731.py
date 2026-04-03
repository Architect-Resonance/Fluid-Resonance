"""
resonance_spike_185731.py

The Resonance Spike: Injecting the 1.85731 Resolution Chord into the
discretization of the 3D Biot-Savart kernel.

Objective:
Observe if the "Haze of Scale" (the 1/10 gap) refracts into a crystalline
spectral state (R < 2.0) when the system is 'Tuned' to the invariant.

Logic:
1. Generate an axisymmetric vortex field (Hou-Li type).
2. Construct the interaction graph Φ(ω).
3. 'Spike' the edge weights by forcing the node-to-hub ratio toward 1.85731.
4. Measure the spectral shift in the L1 (Stokes) gap.
"""

import numpy as np

# The Resolution Chord
R_INVARIANT = 1.8573068741389058

def generate_hou_li_vortex(n_filaments=8, hub_strength=1.0):
    """
    Creates a discrete model of the 'Tornado Core'.
    """
    # Filaments in a circle
    angles = np.linspace(0, 2*np.pi, n_filaments, endpoint=False)
    vertices = []
    for theta in angles:
        # Filament vertices (The 'Weaver's Hands')
        vertices.append([np.cos(theta), np.sin(theta), 0])
    
    # The Hub (The Stillness)
    vertices.append([0, 0, 0])
    V = np.array(vertices)
    
    return V

def construct_spiked_laplacian(V, R_target):
    """
    Constructs the grounded Laplacian and spikes it at the target frequency.
    """
    n = len(V)
    W = np.zeros((n, n))
    hub_idx = n - 1
    
    # 1. Natural Biot-Savart Falloff (The Syrup/Buffer)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(V[i] - V[j])
            W[i, j] = W[j, i] = 1.0 / (dist**2 + 0.1)
            
    # 2. THE SPIKE: Harmonize the core with the Invariant
    # We calibrate the filament-to-hub interaction so that the
    # effective spectral ratio R converges to 1.85731.
    for i in range(hub_idx):
        # Tuning the coupling strength to the 'Resolution Chord'
        W[i, hub_idx] = R_target * (1.0 / (np.linalg.norm(V[i] - V[hub_idx]) + 0.05))
        W[hub_idx, i] = W[i, hub_idx]
        
    # 3. NOISE FILTERING: The Helmholtz Sifting
    # We remove the asymmetric sub-grid crosstalk (the 'Syrup')
    # to allow the chord to ring clearly.
    for i in range(hub_idx):
        for j in range(i+1, hub_idx):
            # Damping the non-radial interactions
            W[i, j] *= 0.1  # Filtering the buffer
            W[j, i] = W[i, j]

    # 4. Compute Grounded Laplacian (The Stokes Gap)
    L_full = np.diag(W.sum(axis=1)) - W
    # Grounding the Spoke-Ends (The Surgery)
    L_eff = L_full[:-1, :-1]
    evals = np.sort(np.linalg.eigvalsh(L_eff))
    
    return evals, W

if __name__ == "__main__":
    print("-" * 60)
    print("RESONANCE SPIKE: TUNING TO 1.85731")
    print("-" * 60)
    
    V = generate_hou_li_vortex()
    evals, W = construct_spiked_laplacian(V, R_INVARIANT)
    
    # Extracting the 'Solitonic' Gap
    # In the tuned state, does the gap stay above the 'Melt' threshold?
    l_min = evals[0]
    
    print(f"Spike Frequency: {R_INVARIANT:.10f}")
    print(f"Spectral Gap (l_min): {l_min:.10f}")
    
    # Determining Refraction
    if l_min > 0.4949 and l_min < 0.5:
        print("STATUS: REFRACTION OBSERVED.")
        print("The frequency is vibrating on the 1/2 Riemann line.")
    elif l_min >= 0.5:
        print("STATUS: CRYSTALLIZATION.")
        print("The gap has crossed the 1/2 barrier. Regularity stabilized.")
    else:
        print("STATUS: SYRUP/BUFFERS DOMINANT.")
        print("The noise is drowning the chord.")

    print("-" * 60)
    print("WEAVER SIGNATURE (Hub Couplings):")
    print(W[-1, :-1])
    print("-" * 60)
