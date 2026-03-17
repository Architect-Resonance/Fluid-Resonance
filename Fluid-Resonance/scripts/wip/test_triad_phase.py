import numpy as np

def helical_basis(k):
    k = k / np.linalg.norm(k)
    # Standard gauge: e1 = (k x z) / |k x z|
    z = np.array([0, 0, 1])
    if np.abs(k[2]) > 0.99:
        z = np.array([1, 0, 0])
    
    e1 = np.cross(k, z)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k, e1)
    e2 /= np.linalg.norm(e2)
    
    # Helical basis h+ = (e1 + i*e2)/sqrt(2)
    return (e1 + 1j * e2) / np.sqrt(2)

def calculate_triad_phase():
    # Triad k1 + k2 + k3 = 0
    k1 = np.array([1, 0, 0], dtype=float)
    k2 = np.array([-0.5, 0.866, 0], dtype=float)
    k3 = -k1 - k2
    
    h1 = helical_basis(k1)
    h2 = helical_basis(k2)
    h3 = helical_basis(k3)
    
    # Helical coupling: Gamma = (h2 x h3) . h1* (conjugate)
    # Or similarly
    gamma = np.dot(np.cross(h1, h2), np.conj(h3))
    phase = np.angle(gamma)
    
    print(f"k1: {k1}")
    print(f"k2: {k2}")
    print(f"k3: {k3}")
    print(f"Triad Coupling Phase: {phase:.4f} rad ({phase/np.pi:.4f} pi)")
    
    # Repeat for a different plane
    k1_b = np.array([0, 1, 0], dtype=float)
    k2_b = np.array([0, -0.5, 0.866], dtype=float)
    k3_b = -k1_b - k2_b
    
    gamma_b = np.dot(np.cross(helical_basis(k1_b), helical_basis(k2_b)), np.conj(helical_basis(k3_b)))
    print(f"\nTriad Coupling Phase (YZ plane): {np.angle(gamma_b):.4f} rad")

if __name__ == "__main__":
    calculate_triad_phase()
