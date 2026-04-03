import sympy as sp

def verify_resonance_invariants():
    print("--- RESONANCE VERIFICATION (SymPy) ---")
    
    t = sp.symbols('t')
    
    # 1. Resolution Chord (R)
    # P7 and P5 are the characteristic polynomials of the 8x8 and 6x6 Laplacians
    p7 = t**7 - 33*t**6 + 443*t**5 - 3097*t**4 + 11948*t**3 - 24634*t**2 + 23588*t - 6916
    p5 = t**5 - 17*t**4 + 104*t**3 - 270*t**2 + 260*t - 52
    
    # Solve for roots numerically to high precision
    roots7 = sorted([r.evalf(20) for r in sp.solve(p7) if r.is_real and r > 0])
    roots5 = sorted([r.evalf(20) for r in sp.solve(p5) if r.is_real and r > 0])
    
    l7 = roots7[0] # Smallest positive root
    l5 = roots5[0] # Smallest positive root
    
    R = l7 / l5
    print(f"P7 smallest root: {l7}")
    print(f"P5 smallest root: {l5}")
    print(f"Resolution Chord R: {R}")
    
    # 2. Hub Couplings (Interaction Matrix L)
    # The Integer Laplacian for the 8x8 'Full' structure (K5 + 2 anchors)
    L_8x8 = sp.Matrix([
        [ 7, -1, -1, -1, -1, -1,  0,  0],
        [-1,  6, -1, -1, -1,  0,  0,  0],
        [-1, -1,  6, -1, -1,  0, -1,  0],
        [-1, -1, -1,  5, -1, -1,  0,  0],
        [-1, -1, -1, -1,  6,  0, -1,  0],
        [-1,  0,  0, -1,  0,  4, -1, -1],
        [ 0,  0, -1,  0, -1, -1,  4, -1],
        [ 0,  0,  0,  0,  0, -1, -1,  2]
    ])
    
    # Check that its characteristic polynomial matches p7
    char_p7 = L_8x8.charpoly(t).as_expr()
    # Note: charpoly might be t^8 - ... but we look at the principal sub-block or grounded version
    # The P7/P5 are likely the grounded/reduced versions.
    print(f"L_8x8 charpoly (grounded): Matches P7? {sp.simplify(char_p7/t - p7) == 0}")

    # 3. Helmholtz Sifting (Leray Projection Filter)
    # Formula: P_{ij}(k) = delta_{ij} - k_i * k_j / |k|^2
    # The 'Sifting' is the energy-weighted projection:
    # E_sol = sum_k |P(k) * u_hat(k)|^2
    print("\nHelmholtz Sifting Definition:")
    print("S(u) = P[ u_target * exp(i * R * phi) ]")
    print("Where P is the Leray Projector and R is the Resonance Chord.")

if __name__ == "__main__":
    verify_resonance_invariants()
