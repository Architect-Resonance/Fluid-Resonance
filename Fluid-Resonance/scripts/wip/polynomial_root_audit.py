import numpy as np

def verify_polynomial_roots():
    print("--- Final Calculus Verification: Polynomial Roots ---")
    
    # P7(t) = t^7 - 33t^6 + 443t^5 - 3097t^4 + 11948t^3 - 24634t^2 + 23588t - 6916
    p7_coeffs = [1, -33, 443, -3097, 11948, -24634, 23588, -6916]
    roots7 = np.roots(p7_coeffs)
    real_roots7 = np.sort(roots7[np.isclose(roots7.imag, 0)].real)
    l_min_7 = real_roots7[0]
    
    # P5(t) = t^5 - 17t^4 + 104t^3 - 270t^2 + 260t - 52
    p5_coeffs = [1, -17, 104, -270, 260, -52]
    roots5 = np.roots(p5_coeffs)
    real_roots5 = np.sort(roots5[np.isclose(roots5.imag, 0)].real)
    l_min_5 = real_roots5[0]
    
    print(f"P7 Roots (Real): {real_roots7}")
    print(f"P5 Roots (Real): {real_roots5}")
    
    print(f"L_min (P7): {l_min_7:.15f}")
    print(f"L_min (P5): {l_min_5:.15f}")
    
    ratio = l_min_7 / l_min_5
    print(f"Ratio R = L_min(P7) / L_min(P5): {ratio:.15f}")
    print(f"Target Constant: 1.8573068741389058")
    print(f"Difference: {ratio - 1.8573068741389058:.2e}")

    print("\n--- Re-verifying the Margin of 32 Expansion ---")
    import sympy as sp
    n = sp.symbols('n')
    lhs = (n-4)*(n+2)**2
    rhs = (n-2)**2*(n+4)
    print(f"LHS Expand: {sp.expand(lhs)}")
    print(f"RHS Expand: {sp.expand(rhs)}")
    print(f"RHS - LHS: {sp.expand(rhs - lhs)}")

if __name__ == "__main__":
    verify_polynomial_roots()
