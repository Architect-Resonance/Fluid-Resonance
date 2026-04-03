import numpy as np
import sympy as sp

def get_char_poly():
    # Matrix from analyze_fano_graph.py (grounded at node 7)
    # L_eff (8x8) provided by user/prev logs:
    L_eff = np.array([
        [ 7, -1, -1, -1, -1, -1,  0,  0],
        [-1,  6, -1, -1, -1,  0,  0,  0],
        [-1, -1,  6, -1, -1,  0, -1,  0],
        [-1, -1, -1,  5, -1, -1,  0,  0],
        [-1, -1, -1, -1,  6,  0, -1,  0],
        [-1,  0,  0, -1,  0,  4, -1, -1],
        [ 0,  0, -1,  0, -1, -1,  4, -1],
        [ 0,  0,  0,  0,  0, -1, -1,  2]
    ])
    
    # Grounded 7x7 (remove node 7)
    L_g = L_eff[:7, :7]
    
    # Use SymPy to get the characteristic polynomial
    M = sp.Matrix(L_g)
    lam = sp.symbols('lambda')
    cp = M.charpoly(lam)
    coeffs = cp.all_coeffs()
    
    print(f"Characteristic Polynomial coefficients (Grounded 7x7):")
    print(coeffs)
    
    # Target P_7 coefficients:
    # P_7(t) = t^7 - 33t^6 + 443t^5 - 3097t^4 + 11948t^3 - 24634t^2 + 23588t - 6916
    target_p7 = [1, -33, 443, -3097, 11948, -24634, 23588, -6916]
    
    print(f"\nTarget P_7 coefficients:")
    print(target_p7)
    
    match = all(sp.nsimplify(c) == tc for c, tc in zip(coeffs, target_p7))
    print(f"\nMatch: {match}")

    # Now the reduced 5x5 (remove nodes 0 and 1)
    L_red = L_g[2:, 2:]
    M_red = sp.Matrix(L_red)
    cp_red = M_red.charpoly(lam)
    coeffs_red = cp_red.all_coeffs()
    
    print(f"\nCharacteristic Polynomial coefficients (Reduced 5x5):")
    print(coeffs_red)
    
    # Target P_5 coefficients:
    # P_5(t) = t^5 - 17t^4 + 104t^3 - 270t^2 + 260t - 52
    target_p5 = [1, -17, 104, -270, 260, -52]
    
    print(f"\nTarget P_5 coefficients:")
    print(target_p5)
    
    match_red = all(sp.nsimplify(c) == tc for c, tc in zip(coeffs_red, target_p5))
    print(f"\nMatch Red: {match_red}")

if __name__ == "__main__":
    get_char_poly()
