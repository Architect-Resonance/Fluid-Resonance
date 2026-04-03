import numpy as np

def check_roots():
    # P_7(t) = t^7 - 33t^6 + 443t^5 - 3097t^4 + 11948t^3 - 24634t^2 + 23588t - 6916
    p7_coeffs = [1, -33, 443, -3097, 11948, -24634, 23588, -6916]
    roots7 = np.sort(np.roots(p7_coeffs))
    l_min_7 = roots7[0]
    
    # P_5(t) = t^5 - 17t^4 + 104t^3 - 270t^2 + 260t - 52
    p5_coeffs = [1, -17, 104, -270, 260, -52]
    roots5 = np.sort(np.roots(p5_coeffs))
    l_min_5 = roots5[0]
    
    print(f"P_7 roots: {roots7}")
    print(f"P_5 roots: {roots5}")
    print(f"\nl_min_7 = {l_min_7:.15f}")
    print(f"l_min_5 = {l_min_5:.15f}")
    
    # The invariant is often cited as l_min_full / l_min_reduced or vice-versa
    # Let's check l_min_7 / l_min_5 and l_min_5 / l_min_7
    print(f"Ratio 7/5: {l_min_7 / l_min_5:.15f}")
    print(f"Ratio 5/7: {l_min_5 / l_min_7:.15f}")

if __name__ == "__main__":
    check_roots()
