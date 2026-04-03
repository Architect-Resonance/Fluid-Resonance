import numpy as np

# Coefficients from RESONANCE_STATE.json
p7_coeffs = [1, -33, 443, -3097, 11948, -24634, 23588, -6916]
p5_coeffs = [1, -17, 104, -270, 260, -52]

def get_roots(coeffs):
    roots = np.roots(coeffs)
    real_positive_roots = [r.real for r in roots if np.isclose(r.imag, 0) and r.real > 0]
    return sorted(real_positive_roots)

if __name__ == "__main__":
    roots7 = get_roots(p7_coeffs)
    roots5 = get_roots(p5_coeffs)
    
    l7 = roots7[0]
    l5 = roots5[0]
    
    print(f"Smallest root of P7: {l7:.15f}")
    print(f"Smallest root of P5: {l5:.15f}")
    print(f"Ratio R: {l7/l5:.15f}")
    print(f"Reference R: 1.8573068741389058")
    print(f"Difference: {abs(l7/l5 - 1.8573068741389058):.15e}")
