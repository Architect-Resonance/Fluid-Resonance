import sympy as sp

def compute_algebraic_r():
    print("--- Scientific Hat: Algebraic Precision Audit ---")
    
    t = sp.symbols('t')
    
    # Characteristic Polynomials (from RESONANCE_STATE.json P7 and P5 approximations)
    # P5 core: K5 with 2 anchors (simplified to its characteristic polynomial)
    # The eigenvalues are the roots. we want the ratio of the smallest non-zero evals.
    
    # The actual polynomials derived for the K_n + 2 anchors are:
    # Pn = (t-1)(t - (n+2+sqrt(...) ) / 2) (t - (n+2-sqrt(...) ) / 2) ...
    # From S79 audit: lambda_min_n = ( (n+2) - sqrt(n^2 + 4n - 28) ) / 2
    
    def get_lambda_min(n):
        disc = n**2 + 4*n - 28
        return ( (n + 2) - sp.sqrt(disc) ) / 2
    
    l7 = get_lambda_min(7)
    l5 = get_lambda_min(5)
    
    R = l7 / l5
    
    print(f"n=7 Lambda_min: {l7}")
    print(f"n=5 Lambda_min: {l5}")
    print(f"Exact R: {R}")
    print(f"Numerical (50 digits): {R.evalf(50)}")
    
    # Is R algebraic? 
    # R = (9 - sqrt(49)) / 2 / ( (7 - sqrt(17)) / 2 )
    # R = (9-7)/2 / ( (7-sqrt(17))/2 )
    # R = 1 / ( (7-sqrt(17))/2 ) = 2 / (7-sqrt(17))
    
    # Rationalize R:
    R_rat = 2 * (7 + sp.sqrt(17)) / (49 - 17)
    R_rat = 2 * (7 + sp.sqrt(17)) / 32
    R_rat = (7 + sp.sqrt(17)) / 16
    
    print(f"Rationalized R: {R_rat}")
    print(f"Simplified Numerical: {R_rat.evalf(50)}")
    
    # Minimal Polynomial of R
    # x = (7 + sqrt(17)) / 16
    # 16x - 7 = sqrt(17)
    # (16x - 7)^2 = 17
    # 256x^2 - 224x + 49 = 17
    # 256x^2 - 224x + 32 = 0
    # 8x^2 - 7x + 1 = 0 (Dividing by 32)
    
    min_poly = 8*t**2 - 7*t + 1
    print(f"Minimal Polynomial of R: {min_poly}")
    
    # Compare with 1.857 hypothesis
    # (7 + sqrt(17))/16 approx (7 + 4.123)/16 = 11.123/16 = 0.695
    # Wait, the 1.857 invariant was R = lambda_8 / lambda_6? 
    # Or R = lambda_7 / lambda_5?
    # Let's check CLAUDE_BRIDGE.md
    
    return R_rat

if __name__ == "__main__":
    compute_algebraic_r()
