import numpy as np

def solve_g_detailed():
    # Equation: 512000g^2 + 770840g - 45639 = 0
    a = 512000
    b = 770840
    c = -45639
    
    # Discriminant delta = b^2 - 4ac
    delta = b**2 - 4*a*c
    sqrt_delta = np.sqrt(delta)
    
    g_plus = (-b + sqrt_delta) / (2*a)
    g_minus = (-b - sqrt_delta) / (2*a)
    
    print(f"Discriminant: {delta}")
    print(f"Sqrt(Delta):  {sqrt_delta}")
    print(f"g+: {g_plus:.15f}")
    print(f"g-: {g_minus:.15f}")
    
    # Target lambda_min was 0.4950
    # Let's see if 0.4950 is a multiple of g
    print(f"\nRatio 0.494998 / g+: {0.494998634 / g_plus}")
    
    # Now let's try to find an 8x8 Laplacian that gives this char poly
    # The coefficients of the quadratic might be related to the trace/det of a 2x2 block
    # or a specific Schur complement.

if __name__ == "__main__":
    solve_g_detailed()
