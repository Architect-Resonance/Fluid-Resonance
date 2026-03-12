import sympy as sp
import numpy as np

def verify_claims():
    print("--- Verification: The Rigor of the Script ---")
    n = sp.symbols('n')
    
    # 1. & 2. R < 2 for pure Kn cores and the Safety Margin 32
    print("\n[Claims 1 & 2: R < 2 and the Margin of 32]")
    # lambda_min_eff = (n+2 - sqrt(n^2+4n-28))/2
    # lambda_min_red = (n - sqrt(n^2-16))/2
    # R < 2 <=> lambda_min_eff < 2 * lambda_min_red
    # As per proof_R_less_than_2.py line 164-175:
    # (n-4)*(n+2)^2 < (n-2)^2*(n+4)
    lhs = (n-4)*(n+2)**2
    rhs = (n-2)**2*(n+4)
    
    lhs_expanded = sp.expand(lhs)
    rhs_expanded = sp.expand(rhs)
    diff = rhs_expanded - lhs_expanded
    
    print(f"LHS Expanded: {lhs_expanded}")
    print(f"RHS Expanded: {rhs_expanded}")
    print(f"Difference (RHS - LHS): {diff}")
    print(f"Inequality: {lhs_expanded} < {rhs_expanded} => -16 < 16 (if we subtract terms)")
    
    # 5. Discriminant at n=7
    print("\n[Claim 5: n^2 + 4n - 28 at n=7]")
    disc = n**2 + 4*n - 28
    valat7 = disc.subs(n, 7)
    print(f"Discriminant at n=7: {valat7} (7^2 = {valat7})")

    # 3. & 4. L0 and Stokes ratios (from SPECTRAL_INVARIANT_RESULTS.md)
    print("\n[Claims 3 & 4: L0/Stokes Ratios and 1.68x increase]")
    # Numbers from SPECTRAL_INVARIANT_RESULTS.md lines 171-173
    l0_full = 0.6724
    l0_red = 0.2571
    l0_ratio = l0_full / l0_red
    
    stokes_full = 0.9047
    stokes_red = 1.5188
    stokes_ratio = stokes_full / stokes_red
    
    # Energy decay time reduction (Line 198)
    # Energy decay time = 1/lambda_min
    decay_full = 1/stokes_full
    decay_red = 1/stokes_red
    decay_increase = decay_full / decay_red
    
    print(f"L0 Ratio: {l0_ratio:.4f} (Target: 2.615)")
    print(f"Stokes Ratio: {stokes_ratio:.4f} (Target: 0.596)")
    print(f"Stokes Gap Inverse (Time) Increase: {decay_increase:.4f}x (Target: 1.68x)")

if __name__ == "__main__":
    verify_claims()
