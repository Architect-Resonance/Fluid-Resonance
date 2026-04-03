import sympy as sp

def compare_alphas():
    x = sp.symbols('x')  # cos(theta)
    
    # 1. My Alpha (Isotropic Cross-helical)
    alpha_ag = (1 - x) / (3 - x)
    
    # 2. Meridian's Alpha (Energy-triad geometric suppression)
    # sin^2(theta)/4 = (1 - cos^2(theta))/4 = (1 - x^2)/4
    alpha_m = (1 - x**2) / sp.Integer(4)
    
    diff = sp.simplify(alpha_ag - alpha_m)
    
    print(f"Alpha_AG: {alpha_ag}")
    print(f"Alpha_M:  {alpha_m}")
    print(f"Difference (AG - M): {diff}")
    
    # Values at theta = pi/2 (x = 0)
    print(f"\nValue at theta = pi/2 (x=0):")
    print(f"  Alpha_AG: {alpha_ag.subs(x, 0)}")
    print(f"  Alpha_M:  {alpha_m.subs(x, 0)}")
    
    # Values at theta = pi (x = -1)
    print(f"\nValue at theta = pi (x=-1):")
    print(f"  Alpha_AG: {alpha_ag.subs(x, -1)}")
    print(f"  Alpha_M:  {alpha_m.subs(x, -1)}")

    # Integral comparison
    int_ag = sp.Rational(1, 2) * sp.integrate(alpha_ag, (x, -1, 1))
    int_m = sp.Rational(1, 2) * sp.integrate(alpha_m, (x, -1, 1))
    
    print(f"\nIsotropic Average (1/2 * integral_{-1}^1 ...):")
    print(f"  Int_AG: {int_ag} (~{float(int_ag.evalf()):.4f})")
    print(f"  Int_M:  {int_m} (~{float(int_m.evalf()):.4f})")

if __name__ == "__main__":
    compare_alphas()
