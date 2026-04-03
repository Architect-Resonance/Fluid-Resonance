import sympy as sp

def derive_alpha_iso():
    x = sp.symbols('x')  # represents cos(theta)
    rho = sp.symbols('rho', real=True, positive=True)
    
    # The cross-helical Leray suppression formula for rho=1
    # alpha(x, 1) = 1 - (1+1)^2 * (1+x) / [(1+1^2+2*x)*(3-x)]
    # alpha(x, 1) = 1 - 4*(1+x) / [(2+2x)*(3-x)]
    # alpha(x, 1) = 1 - 2*(1+x) / [(1+x)*(3-x)]
    # alpha(x, 1) = 1 - 2 / (3-x)
    # alpha(x, 1) = (3-x - 2) / (3-x) = (1-x) / (3-x)
    
    expr = (1 - x) / (3 - x)
    
    # Isotropic average: (1/2) * integral from -1 to 1 of alpha(x, 1) dx
    avg_expr = sp.Rational(1, 2) * sp.integrate(expr, (x, -1, 1))
    
    print(f"Formula for rho=1: alpha(x) = {expr}")
    print(f"Isotropic average integral: 1/2 * integral_{-1}^1 (1-x)/(3-x) dx")
    print(f"Result: {avg_expr}")
    print(f"Numerical value: {float(avg_expr.evalf()):.10f}")
    print(f"1 - ln(2) value: {float((1 - sp.log(2)).evalf()):.10f}")

if __name__ == "__main__":
    derive_alpha_iso()
