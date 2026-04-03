import sympy as sp

# The invariant value from our previous audits
R_val = 1.857312485

def search_algebraic(val, tol=1e-8):
    print(f"Searching for algebraic match for {val}...")
    x = sp.symbols('x')
    
    # Try nsimplify with different tolerances
    try:
        simp = sp.nsimplify(val, tolerance=tol, full=True)
        print(f"Simple match: {simp}")
    except:
        pass

    # Try to see if it's a root of a small polynomial
    # We can use the 'minpoly' approach by constructing a symbolic float
    val_mp = sp.Float(val, 50)
    
    # Check for 13/7
    thirteen_sevenths = sp.Rational(13, 7)
    diff = abs(val - float(thirteen_sevenths))
    print(f"Match with 13/7: {float(thirteen_sevenths)} (Diff: {diff:.10f})")

    # Check for roots of small polynomials manually if nsimplify fails
    # Standard constants
    constants = {
        "sqrt(2) + sqrt(3)/2": sp.sqrt(2) + sp.sqrt(3)/2,
        "phi^2 / sqrt(3)": (sp.GoldenRatio**2) / sp.sqrt(3),
        "(1 + sqrt(21))/3": (1 + sp.sqrt(21)) / 3,
        "2 * cos(pi/7) * 2": 4 * sp.cos(sp.pi/7),
        "1 + 2*sin(pi/7)": 1 + 2*sp.sin(sp.pi/7),
    }
    
    for name, c in constants.items():
        c_val = float(c.evalf())
        print(f"Checking {name}: {c_val} (Diff: {abs(val - c_val):.10f})")

if __name__ == "__main__":
    search_algebraic(R_val)
