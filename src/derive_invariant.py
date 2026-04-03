import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

R = 1.8573068741389058
L2f = 0.4949988739119799
L2r = 0.2665143174799662

# Build cluster adjacency
def c2e(clauses):
    edges = set()
    for c in clauses:
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                edges.add((min(c[i],c[j]), max(c[i],c[j])))
    return edges

cluster_clauses = [(0,1,2),(1,2,3),(2,3,4),(3,4,0),(4,0,1),(5,0,3),(6,2,4),(7,5,6)]
edges8 = c2e(cluster_clauses)
A8 = np.zeros((8,8))
for i,j in edges8: A8[i][j] = 1; A8[j][i] = 1
L8 = np.diag(A8.sum(axis=1)) - A8

print("="*75)
print("STEP 1: Cluster eigenvalue identification")
print("="*75)

eig8 = np.sort(np.linalg.eigvalsh(L8))
print(f"Cluster eigenvalues: {eig8}")
print(f"Non-integer pair: x = {eig8[1]:.16f}, 7-x = {eig8[4]:.16f}")
print(f"Sum: {eig8[1] + eig8[4]:.16f} (should be 7)")
print(f"Product: {eig8[1] * eig8[4]:.16f}")
p = eig8[1] * eig8[4]
disc = 49 - 4*p
print(f"Discriminant of t^2 - 7t + {p:.6f}: {disc:.16f}")
print(f"sqrt(disc) = {np.sqrt(disc):.16f}")
print(f"x = (7 - sqrt({disc:.6f})) / 2 = {(7 - np.sqrt(disc))/2:.16f}")

# Check if product is a nice number
print(f"\nProduct p = {p:.16f}")
print(f"p * 1 = {p:.10f}")
# Try p = a/b
for a in range(1, 100):
    for b in range(1, 100):
        if abs(a/b - p) < 1e-8:
            print(f"  p ~ {a}/{b} = {a/b:.12f} err = {abs(a/b - p):.2e}")

print()
print("="*75)
print("STEP 2: Quotient graph reduction")
print("="*75)
print("For star topology, Fiedler mode sets hub = 0.")
print("Eigenvalue is smallest eigenvalue of grounded spoke Laplacian.")

# Bridge grounding
D_bridge = np.zeros(8)
D_bridge[0] = 2  # spoke var 0: 2 hub connections
D_bridge[1] = 2  # spoke var 1: 2 hub connections
D_bridge[2] = 1  # spoke var 2: 1 hub connection
D_bridge[4] = 1  # spoke var 4: 1 hub connection

L_eff = L8 + np.diag(D_bridge)
eig_eff = np.sort(np.linalg.eigvalsh(L_eff))

print(f"\nGrounded spoke Laplacian (8x8):")
print(f"  D_bridge = {D_bridge}")
print(f"  Eigenvalues: {np.round(eig_eff, 10)}")
print(f"  lambda_min = {eig_eff[0]:.16f}")
print(f"  Expected L2_full = {L2f:.16f}")
print(f"  MATCH: {abs(eig_eff[0] - L2f) < 1e-10}")

# Reduced: remove spoke positions 2 and 4 (valve vars within spoke)
keep = [0, 1, 3, 5, 6, 7]
A_red = A8[np.ix_(keep, keep)]
D_bridge_red = D_bridge[keep]
L_eff_red = np.diag(A_red.sum(axis=1)) - A_red + np.diag(D_bridge_red)
eig_eff_red = np.sort(np.linalg.eigvalsh(L_eff_red))

print(f"\nReduced grounded spoke Laplacian (6x6):")
print(f"  Remaining positions: {keep}")
print(f"  D_bridge_red = {D_bridge_red}")
print(f"  Eigenvalues: {np.round(eig_eff_red, 10)}")
print(f"  lambda_min = {eig_eff_red[0]:.16f}")
print(f"  Expected L2_red = {L2r:.16f}")
print(f"  MATCH: {abs(eig_eff_red[0] - L2r) < 1e-10}")

print(f"\nRATIO = {eig_eff[0] / eig_eff_red[0]:.16f}")

print()
print("="*75)
print("STEP 3: Exact matrices")
print("="*75)

print("\n8x8 grounded spoke Laplacian (integer matrix!):")
L_eff_int = L_eff.astype(int)
print(L_eff_int)
print(f"\nAll entries are integers: {np.allclose(L_eff, L_eff_int)}")

print("\n6x6 reduced grounded Laplacian (integer matrix!):")
L_eff_red_int = L_eff_red.astype(int)
print(L_eff_red_int)
print(f"\nAll entries are integers: {np.allclose(L_eff_red, L_eff_red_int)}")

print()
print("="*75)
print("STEP 4: Characteristic polynomials (integer coefficients)")
print("="*75)

# The eigenvalues are roots of det(L - lambda*I) = 0
# For integer matrices, these are integer-coefficient polynomials!

# Compute characteristic polynomial via numpy
# p(x) = det(xI - L) = x^n - tr(L)*x^(n-1) + ...
coeffs8 = np.round(np.polynomial.polynomial.polyfromroots(eig_eff)).astype(int)
print(f"\n8x8 char poly (ascending powers):")
terms8 = []
for i, c in enumerate(coeffs8):
    if c != 0:
        terms8.append(f"{c}*x^{i}")
        print(f"  x^{i}: {c}")

coeffs6 = np.round(np.polynomial.polynomial.polyfromroots(eig_eff_red)).astype(int)
print(f"\n6x6 char poly (ascending powers):")
terms6 = []
for i, c in enumerate(coeffs6):
    if c != 0:
        terms6.append(f"{c}*x^{i}")
        print(f"  x^{i}: {c}")

# The ratio R = root1(P8) / root1(P6) where root1 = smallest positive root
print()
print("="*75)
print("STEP 5: Identify the minimal polynomials of the eigenvalues")
print("="*75)

# L2_full is the smallest root of the 8x8 char poly
# But maybe it satisfies a smaller polynomial (a factor of the char poly)

# Let me try to factor the characteristic polynomials
# The 8x8 has eigenvalues: let me see which are integers
print(f"\n8x8 eigenvalues: {np.round(eig_eff, 10)}")
print("Integer eigenvalues:")
for v in eig_eff:
    rv = round(v)
    if abs(v - rv) < 1e-8:
        print(f"  {rv} (factor: x - {rv})")

# After dividing out integer roots, the remaining polynomial
# contains the non-integer eigenvalues including L2_full
int_roots_8 = []
non_int_8 = []
for v in eig_eff:
    rv = round(v)
    if abs(v - rv) < 1e-8:
        int_roots_8.append(rv)
    else:
        non_int_8.append(v)

print(f"\nInteger roots (8x8): {int_roots_8}")
print(f"Non-integer roots (8x8): {[f'{v:.10f}' for v in non_int_8]}")

int_roots_6 = []
non_int_6 = []
for v in eig_eff_red:
    rv = round(v)
    if abs(v - rv) < 1e-8:
        int_roots_6.append(rv)
    else:
        non_int_6.append(v)

print(f"\nInteger roots (6x6): {int_roots_6}")
print(f"Non-integer roots (6x6): {[f'{v:.10f}' for v in non_int_6]}")

# Compute the residual polynomial for non-integer roots
# by dividing out (x - int_root) for each integer root
from numpy.polynomial import polynomial as P

poly8 = np.array(coeffs8, dtype=float)
for r in int_roots_8:
    # Divide poly by (x - r) using synthetic division
    # polynomial is in ascending power form
    n = len(poly8) - 1
    result = np.zeros(n)
    result[n-1] = poly8[n]
    for i in range(n-2, -1, -1):
        result[i] = poly8[i+1] + r * result[i+1]
    poly8 = result

print(f"\nResidual polynomial for 8x8 (after dividing integer roots):")
print(f"  Degree: {len(poly8)-1}")
print(f"  Coefficients (ascending): {np.round(poly8, 4)}")
# Check these are close to integers
poly8_int = np.round(poly8).astype(int)
print(f"  Integer coefficients: {poly8_int}")
print(f"  Max rounding error: {np.max(np.abs(poly8 - poly8_int)):.2e}")

# Verify: roots of residual should be the non-integer eigenvalues
res_roots_8 = np.sort(np.roots(poly8_int[::-1]))  # np.roots wants descending
print(f"  Roots: {res_roots_8}")
print(f"  Expected: {non_int_8}")

poly6 = np.array(coeffs6, dtype=float)
for r in int_roots_6:
    n = len(poly6) - 1
    result = np.zeros(n)
    result[n-1] = poly6[n]
    for i in range(n-2, -1, -1):
        result[i] = poly6[i+1] + r * result[i+1]
    poly6 = result

print(f"\nResidual polynomial for 6x6 (after dividing integer roots):")
print(f"  Degree: {len(poly6)-1}")
poly6_int = np.round(poly6).astype(int)
print(f"  Integer coefficients: {poly6_int}")
print(f"  Max rounding error: {np.max(np.abs(poly6 - poly6_int)):.2e}")
res_roots_6 = np.sort(np.roots(poly6_int[::-1]))
print(f"  Roots: {res_roots_6}")

print()
print("="*75)
print("STEP 6: THE EXACT FORM")
print("="*75)

# L2_full is the smallest root of poly8_int
# L2_red is the smallest root of poly6_int
# The ratio is an algebraic number defined by these two polynomials

print(f"\nL2_full is the smallest positive root of:")
terms = []
for i, c in enumerate(poly8_int):
    if c != 0:
        terms.append(f"{'+ ' if c > 0 and i > 0 else ''}{c}*t^{i}")
print(f"  {' '.join(terms)} = 0")

print(f"\nL2_red is the smallest positive root of:")
terms = []
for i, c in enumerate(poly6_int):
    if c != 0:
        terms.append(f"{'+ ' if c > 0 and i > 0 else ''}{c}*t^{i}")
print(f"  {' '.join(terms)} = 0")

# If both are quadratic, the ratio has a nice closed form
if len(poly8_int) == 3 and len(poly6_int) == 3:
    a1, b1, c1 = poly8_int
    a2, b2, c2 = poly6_int
    print(f"\nBOTH ARE QUADRATIC!")
    print(f"  L2_full = (-{b1} - sqrt({b1}^2 - 4*{a1}*{c1})) / (2*{c1})")
    disc1 = b1**2 - 4*a1*c1
    print(f"         = ({-b1} - sqrt({disc1})) / {2*c1}")
    print(f"         = {(-b1 - np.sqrt(disc1)) / (2*c1):.16f}")

    print(f"  L2_red  = (-{b2} - sqrt({b2}^2 - 4*{a2}*{c2})) / (2*{c2})")
    disc2 = b2**2 - 4*a2*c2
    print(f"         = ({-b2} - sqrt({disc2})) / {2*c2}")
    print(f"         = {(-b2 - np.sqrt(disc2)) / (2*c2):.16f}")

    print(f"\n  EXACT RATIO = ({-b1} - sqrt({disc1})) / ({2*c1}) ")
    print(f"              / ({-b2} - sqrt({disc2})) / ({2*c2})")
    print(f"              = {c2}/{c1} * ({-b1} - sqrt({disc1})) / ({-b2} - sqrt({disc2}))")

print()
print("="*75)
print("FINAL ANSWER")
print("="*75)
print(f"  Star invariant = {eig_eff[0] / eig_eff_red[0]:.16f}")
print(f"  = lambda_min(L_grounded_8x8) / lambda_min(L_grounded_6x6)")
print(f"  Both Laplacians are INTEGER matrices.")
print(f"  The ratio is an algebraic number of degree <= {len(poly8_int)-1} * {len(poly6_int)-1} = {(len(poly8_int)-1)*(len(poly6_int)-1)}")
