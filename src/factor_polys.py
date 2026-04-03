import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

print("="*75)
print("FACTORING THE MINIMAL POLYNOMIALS")
print("="*75)

# Degree-5 for L2_red: t^5 - 17t^4 + 104t^3 - 270t^2 + 260t - 52 = 0
print("\nDegree-5 polynomial: t^5 - 17t^4 + 104t^3 - 270t^2 + 260t - 52")

# Try rational roots
poly5_coeffs = [1, -17, 104, -270, 260, -52]
print("Rational root test (divisors of 52):")
for r in [1, 2, 4, 13, 26, 52, -1, -2, -4, -13, -26, -52]:
    val = sum(poly5_coeffs[i] * r**(5-i) for i in range(6))
    if abs(val) < 0.5:
        print(f"  Root found: t = {r}")

# Try (t^2 + at + b)(t^3 + ct^2 + dt + e) = t^5 - 17t^4 + 104t^3 - 270t^2 + 260t - 52
print("\nTrying (t^2 + at + b)(t^3 + ct^2 + dt + e):")
found5 = False
for b in range(-60, 61):
    if b == 0:
        continue
    for e in range(-60, 61):
        if e == 0:
            continue
        if b * e != -52:
            continue
        for a in range(-30, 31):
            c = -17 - a
            d = 104 - b - a * c
            # Check: ad + bc + e = -270
            if a * d + b * c + e != -270:
                continue
            # Check: ae + bd = 260
            if a * e + b * d != 260:
                continue
            print(f"  FOUND: (t^2 + ({a})t + ({b}))(t^3 + ({c})t^2 + ({d})t + ({e}))")
            found5 = True

            r_quad = np.roots([1, a, b])
            r_cubic = np.roots([1, c, d, e])
            print(f"  Quadratic roots: {np.sort(r_quad.real)}")
            print(f"  Cubic roots: {np.sort(r_cubic.real)}")

            for r in r_quad:
                if abs(r.imag) < 1e-6 and abs(r.real - 0.2665) < 0.01:
                    disc = a*a - 4*b
                    print(f"\n  *** L2_red is in QUADRATIC: t^2 + ({a})t + ({b}) = 0")
                    print(f"  Discriminant = {a}^2 - 4*{b} = {disc}")
                    if disc >= 0:
                        print(f"  L2_red = ({-a} - sqrt({disc})) / 2 = {(-a - np.sqrt(disc))/2:.16f}")

            for r in r_cubic:
                if abs(r.imag) < 1e-6 and abs(r.real - 0.2665) < 0.01:
                    print(f"\n  *** L2_red is in CUBIC: t^3 + ({c})t^2 + ({d})t + ({e}) = 0")

if not found5:
    print("  No (quad)(cubic) factorization found.")

# Degree-7 for L2_full: t^7 - 33t^6 + 443t^5 - 3097t^4 + 11948t^3 - 24634t^2 + 23588t - 6916 = 0
print("\n" + "="*75)
print("Degree-7 polynomial: t^7 - 33t^6 + 443t^5 - 3097t^4 + 11948t^3 - 24634t^2 + 23588t - 6916")

# Try (t^2 + at + b)(t^5 + ...) and (t^3 + ...)(t^4 + ...)
print("\nTrying (t^2 + at + b)(t^5 + ct^4 + dt^3 + et^2 + ft + g):")
found7_2_5 = False
for b in range(-100, 101):
    if b == 0:
        continue
    if 6916 % abs(b) != 0:
        continue
    g = -6916 // b
    for a in range(-40, 41):
        c = -33 - a
        d = 443 - b - a*c
        e = -3097 - a*d - b*c
        f = 11948 - a*e - b*d
        # Check remaining:
        check1 = a*f + b*e  # should be -24634 (wait, this is wrong)
        # Actually for (t^2+at+b)(t^5+ct^4+dt^3+et^2+ft+g):
        # t^7 + (a+c)t^6 + (b+ac+d)t^5 + (ad+bc+e)t^4 + (ae+bd+f)t^3
        #     + (af+be+g)t^2 + (ag+bf)t + bg
        c6 = a + c        # = -33 (by construction)
        c5 = b + a*c + d  # = 443 (by construction)
        c4 = a*d + b*c + e  # = -3097 (by construction)
        c3 = a*e + b*d + f  # = 11948 (by construction)
        c2 = a*f + b*e + g  # should be -24634
        c1 = a*g + b*f     # should be 23588
        c0 = b*g           # should be -6916 (by construction)
        if c2 == -24634 and c1 == 23588:
            print(f"  FOUND: (t^2+({a})t+({b}))(t^5+({c})t^4+({d})t^3+({e})t^2+({f})t+({g}))")
            found7_2_5 = True
            r2 = np.roots([1, a, b])
            r5 = np.roots([1, c, d, e, f, g])
            print(f"  Quadratic roots: {r2}")
            print(f"  Quintic roots: {np.sort(r5.real[np.abs(r5.imag)<0.01])}")
            for r in list(r2) + list(r5):
                if abs(r.imag) < 1e-6 and abs(r.real - 0.4950) < 0.01:
                    if r in r2:
                        print(f"  *** L2_full is in QUADRATIC factor")
                    else:
                        print(f"  *** L2_full is in QUINTIC factor")

if not found7_2_5:
    print("  No (t^2)(t^5) factorization found.")

print("\nTrying (t^3 + at^2 + bt + c)(t^4 + dt^3 + et^2 + ft + g):")
found7_3_4 = False
# c*g = -6916
divs6916 = []
for d in range(1, 200):
    if 6916 % d == 0:
        divs6916.extend([d, -d])

for c_val in divs6916:
    if abs(c_val) > 100:
        continue
    if 6916 % abs(c_val) != 0:
        continue
    g_val = -6916 // c_val
    if abs(g_val) > 5000:
        continue
    for a_val in range(-35, 2):
        d_val = -33 - a_val
        for b_val in range(-100, 101):
            e_val = 443 - a_val*d_val - b_val
            f_val = -3097 - (c_val + a_val*e_val + b_val*d_val)
            # Check:
            c3 = a_val*f_val + b_val*e_val + c_val*d_val + g_val
            c2 = a_val*g_val + b_val*f_val + c_val*e_val
            c1 = b_val*g_val + c_val*f_val
            c0 = c_val*g_val
            if c3 == 11948 and c2 == -24634 and c1 == 23588 and c0 == -6916:
                print(f"  FOUND: (t^3+({a_val})t^2+({b_val})t+({c_val}))(t^4+({d_val})t^3+({e_val})t^2+({f_val})t+({g_val}))")
                found7_3_4 = True
                r3 = np.roots([1, a_val, b_val, c_val])
                r4 = np.roots([1, d_val, e_val, f_val, g_val])
                r3_real = np.sort([x.real for x in r3 if abs(x.imag) < 0.01])
                r4_real = np.sort([x.real for x in r4 if abs(x.imag) < 0.01])
                print(f"  Cubic real roots: {r3_real}")
                print(f"  Quartic real roots: {r4_real}")
                for r in list(r3):
                    if abs(r.imag) < 1e-6 and abs(r.real - 0.4950) < 0.01:
                        print(f"  *** L2_full is in CUBIC: t^3+({a_val})t^2+({b_val})t+({c_val}) = 0")
                for r in list(r4):
                    if abs(r.imag) < 1e-6 and abs(r.real - 0.4950) < 0.01:
                        print(f"  *** L2_full is in QUARTIC: t^4+({d_val})t^3+({e_val})t^2+({f_val})t+({g_val}) = 0")
                break
        if found7_3_4:
            break
    if found7_3_4:
        break

if not found7_3_4:
    print("  No (t^3)(t^4) factorization found with |coeffs| < 100.")

# SUMMARY
print()
print("="*75)
print("EXACT CLOSED FORM - FINAL")
print("="*75)
print()
print("The star invariant R = 1.8573068741389...")
print()
print("is the ratio of smallest positive roots of two integer polynomials:")
print()
print("  Numerator eigenvalue from: (grounded 8x8 spoke Laplacian)")
print("    P7(t) = t^7 - 33t^6 + 443t^5 - 3097t^4 + 11948t^3 - 24634t^2 + 23588t - 6916")
print()
print("  Denominator eigenvalue from: (grounded 6x6 reduced spoke Laplacian)")
print("    P5(t) = t^5 - 17t^4 + 104t^3 - 270t^2 + 260t - 52")
print()

# Verify
roots7 = np.sort(np.roots([1, -33, 443, -3097, 11948, -24634, 23588, -6916]).real)
roots5 = np.sort(np.roots([1, -17, 104, -270, 260, -52]).real)
L2f = min(r for r in roots7 if r > 0.01)
L2r = min(r for r in roots5 if r > 0.01)
R = L2f / L2r

print(f"  lambda_min(P7) = {L2f:.16f}")
print(f"  lambda_min(P5) = {L2r:.16f}")
print(f"  R = {R:.16f}")
print()
print("  Fundamental irrational of the cluster: sqrt(17)")
print("  (from the K5 + anchor Laplacian eigenvalue pair: t^2 - 7t + 8 = 0)")
print()
print("  Best rational approximation: 13/7 (within 0.009%)")
print()
print("  The integer Laplacian matrices that generate this constant:")
print()
print("  8x8 (spoke grounded by 3-clause bridge to hub):")
print("  [[ 7 -1 -1 -1 -1 -1  0  0]")
print("   [-1  6 -1 -1 -1  0  0  0]")
print("   [-1 -1  6 -1 -1  0 -1  0]")
print("   [-1 -1 -1  5 -1 -1  0  0]")
print("   [-1 -1 -1 -1  6  0 -1  0]")
print("   [-1  0  0 -1  0  4 -1 -1]")
print("   [ 0  0 -1  0 -1 -1  4 -1]")
print("   [ 0  0  0  0  0 -1 -1  2]]")
print()
print("  6x6 (spoke after valve removal, grounded):")
print("  [[ 5 -1 -1 -1  0  0]")
print("   [-1  4 -1  0  0  0]")
print("   [-1 -1  3 -1  0  0]")
print("   [-1  0 -1  4 -1 -1]")
print("   [ 0  0  0 -1  2 -1]")
print("   [ 0  0  0 -1 -1  2]]")
