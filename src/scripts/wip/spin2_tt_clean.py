"""
Spin-2 TT Projector — Clean Computation
========================================
S96-M1c: Fix spin-1 Lamb vector, add pure geometric version,
identify exact analytical formula.

Two approaches:
  A) Physical: correct source tensors (Lamb vector for spin-1, Isaacson-like for spin-2)
  B) Geometric: pure projector overlap (how much of a helical mode at k1 is TT at k3?)
"""

import numpy as np
from numpy import sin, cos, pi, sqrt
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def orthonormal_basis(k_hat):
    """Find e1, e2 perpendicular to k_hat."""
    if abs(k_hat[2]) < 0.9:
        e1 = np.cross(k_hat, [0, 0, 1])
    else:
        e1 = np.cross(k_hat, [1, 0, 0])
    e1 = normalize(e1)
    e2 = np.cross(k_hat, e1)
    e2 = normalize(e2)
    return e1, e2


def leray(k_hat):
    """Leray projector P_{ij} = delta_{ij} - k_hat_i k_hat_j"""
    return np.eye(3) - np.outer(k_hat, k_hat)


def tt_project(S, k_hat):
    """
    TT-project a symmetric tensor S_{ij} w.r.t. direction k_hat.
    Lambda_{ij,kl} = (P_{ik}P_{jl} + P_{il}P_{jk})/2 - P_{ij}P_{kl}/2
    (symmetrized form for symmetric input)
    """
    P = leray(k_hat)
    # S_T = P S P (transverse part)
    S_T = P @ S @ P
    # Subtract trace: S_TT = S_T - (1/2) P Tr(S_T)
    tr_ST = np.trace(S_T)
    S_TT = S_T - 0.5 * P * tr_ST
    return S_TT


def helical_vectors(k_hat):
    """Spin-1 helical modes: h+ = (e1 + ie2)/sqrt(2), h- = (e1 - ie2)/sqrt(2)"""
    e1, e2 = orthonormal_basis(k_hat)
    return (e1 + 1j * e2) / sqrt(2), (e1 - 1j * e2) / sqrt(2)


def helical_tensors(k_hat):
    """Spin-2 helical modes (circular GW polarization)."""
    e1, e2 = orthonormal_basis(k_hat)
    e_plus = np.outer(e1, e1) - np.outer(e2, e2)
    e_cross = np.outer(e1, e2) + np.outer(e2, e1)
    e_R = (e_plus + 1j * e_cross) / sqrt(2)  # helicity +2
    e_L = (e_plus - 1j * e_cross) / sqrt(2)  # helicity -2
    return e_R, e_L


def tensor_norm2(T):
    """||T||^2 = T_{ij} T*_{ij}"""
    return np.real(np.sum(T * np.conj(T)))


# ============================================================
# SPIN-1: Lamb vector form (CORRECT)
# ============================================================
def spin1_lamb_suppression(theta, rho=1.0):
    """
    Two helical velocity modes h+(k1), h-(k2) at angle theta.
    Lamb vector: L(k3) = u1 x omega2 + u2 x omega1
    where omega = ik x u.

    Leray-projected fraction alpha_{+-}(theta).
    Known analytical: alpha = 1 - 2/(3-cos(theta)) for rho=1.
    """
    k1 = np.array([0, 0, 1.0])
    k2 = rho * np.array([sin(theta), 0, cos(theta)])
    k3 = k1 + k2
    k3_norm = np.linalg.norm(k3)
    if k3_norm < 1e-10:
        return 0.0
    k3_hat = k3 / k3_norm

    hp1, hm1 = helical_vectors(k1 / np.linalg.norm(k1))
    hp2, hm2 = helical_vectors(k2 / np.linalg.norm(k2))

    # Cross-helical: h+(k1) with h-(k2)
    # omega1 = i k1 x h+(k1), omega2 = i k2 x h-(k2)
    omega1 = 1j * np.cross(k1, hp1)
    omega2 = 1j * np.cross(k2, hm2)

    # Lamb vector: L = h+(k1) x omega2 + h-(k2) x omega1
    L = np.cross(hp1, omega2) + np.cross(hm2, omega1)

    # Leray project
    P = leray(k3_hat)
    L_proj = P @ L

    L_norm2 = np.real(np.dot(L, np.conj(L)))
    L_proj_norm2 = np.real(np.dot(L_proj, np.conj(L_proj)))

    if L_norm2 < 1e-15:
        return 0.0
    return L_proj_norm2 / L_norm2


# ============================================================
# SPIN-2: Approach A — Interaction source tensor
# ============================================================
def spin2_interaction_suppression(theta, rho=1.0):
    """
    Two GW modes e_R(k1), e_L(k2) at angle theta.
    Source: S_{ij} = sum of terms from Isaacson-like stress.

    Use the gauge-invariant form: S_{ij} ~ e1_{ab} k2_a k2_b * e2_{ij}
    + crossed terms + symmetrizations.

    Actually, simplest correct form for the quadratic source:
    S_{ij}(k3) = (k1.k2) e1_{ia} e2_{ja} + k1_a k2_b e1_{ab} e2_{ij}
                 + (symmetrize in ij, 1<->2)
    """
    k1 = np.array([0, 0, 1.0])
    k2 = rho * np.array([sin(theta), 0, cos(theta)])
    k3 = k1 + k2
    k3_norm = np.linalg.norm(k3)
    if k3_norm < 1e-10:
        return 0.0
    k3_hat = k3 / k3_norm

    eR1, eL1 = helical_tensors(normalize(k1))
    eR2, eL2 = helical_tensors(normalize(k2))

    # Cross-helical: R(k1) x L(k2)
    e1, e2_tensor = eR1, eL2

    k1k2 = np.dot(k1, k2)
    # Term 1: (k1.k2) * e1_{ia} e2_{ja}  [matrix product]
    t1 = k1k2 * (e1 @ np.conj(e2_tensor).T)
    # Term 2: (k1_a k2_b e1_{ab}) * e2_{ij}
    scalar12 = np.einsum('a,b,ab->', k1, k2, e1)
    t2 = scalar12 * e2_tensor
    # Term 3: (k2_a k1_b e2_{ab}) * e1_{ij}
    scalar21 = np.einsum('a,b,ab->', k2, k1, e2_tensor)
    t3 = scalar21 * e1
    # Symmetrize in ij and 1<->2
    S = (t1 + t1.T + t2 + t2.T + t3 + t3.T) / 6.0

    S_TT = tt_project(S, k3_hat)

    S_n2 = tensor_norm2(S)
    S_TT_n2 = tensor_norm2(S_TT)

    if S_n2 < 1e-15:
        return 0.0
    return S_TT_n2 / S_n2


# ============================================================
# SPIN-2: Approach B — Pure geometric overlap
# ============================================================
def spin2_geometric_suppression(theta, rho=1.0):
    """
    Pure geometric: take e_R(k1), TT-project at k3 = k1 + rho*k2.
    Fraction = ||Lambda(k3) . e_R(k1)||^2 / ||e_R(k1)||^2

    This is the direct spin-2 analogue of:
    spin-1: ||P(k3) . h+(k1)||^2 / ||h+(k1)||^2
    """
    k1 = np.array([0, 0, 1.0])
    k2 = rho * np.array([sin(theta), 0, cos(theta)])
    k3 = k1 + k2
    k3_norm = np.linalg.norm(k3)
    if k3_norm < 1e-10:
        return 0.0
    k3_hat = k3 / k3_norm

    eR1, eL1 = helical_tensors(normalize(k1))

    # TT project e_R(k1) w.r.t. k3
    eR1_TT = tt_project(eR1, k3_hat)

    return tensor_norm2(eR1_TT) / tensor_norm2(eR1)


def spin1_geometric_suppression(theta, rho=1.0):
    """
    Pure geometric spin-1: ||P(k3) . h+(k1)||^2 / ||h+(k1)||^2
    """
    k1 = np.array([0, 0, 1.0])
    k2 = rho * np.array([sin(theta), 0, cos(theta)])
    k3 = k1 + k2
    k3_norm = np.linalg.norm(k3)
    if k3_norm < 1e-10:
        return 0.0
    k3_hat = k3 / k3_norm

    hp1, _ = helical_vectors(normalize(k1))
    P = leray(k3_hat)
    hp1_proj = P @ hp1

    return np.real(np.dot(hp1_proj, np.conj(hp1_proj)))  # |h+|^2 = 1


# ============================================================
# SPIN-2: Approach C — Cross-helical tensor product
# ============================================================
def spin2_cross_product_suppression(theta, rho=1.0):
    """
    Cross-helical product: S_{ij} = h+_i(k1) h-_j(k2) + h-_j(k2) h+_i(k1)
    (symmetrized outer product of spin-1 helical modes → spin-2 object)
    TT-project at k3.

    This is the most natural "cross-helical" spin-2 source.
    """
    k1 = np.array([0, 0, 1.0])
    k2 = rho * np.array([sin(theta), 0, cos(theta)])
    k3 = k1 + k2
    k3_norm = np.linalg.norm(k3)
    if k3_norm < 1e-10:
        return 0.0
    k3_hat = k3 / k3_norm

    hp1, _ = helical_vectors(normalize(k1))
    _, hm2 = helical_vectors(normalize(k2))

    # Symmetric outer product
    S = (np.outer(hp1, hm2) + np.outer(hm2, hp1)) / 2.0

    S_TT = tt_project(S, k3_hat)

    S_n2 = tensor_norm2(S)
    S_TT_n2 = tensor_norm2(S_TT)

    if S_n2 < 1e-15:
        return 0.0
    return S_TT_n2 / S_n2


# ============================================================
# Integration
# ============================================================
def sphere_average(func, N=10000):
    """<f(theta)> = integral_0^pi f(theta) sin(theta) d(theta) / 2"""
    theta = np.linspace(0.001, pi - 0.001, N)
    vals = np.array([func(t) for t in theta])
    return np.trapezoid(vals * sin(theta), theta) / 2.0


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 72)
    print("SPIN-2 TT PROJECTOR — CLEAN COMPUTATION")
    print("=" * 72)

    angles = [pi/6, pi/4, pi/3, pi/2, 2*pi/3, 3*pi/4, 5*pi/6]
    names = ['pi/6', 'pi/4', 'pi/3', 'pi/2', '2pi/3', '3pi/4', '5pi/6']

    # --- SPIN-1 VERIFICATION ---
    print("\n--- Spin-1 Leray: Lamb vector (should give 1-2/(3-cos)) ---")
    print(f"{'theta':>8} {'Lamb alpha':>12} {'Formula':>12} {'Match?':>8}")
    print("-" * 44)
    for name, t in zip(names, angles):
        a = spin1_lamb_suppression(t)
        formula = 1 - 2 / (3 - cos(t))
        match = "YES" if abs(a - formula) < 1e-4 else "NO"
        print(f"{name:>8} {a:>12.6f} {formula:>12.6f} {match:>8}")

    avg_s1_lamb = sphere_average(spin1_lamb_suppression)
    print(f"\nIsotropic avg = {avg_s1_lamb:.6f}  (expected 1-ln2 = {1-np.log(2):.6f})")

    # --- GEOMETRIC COMPARISON ---
    print("\n" + "=" * 72)
    print("GEOMETRIC SUPPRESSION: Spin-1 vs Spin-2")
    print("(Pure projector overlap, no dynamics)")
    print("=" * 72)
    print(f"{'theta':>8} {'S1 geom':>10} {'S2 geom':>10} {'S2 cross':>10} {'S2 interact':>12}")
    print("-" * 54)
    for name, t in zip(names, angles):
        s1g = spin1_geometric_suppression(t)
        s2g = spin2_geometric_suppression(t)
        s2c = spin2_cross_product_suppression(t)
        s2i = spin2_interaction_suppression(t)
        print(f"{name:>8} {s1g:>10.6f} {s2g:>10.6f} {s2c:>10.6f} {s2i:>12.6f}")

    # --- ISOTROPIC AVERAGES ---
    print("\n--- Isotropic averages ---")
    avgs = {}
    for label, func in [
        ("Spin-1 Lamb (physical)", spin1_lamb_suppression),
        ("Spin-1 geometric", spin1_geometric_suppression),
        ("Spin-2 geometric", spin2_geometric_suppression),
        ("Spin-2 cross-product", spin2_cross_product_suppression),
        ("Spin-2 interaction", spin2_interaction_suppression),
    ]:
        a = sphere_average(func)
        avgs[label] = a
        print(f"  {label:>30}: {a:.8f}")

    print(f"\n  Reference: 1-ln(2) = {1-np.log(2):.8f}")
    print(f"  Reference: pi(sqrt2-1)/4 = {pi*(sqrt(2)-1)/4:.8f}")
    print(f"  Reference: 1/6 = {1/6:.8f}")

    # --- ANALYTICAL FORMULA SEARCH for spin-2 geometric ---
    print("\n" + "=" * 72)
    print("ANALYTICAL FORMULA SEARCH — Spin-2 geometric")
    print("=" * 72)

    # Dense sampling
    thetas = np.linspace(0.001, pi - 0.001, 500)
    s2g_vals = np.array([spin2_geometric_suppression(t) for t in thetas])
    c_vals = np.cos(thetas)

    # Check exact values at nice angles
    print("\nExact values at nice angles:")
    nice = [(pi/6, 'pi/6'), (pi/4, 'pi/4'), (pi/3, 'pi/3'),
            (pi/2, 'pi/2'), (2*pi/3, '2pi/3'), (3*pi/4, '3pi/4')]
    for t, name in nice:
        v = spin2_geometric_suppression(t)
        # Try to identify as fraction
        print(f"  {name:>8}: {v:.10f}")

    # Test specific formulas
    print("\nFormula candidates (spin-2 geometric):")
    candidates = {
        'sin^2(t)/4':        lambda t: sin(t)**2/4,
        'sin^4(t)/8':        lambda t: sin(t)**4/8,
        'sin^4(t)/16':       lambda t: sin(t)**4/16,
        'sin^2(t)(1-cos(t))/8': lambda t: sin(t)**2*(1-cos(t))/8,
        '(1-cos^2(t))^2/8':  lambda t: (1-cos(t)**2)**2/8,
        '3sin^4(t)/16':      lambda t: 3*sin(t)**4/16,
        'sin^2(t)sin^2(t/2)/2': lambda t: sin(t)**2*sin(t/2)**2/2,
        '(1-cos(t))^2(1+cos(t))/8': lambda t: (1-cos(t))**2*(1+cos(t))/8,
        '(1-cos(t))^2(2+cos(t))/16': lambda t: (1-cos(t))**2*(2+cos(t))/16,
        'sin^4(t)/(4(1+cos(t))^2+8)': lambda t: sin(t)**4/(4*(1+cos(t))**2+8),
    }

    for label, f in candidates.items():
        vals = np.array([f(t) for t in thetas])
        maxdiff = np.max(np.abs(vals - s2g_vals))
        flag = " *** MATCH ***" if maxdiff < 1e-6 else ""
        print(f"  {label:>40}: max diff = {maxdiff:.2e}{flag}")

    # Try polynomial in cos(theta)
    print("\n--- Polynomial fit in cos(theta) ---")
    from numpy.polynomial import polynomial as Poly
    # Fit: alpha = sum_k a_k cos^k(theta)
    for deg in [2, 3, 4, 5, 6]:
        coeffs = np.polyfit(c_vals, s2g_vals, deg)
        fit = np.polyval(coeffs, c_vals)
        res = np.max(np.abs(fit - s2g_vals))
        if res < 1e-4:
            print(f"  Degree {deg}: max residual = {res:.2e}")
            print(f"    Coefficients: {coeffs}")
            break
        elif deg <= 4:
            print(f"  Degree {deg}: max residual = {res:.2e}")

    # --- SPIN-2 CROSS-PRODUCT formula search ---
    print("\n" + "=" * 72)
    print("ANALYTICAL FORMULA SEARCH — Spin-2 cross-helical product")
    print("=" * 72)

    s2c_vals = np.array([spin2_cross_product_suppression(t) for t in thetas])

    print("\nExact values at nice angles:")
    for t, name in nice:
        v = spin2_cross_product_suppression(t)
        print(f"  {name:>8}: {v:.10f}")

    print("\nFormula candidates:")
    for label, f in candidates.items():
        vals = np.array([f(t) for t in thetas])
        maxdiff = np.max(np.abs(vals - s2c_vals))
        flag = " *** MATCH ***" if maxdiff < 1e-6 else ""
        print(f"  {label:>40}: max diff = {maxdiff:.2e}{flag}")

    # Rational fit
    print("\n--- Polynomial in cos(theta) for cross-product ---")
    for deg in [2, 3, 4, 5, 6]:
        coeffs = np.polyfit(c_vals, s2c_vals, deg)
        fit = np.polyval(coeffs, c_vals)
        res = np.max(np.abs(fit - s2c_vals))
        if res < 1e-4:
            print(f"  Degree {deg}: max residual = {res:.2e}")
            print(f"    Coefficients (highest first): {np.array2string(coeffs, precision=8)}")
            break
        elif deg <= 4:
            print(f"  Degree {deg}: max residual = {res:.2e}")

    # Try extended candidate list for cross-product
    print("\nExtended candidates for cross-product:")
    more_candidates = {
        'sin^2(t)(3+cos(t))/32':   lambda t: sin(t)**2*(3+cos(t))/32,
        'sin^2(t)(1+cos(t))/16':   lambda t: sin(t)**2*(1+cos(t))/16,
        'sin^2(t)(5+3cos(t))/64':  lambda t: sin(t)**2*(5+3*cos(t))/64,
        'sin^4(t/2)cos^2(t/2)':    lambda t: sin(t/2)**4*cos(t/2)**2,
        '(1-cos(t))^2(1+cos(t))/16': lambda t: (1-cos(t))**2*(1+cos(t))/16,
        'sin^4(t/2)(1+cos(t))/4':  lambda t: sin(t/2)**4*(1+cos(t))/4,
        '(1-cos^2(t))^2/(16-8cos(t))': lambda t: sin(t)**4/(16-8*cos(t)) if abs(16-8*cos(t))>1e-10 else 0,
        'sin^2(t)/(4(3-cos(t)))':  lambda t: sin(t)**2/(4*(3-cos(t))),
    }
    for label, f in more_candidates.items():
        vals = np.array([f(t) for t in thetas])
        maxdiff = np.max(np.abs(vals - s2c_vals))
        flag = " *** MATCH ***" if maxdiff < 1e-6 else ""
        print(f"  {label:>40}: max diff = {maxdiff:.2e}{flag}")

    # Check boundary values for cross-product
    print("\nBoundary analysis (cross-product):")
    for t_val in [0.001, 0.01, 0.1, pi-0.1, pi-0.01, pi-0.001]:
        v = spin2_cross_product_suppression(t_val)
        s4 = sin(t_val)**4
        ratio_s4 = v / s4 if s4 > 1e-20 else float('inf')
        print(f"  theta={t_val:.4f}: alpha={v:.10f}, sin^4(t)={s4:.10f}, ratio={ratio_s4:.6f}")

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for label, a in avgs.items():
        print(f"  {label:>30}: <alpha> = {a:.8f}")
    print(f"\n  Key: sin^2(t)/4 average = 1/6 = {1/6:.8f}")
    print(f"  Key: 1-ln(2) = {1-np.log(2):.8f}")
    print(f"  Key: pi(sqrt2-1)/4 = {pi*(sqrt(2)-1)/4:.8f}")
