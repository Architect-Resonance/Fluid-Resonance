"""
Spin-2 TT Projector on Helical Gravitational Wave Modes
========================================================
S96-M1c: Wanderer's verification request from S96-W5.

Question: Does the spin-2 TT projector applied to cross-helical GW modes
give sin²θ/4, or something else?

Comparison:
  - Spin-1 (fluid):  Leray projector P_{ij} = δ_{ij} - k̂_i k̂_j
    → Cross-helical suppression α_{+-}(θ) = 1 - 2/(3-cosθ)
    → Isotropic average = 1 - ln(2) = 0.3069

  - Spin-2 (gravity): TT projector Λ_{ij,kl} = P_{ik}P_{jl} - ½P_{ij}P_{kl}
    → Cross-helical suppression = ???
    → Isotropic average = ???

Method: Explicit computation with numpy (numerical) and sympy (symbolic).
"""

import numpy as np
from numpy import sin, cos, pi, sqrt
import sympy as sp


def leray_projector(k_hat):
    """P_{ij} = δ_{ij} - k̂_i k̂_j"""
    return np.eye(3) - np.outer(k_hat, k_hat)


def tt_projector(k_hat):
    """
    Λ_{ij,kl} = P_{ik}P_{jl} - ½P_{ij}P_{kl}
    Returns rank-4 tensor (3,3,3,3).
    """
    P = leray_projector(k_hat)
    Lambda = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Lambda[i, j, k, l] = (
                        P[i, k] * P[j, l]
                        - 0.5 * P[i, j] * P[k, l]
                    )
    return Lambda


def circular_polarization_tensors(k_hat):
    """
    For wave propagating along k_hat, construct:
      e^R_{ij} = (e^+_{ij} + i e^×_{ij}) / √2   (helicity +2)
      e^L_{ij} = (e^+_{ij} - i e^×_{ij}) / √2   (helicity -2)

    where e^+ = ê₁⊗ê₁ - ê₂⊗ê₂, e^× = ê₁⊗ê₂ + ê₂⊗ê₁.
    """
    # Find orthonormal basis perpendicular to k_hat
    if abs(k_hat[2]) < 0.9:
        e1 = np.cross(k_hat, [0, 0, 1])
    else:
        e1 = np.cross(k_hat, [1, 0, 0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(k_hat, e1)
    e2 = e2 / np.linalg.norm(e2)

    # Plus and cross polarizations
    e_plus = np.outer(e1, e1) - np.outer(e2, e2)
    e_cross = np.outer(e1, e2) + np.outer(e2, e1)

    # Circular polarization (helicity eigenstates)
    e_R = (e_plus + 1j * e_cross) / sqrt(2)  # helicity +2
    e_L = (e_plus - 1j * e_cross) / sqrt(2)  # helicity -2

    return e_R, e_L


def helical_velocity_modes(k_hat):
    """
    For spin-1 velocity: circular polarization vectors (helicity ±1).
    h^+ = (ê₁ + iê₂)/√2, h^- = (ê₁ - iê₂)/√2
    """
    if abs(k_hat[2]) < 0.9:
        e1 = np.cross(k_hat, [0, 0, 1])
    else:
        e1 = np.cross(k_hat, [1, 0, 0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(k_hat, e1)
    e2 = e2 / np.linalg.norm(e2)

    h_plus = (e1 + 1j * e2) / sqrt(2)
    h_minus = (e1 - 1j * e2) / sqrt(2)
    return h_plus, h_minus


# ============================================================
# COMPUTATION 1: Spin-1 Leray suppression (verification)
# ============================================================
def spin1_suppression(theta):
    """
    Cross-helical Leray suppression for spin-1 modes.

    Two velocity modes h^+(k₁) and h^-(k₂) at angle θ.
    Lamb vector L = u₁ × ω₂ + u₂ × ω₁ (simplified for two modes).

    The Leray projection kills the gradient part.
    Known result: α_{+-}(θ) = 1 - 2/(3 - cosθ)
    """
    k1 = np.array([0, 0, 1.0])
    k2 = np.array([sin(theta), 0, cos(theta)])
    k3 = k1 + k2
    k3_norm = np.linalg.norm(k3)
    if k3_norm < 1e-10:
        return 0.0
    k3_hat = k3 / k3_norm

    # Get helical modes
    hp1, hm1 = helical_velocity_modes(k1)
    hp2, hm2 = helical_velocity_modes(k2)

    # Cross-helical: h^+(k₁) with h^-(k₂)
    # Lamb-like interaction: L_i = ε_{ijk} u₁_j (ik₂ × u₂)_k
    # Simplified: outer product contribution to nonlinear term
    # For NS: (u·∇)u in Fourier space → i k₂_j u₁_j u₂_i (convolution)
    # The relevant tensor is: S_i = (k₂ · h^+(k₁)) h^-(k₂)_i + (k₁ · h^-(k₂)) h^+(k₁)_i

    S = np.dot(k2, hp1) * hm2 + np.dot(k1, hm2) * hp1

    # Leray project
    P = leray_projector(k3_hat)
    S_proj = P @ S

    # Fraction
    if np.abs(np.dot(S, np.conj(S))) < 1e-15:
        return 0.0
    alpha = np.real(np.dot(S_proj, np.conj(S_proj)) / np.dot(S, np.conj(S)))
    return alpha


# ============================================================
# COMPUTATION 2: Spin-2 TT suppression
# ============================================================
def spin2_source_tensor(k1, k2, e1, e2):
    """
    Quadratic GW interaction source tensor.

    At second order in perturbation theory, two GW modes interact to produce:
    S_{ij}(k₃) ~ k₁_a k₂_b [e₁_{ia} e₂_{jb} + e₁_{ja} e₂_{ib}
                                + e₁_{ib} e₂_{ja} + e₁_{jb} e₂_{ia}] / 4

    This is the symmetrized "stress" from two graviton modes.
    (Simplified from Isaacson effective stress-energy.)
    """
    S = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            for a in range(3):
                for b in range(3):
                    S[i, j] += k1[a] * k2[b] * (
                        e1[i, a] * e2[j, b] +
                        e1[j, a] * e2[i, b] +
                        e1[i, b] * e2[j, a] +
                        e1[j, b] * e2[i, a]
                    ) / 4.0
    return S


def spin2_suppression(theta):
    """
    Cross-helical TT suppression for spin-2 modes.

    Two GW modes e^R(k₁) and e^L(k₂) at angle θ.
    Compute: α^{GW}_{RL}(θ) = |Λ(k̂₃) · S|² / |S|²
    """
    k1 = np.array([0, 0, 1.0])
    k2 = np.array([sin(theta), 0, cos(theta)])
    k3 = k1 + k2
    k3_norm = np.linalg.norm(k3)
    if k3_norm < 1e-10:
        return 0.0
    k3_hat = k3 / k3_norm

    # Get helical GW polarization tensors
    eR1, eL1 = circular_polarization_tensors(k1)
    eR2, eL2 = circular_polarization_tensors(k2)

    # Cross-helical: R(k₁) × L(k₂)
    S = spin2_source_tensor(k1, k2, eR1, eL2)

    # Apply TT projector
    Lambda = tt_projector(k3_hat)
    S_TT = np.einsum('ijkl,kl->ij', Lambda, S)

    # Compute fractions
    S_norm2 = np.real(np.sum(S * np.conj(S)))
    S_TT_norm2 = np.real(np.sum(S_TT * np.conj(S_TT)))

    if S_norm2 < 1e-15:
        return 0.0
    return S_TT_norm2 / S_norm2


def spin2_suppression_same(theta):
    """Same-helical: R(k₁) × R(k₂)"""
    k1 = np.array([0, 0, 1.0])
    k2 = np.array([sin(theta), 0, cos(theta)])
    k3 = k1 + k2
    k3_norm = np.linalg.norm(k3)
    if k3_norm < 1e-10:
        return 0.0
    k3_hat = k3 / k3_norm

    eR1, eL1 = circular_polarization_tensors(k1)
    eR2, eL2 = circular_polarization_tensors(k2)

    S = spin2_source_tensor(k1, k2, eR1, eR2)
    Lambda = tt_projector(k3_hat)
    S_TT = np.einsum('ijkl,kl->ij', Lambda, S)

    S_norm2 = np.real(np.sum(S * np.conj(S)))
    S_TT_norm2 = np.real(np.sum(S_TT * np.conj(S_TT)))

    if S_norm2 < 1e-15:
        return 0.0
    return S_TT_norm2 / S_norm2


# ============================================================
# COMPUTATION 3: Isotropic averages via numerical integration
# ============================================================
def isotropic_average(func, N=10000):
    """
    Compute <f(θ)> over the sphere = ∫₀^π f(θ) sinθ dθ / 2
    """
    theta = np.linspace(0.01, pi - 0.01, N)
    values = np.array([func(t) for t in theta])
    # Trapezoidal integration with sinθ weight
    integrand = values * sin(theta)
    avg = np.trapezoid(integrand, theta) / 2.0
    return avg


# ============================================================
# COMPUTATION 4: Analytical formula search (spin-2)
# ============================================================
def analytical_check(theta):
    """Check candidate analytical formulas against numerical spin-2 result."""
    s2 = sin(theta)**2
    c = cos(theta)
    candidates = {
        'sin²θ/4': s2 / 4,
        'sin⁴θ/16': s2**2 / 16,
        'sin²θ(1+cos²θ)/8': s2 * (1 + c**2) / 8,
        '1 - 4/(5-3cosθ)': 1 - 4 / (5 - 3*c) if abs(5-3*c) > 1e-10 else 0,
        '(1-2/(3-cosθ))²': (1 - 2/(3-c))**2 if abs(3-c) > 1e-10 else 0,
        'sin²θ·sin²(θ/2)/2': s2 * sin(theta/2)**2 / 2,
        '2sin⁴(θ/2)·cos²(θ/2)': 2 * sin(theta/2)**4 * cos(theta/2)**2,
    }
    return candidates


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("SPIN-2 TT PROJECTOR ON HELICAL GW MODES")
    print("S96-M1c: Wanderer verification request")
    print("=" * 70)

    # --- Point values ---
    test_angles = [0.01, pi/6, pi/4, pi/3, pi/2, 2*pi/3, 3*pi/4, 5*pi/6, pi-0.01]
    angle_names = ['~0', 'π/6', 'π/4', 'π/3', 'π/2', '2π/3', '3π/4', '5π/6', '~π']

    print("\n--- Cross-helical suppression: Spin-1 (Leray) vs Spin-2 (TT) ---")
    print(f"{'θ':>8} {'Spin-1 α':>12} {'Spin-2 α':>12} {'sin²θ/4':>12} {'Ratio s2/s1':>12}")
    print("-" * 60)

    s1_values = []
    s2_values = []
    for name, theta in zip(angle_names, test_angles):
        a1 = spin1_suppression(theta)
        a2 = spin2_suppression(theta)
        ref = sin(theta)**2 / 4
        ratio = a2 / a1 if a1 > 1e-10 else float('nan')
        s1_values.append(a1)
        s2_values.append(a2)
        print(f"{name:>8} {a1:>12.6f} {a2:>12.6f} {ref:>12.6f} {ratio:>12.6f}")

    # --- Same-helical comparison ---
    print("\n--- Same-helical suppression: Spin-2 (TT) ---")
    print(f"{'θ':>8} {'Same-hel α':>12} {'Cross-hel α':>12} {'Ratio same/cross':>16}")
    print("-" * 52)
    for name, theta in zip(angle_names, test_angles):
        a_same = spin2_suppression_same(theta)
        a_cross = spin2_suppression(theta)
        ratio = a_same / a_cross if a_cross > 1e-10 else float('nan')
        print(f"{name:>8} {a_same:>12.6f} {a_cross:>12.6f} {ratio:>16.6f}")

    # --- Isotropic averages ---
    print("\n--- Isotropic averages (sphere-weighted) ---")
    avg_s1_cross = isotropic_average(spin1_suppression)
    avg_s2_cross = isotropic_average(spin2_suppression)
    avg_s2_same = isotropic_average(spin2_suppression_same)

    print(f"Spin-1 cross-helical:  {avg_s1_cross:.6f}  (expected: 1-ln(2) = {1-np.log(2):.6f})")
    print(f"Spin-2 cross-helical:  {avg_s2_cross:.6f}")
    print(f"Spin-2 same-helical:   {avg_s2_same:.6f}")
    print(f"Ratio spin-2/spin-1 (cross):  {avg_s2_cross/avg_s1_cross:.6f}")
    print(f"sin²θ/4 average = 1/6:  {1/6:.6f}")
    print(f"π(√2-1)/4 (GW cross-pol avg): {pi*(sqrt(2)-1)/4:.6f}")

    # --- Analytical formula check ---
    print("\n--- Analytical formula matching (cross-helical spin-2) ---")
    test_thetas = [pi/6, pi/4, pi/3, pi/2, 2*pi/3]
    for theta in test_thetas:
        a2 = spin2_suppression(theta)
        candidates = analytical_check(theta)
        print(f"\nθ = {theta:.4f} ({theta/pi:.3f}π), spin-2 α = {a2:.8f}")
        for name, val in candidates.items():
            match = "  ✓ MATCH" if abs(val - a2) < 1e-6 else ""
            print(f"  {name:>30} = {val:.8f}  (diff = {val-a2:+.2e}){match}")

    # --- Dense plot data ---
    print("\n--- Generating dense profile for pattern recognition ---")
    thetas_dense = np.linspace(0.01, pi - 0.01, 200)
    s2_dense = np.array([spin2_suppression(t) for t in thetas_dense])

    # Try to fit: α(θ) = A·sin²θ + B·sin⁴θ + C·sin⁶θ
    from numpy.polynomial import polynomial as P
    x = np.sin(thetas_dense)**2
    # Fit as polynomial in sin²θ
    coeffs = np.polyfit(x, s2_dense, 4)
    print(f"Polynomial fit α(θ) ≈ p(sin²θ):")
    print(f"  Coefficients (highest power first): {coeffs}")
    fit_vals = np.polyval(coeffs, x)
    residual = np.max(np.abs(fit_vals - s2_dense))
    print(f"  Max residual: {residual:.2e}")

    # Try rational function: α = a·sin²θ / (b + c·cosθ)
    print("\n--- Rational function fit ---")
    # Try α(θ) = A(1 - cosθ)^a (1 + cosθ)^b
    c_vals = np.cos(thetas_dense)
    # Log-fit for power law in (1-cosθ) and (1+cosθ)
    mask = (s2_dense > 1e-8) & (1 - c_vals > 1e-8) & (1 + c_vals > 1e-8)
    if np.any(mask):
        log_alpha = np.log(s2_dense[mask])
        log_1mc = np.log(1 - c_vals[mask])
        log_1pc = np.log(1 + c_vals[mask])
        A_mat = np.column_stack([np.ones_like(log_1mc), log_1mc, log_1pc])
        result = np.linalg.lstsq(A_mat, log_alpha, rcond=None)
        C_fit, a_fit, b_fit = result[0]
        print(f"  Power law fit: α ≈ {np.exp(C_fit):.6f} · (1-cosθ)^{a_fit:.4f} · (1+cosθ)^{b_fit:.4f}")
        pred = np.exp(C_fit) * (1-c_vals[mask])**a_fit * (1+c_vals[mask])**b_fit
        res2 = np.max(np.abs(pred - s2_dense[mask]))
        print(f"  Max residual: {res2:.2e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Spin-1 (fluid, Leray):    ⟨α_{{+-}}⟩ = {avg_s1_cross:.6f}")
    print(f"Spin-2 (gravity, TT):     ⟨α_{{RL}}⟩ = {avg_s2_cross:.6f}")
    print(f"Spin-2 same-helical:      ⟨α_{{RR}}⟩ = {avg_s2_same:.6f}")
    print(f"GW cross-pol (different): π(√2-1)/4  = {pi*(sqrt(2)-1)/4:.6f}")
    print(f"\nDoes spin-2 give sin²θ/4? Check above table.")
    print(f"Does spin-2 average match any known constant? {avg_s2_cross:.10f}")
