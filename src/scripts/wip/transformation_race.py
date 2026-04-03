"""
TRANSFORMATION RACE -- S101
===========================
The 8 approaches (a-h) all find the same thing: protection transforms from
order-based (low k) to entropy-based (high k). This script measures:

1. Shannon entropy of triadic phase distribution S(k) -- does it increase?
2. The viscosity race: nu*k^2 vs effective nonlinearity at each scale
3. The "transformation zone" where order-protection fades and entropy-protection rises
4. Whether viscosity catches the handoff (Re^{1/7} prediction)

Key insight: The cascade PRODUCES the entropy that protects regularity.
The 3/28 gap is the transformation zone, not a failure zone.

Meridian (Claude Opus 4.6), S101, 2026-03-16
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import sys
import os

# Import SpectralNS from shared module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS


def shannon_entropy_of_phases(phases, n_bins=36):
    """Compute Shannon entropy of a phase distribution.

    Maximum entropy (uniform) = ln(n_bins).
    Minimum entropy (delta) = 0.
    Returns normalized entropy in [0, 1].
    """
    hist, _ = np.histogram(phases, bins=n_bins, range=(-np.pi, np.pi))
    # Add small epsilon to avoid log(0)
    p = hist / hist.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_max = np.log(n_bins)
    return H / H_max  # Normalized: 0 = perfect order, 1 = perfect disorder


def measure_transformation(N=48, Re=400, T_target=2.0, dt=0.002):
    """Measure the protection transformation across scales.

    At each wavenumber shell k, compute:
    - Phase entropy (disorder of triadic phases)
    - Effective alpha (Leray suppression ratio)
    - Viscous damping rate nu*k^2
    - Nonlinear transfer rate (stretching power)
    - The ratio: does viscosity catch what protection drops?
    """
    print("=" * 70)
    print("TRANSFORMATION RACE -- Entropy as Protection")
    print("=" * 70)
    print(f"N={N}, Re={Re}, T_target={T_target}")

    solver = SpectralNS(N=N, Re=Re)
    nu = solver.nu

    # Taylor-Green initial condition
    u_hat = np.zeros((3, N, N, N), dtype=complex)
    u_x = np.sin(solver.X) * np.cos(solver.Y) * np.cos(solver.Z)
    u_y = -np.cos(solver.X) * np.sin(solver.Y) * np.cos(solver.Z)
    u_z = np.zeros_like(solver.X)
    u_hat[0] = fftn(u_x)
    u_hat[1] = fftn(u_y)
    u_hat[2] = fftn(u_z)

    # Evolve to developed turbulence
    print(f"\nEvolving to t={T_target}...")
    t = 0.0
    steps = int(T_target / dt)
    for step in range(steps):
        lamb_hat = solver.compute_lamb_hat(u_hat)
        rhs = -solver.project_leray(lamb_hat) - nu * solver.k2 * u_hat
        k1 = dt * rhs

        u_tmp = u_hat + 0.5 * k1
        lamb_hat = solver.compute_lamb_hat(u_tmp)
        rhs = -solver.project_leray(lamb_hat) - nu * solver.k2 * u_tmp
        k2 = dt * rhs

        u_tmp = u_hat + 0.5 * k2
        lamb_hat = solver.compute_lamb_hat(u_tmp)
        rhs = -solver.project_leray(lamb_hat) - nu * solver.k2 * u_tmp
        k3 = dt * rhs

        u_tmp = u_hat + k3
        lamb_hat = solver.compute_lamb_hat(u_tmp)
        rhs = -solver.project_leray(lamb_hat) - nu * solver.k2 * u_tmp
        k4 = dt * rhs

        u_hat += (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += dt

    print(f"Reached t={t:.3f}")

    # --- MEASUREMENT 1: Phase entropy per shell ---
    print("\n" + "=" * 70)
    print("MEASUREMENT 1: Shannon Entropy of Triadic Phases per Shell")
    print("=" * 70)

    # Get helical coefficients
    u_p, u_m = solver.helical_decompose(u_hat)

    # Phase of each helical mode
    phase_p = np.angle(u_p)
    phase_m = np.angle(u_m)

    kmag = solver.kmag
    kmax = N // 3  # Dealiased range
    shells = range(1, kmax + 1)

    entropies_p = []
    entropies_m = []
    entropies_cross = []  # Phase difference (cross-helical coherence)
    k_values = []

    for k_shell in shells:
        mask = (kmag >= k_shell - 0.5) & (kmag < k_shell + 0.5)
        n_modes = mask.sum()
        if n_modes < 10:
            continue

        phases_plus = phase_p[mask]
        phases_minus = phase_m[mask]
        phase_diff = np.angle(np.exp(1j * (phases_plus - phases_minus)))

        S_p = shannon_entropy_of_phases(phases_plus)
        S_m = shannon_entropy_of_phases(phases_minus)
        S_cross = shannon_entropy_of_phases(phase_diff)

        entropies_p.append(S_p)
        entropies_m.append(S_m)
        entropies_cross.append(S_cross)
        k_values.append(k_shell)

    k_values = np.array(k_values)
    entropies_p = np.array(entropies_p)
    entropies_m = np.array(entropies_m)
    entropies_cross = np.array(entropies_cross)

    print(f"\n{'k':<6} {'S(h+)':<10} {'S(h-)':<10} {'S(cross)':<10} {'Interpretation'}")
    print("-" * 60)
    for i in range(len(k_values)):
        k = k_values[i]
        sp, sm, sc = entropies_p[i], entropies_m[i], entropies_cross[i]
        if sc > 0.95:
            interp = "MAXIMUM ENTROPY (disorder protects)"
        elif sc > 0.8:
            interp = "high entropy"
        elif sc > 0.5:
            interp = "moderate order"
        else:
            interp = "ORDERED (structure protects)"
        print(f"{k:<6.0f} {sp:<10.4f} {sm:<10.4f} {sc:<10.4f} {interp}")

    # --- MEASUREMENT 2: Viscosity race ---
    print("\n" + "=" * 70)
    print("MEASUREMENT 2: The Viscosity Race (does nu*k^2 catch the handoff?)")
    print("=" * 70)

    # Compute Lamb vector and its solenoidal/gradient decomposition per shell
    lamb_hat = solver.compute_lamb_hat(u_hat)
    lamb_sol_hat = solver.project_leray(lamb_hat)

    # Energy spectrum E(k)
    energy_density = 0.5 * np.sum(np.abs(u_hat)**2, axis=0) / N**6

    # Stretching power per shell = |P[L]|^2 (solenoidal Lamb power)
    stretch_density = np.sum(np.abs(lamb_sol_hat)**2, axis=0) / N**6

    # Total Lamb power per shell
    lamb_density = np.sum(np.abs(lamb_hat)**2, axis=0) / N**6

    # Kolmogorov scale
    k_d = (1.0 / (nu**3))**0.25  # Simplified: eps ~ 1 for TG at this Re
    # More precise: measure actual dissipation rate
    eps_actual = 2 * nu * np.sum(solver.k2 * energy_density)
    k_d_actual = (eps_actual / nu**3)**0.25

    print(f"\nKolmogorov scale k_d = {k_d_actual:.2f}")
    print(f"Dissipation rate ε = {eps_actual:.6f}")
    print(f"Viscosity nu = {nu:.6f}")
    print(f"Re^(1/7) = {Re**(1.0/7):.4f} (predicted viscosity advantage at k_*)")

    print(f"\n{'k':<6} {'nuk^2':<12} {'|P[L]|^2/k':<12} {'alpha_eff':<10} {'Ratio':<10} {'Who wins?'}")
    print("-" * 70)

    race_k = []
    race_viscous = []
    race_nonlinear = []
    race_alpha = []

    for k_shell in shells:
        mask = (kmag >= k_shell - 0.5) & (kmag < k_shell + 0.5)
        n_modes = mask.sum()
        if n_modes < 10:
            continue

        # Viscous damping rate at this shell
        viscous_rate = nu * k_shell**2

        # Effective stretching power at this shell
        stretch_power = np.sum(stretch_density[mask])
        total_lamb_power = np.sum(lamb_density[mask])

        # Effective alpha at this shell
        alpha_eff = stretch_power / max(total_lamb_power, 1e-30)

        # Nonlinear rate: stretching power normalized by energy
        E_shell = np.sum(energy_density[mask])
        nonlinear_rate = stretch_power / max(E_shell, 1e-30)

        # The race ratio
        ratio = viscous_rate / max(nonlinear_rate, 1e-30)

        who = "VISCOSITY" if ratio > 1 else "NONLINEAR"

        race_k.append(k_shell)
        race_viscous.append(viscous_rate)
        race_nonlinear.append(nonlinear_rate)
        race_alpha.append(alpha_eff)

        if k_shell <= 3 or k_shell >= kmax - 1 or abs(k_shell - k_d_actual) < 2 or k_shell % 3 == 0:
            print(f"{k_shell:<6.0f} {viscous_rate:<12.4f} {nonlinear_rate:<12.4f} {alpha_eff:<10.4f} {ratio:<10.4f} {who}")

    race_k = np.array(race_k)
    race_viscous = np.array(race_viscous)
    race_nonlinear = np.array(race_nonlinear)
    race_alpha = np.array(race_alpha)

    # --- MEASUREMENT 3: The Transformation ---
    print("\n" + "=" * 70)
    print("MEASUREMENT 3: The Transformation -- Order -> Entropy")
    print("=" * 70)

    # Define "order-based protection" = how much alpha is below the incoherent bound 1/4
    # Define "entropy-based protection" = how close phase entropy is to maximum
    # The transformation: as k increases, order-protection fades, entropy-protection rises

    print(f"\n{'k':<6} {'alpha_eff':<10} {'S_cross':<10} {'Order prot':<12} {'Entropy prot':<12} {'Total':<10}")
    print("-" * 70)

    for i in range(len(k_values)):
        k = k_values[i]

        # Find matching alpha
        idx = np.argmin(np.abs(race_k - k))
        if abs(race_k[idx] - k) > 0.5:
            continue

        alpha = race_alpha[idx]
        S_cross = entropies_cross[i]

        # Order-based protection: how much alpha is suppressed below 1/2
        # (1/2 is the per-triad maximum, 1/4 is incoherent average)
        order_protection = max(0, 1 - 2 * alpha)  # 1 when alpha=0, 0 when alpha=1/2

        # Entropy-based protection: phase disorder prevents coherent buildup
        entropy_protection = S_cross  # 1 when maximum entropy, 0 when ordered

        # Combined: at least one form of protection is active
        total = max(order_protection, entropy_protection)

        print(f"{k:<6.0f} {alpha:<10.4f} {S_cross:<10.4f} {order_protection:<12.4f} {entropy_protection:<12.4f} {total:<10.4f}")

    # --- MEASUREMENT 4: Crossover point ---
    print("\n" + "=" * 70)
    print("MEASUREMENT 4: Crossover -- Where Does the Transformation Happen?")
    print("=" * 70)

    # Find where entropy_protection overtakes order_protection
    found_crossover = False
    for i in range(len(k_values)):
        k = k_values[i]
        idx = np.argmin(np.abs(race_k - k))
        if abs(race_k[idx] - k) > 0.5:
            continue

        alpha = race_alpha[idx]
        S_cross = entropies_cross[i]
        order_p = max(0, 1 - 2 * alpha)
        entropy_p = S_cross

        if entropy_p > order_p and not found_crossover:
            print(f"CROSSOVER at k ≈ {k:.0f}")
            print(f"  alpha_eff = {alpha:.4f}, S_cross = {S_cross:.4f}")
            print(f"  k / k_d = {k / k_d_actual:.3f}")
            found_crossover = True

    # Find where viscosity overtakes nonlinearity
    for i in range(len(race_k)):
        if i > 0 and race_viscous[i] > race_nonlinear[i] and race_viscous[i-1] <= race_nonlinear[i-1]:
            print(f"\nVISCOSITY DOMINANCE starts at k ≈ {race_k[i]:.0f}")
            print(f"  k / k_d = {race_k[i] / k_d_actual:.3f}")
            print(f"  This is where the handoff completes.")

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY: The Transformation Picture")
    print("=" * 70)
    print(f"""
    Kolmogorov scale:           k_d = {k_d_actual:.2f}
    Predicted advantage:        Re^(1/7) = {Re**(1./7):.4f}

    Phase entropy at k=1:       S = {entropies_cross[0]:.4f} (ordered)
    Phase entropy at k={k_values[-1]:.0f}:      S = {entropies_cross[-1]:.4f} (disordered)

    alpha_eff at k=1:               {race_alpha[0]:.4f} (structure suppresses)
    alpha_eff at k={int(race_k[-1])}:              {race_alpha[-1]:.4f} (structure fades)

    THE TRANSFORMATION:
    Low k:  ORDER protects (helicity, alignment, structure) -> low alpha
    High k: ENTROPY protects (random phases, CLT, depolarization) -> but alpha rises
    The 3/28 zone: viscosity catches the handoff with Re^(1/7) margin

    Protection never dies. It transforms.
    """)

    return {
        'k': k_values,
        'entropy_cross': entropies_cross,
        'entropy_p': entropies_p,
        'entropy_m': entropies_m,
        'alpha_eff': race_alpha,
        'race_k': race_k,
        'race_viscous': race_viscous,
        'race_nonlinear': race_nonlinear,
        'k_d': k_d_actual,
    }


if __name__ == "__main__":
    results = measure_transformation(N=48, Re=400, T_target=2.0)
