"""
PREDICTION 1 — DEFINITIVE TEST: α(θ, ρ=1) = (1−cosθ)/(3−cosθ)
================================================================
Restrict to SAME-SHELL triads (|k1| ≈ |k2|, i.e. ρ ≈ 1).
This eliminates ρ-dependence and tests the pure θ-formula.

Also measure α(θ,ρ) for several ρ bins to extract the general formula.

Meridian 2, S100.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def measure_triad_alpha(solver, k1_idx, k2_idx):
    """Compute Leray ratio α for a single triad.

    Returns (theta, rho, alpha_pm, alpha_mp) or None if invalid.
    """
    i1, j1, l1 = k1_idx
    i2, j2, l2 = k2_idx
    N = solver.N

    k1_vec = np.array([solver.kx[i1,j1,l1], solver.ky[i1,j1,l1], solver.kz[i1,j1,l1]], dtype=float)
    k2_vec = np.array([solver.kx[i2,j2,l2], solver.ky[i2,j2,l2], solver.kz[i2,j2,l2]], dtype=float)
    k1_mag = np.sqrt(solver.k2[i1,j1,l1])
    k2_mag = np.sqrt(solver.k2[i2,j2,l2])

    if k1_mag < 0.5 or k2_mag < 0.5:
        return None

    # k3 = k1 + k2
    k3_vec = k1_vec + k2_vec
    k3_mag_sq = np.dot(k3_vec, k3_vec)
    if k3_mag_sq < 0.5:
        return None
    k3_mag = np.sqrt(k3_mag_sq)

    kmax = N // 3
    if k3_mag > kmax + 0.5:
        return None

    # Map k3 to grid
    i3 = int(round(k3_vec[0])) % N
    j3 = int(round(k3_vec[1])) % N
    l3 = int(round(k3_vec[2])) % N
    k3_actual = np.array([solver.kx[i3,j3,l3], solver.ky[i3,j3,l3], solver.kz[i3,j3,l3]], dtype=float)
    if np.linalg.norm(k3_actual - k3_vec) > 0.1:
        return None

    # Angles and ratio
    cos_theta = np.clip(np.dot(k1_vec, k2_vec) / (k1_mag * k2_mag), -1, 1)
    theta = np.arccos(cos_theta)
    rho = k1_mag / k2_mag

    # Helical basis vectors
    ep1 = np.array([solver.h_plus[c, i1, j1, l1] for c in range(3)])
    em1 = np.array([solver.h_minus[c, i1, j1, l1] for c in range(3)])
    ep2 = np.array([solver.h_plus[c, i2, j2, l2] for c in range(3)])
    em2 = np.array([solver.h_minus[c, i2, j2, l2] for c in range(3)])

    k3_hat = k3_actual / np.sqrt(np.dot(k3_actual, k3_actual))

    # (+,-) coupling
    lamb_pm = np.cross(ep1, em2)
    pw_pm = np.sum(np.abs(lamb_pm)**2)
    if pw_pm < 1e-30:
        return None
    lamb_pm_sol = lamb_pm - k3_hat * np.dot(k3_hat, lamb_pm)
    alpha_pm = np.sum(np.abs(lamb_pm_sol)**2) / pw_pm

    # (-,+) coupling
    lamb_mp = np.cross(em1, ep2)
    pw_mp = np.sum(np.abs(lamb_mp)**2)
    alpha_mp = np.sum(np.abs(lamb_mp - k3_hat * np.dot(k3_hat, lamb_mp))**2) / pw_mp if pw_mp > 1e-30 else np.nan

    return theta, rho, alpha_pm, alpha_mp


def main():
    print("=" * 70)
    print("  PREDICTION 1 — DEFINITIVE ρ=1 TEST")
    print("  α(θ, ρ=1) should equal (1−cosθ)/(3−cosθ)")
    print("=" * 70)

    N = 48
    solver = SpectralNS(N=N, Re=100)

    kmax = N // 3
    active = np.argwhere((solver.k2 > 0) & (solver.kmag < kmax + 0.5))

    # Build shell lookup: shell_idx -> list of grid indices
    shell_modes = {}
    for idx in active:
        i, j, l = idx
        k_mag = np.sqrt(solver.k2[i, j, l])
        shell = int(round(k_mag))
        if shell not in shell_modes:
            shell_modes[shell] = []
        shell_modes[shell].append((i, j, l))

    print(f"\n  Grid N={N}, kmax={kmax}")
    print(f"  Active modes: {len(active)}")
    print(f"  Shells with modes: {sorted(shell_modes.keys())[:10]}...")

    rng = np.random.default_rng(42)

    # ================================================================
    # TEST 1: Same-shell triads (ρ = 1 exactly)
    # ================================================================
    print("\n  TEST 1: Same-shell triads (ρ = 1 exactly)")

    thetas_rho1 = []
    alphas_rho1 = []

    n_target = 30000
    n_found = 0
    attempts = 0

    shells_list = [s for s in shell_modes.keys() if 2 <= s <= kmax and len(shell_modes[s]) >= 10]

    while n_found < n_target and attempts < n_target * 50:
        attempts += 1
        shell = rng.choice(shells_list)
        modes = shell_modes[shell]

        idx1 = modes[rng.integers(len(modes))]
        idx2 = modes[rng.integers(len(modes))]

        result = measure_triad_alpha(solver, idx1, idx2)
        if result is None:
            continue

        theta, rho, alpha_pm, alpha_mp = result
        # ρ should be exactly 1 since same shell... but check
        if abs(rho - 1.0) > 0.01:
            continue

        thetas_rho1.append(theta)
        alphas_rho1.append(alpha_pm)
        n_found += 1

    thetas_rho1 = np.array(thetas_rho1)
    alphas_rho1 = np.array(alphas_rho1)
    print(f"  Found {n_found} same-shell triads (from {attempts} attempts)")

    # Bin by theta
    n_bins = 30
    theta_bins = np.linspace(0, np.pi, n_bins + 1)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    alpha_binned = np.zeros(n_bins)
    alpha_std = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (thetas_rho1 >= theta_bins[i]) & (thetas_rho1 < theta_bins[i+1])
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            alpha_binned[i] = np.mean(alphas_rho1[mask])
            alpha_std[i] = np.std(alphas_rho1[mask])

    c = np.cos(theta_centers)
    predicted = (1 - c) / (3 - c)
    sin2_4 = np.sin(theta_centers)**2 / 4

    print(f"\n  {'θ/π':>6}  {'α_meas':>10}  {'(1-c)/(3-c)':>12}  {'sin²θ/4':>10}  {'ratio':>8}  {'std':>8}  {'n':>6}")
    print("  " + "-" * 72)

    valid = counts > 5
    for i in range(n_bins):
        if valid[i]:
            ratio = alpha_binned[i] / predicted[i] if predicted[i] > 1e-10 else float('nan')
            print(f"  {theta_centers[i]/np.pi:>6.3f}  {alpha_binned[i]:>10.6f}  "
                  f"{predicted[i]:>12.6f}  {sin2_4[i]:>10.6f}  "
                  f"{ratio:>8.4f}  {alpha_std[i]:>8.4f}  {counts[i]:>6d}")

    # RMS against both formulas
    if np.any(valid):
        rms_ratio = np.sqrt(np.mean((alpha_binned[valid] - predicted[valid])**2))
        rms_sin2 = np.sqrt(np.mean((alpha_binned[valid] - sin2_4[valid])**2))
        mean_ratio = np.nanmean(alpha_binned[valid] / np.maximum(predicted[valid], 1e-10))

        print(f"\n  ρ=1 results:")
        print(f"    RMS vs (1−cosθ)/(3−cosθ): {rms_ratio:.8f}")
        print(f"    RMS vs sin²θ/4:           {rms_sin2:.8f}")
        print(f"    Mean ratio to (1−c)/(3−c): {mean_ratio:.6f}")

        if rms_ratio < 0.001:
            print(f"\n  >>> CONFIRMED: α(θ,ρ=1) = (1−cosθ)/(3−cosθ) at precision {rms_ratio:.2e} <<<")
        elif rms_ratio < 0.01:
            print(f"\n  >>> Good match to (1−cosθ)/(3−cosθ). RMS = {rms_ratio:.6f}")
        else:
            print(f"\n  >>> RMS = {rms_ratio:.6f}. Still not matching. Investigate scatter.")

    # ================================================================
    # TEST 2: ρ-dependence — bin by (θ, ρ) for general triads
    # ================================================================
    print("\n\n  TEST 2: General triads — binned by (θ, ρ)")

    thetas_all = []
    rhos_all = []
    alphas_all = []

    n_target2 = 50000
    n_found2 = 0
    attempts2 = 0

    while n_found2 < n_target2 and attempts2 < n_target2 * 20:
        attempts2 += 1
        idx1 = active[rng.integers(len(active))]
        idx2 = active[rng.integers(len(active))]

        result = measure_triad_alpha(solver, tuple(idx1), tuple(idx2))
        if result is None:
            continue

        theta, rho, alpha_pm, _ = result
        thetas_all.append(theta)
        rhos_all.append(rho)
        alphas_all.append(alpha_pm)
        n_found2 += 1

    thetas_all = np.array(thetas_all)
    rhos_all = np.array(rhos_all)
    alphas_all = np.array(alphas_all)
    print(f"  Found {n_found2} triads")

    # Bin by ρ and check θ-dependence for each ρ bin
    rho_edges = [0.5, 0.8, 0.95, 1.05, 1.25, 2.0, 4.0]
    rho_labels = ['ρ∈[0.5,0.8)', 'ρ∈[0.8,0.95)', 'ρ≈1', 'ρ∈[1.05,1.25)', 'ρ∈[1.25,2)', 'ρ∈[2,4)']

    print(f"\n  RMS vs (1−cosθ)/(3−cosθ) by ρ bin:")
    rms_by_rho = {}
    for rb in range(len(rho_edges) - 1):
        rho_mask = (rhos_all >= rho_edges[rb]) & (rhos_all < rho_edges[rb + 1])
        n_rho = np.sum(rho_mask)
        if n_rho < 100:
            print(f"    {rho_labels[rb]:20s}: too few ({n_rho})")
            continue

        # Bin by theta within this ρ range
        t_sub = thetas_all[rho_mask]
        a_sub = alphas_all[rho_mask]
        a_binned = np.zeros(n_bins)
        c_counts = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            m = (t_sub >= theta_bins[i]) & (t_sub < theta_bins[i+1])
            c_counts[i] = np.sum(m)
            if c_counts[i] > 0:
                a_binned[i] = np.mean(a_sub[m])

        v = c_counts > 3
        if np.sum(v) > 3:
            rms = np.sqrt(np.mean((a_binned[v] - predicted[v])**2))
            rms_by_rho[rho_labels[rb]] = rms
            print(f"    {rho_labels[rb]:20s}: RMS = {rms:.6f}  (n={n_rho})")

    # ================================================================
    # TEST 3: Check ABSOLUTE power |P[e⁺×e⁻]|² vs sin²θ/4
    # ================================================================
    print("\n\n  TEST 3: Absolute power |P[e⁺×e⁻]|² vs sin²θ/4 (ρ=1)")

    abs_powers = []
    abs_thetas = []

    n_found3 = 0
    attempts3 = 0

    while n_found3 < 20000 and attempts3 < 20000 * 50:
        attempts3 += 1
        shell = rng.choice(shells_list)
        modes = shell_modes[shell]

        i1, j1, l1 = modes[rng.integers(len(modes))]
        i2, j2, l2 = modes[rng.integers(len(modes))]

        k1_vec = np.array([solver.kx[i1,j1,l1], solver.ky[i1,j1,l1], solver.kz[i1,j1,l1]], dtype=float)
        k2_vec = np.array([solver.kx[i2,j2,l2], solver.ky[i2,j2,l2], solver.kz[i2,j2,l2]], dtype=float)
        k1_mag = np.sqrt(solver.k2[i1,j1,l1])
        k2_mag = np.sqrt(solver.k2[i2,j2,l2])

        if k1_mag < 0.5 or k2_mag < 0.5 or abs(k1_mag/k2_mag - 1) > 0.01:
            continue

        k3_vec = k1_vec + k2_vec
        k3_mag_sq = np.dot(k3_vec, k3_vec)
        if k3_mag_sq < 0.5 or np.sqrt(k3_mag_sq) > kmax + 0.5:
            continue

        i3 = int(round(k3_vec[0])) % N
        j3 = int(round(k3_vec[1])) % N
        l3 = int(round(k3_vec[2])) % N
        k3_actual = np.array([solver.kx[i3,j3,l3], solver.ky[i3,j3,l3], solver.kz[i3,j3,l3]], dtype=float)
        if np.linalg.norm(k3_actual - k3_vec) > 0.1:
            continue

        cos_theta = np.clip(np.dot(k1_vec, k2_vec) / (k1_mag * k2_mag), -1, 1)
        theta = np.arccos(cos_theta)

        ep1 = np.array([solver.h_plus[c, i1, j1, l1] for c in range(3)])
        em2 = np.array([solver.h_minus[c, i2, j2, l2] for c in range(3)])

        k3_hat = k3_actual / np.sqrt(np.dot(k3_actual, k3_actual))

        lamb = np.cross(ep1, em2)
        lamb_sol = lamb - k3_hat * np.dot(k3_hat, lamb)
        abs_sol_power = np.sum(np.abs(lamb_sol)**2)

        abs_thetas.append(theta)
        abs_powers.append(abs_sol_power)
        n_found3 += 1

    abs_thetas = np.array(abs_thetas)
    abs_powers = np.array(abs_powers)
    print(f"  Found {n_found3} same-shell triads")

    # Bin
    abs_binned = np.zeros(n_bins)
    abs_counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (abs_thetas >= theta_bins[i]) & (abs_thetas < theta_bins[i+1])
        abs_counts[i] = np.sum(mask)
        if abs_counts[i] > 0:
            abs_binned[i] = np.mean(abs_powers[mask])

    v3 = abs_counts > 5
    if np.any(v3):
        rms_abs = np.sqrt(np.mean((abs_binned[v3] - sin2_4[v3])**2))
        mean_ratio_abs = np.nanmean(abs_binned[v3] / np.maximum(sin2_4[v3], 1e-10))

        print(f"\n  {'θ/π':>6}  {'|P[e⁺×e⁻]|²':>12}  {'sin²θ/4':>10}  {'ratio':>8}  {'n':>6}")
        print("  " + "-" * 54)
        for i in range(n_bins):
            if v3[i]:
                r = abs_binned[i] / sin2_4[i] if sin2_4[i] > 1e-10 else float('nan')
                print(f"  {theta_centers[i]/np.pi:>6.3f}  {abs_binned[i]:>12.6f}  "
                      f"{sin2_4[i]:>10.6f}  {r:>8.4f}  {abs_counts[i]:>6d}")

        print(f"\n  Absolute power results:")
        print(f"    RMS vs sin²θ/4: {rms_abs:.8f}")
        print(f"    Mean ratio:     {mean_ratio_abs:.6f}")

        if rms_abs < 0.001:
            print(f"\n  >>> CONFIRMED: |P[e⁺×e⁻]|² = sin²θ/4 at precision {rms_abs:.2e} <<<")
        else:
            # Also check |h⁺×h⁻|² = (1+cosθ)(3-cosθ)/4
            total_binned = np.zeros(n_bins)
            total_count = np.zeros(n_bins, dtype=int)
            # Re-measure total power
            total_powers = []
            for i_t, theta_val in enumerate(abs_thetas):
                pass  # We didn't save total powers, compute from alphas
            print(f"    (sin²θ/4 is absolute power, not ratio — investigating)")

    # ================================================================
    # PLOT
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Prediction 1 — Definitive ρ=1 Test', fontsize=14, fontweight='bold')

    theta_fine = np.linspace(0.01, np.pi - 0.01, 200)
    c_fine = np.cos(theta_fine)

    # Panel 1: ρ=1 ratio vs (1-cosθ)/(3-cosθ)
    ax = axes[0, 0]
    v1 = counts > 5
    ax.errorbar(theta_centers[v1]/np.pi, alpha_binned[v1], yerr=alpha_std[v1]/np.sqrt(counts[v1]),
                fmt='o', color='#364FC7', markersize=4, capsize=2, label='Measured (ρ=1)')
    ax.plot(theta_fine/np.pi, (1-c_fine)/(3-c_fine), 'r-', linewidth=2, label='(1−cosθ)/(3−cosθ)')
    ax.plot(theta_fine/np.pi, np.sin(theta_fine)**2/4, 'g--', linewidth=1, alpha=0.5, label='sin²θ/4')
    ax.set_xlabel('θ/π')
    ax.set_ylabel('α = Leray ratio')
    ax.set_title('Same-shell triads (ρ=1)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel 2: Ratio to (1-cosθ)/(3-cosθ)
    ax = axes[0, 1]
    r1 = alpha_binned[v1] / np.maximum(predicted[v1], 1e-10)
    ax.plot(theta_centers[v1]/np.pi, r1, 'o-', color='#364FC7', markersize=3)
    ax.axhline(1.0, color='red', linestyle='--', label='Exact match')
    ax.fill_between([0, 1], 0.99, 1.01, color='red', alpha=0.1)
    ax.set_xlabel('θ/π')
    ax.set_ylabel('Measured / (1−cosθ)/(3−cosθ)')
    ax.set_title('Ratio to prediction (ρ=1)')
    ax.legend()
    ax.set_ylim(0.8, 1.2)

    # Panel 3: Absolute power vs sin²θ/4
    ax = axes[1, 0]
    v3 = abs_counts > 5
    ax.plot(theta_centers[v3]/np.pi, abs_binned[v3], 'o', color='#FF9800', markersize=4, label='|P[e⁺×e⁻]|²')
    ax.plot(theta_fine/np.pi, np.sin(theta_fine)**2/4, 'r-', linewidth=2, label='sin²θ/4')
    ax.set_xlabel('θ/π')
    ax.set_ylabel('|P[e⁺×e⁻]|²')
    ax.set_title('Absolute solenoidal power (ρ=1)')
    ax.legend(fontsize=8)

    # Panel 4: Scatter of all triads colored by ρ
    ax = axes[1, 1]
    sc = ax.scatter(thetas_all[:5000]/np.pi, alphas_all[:5000], c=rhos_all[:5000],
                    cmap='RdYlBu_r', s=1, alpha=0.3, vmin=0.5, vmax=3)
    ax.plot(theta_fine/np.pi, (1-c_fine)/(3-c_fine), 'r-', linewidth=2, label='(1−c)/(3−c) [ρ=1]')
    fig.colorbar(sc, ax=ax, label='ρ = |k₁|/|k₂|')
    ax.set_xlabel('θ/π')
    ax.set_ylabel('α')
    ax.set_title('All triads colored by ρ')
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'verify_prediction1_rho1.png')
    plt.savefig(outpath, dpi=150)
    print(f"\n  Plot saved to {outpath}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
