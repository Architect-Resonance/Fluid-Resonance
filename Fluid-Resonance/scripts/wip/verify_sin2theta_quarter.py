"""
PREDICTION 1 VERIFICATION: Per-triad solenoidal fraction = sin²θ/4
=====================================================================
PURE GEOMETRY TEST — no flow amplitudes, no statistics.

For a cross-helical triad (k1, k2, k3 = k1+k2):
  The helical coupling involves e⁺(k1) × e⁻(k2) (or with vorticity: ik × e).
  After Leray projection at k3:
    α(θ) = |P_{k3}[e⁺(k1) × (ik2 × e⁻(k2))]|² / |e⁺(k1) × (ik2 × e⁻(k2))|²
  should equal sin²θ₁₂/4 where θ₁₂ = angle between k1 and k2.

This is a property of the helical basis and Leray projector, independent of flow.

Also verify shell-averaged Lamb ratio in a developed flow to check the
isotropic average (geometric: 1−ln2, Lamb-weighted: 1/4).

Meridian 2, S100.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def verify_geometric(N=32, n_triads=50000):
    """Pure geometry: verify sin²θ/4 using helical basis vectors only.

    For random triad geometries (k1, k2, k3=k1+k2), compute:
      α = |P_{k3}[e⁺(k1) × (ik2 × e⁻(k2))]|² / |e⁺(k1) × (ik2 × e⁻(k2))|²
    """
    solver = SpectralNS(N=N, Re=100)  # Re irrelevant for geometry

    thetas = []
    alphas_pm = []  # (+,-) coupling
    alphas_mp = []  # (-,+) coupling

    rng = np.random.default_rng(42)

    # Sample random triads
    kmax = N // 3
    active = np.argwhere((solver.k2 > 0) & (solver.kmag < kmax + 0.5))

    count = 0
    attempts = 0
    max_attempts = n_triads * 20

    while count < n_triads and attempts < max_attempts:
        attempts += 1

        # Random k1
        idx1 = rng.integers(len(active))
        i1, j1, l1 = active[idx1]
        k1_vec = np.array([solver.kx[i1, j1, l1],
                           solver.ky[i1, j1, l1],
                           solver.kz[i1, j1, l1]], dtype=float)
        k1_mag = np.sqrt(solver.k2[i1, j1, l1])

        # Random k2
        idx2 = rng.integers(len(active))
        i2, j2, l2 = active[idx2]
        k2_vec = np.array([solver.kx[i2, j2, l2],
                           solver.ky[i2, j2, l2],
                           solver.kz[i2, j2, l2]], dtype=float)
        k2_mag = np.sqrt(solver.k2[i2, j2, l2])

        if k1_mag < 0.5 or k2_mag < 0.5:
            continue

        # k3 = k1 + k2
        k3_vec = k1_vec + k2_vec
        k3_mag = np.linalg.norm(k3_vec)
        if k3_mag < 0.5 or k3_mag > kmax + 0.5:
            continue

        # Map k3 to grid
        i3 = int(round(k3_vec[0])) % N
        j3 = int(round(k3_vec[1])) % N
        l3 = int(round(k3_vec[2])) % N

        k3_actual = np.array([solver.kx[i3, j3, l3],
                              solver.ky[i3, j3, l3],
                              solver.kz[i3, j3, l3]], dtype=float)
        if np.linalg.norm(k3_actual - k3_vec) > 0.1:
            continue
        k3_mag_sq = solver.k2[i3, j3, l3]
        if k3_mag_sq < 0.5:
            continue

        # Angle between k1 and k2
        cos_theta = np.clip(np.dot(k1_vec, k2_vec) / (k1_mag * k2_mag), -1, 1)
        theta = np.arccos(cos_theta)

        # Helical basis vectors at k1, k2
        ep1 = np.array([solver.h_plus[c, i1, j1, l1] for c in range(3)])   # e⁺(k1)
        em1 = np.array([solver.h_minus[c, i1, j1, l1] for c in range(3)])  # e⁻(k1)
        ep2 = np.array([solver.h_plus[c, i2, j2, l2] for c in range(3)])   # e⁺(k2)
        em2 = np.array([solver.h_minus[c, i2, j2, l2] for c in range(3)])  # e⁻(k2)

        # Vorticity basis: ω±(k) = ik × u±(k).
        # Since ik × e±(k) = ±|k| e±(k), we have:
        #   ω⁺(k2) = +|k2| e⁺(k2)  (times i, absorbed in the coupling)
        #   ω⁻(k2) = -|k2| e⁻(k2)
        # Actually: ik × e±(k) = ±|k| e±(k)
        # So the Lamb contribution u⁺(k1) × ω⁻(k2) = a1 * (-|k2| a2) * [e⁺(k1) × e⁻(k2)]
        # The solenoidal fraction depends only on the DIRECTION of e⁺(k1) × e⁻(k2)
        # relative to k3, not on the scalar amplitudes. So we just need:

        # Cross-helical Lamb direction: e⁺(k1) × e⁻(k2)
        lamb_pm = np.cross(ep1, em2)  # complex 3-vector
        power_pm = np.sum(np.abs(lamb_pm)**2)

        if power_pm < 1e-30:
            continue

        # Leray projection at k3
        k3_hat = k3_actual / np.sqrt(k3_mag_sq)
        lamb_pm_sol = lamb_pm - k3_hat * np.dot(k3_hat, lamb_pm)
        power_pm_sol = np.sum(np.abs(lamb_pm_sol)**2)

        alpha_pm = power_pm_sol / power_pm

        # Also (-,+) coupling: e⁻(k1) × e⁺(k2)
        lamb_mp = np.cross(em1, ep2)
        power_mp = np.sum(np.abs(lamb_mp)**2)
        if power_mp > 1e-30:
            lamb_mp_sol = lamb_mp - k3_hat * np.dot(k3_hat, lamb_mp)
            power_mp_sol = np.sum(np.abs(lamb_mp_sol)**2)
            alpha_mp = power_mp_sol / power_mp
        else:
            alpha_mp = float('nan')

        thetas.append(theta)
        alphas_pm.append(alpha_pm)
        alphas_mp.append(alpha_mp)
        count += 1

    return np.array(thetas), np.array(alphas_pm), np.array(alphas_mp)


def verify_shell_averaged(Re=800, N=48):
    """Verify shell-averaged α in a developed flow."""
    solver = SpectralNS(N=N, Re=Re)
    u_hat = solver.taylor_green_ic()

    # Evolve
    dt = 0.001
    steps = 1000
    print(f"  Evolving TG at Re={Re}, N={N} to t=1.0...")
    for step in range(steps):
        u_hat = solver.step_rk4(u_hat, dt)
    print("  Done.")

    # Compute cross-helical Lamb and its Leray projection
    lamb_cross = solver.compute_lamb_hat_cross_only(u_hat)
    lamb_cross_sol = solver.project_leray(lamb_cross)

    power_total = np.sum(np.abs(lamb_cross)**2, axis=0)
    power_sol = np.sum(np.abs(lamb_cross_sol)**2, axis=0)

    kmax = N // 3
    shells = np.arange(1, kmax + 1)
    alpha_shell = np.zeros(kmax)

    for i, ks in enumerate(shells):
        mask = (solver.kmag >= ks - 0.5) & (solver.kmag < ks + 0.5)
        pt = np.sum(power_total[mask])
        ps = np.sum(power_sol[mask])
        if pt > 1e-30:
            alpha_shell[i] = ps / pt

    # Global ratio
    global_alpha = np.sum(power_sol) / np.sum(power_total)

    return shells, alpha_shell, global_alpha


def main():
    print("=" * 70)
    print("  PREDICTION 1: Per-triad solenoidal fraction = sin²θ/4")
    print("  Part A: Pure geometry (helical basis only)")
    print("  Part B: Shell-averaged in developed flow")
    print("=" * 70)

    # ================================================================
    # PART A: Pure geometric verification
    # ================================================================
    print("\n  PART A: Pure geometric verification (50,000 random triads)...")
    thetas, alphas_pm, alphas_mp = verify_geometric(N=32, n_triads=50000)
    print(f"  {len(thetas)} triads verified.")

    # Bin by theta
    n_bins = 40
    theta_bins = np.linspace(0, np.pi, n_bins + 1)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    alpha_pm_binned = np.zeros(n_bins)
    alpha_mp_binned = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (thetas >= theta_bins[i]) & (thetas < theta_bins[i+1])
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            alpha_pm_binned[i] = np.mean(alphas_pm[mask])
            alpha_mp_binned[i] = np.nanmean(alphas_mp[mask])

    sin2_quarter = np.sin(theta_centers)**2 / 4

    print("\n  Per-triad α(θ) vs sin²θ/4:")
    print(f"  {'θ/π':>6}  {'α(+,-)':>10}  {'α(-,+)':>10}  {'sin²θ/4':>10}  "
          f"{'ratio+':>8}  {'ratio-':>8}  {'n':>6}")
    print("  " + "-" * 68)

    for i in range(n_bins):
        if counts[i] > 10:
            r_pm = alpha_pm_binned[i] / sin2_quarter[i] if sin2_quarter[i] > 1e-10 else float('nan')
            r_mp = alpha_mp_binned[i] / sin2_quarter[i] if sin2_quarter[i] > 1e-10 else float('nan')
            print(f"  {theta_centers[i]/np.pi:>6.3f}  {alpha_pm_binned[i]:>10.6f}  "
                  f"{alpha_mp_binned[i]:>10.6f}  {sin2_quarter[i]:>10.6f}  "
                  f"{r_pm:>8.4f}  {r_mp:>8.4f}  {counts[i]:>6d}")

    # Match quality
    valid = counts > 10
    if np.any(valid):
        residuals_pm = alpha_pm_binned[valid] - sin2_quarter[valid]
        rms_pm = np.sqrt(np.mean(residuals_pm**2))
        mean_ratio_pm = np.mean(alpha_pm_binned[valid] / np.maximum(sin2_quarter[valid], 1e-10))

        print(f"\n  (+,-) RMS error: {rms_pm:.8f}")
        print(f"  (+,-) Mean ratio: {mean_ratio_pm:.6f}")

        if rms_pm < 0.001:
            print("\n  >>> PREDICTION 1 CONFIRMED at machine precision <<<")
        elif rms_pm < 0.01:
            print("\n  >>> PREDICTION 1 CONFIRMED (good match) <<<")
        elif rms_pm < 0.05:
            print("\n  >>> Approximate match — check formula interpretation <<<")
        else:
            print(f"\n  >>> Does NOT match sin²θ/4. RMS = {rms_pm:.4f}")
            print("  >>> The per-triad formula may need re-derivation.")

    # Check: maybe it's sin²θ/2 or (1-cos²θ)/4 with different convention?
    # Also check 1 - cos²θ_k3 (angle between Lamb and k3)
    # And check if it matches some OTHER function of θ
    print("\n  Checking alternative formulas...")
    c = np.cos(theta_centers)
    for label, formula in [
        ("sin²θ/4", np.sin(theta_centers)**2 / 4),
        ("sin²θ/2", np.sin(theta_centers)**2 / 2),
        ("sin²θ", np.sin(theta_centers)**2),
        ("(1-cosθ)/(3-cosθ) [α(θ,ρ=1)]", (1 - c) / (3 - c)),
        ("(1-cos θ)/4", (1 - c) / 4),
        ("(1-cos θ)/2", (1 - c) / 2),
        ("(1-cos²θ)/4", (1 - c**2) / 4),
        ("sin²θ/[(1+cosθ)(3-cosθ)]", np.where(
            np.abs((1 + c) * (3 - c)) > 1e-10,
            np.sin(theta_centers)**2 / ((1 + c) * (3 - c)),
            0.5)),
    ]:
        if np.any(valid):
            rms = np.sqrt(np.mean((alpha_pm_binned[valid] - formula[valid])**2))
            print(f"    {label:30s}: RMS = {rms:.6f}")

    # ================================================================
    # PART B: Shell-averaged verification
    # ================================================================
    print("\n  PART B: Shell-averaged in developed flow...")
    shells, alpha_shell, global_alpha = verify_shell_averaged()

    print(f"\n  Global α = {global_alpha:.6f}")
    print(f"  Expected (1-ln2): {1 - np.log(2):.6f}")
    print(f"  Expected (1/4):   {0.25:.6f}")

    print("\n  Per-shell:")
    for i in range(min(16, len(shells))):
        print(f"    k={shells[i]:2d}: α = {alpha_shell[i]:.6f}")

    # ================================================================
    # PLOT
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Prediction 1: Per-triad solenoidal fraction', fontsize=14, fontweight='bold')

    # Panel 1: Geometric α(θ) vs sin²θ/4
    ax = axes[0]
    valid_mask = counts > 10
    ax.plot(theta_centers[valid_mask] / np.pi, alpha_pm_binned[valid_mask],
            'o', color='#364FC7', markersize=4, label='Measured (+,−)')
    ax.plot(theta_centers[valid_mask] / np.pi, alpha_mp_binned[valid_mask],
            's', color='#2196F3', markersize=3, alpha=0.5, label='Measured (−,+)')
    theta_fine = np.linspace(0.01, np.pi - 0.01, 200)
    ax.plot(theta_fine / np.pi, np.sin(theta_fine)**2 / 4,
            'r-', linewidth=2, label='sin²θ/4')
    ax.plot(theta_fine / np.pi, np.sin(theta_fine)**2 / 2,
            'g--', linewidth=1, alpha=0.5, label='sin²θ/2')
    ax.set_xlabel('θ/π')
    ax.set_ylabel('α = |P[e⁺×e⁻]|² / |e⁺×e⁻|²')
    ax.set_title('Pure geometry: per-triad')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Panel 2: Ratio to sin²θ/4
    ax = axes[1]
    ratio_pm = alpha_pm_binned[valid_mask] / np.maximum(sin2_quarter[valid_mask], 1e-10)
    ax.plot(theta_centers[valid_mask] / np.pi, ratio_pm, 'o-', color='#364FC7', markersize=3)
    ax.axhline(1.0, color='red', linestyle='--', label='Exact match')
    ax.set_xlabel('θ/π')
    ax.set_ylabel('Measured / sin²θ/4')
    ax.set_title('Ratio to prediction')
    ax.legend()

    # Panel 3: Shell-averaged
    ax = axes[2]
    ax.plot(shells, alpha_shell, 'o-', color='#364FC7', markersize=4, label='Measured')
    ax.axhline(0.25, color='red', linestyle='--', label='1/4')
    ax.axhline(1 - np.log(2), color='orange', linestyle=':', label=f'1-ln(2)')
    ax.set_xlabel('k')
    ax.set_ylabel('α_shell')
    ax.set_title('Shell-averaged (developed flow)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'verify_sin2theta_quarter.png')
    plt.savefig(outpath, dpi=150)
    print(f"\n  Plot saved to {outpath}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
