"""
ENERGY-WEIGHTED LERAY SUPPRESSION FACTOR
=========================================
Meridian 2 — S93 follow-up

The Wanderer (S93) showed that the GEOMETRIC average of the Leray suppression
factor alpha_+- is ~0.40 (60% of cross-helical Lamb is gradient).

But the geometric average treats all triads equally. In actual turbulence,
triads carry very different amounts of energy. The critical question:

  Do near-antiparallel triads (k1 ~ -k2, where alpha -> 1) carry enough
  energy to push the ENERGY-WEIGHTED average above the geometric average?

This script computes:
1. alpha_E = ||P_sol(L_sector)||^2 / ||L_sector||^2 for cross/same/full
   (the ACTUAL energy-weighted suppression — no triad enumeration needed)
2. Scale-resolved alpha(|k|) — does suppression vary with wavenumber?
3. Angular distribution of triad energy — where does the energy concentrate?
4. Time evolution through peak enstrophy — does alpha_E change as flow develops?

If alpha_E_cross stays well below 1.0 even at peak enstrophy, the geometric
suppression survives energy weighting. If it approaches 1.0, the near-antiparallel
triads dominate and the bound fails.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import sys
import os
import json

# Import shared infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS
from leray_suppression_geometry import compute_alpha_single_triad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class EnergyWeightedLeray(SpectralNS):
    """Measure the actual energy-weighted Leray suppression in evolving NS flows."""

    def compute_sector_alpha(self, u_hat):
        """Compute energy-weighted alpha for full, cross-helical, and same-helical sectors.

        alpha_sector = ||P_sol(L_sector)||^2 / ||L_sector||^2

        This IS the energy-weighted average — each triad contributes proportionally
        to how much Lamb vector it produces.
        """
        def norm_sq(f_hat):
            return float(np.sum(np.abs(f_hat)**2))

        # Full Lamb vector and its solenoidal projection
        L_hat = self.compute_lamb_hat(u_hat)
        L_sol_hat = self.project_leray(L_hat)

        # Cross-helical Lamb (omega+ x v- + omega- x v+)
        L_cross_hat = self.compute_lamb_hat_cross_only(u_hat)
        L_cross_sol_hat = self.project_leray(L_cross_hat)

        # Same-helical Lamb (omega+ x v+ + omega- x v-)
        L_same_hat = self.compute_lamb_hat_bt_surgery(u_hat)
        L_same_sol_hat = self.project_leray(L_same_hat)

        # Energy-weighted alphas
        nL = norm_sq(L_hat)
        nLc = norm_sq(L_cross_hat)
        nLs = norm_sq(L_same_hat)

        alpha_full = norm_sq(L_sol_hat) / max(nL, 1e-30)
        alpha_cross = norm_sq(L_cross_sol_hat) / max(nLc, 1e-30)
        alpha_same = norm_sq(L_same_sol_hat) / max(nLs, 1e-30)

        return {
            'alpha_full': alpha_full,
            'alpha_cross': alpha_cross,
            'alpha_same': alpha_same,
            'L_norm': np.sqrt(nL),
            'L_cross_norm': np.sqrt(nLc),
            'L_same_norm': np.sqrt(nLs),
        }

    def compute_scale_resolved_alpha(self, u_hat, n_shells=10):
        """Compute alpha at each wavenumber shell.

        For each shell [k_lo, k_hi], restrict L_cross and P_sol(L_cross) to that shell
        and compute the local alpha.
        """
        L_cross_hat = self.compute_lamb_hat_cross_only(u_hat)
        L_cross_sol_hat = self.project_leray(L_cross_hat)

        L_same_hat = self.compute_lamb_hat_bt_surgery(u_hat)
        L_same_sol_hat = self.project_leray(L_same_hat)

        kmag = np.sqrt(self.k2)
        kmax = self.N // 3  # dealiasing limit
        shell_edges = np.linspace(1, kmax, n_shells + 1)

        results = []
        for i in range(n_shells):
            mask = (kmag >= shell_edges[i]) & (kmag < shell_edges[i+1])
            mask3 = mask[np.newaxis, :, :, :]

            # Cross-helical
            Lc_shell = np.sum(np.abs(L_cross_hat * mask3)**2)
            Lc_sol_shell = np.sum(np.abs(L_cross_sol_hat * mask3)**2)
            alpha_c = float(Lc_sol_shell / max(Lc_shell, 1e-30))

            # Same-helical
            Ls_shell = np.sum(np.abs(L_same_hat * mask3)**2)
            Ls_sol_shell = np.sum(np.abs(L_same_sol_hat * mask3)**2)
            alpha_s = float(Ls_sol_shell / max(Ls_shell, 1e-30))

            k_mid = (shell_edges[i] + shell_edges[i+1]) / 2
            results.append({
                'k': float(k_mid),
                'alpha_cross': alpha_c,
                'alpha_same': alpha_s,
                'energy_cross': float(Lc_shell),
                'energy_same': float(Ls_shell),
            })

        return results

    def angular_triad_distribution(self, u_hat, n_bins=18, n_sample=200000):
        """Sample triads, bin by angle, weight by energy.

        For cross-helical triads: omega+(k1) x v-(k2) at k3 = k1+k2.
        Weight: |omega_p(k1)|^2 * |u_m(k2)|^2

        This tells us where the energy concentrates in angle-space
        and whether high-alpha (near-antiparallel) triads carry significant energy.
        """
        u_p, u_m = self.helical_decompose(u_hat)
        omega_hat = self.compute_vorticity_hat(u_hat)
        o_p, o_m = self.helical_decompose(omega_hat)

        N = self.N
        energy_by_angle = np.zeros(n_bins)
        alpha_by_angle = np.zeros(n_bins)
        count_by_angle = np.zeros(n_bins)

        for _ in range(n_sample):
            i1 = tuple(np.random.randint(0, N, 3))
            i2 = tuple(np.random.randint(0, N, 3))

            k1_vec = np.array([float(self.kx[i1]), float(self.ky[i1]), float(self.kz[i1])])
            k2_vec = np.array([float(self.kx[i2]), float(self.ky[i2]), float(self.kz[i2])])

            k1_mag = np.linalg.norm(k1_vec)
            k2_mag = np.linalg.norm(k2_vec)
            if k1_mag < 0.5 or k2_mag < 0.5:
                continue

            # Angle between k1 and k2
            cos_theta = np.dot(k1_vec, k2_vec) / (k1_mag * k2_mag)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)

            bin_idx = min(int(theta / np.pi * n_bins), n_bins - 1)

            # Cross-helical triad weight: |omega+(k1)|^2 * |v-(k2)|^2
            w = float(np.abs(o_p[i1])**2 * np.abs(u_m[i2])**2)
            if w < 1e-40:
                continue

            # Geometric alpha for this triad
            alpha = compute_alpha_single_triad(k1_vec, k2_vec)['+-']

            energy_by_angle[bin_idx] += w
            alpha_by_angle[bin_idx] += w * alpha
            count_by_angle[bin_idx] += 1

        # Weighted average alpha per bin
        weighted_alpha = np.zeros(n_bins)
        for i in range(n_bins):
            if energy_by_angle[i] > 1e-40:
                weighted_alpha[i] = alpha_by_angle[i] / energy_by_angle[i]

        # Normalize energy to fractions
        E_total = energy_by_angle.sum()
        energy_frac = energy_by_angle / max(E_total, 1e-40)

        bin_centers = np.linspace(np.pi / (2 * n_bins), np.pi - np.pi / (2 * n_bins), n_bins)
        return bin_centers, energy_frac, weighted_alpha


def evolve_and_measure(solver, u_hat, label, dt=0.005, t_max=10.0, report_interval=0.5):
    """Evolve flow and measure energy-weighted alpha at each reporting time."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    t = 0.0
    step = 0
    timeseries = []

    print(f"  {'t':>5s}  {'a_full':>7s}  {'a_cross':>7s}  {'a_same':>7s}  "
          f"{'Z':>10s}  {'|L_c|/|L|':>10s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*10}")

    while t <= t_max + 1e-10:
        if step % int(report_interval / dt) == 0 or t < dt:
            alphas = solver.compute_sector_alpha(u_hat)

            # Enstrophy
            omega_hat = solver.compute_vorticity_hat(u_hat)
            Z = float(0.5 * np.sum(np.abs(omega_hat)**2) / solver.N**3)

            # Cross/total ratio
            ratio = alphas['L_cross_norm'] / max(alphas['L_norm'], 1e-30)

            record = {
                't': round(t, 3),
                'alpha_full': round(alphas['alpha_full'], 6),
                'alpha_cross': round(alphas['alpha_cross'], 6),
                'alpha_same': round(alphas['alpha_same'], 6),
                'Z': round(Z, 6),
                'cross_ratio': round(ratio, 4),
            }
            timeseries.append(record)

            print(f"  {t:5.1f}  {alphas['alpha_full']:7.4f}  {alphas['alpha_cross']:7.4f}  "
                  f"{alphas['alpha_same']:7.4f}  {Z:10.4f}  {ratio:10.4f}")

        u_hat = solver.step_rk4(u_hat, dt, mode='full')
        t += dt
        step += 1

    return timeseries


def snapshot_analysis(solver, u_hat, label):
    """Detailed analysis at a single timestep: scale-resolved + angular."""
    print(f"\n  --- Scale-resolved alpha for {label} ---")
    shells = solver.compute_scale_resolved_alpha(u_hat, n_shells=8)

    print(f"  {'k':>5s}  {'a_cross':>8s}  {'a_same':>8s}  {'E_cross':>10s}  {'E_same':>10s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for s in shells:
        print(f"  {s['k']:5.1f}  {s['alpha_cross']:8.4f}  {s['alpha_same']:8.4f}  "
              f"{s['energy_cross']:10.2e}  {s['energy_same']:10.2e}")

    print(f"\n  --- Angular triad distribution for {label} ---")
    angles, energy_frac, weighted_alpha = solver.angular_triad_distribution(u_hat)

    print(f"  {'angle':>7s}  {'E_frac':>8s}  {'w_alpha':>8s}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}")
    for i in range(len(angles)):
        print(f"  {np.degrees(angles[i]):7.1f}  {energy_frac[i]:8.4f}  {weighted_alpha[i]:8.4f}")

    return shells, angles, energy_frac, weighted_alpha


def main():
    print("=" * 70)
    print("  ENERGY-WEIGHTED LERAY SUPPRESSION FACTOR")
    print("  Meridian 2 — Does energy weighting defeat the geometric average?")
    print("=" * 70)

    np.random.seed(42)
    solver = EnergyWeightedLeray(N=32, Re=400)

    # =====================================================================
    # 1. Time evolution for 3 ICs
    # =====================================================================
    ics = [
        ("Taylor-Green", solver.taylor_green_ic()),
        ("Random", solver.random_ic()),
        ("Imbalanced 80/20", solver.imbalanced_helical_ic()),
    ]

    all_timeseries = {}
    peak_states = {}

    for label, u_hat_ic in ics:
        ts = evolve_and_measure(solver, u_hat_ic.copy(), label)
        all_timeseries[label] = ts

        # Find peak enstrophy
        peak_idx = max(range(len(ts)), key=lambda i: ts[i]['Z'])
        peak_states[label] = {
            'time': ts[peak_idx]['t'],
            'Z_peak': ts[peak_idx]['Z'],
            'alpha_cross_at_peak': ts[peak_idx]['alpha_cross'],
            'alpha_same_at_peak': ts[peak_idx]['alpha_same'],
        }

    # =====================================================================
    # 2. Detailed snapshot at peak enstrophy
    # =====================================================================
    print("\n" + "=" * 70)
    print("  DETAILED ANALYSIS AT PEAK ENSTROPHY")
    print("=" * 70)

    all_shells = {}
    all_angular = {}

    for label, u_hat_ic in ics:
        # Re-evolve to peak
        t_peak = peak_states[label]['time']
        u_hat = u_hat_ic.copy()
        t = 0.0
        dt = 0.005
        while t < t_peak - dt/2:
            u_hat = solver.step_rk4(u_hat, dt, mode='full')
            t += dt

        shells, angles, energy_frac, weighted_alpha = snapshot_analysis(
            solver, u_hat, f"{label} (t={t_peak:.1f})")
        all_shells[label] = shells
        all_angular[label] = (angles, energy_frac, weighted_alpha)

    # =====================================================================
    # 3. VERDICT
    # =====================================================================
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    print("\n  Geometric average (Wanderer S93): alpha_+- = 0.398")
    print("\n  Energy-weighted alpha_cross at peak enstrophy:")
    print(f"  {'IC':<20s}  {'t_peak':>6s}  {'Z_peak':>8s}  {'a_cross':>8s}  {'a_same':>8s}  {'vs geom':>8s}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for label in peak_states:
        ps = peak_states[label]
        diff = ps['alpha_cross_at_peak'] - 0.398
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<20s}  {ps['time']:6.1f}  {ps['Z_peak']:8.4f}  "
              f"{ps['alpha_cross_at_peak']:8.4f}  {ps['alpha_same_at_peak']:8.4f}  "
              f"{sign}{diff:7.4f}")

    # Check if any alpha_cross exceeds critical thresholds
    max_alpha_cross = max(
        max(r['alpha_cross'] for r in ts)
        for ts in all_timeseries.values()
    )

    print(f"\n  Maximum alpha_cross observed across all ICs and times: {max_alpha_cross:.4f}")

    if max_alpha_cross < 0.50:
        print("  => STRONG RESULT: energy weighting REDUCES suppression factor")
        print("     Near-antiparallel triads do NOT accumulate enough energy")
        print("     The geometric bound is CONSERVATIVE")
    elif max_alpha_cross < 0.70:
        print("  => MODERATE: energy weighting slightly increases alpha")
        print("     Some concentration toward antiparallel, but still < 1")
        print("     A weighted bound may still close")
    else:
        print("  => WEAK: energy weighting significantly increases alpha")
        print("     Near-antiparallel triads carry substantial energy")
        print("     Geometric average is NOT representative")

    # =====================================================================
    # 4. Save results
    # =====================================================================
    results = {
        'params': {'N': 32, 'Re': 400, 'dt': 0.005, 't_max': 10.0},
        'geometric_reference': {'alpha_cross': 0.398, 'alpha_same': 0.870},
        'peak_enstrophy': peak_states,
        'max_alpha_cross': max_alpha_cross,
    }
    out_path = os.path.join(os.path.dirname(__file__), 'energy_weighted_leray_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # =====================================================================
    # 5. Plot
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Energy-Weighted Leray Suppression Factor', fontsize=14)

    colors = {'Taylor-Green': 'blue', 'Random': 'green', 'Imbalanced 80/20': 'red'}

    # Row 1: Time evolution of alpha
    ax = axes[0, 0]
    for label, ts in all_timeseries.items():
        times = [r['t'] for r in ts]
        a_cross = [r['alpha_cross'] for r in ts]
        ax.plot(times, a_cross, color=colors[label], label=label, linewidth=1.5)
    ax.axhline(y=0.398, color='gray', linestyle=':', label='Geometric avg')
    ax.set_xlabel('t')
    ax.set_ylabel('alpha_cross (energy-weighted)')
    ax.set_title('Cross-helical suppression vs time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    for label, ts in all_timeseries.items():
        times = [r['t'] for r in ts]
        a_same = [r['alpha_same'] for r in ts]
        ax.plot(times, a_same, color=colors[label], label=label, linewidth=1.5)
    ax.axhline(y=0.870, color='gray', linestyle=':', label='Geometric avg')
    ax.set_xlabel('t')
    ax.set_ylabel('alpha_same (energy-weighted)')
    ax.set_title('Same-helical suppression vs time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[0, 2]
    for label, ts in all_timeseries.items():
        times = [r['t'] for r in ts]
        Z = [r['Z'] for r in ts]
        ax.plot(times, Z, color=colors[label], label=label, linewidth=1.5)
    ax.set_xlabel('t')
    ax.set_ylabel('Enstrophy Z')
    ax.set_title('Enstrophy evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: Scale-resolved and angular at peak
    ax = axes[1, 0]
    for label, shells in all_shells.items():
        ks = [s['k'] for s in shells]
        ac = [s['alpha_cross'] for s in shells]
        ax.plot(ks, ac, 'o-', color=colors[label], label=label, linewidth=1.5)
    ax.axhline(y=0.398, color='gray', linestyle=':')
    ax.set_xlabel('|k| (wavenumber shell)')
    ax.set_ylabel('alpha_cross')
    ax.set_title('Scale-resolved alpha_cross at peak Z')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Angular distribution
    ax = axes[1, 1]
    for label, (angles, efrac, walpha) in all_angular.items():
        ax.bar(np.degrees(angles), efrac, width=8, alpha=0.4, color=colors[label], label=label)
    ax.set_xlabel('Angle between k1 and k2 (degrees)')
    ax.set_ylabel('Energy fraction')
    ax.set_title('Angular energy distribution of cross-helical triads')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for label, (angles, efrac, walpha) in all_angular.items():
        ax.plot(np.degrees(angles), walpha, 'o-', color=colors[label], label=label, linewidth=1.5)
    ax.axhline(y=0.398, color='gray', linestyle=':', label='Geometric avg')
    ax.set_xlabel('Angle between k1 and k2 (degrees)')
    ax.set_ylabel('Energy-weighted alpha_cross')
    ax.set_title('Weighted alpha by angle at peak Z')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'energy_weighted_leray.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
