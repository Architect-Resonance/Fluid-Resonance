"""
APPROACH (a1): KURAMOTO PHASE COHERENCE IN NS TRIADS
=====================================================
Measure the Kuramoto order parameter R_K(k) for triadic Fourier phases
in DNS, following Buzzicotti et al. (PRL 2021) and Manfredini & Gürcan (2025).

Key questions:
1. Are triadic phases clustered or uniformly distributed?
2. Does clustering increase with wavenumber (danger sign)?
3. Is the preferred phase near ±π/2 (Murray-Bustamante 2018 optimum)?
4. Does our sin²θ/4 Leray suppression modulate the phase coherence?

Definitions:
- For each triad (k1, k2, k3) with k1 + k2 + k3 = 0:
  - Triadic phase: φ(k1,k2,k3) = arg(û(k1) · û(k2) · û(k3))
    (more precisely: arg of the helicity-weighted triple product)
  - Kuramoto order parameter at shell k:
    R_K(k) = |<exp(i·φ)>_triads_at_scale_k|
    R_K = 0 means random phases, R_K = 1 means perfect alignment

- We also measure the HELICAL Kuramoto parameter:
  R_K^{+-}(k) = order parameter restricted to cross-helical triads
  R_K^{++}(k) = order parameter restricted to same-helical triads

This tells us whether the dangerous sector (cross-helical, 97-99% of
stretching) has MORE or LESS phase coherence than the safe sector.

HONEST: We measure and report. No wishful thinking.

Meridian 1 (for Meridian 2 to run), S96.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time as clock

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def compute_triadic_phases(solver, u_hat, k_shell, dk=1.0):
    """Compute triadic phases for all triads where |k3| is in the shell [k_shell-dk/2, k_shell+dk/2].

    For each triad (k1, k2, k3) with k1 + k2 = -k3:
    - Triadic phase = arg(û_s1(k1) · û_s2(k2) · conj(û_s3(k3)))
      where s1,s2,s3 are helical signs

    We collect phases separately for:
    - cross-helical triads (s1 ≠ s2)
    - same-helical triads (s1 = s2)

    Returns dict with phase arrays and metadata.
    """
    N = solver.N
    u_p, u_m = solver.helical_decompose(u_hat)

    # Shell selection mask for k3
    shell_mask = (solver.kmag >= k_shell - dk / 2) & (solver.kmag < k_shell + dk / 2)
    k3_indices = np.argwhere(shell_mask & (solver.k2 > 0))

    if len(k3_indices) == 0:
        return None

    # Limit number of k3 points to avoid combinatorial explosion
    max_k3 = min(len(k3_indices), 200)
    if len(k3_indices) > max_k3:
        rng = np.random.default_rng(42)
        k3_indices = k3_indices[rng.choice(len(k3_indices), max_k3, replace=False)]

    phases_cross = []
    phases_same = []
    angles_cross = []  # inter-wavevector angles θ for cross-helical triads
    angles_same = []

    # For each k3, sample k1 from active modes, k2 = -k3 - k1
    active_mask = (solver.k2 > 0) & (np.abs(u_p) + np.abs(u_m) > 1e-15)
    active_indices = np.argwhere(active_mask)

    # Subsample active modes
    max_k1 = min(len(active_indices), 100)
    if len(active_indices) > max_k1:
        rng = np.random.default_rng(123)
        active_indices = active_indices[rng.choice(len(active_indices), max_k1, replace=False)]

    for k3_idx in k3_indices:
        i3, j3, l3 = k3_idx
        k3_vec = np.array([solver.kx[i3, j3, l3],
                           solver.ky[i3, j3, l3],
                           solver.kz[i3, j3, l3]])

        for k1_idx in active_indices:
            i1, j1, l1 = k1_idx

            # k2 = -k3 - k1 (triadic condition)
            k1_vec = np.array([solver.kx[i1, j1, l1],
                               solver.ky[i1, j1, l1],
                               solver.kz[i1, j1, l1]])
            k2_vec = -k3_vec - k1_vec

            # Check if k2 is on the grid
            i2 = int(round(k2_vec[0])) % N
            j2 = int(round(k2_vec[1])) % N
            l2 = int(round(k2_vec[2])) % N

            # Verify k2 is valid (on grid and nonzero)
            k2_actual = np.array([solver.kx[i2, j2, l2],
                                  solver.ky[i2, j2, l2],
                                  solver.kz[i2, j2, l2]])
            if np.linalg.norm(k2_actual - k2_vec) > 0.1:
                continue  # k2 not on grid
            if solver.k2[i2, j2, l2] < 0.5:
                continue  # k2 = 0

            # Get helical coefficients
            up1 = u_p[i1, j1, l1]
            um1 = u_m[i1, j1, l1]
            up2 = u_p[i2, j2, l2]
            um2 = u_m[i2, j2, l2]
            up3 = u_p[i3, j3, l3]
            um3 = u_m[i3, j3, l3]

            # Compute angle between k1 and k2
            k1_mag = np.sqrt(solver.k2[i1, j1, l1])
            k2_mag = np.sqrt(solver.k2[i2, j2, l2])
            if k1_mag < 0.5 or k2_mag < 0.5:
                continue
            cos_theta = np.clip(np.dot(k1_vec, k2_actual) / (k1_mag * k2_mag), -1, 1)
            theta = np.arccos(cos_theta)

            # Cross-helical triadic phases: (+,-,s3) and (-,+,s3)
            for (a1, a2, a3, label) in [
                (up1, um2, np.conj(up3), 'cross'),
                (up1, um2, np.conj(um3), 'cross'),
                (um1, up2, np.conj(up3), 'cross'),
                (um1, up2, np.conj(um3), 'cross'),
                (up1, up2, np.conj(up3), 'same'),
                (up1, up2, np.conj(um3), 'same'),
                (um1, um2, np.conj(up3), 'same'),
                (um1, um2, np.conj(um3), 'same'),
            ]:
                triple = a1 * a2 * a3
                if abs(triple) < 1e-30:
                    continue
                phi = np.angle(triple)

                if label == 'cross':
                    phases_cross.append(phi)
                    angles_cross.append(theta)
                else:
                    phases_same.append(phi)
                    angles_same.append(theta)

    return {
        'k_shell': k_shell,
        'phases_cross': np.array(phases_cross) if phases_cross else np.array([]),
        'phases_same': np.array(phases_same) if phases_same else np.array([]),
        'angles_cross': np.array(angles_cross) if angles_cross else np.array([]),
        'angles_same': np.array(angles_same) if angles_same else np.array([]),
    }


def kuramoto_order_parameter(phases):
    """R_K = |<exp(i*phi)>|. Returns (R_K, mean_phase)."""
    if len(phases) == 0:
        return 0.0, 0.0
    z = np.mean(np.exp(1j * phases))
    return abs(z), np.angle(z)


def run_kuramoto_analysis(solver, u_hat, ic_name, k_shells=None):
    """Run full Kuramoto analysis across wavenumber shells."""
    N = solver.N
    if k_shells is None:
        k_shells = np.arange(1, N // 3, 1)

    results = []
    for k in k_shells:
        data = compute_triadic_phases(solver, u_hat, k, dk=1.0)
        if data is None:
            continue

        R_cross, phi_cross = kuramoto_order_parameter(data['phases_cross'])
        R_same, phi_same = kuramoto_order_parameter(data['phases_same'])

        # Also compute R for ALL phases combined
        all_phases = np.concatenate([data['phases_cross'], data['phases_same']])
        R_all, phi_all = kuramoto_order_parameter(all_phases)

        # Phase near ±π/2 fraction (Murray-Bustamante optimum)
        if len(data['phases_cross']) > 0:
            near_pi2 = np.mean(np.abs(np.abs(data['phases_cross']) - np.pi / 2) < np.pi / 6)
        else:
            near_pi2 = 0.0

        results.append({
            'k': k,
            'R_cross': R_cross, 'phi_cross': phi_cross,
            'R_same': R_same, 'phi_same': phi_same,
            'R_all': R_all, 'phi_all': phi_all,
            'n_cross': len(data['phases_cross']),
            'n_same': len(data['phases_same']),
            'near_pi2_frac': near_pi2,
            'mean_angle_cross': np.mean(data['angles_cross']) if len(data['angles_cross']) > 0 else 0,
        })

    return results


def run_time_evolution(solver, ic_name, u_hat_ic, dt=0.005, T=3.0,
                       measure_every=50, k_shells=None):
    """Run NS evolution and measure Kuramoto parameters at intervals."""
    N = solver.N
    if k_shells is None:
        k_shells = np.arange(1, N // 3, 2)  # every other shell for speed

    n_steps = int(T / dt)
    u_hat = u_hat_ic.copy()

    all_snapshots = []
    times = []

    print(f"\n{'='*70}")
    print(f"Kuramoto Phase Coherence — {ic_name}")
    print(f"N={N}, Re={1/solver.nu:.0f}, dt={dt}, T={T}")
    print(f"Shells: {k_shells}")
    print(f"{'='*70}\n")

    for step in range(n_steps + 1):
        t = step * dt

        if step % measure_every == 0:
            print(f"  t = {t:.3f} — measuring phase coherence...")
            results = run_kuramoto_analysis(solver, u_hat, ic_name, k_shells)
            all_snapshots.append(results)
            times.append(t)

            # Quick summary
            if results:
                R_cross_vals = [r['R_cross'] for r in results if r['n_cross'] > 10]
                R_same_vals = [r['R_same'] for r in results if r['n_same'] > 10]
                if R_cross_vals:
                    print(f"    R_K(cross): mean={np.mean(R_cross_vals):.4f}, "
                          f"max={np.max(R_cross_vals):.4f}")
                if R_same_vals:
                    print(f"    R_K(same):  mean={np.mean(R_same_vals):.4f}, "
                          f"max={np.max(R_same_vals):.4f}")

        if step < n_steps:
            u_hat = solver.step_rk4(u_hat, dt, mode='full')

    return times, all_snapshots


def plot_kuramoto_results(times, all_snapshots, ic_name, save_path):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Kuramoto Phase Coherence — {ic_name}", fontsize=14, fontweight='bold')

    # --- Panel 1: R_K vs k at multiple times ---
    ax = axes[0, 0]
    n_times = len(times)
    cmap = plt.cm.viridis
    for i, (t, snap) in enumerate(zip(times, all_snapshots)):
        if not snap:
            continue
        ks = [r['k'] for r in snap if r['n_cross'] > 10]
        Rs = [r['R_cross'] for r in snap if r['n_cross'] > 10]
        if ks:
            color = cmap(i / max(n_times - 1, 1))
            ax.plot(ks, Rs, 'o-', color=color, markersize=3, linewidth=1,
                    label=f't={t:.2f}' if i % max(1, n_times // 5) == 0 else None)

    # Buzzicotti scaling: R ~ k^{-1}
    if all_snapshots and all_snapshots[0]:
        ks_ref = np.array([r['k'] for r in all_snapshots[0] if r['n_cross'] > 10])
        if len(ks_ref) > 2:
            R_ref = 0.5 * ks_ref[0] / ks_ref  # R ~ k^{-1} reference
            ax.plot(ks_ref, R_ref, 'k--', linewidth=1.5, alpha=0.5, label=r'$k^{-1}$ (Buzzicotti)')

    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel(r'$R_K^{+-}$ (cross-helical)')
    ax.set_title('Cross-helical Kuramoto parameter vs k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: R_cross vs R_same at last time ---
    ax = axes[0, 1]
    if all_snapshots:
        last_snap = all_snapshots[-1]
        ks = [r['k'] for r in last_snap if r['n_cross'] > 10 and r['n_same'] > 10]
        R_cross = [r['R_cross'] for r in last_snap if r['n_cross'] > 10 and r['n_same'] > 10]
        R_same = [r['R_same'] for r in last_snap if r['n_cross'] > 10 and r['n_same'] > 10]
        if ks:
            ax.plot(ks, R_cross, 'ro-', markersize=4, linewidth=1.5, label=r'$R_K^{+-}$ (cross)')
            ax.plot(ks, R_same, 'bs-', markersize=4, linewidth=1.5, label=r'$R_K^{++}$ (same)')
            ax.set_xlabel('Wavenumber k')
            ax.set_ylabel(r'$R_K$')
            ax.set_title(f'Cross vs Same helical phase coherence (t={times[-1]:.2f})')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # --- Panel 3: Phase histogram at selected k ---
    ax = axes[1, 0]
    if all_snapshots:
        last_snap = all_snapshots[-1]
        # Find k with most triads
        best_r = max(last_snap, key=lambda r: r['n_cross']) if last_snap else None
        if best_r and best_r['n_cross'] > 0:
            # Recompute phases for this shell
            solver_dummy = None  # We don't have solver here, so we stored phases
            # Use the R and phi values to show the distribution
            ax.text(0.5, 0.5, f"k={best_r['k']}\n"
                    f"R_cross={best_r['R_cross']:.4f}\n"
                    f"mean phase={np.degrees(best_r['phi_cross']):.1f}°\n"
                    f"near ±π/2: {best_r['near_pi2_frac']*100:.1f}%\n"
                    f"N_triads={best_r['n_cross']}",
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_title('Phase statistics at peak-triad shell')

    # --- Panel 4: R_K time evolution at selected k ---
    ax = axes[1, 1]
    if len(all_snapshots) > 1:
        # Track R_K at a few k values over time
        target_ks = [2, 4, 6, 8]
        for k_target in target_ks:
            R_t = []
            t_vals = []
            for t, snap in zip(times, all_snapshots):
                matching = [r for r in snap if r['k'] == k_target and r['n_cross'] > 5]
                if matching:
                    R_t.append(matching[0]['R_cross'])
                    t_vals.append(t)
            if R_t:
                ax.plot(t_vals, R_t, 'o-', markersize=4, linewidth=1.5, label=f'k={k_target}')

        ax.set_xlabel('Time t')
        ax.set_ylabel(r'$R_K^{+-}$')
        ax.set_title('Phase coherence evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("KURAMOTO PHASE COHERENCE IN NS TRIADS")
    print("=" * 70)
    print()
    print("Approach (a1): Measure triadic phase alignment")
    print("  - Kuramoto R_K(k) for cross-helical and same-helical triads")
    print("  - Compare with Buzzicotti et al. PRL 2021: cos(alpha_k) ~ k^{-1}")
    print("  - Check Murray-Bustamante optimum: phases near ±π/2")
    print()

    N = 32
    Re = 400
    dt = 0.005
    T = 3.0
    measure_every = 100  # measure every 100 steps = every 0.5 time units

    solver = SpectralNS(N=N, Re=Re)
    k_shells = np.arange(1, N // 3, 1)

    wall_start = clock.time()

    # Run for 3 ICs
    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Random': solver.random_ic(seed=42),
        'Imbalanced (80/20)': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
    }

    all_results = {}
    for ic_name, u_hat_ic in ics.items():
        times, snapshots = run_time_evolution(
            solver, ic_name, u_hat_ic,
            dt=dt, T=T, measure_every=measure_every, k_shells=k_shells
        )
        all_results[ic_name] = (times, snapshots)

        # Plot per IC
        safe_name = ic_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        plot_kuramoto_results(
            times, snapshots, ic_name,
            f'h:/tmp/kuramoto_{safe_name}.png'
        )

    wall_time = clock.time() - wall_start

    # ============================================================
    # CROSS-IC SUMMARY
    # ============================================================
    print("\n\n" + "=" * 70)
    print("CROSS-IC SUMMARY")
    print("=" * 70)

    for ic_name, (times, snapshots) in all_results.items():
        print(f"\n--- {ic_name} ---")
        if not snapshots or not snapshots[-1]:
            print("  No data collected.")
            continue

        last_snap = snapshots[-1]
        R_cross = [r['R_cross'] for r in last_snap if r['n_cross'] > 10]
        R_same = [r['R_same'] for r in last_snap if r['n_same'] > 10]
        ks = [r['k'] for r in last_snap if r['n_cross'] > 10]

        if R_cross:
            print(f"  Final t={times[-1]:.2f}")
            print(f"  R_K^{{+-}} (cross): mean={np.mean(R_cross):.4f}, "
                  f"max={np.max(R_cross):.4f} at k={ks[np.argmax(R_cross)]}")
            if R_same:
                print(f"  R_K^{{++}} (same):  mean={np.mean(R_same):.4f}, "
                      f"max={np.max(R_same):.4f}")
                ratio = np.mean(R_cross) / max(np.mean(R_same), 1e-10)
                print(f"  R_cross/R_same ratio: {ratio:.4f}")
                if ratio > 1.2:
                    print("  >>> DANGER: Cross-helical triads are MORE phase-coherent!")
                elif ratio < 0.8:
                    print("  >>> GOOD: Cross-helical triads are LESS phase-coherent")
                else:
                    print("  >>> NEUTRAL: Similar phase coherence in both sectors")

            # Check Buzzicotti scaling
            if len(ks) > 3:
                log_k = np.log(np.array(ks))
                log_R = np.log(np.array(R_cross))
                valid = np.isfinite(log_R) & np.isfinite(log_k)
                if np.sum(valid) > 2:
                    slope, intercept = np.polyfit(log_k[valid], log_R[valid], 1)
                    print(f"  R_K(k) ~ k^{{{slope:.2f}}} (Buzzicotti predicts k^{{-1}})")

            # Murray-Bustamante: phase near ±π/2
            pi2_fracs = [r['near_pi2_frac'] for r in last_snap if r['n_cross'] > 10]
            if pi2_fracs:
                print(f"  Phases near ±π/2: {np.mean(pi2_fracs)*100:.1f}% "
                      f"(random would be 33.3%)")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()
    print("Key questions answered:")
    print("  1. R_K(k) < 0.3 everywhere? → Phase coherence is WEAK (good for regularity)")
    print("  2. R_K(k) grows with k? → Phase coherence INCREASES at small scales (danger)")
    print("  3. R_cross > R_same? → Dangerous sector has more coherence (bad)")
    print("  4. Phases near ±π/2? → Murray-Bustamante optimum (maximizes cascade)")
    print()
    print("These measurements will tell us whether phase coherence is the")
    print("missing ingredient that separates α = 1/4 (incoherent bound) from")
    print("α_eff ~ 0.9 (late-time observed). This is the single most important")
    print("open question in our research.")

    print(f"\nTotal wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
