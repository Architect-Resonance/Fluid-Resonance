"""
LERAY-WEIGHTED KURAMOTO PHASE COHERENCE
=========================================
Meridian 1 (S96-M1d) showed the C-F bridge needs phase decoherence exponent
q > 7/6. Raw R_K gives q ≈ 0.3. But the RELEVANT quantity for regularity is
the sin²θ-WEIGHTED R_K — because sin²θ/4 kills triads near θ=0 and θ=π.

Key idea: triads near θ=π/2 contribute MOST to enstrophy production (they have
the largest sin²θ/4 solenoidal fraction). If THOSE triads are more decoherent
than average, the effective q could be much larger.

This script measures:
1. R_K in angular bins:
   - Near-parallel:      θ < π/6      (killed by sin²θ/4 → irrelevant)
   - Local:              π/6 < θ < 5π/6 (critical range)
   - Near-antiparallel:  θ > 5π/6     (killed by sin²θ/4 → irrelevant)

2. Leray-weighted Kuramoto:
   R_K^{Leray}(k) = |⟨sin²θ · exp(iφ)⟩| / ⟨sin²θ⟩

3. Compare k-scaling exponent q for raw vs Leray-weighted R_K.

HONEST: We measure and report. If q_Leray > 7/6, the C-F bridge closes.
If not, we report the gap honestly.

Meridian 2, S97.
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


# Angular bin edges
BIN_PARALLEL = np.pi / 6        # θ < π/6
BIN_ANTIPARALLEL = 5 * np.pi / 6  # θ > 5π/6


def compute_phases_with_angles(solver, u_hat, k_shell, dk=1.0, max_k3=200, max_k1=100):
    """Compute triadic phases AND inter-wavevector angles for cross-helical triads.

    Returns arrays of (phase, theta, sin2theta) for each triad.
    """
    N = solver.N
    u_p, u_m = solver.helical_decompose(u_hat)

    shell_mask = (solver.kmag >= k_shell - dk / 2) & (solver.kmag < k_shell + dk / 2)
    k3_indices = np.argwhere(shell_mask & (solver.k2 > 0))

    if len(k3_indices) == 0:
        return None

    if len(k3_indices) > max_k3:
        rng = np.random.default_rng(42)
        k3_indices = k3_indices[rng.choice(len(k3_indices), max_k3, replace=False)]

    phases = []
    thetas = []

    active_mask = (solver.k2 > 0) & (np.abs(u_p) + np.abs(u_m) > 1e-15)
    active_indices = np.argwhere(active_mask)

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
            k1_vec = np.array([solver.kx[i1, j1, l1],
                               solver.ky[i1, j1, l1],
                               solver.kz[i1, j1, l1]])
            k2_vec = -k3_vec - k1_vec

            i2 = int(round(k2_vec[0])) % N
            j2 = int(round(k2_vec[1])) % N
            l2 = int(round(k2_vec[2])) % N

            k2_actual = np.array([solver.kx[i2, j2, l2],
                                  solver.ky[i2, j2, l2],
                                  solver.kz[i2, j2, l2]])
            if np.linalg.norm(k2_actual - k2_vec) > 0.1:
                continue
            if solver.k2[i2, j2, l2] < 0.5:
                continue

            # Angle between k1 and k2
            k1_mag = np.sqrt(solver.k2[i1, j1, l1])
            k2_mag = np.sqrt(solver.k2[i2, j2, l2])
            if k1_mag < 0.5 or k2_mag < 0.5:
                continue
            cos_theta = np.clip(np.dot(k1_vec, k2_actual) / (k1_mag * k2_mag), -1, 1)
            theta = np.arccos(cos_theta)

            up1 = u_p[i1, j1, l1]
            um1 = u_m[i1, j1, l1]
            up2 = u_p[i2, j2, l2]
            um2 = u_m[i2, j2, l2]
            up3 = u_p[i3, j3, l3]
            um3 = u_m[i3, j3, l3]

            # Cross-helical triadic phases only (the dangerous sector)
            for (a1, a2, a3) in [
                (up1, um2, np.conj(up3)),
                (up1, um2, np.conj(um3)),
                (um1, up2, np.conj(up3)),
                (um1, up2, np.conj(um3)),
            ]:
                triple = a1 * a2 * a3
                if abs(triple) > 1e-30:
                    phases.append(np.angle(triple))
                    thetas.append(theta)

    if not phases:
        return None

    return {
        'phases': np.array(phases),
        'thetas': np.array(thetas),
        'sin2theta': np.sin(np.array(thetas))**2,
    }


def compute_binned_and_weighted_RK(data):
    """Compute R_K in angular bins and sin²θ-weighted R_K.

    Returns dict with:
    - R_parallel, R_local, R_antiparallel (angular bin R_K)
    - R_leray (sin²θ-weighted)
    - R_raw (unweighted, all angles)
    - n_parallel, n_local, n_antiparallel (counts)
    """
    if data is None:
        return None

    phases = data['phases']
    thetas = data['thetas']
    sin2t = data['sin2theta']

    # Angular bins
    mask_par = thetas < BIN_PARALLEL
    mask_anti = thetas > BIN_ANTIPARALLEL
    mask_local = ~mask_par & ~mask_anti

    result = {
        'n_total': len(phases),
        'n_parallel': int(np.sum(mask_par)),
        'n_local': int(np.sum(mask_local)),
        'n_antiparallel': int(np.sum(mask_anti)),
    }

    # Raw R_K (all angles, unweighted)
    if len(phases) > 0:
        result['R_raw'] = abs(np.mean(np.exp(1j * phases)))
    else:
        result['R_raw'] = 0.0

    # Binned R_K
    for name, mask in [('parallel', mask_par), ('local', mask_local),
                        ('antiparallel', mask_anti)]:
        p = phases[mask]
        if len(p) > 5:
            result[f'R_{name}'] = abs(np.mean(np.exp(1j * p)))
        else:
            result[f'R_{name}'] = float('nan')

    # Leray-weighted R_K: |⟨sin²θ · exp(iφ)⟩| / ⟨sin²θ⟩
    if len(phases) > 0 and np.sum(sin2t) > 1e-15:
        weighted_z = np.mean(sin2t * np.exp(1j * phases))
        mean_sin2t = np.mean(sin2t)
        result['R_leray'] = abs(weighted_z) / mean_sin2t
    else:
        result['R_leray'] = 0.0

    # Mean sin²θ (diagnostic: should be ~2/3 for isotropic)
    result['mean_sin2theta'] = np.mean(sin2t) if len(sin2t) > 0 else 0.0

    # Fraction of sin²θ weight in each bin
    total_weight = np.sum(sin2t)
    if total_weight > 0:
        result['weight_parallel'] = np.sum(sin2t[mask_par]) / total_weight
        result['weight_local'] = np.sum(sin2t[mask_local]) / total_weight
        result['weight_antiparallel'] = np.sum(sin2t[mask_anti]) / total_weight
    else:
        result['weight_parallel'] = 0.0
        result['weight_local'] = 0.0
        result['weight_antiparallel'] = 0.0

    return result


def fit_power_law(k, R, min_pts=3):
    """Fit R ~ k^beta, return beta."""
    valid = (k > 0) & (R > 0) & np.isfinite(R) & np.isfinite(np.log(np.maximum(R, 1e-30)))
    if np.sum(valid) < min_pts:
        return float('nan')
    slope, _ = np.polyfit(np.log(k[valid]), np.log(R[valid]), 1)
    return slope


def run_measurement(solver, u_hat, ic_name, k_shells):
    """Measure binned and weighted R_K across shells."""
    results = []
    for k in k_shells:
        data = compute_phases_with_angles(solver, u_hat, k)
        rk = compute_binned_and_weighted_RK(data)
        if rk is not None and rk['n_total'] > 10:
            rk['k'] = k
            results.append(rk)

    return results


def main():
    print("=" * 70)
    print("  LERAY-WEIGHTED KURAMOTO PHASE COHERENCE")
    print("=" * 70)
    print()
    print("  C-F bridge needs q > 7/6 = 1.167 for convergence")
    print("  q >= 5/3 = 1.667 for Holder-1/2 regularity")
    print("  Raw R_K gives q ~ 0.3. Does Leray-weighting improve this?")
    print()

    # Run at multiple Re to check scaling
    configs = [
        (400,  32),
        (1600, 64),
    ]

    dt_base = 0.005
    T = 3.0  # evolve to developed turbulence

    all_data = {}
    wall_start = clock.time()

    for Re, N in configs:
        print(f"\n{'='*70}")
        print(f"  Re = {Re}, N = {N}")
        print(f"{'='*70}")

        dt = dt_base * (32 / N)
        n_steps = int(T / dt)
        k_shells = np.arange(1, N // 3, 1)
        solver = SpectralNS(N=N, Re=Re)

        # Use random IC
        u_hat = solver.random_ic(seed=42)

        # Evolve to T
        print(f"  Evolving to t={T}...")
        t0 = clock.time()
        for step in range(n_steps):
            u_hat = solver.step_rk4(u_hat, dt, mode='full')
        print(f"  Evolution: {clock.time()-t0:.1f}s")

        # Measure
        print(f"  Measuring R_K with angular decomposition...")
        results = run_measurement(solver, u_hat, 'Random', k_shells)
        all_data[(Re, N)] = results

        if not results:
            print("  No data!")
            continue

        # Extract arrays
        ks = np.array([r['k'] for r in results])
        R_raw = np.array([r['R_raw'] for r in results])
        R_leray = np.array([r['R_leray'] for r in results])
        R_local = np.array([r['R_local'] for r in results])
        R_par = np.array([r['R_parallel'] for r in results])
        R_anti = np.array([r['R_antiparallel'] for r in results])

        # Fit power laws
        q_raw = -fit_power_law(ks, R_raw)  # negate: R ~ k^{-q}
        q_leray = -fit_power_law(ks, R_leray)
        q_local = -fit_power_law(ks[np.isfinite(R_local)],
                                  R_local[np.isfinite(R_local)])

        print(f"\n  RESULTS (t={T}, Random IC):")
        print(f"    R_raw:   mean={np.nanmean(R_raw):.4f}, q={q_raw:+.3f}")
        print(f"    R_leray: mean={np.nanmean(R_leray):.4f}, q={q_leray:+.3f}")
        print(f"    R_local: mean={np.nanmean(R_local):.4f}, q={q_local:+.3f}")

        # Angular bin statistics
        n_par = np.mean([r['n_parallel'] for r in results])
        n_loc = np.mean([r['n_local'] for r in results])
        n_anti = np.mean([r['n_antiparallel'] for r in results])
        w_par = np.mean([r['weight_parallel'] for r in results])
        w_loc = np.mean([r['weight_local'] for r in results])
        w_anti = np.mean([r['weight_antiparallel'] for r in results])

        print(f"\n  Angular distribution:")
        print(f"    Parallel (θ<π/6):       n={n_par:.0f} ({n_par/(n_par+n_loc+n_anti)*100:.1f}%), "
              f"sin²θ weight={w_par:.3f}")
        print(f"    Local (π/6<θ<5π/6):     n={n_loc:.0f} ({n_loc/(n_par+n_loc+n_anti)*100:.1f}%), "
              f"sin²θ weight={w_loc:.3f}")
        print(f"    Antiparallel (θ>5π/6):  n={n_anti:.0f} ({n_anti/(n_par+n_loc+n_anti)*100:.1f}%), "
              f"sin²θ weight={w_anti:.3f}")
        print(f"    Mean sin²θ = {np.mean([r['mean_sin2theta'] for r in results]):.4f} "
              f"(isotropic = 2/3 = 0.667)")

        # Per-k detail for a few shells
        print(f"\n  Per-shell detail:")
        print(f"  {'k':>3s}  {'R_raw':>8s}  {'R_leray':>8s}  {'R_local':>8s}  "
              f"{'R_par':>8s}  {'R_anti':>8s}  {'n':>5s}")
        for r in results:
            print(f"  {r['k']:3.0f}  {r['R_raw']:8.4f}  {r['R_leray']:8.4f}  "
                  f"{r['R_local']:8.4f}  "
                  f"{r['R_parallel']:8.4f}  {r['R_antiparallel']:8.4f}  "
                  f"{r['n_total']:5d}")

        # C-F bridge check
        print(f"\n  C-F BRIDGE CHECK:")
        print(f"    Raw q = {q_raw:.3f} (need > 1.167)")
        print(f"    Leray-weighted q = {q_leray:.3f} (need > 1.167)")
        print(f"    Local-only q = {q_local:.3f} (need > 1.167)")
        if q_leray > 7/6:
            print(f"    >>> GAP CLOSED: q_Leray > 7/6!")
        elif q_leray > 1:
            print(f"    >>> CLOSER but gap remains: deficit = {7/6 - q_leray:.3f}")
        else:
            print(f"    >>> Gap still open: deficit = {7/6 - q_leray:.3f}")

    total_time = clock.time() - wall_start

    # ============================================================
    # CROSS-Re COMPARISON
    # ============================================================
    if len(all_data) > 1:
        print(f"\n{'='*70}")
        print("  CROSS-Re COMPARISON")
        print(f"{'='*70}")
        for (Re, N), results in all_data.items():
            if not results:
                continue
            ks = np.array([r['k'] for r in results])
            R_leray = np.array([r['R_leray'] for r in results])
            q = -fit_power_law(ks, R_leray)
            print(f"  Re={Re:5d}: q_Leray = {q:+.3f}, <R_Leray> = {np.nanmean(R_leray):.4f}")

    # ============================================================
    # VERDICT
    # ============================================================
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    print("""
  The question: does sin²θ-weighting change the effective decoherence
  exponent q enough to close the C-F gap (need q > 7/6)?

  Three scenarios:
  1. q_Leray >> q_raw and q_Leray > 7/6 → GAP CLOSED (best case)
  2. q_Leray > q_raw but < 7/6 → Leray weighting helps but not enough
  3. q_Leray ≈ q_raw → angular structure doesn't matter for decoherence

  Note: even if q is insufficient, R_K ~ 0.005 means the ABSOLUTE
  coherence is tiny. The C-F exponent is a worst-case theoretical bound;
  the actual contribution from phase coherence may be bounded by the
  small magnitude of R_K regardless of scaling.
""")

    print(f"  Total wall time: {total_time:.1f}s")

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Leray-Weighted Kuramoto Phase Coherence', fontsize=13)

    colors_re = {400: 'blue', 1600: 'orange', 3200: 'red'}

    # Panel 1: R_raw vs R_leray vs R_local (first Re)
    ax = axes[0, 0]
    for (Re, N), results in all_data.items():
        if not results:
            continue
        ks = np.array([r['k'] for r in results])
        R_raw = np.array([r['R_raw'] for r in results])
        R_leray = np.array([r['R_leray'] for r in results])
        R_local = np.array([r['R_local'] for r in results])

        c = colors_re.get(Re, 'black')
        ax.plot(ks, R_raw, 'o-', color=c, markersize=3, linewidth=1, alpha=0.5,
                label=f'R_raw Re={Re}')
        ax.plot(ks, R_leray, 's-', color=c, markersize=4, linewidth=2,
                label=f'R_Leray Re={Re}')

    k_ref = np.arange(2, 20)
    ax.plot(k_ref, 0.03 * k_ref**(-7/6), 'k--', alpha=0.4, label=r'$k^{-7/6}$ (C-F threshold)')
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel(r'$R_K$')
    ax.set_title('Raw vs Leray-weighted R_K')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Angular bin R_K (first Re only)
    ax = axes[0, 1]
    first_key = list(all_data.keys())[0]
    results = all_data[first_key]
    if results:
        ks = np.array([r['k'] for r in results])
        R_par = np.array([r['R_parallel'] for r in results])
        R_loc = np.array([r['R_local'] for r in results])
        R_anti = np.array([r['R_antiparallel'] for r in results])

        ax.plot(ks[np.isfinite(R_par)], R_par[np.isfinite(R_par)], 'g^-', markersize=4,
                linewidth=1, label=r'$\theta < \pi/6$ (parallel)')
        ax.plot(ks[np.isfinite(R_loc)], R_loc[np.isfinite(R_loc)], 'ro-', markersize=4,
                linewidth=1.5, label=r'$\pi/6 < \theta < 5\pi/6$ (local)')
        ax.plot(ks[np.isfinite(R_anti)], R_anti[np.isfinite(R_anti)], 'bv-', markersize=4,
                linewidth=1, label=r'$\theta > 5\pi/6$ (antiparallel)')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel(r'$R_K$')
        ax.set_title(f'R_K by angular bin (Re={first_key[0]})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 3: sin²θ weight distribution
    ax = axes[1, 0]
    if results:
        w_par = [r['weight_parallel'] for r in results]
        w_loc = [r['weight_local'] for r in results]
        w_anti = [r['weight_antiparallel'] for r in results]
        ks = [r['k'] for r in results]
        ax.bar(ks, w_loc, color='red', alpha=0.6, label='Local')
        ax.bar(ks, w_par, bottom=w_loc, color='green', alpha=0.6, label='Parallel')
        ax.bar(ks, w_anti, bottom=[l+p for l,p in zip(w_loc, w_par)],
               color='blue', alpha=0.6, label='Antiparallel')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('sin²θ weight fraction')
        ax.set_title('Weight distribution by angular bin')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 4: R_leray / R_raw ratio
    ax = axes[1, 1]
    for (Re, N), results in all_data.items():
        if not results:
            continue
        ks = np.array([r['k'] for r in results])
        R_raw = np.array([r['R_raw'] for r in results])
        R_leray = np.array([r['R_leray'] for r in results])
        valid = (R_raw > 0) & (R_leray > 0)
        if np.any(valid):
            ratio = R_leray[valid] / R_raw[valid]
            c = colors_re.get(Re, 'black')
            ax.plot(ks[valid], ratio, 'o-', color=c, markersize=4, linewidth=1.5,
                    label=f'Re={Re}')
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel(r'$R_K^{Leray} / R_K^{raw}$')
    ax.set_title('Leray weighting effect')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'kuramoto_leray_weighted.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
