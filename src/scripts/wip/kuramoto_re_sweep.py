"""
KURAMOTO PHASE COHERENCE vs REYNOLDS NUMBER — Re SWEEP
========================================================
The #1 open question: does phase coherence R_K grow at higher Re?
Buzzicotti et al. (PRL 2021) predict cos(alpha_k) ~ k^{-1}.

We sweep Re = [400, 800, 1600, 3200] with matched resolution:
  Re=400  → N=32  (baseline, already measured)
  Re=800  → N=48
  Re=1600 → N=64
  Re=3200 → N=96  (expensive but tractable)

For each Re, we evolve to t=3.0 and measure R_K at the final snapshot
(after turbulence has developed). We also track R_K at intermediate times
to see if coherence grows during evolution.

HONEST: We measure and report. No wishful thinking.

Meridian 2, S97. Extension of a1.
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


def compute_triadic_phases_fast(solver, u_hat, k_shell, dk=1.0, max_k3=150, max_k1=80):
    """Fast version of triadic phase computation for sweep.
    Reduced sampling for speed at high N."""
    N = solver.N
    u_p, u_m = solver.helical_decompose(u_hat)

    shell_mask = (solver.kmag >= k_shell - dk / 2) & (solver.kmag < k_shell + dk / 2)
    k3_indices = np.argwhere(shell_mask & (solver.k2 > 0))

    if len(k3_indices) == 0:
        return None

    if len(k3_indices) > max_k3:
        rng = np.random.default_rng(42)
        k3_indices = k3_indices[rng.choice(len(k3_indices), max_k3, replace=False)]

    phases_cross = []
    phases_same = []

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

            up1 = u_p[i1, j1, l1]
            um1 = u_m[i1, j1, l1]
            up2 = u_p[i2, j2, l2]
            um2 = u_m[i2, j2, l2]
            up3 = u_p[i3, j3, l3]
            um3 = u_m[i3, j3, l3]

            # Cross-helical: (+,-,s3) and (-,+,s3)
            for (a1, a2, a3) in [
                (up1, um2, np.conj(up3)),
                (up1, um2, np.conj(um3)),
                (um1, up2, np.conj(up3)),
                (um1, up2, np.conj(um3)),
            ]:
                triple = a1 * a2 * a3
                if abs(triple) > 1e-30:
                    phases_cross.append(np.angle(triple))

            # Same-helical: (s,s,s3)
            for (a1, a2, a3) in [
                (up1, up2, np.conj(up3)),
                (up1, up2, np.conj(um3)),
                (um1, um2, np.conj(up3)),
                (um1, um2, np.conj(um3)),
            ]:
                triple = a1 * a2 * a3
                if abs(triple) > 1e-30:
                    phases_same.append(np.angle(triple))

    return {
        'phases_cross': np.array(phases_cross) if phases_cross else np.array([]),
        'phases_same': np.array(phases_same) if phases_same else np.array([]),
    }


def kuramoto_R(phases):
    """Kuramoto order parameter."""
    if len(phases) == 0:
        return 0.0
    return abs(np.mean(np.exp(1j * phases)))


def measure_coherence_spectrum(solver, u_hat, k_shells):
    """Measure R_K(k) for cross and same helical triads."""
    R_cross_k = []
    R_same_k = []
    n_cross_k = []
    valid_k = []

    for k in k_shells:
        data = compute_triadic_phases_fast(solver, u_hat, k)
        if data is None:
            continue

        nc = len(data['phases_cross'])
        ns = len(data['phases_same'])

        if nc < 5:
            continue

        valid_k.append(k)
        R_cross_k.append(kuramoto_R(data['phases_cross']))
        R_same_k.append(kuramoto_R(data['phases_same']) if ns > 5 else 0.0)
        n_cross_k.append(nc)

    return {
        'k': np.array(valid_k),
        'R_cross': np.array(R_cross_k),
        'R_same': np.array(R_same_k),
        'n_cross': np.array(n_cross_k),
    }


def fit_power_law(k, R):
    """Fit R ~ k^beta, return beta."""
    valid = (k > 0) & (R > 0) & np.isfinite(np.log(R))
    if np.sum(valid) < 3:
        return float('nan')
    slope, _ = np.polyfit(np.log(k[valid]), np.log(R[valid]), 1)
    return slope


def main():
    print("=" * 70)
    print("  KURAMOTO PHASE COHERENCE vs REYNOLDS NUMBER")
    print("=" * 70)

    # Re/N pairs: ensure k_max ~ N/3 gives enough inertial range
    configs = [
        (400,  32),
        (800,  48),
        (1600, 64),
        (3200, 96),
    ]

    dt_base = 0.005  # dt for Re=400, N=32
    T = 3.0
    measure_times = [1.0, 2.0, 3.0]  # measure at these times

    all_results = {}
    wall_start = clock.time()

    for Re, N in configs:
        print(f"\n{'='*70}")
        print(f"  Re = {Re}, N = {N}")
        print(f"{'='*70}")

        # Scale dt with N to maintain CFL
        dt = dt_base * (32 / N)
        n_steps = int(T / dt)
        k_shells = np.arange(1, N // 3, 1)

        solver = SpectralNS(N=N, Re=Re)

        # Use random IC (most physically relevant for phase coherence)
        u_hat = solver.random_ic(seed=42)

        step_start = clock.time()
        snapshots = {}

        for step in range(n_steps + 1):
            t = step * dt

            # Check if we should measure
            for tm in measure_times:
                if abs(t - tm) < dt / 2 and tm not in snapshots:
                    print(f"  t={t:.3f} — measuring R_K...")
                    spec = measure_coherence_spectrum(solver, u_hat, k_shells)
                    snapshots[tm] = spec

                    if len(spec['k']) > 0:
                        mean_Rc = np.mean(spec['R_cross'])
                        max_Rc = np.max(spec['R_cross'])
                        beta = fit_power_law(spec['k'], spec['R_cross'])
                        print(f"    R_cross: mean={mean_Rc:.4f}, max={max_Rc:.4f}, "
                              f"k^{{{beta:.2f}}}")

            if step < n_steps:
                u_hat = solver.step_rk4(u_hat, dt, mode='full')

        step_time = clock.time() - step_start
        print(f"  Wall time: {step_time:.1f}s")

        all_results[(Re, N)] = snapshots

    total_time = clock.time() - wall_start

    # ============================================================
    # ANALYSIS: R_K vs Re at final time
    # ============================================================
    print(f"\n\n{'='*70}")
    print("  SUMMARY: R_K vs Re (at t=3.0, Random IC)")
    print(f"{'='*70}")

    t_final = 3.0
    summary_rows = []

    for Re, N in configs:
        snap = all_results.get((Re, N), {}).get(t_final, None)
        if snap is None or len(snap['k']) == 0:
            print(f"  Re={Re:5d}: NO DATA")
            continue

        mean_Rc = np.mean(snap['R_cross'])
        max_Rc = np.max(snap['R_cross'])
        mean_Rs = np.mean(snap['R_same']) if np.any(snap['R_same'] > 0) else 0.0
        beta = fit_power_law(snap['k'], snap['R_cross'])
        ratio = mean_Rc / max(mean_Rs, 1e-10)

        # Near pi/2 fraction (recompute for final snapshot summary)
        summary_rows.append({
            'Re': Re, 'N': N,
            'mean_Rc': mean_Rc, 'max_Rc': max_Rc,
            'mean_Rs': mean_Rs, 'beta': beta, 'ratio': ratio,
        })

        print(f"  Re={Re:5d} (N={N:3d}): "
              f"<R_cross>={mean_Rc:.4f}  max={max_Rc:.4f}  "
              f"<R_same>={mean_Rs:.4f}  "
              f"R_c/R_s={ratio:.2f}  "
              f"beta={beta:+.2f}")

    # Does R_K grow with Re?
    if len(summary_rows) >= 2:
        Re_vals = np.array([r['Re'] for r in summary_rows])
        Rc_vals = np.array([r['mean_Rc'] for r in summary_rows])

        if np.all(Rc_vals > 0):
            Re_slope = np.polyfit(np.log(Re_vals), np.log(Rc_vals), 1)[0]
            print(f"\n  R_K(Re) scaling: R_K ~ Re^{{{Re_slope:+.3f}}}")
            if Re_slope > 0.1:
                print("  >>> DANGER: Phase coherence GROWS with Re!")
            elif Re_slope < -0.1:
                print("  >>> GOOD: Phase coherence DECREASES with Re")
            else:
                print("  >>> NEUTRAL: Phase coherence roughly constant with Re")

    # Does the k-scaling exponent change with Re?
    if len(summary_rows) >= 2:
        print(f"\n  k-scaling exponent vs Re:")
        for r in summary_rows:
            marker = " ← Buzzicotti" if abs(r['beta'] + 1) < 0.2 else ""
            print(f"    Re={r['Re']:5d}: beta = {r['beta']:+.2f}{marker}")

    # ============================================================
    # TEMPORAL EVOLUTION: does R_K grow during evolution?
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEMPORAL EVOLUTION OF R_K")
    print(f"{'='*70}")

    for Re, N in configs:
        snapshots = all_results.get((Re, N), {})
        if not snapshots:
            continue
        print(f"\n  Re={Re}:")
        for tm in sorted(snapshots.keys()):
            snap = snapshots[tm]
            if len(snap['k']) == 0:
                continue
            mean_Rc = np.mean(snap['R_cross'])
            beta = fit_power_law(snap['k'], snap['R_cross'])
            print(f"    t={tm:.1f}: <R_cross>={mean_Rc:.4f}, beta={beta:+.2f}")

    # ============================================================
    # VERDICT
    # ============================================================
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    print("""
  Key questions:
  1. Does <R_K> grow with Re? → If yes, incoherent bound (alpha=1/4) breaks at high Re
  2. Does k-exponent approach -1 (Buzzicotti)? → Phase coherence scaling
  3. Is R_cross/R_same > 1 at high Re? → Dangerous sector more coherent?
  4. Does R_K grow with time at high Re? → Finite-time coherence buildup?

  This is the critical test for our regularity program.
  If R_K stays bounded and small as Re→∞, alpha_eff ~ 1/4 holds.
  If R_K grows, we need a mechanism to control it.
""")

    print(f"  Total wall time: {total_time:.1f}s")

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Kuramoto Phase Coherence vs Reynolds Number (Random IC)', fontsize=13)

    colors_re = {400: 'blue', 800: 'green', 1600: 'orange', 3200: 'red'}

    # Panel 1: R_K(k) at final time for each Re
    ax = axes[0, 0]
    for Re, N in configs:
        snap = all_results.get((Re, N), {}).get(t_final)
        if snap is None or len(snap['k']) == 0:
            continue
        ax.plot(snap['k'], snap['R_cross'], 'o-', color=colors_re[Re],
                markersize=3, linewidth=1.5, label=f'Re={Re}')

    # Reference k^{-1}
    k_ref = np.arange(2, 30)
    ax.plot(k_ref, 0.15 / k_ref, 'k--', alpha=0.4, label=r'$k^{-1}$')
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel(r'$R_K^{+-}$ (cross-helical)')
    ax.set_title(f'Phase coherence spectrum (t={t_final})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: <R_K> vs Re
    ax = axes[0, 1]
    if summary_rows:
        Re_arr = [r['Re'] for r in summary_rows]
        Rc_arr = [r['mean_Rc'] for r in summary_rows]
        Rs_arr = [r['mean_Rs'] for r in summary_rows]
        ax.plot(Re_arr, Rc_arr, 'ro-', markersize=8, linewidth=2, label=r'$\langle R_K^{+-}\rangle$')
        ax.plot(Re_arr, Rs_arr, 'bs-', markersize=6, linewidth=1.5, label=r'$\langle R_K^{++}\rangle$')
        ax.set_xlabel('Reynolds number Re')
        ax.set_ylabel(r'$\langle R_K \rangle$')
        ax.set_title('Mean phase coherence vs Re')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 3: k-exponent vs Re
    ax = axes[1, 0]
    if summary_rows:
        Re_arr = [r['Re'] for r in summary_rows]
        beta_arr = [r['beta'] for r in summary_rows]
        ax.plot(Re_arr, beta_arr, 'ko-', markersize=8, linewidth=2)
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Buzzicotti: -1')
        ax.set_xlabel('Reynolds number Re')
        ax.set_ylabel(r'$\beta$ (R_K ~ k^{$\beta$})')
        ax.set_title('k-scaling exponent vs Re')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 4: R_cross/R_same vs Re
    ax = axes[1, 1]
    if summary_rows:
        Re_arr = [r['Re'] for r in summary_rows]
        ratio_arr = [r['ratio'] for r in summary_rows]
        ax.plot(Re_arr, ratio_arr, 'go-', markersize=8, linewidth=2)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Reynolds number Re')
        ax.set_ylabel(r'$R_{cross} / R_{same}$')
        ax.set_title('Cross/Same coherence ratio vs Re')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'kuramoto_re_sweep.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
