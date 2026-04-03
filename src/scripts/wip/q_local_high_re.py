"""
q_local AT HIGH Re — CRITICAL C-F BRIDGE TEST
================================================
Meridian 1 (S98) resolved: R_local IS the correct C-F measure.
Two data points so far:
  Re=400:  q_local = 1.60 (above 7/6)
  Re=1600: q_local = 1.17 (marginal)

Extrapolation: q ~ Re^{-0.23} → q_local ≈ 1.0 at Re=3200.
If actual q_local > 1.1, the bridge holds further than expected.

This script measures q_local at Re=3200 (N=96) and Re=6400 (N=128).
Also measures:
  - Angular distribution vs k-shell
  - q_local using different k-range cutoffs

Single most important measurement. Honest reporting.

Meridian 2, S98.
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

BIN_PARALLEL = np.pi / 6
BIN_ANTIPARALLEL = 5 * np.pi / 6


def compute_phases_with_angles(solver, u_hat, k_shell, dk=1.0, max_k3=150, max_k1=80):
    """Compute cross-helical triadic phases with inter-wavevector angles."""
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
    }


def measure_at_shell(data):
    """Compute R_local, R_raw, angular fractions for one shell."""
    if data is None:
        return None

    phases = data['phases']
    thetas = data['thetas']
    sin2t = np.sin(thetas)**2

    mask_par = thetas < BIN_PARALLEL
    mask_anti = thetas > BIN_ANTIPARALLEL
    mask_local = ~mask_par & ~mask_anti

    n_total = len(phases)
    n_local = int(np.sum(mask_local))
    n_anti = int(np.sum(mask_anti))

    R_raw = abs(np.mean(np.exp(1j * phases))) if n_total > 0 else 0.0

    if n_local > 5:
        R_local = abs(np.mean(np.exp(1j * phases[mask_local])))
    else:
        R_local = float('nan')

    # Leray-weighted
    if n_total > 0 and np.sum(sin2t) > 1e-15:
        R_leray = abs(np.mean(sin2t * np.exp(1j * phases))) / np.mean(sin2t)
    else:
        R_leray = 0.0

    # Local fraction
    frac_local = n_local / max(n_total, 1)
    frac_anti = n_anti / max(n_total, 1)

    return {
        'n_total': n_total, 'n_local': n_local,
        'R_raw': R_raw, 'R_local': R_local, 'R_leray': R_leray,
        'frac_local': frac_local, 'frac_anti': frac_anti,
    }


def fit_power_law(k, R, min_pts=3):
    """Fit R ~ k^beta. Returns -beta (so positive = decay)."""
    valid = np.isfinite(R) & (R > 0) & (k > 0)
    if np.sum(valid) < min_pts:
        return float('nan')
    slope, _ = np.polyfit(np.log(k[valid]), np.log(R[valid]), 1)
    return -slope  # return q where R ~ k^{-q}


def run_single_re(Re, N, T=2.0):
    """Run one Re value: evolve to T, measure q_local."""
    dt = 0.005 * (32 / N)
    n_steps = int(T / dt)
    k_shells = np.arange(2, N // 3, 1)  # start from k=2

    print(f"\n  Re={Re}, N={N}, T={T}, dt={dt:.5f}, steps={n_steps}")
    solver = SpectralNS(N=N, Re=Re)
    u_hat = solver.random_ic(seed=42)

    # Evolve
    t0 = clock.time()
    for step in range(n_steps):
        u_hat = solver.step_rk4(u_hat, dt, mode='full')
        if step % (n_steps // 5) == 0:
            print(f"    step {step}/{n_steps} ({100*step/n_steps:.0f}%)")
    evolve_time = clock.time() - t0
    print(f"  Evolution: {evolve_time:.1f}s")

    # Measure across shells
    print(f"  Measuring angular-binned R_K...")
    t0 = clock.time()
    results = []
    for k in k_shells:
        data = compute_phases_with_angles(solver, u_hat, k)
        r = measure_at_shell(data)
        if r is not None and r['n_total'] > 10:
            r['k'] = k
            results.append(r)
    measure_time = clock.time() - t0
    print(f"  Measurement: {measure_time:.1f}s")

    if not results:
        print("  NO DATA!")
        return None

    ks = np.array([r['k'] for r in results])
    R_local = np.array([r['R_local'] for r in results])
    R_raw = np.array([r['R_raw'] for r in results])
    R_leray = np.array([r['R_leray'] for r in results])

    # q_local (full range)
    q_local = fit_power_law(ks, R_local)
    q_raw = fit_power_law(ks, R_raw)
    q_leray = fit_power_law(ks, R_leray)

    print(f"\n  RESULTS:")
    print(f"    q_local = {q_local:.3f}  (C-F threshold = 1.167)")
    print(f"    q_raw   = {q_raw:.3f}")
    print(f"    q_leray = {q_leray:.3f}")
    print(f"    <R_local> = {np.nanmean(R_local):.4f}")
    print(f"    <R_raw>   = {np.mean(R_raw):.4f}")

    # q_local with different k-cutoffs
    print(f"\n  q_local vs k-range:")
    for k_min in [2, 4, 6, 8]:
        mask = ks >= k_min
        if np.sum(mask & np.isfinite(R_local)) >= 3:
            q = fit_power_law(ks[mask], R_local[mask])
            print(f"    k >= {k_min}: q_local = {q:.3f}")

    # Angular distribution vs k
    print(f"\n  Angular distribution vs k:")
    print(f"  {'k':>3s}  {'frac_local':>10s}  {'frac_anti':>10s}  {'R_local':>8s}  {'R_raw':>8s}  {'n':>5s}")
    for r in results:
        print(f"  {r['k']:3.0f}  {r['frac_local']:10.3f}  {r['frac_anti']:10.3f}  "
              f"{r['R_local']:8.4f}  {r['R_raw']:8.4f}  {r['n_total']:5d}")

    return {
        'Re': Re, 'N': N,
        'q_local': q_local, 'q_raw': q_raw, 'q_leray': q_leray,
        'ks': ks, 'R_local': R_local, 'R_raw': R_raw, 'R_leray': R_leray,
        'results': results,
    }


def main():
    print("=" * 70)
    print("  q_local AT HIGH Re — CRITICAL C-F BRIDGE TEST")
    print("=" * 70)
    print()
    print("  Prior data: Re=400 q=1.60, Re=1600 q=1.17")
    print("  Extrapolation: q ~ Re^{-0.23} → q ≈ 1.0 at Re=3200")
    print("  C-F threshold: q > 7/6 = 1.167")
    print()

    wall_start = clock.time()
    all_results = {}

    # Re=3200, N=96 — the critical test
    r = run_single_re(3200, 96, T=2.0)
    if r:
        all_results[3200] = r

    # Re=6400, N=128 — if feasible (will be slow)
    # N=128 has 128^3 = 2M grid points, ~8x slower than N=96
    # Try with shorter T
    print(f"\n  Attempting Re=6400, N=128 (T=1.0 for speed)...")
    r = run_single_re(6400, 128, T=1.0)
    if r:
        all_results[6400] = r

    total_time = clock.time() - wall_start

    # ============================================================
    # COMBINED RESULTS
    # ============================================================
    print(f"\n{'='*70}")
    print("  COMBINED q_local vs Re (including prior data)")
    print(f"{'='*70}")

    # Include prior data points
    all_q = [
        (400, 1.60),
        (1600, 1.17),
    ]
    for Re, data in all_results.items():
        all_q.append((Re, data['q_local']))

    all_q.sort()

    print(f"\n  {'Re':>6s}  {'q_local':>8s}  {'vs 7/6':>8s}  {'Status':>12s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*12}")
    for Re, q in all_q:
        diff = q - 7/6
        status = "CLOSED" if q > 7/6 else "MARGINAL" if abs(diff) < 0.05 else "OPEN"
        print(f"  {Re:6d}  {q:8.3f}  {diff:+8.3f}  {status:>12s}")

    # Fit Re-dependence
    Re_arr = np.array([r for r, q in all_q])
    q_arr = np.array([q for r, q in all_q])

    if len(q_arr) >= 3 and np.all(q_arr > 0):
        # Fit q = A * Re^gamma
        valid = q_arr > 0
        slope, intercept = np.polyfit(np.log(Re_arr[valid]), np.log(q_arr[valid]), 1)
        A = np.exp(intercept)
        print(f"\n  Power law fit: q_local ≈ {A:.2f} × Re^{{{slope:+.3f}}}")

        # Predict threshold Re
        # 7/6 = A * Re_crit^slope → Re_crit = (7/(6A))^(1/slope)
        if slope < 0:
            Re_crit = (7 / (6 * A))**(1 / slope)
            print(f"  Predicted Re where q = 7/6: Re ≈ {Re_crit:.0f}")
        else:
            print(f"  q increases with Re — gap stays closed")

    # ============================================================
    # VERDICT
    # ============================================================
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    if all_results.get(3200):
        q3200 = all_results[3200]['q_local']
        if q3200 > 7/6:
            print(f"\n  q_local(Re=3200) = {q3200:.3f} > 7/6")
            print("  >>> C-F BRIDGE HOLDS at Re=3200! Better than predicted.")
        elif q3200 > 1.0:
            print(f"\n  q_local(Re=3200) = {q3200:.3f}")
            print(f"  >>> Below C-F threshold by {7/6 - q3200:.3f}")
            print("  >>> Bridge is failing at high Re, consistent with Iyer et al.")
        else:
            print(f"\n  q_local(Re=3200) = {q3200:.3f}")
            print("  >>> Well below threshold. C-F bridge is a moderate-Re phenomenon.")

    print(f"\n  Total wall time: {total_time:.1f}s")

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('q_local vs Re — C-F Bridge Critical Test', fontsize=13)

    # Panel 1: R_local(k) at each Re
    ax = axes[0]
    colors = {3200: 'red', 6400: 'darkred'}
    for Re, data in all_results.items():
        valid = np.isfinite(data['R_local'])
        ax.plot(data['ks'][valid], data['R_local'][valid], 'o-',
                color=colors.get(Re, 'black'), markersize=3, linewidth=1.5,
                label=f'Re={Re} (q={data["q_local"]:.2f})')
    k_ref = np.arange(2, 30)
    ax.plot(k_ref, 0.05 * k_ref**(-7/6), 'k--', alpha=0.4, label=r'$k^{-7/6}$')
    ax.set_xlabel('k')
    ax.set_ylabel(r'$R_K^{local}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Local-triad phase coherence')

    # Panel 2: q_local vs Re (all data points)
    ax = axes[1]
    Re_plot = [r for r, q in all_q]
    q_plot = [q for r, q in all_q]
    ax.plot(Re_plot, q_plot, 'ro-', markersize=10, linewidth=2)
    ax.axhline(y=7/6, color='blue', linestyle='--', linewidth=2, label='C-F threshold 7/6')
    ax.set_xlabel('Re')
    ax.set_ylabel('q_local')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('q_local vs Re')

    # Panel 3: Local fraction vs k
    ax = axes[2]
    for Re, data in all_results.items():
        frac = [r['frac_local'] for r in data['results']]
        ks = [r['k'] for r in data['results']]
        ax.plot(ks, frac, 'o-', color=colors.get(Re, 'black'),
                markersize=3, linewidth=1.5, label=f'Re={Re}')
    ax.set_xlabel('k')
    ax.set_ylabel('Fraction of local triads')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Local-in-scale triad fraction vs k')

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'q_local_high_re.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
