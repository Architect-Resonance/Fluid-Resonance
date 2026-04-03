"""
HELICAL IMBALANCE σ²(k) — APPROACH (f4)
=========================================
Biferale-Titi (2013): single helical sector → global regularity.
Combined suppression: α_total = sin²θ/4 × (1 − σ²(k))

σ(k) = H(k) / [k E(k)]  where H(k) = helicity spectrum, E(k) = energy spectrum.
|σ(k)| ≤ 1 (realizability).

If ⟨σ²(k)⟩ > 0 at high k, the helical protection factor (1−σ²) < 1 gives
EXTRA suppression beyond sin²θ/4 alone.

Key question: does σ(k) → 0 at high k (Kraichnan 1973), or is there residual
imbalance that helps regularity?

This script measures:
1. σ(k) = H(k)/[kE(k)] at each shell for multiple Re
2. σ²(k) profile — the variance of helical imbalance
3. Time evolution of σ(k) — does it decay or persist?
4. Combined suppression α_total(k) = 1/4 × (1 − σ²(k))

Re = [400, 800, 1600, 3200], multiple time snapshots.

Meridian 2, S98 — Approach (f4).
"""

import numpy as np
from numpy.fft import fftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time as clock

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def measure_helical_spectra(solver, u_hat):
    """Compute E(k), H(k), and σ(k) = H(k)/[kE(k)] per shell.

    Returns dict with arrays indexed by shell k = 1, 2, ..., kmax.
    """
    u_p, u_m = solver.helical_decompose(u_hat)

    # Energy in each helical mode: |h±(k)|²
    Ep = np.abs(u_p)**2
    Em = np.abs(u_m)**2

    kmax = int(solver.N // 2)
    shells = np.arange(1, kmax + 1)
    E_k = np.zeros(kmax)
    H_k = np.zeros(kmax)
    Ep_k = np.zeros(kmax)
    Em_k = np.zeros(kmax)

    for i, ks in enumerate(shells):
        mask = (solver.kmag >= ks - 0.5) & (solver.kmag < ks + 0.5)
        Ep_k[i] = np.sum(Ep[mask])
        Em_k[i] = np.sum(Em[mask])

    E_k = Ep_k + Em_k  # total energy per shell
    H_k = shells * (Ep_k - Em_k)  # helicity spectrum: k(|h+|² - |h-|²)

    # Relative helicity σ(k) = H(k) / [k E(k)]
    sigma_k = np.zeros(kmax)
    valid = E_k > 1e-30
    sigma_k[valid] = H_k[valid] / (shells[valid] * E_k[valid])
    sigma_k = np.clip(sigma_k, -1, 1)

    # Combined suppression
    alpha_total = 0.25 * (1 - sigma_k**2)

    return {
        'shells': shells,
        'E_k': E_k,
        'H_k': H_k,
        'Ep_k': Ep_k,
        'Em_k': Em_k,
        'sigma_k': sigma_k,
        'sigma2_k': sigma_k**2,
        'alpha_total': alpha_total,
    }


def run_measurement(Re, N, T_snapshots, dt_factor=0.5):
    """Evolve NS at given Re and measure σ(k) at multiple time snapshots."""
    solver = SpectralNS(N=N, Re=Re)

    # CFL-adjusted dt
    dt = dt_factor / (N * Re**0.5)
    dt = min(dt, 0.005)
    dt = max(dt, 0.0005)

    # TG initial condition
    u_hat = solver.taylor_green_ic()

    T_max = max(T_snapshots)
    steps_total = int(T_max / dt)

    results = []
    snapshot_steps = [int(t / dt) for t in T_snapshots]
    snapshot_set = set(snapshot_steps)

    print(f"\n  Re={Re}, N={N}, dt={dt:.5f}, T_max={T_max}, steps={steps_total}")

    t0 = clock.time()
    for step in range(steps_total + 1):
        if step in snapshot_set:
            t_current = step * dt
            spectra = measure_helical_spectra(solver, u_hat)
            spectra['t'] = t_current
            spectra['Re'] = Re
            results.append(spectra)
            print(f"    t={t_current:.2f}: <σ²>={np.mean(spectra['sigma2_k'][:N//3]):.4f}, "
                  f"σ_max={np.max(np.abs(spectra['sigma_k'][:N//3])):.4f}, "
                  f"<α_total>={np.mean(spectra['alpha_total'][:N//3]):.4f}")

        if step < steps_total:
            u_hat = solver.step_rk4(u_hat, dt)

        if step > 0 and step % (steps_total // 5) == 0:
            elapsed = clock.time() - t0
            print(f"    step {step}/{steps_total} ({100*step/steps_total:.0f}%) — {elapsed:.0f}s")

    elapsed = clock.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return results


def main():
    print("=" * 70)
    print("  HELICAL IMBALANCE σ²(k) — APPROACH (f4)")
    print("  Combined suppression: α_total = sin²θ/4 × (1 − σ²(k))")
    print("  Biferale-Titi endpoint: σ=±1 → regularity proved")
    print("=" * 70)

    # Configuration: Re → (N, T_snapshots)
    configs = [
        (400,  32, [0.5, 1.0, 1.5, 2.0]),
        (800,  48, [0.5, 1.0, 1.5, 2.0]),
        (1600, 64, [0.5, 1.0, 1.5, 2.0]),
        (3200, 96, [0.5, 1.0, 2.0]),
    ]

    all_results = {}

    for Re, N, T_snaps in configs:
        results = run_measurement(Re, N, T_snaps)
        all_results[Re] = results

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    # Table: σ²(k) averaged over inertial range, per Re and time
    print("\n  σ²(k) averaged over inertial range (k=3 to N/3):")
    print(f"  {'Re':>6}  {'t':>5}  {'<σ²>':>8}  {'<|σ|>':>8}  {'<α_total>':>10}  {'α_gain':>8}")
    print("  " + "-" * 58)

    for Re, N, T_snaps in configs:
        for snap in all_results[Re]:
            kmax_inertial = max(3, len(snap['sigma_k']) // 3)
            inertial = slice(2, kmax_inertial)  # k=3 to N/3
            sigma2_avg = np.mean(snap['sigma2_k'][inertial])
            sigma_abs_avg = np.mean(np.abs(snap['sigma_k'][inertial]))
            alpha_avg = np.mean(snap['alpha_total'][inertial])
            # α_gain: how much better than sin²θ/4 = 0.25 alone
            gain = (0.25 - alpha_avg) / 0.25 * 100
            print(f"  {Re:>6}  {snap['t']:>5.1f}  {sigma2_avg:>8.4f}  {sigma_abs_avg:>8.4f}  "
                  f"{alpha_avg:>10.4f}  {gain:>7.1f}%")

    # Profile: σ²(k) vs k at latest time for each Re
    print("\n  σ²(k) profile at latest snapshot:")
    print(f"  {'k':>4}", end="")
    for Re, N, T_snaps in configs:
        print(f"  {'Re='+str(Re):>10}", end="")
    print()

    max_shells = max(len(all_results[Re][-1]['shells']) for Re, _, _ in configs)
    for ki in range(min(30, max_shells)):
        k_val = ki + 1
        print(f"  {k_val:>4}", end="")
        for Re, N, T_snaps in configs:
            snap = all_results[Re][-1]
            if ki < len(snap['sigma2_k']):
                print(f"  {snap['sigma2_k'][ki]:>10.4f}", end="")
            else:
                print(f"  {'---':>10}", end="")
        print()

    # ================================================================
    # KEY QUESTION: Does σ²(k) → 0 at high k?
    # ================================================================
    print("\n" + "=" * 70)
    print("  KEY QUESTION: σ²(k) scaling at high k")
    print("=" * 70)

    for Re, N, T_snaps in configs:
        snap = all_results[Re][-1]  # latest time
        k_inertial = snap['shells'][2:N//3]
        s2_inertial = snap['sigma2_k'][2:N//3]

        if len(k_inertial) > 3 and np.all(s2_inertial > 1e-10):
            # Fit log(σ²) vs log(k)
            logk = np.log(k_inertial)
            logs2 = np.log(s2_inertial)
            valid = np.isfinite(logs2)
            if np.sum(valid) > 3:
                p = np.polyfit(logk[valid], logs2[valid], 1)
                print(f"  Re={Re}: σ²(k) ~ k^{p[0]:.2f}")
            else:
                print(f"  Re={Re}: insufficient valid data for fit")
        else:
            # Some σ² values are near zero
            mean_s2 = np.mean(s2_inertial)
            print(f"  Re={Re}: <σ²> = {mean_s2:.4f} (some shells near zero)")

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Use Re=1600 latest snapshot as representative
    snap = all_results[1600][-1]
    kmax_i = len(snap['sigma_k']) // 3
    s2_high = np.mean(snap['sigma2_k'][kmax_i//2:kmax_i])
    s2_low = np.mean(snap['sigma2_k'][2:kmax_i//2])
    alpha_high = np.mean(snap['alpha_total'][kmax_i//2:kmax_i])

    print(f"\n  Representative (Re=1600, t={snap['t']:.1f}):")
    print(f"    <σ²> low-k (3 to {kmax_i//2}):  {s2_low:.4f}")
    print(f"    <σ²> high-k ({kmax_i//2} to {kmax_i}): {s2_high:.4f}")
    print(f"    <α_total> high-k:        {alpha_high:.4f}")
    print(f"    sin²θ/4 alone:           0.2500")
    print(f"    Extra suppression:        {(0.25 - alpha_high)/0.25*100:.1f}%")

    if s2_high > 0.01:
        print(f"\n  >>> σ²(k) > 0.01 at high k — helical protection IS present!")
        print(f"  >>> Combined α_total < 0.25 — the Biferale-Titi factor helps.")
    else:
        print(f"\n  >>> σ²(k) ≈ 0 at high k — Kraichnan was right.")
        print(f"  >>> Helical protection vanishes. Only sin²θ/4 remains.")

    # ================================================================
    # PLOT
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Helical Imbalance σ²(k) — Approach (f4)', fontsize=14, fontweight='bold')

    colors = {400: '#2196F3', 800: '#4CAF50', 1600: '#FF9800', 3200: '#F44336'}

    # Panel 1: σ(k) profile at latest time
    ax = axes[0, 0]
    for Re, N, T_snaps in configs:
        snap = all_results[Re][-1]
        kmax_plot = N // 3
        ax.plot(snap['shells'][:kmax_plot], snap['sigma_k'][:kmax_plot],
                'o-', color=colors[Re], markersize=2, label=f'Re={Re}', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(1, color='red', linestyle=':', alpha=0.3, label='Beltrami (σ=±1)')
    ax.axhline(-1, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('k')
    ax.set_ylabel('σ(k) = H(k)/[kE(k)]')
    ax.set_title('Relative helicity σ(k)')
    ax.legend(fontsize=8)
    ax.set_ylim(-1.1, 1.1)

    # Panel 2: σ²(k) profile (log-log)
    ax = axes[0, 1]
    for Re, N, T_snaps in configs:
        snap = all_results[Re][-1]
        kmax_plot = N // 3
        s2 = snap['sigma2_k'][:kmax_plot]
        valid = s2 > 1e-10
        ax.loglog(snap['shells'][:kmax_plot][valid], s2[valid],
                  'o-', color=colors[Re], markersize=2, label=f'Re={Re}', alpha=0.8)
    ax.axhline(1, color='red', linestyle=':', alpha=0.3, label='σ²=1 (Beltrami)')
    ax.set_xlabel('k')
    ax.set_ylabel('σ²(k)')
    ax.set_title('Helical imbalance variance σ²(k)')
    ax.legend(fontsize=8)

    # Panel 3: α_total(k) = 1/4 × (1−σ²)
    ax = axes[1, 0]
    for Re, N, T_snaps in configs:
        snap = all_results[Re][-1]
        kmax_plot = N // 3
        ax.plot(snap['shells'][:kmax_plot], snap['alpha_total'][:kmax_plot],
                'o-', color=colors[Re], markersize=2, label=f'Re={Re}', alpha=0.8)
    ax.axhline(0.25, color='black', linestyle='--', alpha=0.5, label='sin²θ/4 only (σ=0)')
    ax.axhline(0, color='red', linestyle=':', alpha=0.3, label='σ=±1 (full protection)')
    ax.set_xlabel('k')
    ax.set_ylabel('α_total(k)')
    ax.set_title('Combined suppression α_total = ¼(1−σ²)')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.01, 0.30)

    # Panel 4: σ²(k) time evolution at Re=1600
    ax = axes[1, 1]
    if 1600 in all_results:
        for snap in all_results[1600]:
            kmax_plot = 64 // 3
            ax.semilogy(snap['shells'][:kmax_plot], snap['sigma2_k'][:kmax_plot],
                        'o-', markersize=2, label=f"t={snap['t']:.1f}", alpha=0.7)
    ax.set_xlabel('k')
    ax.set_ylabel('σ²(k)')
    ax.set_title('σ²(k) time evolution (Re=1600)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'helical_imbalance_f4.png')
    plt.savefig(outpath, dpi=150)
    print(f"\n  Plot saved to {outpath}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
