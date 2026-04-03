"""
HELICAL IMBALANCE f4b — ABC & IMBALANCED ICs
=============================================
Follow-up to f4 (TG gave sigma^2 = 0 by mirror symmetry).
Now test with HELICAL initial conditions:
  1. Imbalanced random IC (h_plus_frac = 0.8) — sigma(k=1) ~ 0.6
  2. ABC flow IC (A=B=C=1) — maximally helical (sigma=1 at low k)

Key measurement: sigma^2(k) decay in inertial range.
Kraichnan (1973) predicts sigma(k) -> 0 at high k.
If sigma^2(k) ~ k^{-p}, what is p?

Re = [400, 800, 1600], multiple time snapshots.
Meridian 2, S100.
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


def abc_flow_ic(solver, A=1.0, B=1.0, C=1.0):
    """Arnold-Beltrami-Childress flow: maximally helical (eigenfunction of curl).

    u = (B cos(y) + C sin(z), C cos(z) + A sin(x), A cos(x) + B sin(y))
    curl(u) = u (Beltrami property), so sigma(k=1) = 1 exactly.
    """
    N = solver.N
    L = 2 * np.pi
    dx = L / N
    x = np.arange(N) * dx
    y = np.arange(N) * dx
    z = np.arange(N) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    ux = B * np.cos(Y) + C * np.sin(Z)
    uy = C * np.cos(Z) + A * np.sin(X)
    uz = A * np.cos(X) + B * np.sin(Y)

    u_hat = np.zeros((3, N, N, N), dtype=complex)
    u_hat[0] = fftn(ux)
    u_hat[1] = fftn(uy)
    u_hat[2] = fftn(uz)

    # Already solenoidal by construction, but project to be safe
    u_hat = solver.project_leray(u_hat)

    # Normalize energy to 0.5
    u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    E = 0.5 * np.mean(np.sum(u**2, axis=0))
    u_hat *= np.sqrt(0.5 / max(E, 1e-15))

    return u_hat


def measure_helical_spectra(solver, u_hat):
    """Compute E(k), H(k), sigma(k) per shell. Identical to f4."""
    u_p, u_m = solver.helical_decompose(u_hat)

    Ep = np.abs(u_p)**2
    Em = np.abs(u_m)**2

    kmax = int(solver.N // 2)
    shells = np.arange(1, kmax + 1)
    E_k = np.zeros(kmax)
    Ep_k = np.zeros(kmax)
    Em_k = np.zeros(kmax)

    for i, ks in enumerate(shells):
        mask = (solver.kmag >= ks - 0.5) & (solver.kmag < ks + 0.5)
        Ep_k[i] = np.sum(Ep[mask])
        Em_k[i] = np.sum(Em[mask])

    E_k = Ep_k + Em_k
    H_k = shells * (Ep_k - Em_k)

    sigma_k = np.zeros(kmax)
    valid = E_k > 1e-30
    sigma_k[valid] = H_k[valid] / (shells[valid] * E_k[valid])
    sigma_k = np.clip(sigma_k, -1, 1)

    alpha_total = 0.25 * (1 - sigma_k**2)

    return {
        'shells': shells,
        'E_k': E_k,
        'sigma_k': sigma_k,
        'sigma2_k': sigma_k**2,
        'alpha_total': alpha_total,
    }


def run_measurement(solver, u_hat, label, Re, N, T_snapshots, dt_factor=0.5):
    """Evolve and measure sigma(k) at snapshots."""
    dt = dt_factor / (N * Re**0.5)
    dt = min(dt, 0.005)
    dt = max(dt, 0.0005)

    T_max = max(T_snapshots)
    steps_total = int(T_max / dt)

    results = []
    snapshot_steps = {int(t / dt): t for t in T_snapshots}

    print(f"\n  [{label}] Re={Re}, N={N}, dt={dt:.5f}, steps={steps_total}")

    # Measure t=0
    spec0 = measure_helical_spectra(solver, u_hat)
    spec0['t'] = 0.0
    spec0['Re'] = Re
    spec0['label'] = label
    results.append(spec0)
    kmax_i = N // 3
    print(f"    t=0.00: <sigma2>={np.mean(spec0['sigma2_k'][:kmax_i]):.6f}, "
          f"sigma_max={np.max(np.abs(spec0['sigma_k'][:kmax_i])):.4f}")

    t0 = clock.time()
    for step in range(1, steps_total + 1):
        u_hat = solver.step_rk4(u_hat, dt)

        if step in snapshot_steps:
            t_current = snapshot_steps[step]
            spec = measure_helical_spectra(solver, u_hat)
            spec['t'] = t_current
            spec['Re'] = Re
            spec['label'] = label
            results.append(spec)
            print(f"    t={t_current:.2f}: <sigma2>={np.mean(spec['sigma2_k'][:kmax_i]):.6f}, "
                  f"sigma_max={np.max(np.abs(spec['sigma_k'][:kmax_i])):.4f}, "
                  f"<alpha_total>={np.mean(spec['alpha_total'][:kmax_i]):.4f}")

        if step > 0 and step % max(1, steps_total // 5) == 0:
            elapsed = clock.time() - t0
            print(f"    step {step}/{steps_total} ({100*step/steps_total:.0f}%) -- {elapsed:.0f}s")

    elapsed = clock.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return results


def fit_sigma2_decay(shells, sigma2_k, k_range):
    """Fit log(sigma2) vs log(k) in k_range. Return exponent p (sigma2 ~ k^{-p})."""
    mask = (shells >= k_range[0]) & (shells <= k_range[1]) & (sigma2_k > 1e-12)
    if np.sum(mask) < 3:
        return None, None
    logk = np.log(shells[mask])
    logs2 = np.log(sigma2_k[mask])
    p = np.polyfit(logk, logs2, 1)
    return -p[0], p[1]  # exponent (positive = decay), intercept


def main():
    print("=" * 70)
    print("  HELICAL IMBALANCE f4b -- ABC & IMBALANCED ICs")
    print("  Key question: sigma2(k) ~ k^{-p}. What is p?")
    print("  Kraichnan predicts p = 4/3.")
    print("=" * 70)

    configs = [
        (400,  32, [0.5, 1.0, 2.0]),
        (800,  48, [0.5, 1.0, 2.0]),
        (1600, 64, [0.5, 1.0, 2.0]),
    ]

    all_results = {}

    for Re, N, T_snaps in configs:
        solver = SpectralNS(N=N, Re=Re)

        # IC 1: Imbalanced random (h_plus_frac=0.8)
        u_hat_imb = solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8)
        key_imb = f'Imb_Re{Re}'
        all_results[key_imb] = run_measurement(solver, u_hat_imb, 'Imbalanced', Re, N, T_snaps)

        # IC 2: ABC flow (maximally helical)
        u_hat_abc = abc_flow_ic(solver)
        key_abc = f'ABC_Re{Re}'
        all_results[key_abc] = run_measurement(solver, u_hat_abc, 'ABC', Re, N, T_snaps)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'IC':>12}  {'Re':>5}  {'t':>4}  {'<sigma2>':>10}  {'sigma_max':>10}  "
          f"{'<alpha>':>8}  {'gain%':>6}")
    print("  " + "-" * 70)

    for Re, N, T_snaps in configs:
        for ic_label, key in [('Imbalanced', f'Imb_Re{Re}'), ('ABC', f'ABC_Re{Re}')]:
            for snap in all_results[key]:
                kmax_i = N // 3
                s2_avg = np.mean(snap['sigma2_k'][2:kmax_i])
                s_max = np.max(np.abs(snap['sigma_k'][2:kmax_i]))
                a_avg = np.mean(snap['alpha_total'][2:kmax_i])
                gain = (0.25 - a_avg) / 0.25 * 100
                print(f"  {ic_label:>12}  {Re:>5}  {snap['t']:>4.1f}  {s2_avg:>10.6f}  "
                      f"{s_max:>10.4f}  {a_avg:>8.4f}  {gain:>5.1f}%")

    # ================================================================
    # POWER LAW FIT: sigma2(k) ~ k^{-p}
    # ================================================================
    print("\n" + "=" * 70)
    print("  POWER LAW FIT: sigma2(k) ~ k^{-p} (inertial range)")
    print("=" * 70)

    print(f"\n  {'IC':>12}  {'Re':>5}  {'t':>4}  {'p':>8}  {'Kraichnan':>10}")
    print("  " + "-" * 50)

    for Re, N, T_snaps in configs:
        kmax_i = N // 3
        k_fit_range = (3, kmax_i)

        for ic_label, key in [('Imbalanced', f'Imb_Re{Re}'), ('ABC', f'ABC_Re{Re}')]:
            for snap in all_results[key]:
                p, _ = fit_sigma2_decay(snap['shells'], snap['sigma2_k'], k_fit_range)
                if p is not None:
                    print(f"  {ic_label:>12}  {Re:>5}  {snap['t']:>4.1f}  {p:>8.3f}  {'4/3=1.333':>10}")
                else:
                    print(f"  {ic_label:>12}  {Re:>5}  {snap['t']:>4.1f}  {'N/A':>8}  {'4/3=1.333':>10}")

    # ================================================================
    # sigma(k) PROFILE at latest time
    # ================================================================
    print("\n" + "=" * 70)
    print("  sigma(k) PROFILE at latest time")
    print("=" * 70)

    # Show ABC profiles
    print(f"\n  ABC flow sigma(k) at t=2.0:")
    print(f"  {'k':>4}", end="")
    for Re, N, _ in configs:
        print(f"  {'Re='+str(Re):>10}", end="")
    print()

    max_k = max(N // 3 for _, N, _ in configs)
    for ki in range(min(30, max_k)):
        k_val = ki + 1
        print(f"  {k_val:>4}", end="")
        for Re, N, _ in configs:
            key = f'ABC_Re{Re}'
            snap = all_results[key][-1]
            if ki < len(snap['sigma_k']):
                print(f"  {snap['sigma_k'][ki]:>10.4f}", end="")
            else:
                print(f"  {'---':>10}", end="")
        print()

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Use Re=800 ABC as representative
    key = 'ABC_Re800'
    snap_latest = all_results[key][-1]
    snap_t0 = all_results[key][0]
    kmax_i = 48 // 3

    s2_t0 = np.mean(snap_t0['sigma2_k'][2:kmax_i])
    s2_late = np.mean(snap_latest['sigma2_k'][2:kmax_i])
    alpha_late = np.mean(snap_latest['alpha_total'][2:kmax_i])

    p_abc, _ = fit_sigma2_decay(snap_latest['shells'], snap_latest['sigma2_k'], (3, kmax_i))

    print(f"\n  ABC flow, Re=800, t={snap_latest['t']:.1f}:")
    print(f"    <sigma2> at t=0:   {s2_t0:.6f}")
    print(f"    <sigma2> at t=2:   {s2_late:.6f}")
    print(f"    Decay ratio:       {s2_late / max(s2_t0, 1e-30):.4f}")
    print(f"    <alpha_total>:     {alpha_late:.4f}")
    print(f"    sin2theta/4 alone: 0.2500")
    if p_abc is not None:
        print(f"    p (sigma2 ~ k^-p): {p_abc:.3f}  (Kraichnan: 4/3 = 1.333)")
    print(f"    Extra suppression: {(0.25 - alpha_late)/0.25*100:.1f}%")

    if alpha_late < 0.20:
        print(f"\n  >>> Helical protection IS significant! alpha_total = {alpha_late:.4f} < 0.25")
        print(f"  >>> The Biferale-Titi factor (1-sigma2) provides {(0.25-alpha_late)/0.25*100:.0f}% extra suppression.")
    elif alpha_late < 0.24:
        print(f"\n  >>> Moderate helical protection. alpha_total = {alpha_late:.4f}")
    else:
        print(f"\n  >>> Minimal helical protection. alpha_total = {alpha_late:.4f} near 0.25")
        print(f"  >>> sigma2 decays too fast — Kraichnan wins again.")

    # ================================================================
    # PLOT
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('f4b: Helical Imbalance with ABC & Imbalanced ICs', fontsize=14, fontweight='bold')

    colors_re = {400: '#2196F3', 800: '#4CAF50', 1600: '#FF9800'}

    # Panel 1: sigma(k) ABC at various Re (latest time)
    ax = axes[0, 0]
    for Re, N, _ in configs:
        snap = all_results[f'ABC_Re{Re}'][-1]
        kp = N // 3
        ax.plot(snap['shells'][:kp], snap['sigma_k'][:kp],
                'o-', color=colors_re[Re], markersize=2, label=f'Re={Re}', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('sigma(k)')
    ax.set_title(f'ABC: sigma(k) at t={T_snaps[-1]}')
    ax.legend(fontsize=8)

    # Panel 2: sigma2(k) log-log with Kraichnan slope
    ax = axes[0, 1]
    for Re, N, _ in configs:
        snap = all_results[f'ABC_Re{Re}'][-1]
        kp = N // 3
        s2 = snap['sigma2_k'][:kp]
        valid = s2 > 1e-12
        if np.any(valid):
            ax.loglog(snap['shells'][:kp][valid], s2[valid],
                      'o-', color=colors_re[Re], markersize=2, label=f'Re={Re}', alpha=0.8)
    # Kraichnan k^{-4/3} reference
    k_ref = np.arange(2, 20)
    ax.loglog(k_ref, 0.5 * k_ref**(-4/3), 'k--', linewidth=1, alpha=0.5, label='k^{-4/3} (Kraichnan)')
    ax.set_xlabel('k')
    ax.set_ylabel('sigma2(k)')
    ax.set_title('ABC: sigma2(k) decay')
    ax.legend(fontsize=8)

    # Panel 3: alpha_total(k) ABC
    ax = axes[0, 2]
    for Re, N, _ in configs:
        snap = all_results[f'ABC_Re{Re}'][-1]
        kp = N // 3
        ax.plot(snap['shells'][:kp], snap['alpha_total'][:kp],
                'o-', color=colors_re[Re], markersize=2, label=f'Re={Re}', alpha=0.8)
    ax.axhline(0.25, color='black', linestyle='--', alpha=0.5, label='sin2theta/4 (sigma=0)')
    ax.axhline(0, color='red', linestyle=':', alpha=0.3, label='sigma=1 (full protection)')
    ax.set_xlabel('k')
    ax.set_ylabel('alpha_total(k)')
    ax.set_title('ABC: Combined suppression')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.01, 0.30)

    # Panel 4: sigma(k) Imbalanced at various Re
    ax = axes[1, 0]
    for Re, N, _ in configs:
        snap = all_results[f'Imb_Re{Re}'][-1]
        kp = N // 3
        ax.plot(snap['shells'][:kp], snap['sigma_k'][:kp],
                'o-', color=colors_re[Re], markersize=2, label=f'Re={Re}', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('sigma(k)')
    ax.set_title(f'Imbalanced: sigma(k) at t={T_snaps[-1]}')
    ax.legend(fontsize=8)

    # Panel 5: sigma2(k) Imbalanced log-log
    ax = axes[1, 1]
    for Re, N, _ in configs:
        snap = all_results[f'Imb_Re{Re}'][-1]
        kp = N // 3
        s2 = snap['sigma2_k'][:kp]
        valid = s2 > 1e-12
        if np.any(valid):
            ax.loglog(snap['shells'][:kp][valid], s2[valid],
                      'o-', color=colors_re[Re], markersize=2, label=f'Re={Re}', alpha=0.8)
    ax.loglog(k_ref, 0.5 * k_ref**(-4/3), 'k--', linewidth=1, alpha=0.5, label='k^{-4/3}')
    ax.set_xlabel('k')
    ax.set_ylabel('sigma2(k)')
    ax.set_title('Imbalanced: sigma2(k) decay')
    ax.legend(fontsize=8)

    # Panel 6: Time evolution of <sigma2> for ABC Re=800
    ax = axes[1, 2]
    for ic_label, prefix in [('ABC', 'ABC'), ('Imbalanced', 'Imb')]:
        key = f'{prefix}_Re800'
        times = [s['t'] for s in all_results[key]]
        s2_avgs = [np.mean(s['sigma2_k'][2:48//3]) for s in all_results[key]]
        ax.plot(times, s2_avgs, 'o-', markersize=4, label=f'{ic_label} Re=800')
    ax.set_xlabel('t')
    ax.set_ylabel('<sigma2> (inertial range)')
    ax.set_title('sigma2 time evolution')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'helical_imbalance_f4b.png')
    plt.savefig(outpath, dpi=150)
    print(f"\n  Plot saved to {outpath}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
