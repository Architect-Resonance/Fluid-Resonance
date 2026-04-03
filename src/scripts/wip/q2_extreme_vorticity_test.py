"""
Q2: D > S CONDITIONED ON EXTREME VORTICITY
=============================================
Meridian's priority #1 question (2026-03-12):

  "If extreme vorticity regions self-Beltramize (Buaria+ 2020),
   and D > S in the homochiral sector (Investigation 4),
   does the D-S gap WIDEN where |omega| > 95th percentile?"

If YES => self-reinforcing regularization (the flow creates its own brake).
If NO  => the self-Beltramization path is dead.

Method:
  - Run BT-surgery NS solver from shared_algebraic_structure.py
  - At each reporting step, compute LOCAL S and D fields (per grid point)
  - Condition on |omega| > percentile threshold (90th, 95th, 99th)
  - Compare conditional D/S vs global D/S
  - Track whether the gap widens or narrows at extreme points

HONEST TEST: Report what the numbers say.
"""

import sys
import os
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock

# Import the solver from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS


def compute_local_budget(solver, u_hat):
    """Compute LOCAL (per-grid-point) stretching and dissipation fields.

    Returns:
        omega_mag: |omega| at each grid point (N,N,N)
        S_local: 2 * omega_i S_ij omega_j at each grid point (N,N,N)
        D_local: 2*nu * |nabla omega|^2 at each grid point (N,N,N)
    """
    N = solver.N
    K = [solver.kx, solver.ky, solver.kz]

    # Vorticity in Fourier and physical space
    omega_hat = solver.compute_vorticity_hat(u_hat)
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

    # Vorticity magnitude
    omega_mag = np.sqrt(np.sum(omega**2, axis=0))

    # Velocity gradient tensor
    grad_u = np.zeros((3, 3, N, N, N))
    for i in range(3):
        for j in range(3):
            grad_u[i, j] = np.real(ifftn(1j * K[i] * u_hat[j]))

    # Strain rate tensor S_ij = (1/2)(du_i/dx_j + du_j/dx_i)
    S_tensor = np.zeros((3, 3, N, N, N))
    for i in range(3):
        for j in range(3):
            S_tensor[i, j] = 0.5 * (grad_u[j, i] + grad_u[i, j])

    # Local stretching: 2 * omega_i S_ij omega_j (per grid point)
    stretching_field = np.zeros((N, N, N))
    for i in range(3):
        for j in range(3):
            stretching_field += omega[i] * S_tensor[i, j] * omega[j]
    S_local = 2.0 * stretching_field

    # Local dissipation: 2*nu * |nabla omega|^2 (per grid point)
    # nabla omega has components d(omega_i)/d(x_j)
    grad_omega_sq = np.zeros((N, N, N))
    for i in range(3):
        for j_dir in range(3):
            dw_dx = np.real(ifftn(1j * K[j_dir] * omega_hat[i]))
            grad_omega_sq += dw_dx**2
    D_local = 2.0 * solver.nu * grad_omega_sq

    return omega_mag, S_local, D_local


def compute_conditional_stats(omega_mag, S_local, D_local, percentiles=(90, 95, 99)):
    """Compute D vs S statistics conditioned on |omega| exceeding various thresholds.

    Returns dict with keys for each percentile and 'global'.
    """
    stats = {}

    # Global (unconditional)
    S_global = np.mean(S_local)
    D_global = np.mean(D_local)
    SD_global = S_global / D_global if abs(D_global) > 1e-30 else 0.0
    stats['global'] = {
        'S': S_global, 'D': D_global,
        'D_minus_S': D_global - S_global,
        'S_over_D': SD_global,
        'n_points': S_local.size,
    }

    # Conditional on each percentile
    for pct in percentiles:
        threshold = np.percentile(omega_mag, pct)
        mask = omega_mag > threshold
        n_pts = np.sum(mask)

        if n_pts == 0:
            stats[pct] = {
                'S': 0, 'D': 0, 'D_minus_S': 0, 'S_over_D': 0,
                'n_points': 0, 'threshold': threshold,
            }
            continue

        S_cond = np.mean(S_local[mask])
        D_cond = np.mean(D_local[mask])
        SD_cond = S_cond / D_cond if abs(D_cond) > 1e-30 else 0.0

        stats[pct] = {
            'S': S_cond, 'D': D_cond,
            'D_minus_S': D_cond - S_cond,
            'S_over_D': SD_cond,
            'n_points': int(n_pts),
            'threshold': threshold,
        }

    return stats


def compute_beltramization_metric(solver, u_hat, omega_mag, percentile=95):
    """Measure alignment cos(u, omega) at extreme vorticity points.

    Buaria+ 2020 claim: extreme |omega| regions have u || omega.
    cos(u, omega) => 1 means perfect Beltrami (single-helicity).
    """
    N = solver.N
    omega_hat = solver.compute_vorticity_hat(u_hat)
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
    u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])

    # Compute |u| and |omega|
    u_mag = np.sqrt(np.sum(u**2, axis=0))
    # omega_mag already provided

    # cos(u, omega) = u . omega / (|u| |omega|)
    u_dot_omega = np.sum(u * omega, axis=0)
    denom = u_mag * omega_mag
    cos_uo = np.where(denom > 1e-15, np.abs(u_dot_omega) / denom, 0.0)

    threshold = np.percentile(omega_mag, percentile)
    mask_extreme = omega_mag > threshold
    mask_mild = (omega_mag > np.percentile(omega_mag, 25)) & (omega_mag <= np.percentile(omega_mag, 75))

    return {
        'cos_extreme_mean': float(np.mean(cos_uo[mask_extreme])) if np.any(mask_extreme) else 0,
        'cos_extreme_median': float(np.median(cos_uo[mask_extreme])) if np.any(mask_extreme) else 0,
        'cos_mild_mean': float(np.mean(cos_uo[mask_mild])) if np.any(mask_mild) else 0,
        'cos_mild_median': float(np.median(cos_uo[mask_mild])) if np.any(mask_mild) else 0,
        'cos_global_mean': float(np.mean(cos_uo)),
    }


def run_q2_test():
    """Run the Q2 extreme vorticity test."""
    print("=" * 78)
    print("Q2: D > S CONDITIONED ON EXTREME VORTICITY")
    print("=" * 78)
    print()
    print("Question: Does the D-S gap WIDEN at extreme vorticity points?")
    print("  If yes => self-reinforcing regularization")
    print("  If no  => self-Beltramization path is dead")
    print()
    print("Also measuring: cos(u, omega) at extreme points (Buaria 2020)")
    print("  cos -> 1 means Beltrami (u || omega), confirming self-Beltramization")
    print()

    N = 32
    Re = 400
    dt = 0.005
    T = 5.0
    report_every = 20
    percentiles = (90, 95, 99)

    solver = SpectralNS(N=N, Re=Re)
    wall_start = clock.time()

    # Use ICs that are non-degenerate under BT surgery
    ics = {
        'Imbalanced (80/20)': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
        'Imbalanced (95/5)': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.95),
        'Random (seed=42)': solver.random_ic(seed=42),
    }

    all_results = {}

    for ic_name, u_hat_ic in ics.items():
        print(f"\n{'='*78}")
        print(f"IC: {ic_name}")
        print(f"{'='*78}")

        u_hat_full = u_hat_ic.copy()
        u_hat_bt = u_hat_ic.copy()

        n_steps = int(T / dt)
        times = []
        # Track stats for both full and BT
        stats_full_all = []
        stats_bt_all = []
        beltrami_full_all = []
        beltrami_bt_all = []

        print(f"\n{'t':>5} | {'S/D_glob':>8} {'S/D_90':>7} {'S/D_95':>7} {'S/D_99':>7} | "
              f"{'BT S/D_g':>8} {'BT S/D_95':>9} | {'cos_ext':>7} {'cos_mild':>8}")
        print("-" * 90)

        for step in range(n_steps + 1):
            t = step * dt

            if step % report_every == 0:
                # Full NS conditional budget
                omega_mag_f, S_local_f, D_local_f = compute_local_budget(solver, u_hat_full)
                stats_f = compute_conditional_stats(omega_mag_f, S_local_f, D_local_f, percentiles)
                beltrami_f = compute_beltramization_metric(solver, u_hat_full, omega_mag_f, 95)

                # BT surgery conditional budget
                omega_mag_b, S_local_b, D_local_b = compute_local_budget(solver, u_hat_bt)
                stats_b = compute_conditional_stats(omega_mag_b, S_local_b, D_local_b, percentiles)
                beltrami_b = compute_beltramization_metric(solver, u_hat_bt, omega_mag_b, 95)

                times.append(t)
                stats_full_all.append(stats_f)
                stats_bt_all.append(stats_b)
                beltrami_full_all.append(beltrami_f)
                beltrami_bt_all.append(beltrami_b)

                # Print summary line
                if step % (report_every * 5) == 0 and t > 0.01:
                    sf = stats_f
                    sb = stats_b
                    bf = beltrami_f
                    print(f"{t:5.2f} | "
                          f"{sf['global']['S_over_D']:8.4f} "
                          f"{sf[90]['S_over_D']:7.4f} "
                          f"{sf[95]['S_over_D']:7.4f} "
                          f"{sf[99]['S_over_D']:7.4f} | "
                          f"{sb['global']['S_over_D']:8.4f} "
                          f"{sb[95]['S_over_D']:9.4f} | "
                          f"{bf['cos_extreme_mean']:7.4f} "
                          f"{bf['cos_mild_mean']:8.4f}")

            if step < n_steps:
                u_hat_full = solver.step_rk4(u_hat_full, dt, mode='full')
                u_hat_bt = solver.step_rk4(u_hat_bt, dt, mode='bt')

        times = np.array(times)

        # ============================================================
        # ANALYSIS: Does the gap widen at extreme points?
        # ============================================================
        print(f"\n--- Analysis for {ic_name} ---")

        # Extract time series for each percentile
        for system, stats_all, label in [
            ('full', stats_full_all, 'Full NS'),
            ('bt', stats_bt_all, 'BT Surgery'),
        ]:
            print(f"\n  {label}:")
            valid = times > 0.1

            for pct in percentiles:
                sd_series = np.array([s[pct]['S_over_D'] for s in stats_all])
                margin_series = np.array([s[pct]['D_minus_S'] for s in stats_all])
                sd_global = np.array([s['global']['S_over_D'] for s in stats_all])
                margin_global = np.array([s['global']['D_minus_S'] for s in stats_all])

                if np.any(valid):
                    # Compare conditional vs global S/D
                    sd_cond_mean = np.mean(sd_series[valid])
                    sd_glob_mean = np.mean(sd_global[valid])
                    sd_ratio = sd_cond_mean / sd_glob_mean if abs(sd_glob_mean) > 1e-30 else 0

                    # Does gap widen? (conditional D-S > global D-S)
                    margin_cond_mean = np.mean(margin_series[valid])
                    margin_glob_mean = np.mean(margin_global[valid])

                    always_diss_dom = np.all(margin_series[valid] > 0)

                    print(f"    |ω| > {pct}th pct:")
                    print(f"      mean S/D = {sd_cond_mean:.6f} (global: {sd_glob_mean:.6f}, ratio: {sd_ratio:.4f})")
                    print(f"      mean D-S = {margin_cond_mean:.4e} (global: {margin_glob_mean:.4e})")
                    print(f"      D>S always? {'YES' if always_diss_dom else 'NO'}")
                    if sd_ratio < 1.0:
                        print(f"      => Gap WIDENS at extreme points (S/D drops by {(1-sd_ratio)*100:.1f}%)")
                    elif sd_ratio > 1.0:
                        print(f"      => Gap NARROWS at extreme points (S/D rises by {(sd_ratio-1)*100:.1f}%)")
                    else:
                        print(f"      => No change")

        # Beltramization analysis
        print(f"\n  Self-Beltramization (Buaria 2020 test):")
        for system, beltrami_all, label in [
            ('full', beltrami_full_all, 'Full NS'),
            ('bt', beltrami_bt_all, 'BT Surgery'),
        ]:
            cos_ext = np.array([b['cos_extreme_mean'] for b in beltrami_all])
            cos_mild = np.array([b['cos_mild_mean'] for b in beltrami_all])
            if np.any(valid):
                print(f"    {label}:")
                print(f"      cos(u,ω) extreme: mean={np.mean(cos_ext[valid]):.4f}, max={np.max(cos_ext[valid]):.4f}")
                print(f"      cos(u,ω) mild:    mean={np.mean(cos_mild[valid]):.4f}")
                ratio = np.mean(cos_ext[valid]) / np.mean(cos_mild[valid]) if np.mean(cos_mild[valid]) > 1e-10 else 0
                print(f"      extreme/mild ratio: {ratio:.4f}")
                if ratio > 1.2:
                    print(f"      => CONFIRMED: extreme regions are more Beltrami-aligned")
                elif ratio > 1.05:
                    print(f"      => WEAK: slight alignment tendency")
                else:
                    print(f"      => NO self-Beltramization detected")

        all_results[ic_name] = {
            'times': times,
            'stats_full': stats_full_all,
            'stats_bt': stats_bt_all,
            'beltrami_full': beltrami_full_all,
            'beltrami_bt': beltrami_bt_all,
        }

    wall_time = clock.time() - wall_start
    print(f"\nTotal wall time: {wall_time:.1f}s")

    # ============================================================
    # CROSS-IC VERDICT
    # ============================================================
    print("\n\n" + "=" * 78)
    print("Q2 VERDICT")
    print("=" * 78)

    for ic_name, res in all_results.items():
        times = res['times']
        valid = times > 0.1
        if not np.any(valid):
            continue

        print(f"\n{ic_name}:")

        # BT surgery: conditional S/D at 95th percentile
        sd_bt_95 = np.array([s[95]['S_over_D'] for s in res['stats_bt']])
        sd_bt_global = np.array([s['global']['S_over_D'] for s in res['stats_bt']])
        margin_bt_95 = np.array([s[95]['D_minus_S'] for s in res['stats_bt']])

        ratio = np.mean(sd_bt_95[valid]) / np.mean(sd_bt_global[valid]) if np.mean(sd_bt_global[valid]) > 1e-10 else 0

        if ratio < 0.95:
            print(f"  BT S/D at 95th pct: {np.mean(sd_bt_95[valid]):.4f} (global: {np.mean(sd_bt_global[valid]):.4f})")
            print(f"  => POSITIVE: Gap WIDENS by {(1-ratio)*100:.1f}% at extreme vorticity")
        elif ratio > 1.05:
            print(f"  BT S/D at 95th pct: {np.mean(sd_bt_95[valid]):.4f} (global: {np.mean(sd_bt_global[valid]):.4f})")
            print(f"  => NEGATIVE: Gap NARROWS by {(ratio-1)*100:.1f}% at extreme vorticity")
        else:
            print(f"  BT S/D at 95th pct: {np.mean(sd_bt_95[valid]):.4f} (global: {np.mean(sd_bt_global[valid]):.4f})")
            print(f"  => NEUTRAL: Gap unchanged at extreme vorticity (ratio = {ratio:.4f})")

        # Beltramization
        cos_ext = np.array([b['cos_extreme_mean'] for b in res['beltrami_full']])
        cos_mild = np.array([b['cos_mild_mean'] for b in res['beltrami_full']])
        b_ratio = np.mean(cos_ext[valid]) / np.mean(cos_mild[valid]) if np.mean(cos_mild[valid]) > 1e-10 else 0
        print(f"  Beltramization: extreme/mild cos(u,ω) ratio = {b_ratio:.4f}")

    print()
    print("INTERPRETATION:")
    print("  If gap WIDENS + Beltramization confirmed:")
    print("    => Self-reinforcing regularization (flow brakes itself)")
    print("    => D > S strengthens exactly where blow-up risk is highest")
    print("    => Proceed to Q1 (Migdal framework) and Q3 (boundary layer)")
    print()
    print("  If gap NARROWS or Beltramization absent:")
    print("    => Self-Beltramization path is dead for regularity")
    print("    => Focus on Q1 (Migdal loop space) independently")

    # ============================================================
    # PLOTTING
    # ============================================================
    print("\n\nGenerating plots...")

    # Pick the IC with the most interesting results (most enstrophy)
    best_ic = list(all_results.keys())[0]
    res = all_results[best_ic]
    times = res['times']
    valid = times > 0.1

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Q2: D > S at Extreme Vorticity\n"
                 f"N={N}, Re={Re}, IC={best_ic}",
                 fontsize=14, fontweight='bold')

    # Panel 1: S/D at different percentiles (BT)
    ax = axes[0, 0]
    sd_global = np.array([s['global']['S_over_D'] for s in res['stats_bt']])
    ax.plot(times[valid], sd_global[valid], 'k-', linewidth=2, label='Global', alpha=0.8)
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    for i, pct in enumerate(percentiles):
        sd_pct = np.array([s[pct]['S_over_D'] for s in res['stats_bt']])
        ax.plot(times[valid], sd_pct[valid], '-', color=colors[i], linewidth=1.5,
                label=f'|ω| > {pct}th pct')
    ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('t')
    ax.set_ylabel('S / D')
    ax.set_title('BT Surgery: S/D at extreme vorticity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: S/D at different percentiles (Full NS)
    ax = axes[0, 1]
    sd_global = np.array([s['global']['S_over_D'] for s in res['stats_full']])
    ax.plot(times[valid], sd_global[valid], 'k-', linewidth=2, label='Global', alpha=0.8)
    for i, pct in enumerate(percentiles):
        sd_pct = np.array([s[pct]['S_over_D'] for s in res['stats_full']])
        ax.plot(times[valid], sd_pct[valid], '-', color=colors[i], linewidth=1.5,
                label=f'|ω| > {pct}th pct')
    ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('t')
    ax.set_ylabel('S / D')
    ax.set_title('Full NS: S/D at extreme vorticity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: D-S margin comparison (BT, 95th pct vs global)
    ax = axes[0, 2]
    margin_global = np.array([s['global']['D_minus_S'] for s in res['stats_bt']])
    margin_95 = np.array([s[95]['D_minus_S'] for s in res['stats_bt']])
    ax.plot(times[valid], margin_global[valid], 'k-', linewidth=2, label='Global D-S')
    ax.plot(times[valid], margin_95[valid], 'b--', linewidth=2, label='D-S at |ω|>95th')
    ax.axhline(y=0.0, color='r', linestyle=':', linewidth=1)
    ax.set_xlabel('t')
    ax.set_ylabel('D - S')
    ax.set_title('BT: Dissipation margin (global vs extreme)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Beltramization (cos(u,ω)) over time
    ax = axes[1, 0]
    cos_ext = np.array([b['cos_extreme_mean'] for b in res['beltrami_full']])
    cos_mild = np.array([b['cos_mild_mean'] for b in res['beltrami_full']])
    cos_glob = np.array([b['cos_global_mean'] for b in res['beltrami_full']])
    ax.plot(times[valid], cos_ext[valid], 'r-', linewidth=2, label='Extreme (>95th)')
    ax.plot(times[valid], cos_mild[valid], 'b-', linewidth=2, label='Mild (25-75th)')
    ax.plot(times[valid], cos_glob[valid], 'k--', linewidth=1, label='Global', alpha=0.6)
    ax.set_xlabel('t')
    ax.set_ylabel('|cos(u, ω)|')
    ax.set_title('Self-Beltramization: u||ω alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 5: S/D ratio (BT) across all ICs at 95th percentile
    ax = axes[1, 1]
    ic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (ic_name, res_ic) in enumerate(all_results.items()):
        t = res_ic['times']
        v = t > 0.1
        sd_95 = np.array([s[95]['S_over_D'] for s in res_ic['stats_bt']])
        sd_g = np.array([s['global']['S_over_D'] for s in res_ic['stats_bt']])
        ax.plot(t[v], sd_95[v], '-', color=ic_colors[i], linewidth=2, label=f'{ic_name} (95th)')
        ax.plot(t[v], sd_g[v], '--', color=ic_colors[i], linewidth=1, alpha=0.5)
    ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('t')
    ax.set_ylabel('S / D')
    ax.set_title('BT S/D at 95th pct (all ICs)\nDashed = global')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary bar chart — S/D ratio at different thresholds
    ax = axes[1, 2]
    best_res = all_results[best_ic]
    # Average over valid time steps
    labels = ['Global', '90th', '95th', '99th']
    bt_vals = []
    full_vals = []
    for key in ['global', 90, 95, 99]:
        bt_v = np.mean([s[key]['S_over_D'] for s, v in zip(best_res['stats_bt'], valid) if v])
        full_v = np.mean([s[key]['S_over_D'] for s, v in zip(best_res['stats_full'], valid) if v])
        bt_vals.append(bt_v)
        full_vals.append(full_v)

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, bt_vals, width, label='BT Surgery', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, full_vals, width, label='Full NS', color='#ff7f0e', alpha=0.8)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='S/D = 1 (critical)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean S / D')
    ax.set_title(f'S/D by vorticity threshold ({best_ic})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath = 'h:/tmp/q2_extreme_vorticity.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {outpath}")

    # ============================================================
    # SAVE RAW DATA
    # ============================================================
    import json

    summary_data = {}
    for ic_name, res_ic in all_results.items():
        t = res_ic['times']
        v = t > 0.1

        ic_summary = {'percentiles': {}}
        for pct in ['global', 90, 95, 99]:
            sd_bt = np.array([s[pct]['S_over_D'] for s in res_ic['stats_bt']])
            sd_full = np.array([s[pct]['S_over_D'] for s in res_ic['stats_full']])
            margin_bt = np.array([s[pct]['D_minus_S'] for s in res_ic['stats_bt']])

            key = str(pct)
            ic_summary['percentiles'][key] = {
                'bt_sd_mean': float(np.mean(sd_bt[v])) if np.any(v) else 0,
                'full_sd_mean': float(np.mean(sd_full[v])) if np.any(v) else 0,
                'bt_margin_mean': float(np.mean(margin_bt[v])) if np.any(v) else 0,
                'bt_sd_max': float(np.max(sd_bt[v])) if np.any(v) else 0,
            }

        cos_ext = np.array([b['cos_extreme_mean'] for b in res_ic['beltrami_full']])
        cos_mild = np.array([b['cos_mild_mean'] for b in res_ic['beltrami_full']])
        ic_summary['beltramization'] = {
            'cos_extreme_mean': float(np.mean(cos_ext[v])) if np.any(v) else 0,
            'cos_mild_mean': float(np.mean(cos_mild[v])) if np.any(v) else 0,
            'extreme_over_mild': float(np.mean(cos_ext[v]) / np.mean(cos_mild[v]))
                if np.any(v) and np.mean(cos_mild[v]) > 1e-10 else 0,
        }

        summary_data[ic_name] = ic_summary

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q2_extreme_vorticity_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Raw data saved to {json_path}")

    return all_results


if __name__ == "__main__":
    all_results = run_q2_test()
