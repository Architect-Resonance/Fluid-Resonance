"""
LYAPUNOV FUNCTIONAL TEST: SOLENOIDAL LAMB FRACTION
====================================================
Meridian's conjecture (S90-W response):

If there exists a Lyapunov functional measuring distance from ker(N_C),
perhaps F(t) = ||P_sol(omega x v)||^2 / ||omega x v||^2, and it's
monotonically decreasing under NS dynamics, then ker(N_C) is a global
attractor.

The Lamb vector decomposes as:
  omega x v = grad(phi) + P_sol(omega x v)
  (longitudinal)       (solenoidal)

- Tsinober: ~91% of Lamb is longitudinal (absorbed by pressure)
- Only the solenoidal part drives NS dynamics (Leray projection)
- Beltrami = ker(N_C): omega x v = 0 (both parts vanish)
- Distance from kernel ~ solenoidal fraction

TEST: Track F(t) = ||P_sol(omega x v)||^2 / ||omega x v||^2 over time.
- If F(t) is monotonically decreasing: evidence for ker(N_C) as attractor
- If F(t) is non-monotonic: ker(N_C) is NOT a simple attractor

We test full NS and BT surgery separately, with multiple ICs.

HONEST TEST: We report what the numbers say.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


class LyapunovTracker(SpectralNS):
    """Extends SpectralNS with solenoidal Lamb fraction tracking."""

    def compute_solenoidal_lamb_fraction(self, u_hat):
        """Compute F = ||P_sol(omega x v)||^2 / ||omega x v||^2.

        The Lamb vector omega x v decomposes as:
          L = L_longitudinal + L_solenoidal
        where L_solenoidal = P_Leray(L) (divergence-free part).

        F = ||L_sol||^2 / ||L||^2 measures how much of the Lamb vector
        actually drives NS dynamics (vs being absorbed by pressure).

        F = 0 would mean all Lamb is longitudinal (pressure absorbs everything).
        F = 1 would mean all Lamb is solenoidal (maximum nonlinear drive).
        """
        N = self.N

        # Full Lamb vector in Fourier space
        lamb_hat = self.compute_lamb_hat(u_hat)

        # Solenoidal (Leray-projected) Lamb vector
        lamb_sol_hat = self.project_leray(lamb_hat)

        # Compute L2 norms (Parseval: ||f||^2 = (1/N^3) sum |f_hat|^2 / N^3)
        norm_full_sq = 0.0
        norm_sol_sq = 0.0
        for i in range(3):
            norm_full_sq += np.sum(np.abs(lamb_hat[i])**2)
            norm_sol_sq += np.sum(np.abs(lamb_sol_hat[i])**2)

        if norm_full_sq < 1e-30:
            return 0.0, 0.0, 0.0

        F = float(norm_sol_sq / norm_full_sq)

        # Also return absolute solenoidal norm (for tracking if it grows/shrinks)
        norm_sol = float(np.sqrt(norm_sol_sq / N**6))
        norm_full = float(np.sqrt(norm_full_sq / N**6))

        return F, norm_sol, norm_full

    def compute_lamb_rms(self, u_hat):
        """Compute RMS of the Lamb vector ||omega x v||_rms."""
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        lamb = np.array([
            omega[1] * u[2] - omega[2] * u[1],
            omega[2] * u[0] - omega[0] * u[2],
            omega[0] * u[1] - omega[1] * u[0],
        ])
        return float(np.sqrt(np.mean(np.sum(lamb**2, axis=0))))

    def compute_beltrami_distance(self, u_hat):
        """Compute distance from Beltrami: ||omega x u|| / (||omega|| * ||u||).

        = 0 for perfect Beltrami (u || omega everywhere)
        = 1 for maximally non-Beltrami (u perp omega everywhere)
        """
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

        cross = np.array([
            omega[1] * u[2] - omega[2] * u[1],
            omega[2] * u[0] - omega[0] * u[2],
            omega[0] * u[1] - omega[1] * u[0],
        ])
        cross_norm = np.sqrt(np.mean(np.sum(cross**2, axis=0)))
        omega_norm = np.sqrt(np.mean(np.sum(omega**2, axis=0)))
        u_norm = np.sqrt(np.mean(np.sum(u**2, axis=0)))

        denom = omega_norm * u_norm
        if denom < 1e-30:
            return 0.0
        return float(cross_norm / denom)

    def run_lyapunov_test(self, u_hat_init, mode='full', dt=0.005,
                          t_max=10.0, sample_interval=10, label=''):
        """Run NS simulation tracking solenoidal Lamb fraction over time.

        Returns dict with time series of all tracked quantities.
        """
        u_hat = u_hat_init.copy()
        t = 0.0
        step = 0

        results = {
            'label': label,
            'mode': mode,
            'times': [],
            'F_solenoidal': [],      # ||P_sol(L)||^2 / ||L||^2
            'norm_sol': [],           # ||P_sol(L)||
            'norm_full': [],          # ||L||
            'beltrami_distance': [],  # ||omega x u|| / (||omega||*||u||)
            'enstrophy': [],
            'energy': [],
        }

        print(f"\n{'='*60}")
        print(f"  {label} ({mode} NS)")
        print(f"{'='*60}")
        print(f"  {'t':>6s}  {'F_sol':>8s}  {'||L_sol||':>10s}  {'||L||':>10s}  "
              f"{'beta':>8s}  {'Z':>10s}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

        while t <= t_max:
            if step % sample_interval == 0:
                F, norm_sol, norm_full = self.compute_solenoidal_lamb_fraction(u_hat)
                beta = self.compute_beltrami_distance(u_hat)

                u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
                E = 0.5 * np.mean(np.sum(u**2, axis=0))

                omega_hat = self.compute_vorticity_hat(u_hat)
                omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
                Z = 0.5 * np.mean(np.sum(omega**2, axis=0))

                results['times'].append(float(t))
                results['F_solenoidal'].append(F)
                results['norm_sol'].append(norm_sol)
                results['norm_full'].append(norm_full)
                results['beltrami_distance'].append(beta)
                results['enstrophy'].append(float(Z))
                results['energy'].append(float(E))

                if step % (sample_interval * 10) == 0:
                    print(f"  {t:6.2f}  {F:8.5f}  {norm_sol:10.6f}  {norm_full:10.6f}  "
                          f"{beta:8.5f}  {Z:10.6f}")

                # Early stop if energy is negligible
                if E < 1e-12:
                    print(f"  [Energy negligible at t={t:.2f}, stopping]")
                    break

            u_hat = self.step_rk4(u_hat, dt, mode=mode)
            t += dt
            step += 1

        return results


def analyze_monotonicity(times, values, label=''):
    """Check if a time series is monotonically decreasing.

    Returns (is_monotone, n_violations, max_violation, trend_slope).
    """
    if len(values) < 2:
        return True, 0, 0.0, 0.0

    diffs = np.diff(values)
    violations = diffs > 0  # increasing = violation of monotone decrease
    n_violations = int(np.sum(violations))
    max_violation = float(np.max(diffs)) if n_violations > 0 else 0.0

    # Linear trend
    t_arr = np.array(times)
    v_arr = np.array(values)
    if len(t_arr) > 1:
        slope = float(np.polyfit(t_arr, v_arr, 1)[0])
    else:
        slope = 0.0

    is_monotone = (n_violations == 0)
    return is_monotone, n_violations, max_violation, slope


def main():
    print("=" * 60)
    print("  LYAPUNOV FUNCTIONAL TEST")
    print("  F(t) = ||P_sol(omega x v)||^2 / ||omega x v||^2")
    print("  Meridian's conjecture: F(t) monotonically decreasing")
    print("  => ker(N_C) is a global attractor")
    print("=" * 60)

    solver = LyapunovTracker(N=32, Re=400)

    # ICs to test
    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Imbalanced 80/20': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
        'Random': solver.random_ic(seed=42),
    }

    all_results = {}
    dt = 0.005
    t_max = 10.0
    sample_every = 20  # every 0.1 time units

    for ic_name, u_hat_init in ics.items():
        for mode in ['full', 'bt']:
            key = f"{ic_name}_{mode}"
            results = solver.run_lyapunov_test(
                u_hat_init, mode=mode, dt=dt, t_max=t_max,
                sample_interval=sample_every, label=f"{ic_name}"
            )
            all_results[key] = results

    # =========================================================
    # ANALYSIS: Monotonicity check
    # =========================================================
    print("\n" + "=" * 60)
    print("  MONOTONICITY ANALYSIS")
    print("=" * 60)

    summary = {}
    for key, res in all_results.items():
        is_mono, n_viol, max_viol, slope = analyze_monotonicity(
            res['times'], res['F_solenoidal'], key
        )

        # Also check beltrami distance
        is_mono_beta, n_viol_beta, max_viol_beta, slope_beta = analyze_monotonicity(
            res['times'], res['beltrami_distance'], key + '_beta'
        )

        # Also check absolute solenoidal norm
        is_mono_norm, n_viol_norm, max_viol_norm, slope_norm = analyze_monotonicity(
            res['times'], res['norm_sol'], key + '_norm_sol'
        )

        summary[key] = {
            'F_solenoidal': {
                'monotone_decreasing': is_mono,
                'violations': n_viol,
                'max_increase': max_viol,
                'trend_slope': slope,
                'initial': res['F_solenoidal'][0] if res['F_solenoidal'] else None,
                'final': res['F_solenoidal'][-1] if res['F_solenoidal'] else None,
            },
            'beltrami_distance': {
                'monotone_decreasing': is_mono_beta,
                'violations': n_viol_beta,
                'trend_slope': slope_beta,
                'initial': res['beltrami_distance'][0] if res['beltrami_distance'] else None,
                'final': res['beltrami_distance'][-1] if res['beltrami_distance'] else None,
            },
            'norm_sol': {
                'monotone_decreasing': is_mono_norm,
                'violations': n_viol_norm,
                'trend_slope': slope_norm,
            }
        }

        status = "MONOTONE" if is_mono else f"NOT MONOTONE ({n_viol} violations)"
        print(f"\n  {key}:")
        print(f"    F_sol:  {res['F_solenoidal'][0]:.5f} => {res['F_solenoidal'][-1]:.5f}  "
              f"slope={slope:.6f}  {status}")
        print(f"    beta:   {res['beltrami_distance'][0]:.5f} => {res['beltrami_distance'][-1]:.5f}  "
              f"slope={slope_beta:.6f}  "
              f"{'MONOTONE' if is_mono_beta else f'NOT MONOTONE ({n_viol_beta} viol)'}")

    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    any_monotone = any(
        s['F_solenoidal']['monotone_decreasing'] for s in summary.values()
    )
    all_decreasing_trend = all(
        s['F_solenoidal']['trend_slope'] < 0 for s in summary.values()
    )

    if all(s['F_solenoidal']['monotone_decreasing'] for s in summary.values()):
        print("  ALL F(t) are strictly monotone decreasing.")
        print("  => STRONG evidence for ker(N_C) as attractor")
    elif all_decreasing_trend:
        print("  All F(t) have decreasing TREND (slope < 0)")
        print("  but not strictly monotone (local fluctuations).")
        print("  => MODERATE evidence: ker(N_C) attracts on average,")
        print("     but not a simple Lyapunov functional")
    elif any_monotone:
        print("  MIXED: some runs monotone, others not.")
        print("  => INCONCLUSIVE for general attractor claim")
    else:
        print("  NO F(t) is monotone decreasing.")
        if all(s['F_solenoidal']['trend_slope'] > 0 for s in summary.values()):
            print("  => NEGATIVE: solenoidal fraction INCREASES")
            print("     ker(N_C) is NOT an attractor via this functional")
        else:
            print("  => COMPLEX behavior. Not a simple Lyapunov functional.")

    # BT vs Full comparison
    print("\n  BT vs Full NS comparison:")
    for ic_name in ics.keys():
        key_full = f"{ic_name}_full"
        key_bt = f"{ic_name}_bt"
        if key_full in summary and key_bt in summary:
            F_full = summary[key_full]['F_solenoidal']['final']
            F_bt = summary[key_bt]['F_solenoidal']['final']
            if F_full and F_bt:
                print(f"    {ic_name}: F_full={F_full:.5f}, F_bt={F_bt:.5f}  "
                      f"({'BT closer to kernel' if F_bt < F_full else 'Full closer to kernel'})")

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    output_path = os.path.join(os.path.dirname(__file__), 'lyapunov_results.json')
    save_data = {
        'params': {'N': 32, 'Re': 400, 'dt': dt, 't_max': t_max},
        'summary': summary,
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    # =========================================================
    # PLOT
    # =========================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Lyapunov Functional: Solenoidal Lamb Fraction F(t)', fontsize=14)

    colors = {'Taylor-Green': 'blue', 'Imbalanced 80/20': 'red', 'Random': 'green'}
    linestyles = {'full': '-', 'bt': '--'}

    # Row 1: F_solenoidal
    for col, mode in enumerate(['full', 'bt']):
        ax = axes[0, col]
        for ic_name in ics.keys():
            key = f"{ic_name}_{mode}"
            res = all_results[key]
            ax.plot(res['times'], res['F_solenoidal'],
                    color=colors[ic_name], linestyle=linestyles[mode],
                    label=ic_name, linewidth=1.5)
        ax.set_ylabel('F = ||P_sol(L)||^2 / ||L||^2')
        ax.set_title(f'Solenoidal Fraction ({mode.upper()} NS)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: Beltrami distance
    for col, mode in enumerate(['full', 'bt']):
        ax = axes[1, col]
        for ic_name in ics.keys():
            key = f"{ic_name}_{mode}"
            res = all_results[key]
            ax.plot(res['times'], res['beltrami_distance'],
                    color=colors[ic_name], linestyle=linestyles[mode],
                    label=ic_name, linewidth=1.5)
        ax.set_ylabel('beta = ||omega x u|| / (||omega||*||u||)')
        ax.set_title(f'Beltrami Distance ({mode.upper()} NS)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 3: Absolute solenoidal norm
    for col, mode in enumerate(['full', 'bt']):
        ax = axes[2, col]
        for ic_name in ics.keys():
            key = f"{ic_name}_{mode}"
            res = all_results[key]
            ax.plot(res['times'], res['norm_sol'],
                    color=colors[ic_name], linestyle=linestyles[mode],
                    label=ic_name, linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('||P_sol(omega x v)||')
        ax.set_title(f'Absolute Solenoidal Lamb Norm ({mode.upper()} NS)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'lyapunov_solenoidal_lamb.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
