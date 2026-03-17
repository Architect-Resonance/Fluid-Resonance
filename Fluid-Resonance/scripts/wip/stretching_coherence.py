"""
STRETCHING COHERENCE: The Right Measurement
=============================================
S95 continued -- Wanderer

CORRECTION: The alignment_bridge.py measured <omega . P_sol(L)> but the
enstrophy production rate is <omega . curl(P_sol(L))> = 2 <omega_i S_ij omega_j>.

The RIGHT question: Does actual NS produce MORE or LESS vortex stretching
than a phase-scrambled field with the same energy spectrum?

If S_actual > S_scrambled: dynamics build stretching coherence (DANGEROUS)
If S_actual < S_scrambled: dynamics suppress stretching (SAFE)

Also measures: S/D ratio for actual vs scrambled.
If (S/D)_actual < (S/D)_scrambled: dynamics help regularity.

This is the DIRECT test of Servidio-Matthaeus "depression of nonlinearity."
"""

import numpy as np
from numpy.fft import fftn, ifftn
import sys
import os
import time as clock

sys.path.insert(0, os.path.dirname(__file__))
from fourier_physical_bridge import FourierPhysicalBridge

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class StretchingCoherence(FourierPhysicalBridge):
    """Measure stretching coherence: actual vs phase-scrambled."""

    def compute_stretching_diagnostics(self, u_hat, n_scrambles=5):
        """Compare stretching for actual field vs phase-scrambled fields."""
        diag = {}

        # Actual field: enstrophy budget
        Z, S, D = self.compute_enstrophy_budget(u_hat)
        diag['Z'] = Z
        diag['S'] = S
        diag['D'] = D
        diag['S_over_D'] = S / D if D > 1e-30 else 0.0

        # Helical decomposition of stretching
        S_same, S_cross, S_full = self.compute_enstrophy_budget_helical(u_hat)
        diag['S_same'] = S_same
        diag['S_cross'] = S_cross

        # Phase-scrambled fields
        S_scr_list = []
        D_scr_list = []
        Z_scr_list = []
        SD_scr_list = []
        S_same_scr_list = []
        S_cross_scr_list = []

        for i in range(n_scrambles):
            u_scr = self.phase_scramble(u_hat, seed=3000 + i)
            Z_s, S_s, D_s = self.compute_enstrophy_budget(u_scr)
            S_same_s, S_cross_s, _ = self.compute_enstrophy_budget_helical(u_scr)
            S_scr_list.append(S_s)
            D_scr_list.append(D_s)
            Z_scr_list.append(Z_s)
            SD_scr_list.append(S_s / D_s if D_s > 1e-30 else 0.0)
            S_same_scr_list.append(S_same_s)
            S_cross_scr_list.append(S_cross_s)

        diag['S_scrambled'] = float(np.mean(S_scr_list))
        diag['D_scrambled'] = float(np.mean(D_scr_list))
        diag['Z_scrambled'] = float(np.mean(Z_scr_list))
        diag['SD_scrambled'] = float(np.mean(SD_scr_list))
        diag['S_same_scrambled'] = float(np.mean(S_same_scr_list))
        diag['S_cross_scrambled'] = float(np.mean(S_cross_scr_list))

        # Ratios
        diag['S_ratio'] = (
            S / diag['S_scrambled']
            if abs(diag['S_scrambled']) > 1e-30
            else 1.0
        )
        diag['SD_ratio'] = (
            diag['S_over_D'] / diag['SD_scrambled']
            if abs(diag['SD_scrambled']) > 1e-10
            else 1.0
        )

        # Depression of nonlinearity index
        # DoN = 1 - |actual_nonlinear_transfer| / |max_possible|
        # We use: DoN = 1 - S_actual / S_scrambled
        diag['depression'] = 1.0 - diag['S_ratio'] if diag['S_ratio'] > 0 else 0.0

        return diag


def run_stretching_experiment(N=32, Re=400, dt=0.005, T=5.0, report_every=50):
    """Run the stretching coherence experiment."""

    print("=" * 80)
    print("STRETCHING COHERENCE EXPERIMENT (S95)")
    print("=" * 80)
    print()
    print("Question: Does NS dynamics BUILD or SUPPRESS vortex stretching")
    print("compared to phase-scrambled fields with the same spectrum?")
    print()
    print("This is the direct test of Servidio-Matthaeus")
    print("'depression of nonlinearity' (PRL 2008).")
    print()

    solver = StretchingCoherence(N=N, Re=Re)

    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Random': solver.random_ic(seed=42),
        'Imbalanced_80_20': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
    }

    n_steps = int(T / dt)
    all_results = {}

    for ic_name, u_hat_ic in ics.items():
        print(f"\n{'='*80}")
        print(f"IC: {ic_name}")
        print(f"{'='*80}")

        u_hat = u_hat_ic.copy()
        times = []
        diagnostics = []

        header = (
            f"{'t':>5} | "
            f"{'S_act':>10} {'S_scr':>10} {'S_rat':>6} | "
            f"{'S/D_a':>6} {'S/D_s':>6} {'SD_r':>6} | "
            f"{'DoN':>6} | "
            f"{'S_same':>10} {'S_cross':>10} | "
            f"{'Z':>9}"
        )
        print(header)
        print("-" * len(header))

        wall_start = clock.time()

        for step in range(n_steps + 1):
            t = step * dt

            if step % report_every == 0:
                diag = solver.compute_stretching_diagnostics(u_hat, n_scrambles=5)
                diag['time'] = t
                times.append(t)
                diagnostics.append(diag)

                print(
                    f"{t:5.2f} | "
                    f"{diag['S']:10.4e} {diag['S_scrambled']:10.4e} "
                    f"{diag['S_ratio']:6.3f} | "
                    f"{diag['S_over_D']:6.3f} {diag['SD_scrambled']:6.3f} "
                    f"{diag['SD_ratio']:6.3f} | "
                    f"{diag['depression']:6.3f} | "
                    f"{diag['S_same']:10.4e} {diag['S_cross']:10.4e} | "
                    f"{diag['Z']:9.4e}"
                )

            if step < n_steps:
                u_hat = solver.step_rk4(u_hat, dt, mode='full')

        wall_time = clock.time() - wall_start
        print(f"\nWall time: {wall_time:.1f}s")

        all_results[ic_name] = {
            'times': np.array(times),
            'diagnostics': diagnostics,
        }

    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n\n" + "=" * 80)
    print("ANALYSIS: DEPRESSION OF NONLINEARITY")
    print("=" * 80)

    for ic_name, result in all_results.items():
        diags = result['diagnostics']
        times = result['times']

        print(f"\n--- {ic_name} ---")

        S_rat = [d['S_ratio'] for d in diags]
        SD_rat = [d['SD_ratio'] for d in diags]
        don = [d['depression'] for d in diags]

        print(f"\n  Stretching ratio (S_actual / S_scrambled):")
        print(f"    min={min(S_rat):.3f}, max={max(S_rat):.3f}, "
              f"final={S_rat[-1]:.3f}")

        if max(S_rat) < 1.0:
            print(f"    ** DEPRESSION OF NONLINEARITY CONFIRMED")
            print(f"    ** Actual stretching is {(1-min(S_rat))*100:.1f}% "
                  f"below random-phase prediction")
        elif min(S_rat) > 1.0:
            print(f"    ** Dynamics AMPLIFY stretching beyond random prediction")
        else:
            print(f"    ** Mixed: sometimes above, sometimes below random")

        print(f"\n  S/D ratio comparison:")
        sd_a = [d['S_over_D'] for d in diags]
        sd_s = [d['SD_scrambled'] for d in diags]
        print(f"    Actual S/D:    min={min(sd_a):.4f}, max={max(sd_a):.4f}")
        print(f"    Scrambled S/D: min={min(sd_s):.4f}, max={max(sd_s):.4f}")

        print(f"\n  Depression of nonlinearity (DoN = 1 - S_ratio):")
        print(f"    min={min(don):.3f}, max={max(don):.3f}, "
              f"final={don[-1]:.3f}")

        # Cross-helical coherence
        S_cross_a = [d['S_cross'] for d in diags]
        S_cross_s = [d['S_cross_scrambled'] for d in diags]

        # Avoid division by zero
        cross_ratios = []
        for a, s in zip(S_cross_a, S_cross_s):
            if abs(s) > 1e-20:
                cross_ratios.append(a / s)

        if cross_ratios:
            print(f"\n  Cross-helical stretching ratio:")
            print(f"    min={min(cross_ratios):.3f}, "
                  f"max={max(cross_ratios):.3f}")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n\n" + "=" * 80)
    print("VERDICT: DEPRESSION OF NONLINEARITY IN NS")
    print("=" * 80)

    all_depressed = True
    for ic_name, result in all_results.items():
        diags = result['diagnostics']
        # Look at t > 0.5 to avoid initial transients
        late_diags = [d for d in diags if d['time'] > 0.5]
        if late_diags:
            late_S_rat = [d['S_ratio'] for d in late_diags]
            mean_S_rat = np.mean(late_S_rat)
            max_S_rat = max(late_S_rat)

            print(f"\n  {ic_name}:")
            print(f"    Mean S_ratio (t>0.5): {mean_S_rat:.4f}")
            print(f"    Max S_ratio (t>0.5):  {max_S_rat:.4f}")

            if max_S_rat > 1.0:
                print(f"    STATUS: STRETCHING EXCEEDS RANDOM at some times")
                all_depressed = False
            else:
                print(f"    STATUS: DEPRESSION CONFIRMED ({(1-mean_S_rat)*100:.1f}% below random)")

    print(f"\n  OVERALL: {'ALL ICs show depression' if all_depressed else 'Mixed results'}")

    if all_depressed:
        print("\n  INTERPRETATION:")
        print("  NS dynamics organize phases to REDUCE stretching below")
        print("  what random phases would produce. The Leray projector")
        print("  removes MORE of the dangerous Lamb vector when phases")
        print("  are NS-correlated than when they are random.")
        print()
        print("  This is consistent with Servidio-Matthaeus (2008):")
        print("  'Depression of nonlinearity' = turbulence self-organizes")
        print("  to deplete the effective nonlinear coupling.")
        print()
        print("  Combined with our per-triad alpha_E = 1/4:")
        print("  - Random phases: alpha_E ~ 1/4 (geometric suppression)")
        print("  - NS dynamics: alpha_E < 1/4 (depression of nonlinearity)")
        print("  - The gap between geometry and dynamics HELPS, not hurts")
    else:
        print("\n  INTERPRETATION:")
        print("  Results are mixed. At some times, NS dynamics build")
        print("  stretching coherence beyond what random phases produce.")
        print("  This is where the millennium problem lives.")

    # ============================================================
    # PLOTS
    # ============================================================
    print("\n\nGenerating plots...")

    fig, axes = plt.subplots(3, len(all_results), figsize=(6 * len(all_results), 15))
    if len(all_results) == 1:
        axes = axes[:, np.newaxis]

    for col, (ic_name, result) in enumerate(all_results.items()):
        diags = result['diagnostics']
        times = result['times']

        # Row 1: S_actual vs S_scrambled
        ax = axes[0, col]
        S_a = [d['S'] for d in diags]
        S_s = [d['S_scrambled'] for d in diags]
        ax.plot(times, S_a, 'b-', lw=2, label='S actual')
        ax.plot(times, S_s, 'r--', lw=2, label='S scrambled')
        ax.set_title(f'{ic_name}\nStretching: actual vs scrambled')
        ax.set_ylabel('Stretching S')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Row 2: S/D ratio
        ax = axes[1, col]
        SD_a = [d['S_over_D'] for d in diags]
        SD_s = [d['SD_scrambled'] for d in diags]
        ax.plot(times, SD_a, 'b-', lw=2, label='S/D actual')
        ax.plot(times, SD_s, 'r--', lw=2, label='S/D scrambled')
        ax.axhline(y=1.0, color='k', ls=':', lw=1, label='Critical S/D=1')
        ax.set_title('S/D Ratio')
        ax.set_ylabel('S / D')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Row 3: Depression of nonlinearity
        ax = axes[2, col]
        don = [d['depression'] for d in diags]
        S_rat = [d['S_ratio'] for d in diags]
        ax.plot(times, S_rat, 'b-', lw=2, label='S_ratio (actual/scrambled)')
        ax.axhline(y=1.0, color='k', ls=':', lw=1)
        ax.fill_between(times, [min(0, min(S_rat))] * len(times), 1.0,
                        alpha=0.1, color='green', label='Depression zone')
        ax.fill_between(times, 1.0, [max(1, max(S_rat))] * len(times),
                        alpha=0.1, color='red', label='Amplification zone')
        ax.set_xlabel('t')
        ax.set_ylabel('S_actual / S_scrambled')
        ax.set_title('Depression of Nonlinearity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'h:/tmp/stretching_coherence.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    return all_results


if __name__ == "__main__":
    all_results = run_stretching_experiment()
