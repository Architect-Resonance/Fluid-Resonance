"""
ALIGNMENT BRIDGE: The Missing Piece
=====================================
S95 continued -- Wanderer

KEY INSIGHT from fourier_physical_bridge.py:
  - NS dynamics HELP Leray suppression (alpha_actual < alpha_scrambled)
  - But enstrophy production can still be large if P_sol(L) is ALIGNED with
    the vorticity gradient

The real bridge isn't "how much Lamb survives" but "how aligned is the
surviving Lamb with what drives enstrophy growth."

This script measures:
1. ALIGNMENT FACTOR: beta = <omega . P_sol(L)> / (|omega| |P_sol(L)|)
   - For actual NS vs phase-scrambled fields
   - This captures the directional coherence

2. EFFECTIVE FORCING: E_eff = <omega . P_sol(L)>
   - The actual enstrophy production rate from the solenoidal Lamb
   - Decomposed: E_eff = |omega| * |P_sol(L)| * beta
   - Where alpha controls magnitude and beta controls alignment

3. MAGNITUDE vs ALIGNMENT decomposition:
   - Compare alpha_actual/alpha_scrambled (magnitude ratio, from previous script)
   - With beta_actual/beta_scrambled (alignment ratio, new)
   - Product alpha*beta gives total coherence effect on enstrophy production
"""

import numpy as np
from numpy.fft import fftn, ifftn
import sys
import os
import time as clock

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS
from fourier_physical_bridge import FourierPhysicalBridge

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AlignmentBridge(FourierPhysicalBridge):
    """Measure alignment between solenoidal Lamb and vorticity."""

    def compute_alignment_diagnostics(self, u_hat, n_scrambles=5):
        """Compute alignment factor beta and effective forcing."""
        diag = {}

        # Vorticity in physical space
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        omega_mag = np.sqrt(np.sum(omega ** 2, axis=0))

        # Solenoidal Lamb in physical space
        L_hat = self.compute_lamb_hat(u_hat)
        L_sol_hat = self.project_leray(L_hat)
        L_sol = np.array([np.real(ifftn(L_sol_hat[i])) for i in range(3)])
        L_sol_mag = np.sqrt(np.sum(L_sol ** 2, axis=0))
        L = np.array([np.real(ifftn(L_hat[i])) for i in range(3)])

        # Alpha: magnitude suppression
        norm_L = np.sum(np.abs(L_hat) ** 2)
        norm_L_sol = np.sum(np.abs(L_sol_hat) ** 2)
        alpha = float(norm_L_sol / norm_L) if norm_L > 1e-30 else 0.0
        diag['alpha'] = alpha

        # Beta: alignment factor
        # omega . P_sol(L) at each point
        omega_dot_Lsol = np.sum(omega * L_sol, axis=0)

        # Effective enstrophy forcing = <omega . P_sol(L)>
        E_eff = float(np.mean(omega_dot_Lsol))
        diag['E_eff'] = E_eff

        # Pointwise alignment: cos(omega, P_sol(L))
        denom = omega_mag * L_sol_mag + 1e-30
        cos_alignment = omega_dot_Lsol / denom

        # Global alignment factor
        total_omega_Lsol = float(np.mean(omega_dot_Lsol))
        total_omega_mag = float(np.mean(omega_mag * L_sol_mag))
        beta = total_omega_Lsol / total_omega_mag if total_omega_mag > 1e-30 else 0.0
        diag['beta'] = beta

        # Mean pointwise |cos|
        diag['mean_abs_cos'] = float(np.mean(np.abs(cos_alignment)))

        # Conditional alignment at extreme vorticity
        for pct in [50, 90, 95, 99]:
            threshold = np.percentile(omega_mag.ravel(), pct)
            mask = omega_mag > threshold
            if np.sum(mask) > 0:
                diag[f'beta_p{pct}'] = float(
                    np.mean(omega_dot_Lsol[mask])
                    / (np.mean(omega_mag[mask] * L_sol_mag[mask]) + 1e-30)
                )
                diag[f'Lsol_mag_p{pct}'] = float(np.mean(L_sol_mag[mask]))
            else:
                diag[f'beta_p{pct}'] = 0.0
                diag[f'Lsol_mag_p{pct}'] = 0.0

        # Enstrophy for reference
        Z = 0.5 * np.mean(np.sum(omega ** 2, axis=0))
        diag['enstrophy'] = float(Z)

        # Now compute same quantities for phase-scrambled fields
        alpha_scr_list = []
        beta_scr_list = []
        E_eff_scr_list = []
        cos_scr_list = []

        for s in range(n_scrambles):
            u_scr = self.phase_scramble(u_hat, seed=2000 + s)

            # Scrambled vorticity
            om_hat_scr = self.compute_vorticity_hat(u_scr)
            om_scr = np.array([np.real(ifftn(om_hat_scr[i])) for i in range(3)])
            om_scr_mag = np.sqrt(np.sum(om_scr ** 2, axis=0))

            # Scrambled solenoidal Lamb
            L_hat_scr = self.compute_lamb_hat(u_scr)
            L_sol_hat_scr = self.project_leray(L_hat_scr)
            L_sol_scr = np.array(
                [np.real(ifftn(L_sol_hat_scr[i])) for i in range(3)]
            )
            L_sol_scr_mag = np.sqrt(np.sum(L_sol_scr ** 2, axis=0))

            # Alpha scrambled
            nL = np.sum(np.abs(L_hat_scr) ** 2)
            nLs = np.sum(np.abs(L_sol_hat_scr) ** 2)
            alpha_scr_list.append(float(nLs / nL) if nL > 1e-30 else 0.0)

            # Beta scrambled
            om_dot_Ls_scr = np.sum(om_scr * L_sol_scr, axis=0)
            E_eff_scr = float(np.mean(om_dot_Ls_scr))
            E_eff_scr_list.append(E_eff_scr)

            total_om_Ls_scr = float(np.mean(om_dot_Ls_scr))
            total_om_mag_scr = float(
                np.mean(om_scr_mag * L_sol_scr_mag)
            )
            beta_scr = (
                total_om_Ls_scr / total_om_mag_scr
                if total_om_mag_scr > 1e-30
                else 0.0
            )
            beta_scr_list.append(beta_scr)

            cos_scr = om_dot_Ls_scr / (om_scr_mag * L_sol_scr_mag + 1e-30)
            cos_scr_list.append(float(np.mean(np.abs(cos_scr))))

        diag['alpha_scrambled'] = float(np.mean(alpha_scr_list))
        diag['beta_scrambled'] = float(np.mean(beta_scr_list))
        diag['E_eff_scrambled'] = float(np.mean(E_eff_scr_list))
        diag['mean_abs_cos_scrambled'] = float(np.mean(cos_scr_list))

        # Ratios
        diag['alpha_ratio'] = (
            alpha / diag['alpha_scrambled']
            if diag['alpha_scrambled'] > 1e-10
            else 1.0
        )
        diag['beta_ratio'] = (
            beta / diag['beta_scrambled']
            if abs(diag['beta_scrambled']) > 1e-10
            else 1.0
        )
        diag['E_eff_ratio'] = (
            E_eff / diag['E_eff_scrambled']
            if abs(diag['E_eff_scrambled']) > 1e-10
            else 1.0
        )

        return diag


def run_alignment_experiment(N=32, Re=400, dt=0.005, T=5.0, report_every=50):
    """Run the alignment bridge experiment."""

    print("=" * 80)
    print("ALIGNMENT BRIDGE EXPERIMENT (S95)")
    print("=" * 80)
    print()
    print("Key question: Is the enstrophy danger in MAGNITUDE or ALIGNMENT?")
    print()
    print("  E_eff = <omega . P_sol(L)>  (enstrophy forcing)")
    print("        = |omega| * |P_sol(L)| * beta")
    print("        = |omega| * |L| * alpha * beta")
    print()
    print("  alpha = magnitude suppression (Leray kills gradient part)")
    print("  beta  = alignment factor (how well P_sol(L) aims at omega)")
    print()
    print("  If alpha_ratio < 1: dynamics HELP magnitude suppression")
    print("  If beta_ratio > 1:  dynamics BUILD alignment (directional coherence)")
    print("  Product alpha_ratio * beta_ratio = total coherence effect")
    print()

    solver = AlignmentBridge(N=N, Re=Re)

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
            f"{'t':>5} | {'alpha':>6} {'a_scr':>6} {'a_rat':>6} | "
            f"{'beta':>7} {'b_scr':>7} {'b_rat':>6} | "
            f"{'E_eff':>10} {'E_scr':>10} {'E_rat':>6} | "
            f"{'Z':>9}"
        )
        print(header)
        print("-" * len(header))

        wall_start = clock.time()

        for step in range(n_steps + 1):
            t = step * dt

            if step % report_every == 0:
                diag = solver.compute_alignment_diagnostics(u_hat, n_scrambles=5)
                diag['time'] = t
                times.append(t)
                diagnostics.append(diag)

                print(
                    f"{t:5.2f} | "
                    f"{diag['alpha']:6.4f} {diag['alpha_scrambled']:6.4f} "
                    f"{diag['alpha_ratio']:6.3f} | "
                    f"{diag['beta']:7.4f} {diag['beta_scrambled']:7.4f} "
                    f"{diag['beta_ratio']:6.2f} | "
                    f"{diag['E_eff']:10.4e} {diag['E_eff_scrambled']:10.4e} "
                    f"{diag['E_eff_ratio']:6.2f} | "
                    f"{diag['enstrophy']:9.4e}"
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
    print("ANALYSIS: MAGNITUDE vs ALIGNMENT")
    print("=" * 80)

    for ic_name, result in all_results.items():
        diags = result['diagnostics']
        times = result['times']

        print(f"\n--- {ic_name} ---")

        alpha_rat = [d['alpha_ratio'] for d in diags]
        beta_rat = [d['beta_ratio'] for d in diags]
        E_rat = [d['E_eff_ratio'] for d in diags]

        # Combined effect
        combined = [a * b for a, b in zip(alpha_rat, beta_rat)]

        print(f"\n  Magnitude (alpha_ratio = alpha_actual/alpha_scrambled):")
        print(f"    min={min(alpha_rat):.3f}, max={max(alpha_rat):.3f}, "
              f"final={alpha_rat[-1]:.3f}")
        if max(alpha_rat) < 1.0:
            print(f"    ** Dynamics HELP magnitude suppression (ratio < 1)")

        print(f"\n  Alignment (beta_ratio = beta_actual/beta_scrambled):")
        print(f"    min={min(beta_rat):.3f}, max={max(beta_rat):.3f}, "
              f"final={beta_rat[-1]:.3f}")
        if max(beta_rat) > 1.0:
            print(f"    ** Dynamics BUILD alignment (ratio > 1)")

        print(f"\n  Combined effect (alpha_ratio * beta_ratio):")
        print(f"    min={min(combined):.3f}, max={max(combined):.3f}, "
              f"final={combined[-1]:.3f}")
        if max(combined) > 1.0:
            print(f"    ** NET: Dynamics AMPLIFY enstrophy forcing by {max(combined):.2f}x")
        else:
            print(f"    ** NET: Dynamics SUPPRESS enstrophy forcing by "
                  f"{1/max(combined):.2f}x")

        print(f"\n  Direct E_eff ratio (most reliable):")
        print(f"    min={min(E_rat):.3f}, max={max(E_rat):.3f}, "
              f"final={E_rat[-1]:.3f}")

        # Conditional alignment at extreme vorticity
        print(f"\n  Alignment at extreme vorticity (actual beta):")
        for pct in [50, 90, 95, 99]:
            beta_p = [d[f'beta_p{pct}'] for d in diags]
            print(f"    |omega| > p{pct}: beta(final) = {beta_p[-1]:.4f}")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n\n" + "=" * 80)
    print("VERDICT: WHERE DOES THE DANGER LIVE?")
    print("=" * 80)

    print("\n  E_eff = |omega| * |P_sol(L)| * beta")
    print("        = |omega| * |L| * alpha * beta")
    print()
    print("  Three factors control enstrophy production:")
    print("  1. |L| - Lamb vector magnitude (grows with enstrophy)")
    print("  2. alpha - fraction surviving Leray (DYNAMICS HELP: ratio < 1)")
    print("  3. beta - alignment with vorticity (this is the key question)")
    print()

    for ic_name, result in all_results.items():
        diags = result['diagnostics']
        alpha_r = diags[-1]['alpha_ratio']
        beta_r = diags[-1]['beta_ratio']
        E_r = diags[-1]['E_eff_ratio']

        print(f"  {ic_name}:")
        print(f"    alpha_ratio = {alpha_r:.3f} "
              f"({'HELPS' if alpha_r < 1 else 'HURTS'})")
        print(f"    beta_ratio  = {beta_r:.3f} "
              f"({'HURTS' if beta_r > 1 else 'HELPS'})")
        print(f"    E_eff_ratio = {E_r:.3f} "
              f"({'NET DANGER' if E_r > 1 else 'NET SAFE'})")

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

        # Row 1: Alpha and Beta ratios
        ax = axes[0, col]
        alpha_rat = [d['alpha_ratio'] for d in diags]
        beta_rat = [d['beta_ratio'] for d in diags]
        combined = [a * b for a, b in zip(alpha_rat, beta_rat)]
        ax.plot(times, alpha_rat, 'b-', lw=2, label='alpha ratio (magnitude)')
        ax.plot(times, beta_rat, 'r-', lw=2, label='beta ratio (alignment)')
        ax.plot(times, combined, 'k--', lw=2, label='combined (alpha*beta)')
        ax.axhline(y=1.0, color='gray', ls=':', lw=1)
        ax.fill_between(times, 0, 1, alpha=0.05, color='green')
        ax.fill_between(times, 1, max(max(combined), 2), alpha=0.05, color='red')
        ax.set_title(f'{ic_name}\nMagnitude vs Alignment')
        ax.set_ylabel('Actual / Scrambled ratio')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2: E_eff actual vs scrambled
        ax = axes[1, col]
        E_eff = [d['E_eff'] for d in diags]
        E_scr = [d['E_eff_scrambled'] for d in diags]
        Z = [d['enstrophy'] for d in diags]
        ax.plot(times, E_eff, 'b-', lw=2, label='E_eff actual')
        ax.plot(times, E_scr, 'r--', lw=2, label='E_eff scrambled')
        ax2 = ax.twinx()
        ax2.semilogy(times, Z, 'g:', lw=1, alpha=0.5, label='Enstrophy')
        ax.set_title('Enstrophy Forcing (actual vs scrambled)')
        ax.set_ylabel('E_eff = <omega . P_sol(L)>')
        ax2.set_ylabel('Enstrophy', color='g')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 3: Conditional alignment at extreme vorticity
        ax = axes[2, col]
        for pct in [50, 90, 95, 99]:
            beta_p = [d[f'beta_p{pct}'] for d in diags]
            ax.plot(times, beta_p, lw=1.5, label=f'beta p{pct}')
        ax.set_xlabel('t')
        ax.set_ylabel('Alignment factor beta')
        ax.set_title('Alignment at Extreme Vorticity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'h:/tmp/alignment_bridge.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    return all_results


if __name__ == "__main__":
    all_results = run_alignment_experiment()
