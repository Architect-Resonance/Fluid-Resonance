# -*- coding: utf-8 -*-
"""
F-SWEEP: Three Phases of Suppression
=====================================
S113 -- Wanderer. Measures ||Q||/||-dS|| across helicity fractions
f = 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99

Hypothesis (Beltrami catch-22):
  - f ~ 0.5: Leray shield dominates (max geometric suppression)
  - f ~ 0.7-0.8: Transition zone (both shields active)
  - f -> 1.0: Beltramization shield dominates (L_cross -> 0)

The three phases may correspond to three different correlation structures
between vorticity intensity (Omega) and vorticity growth (dOmega/dt).

Phase 1 (combinatorial): Fano algebraic floor -- sparse mode coupling
Phase 2 (angular): Leray geometric suppression -- pressure fights back
Phase 3 (negative feedback): Beltramization -- self-attenuation

We test: does R(f) show three regimes? Is the minimum never > 1?

Method: N=64 Re=1600, multiple narrowband ICs at each f,
measure at t=0 and evolve to T=1.0 to track dynamics.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miller_Q_decomposition import MillerAnalyzer
import time as clock


def run_f_sweep(N=64, Re=1600, dt=0.002, T=1.0, report_interval=0.25):
    """Sweep helicity fraction and measure Miller Q ratio at each f."""

    f_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    print("=" * 90)
    print("F-SWEEP: THREE PHASES OF SUPPRESSION")
    print(f"N={N}, Re={Re}, dt={dt}, T={T}")
    print(f"Helicity fractions: {f_values}")
    print("=" * 90)

    solver = MillerAnalyzer(N=N, Re=Re)
    report_steps = int(report_interval / dt)
    n_steps = int(T / dt)

    # Header for summary table
    print(f"\n{'f':>6} | {'R(t=0)':>10} {'Rs(t=0)':>10} {'Rc(t=0)':>10} "
          f"{'Rc/R':>6} | {'R(peak)':>10} {'t_peak':>6} | {'E_final':>10}")
    print("-" * 95)

    all_results = {}
    total_start = clock.time()

    for f in f_values:
        # Create narrowband IC with this helicity fraction
        # h_plus_frac = f means fraction f of energy in h+ sector
        u_hat_ic = solver.narrowband_imbalanced_ic(seed=42, h_plus_frac=f, k_max_ic=2)

        # Verify actual helicity fraction
        hp, hm = solver.helical_energy_fractions(u_hat_ic)
        E0 = solver.compute_total_energy(u_hat_ic)

        # Measure at t=0
        lap_S_norm = solver.compute_neg_laplacian_strain_norm(u_hat_ic)
        Q_full, Q_same, Q_cross = solver.compute_Q_helical_decomposition(u_hat_ic)
        nQ = solver.tensor_L2_norm(Q_full)
        nQs = solver.tensor_L2_norm(Q_same)
        nQc = solver.tensor_L2_norm(Q_cross)
        safe_lap = max(lap_S_norm, 1e-30)
        safe_Q = max(nQ, 1e-30)

        r_Q_0 = nQ / safe_lap
        r_Qs_0 = nQs / safe_lap
        r_Qc_0 = nQc / safe_lap
        fc_0 = nQc / safe_Q

        # Evolve and track peak
        u_hat = u_hat_ic.copy()
        peak_r_Q = r_Q_0
        t_peak = 0.0
        timeseries = [{'t': 0.0, 'r_Q': r_Q_0, 'r_Qs': r_Qs_0, 'r_Qc': r_Qc_0,
                       'fc': fc_0, 'hp': hp, 'E': E0}]

        ic_start = clock.time()

        for step in range(1, n_steps + 1):
            u_hat = solver.step_rk4(u_hat, dt, mode='full')
            t = step * dt

            if step % report_steps == 0:
                lap_S_norm = solver.compute_neg_laplacian_strain_norm(u_hat)
                Q_full, Q_same, Q_cross = solver.compute_Q_helical_decomposition(u_hat)
                nQ = solver.tensor_L2_norm(Q_full)
                nQs = solver.tensor_L2_norm(Q_same)
                nQc = solver.tensor_L2_norm(Q_cross)
                E = solver.compute_total_energy(u_hat)
                hp_t, hm_t = solver.helical_energy_fractions(u_hat)

                safe_lap = max(lap_S_norm, 1e-30)
                safe_Q = max(nQ, 1e-30)

                r_Q = nQ / safe_lap
                r_Qs = nQs / safe_lap
                r_Qc = nQc / safe_lap
                fc = nQc / safe_Q

                timeseries.append({'t': t, 'r_Q': r_Q, 'r_Qs': r_Qs, 'r_Qc': r_Qc,
                                   'fc': fc, 'hp': hp_t, 'E': E})

                if r_Q > peak_r_Q:
                    peak_r_Q = r_Q
                    t_peak = t

        E_final = solver.compute_total_energy(u_hat)
        elapsed = clock.time() - ic_start

        print(f"{f:6.2f} | {r_Q_0:10.6f} {r_Qs_0:10.6f} {r_Qc_0:10.6f} "
              f"{fc_0:6.3f} | {peak_r_Q:10.6f} {t_peak:6.2f} | {E_final:10.6f}  [{elapsed:.0f}s]")

        all_results[f] = {
            'r_Q_0': r_Q_0, 'r_Qs_0': r_Qs_0, 'r_Qc_0': r_Qc_0,
            'fc_0': fc_0, 'peak_r_Q': peak_r_Q, 't_peak': t_peak,
            'hp_actual': hp, 'E0': E0, 'E_final': E_final,
            'timeseries': timeseries,
        }

    total_elapsed = clock.time() - total_start

    # Analysis
    print(f"\n{'='*90}")
    print(f"ANALYSIS -- Total time: {total_elapsed:.0f}s")
    print(f"{'='*90}")

    # Extract R(f) curve
    fs = sorted(all_results.keys())
    Rs = [all_results[f]['r_Q_0'] for f in fs]
    Rs_peak = [all_results[f]['peak_r_Q'] for f in fs]
    Rc_fracs = [all_results[f]['fc_0'] for f in fs]

    print("\nR(f) at t=0:")
    print(f"  Max R(f) = {max(Rs):.6f} at f = {fs[Rs.index(max(Rs))]:.2f}")
    print(f"  Min R(f) = {min(Rs):.6f} at f = {fs[Rs.index(min(Rs))]:.2f}")

    print("\nR(f) peak over evolution:")
    print(f"  Max peak R(f) = {max(Rs_peak):.6f} at f = {fs[Rs_peak.index(max(Rs_peak))]:.2f}")
    print(f"  Blowup threshold: 1.0")
    print(f"  Safety margin: {1.0 - max(Rs_peak):.3f}")

    # Check for three-phase structure
    print("\nPhase structure:")
    print("  f=0.50: Leray-dominated (balanced helicity)")
    print(f"    R = {all_results[0.50]['r_Q_0']:.6f}, Rc/R = {all_results[0.50]['fc_0']:.3f}")
    print("  f=0.75: Transition zone")
    print(f"    R = {all_results[0.75]['r_Q_0']:.6f}, Rc/R = {all_results[0.75]['fc_0']:.3f}")
    print("  f=0.95: Beltramization zone")
    print(f"    R = {all_results[0.95]['r_Q_0']:.6f}, Rc/R = {all_results[0.95]['fc_0']:.3f}")
    print("  f=0.99: Deep Beltrami")
    print(f"    R = {all_results[0.99]['r_Q_0']:.6f}, Rc/R = {all_results[0.99]['fc_0']:.3f}")

    # Cross-helical fraction trend
    print("\nCross-helical fraction Rc/R vs f:")
    for f in fs:
        bar = "#" * int(all_results[f]['fc_0'] * 50)
        print(f"  f={f:.2f}: Rc/R={all_results[f]['fc_0']:.3f} |{bar}")

    # Theoretical comparison: f(1-f) scaling
    print("\nR_cross vs f(1-f) scaling:")
    for f in fs:
        f1f = f * (1 - f)
        rc = all_results[f]['r_Qc_0']
        rc_50 = all_results[0.50]['r_Qc_0']
        if f1f > 0 and rc_50 > 0:
            # Normalize: if R_cross ~ f(1-f), then R_cross / (4*f*(1-f)) should be constant
            # (factor 4 because f(1-f) max is 0.25 at f=0.5)
            ratio = rc / (4 * f1f) if f1f > 0 else 0
            ratio_ref = rc_50 / (4 * 0.25)
            print(f"  f={f:.2f}: Rc={rc:.6f}, f(1-f)={f1f:.4f}, "
                  f"Rc/(4f(1-f))={ratio:.6f} (ref at f=0.5: {ratio_ref:.6f})")

    print(f"\nAll R(f) < 1.0: {all(r < 1.0 for r in Rs_peak)}")
    print(f"Beltrami catch-22 confirmed: {'YES' if max(Rs_peak) < 1.0 else 'INCONCLUSIVE'}")


if __name__ == '__main__':
    run_f_sweep()
