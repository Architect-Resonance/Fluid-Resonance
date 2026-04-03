# -*- coding: utf-8 -*-
"""
Re=3200 N=96 Miller Q ratio measurement
========================================
S112 -- Wanderer. Measures ||Q||/||-dS|| at Re=3200 with N=96 grid.
Two ICs: Taylor-Green and Narrowband Imbalanced (80/20).
Reports every 0.5 time units.

Previous results (S111):
  Re=400:  TG ratio = 0.458, NB80 ratio = 0.268
  Re=1600: TG ratio = 0.458, NB80 ratio = 0.250
  Both peak at t=0, Re-independent.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miller_Q_decomposition import MillerAnalyzer
import time as clock

def run_re3200(N=96, Re=3200, dt=0.001, T=3.0, report_interval=0.5):
    """Run Miller analysis at Re=3200 with N=96."""
    print("=" * 78)
    print(f"MILLER Q RATIO -- Re={Re}, N={N}")
    print(f"dt={dt}, T={T}, report every {report_interval}s")
    print("=" * 78)

    solver = MillerAnalyzer(N=N, Re=Re)

    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'NB80': solver.narrowband_imbalanced_ic(seed=42, h_plus_frac=0.8, k_max_ic=2),
    }

    results = {}
    report_steps = int(report_interval / dt)

    for ic_name, u_hat_ic in ics.items():
        hp, hm = solver.helical_energy_fractions(u_hat_ic)
        E0 = solver.compute_total_energy(u_hat_ic)
        print(f"\n{'='*70}")
        print(f"IC: {ic_name}  |  h+: {hp:.3f}  h-: {hm:.3f}  E0: {E0:.4f}")
        print(f"{'='*70}")

        # Verify orthogonality at t=0
        ortho = solver.verify_miller_orthogonality(u_hat_ic)
        print(f"Miller orthogonality: <-DeltaS, w*w> = {ortho:.2e}")

        print(f"\n{'t':>5} | {'||Q||/||-dS||':>14} {'||Qs||/||-dS||':>15} "
              f"{'||Qc||/||-dS||':>15} {'Qs/Q':>6} {'Qc/Q':>6} {'E':>10}")
        print("-" * 85)

        u_hat = u_hat_ic.copy()
        n_steps = int(T / dt)
        ic_results = []
        t_start = clock.time()

        for step in range(n_steps + 1):
            t = step * dt

            if step % report_steps == 0:
                lap_S_norm = solver.compute_neg_laplacian_strain_norm(u_hat)
                Q_full, Q_same, Q_cross = solver.compute_Q_helical_decomposition(u_hat)

                nQ = solver.tensor_L2_norm(Q_full)
                nQs = solver.tensor_L2_norm(Q_same)
                nQc = solver.tensor_L2_norm(Q_cross)
                E = solver.compute_total_energy(u_hat)

                safe_lap = max(lap_S_norm, 1e-30)
                safe_Q = max(nQ, 1e-30)

                r_Q = nQ / safe_lap
                r_Qs = nQs / safe_lap
                r_Qc = nQc / safe_lap
                fs = nQs / safe_Q
                fc = nQc / safe_Q

                elapsed = clock.time() - t_start
                print(f"{t:5.2f} | {r_Q:14.6f} {r_Qs:15.6f} {r_Qc:15.6f} "
                      f"{fs:6.3f} {fc:6.3f} {E:10.6f}  [{elapsed:.0f}s]")

                ic_results.append({
                    't': t, 'r_Q': r_Q, 'r_Qs': r_Qs, 'r_Qc': r_Qc,
                    'fs': fs, 'fc': fc, 'E': E
                })

            if step < n_steps:
                u_hat = solver.step_rk4(u_hat, dt, mode='full')

        results[ic_name] = ic_results
        elapsed_total = clock.time() - t_start
        print(f"\n{ic_name} completed in {elapsed_total:.1f}s")

        # Summary
        max_ratio = max(r['r_Q'] for r in ic_results)
        t_max = [r['t'] for r in ic_results if r['r_Q'] == max_ratio][0]
        print(f"Peak ||Q||/||-dS|| = {max_ratio:.6f} at t={t_max:.2f}")

    # Final comparison
    print(f"\n{'='*78}")
    print("SUMMARY -- Re=3200 N=96")
    print(f"{'='*78}")
    for ic_name, res in results.items():
        max_r = max(r['r_Q'] for r in res)
        t_max = [r['t'] for r in res if r['r_Q'] == max_r][0]
        print(f"  {ic_name:20s}: peak ||Q||/||-dS|| = {max_r:.6f} at t={t_max:.2f}")
    print(f"\nPrevious: Re=400 TG=0.458, Re=1600 TG=0.458")
    print(f"Blowup threshold: 1.0")
    print(f"Safety margin: {1.0 - max(max(r['r_Q'] for r in res) for res in results.values()):.3f}")


if __name__ == '__main__':
    run_re3200()
