"""
PART A SCALING TEST: Does R = ||Q||/||-DS|| scale with amplitude?
================================================================
If R ~ ||u|| (linear in amplitude), Part A of the Angle strategy fails.
If R is amplitude-independent, Part A might work.

This is the critical test before pursuing the variational approach.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miller_stress_test import CircleClosingAudit


def measure_R(solver, u_hat):
    """Compute R = ||Q||/||-DS|| and R_NL = ||total_NL||/||-DS||."""
    # ||Q|| via Miller's definition
    Q = solver.compute_Q_components(u_hat)
    norm_Q = np.sqrt(np.sum(Q**2) / solver.N**3)

    # ||-DS||
    norm_DS = solver.compute_neg_laplacian_strain_norm(u_hat)

    # ||total_NL|| via Lamb vector
    lamb_hat = solver.compute_lamb_hat(u_hat)
    Q_lamb = solver.compute_Q_from_lamb(lamb_hat)
    norm_NL = np.sqrt(np.sum(Q_lamb**2) / solver.N**3)

    return norm_Q, norm_NL, norm_DS


def main():
    print("=" * 60)
    print("PART A SCALING TEST")
    print("Does R scale with amplitude?")
    print("=" * 60)

    N = 32
    Re = 400
    solver = CircleClosingAudit(N=N, Re=Re)

    # Taylor-Green IC at unit amplitude
    u_hat_base = solver.taylor_green_ic()
    E_base = solver.compute_total_energy(u_hat_base)
    print(f"\nBase TG: E = {E_base:.6f}")

    amplitudes = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

    print(f"\n{'Amp':>6s} {'E':>10s} {'||Q||':>12s} {'||NL||':>12s} "
          f"{'||-DS||':>12s} {'R_Q':>8s} {'R_NL':>8s} {'R_Q/Amp':>10s}")
    print("-" * 90)

    R_Q_values = []
    for amp in amplitudes:
        u_hat = u_hat_base * amp
        E = solver.compute_total_energy(u_hat)
        norm_Q, norm_NL, norm_DS = measure_R(solver, u_hat)

        R_Q = norm_Q / max(norm_DS, 1e-30)
        R_NL = norm_NL / max(norm_DS, 1e-30)

        R_Q_values.append(R_Q)
        print(f"{amp:6.1f} {E:10.4f} {norm_Q:12.4e} {norm_NL:12.4e} "
              f"{norm_DS:12.4e} {R_Q:8.4f} {R_NL:8.4f} {R_Q/amp:10.4f}")

    # Check scaling
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    # If R ~ amp, then R/amp should be constant
    ratios = [R_Q_values[i] / amplitudes[i] for i in range(len(amplitudes))]
    print(f"\nR_Q / amplitude: {[f'{r:.4f}' for r in ratios]}")
    print(f"Variation: {max(ratios)/min(ratios):.4f}x")

    if max(ratios) / min(ratios) < 1.05:
        print("\nVERDICT: R scales LINEARLY with amplitude.")
        print("Part A (static bound R < 1 for ALL u) is FALSE.")
        print("At amplitude = {:.1f}, R_Q = {:.3f} > 1.".format(
            1.0/ratios[0] * 1.1,
            ratios[0] * (1.0/ratios[0] * 1.1)))
        print("\nImplication: The dynamics (Part B) are ESSENTIAL.")
    else:
        print("\nR does NOT scale linearly. Part A may be viable.")

    # Also check: what amplitude gives R = 1?
    base_R = R_Q_values[amplitudes.index(1.0)]
    critical_amp = 1.0 / base_R
    print(f"\nCritical amplitude for R_Q = 1: {critical_amp:.2f}x")
    print(f"Critical amplitude for R_NL = 1: computed from base ratio")

    # Random IC test
    print("\n" + "=" * 60)
    print("RANDOM IC CHECK")
    print("=" * 60)
    u_hat_rand = solver.random_ic(seed=42)
    for amp in [1.0, 3.0, 10.0]:
        u_hat = u_hat_rand * amp
        norm_Q, norm_NL, norm_DS = measure_R(solver, u_hat)
        R_Q = norm_Q / max(norm_DS, 1e-30)
        R_NL = norm_NL / max(norm_DS, 1e-30)
        print(f"  amp={amp:5.1f}  R_Q={R_Q:.4f}  R_NL={R_NL:.4f}  R_Q/amp={R_Q/amp:.4f}")

    print("\nDONE.")


if __name__ == '__main__':
    main()
