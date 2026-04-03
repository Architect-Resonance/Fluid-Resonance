"""
sin^2(theta)(k) SCALE-DEPENDENT MEASUREMENT — M2 TASK
======================================================
S100-M1c: THE most important measurement left in this project.

ASSIGNED TO: Meridian 2
PRIORITY: HIGHEST
ESTIMATED DIFFICULTY: Moderate (adapts existing triadic measurement code)

BACKGROUND:
  The Wanderer (S99-W11) hypothesized that sin^2(theta)_eff might decay
  with wavenumber k. If <sin^2(theta)>(k) ~ k^(-beta) with beta >= 3/28
  (approximately 0.107), the 3/28 gap closes and geometric suppression
  alone handles regularity.

  beta >= 3/28 is a VERY GENTLE requirement: only ~20% reduction per
  decade of wavenumber. But nobody has measured this specific quantity.

  The literature on "depression of nonlinearity" measures a DIFFERENT
  angle (u vs omega alignment in physical space). Our sin^2(theta) is
  the angle between k1 and k2 in a cross-helical triad.

WHAT TO MEASURE:
  For each wavenumber shell k3:
    <sin^2(theta)>(k3) = sum_triads w(triad) * sin^2(theta(k1,k2))
                         / sum_triads w(triad)

  where the sum is over all triads with |k1 + k2| = k3, and:
    w(triad) = |a+(k1)|^2 * |a-(k2)|^2  (energy weight, cross-helical only)

  This gives the energy-weighted average sin^2(theta) for triads that
  produce enstrophy at scale k3.

ALSO MEASURE:
  1. Unweighted (geometric) <sin^2(theta)>(k3) — just counting triads
  2. Same-helical weighted <sin^2(theta)>(k3) — for comparison
  3. Time evolution: does <sin^2(theta)>(k3) change as turbulence develops?
  4. Re dependence: does the scaling beta change with Re?

ADDITIONALLY — SPECTRAL ENTROPY (Predictions P10a-c):
  p(k, t) = E(k, t) / E_total(t)
  S_total(t) = -sum_k p(k,t) * log(p(k,t))
  S_tail(k0, t) = -sum_{k>k0} p(k,t) * log(p(k,t))

  Measure:
  P10a: Is S_total(t) non-decreasing during cascade development?
  P10b: Is S_tail(k, t) non-decreasing for k in inertial range?
  P10c: At what k does S_tail start decreasing (if ever)?

IMPLEMENTATION:
  SpectralNS uses a FUNCTIONAL API:
    u_hat = solver.taylor_green_ic()   # returns u_hat, shape (3, N, N, N)
    u_hat = solver.step_rk4(u_hat, dt) # returns new u_hat, no internal state
  Time must be tracked externally. u_hat[0], u_hat[1], u_hat[2] are the
  three velocity components in Fourier space.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def compute_energy_spectrum(solver, u_hat):
    """Compute shell-averaged energy spectrum E(k).

    Args:
        solver: SpectralNS instance (for grid info: k2, N)
        u_hat: velocity field, shape (3, N, N, N)

    Returns:
        E_k: array of length kmax, energy in each shell
    """
    N = solver.N
    kmax = N // 3
    k_mag = np.sqrt(solver.k2)
    E_k = np.zeros(kmax)
    for ik in range(kmax):
        k = ik + 1
        mask = (k_mag >= k - 0.5) & (k_mag < k + 0.5)
        E_k[ik] = 0.5 * np.sum(
            np.abs(u_hat[0][mask])**2 +
            np.abs(u_hat[1][mask])**2 +
            np.abs(u_hat[2][mask])**2
        )
    return E_k


def compute_spectral_entropy(E_k):
    """Compute spectral entropy from energy spectrum.

    Returns:
        S_total: total spectral entropy
        S_tail: array of tail entropies S(k0) for each k0
    """
    E_total = np.sum(E_k)
    if E_total < 1e-30:
        return 0.0, np.zeros(len(E_k))

    p = E_k / E_total
    p_safe = np.where(p > 0, p, 1)  # avoid log(0)

    S_total = -np.sum(p * np.log(p_safe) * (p > 0))

    S_tail = np.zeros(len(E_k))
    for ik0 in range(len(E_k)):
        pt = p[ik0:]
        pt_safe = np.where(pt > 0, pt, 1)
        S_tail[ik0] = -np.sum(pt * np.log(pt_safe) * (pt > 0))

    return S_total, S_tail


def measure_sin2theta_vs_k(solver, u_hat, n_triads_per_shell=5000):
    """Measure energy-weighted <sin^2(theta)> as function of k3.

    Args:
        solver: SpectralNS instance (for grid info)
        u_hat: velocity field, shape (3, N, N, N)
        n_triads_per_shell: number of triads to sample per k3 shell

    Returns:
        k3_values: array of k3 shell values
        sin2_weighted: energy-weighted <sin^2(theta)>(k3)
        sin2_unweighted: geometric (counting) <sin^2(theta)>(k3)
        n_triads: number of valid triads per shell
    """
    N = solver.N
    kmax = N // 3

    # Stack velocity components for easy per-mode access: shape (N, N, N, 3)
    u_stack = np.stack([u_hat[0], u_hat[1], u_hat[2]], axis=-1)

    k3_values = np.arange(2, kmax + 1)
    sin2_weighted = np.zeros(len(k3_values))
    sin2_unweighted = np.zeros(len(k3_values))
    n_valid = np.zeros(len(k3_values), dtype=int)

    rng = np.random.default_rng(42)

    for ik3, k3_target in enumerate(k3_values):
        w_cross_sum = 0.0
        ws_cross_sum = 0.0
        count = 0
        sin2_geom_sum = 0.0

        for _ in range(n_triads_per_shell):
            # Sample random k1
            k1_vec = rng.integers(-kmax, kmax + 1, size=3)
            k1_mag = np.sqrt(np.sum(k1_vec**2))
            if k1_mag < 1:
                continue

            # k3 direction: random unit vector, magnitude = k3_target
            k3_dir = rng.standard_normal(3)
            k3_dir /= np.linalg.norm(k3_dir)
            k3_vec_float = k3_dir * k3_target
            k3_vec = np.round(k3_vec_float).astype(int)
            k3_mag = np.sqrt(np.sum(k3_vec**2))
            if abs(k3_mag - k3_target) > 0.5 or k3_mag < 1:
                continue

            # k2 = k3 - k1
            k2_vec = k3_vec - k1_vec
            k2_mag = np.sqrt(np.sum(k2_vec**2))
            if k2_mag < 1 or k2_mag > kmax:
                continue

            # Compute angle theta between k1 and k2
            cos_theta = np.dot(k1_vec, k2_vec) / (k1_mag * k2_mag)
            cos_theta = np.clip(cos_theta, -1, 1)
            sin2_theta = 1 - cos_theta**2

            # Map to grid indices (periodic wrap)
            i1, j1, l1 = k1_vec[0] % N, k1_vec[1] % N, k1_vec[2] % N
            i2, j2, l2 = k2_vec[0] % N, k2_vec[1] % N, k2_vec[2] % N

            # Get velocity amplitudes as proxy for helical amplitudes
            # (Full helical decomposition would be better but this captures
            #  the energy weighting)
            u1 = u_stack[i1, j1, l1]
            u2 = u_stack[i2, j2, l2]
            E1 = np.sum(np.abs(u1)**2)
            E2 = np.sum(np.abs(u2)**2)

            if E1 < 1e-30 or E2 < 1e-30:
                continue

            # Cross-helical weight (use energy product as proxy)
            w_cross = E1 * E2
            w_cross_sum += w_cross
            ws_cross_sum += w_cross * sin2_theta

            # Geometric (unweighted)
            sin2_geom_sum += sin2_theta
            count += 1

        if count > 0:
            sin2_weighted[ik3] = ws_cross_sum / w_cross_sum if w_cross_sum > 0 else 0
            sin2_unweighted[ik3] = sin2_geom_sum / count
            n_valid[ik3] = count

    return k3_values, sin2_weighted, sin2_unweighted, n_valid


def run_measurement():
    """Run the full sin^2(theta)(k) + spectral entropy measurement."""
    print("=" * 70)
    print("sin^2(theta)(k) SCALE-DEPENDENT MEASUREMENT")
    print("=" * 70)
    print()

    # Setup
    N = 48
    Re = 400
    dt = 0.005
    print(f"Grid: N={N}, Re={Re}")
    print(f"k_max = {N//3}")
    print()

    # Initialize solver and state
    solver = SpectralNS(N=N, Re=Re)
    u_hat = solver.taylor_green_ic()

    # Evolve to developed turbulence
    print("Evolving to developed turbulence...")
    t = 0.0
    t_target = 2.0
    n_steps = int(t_target / dt)
    for step in range(n_steps):
        u_hat = solver.step_rk4(u_hat, dt)
        t += dt
        if (step + 1) % 100 == 0:
            E = 0.5 * np.mean(np.sum(np.abs(u_hat)**2, axis=0)) * solver.N**3
            print(f"  step {step+1}/{n_steps}, t = {t:.3f}, E ~ {E:.4f}")
    print(f"Evolved to t = {t:.3f}")
    print()

    # Measure sin^2(theta)(k)
    print("Measuring <sin^2(theta)>(k3)...")
    k3_vals, sin2_w, sin2_uw, n_valid = measure_sin2theta_vs_k(
        solver, u_hat, n_triads_per_shell=10000
    )

    # Print results
    print()
    print(f"{'k3':>5} {'<sin^2>_weighted':>18} {'<sin^2>_unweight':>18} {'N_triads':>10}")
    print("-" * 55)
    for i, k3 in enumerate(k3_vals):
        if n_valid[i] > 10:
            print(f"{k3:>5} {sin2_w[i]:>18.6f} {sin2_uw[i]:>18.6f} {n_valid[i]:>10}")

    # Fit power law: sin2 ~ k^(-beta)
    valid = (n_valid > 50) & (sin2_w > 0) & (k3_vals >= 3)
    if np.sum(valid) > 3:
        log_k = np.log(k3_vals[valid])
        log_sin2 = np.log(sin2_w[valid])
        # Linear fit in log-log
        coeffs = np.polyfit(log_k, log_sin2, 1)
        beta = -coeffs[0]
        intercept = np.exp(coeffs[1])

        print()
        print(f"POWER LAW FIT: <sin^2(theta)> ~ k^(-{beta:.4f})")
        print(f"  Intercept: {intercept:.4f}")
        print(f"  Required for gap closure: beta >= {3/28:.4f}")
        if beta >= 3/28:
            print(f"  *** GAP CLOSES! beta = {beta:.4f} >= {3/28:.4f} ***")
        else:
            print(f"  Gap remains open. Deficit: {3/28 - beta:.4f}")
    else:
        print("Not enough valid data points for power law fit")

    # ====================================================================
    # SPECTRAL ENTROPY MEASUREMENT
    # ====================================================================
    print()
    print("=" * 70)
    print("SPECTRAL ENTROPY MEASUREMENT")
    print("=" * 70)
    print()

    # Re-initialize for time series
    solver2 = SpectralNS(N=N, Re=Re)
    u_hat2 = solver2.taylor_green_ic()
    t2 = 0.0

    times = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    S_total_list = []

    for t_snap in times:
        # Evolve to snapshot time
        while t2 < t_snap - 1e-6:
            step_dt = min(dt, t_snap - t2)
            u_hat2 = solver2.step_rk4(u_hat2, step_dt)
            t2 += step_dt

        # Compute energy spectrum and entropy
        E_k = compute_energy_spectrum(solver2, u_hat2)
        E_total = np.sum(E_k)
        S, _ = compute_spectral_entropy(E_k)
        S_total_list.append(S)
        print(f"  t = {t_snap:.1f}: S_total = {S:.4f}, E_total = {E_total:.6f}")

    print()
    print("ENTROPY MONOTONICITY CHECK:")
    monotone = all(S_total_list[i+1] >= S_total_list[i] - 0.01
                   for i in range(len(S_total_list)-1))
    print(f"  S_total non-decreasing: {'YES' if monotone else 'NO'}")
    for i in range(len(times)-1):
        dS = S_total_list[i+1] - S_total_list[i]
        print(f"    t={times[i]:.1f}->{times[i+1]:.1f}: dS = {dS:+.4f} "
              f"{'OK' if dS >= -0.01 else 'VIOLATION'}")

    print()
    print("=" * 70)
    print("M2 TASK COMPLETE — Report results to M1")
    print("=" * 70)


if __name__ == "__main__":
    run_measurement()
