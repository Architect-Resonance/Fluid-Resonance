"""
NON-MONOTONICITY DIAGNOSTIC: q(Re=6400) > q(Re=3200)?
S98-M1d: Investigate whether the recovery q=0.63 → 0.87 is real or artifact.

HYPOTHESIS: The non-monotonicity is an ARTIFACT of different evolution times.
  Re=3200 used T=2.0 (fully developed turbulence)
  Re=6400 used T=1.0 (may not be fully developed)

TESTS:
  1. Run Re=3200 at both T=1.0 and T=2.0 → does q change with T?
  2. Run Re=6400 at T=2.0 (if feasible) → does q drop below 0.63?
  3. Track q(t) during evolution → does it converge or fluctuate?
  4. Check sensitivity to random seed

If q(T=1.0) > q(T=2.0) consistently, the non-monotonicity is an artifact.
If q(Re=6400, T=2.0) ≈ q(Re=6400, T=1.0) ≈ 0.87, it's a real effect.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import os
import sys
import time as clock

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS

BIN_PARALLEL = np.pi / 6
BIN_ANTIPARALLEL = 5 * np.pi / 6


def quick_q_local(solver, u_hat, max_shells=15, max_k3=80, max_k1=50):
    """Faster q_local measurement with reduced sampling."""
    u_p, u_m = solver.helical_decompose(u_hat)
    N = solver.N
    k_shells = np.arange(2, min(N // 3, 2 + max_shells), 1)

    ks = []
    R_locals = []

    active_mask = (solver.k2 > 0) & (np.abs(u_p) + np.abs(u_m) > 1e-15)
    active_indices = np.argwhere(active_mask)
    if len(active_indices) > max_k1:
        rng = np.random.default_rng(123)
        active_indices = active_indices[rng.choice(len(active_indices), max_k1, replace=False)]

    for k_shell in k_shells:
        dk = 1.0
        shell_mask = (solver.kmag >= k_shell - dk/2) & (solver.kmag < k_shell + dk/2)
        k3_indices = np.argwhere(shell_mask & (solver.k2 > 0))
        if len(k3_indices) == 0:
            continue
        if len(k3_indices) > max_k3:
            rng = np.random.default_rng(42)
            k3_indices = k3_indices[rng.choice(len(k3_indices), max_k3, replace=False)]

        phases_local = []

        for k3_idx in k3_indices:
            i3, j3, l3 = k3_idx
            k3_vec = np.array([solver.kx[i3, j3, l3],
                               solver.ky[i3, j3, l3],
                               solver.kz[i3, j3, l3]])

            for k1_idx in active_indices:
                i1, j1, l1 = k1_idx
                k1_vec = np.array([solver.kx[i1, j1, l1],
                                   solver.ky[i1, j1, l1],
                                   solver.kz[i1, j1, l1]])
                k2_vec = -k3_vec - k1_vec

                i2 = int(round(k2_vec[0])) % N
                j2 = int(round(k2_vec[1])) % N
                l2 = int(round(k2_vec[2])) % N

                k2_actual = np.array([solver.kx[i2, j2, l2],
                                      solver.ky[i2, j2, l2],
                                      solver.kz[i2, j2, l2]])
                if np.linalg.norm(k2_actual - k2_vec) > 0.1:
                    continue
                if solver.k2[i2, j2, l2] < 0.5:
                    continue

                k1_mag = np.sqrt(solver.k2[i1, j1, l1])
                k2_mag = np.sqrt(solver.k2[i2, j2, l2])
                if k1_mag < 0.5 or k2_mag < 0.5:
                    continue

                cos_theta = np.clip(np.dot(k1_vec, k2_actual) / (k1_mag * k2_mag), -1, 1)
                theta = np.arccos(cos_theta)

                # Local triads only
                if theta < BIN_PARALLEL or theta > BIN_ANTIPARALLEL:
                    continue

                up1 = u_p[i1, j1, l1]
                um1 = u_m[i1, j1, l1]
                up2 = u_p[i2, j2, l2]
                um2 = u_m[i2, j2, l2]
                up3 = u_p[i3, j3, l3]
                um3 = u_m[i3, j3, l3]

                for (a1, a2, a3) in [
                    (up1, um2, np.conj(up3)),
                    (up1, um2, np.conj(um3)),
                    (um1, up2, np.conj(up3)),
                    (um1, up2, np.conj(um3)),
                ]:
                    triple = a1 * a2 * a3
                    if abs(triple) > 1e-30:
                        phases_local.append(np.angle(triple))

        if len(phases_local) > 10:
            R = abs(np.mean(np.exp(1j * np.array(phases_local))))
            ks.append(k_shell)
            R_locals.append(R)

    if len(ks) < 3:
        return float('nan'), ks, R_locals

    ks = np.array(ks, dtype=float)
    R_locals = np.array(R_locals)
    valid = (R_locals > 0) & np.isfinite(R_locals)
    if np.sum(valid) < 3:
        return float('nan'), ks, R_locals

    slope, _ = np.polyfit(np.log(ks[valid]), np.log(R_locals[valid]), 1)
    q = -slope
    return q, ks, R_locals


def evolve_and_track(Re, N, T_max, n_checkpoints=5, seed=42):
    """Evolve and measure q_local at multiple time checkpoints."""
    dt = 0.005 * (32 / N)
    n_steps = int(T_max / dt)
    checkpoint_interval = n_steps // n_checkpoints

    solver = SpectralNS(N=N, Re=Re)
    u_hat = solver.random_ic(seed=seed)

    results = []
    t = 0.0

    print(f"  Re={Re}, N={N}, T_max={T_max}, dt={dt:.5f}")

    for step in range(n_steps):
        u_hat = solver.step_rk4(u_hat, dt, mode='full')
        t += dt

        if (step + 1) % checkpoint_interval == 0:
            q, ks, Rs = quick_q_local(solver, u_hat)
            enstrophy = float(np.sum(solver.k2 * np.abs(u_hat)**2))
            energy = float(np.sum(np.abs(u_hat)**2))
            results.append({
                't': t,
                'q_local': q,
                'enstrophy': enstrophy,
                'energy': energy,
                'n_shells': len(ks),
            })
            print(f"    t={t:.3f}: q_local={q:.3f}, Omega={enstrophy:.2e}, E={energy:.2e}")

    return results


def main():
    print("=" * 70)
    print("  NON-MONOTONICITY DIAGNOSTIC: q(Re=6400) vs q(Re=3200)")
    print("=" * 70)
    print()
    print("  M2 data:  Re=3200 (T=2.0) → q=0.63")
    print("            Re=6400 (T=1.0) → q=0.87")
    print("  Question: Is the recovery q=0.63→0.87 real or artifact of T?")
    print()

    wall_start = clock.time()

    # ============================================================
    # TEST 1: Re=3200 at T=1.0 (like-for-like with Re=6400)
    # ============================================================
    print("=" * 70)
    print("  TEST 1: Re=3200, N=64, T_max=2.0 (track q vs time)")
    print("=" * 70)
    results_3200 = evolve_and_track(Re=3200, N=64, T_max=2.0, n_checkpoints=5)

    # ============================================================
    # TEST 2: Re=6400 at T=2.0 (extended evolution)
    # ============================================================
    print()
    print("=" * 70)
    print("  TEST 2: Re=6400, N=64, T_max=2.0 (extended evolution)")
    print("=" * 70)
    results_6400 = evolve_and_track(Re=6400, N=64, T_max=2.0, n_checkpoints=5)

    # ============================================================
    # TEST 3: Seed sensitivity at Re=3200
    # ============================================================
    print()
    print("=" * 70)
    print("  TEST 3: Re=3200, N=64, T=2.0, different seeds")
    print("=" * 70)
    seed_results = []
    for seed in [42, 137, 271, 314, 577]:
        solver = SpectralNS(N=64, Re=3200)
        u_hat = solver.random_ic(seed=seed)
        dt = 0.005 * (32 / 64)
        for step in range(int(2.0 / dt)):
            u_hat = solver.step_rk4(u_hat, dt, mode='full')
        q, _, _ = quick_q_local(solver, u_hat)
        seed_results.append((seed, q))
        print(f"    seed={seed}: q_local = {q:.3f}")

    # ============================================================
    # TEST 4: Seed sensitivity at Re=6400
    # ============================================================
    print()
    print("=" * 70)
    print("  TEST 4: Re=6400, N=64, T=2.0, different seeds")
    print("=" * 70)
    seed_results_6400 = []
    for seed in [42, 137, 271]:
        solver = SpectralNS(N=64, Re=6400)
        u_hat = solver.random_ic(seed=seed)
        dt = 0.005 * (32 / 64)
        for step in range(int(2.0 / dt)):
            u_hat = solver.step_rk4(u_hat, dt, mode='full')
        q, _, _ = quick_q_local(solver, u_hat)
        seed_results_6400.append((seed, q))
        print(f"    seed={seed}: q_local = {q:.3f}")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print()
    print("=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    # q vs time
    print("\n  q_local vs time:")
    print(f"  {'t':>6s}  {'Re=3200':>10s}  {'Re=6400':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
    for r1, r2 in zip(results_3200, results_6400):
        print(f"  {r1['t']:6.2f}  {r1['q_local']:10.3f}  {r2['q_local']:10.3f}")

    # T-dependence
    q_3200_early = results_3200[0]['q_local'] if results_3200 else float('nan')
    q_3200_late = results_3200[-1]['q_local'] if results_3200 else float('nan')
    q_6400_early = results_6400[0]['q_local'] if results_6400 else float('nan')
    q_6400_late = results_6400[-1]['q_local'] if results_6400 else float('nan')

    print(f"\n  T-dependence:")
    print(f"    Re=3200: q(T~0.4) = {q_3200_early:.3f}, q(T=2.0) = {q_3200_late:.3f}")
    print(f"    Re=6400: q(T~0.4) = {q_6400_early:.3f}, q(T=2.0) = {q_6400_late:.3f}")

    if q_3200_early > q_3200_late:
        print(f"    Re=3200: q DECREASES with T (coherence builds over time)")
    else:
        print(f"    Re=3200: q INCREASES or stable with T")

    if q_6400_early > q_6400_late:
        print(f"    Re=6400: q DECREASES with T (coherence builds over time)")
    else:
        print(f"    Re=6400: q INCREASES or stable with T")

    # Seed sensitivity
    q_3200_seeds = [q for _, q in seed_results]
    q_6400_seeds = [q for _, q in seed_results_6400]
    q_3200_mean = np.mean(q_3200_seeds)
    q_3200_std = np.std(q_3200_seeds)
    q_6400_mean = np.mean(q_6400_seeds)
    q_6400_std = np.std(q_6400_seeds)

    print(f"\n  Seed sensitivity:")
    print(f"    Re=3200: q = {q_3200_mean:.3f} +/- {q_3200_std:.3f} (n={len(q_3200_seeds)})")
    print(f"    Re=6400: q = {q_6400_mean:.3f} +/- {q_6400_std:.3f} (n={len(q_6400_seeds)})")

    # Verdict
    print(f"\n  {'='*60}")
    print(f"  VERDICT")
    print(f"  {'='*60}")

    # Compare same-T values
    if results_3200 and results_6400:
        # Find closest time to T=1.0 for both
        idx_3200_t1 = min(range(len(results_3200)),
                          key=lambda i: abs(results_3200[i]['t'] - 1.0))
        idx_6400_t1 = min(range(len(results_6400)),
                          key=lambda i: abs(results_6400[i]['t'] - 1.0))

        q_3200_at_t1 = results_3200[idx_3200_t1]['q_local']
        q_6400_at_t1 = results_6400[idx_6400_t1]['q_local']

        print(f"\n  Like-for-like comparison at T~1.0:")
        print(f"    Re=3200: q = {q_3200_at_t1:.3f}")
        print(f"    Re=6400: q = {q_6400_at_t1:.3f}")

        if q_6400_at_t1 > q_3200_at_t1:
            print(f"    => Non-monotonicity PERSISTS at same T")
            print(f"       This suggests a REAL EFFECT (possibly inertial range lengthening)")
        else:
            print(f"    => Non-monotonicity DISAPPEARS at same T")
            print(f"       This confirms the ARTIFACT hypothesis (different T values)")

    print(f"\n  At T=2.0 (fully developed):")
    print(f"    Re=3200: q = {q_3200_late:.3f}")
    print(f"    Re=6400: q = {q_6400_late:.3f}")

    if q_6400_late > q_3200_late:
        print(f"    => Non-monotonicity REAL at long times")
    elif abs(q_6400_late - q_3200_late) < 0.1:
        print(f"    => q approximately EQUAL — non-monotonicity was transient")
    else:
        print(f"    => q_6400 < q_3200 at long times — MONOTONE decrease confirmed")

    total_time = clock.time() - wall_start
    print(f"\n  Total wall time: {total_time:.1f}s")

    # NOTE about resolution
    print(f"""
  IMPORTANT CAVEATS:
  1. N=64 is LOWER resolution than M2's N=96/128.
     This means shorter inertial range and different dissipation scale.
     Results are QUALITATIVE — the trends matter, not exact q values.

  2. The M2 data used N=96 (Re=3200) and N=128 (Re=6400).
     Resolution difference could also contribute to non-monotonicity:
     - N=128 has 128/3 ≈ 42 resolved shells vs N=96's 32 shells
     - More shells in the fit → different power law slope

  3. The eddy turnover time tau_e ~ L/U ~ 1/sqrt(E*k_peak) may differ
     between Re=3200 and Re=6400. T=1.0 might be < tau_e at Re=6400.
""")


if __name__ == '__main__':
    main()
