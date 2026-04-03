# -*- coding: utf-8 -*-
"""
PHASE RESONANCE COMPETITION TEST
=================================
S113-W — Wanderer + Antigravity insight.

Key hypothesis: The cascade phase lock and the blowup phase lock are INCOMPATIBLE.

For blowup (g_N large):
  - Need Q aligned with -ΔS → specific phase coherence between modes
  - cos(angle(Q, -ΔS)) must be large and positive

For cascade (g_D large):
  - Need forward energy flux → triadic phases locked with ⟨sin θ_triad⟩ < 0
  - Benavides-Bustamante 2025: cascade requires sign-indefinite helicity

For Beltramization:
  - Need ω ∥ u → cos(angle(ω, u)) → 1
  - This kills L = u × ω → kills Q_cross

QUESTION: When g_N is largest, what are the cascade phases doing?
          When cascade is strongest, what's happening to Q-alignment?
          Can all three resonance conditions be simultaneously satisfied?

Method: Track all three phase indicators during DNS evolution at high time resolution.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import sys, os
import time as clock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miller_Q_decomposition import MillerAnalyzer


def compute_cascade_flux(solver, u_hat, k_cut):
    """
    Compute energy flux through shell k = k_cut.
    Π(k_cut) = -⟨u_<, (u·∇)u⟩  where u_< is the low-pass filtered field.
    Positive = forward cascade.
    """
    N = solver.N
    kmag = solver.kmag
    K = [solver.kx, solver.ky, solver.kz]

    # Low-pass filter
    mask = (kmag <= k_cut).astype(np.complex128)
    u_hat_low = u_hat * mask[np.newaxis, :, :, :]

    # Full nonlinear term in physical space: (u.grad)u
    u_phys = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    # du_j/dx_i
    grad_u = np.zeros((3, 3, N, N, N))
    for j in range(3):
        for i in range(3):
            grad_u[i, j] = np.real(ifftn(1j * K[i] * u_hat[j]))

    # (u·∇)u_j = sum_i u_i ∂u_j/∂x_i
    advection = np.zeros((3, N, N, N))
    for j in range(3):
        for i in range(3):
            advection[j] += u_phys[i] * grad_u[i, j]

    # Low-pass velocity
    u_low_phys = np.array([np.real(ifftn(u_hat_low[i])) for i in range(3)])

    # Flux = -⟨u_<, (u·∇)u⟩
    flux = 0.0
    for j in range(3):
        flux -= np.mean(u_low_phys[j] * advection[j])

    return flux


def compute_beltramization(solver, u_hat):
    """
    Compute average cos(angle(ω, u)) weighted by |ω|.
    Returns: mean cos(ω, u), weighted by vorticity magnitude.
    """
    u_phys = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    w_hat = solver.compute_vorticity_hat(u_hat)
    w_phys = np.array([np.real(ifftn(w_hat[i])) for i in range(3)])

    # |u| and |ω| at each point
    u_mag = np.sqrt(sum(u_phys[i]**2 for i in range(3))) + 1e-30
    w_mag = np.sqrt(sum(w_phys[i]**2 for i in range(3))) + 1e-30

    # cos(ω, u) at each point
    dot = sum(u_phys[i] * w_phys[i] for i in range(3))
    cos_wu = dot / (u_mag * w_mag)

    # Vorticity-weighted average of |cos|
    weights = w_mag**2  # weight by enstrophy density
    cos_mean = np.sum(np.abs(cos_wu) * weights) / np.sum(weights)

    return cos_mean


def compute_Q_alignment(solver, u_hat):
    """
    Compute cos(angle(Q, -ΔS)) = ⟨Q, -ΔS⟩ / (||Q|| · ||-ΔS||).
    This is the "blowup resonance" — how efficiently Q drives enstrophy growth.
    """
    Q_full, _, _ = solver.compute_Q_helical_decomposition(u_hat)
    nQ = solver.tensor_L2_norm(Q_full)

    # Compute -ΔS
    S_hat = solver.compute_strain_hat(u_hat)
    neg_lap_S_hat = np.zeros_like(S_hat)
    ksq = solver.kmag**2
    for i in range(3):
        for j in range(3):
            neg_lap_S_hat[i, j] = ksq * S_hat[i, j]
    n_neg_lap_S = solver.tensor_L2_norm(neg_lap_S_hat)

    if nQ < 1e-30 or n_neg_lap_S < 1e-30:
        return 0.0

    # Inner product ⟨Q, -ΔS⟩
    Q_phys = np.zeros((3, 3, solver.N, solver.N, solver.N))
    neg_lap_S_phys = np.zeros_like(Q_phys)
    for i in range(3):
        for j in range(3):
            Q_phys[i, j] = np.real(ifftn(Q_full[i, j]))
            neg_lap_S_phys[i, j] = np.real(ifftn(neg_lap_S_hat[i, j]))

    inner = np.real(np.sum(Q_phys * neg_lap_S_phys)) / solver.N**3
    cos_QdS = float(np.real(inner / (nQ * n_neg_lap_S)))

    return cos_QdS


def compute_growth_rates(solver, u_hat, dt_small=1e-4):
    """
    Estimate instantaneous growth rates g_N and g_D by finite difference.
    g_N = d/dt ln||Q||, g_D = d/dt ln||-ΔS||
    """
    # Current norms
    Q_full, _, _ = solver.compute_Q_helical_decomposition(u_hat)
    nQ_0 = solver.tensor_L2_norm(Q_full)
    nDS_0 = solver.compute_neg_laplacian_strain_norm(u_hat)

    # One tiny step
    u_hat_1 = solver.step_rk4(u_hat, dt_small, mode='full')

    Q_full_1, _, _ = solver.compute_Q_helical_decomposition(u_hat_1)
    nQ_1 = solver.tensor_L2_norm(Q_full_1)
    nDS_1 = solver.compute_neg_laplacian_strain_norm(u_hat_1)

    g_N = (np.log(max(nQ_1, 1e-30)) - np.log(max(nQ_0, 1e-30))) / dt_small
    g_D = (np.log(max(nDS_1, 1e-30)) - np.log(max(nDS_0, 1e-30))) / dt_small

    return g_N, g_D


def run_phase_resonance_test(N=32, Re=800, dt=0.005, T=2.0, report_dt=0.05):
    """
    Track three phase indicators simultaneously during DNS evolution.
    """
    solver = MillerAnalyzer(N=N, Re=Re)
    n_steps = int(T / dt)
    report_steps = max(1, int(report_dt / dt))

    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Narrowband-80': solver.narrowband_imbalanced_ic(seed=42, h_plus_frac=0.8),
    }

    print("=" * 100)
    print("PHASE RESONANCE COMPETITION TEST")
    print(f"N={N}, Re={Re}, dt={dt}, T={T}")
    print("Hypothesis: cascade phase lock and blowup phase lock are incompatible")
    print("=" * 100)

    for ic_name, u_hat_ic in ics.items():
        print(f"\n{'-' * 80}")
        print(f"IC: {ic_name}")
        print(f"{'-' * 80}")

        header = (f"{'t':>6} | {'R':>8} {'g_N':>10} {'g_D':>10} {'g_D-g_N':>10} | "
                  f"{'cos_QdS':>8} {'cos_wu':>8} {'Pi(k=4)':>10} | "
                  f"{'cascade':>8} {'blowup':>8} {'beltrami':>8}")
        print(header)
        print("-" * 120)

        u_hat = u_hat_ic.copy()
        timeseries = []

        for step in range(n_steps + 1):
            t = step * dt

            if step % report_steps == 0:
                t_start = clock.time()

                # 1. Miller ratio R
                nDS = solver.compute_neg_laplacian_strain_norm(u_hat)
                Q_full, Q_same, Q_cross = solver.compute_Q_helical_decomposition(u_hat)
                nQ = solver.tensor_L2_norm(Q_full)
                R = nQ / max(nDS, 1e-30)

                # 2. Growth rates
                g_N, g_D = compute_growth_rates(solver, u_hat, dt_small=dt/10)

                # 3. Q-alignment (blowup resonance)
                cos_QdS = compute_Q_alignment(solver, u_hat)

                # 4. Beltramization (alignment resonance)
                cos_wu = compute_beltramization(solver, u_hat)

                # 5. Cascade flux at k=4 (mid-range)
                Pi = compute_cascade_flux(solver, u_hat, k_cut=4)

                # Classify which "resonance" is dominant
                # cascade_strength: how much does the cascade drive g_D?
                # blowup_strength: how efficiently does Q drive enstrophy?
                # beltrami_strength: how aligned is ω with u?
                cascade_str = "STRONG" if Pi > 0.001 else ("weak" if Pi > 0 else "reverse")
                blowup_str = "STRONG" if cos_QdS > 0.5 else ("medium" if cos_QdS > 0.2 else "weak")
                beltrami_str = "HIGH" if cos_wu > 0.7 else ("medium" if cos_wu > 0.4 else "low")

                elapsed = clock.time() - t_start

                record = {
                    't': t, 'R': R, 'g_N': g_N, 'g_D': g_D,
                    'cos_QdS': cos_QdS, 'cos_wu': cos_wu, 'Pi': Pi,
                }
                timeseries.append(record)

                print(f"{t:6.3f} | {R:8.5f} {g_N:10.3f} {g_D:10.3f} {g_D-g_N:10.3f} | "
                      f"{cos_QdS:8.4f} {cos_wu:8.4f} {Pi:10.6f} | "
                      f"{cascade_str:>8} {blowup_str:>8} {beltrami_str:>8}")

            if step < n_steps:
                u_hat = solver.step_rk4(u_hat, dt, mode='full')

        # Analysis: correlations between indicators
        print(f"\n{'-' * 40}")
        print("CORRELATION ANALYSIS")
        print(f"{'-' * 40}")

        ts = timeseries
        if len(ts) > 3:
            Rs = np.array([r['R'] for r in ts])
            g_Ns = np.array([r['g_N'] for r in ts])
            g_Ds = np.array([r['g_D'] for r in ts])
            cos_QdSs = np.array([r['cos_QdS'] for r in ts])
            cos_wus = np.array([r['cos_wu'] for r in ts])
            Pis = np.array([r['Pi'] for r in ts])

            # Key question 1: When g_N is large, is cascade also strong?
            # If incompatible: corr(g_N, Pi) should be negative
            if np.std(g_Ns) > 1e-10 and np.std(Pis) > 1e-10:
                corr_gN_Pi = np.corrcoef(g_Ns, Pis)[0, 1]
                print(f"corr(g_N, Pi):    {corr_gN_Pi:+.4f}  "
                      f"{'<- INCOMPATIBLE (cascade opposes blowup growth)' if corr_gN_Pi < -0.3 else ''}")

            # Key question 2: When Q-alignment is strong, is Beltramization also strong?
            # If incompatible: corr(cos_QdS, cos_wu) should be negative
            if np.std(cos_QdSs) > 1e-10 and np.std(cos_wus) > 1e-10:
                corr_QdS_wu = np.corrcoef(cos_QdSs, cos_wus)[0, 1]
                print(f"corr(cos_QdS, cos_wu): {corr_QdS_wu:+.4f}  "
                      f"{'<- INCOMPATIBLE (Beltramization opposes Q-alignment)' if corr_QdS_wu < -0.3 else ''}")

            # Key question 3: When cascade is strong, does floor outgrow numerator?
            gap = g_Ds - g_Ns
            if np.std(Pis) > 1e-10 and np.std(gap) > 1e-10:
                corr_Pi_gap = np.corrcoef(Pis, gap)[0, 1]
                print(f"corr(Pi, g_D-g_N): {corr_Pi_gap:+.4f} "
                      f"{'<- CASCADE DRIVES RESTORING FORCE' if corr_Pi_gap > 0.3 else ''}")

            # Summary statistics
            print(f"\nMean g_D - g_N:     {np.mean(gap):+.4f}  "
                  f"{'<- floor wins on average' if np.mean(gap) > 0 else '<- WARNING: numerator wins'}")
            print(f"Max R:              {np.max(Rs):.6f}")
            print(f"Fraction g_D > g_N: {np.mean(gap > 0)*100:.1f}%")
            print(f"Mean cos(w,u):      {np.mean(cos_wus):.4f}")
            print(f"Mean cos(Q,-dS):    {np.mean(cos_QdSs):.4f}")
            print(f"Mean Pi(k=4):       {np.mean(Pis):.6f}")

            # THE KEY TEST: Are there ANY moments where all three resonances
            # are simultaneously strong?
            strong_cascade = Pis > np.percentile(Pis, 75)
            strong_blowup = cos_QdSs > np.percentile(cos_QdSs, 75)
            strong_beltrami = cos_wus > np.percentile(cos_wus, 75)

            triple = strong_cascade & strong_blowup & strong_beltrami
            print(f"\nTriple resonance (all 3 strong simultaneously): "
                  f"{np.sum(triple)}/{len(triple)} timesteps "
                  f"({np.sum(triple)/len(triple)*100:.1f}%)")
            if np.sum(triple) > 0:
                print(f"  R during triple resonance: {np.mean(Rs[triple]):.6f}")
                print(f"  g_D-g_N during triple:     {np.mean(gap[triple]):+.4f}")
            else:
                print("  <- CONFIRMED: triple resonance never occurs!")


if __name__ == '__main__':
    run_phase_resonance_test()
