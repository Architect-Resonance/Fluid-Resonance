"""
TRIAD COUNTING & N_eff AT THE CLT CROSSOVER
=============================================
Question: "The crossover happens at k≈3 because k²≈9 independent triads
is the CLT activation threshold."

Previous finding (S98-M1c): N_eff ~ k³ (3D volume), NOT k² (surface).
This script verifies which scaling is correct at k=3 specifically.

Method:
  1. Run SpectralNS (N=48, Re=800, Taylor-Green IC) to t=2.0
  2. For each target shell k, enumerate ALL triadic interactions k1+k2=k3
     where |k3| is in shell k (within ±0.5)
  3. Compute the actual enstrophy transfer contribution from each triad
  4. Compute N_eff = (sum|t_i|)² / sum(|t_i|²) — effective independent count
  5. Compare N_eff(k) to k², k³, (4π/3)k³ predictions
  6. Check CLT suppression factor 1/√N_eff against protection floor 0.25

HONEST TEST: We report what the numbers say.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import sys
import time as clock

# Import SpectralNS from shared module
sys.path.insert(0, 'h:/Project/Entropy/Fluid-Resonance/scripts/wip')
from shared_algebraic_structure import SpectralNS


def count_triads_in_shell(N, k_target, shell_width=0.5):
    """Count exact number of triads k1 + k2 = k3 where |k3| is in shell k_target.

    A triad is: k1 + k2 = k3, with |k3| in [k_target - shell_width, k_target + shell_width].
    We restrict to the dealiased region |k_i| <= N/3.

    Returns:
        n_triads: total number of triads
        triad_list: list of (k1, k2, k3) tuples (as integer triplets)
    """
    kmax = N // 3  # dealiasing limit
    k_lo = k_target - shell_width
    k_hi = k_target + shell_width

    # First, find all k3 vectors in the target shell
    k3_list = []
    for ix in range(-kmax, kmax + 1):
        for iy in range(-kmax, kmax + 1):
            for iz in range(-kmax, kmax + 1):
                kmag = np.sqrt(ix**2 + iy**2 + iz**2)
                if k_lo <= kmag <= k_hi:
                    k3_list.append((ix, iy, iz))

    n_modes_in_shell = len(k3_list)

    # For each k3, count all decompositions k1 + k2 = k3
    # where both k1 and k2 are in dealiased region
    # This is expensive for large shells, so we sample for large k
    if k_target > 10 and N >= 48:
        # Sample approach: pick random k3 vectors, enumerate k1
        n_sample = min(50, n_modes_in_shell)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_modes_in_shell, n_sample, replace=False)
        total_per_k3 = 0
        for idx in sample_idx:
            k3 = k3_list[idx]
            count = 0
            for ix1 in range(-kmax, kmax + 1):
                for iy1 in range(-kmax, kmax + 1):
                    for iz1 in range(-kmax, kmax + 1):
                        ix2, iy2, iz2 = k3[0] - ix1, k3[1] - iy1, k3[2] - iz1
                        if abs(ix2) <= kmax and abs(iy2) <= kmax and abs(iz2) <= kmax:
                            count += 1
            total_per_k3 += count
        avg_per_k3 = total_per_k3 / n_sample
        n_triads = int(avg_per_k3 * n_modes_in_shell)
        return n_triads, n_modes_in_shell, avg_per_k3
    else:
        # Exact enumeration for small k
        n_triads = 0
        for k3 in k3_list:
            for ix1 in range(-kmax, kmax + 1):
                for iy1 in range(-kmax, kmax + 1):
                    for iz1 in range(-kmax, kmax + 1):
                        ix2, iy2, iz2 = k3[0] - ix1, k3[1] - iy1, k3[2] - iz1
                        if abs(ix2) <= kmax and abs(iy2) <= kmax and abs(iz2) <= kmax:
                            n_triads += 1
        avg_per_k3 = n_triads / max(n_modes_in_shell, 1)
        return n_triads, n_modes_in_shell, avg_per_k3


def compute_shell_enstrophy_transfer(solver, u_hat, k_target, shell_width=0.5):
    """Compute enstrophy transfer into shell k_target from the nonlinear term.

    The enstrophy transfer spectrum T_Z(k) = Re[ omega_hat*(k) . (curl(u x omega))_hat(k) ]
    summed over modes in the shell.

    We also decompose into same-helical and cross-helical contributions.

    Returns:
        T_total: total enstrophy transfer into shell
        T_same: same-helicity contribution
        T_cross: cross-helicity contribution
        contributions: array of per-mode |contribution| values for N_eff calculation
        contributions_same: same-helicity per-mode contributions
        contributions_cross: cross-helicity per-mode contributions
    """
    N = solver.N
    kmax = N // 3
    k_lo = k_target - shell_width
    k_hi = k_target + shell_width

    # Shell mask
    kmag = solver.kmag
    shell_mask = (kmag >= k_lo) & (kmag <= k_hi) & solver.dealias_mask

    # Vorticity
    omega_hat = solver.compute_vorticity_hat(u_hat)

    # Full nonlinear term: Leray-projected Lamb vector
    lamb_hat_full = solver.compute_lamb_hat(u_hat)
    nl_hat_full = solver.project_leray(lamb_hat_full)

    # Same-helicity nonlinear term
    lamb_hat_same = solver.compute_lamb_hat_bt_surgery(u_hat)
    nl_hat_same = solver.project_leray(lamb_hat_same)

    # Cross-helicity nonlinear term
    lamb_hat_cross = solver.compute_lamb_hat_cross_only(u_hat)
    nl_hat_cross = solver.project_leray(lamb_hat_cross)

    # Enstrophy transfer: T_Z(k) = Re[ sum_i conj(omega_hat_i) * (i k x nl)_i ]
    # Actually, enstrophy equation: dZ/dt = <omega . (omega . grad u)> - nu <|nabla omega|^2>
    # In Fourier: the stretching contribution per mode is
    #   Re[ conj(omega_hat(k)) . F{omega . grad u}(k) ]
    # But we can use the vorticity equation:
    #   d omega_hat / dt = ik x nl_hat - nu k^2 omega_hat
    # So enstrophy transfer per mode = Re[ conj(omega_hat) . (ik x nl_hat) ] * k^2
    # Actually simpler: just use omega_hat and the RHS
    #
    # The enstrophy transfer into shell k is:
    #   T_Z(k) = sum_{|k|~k_target} Re[ conj(omega_hat(k)) . NL_omega(k) ]
    # where NL_omega = curl of (u x omega) projected = i k x Leray(u x omega)
    #
    # Even simpler: use the fact that
    #   d|omega_hat(k)|^2/dt = 2 Re[ conj(omega_hat(k)) . d(omega_hat(k))/dt ]
    # and d(omega_hat)/dt = i k x nl_hat - nu k^2 omega_hat
    # So the nonlinear enstrophy transfer per mode is:
    #   t(k) = 2 Re[ conj(omega_hat(k)) . (i k_vec x nl_hat(k)) ]
    # But this is NOT quite right because enstrophy = (1/2)|omega|^2
    # and we want the transfer in terms of the vorticity equation.
    #
    # Let's just compute it directly: for each mode in the shell,
    # the contribution to enstrophy stretching is related to
    #   Re[ conj(omega_hat(k)) . NL_hat(k) ] * |k|^2
    # where NL_hat is the projected nonlinear term in the VELOCITY equation.
    #
    # Actually, the cleanest approach: enstrophy transfer per mode is
    #   T(k) = -Re[ |k|^2 conj(u_hat(k)) . NL_hat(k) ]  (from omega = curl u, Z = sum k^2 E(k))
    # No wait, let me think clearly.
    #
    # Energy transfer per mode: T_E(k) = -Re[ conj(u_hat(k)) . NL_hat(k) ]
    #   where NL_hat = -Leray(u x omega)_hat  (the nonlinear term in du/dt)
    # Enstrophy transfer per mode: T_Z(k) = k^2 * T_E(k) (for each Fourier mode)
    #   This is because Z = sum k^2 |u_hat|^2 (enstrophy = k^2 weighted energy)
    #
    # Wait no: Z = (1/2)<|omega|^2> = (1/2N^3) sum_k |omega_hat_k|^2 / N^3
    # and omega_hat = ik x u_hat, so |omega_hat|^2 = k^2 |u_hat_perp|^2
    # For solenoidal fields, |u_hat|^2 = |u_hat_perp|^2, so |omega_hat|^2 = k^2 |u_hat|^2
    # Therefore: Z = (1/(2N^6)) sum_k k^2 |u_hat_k|^2
    #
    # The velocity equation: du_hat/dt = NL_hat - nu k^2 u_hat
    # where NL_hat = Leray(lamb_hat) (note sign: lamb = omega x u, not u x omega)
    # Actually in the code: lamb = u x omega, and RHS = Leray(lamb) - nu k^2 u
    # So du_hat/dt = Leray(lamb_hat) - nu k^2 u_hat
    #
    # Energy per mode: E(k) = |u_hat(k)|^2 / (2 N^6)
    # dE(k)/dt = Re[ conj(u_hat(k)) . du_hat(k)/dt ] / N^6
    #          = Re[ conj(u_hat(k)) . NL_hat(k) ] / N^6 - nu k^2 |u_hat(k)|^2 / N^6
    #
    # Enstrophy per mode: Z(k) = k^2 |u_hat(k)|^2 / (2 N^6)
    # Nonlinear enstrophy transfer: k^2 Re[ conj(u_hat(k)) . NL_hat(k) ] / N^6
    #
    # So per-mode contribution to enstrophy transfer:
    #   t_i = k^2 * Re[ conj(u_hat(k)) . NL_hat(k) ] / N^6

    N6 = N**6

    # Per-mode enstrophy transfer: full, same, cross
    def per_mode_transfer(u_h, nl_h, mask):
        """Return per-mode enstrophy transfer as a 1D array of values in the shell."""
        # k^2 * Re[ conj(u_hat) . nl_hat ] / N^6
        dot = np.zeros((N, N, N), dtype=complex)
        for i in range(3):
            dot += np.conj(u_h[i]) * nl_h[i]
        transfer = solver.k2 * np.real(dot) / N6
        return transfer[mask]

    t_full = per_mode_transfer(u_hat, nl_hat_full, shell_mask)
    t_same = per_mode_transfer(u_hat, nl_hat_same, shell_mask)
    t_cross = per_mode_transfer(u_hat, nl_hat_cross, shell_mask)

    T_total = np.sum(t_full)
    T_same = np.sum(t_same)
    T_cross = np.sum(t_cross)

    return T_total, T_same, T_cross, t_full, t_same, t_cross


def compute_N_eff(contributions):
    """Compute effective number of independent contributions.

    N_eff = (sum |t_i|)^2 / sum(|t_i|^2)

    This is the "participation ratio" — equals N for uniform contributions,
    equals 1 if one contribution dominates.
    """
    abs_t = np.abs(contributions)
    sum_abs = np.sum(abs_t)
    sum_sq = np.sum(abs_t**2)
    if sum_sq < 1e-30:
        return 0.0
    return sum_abs**2 / sum_sq


def main():
    print("=" * 78)
    print("TRIAD COUNTING & N_eff AT THE CLT CROSSOVER")
    print("=" * 78)
    print()
    print("Claim: 'crossover at k≈3 because k²≈9 triads is the CLT threshold'")
    print("Counter-claim (S98-M1c): N_eff ~ k³ (3D volume), not k² (surface)")
    print()

    # =====================================================
    # PART 1: Geometric triad counting (no dynamics needed)
    # =====================================================
    print("=" * 78)
    print("PART 1: GEOMETRIC TRIAD COUNTING")
    print("(How many triads k1+k2=k3 exist for each shell k?)")
    print("=" * 78)

    target_shells = [1, 2, 3, 4, 5, 8, 12, 16]

    for N in [32, 48]:
        print(f"\n--- N = {N} (kmax = {N//3}) ---")
        print(f"{'k':>4} | {'N_modes':>8} | {'N_triads':>12} | {'triads/mode':>12} | "
              f"{'k²':>6} | {'k³':>6} | {'4π/3·k³':>8}")
        print("-" * 75)

        for k in target_shells:
            if k > N // 3:
                continue
            t0 = clock.time()
            n_triads, n_modes, avg_per_mode = count_triads_in_shell(N, k)
            dt = clock.time() - t0
            k2_pred = k**2
            k3_pred = k**3
            k3_sphere = 4 * np.pi / 3 * k**3
            print(f"{k:4d} | {n_modes:8d} | {n_triads:12d} | {avg_per_mode:12.1f} | "
                  f"{k2_pred:6d} | {k3_pred:6d} | {k3_sphere:8.1f}  ({dt:.1f}s)")

    # =====================================================
    # PART 2: Dynamical N_eff from actual NS simulation
    # =====================================================
    print()
    print("=" * 78)
    print("PART 2: DYNAMICAL N_eff FROM NS SIMULATION")
    print("(N_eff = participation ratio of enstrophy transfer contributions)")
    print("=" * 78)

    N = 48
    Re = 800
    dt_sim = 0.004
    T_target = 2.0
    n_steps = int(T_target / dt_sim)

    print(f"\nSolver: N={N}, Re={Re}, Taylor-Green IC, RK4, dt={dt_sim}")
    print(f"Evolving to t={T_target}...")

    solver = SpectralNS(N=N, Re=Re)
    u_hat = solver.taylor_green_ic()

    t0 = clock.time()
    for step in range(n_steps):
        u_hat = solver.step_rk4(u_hat, dt_sim, mode='full')
        if (step + 1) % 100 == 0:
            Z = solver.compute_enstrophy(u_hat)
            E = solver.compute_total_energy(u_hat)
            t_now = (step + 1) * dt_sim
            print(f"  t={t_now:.2f}: Z={Z:.4e}, E={E:.4e}")
    wall_evolve = clock.time() - t0
    print(f"Evolution took {wall_evolve:.1f}s")

    # Now compute N_eff for each target shell
    print()
    print(f"{'k':>4} | {'T_total':>12} | {'T_same':>12} | {'T_cross':>12} | "
          f"{'N_modes':>8} | {'N_eff_full':>10} | {'N_eff_same':>10} | {'N_eff_cross':>10} | "
          f"{'k²':>5} | {'k³':>5} | {'1/√N_eff':>8}")
    print("-" * 130)

    results = {}
    target_shells_sim = [1, 2, 3, 4, 5, 8, 12, 16]

    for k in target_shells_sim:
        if k > N // 3:
            continue
        T_tot, T_same, T_cross, t_full, t_same, t_cross = \
            compute_shell_enstrophy_transfer(solver, u_hat, k)

        N_eff_full = compute_N_eff(t_full)
        N_eff_same = compute_N_eff(t_same)
        N_eff_cross = compute_N_eff(t_cross)
        n_modes = len(t_full)
        clt_factor = 1.0 / np.sqrt(max(N_eff_full, 1e-30))

        results[k] = {
            'T_total': T_tot, 'T_same': T_same, 'T_cross': T_cross,
            'N_eff_full': N_eff_full, 'N_eff_same': N_eff_same,
            'N_eff_cross': N_eff_cross,
            'n_modes': n_modes, 'clt_factor': clt_factor,
        }

        print(f"{k:4d} | {T_tot:12.4e} | {T_same:12.4e} | {T_cross:12.4e} | "
              f"{n_modes:8d} | {N_eff_full:10.1f} | {N_eff_same:10.1f} | {N_eff_cross:10.1f} | "
              f"{k**2:5d} | {k**3:5d} | {clt_factor:8.4f}")

    # =====================================================
    # PART 3: SCALING ANALYSIS
    # =====================================================
    print()
    print("=" * 78)
    print("PART 3: SCALING ANALYSIS — N_eff(k) vs k² vs k³")
    print("=" * 78)

    ks = sorted(results.keys())
    ks_arr = np.array(ks, dtype=float)
    neff_arr = np.array([results[k]['N_eff_full'] for k in ks])

    # Fit log(N_eff) = alpha * log(k) + beta
    valid = neff_arr > 1.0  # only fit where N_eff is meaningful
    if np.sum(valid) >= 2:
        log_k = np.log(ks_arr[valid])
        log_neff = np.log(neff_arr[valid])
        alpha, beta = np.polyfit(log_k, log_neff, 1)
        print(f"\nPower-law fit: N_eff ∝ k^{alpha:.2f}  (prefactor = {np.exp(beta):.2f})")
        print(f"  k² prediction: exponent = 2.00")
        print(f"  k³ prediction: exponent = 3.00")
        print(f"  Measured exponent: {alpha:.2f}")

        # Residuals
        print(f"\n{'k':>4} | {'N_eff':>10} | {'k²':>8} | {'k³':>8} | {'fit':>10} | "
              f"{'N_eff/k²':>8} | {'N_eff/k³':>8}")
        print("-" * 70)
        for k in ks:
            neff = results[k]['N_eff_full']
            fit_val = np.exp(beta) * k**alpha
            ratio_k2 = neff / k**2 if k > 0 else 0
            ratio_k3 = neff / k**3 if k > 0 else 0
            print(f"{k:4d} | {neff:10.1f} | {k**2:8d} | {k**3:8d} | {fit_val:10.1f} | "
                  f"{ratio_k2:8.2f} | {ratio_k3:8.2f}")

    # =====================================================
    # PART 4: CLT THRESHOLD ANALYSIS AT k=3
    # =====================================================
    print()
    print("=" * 78)
    print("PART 4: CLT THRESHOLD AT k=3")
    print("=" * 78)

    if 3 in results:
        r3 = results[3]
        print(f"\nAt k = 3:")
        print(f"  N_modes in shell: {r3['n_modes']}")
        print(f"  N_eff (full):     {r3['N_eff_full']:.1f}")
        print(f"  N_eff (same-hel): {r3['N_eff_same']:.1f}")
        print(f"  N_eff (cross-hel):{r3['N_eff_cross']:.1f}")
        print(f"  k² = 9, k³ = 27, 4π/3·k³ = {4*np.pi/3*27:.1f}")
        print(f"  1/√N_eff = {r3['clt_factor']:.4f}")
        print()

        if r3['N_eff_full'] > 0:
            closest_k2 = abs(r3['N_eff_full'] - 9)
            closest_k3 = abs(r3['N_eff_full'] - 27)
            if closest_k2 < closest_k3:
                print(f"  → N_eff({r3['N_eff_full']:.1f}) is CLOSER to k²=9 than k³=27")
            else:
                print(f"  → N_eff({r3['N_eff_full']:.1f}) is CLOSER to k³=27 than k²=9")

    # CLT suppression factor crossing
    print(f"\nCLT suppression factor 1/√N_eff at each k:")
    print(f"{'k':>4} | {'1/√N_eff':>10} | {'< 0.25?':>8} | {'< 0.33?':>8}")
    print("-" * 40)
    crossover_025 = None
    crossover_033 = None
    for k in ks:
        cf = results[k]['clt_factor']
        below_025 = cf < 0.25
        below_033 = cf < 0.33
        print(f"{k:4d} | {cf:10.4f} | {'YES' if below_025 else 'no':>8} | {'YES' if below_033 else 'no':>8}")
        if below_025 and crossover_025 is None:
            crossover_025 = k
        if below_033 and crossover_033 is None:
            crossover_033 = k

    print()
    if crossover_025:
        print(f"  1/√N_eff crosses 0.25 (protection floor) at k = {crossover_025}")
    if crossover_033:
        print(f"  1/√N_eff crosses 0.33 (1/3 threshold) at k = {crossover_033}")

    # =====================================================
    # PART 5: MULTIPLE TIME SNAPSHOTS (robustness check)
    # =====================================================
    print()
    print("=" * 78)
    print("PART 5: N_eff AT k=3 ACROSS MULTIPLE TIMES")
    print("(Check if N_eff is stable or time-dependent)")
    print("=" * 78)

    # Re-evolve from t=0, sample at several times
    print(f"\nRe-evolving from t=0, sampling N_eff(k=3) at t=0.5, 1.0, 1.5, 2.0, 2.5, 3.0")
    u_hat2 = solver.taylor_green_ic()
    sample_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    next_sample = 0
    step = 0

    print(f"{'t':>5} | {'Z':>12} | {'N_eff(k=3)':>12} | {'N_eff_same':>12} | {'N_eff_cross':>12} | {'1/√N_eff':>10}")
    print("-" * 75)

    while next_sample < len(sample_times):
        t_current = step * dt_sim
        if t_current >= sample_times[next_sample] - 1e-8:
            Z = solver.compute_enstrophy(u_hat2)
            _, _, _, t_full, t_same, t_cross = \
                compute_shell_enstrophy_transfer(solver, u_hat2, 3)
            neff_f = compute_N_eff(t_full)
            neff_s = compute_N_eff(t_same)
            neff_c = compute_N_eff(t_cross)
            cf = 1.0 / np.sqrt(max(neff_f, 1e-30))
            print(f"{t_current:5.2f} | {Z:12.4e} | {neff_f:12.1f} | {neff_s:12.1f} | {neff_c:12.1f} | {cf:10.4f}")
            next_sample += 1

        u_hat2 = solver.step_rk4(u_hat2, dt_sim, mode='full')
        step += 1

    # =====================================================
    # VERDICT
    # =====================================================
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)

    if 3 in results:
        r3 = results[3]
        neff3 = r3['N_eff_full']
        print(f"\n1. N_eff at k=3: {neff3:.1f}")
        print(f"   - k² = 9 prediction: {'MATCHES' if 5 < neff3 < 15 else 'DOES NOT MATCH'}")
        print(f"   - k³ = 27 prediction: {'MATCHES' if 18 < neff3 < 40 else 'DOES NOT MATCH'}")

    if np.sum(valid) >= 2:
        print(f"\n2. Scaling: N_eff ∝ k^{alpha:.2f}")
        if 1.5 < alpha < 2.5:
            print(f"   → SUPPORTS k² scaling (surface counting)")
        elif 2.5 < alpha < 3.5:
            print(f"   → SUPPORTS k³ scaling (volume counting)")
        else:
            print(f"   → Neither k² nor k³ — anomalous scaling")

    if crossover_025:
        print(f"\n3. CLT crossover: 1/√N_eff drops below 0.25 at k={crossover_025}")
        if crossover_025 == 3:
            print(f"   → CONFIRMS the claim: crossover at k≈3")
        elif crossover_025 < 3:
            print(f"   → Crossover EARLIER than k=3 (at k={crossover_025})")
        else:
            print(f"   → Crossover LATER than k=3 (at k={crossover_025})")
    else:
        print(f"\n3. 1/√N_eff never drops below 0.25 in measured range — no CLT activation")

    print()
    print("NOTE: N_eff is the DYNAMICAL effective count (participation ratio of")
    print("actual enstrophy transfer contributions), not just geometric triad count.")
    print("The geometric count gives an upper bound; N_eff accounts for the fact")
    print("that some triads contribute much more than others.")


if __name__ == "__main__":
    main()
