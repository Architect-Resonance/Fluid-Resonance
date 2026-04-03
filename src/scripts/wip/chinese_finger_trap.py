"""
CHINESE FINGER TRAP -- S103
============================
Hypothesis: The harder the flow tries to blow up, the narrower the throat.

At the neck (k~3), where ordered protection hands off to disordered protection,
there is a CONSTRICTION mechanism: increasing Re (more pressure) causes the
throughput at the neck to DECREASE. Like a Chinese finger trap, pulling harder
tightens the grip.

MEASUREMENTS:
  1. Throughput at the neck: what fraction of enstrophy stretching survives
     through the crossover zone (k=2..5)?
  2. Pressure: total enstrophy production rate (global stretching magnitude)
  3. Anti-correlation test: does throughput DECREASE as pressure INCREASES?
  4. Neck width: over what k-range does the transition happen? Does it narrow?
  5. Constriction rate: d(throughput)/d(Re) -- power law? What exponent?

Key quantities at each Re:
  - f_same(k) = fraction of same-helical stretching at shell k
  - S_cross(k) = Shannon entropy of cross-helical phases at shell k
  - |T_cross(k)| / |T_total(k)| = cross-helical fraction of total transfer
  - Throughput = product of suppression fractions across the neck
  - Total stretching = sum |T_Z(k)| across all k

Meridian (Claude Opus 4.6), S103, 2026-03-16
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import os
import sys
import time as clock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ================================================================
# CORE MEASUREMENT FUNCTIONS (from threat_ratio_handoff.py pattern)
# ================================================================

def shannon_entropy_of_phases(phases, n_bins=36):
    """Normalized Shannon entropy of a phase distribution. 0=delta, 1=uniform."""
    hist, _ = np.histogram(phases, bins=n_bins, range=(-np.pi, np.pi))
    p = hist / hist.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_max = np.log(n_bins)
    return H / H_max


def compute_stretching_hat_helical(solver, u_hat):
    """Decompose vortex stretching into same-helical and cross-helical parts.

    Returns: stretch_same_hat, stretch_cross_hat (each shape (3, N, N, N))
    """
    N = solver.N
    K = [solver.kx, solver.ky, solver.kz]

    u_p, u_m = solver.helical_decompose(u_hat)
    u_hat_plus = solver.helical_reconstruct(u_p, np.zeros_like(u_m))
    u_hat_minus = solver.helical_reconstruct(np.zeros_like(u_p), u_m)

    omega_hat_plus = solver.compute_vorticity_hat(u_hat_plus)
    omega_hat_minus = solver.compute_vorticity_hat(u_hat_minus)
    omega_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
    omega_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

    def compute_stretch_pair(omega_phys, u_h):
        grad_u = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                grad_u[i, j] = np.real(ifftn(1j * K[j] * u_h[i]))
        s = np.zeros((3, N, N, N))
        for i in range(3):
            for j in range(3):
                s[i] += omega_phys[j] * grad_u[i, j]
        return s

    stretch_same = (compute_stretch_pair(omega_plus, u_hat_plus) +
                    compute_stretch_pair(omega_minus, u_hat_minus))
    stretch_cross = (compute_stretch_pair(omega_plus, u_hat_minus) +
                     compute_stretch_pair(omega_minus, u_hat_plus))

    def to_fourier_dealiased(field):
        f_hat = np.array([fftn(field[i]) for i in range(3)])
        for i in range(3):
            f_hat[i] *= solver.dealias_mask
        return f_hat

    return to_fourier_dealiased(stretch_same), to_fourier_dealiased(stretch_cross)


def compute_total_stretching_hat(solver, u_hat):
    """Compute total vortex stretching (omega.grad)u in Fourier space."""
    N = solver.N
    K = [solver.kx, solver.ky, solver.kz]

    omega_hat = solver.compute_vorticity_hat(u_hat)
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

    grad_u = np.zeros((3, 3, N, N, N))
    for i in range(3):
        for j in range(3):
            grad_u[i, j] = np.real(ifftn(1j * K[j] * u_hat[i]))

    stretch = np.zeros((3, N, N, N))
    for i in range(3):
        for j in range(3):
            stretch[i] += omega[j] * grad_u[i, j]

    stretch_hat = np.array([fftn(stretch[i]) for i in range(3)])
    for i in range(3):
        stretch_hat[i] *= solver.dealias_mask
    return stretch_hat


def measure_finger_trap(solver, u_hat):
    """Compute all finger-trap diagnostics for a given flow state.

    Returns dict with per-shell arrays and scalar summaries.
    """
    N = solver.N
    kmax = N // 3  # dealiased range
    k_mag = np.sqrt(solver.k2)
    norm = 1.0 / N**6

    omega_hat = solver.compute_vorticity_hat(u_hat)
    stretch_hat = compute_total_stretching_hat(solver, u_hat)
    stretch_same_hat, stretch_cross_hat = compute_stretching_hat_helical(solver, u_hat)

    # Helical decomposition for phase entropy
    u_p, u_m = solver.helical_decompose(u_hat)
    phase_p = np.angle(u_p)
    phase_m = np.angle(u_m)

    # Per-shell quantities
    k_shells = np.arange(1, kmax + 1, dtype=float)
    n_shells = len(k_shells)

    T_Z = np.zeros(n_shells)        # total enstrophy transfer
    T_Z_same = np.zeros(n_shells)   # same-helical contribution
    T_Z_cross = np.zeros(n_shells)  # cross-helical contribution
    D_Z = np.zeros(n_shells)        # viscous dissipation
    Z_k = np.zeros(n_shells)        # enstrophy spectrum
    S_cross = np.zeros(n_shells)    # cross-helical phase entropy
    f_same = np.zeros(n_shells)     # same-helical fraction

    for ik in range(n_shells):
        k = ik + 1
        mask = (k_mag >= k - 0.5) & (k_mag < k + 0.5)
        n_modes = mask.sum()

        # Enstrophy transfer
        for i in range(3):
            T_Z[ik] += np.real(np.sum(
                np.conj(omega_hat[i][mask]) * stretch_hat[i][mask])) * norm
            T_Z_same[ik] += np.real(np.sum(
                np.conj(omega_hat[i][mask]) * stretch_same_hat[i][mask])) * norm
            T_Z_cross[ik] += np.real(np.sum(
                np.conj(omega_hat[i][mask]) * stretch_cross_hat[i][mask])) * norm

        # Viscous dissipation
        for i in range(3):
            D_Z[ik] += 2.0 * solver.nu * np.sum(
                solver.k2[mask] * np.abs(omega_hat[i][mask])**2) * norm

        # Enstrophy spectrum
        for i in range(3):
            Z_k[ik] += 0.5 * np.sum(np.abs(omega_hat[i][mask])**2) * norm

        # Protection fraction
        T_abs = np.abs(T_Z_same[ik]) + np.abs(T_Z_cross[ik])
        f_same[ik] = np.abs(T_Z_same[ik]) / T_abs if T_abs > 1e-30 else 0.5

        # Cross-helical phase entropy
        if n_modes >= 10:
            phases_plus = phase_p[mask]
            phases_minus = phase_m[mask]
            phase_diff = np.angle(np.exp(1j * (phases_plus - phases_minus)))
            S_cross[ik] = shannon_entropy_of_phases(phase_diff)
        else:
            S_cross[ik] = np.nan

    # --- SCALAR SUMMARIES ---

    # Total stretching magnitude across all shells
    total_stretching = np.sum(np.abs(T_Z))

    # Cross-helical fraction of total transfer at each shell
    T_abs_total = np.abs(T_Z)
    T_abs_cross = np.abs(T_Z_cross)
    cross_frac = np.where(T_abs_total > 1e-30, T_abs_cross / T_abs_total, 0.5)

    # --- NECK / THROAT ANALYSIS ---
    # The "neck" is the crossover zone where f_same transitions from >0.5 to <0.5
    # Find the transition: where does f_same cross 0.5?
    neck_center = 3  # default
    for ik in range(n_shells - 1):
        if f_same[ik] >= 0.5 and f_same[ik + 1] < 0.5:
            # Linear interpolation
            w = (0.5 - f_same[ik]) / (f_same[ik + 1] - f_same[ik])
            neck_center = k_shells[ik] + w
            break

    # Neck width: range of k where 0.3 < f_same < 0.7 (transition zone)
    in_transition = (f_same > 0.3) & (f_same < 0.7)
    if np.any(in_transition):
        k_trans = k_shells[in_transition]
        neck_width = k_trans[-1] - k_trans[0] + 1  # inclusive
        neck_lo = k_trans[0]
        neck_hi = k_trans[-1]
    else:
        neck_width = 1.0
        neck_lo = neck_center - 0.5
        neck_hi = neck_center + 0.5

    # Throughput at the neck: the MINIMUM of (1 - f_same) in the neck region
    # This measures how much cross-helical transfer gets through
    # Low throughput = finger trap is tight
    # Alternative: the suppression factor = (1 - max(cross_frac in neck))
    # Better metric: the "constriction" = min(f_same) * max(1-f_same) product
    # Actually simplest: throughput = cross_frac at neck center
    ik_neck = max(0, min(int(round(neck_center)) - 1, n_shells - 1))

    # Throughput = how efficiently stretching passes through the neck
    # = |T_Z(k_neck)| / max(|T_Z|) -- fraction of peak transfer at neck
    peak_T = np.max(np.abs(T_Z))
    throughput_transfer = np.abs(T_Z[ik_neck]) / peak_T if peak_T > 1e-30 else 0

    # Alternative throughput: ratio of enstrophy flux at neck to total production
    Pi_Z = np.cumsum(T_Z)
    throughput_flux = Pi_Z[ik_neck] / Pi_Z[-1] if abs(Pi_Z[-1]) > 1e-30 else 0

    # Suppression ratio at neck: how much does Leray projection suppress?
    # = 1 - |T_Z_cross(neck)| / (|T_Z_same(neck)| + |T_Z_cross(neck)|)
    suppression_at_neck = f_same[ik_neck]  # Higher f_same = more ordered = more suppressed

    return {
        'k_shells': k_shells,
        'T_Z': T_Z,
        'T_Z_same': T_Z_same,
        'T_Z_cross': T_Z_cross,
        'D_Z': D_Z,
        'Z_k': Z_k,
        'f_same': f_same,
        'S_cross': S_cross,
        'cross_frac': cross_frac,
        'Pi_Z': Pi_Z,
        'total_stretching': total_stretching,
        'peak_T': peak_T,
        'throughput_transfer': throughput_transfer,
        'throughput_flux': throughput_flux,
        'neck_center': neck_center,
        'neck_width': neck_width,
        'neck_lo': neck_lo,
        'neck_hi': neck_hi,
        'suppression_at_neck': suppression_at_neck,
        'enstrophy': solver.compute_enstrophy(u_hat),
        'energy': solver.compute_total_energy(u_hat),
    }


# ================================================================
# MAIN EXPERIMENT: Re SWEEP
# ================================================================

def run_finger_trap_experiment():
    """Run the Chinese finger trap test across Reynolds numbers."""
    print("=" * 70)
    print("CHINESE FINGER TRAP HYPOTHESIS TEST")
    print("=" * 70)
    print()
    print("Hypothesis: the harder the flow tries to blow up, the narrower")
    print("the throat at the neck (k~3). Anti-correlation between pressure")
    print("(total stretching) and throughput (transfer at neck).")
    print()

    # Use N=64 for better shell resolution; N=128 too slow for 5 Re values
    # N=64 gives kmax=21, enough to see the neck clearly
    N = 64
    dt = 0.002
    t_develop = 2.0

    Re_values = [400, 800, 1600, 3200, 6400]

    results = {}
    wall_start = clock.time()

    for Re in Re_values:
        print(f"\n{'='*50}")
        print(f"Re = {Re}, N = {N}")
        print(f"{'='*50}")

        solver = SpectralNS(N=N, Re=Re)
        u_hat = solver.taylor_green_ic()

        # Evolve
        t = 0.0
        n_steps = int(t_develop / dt)
        for step in range(n_steps):
            u_hat = solver.step_rk4(u_hat, dt)
            t += dt
            # Safety: check for NaN
            if step % 200 == 0:
                E = solver.compute_total_energy(u_hat)
                if np.isnan(E) or E > 1e10:
                    print(f"  WARNING: blowup at step {step}, t={t:.3f}, E={E}")
                    break

        E = solver.compute_total_energy(u_hat)
        Z = solver.compute_enstrophy(u_hat)
        print(f"  t = {t:.3f}, E = {E:.4e}, Z = {Z:.4e}")

        if np.isnan(E):
            print(f"  SKIPPING Re={Re} (numerical blowup)")
            continue

        # Measure finger trap diagnostics
        meas = measure_finger_trap(solver, u_hat)
        results[Re] = meas

        # Print summary
        print(f"  Total stretching:  {meas['total_stretching']:.6e}")
        print(f"  Throughput (flux): {meas['throughput_flux']:.4f}")
        print(f"  Throughput (xfer): {meas['throughput_transfer']:.4f}")
        print(f"  Neck center:      k ~ {meas['neck_center']:.1f}")
        print(f"  Neck width:       {meas['neck_width']:.1f} shells")
        print(f"  f_same at neck:   {meas['suppression_at_neck']:.4f}")

        # Print f_same at specific k values
        for k_check in [1, 2, 3, 5, 10]:
            if k_check <= len(meas['f_same']):
                print(f"  f_same(k={k_check:>2}):    {meas['f_same'][k_check-1]:.4f}")

    wall_time = clock.time() - wall_start
    print(f"\n\nTotal wall time: {wall_time:.1f}s")

    return results


# ================================================================
# ANALYSIS
# ================================================================

def analyze_finger_trap(results):
    """Quantitative analysis of the finger trap hypothesis."""
    print("\n\n" + "=" * 70)
    print("CHINESE FINGER TRAP -- QUANTITATIVE ANALYSIS")
    print("=" * 70)

    Re_arr = np.array(sorted(results.keys()))
    n_re = len(Re_arr)

    # Extract scalar quantities
    total_stretch = np.array([results[Re]['total_stretching'] for Re in Re_arr])
    throughput_flux = np.array([results[Re]['throughput_flux'] for Re in Re_arr])
    throughput_xfer = np.array([results[Re]['throughput_transfer'] for Re in Re_arr])
    neck_width = np.array([results[Re]['neck_width'] for Re in Re_arr])
    neck_center = np.array([results[Re]['neck_center'] for Re in Re_arr])
    enstrophy = np.array([results[Re]['enstrophy'] for Re in Re_arr])

    # --- TEST 1: Anti-correlation (throughput vs pressure) ---
    print("\n--- TEST 1: ANTI-CORRELATION (Finger Trap) ---")
    print(f"\n{'Re':>6} {'TotalStretch':>14} {'Throughput':>12} {'NeckWidth':>10} {'NeckCenter':>11}")
    print("-" * 60)
    for i, Re in enumerate(Re_arr):
        print(f"{Re:>6} {total_stretch[i]:>14.6e} {throughput_xfer[i]:>12.4f} "
              f"{neck_width[i]:>10.1f} {neck_center[i]:>11.1f}")

    # Correlation between total stretching and throughput
    if n_re >= 3:
        corr_stretch_thru = np.corrcoef(total_stretch, throughput_xfer)[0, 1]
        corr_stretch_flux = np.corrcoef(total_stretch, throughput_flux)[0, 1]
        print(f"\n  Pearson correlation (stretch vs throughput_xfer): {corr_stretch_thru:.4f}")
        print(f"  Pearson correlation (stretch vs throughput_flux): {corr_stretch_flux:.4f}")

        if corr_stretch_thru < -0.5:
            print("  --> ANTI-CORRELATION DETECTED: finger trap hypothesis SUPPORTED")
        elif corr_stretch_thru < 0:
            print("  --> WEAK anti-correlation: partial support")
        else:
            print("  --> POSITIVE correlation: finger trap hypothesis NOT supported")

    # --- TEST 2: Neck width vs Re ---
    print("\n--- TEST 2: NECK WIDTH SCALING ---")
    if n_re >= 3:
        log_Re = np.log(Re_arr)
        log_nw = np.log(np.maximum(neck_width, 0.1))
        coeffs_nw = np.polyfit(log_Re, log_nw, 1)
        print(f"  Neck width ~ Re^({coeffs_nw[0]:.3f})")
        if coeffs_nw[0] < 0:
            print("  --> Neck NARROWS with Re (finger trap tightens)")
        else:
            print("  --> Neck WIDENS with Re")

    # --- TEST 3: Throughput power law ---
    print("\n--- TEST 3: THROUGHPUT POWER LAW ---")
    if n_re >= 3:
        log_Re = np.log(Re_arr)
        # Use transfer throughput (more direct)
        log_tp = np.log(np.maximum(throughput_xfer, 1e-10))
        coeffs_tp = np.polyfit(log_Re, log_tp, 1)
        print(f"  Throughput (xfer) ~ Re^({coeffs_tp[0]:.3f})")
        print(f"  Constriction rate: d(throughput)/d(Re) ~ Re^({coeffs_tp[0]-1:.3f})")

        # Flux throughput
        log_fp = np.log(np.maximum(np.abs(throughput_flux), 1e-10))
        coeffs_fp = np.polyfit(log_Re, log_fp, 1)
        print(f"  Throughput (flux)  ~ Re^({coeffs_fp[0]:.3f})")

    # --- TEST 4: f_same sharpening ---
    print("\n--- TEST 4: TRANSITION SHARPENING ---")
    print("  f_same at selected wavenumbers across Re:")
    k_check = [1, 2, 3, 5, 10]
    print(f"  {'Re':>6}", end="")
    for k in k_check:
        print(f"  {'k='+str(k):>8}", end="")
    print()
    print("  " + "-" * (8 + 10 * len(k_check)))
    for Re in Re_arr:
        fs = results[Re]['f_same']
        print(f"  {Re:>6}", end="")
        for k in k_check:
            if k <= len(fs):
                print(f"  {fs[k-1]:>8.4f}", end="")
            else:
                print(f"  {'--':>8}", end="")
        print()

    # Measure "sharpness" of transition: |f_same(k=1) - f_same(k=5)| / (5-1)
    print("\n  Transition gradient (f_same(k=1) - f_same(k=5)) / 4:")
    for Re in Re_arr:
        fs = results[Re]['f_same']
        if len(fs) >= 5:
            grad = (fs[0] - fs[4]) / 4.0
            print(f"    Re={Re}: gradient = {grad:.4f}")

    # --- TEST 5: Total stretching scaling ---
    print("\n--- TEST 5: TOTAL STRETCHING SCALING ---")
    if n_re >= 3:
        log_Re = np.log(Re_arr)
        log_ts = np.log(total_stretch)
        coeffs_ts = np.polyfit(log_Re, log_ts, 1)
        print(f"  Total stretching ~ Re^({coeffs_ts[0]:.3f})")

    # --- OVERALL VERDICT ---
    print("\n" + "=" * 70)
    print("FINGER TRAP VERDICT")
    print("=" * 70)

    verdicts = []

    if n_re >= 3:
        # Check 1: anti-correlation
        if corr_stretch_thru < -0.5:
            verdicts.append("PASS: Strong anti-correlation between pressure and throughput")
        elif corr_stretch_thru < 0:
            verdicts.append("PARTIAL: Weak anti-correlation (suggestive but not decisive)")
        else:
            verdicts.append("FAIL: No anti-correlation found")

        # Check 2: neck narrows
        if coeffs_nw[0] < -0.1:
            verdicts.append("PASS: Neck narrows with Re (constriction)")
        elif coeffs_nw[0] < 0.1:
            verdicts.append("PARTIAL: Neck width roughly constant")
        else:
            verdicts.append("FAIL: Neck widens with Re (no constriction)")

        # Check 3: throughput decreases
        if coeffs_tp[0] < -0.1:
            verdicts.append("PASS: Throughput decreases as power law in Re")
        elif coeffs_tp[0] < 0.1:
            verdicts.append("PARTIAL: Throughput roughly constant")
        else:
            verdicts.append("FAIL: Throughput increases with Re")

    for v in verdicts:
        print(f"  {v}")

    n_pass = sum(1 for v in verdicts if v.startswith("PASS"))
    n_partial = sum(1 for v in verdicts if v.startswith("PARTIAL"))
    n_fail = sum(1 for v in verdicts if v.startswith("FAIL"))

    print(f"\n  Score: {n_pass} PASS, {n_partial} PARTIAL, {n_fail} FAIL out of {len(verdicts)}")

    if n_pass >= 2:
        print("\n  CONCLUSION: Chinese finger trap mechanism is OPERATIVE.")
        print("  The neck constricts under pressure -- pulling harder tightens the grip.")
    elif n_pass + n_partial >= 2:
        print("\n  CONCLUSION: Partial evidence for finger trap. Suggestive but needs")
        print("  higher resolution / wider Re range to confirm.")
    else:
        print("\n  CONCLUSION: Finger trap mechanism NOT detected at these parameters.")
        print("  The throat does not constrict in response to increased pressure.")

    return {
        'Re': Re_arr,
        'total_stretch': total_stretch,
        'throughput_xfer': throughput_xfer,
        'throughput_flux': throughput_flux,
        'neck_width': neck_width,
        'neck_center': neck_center,
        'coeffs_tp': coeffs_tp if n_re >= 3 else None,
        'coeffs_nw': coeffs_nw if n_re >= 3 else None,
    }


# ================================================================
# PLOTTING
# ================================================================

def plot_finger_trap(results, analysis):
    """4-panel plot for the finger trap hypothesis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Chinese Finger Trap: Does the Throat Constrict Under Pressure?\n"
                 "Throughput at neck vs Reynolds number",
                 fontsize=13, fontweight='bold')

    colors = ['#364FC7', '#e67700', '#2b8a3e', '#c92a2a', '#7048e8']
    Re_arr = analysis['Re']

    # ---- Panel 1: Throughput vs Re (log-log, fit power law) ----
    ax = axes[0, 0]
    ax.loglog(Re_arr, analysis['throughput_xfer'], 'ko-', linewidth=2.5,
              markersize=8, label='Transfer throughput', zorder=5)
    ax.loglog(Re_arr, np.abs(analysis['throughput_flux']), 's--', color='#364FC7',
              linewidth=2, markersize=7, label='Flux throughput', zorder=4)

    # Power law fit
    if analysis['coeffs_tp'] is not None:
        Re_fit = np.logspace(np.log10(Re_arr[0] * 0.8),
                             np.log10(Re_arr[-1] * 1.2), 50)
        tp_fit = np.exp(np.polyval(analysis['coeffs_tp'], np.log(Re_fit)))
        exponent = analysis['coeffs_tp'][0]
        ax.loglog(Re_fit, tp_fit, 'r:', linewidth=1.5,
                  label=f'Fit: Re^({exponent:.2f})', alpha=0.7)

    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel('Throughput at neck')
    ax.set_title('Panel 1: Throughput vs Re\n'
                 'Decreasing = finger trap tightens')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # ---- Panel 2: Neck width vs Re ----
    ax = axes[0, 1]
    ax.semilogx(Re_arr, analysis['neck_width'], 'ko-', linewidth=2.5,
                markersize=8, label='Transition width', zorder=5)
    ax.semilogx(Re_arr, analysis['neck_center'], 's--', color='#e67700',
                linewidth=2, markersize=7, label='Neck center', zorder=4)

    if analysis['coeffs_nw'] is not None:
        exponent_nw = analysis['coeffs_nw'][0]
        ax.set_title(f'Panel 2: Neck Geometry vs Re\n'
                     f'Width ~ Re^({exponent_nw:.2f})')
    else:
        ax.set_title('Panel 2: Neck Geometry vs Re')

    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel('Wavenumber shells')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: f_same at k=1,2,3,5,10 vs Re ----
    ax = axes[1, 0]
    k_targets = [1, 2, 3, 5, 10]
    for i, k_tgt in enumerate(k_targets):
        fs_at_k = []
        for Re in Re_arr:
            fs = results[Re]['f_same']
            if k_tgt <= len(fs):
                fs_at_k.append(fs[k_tgt - 1])
            else:
                fs_at_k.append(np.nan)
        ax.semilogx(Re_arr, fs_at_k, 'o-', color=colors[i % len(colors)],
                     linewidth=2, markersize=6, label=f'k={k_tgt}')

    ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5,
               label='50/50 equipartition')
    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel('f_same(k)')
    ax.set_title('Panel 3: Same-Helical Fraction vs Re\n'
                 'Spreading = transition sharpens')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # ---- Panel 4: Total stretching vs throughput (anti-correlation) ----
    ax = axes[1, 1]
    ax.scatter(analysis['total_stretch'], analysis['throughput_xfer'],
               s=100, c='#364FC7', edgecolors='k', linewidth=1, zorder=5)

    # Annotate each point with Re
    for i, Re in enumerate(Re_arr):
        ax.annotate(f'Re={Re}',
                    (analysis['total_stretch'][i], analysis['throughput_xfer'][i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Trend line
    if len(Re_arr) >= 3:
        corr = np.corrcoef(analysis['total_stretch'], analysis['throughput_xfer'])[0, 1]
        # Linear fit for trend line
        coeffs_trend = np.polyfit(analysis['total_stretch'], analysis['throughput_xfer'], 1)
        x_trend = np.linspace(analysis['total_stretch'].min() * 0.9,
                              analysis['total_stretch'].max() * 1.1, 50)
        ax.plot(x_trend, np.polyval(coeffs_trend, x_trend), 'r--', linewidth=1.5,
                alpha=0.6, label=f'r = {corr:.3f}')

    ax.set_xlabel('Total stretching (pressure)')
    ax.set_ylabel('Throughput at neck')
    ax.set_title('Panel 4: Pressure vs Throughput\n'
                 'Negative slope = finger trap')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = 'h:/tmp/chinese_finger_trap.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
    return outpath


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 70)
    print("CHINESE FINGER TRAP -- NAVIER-STOKES TURBULENCE")
    print("The harder you pull, the tighter the grip")
    print("=" * 70)
    print()

    wall_start = clock.time()

    # Run the Re sweep
    results = run_finger_trap_experiment()

    if len(results) < 3:
        print("\nNot enough successful Re values for analysis. Need >= 3.")
        return results, None, None

    # Analyze
    analysis = analyze_finger_trap(results)

    # Plot
    print("\nGenerating plots...")
    outpath = plot_finger_trap(results, analysis)

    total_time = clock.time() - wall_start
    print(f"\nTotal wall time: {total_time:.1f}s")

    return results, analysis, outpath


if __name__ == "__main__":
    main()
