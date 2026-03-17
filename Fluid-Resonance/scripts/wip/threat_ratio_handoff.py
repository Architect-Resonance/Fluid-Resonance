"""
THREAT RATIO R(k) -- SCALE-RESOLVED PROTECTION MEASUREMENT
==========================================================
S100-M1e: The Wanderer's insight -- protection TRANSFORMS from order to disorder.

R(k) = T_Z(k) / [2 nu k^2 Z(k)]

where:
  T_Z(k) = enstrophy transfer into shell k (from vortex stretching)
  2 nu k^2 Z(k) = viscous dissipation at shell k

R(k) < 1 means viscosity wins at that scale. The Wanderer's claim:
  - Low k: R(k) < 1 because ordered structures (Beltrami alignment, helicity
    conservation) suppress stretching -> "order protection"
  - High k: R(k) < 1 because CLT, depolarization, Berry frustration, isotropy
    suppress stretching -> "disorder protection"
  - The handoff between these two regimes is the key to regularity.

MEASUREMENTS:
  1. R(k) vs k at multiple Re (400, 800, 1600, 3200)
  2. Decompose T_Z into same-helical (ordered) and cross-helical (disordered)
  3. Time-resolve the handoff during turbulence development
  4. max_k R(k) scaling with Re -- the killer test

METHOD:
  T_Z(k) = Re[ sum_{|k| in shell} omega_hat*(k) . F[omega . grad(u)](k) ]
  This is the stretching term projected shell-by-shell.
  Decompose velocity into h+/h- to separate ordered/disordered contributions.

SpectralNS functional API: u_hat = solver.step_rk4(u_hat, dt), external time.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import os
import sys
import time as clock

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ================================================================
# CORE MEASUREMENT FUNCTIONS
# ================================================================

def compute_stretching_hat(solver, u_hat):
    """Compute vortex stretching term (omega . grad)u in Fourier space.

    The stretching term in the vorticity equation is:
        (omega . nabla) u  =  omega_j * du_i/dx_j

    Returns shape (3, N, N, N) complex array.
    """
    N = solver.N
    K = [solver.kx, solver.ky, solver.kz]

    omega_hat = solver.compute_vorticity_hat(u_hat)
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

    # Velocity gradient: du_i/dx_j in physical space
    grad_u = np.zeros((3, 3, N, N, N))
    for i in range(3):
        for j in range(3):
            grad_u[i, j] = np.real(ifftn(1j * K[j] * u_hat[i]))

    # Stretching: (omega . nabla) u_i = omega_j * du_i/dx_j
    stretch = np.zeros((3, N, N, N))
    for i in range(3):
        for j in range(3):
            stretch[i] += omega[j] * grad_u[i, j]

    # To Fourier + dealias
    stretch_hat = np.array([fftn(stretch[i]) for i in range(3)])
    for i in range(3):
        stretch_hat[i] *= solver.dealias_mask
    return stretch_hat


def compute_stretching_hat_helical(solver, u_hat):
    """Decompose stretching into same-helical and cross-helical contributions.

    Same-helical: omega_s . grad(u_s) for s in {+, -}
    Cross-helical: omega_+ . grad(u_-) + omega_- . grad(u_+)

    Returns: stretch_same_hat, stretch_cross_hat (each shape (3, N, N, N))
    """
    N = solver.N
    K = [solver.kx, solver.ky, solver.kz]

    # Decompose velocity
    u_p, u_m = solver.helical_decompose(u_hat)
    u_hat_plus = solver.helical_reconstruct(u_p, np.zeros_like(u_m))
    u_hat_minus = solver.helical_reconstruct(np.zeros_like(u_p), u_m)

    # Vorticities
    omega_hat_plus = solver.compute_vorticity_hat(u_hat_plus)
    omega_hat_minus = solver.compute_vorticity_hat(u_hat_minus)
    omega_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
    omega_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

    def compute_stretch_pair(omega_phys, u_h):
        """(omega . nabla) u from given omega (physical) and u_hat (Fourier)."""
        grad_u = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                grad_u[i, j] = np.real(ifftn(1j * K[j] * u_h[i]))
        s = np.zeros((3, N, N, N))
        for i in range(3):
            for j in range(3):
                s[i] += omega_phys[j] * grad_u[i, j]
        return s

    # Same-helical: omega+ . grad(u+) + omega- . grad(u-)
    stretch_same = (compute_stretch_pair(omega_plus, u_hat_plus) +
                    compute_stretch_pair(omega_minus, u_hat_minus))

    # Cross-helical: omega+ . grad(u-) + omega- . grad(u+)
    stretch_cross = (compute_stretch_pair(omega_plus, u_hat_minus) +
                     compute_stretch_pair(omega_minus, u_hat_plus))

    def to_fourier_dealiased(field):
        f_hat = np.array([fftn(field[i]) for i in range(3)])
        for i in range(3):
            f_hat[i] *= solver.dealias_mask
        return f_hat

    return to_fourier_dealiased(stretch_same), to_fourier_dealiased(stretch_cross)


def compute_shell_enstrophy_transfer(solver, u_hat):
    """Compute shell-by-shell enstrophy transfer T_Z(k) and dissipation D_Z(k).

    T_Z(k) = Re[ sum_{|k'| in shell k} omega_hat*(k') . stretch_hat(k') ]
    D_Z(k) = 2 * nu * sum_{|k'| in shell k} k'^2 |omega_hat(k')|^2

    Returns:
        k_shells: array of shell wavenumbers
        T_Z: enstrophy transfer per shell (total)
        T_Z_same: same-helical contribution
        T_Z_cross: cross-helical contribution
        D_Z: viscous dissipation per shell
        Z_k: enstrophy spectrum (per shell)
    """
    N = solver.N
    # IMPORTANT: Use N//2 for shells, not N//3. The dealiasing mask only affects
    # the nonlinear (stretching) term — viscous dissipation acts on ALL modes.
    # Truncating shells at N//3 misses most dissipation, giving spurious S/D >> 1.
    kmax_full = N // 2  # full Nyquist range for D_Z
    kmax_nonlin = N // 3  # dealiased range for T_Z (stretching is 0 beyond this)
    k_mag = np.sqrt(solver.k2)

    omega_hat = solver.compute_vorticity_hat(u_hat)
    stretch_hat = compute_stretching_hat(solver, u_hat)
    stretch_same_hat, stretch_cross_hat = compute_stretching_hat_helical(solver, u_hat)

    k_shells = np.arange(1, kmax_full + 1, dtype=float)
    n_shells = len(k_shells)
    T_Z = np.zeros(n_shells)
    T_Z_same = np.zeros(n_shells)
    T_Z_cross = np.zeros(n_shells)
    D_Z = np.zeros(n_shells)
    Z_k = np.zeros(n_shells)

    norm = 1.0 / N**6  # FFT normalization (full FFT: fftn gives N^3 * coeff)

    for ik in range(n_shells):
        k = ik + 1
        mask = (k_mag >= k - 0.5) & (k_mag < k + 0.5)

        # Enstrophy transfer: Re[ omega* . stretch ]
        # Only meaningful for k <= kmax_nonlin (stretch_hat is dealiased)
        if k <= kmax_nonlin:
            for i in range(3):
                T_Z[ik] += np.real(np.sum(
                    np.conj(omega_hat[i][mask]) * stretch_hat[i][mask]
                )) * norm

                T_Z_same[ik] += np.real(np.sum(
                    np.conj(omega_hat[i][mask]) * stretch_same_hat[i][mask]
                )) * norm

                T_Z_cross[ik] += np.real(np.sum(
                    np.conj(omega_hat[i][mask]) * stretch_cross_hat[i][mask]
                )) * norm

        # Viscous dissipation: 2 nu k^2 |omega|^2 (ALL modes)
        for i in range(3):
            D_Z[ik] += 2.0 * solver.nu * np.sum(
                solver.k2[mask] * np.abs(omega_hat[i][mask])**2
            ) * norm

        # Enstrophy spectrum (ALL modes)
        for i in range(3):
            Z_k[ik] += 0.5 * np.sum(np.abs(omega_hat[i][mask])**2) * norm

    return k_shells, T_Z, T_Z_same, T_Z_cross, D_Z, Z_k


def compute_threat_ratio(T_Z, D_Z):
    """R(k) = T_Z(k) / D_Z(k). Safe division."""
    return np.where(D_Z > 1e-30, T_Z / D_Z, 0.0)


def compute_enstrophy_flux(T_Z):
    """Enstrophy flux: Pi_Z(k) = cumulative sum of T_Z from shell 1 to k.

    Pi_Z(k) > 0 means net enstrophy flow TOWARD higher k (forward cascade).
    Pi_Z(k) = 0 at k_max if budget is balanced (all transferred enstrophy
    is eventually dissipated).
    """
    return np.cumsum(T_Z)


def compute_protection_fraction(T_Z_same, T_Z_cross):
    """At each shell, fraction of stretching from same vs cross helical.

    This is the key diagnostic for the Wanderer's handoff:
    - Low k: same-helical dominates (ordered protection)
    - High k: cross-helical dominates (disordered protection)
    """
    T_abs = np.abs(T_Z_same) + np.abs(T_Z_cross)
    safe = T_abs > 1e-30
    f_same = np.where(safe, np.abs(T_Z_same) / T_abs, 0.5)
    f_cross = np.where(safe, np.abs(T_Z_cross) / T_abs, 0.5)
    return f_same, f_cross


def compute_net_budget(T_Z, D_Z):
    """Net enstrophy budget per shell: B(k) = D_Z(k) - T_Z(k).

    B(k) > 0: dissipation wins at this shell (enstrophy sink)
    B(k) < 0: stretching wins (enstrophy source, cascade forward)

    For regularity, we need sum(B) > 0 (global dissipation dominance).
    """
    return D_Z - T_Z


# ================================================================
# EXPERIMENT 1: R(k) vs k AT MULTIPLE Re
# ================================================================

def experiment_re_scaling():
    """Measure R(k) at multiple Re values after turbulence develops."""
    print("=" * 70)
    print("EXPERIMENT 1: R(k) vs k AT MULTIPLE Re")
    print("=" * 70)
    print()

    N = 32
    Re_values = [400, 800, 1600, 3200]
    dt = 0.003
    t_develop = 2.0  # time to develop turbulence

    results = {}

    for Re in Re_values:
        print(f"\n--- Re = {Re} ---")
        solver = SpectralNS(N=N, Re=Re)
        u_hat = solver.taylor_green_ic()

        # Evolve to developed turbulence
        t = 0.0
        n_steps = int(t_develop / dt)
        for step in range(n_steps):
            u_hat = solver.step_rk4(u_hat, dt)
            t += dt

        # Check enstrophy
        Z = solver.compute_enstrophy(u_hat)
        E = solver.compute_total_energy(u_hat)
        print(f"  t = {t:.2f}, Z = {Z:.4e}, E = {E:.4e}")

        # Measure R(k)
        k_shells, T_Z, T_Z_same, T_Z_cross, D_Z, Z_k = \
            compute_shell_enstrophy_transfer(solver, u_hat)
        R = compute_threat_ratio(T_Z, D_Z)
        R_same = compute_threat_ratio(T_Z_same, D_Z)
        R_cross = compute_threat_ratio(T_Z_cross, D_Z)

        # Protection fraction decomposition
        f_same, f_cross = compute_protection_fraction(T_Z_same, T_Z_cross)
        # Enstrophy flux
        Pi_Z = compute_enstrophy_flux(T_Z)
        Pi_same = compute_enstrophy_flux(T_Z_same)
        Pi_cross = compute_enstrophy_flux(T_Z_cross)
        # Net budget
        B = compute_net_budget(T_Z, D_Z)

        results[Re] = {
            'k': k_shells, 'R': R, 'R_same': R_same, 'R_cross': R_cross,
            'T_Z': T_Z, 'T_Z_same': T_Z_same, 'T_Z_cross': T_Z_cross,
            'D_Z': D_Z, 'Z_k': Z_k,
            'f_same': f_same, 'f_cross': f_cross,
            'Pi_Z': Pi_Z, 'Pi_same': Pi_same, 'Pi_cross': Pi_cross,
            'B': B,
        }

        # Print key values
        active = Z_k > 1e-10 * np.max(Z_k)
        # Global enstrophy budget from existing method (validated, factor-of-2 convention)
        Z_total, S_total, D_total = solver.compute_enstrophy_budget(u_hat)
        global_SD = S_total / D_total if D_total > 0 else 0
        # Cross-validate: shell sums should match global
        shell_S = np.sum(T_Z)
        shell_D = np.sum(D_Z)
        shell_SD = shell_S / shell_D if shell_D > 0 else 0
        # Note: existing code uses 2x convention: S_exist = 2*<omega.S.omega>, D_exist = 2*nu*<|nabla omega|^2>
        # Our shell T_Z = <omega . stretch> per shell, D_Z = 2*nu*k^2*|omega|^2 per shell
        # So sum(T_Z)/sum(D_Z) should ~ S_exist/(2*D_exist) = (S/D)/2... No wait:
        # S_exist = 2 * sum(T_Z) because S_exist = 2*<omega.S.omega> and <omega.S.omega> = <omega.(omega.nabla)u>
        # Hmm actually omega . S . omega = omega_i S_ij omega_j, and (omega.nabla)u_i = omega_j du_i/dx_j
        # omega . (omega.nabla)u = omega_i omega_j du_i/dx_j = omega_i omega_j S_ij (antisym part vanishes)
        # So <omega . (omega.nabla)u> = <omega . S . omega>, and S_exist = 2 * this.
        # D_exist = 2*nu*<|nabla omega|^2> = sum(D_Z) since our D_Z already has the factor 2.
        # So S_exist/D_exist = 2*sum(T_Z)/sum(D_Z) = 2*shell_SD
        print(f"  [validation] existing S/D={global_SD:.4f}, "
              f"2*shell_S/shell_D={2*shell_SD:.4f}, "
              f"ratio match: {abs(global_SD - 2*shell_SD) < 0.1*abs(global_SD)}")
        # Protection crossover
        crossover_k = 'N/A'
        for ik in range(len(k_shells) - 1):
            if active[ik] and f_same[ik] > f_cross[ik] and f_same[ik+1] < f_cross[ik+1]:
                crossover_k = f'{k_shells[ik]:.0f}'
                break
        # kmax for inertial range (where stretching is nonzero)
        kmax_inertial = N // 3
        fs_inertial = f_same[:kmax_inertial]
        print(f"  GLOBAL S/D = {global_SD:.4f} (S>D during cascade is normal)")
        print(f"  Protection: f_same(k=1)={f_same[0]:.3f} -> "
              f"f_same(k={kmax_inertial})={fs_inertial[-1]:.3f}")
        print(f"  Crossover: k ~ {crossover_k}")
        results[Re]['global_SD'] = global_SD
        results[Re]['kmax_inertial'] = kmax_inertial

    # Print summary table
    print("\n\n" + "=" * 70)
    print("Re SCALING -- PROTECTION HANDOFF")
    print("=" * 70)
    print(f"{'Re':>6} {'k_cross':>8} {'f_same(k=1)':>12} "
          f"{'f_same(k=10)':>13} {'drop':>7}")
    print("-" * 50)
    for Re, r in results.items():
        kmi = r['kmax_inertial']
        # Find crossover in inertial range
        k_cross_str = '--'
        for ik in range(min(kmi, len(r['k'])) - 1):
            if r['f_same'][ik] > r['f_cross'][ik] and \
               r['f_same'][ik+1] < r['f_cross'][ik+1]:
                k_cross_str = f"{r['k'][ik]:.0f}"
                break
        fs_low = r['f_same'][0]
        fs_high = r['f_same'][kmi - 1]  # last inertial shell
        drop = fs_low - fs_high
        print(f"{Re:>6} {k_cross_str:>8} {fs_low:>12.4f} "
              f"{fs_high:>13.4f} {drop:>7.4f}")

    return results


# ================================================================
# EXPERIMENT 2: TIME EVOLUTION OF R(k) -- THE HANDOFF
# ================================================================

def experiment_time_evolution():
    """Track R(k) over time to see the order->disorder handoff."""
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 2: TIME EVOLUTION OF R(k) -- ORDER->DISORDER HANDOFF")
    print("=" * 70)
    print()

    N = 32
    Re = 800
    dt = 0.003
    snap_times = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    solver = SpectralNS(N=N, Re=Re)
    u_hat = solver.taylor_green_ic()
    t = 0.0

    snapshots = {}

    for t_snap in snap_times:
        # Evolve to snapshot
        while t < t_snap - 1e-6:
            step_dt = min(dt, t_snap - t)
            u_hat = solver.step_rk4(u_hat, step_dt)
            t += step_dt

        k_shells, T_Z, T_Z_same, T_Z_cross, D_Z, Z_k = \
            compute_shell_enstrophy_transfer(solver, u_hat)
        R = compute_threat_ratio(T_Z, D_Z)
        R_same = compute_threat_ratio(T_Z_same, D_Z)
        R_cross = compute_threat_ratio(T_Z_cross, D_Z)

        Z = solver.compute_enstrophy(u_hat)
        E = solver.compute_total_energy(u_hat)

        f_same, f_cross = compute_protection_fraction(T_Z_same, T_Z_cross)
        Pi_Z = compute_enstrophy_flux(T_Z)
        Pi_same = compute_enstrophy_flux(T_Z_same)
        Pi_cross = compute_enstrophy_flux(T_Z_cross)

        # Global budget
        _, S_glob, D_glob = solver.compute_enstrophy_budget(u_hat)
        global_SD = S_glob / D_glob if D_glob > 0 else 0

        snapshots[t_snap] = {
            'k': k_shells, 'R': R, 'R_same': R_same, 'R_cross': R_cross,
            'T_Z': T_Z, 'T_Z_same': T_Z_same, 'T_Z_cross': T_Z_cross,
            'D_Z': D_Z, 'Z_k': Z_k, 'Z_total': Z, 'E_total': E,
            'f_same': f_same, 'f_cross': f_cross,
            'Pi_Z': Pi_Z, 'Pi_same': Pi_same, 'Pi_cross': Pi_cross,
            'global_SD': global_SD,
        }

        active = Z_k > 1e-10 * np.max(Z_k)
        # Find crossover
        k_cross_str = '--'
        for ik in range(len(k_shells) - 1):
            if active[ik] and f_same[ik] > f_cross[ik] and f_same[ik+1] < f_cross[ik+1]:
                k_cross_str = f"{k_shells[ik]:.0f}"
                break

        kmi = N // 3
        fs_end = f_same[kmi - 1] if kmi <= len(f_same) else f_same[-1]
        print(f"  t={t_snap:.1f}: Z={Z:.3e} S/D={global_SD:.4f} "
              f"k_cross={k_cross_str:>3} "
              f"f_same(k=1)={f_same[0]:.3f} f_same(k={kmi})={fs_end:.3f}")

    return snapshots


# ================================================================
# EXPERIMENT 3: MULTIPLE ICs -- ROBUSTNESS
# ================================================================

def experiment_multiple_ics():
    """Test R(k) with different ICs: TG, random, imbalanced helical."""
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 3: R(k) ROBUSTNESS ACROSS INITIAL CONDITIONS")
    print("=" * 70)
    print()

    N = 32
    Re = 800
    dt = 0.003
    t_develop = 2.0

    ics = {}
    solver = SpectralNS(N=N, Re=Re)
    ics['Taylor-Green'] = solver.taylor_green_ic()
    ics['Pelz'] = solver.pelz_ic()
    ics['Random'] = solver.random_ic(seed=42)
    ics['Imbalanced-80/20'] = solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8)
    ics['Imbalanced-95/5'] = solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.95)

    results = {}

    for ic_name, u_hat_ic in ics.items():
        print(f"\n--- {ic_name} ---")
        u_hat = u_hat_ic.copy()
        t = 0.0
        n_steps = int(t_develop / dt)
        for step in range(n_steps):
            u_hat = solver.step_rk4(u_hat, dt)
            t += dt

        k_shells, T_Z, T_Z_same, T_Z_cross, D_Z, Z_k = \
            compute_shell_enstrophy_transfer(solver, u_hat)
        R = compute_threat_ratio(T_Z, D_Z)
        R_same = compute_threat_ratio(T_Z_same, D_Z)
        R_cross = compute_threat_ratio(T_Z_cross, D_Z)

        results[ic_name] = {
            'k': k_shells, 'R': R, 'R_same': R_same, 'R_cross': R_cross,
            'T_Z': T_Z, 'D_Z': D_Z, 'Z_k': Z_k,
        }

        active = Z_k > 1e-10 * np.max(Z_k)
        if np.any(active):
            max_R = np.max(R[active])
            k_max_R = k_shells[active][np.argmax(R[active])]
            print(f"  max R(k) = {max_R:.4f} at k = {k_max_R:.0f}, "
                  f"R<1: {'YES' if max_R < 1 else 'NO'}")

    return results


# ================================================================
# PLOTTING
# ================================================================

def plot_all(re_results, time_snapshots, ic_results):
    """Generate comprehensive plots focused on protection transformation."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle("Protection Transformation: Order -> Disorder Handoff\n"
                 "Enstrophy stretching decomposed by helical sector",
                 fontsize=14, fontweight='bold')

    colors = ['#364FC7', '#e67700', '#2b8a3e', '#c92a2a', '#7048e8']

    # ---- Panel 1: Protection fraction f_same(k) vs k at multiple Re ----
    ax = axes[0, 0]
    for i, (Re, r) in enumerate(re_results.items()):
        kmi = r['kmax_inertial']
        k_a = r['k'][:kmi]
        ax.plot(k_a, r['f_same'][:kmi], 'o-', color=colors[i % len(colors)],
                linewidth=2, markersize=4, label=f'Re={Re}')
    ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5,
               label='50/50 (equipartition)')
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('f_same(k)')
    ax.set_title('Same-helical fraction of stretching\n'
                 'High = ordered protection, Low = disordered')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # ---- Panel 2: Enstrophy flux decomposition at Re=800 ----
    ax = axes[0, 1]
    r = re_results.get(800, list(re_results.values())[0])
    active = r['Z_k'] > 1e-10 * np.max(r['Z_k'])
    k_a = r['k'][active]
    ax.plot(k_a, r['Pi_Z'][active], 'ko-', linewidth=2, markersize=4,
            label='Pi_Z total (enstrophy flux)')
    ax.plot(k_a, r['Pi_same'][active], 's--', color='#364FC7', linewidth=2,
            markersize=4, label='Pi_same (ordered)')
    ax.plot(k_a, r['Pi_cross'][active], 'D--', color='#e67700', linewidth=2,
            markersize=4, label='Pi_cross (disordered)')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('Cumulative enstrophy flux')
    ax.set_title('Enstrophy Flux Decomposition (Re=800)\n'
                 'Same vs cross helical contributions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Protection fraction f_same(k) time evolution ----
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    snap_times = sorted(time_snapshots.keys())
    kmi_t = 32 // 3  # N//3 for time evolution (N=32)
    for i, t_snap in enumerate(snap_times):
        s = time_snapshots[t_snap]
        if kmi_t <= len(s['k']):
            c = cmap(i / max(len(snap_times) - 1, 1))
            ax.plot(s['k'][:kmi_t], s['f_same'][:kmi_t], 'o-', color=c,
                    linewidth=1.5, markersize=3, label=f't={t_snap:.1f}')
    ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('f_same(k)')
    ax.set_title('Protection Fraction Time Evolution (Re=800)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # ---- Panel 4: Global S/D vs Re ----
    ax = axes[1, 1]
    Re_list = []
    SD_list = []
    for Re, r in re_results.items():
        Re_list.append(Re)
        SD_list.append(r['global_SD'])
    ax.plot(Re_list, SD_list, 'ko-', linewidth=2, markersize=8, label='Global S/D')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Critical: S/D = 1')
    ax.set_xlabel('Reynolds number Re')
    ax.set_ylabel('Global S/D')
    ax.set_title('Global Enstrophy Budget vs Re\n'
                 'S/D < 1 means dissipation wins globally')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # ---- Panel 5: Net budget B(k) = D(k) - T(k) at multiple Re ----
    ax = axes[2, 0]
    for i, (Re, r) in enumerate(re_results.items()):
        active = r['Z_k'] > 1e-10 * np.max(r['Z_k'])
        k_a = r['k'][active]
        ax.plot(k_a, r['B'][active], 'o-', color=colors[i % len(colors)],
                linewidth=2, markersize=4, label=f'Re={Re}')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('B(k) = D(k) - T(k)')
    ax.set_title('Net Budget per Shell\n'
                 'B > 0 = dissipation sink, B < 0 = stretching source')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 6: Global S/D time evolution + protection fraction ----
    ax = axes[2, 1]
    t_list = []
    sd_list = []
    same_frac_list = []
    cross_frac_list = []
    for t_snap in snap_times:
        s = time_snapshots[t_snap]
        active = s['Z_k'] > 1e-10 * np.max(s['Z_k'])
        if np.any(active):
            t_list.append(t_snap)
            sd_list.append(s['global_SD'])
            T_tot = np.sum(np.abs(s['T_Z'][active]))
            if T_tot > 1e-30:
                same_frac_list.append(np.sum(np.abs(s['T_Z_same'][active])) / T_tot)
                cross_frac_list.append(np.sum(np.abs(s['T_Z_cross'][active])) / T_tot)
            else:
                same_frac_list.append(0.5)
                cross_frac_list.append(0.5)

    ax2 = ax.twinx()
    l1, = ax.plot(t_list, sd_list, 'ko-', linewidth=2, markersize=6, label='Global S/D')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
    l2, = ax2.plot(t_list, same_frac_list, 's--', color='#364FC7', linewidth=2,
                   markersize=5, label='f_same (ordered)')
    l3, = ax2.plot(t_list, cross_frac_list, 'D--', color='#e67700', linewidth=2,
                   markersize=5, label='f_cross (disordered)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Global S/D', color='k')
    ax2.set_ylabel('Protection fraction', color='#364FC7')
    ax.set_title('Budget + Protection Evolution (Re=800)')
    ax.legend(handles=[l1, l2, l3], fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = 'h:/tmp/threat_ratio_handoff.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
    return outpath


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 70)
    print("THREAT RATIO R(k) -- PROTECTION TRANSFORMATION MEASUREMENT")
    print("=" * 70)
    print()
    print("The Wanderer's hypothesis: protection doesn't die, it TRANSFORMS.")
    print("  Low k: order (Beltrami, helicity) -> R_same dominates")
    print("  High k: disorder (CLT, depolarization) -> R_cross suppressed")
    print("  Key tests:")
    print("    1. Does f_same(k) decrease with k? (order->disorder handoff)")
    print("    2. Is the handoff robust across Re and ICs?")
    print("    3. Where does dissipation dominate stretching (net budget B(k)>0)?")
    print()

    wall_start = clock.time()

    # Run all three experiments
    re_results = experiment_re_scaling()
    time_snapshots = experiment_time_evolution()
    ic_results = experiment_multiple_ics()

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print("\n\n" + "=" * 70)
    print("FINAL VERDICT: PROTECTION TRANSFORMATION")
    print("=" * 70)

    # Context: Global S > D during cascade development is NORMAL — enstrophy
    # grows until the dissipation range is populated. This does NOT indicate blowup.
    # The Wanderer's insight is about the COMPOSITION of stretching, not total budget.

    # Test 1: Handoff structure (THE MAIN RESULT)
    print("\n  TEST 1 -- PROTECTION HANDOFF (f_same vs k):")
    r = re_results.get(800, list(re_results.values())[0])
    kmi = r['kmax_inertial']
    # Only look at inertial range (k=1..kmax_nonlin) where stretching is nonzero
    fs = r['f_same'][:kmi]
    fc = r['f_cross'][:kmi]
    k_inertial = r['k'][:kmi]

    # Does f_same decrease?
    if len(fs) >= 3:
        decreasing = np.sum(np.diff(fs) < 0) / max(len(fs) - 1, 1)
        print(f"    f_same decreasing fraction: {decreasing:.1%} of inertial shells")
        print(f"    f_same(k=1) = {fs[0]:.4f}, f_same(k={kmi}) = {fs[-1]:.4f}")

    # Find crossover
    crossover_found = False
    for ik in range(len(fs) - 1):
        if fs[ik] > fc[ik] and fs[ik+1] < fc[ik+1]:
            print(f"    Crossover at k ~ {k_inertial[ik]:.0f}: "
                  f"ordered above, disordered below")
            crossover_found = True
            break
    if not crossover_found:
        if np.all(fs > fc):
            print("    Same-helical dominates entire inertial range (NO crossover)")
        else:
            print("    Cross-helical dominates entire inertial range (NO crossover)")

    if fs[0] > fs[-1]:
        print("  -> HANDOFF DETECTED: f_same decreases with k.")
        print("     Ordered protection yields to disordered protection.")
    else:
        print("  -> NO HANDOFF: f_same does not decrease with k.")

    # Test 2: Robustness across Re
    print("\n  TEST 2 -- RE-INDEPENDENCE OF HANDOFF:")
    for Re, r in re_results.items():
        kmi_r = r['kmax_inertial']
        fs_r = r['f_same'][:kmi_r]
        if len(fs_r) >= 2:
            print(f"    Re={Re}: f_same(k=1)={fs_r[0]:.4f} -> "
                  f"f_same(k={kmi_r})={fs_r[-1]:.4f} "
                  f"(drop = {fs_r[0]-fs_r[-1]:.4f})")

    # Test 3: Where does dissipation win? (Net budget per shell)
    print("\n  TEST 3 -- NET BUDGET B(k) = D(k) - T(k):")
    r = re_results.get(800, list(re_results.values())[0])
    active = r['Z_k'] > 1e-10 * np.max(r['Z_k'])
    B_active = r['B'][active]
    k_active = r['k'][active]
    diss_wins = B_active > 0
    if np.any(diss_wins):
        k_diss = k_active[diss_wins]
        print(f"    Dissipation dominates at shells: k = {k_diss}")
        print(f"    Stretching dominates at shells: k = {k_active[~diss_wins]}")
    else:
        print(f"    Stretching dominates at ALL resolved shells")
        print(f"    (Expected: dissipation acts at k_d ~ Re^(3/4) >> k_max)")

    # Note on global budget
    print("\n  NOTE ON GLOBAL S/D:")
    print("    S > D globally during cascade development (enstrophy GROWING)")
    print("    This is normal physics, not a blowup indicator.")
    print("    At N=32, k_max=16 but k_d(Re=400) ~ 89 -- massively under-resolved.")
    print("    The protection transformation (f_same handoff) is the structural result.")

    wall_time = clock.time() - wall_start
    print(f"\nTotal wall time: {wall_time:.1f}s")

    # Generate plots
    print("\nGenerating plots...")
    plot_all(re_results, time_snapshots, ic_results)

    return re_results, time_snapshots, ic_results


if __name__ == "__main__":
    main()
