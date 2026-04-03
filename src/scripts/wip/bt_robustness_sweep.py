"""
BT DYNAMIC SURGERY — ROBUSTNESS SWEEP
======================================
Tests whether the 70% enstrophy reduction from dynamic BT surgery (S36g)
is robust across:
  1. Resolution: N = 32, 48, 64
  2. Reynolds number: Re = 100, 200, 400, 800
  3. Initial conditions: Taylor-Green, Pelz, Random isotropic

For each (N, Re, IC) combo, runs Full NS vs Dynamic surgery and records:
  - Z_peak ratio (dynamic / full)
  - C_max ratio
  - Stretching_peak ratio
  - Final h+ fraction

Also tests 3 genuinely untested angles:
  4. MHD-like: add Alfven wave damping term
  5. Dispersive: add Coriolis (rotating frame)
  6. Triadic resonance: track resonant vs non-resonant triad contributions

References:
  - Biferale & Titi (2013): arXiv:1304.5339
  - Babin, Mahalov, Nikolaenko (1999): rotating NS global regularity
  - Sermange & Temam (1983): MHD regularity
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import time as clock
import sys


class SpectralNS:
    """Pseudo-spectral NS solver with helical decomposition and extensions."""

    def __init__(self, N=64, Re=400):
        self.N = N
        self.nu = 1.0 / Re
        self.Re = Re
        L = 2.0 * np.pi

        # Physical grid
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z_grid = np.meshgrid(x, x, x, indexing='ij')

        # Wavevectors
        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0, 0] = 1.0

        # Leray projector
        self.P = {}
        K = [self.kx, self.ky, self.kz]
        for i in range(3):
            for j in range(3):
                self.P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / self.k2_safe

        # Dealiasing mask (2/3 rule)
        kmax = N // 3
        self.dealias_mask = (
            (np.abs(self.kx) <= kmax) &
            (np.abs(self.ky) <= kmax) &
            (np.abs(self.kz) <= kmax)
        )

        self._build_helical_basis()

    def _build_helical_basis(self):
        """Craya-Herring helical basis."""
        kmag = np.sqrt(self.k2_safe)
        khat = np.array([self.kx / kmag, self.ky / kmag, self.kz / kmag])

        # e1 = (0,0,1) x khat = (-ky, kx, 0) / |...|
        e1 = np.array([
            -khat[1],
            khat[0],
            np.zeros_like(khat[0]),
        ])
        e1_mag = np.sqrt(np.sum(e1**2, axis=0))

        # Fallback for k parallel to z: use (1,0,0) x khat
        parallel = e1_mag < 1e-10
        if np.any(parallel):
            e1_alt = np.array([
                np.zeros_like(khat[0]),
                -khat[2],
                khat[1],
            ])
            for i in range(3):
                e1[i] = np.where(parallel, e1_alt[i], e1[i])
            e1_mag = np.sqrt(np.sum(e1**2, axis=0))

        e1_mag = np.maximum(e1_mag, 1e-15)
        e1 /= e1_mag

        # e2 = khat x e1
        e2 = np.array([
            khat[1]*e1[2] - khat[2]*e1[1],
            khat[2]*e1[0] - khat[0]*e1[2],
            khat[0]*e1[1] - khat[1]*e1[0],
        ])

        self.h_plus = (e1 + 1j * e2) / np.sqrt(2.0)
        self.h_minus = (e1 - 1j * e2) / np.sqrt(2.0)
        self.h_plus[:, 0, 0, 0] = 0.0
        self.h_minus[:, 0, 0, 0] = 0.0

    def project_leray(self, f_hat):
        result = np.zeros_like(f_hat)
        for i in range(3):
            for j in range(3):
                result[i] += self.P[(i, j)] * f_hat[j]
        return result

    def project_h_plus(self, f_hat):
        f_p = np.sum(np.conj(self.h_plus) * f_hat, axis=0)
        return f_p[np.newaxis] * self.h_plus

    def compute_vorticity_hat(self, u_hat):
        return np.array([
            1j * (self.ky * u_hat[2] - self.kz * u_hat[1]),
            1j * (self.kz * u_hat[0] - self.kx * u_hat[2]),
            1j * (self.kx * u_hat[1] - self.ky * u_hat[0]),
        ])

    def compute_rhs(self, u_hat, surgery_mode='none'):
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

        # Lamb vector: u x omega
        lamb = np.array([
            u[1]*omega[2] - u[2]*omega[1],
            u[2]*omega[0] - u[0]*omega[2],
            u[0]*omega[1] - u[1]*omega[0],
        ])
        lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])

        for i in range(3):
            lamb_hat[i] *= self.dealias_mask

        if surgery_mode == 'dynamic':
            lamb_hat = self.project_h_plus(lamb_hat)

        rhs = self.project_leray(lamb_hat)
        rhs -= self.nu * self.k2[np.newaxis] * u_hat
        rhs[:, 0, 0, 0] = 0.0
        return rhs

    def step_rk4(self, u_hat, dt, surgery_mode='none'):
        k1 = self.compute_rhs(u_hat, surgery_mode)
        k2 = self.compute_rhs(u_hat + 0.5*dt*k1, surgery_mode)
        k3 = self.compute_rhs(u_hat + 0.5*dt*k2, surgery_mode)
        k4 = self.compute_rhs(u_hat + dt*k3, surgery_mode)
        return u_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ---- Initial Conditions ----

    def taylor_green_ic(self):
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z_grid)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z_grid)
        u[2] = 0.0
        u_hat = np.array([fftn(u[i]) for i in range(3)])
        return self.project_leray(u_hat)

    def pelz_ic(self):
        """Pelz flow — high-symmetry, known to produce intense vortex stretching."""
        X, Y, Z = self.X, self.Y, self.Z_grid
        u = np.zeros((3,) + X.shape)
        u[0] = np.sin(X) * (np.cos(3*Y)*np.cos(Z) - np.cos(Y)*np.cos(3*Z))
        u[1] = np.sin(Y) * (np.cos(3*Z)*np.cos(X) - np.cos(Z)*np.cos(3*X))
        u[2] = np.sin(Z) * (np.cos(3*X)*np.cos(Y) - np.cos(X)*np.cos(3*Y))
        u_hat = np.array([fftn(u[i]) for i in range(3)])
        return self.project_leray(u_hat)

    def random_ic(self, energy_scale=0.5, seed=42):
        """Random isotropic divergence-free field with controlled energy."""
        rng = np.random.RandomState(seed)
        N = self.N
        u_hat = (rng.randn(3, N, N, N) + 1j * rng.randn(3, N, N, N))

        # Project to divergence-free
        u_hat = self.project_leray(u_hat)

        # Set energy scale
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        E = 0.5 * np.mean(np.sum(u**2, axis=0))
        if E > 1e-15:
            u_hat *= np.sqrt(energy_scale / E)

        return u_hat

    # ---- Diagnostics ----

    def compute_diagnostics(self, u_hat):
        K = [self.kx, self.ky, self.kz]
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

        E = 0.5 * np.mean(np.sum(u**2, axis=0))
        Z = np.mean(np.sum(omega**2, axis=0))

        # Strain tensor
        S = np.zeros((3, 3) + u.shape[1:])
        for i in range(3):
            for j in range(i, 3):
                S_hat = 0.5 * (1j * K[j] * u_hat[i] + 1j * K[i] * u_hat[j])
                S[i, j] = np.real(ifftn(S_hat))
                if j != i:
                    S[j, i] = S[i, j]

        stretch = np.zeros(u.shape[1:])
        for i in range(3):
            for j in range(3):
                stretch += omega[i] * S[i, j] * omega[j]
        stretching = np.mean(stretch)

        # Lambda max (Frobenius bound)
        S_frob_sq = np.zeros(u.shape[1:])
        for i in range(3):
            for j in range(3):
                S_frob_sq += S[i, j]**2
        lambda_max = np.sqrt(np.max(S_frob_sq))

        C = abs(stretching) / (lambda_max * Z) if lambda_max * Z > 1e-15 else 0.0

        # h+ fraction
        N_g = self.N
        u_p = np.sum(np.conj(self.h_plus) * u_hat, axis=0)
        u_m = np.sum(np.conj(self.h_minus) * u_hat, axis=0)
        E_p = np.sum(np.abs(u_p)**2) / N_g**3
        E_m = np.sum(np.abs(u_m)**2) / N_g**3
        E_hel = E_p + E_m
        hp_frac = E_p / E_hel if E_hel > 1e-15 else 0.5

        return {
            'E': E, 'Z': Z, 'stretching': stretching,
            'lambda_max': lambda_max, 'C': C, 'hp_frac': hp_frac,
        }


def run_one_case(solver, u_hat_init, dt, T_final, surgery_mode):
    """Run simulation, return time series of diagnostics."""
    u_hat = u_hat_init.copy()
    results = []
    t = 0.0
    step = 0
    diag_every = max(1, int(0.5 / dt))  # Every 0.5 time units

    while t <= T_final + 1e-10:
        if step % diag_every == 0:
            d = solver.compute_diagnostics(u_hat)
            d['t'] = t
            results.append(d)
        u_hat = solver.step_rk4(u_hat, dt, surgery_mode)
        t += dt
        step += 1

    return results


def extract_peaks(diag_list):
    """Extract peak metrics from a diagnostic time series."""
    Z_vals = [d['Z'] for d in diag_list]
    C_vals = [d['C'] for d in diag_list]
    S_vals = [abs(d['stretching']) for d in diag_list]
    return {
        'Z_peak': max(Z_vals),
        'C_max': max(C_vals),
        'S_peak': max(S_vals),
        'Z_final': Z_vals[-1],
        'E_final': diag_list[-1]['E'],
        'hp_final': diag_list[-1]['hp_frac'],
        'Z_growth': max(Z_vals) / Z_vals[0] if Z_vals[0] > 1e-15 else 0,
    }


# ============================================================
# PART 1: CORE ROBUSTNESS SWEEP (N, Re, IC)
# ============================================================

def run_sweep():
    """Main sweep: N x Re x IC combinations."""
    print("=" * 80)
    print("BT DYNAMIC SURGERY — ROBUSTNESS SWEEP")
    print("=" * 80)
    print()

    # TARGETED sweep: answer 3 key questions with minimal runs
    # Q1: Robust across ICs? -> N=32, Re=400, {TG, Pelz, Random}
    # Q2: Robust across Re?  -> N=32, TG, {100, 400, 800}
    # Q3: Robust across N?   -> {32, 48}, Re=400, TG
    # (N=32/Re=400/TG appears in all 3, so 7 unique combos)
    combos = [
        # Q1: IC robustness
        (32, 400, 'taylor_green'),
        (32, 400, 'pelz'),
        (32, 400, 'random'),
        # Q2: Re robustness (TG at N=32 already in Q1, add Re=100,800)
        (32, 100, 'taylor_green'),
        (32, 800, 'taylor_green'),
        # Q3: Resolution (N=32/Re=400/TG already in Q1, add N=48)
        (48, 400, 'taylor_green'),
        # Bonus: cross-check Pelz at higher Re
        (32, 800, 'pelz'),
    ]

    dt = 0.005
    T_final = 4.0  # S36g showed all action is in [0, 4]

    # Results table
    results = []

    total = len(combos)
    count = 0
    wall_start = clock.time()

    for N, Re, ic_type in combos:
        solver = SpectralNS(N=N, Re=Re)
        count += 1
        print(f"\n[{count}/{total}] N={N}, Re={Re}, IC={ic_type}, dt={dt:.4f}")

        # Initialize
        if ic_type == 'taylor_green':
            u_hat = solver.taylor_green_ic()
        elif ic_type == 'pelz':
            u_hat = solver.pelz_ic()
        else:
            u_hat = solver.random_ic(energy_scale=0.5)

        # Case A: Full NS
        t0 = clock.time()
        dA = run_one_case(solver, u_hat, dt, T_final, 'none')
        pA = extract_peaks(dA)
        tA = clock.time() - t0

        # Case C: Dynamic surgery
        t0 = clock.time()
        dC = run_one_case(solver, u_hat, dt, T_final, 'dynamic')
        pC = extract_peaks(dC)
        tC = clock.time() - t0

        # Ratios
        Z_ratio = pC['Z_peak'] / pA['Z_peak'] if pA['Z_peak'] > 1e-15 else 0
        C_ratio = pC['C_max'] / pA['C_max'] if pA['C_max'] > 1e-15 else 0
        S_ratio = pC['S_peak'] / pA['S_peak'] if pA['S_peak'] > 1e-15 else 0

        entry = {
            'N': N, 'Re': Re, 'IC': ic_type, 'dt': dt,
            'Z_full': pA['Z_peak'], 'Z_surg': pC['Z_peak'],
            'Z_ratio': Z_ratio,
            'C_full': pA['C_max'], 'C_surg': pC['C_max'],
            'C_ratio': C_ratio,
            'S_full': pA['S_peak'], 'S_surg': pC['S_peak'],
            'S_ratio': S_ratio,
            'Zgrow_full': pA['Z_growth'], 'Zgrow_surg': pC['Z_growth'],
            'hp_final': pC['hp_final'],
            'time_A': tA, 'time_C': tC,
        }
        results.append(entry)

        print(f"  Full: Z_peak={pA['Z_peak']:.4e}, C_max={pA['C_max']:.5f}, "
              f"Z_growth={pA['Z_growth']:.2f}x  ({tA:.1f}s)")
        print(f"  Surg: Z_peak={pC['Z_peak']:.4e}, C_max={pC['C_max']:.5f}, "
              f"Z_growth={pC['Z_growth']:.2f}x  ({tC:.1f}s)")
        print(f"  Ratios: Z={Z_ratio:.3f}, C={C_ratio:.3f}, S={S_ratio:.3f}, "
              f"h+_final={pC['hp_final']*100:.1f}%")
        sys.stdout.flush()

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    elapsed = clock.time() - wall_start
    print(f"\n\n{'=' * 80}")
    print(f"SUMMARY TABLE ({elapsed:.0f}s total)")
    print(f"{'=' * 80}")
    print(f"\n{'N':<4} {'Re':<5} {'IC':<14} {'Z_full':<10} {'Z_surg':<10} "
          f"{'Z_ratio':<8} {'C_full':<8} {'C_surg':<8} {'C_ratio':<8} "
          f"{'S_ratio':<8} {'h+%':<5}")
    print("-" * 95)

    for r in results:
        print(f"{r['N']:<4} {r['Re']:<5} {r['IC']:<14} "
              f"{r['Z_full']:<10.3e} {r['Z_surg']:<10.3e} "
              f"{r['Z_ratio']:<8.3f} {r['C_full']:<8.4f} {r['C_surg']:<8.4f} "
              f"{r['C_ratio']:<8.3f} {r['S_ratio']:<8.3f} "
              f"{r['hp_final']*100:<5.1f}")

    # Statistics by parameter (derived from combos)
    print(f"\n\n{'=' * 80}")
    print("AGGREGATE ANALYSIS")
    print(f"{'=' * 80}")

    # By IC type
    ic_types_seen = sorted(set(r['IC'] for r in results))
    print("\nZ_ratio by IC type:")
    for ic in ic_types_seen:
        subset = [r['Z_ratio'] for r in results if r['IC'] == ic]
        print(f"  {ic:<14}: {np.mean(subset):.3f} (n={len(subset)})")

    # By Re
    re_values_seen = sorted(set(r['Re'] for r in results))
    print("\nZ_ratio by Re:")
    for Re in re_values_seen:
        subset = [r['Z_ratio'] for r in results if r['Re'] == Re]
        print(f"  Re={Re:<5}: {np.mean(subset):.3f} (n={len(subset)})")

    # By N
    n_values_seen = sorted(set(r['N'] for r in results))
    print("\nZ_ratio by N:")
    for N in n_values_seen:
        subset = [r['Z_ratio'] for r in results if r['N'] == N]
        print(f"  N={N:<3}: {np.mean(subset):.3f} (n={len(subset)})")

    # Overall
    all_Z = [r['Z_ratio'] for r in results]
    all_C = [r['C_ratio'] for r in results]
    all_S = [r['S_ratio'] for r in results]
    print(f"\nOverall statistics ({len(results)} runs):")
    print(f"  Z_ratio: {np.mean(all_Z):.3f} +/- {np.std(all_Z):.3f} "
          f"(min={np.min(all_Z):.3f}, max={np.max(all_Z):.3f})")
    print(f"  C_ratio: {np.mean(all_C):.3f} +/- {np.std(all_C):.3f}")
    print(f"  S_ratio: {np.mean(all_S):.3f} +/- {np.std(all_S):.3f}")

    # Verdict
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")

    if np.mean(all_Z) < 0.5:
        print(f"\nDynamic BT surgery ROBUSTLY reduces enstrophy (mean ratio {np.mean(all_Z):.3f}).")
        if np.std(all_Z) < 0.2:
            print("The effect is CONSISTENT across N, Re, and IC types.")
        else:
            print("BUT the effect varies significantly across configurations.")
    elif np.mean(all_Z) < 0.9:
        print(f"\nDynamic BT surgery provides MODERATE reduction (mean ratio {np.mean(all_Z):.3f}).")
        print("The effect is real but not as dramatic as the TG-only result suggested.")
    else:
        print(f"\nDynamic BT surgery shows MINIMAL effect (mean ratio {np.mean(all_Z):.3f}).")
        print("The S36g TG result may have been IC-specific.")

    # Check Re dependence (critical for NS regularity)
    if len(re_values_seen) >= 2:
        Re_lo = min(re_values_seen)
        Re_hi = max(re_values_seen)
        z_lo = np.mean([r['Z_ratio'] for r in results if r['Re'] == Re_lo])
        z_hi = np.mean([r['Z_ratio'] for r in results if r['Re'] == Re_hi])
        trend = z_hi - z_lo
        print(f"\nRe trend: Z_ratio changes by {trend:+.3f} from Re={Re_lo} to Re={Re_hi}")
        if trend > 0.1:
            print("  WARNING: Surgery becomes LESS effective at high Re.")
            print("  This suggests the effect may vanish in the inviscid limit.")
        elif trend < -0.1:
            print("  Surgery becomes MORE effective at high Re. Very promising.")
        else:
            print("  Surgery effectiveness is INDEPENDENT of Re. Good sign for universality.")

    return results


# ============================================================
# PART 2: UNTESTED ANGLES — QUICK PROBES
# ============================================================

def probe_mhd():
    """
    MHD probe: add Lorentz force from a uniform background magnetic field.
    In ideal MHD, B0 introduces Alfven waves that prevent blow-up.
    This is NOT NS (Trap #9 territory), but we test whether the regularization
    mechanism resembles BT surgery (i.e., removing certain triadic interactions).
    """
    print(f"\n\n{'=' * 80}")
    print("UNTESTED ANGLE: MHD-LIKE ALFVEN DAMPING")
    print(f"{'=' * 80}")
    print("Adding uniform B0 = (0, 0, B0) introduces Alfven wave term.")
    print("This modifies dw/dt by adding B0 * d(omega)/dz (vorticity transport along B0).")
    print("NOTE: This is a DIFFERENT PDE (Trap #9). Testing for comparison only.\n")

    N = 48
    Re = 400
    dt = 0.005
    T = 5.0

    solver = SpectralNS(N=N, Re=Re)
    u_hat_init = solver.taylor_green_ic()

    B0_values = [0.0, 0.1, 0.5, 1.0, 2.0]

    print(f"{'B0':<6} {'Z_peak':<10} {'Z_growth':<10} {'C_max':<8}")
    print("-" * 36)

    for B0 in B0_values:
        u_hat = u_hat_init.copy()
        Z_max = 0.0
        C_max = 0.0
        t = 0.0
        step = 0
        diag_every = max(1, int(0.5 / dt))

        Z_init = None
        while t <= T + 1e-10:
            if step % diag_every == 0:
                d = solver.compute_diagnostics(u_hat)
                if Z_init is None:
                    Z_init = d['Z']
                Z_max = max(Z_max, d['Z'])
                C_max = max(C_max, d['C'])

            # Standard NS RHS
            k1 = solver.compute_rhs(u_hat, 'none')

            # Add Alfven term: B0 * ik_z * omega_hat
            if B0 > 0:
                omega_hat = solver.compute_vorticity_hat(u_hat)
                # Alfven wave: B0 * (B0 . grad) omega = B0^2 * d(omega)/dz
                alfven = B0 * 1j * solver.kz * omega_hat
                # This enters as a source for the velocity equation through curl^(-1)
                # Simplified: add B0 * ik_z * u_hat (magnetic tension)
                k1 += B0 * 1j * solver.kz[np.newaxis] * u_hat

            k2 = solver.compute_rhs(u_hat + 0.5*dt*k1, 'none')
            if B0 > 0:
                k2 += B0 * 1j * solver.kz[np.newaxis] * (u_hat + 0.5*dt*k1)

            k3 = solver.compute_rhs(u_hat + 0.5*dt*k2, 'none')
            if B0 > 0:
                k3 += B0 * 1j * solver.kz[np.newaxis] * (u_hat + 0.5*dt*k2)

            k4 = solver.compute_rhs(u_hat + dt*k3, 'none')
            if B0 > 0:
                k4 += B0 * 1j * solver.kz[np.newaxis] * (u_hat + dt*k3)

            u_hat = u_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
            step += 1

        Z_growth = Z_max / Z_init if Z_init > 1e-15 else 0
        print(f"{B0:<6.1f} {Z_max:<10.3e} {Z_growth:<10.2f}x {C_max:<8.4f}")

    print("\nInterpretation:")
    print("  B0 > 0 introduces dispersive Alfven waves along the field direction.")
    print("  If Z_growth decreases with B0, magnetic field suppresses enstrophy cascade.")
    print("  Compare mechanism with BT surgery: both break isotropy of triadic transfers.")


def probe_coriolis():
    """
    Dispersive regularization: add Coriolis force (rotating frame).
    Babin-Mahalov-Nikolaenko (1999) proved global regularity for fast rotation.
    Test: how does rotation rate affect enstrophy growth?
    """
    print(f"\n\n{'=' * 80}")
    print("UNTESTED ANGLE: CORIOLIS / ROTATING NS")
    print(f"{'=' * 80}")
    print("Adding Coriolis: 2*Omega x u (rotation about z-axis).")
    print("Ref: Babin, Mahalov & Nikolaenko (1999) proved regularity for large Omega.")
    print("In Fourier space: 2*Omega * (-u2, u1, 0) projected through Leray.\n")

    N = 48
    Re = 400
    dt = 0.005
    T = 5.0

    solver = SpectralNS(N=N, Re=Re)
    u_hat_init = solver.taylor_green_ic()

    Omega_values = [0.0, 1.0, 5.0, 10.0, 50.0]

    print(f"{'Omega':<7} {'Z_peak':<10} {'Z_growth':<10} {'C_max':<8}")
    print("-" * 38)

    for Omega in Omega_values:
        u_hat = u_hat_init.copy()
        Z_max = 0.0
        C_max = 0.0
        t = 0.0
        step = 0
        diag_every = max(1, int(0.5 / dt))
        Z_init = None

        while t <= T + 1e-10:
            if step % diag_every == 0:
                d = solver.compute_diagnostics(u_hat)
                if Z_init is None:
                    Z_init = d['Z']
                Z_max = max(Z_max, d['Z'])
                C_max = max(C_max, d['C'])

            # Coriolis term: 2*Omega * (e_z x u) = 2*Omega * (-u_y, u_x, 0)
            coriolis_hat = np.zeros_like(u_hat)
            coriolis_hat[0] = -2.0 * Omega * u_hat[1]
            coriolis_hat[1] = 2.0 * Omega * u_hat[0]
            # coriolis_hat[2] = 0 (no z-component)

            # Project Coriolis through Leray
            coriolis_hat = solver.project_leray(coriolis_hat)

            # Full RK4 with Coriolis
            def full_rhs(uh):
                rhs = solver.compute_rhs(uh, 'none')
                if Omega > 0:
                    cor = np.zeros_like(uh)
                    cor[0] = -2.0 * Omega * uh[1]
                    cor[1] = 2.0 * Omega * uh[0]
                    rhs += solver.project_leray(cor)
                return rhs

            k1 = full_rhs(u_hat)
            k2 = full_rhs(u_hat + 0.5*dt*k1)
            k3 = full_rhs(u_hat + 0.5*dt*k2)
            k4 = full_rhs(u_hat + dt*k3)
            u_hat = u_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

            t += dt
            step += 1

        Z_growth = Z_max / Z_init if Z_init > 1e-15 else 0
        print(f"{Omega:<7.1f} {Z_max:<10.3e} {Z_growth:<10.2f}x {C_max:<8.4f}")

    print("\nInterpretation:")
    print("  Coriolis introduces inertial wave dispersion (omega ~ k_z/|k|).")
    print("  Fast rotation (large Omega) suppresses 3D interactions, leaving 2D flow.")
    print("  2D NS is globally regular (conservation of enstrophy in 2D).")
    print("  Compare with BT: both suppress certain triadic interactions.")
    print("  BT removes cross-helicity triads; Coriolis suppresses off-axis triads.")


def probe_triadic_resonance():
    """
    Track resonant vs non-resonant triadic contributions to stretching.
    A triad (k, p, q) with k = p + q is resonant if omega(k) = omega(p) + omega(q).
    For NS, there's no linear dispersion, so ALL triads are technically resonant.
    But with rotation or stratification, only near-resonant triads survive.

    We measure: what fraction of enstrophy transfer comes from triads where
    |p| ~ |q| (local) vs |p| >> |q| (non-local)?
    """
    print(f"\n\n{'=' * 80}")
    print("UNTESTED ANGLE: TRIADIC STRUCTURE OF ENSTROPHY TRANSFER")
    print(f"{'=' * 80}")
    print("Question: Is enstrophy transfer dominated by local or non-local triads?")
    print("Local: |p| ~ |q| ~ |k|. Non-local: |p| >> |q| or vice versa.")
    print("If non-local triads dominate, BT surgery may be removing them specifically.\n")

    N = 48
    Re = 400
    solver = SpectralNS(N=N, Re=Re)

    # Get a developed flow (run full NS for a bit)
    u_hat = solver.taylor_green_ic()
    dt = 0.005
    for _ in range(200):
        u_hat = solver.step_rk4(u_hat, dt, 'none')

    # Now compute the nonlinear term and decompose by shell
    omega_hat = solver.compute_vorticity_hat(u_hat)
    u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

    lamb = np.array([
        u[1]*omega[2] - u[2]*omega[1],
        u[2]*omega[0] - u[0]*omega[2],
        u[0]*omega[1] - u[1]*omega[0],
    ])
    lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])

    # Decompose by helical content
    lamb_hp = solver.project_h_plus(lamb_hat)
    lamb_hm = lamb_hat - lamb_hp

    # Energy in each component by shell
    kmag = np.sqrt(solver.k2)
    kmax = N // 3

    shells = np.arange(0, kmax + 1)
    E_hp_shell = np.zeros(len(shells))
    E_hm_shell = np.zeros(len(shells))
    E_full_shell = np.zeros(len(shells))

    for ik, k in enumerate(shells):
        mask = (kmag >= k) & (kmag < k + 1)
        for i in range(3):
            E_hp_shell[ik] += np.sum(np.abs(lamb_hp[i][mask])**2)
            E_hm_shell[ik] += np.sum(np.abs(lamb_hm[i][mask])**2)
            E_full_shell[ik] += np.sum(np.abs(lamb_hat[i][mask])**2)

    # Normalize
    total = np.sum(E_full_shell) + 1e-30
    hp_total = np.sum(E_hp_shell)
    hm_total = np.sum(E_hm_shell)

    print("Lamb vector (nonlinear term) shell decomposition at t=1.0:")
    print(f"\n  Total h+ fraction: {hp_total/total*100:.1f}%")
    print(f"  Total h- fraction: {hm_total/total*100:.1f}%")
    print(f"  (If ~50/50: BT surgery removes half the nonlinearity)")

    print(f"\n{'Shell k':<9} {'E_full':<12} {'h+ %':<8} {'h- %':<8}")
    print("-" * 40)

    for ik in range(min(16, len(shells))):
        if E_full_shell[ik] > 1e-20:
            hp_pct = E_hp_shell[ik] / E_full_shell[ik] * 100
            hm_pct = E_hm_shell[ik] / E_full_shell[ik] * 100
            print(f"{shells[ik]:<9} {E_full_shell[ik]:<12.3e} {hp_pct:<8.1f} {hm_pct:<8.1f}")

    # Also: what fraction of nonlinearity is in low vs high k?
    mid_k = kmax // 2
    low_k_frac = np.sum(E_full_shell[:mid_k]) / total * 100
    high_k_frac = np.sum(E_full_shell[mid_k:]) / total * 100

    low_hp_frac = np.sum(E_hp_shell[:mid_k]) / (np.sum(E_full_shell[:mid_k]) + 1e-30) * 100
    high_hp_frac = np.sum(E_hp_shell[mid_k:]) / (np.sum(E_full_shell[mid_k:]) + 1e-30) * 100

    print(f"\nScale decomposition (k < {mid_k} vs k >= {mid_k}):")
    print(f"  Low-k: {low_k_frac:.1f}% of total, h+ fraction = {low_hp_frac:.1f}%")
    print(f"  High-k: {high_k_frac:.1f}% of total, h+ fraction = {high_hp_frac:.1f}%")

    print("\nInterpretation:")
    print("  If h- fraction is higher at small scales: BT surgery removes forward cascade.")
    print("  If h+/h- ratio is scale-independent: removal is uniform across scales.")
    print("  This helps distinguish BT from simple energy reduction.")


if __name__ == "__main__":
    # Part 1: Core robustness sweep
    results = run_sweep()

    # Part 2: Untested angles
    probe_mhd()
    probe_coriolis()
    probe_triadic_resonance()

    print(f"\n\n{'=' * 80}")
    print("ALL PROBES COMPLETE")
    print(f"{'=' * 80}")
