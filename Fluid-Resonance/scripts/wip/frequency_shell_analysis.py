"""
FREQUENCY-SHELL ENSTROPHY ANALYSIS & WILSON LOOP CIRCULATION TEST
==================================================================
S36j — Two highest-priority experiments from the 22-word path analysis.

Experiment 1: LITTLEWOOD-PALEY SHELL DECOMPOSITION
  Decomposes enstrophy into dyadic frequency shells S_j = {k : 2^(j-1) <= |k| < 2^j}.
  Tracks Z_j(t) per shell, measures triadic transfer rates T_{j,l,m},
  and checks whether the forward cascade is local in frequency (Eyink 2005).

  This is THE central tool in modern NS regularity theory (Chemin-Lerner 1995,
  Koch-Tataru 2001, Cannone-Meyer-Planchon 1994). The BKM criterion says
  blow-up requires Z to concentrate at arbitrarily high j.

Experiment 2: CIRCULATION PDF & WILSON LOOP (Migdal test)
  Computes Gamma_C = oint_C v.dl for square loops of various sizes.
  Tests Migdal's (1993) loop-space reformulation of NS:
    - Kolmogorov prediction: Gamma_rms(L) ~ epsilon^{1/3} * L^{4/3}
    - Area-law test: |Psi[C]| = |<exp(i*Gamma/nu)>| vs L^2

  No published numerical tests of Migdal's framework exist (gap in literature).

Experiment 3: FRACTAL DIMENSION OF HIGH-VORTICITY REGIONS
  Measures box-counting dimension of {|omega| > lambda * omega_rms}.
  Connects to CKN partial regularity (1982): singular set has dim <= 1.

Author: Meridian (Claude Opus 4.6)
Date: 2026-03-12
Checkpoint: 19.0.0 -> 20.0.0
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import time as clock


class SpectralNS:
    """Minimal pseudo-spectral NS solver (from spectral_bt_surgery.py)."""

    def __init__(self, N=64, Re=400):
        self.N = N
        self.nu = 1.0 / Re
        L = 2.0 * np.pi

        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z_grid = np.meshgrid(x, x, x, indexing='ij')
        self.dx = L / N

        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.kmag = np.sqrt(self.k2)
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0, 0] = 1.0

        K = [self.kx, self.ky, self.kz]
        self.P = {}
        for i in range(3):
            for j in range(3):
                self.P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / self.k2_safe

        kmax = N // 3
        self.dealias_mask = (
            (np.abs(self.kx) <= kmax) &
            (np.abs(self.ky) <= kmax) &
            (np.abs(self.kz) <= kmax)
        )

    def project_leray(self, f_hat):
        result = np.zeros_like(f_hat)
        for i in range(3):
            for j in range(3):
                result[i] += self.P[(i, j)] * f_hat[j]
        return result

    def compute_vorticity_hat(self, u_hat):
        return np.array([
            1j * (self.ky * u_hat[2] - self.kz * u_hat[1]),
            1j * (self.kz * u_hat[0] - self.kx * u_hat[2]),
            1j * (self.kx * u_hat[1] - self.ky * u_hat[0]),
        ])

    def compute_rhs(self, u_hat):
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        lamb = np.array([
            u[1]*omega[2] - u[2]*omega[1],
            u[2]*omega[0] - u[0]*omega[2],
            u[0]*omega[1] - u[1]*omega[0],
        ])
        lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])
        for i in range(3):
            lamb_hat[i] *= self.dealias_mask
        rhs = self.project_leray(lamb_hat)
        rhs -= self.nu * self.k2[np.newaxis] * u_hat
        rhs[:, 0, 0, 0] = 0.0
        return rhs

    def step_rk4(self, u_hat, dt):
        k1 = self.compute_rhs(u_hat)
        k2 = self.compute_rhs(u_hat + 0.5*dt*k1)
        k3 = self.compute_rhs(u_hat + 0.5*dt*k2)
        k4 = self.compute_rhs(u_hat + dt*k3)
        return u_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def taylor_green_ic(self):
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z_grid)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z_grid)
        u[2] = 0.0
        u_hat = np.array([fftn(u[i]) for i in range(3)])
        return self.project_leray(u_hat)

    def random_ic(self, seed=42, k_peak=4):
        """Random solenoidal IC with energy peaked at k_peak."""
        rng = np.random.RandomState(seed)
        u_hat = np.zeros((3, self.N, self.N, self.N), dtype=complex)
        for i in range(3):
            phase = rng.uniform(0, 2*np.pi, (self.N, self.N, self.N))
            amp = np.exp(-0.5 * (self.kmag - k_peak)**2 / 2.0**2)
            u_hat[i] = amp * np.exp(1j * phase)
        u_hat = self.project_leray(u_hat)
        # Normalize
        E = 0.5 * np.sum(np.abs(u_hat)**2) / self.N**3
        u_hat *= np.sqrt(0.5 / max(E, 1e-15))
        return u_hat


# ============================================================
# EXPERIMENT 1: FREQUENCY-SHELL ENSTROPHY DECOMPOSITION
# ============================================================

def compute_shell_enstrophy(solver, u_hat, n_shells=None):
    """
    Decompose enstrophy into dyadic frequency shells.
    Shell j: 2^(j-1) <= |k| < 2^j  (shell 0: |k| < 1)

    Returns dict with:
      Z_shells[j] = enstrophy in shell j
      E_shells[j] = energy in shell j
      k_shells[j] = characteristic wavenumber of shell j
    """
    N = solver.N
    if n_shells is None:
        n_shells = int(np.log2(N)) + 1

    omega_hat = solver.compute_vorticity_hat(u_hat)

    Z_shells = np.zeros(n_shells)
    E_shells = np.zeros(n_shells)
    k_shells = np.zeros(n_shells)
    mode_counts = np.zeros(n_shells)

    for j in range(n_shells):
        if j == 0:
            mask = solver.kmag < 1.0
            k_char = 0.5
        else:
            k_lo = 2.0**(j-1)
            k_hi = 2.0**j
            mask = (solver.kmag >= k_lo) & (solver.kmag < k_hi)
            k_char = 0.5 * (k_lo + k_hi)

        k_shells[j] = k_char
        mode_counts[j] = np.sum(mask)

        for i in range(3):
            Z_shells[j] += np.sum(np.abs(omega_hat[i][mask])**2) / N**3
            E_shells[j] += np.sum(np.abs(u_hat[i][mask])**2) / N**3

    return {
        'Z_shells': Z_shells,
        'E_shells': E_shells,
        'k_shells': k_shells,
        'mode_counts': mode_counts,
        'n_shells': n_shells,
    }


def compute_triadic_transfer(solver, u_hat, n_shells=None):
    """
    Measure energy transfer INTO each shell from the nonlinear term.

    T_j = Re[ sum_{k in S_j} conj(u_hat(k)) . F_hat(k) ] / N^3

    where F_hat = Leray(u x omega) is the nonlinear forcing in Fourier space.

    Positive T_j = energy flows INTO shell j (forward cascade).
    Negative T_j = energy flows OUT OF shell j (inverse cascade).
    """
    N = solver.N
    if n_shells is None:
        n_shells = int(np.log2(N)) + 1

    # Compute nonlinear term
    omega_hat = solver.compute_vorticity_hat(u_hat)
    u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
    lamb = np.array([
        u[1]*omega[2] - u[2]*omega[1],
        u[2]*omega[0] - u[0]*omega[2],
        u[0]*omega[1] - u[1]*omega[0],
    ])
    F_hat = np.array([fftn(lamb[i]) for i in range(3)])
    for i in range(3):
        F_hat[i] *= solver.dealias_mask
    F_hat = solver.project_leray(F_hat)

    T_shells = np.zeros(n_shells)
    # Also compute enstrophy transfer
    Z_transfer = np.zeros(n_shells)

    for j in range(n_shells):
        if j == 0:
            mask = solver.kmag < 1.0
        else:
            k_lo = 2.0**(j-1)
            k_hi = 2.0**j
            mask = (solver.kmag >= k_lo) & (solver.kmag < k_hi)

        for i in range(3):
            T_shells[j] += np.real(np.sum(np.conj(u_hat[i][mask]) * F_hat[i][mask])) / N**3
            # Enstrophy transfer: -k^2 * Re[conj(u_hat) . F_hat] (via curl)
            Z_transfer[j] += np.real(np.sum(
                solver.k2[mask] * np.conj(u_hat[i][mask]) * F_hat[i][mask]
            )) / N**3

    return {
        'T_energy': T_shells,
        'T_enstrophy': Z_transfer,
    }


def experiment_1_frequency_shells():
    """
    Littlewood-Paley frequency shell analysis.
    Track Z_j(t) and triadic transfer T_j(t) through NS evolution.
    """
    print("=" * 72)
    print("EXPERIMENT 1: FREQUENCY-SHELL ENSTROPHY DECOMPOSITION")
    print("  Littlewood-Paley analysis for NS regularity")
    print("=" * 72)

    N = 64
    Re = 400
    dt = 0.005
    T_final = 5.0
    diag_every = 40  # every 0.2 time units

    solver = SpectralNS(N=N, Re=Re)
    n_shells = int(np.log2(N)) + 1  # 7 shells for N=64

    print(f"N={N}, Re={Re}, dt={dt}, T={T_final}")
    print(f"Shells: {n_shells} (dyadic: |k| in [2^(j-1), 2^j))")
    print(f"Shell wavenumbers: {[f'2^{j}={2**j}' for j in range(n_shells)]}")

    # Run for two ICs: Taylor-Green and Random
    for ic_name, ic_func in [("Taylor-Green", solver.taylor_green_ic),
                              ("Random (k_peak=4)", lambda: solver.random_ic(seed=42, k_peak=4))]:
        print(f"\n--- IC: {ic_name} ---")
        u_hat = ic_func()

        all_times = []
        all_Z_shells = []
        all_T_energy = []
        all_T_enstrophy = []
        all_Z_total = []

        t = 0.0
        step = 0

        # Header
        shell_labels = " | ".join([f"Z_{j}" for j in range(n_shells)])
        print(f"\n{'t':<6} | {'Z_tot':<11} | {shell_labels}")
        print("-" * (20 + 14 * n_shells))

        while t <= T_final + 1e-10:
            if step % diag_every == 0:
                shells = compute_shell_enstrophy(solver, u_hat, n_shells)
                transfer = compute_triadic_transfer(solver, u_hat, n_shells)

                Z_total = np.sum(shells['Z_shells'])
                all_times.append(t)
                all_Z_shells.append(shells['Z_shells'].copy())
                all_T_energy.append(transfer['T_energy'].copy())
                all_T_enstrophy.append(transfer['T_enstrophy'].copy())
                all_Z_total.append(Z_total)

                # Print
                z_str = " | ".join([f"{z:.3e}" for z in shells['Z_shells']])
                print(f"{t:<6.2f} | {Z_total:<11.4e} | {z_str}")

            u_hat = solver.step_rk4(u_hat, dt)
            t += dt
            step += 1

        # Analysis
        print(f"\n--- ANALYSIS: {ic_name} ---")

        # Find peak enstrophy time
        Z_arr = np.array(all_Z_total)
        Z_shells_arr = np.array(all_Z_shells)
        T_energy_arr = np.array(all_T_energy)
        T_enstrophy_arr = np.array(all_T_enstrophy)
        t_arr = np.array(all_times)

        peak_idx = np.argmax(Z_arr)
        print(f"\nPeak enstrophy at t={t_arr[peak_idx]:.2f}, Z={Z_arr[peak_idx]:.4e}")
        print(f"Shell fractions at peak:")
        for j in range(n_shells):
            frac = Z_shells_arr[peak_idx, j] / Z_arr[peak_idx] if Z_arr[peak_idx] > 0 else 0
            dissip = solver.nu * 4.0**j * Z_shells_arr[peak_idx, j]
            print(f"  Shell {j} (k~{2**j:3d}): Z_j/Z = {frac:6.1%}, "
                  f"T_energy = {T_energy_arr[peak_idx, j]:+.3e}, "
                  f"T_enstrophy = {T_enstrophy_arr[peak_idx, j]:+.3e}, "
                  f"Dissip = {dissip:.3e}")

        # KEY TEST: Is the cascade spectrally local?
        # At peak enstrophy, the enstrophy transfer should peak at shells near
        # the dissipation wavenumber k_d ~ (Z/nu^2)^{1/4}
        k_d = (Z_arr[peak_idx] / solver.nu**2)**0.25
        print(f"\nDissipation wavenumber k_d = {k_d:.1f}")

        # Check locality: does the energy transfer spectrum decay away from k_d?
        peak_transfer_shell = np.argmax(np.abs(T_enstrophy_arr[peak_idx]))
        print(f"Peak enstrophy transfer at shell {peak_transfer_shell} (k~{2**peak_transfer_shell})")

        # BKM blow-up test: does enstrophy concentrate at the highest shells?
        high_shell_frac = np.sum(Z_shells_arr[peak_idx, -2:]) / Z_arr[peak_idx]
        print(f"Enstrophy in top 2 shells: {high_shell_frac:.1%}")
        if high_shell_frac > 0.5:
            print("  WARNING: Enstrophy concentrated at high k -- potential under-resolution!")
        else:
            print("  OK: Enstrophy well-distributed across shells")

        # Dissipation vs stretching balance per shell
        print(f"\nDissipation/Transfer ratio per shell (>1 means dissipation dominates):")
        for j in range(n_shells):
            dissip = solver.nu * 4.0**j * Z_shells_arr[peak_idx, j]
            transfer = abs(T_enstrophy_arr[peak_idx, j])
            ratio = dissip / transfer if transfer > 1e-15 else float('inf')
            bar = "D>>T" if ratio > 10 else ("D>T" if ratio > 1 else "T>D")
            print(f"  Shell {j}: D/T = {ratio:8.2f}  [{bar}]")


# ============================================================
# EXPERIMENT 2: WILSON LOOP / CIRCULATION PDF (MIGDAL TEST)
# ============================================================

def compute_circulation_square(u_phys, dx, center, L_side, plane='xy'):
    """
    Compute circulation Gamma = oint v.dl around a square loop.

    Args:
        u_phys: (3, N, N, N) velocity field in physical space
        dx: grid spacing
        center: (i0, j0, k0) grid indices of loop center
        L_side: side length in grid points
        plane: 'xy', 'xz', or 'yz'

    Returns: Gamma (scalar)
    """
    N = u_phys.shape[1]
    half = L_side // 2
    i0, j0, k0 = center

    Gamma = 0.0

    if plane == 'xy':
        # Bottom edge: x increases, y = j0 - half
        for di in range(-half, half):
            ix = (i0 + di) % N
            iy = (j0 - half) % N
            Gamma += u_phys[0, ix, iy, k0] * dx
        # Right edge: y increases, x = i0 + half
        for dj in range(-half, half):
            ix = (i0 + half) % N
            iy = (j0 + dj) % N
            Gamma += u_phys[1, ix, iy, k0] * dx
        # Top edge: x decreases, y = j0 + half
        for di in range(half, -half, -1):
            ix = (i0 + di) % N
            iy = (j0 + half) % N
            Gamma -= u_phys[0, ix, iy, k0] * dx
        # Left edge: y decreases, x = i0 - half
        for dj in range(half, -half, -1):
            ix = (i0 - half) % N
            iy = (j0 + dj) % N
            Gamma -= u_phys[1, ix, iy, k0] * dx

    elif plane == 'xz':
        for di in range(-half, half):
            ix = (i0 + di) % N
            iz = (k0 - half) % N
            Gamma += u_phys[0, ix, j0, iz] * dx
        for dk in range(-half, half):
            ix = (i0 + half) % N
            iz = (k0 + dk) % N
            Gamma += u_phys[2, ix, j0, iz] * dx
        for di in range(half, -half, -1):
            ix = (i0 + di) % N
            iz = (k0 + half) % N
            Gamma -= u_phys[0, ix, j0, iz] * dx
        for dk in range(half, -half, -1):
            ix = (i0 - half) % N
            iz = (k0 + dk) % N
            Gamma -= u_phys[2, ix, j0, iz] * dx

    elif plane == 'yz':
        for dj in range(-half, half):
            iy = (j0 + dj) % N
            iz = (k0 - half) % N
            Gamma += u_phys[1, i0, iy, iz] * dx
        for dk in range(-half, half):
            iy = (j0 + half) % N
            iz = (k0 + dk) % N
            Gamma += u_phys[2, i0, iy, iz] * dx
        for dj in range(half, -half, -1):
            iy = (j0 + dj) % N
            iz = (k0 + half) % N
            Gamma -= u_phys[1, i0, iy, iz] * dx
        for dk in range(half, -half, -1):
            iy = (j0 - half) % N
            iz = (k0 + dk) % N
            Gamma -= u_phys[2, i0, iy, iz] * dx

    return Gamma


def experiment_2_wilson_loop():
    """
    Migdal's loop-space test: circulation PDFs and Wilson loop functional.

    Tests:
    1. Gamma_rms(L) scaling -- Kolmogorov predicts L^{4/3}
    2. Psi[C] = <exp(i*Gamma/nu)> area-law decay
    3. PDF of Gamma/Gamma_rms for different loop sizes
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: WILSON LOOP / CIRCULATION PDF (MIGDAL TEST)")
    print("  Testing Migdal (1993) loop-space reformulation of NS")
    print("=" * 72)

    N = 64
    Re = 400
    dt = 0.005
    solver = SpectralNS(N=N, Re=Re)
    nu = solver.nu

    # Evolve to developed flow (t ~ 2-3 for TG at Re=400)
    print(f"\nEvolving TG to t=3.0 (developed turbulence)...")
    u_hat = solver.taylor_green_ic()
    t = 0.0
    while t < 3.0:
        u_hat = solver.step_rk4(u_hat, dt)
        t += dt

    u_phys = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    omega_hat = solver.compute_vorticity_hat(u_hat)
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
    Z = np.mean(np.sum(omega**2, axis=0))
    E = 0.5 * np.mean(np.sum(u_phys**2, axis=0))
    print(f"  E = {E:.4e}, Z = {Z:.4e}")

    # Dissipation rate
    epsilon = 2 * nu * Z
    print(f"  epsilon = {epsilon:.4e}")

    # Loop sizes: 2, 4, 8, 16, 32 grid points per side
    loop_sizes = [2, 4, 8, 16, 24]
    n_samples = 200  # spatial samples per loop size

    print(f"\nComputing circulation for {len(loop_sizes)} loop sizes, "
          f"{n_samples} spatial samples each...")

    dx = 2 * np.pi / N
    rng = np.random.RandomState(0)

    results = {}

    for L_grid in loop_sizes:
        L_phys = L_grid * dx
        circulations = []

        # Sample random loop centers and all 3 planes
        for _ in range(n_samples):
            i0, j0, k0 = rng.randint(0, N, 3)
            for plane in ['xy', 'xz', 'yz']:
                Gamma = compute_circulation_square(u_phys, dx, (i0, j0, k0), L_grid, plane)
                circulations.append(Gamma)

        circulations = np.array(circulations)
        Gamma_rms = np.sqrt(np.mean(circulations**2))

        # Wilson loop functional: Psi = <exp(i*Gamma/nu)>
        Psi = np.mean(np.exp(1j * circulations / nu))
        Psi_mag = np.abs(Psi)

        # Also test at different "nu" values (probing semiclassical limit)
        Psi_10nu = np.abs(np.mean(np.exp(1j * circulations / (10*nu))))
        Psi_01nu = np.abs(np.mean(np.exp(1j * circulations / (0.1*nu))))

        results[L_grid] = {
            'L_phys': L_phys,
            'Gamma_rms': Gamma_rms,
            'Psi_mag': Psi_mag,
            'Psi_10nu': Psi_10nu,
            'Psi_01nu': Psi_01nu,
            'n_samples': len(circulations),
            'mean': np.mean(circulations),
            'skewness': np.mean((circulations/Gamma_rms)**3) if Gamma_rms > 0 else 0,
            'kurtosis': np.mean((circulations/Gamma_rms)**4) if Gamma_rms > 0 else 0,
        }

    # Print results
    print(f"\n{'L_grid':<8} {'L_phys':<8} {'Gamma_rms':<12} {'|Psi|':<10} "
          f"{'|Psi|@10nu':<12} {'|Psi|@0.1nu':<12} {'Kurt':<8}")
    print("-" * 80)

    for L_grid in loop_sizes:
        r = results[L_grid]
        print(f"{L_grid:<8} {r['L_phys']:<8.3f} {r['Gamma_rms']:<12.4e} "
              f"{r['Psi_mag']:<10.4f} {r['Psi_10nu']:<12.4f} {r['Psi_01nu']:<12.4f} "
              f"{r['kurtosis']:<8.2f}")

    # Test Kolmogorov scaling: Gamma_rms ~ epsilon^{1/3} * L^{4/3}
    print(f"\n--- KOLMOGOROV SCALING TEST: Gamma_rms ~ L^alpha ---")
    L_arr = np.array([results[L]['L_phys'] for L in loop_sizes])
    G_arr = np.array([results[L]['Gamma_rms'] for L in loop_sizes])

    # Log-log fit (skip smallest loop which may be in dissipation range)
    if len(loop_sizes) >= 3:
        log_L = np.log(L_arr[1:])
        log_G = np.log(G_arr[1:])
        coeffs = np.polyfit(log_L, log_G, 1)
        alpha = coeffs[0]
        print(f"Measured scaling exponent: alpha = {alpha:.3f}")
        print(f"Kolmogorov prediction: alpha = 4/3 = {4/3:.3f}")
        print(f"Deviation: {abs(alpha - 4/3) / (4/3) * 100:.1f}%")

    # Test area law: |Psi| ~ exp(-sigma * L^2)
    print(f"\n--- AREA LAW TEST: log|Psi| ~ -sigma * A ---")
    areas = np.array([results[L]['L_phys']**2 for L in loop_sizes])
    log_psi = np.array([np.log(max(results[L]['Psi_mag'], 1e-15)) for L in loop_sizes])

    # Check if log|Psi| is linear in A (area law) or in L (perimeter law)
    if np.any(log_psi > -14):  # Not all zero
        valid = log_psi > -14
        if np.sum(valid) >= 2:
            perimeters = np.array([results[L]['L_phys'] * 4 for L in loop_sizes])

            # Area fit
            coeffs_area = np.polyfit(areas[valid], log_psi[valid], 1)
            resid_area = np.sum((log_psi[valid] - np.polyval(coeffs_area, areas[valid]))**2)

            # Perimeter fit
            coeffs_perim = np.polyfit(perimeters[valid], log_psi[valid], 1)
            resid_perim = np.sum((log_psi[valid] - np.polyval(coeffs_perim, perimeters[valid]))**2)

            print(f"  Area law fit:      sigma = {-coeffs_area[0]:.4f}, residual = {resid_area:.4f}")
            print(f"  Perimeter law fit: mu    = {-coeffs_perim[0]:.4f}, residual = {resid_perim:.4f}")

            if resid_area < resid_perim:
                print(f"  -> AREA LAW fits better (consistent with Migdal)")
            else:
                print(f"  -> PERIMETER LAW fits better (inconsistent with Migdal)")

    # PDF shape: Gaussian vs non-Gaussian
    print(f"\n--- PDF SHAPE ANALYSIS ---")
    print(f"{'L_grid':<8} {'Skewness':<12} {'Kurtosis':<12} {'Gaussian?':<12}")
    for L_grid in loop_sizes:
        r = results[L_grid]
        is_gaussian = abs(r['kurtosis'] - 3.0) < 0.5
        print(f"{L_grid:<8} {r['skewness']:<12.3f} {r['kurtosis']:<12.3f} "
              f"{'YES' if is_gaussian else 'NO (non-Gaussian tails)'}")


# ============================================================
# EXPERIMENT 3: FRACTAL DIMENSION OF HIGH-VORTICITY REGIONS
# ============================================================

def experiment_3_fractal_dimension():
    """
    Measure box-counting dimension of high-vorticity regions.
    Connects to CKN partial regularity: singular set has parabolic dim <= 1.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: FRACTAL DIMENSION OF HIGH-VORTICITY REGIONS")
    print("  CKN (1982): singular set has Hausdorff dimension <= 1")
    print("=" * 72)

    N = 64
    Re = 400
    dt = 0.005
    solver = SpectralNS(N=N, Re=Re)

    print(f"\nEvolving TG to peak enstrophy...")
    u_hat = solver.taylor_green_ic()
    t = 0.0
    Z_max = 0
    t_peak = 0
    u_hat_peak = u_hat.copy()

    while t < 5.0:
        u_hat = solver.step_rk4(u_hat, dt)
        t += dt
        if int(t / dt) % 40 == 0:
            omega_hat = solver.compute_vorticity_hat(u_hat)
            omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
            Z = np.mean(np.sum(omega**2, axis=0))
            if Z > Z_max:
                Z_max = Z
                t_peak = t
                u_hat_peak = u_hat.copy()

    print(f"  Peak enstrophy at t={t_peak:.2f}, Z={Z_max:.4e}")

    # Compute vorticity magnitude at peak
    omega_hat = solver.compute_vorticity_hat(u_hat_peak)
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
    omega_mag = np.sqrt(np.sum(omega**2, axis=0))
    omega_rms = np.sqrt(np.mean(omega_mag**2))
    omega_max = np.max(omega_mag)

    print(f"  omega_rms = {omega_rms:.3f}, omega_max = {omega_max:.3f}")
    print(f"  omega_max / omega_rms = {omega_max / omega_rms:.2f}")

    # Box-counting dimension for different thresholds
    thresholds = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    print(f"\n{'Threshold':<12} {'lambda*rms':<12} {'Filled%':<10} {'D_box':<10} {'Interpretation'}")
    print("-" * 60)

    for lam in thresholds:
        thresh = lam * omega_rms
        if thresh > omega_max:
            print(f"{lam:<12.1f} {thresh:<12.3f} {'0.0%':<10} {'---':<10} (above max)")
            continue

        active = omega_mag > thresh
        n_active = np.sum(active)
        fill_frac = n_active / N**3

        if n_active == 0:
            print(f"{lam:<12.1f} {thresh:<12.3f} {'0.0%':<10} {'---':<10} (empty)")
            continue

        # Box-counting: count occupied boxes at different scales
        # Scale s: divide N^3 grid into (N/s)^3 boxes of side s
        scales = [1, 2, 4, 8, 16]
        log_N_boxes = []
        log_inv_s = []

        for s in scales:
            if s > N // 2:
                continue
            n_boxes = 0
            for ix in range(0, N, s):
                for iy in range(0, N, s):
                    for iz in range(0, N, s):
                        box = active[ix:ix+s, iy:iy+s, iz:iz+s]
                        if np.any(box):
                            n_boxes += 1
            if n_boxes > 0:
                log_N_boxes.append(np.log(n_boxes))
                log_inv_s.append(np.log(N / s))

        # Fit dimension
        if len(log_N_boxes) >= 3:
            coeffs = np.polyfit(log_inv_s, log_N_boxes, 1)
            D_box = coeffs[0]
        else:
            D_box = float('nan')

        interp = ""
        if D_box < 1.5:
            interp = "filaments (tubes)"
        elif D_box < 2.5:
            interp = "sheets"
        else:
            interp = "volume-filling"

        print(f"{lam:<12.1f} {thresh:<12.3f} {fill_frac*100:<10.2f} {D_box:<10.2f} {interp}")

    print(f"\nInterpretation:")
    print(f"  D_box < 2: vorticity concentrates on filaments -> supports regularity")
    print(f"  D_box >= 2: sheet-like blow-up geometry possible -> CKN says dim(S) <= 1")
    print(f"  Note: At N=64, resolution limits D_box accuracy for extreme thresholds")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    t0 = clock.time()

    experiment_1_frequency_shells()
    experiment_2_wilson_loop()
    experiment_3_fractal_dimension()

    elapsed = clock.time() - t0
    print(f"\n{'='*72}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*72}")
