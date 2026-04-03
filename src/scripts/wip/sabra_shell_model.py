"""
SABRA SHELL MODEL: Leray suppression at Re up to 10^6
======================================================
The Sabra model (L'vov et al. 1998) is a 1D reduction of the NS cascade:
  du_n/dt = i*Nonlinear_n - nu*k_n^2*u_n + f_n

Each u_n is a complex velocity at shell k_n = k0 * lambda^n.
Triadic interactions couple (n-1, n, n+1).

We decompose u_n = u_n^+ + u_n^- (helical), measure cross-helical
energy transfer, and compute the effective Leray suppression alpha(Re).

Reference: L'vov, Podivilov, Pomyalov, Procaccia, Vandembroucq (1998)
           Biferale (2003) Ann. Rev. Fluid Mech. — shell model review

S100-M2, Meridian 2.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time


class SabraShellModel:
    """Sabra shell model with helical decomposition."""

    def __init__(self, N_shells=30, Re=1000, lam=2.0, k0=1.0,
                 eps_a=0.5, forcing_shell=0):
        self.N = N_shells
        self.lam = lam
        self.k0 = k0
        self.nu = 1.0 / Re
        self.Re = Re
        self.forcing_shell = forcing_shell

        # Wavenumbers
        self.k = k0 * lam ** np.arange(N_shells)

        # Sabra coupling constants (conserve energy + helicity)
        # Standard: a=1, b=-eps, c=-(1-eps) with eps in (0,1)
        # eps=0.5 is canonical (equal forward/backward)
        self.a = 1.0
        self.b = -eps_a
        self.c = -(1.0 - eps_a)

        # Forcing amplitude (maintain constant energy injection)
        self.f_amp = 1e-2 * (1.0 + 0j)

    def rhs(self, u):
        """Compute du/dt for the Sabra model."""
        N = self.N
        k = self.k
        dudt = np.zeros(N, dtype=complex)

        # Nonlinear term: triadic interactions
        for n in range(N):
            nl = 0j
            # Forward coupling: n, n+1, n+2
            if n + 2 < N:
                nl += self.a * k[n+1] * np.conj(u[n+1]) * u[n+2]
            # Local coupling: n-1, n, n+1
            if n - 1 >= 0 and n + 1 < N:
                nl += self.b * k[n] * np.conj(u[n-1]) * u[n+1]
            # Backward coupling: n-2, n-1, n
            if n - 2 >= 0:
                nl += self.c * k[n-1] * u[n-1] * u[n-2]

            dudt[n] = 1j * nl

        # Viscous dissipation
        dudt -= self.nu * k**2 * u

        # Forcing (at large scale)
        dudt[self.forcing_shell] += self.f_amp

        return dudt

    def rhs_helical(self, u_plus, u_minus):
        """Compute du+/dt and du-/dt separately, tracking cross-helical transfer."""
        N = self.N
        k = self.k
        u = u_plus + u_minus

        # Full RHS
        dudt_full = self.rhs(u)

        # Same-helicity RHS (BT surgery: only ++ and -- interactions)
        dudt_pp = np.zeros(N, dtype=complex)
        dudt_mm = np.zeros(N, dtype=complex)

        for n in range(N):
            nl_pp = 0j
            nl_mm = 0j

            if n + 2 < N:
                nl_pp += self.a * k[n+1] * np.conj(u_plus[n+1]) * u_plus[n+2]
                nl_mm += self.a * k[n+1] * np.conj(u_minus[n+1]) * u_minus[n+2]
            if n - 1 >= 0 and n + 1 < N:
                nl_pp += self.b * k[n] * np.conj(u_plus[n-1]) * u_plus[n+1]
                nl_mm += self.b * k[n] * np.conj(u_minus[n-1]) * u_minus[n+1]
            if n - 2 >= 0:
                nl_pp += self.c * k[n-1] * u_plus[n-1] * u_plus[n-2]
                nl_mm += self.c * k[n-1] * u_minus[n-1] * u_minus[n-2]

            dudt_pp[n] = 1j * nl_pp
            dudt_mm[n] = 1j * nl_mm

        dudt_same = dudt_pp + dudt_mm
        # Cross-helical = full - same
        dudt_cross = dudt_full - dudt_same - (-self.nu * k**2 * u)
        # Add back viscosity to full
        dudt_cross_with_visc = dudt_cross  # pure nonlinear cross part

        return dudt_full, dudt_same, dudt_cross

    def step_rk4(self, u, dt):
        """RK4 time step."""
        k1 = self.rhs(u)
        k2 = self.rhs(u + 0.5 * dt * k1)
        k3 = self.rhs(u + 0.5 * dt * k2)
        k4 = self.rhs(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def energy_spectrum(self, u):
        """Energy per shell: E_n = |u_n|^2 / 2."""
        return 0.5 * np.abs(u)**2

    def total_energy(self, u):
        return np.sum(self.energy_spectrum(u))

    def total_enstrophy(self, u):
        return 0.5 * np.sum(self.k**2 * np.abs(u)**2)

    def dissipation_rate(self, u):
        return 2.0 * self.nu * self.total_enstrophy(u)

    def measure_alpha(self, u):
        """Measure effective Leray suppression.

        Decompose u into helical halves, compute cross-helical energy transfer,
        and measure what fraction survives (analogous to solenoidal projection).

        In the shell model, the "Leray projection" analogue is:
        alpha = |cross-helical enstrophy production| / |total enstrophy production|

        This measures how much of the nonlinear stretching comes from
        cross-helical interactions (the part Leray suppresses in 3D).
        """
        # Split into helical halves (random phase split, averaged over realizations)
        # For a single snapshot, use amplitude-based split
        u_plus = u / 2.0   # Equal split (isotropic assumption)
        u_minus = u / 2.0

        _, dudt_same, dudt_cross = self.rhs_helical(u_plus, u_minus)

        # Enstrophy production: sum_n k_n^2 * Re(u_n^* * dudt_n)
        P_cross = np.sum(self.k**2 * np.real(np.conj(u) * dudt_cross))
        P_same = np.sum(self.k**2 * np.real(np.conj(u) * dudt_same))
        P_total = P_cross + P_same

        if abs(P_total) < 1e-30:
            return 0.0, 0.0, 0.0

        alpha = abs(P_cross) / (abs(P_cross) + abs(P_same))

        return alpha, P_cross, P_same

    def measure_alpha_phase_averaged(self, u, n_phases=16):
        """Average alpha over random helical decompositions.

        The equal split u/2 is one choice. Average over random phase rotations
        to get a robust estimate.
        """
        alphas = []
        rng = np.random.default_rng(42)

        for _ in range(n_phases):
            # Random phase rotation for helical split
            phase = rng.uniform(0, 2*np.pi, self.N)
            u_plus = u * np.exp(1j * phase) / 2.0
            u_minus = u * np.exp(-1j * phase) / 2.0

            _, dudt_same, dudt_cross = self.rhs_helical(u_plus, u_minus)

            P_cross = np.sum(self.k**2 * np.abs(np.real(np.conj(u) * dudt_cross)))
            P_same = np.sum(self.k**2 * np.abs(np.real(np.conj(u) * dudt_same)))

            if P_cross + P_same > 1e-30:
                alphas.append(P_cross / (P_cross + P_same))

        return np.mean(alphas) if alphas else 0.0


def run_single_re(Re, N_shells=30, n_spinup=50000, n_measure=20000,
                  sample_every=100, verbose=True):
    """Run Sabra model at given Re, return time-averaged alpha."""
    model = SabraShellModel(N_shells=N_shells, Re=Re)

    # CFL-like dt: based on largest wavenumber and typical velocity
    # For shell models, dt ~ 1/(k_max * u_max). Start conservative.
    dt = min(1e-4, 1.0 / model.k[N_shells-1])

    # Initial condition: Kolmogorov-like with random phases
    rng = np.random.default_rng(123)
    u = np.zeros(N_shells, dtype=complex)
    for n in range(N_shells):
        amp = model.k[n]**(-1.0/3.0) * np.exp(-model.nu * model.k[n]**2 * 0.1)
        u[n] = amp * np.exp(2j * np.pi * rng.random())

    # Spinup
    if verbose:
        print(f"    Re={Re:.0e} (N={N_shells}, dt={dt:.1e}): spinup {n_spinup}...", end='', flush=True)

    for step in range(n_spinup):
        u = model.step_rk4(u, dt)
        if np.any(np.isnan(u)) or np.max(np.abs(u)) > 1e10:
            # Try smaller dt
            dt *= 0.1
            if dt < 1e-10:
                if verbose:
                    print(f" BLOWUP at step {step}!")
                return np.nan, np.nan, np.nan
            # Reset and retry
            for n in range(N_shells):
                amp = model.k[n]**(-1.0/3.0) * np.exp(-model.nu * model.k[n]**2 * 0.1)
                u[n] = amp * np.exp(2j * np.pi * rng.random())
            if verbose:
                print(f" (dt->{dt:.1e})", end='', flush=True)

    # Measurement phase
    alphas = []
    energies = []

    for step in range(n_measure):
        u = model.step_rk4(u, dt)

        if step % sample_every == 0:
            a = model.measure_alpha_phase_averaged(u, n_phases=8)
            alphas.append(a)
            energies.append(model.total_energy(u))

        if np.any(np.isnan(u)) or np.max(np.abs(u)) > 1e10:
            if verbose:
                print(f" BLOWUP during measurement at step {step}!")
            return np.nan, np.nan, np.nan

    alpha_mean = np.mean(alphas)
    alpha_std = np.std(alphas)
    E_mean = np.mean(energies)

    if verbose:
        print(f" alpha = {alpha_mean:.4f} +/- {alpha_std:.4f}, E = {E_mean:.2e}")

    return alpha_mean, alpha_std, E_mean


def main():
    print("=" * 70)
    print("  SABRA SHELL MODEL: Leray Suppression vs Reynolds Number")
    print("  Re range: 400 to 10^6")
    print("=" * 70)

    # Reynolds number sweep
    Re_values = [400, 800, 1600, 3200, 6400, 1e4, 3e4, 1e5, 3e5, 1e6]
    N_shells_values = [20, 20, 22, 24, 26, 28, 30, 32, 34, 36]

    results = []
    t0 = time.time()

    for Re, N_sh in zip(Re_values, N_shells_values):
        alpha, alpha_std, E = run_single_re(
            Re, N_shells=N_sh, n_spinup=50000, n_measure=20000,
            sample_every=100, verbose=True)
        results.append((Re, alpha, alpha_std, E))

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Results table
    print(f"\n  {'Re':>10}  {'alpha':>8}  {'std':>8}  {'Energy':>10}")
    print("  " + "-" * 45)
    for Re, alpha, std, E in results:
        if not np.isnan(alpha):
            print(f"  {Re:>10.0f}  {alpha:>8.4f}  {std:>8.4f}  {E:>10.2e}")
        else:
            print(f"  {Re:>10.0f}  {'BLOWUP':>8}  {'---':>8}  {'---':>10}")

    # Filter valid results
    valid = [(Re, a, s) for Re, a, s, E in results if not np.isnan(a)]
    if len(valid) < 3:
        print("\n  Not enough valid data points for scaling analysis.")
        return

    Re_arr = np.array([r[0] for r in valid])
    alpha_arr = np.array([r[1] for r in valid])
    std_arr = np.array([r[2] for r in valid])

    # Power law fit: alpha(Re) ~ A * Re^p
    log_Re = np.log(Re_arr)
    log_alpha = np.log(alpha_arr)
    p, log_A = np.polyfit(log_Re, log_alpha, 1)
    A = np.exp(log_A)

    print(f"\n  Power law fit: alpha(Re) ~ {A:.4f} * Re^({p:.4f})")
    print(f"  Expected for Re^(1/7) margin: exponent ~ -0.143")
    print(f"  Measured exponent: {p:.4f}")

    # Safety margin: how far is alpha from 1/2 (the blowup threshold)?
    # Margin = (1/2 - alpha) / (1/2)
    margins = (0.5 - alpha_arr) / 0.5
    log_margins = np.log(margins)
    p_margin, log_A_margin = np.polyfit(log_Re, log_margins, 1)

    print(f"\n  Safety margin (1/2 - alpha)/(1/2):")
    print(f"  Power law: margin ~ {np.exp(log_A_margin):.4f} * Re^({p_margin:.4f})")

    # Also check: alpha relative to 1/4 (the incoherent bound)
    print(f"\n  Comparison with theoretical bounds:")
    print(f"  {'Re':>10}  {'alpha':>8}  {'alpha/0.25':>10}  {'alpha/0.307':>11}  {'margin':>8}")
    print("  " + "-" * 55)
    for Re, a, s in valid:
        print(f"  {Re:>10.0f}  {a:>8.4f}  {a/0.25:>10.4f}  {a/0.307:>11.4f}  {(0.5-a)/0.5:>8.4f}")

    # ================================================================
    # 4-panel figure
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sabra Shell Model: Leray Suppression vs Re', fontsize=14)

    # Panel 1: alpha vs Re (log-log)
    ax = axes[0, 0]
    ax.errorbar(Re_arr, alpha_arr, yerr=std_arr, fmt='bo-', lw=2, ms=8, capsize=4)
    Re_fit = np.logspace(np.log10(Re_arr[0]), np.log10(Re_arr[-1]), 100)
    ax.plot(Re_fit, A * Re_fit**p, 'r--', lw=1.5,
            label=f'Fit: {A:.3f} * Re^({p:.3f})')
    ax.axhline(0.25, color='orange', ls=':', lw=1, label='alpha_E = 1/4')
    ax.axhline(0.307, color='green', ls=':', lw=1, label='1 - ln2 = 0.307')
    ax.axhline(0.5, color='red', ls=':', lw=1, label='Blowup threshold = 1/2')
    ax.set_xscale('log')
    ax.set_xlabel('Re')
    ax.set_ylabel('alpha (cross-helical fraction)')
    ax.set_title('Leray suppression vs Re')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 0.6)

    # Panel 2: Safety margin vs Re (log-log)
    ax = axes[0, 1]
    ax.plot(Re_arr, margins, 'rs-', lw=2, ms=8)
    ax.plot(Re_fit, np.exp(log_A_margin) * Re_fit**p_margin, 'b--', lw=1.5,
            label=f'Fit: margin ~ Re^({p_margin:.3f})')
    # Show Re^{1/7} for comparison
    ax.plot(Re_fit, margins[0] * (Re_fit/Re_arr[0])**(-1.0/7), 'g:',
            lw=1.5, label='Re^{-1/7} reference')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Re')
    ax.set_ylabel('Safety margin (1/2 - alpha)/(1/2)')
    ax.set_title('Safety margin scaling')
    ax.legend(fontsize=8)

    # Panel 3: alpha/0.25 ratio (how close to incoherent bound)
    ax = axes[1, 0]
    ratios = alpha_arr / 0.25
    ax.plot(Re_arr, ratios, 'go-', lw=2, ms=8)
    ax.axhline(1.0, color='orange', ls='--', lw=1, label='alpha_E = 1/4')
    ax.axhline(0.307/0.25, color='green', ls=':', lw=1, label='(1-ln2)/0.25 = 1.23')
    ax.set_xscale('log')
    ax.set_xlabel('Re')
    ax.set_ylabel('alpha / 0.25')
    ax.set_title('Ratio to incoherent bound')
    ax.legend(fontsize=8)

    # Panel 4: Energy spectrum at highest Re
    ax = axes[1, 1]
    N_sh_spec = N_shells_values[-1]
    model = SabraShellModel(N_shells=N_sh_spec, Re=Re_values[-1])
    # Quick run for spectrum
    rng = np.random.default_rng(123)
    u = np.zeros(model.N, dtype=complex)
    for n in range(model.N):
        u[n] = model.k[n]**(-1.0/3.0) * np.exp(2j*np.pi*rng.random())
    dt_spec = min(1e-4, 1.0 / model.k[N_sh_spec-1])
    for _ in range(50000):
        u = model.step_rk4(u, dt_spec)
        if np.any(np.isnan(u)):
            break
    E_spec = model.energy_spectrum(u)
    ax.loglog(model.k, E_spec, 'b-', lw=2, label=f'Re={Re_values[-1]:.0e}')
    # K41 reference
    k_ref = model.k[2:-5]
    ax.loglog(k_ref, 0.1 * k_ref**(-2.0/3.0), 'r--', lw=1, label='k^{-2/3} (K41)')
    ax.set_xlabel('k')
    ax.set_ylabel('E(k) = |u_n|^2/2')
    ax.set_title('Energy spectrum (steady state)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'sabra_alpha_vs_re.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved to {out_path}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
