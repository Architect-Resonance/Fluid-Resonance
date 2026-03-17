"""
SABRA SHELL MODEL: Enstrophy margin vs Re
==========================================
Instead of measuring alpha (which needs 3D Leray projector),
measure the ENSTROPHY SAFETY MARGIN directly:

  margin = 1 - Z_dot / Z_dot_max

where Z_dot = actual enstrophy growth rate
      Z_dot_max = maximum possible given the energy spectrum

This is what matters for regularity: does Z grow slower than
the worst case? And how does the margin scale with Re?

The Re^{1/7} prediction says: margin ~ Re^{-1/7} -> 0 slowly.

S100-M2, Meridian 2.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time


class SabraModel:
    """Minimal Sabra shell model for enstrophy margin measurement."""

    def __init__(self, N_shells=30, Re=1000, lam=2.0, eps=0.5):
        self.N = N_shells
        self.lam = lam
        self.k = lam ** np.arange(N_shells, dtype=float)
        self.nu = 1.0 / Re
        self.Re = Re
        self.a = 1.0
        self.b = -eps
        self.c = -(1.0 - eps)
        self.f_amp = 1e-2 * (1.0 + 0j)

    def nonlinear(self, u):
        """Nonlinear term only (no viscosity, no forcing)."""
        N = self.N
        k = self.k
        nl = np.zeros(N, dtype=complex)
        for n in range(N):
            if n + 2 < N:
                nl[n] += self.a * k[n+1] * np.conj(u[n+1]) * u[n+2]
            if 0 <= n - 1 and n + 1 < N:
                nl[n] += self.b * k[n] * np.conj(u[n-1]) * u[n+1]
            if n - 2 >= 0:
                nl[n] += self.c * k[n-1] * u[n-1] * u[n-2]
            nl[n] *= 1j
        return nl

    def rhs(self, u):
        """Full RHS = nonlinear - viscous + forcing."""
        r = self.nonlinear(u)
        r -= self.nu * self.k**2 * u
        r[0] += self.f_amp
        return r

    def step_rk4(self, u, dt):
        k1 = self.rhs(u)
        k2 = self.rhs(u + 0.5*dt*k1)
        k3 = self.rhs(u + 0.5*dt*k2)
        k4 = self.rhs(u + dt*k3)
        return u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    def energy(self, u):
        return 0.5 * np.sum(np.abs(u)**2)

    def enstrophy(self, u):
        return 0.5 * np.sum(self.k**2 * np.abs(u)**2)

    def palinstrophy(self, u):
        return 0.5 * np.sum(self.k**4 * np.abs(u)**2)

    def enstrophy_production(self, u):
        """dZ/dt from nonlinear term = sum k_n^2 Re(u_n^* NL_n)."""
        nl = self.nonlinear(u)
        return np.sum(self.k**2 * np.real(np.conj(u) * nl))

    def max_enstrophy_production(self, u):
        """Upper bound on |dZ/dt| from Cauchy-Schwarz.

        |dZ/dt_NL| <= sum k_n^2 |u_n| |NL_n|

        Also: analytic bound from energy/enstrophy: |dZ/dt| <= C * Z^{3/2} / E^{1/2}
        (shell model analogue of the 3D bound).
        """
        nl = self.nonlinear(u)
        # Pointwise bound
        pw_bound = np.sum(self.k**2 * np.abs(u) * np.abs(nl))
        # Analytic bound (3D analogue)
        E = self.energy(u)
        Z = self.enstrophy(u)
        P = self.palinstrophy(u)
        analytic_bound = np.sqrt(P * Z) if Z > 0 else 0.0  # Schwarz on Z_dot

        return pw_bound, analytic_bound

    def dissipation_rate(self, u):
        """Energy dissipation = 2*nu*Z."""
        return 2 * self.nu * self.enstrophy(u)

    def energy_spectrum(self, u):
        return 0.5 * np.abs(u)**2


def run_re(Re, N_shells, n_spinup=80000, n_measure=40000, sample_every=200):
    """Run at given Re, return statistics."""
    model = SabraModel(N_shells=N_shells, Re=Re)
    dt = min(1e-4, 0.5 / model.k[N_shells-1])

    # IC
    rng = np.random.default_rng(42)
    u = np.zeros(N_shells, dtype=complex)
    for n in range(N_shells):
        u[n] = model.k[n]**(-1.0/3.0) * np.exp(2j*np.pi*rng.random()) * 0.01

    print(f"    Re={Re:.0e} N={N_shells} dt={dt:.1e}: ", end='', flush=True)

    # Spinup with adaptive dt
    for step in range(n_spinup):
        u = model.step_rk4(u, dt)
        if np.any(np.isnan(u)) or np.max(np.abs(u)) > 1e15:
            dt *= 0.2
            if dt < 1e-15:
                print("BLOWUP")
                return None
            rng2 = np.random.default_rng(step)
            u = np.zeros(N_shells, dtype=complex)
            for n in range(N_shells):
                u[n] = model.k[n]**(-1.0/3.0) * np.exp(2j*np.pi*rng2.random()) * 0.01
            print(f"(dt->{dt:.0e}) ", end='', flush=True)

    # Measure
    Z_dots = []
    Z_dot_bounds_pw = []
    Z_dot_bounds_an = []
    energies = []
    enstrophies = []
    epsilons = []

    for step in range(n_measure):
        u = model.step_rk4(u, dt)
        if np.any(np.isnan(u)):
            print("BLOWUP in measure")
            return None

        if step % sample_every == 0:
            Zd = model.enstrophy_production(u)
            bpw, ban = model.max_enstrophy_production(u)
            Z_dots.append(Zd)
            Z_dot_bounds_pw.append(bpw)
            Z_dot_bounds_an.append(ban)
            energies.append(model.energy(u))
            enstrophies.append(model.enstrophy(u))
            epsilons.append(model.dissipation_rate(u))

    Z_dots = np.array(Z_dots)
    Z_dot_bounds_pw = np.array(Z_dot_bounds_pw)
    Z_dot_bounds_an = np.array(Z_dot_bounds_an)

    # Margin = 1 - |actual|/bound (how far from worst case)
    # Use pointwise bound
    ratios_pw = np.abs(Z_dots) / np.maximum(Z_dot_bounds_pw, 1e-30)
    ratios_an = np.abs(Z_dots) / np.maximum(Z_dot_bounds_an, 1e-30)
    margin_pw = 1.0 - np.mean(ratios_pw)
    margin_an = 1.0 - np.mean(ratios_an)

    # Depression of nonlinearity: |actual Z_dot| / |analytic bound|
    depression = np.mean(ratios_an)

    E_mean = np.mean(energies)
    Z_mean = np.mean(enstrophies)
    eps_mean = np.mean(epsilons)

    print(f"margin_pw={margin_pw:.4f}, margin_an={margin_an:.4f}, "
          f"depression={depression:.4f}, E={E_mean:.2e}, Z={Z_mean:.2e}")

    return {
        'Re': Re,
        'margin_pw': margin_pw,
        'margin_an': margin_an,
        'depression': depression,
        'E': E_mean,
        'Z': Z_mean,
        'eps': eps_mean,
        'ratio_pw_mean': np.mean(ratios_pw),
        'ratio_pw_std': np.std(ratios_pw),
        'ratio_an_mean': np.mean(ratios_an),
        'ratio_an_std': np.std(ratios_an),
    }


def main():
    print("=" * 70)
    print("  SABRA SHELL MODEL: Enstrophy Safety Margin vs Re")
    print("  Does the margin scale as Re^{-1/7}?")
    print("=" * 70)

    Re_values = [400, 800, 1600, 3200, 6400, 1e4, 3e4, 1e5, 3e5, 1e6]
    N_shells =  [20,  20,  22,   24,   26,   28,  30,  32,  34,  36]

    t0 = time.time()
    results = []
    for Re, Ns in zip(Re_values, N_shells):
        r = run_re(Re, Ns)
        if r is not None:
            results.append(r)

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s")

    # Summary table
    print(f"\n  {'Re':>10}  {'margin_pw':>10}  {'margin_an':>10}  {'depression':>11}  {'Z/E':>10}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['Re']:>10.0f}  {r['margin_pw']:>10.4f}  {r['margin_an']:>10.4f}  "
              f"{r['depression']:>11.4f}  {r['Z']/r['E']:>10.2f}")

    # Fit power law to margin
    Re_arr = np.array([r['Re'] for r in results])
    margin_pw = np.array([r['margin_pw'] for r in results])
    margin_an = np.array([r['margin_an'] for r in results])
    depression = np.array([r['depression'] for r in results])

    # Only fit where margin > 0
    valid_pw = margin_pw > 0
    if np.sum(valid_pw) >= 3:
        p_pw, logA_pw = np.polyfit(np.log(Re_arr[valid_pw]), np.log(margin_pw[valid_pw]), 1)
        print(f"\n  Margin (pointwise) scaling: margin ~ Re^({p_pw:.4f})")
        print(f"  Expected Re^(-1/7) = Re^(-0.143)")
    else:
        p_pw = np.nan
        print(f"\n  Not enough positive margins for fit")

    # Depression of nonlinearity scaling
    p_dep, logA_dep = np.polyfit(np.log(Re_arr), np.log(depression), 1)
    print(f"  Depression ratio scaling: depression ~ Re^({p_dep:.4f})")

    # ================================================================
    # 4-panel figure
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sabra Shell Model: Enstrophy Margin vs Re', fontsize=14)

    # Panel 1: Depression of nonlinearity
    ax = axes[0, 0]
    ax.plot(Re_arr, depression, 'bo-', lw=2, ms=8)
    Re_fit = np.logspace(np.log10(Re_arr[0]), np.log10(Re_arr[-1]), 100)
    ax.plot(Re_fit, np.exp(logA_dep) * Re_fit**p_dep, 'r--', lw=1.5,
            label=f'Fit: Re^({p_dep:.3f})')
    ax.axhline(1.0, color='red', ls=':', lw=1, label='No depression (worst case)')
    ax.set_xscale('log')
    ax.set_xlabel('Re')
    ax.set_ylabel('|Z_dot| / Z_dot_max')
    ax.set_title('Depression of nonlinearity')
    ax.legend()

    # Panel 2: Margin vs Re
    ax = axes[0, 1]
    ax.plot(Re_arr, margin_pw, 'rs-', lw=2, ms=8, label='Pointwise margin')
    ax.plot(Re_arr, margin_an, 'go-', lw=2, ms=8, label='Analytic margin')
    if not np.isnan(p_pw) and np.sum(valid_pw) >= 3:
        ax.plot(Re_fit, np.exp(logA_pw) * Re_fit**p_pw, 'r--', lw=1,
                label=f'Fit pw: Re^({p_pw:.3f})')
    # Reference Re^{-1/7}
    if np.any(valid_pw):
        ref_val = margin_pw[valid_pw][0]
        ref_re = Re_arr[valid_pw][0]
        ax.plot(Re_fit, ref_val * (Re_fit/ref_re)**(-1.0/7), 'k:',
                lw=1.5, label='Re^{-1/7} reference')
    ax.set_xscale('log')
    ax.set_xlabel('Re')
    ax.set_ylabel('Margin = 1 - |Z_dot|/Z_dot_max')
    ax.set_title('Safety margin vs Re')
    ax.legend(fontsize=7)

    # Panel 3: Z/E ratio (effective Re_lambda)
    ax = axes[1, 0]
    ZE = np.array([r['Z']/r['E'] for r in results])
    ax.plot(Re_arr, ZE, 'mo-', lw=2, ms=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Re')
    ax.set_ylabel('Z/E')
    ax.set_title('Enstrophy-to-energy ratio')

    # Panel 4: Dissipation rate
    ax = axes[1, 1]
    eps_arr = np.array([r['eps'] for r in results])
    ax.plot(Re_arr, eps_arr, 'co-', lw=2, ms=8)
    ax.set_xscale('log')
    ax.set_xlabel('Re')
    ax.set_ylabel('epsilon = 2*nu*Z')
    ax.set_title('Energy dissipation rate')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'sabra_margin_vs_re.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved to {out_path}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
