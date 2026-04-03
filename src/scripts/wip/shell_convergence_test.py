"""
WANDERER'S TEST: Shell-by-Shell Enstrophy Production Convergence
================================================================
Question: Does Σ_k |(dZ/dt)_k| converge as K → ∞?

If shell contributions |(dZ/dt)_k| fall faster than k^{-1},
the sum converges and the global ratio stays bounded.

Three zones:
  Shell 1 (Fano, k≤3): bounded by Fano geometry
  Shell 2 (Inertial, 3<k<k_d): need decay with k
  Shell 3 (Dissipation, k>k_d): negative (helps)

Plot: |(dZ/dt)_k| / Z_total^{3/2} vs k for NS-evolved fields.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock


def P(*args, **kwargs):
    print(*args, **kwargs, flush=True)


class SpectralNS:
    def __init__(self, N, nu):
        self.N = N
        self.nu = nu
        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0,0,0] = 1.0
        self.kmag = np.sqrt(self.k2_safe)
        kmax = N // 3
        self.mask = ((np.abs(self.kx) <= kmax) &
                     (np.abs(self.ky) <= kmax) &
                     (np.abs(self.kz) <= kmax)).astype(float)

    def leray_project(self, fx, fy, fz):
        kdotf = self.kx*fx + self.ky*fy + self.kz*fz
        fx = fx - self.kx * kdotf / self.k2_safe
        fy = fy - self.ky * kdotf / self.k2_safe
        fz = fz - self.kz * kdotf / self.k2_safe
        return fx, fy, fz

    def taylor_green_ic(self):
        N = self.N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        ux = np.sin(X) * np.cos(Y) * np.cos(Z)
        uy = -np.cos(X) * np.sin(Y) * np.cos(Z)
        uz = np.zeros_like(X)
        ux_hat = fftn(ux) * self.mask
        uy_hat = fftn(uy) * self.mask
        uz_hat = fftn(uz) * self.mask
        return self.leray_project(ux_hat, uy_hat, uz_hat)

    def nonlinear(self, ux_hat, uy_hat, uz_hat):
        m = self.mask
        ux = np.real(ifftn(ux_hat * m))
        uy = np.real(ifftn(uy_hat * m))
        uz = np.real(ifftn(uz_hat * m))
        ikx, iky, ikz = 1j*self.kx, 1j*self.ky, 1j*self.kz
        dux_dx = np.real(ifftn(ikx*ux_hat*m))
        dux_dy = np.real(ifftn(iky*ux_hat*m))
        dux_dz = np.real(ifftn(ikz*ux_hat*m))
        duy_dx = np.real(ifftn(ikx*uy_hat*m))
        duy_dy = np.real(ifftn(iky*uy_hat*m))
        duy_dz = np.real(ifftn(ikz*uy_hat*m))
        duz_dx = np.real(ifftn(ikx*uz_hat*m))
        duz_dy = np.real(ifftn(iky*uz_hat*m))
        duz_dz = np.real(ifftn(ikz*uz_hat*m))
        nlx = -(ux*dux_dx + uy*dux_dy + uz*dux_dz)
        nly = -(ux*duy_dx + uy*duy_dy + uz*duy_dz)
        nlz = -(ux*duz_dx + uy*duz_dy + uz*duz_dz)
        nlx_hat = fftn(nlx) * m
        nly_hat = fftn(nly) * m
        nlz_hat = fftn(nlz) * m
        return self.leray_project(nlx_hat, nly_hat, nlz_hat)

    def rk4_step(self, ux, uy, uz, dt):
        def rhs(ux, uy, uz):
            nlx, nly, nlz = self.nonlinear(ux, uy, uz)
            return (nlx - self.nu*self.k2*ux,
                    nly - self.nu*self.k2*uy,
                    nlz - self.nu*self.k2*uz)
        k1x, k1y, k1z = rhs(ux, uy, uz)
        k2x, k2y, k2z = rhs(ux+dt/2*k1x, uy+dt/2*k1y, uz+dt/2*k1z)
        k3x, k3y, k3z = rhs(ux+dt/2*k2x, uy+dt/2*k2y, uz+dt/2*k2z)
        k4x, k4y, k4z = rhs(ux+dt*k3x, uy+dt*k3y, uz+dt*k3z)
        ux_new = ux + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
        uy_new = uy + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
        uz_new = uz + dt/6*(k1z + 2*k2z + 2*k3z + k4z)
        return self.leray_project(ux_new, uy_new, uz_new)


def shell_production(ns, ux_hat, uy_hat, uz_hat, max_k=None):
    """
    Returns per-shell: signed production, unsigned production, shell enstrophy.
    """
    nlx, nly, nlz = ns.nonlinear(ux_hat, uy_hat, uz_hat)
    norm = 1.0 / ns.N**6

    # Per-mode enstrophy production
    prod_per_mode = ns.k2 * 2 * np.real(
        np.conj(ux_hat)*nlx + np.conj(uy_hat)*nly + np.conj(uz_hat)*nlz
    ) * norm

    # Per-mode enstrophy
    Z_per_mode = ns.k2 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2 + np.abs(uz_hat)**2) * norm

    kmag = ns.kmag.copy()
    kmag[0,0,0] = 0.0
    if max_k is None:
        max_k = ns.N // 3

    shells = []
    for k in range(1, max_k + 1):
        mask = (kmag >= k - 0.5) & (kmag < k + 0.5)
        if not np.any(mask):
            shells.append((k, 0.0, 0.0, 0.0, 0))
            continue
        signed = np.sum(prod_per_mode[mask])
        unsigned = np.sum(np.abs(prod_per_mode[mask]))
        z_shell = np.sum(Z_per_mode[mask])
        n_modes = int(np.sum(mask))
        shells.append((k, signed, unsigned, z_shell, n_modes))

    return shells


def run_experiment(N, Re, t_final=3.0, dt=0.001, snapshots=None):
    """Run NS and collect shell-by-shell production at snapshots."""
    nu = 1.0 / Re
    ns = SpectralNS(N, nu)
    ux, uy, uz = ns.taylor_green_ic()

    if snapshots is None:
        snapshots = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    n_steps = int(t_final / dt)
    snap_steps = [int(t / dt) for t in snapshots]

    results = {}
    t0 = clock.time()

    # t=0 snapshot
    E0, Z0 = 1.0 / ns.N**6 * np.sum(np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2), \
             1.0 / ns.N**6 * np.sum(ns.k2 * (np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2))

    for step in range(1, n_steps + 1):
        ux, uy, uz = ns.rk4_step(ux, uy, uz, dt)

        if step in snap_steps:
            t = step * dt
            shells = shell_production(ns, ux, uy, uz)
            norm = 1.0 / ns.N**6
            E = np.sum(np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2) * norm
            Z = np.sum(ns.k2 * (np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2)) * norm
            results[t] = {'shells': shells, 'E': E, 'Z': Z}
            elapsed = clock.time() - t0
            P(f"  t={t:.1f}: E={E:.6f}, Z={Z:.3f}, elapsed={elapsed:.0f}s")

        if step % 500 == 0 and step not in snap_steps:
            elapsed = clock.time() - t0
            P(f"  [step {step}, t={step*dt:.3f}, elapsed={elapsed:.0f}s]")

    return results


def plot_shell_convergence(all_results, filename):
    """
    Plot |(dZ/dt)_k| vs k for each experiment at t=3.
    The Wanderer's test: does it decay fast enough for convergence?
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: |signed production| per shell (log-log)
    ax = axes[0, 0]
    for label, results in all_results.items():
        t_key = max(results.keys())  # latest time
        shells = results[t_key]['shells']
        Z_total = results[t_key]['Z']
        ks = [s[0] for s in shells if s[2] > 1e-30]
        signed_abs = [abs(s[1]) for s in shells if s[2] > 1e-30]
        ax.loglog(ks, signed_abs, 'o-', label=f'{label} (t={t_key:.1f})', markersize=4)

    # Reference slopes
    k_ref = np.array([3, 10, 20])
    ax.loglog(k_ref, 0.5 * k_ref**(-5./3), 'k--', alpha=0.3, label='k^{-5/3}')
    ax.loglog(k_ref, 0.5 * k_ref**(-1.0), 'k:', alpha=0.3, label='k^{-1}')
    ax.set_xlabel('Shell k')
    ax.set_ylabel('|signed dZ/dt per shell|')
    ax.set_title('Shell Production (signed magnitude)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: unsigned production per shell
    ax = axes[0, 1]
    for label, results in all_results.items():
        t_key = max(results.keys())
        shells = results[t_key]['shells']
        ks = [s[0] for s in shells if s[2] > 1e-30]
        unsigned = [s[2] for s in shells if s[2] > 1e-30]
        ax.loglog(ks, unsigned, 's-', label=f'{label}', markersize=4)
    ax.set_xlabel('Shell k')
    ax.set_ylabel('unsigned |dZ/dt| per shell')
    ax.set_title('Shell Production (unsigned)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: cancellation per shell
    ax = axes[1, 0]
    for label, results in all_results.items():
        t_key = max(results.keys())
        shells = results[t_key]['shells']
        ks = [s[0] for s in shells if s[2] > 1e-30]
        cancel = [1 - abs(s[1])/s[2] if s[2] > 1e-30 else 0 for s in shells if s[2] > 1e-30]
        ax.plot(ks, cancel, 'o-', label=f'{label}', markersize=4)
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='sin²θ/4 ceiling')
    ax.set_xlabel('Shell k')
    ax.set_ylabel('Cancellation fraction')
    ax.set_title('Per-Shell Cancellation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: normalized ratio per shell |prod_k| / Z_k^{3/2}
    ax = axes[1, 1]
    for label, results in all_results.items():
        t_key = max(results.keys())
        shells = results[t_key]['shells']
        ks = []
        ratios = []
        for s in shells:
            k, signed, unsigned, z_shell, n = s
            if z_shell > 1e-30:
                ks.append(k)
                ratios.append(abs(signed) / z_shell**1.5)
        ax.semilogy(ks, ratios, 'o-', label=f'{label}', markersize=4)
    ax.set_xlabel('Shell k')
    ax.set_ylabel('|dZ_k/dt| / Z_k^{3/2}')
    ax.set_title('Per-Shell Ratio (Wanderer\'s convergence test)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Shell-by-Shell Enstrophy Production — Convergence Test", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    P(f"  Plot saved: {filename}")


def plot_time_evolution(results, label, filename):
    """Plot shell production at multiple times for one experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    times = sorted(results.keys())
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(times)))

    # Signed magnitude vs k at each time
    ax = axes[0]
    for i, t in enumerate(times):
        shells = results[t]['shells']
        ks = [s[0] for s in shells if s[2] > 1e-30]
        signed = [abs(s[1]) for s in shells if s[2] > 1e-30]
        if signed:
            ax.semilogy(ks, signed, 'o-', color=cmap[i], label=f't={t:.1f}', markersize=3)
    ax.set_xlabel('Shell k')
    ax.set_ylabel('|signed dZ/dt per shell|')
    ax.set_title(f'{label}: Signed Production')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Cancellation vs k at each time
    ax = axes[1]
    for i, t in enumerate(times):
        shells = results[t]['shells']
        ks = [s[0] for s in shells if s[2] > 1e-30]
        cancel = [1 - abs(s[1])/s[2] if s[2] > 1e-30 else 0 for s in shells if s[2] > 1e-30]
        ax.plot(ks, cancel, 'o-', color=cmap[i], label=f't={t:.1f}', markersize=3)
    ax.set_xlabel('Shell k')
    ax.set_ylabel('Cancellation')
    ax.set_title(f'{label}: Per-Shell Cancellation')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Cumulative sum test: Σ_{k'=1}^{k} |signed_k'| / Z^{3/2}
    ax = axes[2]
    for i, t in enumerate(times):
        shells = results[t]['shells']
        Z = results[t]['Z']
        ks = [s[0] for s in shells]
        cumsum = np.cumsum([abs(s[1]) for s in shells])
        if Z > 1e-30:
            ax.plot(ks, cumsum / Z**1.5, 'o-', color=cmap[i], label=f't={t:.1f}', markersize=3)
    ax.set_xlabel('Shell k (cumulative up to)')
    ax.set_ylabel('Σ|signed_k| / Z^{3/2}')
    ax.set_title(f'{label}: Cumulative Ratio (converges?)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    P(f"  Plot saved: {filename}")


# ============================================================
# MAIN
# ============================================================

P("="*72)
P("  WANDERER'S TEST: Shell-by-Shell Enstrophy Production Convergence")
P("="*72)

all_results = {}

# Experiment 1: N=32, Re=400
P("\n--- N=32, Re=400 ---")
r1 = run_experiment(N=32, Re=400, t_final=3.0, dt=0.001)
all_results['N32_Re400'] = r1

# Experiment 2: N=32, Re=1600
P("\n--- N=32, Re=1600 ---")
r2 = run_experiment(N=32, Re=1600, t_final=3.0, dt=0.0005)
all_results['N32_Re1600'] = r2

# Experiment 3: N=64, Re=400
P("\n--- N=64, Re=400 ---")
r3 = run_experiment(N=64, Re=400, t_final=3.0, dt=0.001)
all_results['N64_Re400'] = r3

# Experiment 4: N=64, Re=1600
P("\n--- N=64, Re=1600 ---")
r4 = run_experiment(N=64, Re=1600, t_final=3.0, dt=0.0005)
all_results['N64_Re1600'] = r4

# ---- ANALYSIS ----
P("\n" + "="*72)
P("  SHELL-BY-SHELL ANALYSIS AT LATEST TIME")
P("="*72)

for label, results in all_results.items():
    t_key = max(results.keys())
    shells = results[t_key]['shells']
    Z = results[t_key]['Z']
    E = results[t_key]['E']
    P(f"\n  {label} at t={t_key:.1f}: E={E:.6f}, Z={Z:.3f}")
    P(f"  {'k':>4}  {'|signed|':>12}  {'unsigned':>12}  {'cancel%':>8}  {'Z_shell':>12}  {'|s|/Z_s^1.5':>12}  {'n_modes':>8}")
    for k, signed, unsigned, z_shell, n in shells:
        if unsigned < 1e-30:
            continue
        cancel = 1 - abs(signed)/unsigned
        ratio = abs(signed) / z_shell**1.5 if z_shell > 1e-30 else 0
        P(f"  {k:4d}  {abs(signed):12.6f}  {unsigned:12.6f}  {cancel:7.1%}  {z_shell:12.6f}  {ratio:12.6f}  {n:8d}")

    # Cumulative sum
    P(f"\n  Cumulative convergence test:")
    cumsum_signed = 0.0
    cumsum_unsigned = 0.0
    for k, signed, unsigned, z_shell, n in shells:
        cumsum_signed += abs(signed)
        cumsum_unsigned += unsigned
        if unsigned > 1e-30:
            P(f"    k<={k:2d}: Sum|signed|/Z^1.5 = {cumsum_signed/Z**1.5:.6f}  "
              f"Sum_unsigned/Z^1.5 = {cumsum_unsigned/Z**1.5:.6f}  "
              f"cancel_cumul = {1 - cumsum_signed/cumsum_unsigned:.1%}")

# ---- PLOTS ----
P("\n  Generating plots...")
outdir = 'Fluid-Resonance/scripts/wip'

plot_shell_convergence(all_results, f'{outdir}/shell_convergence_comparison.png')

for label, results in all_results.items():
    plot_time_evolution(results, label, f'{outdir}/shell_evolution_{label}.png')

P("\n  DONE.")
