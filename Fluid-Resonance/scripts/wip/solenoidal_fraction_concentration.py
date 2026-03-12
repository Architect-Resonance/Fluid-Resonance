"""
Solenoidal Fraction of the Lamb Vector under Vorticity Concentration
=====================================================================

Investigation: As vorticity concentrates (approaching potential blow-up),
does the solenoidal (divergence-free) fraction of the Lamb vector L = u x omega
DECREASE (self-limiting), INCREASE (dangerous), or stay CONSTANT?

The Lamb vector decomposes via Helmholtz:
  L = L_irrot + L_sol
where L_irrot is absorbed by pressure (dynamically inactive)
and L_sol is the actual nonlinearity driving the flow.

Method: 3D pseudo-spectral Navier-Stokes, periodic box, RK4, 2/3 dealiasing.

Author: Claude Opus 4.6 / Brendan — Entropy Project
Date: 2026-03-12
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time as clock

# =============================================================================
# Core spectral utilities
# =============================================================================

def setup_grid(N):
    """Create 3D periodic grid and wavenumber arrays."""
    L = 2 * np.pi
    dx = L / N
    x = np.arange(N) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    k = fftfreq(N, d=1.0/N)  # wavenumbers: 0, 1, ..., N/2, -N/2+1, ..., -1
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2_safe = np.where(K2 == 0, 1.0, K2)  # avoid division by zero at k=0

    # 2/3 dealiasing mask
    kmax = N // 3
    dealias = ((np.abs(KX) <= kmax) & (np.abs(KY) <= kmax) & (np.abs(KZ) <= kmax)).astype(float)

    return X, Y, Z, KX, KY, KZ, K2, K2_safe, dealias


def curl_spectral(uh, vh, wh, KX, KY, KZ):
    """Compute curl in Fourier space: omega = curl(u)."""
    # omega_x = dw/dy - dv/dz
    # omega_y = du/dz - dw/dx
    # omega_z = dv/dx - du/dy
    ox = 1j * KY * wh - 1j * KZ * vh
    oy = 1j * KZ * uh - 1j * KX * wh
    oz = 1j * KX * vh - 1j * KY * uh
    return ox, oy, oz


def leray_project(fxh, fyh, fzh, KX, KY, KZ, K2_safe):
    """Leray (solenoidal) projection: remove irrotational part.
    P_sol f = f - k (k . f) / |k|^2
    """
    kdotf = KX * fxh + KY * fyh + KZ * fzh
    proj_x = fxh - KX * kdotf / K2_safe
    proj_y = fyh - KY * kdotf / K2_safe
    proj_z = fzh - KZ * kdotf / K2_safe
    return proj_x, proj_y, proj_z


def compute_rhs(uh, vh, wh, KX, KY, KZ, K2, K2_safe, dealias, nu):
    """Compute RHS of Navier-Stokes in rotational form:
    du/dt = P_sol( u x omega ) - nu * |k|^2 * u_hat

    Actually using the standard form:
    du/dt = -P_sol(u . grad u) - nu |k|^2 u

    We use rotational form: u . grad u = omega x u + grad(|u|^2/2)
    So: du/dt = P_sol(u x omega) - nu |k|^2 u  (pressure + Bernoulli absorbed by projection)
    """
    N = uh.shape[0]

    # Get physical space velocity
    u = np.real(ifftn(uh))
    v = np.real(ifftn(vh))
    w = np.real(ifftn(wh))

    # Compute vorticity in spectral space, then transform
    ox_h, oy_h, oz_h = curl_spectral(uh, vh, wh, KX, KY, KZ)
    ox = np.real(ifftn(ox_h))
    oy = np.real(ifftn(oy_h))
    oz = np.real(ifftn(oz_h))

    # Lamb vector: L = u x omega (note: NOT omega x u)
    # L_x = v*oz - w*oy
    # L_y = w*ox - u*oz
    # L_z = u*oy - v*ox
    Lx = v * oz - w * oy
    Ly = w * ox - u * oz
    Lz = u * oy - v * ox

    # FFT and dealias
    Lxh = fftn(Lx) * dealias
    Lyh = fftn(Ly) * dealias
    Lzh = fftn(Lz) * dealias

    # Leray projection (removes pressure + Bernoulli gradient)
    rhsx, rhsy, rhsz = leray_project(Lxh, Lyh, Lzh, KX, KY, KZ, K2_safe)

    # Add viscous dissipation
    rhsx -= nu * K2 * uh
    rhsy -= nu * K2 * vh
    rhsz -= nu * K2 * wh

    return rhsx, rhsy, rhsz


def compute_diagnostics(uh, vh, wh, KX, KY, KZ, K2_safe, dealias, N):
    """Compute all diagnostic quantities at current state."""
    # Physical space fields
    u = np.real(ifftn(uh))
    v = np.real(ifftn(vh))
    w = np.real(ifftn(wh))

    # Vorticity
    ox_h, oy_h, oz_h = curl_spectral(uh, vh, wh, KX, KY, KZ)
    ox = np.real(ifftn(ox_h))
    oy = np.real(ifftn(oy_h))
    oz = np.real(ifftn(oz_h))

    # Enstrophy and vorticity stats
    omega_mag2 = ox**2 + oy**2 + oz**2
    enstrophy = np.mean(omega_mag2)  # volume-averaged
    omega_max = np.sqrt(np.max(omega_mag2))
    omega_rms = np.sqrt(np.mean(omega_mag2))
    concentration = omega_max / omega_rms if omega_rms > 1e-15 else 1.0

    # Energy
    energy = 0.5 * np.mean(u**2 + v**2 + w**2)

    # Lamb vector L = u x omega
    Lx = v * oz - w * oy
    Ly = w * ox - u * oz
    Lz = u * oy - v * ox

    L_norm2 = np.mean(Lx**2 + Ly**2 + Lz**2)

    if L_norm2 < 1e-30:
        return energy, enstrophy, omega_max, omega_rms, concentration, 0.0

    # FFT of Lamb vector
    Lxh = fftn(Lx) * dealias
    Lyh = fftn(Ly) * dealias
    Lzh = fftn(Lz) * dealias

    # Solenoidal projection
    Lsx, Lsy, Lsz = leray_project(Lxh, Lyh, Lzh, KX, KY, KZ, K2_safe)

    # Compute norms in Fourier space (Parseval)
    L_total_norm2 = (np.sum(np.abs(Lxh)**2) + np.sum(np.abs(Lyh)**2) + np.sum(np.abs(Lzh)**2))
    L_sol_norm2 = (np.sum(np.abs(Lsx)**2) + np.sum(np.abs(Lsy)**2) + np.sum(np.abs(Lsz)**2))

    sol_fraction = L_sol_norm2 / L_total_norm2 if L_total_norm2 > 0 else 0.0

    return energy, enstrophy, omega_max, omega_rms, concentration, float(np.real(sol_fraction))


def rk4_step(uh, vh, wh, dt, KX, KY, KZ, K2, K2_safe, dealias, nu):
    """Standard RK4 time integration."""
    k1x, k1y, k1z = compute_rhs(uh, vh, wh, KX, KY, KZ, K2, K2_safe, dealias, nu)

    k2x, k2y, k2z = compute_rhs(
        uh + 0.5*dt*k1x, vh + 0.5*dt*k1y, wh + 0.5*dt*k1z,
        KX, KY, KZ, K2, K2_safe, dealias, nu)

    k3x, k3y, k3z = compute_rhs(
        uh + 0.5*dt*k2x, vh + 0.5*dt*k2y, wh + 0.5*dt*k2z,
        KX, KY, KZ, K2, K2_safe, dealias, nu)

    k4x, k4y, k4z = compute_rhs(
        uh + dt*k3x, vh + dt*k3y, wh + dt*k3z,
        KX, KY, KZ, K2, K2_safe, dealias, nu)

    uh_new = uh + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
    vh_new = vh + (dt/6) * (k1y + 2*k2y + 2*k3y + k4y)
    wh_new = wh + (dt/6) * (k1z + 2*k2z + 2*k3z + k4z)

    # Enforce dealiasing
    uh_new *= dealias
    vh_new *= dealias
    wh_new *= dealias

    return uh_new, vh_new, wh_new


# =============================================================================
# Initial conditions
# =============================================================================

def ic_taylor_green(X, Y, Z):
    """Classic Taylor-Green vortex."""
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(X)
    return u, v, w


def ic_pelz(X, Y, Z):
    """Pelz-type high-symmetry flow (tends to concentrate vorticity)."""
    u = (np.sin(X) * np.cos(Y) * np.cos(Z)
       + np.sin(Y) * np.cos(Z) * np.cos(X)
       + np.sin(Z) * np.cos(X) * np.cos(Y))
    v = -(np.cos(X) * np.sin(Y) * np.cos(Z)
        + np.cos(Y) * np.sin(Z) * np.cos(X)
        + np.cos(Z) * np.sin(X) * np.cos(Y))
    w = np.zeros_like(X)
    # Project to divergence-free (this IC isn't exactly solenoidal)
    return u, v, w


def ic_random_solenoidal(N, seed=42):
    """Random solenoidal field via random Fourier modes."""
    rng = np.random.RandomState(seed)
    k = fftfreq(N, d=1.0/N)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2_safe = np.where(K2 == 0, 1.0, K2)

    # Random complex field in Fourier space
    shape = (N, N, N)
    fxh = (rng.randn(*shape) + 1j * rng.randn(*shape))
    fyh = (rng.randn(*shape) + 1j * rng.randn(*shape))
    fzh = (rng.randn(*shape) + 1j * rng.randn(*shape))

    # Restrict to low wavenumbers (smooth field)
    mask = (K2 <= 16).astype(float)  # |k| <= 4
    fxh *= mask
    fyh *= mask
    fzh *= mask

    # Leray project to make divergence-free
    kdotf = KX * fxh + KY * fyh + KZ * fzh
    fxh -= KX * kdotf / K2_safe
    fyh -= KY * kdotf / K2_safe
    fzh -= KZ * kdotf / K2_safe

    u = np.real(ifftn(fxh))
    v = np.real(ifftn(fyh))
    w = np.real(ifftn(fzh))

    # Normalize to unit energy
    E = 0.5 * np.mean(u**2 + v**2 + w**2)
    scale = 1.0 / np.sqrt(2 * E) if E > 0 else 1.0
    return u * scale, v * scale, w * scale


def ic_concentrated_vortex(X, Y, Z, sigma=0.3):
    """Taylor-Green + concentrated Gaussian vortex tube at center.
    This creates a field with high |omega|_max / |omega|_rms."""
    # Base Taylor-Green
    u, v, w = ic_taylor_green(X, Y, Z)

    # Add concentrated vortex tube along z-axis at (pi, pi, z)
    r2 = (X - np.pi)**2 + (Y - np.pi)**2
    amp = 3.0  # strength of concentrated vortex
    gauss = amp * np.exp(-r2 / (2 * sigma**2))

    # Vortex tube: circular velocity around z-axis
    # u_theta = Gamma/(2*pi*r) * (1 - exp(-r^2/(2*sigma^2)))
    # In Cartesian: u += -gauss * (Y - pi), v += gauss * (X - pi)
    u += -gauss * (Y - np.pi)
    v += gauss * (X - np.pi)

    return u, v, w


def ensure_divergence_free(u, v, w, KX, KY, KZ, K2_safe, dealias):
    """Project velocity field to be divergence-free."""
    uh, vh, wh = fftn(u), fftn(v), fftn(w)
    uh, vh, wh = leray_project(uh, vh, wh, KX, KY, KZ, K2_safe)
    uh *= dealias; vh *= dealias; wh *= dealias
    return uh, vh, wh


# =============================================================================
# Main simulation runner
# =============================================================================

def run_simulation(ic_name, ic_func, N, nu, dt, T_final, diag_interval=5):
    """Run NS simulation with given IC and collect diagnostics."""
    print(f"\n{'='*70}")
    print(f"  Simulation: {ic_name}")
    print(f"  N={N}, Re={1/nu:.0f}, dt={dt}, T_final={T_final}")
    print(f"{'='*70}")

    X, Y, Z, KX, KY, KZ, K2, K2_safe, dealias = setup_grid(N)

    # Initialize
    if ic_name == "Random Solenoidal":
        u, v, w = ic_func(N)
    else:
        u, v, w = ic_func(X, Y, Z)

    # Project to ensure divergence-free
    uh, vh, wh = ensure_divergence_free(u, v, w, KX, KY, KZ, K2_safe, dealias)

    # Storage
    times = []
    energies = []
    enstrophies = []
    omega_maxes = []
    omega_rmses = []
    concentrations = []
    sol_fractions = []

    n_steps = int(T_final / dt)

    t0 = clock.time()
    for step in range(n_steps + 1):
        t = step * dt

        if step % diag_interval == 0:
            E, Z_enst, om_max, om_rms, conc, sf = compute_diagnostics(
                uh, vh, wh, KX, KY, KZ, K2_safe, dealias, N)

            times.append(t)
            energies.append(E)
            enstrophies.append(Z_enst)
            omega_maxes.append(om_max)
            omega_rmses.append(om_rms)
            concentrations.append(conc)
            sol_fractions.append(sf)

            if step % (diag_interval * 20) == 0:
                print(f"  t={t:.3f}  E={E:.6f}  Z={Z_enst:.4f}  "
                      f"|w|_max={om_max:.3f}  sol_frac={sf:.4f}")

        if step < n_steps:
            uh, vh, wh = rk4_step(uh, vh, wh, dt, KX, KY, KZ, K2, K2_safe, dealias, nu)

    wall_time = clock.time() - t0
    print(f"  Wall time: {wall_time:.1f}s")

    return {
        'name': ic_name,
        'times': np.array(times),
        'energy': np.array(energies),
        'enstrophy': np.array(enstrophies),
        'omega_max': np.array(omega_maxes),
        'omega_rms': np.array(omega_rmses),
        'concentration': np.array(concentrations),
        'sol_fraction': np.array(sol_fractions),
    }


# =============================================================================
# Static concentration test (artificially constructed fields)
# =============================================================================

def static_concentration_test(N=32):
    """Create fields with varying vorticity concentration and measure solenoidal fraction.

    Strategy: superpose a background solenoidal field with a concentrated vortex
    of increasing strength to control |omega|_max / |omega|_rms.
    """
    print(f"\n{'='*70}")
    print(f"  Static Concentration Test")
    print(f"{'='*70}")

    X, Y, Z, KX, KY, KZ, K2, K2_safe, dealias = setup_grid(N)

    # Base field: Taylor-Green
    u0, v0, w0 = ic_taylor_green(X, Y, Z)

    sigmas = [0.8, 0.5, 0.3, 0.2, 0.15, 0.1]
    amplitudes = [0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    results = []

    for amp in amplitudes:
        sigma = 0.25
        u = u0.copy()
        v = v0.copy()
        w = w0.copy()

        if amp > 0:
            r2 = (X - np.pi)**2 + (Y - np.pi)**2
            gauss = amp * np.exp(-r2 / (2 * sigma**2))
            u += -gauss * (Y - np.pi)
            v += gauss * (X - np.pi)

        # Project divergence-free
        uh, vh, wh = ensure_divergence_free(u, v, w, KX, KY, KZ, K2_safe, dealias)

        E, Z_enst, om_max, om_rms, conc, sf = compute_diagnostics(
            uh, vh, wh, KX, KY, KZ, K2_safe, dealias, N)

        results.append({
            'amplitude': amp,
            'concentration': conc,
            'sol_fraction': sf,
            'enstrophy': Z_enst,
            'omega_max': om_max,
            'omega_rms': om_rms,
        })

        print(f"  amp={amp:5.1f}  |w|_max/|w|_rms={conc:6.2f}  "
              f"sol_frac={sf:.4f}  Z={Z_enst:.2f}")

    # Also try varying sigma (width of concentration) at fixed amplitude
    print(f"\n  --- Varying sigma at fixed amplitude=4.0 ---")
    for sigma in sigmas:
        u = u0.copy()
        v = v0.copy()
        w = w0.copy()

        r2 = (X - np.pi)**2 + (Y - np.pi)**2
        gauss = 4.0 * np.exp(-r2 / (2 * sigma**2))
        u += -gauss * (Y - np.pi)
        v += gauss * (X - np.pi)

        uh, vh, wh = ensure_divergence_free(u, v, w, KX, KY, KZ, K2_safe, dealias)

        E, Z_enst, om_max, om_rms, conc, sf = compute_diagnostics(
            uh, vh, wh, KX, KY, KZ, K2_safe, dealias, N)

        results.append({
            'amplitude': 4.0,
            'sigma': sigma,
            'concentration': conc,
            'sol_fraction': sf,
            'enstrophy': Z_enst,
            'omega_max': om_max,
            'omega_rms': om_rms,
        })

        print(f"  sigma={sigma:.2f}  |w|_max/|w|_rms={conc:6.2f}  "
              f"sol_frac={sf:.4f}  Z={Z_enst:.2f}")

    return results


# =============================================================================
# Analysis and plotting
# =============================================================================

def analyze_and_plot(sim_results, static_results):
    """Comprehensive analysis and multi-panel plot."""

    fig = plt.figure(figsize=(20, 24))

    colors_list = ['#364FC7', '#E03131', '#2F9E44', '#F08C00']

    # =========================================================================
    # Panel 1: Solenoidal fraction vs time (all ICs)
    # =========================================================================
    ax1 = fig.add_subplot(4, 2, 1)
    for i, r in enumerate(sim_results):
        ax1.plot(r['times'], r['sol_fraction'], color=colors_list[i],
                 label=r['name'], linewidth=1.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Solenoidal Fraction ||L_sol||² / ||L||²')
    ax1.set_title('Solenoidal Fraction of Lamb Vector vs Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: Enstrophy vs time (all ICs)
    # =========================================================================
    ax2 = fig.add_subplot(4, 2, 2)
    for i, r in enumerate(sim_results):
        ax2.plot(r['times'], r['enstrophy'], color=colors_list[i],
                 label=r['name'], linewidth=1.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Enstrophy Z(t)')
    ax2.set_title('Enstrophy vs Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Solenoidal fraction vs enstrophy (parametric, all ICs)
    # =========================================================================
    ax3 = fig.add_subplot(4, 2, 3)
    for i, r in enumerate(sim_results):
        ax3.scatter(r['enstrophy'], r['sol_fraction'], c=colors_list[i],
                    label=r['name'], s=10, alpha=0.6)
    ax3.set_xlabel('Enstrophy Z(t)')
    ax3.set_ylabel('Solenoidal Fraction')
    ax3.set_title('Sol. Fraction vs Enstrophy (Parametric in Time)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Solenoidal fraction vs concentration ratio (parametric, all ICs)
    # =========================================================================
    ax4 = fig.add_subplot(4, 2, 4)
    for i, r in enumerate(sim_results):
        ax4.scatter(r['concentration'], r['sol_fraction'], c=colors_list[i],
                    label=r['name'], s=10, alpha=0.6)
    ax4.set_xlabel('Concentration |ω|_max / |ω|_rms')
    ax4.set_ylabel('Solenoidal Fraction')
    ax4.set_title('Sol. Fraction vs Vorticity Concentration (Dynamic)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 5: Energy decay (all ICs)
    # =========================================================================
    ax5 = fig.add_subplot(4, 2, 5)
    for i, r in enumerate(sim_results):
        ax5.plot(r['times'], r['energy'], color=colors_list[i],
                 label=r['name'], linewidth=1.5)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Energy E(t)')
    ax5.set_title('Energy Decay')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Max vorticity vs time
    # =========================================================================
    ax6 = fig.add_subplot(4, 2, 6)
    for i, r in enumerate(sim_results):
        ax6.plot(r['times'], r['omega_max'], color=colors_list[i],
                 label=r['name'], linewidth=1.5)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('|ω|_max')
    ax6.set_title('Maximum Vorticity vs Time')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 7: Static test — solenoidal fraction vs concentration
    # =========================================================================
    ax7 = fig.add_subplot(4, 2, 7)
    concs = [r['concentration'] for r in static_results]
    sfs = [r['sol_fraction'] for r in static_results]
    ax7.scatter(concs, sfs, c='#364FC7', s=40, zorder=5)
    ax7.set_xlabel('Concentration |ω|_max / |ω|_rms')
    ax7.set_ylabel('Solenoidal Fraction')
    ax7.set_title('Static Test: Sol. Fraction vs Constructed Concentration')
    ax7.grid(True, alpha=0.3)

    # Fit trend line
    if len(concs) > 2:
        concs_arr = np.array(concs)
        sfs_arr = np.array(sfs)
        z = np.polyfit(concs_arr, sfs_arr, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(concs), max(concs), 100)
        ax7.plot(x_fit, p(x_fit), 'r--', alpha=0.7, label=f'Linear fit (slope={z[0]:.4f})')
        ax7.legend(fontsize=8)

    # =========================================================================
    # Panel 8: Normalized solenoidal fraction (relative to t=0) vs normalized enstrophy
    # =========================================================================
    ax8 = fig.add_subplot(4, 2, 8)
    for i, r in enumerate(sim_results):
        sf0 = r['sol_fraction'][0] if r['sol_fraction'][0] > 0 else 1e-10
        Z0 = r['enstrophy'][0] if r['enstrophy'][0] > 0 else 1e-10
        ax8.plot(r['enstrophy'] / Z0, r['sol_fraction'] / sf0,
                 color=colors_list[i], label=r['name'], linewidth=1.5)
    ax8.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial value')
    ax8.set_xlabel('Z(t) / Z(0)')
    ax8.set_ylabel('Sol. Fraction / Sol. Fraction(0)')
    ax8.set_title('Relative Change: Sol. Fraction vs Enstrophy Growth')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r'h:\tmp\solenoidal_concentration.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to h:\\tmp\\solenoidal_concentration.png")
    plt.close()


def print_detailed_analysis(sim_results, static_results):
    """Print detailed numerical analysis."""
    print(f"\n{'#'*70}")
    print(f"  DETAILED ANALYSIS")
    print(f"{'#'*70}")

    for r in sim_results:
        name = r['name']
        sf = r['sol_fraction']
        Z = r['enstrophy']
        conc = r['concentration']

        # Initial and final values
        sf_init = sf[0]
        sf_final = sf[-1]

        # At peak enstrophy
        idx_peak = np.argmax(Z)
        sf_at_peak = sf[idx_peak]
        Z_peak = Z[idx_peak]
        t_peak = r['times'][idx_peak]

        # Min and max solenoidal fraction
        sf_min = np.min(sf)
        sf_max = np.max(sf)

        # Correlations (only if enough variation)
        if np.std(sf) > 1e-10 and np.std(Z) > 1e-10:
            corr_Z, pval_Z = pearsonr(Z, sf)
        else:
            corr_Z, pval_Z = 0.0, 1.0

        if np.std(sf) > 1e-10 and np.std(conc) > 1e-10:
            corr_conc, pval_conc = pearsonr(conc, sf)
        else:
            corr_conc, pval_conc = 0.0, 1.0

        print(f"\n  --- {name} ---")
        print(f"  Sol. fraction at t=0:           {sf_init:.6f}  ({sf_init*100:.2f}%)")
        print(f"  Sol. fraction at t=final:       {sf_final:.6f}  ({sf_final*100:.2f}%)")
        print(f"  Sol. fraction at peak Z (t={t_peak:.3f}): {sf_at_peak:.6f}  ({sf_at_peak*100:.2f}%)")
        print(f"  Sol. fraction range:            [{sf_min:.6f}, {sf_max:.6f}]")
        print(f"  Change from t=0 to peak Z:      {(sf_at_peak - sf_init):.6f}  ({(sf_at_peak/sf_init - 1)*100:+.2f}%)")
        print(f"  Enstrophy at peak:              {Z_peak:.4f} (at t={t_peak:.3f})")
        print(f"  Enstrophy growth factor:        {Z_peak / Z[0]:.2f}x")
        print(f"  Concentration range:            [{np.min(conc):.2f}, {np.max(conc):.2f}]")
        print(f"  Pearson corr(Z, sol_frac):      {corr_Z:+.4f}  (p={pval_Z:.2e})")
        print(f"  Pearson corr(conc, sol_frac):   {corr_conc:+.4f}  (p={pval_conc:.2e})")

        # Interpretation
        if corr_Z > 0.3 and pval_Z < 0.05:
            print(f"  >> INTERPRETATION: Sol. fraction INCREASES with enstrophy (DANGEROUS)")
        elif corr_Z < -0.3 and pval_Z < 0.05:
            print(f"  >> INTERPRETATION: Sol. fraction DECREASES with enstrophy (SELF-LIMITING)")
        else:
            print(f"  >> INTERPRETATION: No strong correlation with enstrophy")

    # Static test analysis
    print(f"\n  --- Static Concentration Test ---")
    concs = np.array([r['concentration'] for r in static_results])
    sfs = np.array([r['sol_fraction'] for r in static_results])

    if np.std(concs) > 1e-10 and np.std(sfs) > 1e-10:
        corr, pval = pearsonr(concs, sfs)
        print(f"  Pearson corr(concentration, sol_frac): {corr:+.4f}  (p={pval:.2e})")
        slope = np.polyfit(concs, sfs, 1)[0]
        print(f"  Linear fit slope:               {slope:+.6f}")

        if corr > 0.3 and pval < 0.05:
            print(f"  >> Static fields: more concentrated = MORE solenoidal (DANGEROUS)")
        elif corr < -0.3 and pval < 0.05:
            print(f"  >> Static fields: more concentrated = LESS solenoidal (SELF-LIMITING)")
        else:
            print(f"  >> Static fields: no strong relationship")

    # Final verdict
    print(f"\n{'='*70}")
    print(f"  OVERALL VERDICT")
    print(f"{'='*70}")

    # Collect all dynamic correlations
    all_corrs_Z = []
    for r in sim_results:
        sf = r['sol_fraction']
        Z = r['enstrophy']
        if np.std(sf) > 1e-10 and np.std(Z) > 1e-10:
            c, p = pearsonr(Z, sf)
            all_corrs_Z.append((r['name'], c, p))

    n_increasing = sum(1 for _, c, p in all_corrs_Z if c > 0.3 and p < 0.05)
    n_decreasing = sum(1 for _, c, p in all_corrs_Z if c < -0.3 and p < 0.05)
    n_neutral = len(all_corrs_Z) - n_increasing - n_decreasing

    print(f"  Dynamic simulations: {n_increasing} increasing, {n_decreasing} decreasing, {n_neutral} neutral")

    if n_increasing > n_decreasing:
        print(f"  RESULT: Solenoidal fraction tends to INCREASE as vorticity intensifies.")
        print(f"  This suggests the nonlinearity becomes MORE active (less pressure shielding)")
        print(f"  when approaching potential blow-up — a DANGEROUS scenario.")
    elif n_decreasing > n_increasing:
        print(f"  RESULT: Solenoidal fraction tends to DECREASE as vorticity intensifies.")
        print(f"  This suggests a SELF-LIMITING mechanism: pressure absorbs more of the")
        print(f"  nonlinearity precisely when it matters most.")
    else:
        print(f"  RESULT: No clear universal trend. The relationship between vorticity")
        print(f"  concentration and solenoidal fraction appears IC-dependent.")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    N = 32
    Re = 400
    nu = 1.0 / Re
    dt = 0.005
    T_final = 3.0  # enough to see enstrophy peak for Taylor-Green at Re=400
    diag_every = 4  # diagnose every 4 steps

    print(f"Solenoidal Fraction of Lamb Vector — Concentration Investigation")
    print(f"Grid: {N}^3, Re={Re}, dt={dt}, T_final={T_final}")
    print(f"Total steps: {int(T_final/dt)}")

    # Run simulations
    results = []

    results.append(run_simulation(
        "Taylor-Green", ic_taylor_green, N, nu, dt, T_final, diag_every))

    results.append(run_simulation(
        "Pelz (High Symmetry)", ic_pelz, N, nu, dt, T_final, diag_every))

    results.append(run_simulation(
        "Random Solenoidal", lambda *args: ic_random_solenoidal(N, seed=42),
        N, nu, dt, T_final, diag_every))

    results.append(run_simulation(
        "Concentrated Vortex", ic_concentrated_vortex, N, nu, dt, T_final, diag_every))

    # Static concentration test
    static = static_concentration_test(N)

    # Analysis
    print_detailed_analysis(results, static)

    # Plot
    analyze_and_plot(results, static)

    print("Done.")
