"""
R(t) DYNAMICS DECOMPOSITION — Why does dR/dt < 0 when R is large?
=================================================================

Analytical structure:
    R = ||NL|| / ||-ΔS||  where NL = sym-grad(P(L)), D = -ΔS

    dR/dt = R · (d/dt ln||NL|| - d/dt ln||-ΔS||)

The key decomposition: in Fourier space,
    ||NL||² = Σ_k |NL^(k)|²   (effective k² weighting of P(L))
    ||-ΔS||² = Σ_k k⁴|Ŝ(k)|²  (effective k⁶ weighting of û)

When the forward cascade fills in high-k modes, the k⁶ weighting in ||-ΔS||²
amplifies much faster than the effective k² in ||NL||². This makes D grow
fractionally faster than N → R decreases.

This script verifies:
1. Spectral decomposition: which k-bands drive Ṅ/N vs Ḋ/D
2. Whether the forward cascade preferentially amplifies ||-ΔS|| over ||NL||
3. The role of nonlinear vs dissipative terms in the restoring force
4. Formal dR/dt expression with all terms measured

Uses shared_algebraic_structure.py SpectralNS base class.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


class RDecompositionAnalyzer(SpectralNS):
    """SpectralNS extended with R(t) decomposition diagnostics."""

    def compute_strain_hat(self, u_hat):
        K = [self.kx, self.ky, self.kz]
        N = self.N
        S_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                S_hat[i, j] = 0.5j * (K[i] * u_hat[j] + K[j] * u_hat[i])
        return S_hat

    def compute_neg_laplacian_strain_hat(self, u_hat):
        S_hat = self.compute_strain_hat(u_hat)
        return self.k2[np.newaxis, np.newaxis] * S_hat

    def compute_total_NL_hat(self, u_hat):
        """total_NL = sym-grad(P(L)) where L = u × ω."""
        K = [self.kx, self.ky, self.kz]
        N = self.N
        lamb_hat = self.compute_lamb_hat(u_hat)
        PL_hat = self.project_leray(lamb_hat)
        NL_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                NL_hat[i, j] = 0.5j * (K[i] * PL_hat[j] + K[j] * PL_hat[i])
        return NL_hat

    def tensor_norm_sq_hat(self, T_hat):
        """||T||² = (1/N⁶) Σ |T_{ij}(k)|²."""
        N = self.N
        return np.sum(np.abs(T_hat)**2) / N**6

    def tensor_inner_product_hat(self, A_hat, B_hat):
        """⟨A, B⟩ = (1/N⁶) Re Σ A*B."""
        N = self.N
        return np.real(np.sum(np.conj(A_hat) * B_hat)) / N**6

    def spectral_energy_density(self, T_hat):
        """Return |T(k)|² summed over tensor indices, as a 3D field."""
        return np.sum(np.abs(T_hat)**2, axis=(0, 1))

    def shell_spectrum(self, field_3d, n_shells=None):
        """Bin a 3D field by |k| into radial shells.
        Returns (k_centers, shell_sums).
        """
        kmag = np.sqrt(self.k2)
        if n_shells is None:
            n_shells = self.N // 2
        k_edges = np.arange(0.5, n_shells + 0.5, 1.0)
        k_centers = np.arange(1, n_shells + 1, dtype=float)
        shell_sums = np.zeros(n_shells)
        for i in range(n_shells):
            mask = (kmag >= k_edges[i]) & (kmag < k_edges[i] + 1.0)
            shell_sums[i] = np.sum(field_3d[mask])
        return k_centers, shell_sums

    def compute_R_decomposition(self, u_hat, dt_fd=1e-5):
        """Full decomposition of R(t) dynamics.

        Returns dict with:
        - R, N_norm, D_norm: current values
        - dNdt, dDdt: time derivatives (finite difference)
        - dRdt: = R * (dNdt/N - dDdt/D)
        - NL_spectrum, D_spectrum: spectral densities by shell
        - dNdt_spectrum, dDdt_spectrum: spectral growth rates by shell
        - NL_nonlinear, NL_dissipative: decomposed contributions to dN/dt
        - D_nonlinear, D_dissipative: decomposed contributions to dD/dt
        """
        N = self.N

        # Current state
        NL_hat = self.compute_total_NL_hat(u_hat)
        neglapS_hat = self.compute_neg_laplacian_strain_hat(u_hat)

        N_sq = self.tensor_norm_sq_hat(NL_hat)
        D_sq = self.tensor_norm_sq_hat(neglapS_hat)
        N_norm = np.sqrt(N_sq)
        D_norm = np.sqrt(D_sq)
        R = N_norm / max(D_norm, 1e-30)

        # Spectral densities
        NL_density = self.spectral_energy_density(NL_hat) / N**6
        D_density = self.spectral_energy_density(neglapS_hat) / N**6
        k_centers, NL_spectrum = self.shell_spectrum(NL_density)
        _, D_spectrum = self.shell_spectrum(D_density)

        # Time derivatives via finite difference
        # Forward step
        u_hat_fwd = self.step_rk4(u_hat, dt_fd, mode='full')
        NL_hat_fwd = self.compute_total_NL_hat(u_hat_fwd)
        neglapS_hat_fwd = self.compute_neg_laplacian_strain_hat(u_hat_fwd)
        N_sq_fwd = self.tensor_norm_sq_hat(NL_hat_fwd)
        D_sq_fwd = self.tensor_norm_sq_hat(neglapS_hat_fwd)

        # Backward step
        u_hat_bwd = self.step_rk4(u_hat, -dt_fd, mode='full')
        NL_hat_bwd = self.compute_total_NL_hat(u_hat_bwd)
        neglapS_hat_bwd = self.compute_neg_laplacian_strain_hat(u_hat_bwd)
        N_sq_bwd = self.tensor_norm_sq_hat(NL_hat_bwd)
        D_sq_bwd = self.tensor_norm_sq_hat(neglapS_hat_bwd)

        # Central differences
        dN_sq_dt = (N_sq_fwd - N_sq_bwd) / (2 * dt_fd)
        dD_sq_dt = (D_sq_fwd - D_sq_bwd) / (2 * dt_fd)
        dNdt = dN_sq_dt / (2 * N_norm) if N_norm > 1e-30 else 0.0
        dDdt = dD_sq_dt / (2 * D_norm) if D_norm > 1e-30 else 0.0

        # Fractional growth rates
        gN = dNdt / N_norm if N_norm > 1e-30 else 0.0  # d/dt ln||NL||
        gD = dDdt / D_norm if D_norm > 1e-30 else 0.0  # d/dt ln||-ΔS||

        dRdt = R * (gN - gD)

        # Spectral decomposition of growth rates
        NL_density_fwd = self.spectral_energy_density(NL_hat_fwd) / N**6
        NL_density_bwd = self.spectral_energy_density(NL_hat_bwd) / N**6
        D_density_fwd = self.spectral_energy_density(neglapS_hat_fwd) / N**6
        D_density_bwd = self.spectral_energy_density(neglapS_hat_bwd) / N**6

        dNL_density_dt = (NL_density_fwd - NL_density_bwd) / (2 * dt_fd)
        dD_density_dt = (D_density_fwd - D_density_bwd) / (2 * dt_fd)

        _, dNL_spectrum = self.shell_spectrum(dNL_density_dt)
        _, dD_spectrum = self.shell_spectrum(dD_density_dt)

        # Analytical decomposition of d||-ΔS||²/dt:
        # d||-ΔS||²/dt = 2⟨-ΔS, -Δ(NL)⟩ - 2ν||∇(-ΔS)||²
        #
        # The first term is the nonlinear contribution (production)
        # The second term is the viscous dissipation
        #
        # -Δ(NL) in Fourier: k² * NL^(k)
        neg_lap_NL_hat = self.k2[np.newaxis, np.newaxis] * NL_hat
        D_nonlinear = 2 * self.tensor_inner_product_hat(neglapS_hat, neg_lap_NL_hat)

        # ||∇(-ΔS)||² in Fourier: Σ k² |k²Ŝ(k)|² = Σ k⁶ |Ŝ(k)|²
        grad_neglapS_sq = np.sum(self.k2 * D_density) * N**6 / N**6  # already normalized
        # Actually: ||∇(-ΔS)||² = Σ_k k² * |(-ΔS)^(k)|² / N⁶
        grad_neglapS_sq = np.sum(self.k2 * self.spectral_energy_density(neglapS_hat)) / N**6
        D_dissipative = -2 * self.nu * grad_neglapS_sq

        # Check: D_nonlinear + D_dissipative should ≈ dD_sq_dt
        D_check = D_nonlinear + D_dissipative

        # For ||NL||², the decomposition is harder because NL depends on u nonlinearly.
        # Instead of analytical decomposition, we measure the viscous contribution
        # by comparing full NS evolution with inviscid (ν=0) evolution.
        #
        # Full: dN²/dt = dN²/dt|_full
        # Inviscid: evolve with ν=0, measure dN²/dt|_inviscid
        # Then: NL_dissipative ≈ dN²/dt|_full - dN²/dt|_inviscid

        # Save ν, temporarily set to 0
        nu_save = self.nu
        self.nu = 0.0
        u_hat_fwd_inv = self.step_rk4(u_hat, dt_fd, mode='full')
        u_hat_bwd_inv = self.step_rk4(u_hat, -dt_fd, mode='full')
        self.nu = nu_save

        NL_hat_fwd_inv = self.compute_total_NL_hat(u_hat_fwd_inv)
        NL_hat_bwd_inv = self.compute_total_NL_hat(u_hat_bwd_inv)
        N_sq_fwd_inv = self.tensor_norm_sq_hat(NL_hat_fwd_inv)
        N_sq_bwd_inv = self.tensor_norm_sq_hat(NL_hat_bwd_inv)
        dN_sq_dt_inv = (N_sq_fwd_inv - N_sq_bwd_inv) / (2 * dt_fd)

        N_nonlinear = dN_sq_dt_inv
        N_dissipative = dN_sq_dt - dN_sq_dt_inv

        # Similarly for D
        neglapS_hat_fwd_inv = self.compute_neg_laplacian_strain_hat(u_hat_fwd_inv)
        neglapS_hat_bwd_inv = self.compute_neg_laplacian_strain_hat(u_hat_bwd_inv)
        D_sq_fwd_inv = self.tensor_norm_sq_hat(neglapS_hat_fwd_inv)
        D_sq_bwd_inv = self.tensor_norm_sq_hat(neglapS_hat_bwd_inv)
        dD_sq_dt_inv = (D_sq_fwd_inv - D_sq_bwd_inv) / (2 * dt_fd)

        D_nonlinear_check = dD_sq_dt_inv
        D_dissipative_check = dD_sq_dt - dD_sq_dt_inv

        return {
            'R': R,
            'N_norm': N_norm,
            'D_norm': D_norm,
            'dNdt': dNdt,
            'dDdt': dDdt,
            'gN': gN,  # d/dt ln||NL||
            'gD': gD,  # d/dt ln||-ΔS||
            'dRdt': dRdt,
            'k_centers': k_centers,
            'NL_spectrum': NL_spectrum,
            'D_spectrum': D_spectrum,
            'dNL_spectrum': dNL_spectrum,
            'dD_spectrum': dD_spectrum,
            # Nonlinear vs dissipative decomposition (of d||·||²/dt)
            'dN_sq_dt': dN_sq_dt,
            'dD_sq_dt': dD_sq_dt,
            'N_nonlinear': N_nonlinear,   # d||NL||²/dt from nonlinearity alone
            'N_dissipative': N_dissipative,  # d||NL||²/dt from viscosity
            'D_nonlinear_inv': D_nonlinear_check,   # d||-ΔS||²/dt from nonlinearity
            'D_dissipative_inv': D_dissipative_check,  # d||-ΔS||²/dt from viscosity
            'D_nonlinear_exact': D_nonlinear,   # analytical: 2⟨-ΔS, -Δ(NL)⟩
            'D_dissipative_exact': D_dissipative,  # analytical: -2ν||∇(-ΔS)||²
            'D_check_error': abs(D_check - dD_sq_dt) / max(abs(dD_sq_dt), 1e-30),
        }


def run_decomposition(Re=800, N=32, n_steps=400, ic_type='taylor_green'):
    """Run DNS and collect R decomposition at key timepoints."""
    print(f"\n{'='*70}")
    print(f"R DECOMPOSITION: Re={Re}, {ic_type}, N={N}")
    print(f"{'='*70}")

    solver = RDecompositionAnalyzer(N=N, Re=Re)

    if ic_type == 'taylor_green':
        u_hat = solver.taylor_green_ic()
    elif ic_type == 'imbalanced_80_20':
        u_hat = solver.narrowband_imbalanced_ic(seed=42, h_plus_frac=0.8)
    elif ic_type == 'random':
        u_hat = solver.random_ic(seed=42)
    else:
        u_hat = solver.taylor_green_ic()

    dt = 0.5 / max(N, Re**0.5)

    # Collect full decomposition at regular intervals
    times = []
    decomps = []
    t = 0.0

    # Initial state
    d = solver.compute_R_decomposition(u_hat)
    times.append(t)
    decomps.append(d)

    for step in range(1, n_steps + 1):
        u_hat = solver.step_rk4(u_hat, dt, mode='full')
        t += dt

        if step % 10 == 0:
            d = solver.compute_R_decomposition(u_hat)
            times.append(t)
            decomps.append(d)

            if step % 50 == 0:
                print(f"  step {step:4d}, t={t:.4f}:")
                print(f"    R={d['R']:.6f}, gN={d['gN']:.4f}, gD={d['gD']:.4f}, "
                      f"gN-gD={d['gN']-d['gD']:.4f}")
                print(f"    d||NL||²/dt: nonlinear={d['N_nonlinear']:.4e}, "
                      f"dissipative={d['N_dissipative']:.4e}")
                print(f"    d||-ΔS||²/dt: nonlinear={d['D_nonlinear_inv']:.4e}, "
                      f"dissipative={d['D_dissipative_inv']:.4e}")
                print(f"    D decomp check error: {d['D_check_error']:.2e}")

    return np.array(times), decomps


def analyze_and_plot(times, decomps, label, filename):
    """Analyze the R decomposition and create diagnostic plots."""

    R_arr = np.array([d['R'] for d in decomps])
    gN_arr = np.array([d['gN'] for d in decomps])
    gD_arr = np.array([d['gD'] for d in decomps])
    dRdt_arr = np.array([d['dRdt'] for d in decomps])

    N_nonlin = np.array([d['N_nonlinear'] for d in decomps])
    N_dissip = np.array([d['N_dissipative'] for d in decomps])
    D_nonlin = np.array([d['D_nonlinear_inv'] for d in decomps])
    D_dissip = np.array([d['D_dissipative_inv'] for d in decomps])

    N_sq = np.array([d['N_norm']**2 for d in decomps])
    D_sq = np.array([d['D_norm']**2 for d in decomps])

    # Fractional nonlinear/dissipative rates
    gN_nonlin = np.array([d['N_nonlinear'] / (2 * d['N_norm']**2)
                          if d['N_norm'] > 1e-20 else 0 for d in decomps])
    gN_dissip = np.array([d['N_dissipative'] / (2 * d['N_norm']**2)
                          if d['N_norm'] > 1e-20 else 0 for d in decomps])
    gD_nonlin = np.array([d['D_nonlinear_inv'] / (2 * d['D_norm']**2)
                          if d['D_norm'] > 1e-20 else 0 for d in decomps])
    gD_dissip = np.array([d['D_dissipative_inv'] / (2 * d['D_norm']**2)
                          if d['D_norm'] > 1e-20 else 0 for d in decomps])

    print(f"\n{'='*70}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*70}")

    # Key question: what makes gD > gN (so dR/dt < 0)?
    # gN = gN_nonlin + gN_dissip
    # gD = gD_nonlin + gD_dissip
    # dR/dt < 0 ⟺ gN < gD
    # ⟺ (gN_nonlin - gD_nonlin) + (gN_dissip - gD_dissip) < 0

    nonlin_diff = gN_nonlin - gD_nonlin  # nonlinear contribution to gN - gD
    dissip_diff = gN_dissip - gD_dissip  # dissipative contribution to gN - gD

    # Skip initial points (t ≈ 0) where everything is small
    valid = times > times[1]  # skip t=0

    print(f"\n  Time-averaged contributions to dR/dt = R * (gN - gD):")
    print(f"    mean(gN - gD) = {np.mean((gN_arr - gD_arr)[valid]):.6f}")
    print(f"    mean(gN_nonlin - gD_nonlin) = {np.mean(nonlin_diff[valid]):.6f}")
    print(f"    mean(gN_dissip - gD_dissip) = {np.mean(dissip_diff[valid]):.6f}")
    print()

    if np.mean(nonlin_diff[valid]) < 0:
        print(f"  → NONLINEAR TERMS create the restoring force!")
        print(f"    The cascade amplifies ||-ΔS|| fractionally faster than ||NL||.")
    if np.mean(dissip_diff[valid]) < 0:
        print(f"  → DISSIPATIVE TERMS create the restoring force!")
        print(f"    Higher derivatives in ||-ΔS|| → stronger viscous damping of D.")
        print(f"    But wait — stronger damping of D would INCREASE R, not decrease.")
        print(f"    This means dissipation is actually helping ||NL|| decay faster.")

    if np.mean(nonlin_diff[valid]) < 0 and np.mean(dissip_diff[valid]) > 0:
        print(f"\n  → PICTURE: Nonlinear cascade drives R down (forward cascade)")
        print(f"    while viscosity fights back (damps D more than N).")
        print(f"    The cascade wins: net gN - gD < 0 → R decreases.")
    elif np.mean(nonlin_diff[valid]) > 0 and np.mean(dissip_diff[valid]) < 0:
        print(f"\n  → PICTURE: Dissipation drives R down (NL decays faster than D)")
        print(f"    while nonlinear terms fight back.")
        print(f"    Dissipation wins: net gN - gD < 0 → R decreases.")

    # Spectral analysis at a representative time (peak enstrophy)
    enstrophy = np.array([d['D_norm']**2 for d in decomps])
    peak_idx = np.argmax(enstrophy[5:]) + 5  # skip initial transient
    d_peak = decomps[peak_idx]
    print(f"\n  Spectral analysis at peak ||-ΔS|| (t = {times[peak_idx]:.4f}):")
    print(f"    R = {d_peak['R']:.6f}")

    k = d_peak['k_centers']
    NL_spec = d_peak['NL_spectrum']
    D_spec = d_peak['D_spectrum']

    # Ratio spectrum: ||NL(k)||² / ||-ΔS(k)||²
    ratio_spec = NL_spec / np.maximum(D_spec, 1e-30)
    valid_k = D_spec > 1e-20
    if np.any(valid_k):
        print(f"    k-shell ratio ||NL(k)||²/||-ΔS(k)||²:")
        for i in range(min(10, len(k))):
            if valid_k[i]:
                print(f"      k={k[i]:4.0f}: {ratio_spec[i]:.6f}")

    # Growth rate spectra
    dNL_spec = d_peak['dNL_spectrum']
    dD_spec = d_peak['dD_spectrum']
    gNL_spec = dNL_spec / np.maximum(NL_spec, 1e-30)
    gD_spec = dD_spec / np.maximum(D_spec, 1e-30)

    print(f"\n    Spectral growth rates d/dt ln(·) at peak:")
    print(f"    {'k':>4}  {'g_NL':>10}  {'g_D':>10}  {'g_NL-g_D':>10}  {'drives R':>10}")
    for i in range(min(10, len(k))):
        if valid_k[i] and NL_spec[i] > 1e-20:
            diff = gNL_spec[i] - gD_spec[i]
            direction = "DOWN" if diff < 0 else "UP"
            print(f"    {k[i]:4.0f}  {gNL_spec[i]:10.4f}  {gD_spec[i]:10.4f}  "
                  f"{diff:10.4f}  {direction:>10}")

    # ---- PLOTS ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'R(t) Dynamics Decomposition — {label}', fontsize=14)

    # (0,0) R(t) and dR/dt
    ax = axes[0, 0]
    ax.plot(times, R_arr, 'b-', linewidth=1.5, label='R(t)')
    ax.set_xlabel('t')
    ax.set_ylabel('R', color='b')
    ax.set_title('R(t) = ||NL|| / ||-ΔS||')
    ax2 = ax.twinx()
    ax2.plot(times, dRdt_arr, 'r-', linewidth=0.8, alpha=0.7, label='dR/dt')
    ax2.axhline(y=0, color='k', linewidth=0.3)
    ax2.set_ylabel('dR/dt', color='r')

    # (0,1) Fractional growth rates gN, gD
    ax = axes[0, 1]
    ax.plot(times, gN_arr, 'b-', linewidth=1, label='gN = d/dt ln||NL||')
    ax.plot(times, gD_arr, 'r-', linewidth=1, label='gD = d/dt ln||-ΔS||')
    ax.axhline(y=0, color='k', linewidth=0.3)
    ax.set_xlabel('t')
    ax.set_ylabel('fractional growth rate')
    ax.set_title('gN vs gD (dR/dt < 0 when gN < gD)')
    ax.legend(fontsize=8)

    # (0,2) Nonlinear vs dissipative contributions to gN - gD
    ax = axes[0, 2]
    ax.plot(times, nonlin_diff, 'g-', linewidth=1, label='nonlinear (gN-gD)')
    ax.plot(times, dissip_diff, 'm-', linewidth=1, label='dissipative (gN-gD)')
    ax.plot(times, gN_arr - gD_arr, 'k--', linewidth=1.5, label='total (gN-gD)')
    ax.axhline(y=0, color='r', linewidth=0.5, linestyle='--')
    ax.set_xlabel('t')
    ax.set_ylabel('contribution to gN - gD')
    ax.set_title('What drives dR/dt < 0?')
    ax.legend(fontsize=8)

    # (1,0) Spectral ratio at peak
    ax = axes[1, 0]
    if np.any(valid_k):
        ax.semilogy(k[valid_k], ratio_spec[valid_k], 'b.-')
        ax.set_xlabel('k')
        ax.set_ylabel('||NL(k)||² / ||-ΔS(k)||²')
        ax.set_title(f'Spectral ratio at peak (t={times[peak_idx]:.3f})')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)

    # (1,1) Spectra at peak
    ax = axes[1, 1]
    ax.semilogy(k[valid_k], NL_spec[valid_k], 'b.-', label='||NL(k)||²')
    ax.semilogy(k[valid_k], D_spec[valid_k], 'r.-', label='||-ΔS(k)||²')
    ax.set_xlabel('k')
    ax.set_ylabel('spectral density')
    ax.set_title('NL vs -ΔS spectra at peak')
    ax.legend(fontsize=8)

    # (1,2) Phase portrait with coloring by nonlinear contribution
    ax = axes[1, 2]
    sc = ax.scatter(R_arr[1:], (gN_arr - gD_arr)[1:],
                   c=nonlin_diff[1:], cmap='RdBu_r', s=8, alpha=0.7,
                   vmin=-abs(nonlin_diff[1:]).max(),
                   vmax=abs(nonlin_diff[1:]).max())
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlabel('R')
    ax.set_ylabel('gN - gD')
    ax.set_title('Phase portrait (color = nonlinear contribution)')
    plt.colorbar(sc, ax=ax, label='nonlinear (gN-gD)')

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(outpath, dpi=150)
    print(f"\n  Plot saved: {outpath}")
    plt.close()

    return {
        'times': times,
        'R': R_arr,
        'gN': gN_arr,
        'gD': gD_arr,
        'nonlin_diff': nonlin_diff,
        'dissip_diff': dissip_diff,
    }


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    print("R(t) DYNAMICS DECOMPOSITION")
    print("Why does dR/dt < 0 when R is large?")
    print("=" * 70)

    # Main run: TG at Re=800
    times1, decomps1 = run_decomposition(Re=800, N=32, n_steps=400,
                                          ic_type='taylor_green')
    res1 = analyze_and_plot(times1, decomps1, "TG Re=800 N=32",
                           "r_decomp_TG_800.png")

    # Comparison: Re=1600
    times2, decomps2 = run_decomposition(Re=1600, N=32, n_steps=400,
                                          ic_type='taylor_green')
    res2 = analyze_and_plot(times2, decomps2, "TG Re=1600 N=32",
                           "r_decomp_TG_1600.png")

    # Comparison: Random IC
    times3, decomps3 = run_decomposition(Re=800, N=32, n_steps=400,
                                          ic_type='random')
    res3 = analyze_and_plot(times3, decomps3, "Random Re=800 N=32",
                           "r_decomp_random_800.png")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for label, res in [("TG Re=800", res1), ("TG Re=1600", res2), ("Random Re=800", res3)]:
        valid = res['times'] > res['times'][1]
        mean_total = np.mean((res['gN'] - res['gD'])[valid])
        mean_nonlin = np.mean(res['nonlin_diff'][valid])
        mean_dissip = np.mean(res['dissip_diff'][valid])
        print(f"\n  {label}:")
        print(f"    mean(gN - gD) = {mean_total:.6f}  {'RESTORING' if mean_total < 0 else 'GROWING'}")
        print(f"    from nonlinear: {mean_nonlin:.6f}  ({mean_nonlin/mean_total*100:.1f}% of total)")
        print(f"    from dissipative: {mean_dissip:.6f}  ({mean_dissip/mean_total*100:.1f}% of total)")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If nonlinear contribution dominates (gN_nonlin - gD_nonlin < 0):
  → The FORWARD CASCADE drives R down.
  → The cascade transfers energy to high k, which amplifies ||-ΔS||
     (k⁶ weighting) much more than ||NL|| (k² weighting).
  → This is structural: any broadband flow has this property.
  → Near blowup, the spectrum would need to FLATTEN to make
     ||NL|| grow as fast as ||-ΔS|| — but flattening requires
     the cascade, which drives R down. CIRCULAR TRAP.

If dissipative contribution dominates (gN_dissip - gD_dissip < 0):
  → Viscosity damps ||NL|| faster than ||-ΔS||.
  → This is surprising because ||-ΔS|| has MORE derivatives.
  → Would indicate the nonlinear structure of NL makes it more
     sensitive to viscous damping than the linear -ΔS.
""")
    print("Done.")
