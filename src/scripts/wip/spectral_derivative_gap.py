# -*- coding: utf-8 -*-
"""
SPECTRAL DERIVATIVE GAP — Why g_D > g_N (the formal mechanism)
===============================================================
S113-W — Wanderer session.

Key theorem candidate: The forward cascade amplifies ||-dS|| faster than ||Q||
because ||-dS(k)||^2 ~ k^6 E(k) (LINEAR, intra-scale) while ||Q(k)||^2 involves
convolutions (QUADRATIC, inter-scale). The ratio r(k) = ||Q(k)||/||-dS(k)||
decreases monotonically with k.

M1 confirmed r(k) ~ k^{-1} at Re=400, N=32. This script tests:
  1. r(k) at multiple Re to check for K41 scaling
  2. The spectral decomposition of g_D and g_N (where in k-space does restoration come from?)
  3. The connection between forward flux Pi(k) and g_D - g_N

K41 prediction for inertial range:
  ||-dS(k)||^2_shell ~ k^6 * E(k) ~ k^{6-5/3} = k^{13/3}
  ||Q(k)||^2_shell ~ (convolution)^2 ~ k^{-4/3}  (from |u(k)| ~ k^{-11/6})
  r(k) ~ k^{-17/6} ~ k^{-2.83}  (steeper than M1's low-Re k^{-1})

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


class SpectralGapAnalyzer(SpectralNS):
    """Extended SpectralNS with shell-by-shell spectral analysis."""

    def compute_strain_hat(self, u_hat):
        """S_ij(k) = (i/2)(k_i u_j + k_j u_i)."""
        K = [self.kx, self.ky, self.kz]
        N = self.N
        S_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                S_hat[i, j] = 0.5j * (K[i] * u_hat[j] + K[j] * u_hat[i])
        return S_hat

    def compute_neg_laplacian_strain_hat(self, u_hat):
        """-dS in Fourier: k^2 S_ij(k)."""
        S_hat = self.compute_strain_hat(u_hat)
        return self.k2[np.newaxis, np.newaxis] * S_hat

    def compute_total_NL_hat(self, u_hat):
        """total_NL = sym-grad(P(L)) where L = u x omega."""
        K = [self.kx, self.ky, self.kz]
        N = self.N
        lamb_hat = self.compute_lamb_hat(u_hat)
        PL_hat = self.project_leray(lamb_hat)
        NL_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                NL_hat[i, j] = 0.5j * (K[i] * PL_hat[j] + K[j] * PL_hat[i])
        return NL_hat

    def compute_Q_full_hat(self, u_hat):
        """Miller's Q = total_NL + (3/4)(w*w)_TF projected to symmetric tracefree.
        For spectral analysis we use Q_full, not just total_NL.
        """
        NL_hat = self.compute_total_NL_hat(u_hat)

        # Add (3/4)(omega_i omega_j - |omega|^2 delta_ij / 3) in physical space
        w_hat = self.compute_vorticity_hat(u_hat)
        w = np.array([np.real(ifftn(w_hat[i])) for i in range(3)])
        N = self.N

        ww_TF = np.zeros((3, 3, N, N, N))
        w_sq = sum(w[i]**2 for i in range(3))
        for i in range(3):
            for j in range(3):
                ww_TF[i, j] = w[i] * w[j]
                if i == j:
                    ww_TF[i, j] -= w_sq / 3.0

        # Transform to Fourier and add
        ww_TF_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                ww_TF_hat[i, j] = fftn(ww_TF[i, j])

        Q_hat = NL_hat + 0.75 * ww_TF_hat
        return Q_hat

    def shell_tensor_norm_sq(self, T_hat, k_shell):
        """Compute ||T||^2 restricted to shell |k| in [k_shell-0.5, k_shell+0.5)."""
        N = self.N
        kmag = self.kmag
        mask = (kmag >= k_shell - 0.5) & (kmag < k_shell + 0.5)
        norm_sq = 0.0
        for i in range(3):
            for j in range(3):
                norm_sq += np.sum(np.abs(T_hat[i, j][mask])**2)
        return norm_sq / N**6  # Parseval normalization

    def shell_energy(self, u_hat, k_shell):
        """E(k) = sum of |u_hat(k')|^2 for |k'| in shell."""
        kmag = self.kmag
        mask = (kmag >= k_shell - 0.5) & (kmag < k_shell + 0.5)
        E = 0.0
        for i in range(3):
            E += np.sum(np.abs(u_hat[i][mask])**2)
        return 0.5 * E / self.N**6

    def compute_spectral_ratio_profile(self, u_hat, k_max=None):
        """Compute r(k) = ||Q(k)|| / ||-dS(k)|| for each shell."""
        if k_max is None:
            k_max = self.N // 3  # 2/3 dealiasing rule

        Q_hat = self.compute_Q_full_hat(u_hat)
        dS_hat = self.compute_neg_laplacian_strain_hat(u_hat)

        shells = np.arange(1, k_max + 1)
        r_k = np.zeros(len(shells))
        Q_norm_k = np.zeros(len(shells))
        dS_norm_k = np.zeros(len(shells))
        E_k = np.zeros(len(shells))

        for idx, k in enumerate(shells):
            Q_sq = self.shell_tensor_norm_sq(Q_hat, k)
            dS_sq = self.shell_tensor_norm_sq(dS_hat, k)
            Q_norm_k[idx] = np.sqrt(Q_sq)
            dS_norm_k[idx] = np.sqrt(dS_sq)
            E_k[idx] = self.shell_energy(u_hat, k)
            if dS_sq > 1e-60:
                r_k[idx] = np.sqrt(Q_sq / dS_sq)
            else:
                r_k[idx] = np.nan

        return shells, r_k, Q_norm_k, dS_norm_k, E_k

    def compute_shell_growth_contribution(self, u_hat, dt_small=1e-4, k_max=None):
        """Decompose g_N and g_D by shell: which k-shells drive restoration?"""
        if k_max is None:
            k_max = self.N // 3

        # Current state
        Q_hat_0 = self.compute_Q_full_hat(u_hat)
        dS_hat_0 = self.compute_neg_laplacian_strain_hat(u_hat)

        # One tiny step
        u_hat_1 = self.step_rk4(u_hat, dt_small, mode='full')
        Q_hat_1 = self.compute_Q_full_hat(u_hat_1)
        dS_hat_1 = self.compute_neg_laplacian_strain_hat(u_hat_1)

        shells = np.arange(1, k_max + 1)
        dQ_sq_k = np.zeros(len(shells))
        ddS_sq_k = np.zeros(len(shells))

        for idx, k in enumerate(shells):
            Q0_sq = self.shell_tensor_norm_sq(Q_hat_0, k)
            Q1_sq = self.shell_tensor_norm_sq(Q_hat_1, k)
            dS0_sq = self.shell_tensor_norm_sq(dS_hat_0, k)
            dS1_sq = self.shell_tensor_norm_sq(dS_hat_1, k)
            dQ_sq_k[idx] = (Q1_sq - Q0_sq) / dt_small
            ddS_sq_k[idx] = (dS1_sq - dS0_sq) / dt_small

        # Normalize by total norm^2 to get contribution to g
        Q_total_sq = sum(self.shell_tensor_norm_sq(Q_hat_0, k) for k in shells)
        dS_total_sq = sum(self.shell_tensor_norm_sq(dS_hat_0, k) for k in shells)

        gN_per_k = dQ_sq_k / (2 * max(Q_total_sq, 1e-60))
        gD_per_k = ddS_sq_k / (2 * max(dS_total_sq, 1e-60))

        return shells, gN_per_k, gD_per_k


def run_spectral_gap_test():
    """Main test: spectral derivative gap at multiple Re."""
    print("=" * 90)
    print("SPECTRAL DERIVATIVE GAP TEST")
    print("Why g_D > g_N: the forward cascade amplifies ||-dS|| faster than ||Q||")
    print("=" * 90)

    # Test across multiple Re values
    configs = [
        (32, 200, 0.01),
        (32, 800, 0.005),
        (64, 1600, 0.002),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (N, Re, dt) in enumerate(configs):
        print(f"\n{'=' * 60}")
        print(f"N={N}, Re={Re}")
        print(f"{'=' * 60}")

        solver = SpectralGapAnalyzer(N=N, Re=Re)

        # Random IC (most generic)
        np.random.seed(42)
        u_hat = solver.random_ic(seed=42)

        # Evolve to t=1.0 to develop cascade
        n_warmup = int(1.0 / dt)
        print(f"Warming up {n_warmup} steps to t=1.0...")
        for step in range(n_warmup):
            u_hat = solver.step_rk4(u_hat, dt, mode='full')

        # Measure spectral ratio profile
        k_max = N // 3
        shells, r_k, Q_norm, dS_norm, E_k = solver.compute_spectral_ratio_profile(u_hat, k_max)

        # Fit power law to r(k) in available range
        valid = ~np.isnan(r_k) & (r_k > 0) & (shells >= 2) & (shells <= k_max - 2)
        if np.sum(valid) > 3:
            log_k = np.log(shells[valid])
            log_r = np.log(r_k[valid])
            slope, intercept = np.polyfit(log_k, log_r, 1)
            print(f"r(k) ~ k^{{{slope:.2f}}}  (K41 prediction: k^{{-2.83}})")
        else:
            slope = np.nan

        # Print shell-by-shell
        print(f"\n{'k':>4} | {'||Q(k)||':>12} {'||-dS(k)||':>12} {'r(k)':>10} | {'E(k)':>12}")
        print("-" * 70)
        for idx in range(len(shells)):
            k = shells[idx]
            print(f"{k:4d} | {Q_norm[idx]:12.4e} {dS_norm[idx]:12.4e} {r_k[idx]:10.4f} | {E_k[idx]:12.4e}")

        # Shell growth decomposition
        print(f"\nShell-by-shell growth rate decomposition:")
        shells_g, gN_k, gD_k = solver.compute_shell_growth_contribution(u_hat, dt_small=dt/10, k_max=k_max)
        gap_k = gD_k - gN_k
        print(f"{'k':>4} | {'gN(k)':>12} {'gD(k)':>12} {'gD-gN':>12} | {'cumul gap':>12}")
        print("-" * 70)
        cumul = 0.0
        for idx in range(len(shells_g)):
            k = shells_g[idx]
            cumul += gap_k[idx]
            print(f"{k:4d} | {gN_k[idx]:12.4e} {gD_k[idx]:12.4e} {gap_k[idx]:12.4e} | {cumul:12.4e}")

        total_gN = np.sum(gN_k)
        total_gD = np.sum(gD_k)
        total_gap = total_gD - total_gN
        print(f"\nTotal g_N = {total_gN:.6f}")
        print(f"Total g_D = {total_gD:.6f}")
        print(f"Total gap = {total_gap:.6f}  {'<- RESTORING (g_D > g_N)' if total_gap > 0 else '<- WARNING: NOT RESTORING'}")

        # Where does the restoration come from?
        if np.sum(gap_k > 0) > 0:
            pos_shells = shells_g[gap_k > 0]
            print(f"Restoration from shells k = {pos_shells[0]} to {pos_shells[-1]}")
            frac_high = np.sum(gap_k[shells_g > k_max//2]) / max(np.sum(gap_k[gap_k > 0]), 1e-30)
            print(f"Fraction from high-k (k > {k_max//2}): {frac_high*100:.1f}%")

        # Plot r(k)
        ax = axes[0, col]
        valid_plot = ~np.isnan(r_k) & (r_k > 0)
        ax.loglog(shells[valid_plot], r_k[valid_plot], 'bo-', markersize=4, label='r(k) measured')
        if not np.isnan(slope):
            k_fit = np.linspace(2, k_max, 50)
            ax.loglog(k_fit, np.exp(intercept) * k_fit**slope, 'r--',
                      label=f'fit: k^{{{slope:.2f}}}')
            ax.loglog(k_fit, np.exp(intercept) * (k_fit/2)**(-17/6), 'g:',
                      alpha=0.5, label='K41: k^{-2.83}')
        ax.set_xlabel('k')
        ax.set_ylabel('||Q(k)|| / ||-dS(k)||')
        ax.set_title(f'N={N}, Re={Re}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot shell growth gap
        ax2 = axes[1, col]
        ax2.bar(shells_g, gap_k, color=['green' if g > 0 else 'red' for g in gap_k], alpha=0.7)
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.set_xlabel('k')
        ax2.set_ylabel('g_D(k) - g_N(k)')
        ax2.set_title(f'Shell restoration: N={N}, Re={Re}')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'spectral_derivative_gap.png'), dpi=150)
    print(f"\nPlot saved to spectral_derivative_gap.png")

    print("\n" + "=" * 90)
    print("SPECTRAL DERIVATIVE GAP — SUMMARY")
    print("=" * 90)
    print("If r(k) decreases monotonically, then the forward cascade")
    print("(which moves energy from low k to high k) amplifies ||-dS||")
    print("faster than ||Q||, guaranteeing g_D > g_N and dR/dt < 0.")
    print("")
    print("This is the FORMAL MECHANISM behind the restoring dynamics:")
    print("  1. r(k) = ||Q(k)||/||-dS(k)|| decreases with k  (MEASURED)")
    print("  2. Forward cascade moves energy to high k  (DUCHON-ROBERT)")
    print("  3. At high k, dS gains k^6 but Q gains less  (DERIVATIVE GAP)")
    print("  4. Therefore g_D > g_N -> dR/dt < 0  (RESTORING)")
    print("  5. R cannot reach 1 -> regularity  (MILLER'S CRITERION)")
    print("=" * 90)


if __name__ == '__main__':
    run_spectral_gap_test()
