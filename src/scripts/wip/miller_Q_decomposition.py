# -*- coding: utf-8 -*-
"""
MILLER Q PERTURBATION -- HELICAL DECOMPOSITION
===============================================
Based on Miller 2024 (arXiv:2407.02691): "Strain-Vorticity Interaction Model"

Key identity (Miller):
  Full NS strain evolution = mu=0 model (globally regular) + Q perturbation

  Q = P_st( (u.grad)S + S^2 + 3/4w*w )

  where P_st is the symmetric trace-free projection.

Blowup criterion (Miller, Thm 1.1):
  If blowup occurs at time T*, then  limsup ||Q|| / ||-dS|| >= 1

What we test:
  1. Decompose Q into helical sectors: Q = Q_same + Q_cross
     - Q_same: contributions from same-helicity interactions only
     - Q_cross: contributions from cross-helicity interactions
  2. Track ||Q|| / ||-dS|| during DNS evolution
  3. Check: does alpha ~ 1-ln2 suppress Q_cross, leaving only Q_same?
  4. Check: can ||Q_same|| / ||-dS|| reach 1 alone?

Connection to our framework:
  - Miller: <-dS, w*w> = 0 (orthogonality -- vorticity part is harmless)
  - Us: cross-helical Lamb vector is 69% gradient (Leray-killed)
  - If Q_cross is suppressed by alpha, only Q_same remains
  - But same-helicity NS is globally regular (Biferale-Titi 2003)
  - Circle closes: ||Q_same||/||-dS|| < 1 always? (THIS IS THE TEST)

Method: Pseudo-spectral 3D NS, N=32, Re=400, TG + imbalanced ICs
"""

import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock
import sys
import os

# Import base solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS


class MillerAnalyzer(SpectralNS):
    """Extends SpectralNS with Miller's Q perturbation diagnostics."""

    def compute_strain_hat(self, u_hat):
        """Compute strain tensor S_ij in Fourier space.
        S_hat_ij = (1/2)(ik_j u_hat_i + ik_i u_hat_j)
        """
        K = [self.kx, self.ky, self.kz]
        S_hat = np.zeros((3, 3) + u_hat.shape[1:], dtype=complex)
        for i in range(3):
            for j in range(3):
                S_hat[i, j] = 0.5 * (1j * K[j] * u_hat[i] + 1j * K[i] * u_hat[j])
        return S_hat

    def compute_strain_physical(self, u_hat):
        """Compute strain tensor in physical space."""
        S_hat = self.compute_strain_hat(u_hat)
        S = np.zeros((3, 3, self.N, self.N, self.N))
        for i in range(3):
            for j in range(3):
                S[i, j] = np.real(ifftn(S_hat[i, j]))
        return S

    def compute_neg_laplacian_strain_norm(self, u_hat):
        """Compute ||-dS||_L^2 = sqrt( sum_{ij} <|k^2 S_hat_ij|^2> ).
        Uses Parseval: <|f|^2> = (1/N^3) sum |f_hat|^2 / N^3.
        """
        S_hat = self.compute_strain_hat(u_hat)
        N = self.N
        norm_sq = 0.0
        for i in range(3):
            for j in range(3):
                lap_S_ij = self.k2 * S_hat[i, j]
                norm_sq += np.sum(np.abs(lap_S_ij)**2)
        return np.sqrt(norm_sq / N**6)

    def compute_advection_of_strain(self, u_hat):
        """Compute (u.grad)S in physical space (dealiased).
        [(u.grad)S]_ij = u_k dS_ij/dx_k
        """
        N = self.N
        K = [self.kx, self.ky, self.kz]
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        S_hat = self.compute_strain_hat(u_hat)

        result = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                for k_idx in range(3):
                    dS_dx = np.real(ifftn(1j * K[k_idx] * S_hat[i, j]))
                    result[i, j] += u[k_idx] * dS_dx
        return result

    def compute_S_squared(self, u_hat):
        """Compute S^2 = S_ik S_kj in physical space."""
        N = self.N
        S = self.compute_strain_physical(u_hat)
        S2 = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                for k_idx in range(3):
                    S2[i, j] += S[i, k_idx] * S[k_idx, j]
        return S2

    def compute_omega_tensor(self, u_hat):
        """Compute w*w tensor: (w*w)_ij = omega_i omega_j."""
        N = self.N
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        OO = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                OO[i, j] = omega[i] * omega[j]
        return OO

    def project_symmetric_tracefree(self, T):
        """Project a 3x3 tensor field to symmetric trace-free part.
        P_st(T)_ij = (1/2)(T_ij + T_ji) - (1/3) delta_ij tr(T)
        """
        T_sym = np.zeros_like(T)
        for i in range(3):
            for j in range(3):
                T_sym[i, j] = 0.5 * (T[i, j] + T[j, i])
        tr = T_sym[0, 0] + T_sym[1, 1] + T_sym[2, 2]
        for i in range(3):
            T_sym[i, i] -= tr / 3.0
        return T_sym

    def compute_Q_perturbation(self, u_hat):
        """Compute Miller's Q = P_st( (u.grad)S + S^2 + 3/4w*w ).
        Returns Q as a 3x3 physical-space tensor field (dealiased).
        """
        advS = self.compute_advection_of_strain(u_hat)
        S2 = self.compute_S_squared(u_hat)
        OO = self.compute_omega_tensor(u_hat)

        T = advS + S2 + 0.75 * OO
        Q = self.project_symmetric_tracefree(T)

        # Dealias
        N = self.N
        for i in range(3):
            for j in range(3):
                Q_hat_ij = fftn(Q[i, j]) * self.dealias_mask
                Q[i, j] = np.real(ifftn(Q_hat_ij))
        return Q

    def tensor_L2_norm(self, T):
        """L^2 norm of tensor field: ||T|| = sqrt( sum_ij <T_ij^2> )."""
        norm_sq = 0.0
        for i in range(3):
            for j in range(3):
                norm_sq += np.mean(T[i, j]**2)
        return np.sqrt(norm_sq)

    def compute_Q_helical_decomposition(self, u_hat):
        """Decompose Q into same-helicity and cross-helicity contributions.

        Q_same = Q(u+) + Q(u-)   (each sector computed independently)
        Q_cross = Q(u) - Q_same  (cross-sector interactions)

        Approximate because Q is nonlinear, but measures how much of Q
        comes from within-sector vs cross-sector dynamics.
        """
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)

        Q_full = self.compute_Q_perturbation(u_hat)
        Q_plus = self.compute_Q_perturbation(u_hat_plus)
        Q_minus = self.compute_Q_perturbation(u_hat_minus)

        Q_same = Q_plus + Q_minus
        Q_cross = Q_full - Q_same

        return Q_full, Q_same, Q_cross

    def verify_miller_orthogonality(self, u_hat):
        """Verify Miller's identity: <-dS, w*w> = 0.
        Returns the inner product (should be ~0 to machine precision).
        """
        S_hat = self.compute_strain_hat(u_hat)
        N = self.N

        # -Delta_S in physical space
        neg_lap_S = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                neg_lap_S[i, j] = np.real(ifftn(self.k2 * S_hat[i, j]))

        OO = self.compute_omega_tensor(u_hat)

        # Inner product: sum_ij <(-Delta_S)_ij * (w*w)_ij>
        inner = 0.0
        for i in range(3):
            for j in range(3):
                inner += np.mean(neg_lap_S[i, j] * OO[i, j])
        return inner


def run_miller_analysis(N=32, Re=400, dt=0.005, T=3.0, report_every=20):
    """Run DNS and track Miller's Q decomposition."""
    print("=" * 78)
    print("MILLER Q PERTURBATION -- HELICAL DECOMPOSITION")
    print(f"Based on Miller 2024 (arXiv:2407.02691)")
    print("=" * 78)
    print(f"N={N}, Re={Re}, dt={dt}, T={T}")
    print()
    print("Measuring: ||Q||/||-dS||, ||Q_same||/||-dS||, ||Q_cross||/||-dS||")
    print("Blowup requires ||Q||/||-dS|| >= 1 (Miller Thm 1.1)")
    print()

    solver = MillerAnalyzer(N=N, Re=Re)
    n_steps = int(T / dt)

    # Initial conditions
    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Imbalanced (80/20)': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
        'Imbalanced (95/5)': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.95),
    }

    all_results = {}

    for ic_name, u_hat_ic in ics.items():
        hp, hm = solver.helical_energy_fractions(u_hat_ic)
        print(f"\n{'='*70}")
        print(f"IC: {ic_name}  |  h+: {hp:.3f}  h-: {hm:.3f}")
        print(f"{'='*70}")

        # Verify Miller orthogonality at t=0
        ortho = solver.verify_miller_orthogonality(u_hat_ic)
        print(f"Miller orthogonality check: <-dS, w*w> = {ortho:.2e} (should be ~0)")
        print()

        print(f"{'t':>5} | {'||Q||/||-dS||':>14} {'||Qs||/||-dS||':>15} "
              f"{'||Qc||/||-dS||':>15} {'Qs/Q':>6} {'Qc/Q':>6} "
              f"{'E':>10}")
        print("-" * 85)

        u_hat = u_hat_ic.copy()
        times = []
        ratio_Q = []
        ratio_Q_same = []
        ratio_Q_cross = []
        frac_same = []
        frac_cross = []
        energies = []
        orthogonality = []

        for step in range(n_steps + 1):
            t = step * dt

            if step % report_every == 0:
                lap_S_norm = solver.compute_neg_laplacian_strain_norm(u_hat)
                Q_full, Q_same, Q_cross = solver.compute_Q_helical_decomposition(u_hat)

                nQ = solver.tensor_L2_norm(Q_full)
                nQs = solver.tensor_L2_norm(Q_same)
                nQc = solver.tensor_L2_norm(Q_cross)
                E = solver.compute_total_energy(u_hat)
                ortho = solver.verify_miller_orthogonality(u_hat)

                safe_lap = max(lap_S_norm, 1e-30)
                safe_Q = max(nQ, 1e-30)

                r_Q = nQ / safe_lap
                r_Qs = nQs / safe_lap
                r_Qc = nQc / safe_lap
                fs = nQs / safe_Q
                fc = nQc / safe_Q

                times.append(t)
                ratio_Q.append(r_Q)
                ratio_Q_same.append(r_Qs)
                ratio_Q_cross.append(r_Qc)
                frac_same.append(fs)
                frac_cross.append(fc)
                energies.append(E)
                orthogonality.append(ortho)

                if step % (report_every * 5) == 0:
                    print(f"{t:5.2f} | {r_Q:14.6f} {r_Qs:15.6f} "
                          f"{r_Qc:15.6f} {fs:6.3f} {fc:6.3f} "
                          f"{E:10.4e}")

            if step < n_steps:
                u_hat = solver.step_rk4(u_hat, dt, mode='full')

        all_results[ic_name] = {
            'times': np.array(times),
            'ratio_Q': np.array(ratio_Q),
            'ratio_Q_same': np.array(ratio_Q_same),
            'ratio_Q_cross': np.array(ratio_Q_cross),
            'frac_same': np.array(frac_same),
            'frac_cross': np.array(frac_cross),
            'energies': np.array(energies),
            'orthogonality': np.array(orthogonality),
        }

    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n\n" + "=" * 78)
    print("ANALYSIS")
    print("=" * 78)

    for ic_name, r in all_results.items():
        valid = r['times'] > 0.1
        if not np.any(valid):
            continue

        max_Q = np.max(r['ratio_Q'][valid])
        max_Qs = np.max(r['ratio_Q_same'][valid])
        max_Qc = np.max(r['ratio_Q_cross'][valid])
        mean_fs = np.mean(r['frac_same'][valid])
        mean_fc = np.mean(r['frac_cross'][valid])
        max_ortho = np.max(np.abs(r['orthogonality'][valid]))

        print(f"\n--- {ic_name} ---")
        print(f"  max ||Q||/||-dS||       = {max_Q:.6f}")
        print(f"  max ||Q_same||/||-dS||  = {max_Qs:.6f}")
        print(f"  max ||Q_cross||/||-dS|| = {max_Qc:.6f}")
        print(f"  mean Q_same fraction    = {mean_fs:.4f}")
        print(f"  mean Q_cross fraction   = {mean_fc:.4f}")
        print(f"  max |<-dS, w*w>|      = {max_ortho:.2e} (Miller orthogonality)")

        if max_Q < 1.0:
            print(f"  -> ||Q||/||-dS|| < 1: blowup EXCLUDED (Miller criterion)")
            print(f"     gap to blowup: {1.0 - max_Q:.4f}")
        if max_Qs < 1.0:
            gap = 1.0 - max_Qs
            print(f"  -> ||Q_same||/||-dS|| < 1 (gap = {gap:.4f})")
            print(f"    If Q_cross suppressed by alpha -> blowup excluded by same-helicity alone")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print()
    print("Miller 2024 framework:")
    print("  - mu=0 model (keeps only w*w) is globally regular")
    print("  - <-dS, w*w> = 0: vorticity self-interaction is orthogonal to strain Laplacian")
    print("  - Blowup requires ||Q||/||-dS|| >= 1 where Q = P_st((u.grad)S + S^2 + 3/4w*w)")
    print()
    print("Our helical decomposition of Q:")
    for ic_name, r in all_results.items():
        valid = r['times'] > 0.1
        if not np.any(valid):
            continue
        max_Qs = np.max(r['ratio_Q_same'][valid])
        max_Qc = np.max(r['ratio_Q_cross'][valid])
        max_Q = np.max(r['ratio_Q'][valid])
        mean_fc = np.mean(r['frac_cross'][valid])
        print(f"  {ic_name}: Q_cross/Q = {mean_fc*100:.1f}%, "
              f"||Q||/||-dS|| max = {max_Q:.4f}, "
              f"||Q_same||/||-dS|| max = {max_Qs:.4f}")

    print()
    print("Potential circle-closing argument:")
    print("  1. Miller: blowup requires ||Q||/||-dS|| >= 1")
    print("  2. Us: Q_cross suppressed by alpha ~ 1-ln2 (Leray + helical geometry)")
    print("  3. Biferale-Titi: same-helicity NS is globally regular")
    print("  4. If ||Q_same||/||-dS|| < 1 universally -> no blowup -> regularity")
    print()
    print("WARNING: This is at moderate Re. Higher Re needed to check persistence.")

    # ============================================================
    # PLOTS
    # ============================================================
    print("\nGenerating plots...")

    n_ics = len(all_results)
    fig, axes = plt.subplots(n_ics, 3, figsize=(18, 5 * n_ics))
    if n_ics == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Miller Q Perturbation -- Helical Decomposition\n"
                 f"N={N}, Re={Re}, arXiv:2407.02691",
                 fontsize=14, fontweight='bold')

    for row, (ic_name, r) in enumerate(all_results.items()):
        t = r['times']

        # Panel 1: blowup ratios
        ax = axes[row, 0]
        ax.plot(t, r['ratio_Q'], 'k-', linewidth=2, label='||Q|| / ||-dS||')
        ax.plot(t, r['ratio_Q_same'], 'b--', linewidth=1.5, label='||Q_same|| / ||-dS||')
        ax.plot(t, r['ratio_Q_cross'], 'r:', linewidth=1.5, label='||Q_cross|| / ||-dS||')
        ax.axhline(y=1.0, color='k', linestyle='-.', linewidth=1, alpha=0.5,
                    label='Blowup threshold')
        ax.set_xlabel('t')
        ax.set_ylabel('Ratio')
        ax.set_title(f'{ic_name}: Miller blowup ratio')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: same vs cross fraction
        ax = axes[row, 1]
        ax.fill_between(t, 0, r['frac_same'], alpha=0.4, color='blue', label='Same-helicity')
        ax.fill_between(t, r['frac_same'], r['frac_same'] + r['frac_cross'],
                        alpha=0.4, color='red', label='Cross-helicity')
        ax.set_xlabel('t')
        ax.set_ylabel('Fraction of ||Q||')
        ax.set_title(f'{ic_name}: Q helical composition')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(2.0, np.max(r['frac_same'] + r['frac_cross']) * 1.1))

        # Panel 3: Miller orthogonality check
        ax = axes[row, 2]
        ax.semilogy(t, np.abs(r['orthogonality']), 'g-', linewidth=1.5)
        ax.set_xlabel('t')
        ax.set_ylabel('|<-dS, w*w>|')
        ax.set_title(f'{ic_name}: Miller orthogonality (should be ~0)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'miller_Q_decomposition.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {outpath}")

    return all_results


if __name__ == "__main__":
    wall_start = clock.time()
    results = run_miller_analysis(N=32, Re=400, dt=0.005, T=3.0, report_every=20)
    wall_time = clock.time() - wall_start
    print(f"\nTotal wall time: {wall_time:.1f}s")
