"""
R(t) DYNAMICS STUDY — Time evolution of the enstrophy production ratio
=====================================================================

Key question: How does R(t) = ||total_NL(t)|| / ||-ΔS(t)|| evolve in time?

If R(t) has a RESTORING mechanism (dR/dt < 0 when R is large), then R cannot
reach 1 and blowup is impossible. This would close Gap 1 (a priori bound).

We measure:
  1. R(t) = ||total_NL|| / ||-ΔS||  (the enstrophy production ratio)
  2. dR/dt computed from the time series
  3. γ(t) = ||∇²u||² / (||∇u|| · ||∇³u||)  (G-N tightness ratio, ≤ 1 always)
     γ < 1 means G-N is loose → actual R much smaller than Sobolev bound
  4. Phase portrait: R vs dR/dt — look for restoring dynamics

If the DNS shows dR/dt < 0 when R > R* for some threshold R*, and γ stays
small (G-N loose), then the combination closes the regularity argument:
  - R is bounded by a restoring force
  - The Sobolev bound C√Ω is never approached because γ << 1

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


class RDynamicsAnalyzer(SpectralNS):
    """Extended SpectralNS with R(t) dynamics measurement."""

    def compute_strain_hat(self, u_hat):
        """S_ij(k) = (i/2)(k_i û_j + k_j û_i)."""
        K = [self.kx, self.ky, self.kz]
        N = self.N
        S_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                S_hat[i, j] = 0.5j * (K[i] * u_hat[j] + K[j] * u_hat[i])
        return S_hat

    def compute_neg_laplacian_strain_hat(self, u_hat):
        """-ΔS in Fourier: k² S_ij(k)."""
        S_hat = self.compute_strain_hat(u_hat)
        return self.k2[np.newaxis, np.newaxis] * S_hat

    def compute_total_NL_hat(self, u_hat):
        """total_NL = sym-grad(P(L)) where L = u × ω.
        This is the nonlinear strain production tensor.
        """
        K = [self.kx, self.ky, self.kz]
        N = self.N
        lamb_hat = self.compute_lamb_hat(u_hat)
        PL_hat = self.project_leray(lamb_hat)
        NL_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i in range(3):
            for j in range(3):
                NL_hat[i, j] = 0.5j * (K[i] * PL_hat[j] + K[j] * PL_hat[i])
        return NL_hat

    def l2_norm_tensor_hat(self, T_hat):
        """||T||² = (1/N⁶) Σ_{i,j,k} |T_{ij}(k)|²."""
        N = self.N
        norm_sq = 0.0
        for i in range(T_hat.shape[0]):
            for j in range(T_hat.shape[1]):
                norm_sq += np.sum(np.abs(T_hat[i, j])**2)
        return np.sqrt(norm_sq / N**6)

    def compute_sobolev_norms(self, u_hat):
        """Compute ||∇u||, ||∇²u||, ||∇³u|| for G-N tightness ratio.

        ||∇^n u||² = Σ_k |k|^{2n} |û(k)|²
        """
        N = self.N
        u2 = sum(np.abs(u_hat[i])**2 for i in range(3))

        norm_grad_sq = np.sum(self.k2 * u2) / N**6       # ||∇u||²
        norm_lap_sq = np.sum(self.k2**2 * u2) / N**6      # ||∇²u||²
        norm_trilap_sq = np.sum(self.k2**3 * u2) / N**6   # ||∇³u||²

        return (np.sqrt(norm_grad_sq),
                np.sqrt(norm_lap_sq),
                np.sqrt(norm_trilap_sq))

    def compute_inner_product_hat(self, A_hat, B_hat):
        """⟨A, B⟩ = (1/N⁶) Re Σ_{i,j,k} A_{ij}(k) B_{ij}(k)*."""
        N = self.N
        ip = 0.0
        for i in range(A_hat.shape[0]):
            for j in range(A_hat.shape[1]):
                ip += np.sum(np.real(A_hat[i, j] * np.conj(B_hat[i, j])))
        return ip / N**6

    def measure_R_state(self, u_hat):
        """Measure R(t) and all related quantities at current state.

        Returns dict with:
            R: ||total_NL|| / ||-ΔS||
            norm_NL: ||total_NL||
            norm_neglapS: ||-ΔS||
            gamma: ||∇²u||² / (||∇u|| · ||∇³u||)  (G-N tightness, ≤ 1)
            enstrophy_production: ⟨total_NL, -ΔS⟩  (= dΩ/dt + ν||-ΔS||²)
            enstrophy: ||ω||² / 2
            energy: ||u||² / 2
        """
        N = self.N

        # Enstrophy production ratio
        NL_hat = self.compute_total_NL_hat(u_hat)
        neglapS_hat = self.compute_neg_laplacian_strain_hat(u_hat)

        norm_NL = self.l2_norm_tensor_hat(NL_hat)
        norm_neglapS = self.l2_norm_tensor_hat(neglapS_hat)

        R = norm_NL / max(norm_neglapS, 1e-30)

        # Inner product ⟨total_NL, -ΔS⟩ = enstrophy production from nonlinearity
        ip = self.compute_inner_product_hat(NL_hat, neglapS_hat)

        # Directional cosine: cos(θ) = ⟨NL, -ΔS⟩ / (||NL|| · ||-ΔS||)
        cos_theta = ip / max(norm_NL * norm_neglapS, 1e-30)

        # G-N tightness ratio: γ = ||∇²u||² / (||∇u|| · ||∇³u||)
        norm_grad, norm_lap, norm_trilap = self.compute_sobolev_norms(u_hat)
        gamma = norm_lap**2 / max(norm_grad * norm_trilap, 1e-30)

        # Energy and enstrophy
        u2 = sum(np.abs(u_hat[i])**2 for i in range(3))
        energy = 0.5 * np.sum(u2) / N**6
        enstrophy = 0.5 * np.sum(self.k2 * u2) / N**6

        # Helical fractions
        f_plus, f_minus = self.helical_energy_fractions(u_hat)

        return {
            'R': R,
            'norm_NL': norm_NL,
            'norm_neglapS': norm_neglapS,
            'ip_NL_neglapS': ip,
            'cos_theta': cos_theta,
            'gamma': gamma,
            'norm_grad': norm_grad,
            'norm_lap': norm_lap,
            'norm_trilap': norm_trilap,
            'energy': energy,
            'enstrophy': enstrophy,
            'f_plus': f_plus,
            'f_minus': f_minus,
        }


def run_R_dynamics(Re, N=32, n_steps=600, ic_type='taylor_green',
                   measure_every=1, label=None):
    """Run DNS and measure R(t) at high temporal resolution.

    Returns time array and list of measurement dicts.
    """
    if label is None:
        label = f"Re={Re}, {ic_type}, N={N}"
    print(f"\n{'='*70}")
    print(f"R(t) DYNAMICS: {label}")
    print(f"{'='*70}")

    solver = RDynamicsAnalyzer(N=N, Re=Re)

    if ic_type == 'taylor_green':
        u_hat = solver.taylor_green_ic()
    elif ic_type == 'imbalanced_80_20':
        u_hat = solver.narrowband_imbalanced_ic(seed=42, h_plus_frac=0.8)
    elif ic_type == 'random':
        u_hat = solver.random_ic(seed=42)
    else:
        u_hat = solver.taylor_green_ic()

    dt = 0.5 / max(N, Re**0.5)

    times = []
    measurements = []
    t = 0.0

    # Initial measurement
    m = solver.measure_R_state(u_hat)
    times.append(t)
    measurements.append(m)

    for step in range(1, n_steps + 1):
        u_hat = solver.step_rk4(u_hat, dt, mode='full')
        t += dt

        if step % measure_every == 0:
            m = solver.measure_R_state(u_hat)
            times.append(t)
            measurements.append(m)

            if step % 100 == 0:
                print(f"  step {step:4d}, t={t:.4f}: R={m['R']:.6f}, "
                      f"γ={m['gamma']:.4f}, cos θ={m['cos_theta']:.4f}, "
                      f"Ω={m['enstrophy']:.4e}, E={m['energy']:.4e}")

    return np.array(times), measurements


def compute_dRdt(times, measurements):
    """Compute dR/dt from time series using central differences."""
    R_arr = np.array([m['R'] for m in measurements])
    dRdt = np.zeros_like(R_arr)

    # Central difference (2nd order) for interior points
    for i in range(1, len(R_arr) - 1):
        dRdt[i] = (R_arr[i+1] - R_arr[i-1]) / (times[i+1] - times[i-1])

    # Forward/backward at endpoints
    if len(R_arr) > 1:
        dRdt[0] = (R_arr[1] - R_arr[0]) / (times[1] - times[0])
        dRdt[-1] = (R_arr[-1] - R_arr[-2]) / (times[-1] - times[-2])

    return R_arr, dRdt


def analyze_restoring(R_arr, dRdt):
    """Analyze whether dR/dt shows restoring dynamics (negative when R large)."""
    # Bin R values and compute mean dR/dt in each bin
    n_bins = 20
    R_min, R_max = R_arr.min(), R_arr.max()
    if R_max - R_min < 1e-10:
        print("  R essentially constant — no dynamics to analyze.")
        return None

    bin_edges = np.linspace(R_min, R_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_mean_dRdt = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for i in range(len(R_arr)):
        idx = np.searchsorted(bin_edges[1:], R_arr[i])
        if idx >= n_bins:
            idx = n_bins - 1
        bin_mean_dRdt[idx] += dRdt[i]
        bin_count[idx] += 1

    valid = bin_count > 2
    bin_mean_dRdt[valid] /= bin_count[valid]

    # Check: is mean(dR/dt) < 0 in the upper half of R values?
    upper_mask = valid & (bin_centers > 0.5 * (R_min + R_max))
    if np.any(upper_mask):
        upper_mean = np.mean(bin_mean_dRdt[upper_mask])
        print(f"  Mean dR/dt in upper R half: {upper_mean:.6f}")
        if upper_mean < 0:
            print(f"  → RESTORING: dR/dt < 0 when R is large!")
        else:
            print(f"  → No clear restoring force detected.")
    else:
        upper_mean = 0.0
        print(f"  Not enough data in upper R range.")

    return {
        'bin_centers': bin_centers,
        'bin_mean_dRdt': bin_mean_dRdt,
        'bin_count': bin_count,
        'valid': valid,
        'upper_mean': upper_mean,
    }


def plot_results(all_results, filename='r_dynamics_study.png'):
    """Plot R(t) dynamics results."""
    n_runs = len(all_results)
    fig, axes = plt.subplots(n_runs, 4, figsize=(20, 5 * n_runs))
    if n_runs == 1:
        axes = axes[np.newaxis, :]

    for idx, (label, times, meas, R_arr, dRdt, restoring) in enumerate(all_results):
        gamma_arr = np.array([m['gamma'] for m in meas])
        cos_arr = np.array([m['cos_theta'] for m in meas])
        enstrophy_arr = np.array([m['enstrophy'] for m in meas])

        # Panel 1: R(t) vs time
        ax = axes[idx, 0]
        ax.plot(times, R_arr, 'b-', linewidth=0.8)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Blowup threshold')
        ax.set_xlabel('t')
        ax.set_ylabel('R(t) = ||total_NL|| / ||-ΔS||')
        ax.set_title(f'{label}\nR(t) time series')
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

        # Panel 2: Phase portrait R vs dR/dt
        ax = axes[idx, 1]
        sc = ax.scatter(R_arr[1:-1], dRdt[1:-1], c=times[1:-1], s=2,
                       cmap='viridis', alpha=0.7)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlabel('R')
        ax.set_ylabel('dR/dt')
        ax.set_title('Phase portrait')
        plt.colorbar(sc, ax=ax, label='time')

        # Overlay binned average if available
        if restoring is not None:
            v = restoring['valid']
            ax.plot(restoring['bin_centers'][v], restoring['bin_mean_dRdt'][v],
                   'r-o', markersize=4, linewidth=2, label='binned mean')
            ax.legend(fontsize=8)

        # Panel 3: γ(t) — G-N tightness
        ax = axes[idx, 2]
        ax.plot(times, gamma_arr, 'g-', linewidth=0.8)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='G-N equality')
        ax.set_xlabel('t')
        ax.set_ylabel('γ = ||∇²u||² / (||∇u|| · ||∇³u||)')
        ax.set_title(f'G-N tightness (mean={gamma_arr.mean():.3f})')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)

        # Panel 4: cos(θ) and enstrophy
        ax = axes[idx, 3]
        ax.plot(times, cos_arr, 'm-', linewidth=0.8, label='cos θ (NL vs -ΔS)')
        ax.set_xlabel('t')
        ax.set_ylabel('cos θ', color='m')
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='k', linewidth=0.3)
        ax.legend(loc='upper left', fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(times, enstrophy_arr, 'c-', linewidth=0.8, alpha=0.5)
        ax2.set_ylabel('Enstrophy Ω', color='c')
        ax.set_title('Directional alignment')

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(outpath, dpi=150)
    print(f"\n  Plot saved: {outpath}")
    plt.close()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    print("R(t) DYNAMICS STUDY — Enstrophy production ratio evolution")
    print("=" * 70)
    print()
    print("Key question: Does R(t) = ||total_NL|| / ||-ΔS|| have a restoring")
    print("mechanism that prevents it from reaching 1?")
    print()

    all_results = []

    # ---- Sweep across Re for Taylor-Green ----
    print("\n" + "#" * 70)
    print("# SECTION 1: Re sweep with Taylor-Green IC")
    print("#" * 70)

    for Re in [200, 400, 800, 1600]:
        times, meas = run_R_dynamics(
            Re=Re, N=32, n_steps=600, ic_type='taylor_green',
            measure_every=2
        )
        R_arr, dRdt = compute_dRdt(times, meas)
        restoring = analyze_restoring(R_arr, dRdt)
        label = f"TG Re={Re}"
        all_results.append((label, times, meas, R_arr, dRdt, restoring))

        # Summary
        print(f"\n  SUMMARY for {label}:")
        print(f"    max R = {R_arr.max():.6f}")
        print(f"    mean R = {R_arr.mean():.6f}")
        gamma_arr = np.array([m['gamma'] for m in meas])
        print(f"    mean γ = {gamma_arr.mean():.4f} (G-N tightness)")
        cos_arr = np.array([m['cos_theta'] for m in meas])
        print(f"    mean cos θ = {cos_arr.mean():.4f}")

    # ---- Different ICs at Re=800 ----
    print("\n" + "#" * 70)
    print("# SECTION 2: IC comparison at Re=800")
    print("#" * 70)

    for ic_type in ['imbalanced_80_20', 'random']:
        times, meas = run_R_dynamics(
            Re=800, N=32, n_steps=600, ic_type=ic_type,
            measure_every=2
        )
        R_arr, dRdt = compute_dRdt(times, meas)
        restoring = analyze_restoring(R_arr, dRdt)
        label = f"{ic_type} Re=800"
        all_results.append((label, times, meas, R_arr, dRdt, restoring))

        print(f"\n  SUMMARY for {label}:")
        print(f"    max R = {R_arr.max():.6f}")
        print(f"    mean R = {R_arr.mean():.6f}")
        gamma_arr = np.array([m['gamma'] for m in meas])
        print(f"    mean γ = {gamma_arr.mean():.4f}")

    # ---- Higher resolution check ----
    print("\n" + "#" * 70)
    print("# SECTION 3: Resolution check N=64 at Re=800")
    print("#" * 70)

    times, meas = run_R_dynamics(
        Re=800, N=64, n_steps=400, ic_type='taylor_green',
        measure_every=2
    )
    R_arr, dRdt = compute_dRdt(times, meas)
    restoring = analyze_restoring(R_arr, dRdt)
    all_results.append(("TG Re=800 N=64", times, meas, R_arr, dRdt, restoring))

    print(f"\n  SUMMARY for TG Re=800 N=64:")
    print(f"    max R = {R_arr.max():.6f}")
    gamma_arr = np.array([m['gamma'] for m in meas])
    print(f"    mean γ = {gamma_arr.mean():.4f}")

    # ---- Plot everything ----
    plot_results(all_results)

    # ---- Final synthesis ----
    print("\n" + "=" * 70)
    print("FINAL SYNTHESIS")
    print("=" * 70)
    print()

    max_R_all = max(r[3].max() for r in all_results)
    mean_gamma_all = np.mean([np.array([m['gamma'] for m in r[2]]).mean()
                             for r in all_results])
    restoring_count = sum(1 for r in all_results
                         if r[5] is not None and r[5]['upper_mean'] < 0)

    print(f"  Across all {len(all_results)} runs:")
    print(f"    Global max R = {max_R_all:.6f}  (blowup threshold = 1.0)")
    print(f"    Mean G-N tightness γ = {mean_gamma_all:.4f}")
    print(f"    Runs with restoring dynamics: {restoring_count}/{len(all_results)}")
    print()

    if max_R_all < 0.3:
        print("  RESULT: R stays FAR below 1.0 across all tested conditions.")
        print("  The enstrophy production ratio never approaches blowup threshold.")
    elif max_R_all < 1.0:
        print(f"  RESULT: R reaches {max_R_all:.4f} but stays below 1.0.")
        print("  Suggestive of safety margin but further investigation needed.")
    else:
        print(f"  WARNING: R reaches {max_R_all:.4f} — above 1.0!")
        print("  This needs immediate investigation.")

    if mean_gamma_all < 0.5:
        print(f"  G-N tightness γ ≈ {mean_gamma_all:.3f} << 1: Gagliardo-Nirenberg")
        print("  is very loose for NS solutions. The Sobolev bound C√Ω is")
        print("  pessimistic by a large factor — this is WHY R stays small.")

    if restoring_count == len(all_results):
        print("  ALL runs show restoring dynamics (dR/dt < 0 when R large).")
        print("  → R has a self-correcting mechanism preventing approach to 1.")
    elif restoring_count > 0:
        print(f"  {restoring_count}/{len(all_results)} runs show restoring dynamics.")

    print()
    print("Done.")
