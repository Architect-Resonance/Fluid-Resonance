"""
FOURIER-PHYSICAL BRIDGE EXPERIMENT
====================================
S95 — Wanderer

THE KEY QUESTION: What connects spectral phase coherence to physical-space
vortex structures? Why does per-triad alpha_E = 1/4 but actual NS gives ~0.9?

This script measures FIVE things simultaneously during NS evolution:

1. PHASE SCRAMBLE TEST
   - Keep |u_hat(k)| but randomize phases → compute alpha_E
   - Compare alpha_E(actual) vs alpha_E(random)
   - The RATIO directly measures phase coherence amplification
   - If ratio >> 1: dynamics builds dangerous coherence
   - If ratio ~ 1: per-triad bounds suffice

2. VORTEX TUBE CONDITIONING
   - Identify tubes: regions where |omega| > mean + 2*sigma
   - Compute solenoidal Lamb fraction INSIDE tubes vs OUTSIDE
   - Tests: do tubes concentrate the solenoidal forcing?

3. ANTI-TWIST PROXY
   - omega . S . omega (stretching rate) conditioned on |omega| percentile
   - omega-u alignment cos(omega, u) as proxy for Beltramization
   - Tests: do high-vorticity regions self-regulate?

4. PHASE COHERENCE DYNAMICS
   - Track coherence amplification ratio over time
   - Does it saturate? Oscillate? Grow monotonically?
   - Compare with enstrophy evolution

5. SPATIAL CORRELATION
   - Correlation between |P_sol(L)(x)| and |omega(x)|
   - Do solenoidal forcing and vorticity co-locate?
   - If yes: tubes drive themselves via coherent forcing

HONEST TEST. We report what the numbers say.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import sys
import os
import time as clock

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FourierPhysicalBridge(SpectralNS):
    """Extended NS solver measuring the Fourier <-> physical-space bridge."""

    def phase_scramble(self, u_hat, seed=None):
        """Return a copy of u_hat with randomized phases but identical amplitudes.

        Preserves: |u_hat(k)|, solenoidality, reality condition u(-k) = u*(k).
        Destroys: phase relationships between modes (the coherence).

        Method: Generate random real-valued fields in physical space (guaranteeing
        conjugate symmetry after FFT), then set amplitudes to match original.
        """
        rng = np.random.RandomState(seed)
        N = self.N

        # Original amplitudes per component
        amp = np.abs(u_hat)

        # Generate random real fields → FFT → guaranteed conjugate symmetry
        u_scrambled = np.zeros_like(u_hat)
        for i in range(3):
            rand_field = rng.randn(N, N, N)
            rand_hat = fftn(rand_field)
            rand_amp = np.abs(rand_hat)
            # Replace amplitudes but keep random phases
            safe_amp = np.where(rand_amp > 1e-30, rand_amp, 1.0)
            u_scrambled[i] = amp[i] * rand_hat / safe_amp

        # Re-project to enforce solenoidality
        u_scrambled = self.project_leray(u_scrambled)

        # Rescale to match original energy exactly
        E_orig = np.sum(np.abs(u_hat) ** 2)
        E_scr = np.sum(np.abs(u_scrambled) ** 2)
        if E_scr > 1e-30:
            u_scrambled *= np.sqrt(E_orig / E_scr)

        return u_scrambled

    def compute_alpha_full(self, u_hat):
        """Compute energy-weighted alpha = ||P_sol(L)||^2 / ||L||^2."""
        L_hat = self.compute_lamb_hat(u_hat)
        L_sol_hat = self.project_leray(L_hat)

        norm_L = np.sum(np.abs(L_hat) ** 2)
        norm_L_sol = np.sum(np.abs(L_sol_hat) ** 2)

        if norm_L < 1e-30:
            return 0.0
        return float(norm_L_sol / norm_L)

    def compute_alpha_cross(self, u_hat):
        """Compute alpha for cross-helical Lamb only."""
        L_cross_hat = self.compute_lamb_hat_cross_only(u_hat)
        L_cross_sol_hat = self.project_leray(L_cross_hat)

        norm_L = np.sum(np.abs(L_cross_hat) ** 2)
        norm_L_sol = np.sum(np.abs(L_cross_sol_hat) ** 2)

        if norm_L < 1e-30:
            return 0.0
        return float(norm_L_sol / norm_L)

    def vortex_tube_mask(self, u_hat, threshold_sigma=2.0):
        """Identify vortex tube regions: |omega| > mean + threshold * sigma.
        Returns boolean mask and vorticity magnitude field."""
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        omega_mag = np.sqrt(np.sum(omega ** 2, axis=0))

        mean_omega = np.mean(omega_mag)
        std_omega = np.std(omega_mag)
        threshold = mean_omega + threshold_sigma * std_omega

        mask = omega_mag > threshold
        return mask, omega_mag, omega

    def compute_strain_tensor(self, u_hat):
        """Return strain rate tensor S_ij in physical space."""
        N = self.N
        K = [self.kx, self.ky, self.kz]
        grad_u = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                grad_u[i, j] = np.real(ifftn(1j * K[i] * u_hat[j]))

        S = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                S[i, j] = 0.5 * (grad_u[j, i] + grad_u[i, j])
        return S

    def stretching_field(self, omega, S):
        """Compute omega_i S_ij omega_j at each point."""
        N = omega.shape[1]
        field = np.zeros((N, N, N))
        for i in range(3):
            for j in range(3):
                field += omega[i] * S[i, j] * omega[j]
        return field

    def compute_bridge_diagnostics(self, u_hat, n_scrambles=5):
        """Compute all bridge diagnostics for a given field state.

        Returns dict with all measurements.
        """
        N = self.N
        diag = {}

        # --- 1. ALPHA: actual vs scrambled ---
        alpha_actual = self.compute_alpha_full(u_hat)
        alpha_cross_actual = self.compute_alpha_cross(u_hat)

        alpha_scrambled_list = []
        alpha_cross_scrambled_list = []
        for s in range(n_scrambles):
            u_scr = self.phase_scramble(u_hat, seed=1000 + s)
            alpha_scrambled_list.append(self.compute_alpha_full(u_scr))
            alpha_cross_scrambled_list.append(self.compute_alpha_cross(u_scr))

        alpha_scrambled = np.mean(alpha_scrambled_list)
        alpha_cross_scrambled = np.mean(alpha_cross_scrambled_list)

        diag['alpha_actual'] = alpha_actual
        diag['alpha_scrambled'] = alpha_scrambled
        diag['alpha_cross_actual'] = alpha_cross_actual
        diag['alpha_cross_scrambled'] = alpha_cross_scrambled
        diag['coherence_amplification'] = (
            alpha_actual / alpha_scrambled if alpha_scrambled > 1e-10 else 1.0
        )
        diag['coherence_amplification_cross'] = (
            alpha_cross_actual / alpha_cross_scrambled
            if alpha_cross_scrambled > 1e-10
            else 1.0
        )

        # --- 2. VORTEX TUBE CONDITIONING ---
        tube_mask, omega_mag, omega = self.vortex_tube_mask(u_hat)
        tube_frac = np.mean(tube_mask)
        diag['tube_volume_fraction'] = float(tube_frac)

        # Compute solenoidal Lamb in physical space
        L_hat = self.compute_lamb_hat(u_hat)
        L_sol_hat = self.project_leray(L_hat)
        L_sol = np.array([np.real(ifftn(L_sol_hat[i])) for i in range(3)])
        L_sol_mag = np.sqrt(np.sum(L_sol ** 2, axis=0))
        L = np.array([np.real(ifftn(L_hat[i])) for i in range(3)])
        L_mag = np.sqrt(np.sum(L ** 2, axis=0))

        # Alpha inside tubes vs outside
        if np.sum(tube_mask) > 0:
            L2_inside = np.mean(L_mag[tube_mask] ** 2)
            Lsol2_inside = np.mean(L_sol_mag[tube_mask] ** 2)
            diag['alpha_inside_tubes'] = (
                float(Lsol2_inside / L2_inside) if L2_inside > 1e-30 else 0.0
            )
        else:
            diag['alpha_inside_tubes'] = 0.0

        outside_mask = ~tube_mask
        if np.sum(outside_mask) > 0:
            L2_outside = np.mean(L_mag[outside_mask] ** 2)
            Lsol2_outside = np.mean(L_sol_mag[outside_mask] ** 2)
            diag['alpha_outside_tubes'] = (
                float(Lsol2_outside / L2_outside) if L2_outside > 1e-30 else 0.0
            )
        else:
            diag['alpha_outside_tubes'] = 0.0

        # Fraction of solenoidal Lamb energy in tubes
        total_Lsol2 = np.mean(L_sol_mag ** 2)
        if total_Lsol2 > 1e-30 and tube_frac > 0:
            Lsol2_tube_total = np.mean(L_sol_mag[tube_mask] ** 2) * tube_frac
            diag['tube_Lsol_fraction'] = float(Lsol2_tube_total / total_Lsol2)
        else:
            diag['tube_Lsol_fraction'] = 0.0

        # --- 3. ANTI-TWIST PROXY ---
        S_tensor = self.compute_strain_tensor(u_hat)
        stretch_field = self.stretching_field(omega, S_tensor)

        # Conditional stretching at different |omega| percentiles
        omega_flat = omega_mag.ravel()
        stretch_flat = stretch_field.ravel()

        for pct in [50, 90, 95, 99]:
            threshold = np.percentile(omega_flat, pct)
            high_mask = omega_flat > threshold
            if np.sum(high_mask) > 0:
                mean_stretch_high = np.mean(stretch_flat[high_mask])
                # Normalize by |omega|^2 in high region to get stretching efficiency
                omega2_high = np.mean(omega_flat[high_mask] ** 2)
                stretch_efficiency = (
                    mean_stretch_high / omega2_high if omega2_high > 1e-30 else 0.0
                )
                diag[f'stretch_efficiency_p{pct}'] = float(stretch_efficiency)
            else:
                diag[f'stretch_efficiency_p{pct}'] = 0.0

        # omega-u alignment (Beltramization proxy)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        u_mag = np.sqrt(np.sum(u ** 2, axis=0))
        dot_ou = np.sum(omega * u, axis=0)
        cos_ou = dot_ou / (omega_mag * u_mag + 1e-30)

        # Mean |cos(omega, u)| in tubes vs outside
        if np.sum(tube_mask) > 0:
            diag['beltrami_tubes'] = float(np.mean(np.abs(cos_ou[tube_mask])))
        else:
            diag['beltrami_tubes'] = 0.0
        diag['beltrami_outside'] = float(np.mean(np.abs(cos_ou[outside_mask])))
        diag['beltrami_global'] = float(np.mean(np.abs(cos_ou)))

        # --- 4 & 5: SPATIAL CORRELATION ---
        # Pearson correlation between |P_sol(L)| and |omega|
        Lsol_flat = L_sol_mag.ravel()
        omega_flat_full = omega_mag.ravel()
        if np.std(Lsol_flat) > 1e-30 and np.std(omega_flat_full) > 1e-30:
            corr = np.corrcoef(Lsol_flat, omega_flat_full)[0, 1]
            diag['Lsol_omega_correlation'] = float(corr)
        else:
            diag['Lsol_omega_correlation'] = 0.0

        # --- Enstrophy ---
        Z = 0.5 * np.mean(np.sum(omega ** 2, axis=0))
        diag['enstrophy'] = float(Z)

        return diag


def run_bridge_experiment(N=32, Re=400, dt=0.005, T=5.0, report_every=40):
    """Run the full bridge experiment for multiple ICs."""

    print("=" * 80)
    print("FOURIER-PHYSICAL BRIDGE EXPERIMENT (S95)")
    print("=" * 80)
    print()
    print("Question: What connects spectral phase coherence to physical-space")
    print("vortex structures? Why does per-triad alpha = 1/4 but actual NS ~ 0.9?")
    print()
    print(f"Parameters: N={N}, Re={Re}, dt={dt}, T={T}")
    print(f"Phase scrambles per timestep: 5")
    print()

    solver = FourierPhysicalBridge(N=N, Re=Re)

    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Random': solver.random_ic(seed=42),
        'Imbalanced_80_20': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
    }

    n_steps = int(T / dt)
    all_results = {}

    for ic_name, u_hat_ic in ics.items():
        print(f"\n{'='*80}")
        print(f"IC: {ic_name}")
        print(f"{'='*80}")

        u_hat = u_hat_ic.copy()
        times = []
        diagnostics = []

        header = (
            f"{'t':>5} | {'a_act':>6} {'a_scr':>6} {'ratio':>6} | "
            f"{'a_cr_a':>6} {'a_cr_s':>6} {'cr_rat':>6} | "
            f"{'tube%':>5} {'a_in':>5} {'a_out':>5} | "
            f"{'Bel_t':>5} {'corr':>5} | {'Z':>9}"
        )
        print(header)
        print("-" * len(header))

        wall_start = clock.time()

        for step in range(n_steps + 1):
            t = step * dt

            if step % report_every == 0:
                diag = solver.compute_bridge_diagnostics(u_hat, n_scrambles=5)
                diag['time'] = t
                times.append(t)
                diagnostics.append(diag)

                print(
                    f"{t:5.2f} | "
                    f"{diag['alpha_actual']:6.3f} {diag['alpha_scrambled']:6.3f} "
                    f"{diag['coherence_amplification']:6.2f} | "
                    f"{diag['alpha_cross_actual']:6.3f} "
                    f"{diag['alpha_cross_scrambled']:6.3f} "
                    f"{diag['coherence_amplification_cross']:6.2f} | "
                    f"{diag['tube_volume_fraction']*100:5.1f} "
                    f"{diag['alpha_inside_tubes']:5.3f} "
                    f"{diag['alpha_outside_tubes']:5.3f} | "
                    f"{diag['beltrami_tubes']:5.3f} "
                    f"{diag['Lsol_omega_correlation']:5.3f} | "
                    f"{diag['enstrophy']:9.4e}"
                )

            if step < n_steps:
                u_hat = solver.step_rk4(u_hat, dt, mode='full')

        wall_time = clock.time() - wall_start
        print(f"\nWall time: {wall_time:.1f}s")

        all_results[ic_name] = {
            'times': np.array(times),
            'diagnostics': diagnostics,
        }

    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n\n" + "=" * 80)
    print("ANALYSIS: THE BRIDGE")
    print("=" * 80)

    for ic_name, result in all_results.items():
        diags = result['diagnostics']
        times = result['times']

        print(f"\n--- {ic_name} ---")

        # Phase coherence amplification
        ca = [d['coherence_amplification'] for d in diags]
        ca_cross = [d['coherence_amplification_cross'] for d in diags]
        alpha_act = [d['alpha_actual'] for d in diags]
        alpha_scr = [d['alpha_scrambled'] for d in diags]

        print(f"\n  Phase Coherence Amplification (alpha_actual / alpha_scrambled):")
        print(f"    Full Lamb:  min={min(ca):.3f}, max={max(ca):.3f}, "
              f"final={ca[-1]:.3f}")
        print(f"    Cross only: min={min(ca_cross):.3f}, max={max(ca_cross):.3f}, "
              f"final={ca_cross[-1]:.3f}")
        print(f"    Alpha actual (final): {alpha_act[-1]:.4f}")
        print(f"    Alpha scrambled (final): {alpha_scr[-1]:.4f}")

        if max(ca) > 1.5:
            print(f"    ** PHASE COHERENCE IS SIGNIFICANT (max amplification {max(ca):.2f}x)")
        else:
            print(f"    Phase coherence effect is modest (max amplification {max(ca):.2f}x)")

        # Vortex tube analysis
        alpha_in = [d['alpha_inside_tubes'] for d in diags]
        alpha_out = [d['alpha_outside_tubes'] for d in diags]
        tube_frac = [d['tube_volume_fraction'] for d in diags]
        tube_Lsol = [d['tube_Lsol_fraction'] for d in diags]

        print(f"\n  Vortex Tube Conditioning:")
        print(f"    Tube volume fraction: {tube_frac[-1]*100:.1f}%")
        print(f"    Alpha INSIDE tubes (final): {alpha_in[-1]:.4f}")
        print(f"    Alpha OUTSIDE tubes (final): {alpha_out[-1]:.4f}")
        ratio_in_out = alpha_in[-1] / alpha_out[-1] if alpha_out[-1] > 1e-10 else 0
        print(f"    Inside/Outside ratio: {ratio_in_out:.3f}")
        print(f"    Fraction of |P_sol(L)|^2 in tubes: {tube_Lsol[-1]*100:.1f}%")

        if ratio_in_out > 1.2:
            print(f"    ** TUBES CONCENTRATE SOLENOIDAL FORCING "
                  f"({ratio_in_out:.2f}x more inside)")
        else:
            print(f"    Solenoidal forcing similar inside/outside tubes")

        # Anti-twist
        print(f"\n  Anti-Twist (stretching efficiency omega.S.omega / |omega|^2):")
        for pct in [50, 90, 95, 99]:
            se = [d[f'stretch_efficiency_p{pct}'] for d in diags]
            print(f"    |omega| > p{pct}: final efficiency = {se[-1]:.4f}")

        se_50_final = diags[-1]['stretch_efficiency_p50']
        se_99_final = diags[-1]['stretch_efficiency_p99']
        if se_99_final > se_50_final * 1.5:
            print(f"    ** Stretching INCREASES at extreme vorticity "
                  f"(p99/p50 = {se_99_final/se_50_final:.2f}x)")
            print(f"    ** NO anti-twist at this Re (consistent with Q2 result)")
        elif se_99_final < se_50_final * 0.8:
            print(f"    ** Stretching DECREASES at extreme vorticity "
                  f"(anti-twist signal!)")

        # Beltramization
        bel_t = [d['beltrami_tubes'] for d in diags]
        bel_out = [d['beltrami_outside'] for d in diags]
        print(f"\n  Beltramization |cos(omega, u)|:")
        print(f"    Inside tubes: {bel_t[-1]:.4f}")
        print(f"    Outside tubes: {bel_out[-1]:.4f}")

        # Spatial correlation
        corr = [d['Lsol_omega_correlation'] for d in diags]
        print(f"\n  Spatial Correlation (|P_sol(L)| vs |omega|):")
        print(f"    Final: {corr[-1]:.4f}")
        if corr[-1] > 0.5:
            print(f"    ** STRONG co-location: solenoidal forcing concentrates "
                  f"where vorticity is high")

    # ============================================================
    # THE BRIDGE VERDICT
    # ============================================================
    print("\n\n" + "=" * 80)
    print("THE BRIDGE: SPECTRAL <-> PHYSICAL CONNECTION")
    print("=" * 80)

    # Collect cross-IC patterns
    all_ca_max = []
    all_tube_ratio = []
    all_corr_final = []

    for ic_name, result in all_results.items():
        diags = result['diagnostics']
        ca = [d['coherence_amplification'] for d in diags]
        all_ca_max.append(max(ca))

        alpha_in = diags[-1]['alpha_inside_tubes']
        alpha_out = diags[-1]['alpha_outside_tubes']
        all_tube_ratio.append(
            alpha_in / alpha_out if alpha_out > 1e-10 else 1.0
        )
        all_corr_final.append(diags[-1]['Lsol_omega_correlation'])

    print(f"\n  Cross-IC coherence amplification max: {all_ca_max}")
    print(f"  Cross-IC tube alpha ratio (in/out):   {all_tube_ratio}")
    print(f"  Cross-IC Lsol-omega correlation:       {all_corr_final}")

    # Test the feedback loop hypothesis
    print("\n  FEEDBACK LOOP TEST:")
    print("  1. Random phases → alpha ~ 1/4 (theoretical)")
    print(f"     Scrambled alpha ~ {np.mean([d['alpha_scrambled'] for d in all_results[list(all_results.keys())[0]]['diagnostics'][-3:]]):.3f} (measured)")

    print(f"  2. NS builds coherence → amplification ratio ~ "
          f"{np.mean(all_ca_max):.2f}x")

    print(f"  3. Coherence concentrates in tubes? "
          f"{'YES' if np.mean(all_tube_ratio) > 1.1 else 'UNCLEAR'} "
          f"(ratio = {np.mean(all_tube_ratio):.2f})")

    print(f"  4. Tubes self-regulate (anti-twist)? "
          f"See stretching efficiency above")

    print(f"  5. Solenoidal forcing co-locates with vorticity? "
          f"{'YES' if np.mean(all_corr_final) > 0.3 else 'NO'} "
          f"(r = {np.mean(all_corr_final):.3f})")

    # ============================================================
    # PLOTS
    # ============================================================
    print("\n\nGenerating plots...")

    n_ics = len(all_results)
    fig, axes = plt.subplots(4, n_ics, figsize=(6 * n_ics, 20))
    if n_ics == 1:
        axes = axes[:, np.newaxis]

    for col, (ic_name, result) in enumerate(all_results.items()):
        diags = result['diagnostics']
        times = result['times']

        # Row 1: Alpha actual vs scrambled
        ax = axes[0, col]
        alpha_act = [d['alpha_actual'] for d in diags]
        alpha_scr = [d['alpha_scrambled'] for d in diags]
        alpha_cr_act = [d['alpha_cross_actual'] for d in diags]
        alpha_cr_scr = [d['alpha_cross_scrambled'] for d in diags]
        ax.plot(times, alpha_act, 'b-', lw=2, label='a actual')
        ax.plot(times, alpha_scr, 'b--', lw=1.5, label='a scrambled')
        ax.plot(times, alpha_cr_act, 'r-', lw=2, label='a_cross actual')
        ax.plot(times, alpha_cr_scr, 'r--', lw=1.5, label='a_cross scrambled')
        ax.axhline(y=0.25, color='k', ls=':', lw=1, label='Theory: 1/4')
        ax.set_title(f'{ic_name}\nPhase Coherence Effect')
        ax.set_ylabel('a (solenoidal fraction)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Row 2: Coherence amplification ratio
        ax = axes[1, col]
        ca = [d['coherence_amplification'] for d in diags]
        ca_cross = [d['coherence_amplification_cross'] for d in diags]
        Z = [d['enstrophy'] for d in diags]
        ax2 = ax.twinx()
        ax.plot(times, ca, 'b-', lw=2, label='Full amplification')
        ax.plot(times, ca_cross, 'r-', lw=2, label='Cross amplification')
        ax.axhline(y=1.0, color='k', ls=':', lw=1)
        ax2.semilogy(times, Z, 'g--', lw=1, alpha=0.5, label='Enstrophy')
        ax.set_ylabel('Coherence amplification ratio')
        ax2.set_ylabel('Enstrophy', color='g')
        ax.set_title('Amplification vs Enstrophy')
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 3: Tube conditioning
        ax = axes[2, col]
        alpha_in = [d['alpha_inside_tubes'] for d in diags]
        alpha_out = [d['alpha_outside_tubes'] for d in diags]
        tube_Lsol = [d['tube_Lsol_fraction'] for d in diags]
        ax.plot(times, alpha_in, 'r-', lw=2, label='a inside tubes')
        ax.plot(times, alpha_out, 'b-', lw=2, label='a outside tubes')
        ax.plot(times, tube_Lsol, 'g--', lw=1.5, label='|P_sol(L)|² frac in tubes')
        ax.set_title('Vortex Tube Conditioning')
        ax.set_ylabel('a / fraction')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 4: Anti-twist + correlation
        ax = axes[3, col]
        for pct in [50, 90, 99]:
            se = [d[f'stretch_efficiency_p{pct}'] for d in diags]
            ax.plot(times, se, lw=1.5, label=f'Stretch eff p{pct}')
        corr = [d['Lsol_omega_correlation'] for d in diags]
        ax.plot(times, corr, 'k--', lw=2, label='Lsol-w corr')
        ax.set_xlabel('t')
        ax.set_ylabel('Efficiency / Correlation')
        ax.set_title('Anti-Twist + Spatial Correlation')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'h:/tmp/fourier_physical_bridge.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    return all_results


if __name__ == "__main__":
    all_results = run_bridge_experiment()
