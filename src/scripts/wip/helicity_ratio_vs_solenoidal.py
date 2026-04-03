"""
TASK 2: HELICITY RATIO vs SOLENOIDAL FRACTION
==============================================
Meridian's Approach A, Task 2 (S92):

Question: Does the solenoidal fraction s(t) decrease as the helicity
ratio r(t) = ||v-||^2 / ||v+||^2 approaches 0?

If s ~ r^alpha with alpha > 0, then helicity conservation (which keeps
H = E+ - E- ~ const, forcing r -> 0 as E -> 0) automatically bounds
the solenoidal fraction and hence the nonlinear drive.

Method:
1. Generate ICs with varying helicity levels:
   H/H_max ~ 0.1, 0.3, 0.5, 0.7, 0.9
2. Evolve each under full NS and BT surgery
3. At each timestep measure:
   - r(t) = ||v-||^2 / ||v+||^2 (helicity ratio)
   - s(t) = ||P_sol(omega x v)|| / ||omega x v|| (solenoidal fraction)
4. Plot s vs r across all ICs and times

HONEST TEST: We report what the numbers say.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


class HelicityRatioTracker(SpectralNS):
    """Extends SpectralNS with helicity ratio and solenoidal fraction tracking."""

    def make_chiral_ic(self, h_plus_frac=0.5, seed=42):
        """Create IC with specified h+ energy fraction.

        h_plus_frac = 0.5 means achiral (equal h+/h-)
        h_plus_frac = 0.95 means strongly chiral (mostly h+)
        """
        return self.imbalanced_helical_ic(seed=seed, h_plus_frac=h_plus_frac)

    def compute_helicity_ratio(self, u_hat):
        """Compute r = E- / E+ = ||v-||^2 / ||v+||^2."""
        u_p, u_m = self.helical_decompose(u_hat)
        E_plus = float(np.sum(np.abs(u_p)**2))
        E_minus = float(np.sum(np.abs(u_m)**2))
        if E_plus < 1e-30:
            return float('inf')
        return E_minus / E_plus

    def compute_solenoidal_fraction(self, u_hat):
        """Compute s = ||P_sol(omega x v)|| / ||omega x v||."""
        N = self.N
        lamb_hat = self.compute_lamb_hat(u_hat)
        lamb_sol_hat = self.project_leray(lamb_hat)

        norm_full_sq = sum(float(np.sum(np.abs(lamb_hat[i])**2)) for i in range(3))
        norm_sol_sq = sum(float(np.sum(np.abs(lamb_sol_hat[i])**2)) for i in range(3))

        if norm_full_sq < 1e-30:
            return 0.0

        # Return sqrt ratio (norm ratio, not energy ratio)
        return float(np.sqrt(norm_sol_sq / norm_full_sq))

    def compute_helicity_normalized(self, u_hat):
        """Compute H / H_max where H_max = 2*E*k_max_effective."""
        u_p, u_m = self.helical_decompose(u_hat)
        E_plus = float(np.sum(np.abs(u_p)**2))
        E_minus = float(np.sum(np.abs(u_m)**2))

        # Helicity: H = sum_k |k| * (|u_p|^2 - |u_m|^2)
        H = float(np.sum(self.kmag * (np.abs(u_p)**2 - np.abs(u_m)**2)))
        E_total = E_plus + E_minus
        if E_total < 1e-30:
            return 0.0
        # Approximate H/H_max
        return float((E_plus - E_minus) / (E_plus + E_minus))

    def run_sweep(self, u_hat_init, mode='full', dt=0.005, t_max=10.0,
                  sample_interval=20, label=''):
        """Run simulation tracking r(t) and s(t)."""
        u_hat = u_hat_init.copy()
        t = 0.0
        step = 0

        data = {
            'label': label,
            'mode': mode,
            'times': [],
            'r': [],       # helicity ratio E-/E+
            's': [],       # solenoidal fraction
            'h_norm': [],  # normalized helicity
            'energy': [],
        }

        while t <= t_max:
            if step % sample_interval == 0:
                r = self.compute_helicity_ratio(u_hat)
                s = self.compute_solenoidal_fraction(u_hat)
                h = self.compute_helicity_normalized(u_hat)

                u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
                E = 0.5 * np.mean(np.sum(u**2, axis=0))

                data['times'].append(float(t))
                data['r'].append(float(r))
                data['s'].append(float(s))
                data['h_norm'].append(float(h))
                data['energy'].append(float(E))

                if E < 1e-12:
                    break

            u_hat = self.step_rk4(u_hat, dt, mode=mode)
            t += dt
            step += 1

        return data


def fit_power_law(r_arr, s_arr):
    """Fit s = C * r^alpha via log-log regression. Returns (alpha, C, R^2)."""
    mask = (r_arr > 1e-10) & (s_arr > 1e-10) & np.isfinite(r_arr) & np.isfinite(s_arr)
    if np.sum(mask) < 3:
        return 0.0, 0.0, 0.0

    log_r = np.log(r_arr[mask])
    log_s = np.log(s_arr[mask])

    # Linear regression: log_s = alpha * log_r + log_C
    coeffs = np.polyfit(log_r, log_s, 1)
    alpha = coeffs[0]
    C = np.exp(coeffs[1])

    # R^2
    predicted = np.polyval(coeffs, log_r)
    ss_res = np.sum((log_s - predicted)**2)
    ss_tot = np.sum((log_s - np.mean(log_s))**2)
    R2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return float(alpha), float(C), float(R2)


def main():
    print("=" * 70)
    print("  TASK 2: HELICITY RATIO vs SOLENOIDAL FRACTION")
    print("  s(t) = ||P_sol(L)|| / ||L|| vs r(t) = E-/E+")
    print("  Looking for s ~ r^alpha with alpha > 0")
    print("=" * 70)

    solver = HelicityRatioTracker(N=32, Re=400)

    # Generate ICs with varying helicity levels
    h_plus_fracs = [0.55, 0.65, 0.75, 0.85, 0.95]
    # This gives H/H_max ~ 0.1, 0.3, 0.5, 0.7, 0.9

    all_data = {}
    dt = 0.005
    t_max = 10.0

    for h_frac in h_plus_fracs:
        h_label = f"h+={h_frac:.0%}"
        u_hat_init = solver.make_chiral_ic(h_plus_frac=h_frac, seed=42)

        # Check actual helicity ratio
        r0 = solver.compute_helicity_ratio(u_hat_init)
        h0 = solver.compute_helicity_normalized(u_hat_init)
        print(f"\n  IC: {h_label}  r0={r0:.4f}  H/H_max={h0:.3f}")

        for mode in ['full', 'bt']:
            key = f"{h_label}_{mode}"
            data = solver.run_sweep(
                u_hat_init, mode=mode, dt=dt, t_max=t_max,
                sample_interval=20, label=h_label
            )
            all_data[key] = data

    # =========================================================
    # ANALYSIS: s vs r scatter and power law fit
    # =========================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS: s vs r")
    print("=" * 70)

    # Collect all (r, s) pairs for each mode
    for mode in ['full', 'bt']:
        all_r = []
        all_s = []
        print(f"\n  Mode: {mode.upper()} NS")
        print(f"  {'IC':<12s}  {'r_init':>8s}  {'r_final':>8s}  "
              f"{'s_init':>8s}  {'s_final':>8s}  {'alpha':>8s}  {'R^2':>6s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

        for h_frac in h_plus_fracs:
            h_label = f"h+={h_frac:.0%}"
            key = f"{h_label}_{mode}"
            data = all_data[key]

            r_arr = np.array(data['r'])
            s_arr = np.array(data['s'])
            all_r.extend(data['r'])
            all_s.extend(data['s'])

            alpha, C, R2 = fit_power_law(r_arr, s_arr)

            print(f"  {h_label:<12s}  {r_arr[0]:8.4f}  {r_arr[-1]:8.4f}  "
                  f"{s_arr[0]:8.4f}  {s_arr[-1]:8.4f}  {alpha:8.4f}  {R2:6.3f}")

        # Global fit across all ICs
        all_r_arr = np.array(all_r)
        all_s_arr = np.array(all_s)
        global_alpha, global_C, global_R2 = fit_power_law(all_r_arr, all_s_arr)
        print(f"\n  GLOBAL FIT: s = {global_C:.4f} * r^{global_alpha:.4f}  (R^2={global_R2:.3f})")

    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Check global fits
    for mode in ['full', 'bt']:
        all_r = []
        all_s = []
        for h_frac in h_plus_fracs:
            key = f"h+={h_frac:.0%}_{mode}"
            all_r.extend(all_data[key]['r'])
            all_s.extend(all_data[key]['s'])

        alpha, C, R2 = fit_power_law(np.array(all_r), np.array(all_s))

        if alpha > 0.05 and R2 > 0.3:
            print(f"\n  {mode.upper()} NS: s ~ r^{alpha:.3f} (R^2={R2:.3f})")
            print(f"    => POSITIVE: solenoidal fraction decreases as helicity ratio -> 0")
            print(f"    => Helicity conservation DOES constrain nonlinear drive")
        elif alpha > 0:
            print(f"\n  {mode.upper()} NS: s ~ r^{alpha:.3f} (R^2={R2:.3f})")
            print(f"    => WEAK: trend exists but not clean power law")
        else:
            print(f"\n  {mode.upper()} NS: alpha = {alpha:.3f} (R^2={R2:.3f})")
            print(f"    => NEGATIVE: no s-r correlation")

    # =========================================================
    # SAVE & PLOT
    # =========================================================
    output_path = os.path.join(os.path.dirname(__file__), 'helicity_ratio_vs_solenoidal.json')
    save_data = {'params': {'N': 32, 'Re': 400, 'dt': dt, 't_max': t_max}}
    for key, data in all_data.items():
        save_data[key] = {
            'r_range': [min(data['r']), max(data['r'])],
            's_range': [min(data['s']), max(data['s'])],
        }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # Plot: s vs r scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Solenoidal Fraction s vs Helicity Ratio r = E-/E+', fontsize=13)

    colors_h = plt.cm.viridis(np.linspace(0.1, 0.9, len(h_plus_fracs)))

    for col, mode in enumerate(['full', 'bt']):
        ax = axes[col]
        all_r = []
        all_s = []

        for idx, h_frac in enumerate(h_plus_fracs):
            key = f"h+={h_frac:.0%}_{mode}"
            data = all_data[key]
            ax.scatter(data['r'], data['s'], c=[colors_h[idx]]*len(data['r']),
                       s=10, alpha=0.6, label=f'h+={h_frac:.0%}')
            all_r.extend(data['r'])
            all_s.extend(data['s'])

        # Global fit line
        all_r_arr = np.array(all_r)
        all_s_arr = np.array(all_s)
        alpha, C, R2 = fit_power_law(all_r_arr, all_s_arr)

        if R2 > 0.1:
            r_fit = np.linspace(max(all_r_arr.min(), 0.001), all_r_arr.max(), 100)
            s_fit = C * r_fit**alpha
            ax.plot(r_fit, s_fit, 'r--', linewidth=2,
                    label=f's = {C:.3f}*r^{alpha:.3f} (R2={R2:.2f})')

        ax.set_xlabel('r = E-/E+')
        ax.set_ylabel('s = ||P_sol(L)|| / ||L||')
        ax.set_title(f'{mode.upper()} NS')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'helicity_ratio_vs_solenoidal.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved to {plot_path}")
    plt.close()

    # Second plot: time evolution of r and s
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Time Evolution of Helicity Ratio r and Solenoidal Fraction s', fontsize=13)

    for col, mode in enumerate(['full', 'bt']):
        for idx, h_frac in enumerate(h_plus_fracs):
            key = f"h+={h_frac:.0%}_{mode}"
            data = all_data[key]

            axes[0, col].plot(data['times'], data['r'],
                              color=colors_h[idx], label=f'h+={h_frac:.0%}')
            axes[1, col].plot(data['times'], data['s'],
                              color=colors_h[idx], label=f'h+={h_frac:.0%}')

        axes[0, col].set_ylabel('r = E-/E+')
        axes[0, col].set_title(f'{mode.upper()} NS — Helicity Ratio')
        axes[0, col].set_yscale('log')
        axes[0, col].legend(fontsize=7)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].set_ylabel('s = ||P_sol(L)|| / ||L||')
        axes[1, col].set_xlabel('Time')
        axes[1, col].set_title(f'{mode.upper()} NS — Solenoidal Fraction')
        axes[1, col].legend(fontsize=7)
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path2 = os.path.join(os.path.dirname(__file__), 'helicity_ratio_timeseries.png')
    plt.savefig(plot_path2, dpi=150)
    print(f"  Plot saved to {plot_path2}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
