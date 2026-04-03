"""
LAMB VECTOR HELICITY SECTOR DECOMPOSITION
==========================================
Meridian's Approach A (S91 audit):

Decompose the Lamb vector L = omega x v into four helicity sectors:
  L = (omega+ x v+) + (omega- x v-) + (omega+ x v-) + (omega- x v+)
      (same-plus)      (same-minus)     (cross-1)       (cross-2)

Then Leray-project each sector separately:
  P_sol(L) = P_sol(L_s+) + P_sol(L_s-) + P_sol(L_c1) + P_sol(L_c2)

Question: which helicity sector contributes most to the SOLENOIDAL
(dynamically active) part of the Lamb vector?

If cross-helicity terms dominate P_sol(L), then helicity-sector
decomposition IS the proof strategy: bounding cross-helicity coupling
bounds the nonlinear drive.

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


class LambDecomposer(SpectralNS):
    """Extends SpectralNS with helicity-sector Lamb vector decomposition."""

    def decompose_lamb_by_helicity(self, u_hat):
        """Decompose Lamb vector into 4 helicity sectors and Leray-project each.

        Returns dict with norms of each sector (full and solenoidal).
        """
        N = self.N

        # Decompose velocity into h+ and h-
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)

        # Vorticity for each sector
        omega_hat_plus = self.compute_vorticity_hat(u_hat_plus)
        omega_hat_minus = self.compute_vorticity_hat(u_hat_minus)

        # Physical space fields
        v_plus = np.array([np.real(ifftn(u_hat_plus[i])) for i in range(3)])
        v_minus = np.array([np.real(ifftn(u_hat_minus[i])) for i in range(3)])
        w_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
        w_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

        def cross(a, b):
            return np.array([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])

        def to_fourier_dealiased(field):
            f_hat = np.array([fftn(field[i]) for i in range(3)])
            for i in range(3):
                f_hat[i] *= self.dealias_mask
            return f_hat

        def norm_sq(f_hat):
            return sum(float(np.sum(np.abs(f_hat[i])**2)) for i in range(3))

        # Four helicity sectors of the Lamb vector
        sectors = {
            'same_plus':  cross(w_plus, v_plus),
            'same_minus': cross(w_minus, v_minus),
            'cross_1':    cross(w_plus, v_minus),
            'cross_2':    cross(w_minus, v_plus),
        }

        results = {}
        total_sol_sq = 0.0
        total_full_sq = 0.0

        for name, lamb_phys in sectors.items():
            lamb_hat = to_fourier_dealiased(lamb_phys)
            lamb_sol_hat = self.project_leray(lamb_hat)

            full_sq = norm_sq(lamb_hat)
            sol_sq = norm_sq(lamb_sol_hat)

            results[name] = {
                'full_norm_sq': full_sq,
                'sol_norm_sq': sol_sq,
                'full_norm': float(np.sqrt(full_sq / N**6)),
                'sol_norm': float(np.sqrt(sol_sq / N**6)),
            }
            total_full_sq += full_sq
            total_sol_sq += sol_sq

        for name in results:
            results[name]['full_fraction'] = results[name]['full_norm_sq'] / max(total_full_sq, 1e-30)
            results[name]['sol_fraction'] = results[name]['sol_norm_sq'] / max(total_sol_sq, 1e-30)

        same_full = results['same_plus']['full_norm_sq'] + results['same_minus']['full_norm_sq']
        same_sol = results['same_plus']['sol_norm_sq'] + results['same_minus']['sol_norm_sq']
        cross_full = results['cross_1']['full_norm_sq'] + results['cross_2']['full_norm_sq']
        cross_sol = results['cross_1']['sol_norm_sq'] + results['cross_2']['sol_norm_sq']

        results['_aggregate'] = {
            'same_full_frac': float(same_full / max(total_full_sq, 1e-30)),
            'same_sol_frac': float(same_sol / max(total_sol_sq, 1e-30)),
            'cross_full_frac': float(cross_full / max(total_full_sq, 1e-30)),
            'cross_sol_frac': float(cross_sol / max(total_sol_sq, 1e-30)),
            'total_full_norm': float(np.sqrt(total_full_sq / N**6)),
            'total_sol_norm': float(np.sqrt(total_sol_sq / N**6)),
        }

        return results

    def run_decomposition_timeseries(self, u_hat_init, mode='full', dt=0.005,
                                      t_max=10.0, sample_interval=40, label=''):
        """Run simulation tracking helicity-sector Lamb decomposition over time."""
        u_hat = u_hat_init.copy()
        t = 0.0
        step = 0

        timeseries = {
            'label': label,
            'mode': mode,
            'times': [],
            'same_sol_frac': [],
            'cross_sol_frac': [],
            'same_full_frac': [],
            'cross_full_frac': [],
            'total_sol_norm': [],
            'total_full_norm': [],
        }

        print(f"\n{'='*70}")
        print(f"  {label} ({mode} NS)")
        print(f"{'='*70}")
        print(f"  {'t':>5s}  {'same_sol%':>9s}  {'cross_sol%':>10s}  "
              f"{'same_full%':>10s}  {'cross_full%':>11s}  {'||L_sol||':>9s}")
        print(f"  {'-'*5}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*9}")

        while t <= t_max:
            if step % sample_interval == 0:
                decomp = self.decompose_lamb_by_helicity(u_hat)
                agg = decomp['_aggregate']

                timeseries['times'].append(float(t))
                timeseries['same_sol_frac'].append(agg['same_sol_frac'])
                timeseries['cross_sol_frac'].append(agg['cross_sol_frac'])
                timeseries['same_full_frac'].append(agg['same_full_frac'])
                timeseries['cross_full_frac'].append(agg['cross_full_frac'])
                timeseries['total_sol_norm'].append(agg['total_sol_norm'])
                timeseries['total_full_norm'].append(agg['total_full_norm'])

                print(f"  {t:5.1f}  {agg['same_sol_frac']*100:8.2f}%  "
                      f"{agg['cross_sol_frac']*100:9.2f}%  "
                      f"{agg['same_full_frac']*100:9.2f}%  "
                      f"{agg['cross_full_frac']*100:10.2f}%  "
                      f"{agg['total_sol_norm']:9.6f}")

                u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
                E = 0.5 * np.mean(np.sum(u**2, axis=0))
                if E < 1e-12:
                    print(f"  [Energy negligible, stopping]")
                    break

            u_hat = self.step_rk4(u_hat, dt, mode=mode)
            t += dt
            step += 1

        return timeseries


def main():
    print("=" * 70)
    print("  LAMB VECTOR HELICITY SECTOR DECOMPOSITION")
    print("  Meridian's Approach A: which sector drives P_sol(omega x v)?")
    print("=" * 70)

    solver = LambDecomposer(N=32, Re=400)

    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Imbalanced 80/20': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
        'Random': solver.random_ic(seed=42),
    }

    # =========================================================
    # SNAPSHOT AT t=0: detailed sector breakdown
    # =========================================================
    print("\n" + "=" * 70)
    print("  SNAPSHOT AT t=0")
    print("=" * 70)

    for ic_name, u_hat in ics.items():
        decomp = solver.decompose_lamb_by_helicity(u_hat)
        print(f"\n  {ic_name}:")
        print(f"    {'Sector':<14s}  {'||L||':>10s}  {'||P_sol(L)||':>12s}  "
              f"{'% of full':>9s}  {'% of sol':>8s}")
        print(f"    {'-'*14}  {'-'*10}  {'-'*12}  {'-'*9}  {'-'*8}")

        for name in ['same_plus', 'same_minus', 'cross_1', 'cross_2']:
            s = decomp[name]
            print(f"    {name:<14s}  {s['full_norm']:10.6f}  {s['sol_norm']:12.6f}  "
                  f"{s['full_fraction']*100:8.2f}%  {s['sol_fraction']*100:7.2f}%")

        agg = decomp['_aggregate']
        print(f"    {'--- TOTALS ---'}")
        print(f"    Same-helicity:   full={agg['same_full_frac']*100:.1f}%  "
              f"sol={agg['same_sol_frac']*100:.1f}%")
        print(f"    Cross-helicity:  full={agg['cross_full_frac']*100:.1f}%  "
              f"sol={agg['cross_sol_frac']*100:.1f}%")

    # =========================================================
    # TIME SERIES
    # =========================================================
    all_timeseries = {}
    dt = 0.005
    t_max = 10.0

    for ic_name, u_hat_init in ics.items():
        for mode in ['full', 'bt']:
            key = f"{ic_name}_{mode}"
            ts = solver.run_decomposition_timeseries(
                u_hat_init, mode=mode, dt=dt, t_max=t_max,
                sample_interval=40, label=ic_name
            )
            all_timeseries[key] = ts

    # =========================================================
    # ANALYSIS
    # =========================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS: Cross vs Same helicity contribution to P_sol(L)")
    print("=" * 70)

    for key, ts in all_timeseries.items():
        cross_avg = np.mean(ts['cross_sol_frac'])
        same_avg = np.mean(ts['same_sol_frac'])
        cross_initial = ts['cross_sol_frac'][0]
        cross_final = ts['cross_sol_frac'][-1]

        dominant = "CROSS" if cross_avg > 0.5 else "SAME"
        print(f"\n  {key}:")
        print(f"    Cross-helicity sol fraction: {cross_avg*100:.1f}% avg  "
              f"({cross_initial*100:.1f}% => {cross_final*100:.1f}%)")
        print(f"    Same-helicity sol fraction:  {same_avg*100:.1f}% avg")
        print(f"    Dominant sector in P_sol(L): {dominant}")

    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    cross_dominates_all = all(
        np.mean(ts['cross_sol_frac']) > 0.5 for ts in all_timeseries.values()
    )

    if cross_dominates_all:
        print("  Cross-helicity terms DOMINATE P_sol(omega x v) in ALL runs.")
        print("  => Helicity-sector decomposition IS the proof strategy.")
        print("     Bounding cross-helicity coupling bounds nonlinear drive.")
    else:
        # Check per-mode
        full_cross = all(
            np.mean(all_timeseries[k]['cross_sol_frac']) > 0.5
            for k in all_timeseries if 'full' in k
        )
        bt_cross = all(
            np.mean(all_timeseries[k]['cross_sol_frac']) > 0.5
            for k in all_timeseries if 'bt' in k
        )
        if full_cross and not bt_cross:
            print("  Cross-helicity dominates in full NS but not BT.")
            print("  => Cross terms ARE the main solenoidal driver in full NS.")
            print("     BT removes them, shifting balance to same-helicity.")
        elif not full_cross and not bt_cross:
            print("  Same-helicity terms contribute significantly to P_sol(L).")
            print("  => Helicity-sector decomposition alone is INSUFFICIENT.")
        else:
            print("  Mixed results across ICs and modes.")

    # BT comparison
    print("\n  BT effect on cross-helicity solenoidal contribution:")
    for ic_name in ics.keys():
        key_full = f"{ic_name}_full"
        key_bt = f"{ic_name}_bt"
        if key_full in all_timeseries and key_bt in all_timeseries:
            cross_full = np.mean(all_timeseries[key_full]['cross_sol_frac'])
            cross_bt = np.mean(all_timeseries[key_bt]['cross_sol_frac'])
            print(f"    {ic_name}: full={cross_full*100:.1f}%  "
                  f"bt={cross_bt*100:.1f}%  "
                  f"(BT {'reduces' if cross_bt < cross_full else 'increases'} "
                  f"cross contribution)")

    # =========================================================
    # SAVE & PLOT
    # =========================================================
    output_path = os.path.join(os.path.dirname(__file__), 'lamb_decomposition_results.json')
    save_data = {
        'params': {'N': 32, 'Re': 400, 'dt': dt, 't_max': t_max},
        'timeseries_summary': {}
    }
    for key, ts in all_timeseries.items():
        save_data['timeseries_summary'][key] = {
            'cross_sol_avg': float(np.mean(ts['cross_sol_frac'])),
            'same_sol_avg': float(np.mean(ts['same_sol_frac'])),
            'cross_full_avg': float(np.mean(ts['cross_full_frac'])),
            'same_full_avg': float(np.mean(ts['same_full_frac'])),
        }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Lamb Vector Helicity Decomposition: Cross vs Same in P_sol(L)', fontsize=13)

    colors_mode = {'full': ('tab:red', 'tab:blue'), 'bt': ('tab:orange', 'tab:cyan')}
    ic_names = list(ics.keys())

    for col, ic_name in enumerate(ic_names):
        ax = axes[0, col]
        for mode in ['full', 'bt']:
            key = f"{ic_name}_{mode}"
            ts = all_timeseries[key]
            c_cross, c_same = colors_mode[mode]
            ls = '-' if mode == 'full' else '--'
            ax.plot(ts['times'], [x*100 for x in ts['cross_sol_frac']],
                    color=c_cross, linestyle=ls, label=f'Cross ({mode})', linewidth=1.5)
            ax.plot(ts['times'], [x*100 for x in ts['same_sol_frac']],
                    color=c_same, linestyle=ls, label=f'Same ({mode})', linewidth=1.5)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(ic_name)
        ax.set_ylabel('% of P_sol(L)' if col == 0 else '')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        ax = axes[1, col]
        for mode in ['full', 'bt']:
            key = f"{ic_name}_{mode}"
            ts = all_timeseries[key]
            ls = '-' if mode == 'full' else '--'
            ax.plot(ts['times'], ts['total_sol_norm'],
                    color='black', linestyle=ls, label=f'||P_sol(L)|| ({mode})', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('||P_sol(L)||' if col == 0 else '')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'lamb_helicity_decomposition.png')
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
