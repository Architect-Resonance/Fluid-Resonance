"""
H(t)/E(t) TIME SERIES — Helicity-to-Energy Ratio Evolution
============================================================
Track how helicity relative to energy evolves under full NS and BT surgery.
Uses existing solver infrastructure.

Meridian 2, S96.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS, run_single_ic


def main():
    print("=" * 70)
    print("  H(t)/E(t) TIME SERIES")
    print("=" * 70)

    solver = SpectralNS(N=32, Re=400)

    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Random': solver.random_ic(seed=42),
        'Imbalanced 80/20': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Helicity/Energy Evolution under Full NS vs BT Surgery', fontsize=13)

    for col, (ic_name, u_hat_ic) in enumerate(ics.items()):
        print(f"\n  Running {ic_name}...")
        data = run_single_ic(solver, ic_name, u_hat_ic, dt=0.005, T=5.0,
                            report_every=10, verbose=False)

        t = data['times']
        E_full = data['E_full']
        H_full = data['H_full']
        E_bt = data['E_bt']
        H_bt = data['H_bt']
        Z_full = data['Z_full']
        Z_bt = data['Z_bt']

        # H/E ratio (handle near-zero E)
        HE_full = np.where(E_full > 1e-15, H_full / E_full, 0.0)
        HE_bt = np.where(E_bt > 1e-15, H_bt / E_bt, 0.0)

        # Relative helicity: H / (2*E*Z)^{1/2} (Schwarz bound: |H| <= 2*sqrt(E*Z))
        schwarz_full = 2 * np.sqrt(E_full * Z_full)
        schwarz_bt = 2 * np.sqrt(E_bt * Z_bt)
        H_rel_full = np.where(schwarz_full > 1e-15, H_full / schwarz_full, 0.0)
        H_rel_bt = np.where(schwarz_bt > 1e-15, H_bt / schwarz_bt, 0.0)

        print(f"    H/E range (full): [{np.min(HE_full):.4f}, {np.max(HE_full):.4f}]")
        print(f"    H/E range (BT):   [{np.min(HE_bt):.4f}, {np.max(HE_bt):.4f}]")
        print(f"    |H_rel| max (full): {np.max(np.abs(H_rel_full)):.4f}")
        print(f"    |H_rel| max (BT):   {np.max(np.abs(H_rel_bt)):.4f}")

        # Helicity dissipation rate: dH/dt
        dH_full = np.gradient(H_full, t)
        dH_bt = np.gradient(H_bt, t)

        # Top row: H(t), E(t), Z(t) together
        ax = axes[0, col]
        ax.plot(t, E_full, 'b-', linewidth=2, label='E(t) full')
        ax.plot(t, E_bt, 'b--', linewidth=1, label='E(t) BT')
        ax2 = ax.twinx()
        ax2.plot(t, H_full, 'r-', linewidth=2, label='H(t) full')
        ax2.plot(t, H_bt, 'r--', linewidth=1, label='H(t) BT')
        ax.set_title(ic_name, fontsize=11)
        ax.set_xlabel('t')
        ax.set_ylabel('E(t)', color='b')
        ax2.set_ylabel('H(t)', color='r')
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')
        ax.grid(True, alpha=0.3)

        # Bottom row: H/E and H_rel
        ax = axes[1, col]
        ax.plot(t, HE_full, 'k-', linewidth=2, label='H/E full')
        ax.plot(t, HE_bt, 'k--', linewidth=1, label='H/E BT')
        ax.plot(t, H_rel_full, 'g-', linewidth=1.5, label='H_rel full')
        ax.plot(t, H_rel_bt, 'g--', linewidth=1, label='H_rel BT')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('t')
        ax.set_ylabel('H/E and H_rel')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'helicity_energy_tracking.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
