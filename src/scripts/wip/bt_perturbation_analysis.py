"""
ANGLE 2: BT SURGERY PERTURBATION ANALYSIS
==========================================
Key idea: BT surgery (removing cross-helical) is globally regular.
Can we treat cross-helical restoration as a perturbation?

Enstrophy budget: dZ/dt = S_same + S_cross - D

Under BT: S_cross = 0, and BT is regular => S_same < D always (BT margin).
Full NS: if |S_cross| < D - S_same, then dZ/dt < 0 still.

This script measures:
1. BT margin: M = D - S_same under BT dynamics
2. Cross-helical enstrophy production: S_cross under full NS
3. Whether |S_cross| < M (the perturbation is within the safety margin)

If the geometric suppression (alpha = 1-ln2) makes S_cross small enough,
the BT regularity survives cross-helical restoration.

HONEST TEST: We report what the numbers say.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


class EnstrophyBudgetTracker(SpectralNS):
    """Track enstrophy budget decomposed by helicity sector."""

    def compute_enstrophy_budget(self, u_hat):
        """Decompose dZ/dt = S_same + S_cross - D.

        Returns dict with all components.
        """
        N = self.N

        # Enstrophy
        omega_hat = [None]*3
        omega_hat[0] = 1j*(self.ky[None,:,None]*u_hat[2] - self.kz[None,None,:]*u_hat[1])
        omega_hat[1] = 1j*(self.kz[None,None,:]*u_hat[0] - self.kx[:,None,None]*u_hat[2])
        omega_hat[2] = 1j*(self.kx[:,None,None]*u_hat[1] - self.ky[None,:,None]*u_hat[0])

        Z = sum(float(np.sum(np.abs(omega_hat[i])**2)) for i in range(3)) / N**3

        # Dissipation: D = nu * sum |k|^2 |omega|^2
        D = self.nu * sum(float(np.sum(self.k2 * np.abs(omega_hat[i])**2))
                         for i in range(3)) / N**3

        # Helical decomposition
        u_p, u_m = self.helical_decompose(u_hat)

        # Same-helical Lamb: omega_+ x v_+ + omega_- x v_-
        # Cross-helical Lamb: omega_+ x v_- + omega_- x v_+
        def compute_curl(u_hat_sector):
            w = [None]*3
            w[0] = 1j*(self.ky[None,:,None]*u_hat_sector[2] - self.kz[None,None,:]*u_hat_sector[1])
            w[1] = 1j*(self.kz[None,None,:]*u_hat_sector[0] - self.kx[:,None,None]*u_hat_sector[2])
            w[2] = 1j*(self.kx[:,None,None]*u_hat_sector[1] - self.ky[None,:,None]*u_hat_sector[0])
            return w

        w_p = compute_curl(u_p)
        w_m = compute_curl(u_m)

        # Physical space
        w_p_phys = [np.real(ifftn(w_p[i])) for i in range(3)]
        w_m_phys = [np.real(ifftn(w_m[i])) for i in range(3)]
        u_p_phys = [np.real(ifftn(u_p[i])) for i in range(3)]
        u_m_phys = [np.real(ifftn(u_m[i])) for i in range(3)]

        def cross_product_phys(a, b):
            """Compute a x b in physical space."""
            c = [None]*3
            c[0] = a[1]*b[2] - a[2]*b[1]
            c[1] = a[2]*b[0] - a[0]*b[2]
            c[2] = a[0]*b[1] - a[1]*b[0]
            return c

        # Same-helical Lamb (physical space)
        L_same_pp = cross_product_phys(w_p_phys, u_p_phys)
        L_same_mm = cross_product_phys(w_m_phys, u_m_phys)
        L_same_phys = [L_same_pp[i] + L_same_mm[i] for i in range(3)]

        # Cross-helical Lamb (physical space)
        L_cross_pm = cross_product_phys(w_p_phys, u_m_phys)
        L_cross_mp = cross_product_phys(w_m_phys, u_p_phys)
        L_cross_phys = [L_cross_pm[i] + L_cross_mp[i] for i in range(3)]

        # FFT and Leray project
        L_same_hat = [fftn(L_same_phys[i]) for i in range(3)]
        L_cross_hat = [fftn(L_cross_phys[i]) for i in range(3)]

        L_same_sol = self.project_leray(L_same_hat)
        L_cross_sol = self.project_leray(L_cross_hat)

        # Enstrophy production: S = sum_k |k|^2 Re[omega*(k) . P_sol(L(k))]
        # (This is the contribution to dZ/dt from the nonlinear term)
        # Note: dZ/dt = -2 * Re[sum_k |k|^2 u*(k) . P_sol(L(k))] - 2*nu*sum|k|^4|u|^2
        # Actually: using vorticity form:
        # dZ/dt = 2 * Re[sum omega*(k) . FFT(omega x v)(k)] - D
        # But FFT(omega x v) = L_hat, and we need the solenoidal part

        # The enstrophy production from the Lamb vector is:
        # S = Re[sum omega*(k) . (ik x P_sol(L))(k)]
        # Hmm, this is getting complicated. Let me use the direct formula:
        # dZ/dt = sum_k Re[omega_hat*(k) . FFT((omega . nabla)v - (v . nabla)omega)(k)] - D

        # Simpler: use the fact that
        # du/dt = P_sol(L) - nu k^2 u
        # d(omega)/dt = curl(L_sol) - nu k^2 omega
        # dZ/dt = 2 Re[sum omega* . curl(L_sol)] - 2*nu*sum|k|^2|omega|^2

        # curl(L_sol) in Fourier: ik x L_sol_hat
        def curl_hat(v_hat):
            c = [None]*3
            c[0] = 1j*(self.ky[None,:,None]*v_hat[2] - self.kz[None,None,:]*v_hat[1])
            c[1] = 1j*(self.kz[None,None,:]*v_hat[0] - self.kx[:,None,None]*v_hat[2])
            c[2] = 1j*(self.kx[:,None,None]*v_hat[1] - self.ky[None,:,None]*v_hat[0])
            return c

        curl_L_same = curl_hat(L_same_sol)
        curl_L_cross = curl_hat(L_cross_sol)

        # S_same = 2 Re[sum omega* . curl(L_same_sol)] / N^3
        S_same = 2.0 * sum(
            float(np.sum(np.real(np.conj(omega_hat[i]) * curl_L_same[i])))
            for i in range(3)
        ) / N**3

        # S_cross = 2 Re[sum omega* . curl(L_cross_sol)] / N^3
        S_cross = 2.0 * sum(
            float(np.sum(np.real(np.conj(omega_hat[i]) * curl_L_cross[i])))
            for i in range(3)
        ) / N**3

        # Total stretching
        S_total = S_same + S_cross

        # dZ/dt = S_total - D
        dZdt = S_total - D

        # Energy for reference
        u_phys = [np.real(ifftn(u_hat[i])) for i in range(3)]
        E = 0.5 * np.mean(sum(u_phys[i]**2 for i in range(3)))

        # Norms of solenoidal Lamb
        norm_L_same_sol = sum(float(np.sum(np.abs(L_same_sol[i])**2))
                              for i in range(3)) / N**3
        norm_L_cross_sol = sum(float(np.sum(np.abs(L_cross_sol[i])**2))
                               for i in range(3)) / N**3

        return {
            'Z': Z,
            'E': E,
            'D': D,
            'S_same': S_same,
            'S_cross': S_cross,
            'S_total': S_total,
            'dZdt': dZdt,
            'margin': D - S_same,  # BT safety margin
            'ratio': abs(S_cross) / max(D - S_same, 1e-30),  # KEY: is this < 1?
            'norm_L_same_sol': norm_L_same_sol,
            'norm_L_cross_sol': norm_L_cross_sol,
        }


def run_analysis():
    """Run BT perturbation analysis."""
    print("=" * 70)
    print("  ANGLE 2: BT SURGERY PERTURBATION ANALYSIS")
    print("  Is |S_cross| < D - S_same ? (cross-helical within BT margin)")
    print("=" * 70)

    solver = EnstrophyBudgetTracker(N=32, Re=400)

    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Random': solver.random_ic(seed=42),
        'Imbalanced 80/20': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
    }

    dt = 0.005
    n_steps = 2000  # t = 10.0
    sample_every = 40  # every 0.2 time units

    all_results = {}

    for ic_name, u_hat_init in ics.items():
        print(f"\n  IC: {ic_name}")
        print(f"  {'t':>6s}  {'Z':>10s}  {'D':>10s}  {'S_same':>10s}  {'S_cross':>10s}  "
              f"{'Margin':>10s}  {'|Sc|/M':>8s}  {'Safe?':>6s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  "
              f"{'-'*10}  {'-'*8}  {'-'*6}")

        results = {
            'times': [], 'Z': [], 'D': [], 'S_same': [], 'S_cross': [],
            'margin': [], 'ratio': [], 'E': [],
        }

        # Run FULL NS (not BT) — track budget of the actual flow
        u_hat = u_hat_init.copy()
        for step in range(n_steps + 1):
            t = step * dt

            if step % sample_every == 0:
                budget = solver.compute_enstrophy_budget(u_hat)

                results['times'].append(t)
                for key in ['Z', 'D', 'S_same', 'S_cross', 'margin', 'ratio', 'E']:
                    results[key].append(budget[key])

                safe = "YES" if budget['ratio'] < 1.0 else "NO"
                print(f"  {t:6.2f}  {budget['Z']:10.4f}  {budget['D']:10.4f}  "
                      f"{budget['S_same']:10.4f}  {budget['S_cross']:10.4f}  "
                      f"{budget['margin']:10.4f}  {budget['ratio']:8.4f}  {safe:>6s}")

                if budget['E'] < 1e-12:
                    break

            if step < n_steps:
                u_hat = solver.step_rk4(u_hat, dt, mode='full')

        all_results[ic_name] = results

    # =========================================================
    # ANALYSIS
    # =========================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for ic_name, results in all_results.items():
        ratios = np.array(results['ratio'])
        margins = np.array(results['margin'])
        S_cross = np.array(results['S_cross'])

        # Find peak enstrophy time
        Z_arr = np.array(results['Z'])
        peak_idx = np.argmax(Z_arr)
        peak_t = results['times'][peak_idx]

        max_ratio = np.max(ratios)
        mean_ratio = np.mean(ratios)
        frac_safe = np.mean(ratios < 1.0)

        # At peak enstrophy
        peak_ratio = ratios[peak_idx]

        print(f"\n  {ic_name}:")
        print(f"    Peak Z at t={peak_t:.2f}")
        print(f"    |S_cross|/Margin at peak: {peak_ratio:.4f}")
        print(f"    Max |S_cross|/Margin:     {max_ratio:.4f}")
        print(f"    Mean |S_cross|/Margin:    {mean_ratio:.4f}")
        print(f"    Fraction of time safe:    {frac_safe:.1%}")

        if max_ratio < 1.0:
            print(f"    => SAFE: Cross-helical NEVER exceeds BT margin")
        elif frac_safe > 0.9:
            print(f"    => MOSTLY SAFE: Exceeds margin only {1-frac_safe:.1%} of the time")
        else:
            print(f"    => UNSAFE: Cross-helical frequently exceeds BT margin")

    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    all_safe = all(
        np.max(np.array(r['ratio'])) < 1.0
        for r in all_results.values()
    )

    if all_safe:
        print("""
  POSITIVE: Cross-helical enstrophy production |S_cross| stays within
  the BT safety margin D - S_same at ALL times for ALL ICs tested.

  This means: the geometric suppression (alpha = 1-ln2) makes the
  cross-helical perturbation weak enough that BT regularity survives.

  Caveat: This is numerical evidence at N=32, Re=400. Higher Re or
  resolution could change the picture. Need analytical argument.
""")
    else:
        # Find the worst case
        worst_ic = max(all_results.keys(),
                       key=lambda k: np.max(np.array(all_results[k]['ratio'])))
        worst_ratio = np.max(np.array(all_results[worst_ic]['ratio']))
        print(f"""
  MIXED: Cross-helical exceeds BT margin in some cases.
  Worst case: {worst_ic}, max |S_cross|/Margin = {worst_ratio:.4f}

  If ratio >> 1: BT perturbation approach fails — cross-helical
  contribution is too strong relative to the dissipation surplus.

  If ratio is O(1): borderline — geometric suppression helps but
  doesn't fully absorb the cross-helical contribution.
""")

    # =========================================================
    # PLOT
    # =========================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('BT Surgery Perturbation: Is |S_cross| < D - S_same?', fontsize=13)

    for col, (ic_name, results) in enumerate(all_results.items()):
        t = results['times']

        ax = axes[0, col]
        ax.plot(t, results['D'], 'b-', label='D (dissipation)', linewidth=1.5)
        ax.plot(t, results['S_same'], 'g-', label='S_same', linewidth=1.5)
        ax.plot(t, np.abs(results['S_cross']), 'r-', label='|S_cross|', linewidth=1.5)
        ax.fill_between(t, results['S_same'], results['D'], alpha=0.2, color='green',
                        label='BT margin')
        ax.set_xlabel('Time')
        ax.set_ylabel('Enstrophy rate')
        ax.set_title(ic_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(t, results['ratio'], 'k-', linewidth=2)
        ax.axhline(y=1.0, color='red', linestyle='--', label='Safety threshold')
        ax.set_xlabel('Time')
        ax.set_ylabel('|S_cross| / (D - S_same)')
        ax.set_title(f'{ic_name} — Ratio')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(2.0, 1.2 * max(results['ratio'])))

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'bt_perturbation_analysis.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to {plot_path}")
    plt.close()

    # Save data
    save_path = os.path.join(os.path.dirname(__file__), 'bt_perturbation_results.json')
    save_data = {}
    for ic_name, results in all_results.items():
        save_data[ic_name] = {
            'max_ratio': float(np.max(np.array(results['ratio']))),
            'mean_ratio': float(np.mean(np.array(results['ratio']))),
            'peak_Z_ratio': float(np.array(results['ratio'])[np.argmax(np.array(results['Z']))]),
        }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to {save_path}")

    print("\n  DONE.")


if __name__ == '__main__':
    run_analysis()
