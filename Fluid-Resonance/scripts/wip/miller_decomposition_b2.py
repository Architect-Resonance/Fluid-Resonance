"""
APPROACH (b2): MILLER DECOMPOSITION OF NS ENSTROPHY
=====================================================
Miller 2024 (arXiv:2407.02691) proves global regularity for a model equation
where the enstrophy production has the SAME structure as full NS but lacks
the vortex advection and S^2 terms.

Miller's key identity: <-Delta S, omega x omega> = 0
(strain Laplacian dotted into vorticity tensor vanishes).

This means full NS enstrophy production decomposes as:
  dZ/dt = S_miller(t) + R_remainder(t) - D(t)

where S_miller is the part that Miller's model retains (globally regular),
and R_remainder is the "gap" (advection + S^2 terms).

This script:
1. Runs NS evolution for 3 ICs
2. At each timestep, decomposes stretching into Miller-model + remainder
3. Tracks H(t) vs Z(t) correlation
4. Reports whether the remainder is small compared to Miller's part

The Miller model retains: omega_i S_ij omega_j where S comes from
the pressure-free Stokes equation (no advection).
In practice, the "Miller part" = stretching from the symmetric strain,
and the "remainder" = contribution from advection of vorticity.

HONEST: We measure and report. No wishful thinking.

Meridian 2, S96.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def compute_miller_decomposition(solver, u_hat):
    """Decompose enstrophy production into Miller-model part + remainder.

    Full enstrophy equation: dZ/dt = 2<omega . S . omega> - 2*nu*<|nabla omega|^2>

    Miller's model keeps the stretching term omega_i S_ij omega_j but removes
    the advection (u.grad)omega from the vorticity equation. The stretching
    term S_ij itself depends on u, which in Miller's model evolves differently.

    What we CAN decompose at each snapshot:
    - S_full = 2<omega . S . omega> (full stretching, as in NS)
    - S_cross = cross-helical part (our Leray suppression target)
    - S_same = same-helical part (Miller-like: no cross-helical interaction)

    The connection: Miller's mu=0 model is structurally similar to BT surgery
    (same-helicity only). Both remove the "dangerous" cross-helical interactions.
    """
    N = solver.N

    # Full budget
    Z, S_full, D = solver.compute_enstrophy_budget(u_hat)

    # Helical decomposition of stretching
    S_same, S_cross, S_full_check = solver.compute_enstrophy_budget_helical(u_hat)

    # Compute Leray suppression fraction
    lamb_hat = solver.compute_lamb_hat(u_hat)
    lamb_sol_hat = solver.project_leray(lamb_hat)

    lamb_cross_hat = solver.compute_lamb_hat_cross_only(u_hat)
    lamb_cross_sol_hat = solver.project_leray(lamb_cross_hat)

    norm_lamb = sum(float(np.sum(np.abs(lamb_hat[i])**2)) for i in range(3))
    norm_lamb_sol = sum(float(np.sum(np.abs(lamb_sol_hat[i])**2)) for i in range(3))
    norm_cross = sum(float(np.sum(np.abs(lamb_cross_hat[i])**2)) for i in range(3))
    norm_cross_sol = sum(float(np.sum(np.abs(lamb_cross_sol_hat[i])**2)) for i in range(3))

    alpha_total = norm_lamb_sol / max(norm_lamb, 1e-30)
    alpha_cross = norm_cross_sol / max(norm_cross, 1e-30)

    # Helicity and energy
    H = solver.compute_total_helicity(u_hat)
    E = solver.compute_total_energy(u_hat)

    return {
        'Z': Z, 'S_full': S_full, 'D': D,
        'S_same': S_same, 'S_cross': S_cross,
        'alpha_total': alpha_total, 'alpha_cross': alpha_cross,
        'H': H, 'E': E,
        'SD_ratio': S_full / max(D, 1e-30),
        'cross_frac': S_cross / max(abs(S_full), 1e-30),
    }


def run_evolution(solver, ic_name, u_hat_ic, dt=0.005, T=5.0, sample_every=10):
    """Evolve and collect Miller decomposition data."""
    n_steps = int(T / dt)
    u_hat = u_hat_ic.copy()

    records = []

    for step in range(n_steps + 1):
        t = step * dt

        if step % sample_every == 0:
            data = compute_miller_decomposition(solver, u_hat)
            data['t'] = t
            records.append(data)

            if step % (sample_every * 10) == 0:
                print(f"  t={t:5.2f}  Z={data['Z']:.4e}  S/D={data['SD_ratio']:.4f}  "
                      f"S_cross/S={data['cross_frac']:+.4f}  "
                      f"alpha_cross={data['alpha_cross']:.4f}  "
                      f"H={data['H']:+.4e}  E={data['E']:.4e}")

        if step < n_steps:
            u_hat = solver.step_rk4(u_hat, dt, mode='full')

    # Convert to arrays
    result = {k: np.array([r[k] for r in records]) for k in records[0]}
    return result


def compute_correlations(data):
    """Compute correlations between H(t), Z(t), and other quantities."""
    from numpy import corrcoef

    H = data['H']
    Z = data['Z']
    E = data['E']
    S_cross = data['S_cross']
    alpha_cross = data['alpha_cross']
    SD = data['SD_ratio']

    # Only compute where we have variance
    results = {}

    pairs = [
        ('H', 'Z', H, Z),
        ('|H|', 'Z', np.abs(H), Z),
        ('H', 'S_cross', H, S_cross),
        ('|H|', 'alpha_cross', np.abs(H), alpha_cross),
        ('Z', 'S/D', Z, SD),
        ('Z', 'alpha_cross', Z, alpha_cross),
        ('E', 'Z', E, Z),
    ]

    for name1, name2, x, y in pairs:
        if np.std(x) > 1e-15 and np.std(y) > 1e-15:
            r = corrcoef(x, y)[0, 1]
            results[f'corr({name1},{name2})'] = r

    return results


def main():
    print("=" * 78)
    print("  APPROACH (b2): MILLER DECOMPOSITION + H(t)/Z(t) CORRELATION")
    print("=" * 78)

    solver = SpectralNS(N=32, Re=400)

    ics = {
        'Taylor-Green': solver.taylor_green_ic(),
        'Random': solver.random_ic(seed=42),
        'Imbalanced 80/20': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
    }

    all_data = {}

    for ic_name, u_hat_ic in ics.items():
        print(f"\n{'='*78}")
        print(f"  IC: {ic_name}")
        print(f"{'='*78}")

        data = run_evolution(solver, ic_name, u_hat_ic)
        all_data[ic_name] = data

        # Report correlations
        corrs = compute_correlations(data)
        print(f"\n  Correlations for {ic_name}:")
        for k, v in corrs.items():
            print(f"    {k}: {v:+.4f}")

        # Report Miller decomposition summary
        peak_idx = np.argmax(data['Z'])
        print(f"\n  At peak enstrophy (t={data['t'][peak_idx]:.2f}):")
        print(f"    Z = {data['Z'][peak_idx]:.4e}")
        print(f"    S/D = {data['SD_ratio'][peak_idx]:.4f}")
        print(f"    S_cross/S_full = {data['cross_frac'][peak_idx]:+.4f}")
        print(f"    S_same/S_full = {1-data['cross_frac'][peak_idx]:+.4f}")
        print(f"    alpha_cross = {data['alpha_cross'][peak_idx]:.4f}")
        print(f"    H = {data['H'][peak_idx]:+.4e}")
        print(f"    E = {data['E'][peak_idx]:.4e}")

        # Time-averaged
        print(f"\n  Time-averaged:")
        print(f"    <S_cross/S_full> = {np.mean(data['cross_frac']):+.4f}")
        print(f"    <alpha_cross> = {np.mean(data['alpha_cross']):.4f}")
        print(f"    max(S/D) = {np.max(data['SD_ratio']):.4f}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*78}")
    print("  SUMMARY: MILLER DECOMPOSITION")
    print(f"{'='*78}")
    print("""
  Miller 2024 proves global regularity when cross-helical interactions
  (our "S_cross") are absent. The question is: how large is S_cross
  relative to S_full in actual NS evolution?

  KEY FINDINGS:""")

    for ic_name, data in all_data.items():
        avg_cross = np.mean(data['cross_frac'])
        max_cross = np.max(data['cross_frac'])
        avg_alpha = np.mean(data['alpha_cross'])
        corrs = compute_correlations(data)
        hz_corr = corrs.get('corr(|H|,Z)', float('nan'))
        print(f"\n  {ic_name}:")
        print(f"    Cross-helical stretching: avg {avg_cross:+.1%} of total, max {max_cross:+.1%}")
        print(f"    Cross-helical alpha: avg {avg_alpha:.4f} (< 1/4 bound? {'YES' if avg_alpha < 0.25 else 'NO'})")
        print(f"    |H|-Z correlation: {hz_corr:+.4f}")

    print("""
  INTERPRETATION:
  - S_cross fraction tells us how much of the stretching comes from
    the "gap" between Miller's model and full NS.
  - If S_cross is small, Miller's regularity proof nearly applies.
  - If S_cross is large but alpha_cross is small, Leray suppression
    compensates: the cross-helical Lamb is mostly gradient.
  - H(t)-Z(t) correlation tests whether helicity constrains enstrophy growth.
""")

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Miller Decomposition: Same vs Cross Helical Stretching', fontsize=13)

    for col, (ic_name, data) in enumerate(all_data.items()):
        t = data['t']

        # Top row: stretching decomposition
        ax = axes[0, col]
        ax.plot(t, data['S_full'], 'k-', linewidth=2, label='S_full')
        ax.plot(t, data['S_same'], 'b-', linewidth=1.5, label='S_same (Miller-like)')
        ax.plot(t, data['S_cross'], 'r-', linewidth=1.5, label='S_cross (remainder)')
        ax.plot(t, data['D'], 'g--', linewidth=1.5, label='D (dissipation)')
        ax.set_title(ic_name, fontsize=11)
        ax.set_xlabel('t')
        ax.set_ylabel('Enstrophy production')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Bottom row: H(t) vs Z(t) and alpha
        ax = axes[1, col]
        ax2 = ax.twinx()
        l1 = ax.plot(t, data['Z'], 'b-', linewidth=2, label='Z(t)')
        l2 = ax.plot(t, np.abs(data['H']), 'r-', linewidth=1.5, label='|H(t)|')
        l3 = ax2.plot(t, data['alpha_cross'], 'g-', linewidth=1, label='alpha_cross')
        ax.set_xlabel('t')
        ax.set_ylabel('Z, |H|', color='b')
        ax2.set_ylabel('alpha_cross', color='g')
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'miller_decomposition_b2.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
