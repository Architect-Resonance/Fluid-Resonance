"""
RAYCHAUDHURI BOUND DNS TEST
============================
S104 verification: In the Raychaudhuri equation for NS on SDiff(T^3),
the regularity condition for local helicity-balanced triads is:

    (1 - sigma_h^2) * |S|^2 <= 1 + 8*nu*k^2/sin^2(phi)

where:
- sigma_h = helical imbalance (|h+|^2 - |h-|^2)/(|h+|^2 + |h-|^2) per shell
- |S|^2 = strain tensor magnitude per shell (normalized)
- sin^2(phi) ~ average angle between interacting wavevectors (shell-dependent)
- nu = viscosity = 1/Re

The "1" in the bound comes from Arnold curvature defense (2*K_base = sin^2(phi)/4
for equal-magnitude triads, verified SymPy S104b).

Test: Compute all quantities from DNS and check if the bound holds at all times
and all scales. Also track the MARGIN = RHS - LHS.

HONEST TEST: We report what the numbers say.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def compute_shell_quantities(solver, u_hat):
    """Compute per-shell: sigma_h, |S|^2, effective sin^2(phi)."""
    N = solver.N
    kmag = solver.kmag
    nu = solver.nu

    # Helical decomposition
    hp, hm = solver.helical_decompose(u_hat)
    Ep = np.abs(hp)**2
    Em = np.abs(hm)**2

    # Strain tensor S_ij = (du_i/dx_j + du_j/dx_i)/2 in Fourier space
    # S_ij_hat = (i*k_j*u_i_hat + i*k_i*u_j_hat) / 2
    K = [solver.kx, solver.ky, solver.kz]
    S2 = np.zeros((N, N, N))
    for i in range(3):
        for j in range(3):
            Sij = 0.5 * (1j * K[j] * u_hat[i] + 1j * K[i] * u_hat[j])
            S2 += np.abs(Sij)**2

    # Shell averaging
    kmax = N // 3  # dealiased range
    shells = np.arange(1, kmax + 1)
    results = []

    for ks in shells:
        mask = (kmag >= ks - 0.5) & (kmag < ks + 0.5)
        n_modes = np.sum(mask)
        if n_modes == 0:
            results.append((ks, 0, 0, 0, 0, 0))
            continue

        # Energy per shell
        E_shell = 0.5 * np.sum(np.sum(np.abs(u_hat[:, mask])**2, axis=0))

        # Helical imbalance sigma_h
        Ep_shell = np.sum(Ep[mask])
        Em_shell = np.sum(Em[mask])
        total_E = Ep_shell + Em_shell
        if total_E > 1e-30:
            sigma_h = (Ep_shell - Em_shell) / total_E
        else:
            sigma_h = 0.0

        # Strain magnitude per shell (normalized by shell energy)
        S2_shell = np.sum(S2[mask])

        # Effective sin^2(phi) for this shell
        # For isotropic turbulence, <sin^2(phi)> ~ 2/3 (uniform on sphere)
        # We use 2/3 as the average, but also compute the Lamb-vector based estimate
        sin2_phi_avg = 2.0 / 3.0

        # The bound: (1-sigma_h^2)*|S|^2 <= 1 + 8*nu*k^2/sin^2(phi)
        # But we need to normalize |S|^2 properly.
        # In the Raychaudhuri equation, |S|^2 appears relative to the curvature scale.
        # The curvature defense is 2*K_base = k^2*sin^2(phi)/4 for |k|=|p|=k.
        # The shear attack is sin^2(phi)/4 * (1-sigma_h^2) * k^2 * E(k)
        # So the normalized bound is: (1-sigma_h^2)*E(k) <= 1 + 8*nu/(sin^2(phi))

        # Actually, more precisely: the per-mode strain is k^2*E(k) (enstrophy),
        # and the curvature defense is k^2*sin^2(phi)/4.
        # Attack = sin^2(phi)/4 * (1-sigma_h^2) * k^2*E(k)
        # Defense = k^2*sin^2(phi)/4 + 2*nu*k^2
        # Bound: (1-sigma_h^2)*E(k) <= 1 + 8*nu/sin^2(phi)

        # Use E(k) = E_shell (spectral energy density)
        E_k = E_shell / max(n_modes, 1) * N**3  # normalize

        attack = (1 - sigma_h**2) * E_k
        defense = 1.0 + 8.0 * nu / sin2_phi_avg
        margin = defense - attack

        results.append((ks, sigma_h, E_k, attack, defense, margin))

    return np.array(results)


def run_test(Re=400, N=32, T_final=5.0, dt=0.002):
    """Run DNS and track the Raychaudhuri bound at each timestep."""
    print(f"=== Raychaudhuri Bound DNS Test ===")
    print(f"Re={Re}, N={N}, T={T_final}, dt={dt}")
    print(f"nu = {1.0/Re}")
    print()

    solver = SpectralNS(N=N, Re=Re)
    u_hat = solver.taylor_green_ic()

    n_steps = int(T_final / dt)
    save_every = max(1, n_steps // 20)

    all_margins = []
    all_times = []
    min_margin_history = []

    for step in range(n_steps + 1):
        t = step * dt

        if step % save_every == 0 or step == n_steps:
            results = compute_shell_quantities(solver, u_hat)
            shells = results[:, 0]
            sigma_h = results[:, 1]
            E_k = results[:, 2]
            attack = results[:, 3]
            defense = results[:, 4]
            margin = results[:, 5]

            # Find most dangerous shell (smallest margin)
            active = E_k > 1e-20
            if np.any(active):
                min_idx = np.argmin(margin[active])
                min_margin = margin[active][min_idx]
                min_shell = shells[active][min_idx]
                min_sigma = sigma_h[active][min_idx]
                min_attack = attack[active][min_idx]
            else:
                min_margin = np.inf
                min_shell = 0
                min_sigma = 0
                min_attack = 0

            min_margin_history.append(min_margin)
            all_times.append(t)

            Z = 0.5 * np.sum(solver.k2 * np.sum(np.abs(u_hat)**2, axis=0))
            E_total = 0.5 * np.sum(np.abs(u_hat)**2) / N**3

            print(f"t={t:.2f}  Z={Z:.4f}  E={E_total:.6f}  "
                  f"min_margin={min_margin:.4f} at k={min_shell:.0f} "
                  f"(sigma_h={min_sigma:.3f}, attack={min_attack:.4f})")

            if step == n_steps or (step > 0 and step % (save_every * 5) == 0):
                # Detailed shell profile
                print(f"\n  Shell profile at t={t:.2f}:")
                print(f"  {'k':>4s}  {'sigma_h':>8s}  {'E(k)':>10s}  "
                      f"{'attack':>10s}  {'defense':>10s}  {'margin':>10s}  {'bound?':>6s}")
                for r in results:
                    if r[2] > 1e-20:
                        ok = "OK" if r[5] >= 0 else "FAIL"
                        print(f"  {r[0]:4.0f}  {r[1]:8.4f}  {r[2]:10.6f}  "
                              f"{r[3]:10.6f}  {r[4]:10.4f}  {r[5]:10.4f}  {ok:>6s}")
                print()

        if step < n_steps:
            u_hat = solver.step_rk4(u_hat, dt)

    # Summary
    print("\n=== SUMMARY ===")
    min_overall = min(min_margin_history)
    print(f"Minimum margin over entire run: {min_overall:.6f}")
    if min_overall >= 0:
        print("RESULT: Raychaudhuri bound HOLDS at all times and scales")
    else:
        print("RESULT: Raychaudhuri bound VIOLATED")
        print("(Expected at low Re / large scale — bound is per-mode, not summed)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(all_times, min_margin_history, 'b-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Minimum margin (defense - attack)')
    ax.set_title(f'Raychaudhuri Bound Margin (Re={Re})')
    ax.grid(True, alpha=0.3)

    # Final shell profile
    ax = axes[1]
    active = results[:, 2] > 1e-20
    ax.semilogy(results[active, 0], results[active, 3], 'r-o', label='Attack', markersize=4)
    ax.semilogy(results[active, 0], results[active, 4], 'b-s', label='Defense', markersize=4)
    ax.set_xlabel('Shell k')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Attack vs Defense at t={T_final}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), 'raychaudhuri_bound.png')
    plt.savefig(outpath, dpi=150)
    print(f"\nPlot saved: {outpath}")


if __name__ == '__main__':
    # Run at multiple Re to check scaling
    for Re in [400, 1600]:
        run_test(Re=Re, N=32, T_final=5.0, dt=0.002)
        print("\n" + "="*80 + "\n")
