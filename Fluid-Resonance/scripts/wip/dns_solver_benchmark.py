"""
VERIFICATION OPTION 2: DNS Solver Benchmark
=============================================
Verify the SpectralNS pseudo-spectral solver against known results:

  Test 1: Energy balance dE/dt = -2*nu*Z (exact NS identity)
  Test 2: Short-time Taylor expansion of TG vortex energy
  Test 3: Resolution convergence (N=32 vs N=48 vs N=64 at same Re)
  Test 4: Inviscid energy conservation (Euler, nu=0 limit)
  Test 5: Divergence-free constraint (Leray projector correctness)

Reference: Brachet et al. 1983, J. Fluid Mech. 130, 411-452

S100-M2d, Meridian 2.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def compute_energy(solver, u_hat):
    """Total kinetic energy E = (1/2) <|u|^2>."""
    u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    return 0.5 * np.mean(np.sum(u**2, axis=0))


def compute_enstrophy(solver, u_hat):
    """Total enstrophy Z = (1/2) <|omega|^2>."""
    omega_hat = solver.compute_vorticity_hat(u_hat)
    omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
    return 0.5 * np.mean(np.sum(omega**2, axis=0))


def compute_divergence_rms(solver, u_hat):
    """RMS of div(u) in Fourier space — should be zero for solenoidal field."""
    div_hat = (1j * solver.kx * u_hat[0] +
               1j * solver.ky * u_hat[1] +
               1j * solver.kz * u_hat[2])
    return np.sqrt(np.mean(np.abs(div_hat)**2))


def compute_max_k_energy(solver, u_hat):
    """Fraction of energy in highest 1/3 of modes (dealiasing zone)."""
    kmax = solver.N // 3
    outer = ~solver.dealias_mask
    E_outer = 0.5 * np.sum(np.sum(np.abs(u_hat[:, outer])**2, axis=0)) / solver.N**6
    E_total = compute_energy(solver, u_hat)
    return E_outer / max(E_total, 1e-30)


def evolve_and_track(solver, u_hat, T, dt, label=""):
    """Evolve u_hat to time T, tracking E, Z, dE/dt, -2*nu*Z at each step."""
    n_steps = int(T / dt)
    times = [0.0]
    energies = [compute_energy(solver, u_hat)]
    enstrophies = [compute_enstrophy(solver, u_hat)]
    div_rms = [compute_divergence_rms(solver, u_hat)]

    for step in range(n_steps):
        u_hat = solver.step_rk4(u_hat, dt)
        t = (step + 1) * dt
        times.append(t)
        energies.append(compute_energy(solver, u_hat))
        enstrophies.append(compute_enstrophy(solver, u_hat))
        div_rms.append(compute_divergence_rms(solver, u_hat))

    times = np.array(times)
    energies = np.array(energies)
    enstrophies = np.array(enstrophies)
    div_rms = np.array(div_rms)

    # Compute dE/dt via central differences
    dEdt = np.gradient(energies, times)
    dissipation = -2 * solver.nu * enstrophies

    return u_hat, times, energies, enstrophies, dEdt, dissipation, div_rms


def main():
    print("=" * 70)
    print("  DNS SOLVER BENCHMARK")
    print("  SpectralNS vs known Taylor-Green vortex results")
    print("=" * 70)

    # ================================================================
    # TEST 1: Energy balance dE/dt = -2*nu*Z
    # ================================================================
    print("\n  TEST 1: Energy balance identity dE/dt = -2*nu*Z")
    print("  " + "-" * 55)

    Re = 400
    N = 48
    solver = SpectralNS(N=N, Re=Re)
    u_hat = solver.taylor_green_ic()
    dt = 0.005
    T = 2.0

    t0 = time.time()
    u_hat_final, times, E, Z, dEdt, diss, div_rms = evolve_and_track(
        solver, u_hat, T, dt, label=f"Re={Re}, N={N}")
    elapsed = time.time() - t0
    print(f"  Evolved Re={Re}, N={N}, T={T}, dt={dt} in {elapsed:.1f}s")

    # Compare dE/dt with -2*nu*Z
    balance_error = np.abs(dEdt - diss)
    # Skip endpoints (gradient artifacts)
    interior = slice(2, -2)
    rms_balance = np.sqrt(np.mean(balance_error[interior]**2))
    max_balance = np.max(balance_error[interior])
    rel_balance = rms_balance / np.max(np.abs(diss[interior]))

    print(f"  Energy balance |dE/dt - (-2*nu*Z)|:")
    print(f"    RMS absolute:  {rms_balance:.2e}")
    print(f"    Max absolute:  {max_balance:.2e}")
    print(f"    RMS relative:  {rel_balance:.2e}")

    if rel_balance < 0.01:
        print(f"  >>> PASS: Energy balance holds to {rel_balance:.1e} relative error")
    else:
        print(f"  >>> CONCERN: Relative error = {rel_balance:.4f}")

    # ================================================================
    # TEST 2: Short-time Taylor expansion
    # ================================================================
    print(f"\n  TEST 2: Short-time TG energy (analytical at t=0)")
    print("  " + "-" * 55)

    # TG vortex: E(0) = 0.125 (for standard normalization u = sin(x)cos(y)cos(z))
    # E(0) = (1/2) * <u^2> = (1/2) * (1/4 + 1/4 + 0) = 1/4 ... wait
    # u1 = sin(x)cos(y)cos(z), u2 = -cos(x)sin(y)cos(z), u3 = 0
    # <u1^2> = <sin^2(x)><cos^2(y)><cos^2(z)> = (1/2)(1/2)(1/2) = 1/8
    # <u2^2> = same = 1/8
    # E(0) = (1/2)(1/8 + 1/8) = 1/8 = 0.125
    E0_exact = 0.125

    # Z(0): omega = curl(u)
    # omega_x = du3/dy - du2/dz = 0 - cos(x)sin(y)(-sin(z)) = cos(x)sin(y)sin(z)
    # omega_y = du1/dz - du3/dx = sin(x)cos(y)(-sin(z)) - 0 = -sin(x)cos(y)sin(z)
    # omega_z = du2/dx - du1/dy = sin(x)sin(y)cos(z) - sin(x)(-sin(y))cos(z)
    #         = sin(x)sin(y)cos(z) + sin(x)sin(y)cos(z) = 2*sin(x)sin(y)cos(z)
    # <omega_x^2> = (1/2)(1/2)(1/2) = 1/8
    # <omega_y^2> = 1/8
    # <omega_z^2> = 4*(1/2)(1/2)(1/2) = 1/2
    # Z(0) = (1/2)(1/8+1/8+1/2) = (1/2)(3/4) = 3/8 = 0.375
    Z0_exact = 0.375

    # dE/dt(0) = -2*nu*Z(0) = -2*(1/Re)*Z0
    dEdt0_exact = -2.0 / Re * Z0_exact

    E0_meas = E[0]
    Z0_meas = Z[0]

    print(f"  E(0):   measured = {E0_meas:.10f},  exact = {E0_exact:.10f},  error = {abs(E0_meas - E0_exact):.2e}")
    print(f"  Z(0):   measured = {Z0_meas:.10f},  exact = {Z0_exact:.10f},  error = {abs(Z0_meas - Z0_exact):.2e}")
    print(f"  dE/dt(0): measured = {dEdt[0]:.6e},  exact = {dEdt0_exact:.6e}")

    if abs(E0_meas - E0_exact) < 1e-8:
        print(f"  >>> PASS: E(0) matches to machine precision")
    else:
        print(f"  >>> CONCERN: E(0) error = {abs(E0_meas - E0_exact):.2e}")

    if abs(Z0_meas - Z0_exact) < 1e-8:
        print(f"  >>> PASS: Z(0) matches to machine precision")
    else:
        print(f"  >>> CONCERN: Z(0) error = {abs(Z0_meas - Z0_exact):.2e}")

    # Short-time expansion: E(t) ~ E0 + dEdt0 * t + (1/2) d2E/dt2 * t^2
    # d2E/dt2 = -2*nu*dZ/dt. At t=0 for TG: dZ/dt = -2*nu*P(0) + N(0)
    # where P = palinstrophy, N = nonlinear enstrophy production.
    # Rather than computing these analytically, let's just check that
    # E(small t) ~ E0 - (2*nu*Z0)*t to first order
    t_check = 0.05
    idx_check = int(t_check / dt)
    E_linear = E0_exact + dEdt0_exact * times[idx_check]
    E_meas_check = E[idx_check]
    print(f"\n  Linear check at t={times[idx_check]:.3f}:")
    print(f"    E(t) measured:   {E_meas_check:.10f}")
    print(f"    E(0) - 2*nu*Z0*t: {E_linear:.10f}")
    print(f"    Difference:      {abs(E_meas_check - E_linear):.2e} (expected O(t^2) ~ {times[idx_check]**2:.2e})")

    # ================================================================
    # TEST 3: Resolution convergence
    # ================================================================
    print(f"\n  TEST 3: Resolution convergence (Re={Re})")
    print("  " + "-" * 55)

    resolutions = [32, 48, 64]
    E_final = {}
    Z_final = {}

    for N_test in resolutions:
        s = SpectralNS(N=N_test, Re=Re)
        uh = s.taylor_green_ic()
        T_conv = 1.0
        n_steps = int(T_conv / dt)
        for _ in range(n_steps):
            uh = s.step_rk4(uh, dt)
        E_final[N_test] = compute_energy(s, uh)
        Z_final[N_test] = compute_enstrophy(s, uh)
        print(f"  N={N_test:3d}: E(T={T_conv}) = {E_final[N_test]:.10f},  Z = {Z_final[N_test]:.8f}")

    # Convergence: compare N=32 and N=64 to N=48
    if len(resolutions) >= 3:
        dE_32_48 = abs(E_final[32] - E_final[48])
        dE_48_64 = abs(E_final[48] - E_final[64])
        if dE_32_48 > 1e-15:
            conv_rate = np.log2(dE_32_48 / max(dE_48_64, 1e-15))
            print(f"\n  |E(N=32)-E(N=48)| = {dE_32_48:.2e}")
            print(f"  |E(N=48)-E(N=64)| = {dE_48_64:.2e}")
            print(f"  Convergence rate (log2 ratio): {conv_rate:.1f}")
            if dE_48_64 < dE_32_48:
                print(f"  >>> PASS: Resolution convergence confirmed")
            else:
                print(f"  >>> CONCERN: Not converging")
        else:
            print(f"  >>> N=32 and N=48 already agree to machine precision")

    # ================================================================
    # TEST 4: Inviscid energy conservation (Euler)
    # ================================================================
    print(f"\n  TEST 4: Inviscid energy conservation (Re=1e10 ~ Euler)")
    print("  " + "-" * 55)

    Re_euler = int(1e10)
    solver_e = SpectralNS(N=48, Re=Re_euler)
    u_hat_e = solver_e.taylor_green_ic()
    E0_euler = compute_energy(solver_e, u_hat_e)

    T_euler = 1.0
    n_euler = int(T_euler / dt)
    for _ in range(n_euler):
        u_hat_e = solver_e.step_rk4(u_hat_e, dt)

    E_final_euler = compute_energy(solver_e, u_hat_e)
    dE_euler = abs(E_final_euler - E0_euler)
    rel_dE_euler = dE_euler / E0_euler

    print(f"  E(0) = {E0_euler:.10f}")
    print(f"  E(T={T_euler}) = {E_final_euler:.10f}")
    print(f"  |Delta E|/E(0) = {rel_dE_euler:.2e}")

    if rel_dE_euler < 1e-4:
        print(f"  >>> PASS: Energy conserved to {rel_dE_euler:.1e}")
    else:
        print(f"  >>> CONCERN: Energy drift = {rel_dE_euler:.4f}")

    # ================================================================
    # TEST 5: Divergence-free constraint
    # ================================================================
    print(f"\n  TEST 5: Divergence-free constraint")
    print("  " + "-" * 55)

    max_div = np.max(div_rms)
    final_div = div_rms[-1]
    print(f"  Max div(u) RMS over evolution:  {max_div:.2e}")
    print(f"  Final div(u) RMS:               {final_div:.2e}")

    if max_div < 1e-10:
        print(f"  >>> PASS: Divergence-free to machine precision")
    else:
        print(f"  >>> CONCERN: div(u) RMS = {max_div:.2e}")

    # ================================================================
    # TEST 6: Known TG dissipation rate comparison
    # ================================================================
    print(f"\n  TEST 6: TG vortex dissipation rate profile")
    print("  " + "-" * 55)

    # At Re=400, the dissipation rate epsilon(t) = 2*nu*Z(t) should peak
    # around t ~ 5-9 (Brachet 1983). But that requires long evolution.
    # At T=2, we can check that epsilon is INCREASING (pre-peak).
    epsilon = 2 * solver.nu * Z
    eps_increasing = epsilon[-1] > epsilon[0]
    print(f"  epsilon(0) = {epsilon[0]:.6e}")
    print(f"  epsilon(T={T}) = {epsilon[-1]:.6e}")
    print(f"  epsilon increasing: {eps_increasing} (expected: True for t < t_peak)")

    if eps_increasing:
        print(f"  >>> PASS: Dissipation rate increasing before peak (consistent with Brachet 1983)")
    else:
        print(f"  >>> CONCERN: Dissipation should increase in early phase")

    # Energy decay monotonic?
    E_mono = np.all(np.diff(E) <= 1e-15)
    print(f"  Energy monotonically decreasing: {E_mono}")
    if E_mono:
        print(f"  >>> PASS: Second law satisfied (viscous case)")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    tests = [
        ("Energy balance dE/dt = -2*nu*Z", rel_balance < 0.01),
        ("E(0) = 1/8 exact", abs(E0_meas - E0_exact) < 1e-8),
        ("Z(0) = 3/8 exact", abs(Z0_meas - Z0_exact) < 1e-8),
        ("Resolution convergence", dE_48_64 < dE_32_48 if len(resolutions) >= 3 else True),
        ("Inviscid energy conservation", rel_dE_euler < 1e-4),
        ("Divergence-free", max_div < 1e-10),
        ("Dissipation increasing pre-peak", eps_increasing),
        ("Energy monotone decreasing", E_mono),
    ]

    all_pass = True
    for name, passed in tests:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    if all_pass:
        print(f"\n  >>> ALL {len(tests)} TESTS PASS. Solver is trustworthy. <<<")
    else:
        n_fail = sum(1 for _, p in tests if not p)
        print(f"\n  >>> {n_fail}/{len(tests)} tests FAILED. Investigate. <<<")

    # ================================================================
    # 4-panel figure
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DNS Solver Benchmark: Taylor-Green Vortex', fontsize=14)

    # Panel 1: E(t) and Z(t)
    ax = axes[0, 0]
    ax.plot(times, E, 'b-', lw=2, label=f'E(t), Re={Re}')
    ax2 = ax.twinx()
    ax2.plot(times, Z, 'r-', lw=2, label='Z(t)')
    ax.set_xlabel('t')
    ax.set_ylabel('Energy E(t)', color='b')
    ax2.set_ylabel('Enstrophy Z(t)', color='r')
    ax.set_title(f'TG vortex evolution (N={N}, Re={Re})')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Panel 2: Energy balance
    ax = axes[0, 1]
    ax.plot(times[interior], dEdt[interior], 'b-', lw=1.5, label='dE/dt (numerical)')
    ax.plot(times[interior], diss[interior], 'r--', lw=1.5, label='-2*nu*Z')
    ax.set_xlabel('t')
    ax.set_ylabel('Energy rate')
    ax.set_title(f'Energy balance (rel error = {rel_balance:.1e})')
    ax.legend()

    # Panel 3: Divergence over time
    ax = axes[1, 0]
    ax.semilogy(times, div_rms + 1e-20, 'g-', lw=1.5)
    ax.set_xlabel('t')
    ax.set_ylabel('RMS div(u)')
    ax.set_title('Divergence-free constraint')
    ax.set_ylim(1e-18, 1e-8)

    # Panel 4: Resolution convergence E(t) curves
    ax = axes[1, 1]
    for N_test in resolutions:
        s = SpectralNS(N=N_test, Re=Re)
        uh = s.taylor_green_ic()
        ts_conv = [0.0]
        Es_conv = [compute_energy(s, uh)]
        T_conv = 1.0
        n_conv = int(T_conv / dt)
        for step in range(n_conv):
            uh = s.step_rk4(uh, dt)
            if (step + 1) % 10 == 0:
                ts_conv.append((step + 1) * dt)
                Es_conv.append(compute_energy(s, uh))
        ax.plot(ts_conv, Es_conv, lw=1.5, label=f'N={N_test}')
    ax.set_xlabel('t')
    ax.set_ylabel('E(t)')
    ax.set_title(f'Resolution convergence (Re={Re})')
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'dns_solver_benchmark.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved to {out_path}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
