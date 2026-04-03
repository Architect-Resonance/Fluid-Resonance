"""
BELTRAMI KERNEL VALIDATION
===========================
Tests the key claims from BELTRAMI_LOOP_HAMILTONIAN.md:

1. ABC flow is exactly Beltrami (nabla x u = lambda u)
2. ABC flow decays at exact rate nu*lambda^2
3. Lamb vector omega x u = 0 identically for Beltrami
4. Wilson loop circulation decays uniformly
5. General flows approach Beltrami alignment at extreme vorticity (anti-twist)
6. BT surgery brings the flow closer to the Beltrami kernel (smaller |Lamb|)

HONEST TEST: Report what the numbers say.
"""

import sys
import os
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS


class BeltramiValidator(SpectralNS):
    """Extended solver with Beltrami-specific diagnostics."""

    def abc_flow_ic(self, A=1.0, B=1.0, C=1.0):
        """ABC flow: an exact Beltrami solution with lambda=1.

        u = (A sin(z) + C cos(y), B sin(x) + A cos(z), C sin(y) + B cos(x))
        Satisfies: nabla x u = u  (lambda = 1)
        """
        u = np.zeros((3,) + self.X.shape)
        u[0] = A * np.sin(self.Z) + C * np.cos(self.Y)
        u[1] = B * np.sin(self.X) + A * np.cos(self.Z)
        u[2] = C * np.sin(self.Y) + B * np.cos(self.X)
        return self.project_leray(np.array([fftn(u[i]) for i in range(3)]))

    def compute_lamb_vector_field(self, u_hat):
        """Compute the Lamb vector field omega x u in physical space.

        Returns:
            lamb: (3, N, N, N) array - the Lamb vector at each grid point
            lamb_mag: (N, N, N) array - |omega x u|
            omega_mag: (N, N, N) array - |omega|
            u_mag: (N, N, N) array - |u|
        """
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])

        # Lamb vector: omega x u
        lamb = np.array([
            omega[1] * u[2] - omega[2] * u[1],
            omega[2] * u[0] - omega[0] * u[2],
            omega[0] * u[1] - omega[1] * u[0],
        ])

        lamb_mag = np.sqrt(np.sum(lamb**2, axis=0))
        omega_mag = np.sqrt(np.sum(omega**2, axis=0))
        u_mag = np.sqrt(np.sum(u**2, axis=0))

        return lamb, lamb_mag, omega_mag, u_mag

    def compute_beltrami_metrics(self, u_hat):
        """Compute all Beltrami-relevant metrics.

        Returns dict with:
            - lamb_rms: RMS of |Lamb vector| (should be 0 for Beltrami)
            - lamb_over_ou: |omega x u| / (|omega| |u|) = sin(angle), per grid point
            - cos_ou: |cos(omega, u)| per grid point (1 = Beltrami)
            - energy: kinetic energy
            - enstrophy: enstrophy
            - helicity: int u . omega
        """
        N = self.N
        lamb, lamb_mag, omega_mag, u_mag = self.compute_lamb_vector_field(u_hat)

        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])

        # Normalized Lamb vector (sin of angle between omega and u)
        denom = omega_mag * u_mag
        sin_angle = np.where(denom > 1e-15, lamb_mag / denom, 0.0)

        # cos(omega, u)
        u_dot_omega = np.sum(u * omega, axis=0)
        cos_angle = np.where(denom > 1e-15, np.abs(u_dot_omega) / denom, 0.0)

        # Global quantities
        energy = 0.5 * np.mean(np.sum(u**2, axis=0))
        enstrophy = 0.5 * np.mean(np.sum(omega**2, axis=0))
        helicity = np.mean(np.sum(u * omega, axis=0))

        return {
            'lamb_rms': float(np.sqrt(np.mean(lamb_mag**2))),
            'lamb_max': float(np.max(lamb_mag)),
            'sin_angle_mean': float(np.mean(sin_angle)),
            'cos_angle_mean': float(np.mean(cos_angle)),
            'energy': float(energy),
            'enstrophy': float(enstrophy),
            'helicity': float(helicity),
            'omega_mag': omega_mag,
            'sin_angle': sin_angle,
            'cos_angle': cos_angle,
        }

    def compute_wilson_loops(self, u_hat, n_loops=10, seed=42):
        """Compute Wilson loop circulation for several test loops.

        Returns array of circulations Gamma[C] for n_loops random circular loops.
        """
        np.random.seed(seed)
        N = self.N
        L = 2 * np.pi

        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])

        circulations = []
        for _ in range(n_loops):
            # Random circular loop in a random plane
            center = np.random.rand(3) * L
            radius = 0.5 + np.random.rand() * 1.0  # radius 0.5 to 1.5
            n_pts = 64
            theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

            # Random plane: two orthogonal vectors
            v1 = np.random.randn(3)
            v1 /= np.linalg.norm(v1)
            v2 = np.random.randn(3)
            v2 -= v2.dot(v1) * v1
            v2 /= np.linalg.norm(v2)

            # Loop points
            pts = center[np.newaxis, :] + radius * (
                np.cos(theta)[:, np.newaxis] * v1[np.newaxis, :] +
                np.sin(theta)[:, np.newaxis] * v2[np.newaxis, :]
            )

            # Compute circulation by trapezoidal rule
            gamma = 0.0
            for k in range(n_pts):
                k_next = (k + 1) % n_pts
                dl = pts[k_next] - pts[k]

                # Interpolate u at midpoint (nearest grid point)
                mid = 0.5 * (pts[k] + pts[k_next])
                ix = int(mid[0] / L * N) % N
                iy = int(mid[1] / L * N) % N
                iz = int(mid[2] / L * N) % N

                u_mid = np.array([u[i, ix, iy, iz] for i in range(3)])
                gamma += u_mid.dot(dl)

            circulations.append(gamma)

        return np.array(circulations)


def run_beltrami_kernel_test():
    """Run all validation tests."""
    print("=" * 78)
    print("BELTRAMI KERNEL VALIDATION")
    print("=" * 78)
    print()

    N = 32
    Re = 400
    nu = 1.0 / Re
    dt = 0.005
    T = 3.0  # shorter run, ABC decays fast
    report_every = 10

    solver = BeltramiValidator(N=N, Re=Re)
    wall_start = clock.time()

    # ================================================================
    # TEST 1: ABC flow is Beltrami (Lamb vector = 0)
    # ================================================================
    print("=" * 78)
    print("TEST 1: ABC flow is Beltrami")
    print("=" * 78)

    u_abc = solver.abc_flow_ic(A=1.0, B=1.0, C=1.0)
    abc_lambda = 1.0  # eigenvalue for standard ABC flow

    m = solver.compute_beltrami_metrics(u_abc)
    print(f"  Lamb vector RMS:    {m['lamb_rms']:.2e} (should be ~0)")
    print(f"  Lamb vector max:    {m['lamb_max']:.2e} (should be ~0)")
    print(f"  sin(omega,u) mean:  {m['sin_angle_mean']:.6f} (should be ~0)")
    print(f"  cos(omega,u) mean:  {m['cos_angle_mean']:.6f} (should be ~1)")
    print(f"  Energy:             {m['energy']:.6f}")
    print(f"  Enstrophy:          {m['enstrophy']:.6f}")
    print(f"  Helicity:           {m['helicity']:.6f}")
    print(f"  H/E = 2*lambda:     {m['helicity']/m['energy']:.6f} (should be {2*abc_lambda:.1f})")

    is_beltrami = m['lamb_rms'] < 1e-10
    print(f"\n  VERDICT: {'PASS' if is_beltrami else 'FAIL'} - ABC flow is {'Beltrami' if is_beltrami else 'NOT Beltrami'}")

    # ================================================================
    # TEST 2: ABC flow decays at rate nu*lambda^2
    # ================================================================
    print(f"\n{'='*78}")
    print("TEST 2: Decay rate = nu * lambda^2")
    print("=" * 78)

    expected_rate = nu * abc_lambda**2
    print(f"  Expected decay rate: nu*lambda^2 = {expected_rate:.6f}")
    print(f"  Expected E(t)/E(0) at t=1: {np.exp(-2*expected_rate*1.0):.6f}")
    print()

    u_hat = u_abc.copy()
    n_steps = int(T / dt)
    times, energies, enstrophies, helicities = [], [], [], []
    lamb_rms_series, cos_mean_series = [], []

    # Also compute Wilson loop circulations at t=0
    gamma_0 = solver.compute_wilson_loops(u_hat, n_loops=10, seed=42)
    gamma_series = [gamma_0.copy()]
    wilson_times = [0.0]

    for step in range(n_steps + 1):
        t = step * dt

        if step % report_every == 0:
            met = solver.compute_beltrami_metrics(u_hat)
            times.append(t)
            energies.append(met['energy'])
            enstrophies.append(met['enstrophy'])
            helicities.append(met['helicity'])
            lamb_rms_series.append(met['lamb_rms'])
            cos_mean_series.append(met['cos_angle_mean'])

            if step % (report_every * 10) == 0:
                E0 = energies[0]
                expected_E = E0 * np.exp(-2 * expected_rate * t)
                ratio = met['energy'] / expected_E if expected_E > 1e-30 else 0
                print(f"  t={t:.2f}: E={met['energy']:.6e}, "
                      f"E_expected={expected_E:.6e}, ratio={ratio:.6f}, "
                      f"|Lamb|={met['lamb_rms']:.2e}")

        if step % (report_every * 20) == 0 and step > 0:
            gamma_t = solver.compute_wilson_loops(u_hat, n_loops=10, seed=42)
            gamma_series.append(gamma_t)
            wilson_times.append(t)

        if step < n_steps:
            u_hat = solver.step_rk4(u_hat, dt, mode='full')

    times = np.array(times)
    energies = np.array(energies)
    enstrophies = np.array(enstrophies)
    helicities = np.array(helicities)
    lamb_rms_series = np.array(lamb_rms_series)

    # Check decay rate
    E0 = energies[0]
    expected_energies = E0 * np.exp(-2 * expected_rate * times)
    max_deviation = np.max(np.abs(energies / expected_energies - 1.0))
    print(f"\n  Max deviation from E(t) = E(0)*exp(-2*nu*lambda^2*t): {max_deviation:.2e}")
    print(f"  VERDICT: {'PASS' if max_deviation < 0.01 else 'FAIL'} - decay rate is {'exact' if max_deviation < 0.01 else 'NOT exact'}")

    # Check H/E conservation
    he_ratio = helicities / energies
    he_deviation = np.max(np.abs(he_ratio / he_ratio[0] - 1.0))
    print(f"\n  H/E ratio at t=0: {he_ratio[0]:.6f}")
    print(f"  H/E max deviation: {he_deviation:.2e}")
    print(f"  VERDICT: {'PASS' if he_deviation < 0.01 else 'FAIL'} - H/E is {'conserved' if he_deviation < 0.01 else 'NOT conserved'}")

    # Check Wilson loop uniform decay
    wilson_times = np.array(wilson_times)
    gamma_series = np.array(gamma_series)  # (n_times, n_loops)
    if len(wilson_times) > 1:
        print(f"\n  Wilson loop circulations (10 random loops):")
        for i_loop in range(min(5, gamma_series.shape[1])):
            g0 = gamma_series[0, i_loop]
            if abs(g0) > 1e-10:
                ratios = gamma_series[:, i_loop] / g0
                expected_ratios = np.exp(-expected_rate * wilson_times)
                max_dev = np.max(np.abs(ratios - expected_ratios))
                print(f"    Loop {i_loop}: Gamma(0)={g0:.4f}, max decay deviation: {max_dev:.2e}")

    # ================================================================
    # TEST 3: Compare Beltrami vs General flow — Lamb vector
    # ================================================================
    print(f"\n{'='*78}")
    print("TEST 3: Beltrami kernel — Lamb vector comparison")
    print("=" * 78)

    # Run Taylor-Green and imbalanced IC for comparison
    ics = {
        'ABC (Beltrami)': solver.abc_flow_ic(),
        'Imbalanced (80/20)': solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8),
        'Taylor-Green': solver.taylor_green_ic(),
    }

    T_test = 2.0
    n_steps_test = int(T_test / dt)

    comparison_data = {}
    for ic_name, u_ic in ics.items():
        print(f"\n  IC: {ic_name}")
        u_hat_f = u_ic.copy()
        u_hat_b = u_ic.copy()

        ts, lamb_f, lamb_b, cos_f, cos_b = [], [], [], [], []
        lamb_extreme_f, lamb_extreme_b = [], []

        for step in range(n_steps_test + 1):
            t = step * dt
            if step % (report_every * 5) == 0:
                met_f = solver.compute_beltrami_metrics(u_hat_f)
                met_b = solver.compute_beltrami_metrics(u_hat_b)

                ts.append(t)
                lamb_f.append(met_f['lamb_rms'])
                lamb_b.append(met_b['lamb_rms'])
                cos_f.append(met_f['cos_angle_mean'])
                cos_b.append(met_b['cos_angle_mean'])

                # Lamb vector at extreme vorticity (>95th pct)
                threshold_f = np.percentile(met_f['omega_mag'], 95)
                threshold_b = np.percentile(met_b['omega_mag'], 95)
                mask_f = met_f['omega_mag'] > threshold_f
                mask_b = met_b['omega_mag'] > threshold_b

                sin_extreme_f = np.mean(met_f['sin_angle'][mask_f]) if np.any(mask_f) else 0
                sin_extreme_b = np.mean(met_b['sin_angle'][mask_b]) if np.any(mask_b) else 0
                lamb_extreme_f.append(sin_extreme_f)
                lamb_extreme_b.append(sin_extreme_b)

            if step < n_steps_test:
                u_hat_f = solver.step_rk4(u_hat_f, dt, mode='full')
                u_hat_b = solver.step_rk4(u_hat_b, dt, mode='bt')

        comparison_data[ic_name] = {
            'times': np.array(ts),
            'lamb_full': np.array(lamb_f),
            'lamb_bt': np.array(lamb_b),
            'cos_full': np.array(cos_f),
            'cos_bt': np.array(cos_b),
            'lamb_extreme_full': np.array(lamb_extreme_f),
            'lamb_extreme_bt': np.array(lamb_extreme_b),
        }

        print(f"    Full NS: |Lamb| = {lamb_f[0]:.4e} => {lamb_f[-1]:.4e}, "
              f"cos = {cos_f[0]:.4f} => {cos_f[-1]:.4f}")
        print(f"    BT surg: |Lamb| = {lamb_b[0]:.4e} => {lamb_b[-1]:.4e}, "
              f"cos = {cos_b[0]:.4f} => {cos_b[-1]:.4f}")
        print(f"    sin(extreme, full): {lamb_extreme_f[0]:.4f} => {lamb_extreme_f[-1]:.4f}")
        print(f"    sin(extreme, BT):   {lamb_extreme_b[0]:.4f} => {lamb_extreme_b[-1]:.4f}")

        # Does BT surgery bring the flow closer to Beltrami?
        if len(lamb_f) > 1 and len(lamb_b) > 1:
            # Compare late-time Lamb vector
            late = len(lamb_f) // 2
            ratio = np.mean(lamb_b[late:]) / np.mean(lamb_f[late:]) if np.mean(lamb_f[late:]) > 1e-15 else 0
            print(f"    BT/Full Lamb ratio (late time): {ratio:.4f}")
            if ratio < 0.9:
                print(f"    => BT brings flow CLOSER to Beltrami kernel")
            else:
                print(f"    => BT does NOT significantly reduce Lamb vector")

    wall_time = clock.time() - wall_start
    print(f"\nTotal wall time: {wall_time:.1f}s")

    # ================================================================
    # PLOTTING
    # ================================================================
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Beltrami Kernel Validation\n"
                 f"N={N}, Re={Re}", fontsize=14, fontweight='bold')

    # Panel 1: ABC energy decay vs theoretical
    ax = axes[0, 0]
    ax.semilogy(times, energies, 'b-', linewidth=2, label='Measured E(t)')
    ax.semilogy(times, expected_energies, 'r--', linewidth=2, label=r'$E_0 e^{-2\nu\lambda^2 t}$')
    ax.set_xlabel('t')
    ax.set_ylabel('Energy')
    ax.set_title('ABC flow: energy decay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Lamb vector RMS for ABC (should stay ~0)
    ax = axes[0, 1]
    ax.semilogy(times, lamb_rms_series + 1e-16, 'b-', linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel('|Lamb vector| RMS')
    ax.set_title('ABC flow: Lamb vector (should be ~0)')
    ax.grid(True, alpha=0.3)

    # Panel 3: H/E ratio conservation
    ax = axes[0, 2]
    ax.plot(times, he_ratio, 'b-', linewidth=2)
    ax.axhline(y=2*abc_lambda, color='r', linestyle='--', label=f'2*lambda={2*abc_lambda}')
    ax.set_xlabel('t')
    ax.set_ylabel('H / E')
    ax.set_title('ABC flow: H/E conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Lamb vector comparison (Full vs BT)
    ax = axes[1, 0]
    colors_ic = ['#2ca02c', '#1f77b4', '#ff7f0e']
    for i, (ic_name, d) in enumerate(comparison_data.items()):
        if ic_name == 'ABC (Beltrami)':
            continue
        ax.plot(d['times'], d['lamb_full'], '-', color=colors_ic[i], linewidth=2,
                label=f'{ic_name} Full')
        ax.plot(d['times'], d['lamb_bt'], '--', color=colors_ic[i], linewidth=2,
                label=f'{ic_name} BT')
    ax.set_xlabel('t')
    ax.set_ylabel('|Lamb vector| RMS')
    ax.set_title('Lamb vector: Full NS vs BT Surgery')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: cos(omega,u) at extreme vorticity (approach to Beltrami)
    ax = axes[1, 1]
    for i, (ic_name, d) in enumerate(comparison_data.items()):
        if ic_name == 'ABC (Beltrami)':
            continue
        ax.plot(d['times'], 1 - np.array(d['lamb_extreme_full']), '-',
                color=colors_ic[i], linewidth=2, label=f'{ic_name} Full (extreme)')
        ax.plot(d['times'], 1 - np.array(d['lamb_extreme_bt']), '--',
                color=colors_ic[i], linewidth=2, label=f'{ic_name} BT (extreme)')
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Beltrami limit')
    ax.set_xlabel('t')
    ax.set_ylabel('1 - sin(omega, u) at >95th pct')
    ax.set_title('Approach to Beltrami at extreme vorticity')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 6: BT/Full Lamb ratio over time
    ax = axes[1, 2]
    for i, (ic_name, d) in enumerate(comparison_data.items()):
        if ic_name == 'ABC (Beltrami)':
            continue
        valid = d['lamb_full'] > 1e-15
        ratio = np.where(valid, d['lamb_bt'] / d['lamb_full'], 1.0)
        ax.plot(d['times'], ratio, '-', color=colors_ic[i], linewidth=2, label=ic_name)
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Equal')
    ax.set_xlabel('t')
    ax.set_ylabel('|Lamb_BT| / |Lamb_Full|')
    ax.set_title('BT surgery reduces Lamb vector?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = 'h:/tmp/beltrami_kernel.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {outpath}")

    return comparison_data


if __name__ == "__main__":
    comparison_data = run_beltrami_kernel_test()
