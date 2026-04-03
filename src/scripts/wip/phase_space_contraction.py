"""
TASK 1: Direct measurement of phase-space contraction rate sigma_dot_K
======================================================================

sigma_dot_K = Sum_{|k|~K} dF_k/du_k  (Jacobian trace at shell K)

For incompressible NS with zero mean flow, the nonlinear Jacobian diagonal
vanishes identically:
  dN_k/du_k involves triads k = p + q where p=k,q=0 or q=k,p=0
  Since u_0 = 0 (zero mean), these terms vanish.
  Therefore sigma_dot_K = -2*nu * Z_K  (viscous dissipation only)

We VERIFY this numerically via finite differences, then compute the
MORE interesting quantity: finite-time Lyapunov exponents (FTLE) per shell.
The FTLE measures actual stretching rates and connects to Tanogami-Araki's
information flow rate I_K.

Measurements:
  A) Jacobian trace per shell (expect: -2*nu*Z_K exactly)
  B) Jacobian eigenvalues for Fano shell |k|<=3 (stretching vs compression)
  C) FTLE per shell (perturbation growth rate)
  D) Compare FTLE with 1/tau_K
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock

def P(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ========== DNS ENGINE (shared with other scripts) ==========

class SpectralNS:
    def __init__(self, N, nu):
        self.N = N
        self.nu = nu
        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0,0,0] = 1.0
        self.kmag = np.sqrt(self.k2_safe)
        kmax = N // 3
        self.mask = ((np.abs(self.kx) <= kmax) &
                     (np.abs(self.ky) <= kmax) &
                     (np.abs(self.kz) <= kmax)).astype(float)

    def leray_project(self, fx, fy, fz):
        kdotf = self.kx*fx + self.ky*fy + self.kz*fz
        fx = fx - self.kx * kdotf / self.k2_safe
        fy = fy - self.ky * kdotf / self.k2_safe
        fz = fz - self.kz * kdotf / self.k2_safe
        return fx, fy, fz

    def taylor_green_ic(self):
        N = self.N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        ux = np.sin(X) * np.cos(Y) * np.cos(Z)
        uy = -np.cos(X) * np.sin(Y) * np.cos(Z)
        uz = np.zeros_like(X)
        ux_hat = fftn(ux) * self.mask
        uy_hat = fftn(uy) * self.mask
        uz_hat = fftn(uz) * self.mask
        return self.leray_project(ux_hat, uy_hat, uz_hat)

    def nonlinear(self, ux_hat, uy_hat, uz_hat):
        m = self.mask
        ux = np.real(ifftn(ux_hat * m))
        uy = np.real(ifftn(uy_hat * m))
        uz = np.real(ifftn(uz_hat * m))
        ikx, iky, ikz = 1j*self.kx, 1j*self.ky, 1j*self.kz
        dux_dx = np.real(ifftn(ikx*ux_hat*m))
        dux_dy = np.real(ifftn(iky*ux_hat*m))
        dux_dz = np.real(ifftn(ikz*ux_hat*m))
        duy_dx = np.real(ifftn(ikx*uy_hat*m))
        duy_dy = np.real(ifftn(iky*uy_hat*m))
        duy_dz = np.real(ifftn(ikz*uy_hat*m))
        duz_dx = np.real(ifftn(ikx*uz_hat*m))
        duz_dy = np.real(ifftn(iky*uz_hat*m))
        duz_dz = np.real(ifftn(ikz*uz_hat*m))
        nlx = -(ux*dux_dx + uy*dux_dy + uz*dux_dz)
        nly = -(ux*duy_dx + uy*duy_dy + uz*duy_dz)
        nlz = -(ux*duz_dx + uy*duz_dy + uz*duz_dz)
        nlx_hat = fftn(nlx) * m
        nly_hat = fftn(nly) * m
        nlz_hat = fftn(nlz) * m
        return self.leray_project(nlx_hat, nly_hat, nlz_hat)

    def rhs(self, ux, uy, uz):
        """Full RHS: F_k = NL_k - nu*k^2*u_k"""
        nlx, nly, nlz = self.nonlinear(ux, uy, uz)
        return (nlx - self.nu*self.k2*ux,
                nly - self.nu*self.k2*uy,
                nlz - self.nu*self.k2*uz)

    def rk4_step(self, ux, uy, uz, dt):
        k1x, k1y, k1z = self.rhs(ux, uy, uz)
        k2x, k2y, k2z = self.rhs(ux+dt/2*k1x, uy+dt/2*k1y, uz+dt/2*k1z)
        k3x, k3y, k3z = self.rhs(ux+dt/2*k2x, uy+dt/2*k2y, uz+dt/2*k2z)
        k4x, k4y, k4z = self.rhs(ux+dt*k3x, uy+dt*k3y, uz+dt*k3z)
        ux_new = ux + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
        uy_new = uy + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
        uz_new = uz + dt/6*(k1z + 2*k2z + 2*k3z + k4z)
        return self.leray_project(ux_new, uy_new, uz_new)


def get_shell_modes(ns, shell_k):
    """Get indices of modes in shell k (k-0.5 < |k| <= k+0.5)"""
    kmag = ns.kmag
    mask = ns.mask > 0
    in_shell = mask & (kmag > shell_k - 0.5) & (kmag <= shell_k + 0.5)
    return np.where(in_shell)


# ========== MEASUREMENT A: Jacobian trace per shell ==========

def measure_jacobian_trace(ns, ux, uy, uz, max_shell=10, eps=1e-7):
    """
    Compute Tr(dF_k/du_k) for each shell via central finite differences.
    Also compute analytical viscous prediction: -2*nu*Z_K.
    """
    P("  Computing Jacobian trace per shell (finite differences)...")
    N = ns.N
    norm = 1.0 / N**6

    results = []
    for sk in range(1, max_shell + 1):
        idx = get_shell_modes(ns, sk)
        n_modes = len(idx[0])
        if n_modes == 0:
            continue

        # Analytical viscous prediction
        k2_shell = ns.k2[idx]
        Z_shell = np.sum((np.abs(ux[idx])**2 + np.abs(uy[idx])**2 + np.abs(uz[idx])**2) * k2_shell) * norm
        sigma_visc = -2.0 * ns.nu * np.sum(k2_shell) * n_modes  # WRONG — need actual enstrophy per mode

        # Actually: sigma_dot_K^visc = sum over modes in K of (-nu*|k|^2) per component (x,y,z)
        # Each mode contributes -nu*|k|^2 for EACH of 3 components
        # But for complex modes, Re and Im parts both contribute
        # Total: sigma_dot_K^visc = -2 * nu * sum_{|k|~K} |k|^2 * 3 (3D, real+imag → factor 2 absorbed)
        # Wait — need to be careful about real vs complex DOF

        # For a real velocity field, u_hat(-k) = conj(u_hat(k))
        # So independent real DOF at shell K = 2 * n_modes * 3 (Re and Im of 3 components)
        # (minus conjugate pairs, but for the trace this doesn't matter since
        #  the trace counts each independent DOF once)

        # Simpler: sigma_dot = sum_k div(F) where the sum is over ALL Fourier modes
        # For each mode k, each component alpha:
        #   dF_k^alpha / du_k^alpha = d(NL_k^alpha)/d(u_k^alpha) - nu*|k|^2
        # The NL diagonal vanishes (zero mean). So:
        #   sigma_dot_K = sum_{|k|~K} 3 * (-nu*|k|^2)  [3 components]
        # But this double-counts conjugate pairs. For k != -k:
        #   modes k and -k are not independent. The real DOF is 6 per (k,-k) pair.
        # For the TRACE of the Jacobian in real variables, we get:
        #   sigma_dot_K = sum_{independent k in shell K} 2 * 3 * (-nu*|k|^2)
        # = -6*nu * sum_{independent k} |k|^2

        # Let's just do it numerically and compare.

        # Numerical: perturb each mode, measure dF/du diagonal
        # Sample a subset of modes for efficiency
        n_sample = min(n_modes, 50)
        sample_idx = np.random.choice(n_modes, n_sample, replace=False)

        trace_NL = 0.0
        trace_visc_check = 0.0

        for s in range(n_sample):
            i, j, l = idx[0][sample_idx[s]], idx[1][sample_idx[s]], idx[2][sample_idx[s]]
            k2_val = ns.k2[i, j, l]

            for comp in range(3):  # x, y, z components
                # Perturb +eps
                ux_p, uy_p, uz_p = ux.copy(), uy.copy(), uz.copy()
                if comp == 0: ux_p[i,j,l] += eps
                elif comp == 1: uy_p[i,j,l] += eps
                else: uz_p[i,j,l] += eps

                # Perturb -eps
                ux_m, uy_m, uz_m = ux.copy(), uy.copy(), uz.copy()
                if comp == 0: ux_m[i,j,l] -= eps
                elif comp == 1: uy_m[i,j,l] -= eps
                else: uz_m[i,j,l] -= eps

                # Full RHS at both perturbations
                fp_x, fp_y, fp_z = ns.rhs(ux_p, uy_p, uz_p)
                fm_x, fm_y, fm_z = ns.rhs(ux_m, uy_m, uz_m)

                # Central difference for diagonal element
                if comp == 0:
                    dF_du = (fp_x[i,j,l] - fm_x[i,j,l]) / (2*eps)
                elif comp == 1:
                    dF_du = (fp_y[i,j,l] - fm_y[i,j,l]) / (2*eps)
                else:
                    dF_du = (fp_z[i,j,l] - fm_z[i,j,l]) / (2*eps)

                trace_NL += np.real(dF_du) + ns.nu * k2_val  # subtract viscous to isolate NL
                trace_visc_check += -ns.nu * k2_val

        # Scale from sample to full shell
        scale = n_modes / n_sample
        trace_NL_full = trace_NL * scale
        trace_visc_full = trace_visc_check * scale
        trace_total = (trace_NL + trace_visc_check) * scale

        # Analytical prediction
        k2_sum = np.sum(k2_shell)
        sigma_visc_analytical = -3.0 * ns.nu * k2_sum  # 3 components per mode

        results.append({
            'k': sk,
            'n_modes': n_modes,
            'trace_total': np.real(trace_total),
            'trace_NL': np.real(trace_NL_full),
            'trace_visc_num': np.real(trace_visc_full),
            'sigma_visc_analytical': sigma_visc_analytical,
            'Z_shell': Z_shell,
            'k2_sum': k2_sum,
        })

    return results


# ========== MEASUREMENT B: FTLE per shell ==========

def measure_ftle(ns, ux, uy, uz, max_shell=10, n_perturbations=5, dt_ftle=0.01, n_steps_ftle=10):
    """
    Finite-Time Lyapunov Exponent per shell.

    Perturb u at shell K only, evolve for short time, measure growth rate.
    This gives the actual stretching rate that Tanogami-Araki connect to I_K.
    """
    P("  Computing FTLE per shell...")
    N = ns.N
    norm = 1.0 / N**6
    dt = dt_ftle / n_steps_ftle  # Short integration steps

    total_time = dt * n_steps_ftle

    results = []
    for sk in range(1, max_shell + 1):
        idx = get_shell_modes(ns, sk)
        n_modes = len(idx[0])
        if n_modes == 0:
            continue

        # Shell energy for tau_K computation
        E_shell = np.sum(np.abs(ux[idx])**2 + np.abs(uy[idx])**2 + np.abs(uz[idx])**2) * norm
        u_shell = np.sqrt(2 * E_shell / n_modes) if n_modes > 0 else 0
        tau_K = 1.0 / (sk * u_shell) if u_shell > 0 else np.inf

        ftle_vals = []
        for _ in range(n_perturbations):
            # Random perturbation at shell K only
            eps = 1e-8
            delta_ux = np.zeros_like(ux)
            delta_uy = np.zeros_like(uy)
            delta_uz = np.zeros_like(uz)

            # Random complex perturbation
            pert_x = (np.random.randn(n_modes) + 1j*np.random.randn(n_modes))
            pert_y = (np.random.randn(n_modes) + 1j*np.random.randn(n_modes))
            pert_z = (np.random.randn(n_modes) + 1j*np.random.randn(n_modes))

            delta_ux[idx] = pert_x
            delta_uy[idx] = pert_y
            delta_uz[idx] = pert_z

            # Project to solenoidal
            delta_ux, delta_uy, delta_uz = ns.leray_project(delta_ux, delta_uy, delta_uz)

            # Normalize
            amp = np.sqrt(np.sum(np.abs(delta_ux)**2 + np.abs(delta_uy)**2 + np.abs(delta_uz)**2))
            delta_ux *= eps / amp
            delta_uy *= eps / amp
            delta_uz *= eps / amp

            # Initial separation (at shell K only)
            d0 = np.sqrt(np.sum(np.abs(delta_ux[idx])**2 + np.abs(delta_uy[idx])**2 + np.abs(delta_uz[idx])**2))

            # Evolve both base and perturbed
            ux_b, uy_b, uz_b = ux.copy(), uy.copy(), uz.copy()
            ux_p, uy_p, uz_p = ux + delta_ux, uy + delta_uy, uz + delta_uz

            for step in range(n_steps_ftle):
                ux_b, uy_b, uz_b = ns.rk4_step(ux_b, uy_b, uz_b, dt)
                ux_p, uy_p, uz_p = ns.rk4_step(ux_p, uy_p, uz_p, dt)

            # Final separation at shell K
            diff_x = ux_p[idx] - ux_b[idx]
            diff_y = uy_p[idx] - uy_b[idx]
            diff_z = uz_p[idx] - uz_b[idx]
            df = np.sqrt(np.sum(np.abs(diff_x)**2 + np.abs(diff_y)**2 + np.abs(diff_z)**2))

            if d0 > 0 and df > 0:
                ftle = np.log(df / d0) / total_time
                ftle_vals.append(ftle)

        if len(ftle_vals) > 0:
            ftle_mean = np.mean(ftle_vals)
            ftle_std = np.std(ftle_vals)
        else:
            ftle_mean = 0.0
            ftle_std = 0.0

        results.append({
            'k': sk,
            'n_modes': n_modes,
            'ftle_mean': ftle_mean,
            'ftle_std': ftle_std,
            'tau_K': tau_K,
            'inv_tau_K': 1.0/tau_K if tau_K < np.inf else 0,
            'E_shell': E_shell,
            'u_shell': u_shell,
        })

    return results


# ========== MAIN ==========

def run_experiment(N, Re, t_final=3.0):
    """Evolve Taylor-Green to t_final, then measure."""
    nu = 1.0 / Re
    ns = SpectralNS(N, nu)

    P(f"\n--- N={N}, Re={Re} ---")
    ux, uy, uz = ns.taylor_green_ic()

    # Timestep: proven stable in previous runs
    dt = 0.001 if Re <= 400 else 0.0005
    t = 0.0
    step = 0
    t0 = clock.time()

    while t < t_final - dt/2:
        ux, uy, uz = ns.rk4_step(ux, uy, uz, dt)
        t += dt
        step += 1
        if step % 500 == 0:
            P(f"  [step {step}, t={t:.3f}, elapsed={clock.time()-t0:.0f}s]")

    P(f"  Evolved to t={t:.3f} (elapsed={clock.time()-t0:.0f}s)")

    # Compute total energy and enstrophy
    norm = 1.0 / N**6
    E = 0.5 * np.sum(np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2) * norm
    Z = 0.5 * np.sum(ns.k2 * (np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2)) * norm
    P(f"  E={E:.6f}, Z={Z:.4f}")

    max_shell = N // 3

    # Measurement A: Jacobian trace
    P(f"\n  === MEASUREMENT A: Jacobian Trace (shell K=1..{min(max_shell, 10)}) ===")
    jac_results = measure_jacobian_trace(ns, ux, uy, uz, max_shell=min(max_shell, 10))
    P(f"  {'k':>5}  {'Tr(total)':>12}  {'Tr(NL)':>12}  {'Tr(visc_num)':>12}  {'visc_analyt':>12}  n_modes")
    for r in jac_results:
        P(f"  {r['k']:5d}  {r['trace_total']:12.2f}  {r['trace_NL']:12.4f}  {r['trace_visc_num']:12.2f}  "
          f"{r['sigma_visc_analytical']:12.2f}  {r['n_modes']:6d}")

    # Measurement B: FTLE
    P(f"\n  === MEASUREMENT B: Finite-Time Lyapunov Exponents ===")
    ftle_results = measure_ftle(ns, ux, uy, uz, max_shell=min(max_shell, 10),
                                 n_perturbations=8, n_steps_ftle=20)
    P(f"  {'k':>5}  {'FTLE':>12}  {'std':>10}  {'1/tau_K':>12}  {'FTLE*tau_K':>12}  {'tau_K':>10}")
    for r in ftle_results:
        product = r['ftle_mean'] * r['tau_K'] if r['tau_K'] < 1e10 else 0
        P(f"  {r['k']:5d}  {r['ftle_mean']:12.4f}  {r['ftle_std']:10.4f}  {r['inv_tau_K']:12.6f}  "
          f"{product:12.6f}  {r['tau_K']:10.4f}")

    return jac_results, ftle_results


if __name__ == '__main__':
    P("=" * 72)
    P("  TASK 1: Phase-Space Contraction Rate sigma_dot_K")
    P("=" * 72)
    P()
    P("  Theory: For incompressible NS with zero mean flow,")
    P("  the nonlinear Jacobian diagonal vanishes (triads need u_0=0 modes).")
    P("  So sigma_dot_K = -nu * sum_{|k|~K} |k|^2 * (DOF per mode)")
    P("  = purely viscous dissipation.")
    P()
    P("  The INTERESTING quantity is the FTLE: how fast perturbations")
    P("  at shell K actually grow. This involves OFF-diagonal Jacobian")
    P("  terms (inter-shell coupling) and is what Tanogami-Araki's I_K")
    P("  really measures.")
    P()

    all_jac = {}
    all_ftle = {}

    for N, Re in [(32, 400), (32, 1600)]:
        label = f"N{N}_Re{Re}"
        jac, ftle = run_experiment(N, Re, t_final=3.0)
        all_jac[label] = jac
        all_ftle[label] = ftle

    # Summary table
    P("\n" + "=" * 72)
    P("  SUMMARY: FTLE vs 1/tau_K")
    P("=" * 72)
    P()
    P("  If I_K ~ 1/tau_K (Tanogami), then FTLE*tau_K = const across shells")
    P()

    for label, ftle_list in all_ftle.items():
        P(f"  {label}:")
        P(f"    {'k':>4}  {'FTLE':>10}  {'1/tau':>10}  {'FTLE*tau':>10}  {'FTLE/(1/tau)':>12}")
        for r in ftle_list:
            if r['inv_tau_K'] > 0 and r['tau_K'] < 1e10:
                ratio = r['ftle_mean'] / r['inv_tau_K']
                product = r['ftle_mean'] * r['tau_K']
            else:
                ratio = 0
                product = 0
            P(f"    {r['k']:4d}  {r['ftle_mean']:10.4f}  {r['inv_tau_K']:10.4f}  {product:10.4f}  {ratio:12.4f}")
        P()

    # Generate plots
    P("  Generating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Jacobian trace — NL part (should be ~0)
    ax = axes[0]
    for label, jac_list in all_jac.items():
        ks = [r['k'] for r in jac_list]
        nl_trace = [r['trace_NL'] for r in jac_list]
        ax.plot(ks, nl_trace, 'o-', label=label)
    ax.axhline(y=0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Shell k')
    ax.set_ylabel('Tr(J_NL) per shell')
    ax.set_title('Nonlinear Jacobian Trace\n(should be ~0 for zero mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: FTLE vs 1/tau_K
    ax = axes[1]
    for label, ftle_list in all_ftle.items():
        ks = [r['k'] for r in ftle_list]
        ftle_vals = [r['ftle_mean'] for r in ftle_list]
        inv_tau = [r['inv_tau_K'] for r in ftle_list]
        ax.plot(ks, ftle_vals, 'o-', label=f'FTLE {label}')
        ax.plot(ks, inv_tau, 's--', label=f'1/tau_K {label}', alpha=0.6)
    ax.set_xlabel('Shell k')
    ax.set_ylabel('Rate')
    ax.set_title('FTLE vs 1/tau_K\n(Tanogami: should be proportional)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: FTLE * tau_K (should be const if I_K ~ 1/tau_K)
    ax = axes[2]
    for label, ftle_list in all_ftle.items():
        ks = [r['k'] for r in ftle_list]
        products = [r['ftle_mean'] * r['tau_K'] if r['tau_K'] < 1e10 else 0 for r in ftle_list]
        ax.plot(ks, products, 'o-', label=label)
    ax.axhline(y=0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Shell k')
    ax.set_ylabel('FTLE * tau_K')
    ax.set_title('Tanogami Test: FTLE*tau_K\n(const = Tanogami confirmed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = 'Fluid-Resonance/scripts/wip/phase_space_contraction.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    P(f"  Plot saved: {outpath}")

    P("\n  DONE.")
