"""
VERIFICATION OPTION 3: General rho cross-check
================================================
Verify the full formula:
  alpha(theta, rho) = 1 - (1+rho)^2*(1+cos(theta)) / [(1+rho^2+2*rho*cos(theta))*(3-cos(theta))]

against numerical triad measurements at various rho values.

S100-M2d, Meridian 2.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def alpha_formula(theta, rho):
    """Exact alpha(theta, rho) formula. Handles rho=1 singularity."""
    rho = np.asarray(rho, dtype=float)
    c = np.cos(np.asarray(theta, dtype=float))
    num = (1 + rho)**2 * (1 + c)
    den = (1 + rho**2 + 2*rho*c) * (3 - c)
    # Handle den=0 (theta=pi, rho=1 gives 0/0)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = 1 - num / den
    # At rho=1 exactly: use simplified form (1-cos)/(3-cos)
    rho_is_one = np.abs(np.asarray(rho) - 1.0) < 1e-12
    if np.any(rho_is_one):
        result = np.where(rho_is_one, (1 - c) / (3 - c), result)
    return result


def measure_triad_alpha(solver, k1_idx, k2_idx):
    """Compute Leray ratio alpha for a single triad. Returns (theta, rho, alpha_pm) or None."""
    i1, j1, l1 = k1_idx
    i2, j2, l2 = k2_idx
    N = solver.N

    k1_vec = np.array([solver.kx[i1,j1,l1], solver.ky[i1,j1,l1], solver.kz[i1,j1,l1]], dtype=float)
    k2_vec = np.array([solver.kx[i2,j2,l2], solver.ky[i2,j2,l2], solver.kz[i2,j2,l2]], dtype=float)
    k1_mag = np.sqrt(solver.k2[i1,j1,l1])
    k2_mag = np.sqrt(solver.k2[i2,j2,l2])

    if k1_mag < 0.5 or k2_mag < 0.5:
        return None

    k3_vec = k1_vec + k2_vec
    k3_mag_sq = np.dot(k3_vec, k3_vec)
    if k3_mag_sq < 0.5:
        return None
    k3_mag = np.sqrt(k3_mag_sq)

    kmax = N // 3
    if k3_mag > kmax + 0.5:
        return None

    i3 = int(round(k3_vec[0])) % N
    j3 = int(round(k3_vec[1])) % N
    l3 = int(round(k3_vec[2])) % N
    k3_actual = np.array([solver.kx[i3,j3,l3], solver.ky[i3,j3,l3], solver.kz[i3,j3,l3]], dtype=float)
    if np.linalg.norm(k3_actual - k3_vec) > 0.1:
        return None

    cos_theta = np.clip(np.dot(k1_vec, k2_vec) / (k1_mag * k2_mag), -1, 1)
    theta = np.arccos(cos_theta)
    rho = k1_mag / k2_mag

    # Helical basis vectors
    ep1 = np.array([solver.h_plus[c, i1, j1, l1] for c in range(3)])
    em2 = np.array([solver.h_minus[c, i2, j2, l2] for c in range(3)])

    k3_hat = k3_actual / np.sqrt(np.dot(k3_actual, k3_actual))

    # (+,-) coupling
    lamb_pm = np.cross(ep1, em2)
    pw_pm = np.sum(np.abs(lamb_pm)**2)
    if pw_pm < 1e-30:
        return None
    lamb_pm_sol = lamb_pm - k3_hat * np.dot(k3_hat, lamb_pm)
    alpha_pm = np.sum(np.abs(lamb_pm_sol)**2) / pw_pm

    return theta, rho, alpha_pm


def main():
    print("=" * 70)
    print("  VERIFICATION 3: General rho cross-check")
    print("  alpha(theta, rho) = 1 - (1+rho)^2*(1+cos theta)")
    print("                        / [(1+rho^2+2*rho*cos theta)*(3-cos theta)]")
    print("=" * 70)

    N = 48
    solver = SpectralNS(N=N, Re=100)

    kmax = N // 3
    active = np.argwhere((solver.k2 > 0) & (solver.kmag < kmax + 0.5))
    print(f"\n  Grid N={N}, kmax={kmax}, active modes: {len(active)}")

    rng = np.random.default_rng(42)

    # Collect triads across all rho values
    print("\n  Sampling 80,000 random triads...")
    thetas = []
    rhos = []
    alphas_meas = []
    alphas_pred = []

    n_target = 80000
    n_found = 0
    attempts = 0

    while n_found < n_target and attempts < n_target * 20:
        attempts += 1
        idx1 = active[rng.integers(len(active))]
        idx2 = active[rng.integers(len(active))]

        result = measure_triad_alpha(solver, tuple(idx1), tuple(idx2))
        if result is None:
            continue

        theta, rho, alpha_pm = result
        pred = alpha_formula(theta, rho)

        thetas.append(theta)
        rhos.append(rho)
        alphas_meas.append(alpha_pm)
        alphas_pred.append(pred)
        n_found += 1

    thetas = np.array(thetas)
    rhos = np.array(rhos)
    alphas_meas = np.array(alphas_meas)
    alphas_pred = np.array(alphas_pred)
    print(f"  Found {n_found} triads (from {attempts} attempts)")

    # Global RMS
    residuals = alphas_meas - alphas_pred
    rms_global = np.sqrt(np.mean(residuals**2))
    mean_ratio = np.mean(alphas_meas / np.maximum(alphas_pred, 1e-10))
    print(f"\n  GLOBAL RMS (all triads): {rms_global:.8f}")
    print(f"  GLOBAL mean ratio:       {mean_ratio:.6f}")

    # Bin by rho and report
    rho_edges = [0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.18, 1.43, 2.0, 3.3]
    print(f"\n  {'rho bin':25s}  {'n':>6}  {'RMS':>10}  {'mean ratio':>12}  {'max |resid|':>12}")
    print("  " + "-" * 75)

    n_theta_bins = 25
    theta_edges = np.linspace(0, np.pi, n_theta_bins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    rho_bin_data = []  # For plotting

    for rb in range(len(rho_edges) - 1):
        rho_lo, rho_hi = rho_edges[rb], rho_edges[rb + 1]
        mask = (rhos >= rho_lo) & (rhos < rho_hi)
        n_rho = np.sum(mask)
        if n_rho < 50:
            print(f"  rho in [{rho_lo:.2f}, {rho_hi:.2f}){' ':10s}  {n_rho:>6}  (too few)")
            continue

        res_bin = residuals[mask]
        rms_bin = np.sqrt(np.mean(res_bin**2))
        mr_bin = np.mean(alphas_meas[mask] / np.maximum(alphas_pred[mask], 1e-10))
        max_res = np.max(np.abs(res_bin))

        label = f"rho in [{rho_lo:.2f}, {rho_hi:.2f})"
        print(f"  {label:25s}  {n_rho:>6}  {rms_bin:>10.6f}  {mr_bin:>12.6f}  {max_res:>12.6f}")

        # Also bin by theta within this rho range for plotting
        t_sub = thetas[mask]
        a_meas_sub = alphas_meas[mask]
        rho_med = np.median(rhos[mask])

        a_binned_meas = np.full(n_theta_bins, np.nan)
        a_binned_pred = np.full(n_theta_bins, np.nan)
        for i in range(n_theta_bins):
            m = (t_sub >= theta_edges[i]) & (t_sub < theta_edges[i+1])
            if np.sum(m) > 3:
                a_binned_meas[i] = np.mean(a_meas_sub[m])
                a_binned_pred[i] = alpha_formula(theta_centers[i], rho_med)

        rho_bin_data.append((label, rho_med, a_binned_meas, a_binned_pred, n_rho, rms_bin))

    # Compare against WRONG formula (rho=1) to show improvement
    alphas_wrong = (1 - np.cos(thetas)) / (3 - np.cos(thetas))
    rms_wrong = np.sqrt(np.mean((alphas_meas - alphas_wrong)**2))
    print(f"\n  Comparison:")
    print(f"    RMS vs full alpha(theta, rho):    {rms_global:.8f}")
    print(f"    RMS vs rho=1 formula only:        {rms_wrong:.8f}")
    print(f"    Improvement factor:               {rms_wrong/rms_global:.1f}x")

    # ================================================================
    # Check rho-symmetry numerically: alpha(theta, rho) vs alpha(theta, 1/rho)
    # ================================================================
    print("\n\n  RHO-SYMMETRY CHECK: alpha(theta,rho) vs alpha(theta,1/rho)")
    pred_rho = alpha_formula(thetas, rhos)
    pred_inv = alpha_formula(thetas, 1.0/rhos)
    sym_diff = np.abs(pred_rho - pred_inv)
    print(f"  Max |alpha(theta,rho) - alpha(theta,1/rho)|: {np.max(sym_diff):.2e}")
    print(f"  Mean:                                        {np.mean(sym_diff):.2e}")

    # ================================================================
    # Verify isotropic average <alpha>(rho) for several rho values
    # ================================================================
    print("\n\n  ISOTROPIC AVERAGE <alpha>(rho) vs numerical quadrature")
    from scipy import integrate

    def avg_alpha_numerical(rho_val, n_quad=5000):
        """Compute <alpha>(rho) = (1/2) * integral_0^pi alpha(theta,rho)*sin(theta) dtheta."""
        th = np.linspace(0, np.pi, n_quad)
        integrand = alpha_formula(th, rho_val) * np.sin(th)
        try:
            return np.trapz(integrand, th) / 2.0
        except AttributeError:
            return np.trapezoid(integrand, th) / 2.0

    rho_test = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0]
    print(f"  {'rho':>6}  {'<alpha>(rho)':>14}  {'2*(1-ln2)':>12}  {'ratio to min':>14}")
    print("  " + "-" * 55)
    min_alpha = 1 - np.log(2)  # = <alpha>(rho=1)
    for r in rho_test:
        avg = avg_alpha_numerical(r)
        print(f"  {r:>6.2f}  {avg:>14.8f}  {2*min_alpha:>12.8f}  {avg/min_alpha:>14.6f}")

    # ================================================================
    # 4-panel figure
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Verification 3: General rho cross-check', fontsize=14)

    # Panel 1: Scatter of measured vs predicted
    ax = axes[0, 0]
    sc = ax.scatter(alphas_pred, alphas_meas, c=np.log10(rhos), s=1, alpha=0.3, cmap='coolwarm')
    ax.plot([0, 0.7], [0, 0.7], 'k--', lw=1, label='y=x')
    ax.set_xlabel('Predicted alpha(theta, rho)')
    ax.set_ylabel('Measured alpha')
    ax.set_title(f'All triads (RMS={rms_global:.6f})')
    ax.legend()
    plt.colorbar(sc, ax=ax, label='log10(rho)')

    # Panel 2: Residuals vs rho
    ax = axes[0, 1]
    ax.scatter(rhos, residuals, s=1, alpha=0.1, c='steelblue')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('rho')
    ax.set_ylabel('Residual (measured - predicted)')
    ax.set_title('Residuals vs rho')
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.15, 0.15)

    # Panel 3: theta profiles for different rho bins
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    n_curves = len(rho_bin_data)
    for idx, (label, rho_med, a_meas, a_pred, n_rho, rms_bin) in enumerate(rho_bin_data):
        color = cmap(idx / max(n_curves - 1, 1))
        valid = ~np.isnan(a_meas)
        ax.plot(theta_centers[valid] / np.pi, a_meas[valid], 'o', color=color, ms=3, alpha=0.7)
        ax.plot(theta_centers[valid] / np.pi, a_pred[valid], '-', color=color, lw=1.5,
                label=f'rho~{rho_med:.2f} (RMS={rms_bin:.4f})')
    ax.set_xlabel('theta / pi')
    ax.set_ylabel('alpha')
    ax.set_title('alpha(theta) by rho bin: dots=measured, lines=formula')
    ax.legend(fontsize=6, ncol=2)

    # Panel 4: <alpha>(rho) curve
    ax = axes[1, 1]
    rho_curve = np.logspace(-1, 1, 200)
    avg_curve = [avg_alpha_numerical(r) for r in rho_curve]
    ax.plot(rho_curve, avg_curve, 'b-', lw=2, label='<alpha>(rho) numerical quadrature')
    ax.axhline(min_alpha, color='r', ls='--', lw=1, label=f'1-ln2 = {min_alpha:.4f} (rho=1 min)')
    ax.axhline(2*min_alpha, color='orange', ls=':', lw=1, label=f'2(1-ln2) = {2*min_alpha:.4f} (rho->0,inf)')
    ax.set_xlabel('rho')
    ax.set_ylabel('<alpha>(rho)')
    ax.set_title('Isotropic average vs rho')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'verify_alpha_general_rho.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved to {out_path}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
