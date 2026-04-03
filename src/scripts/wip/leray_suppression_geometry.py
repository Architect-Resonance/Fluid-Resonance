"""
LERAY SUPPRESSION FACTOR: GEOMETRIC CALCULATION
================================================
The single question: what fraction of h+(k1) x h-(k2) survives
Leray projection onto the plane perpendicular to k3 = k1 + k2?

This is pure geometry. No dynamics, no helicity conservation.

For helical basis vectors:
  h+(k) = (e1 + i * e2) / sqrt(2)   where e1, e2 perp to k
  h-(k) = (e1 - i * e2) / sqrt(2) = conj(h+(k))

The cross product h+(k1) x h-(k2) lives at wavevector k3 = k1 + k2.
The Leray projector at k3 removes the k3-component:
  P_sol(k3) = I - k3_hat * k3_hat^T

The suppression factor for this triad:
  alpha(k1, k2) = |P_sol(k3) [h+(k1) x h-(k2)]|^2 / |h+(k1) x h-(k2)|^2

We compute this for:
1. Analytic special cases (k1 perp k2, k1 || k2, etc.)
2. Monte Carlo average over random (k1, k2) orientations
3. Average over actual wavenumber grid pairs

If <alpha> < 1 with a clean bound, that's the geometric proof.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def build_helical_basis(k_vec):
    """Construct helical basis vectors h+, h- for wavevector k.

    h+ = (e1 + i*e2) / sqrt(2)
    h- = (e1 - i*e2) / sqrt(2)
    where e1, e2 form an orthonormal basis in the plane perp to k.
    """
    k = np.array(k_vec, dtype=float)
    kmag = np.linalg.norm(k)
    if kmag < 1e-15:
        return np.zeros(3, dtype=complex), np.zeros(3, dtype=complex)

    khat = k / kmag

    # Choose reference vector not parallel to k
    if abs(khat[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(khat, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(khat, e1)

    h_plus = (e1 + 1j * e2) / np.sqrt(2.0)
    h_minus = (e1 - 1j * e2) / np.sqrt(2.0)
    return h_plus, h_minus


def cross_complex(a, b):
    """Cross product of two complex 3-vectors."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


def leray_project(f, k_vec):
    """Leray projection: remove component along k.
    P_sol(k) f = f - (f . khat) khat
    """
    k = np.array(k_vec, dtype=float)
    kmag = np.linalg.norm(k)
    if kmag < 1e-15:
        return f
    khat = k / kmag
    return f - np.dot(f, khat) * khat


def compute_alpha_single_triad(k1_vec, k2_vec):
    """Compute the Leray suppression factor for a single triad.

    alpha = |P_sol(k3) [h+(k1) x h-(k2)]|^2 / |h+(k1) x h-(k2)|^2
    where k3 = k1 + k2.

    Returns alpha for all four sector combinations:
    (++), (--), (+-), (-+)
    """
    k1 = np.array(k1_vec, dtype=float)
    k2 = np.array(k2_vec, dtype=float)
    k3 = k1 + k2

    if np.linalg.norm(k1) < 1e-15 or np.linalg.norm(k2) < 1e-15:
        return {'++': 0, '--': 0, '+-': 0, '-+': 0}
    if np.linalg.norm(k3) < 1e-15:
        return {'++': 1, '--': 1, '+-': 1, '-+': 1}  # k3=0 means no projection

    h1p, h1m = build_helical_basis(k1)
    h2p, h2m = build_helical_basis(k2)

    results = {}
    for label, ha, hb in [('++', h1p, h2p), ('--', h1m, h2m),
                           ('+-', h1p, h2m), ('-+', h1m, h2p)]:
        cross = cross_complex(ha, hb)
        cross_norm_sq = float(np.sum(np.abs(cross)**2))

        if cross_norm_sq < 1e-30:
            results[label] = 0.0
            continue

        projected = leray_project(cross, k3)
        proj_norm_sq = float(np.sum(np.abs(projected)**2))

        results[label] = proj_norm_sq / cross_norm_sq

    return results


def monte_carlo_average(n_samples=100000):
    """Monte Carlo average of alpha over random (k1, k2) orientations.

    Sample k1 and k2 uniformly on the unit sphere, compute alpha for each.
    """
    print(f"\n  Monte Carlo average ({n_samples:,} samples)...")

    alphas = {'++': [], '--': [], '+-': [], '-+': []}

    for _ in range(n_samples):
        # Random unit vectors (uniform on sphere)
        k1 = np.random.randn(3)
        k1 /= np.linalg.norm(k1)
        k2 = np.random.randn(3)
        k2 /= np.linalg.norm(k2)

        # Random magnitudes (uniform in [1, 10])
        k1 *= np.random.uniform(1, 10)
        k2 *= np.random.uniform(1, 10)

        result = compute_alpha_single_triad(k1, k2)
        for key in alphas:
            alphas[key].append(result[key])

    stats = {}
    for key in alphas:
        arr = np.array(alphas[key])
        stats[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'max': float(np.max(arr)),
            'min': float(np.min(arr)),
        }
        print(f"    alpha_{key}: mean={stats[key]['mean']:.6f}  "
              f"std={stats[key]['std']:.6f}  "
              f"max={stats[key]['max']:.6f}")

    return alphas, stats


def angular_dependence(n_angles=200):
    """Compute alpha as a function of angle between k1 and k2.

    Fix |k1| = |k2| = 1, vary angle theta between them.
    """
    print(f"\n  Angular dependence ({n_angles} angles)...")

    thetas = np.linspace(0.01, np.pi - 0.01, n_angles)
    alpha_vs_theta = {'++': [], '--': [], '+-': [], '-+': []}

    k1 = np.array([0, 0, 1.0])

    for theta in thetas:
        # k2 in the xz-plane at angle theta from k1
        k2 = np.array([np.sin(theta), 0, np.cos(theta)])

        result = compute_alpha_single_triad(k1, k2)
        for key in alpha_vs_theta:
            alpha_vs_theta[key].append(result[key])

    return thetas, alpha_vs_theta


def magnitude_dependence(n_ratios=100):
    """Compute alpha as a function of |k1|/|k2| at fixed angle."""
    print(f"\n  Magnitude ratio dependence ({n_ratios} ratios)...")

    theta = np.pi / 3  # 60 degrees
    ratios = np.logspace(-1, 1, n_ratios)  # |k1|/|k2| from 0.1 to 10

    alpha_vs_ratio = {'++': [], '--': [], '+-': [], '-+': []}

    for ratio in ratios:
        k1 = np.array([0, 0, ratio])
        k2 = np.array([np.sin(theta), 0, np.cos(theta)])

        result = compute_alpha_single_triad(k1, k2)
        for key in alpha_vs_ratio:
            alpha_vs_ratio[key].append(result[key])

    return ratios, alpha_vs_ratio


def grid_average(N=32):
    """Average alpha over all valid wavenumber pairs on an N^3 grid.

    This is the actual average relevant to spectral NS.
    """
    print(f"\n  Grid average (N={N}, sampling subset)...")

    from numpy.fft import fftfreq
    k1d = fftfreq(N, d=1.0/N)

    # Sample a subset (full grid is N^6 pairs = too many)
    n_samples = 50000
    alphas = {'++': [], '--': [], '+-': [], '-+': []}

    for _ in range(n_samples):
        # Random grid points
        i1 = np.random.randint(0, N, 3)
        i2 = np.random.randint(0, N, 3)

        k1 = np.array([k1d[i1[0]], k1d[i1[1]], k1d[i1[2]]])
        k2 = np.array([k1d[i2[0]], k1d[i2[1]], k1d[i2[2]]])

        if np.linalg.norm(k1) < 0.5 or np.linalg.norm(k2) < 0.5:
            continue

        result = compute_alpha_single_triad(k1, k2)
        for key in alphas:
            alphas[key].append(result[key])

    stats = {}
    for key in alphas:
        arr = np.array(alphas[key])
        stats[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'max': float(np.max(arr)),
        }
        print(f"    alpha_{key} (grid): mean={stats[key]['mean']:.6f}  "
              f"std={stats[key]['std']:.6f}  "
              f"max={stats[key]['max']:.6f}")

    return stats


def main():
    print("=" * 70)
    print("  LERAY SUPPRESSION FACTOR: GEOMETRIC CALCULATION")
    print("  alpha = |P_sol(k3) [h+(k1) x h-(k2)]|^2 / |h+(k1) x h-(k2)|^2")
    print("=" * 70)

    np.random.seed(42)

    # =========================================================
    # 1. Special cases
    # =========================================================
    print("\n" + "=" * 70)
    print("  SPECIAL CASES")
    print("=" * 70)

    cases = [
        ("k1 || k2 (parallel)", [0,0,1], [0,0,2]),
        ("k1 perp k2 (orthogonal)", [1,0,0], [0,1,0]),
        ("k1 = -k2 (antiparallel)", [0,0,1], [0,0,-1]),
        ("45 degrees, equal mag", [0,0,1], [0,1,1]),
        ("60 degrees, equal mag", [0,0,1], [np.sin(np.pi/3),0,np.cos(np.pi/3)]),
        ("30 degrees, |k2|=3|k1|", [0,0,1], [3*np.sin(np.pi/6),0,3*np.cos(np.pi/6)]),
    ]

    print(f"  {'Case':<35s}  {'a_++':>8s}  {'a_--':>8s}  {'a_+-':>8s}  {'a_-+':>8s}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for name, k1, k2 in cases:
        result = compute_alpha_single_triad(k1, k2)
        print(f"  {name:<35s}  {result['++']:8.5f}  {result['--']:8.5f}  "
              f"{result['+-']:8.5f}  {result['-+']:8.5f}")

    # =========================================================
    # 2. Monte Carlo average
    # =========================================================
    print("\n" + "=" * 70)
    print("  MONTE CARLO AVERAGE")
    print("=" * 70)
    mc_alphas, mc_stats = monte_carlo_average(n_samples=200000)

    # =========================================================
    # 3. Angular dependence
    # =========================================================
    print("\n" + "=" * 70)
    print("  ANGULAR DEPENDENCE")
    print("=" * 70)
    thetas, alpha_theta = angular_dependence(n_angles=500)

    # =========================================================
    # 4. Grid average
    # =========================================================
    print("\n" + "=" * 70)
    print("  GRID AVERAGE (N=32)")
    print("=" * 70)
    grid_stats = grid_average(N=32)

    # =========================================================
    # VERDICT
    # =========================================================
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    cross_mean = (mc_stats['+-']['mean'] + mc_stats['-+']['mean']) / 2
    same_mean = (mc_stats['++']['mean'] + mc_stats['--']['mean']) / 2

    print(f"\n  Cross-helical alpha (Monte Carlo average): {cross_mean:.6f}")
    print(f"  Same-helical alpha (Monte Carlo average):  {same_mean:.6f}")

    cross_max = max(mc_stats['+-']['max'], mc_stats['-+']['max'])
    same_max = max(mc_stats['++']['max'], mc_stats['--']['max'])

    print(f"\n  Cross-helical alpha (max observed): {cross_max:.6f}")
    print(f"  Same-helical alpha (max observed):  {same_max:.6f}")

    if cross_mean < 0.5:
        print(f"\n  => Cross-helical Leray suppression is GEOMETRIC.")
        print(f"     On average, {(1-cross_mean)*100:.1f}% of cross-helical Lamb is gradient.")
        print(f"     This is a PROPERTY OF THE HELICAL BASIS, not of the dynamics.")
    else:
        print(f"\n  => Cross-helical Leray suppression is WEAK geometrically.")
        print(f"     The ~9% observed in simulations must come from dynamics, not geometry alone.")

    if same_mean < cross_mean:
        print(f"\n  => Same-helical has STRONGER suppression ({(1-same_mean)*100:.1f}% gradient)")
    else:
        print(f"\n  => Same-helical has WEAKER suppression ({(1-same_mean)*100:.1f}% gradient)")

    # =========================================================
    # SAVE & PLOT
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Leray Suppression Factor: Geometry of Helical Basis', fontsize=13)

    # Plot 1: Angular dependence
    ax = axes[0]
    for key, color in [('++', 'blue'), ('--', 'cyan'), ('+-', 'red'), ('-+', 'orange')]:
        ax.plot(np.degrees(thetas), alpha_theta[key], color=color, label=key, linewidth=1.5)
    ax.set_xlabel('Angle between k1 and k2 (degrees)')
    ax.set_ylabel('alpha (Leray suppression factor)')
    ax.set_title('alpha vs angle (|k1|=|k2|=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Plot 2: Monte Carlo histogram
    ax = axes[1]
    ax.hist(mc_alphas['+-'], bins=100, alpha=0.5, color='red', label='+-', density=True)
    ax.hist(mc_alphas['++'], bins=100, alpha=0.5, color='blue', label='++', density=True)
    ax.axvline(x=mc_stats['+-']['mean'], color='red', linestyle='--', label=f'+- mean={mc_stats["+-"]["mean"]:.3f}')
    ax.axvline(x=mc_stats['++']['mean'], color='blue', linestyle='--', label=f'++ mean={mc_stats["++"]["mean"]:.3f}')
    ax.set_xlabel('alpha')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of alpha (Monte Carlo)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: alpha_+- and alpha_++ vs angle (zoomed)
    ax = axes[2]
    ax.plot(np.degrees(thetas), alpha_theta['+-'], 'r-', label='Cross (+-)', linewidth=2)
    ax.plot(np.degrees(thetas), alpha_theta['++'], 'b-', label='Same (++)', linewidth=2)
    ax.axhline(y=mc_stats['+-']['mean'], color='red', linestyle=':', alpha=0.5)
    ax.axhline(y=mc_stats['++']['mean'], color='blue', linestyle=':', alpha=0.5)
    ax.fill_between(np.degrees(thetas), 0, alpha_theta['+-'], alpha=0.1, color='red')
    ax.fill_between(np.degrees(thetas), 0, alpha_theta['++'], alpha=0.1, color='blue')
    ax.set_xlabel('Angle between k1 and k2 (degrees)')
    ax.set_ylabel('alpha')
    ax.set_title('Cross vs Same suppression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'leray_suppression_geometry.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to {plot_path}")
    plt.close()

    print("\n  DONE.")


if __name__ == '__main__':
    main()
