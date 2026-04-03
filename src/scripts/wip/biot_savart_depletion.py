"""
BIOT-SAVART STRETCHING DEPLETION UNDER DIRECTIONAL ISOTROPY

THE QUESTION:
    Given n vortex tubes with directions spanning S^2, concentrated
    in a ball of radius epsilon, what is the average stretching
    omega . S omega compared to the maximum possible?

    If there's a UNIVERSAL depletion factor (independent of n, epsilon),
    that's a concrete result toward NS regularity.

THE SETUP:
    - n vortex filaments, each at position x_i in B(0, epsilon)
    - Each has direction xi_i on S^2 and strength Gamma_i
    - The velocity at x_i from filament j is given by Biot-Savart:
        u_j(x_i) = (Gamma_j / 4pi) * xi_j x (x_i - x_j) / |x_i - x_j|^3
    - The strain tensor S at x_i from filament j:
        S_j = symmetric part of grad(u_j)
    - The stretching at filament i:
        sigma_i = xi_i . S(x_i) . xi_i  (summed over all j != i)

    We compare:
        Z_stretch = sum_i Gamma_i^2 * sigma_i  (total enstrophy production)
    against:
        Z_max = maximum possible Z_stretch over all direction configurations

THE KEY: If directions are ISOTROPIC (span S^2), is Z_stretch depleted?

Author: Claude (Opus 4.6), 2026-03-11
"""

import numpy as np
from itertools import combinations
import sys

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(precision=8, linewidth=120)


# ============================================================
# SECTION 1: Biot-Savart strain from a vortex filament
# ============================================================

def biot_savart_velocity(x, x_j, xi_j, gamma_j, delta=0.01):
    """
    Velocity at point x due to a vortex filament at x_j
    with direction xi_j and circulation gamma_j.

    Uses regularized Biot-Savart (Rosenhead-Moore kernel) to avoid
    singularity at x = x_j.

    u = (gamma / 4pi) * xi x r / (|r|^2 + delta^2)^{3/2}
    where r = x - x_j.
    """
    r = x - x_j
    r_mag_sq = np.dot(r, r) + delta**2
    r_mag_32 = r_mag_sq ** 1.5

    cross = np.cross(xi_j, r)
    u = (gamma_j / (4 * np.pi)) * cross / r_mag_32
    return u


def biot_savart_strain(x, x_j, xi_j, gamma_j, delta=0.01):
    """
    Strain tensor S_{ij} at point x due to a vortex filament at x_j.

    S = (grad u + grad u^T) / 2

    Computed via finite differences (simple, robust).
    """
    h = 1e-6
    S = np.zeros((3, 3))

    for k in range(3):
        dx = np.zeros(3)
        dx[k] = h
        u_plus = biot_savart_velocity(x + dx, x_j, xi_j, gamma_j, delta)
        u_minus = biot_savart_velocity(x - dx, x_j, xi_j, gamma_j, delta)
        grad_k = (u_plus - u_minus) / (2 * h)
        S[k, :] += grad_k
        S[:, k] += grad_k

    S /= 2.0
    return S


def compute_stretching(positions, directions, gammas, delta=0.01):
    """
    Compute the total stretching for a system of vortex filaments.

    sigma_i = xi_i . S(x_i) . xi_i
    where S(x_i) = sum_{j != i} S_j(x_i)

    Returns:
        stretching: array of sigma_i values
        total: sum of gamma_i^2 * sigma_i (enstrophy production rate)
    """
    n = len(positions)
    stretching = np.zeros(n)

    for i in range(n):
        S_total = np.zeros((3, 3))
        for j in range(n):
            if i == j:
                continue
            S_j = biot_savart_strain(positions[i], positions[j],
                                     directions[j], gammas[j], delta)
            S_total += S_j

        stretching[i] = directions[i] @ S_total @ directions[i]

    total = np.sum(gammas**2 * stretching)
    return stretching, total


# ============================================================
# SECTION 2: Direction distributions on S^2
# ============================================================

def random_directions(n, seed=None):
    """Random uniform directions on S^2."""
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def cone_directions(n, half_angle_deg=30, seed=None):
    """Directions confined to a cone around e_3 with given half-angle."""
    rng = np.random.RandomState(seed)
    half_angle = np.radians(half_angle_deg)

    dirs = []
    while len(dirs) < n:
        d = rng.randn(3)
        d /= np.linalg.norm(d)
        # Check if within cone around e_3
        if abs(d[2]) >= np.cos(half_angle):
            dirs.append(d)
    return np.array(dirs)


def isotropic_directions(n, seed=None):
    """
    Directions that SPAN S^2 (satisfy Lei et al. condition).
    We use a quasi-uniform distribution: Fibonacci sphere.
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n)

    dirs = np.zeros((n, 3))
    dirs[:, 0] = np.sin(phi) * np.cos(theta)
    dirs[:, 1] = np.sin(phi) * np.sin(theta)
    dirs[:, 2] = np.cos(phi)
    return dirs


def aligned_directions(n):
    """All directions along e_3 (maximum self-consistent stretching)."""
    dirs = np.zeros((n, 3))
    dirs[:, 2] = 1.0
    return dirs


def orthogonal_directions(n):
    """
    Directions split equally among e_1, e_2, e_3.
    This satisfies Lei et al. (spans S^2) but is highly structured.
    """
    dirs = np.zeros((n, 3))
    for i in range(n):
        axis = i % 3
        sign = 1 if (i // 3) % 2 == 0 else -1
        dirs[i, axis] = sign
    return dirs


# ============================================================
# SECTION 3: Position distributions in B(0, epsilon)
# ============================================================

def random_positions_ball(n, epsilon=0.1, seed=None):
    """Random uniform positions in a ball of radius epsilon."""
    rng = np.random.RandomState(seed)
    positions = []
    while len(positions) < n:
        p = rng.uniform(-epsilon, epsilon, 3)
        if np.linalg.norm(p) <= epsilon:
            positions.append(p)
    return np.array(positions)


def ring_positions(n, epsilon=0.1):
    """Positions on a ring of radius epsilon in the xy-plane."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.zeros((n, 3))
    positions[:, 0] = epsilon * np.cos(angles)
    positions[:, 1] = epsilon * np.sin(angles)
    return positions


# ============================================================
# SECTION 4: Depletion factor computation
# ============================================================

def depletion_experiment(n, direction_fn, position_fn, n_trials=20,
                          epsilon=0.1, delta=0.01):
    """
    Run n_trials experiments computing stretching for given
    direction and position distributions.

    Returns: array of total stretching values
    """
    results = []

    for trial in range(n_trials):
        seed = trial * 137 + n

        # Positions
        if position_fn == "random_ball":
            pos = random_positions_ball(n, epsilon, seed)
        elif position_fn == "ring":
            pos = ring_positions(n, epsilon)
        else:
            pos = random_positions_ball(n, epsilon, seed)

        # Directions
        if direction_fn == "aligned":
            dirs = aligned_directions(n)
        elif direction_fn == "isotropic":
            dirs = isotropic_directions(n)
        elif direction_fn == "random":
            dirs = random_directions(n, seed)
        elif direction_fn == "cone_30":
            dirs = cone_directions(n, 30, seed)
        elif direction_fn == "cone_60":
            dirs = cone_directions(n, 60, seed)
        elif direction_fn == "orthogonal":
            dirs = orthogonal_directions(n)
        else:
            dirs = random_directions(n, seed)

        gammas = np.ones(n)

        _, total = compute_stretching(pos, dirs, gammas, delta)
        results.append(total)

    return np.array(results)


# ============================================================
# SECTION 5: Main experiments
# ============================================================

print("=" * 75)
print("BIOT-SAVART STRETCHING DEPLETION ANALYSIS")
print("=" * 75)
print()
print("For each configuration, we compute the total enstrophy production rate")
print("Z_stretch = sum_i gamma_i^2 * (xi_i . S(x_i) . xi_i)")
print("and compare across direction distributions.")
print()

# Test with different n values
for n in [6, 10, 20]:
    print(f"\n{'='*75}")
    print(f"  n = {n} vortex filaments, epsilon = 0.1")
    print(f"{'='*75}")

    configs = [
        ("aligned", "All parallel (e_3)"),
        ("cone_30", "Cone 30 deg"),
        ("cone_60", "Cone 60 deg"),
        ("random", "Random on S^2"),
        ("isotropic", "Isotropic (Fibonacci)"),
        ("orthogonal", "Orthogonal (e1/e2/e3)"),
    ]

    n_trials = 15
    results = {}

    for dir_fn, label in configs:
        vals = depletion_experiment(n, dir_fn, "random_ball",
                                     n_trials=n_trials, epsilon=0.1)
        results[dir_fn] = vals
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"\n  {label:30s}: mean = {mean_val:+10.4f}, std = {std_val:8.4f}")

    # Compute depletion factor relative to aligned
    aligned_mean = np.mean(results["aligned"])
    if abs(aligned_mean) > 1e-10:
        print(f"\n  DEPLETION FACTORS (relative to aligned):")
        for dir_fn, label in configs:
            ratio = np.mean(results[dir_fn]) / aligned_mean
            print(f"    {label:30s}: {ratio:+.4f}")
    else:
        print(f"\n  Aligned mean ~ 0, computing absolute values instead:")
        for dir_fn, label in configs:
            print(f"    {label:30s}: |mean| = {abs(np.mean(results[dir_fn])):.6f}")


# ============================================================
# SECTION 6: Detailed analysis for n=10
# ============================================================

print(f"\n\n{'='*75}")
print("DETAILED ANALYSIS: n=10, single configuration")
print(f"{'='*75}")

n = 10
epsilon = 0.1
np.random.seed(42)
pos = random_positions_ball(n, epsilon, seed=42)
gammas = np.ones(n)

for dir_name, dirs in [
    ("Aligned (e_3)", aligned_directions(n)),
    ("Isotropic (Fibonacci)", isotropic_directions(n)),
    ("Random", random_directions(n, seed=42)),
]:
    stretching, total = compute_stretching(pos, dirs, gammas)

    # Also compute the strain eigenvalues at each point
    print(f"\n  {dir_name}:")
    print(f"    Total stretching: {total:+.6f}")
    print(f"    Per-filament stretching:")
    for i in range(n):
        print(f"      filament {i}: sigma = {stretching[i]:+.6f}")

    # Compute strain tensor at first filament from all others
    S_total = np.zeros((3, 3))
    for j in range(n):
        if j == 0:
            continue
        S_j = biot_savart_strain(pos[0], pos[j], dirs[j], gammas[j])
        S_total += S_j

    eigs = np.linalg.eigvalsh(S_total)
    print(f"    Strain eigenvalues at filament 0: {eigs}")
    print(f"    Trace (should be ~0): {np.sum(eigs):.8f}")
    print(f"    Max stretching eigenvalue: {eigs[-1]:.6f}")
    print(f"    Actual stretching (xi.S.xi): {stretching[0]:.6f}")
    if abs(eigs[-1]) > 1e-10:
        print(f"    Alignment ratio (actual/max): {stretching[0]/eigs[-1]:.4f}")


# ============================================================
# SECTION 7: Scaling with epsilon (separation distance)
# ============================================================

print(f"\n\n{'='*75}")
print("SCALING WITH EPSILON (concentration radius)")
print(f"{'='*75}")

n = 10
for epsilon in [1.0, 0.5, 0.1, 0.05, 0.01]:
    aligned_vals = depletion_experiment(n, "aligned", "random_ball",
                                         n_trials=10, epsilon=epsilon)
    iso_vals = depletion_experiment(n, "isotropic", "random_ball",
                                     n_trials=10, epsilon=epsilon)

    a_mean = np.mean(aligned_vals)
    i_mean = np.mean(iso_vals)

    if abs(a_mean) > 1e-10:
        ratio = i_mean / a_mean
        print(f"  eps={epsilon:.2f}: aligned={a_mean:+.4f}, iso={i_mean:+.4f}, depletion={ratio:+.4f}")
    else:
        print(f"  eps={epsilon:.2f}: aligned={a_mean:+.6f}, iso={i_mean:+.6f}")


# ============================================================
# SECTION 8: The key test — does depletion survive concentration?
# ============================================================

print(f"\n\n{'='*75}")
print("KEY TEST: Does depletion factor converge as n -> infinity?")
print(f"{'='*75}")

epsilon = 0.1
n_trials = 10

print(f"\n  {'n':>5} | {'aligned_mean':>12} | {'iso_mean':>12} | {'depletion':>10} | {'|depl|':>8}")
print("  " + "-" * 55)

for n in [4, 6, 8, 10, 15, 20, 30]:
    aligned_vals = depletion_experiment(n, "aligned", "random_ball",
                                         n_trials=n_trials, epsilon=epsilon)
    iso_vals = depletion_experiment(n, "isotropic", "random_ball",
                                     n_trials=n_trials, epsilon=epsilon)

    a_mean = np.mean(aligned_vals)
    i_mean = np.mean(iso_vals)

    if abs(a_mean) > 1e-10:
        ratio = i_mean / a_mean
        print(f"  {n:5d} | {a_mean:+12.4f} | {i_mean:+12.4f} | {ratio:+10.4f} | {abs(ratio):8.4f}")
    else:
        print(f"  {n:5d} | {a_mean:+12.6f} | {i_mean:+12.6f} | N/A        |")


# ============================================================
# SECTION 9: Statistical analysis — is the depletion significant?
# ============================================================

print(f"\n\n{'='*75}")
print("STATISTICAL TEST: Is isotropic stretching significantly less?")
print(f"{'='*75}")

n = 15
n_trials = 50

aligned_vals = depletion_experiment(n, "aligned", "random_ball",
                                     n_trials=n_trials, epsilon=0.1)
iso_vals = depletion_experiment(n, "isotropic", "random_ball",
                                 n_trials=n_trials, epsilon=0.1)
random_vals = depletion_experiment(n, "random", "random_ball",
                                    n_trials=n_trials, epsilon=0.1)

print(f"\n  n = {n}, {n_trials} trials each:")
print(f"  Aligned:   mean = {np.mean(aligned_vals):+.4f}, std = {np.std(aligned_vals):.4f}")
print(f"  Isotropic: mean = {np.mean(iso_vals):+.4f}, std = {np.std(iso_vals):.4f}")
print(f"  Random:    mean = {np.mean(random_vals):+.4f}, std = {np.std(random_vals):.4f}")

# Sign analysis
print(f"\n  Aligned > 0: {np.sum(aligned_vals > 0)}/{n_trials}")
print(f"  Iso > 0:     {np.sum(iso_vals > 0)}/{n_trials}")
print(f"  Random > 0:  {np.sum(random_vals > 0)}/{n_trials}")

# Absolute values (stretching magnitude regardless of sign)
print(f"\n  |Aligned|:   mean = {np.mean(np.abs(aligned_vals)):.4f}")
print(f"  |Isotropic|: mean = {np.mean(np.abs(iso_vals)):.4f}")
print(f"  |Random|:    mean = {np.mean(np.abs(random_vals)):.4f}")

if np.mean(np.abs(aligned_vals)) > 1e-10:
    depl_iso = np.mean(np.abs(iso_vals)) / np.mean(np.abs(aligned_vals))
    depl_rand = np.mean(np.abs(random_vals)) / np.mean(np.abs(aligned_vals))
    print(f"\n  |Depletion| (iso/aligned):    {depl_iso:.4f}")
    print(f"  |Depletion| (random/aligned): {depl_rand:.4f}")


# ============================================================
# SECTION 10: Summary
# ============================================================

print(f"\n\n{'='*75}")
print("SUMMARY")
print(f"{'='*75}")

print("""
This script computes the Biot-Savart vortex stretching for different
directional distributions, seeking a UNIVERSAL depletion factor when
vorticity directions span S^2 (Lei et al. condition).

Key questions answered:
1. Does isotropic direction distribution reduce total stretching?
2. Is the depletion factor universal (independent of n, epsilon)?
3. How does it compare to cone-confined and aligned configurations?

INTERPRETATION GUIDE:
- If depletion factor ~ constant < 1 for all n: STRONG result
- If depletion factor -> 1 as n -> infinity: WEAK result (depletion vanishes)
- If depletion factor -> 0 as n -> infinity: VERY STRONG (but suspicious)
- If results are noisy/sign-varying: geometry matters more than direction

IMPORTANT CAVEAT:
This is a NUMERICAL experiment with regularized Biot-Savart (delta > 0).
The regularization parameter delta prevents the 1/r^3 singularity but
also smooths out short-range interactions. Results should be verified
with different delta values to check robustness.
""")
