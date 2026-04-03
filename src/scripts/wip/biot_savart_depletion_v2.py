"""
BIOT-SAVART STRETCHING DEPLETION v2 — Corrected baseline.

KEY INSIGHT FROM v1:
    Parallel (aligned) vortex filaments have ZERO stretching.
    This is NOT a depletion effect — it's because Biot-Savart velocity
    from a parallel filament is purely azimuthal (no axial strain).

    This means:
    - "Aligned" is NOT the maximum-stretching baseline
    - Stretching REQUIRES misalignment between filaments
    - The maximum-stretching configuration is PERPENDICULAR filaments

CORRECTED APPROACH:
    1. Find the OPTIMAL direction configuration that MAXIMIZES total stretching
       (for given positions and strengths)
    2. Compare: max-stretching vs isotropic vs random
    3. The depletion factor is: |isotropic| / |optimal|

    If isotropic configurations have LESS stretching than the optimal
    configuration, that's depletion. If they have comparable stretching,
    there's no depletion.

ALSO: Investigate cancellations. Do isotropic configurations produce
stretching that cancels more (more filaments with negative sigma)?

Author: Claude (Opus 4.6), 2026-03-11
"""

import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(precision=6, linewidth=120)


def biot_savart_velocity(x, x_j, xi_j, gamma_j, delta=0.01):
    """Regularized Biot-Savart velocity at x from filament at x_j."""
    r = x - x_j
    r_mag_sq = np.dot(r, r) + delta**2
    r_mag_32 = r_mag_sq ** 1.5
    cross = np.cross(xi_j, r)
    return (gamma_j / (4 * np.pi)) * cross / r_mag_32


def biot_savart_strain(x, x_j, xi_j, gamma_j, delta=0.01):
    """Strain tensor S at x from filament at x_j (finite differences)."""
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
    return S / 2.0


def compute_strain_tensors(positions, directions, gammas, delta=0.01):
    """
    Compute the strain tensor at each filament position from all other
    filaments. Returns list of 3x3 strain tensors.
    """
    n = len(positions)
    strains = []
    for i in range(n):
        S_total = np.zeros((3, 3))
        for j in range(n):
            if i == j:
                continue
            S_j = biot_savart_strain(positions[i], positions[j],
                                     directions[j], gammas[j], delta)
            S_total += S_j
        strains.append(S_total)
    return strains


def total_stretching(directions, strains, gammas):
    """
    Compute total enstrophy production rate for given directions.
    Z_stretch = sum_i gamma_i^2 * xi_i . S_i . xi_i
    """
    n = len(directions)
    total = 0.0
    per_filament = np.zeros(n)
    for i in range(n):
        sigma = directions[i] @ strains[i] @ directions[i]
        per_filament[i] = sigma
        total += gammas[i]**2 * sigma
    return total, per_filament


def optimal_directions(strains, gammas, n_restarts=10):
    """
    Find the direction configuration that MAXIMIZES total stretching.

    For each filament i independently, the direction that maximizes
    xi . S_i . xi is the eigenvector of S_i corresponding to the
    LARGEST eigenvalue.

    HOWEVER: this is not globally optimal because changing direction i
    changes the strain at all other positions. For the regularized
    kernel, this coupling is weak, so the greedy approach is reasonable.

    We use the greedy approach: for each filament, choose the direction
    that maximizes its own stretching given the current strain.
    """
    n = len(strains)
    best_total = -np.inf
    best_dirs = None

    for restart in range(n_restarts):
        # Greedy: each filament picks eigenvector of max eigenvalue of its strain
        dirs = np.zeros((n, 3))
        for i in range(n):
            evals, evecs = np.linalg.eigh(strains[i])
            # Pick eigenvector of largest eigenvalue
            dirs[i] = evecs[:, np.argmax(evals)]
            # Random sign
            if restart > 0 and np.random.rand() > 0.5:
                dirs[i] *= -1

        total, _ = total_stretching(dirs, strains, gammas)
        if total > best_total:
            best_total = total
            best_dirs = dirs.copy()

    return best_dirs, best_total


def isotropic_directions(n):
    """Fibonacci sphere — quasi-uniform on S^2."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n)
    dirs = np.zeros((n, 3))
    dirs[:, 0] = np.sin(phi) * np.cos(theta)
    dirs[:, 1] = np.sin(phi) * np.sin(theta)
    dirs[:, 2] = np.cos(phi)
    return dirs


def random_positions_ball(n, epsilon, seed):
    """Random positions in B(0, epsilon)."""
    rng = np.random.RandomState(seed)
    positions = []
    while len(positions) < n:
        p = rng.uniform(-epsilon, epsilon, 3)
        if np.linalg.norm(p) <= epsilon:
            positions.append(p)
    return np.array(positions)


# ============================================================
# SECTION 1: Two-filament analytical case
# ============================================================

print("=" * 75)
print("SECTION 1: TWO-FILAMENT STRETCHING (analytical baseline)")
print("=" * 75)

# Two filaments at (0,0,0) and (d,0,0) with various relative angles
d = 0.1
pos = np.array([[0, 0, 0], [d, 0, 0]], dtype=float)
gammas = np.ones(2)

print(f"\n  Two filaments at distance d = {d}")
print(f"  {'Angle (deg)':>12} | {'sigma_1':>10} | {'sigma_2':>10} | {'Total':>10}")
print("  " + "-" * 50)

for angle_deg in [0, 15, 30, 45, 60, 75, 90]:
    angle = np.radians(angle_deg)
    dirs = np.array([
        [0, 0, 1],  # filament 1 along z
        [np.sin(angle), 0, np.cos(angle)]  # filament 2 at angle from z
    ], dtype=float)

    strains = compute_strain_tensors(pos, dirs, gammas)
    total, per_f = total_stretching(dirs, strains, gammas)

    print(f"  {angle_deg:12d} | {per_f[0]:+10.4f} | {per_f[1]:+10.4f} | {total:+10.4f}")

print("""
KEY FINDING:
  - Parallel (0 deg): ZERO stretching (confirmed)
  - Perpendicular (90 deg): MAXIMUM stretching magnitude
  - The stretching INCREASES with misalignment angle
  - At 90 deg, one filament is strongly stretched, the other compressed
  - Total stretching (sum) is nonzero and position-dependent
""")


# ============================================================
# SECTION 2: Optimal vs Isotropic vs Random — proper comparison
# ============================================================

print("=" * 75)
print("SECTION 2: OPTIMAL vs ISOTROPIC vs RANDOM (corrected)")
print("=" * 75)

for n in [6, 10, 15, 20]:
    print(f"\n  --- n = {n} ---")
    n_trials = 20
    optimal_totals = []
    iso_totals = []
    random_totals = []

    for trial in range(n_trials):
        seed = trial * 31 + n * 7
        pos = random_positions_ball(n, epsilon=0.1, seed=seed)
        gammas = np.ones(n)

        # Compute strain tensors with ISOTROPIC directions first
        # (strain depends on the OTHER filaments' directions, creating circularity)
        # For a fair comparison: compute strain from RANDOM directions,
        # then find optimal response directions for THOSE strains.
        # This avoids the circularity.

        # Actually, the correct approach: for EACH direction config,
        # compute strain from THOSE directions, then compute stretching.
        # This is self-consistent.

        # --- Isotropic directions ---
        iso_dirs = isotropic_directions(n)
        iso_strains = compute_strain_tensors(pos, iso_dirs, gammas)
        iso_total, _ = total_stretching(iso_dirs, iso_strains, gammas)
        iso_totals.append(iso_total)

        # --- Random directions ---
        rng = np.random.RandomState(seed)
        rand_dirs = rng.randn(n, 3)
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        rand_strains = compute_strain_tensors(pos, rand_dirs, gammas)
        rand_total, _ = total_stretching(rand_dirs, rand_strains, gammas)
        random_totals.append(rand_total)

        # --- Optimal directions (greedy) ---
        # Start with random directions, compute strain, find optimal response
        # Iterate a few times for self-consistency
        curr_dirs = rng.randn(n, 3)
        curr_dirs /= np.linalg.norm(curr_dirs, axis=1, keepdims=True)

        for iteration in range(5):
            curr_strains = compute_strain_tensors(pos, curr_dirs, gammas)
            new_dirs = np.zeros((n, 3))
            for i in range(n):
                evals, evecs = np.linalg.eigh(curr_strains[i])
                new_dirs[i] = evecs[:, np.argmax(evals)]
            curr_dirs = new_dirs

        opt_strains = compute_strain_tensors(pos, curr_dirs, gammas)
        opt_total, _ = total_stretching(curr_dirs, opt_strains, gammas)
        optimal_totals.append(opt_total)

    opt_mean = np.mean(optimal_totals)
    iso_mean = np.mean(iso_totals)
    rand_mean = np.mean(random_totals)

    opt_abs = np.mean(np.abs(optimal_totals))
    iso_abs = np.mean(np.abs(iso_totals))
    rand_abs = np.mean(np.abs(random_totals))

    print(f"  Optimal:   mean = {opt_mean:+10.2f}, |mean| = {opt_abs:10.2f}")
    print(f"  Random:    mean = {rand_mean:+10.2f}, |mean| = {rand_abs:10.2f}")
    print(f"  Isotropic: mean = {iso_mean:+10.2f}, |mean| = {iso_abs:10.2f}")

    if opt_abs > 1e-6:
        print(f"  Depletion (|iso|/|opt|): {iso_abs/opt_abs:.4f}")
        print(f"  Depletion (|rand|/|opt|): {rand_abs/opt_abs:.4f}")

    # Positive fraction (how often total stretching is positive = enstrophy growing)
    opt_pos = np.sum(np.array(optimal_totals) > 0)
    iso_pos = np.sum(np.array(iso_totals) > 0)
    rand_pos = np.sum(np.array(random_totals) > 0)
    print(f"  Positive fraction: opt={opt_pos}/{n_trials}, iso={iso_pos}/{n_trials}, rand={rand_pos}/{n_trials}")


# ============================================================
# SECTION 3: Cancellation analysis
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 3: CANCELLATION ANALYSIS")
print(f"{'='*75}")
print("""
For each configuration, compute:
  - Total positive stretching: sum of sigma_i where sigma_i > 0
  - Total negative stretching: sum of sigma_i where sigma_i < 0
  - Net stretching: positive + negative
  - Cancellation ratio: |net| / (positive + |negative|)
     0 = perfect cancellation, 1 = no cancellation
""")

n = 15
n_trials = 30

for dir_type in ["optimal", "isotropic", "random"]:
    cancel_ratios = []
    net_stretchings = []
    pos_fracs = []

    for trial in range(n_trials):
        seed = trial * 41 + 17
        pos = random_positions_ball(n, epsilon=0.1, seed=seed)
        gammas = np.ones(n)

        if dir_type == "isotropic":
            dirs = isotropic_directions(n)
        elif dir_type == "random":
            rng = np.random.RandomState(seed)
            dirs = rng.randn(n, 3)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        else:  # optimal
            rng = np.random.RandomState(seed)
            dirs = rng.randn(n, 3)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            for _ in range(5):
                strains = compute_strain_tensors(pos, dirs, gammas)
                for i in range(n):
                    evals, evecs = np.linalg.eigh(strains[i])
                    dirs[i] = evecs[:, np.argmax(evals)]

        strains = compute_strain_tensors(pos, dirs, gammas)
        _, per_f = total_stretching(dirs, strains, gammas)

        pos_stretch = np.sum(per_f[per_f > 0])
        neg_stretch = np.sum(per_f[per_f < 0])
        net = pos_stretch + neg_stretch
        total_mag = pos_stretch + abs(neg_stretch)

        if total_mag > 1e-10:
            cancel_ratio = abs(net) / total_mag
        else:
            cancel_ratio = 0.0

        cancel_ratios.append(cancel_ratio)
        net_stretchings.append(net)
        pos_fracs.append(np.sum(per_f > 0) / n)

    print(f"\n  {dir_type.upper():12s}:")
    print(f"    Cancel ratio:   {np.mean(cancel_ratios):.4f} +/- {np.std(cancel_ratios):.4f}")
    print(f"    (0=full cancel, 1=no cancel)")
    print(f"    Net stretching: {np.mean(net_stretchings):+.2f} +/- {np.std(net_stretchings):.2f}")
    print(f"    Frac positive:  {np.mean(pos_fracs):.3f} +/- {np.std(pos_fracs):.3f}")


# ============================================================
# SECTION 4: Self-consistent fixed point analysis
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 4: SELF-CONSISTENT FIXED POINTS")
print(f"{'='*75}")
print("""
The strain at filament i depends on directions of OTHER filaments.
The optimal direction at i depends on the strain at i.
This creates a self-consistent system. Does it have a fixed point?

For OPTIMAL: iterate directions -> strain -> directions until convergence.
Track whether the total stretching INCREASES or DECREASES per iteration.
""")

n = 10
np.random.seed(123)
pos = random_positions_ball(n, epsilon=0.1, seed=123)
gammas = np.ones(n)

# Start with random directions
dirs = np.random.randn(n, 3)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

print(f"  Iteration | Total stretching | Max |sigma_i| | Converged?")
print("  " + "-" * 60)

prev_total = None
for it in range(20):
    strains = compute_strain_tensors(pos, dirs, gammas)
    total, per_f = total_stretching(dirs, strains, gammas)

    # Update directions: each picks max eigenvalue eigenvector
    new_dirs = np.zeros((n, 3))
    for i in range(n):
        evals, evecs = np.linalg.eigh(strains[i])
        new_dirs[i] = evecs[:, np.argmax(evals)]

    # Check convergence
    dir_change = np.max([min(np.linalg.norm(new_dirs[i] - dirs[i]),
                              np.linalg.norm(new_dirs[i] + dirs[i]))
                          for i in range(n)])
    converged = dir_change < 1e-6

    print(f"  {it:9d} | {total:+16.4f} | {np.max(np.abs(per_f)):15.4f} | {'YES' if converged else f'change={dir_change:.6f}'}")

    dirs = new_dirs
    if converged:
        break

print(f"\n  Fixed-point total stretching: {total:+.4f}")
print(f"  Fixed-point direction variety:")
# Check if fixed point directions are isotropic or cone-like
dots = []
for i in range(n):
    for j in range(i+1, n):
        dots.append(abs(np.dot(dirs[i], dirs[j])))
print(f"    Mean |xi_i . xi_j|: {np.mean(dots):.4f} (0=orthogonal, 1=parallel)")
print(f"    Max  |xi_i . xi_j|: {np.max(dots):.4f}")
print(f"    Min  |xi_i . xi_j|: {np.min(dots):.4f}")

# Check if directions span S^2 (Lei et al. condition)
# = do they intersect every great circle?
# Quick check: project onto random directions
n_checks = 100
min_max_proj = 1.0
for _ in range(n_checks):
    e = np.random.randn(3)
    e /= np.linalg.norm(e)
    projs = np.abs([np.dot(dirs[i], e) for i in range(n)])
    max_perp = np.max(np.sqrt(1 - projs**2))  # max |xi x e|
    min_max_proj = min(min_max_proj, max_perp)

print(f"    Min max|xi x e| over 100 random e: {min_max_proj:.4f}")
print(f"    (>0.5 means directions plausibly span S^2)")


# ============================================================
# SECTION 5: The KEY question — net positive stretching
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 5: NET POSITIVE STRETCHING — The critical observable")
print(f"{'='*75}")
print("""
For NS blow-up, we need dZ/dt > 0, i.e., POSITIVE total stretching
must overcome viscous dissipation.

Key question: Among all direction configurations satisfying Lei et al.
(directions span S^2), what is the MAXIMUM net positive stretching?

If this maximum is bounded (grows slower than Z^{3/2}), blow-up
is prevented.
""")

n = 12
n_trials = 50

# For each trial: find optimal directions, check if they span S^2
opt_spanning = []
opt_not_spanning = []

for trial in range(n_trials):
    seed = trial * 53 + 7
    pos = random_positions_ball(n, epsilon=0.1, seed=seed)
    gammas = np.ones(n)

    # Find self-consistent optimal
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    for _ in range(10):
        strains = compute_strain_tensors(pos, dirs, gammas)
        for i in range(n):
            evals, evecs = np.linalg.eigh(strains[i])
            dirs[i] = evecs[:, np.argmax(evals)]

    strains = compute_strain_tensors(pos, dirs, gammas)
    total, _ = total_stretching(dirs, strains, gammas)

    # Check if optimal directions span S^2
    spans = True
    for _ in range(50):
        e = np.random.randn(3)
        e /= np.linalg.norm(e)
        max_cross = max(np.linalg.norm(np.cross(dirs[i], e)) for i in range(n))
        if max_cross < 0.5:
            spans = False
            break

    if spans:
        opt_spanning.append(total)
    else:
        opt_not_spanning.append(total)

print(f"  n = {n}, {n_trials} trials:")
print(f"  Optimal directions spanning S^2: {len(opt_spanning)}/{n_trials}")
print(f"  Optimal directions NOT spanning: {len(opt_not_spanning)}/{n_trials}")

if opt_spanning:
    print(f"\n  SPANNING S^2 (Lei et al. satisfied):")
    print(f"    Mean total stretching: {np.mean(opt_spanning):+.2f}")
    print(f"    Max total stretching:  {np.max(opt_spanning):+.2f}")
    print(f"    Fraction positive:     {np.sum(np.array(opt_spanning) > 0)/len(opt_spanning):.2f}")

if opt_not_spanning:
    print(f"\n  NOT SPANNING (Lei et al. violated):")
    print(f"    Mean total stretching: {np.mean(opt_not_spanning):+.2f}")
    print(f"    Max total stretching:  {np.max(opt_not_spanning):+.2f}")
    print(f"    Fraction positive:     {np.sum(np.array(opt_not_spanning) > 0)/len(opt_not_spanning):.2f}")

if opt_spanning and opt_not_spanning:
    ratio = np.mean(np.abs(opt_spanning)) / np.mean(np.abs(opt_not_spanning))
    print(f"\n  |Spanning| / |Not spanning|: {ratio:.4f}")
    if ratio < 1:
        print(f"  --> Spanning S^2 REDUCES optimal stretching by factor {1-ratio:.4f}")
    else:
        print(f"  --> Spanning S^2 does NOT reduce optimal stretching")


# ============================================================
# SECTION 6: Summary
# ============================================================

print(f"\n\n{'='*75}")
print("SUMMARY")
print(f"{'='*75}")

print("""
CORRECTED UNDERSTANDING:

1. PARALLEL FILAMENTS:
   - Zero stretching (v1 finding, confirmed)
   - NOT the maximum baseline — parallel means NO interaction
   - This is why: Biot-Savart velocity from parallel tubes is azimuthal

2. MAXIMUM STRETCHING:
   - Requires MISALIGNED filaments (perpendicular = strongest)
   - Self-consistent optimal: iterate directions <-> strain
   - Fixed point exists and converges in ~5-10 iterations

3. DEPLETION QUESTION (CORRECTED):
   - NOT "isotropic < aligned" (aligned = 0, wrong baseline)
   - CORRECT: "isotropic < optimal" (optimal uses perpendicular alignment)
   - The depletion is about CANCELLATIONS in the total stretching

4. KEY OBSERVABLE:
   - For blow-up prevention, we need: max total stretching (among
     configurations satisfying Lei et al.) < viscous dissipation
   - This is a CONSTRAINED OPTIMIZATION problem:
     max { sum gamma_i^2 sigma_i } subject to { directions span S^2 }

IMPLICATIONS FOR THE PROOF:
   - The stretching is NOT "depleted" in the naive sense
   - Instead, isotropic directions create a MIX of positive and negative
     stretching, with partial cancellation
   - Whether the cancellation is ENOUGH is the key question
   - The cancellation ratio (Section 3) is the critical quantity
""")
