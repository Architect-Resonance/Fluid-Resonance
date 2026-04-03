"""
stretching_frustration.py

SPIN-GLASS FRUSTRATION IN VORTEX STRETCHING
=============================================

The observation: C = |sum stretching_i| / sum |max_stretching_i| ~ 1/N.
This means the individual stretching contributions CANCEL.

Two possible explanations:
A) Random sign cancellation: stretching_i has random sign -> sum ~ sqrt(N)
   -> C ~ sqrt(N)/N = 1/sqrt(N). But observed C ~ 1/N, which is BETTER.

B) Anti-correlation: Biot-Savart coupling forces ANTI-CORRELATED stretching
   at nearby points. Positive at i implies negative at neighbors.
   -> sum ~ O(1) -> C ~ 1/N.

If (B) is true, the Biot-Savart coupling creates a "spin-glass frustration"
where the stretching at each point can't be maximized independently.

This experiment tests:
1. Are stretching contributions correlated or anti-correlated?
2. Does the correlation depend on distance?
3. Does the sum grow as sqrt(N) (random) or O(1) (anti-correlated)?
4. Does this explain C ~ 1/N?
5. Can the frustration be proved analytically?
"""

import numpy as np
from scipy.linalg import eigh


# Precompute Levi-Civita
LC = np.zeros((3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            if (i, j, k) in [(0,1,2), (1,2,0), (2,0,1)]:
                LC[i, j, k] = 1
            elif (i, j, k) in [(0,2,1), (2,1,0), (1,0,2)]:
                LC[i, j, k] = -1


def compute_velocity_gradient(x, positions, vorticities, epsilon=0.05):
    """Compute full velocity gradient tensor at x from Biot-Savart."""
    grad_u = np.zeros((3, 3))
    for j in range(len(positions)):
        r = x - positions[j]
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            continue
        r_reg = np.sqrt(r_norm**2 + epsilon**2)
        for a in range(3):
            for kk in range(3):
                for b in range(3):
                    for m in range(3):
                        e = LC[a, b, m]
                        if e == 0:
                            continue
                        t1 = (1 if m == kk else 0) / r_reg**3
                        t2 = -3 * r[m] * r[kk] / r_reg**5
                        grad_u[a, kk] += e * vorticities[j][b] * (t1 + t2) / (4 * np.pi)
    return grad_u


def compute_strain(grad_u):
    return 0.5 * (grad_u + grad_u.T)


def compute_per_point_stretching(positions, vorticities, epsilon=0.05):
    """Return the stretching at each point: s_i = omega_i . S_i . omega_i."""
    N = len(positions)
    stretching = np.zeros(N)
    max_stretching = np.zeros(N)
    for i in range(N):
        S_i = compute_strain(compute_velocity_gradient(positions[i], positions, vorticities, epsilon))
        stretching[i] = vorticities[i] @ S_i @ vorticities[i]
        evals = np.sort(np.linalg.eigvalsh(S_i))[::-1]
        max_stretching[i] = evals[0] * np.dot(vorticities[i], vorticities[i])
    return stretching, max_stretching


# ===== PART 1: CORRELATION STRUCTURE =====

def part1_stretching_correlations():
    """
    Compute the correlation between stretching_i and stretching_j
    as a function of distance |x_i - x_j|.

    If anti-correlated at short range -> frustration mechanism confirmed.
    """
    print("=" * 70)
    print("PART 1: STRETCHING CORRELATION STRUCTURE")
    print("Are nearby stretching contributions correlated or anti-correlated?")
    print("=" * 70)

    all_pairs = []  # (distance, s_i * s_j, |s_i * s_j|)

    for seed in range(20):
        np.random.seed(seed * 41 + 7)
        N = 30
        pos = np.random.randn(N, 3) * 0.5
        vor = np.random.randn(N, 3)
        vor /= np.linalg.norm(vor, axis=1, keepdims=True)

        stretching, _ = compute_per_point_stretching(pos, vor)

        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                product = stretching[i] * stretching[j]
                all_pairs.append((dist, product))

    # Bin by distance
    distances = np.array([p[0] for p in all_pairs])
    products = np.array([p[1] for p in all_pairs])

    n_bins = 12
    bin_edges = np.linspace(0, distances.max(), n_bins + 1)

    print(f"\n  {'Dist range':<20} | {'Mean s_i*s_j':>14} | {'Fraction negative':>17} | {'Count':>5}")
    print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*17}-+-{'-'*5}")

    for b in range(n_bins):
        mask = (distances >= bin_edges[b]) & (distances < bin_edges[b + 1])
        if mask.sum() < 5:
            continue
        mean_prod = products[mask].mean()
        frac_neg = (products[mask] < 0).mean()
        sign = "ANTI" if frac_neg > 0.55 else ("CORR" if frac_neg < 0.45 else "~0")
        print(f"  [{bin_edges[b]:.2f}, {bin_edges[b+1]:.2f})"
              f" | {mean_prod:>14.6f} | {frac_neg:>14.2%} ({sign}) | {mask.sum():>5}")

    # Global correlation coefficient
    mean_s = np.mean([p[1] for p in all_pairs])
    frac_neg_all = np.mean([p[1] < 0 for p in all_pairs])
    print(f"\n  Overall: mean(s_i * s_j) = {mean_s:.6f}")
    print(f"  Overall fraction negative: {frac_neg_all:.2%}")
    if frac_neg_all > 0.5:
        print(f"  -> ANTI-CORRELATED (frustration present)")
    else:
        print(f"  -> NOT anti-correlated")


# ===== PART 2: SUM GROWTH RATE =====

def part2_sum_scaling():
    """
    How does |sum_i stretching_i| grow with N?

    Random cancellation: |sum| ~ sqrt(N) -> C ~ 1/sqrt(N)
    Anti-correlation: |sum| ~ O(1) -> C ~ 1/N
    Perfect alignment: |sum| ~ N -> C ~ 1

    Test by varying N while keeping density fixed.
    """
    print("\n" + "=" * 70)
    print("PART 2: SUM SCALING — HOW DOES |SUM STRETCHING| GROW WITH N?")
    print("Random: sqrt(N). Anti-correlated: O(1). Aligned: N.")
    print("=" * 70)

    N_values = [10, 15, 20, 30, 40, 50, 60]
    n_trials = 15

    print(f"\n  {'N':<6} | {'Mean |sum s_i|':>14} | {'Mean sum |s_i|':>14} | {'Mean C':>8} | {'C*N':>8} | {'C*sqrt(N)':>10}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    N_arr = []
    C_arr = []

    for N in N_values:
        abs_sums = []
        total_maxes = []
        C_vals = []

        for trial in range(n_trials):
            np.random.seed(trial * 71 + N * 13)
            # Fixed density: scale ~ N^{1/3} so points don't crowd
            scale = 0.3 * (N / 20.0)**(1.0/3.0)
            pos = np.random.randn(N, 3) * scale
            vor = np.random.randn(N, 3)
            vor /= np.linalg.norm(vor, axis=1, keepdims=True)

            s, ms = compute_per_point_stretching(pos, vor)
            abs_sum = abs(np.sum(s))
            total_max = np.sum(ms)

            abs_sums.append(abs_sum)
            total_maxes.append(total_max)
            if total_max > 1e-10:
                C_vals.append(abs_sum / total_max)

        mean_abs_sum = np.mean(abs_sums)
        mean_total_max = np.mean(total_maxes)
        mean_C = np.mean(C_vals) if C_vals else 0

        N_arr.append(N)
        C_arr.append(mean_C)

        print(f"  {N:<6} | {mean_abs_sum:>14.4f} | {mean_total_max:>14.4f}"
              f" | {mean_C:>8.5f} | {mean_C*N:>8.3f} | {mean_C*np.sqrt(N):>10.4f}")

    # Fit C vs N: log(C) = a * log(N) + b -> C ~ N^a
    if len(N_arr) > 3:
        log_N = np.log(np.array(N_arr))
        log_C = np.log(np.array(C_arr))
        coeffs = np.polyfit(log_N, log_C, 1)
        exponent = coeffs[0]
        print(f"\n  Power-law fit: C ~ N^({exponent:.4f})")
        if abs(exponent + 1) < 0.15:
            print(f"  -> C ~ 1/N CONFIRMED (exponent ~ -1)")
            print(f"  -> Stretching sum grows as O(1) — ANTI-CORRELATION!")
        elif abs(exponent + 0.5) < 0.15:
            print(f"  -> C ~ 1/sqrt(N) (random cancellation)")
        else:
            print(f"  -> Intermediate scaling (exponent {exponent:.3f})")


# ===== PART 3: BIOT-SAVART RECIPROCITY =====

def part3_reciprocity():
    """
    THE MECHANISM: Biot-Savart reciprocity.

    The strain S_i at point i is caused by all OTHER vortices j != i.
    Specifically, vortex j contributes a strain S_{ij} at point i.
    The stretching of i by j is: omega_i . S_{ij} . omega_i

    Key question: is stretching_{ij} = omega_i . S_{ij} . omega_i
    anti-correlated with stretching_{ji} = omega_j . S_{ji} . omega_j?

    If yes: when j stretches i, i COMPRESSES j (and vice versa).
    This would be a CONSERVATION-LIKE constraint on total stretching.
    """
    print("\n" + "=" * 70)
    print("PART 3: BIOT-SAVART RECIPROCITY")
    print("When j stretches i, does i compress j?")
    print("=" * 70)

    all_pairs = []  # (s_ij, s_ji, dist)

    for seed in range(20):
        np.random.seed(seed * 53 + 3)
        N = 25
        pos = np.random.randn(N, 3) * 0.5
        vor = np.random.randn(N, 3)
        vor /= np.linalg.norm(vor, axis=1, keepdims=True)

        for i in range(N):
            for j in range(i + 1, N):
                # Strain at i from ONLY vortex j
                S_ij = compute_strain(compute_velocity_gradient(
                    pos[i], pos[j:j+1], vor[j:j+1]))
                s_ij = vor[i] @ S_ij @ vor[i]

                # Strain at j from ONLY vortex i
                S_ji = compute_strain(compute_velocity_gradient(
                    pos[j], pos[i:i+1], vor[i:i+1]))
                s_ji = vor[j] @ S_ji @ vor[j]

                dist = np.linalg.norm(pos[i] - pos[j])
                all_pairs.append((s_ij, s_ji, dist))

    s_ij_arr = np.array([p[0] for p in all_pairs])
    s_ji_arr = np.array([p[1] for p in all_pairs])
    dists = np.array([p[2] for p in all_pairs])

    # Correlation
    corr = np.corrcoef(s_ij_arr, s_ji_arr)[0, 1]
    print(f"\n  Correlation(s_ij, s_ji) = {corr:.6f}")

    # Fraction where signs differ
    opposite_sign = ((s_ij_arr > 0) & (s_ji_arr < 0)) | ((s_ij_arr < 0) & (s_ji_arr > 0))
    frac_opposite = opposite_sign.mean()
    print(f"  Fraction with opposite sign: {frac_opposite:.2%}")

    # Sum s_ij + s_ji (should be ~0 if anti-correlated)
    pair_sums = s_ij_arr + s_ji_arr
    print(f"  Mean(s_ij + s_ji) = {pair_sums.mean():.6f}")
    print(f"  Mean(|s_ij + s_ji|) / Mean(|s_ij| + |s_ji|) = "
          f"{np.mean(np.abs(pair_sums)) / np.mean(np.abs(s_ij_arr) + np.abs(s_ji_arr)):.4f}")

    # Distance dependence
    print(f"\n  Distance dependence of reciprocity:")
    print(f"  {'Dist range':<20} | {'Corr(s_ij, s_ji)':>18} | {'Frac opposite':>14} | {'Count':>5}")
    print(f"  {'-'*20}-+-{'-'*18}-+-{'-'*14}-+-{'-'*5}")

    n_bins = 8
    bin_edges = np.linspace(0, dists.max(), n_bins + 1)
    for b in range(n_bins):
        mask = (dists >= bin_edges[b]) & (dists < bin_edges[b + 1])
        if mask.sum() < 10:
            continue
        c = np.corrcoef(s_ij_arr[mask], s_ji_arr[mask])[0, 1]
        fo = opposite_sign[mask].mean()
        print(f"  [{bin_edges[b]:.2f}, {bin_edges[b+1]:.2f})"
              f" | {c:>18.4f} | {fo:>11.2%} | {mask.sum():>5}")

    if corr < -0.1:
        print(f"\n  ANTI-RECIPROCITY CONFIRMED!")
        print(f"  When j stretches i, i tends to compress j.")
        print(f"  This is the frustration mechanism explaining C ~ 1/N.")
    elif corr > 0.1:
        print(f"\n  POSITIVE reciprocity — mutual reinforcement.")
    else:
        print(f"\n  No significant reciprocity — stretching pairs are independent.")


# ===== PART 4: TRACE-FREE CONSTRAINT =====

def part4_trace_free_cancellation():
    """
    The trace-free constraint: tr(S) = 0 means alpha + beta + gamma = 0.

    For the TOTAL stretching sum over all points:
    sum_i omega_i . S_i . omega_i = sum_i sum_j!=i (omega_i . S_{ij} . omega_i)

    Can we show that the trace-free constraint on each S_{ij} forces
    cancellation in the double sum?

    Key identity: S_{ij} is trace-free. If omega_i is aligned with alpha(S_{ij}),
    it gets stretched. But the trace-free condition means there MUST be
    compression in other directions. If omega_k at a third point k is
    affected by the same S field, it may be compressed.
    """
    print("\n" + "=" * 70)
    print("PART 4: TRACE-FREE CANCELLATION MECHANISM")
    print("Does the trace-free constraint force global cancellation?")
    print("=" * 70)

    np.random.seed(42)
    N = 30
    pos = np.random.randn(N, 3) * 0.5
    vor = np.random.randn(N, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)

    # Decompose total stretching into pairwise contributions
    stretching_matrix = np.zeros((N, N))  # s_ij = contribution of j to stretching at i
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            S_ij = compute_strain(compute_velocity_gradient(
                pos[i], pos[j:j+1], vor[j:j+1]))
            stretching_matrix[i, j] = vor[i] @ S_ij @ vor[i]

    total = stretching_matrix.sum()
    total_abs = np.abs(stretching_matrix).sum()

    print(f"\n  Total stretching (sum of matrix): {total:.6f}")
    print(f"  Sum of |s_ij|: {total_abs:.6f}")
    print(f"  Cancellation ratio: |sum| / sum|s_ij| = {abs(total) / total_abs:.6f}")

    # Check: for each source j, what is sum_i s_ij?
    # (total effect of vortex j on ALL other vortices)
    print(f"\n  Per-source total effect (sum_i s_ij for each j):")
    col_sums = stretching_matrix.sum(axis=0)
    print(f"  Mean: {col_sums.mean():.6f}")
    print(f"  Std:  {col_sums.std():.6f}")
    print(f"  Max:  {col_sums.max():.6f}")
    print(f"  Min:  {col_sums.min():.6f}")

    # Check: for each receiver i, what is sum_j s_ij?
    # (total stretching experienced by vortex i)
    print(f"\n  Per-receiver total stretching (sum_j s_ij for each i):")
    row_sums = stretching_matrix.sum(axis=1)
    print(f"  Mean: {row_sums.mean():.6f}")
    print(f"  Std:  {row_sums.std():.6f}")
    # Positive means stretched, negative means compressed
    n_pos = (row_sums > 0).sum()
    n_neg = (row_sums < 0).sum()
    print(f"  Stretched (>0): {n_pos}/{N}, Compressed (<0): {n_neg}/{N}")

    # The key: check if the SYMMETRIC part (s_ij + s_ji)/2 cancels
    sym_part = 0.5 * (stretching_matrix + stretching_matrix.T)
    anti_part = 0.5 * (stretching_matrix - stretching_matrix.T)
    print(f"\n  Symmetric part (s_ij + s_ji)/2:")
    print(f"    Total: {sym_part.sum():.6f}")
    print(f"    |Total|/sum|entries|: {abs(sym_part.sum()) / (np.abs(sym_part).sum() + 1e-15):.6f}")
    print(f"  Anti-symmetric part (s_ij - s_ji)/2:")
    print(f"    Total: {anti_part.sum():.6f}")
    print(f"    |Total|/sum|entries|: {abs(anti_part.sum()) / (np.abs(anti_part).sum() + 1e-15):.6f}")


# ===== PART 5: THE FRUSTRATION INDEX =====

def part5_frustration_index():
    """
    Define a frustration index F analogous to spin-glass theory.

    In spin glasses: F = 1 - E_actual / E_max
    Here: F = 1 - C = 1 - |sum s_i| / sum |s_i|_max

    Test:
    - F vs N (does frustration increase with system size?)
    - F vs geometry (which arrangements are most frustrated?)
    - F vs vorticity structure (parallel vs random directions)
    """
    print("\n" + "=" * 70)
    print("PART 5: FRUSTRATION INDEX vs N AND GEOMETRY")
    print("F = 1 - C (fraction of stretching lost to frustration)")
    print("=" * 70)

    # Test 1: F vs N
    print(f"\n  --- F vs N (random configs, fixed density) ---")
    print(f"  {'N':<6} | {'F (frustration)':>15} | {'C (alignment)':>14} | {'C*N':>8}")
    print(f"  {'-'*6}-+-{'-'*15}-+-{'-'*14}-+-{'-'*8}")

    for N in [8, 12, 16, 20, 25, 30, 40, 50]:
        F_vals = []
        C_vals = []
        for trial in range(10):
            np.random.seed(trial * 97 + N * 7)
            scale = 0.3 * (N / 20.0)**(1.0/3.0)
            pos = np.random.randn(N, 3) * scale
            vor = np.random.randn(N, 3)
            vor /= np.linalg.norm(vor, axis=1, keepdims=True)

            s, ms = compute_per_point_stretching(pos, vor)
            total = abs(np.sum(s))
            total_max = np.sum(ms)
            C = total / (total_max + 1e-15)
            F = 1 - C
            F_vals.append(F)
            C_vals.append(C)

        mean_F = np.mean(F_vals)
        mean_C = np.mean(C_vals)
        print(f"  {N:<6} | {mean_F:>15.6f} | {mean_C:>14.6f} | {mean_C*N:>8.4f}")

    # Test 2: Parallel vs random directions
    print(f"\n  --- F for different vorticity structures (N=30) ---")
    np.random.seed(42)
    N = 30
    pos = np.random.randn(N, 3) * 0.5

    configs = {
        "Random": np.random.randn(N, 3),
        "All parallel (z)": np.tile([0, 0, 1.0], (N, 1)),
        "All parallel (x)": np.tile([1.0, 0, 0], (N, 1)),
        "Radial (outward)": pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-10),
        "Tangential": np.cross(pos, [0, 0, 1]) / (np.linalg.norm(np.cross(pos, [0, 0, 1]), axis=1, keepdims=True) + 1e-10),
    }

    print(f"  {'Config':<25} | {'C':>10} | {'F':>10} | {'Total stretch':>14}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}")

    for name, vor in configs.items():
        s, ms = compute_per_point_stretching(pos, vor)
        total = abs(np.sum(s))
        total_max = np.sum(ms)
        C = total / (total_max + 1e-15)
        F = 1 - C
        print(f"  {name:<25} | {C:>10.6f} | {F:>10.6f} | {np.sum(s):>14.6f}")


def synthesis():
    print("\n" + "=" * 70)
    print("SYNTHESIS: THE FRUSTRATION MECHANISM")
    print("=" * 70)
    print("""
  THE HYPOTHESIS: Biot-Savart creates "frustration" in vortex stretching.

  Each vortex i wants to be stretched by S_i (aligning omega_i with alpha_i).
  But S_i depends on all OTHER vortices. Aligning omega_i changes S_j for
  all j != i. The system cannot simultaneously maximize all stretching_i.

  This is mathematically analogous to SPIN-GLASS FRUSTRATION:
  - Spins = vorticity directions omega_i
  - Interactions = Biot-Savart coupling
  - Energy = negative stretching (want to maximize)
  - Frustration = impossibility of simultaneous maximization

  KEY PREDICTIONS:
  1. Stretching pairs (s_ij, s_ji) are anti-correlated
  2. Total stretching |sum s_i| grows as O(1), not O(N)
  3. C ~ 1/N (confirmed empirically)
  4. Frustration F = 1 - C increases with N

  WHY THIS MATTERS FOR REGULARITY:
  If C ~ 1/N for N vortex modes, then:
  |Stretching| <= C(N) * lambda_max * Z = (1/N) * lambda_max * Z

  For a continuous field with N effective modes:
  N ~ (L/eta)^3 where eta = Kolmogorov scale
  lambda_max ~ Z^{1/2} (standard scaling)

  So: |Stretching| <= Z^{3/2} / N
  And: dZ/dt <= Z^{3/2}/N - nu*Dissipation

  If N grows fast enough with Z (blow-up requires more modes),
  the stretching is SELF-LIMITING.

  TO PROVE: C ~ 1/N for the Biot-Savart kernel.
  This would be a THEOREM about the kernel structure,
  not about NS dynamics, and could be proved independently.
""")


if __name__ == "__main__":
    part1_stretching_correlations()
    part2_sum_scaling()
    part3_reciprocity()
    part4_trace_free_cancellation()
    part5_frustration_index()
    synthesis()
