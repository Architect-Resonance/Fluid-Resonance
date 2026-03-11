"""
Path 2: Random 3-SAT Spectral Analysis at Critical Threshold

Steps:
1. Generate random 3-SAT instances at alpha = 4.267 (critical threshold)
2. Build Variable Interaction Graphs (VIGs)
3. Spectral clustering to find natural bottlenecks
4. Compute spectral ratios at those bottlenecks
5. Compare with star invariant predictions

Key question: Do random 3-SAT instances near phase transition
exhibit spectral gap ratios near 1.857 at natural bottlenecks?
"""
import numpy as np
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
np.random.seed(42)  # reproducibility


def generate_random_3sat(n_vars, alpha, seed=None):
    """Generate a random 3-SAT instance.

    Args:
        n_vars: number of Boolean variables
        alpha: clause-to-variable ratio
        seed: random seed

    Returns:
        list of clauses, each clause is a tuple of 3 literals (signed ints)
    """
    if seed is not None:
        np.random.seed(seed)

    n_clauses = int(n_vars * alpha)
    clauses = []

    for _ in range(n_clauses):
        # Pick 3 distinct variables
        vars_chosen = np.random.choice(n_vars, size=3, replace=False)
        # Each literal is positive or negative with equal probability
        signs = np.random.choice([-1, 1], size=3)
        clause = tuple(int(v * s) for v, s in zip(vars_chosen + 1, signs))  # 1-indexed
        clauses.append(clause)

    return clauses


def build_vig(n_vars, clauses):
    """Build Variable Interaction Graph from 3-SAT clauses.

    Unsigned VIG: edge between variables that appear in the same clause.
    """
    A = np.zeros((n_vars, n_vars))
    for clause in clauses:
        vars_in_clause = [abs(lit) - 1 for lit in clause]  # 0-indexed
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = vars_in_clause[i], vars_in_clause[j]
                A[vi][vj] += 1
                A[vj][vi] += 1

    # Binary (unweighted) version
    A_binary = (A > 0).astype(float)
    return A_binary, A  # return both binary and weighted


def spectral_partition(A):
    """Find the spectral bisection (Fiedler partition).

    Returns: (partition_0, partition_1, fiedler_value, fiedler_vector)
    """
    n = A.shape[0]
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Fiedler vector is the eigenvector for the 2nd smallest eigenvalue
    fiedler_idx = 1  # 0 is the trivial zero eigenvalue
    fiedler_value = eigenvalues[fiedler_idx]
    fiedler_vector = eigenvectors[:, fiedler_idx]

    # Partition by sign of Fiedler vector
    p0 = set(i for i in range(n) if fiedler_vector[i] <= 0)
    p1 = set(i for i in range(n) if fiedler_vector[i] > 0)

    return p0, p1, fiedler_value, fiedler_vector


def find_bottleneck_vars(A, p0, p1, k=3):
    """Find the k variables most involved in cross-partition edges.

    These are the "bridge variables" — natural candidates for the valve.
    """
    n = A.shape[0]
    cross_degree = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if A[i][j] > 0:
                if (i in p0 and j in p1) or (i in p1 and j in p0):
                    cross_degree[i] += A[i][j]

    # Top k by cross-degree
    top_k = np.argsort(cross_degree)[-k:][::-1]
    return set(top_k), cross_degree


def compute_spectral_ratio(A, valve_vars):
    """Compute L2(full) / L2(reduced after valve removal)."""
    n = A.shape[0]
    D = np.diag(A.sum(axis=1))
    L = D - A
    eig_full = np.sort(np.linalg.eigvalsh(L))

    if eig_full[1] < 1e-10:
        return None, eig_full[1], None

    remaining = sorted(set(range(n)) - valve_vars)
    if len(remaining) < 2:
        return None, eig_full[1], None

    A_red = A[np.ix_(remaining, remaining)]
    D_red = np.diag(A_red.sum(axis=1))
    L_red = D_red - A_red
    eig_red = np.sort(np.linalg.eigvalsh(L_red))

    if eig_red[1] < 1e-10:
        return float('inf'), eig_full[1], 0

    return eig_full[1] / eig_red[1], eig_full[1], eig_red[1]


def recursive_partition(A, depth=0, max_depth=3, min_size=8):
    """Recursively bisect the graph and compute ratios at each cut."""
    n = A.shape[0]
    results = []

    if n < min_size or depth >= max_depth:
        return results

    p0, p1, fiedler, fvec = spectral_partition(A)

    if len(p0) < 3 or len(p1) < 3:
        return results

    # Find bottleneck variables
    for k in [3, 5, 7]:
        valve, cross_deg = find_bottleneck_vars(A, p0, p1, k=k)
        ratio, l2_full, l2_red = compute_spectral_ratio(A, valve)
        results.append({
            "depth": depth,
            "n": n,
            "cut_sizes": (len(p0), len(p1)),
            "valve_size": k,
            "ratio": ratio,
            "fiedler": fiedler,
            "l2_full": l2_full,
            "l2_red": l2_red,
        })

    # Recurse on each partition
    if len(p0) >= min_size:
        p0_list = sorted(p0)
        A_sub = A[np.ix_(p0_list, p0_list)]
        results.extend(recursive_partition(A_sub, depth + 1, max_depth, min_size))

    if len(p1) >= min_size:
        p1_list = sorted(p1)
        A_sub = A[np.ix_(p1_list, p1_list)]
        results.extend(recursive_partition(A_sub, depth + 1, max_depth, min_size))

    return results


# ============================================================
# MAIN EXPERIMENT
# ============================================================
print("=" * 90)
print("PATH 2: RANDOM 3-SAT SPECTRAL ANALYSIS AT CRITICAL THRESHOLD")
print("=" * 90)

# Parameters
ALPHAS = [3.0, 4.0, 4.267, 4.5, 5.0]
N_VARS_LIST = [50, 100, 200]
N_INSTANCES = 20  # per (alpha, n_vars) pair
VALVE_SIZES = [3, 5, 7]

all_results = []
t_start = time.time()

for n_vars in N_VARS_LIST:
    for alpha in ALPHAS:
        print(f"\n--- N={n_vars}, alpha={alpha} ({N_INSTANCES} instances) ---")

        ratios_by_valve = defaultdict(list)
        fiedler_values = []
        disconnected = 0

        for inst in range(N_INSTANCES):
            seed = 1000 * n_vars + 100 * int(alpha * 100) + inst
            clauses = generate_random_3sat(n_vars, alpha, seed=seed)
            A_bin, A_weighted = build_vig(n_vars, clauses)

            # Check connectivity
            D = np.diag(A_bin.sum(axis=1))
            L = D - A_bin
            eig = np.sort(np.linalg.eigvalsh(L))

            if eig[1] < 1e-10:
                disconnected += 1
                continue

            fiedler_values.append(eig[1])

            # Spectral partition
            p0, p1, fiedler, fvec = spectral_partition(A_bin)

            # Compute ratios for different valve sizes
            for k in VALVE_SIZES:
                valve, _ = find_bottleneck_vars(A_bin, p0, p1, k=k)
                ratio, l2f, l2r = compute_spectral_ratio(A_bin, valve)
                if ratio is not None and ratio != float('inf'):
                    ratios_by_valve[k].append(ratio)

        # Report
        if disconnected > 0:
            print(f"  Disconnected: {disconnected}/{N_INSTANCES}")

        if fiedler_values:
            print(f"  Fiedler (spectral gap): mean={np.mean(fiedler_values):.4f}, "
                  f"std={np.std(fiedler_values):.4f}")

        for k in VALVE_SIZES:
            vals = ratios_by_valve[k]
            if vals:
                near_target = sum(1 for v in vals if abs(v - 1.857) < 0.15)
                print(f"  Valve k={k}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                      f"range=[{min(vals):.4f}, {max(vals):.4f}], "
                      f"near 1.857 (+/-0.15): {near_target}/{len(vals)}")

                all_results.append({
                    "n_vars": n_vars,
                    "alpha": alpha,
                    "valve_k": k,
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "near_target": near_target,
                    "count": len(vals),
                })

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.1f}s")

# ============================================================
# Step 3: Analysis — does alpha=4.267 show special behavior?
# ============================================================
print("\n" + "=" * 90)
print("ANALYSIS: SPECTRAL RATIO VS ALPHA (at spectral bisection bottleneck)")
print("=" * 90)

# Group by (n_vars, valve_k) and compare across alpha
for n_vars in N_VARS_LIST:
    for k in VALVE_SIZES:
        print(f"\n  N={n_vars}, valve_k={k}:")
        subset = [r for r in all_results if r["n_vars"] == n_vars and r["valve_k"] == k]
        for r in subset:
            bar_len = int(r["mean"] * 10)
            bar = "#" * bar_len
            marker = " <--- critical" if abs(r["alpha"] - 4.267) < 0.01 else ""
            print(f"    alpha={r['alpha']:.3f}: mean={r['mean']:.4f} +/- {r['std']:.4f} "
                  f"[{r['min']:.3f}, {r['max']:.3f}] {bar}{marker}")

# ============================================================
# Step 4: Recursive decomposition at critical alpha
# ============================================================
print("\n" + "=" * 90)
print("RECURSIVE SPECTRAL DECOMPOSITION (alpha=4.267, N=200)")
print("=" * 90)

for inst in range(5):
    seed = 99999 + inst
    clauses = generate_random_3sat(200, 4.267, seed=seed)
    A_bin, _ = build_vig(200, clauses)

    print(f"\n  Instance {inst} (seed={seed}):")
    results = recursive_partition(A_bin, max_depth=3, min_size=15)

    for r in results:
        ratio_str = f"{r['ratio']:.4f}" if r['ratio'] is not None else "DISC"
        near = "*" if r['ratio'] is not None and abs(r['ratio'] - 1.857) < 0.15 else " "
        print(f"    depth={r['depth']}, n={r['n']:>3d}, cut={r['cut_sizes']}, "
              f"valve={r['valve_size']}, ratio={ratio_str}{near}, "
              f"fiedler={r['fiedler']:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("SUMMARY: Does the star invariant appear in random 3-SAT?")
print("=" * 90)

# Collect all ratios at critical alpha
crit_ratios = []
for r in all_results:
    if abs(r["alpha"] - 4.267) < 0.01:
        crit_ratios.append(r)

if crit_ratios:
    all_means = [r["mean"] for r in crit_ratios]
    print(f"\n  At alpha=4.267 (critical threshold):")
    print(f"    Overall mean ratio: {np.mean(all_means):.4f}")
    print(f"    Range of means: [{min(all_means):.4f}, {max(all_means):.4f}]")
    print(f"    Star invariant target: 1.8573")
    print(f"    Deviation from target: {abs(np.mean(all_means) - 1.8573):.4f}")

    # Compare with off-critical
    off_crit = [r for r in all_results if abs(r["alpha"] - 4.267) > 0.1]
    if off_crit:
        off_means = [r["mean"] for r in off_crit]
        print(f"\n  At other alpha values:")
        print(f"    Overall mean ratio: {np.mean(off_means):.4f}")
        print(f"    Range of means: [{min(off_means):.4f}, {max(off_means):.4f}]")

    # Statistical test: is alpha=4.267 closer to 1.857?
    crit_dev = [abs(r["mean"] - 1.857) for r in crit_ratios]
    off_dev = [abs(r["mean"] - 1.857) for r in off_crit] if off_crit else []

    if crit_dev and off_dev:
        print(f"\n  Mean deviation from 1.857:")
        print(f"    Critical (alpha=4.267): {np.mean(crit_dev):.4f}")
        print(f"    Off-critical:           {np.mean(off_dev):.4f}")

        if np.mean(crit_dev) < np.mean(off_dev):
            print(f"    --> Critical is CLOSER to star invariant")
        else:
            print(f"    --> No special proximity at critical threshold")

print(f"\n  VERDICT: ", end="")
# The verdict depends on whether critical-alpha ratios are closer to 1.857
# This is the empirical test the user needs for their paper
if crit_ratios:
    mean_crit = np.mean([r["mean"] for r in crit_ratios])
    if abs(mean_crit - 1.857) < 0.1:
        print("Spectral ratio at critical alpha is within 0.1 of star invariant.")
        print("  This supports a connection but is NOT proof — could be coincidence.")
    elif abs(mean_crit - 1.857) < 0.3:
        print("Spectral ratio at critical alpha is within 0.3 of star invariant.")
        print("  Suggestive but NOT compelling — more data needed.")
    else:
        print(f"Spectral ratio at critical alpha ({mean_crit:.4f}) is NOT close to 1.857.")
        print("  The star invariant does NOT appear to govern random 3-SAT bottlenecks.")
