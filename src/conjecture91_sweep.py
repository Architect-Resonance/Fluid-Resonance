"""
Conjecture 9.1 — Extended Parametric Sweep
===========================================
Goal: Find the supremum of R = lambda_min(L_eff) / lambda_min(L_eff_red)
across all symmetric star-cluster systems with bridge width >= 4.

Sweep: K3-K30 cores, 0-15 anchors, bridge widths 4-15.
Also computes analytic quantities needed for the proof:
- max neighbor count of remaining vertices to valve (Delta_max)
- ratio of bridge grounding min/max
- eigenvalue gap structure

Author: Claude (Opus 4.6), 2026-03-11
"""
import numpy as np
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")


def build_cluster_edges(core_size, n_anchors):
    """Build edge set for a cluster: K_core + anchors + connector."""
    edges = set()
    for i in range(core_size):
        for j in range(i + 1, core_size):
            edges.add((i, j))
    for a in range(n_anchors):
        anchor_var = core_size + a
        c1 = (2 * a) % core_size
        c2 = (2 * a + 2) % core_size
        if c2 == c1:
            c2 = (c1 + 1) % core_size
        edges.add((min(anchor_var, c1), max(anchor_var, c1)))
        edges.add((min(anchor_var, c2), max(anchor_var, c2)))
    n_vars = core_size + n_anchors
    if n_anchors >= 2:
        conn = n_vars
        n_vars += 1
        for a in range(n_anchors):
            anchor_var = core_size + a
            edges.add((min(conn, anchor_var), max(conn, anchor_var)))
        for i in range(n_anchors):
            for j in range(i + 1, n_anchors):
                edges.add((core_size + i, core_size + j))
    return n_vars, edges


def compute_invariant(core_size, n_anchors, bridge_width):
    """Compute R and diagnostic quantities for one configuration."""
    n_vars, cluster_edges = build_cluster_edges(core_size, n_anchors)

    A = np.zeros((n_vars, n_vars))
    for i, j in cluster_edges:
        A[i][j] = 1
        A[j][i] = 1

    # Bridge grounding: each clause i adds 2 to D_bridge[i % core_size]
    D_bridge = np.zeros(n_vars)
    for clause_i in range(bridge_width):
        p2 = clause_i % core_size
        D_bridge[p2] += 2

    # Grounded spoke Laplacian
    L_eff = np.diag(A.sum(axis=1) + D_bridge) - A
    eig_full = np.sort(np.linalg.eigvalsh(L_eff))
    lmin_full = eig_full[0]

    if lmin_full < 1e-12:
        return None

    # Valve: remove last 2 bridge-touched spoke positions
    spoke_positions = sorted(set(i % core_size for i in range(bridge_width)))
    if len(spoke_positions) >= 2:
        valve = set(spoke_positions[-2:])
    elif len(spoke_positions) == 1:
        valve = set(spoke_positions)
    else:
        return None

    keep = sorted(set(range(n_vars)) - valve)
    if len(keep) < 2:
        return None

    A_red = A[np.ix_(keep, keep)]
    D_bridge_red = D_bridge[keep]
    L_red = np.diag(A_red.sum(axis=1) + D_bridge_red) - A_red
    eig_red = np.sort(np.linalg.eigvalsh(L_red))
    lmin_red = eig_red[0]

    if lmin_red < 1e-12:
        return None

    R = lmin_full / lmin_red

    # Diagnostic: neighbor count of remaining vertices to valve
    delta = np.zeros(len(keep))
    for idx, v in enumerate(keep):
        for u in valve:
            if A[v, u] > 0:
                delta[idx] += 1
    delta_max = delta.max()

    # Diagnostic: grounding structure
    grounding_min = D_bridge.min()
    grounding_max = D_bridge.max()
    grounding_min_remaining = D_bridge_red.min()

    return {
        "R": R,
        "lmin_full": lmin_full,
        "lmin_red": lmin_red,
        "n_vars": n_vars,
        "n_red": len(keep),
        "valve_size": len(valve),
        "delta_max": delta_max,
        "grounding_min": grounding_min,
        "grounding_max": grounding_max,
        "grounding_min_remaining": grounding_min_remaining,
        "eig_full_2nd": eig_full[1] if len(eig_full) > 1 else None,
        "eig_red_2nd": eig_red[1] if len(eig_red) > 1 else None,
    }


# ============================================================
# MAIN SWEEP
# ============================================================
print("=" * 90)
print("CONJECTURE 9.1 — EXTENDED PARAMETRIC SWEEP")
print("Goal: Find sup(R) for bridge width >= 4")
print("=" * 90)

t_start = time.time()

# Phase 1: Wide sweep
results = []
max_R = 0
max_R_config = None
count = 0
degen = 0

core_range = range(3, 31)
anchor_range = range(0, 16)
bridge_range = range(4, 16)

for core_size in core_range:
    for n_anchors in anchor_range:
        for bridge_width in bridge_range:
            if bridge_width > core_size:
                continue
            r = compute_invariant(core_size, n_anchors, bridge_width)
            count += 1
            if r is None:
                degen += 1
                continue
            results.append({
                "core": core_size,
                "anchors": n_anchors,
                "bridge": bridge_width,
                **r,
            })
            if r["R"] > max_R:
                max_R = r["R"]
                max_R_config = (core_size, n_anchors, bridge_width)

elapsed = time.time() - t_start
print(f"\nSwept {count} configurations ({degen} degenerate) in {elapsed:.1f}s")
print(f"Valid results: {len(results)}")

# ============================================================
# ANALYSIS
# ============================================================

print("\n" + "=" * 90)
print("1. GLOBAL MAXIMUM R")
print("=" * 90)
print(f"\n  sup(R) = {max_R:.10f}")
print(f"  Achieved at: K{max_R_config[0]}, {max_R_config[1]} anchors, bridge width {max_R_config[2]}")

# Top 20 highest R values
results_sorted = sorted(results, key=lambda x: -x["R"])
print(f"\n  Top 20 highest R values:")
print(f"  {'Core':>5s} {'Anch':>5s} {'Brg':>5s} {'R':>12s} {'lmin_full':>12s} {'lmin_red':>12s} {'delta_max':>10s}")
for r in results_sorted[:20]:
    print(f"  K{r['core']:<4d} {r['anchors']:>5d} {r['bridge']:>5d} {r['R']:>12.8f} {r['lmin_full']:>12.8f} {r['lmin_red']:>12.8f} {r['delta_max']:>10.1f}")

# Check if ALL R < 2
all_below_2 = all(r["R"] < 2.0 for r in results)
print(f"\n  ALL R < 2.0? {'YES' if all_below_2 else 'NO'}")
if not all_below_2:
    violators = [r for r in results if r["R"] >= 2.0]
    print(f"  VIOLATIONS: {len(violators)}")
    for v in violators[:10]:
        print(f"    K{v['core']}, {v['anchors']}a, w={v['bridge']}: R = {v['R']:.8f}")

print("\n" + "=" * 90)
print("2. MONOTONICITY IN BRIDGE WIDTH")
print("=" * 90)
print("  For each (core, anchors), how does R change as bridge_width increases?")

by_core_anchor = defaultdict(list)
for r in results:
    by_core_anchor[(r["core"], r["anchors"])].append(r)

monotone_count = 0
non_monotone_count = 0
non_monotone_examples = []

for key, configs in by_core_anchor.items():
    configs.sort(key=lambda x: x["bridge"])
    R_values = [c["R"] for c in configs]
    # Check if R is monotonically decreasing in bridge width
    is_decreasing = all(R_values[i] >= R_values[i+1] for i in range(len(R_values)-1))
    if is_decreasing:
        monotone_count += 1
    else:
        non_monotone_count += 1
        if len(non_monotone_examples) < 5:
            non_monotone_examples.append((key, [(c["bridge"], c["R"]) for c in configs]))

print(f"\n  Monotonically decreasing in w: {monotone_count}")
print(f"  Non-monotonic in w: {non_monotone_count}")
for key, vals in non_monotone_examples:
    print(f"    K{key[0]}, {key[1]}a: {[(w, f'{R:.6f}') for w, R in vals]}")

print("\n" + "=" * 90)
print("3. MONOTONICITY IN CORE SIZE (fixed anchors, fixed bridge)")
print("=" * 90)

by_anchor_bridge = defaultdict(list)
for r in results:
    by_anchor_bridge[(r["anchors"], r["bridge"])].append(r)

core_monotone = 0
core_non_monotone = 0
core_examples = []

for key, configs in by_anchor_bridge.items():
    configs.sort(key=lambda x: x["core"])
    R_values = [c["R"] for c in configs]
    is_decreasing = all(R_values[i] >= R_values[i+1] for i in range(len(R_values)-1))
    if is_decreasing:
        core_monotone += 1
    else:
        core_non_monotone += 1
        if len(core_examples) < 5:
            core_examples.append((key, [(c["core"], c["R"]) for c in configs]))

print(f"\n  Monotonically decreasing in core size: {core_monotone}")
print(f"  Non-monotonic in core size: {core_non_monotone}")
for key, vals in core_examples:
    print(f"    {key[0]}a, w={key[1]}: {[(n, f'{R:.6f}') for n, R in vals[:10]]}")

print("\n" + "=" * 90)
print("4. ASYMPTOTIC BEHAVIOR (large bridge width)")
print("=" * 90)
print("  R as bridge_width -> core_size (maximum for given core)")

for core in [3, 5, 7, 10, 15, 20, 30]:
    subset = [r for r in results if r["core"] == core and r["anchors"] == 0]
    if subset:
        subset.sort(key=lambda x: x["bridge"])
        print(f"\n  K{core}, 0 anchors:")
        for r in subset:
            print(f"    w={r['bridge']:>2d}: R = {r['R']:.10f}  (lmin_full={r['lmin_full']:.6f}, lmin_red={r['lmin_red']:.6f})")

print("\n" + "=" * 90)
print("5. BRIDGE WIDTH = 4 (worst case for conjecture)")
print("=" * 90)

bw4 = [r for r in results if r["bridge"] == 4]
bw4.sort(key=lambda x: -x["R"])
print(f"\n  {len(bw4)} configurations with bridge width 4")
print(f"  Max R = {bw4[0]['R']:.10f} at K{bw4[0]['core']}, {bw4[0]['anchors']}a")
print(f"  Min R = {bw4[-1]['R']:.10f} at K{bw4[-1]['core']}, {bw4[-1]['anchors']}a")
print(f"\n  Top 10:")
for r in bw4[:10]:
    print(f"    K{r['core']:<3d} {r['anchors']:>2d}a: R = {r['R']:.10f}")

print("\n" + "=" * 90)
print("6. KEY QUANTITY: delta_max vs R")
print("=" * 90)
print("  delta_max = max neighbors of remaining vertex to valve")
print("  This is the 'interlacing gap' — how much diagonal drops")

by_delta = defaultdict(list)
for r in results:
    by_delta[int(r["delta_max"])].append(r["R"])

for delta in sorted(by_delta.keys()):
    vals = by_delta[delta]
    print(f"  delta_max = {delta}: count={len(vals):>5d}, R in [{min(vals):.6f}, {max(vals):.6f}], mean={np.mean(vals):.6f}")

print("\n" + "=" * 90)
print("7. EIGENVALUE RATIO STRUCTURE")
print("=" * 90)
print("  Looking for R = lmin_full / lmin_red patterns")

# For bridge width 4, plot R vs core size for 0 anchors
print("\n  R vs core_size (0 anchors, bridge width 4):")
for core in range(4, 31):
    r = next((x for x in results if x["core"] == core and x["anchors"] == 0 and x["bridge"] == 4), None)
    if r:
        gap_ratio = r["eig_full_2nd"] / r["lmin_full"] if r["lmin_full"] > 0 else 0
        print(f"    K{core:>2d}: R = {r['R']:.10f}  gap_ratio = {gap_ratio:.4f}  delta_max = {r['delta_max']:.0f}")

# Check: does R converge as core_size -> infinity?
print("\n" + "=" * 90)
print("8. CONVERGENCE CHECK (does R stabilize for large cores?)")
print("=" * 90)

for bw in [4, 5, 6, 8, 10]:
    subset = [r for r in results if r["bridge"] == bw and r["anchors"] == 0]
    subset.sort(key=lambda x: x["core"])
    if len(subset) >= 5:
        R_last5 = [r["R"] for r in subset[-5:]]
        R_range = max(R_last5) - min(R_last5)
        print(f"  w={bw:>2d}: last 5 R values = {[f'{v:.8f}' for v in R_last5]}, range = {R_range:.2e}")

# Phase 2: Extra-large cores for convergence (K50, K100)
print("\n" + "=" * 90)
print("9. EXTRA-LARGE CORE TEST (K50, K100)")
print("=" * 90)

for core in [50, 100]:
    for bw in [4, 5, 10, 20]:
        if bw > core:
            continue
        r = compute_invariant(core, 0, bw)
        if r:
            print(f"  K{core}, 0a, w={bw}: R = {r['R']:.10f}")

print("\n" + "=" * 90)
print("10. THEORETICAL BOUND ATTEMPT")
print("=" * 90)

# For every config, compute: R vs (1 + delta_max / lmin_full)
# If R <= 1 + delta_max / (lmin_full + epsilon), this gives a bound
print("\n  Checking if R <= 1 + delta_max / lmin_red (Weyl-type bound):")
weyl_holds = 0
weyl_fails = 0
for r in results:
    bound = 1 + r["delta_max"] / r["lmin_red"]
    if r["R"] <= bound + 1e-10:
        weyl_holds += 1
    else:
        weyl_fails += 1

print(f"  R <= 1 + delta_max/lmin_red: holds={weyl_holds}, fails={weyl_fails}")

# Check a tighter bound: R <= 1 + delta_max / (lmin_full - delta_max) for configs where lmin_full > delta_max
print("\n  Checking R <= delta_max / (lmin_red - something) patterns:")
for r in results_sorted[:5]:
    ratio_diag = r["delta_max"] / r["lmin_red"] if r["lmin_red"] > 0 else float("inf")
    print(f"    K{r['core']}, {r['anchors']}a, w={r['bridge']}: R={r['R']:.8f}, delta_max={r['delta_max']:.0f}, "
          f"lmin_red={r['lmin_red']:.6f}, delta_max/lmin_red={ratio_diag:.4f}")

# Final summary
print("\n" + "=" * 90)
print("FINAL VERDICT")
print("=" * 90)
print(f"  Total configurations tested: {len(results)}")
print(f"  Bridge width range: 4-15")
print(f"  Core size range: K3-K30 (+ K50, K100)")
print(f"  Anchor range: 0-15")
print(f"  Global maximum R: {max_R:.10f}")
print(f"  At configuration: K{max_R_config[0]}, {max_R_config[1]} anchors, bridge width {max_R_config[2]}")
print(f"  R < 2.0 for ALL tested configurations: {'YES' if all_below_2 else 'NO'}")
print(f"  Margin to 2.0: {2.0 - max_R:.10f}")
