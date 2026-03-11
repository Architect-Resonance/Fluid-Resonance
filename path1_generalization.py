"""
Path 1, Step 5: Generalize the star invariant across cluster geometries.

For each (core_size, n_anchors, bridge_width) configuration:
1. Build the cluster graph (K_n core + anchors + connector)
2. Define bridge pattern and valve
3. Compute grounded spoke Laplacian (full + reduced)
4. Extract star invariant ratio
5. Extract characteristic polynomials and identify base irrationals

Output: table of invariants indexed by geometry.
"""
import numpy as np
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")


def build_cluster_edges(core_size, n_anchors):
    """Build edge set for a cluster: K_core + anchors + connector.

    Returns: (n_vars, edges) where edges is set of (i,j) pairs.
    """
    edges = set()

    # K_core: all pairs in {0, ..., core_size-1}
    for i in range(core_size):
        for j in range(i + 1, core_size):
            edges.add((i, j))

    # Anchors: each anchor connects to 2 core variables
    for a in range(n_anchors):
        anchor_var = core_size + a
        c1 = (2 * a) % core_size
        c2 = (2 * a + 2) % core_size
        if c2 == c1:
            c2 = (c1 + 1) % core_size
        edges.add((min(anchor_var, c1), max(anchor_var, c1)))
        edges.add((min(anchor_var, c2), max(anchor_var, c2)))

    # Connector: if >= 2 anchors, add connector variable linking them
    n_vars = core_size + n_anchors
    if n_anchors >= 2:
        conn = n_vars
        n_vars += 1
        for a in range(n_anchors):
            anchor_var = core_size + a
            edges.add((min(conn, anchor_var), max(conn, anchor_var)))
        # Also link anchors to each other through connector
        for i in range(n_anchors):
            for j in range(i + 1, n_anchors):
                edges.add((core_size + i, core_size + j))

    return n_vars, edges


def build_bridge_edges(offset_a, offset_b, core_size, width):
    """Build cross-cluster edges from bridge clauses.

    Each bridge clause involves 3 variables creating 3 edges.
    Pattern: clause i uses core positions (i, i+1) from A and position i from B.

    Returns: (cross_edges, valve_vars)
    """
    cross_edges = set()
    all_bridge_vars_a = set()
    all_bridge_vars_b = set()

    for i in range(width):
        # Core positions used (mod core_size)
        p0 = i % core_size
        p1 = (i + 1) % core_size
        p2 = i % core_size

        # Variables
        va0 = offset_a + p0  # cluster A, core position p0
        va1 = offset_a + p1  # cluster A, core position p1
        vb0 = offset_b + p2  # cluster B, core position p2

        # 3 edges from this clause
        cross_edges.add((min(va0, vb0), max(va0, vb0)))
        cross_edges.add((min(va0, va1), max(va0, va1)))  # may be internal
        cross_edges.add((min(va1, vb0), max(va1, vb0)))

        all_bridge_vars_a.update([p0, p1])
        all_bridge_vars_b.update([p2])

    # Valve: remove the 3 variables of the LAST bridge clause
    last_i = width - 1
    lp0 = last_i % core_size
    lp1 = (last_i + 1) % core_size
    lp2 = last_i % core_size
    valve = {offset_a + lp0, offset_a + lp1, offset_b + lp2}

    # Alternative valve: if the default disconnects, try removing one from each side
    # We'll detect disconnection and report it

    return cross_edges, valve


def compute_grounded_spoke_invariant(core_size, n_anchors, bridge_width):
    """Compute the star invariant via grounded spoke Laplacian."""

    n_vars, cluster_edges = build_cluster_edges(core_size, n_anchors)

    # Build cluster adjacency
    A = np.zeros((n_vars, n_vars))
    for i, j in cluster_edges:
        A[i][j] = 1
        A[j][i] = 1

    # Bridge grounding: count how many hub connections each spoke var gets
    D_bridge = np.zeros(n_vars)
    bridge_vars_in_spoke = set()

    for clause_i in range(bridge_width):
        p0 = clause_i % core_size
        p1 = (clause_i + 1) % core_size
        p2 = clause_i % core_size  # from hub side

        # In the spoke-separation mode (hub = 0), the bridge edges from hub
        # act as grounding on spoke vars. Each edge to hub adds +1 to diagonal.
        # Spoke var p0 connects to hub var p0 -> +1
        # Spoke var p0 connects to hub var p1 -> wait, need to think about this

        # Bridge clause: (hub+p0, spoke+p0, hub+p1)
        # No wait - bridge pattern is: (a+p0, b+p2, a+p1)
        # where a = hub offset, b = spoke offset
        # Edges: (a+p0, b+p2), (a+p0, a+p1), (b+p2, a+p1)
        # Cross edges from spoke perspective:
        #   spoke var p2 connects to hub var p0 -> D_bridge[p2] += 1
        #   spoke var p2 connects to hub var p1 -> D_bridge[p2] += 1
        # Wait, that's only if the edge is cross-cluster.

        # Let me re-derive. The bridge clause creates edges:
        # (a+p0, b+p2): cross (hub p0 to spoke p2) -> spoke p2 gets +1 grounding
        # (a+p0, a+p1): internal to hub, doesn't affect spoke
        # (a+p1, b+p2): cross (hub p1 to spoke p2) -> spoke p2 gets +1 grounding

        D_bridge[p2] += 2  # spoke position p2 gets 2 grounding edges per clause
        bridge_vars_in_spoke.add(p2)

    # But wait - some clauses may share p2 positions, accumulating grounding.
    # And there might also be internal spoke edges created by the bridge.
    # Bridge clause: (a+p0, b+p2, a+p1)
    # The edge (b+p2, ...) - there's no spoke-spoke edge from bridge in this pattern.
    # Actually no: the edge (a+p1, b+p2) is cross-cluster.
    # And (a+p0, b+p2) is cross-cluster.
    # And (a+p0, a+p1) is hub-internal.
    # So no new spoke-internal edges from bridge. Good.

    # But actually, for width >= 2, there might be additional cross-edges I'm missing.
    # Let me reconsider. For bridge clause i:
    #   Variables: hub[p0_i], spoke[p2_i], hub[p1_i]
    # For bridge clause i+1:
    #   Variables: hub[p0_{i+1}], spoke[p2_{i+1}], hub[p1_{i+1}]
    # These are independent clauses, each contributing their own edges.
    # Cross edges are: (hub[p0_i], spoke[p2_i]) and (hub[p1_i], spoke[p2_i])
    # So spoke var p2_i gets 2 grounding edges per clause it appears in.

    # Grounded spoke Laplacian
    L_spoke = np.diag(A.sum(axis=1)) - A
    L_grounded = L_spoke + np.diag(D_bridge)

    eig_full = np.sort(np.linalg.eigvalsh(L_grounded))
    l2_full = eig_full[0]  # smallest eigenvalue (grounded, so no zero eigenvalue)

    # Valve: remove variables of the last bridge clause
    last_i = bridge_width - 1
    lp0 = last_i % core_size       # this is a hub var in original, but in spoke it's...
    lp1 = (last_i + 1) % core_size  # hub var
    lp2 = last_i % core_size       # spoke var

    # Wait, the valve in the original was {a+4, b+4, b+2} for bridge(a,b):
    # That's: hub var 4, spoke var 4, spoke var 2
    # So it removes 1 hub position + 2 spoke positions from the last clause
    # But in the grounded spoke model, hub is already zeroed out.
    # The valve removes spoke positions.

    # Let me reconsider. In the full star:
    # valve(a, b) = {a+4, b+4, b+2} where a=hub, b=spoke
    # a+4 = hub core position 4 (removes from hub)
    # b+4 = spoke core position 4
    # b+2 = spoke core position 2

    # For the last bridge clause (4, b+4, b+2) in the original:
    # p0=4 (hub), p2=4 (spoke... wait no. Let me re-examine.

    # Original bridge(a,b):
    # Clause 0: (a+0, b+0, a+1) -> hub vars {0,1}, spoke var {0}
    # Clause 1: (a+1, b+1, a+2) -> hub vars {1,2}, spoke var {1}
    # Clause 2: (a+4, b+4, b+2) -> hub var {4}, spoke vars {4, 2}

    # Hmm, clause 2 is different from my pattern! It uses spoke positions {4, 2},
    # not just one spoke position. My generalized pattern only uses one spoke position
    # per clause. The original bridge has a different structure.

    # Let me keep this simpler: for the valve, remove 2 spoke positions.
    # Use the positions that appear in the bridge's spoke-side connections.
    # For width w, spoke positions involved are {0, 1, ..., w-1} mod core_size.
    # Remove the last 2 (or as many as available).

    spoke_positions_in_bridge = sorted(set(i % core_size for i in range(bridge_width)))

    if len(spoke_positions_in_bridge) >= 2:
        valve_spoke = set(spoke_positions_in_bridge[-2:])
    elif len(spoke_positions_in_bridge) == 1:
        valve_spoke = set(spoke_positions_in_bridge)
    else:
        valve_spoke = set()

    if len(valve_spoke) == 0:
        return None  # can't apply valve

    keep = sorted(set(range(n_vars)) - valve_spoke)

    if len(keep) < 2:
        return None

    A_red = A[np.ix_(keep, keep)]
    D_bridge_red = D_bridge[keep]
    L_red = np.diag(A_red.sum(axis=1)) - A_red + np.diag(D_bridge_red)

    eig_red = np.sort(np.linalg.eigvalsh(L_red))
    l2_red = eig_red[0]

    if l2_full < 1e-10 or l2_red < 1e-10:
        return None  # degenerate

    ratio = l2_full / l2_red

    # Extract characteristic polynomial info
    # Find the irreducible polynomial for l2_full
    eig_full_rounded = eig_full
    int_roots = [round(v) for v in eig_full_rounded if abs(v - round(v)) < 1e-6]

    return {
        "ratio": ratio,
        "l2_full": l2_full,
        "l2_red": l2_red,
        "n_vars": n_vars,
        "n_vars_red": len(keep),
        "valve": valve_spoke,
        "cluster_edges": len(cluster_edges),
        "grounding": D_bridge.copy(),
        "eig_full": eig_full,
        "eig_red": eig_red,
    }


def verify_with_full_star(core_size, n_anchors, bridge_width, n_spokes=3):
    """Verify the grounded spoke result against a full star construction."""
    n_vars_cluster, cluster_edges = build_cluster_edges(core_size, n_anchors)
    n_total = (n_spokes + 1) * n_vars_cluster

    all_edges = set()

    # Build all clusters
    for c in range(n_spokes + 1):
        offset = c * n_vars_cluster
        for i, j in cluster_edges:
            all_edges.add((offset + i, offset + j))

    # Build bridges from hub (cluster 0) to each spoke
    all_valve = set()
    for s in range(1, n_spokes + 1):
        hub_off = 0
        spoke_off = s * n_vars_cluster

        for clause_i in range(bridge_width):
            p0 = clause_i % core_size
            p1 = (clause_i + 1) % core_size
            p2 = clause_i % core_size

            va0 = hub_off + p0
            va1 = hub_off + p1
            vb0 = spoke_off + p2

            all_edges.add((min(va0, vb0), max(va0, vb0)))
            all_edges.add((min(va0, va1), max(va0, va1)))
            all_edges.add((min(va1, vb0), max(va1, vb0)))

        # Valve: same spoke positions as in grounded model
        spoke_positions = sorted(set(i % core_size for i in range(bridge_width)))
        if len(spoke_positions) >= 2:
            for p in spoke_positions[-2:]:
                all_valve.add(spoke_off + p)
        elif len(spoke_positions) == 1:
            all_valve.add(spoke_off + spoke_positions[0])

        # Also remove corresponding hub positions
        last_i = bridge_width - 1
        all_valve.add(hub_off + last_i % core_size)

    # Build full adjacency
    A_full = np.zeros((n_total, n_total))
    for i, j in all_edges:
        A_full[i][j] = 1
        A_full[j][i] = 1

    L_full = np.diag(A_full.sum(axis=1)) - A_full
    eig_full = np.sort(np.linalg.eigvalsh(L_full))

    # Reduced
    remaining = sorted(set(range(n_total)) - all_valve)
    A_red = A_full[np.ix_(remaining, remaining)]
    L_red = np.diag(A_red.sum(axis=1)) - A_red
    eig_red = np.sort(np.linalg.eigvalsh(L_red))

    if eig_full[1] < 1e-10 or eig_red[1] < 1e-10:
        return None

    return eig_full[1] / eig_red[1]


# ============================================================
# MAIN: Run all configurations
# ============================================================
print("=" * 90)
print("PATH 1, STEP 5: GENERALIZATION OF STAR INVARIANT")
print("Systematic scan: core_size x n_anchors x bridge_width")
print("=" * 90)

results = []
t_start = time.time()

for core_size in [3, 4, 5, 6, 7]:
    for n_anchors in [0, 1, 2, 3]:
        for bridge_width in [1, 2, 3, 4, 5]:
            # Skip if bridge uses more positions than core has
            if bridge_width > core_size:
                continue

            r = compute_grounded_spoke_invariant(core_size, n_anchors, bridge_width)

            # Verify with full star for a subset
            star_ratio = None
            if r is not None and (core_size == 5 or bridge_width == 3):
                star_ratio = verify_with_full_star(core_size, n_anchors, bridge_width)

            config = {
                "core": f"K{core_size}",
                "anchors": n_anchors,
                "bridge": bridge_width,
                "n_vars": r["n_vars"] if r else None,
            }

            if r is not None:
                config["ratio"] = r["ratio"]
                config["l2_full"] = r["l2_full"]
                config["l2_red"] = r["l2_red"]
                config["valve"] = r["valve"]
                config["star_verified"] = star_ratio

                match = ""
                if star_ratio is not None:
                    if abs(star_ratio - r["ratio"]) < 0.01:
                        match = "MATCH"
                    else:
                        match = f"DIFF ({star_ratio:.4f})"
                config["verify"] = match
            else:
                config["ratio"] = None
                config["verify"] = "DEGEN"

            results.append(config)

elapsed = time.time() - t_start

# Print results table
print(f"\nComputed {len(results)} configurations in {elapsed:.1f}s\n")

print(f"{'Core':<6s} {'Anch':>4s} {'Brg':>4s} {'Vars':>4s} {'Ratio':>10s} {'L2_full':>10s} {'L2_red':>10s} {'Verify':>12s}")
print(f"{'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

for c in results:
    if c["ratio"] is not None:
        verify_str = c.get("verify", "")
        print(f"{c['core']:<6s} {c['anchors']:>4d} {c['bridge']:>4d} {c['n_vars']:>4d} "
              f"{c['ratio']:>10.6f} {c['l2_full']:>10.6f} {c['l2_red']:>10.6f} {verify_str:>12s}")
    else:
        print(f"{c['core']:<6s} {c['anchors']:>4d} {c['bridge']:>4d} {'?':>4s} "
              f"{'DEGEN':>10s} {'':>10s} {'':>10s} {'':>12s}")

# Group by bridge width and show patterns
print("\n" + "=" * 90)
print("INVARIANTS GROUPED BY BRIDGE WIDTH")
print("=" * 90)

for bw in [1, 2, 3, 4, 5]:
    subset = [c for c in results if c["bridge"] == bw and c["ratio"] is not None]
    if subset:
        ratios = [c["ratio"] for c in subset]
        print(f"\n  Bridge width = {bw}:")
        print(f"    Count: {len(subset)}")
        print(f"    Range: [{min(ratios):.6f}, {max(ratios):.6f}]")
        print(f"    Mean:  {np.mean(ratios):.6f}")
        print(f"    Std:   {np.std(ratios):.6f}")

# Group by core size
print("\n" + "=" * 90)
print("INVARIANTS GROUPED BY CORE SIZE")
print("=" * 90)

for cs in [3, 4, 5, 6, 7]:
    subset = [c for c in results if c["core"] == f"K{cs}" and c["ratio"] is not None]
    if subset:
        ratios = [c["ratio"] for c in subset]
        print(f"\n  Core = K{cs}:")
        print(f"    Count: {len(subset)}")
        print(f"    Range: [{min(ratios):.6f}, {max(ratios):.6f}]")
        print(f"    Mean:  {np.mean(ratios):.6f}")
        print(f"    Std:   {np.std(ratios):.6f}")

# Find the configurations closest to 1.857 (the original K5 value)
print("\n" + "=" * 90)
print("CONFIGURATIONS CLOSEST TO 1.857 (original K5 star invariant)")
print("=" * 90)

valid = [(c, abs(c["ratio"] - 1.857)) for c in results if c["ratio"] is not None]
valid.sort(key=lambda x: x[1])
print(f"\n  {'Core':<6s} {'Anch':>4s} {'Brg':>4s} {'Ratio':>10s} {'Delta':>10s}")
for c, d in valid[:15]:
    print(f"  {c['core']:<6s} {c['anchors']:>4d} {c['bridge']:>4d} {c['ratio']:>10.6f} {d:>10.6f}")

# Find ALL unique invariant values (cluster by proximity)
print("\n" + "=" * 90)
print("DISTINCT INVARIANT VALUES (clustered within 0.001)")
print("=" * 90)

all_ratios = sorted(set(round(c["ratio"], 3) for c in results if c["ratio"] is not None))
print(f"\n  {len(all_ratios)} distinct values (rounded to 3 decimals):")
for rv in all_ratios:
    configs = [c for c in results if c["ratio"] is not None and abs(c["ratio"] - rv) < 0.001]
    labels = [f"{c['core']}/a{c['anchors']}/b{c['bridge']}" for c in configs]
    print(f"    {rv:.3f}: {', '.join(labels)}")

# Characteristic polynomial analysis for select configs
print("\n" + "=" * 90)
print("BASE IRRATIONALS (from cluster Laplacian eigenvalue pairs)")
print("=" * 90)

for cs in [3, 4, 5, 6, 7]:
    for na in [0, 2]:
        nv, edges = build_cluster_edges(cs, na)
        A = np.zeros((nv, nv))
        for i, j in edges:
            A[i][j] = 1
            A[j][i] = 1
        L = np.diag(A.sum(axis=1)) - A
        eig = np.sort(np.linalg.eigvalsh(L))

        # Find non-integer eigenvalues
        non_int = [(i, v) for i, v in enumerate(eig) if abs(v - round(v)) > 0.01]

        # For pairs that sum to an integer, find discriminant
        pairs = []
        for i in range(len(non_int)):
            for j in range(i+1, len(non_int)):
                s = non_int[i][1] + non_int[j][1]
                if abs(s - round(s)) < 0.01:
                    p = non_int[i][1] * non_int[j][1]
                    S = round(s)
                    disc = S*S - 4*p
                    pairs.append((S, p, disc))

        if pairs:
            S, p, disc = pairs[0]
            p_round = round(p)
            disc_check = S*S - 4*p_round
            if abs(p - p_round) < 0.01 and disc_check > 0:
                print(f"  K{cs}, {na} anchors ({nv} vars): t^2 - {S}t + {p_round} = 0, disc = {disc_check}, sqrt({disc_check})")
            else:
                print(f"  K{cs}, {na} anchors ({nv} vars): non-integer product {p:.4f}")
        else:
            int_eigs = [round(v) for v in eig if abs(v - round(v)) < 0.01]
            print(f"  K{cs}, {na} anchors ({nv} vars): all eigenvalues integer: {int_eigs}")
