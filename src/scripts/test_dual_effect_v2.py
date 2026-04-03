"""
Test the Dual Effect Conjecture for Simplicial Surgery — Version 2

  Conjecture: For simplicial surgery on a graph that reduces b_1,
  the graph Laplacian gap (Fiedler value of L_0) DECREASES
  but the Stokes gap (smallest nonzero eigenvalue of the Hodge 1-Laplacian L_1
  restricted to divergence-free 1-forms) INCREASES.

Version 2 improvements:
  - Focus on triangle-rich graphs where edge removal changes the simplicial complex
  - Test both single-edge surgery AND vertex-removal (valve) surgery
  - Track whether the removed edge participates in triangles
  - Separate "up" and "down" Laplacian contributions
  - Include the exact K5+2anchor valve surgery from the Fluid-Resonance project

Author: Claude Opus 4.6 (Meridian), 2026-03-12
"""

import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.set_printoptions(precision=6, suppress=True)

EPS = 1e-9


# ============================================================
# GRAPH AND SIMPLICIAL COMPLEX
# ============================================================

def adj_from_edges(edges, n_vertices):
    adj = {i: set() for i in range(n_vertices)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def find_triangles(edges, n_vertices):
    adj = adj_from_edges(edges, n_vertices)
    triangles = set()
    for u, v in edges:
        common = adj[u] & adj[v]
        for w in common:
            triangles.add(tuple(sorted([u, v, w])))
    return sorted(triangles)


def build_d0(edges, n_vertices):
    """Coboundary d_0: |E| x |V|. For edge e=(i,j), i<j: d0[e,i]=-1, d0[e,j]=+1."""
    n_e = len(edges)
    d0 = np.zeros((n_e, n_vertices))
    for idx, (i, j) in enumerate(edges):
        d0[idx, i] = -1.0
        d0[idx, j] = 1.0
    return d0


def build_d1(edges, triangles):
    """Coboundary d_1: |T| x |E|. For triangle (i,j,k), i<j<k:
       boundary = (j,k) - (i,k) + (i,j), so d1[t, e_ij]=+1, d1[t, e_ik]=-1, d1[t, e_jk]=+1."""
    edge_idx = {e: i for i, e in enumerate(edges)}
    n_e = len(edges)
    n_t = len(triangles)
    d1 = np.zeros((n_t, n_e))
    for t_i, (a, b, c) in enumerate(triangles):
        if (a, b) in edge_idx:
            d1[t_i, edge_idx[(a, b)]] = 1.0
        if (a, c) in edge_idx:
            d1[t_i, edge_idx[(a, c)]] = -1.0
        if (b, c) in edge_idx:
            d1[t_i, edge_idx[(b, c)]] = 1.0
    return d1


def is_connected(n_vertices, edges):
    if n_vertices == 0:
        return True
    adj = adj_from_edges(edges, n_vertices)
    visited = set()
    stack = [0]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        for w in adj[u]:
            if w not in visited:
                stack.append(w)
    return len(visited) == n_vertices


def compute_b1(n_vertices, edges, triangles):
    """Simplicial b_1 = |E| - rank(d0) - rank(d1)."""
    d0 = build_d0(edges, n_vertices)
    rank_d0 = np.linalg.matrix_rank(d0, tol=1e-10)
    if len(triangles) > 0:
        d1 = build_d1(edges, triangles)
        rank_d1 = np.linalg.matrix_rank(d1, tol=1e-10)
    else:
        rank_d1 = 0
    return len(edges) - rank_d0 - rank_d1


def compute_L0_gap(d0):
    """Smallest nonzero eigenvalue of L_0 = d_0^T d_0."""
    L0 = d0.T @ d0
    evals = np.linalg.eigvalsh(L0)
    nz = evals[evals > EPS]
    return float(nz[0]) if len(nz) > 0 else 0.0


def compute_stokes_gap(d0, d1, n_edges):
    """
    Stokes gap = smallest nonzero eigenvalue of L_1 restricted to ker(d_0^T).

    L_1 = d_0 d_0^T + d_1^T d_1  (both |E|x|E|)

    ker(d_0^T) = divergence-free subspace of R^|E|.
    """
    L1_down = d0 @ d0.T
    if d1.shape[0] > 0:
        L1_up = d1.T @ d1
    else:
        L1_up = np.zeros((n_edges, n_edges))
    L1 = L1_down + L1_up

    # Null space of d_0^T (|V| x |E|)
    d0T = d0.T
    U, S, Vt = np.linalg.svd(d0T, full_matrices=True)
    rank = np.sum(S > EPS)
    null_vecs = Vt[rank:].T  # |E| x dim(null)

    dim_null = null_vecs.shape[1]
    if dim_null == 0:
        return float('inf'), 0

    # Project L_1 onto divergence-free subspace
    L1_proj = null_vecs.T @ L1 @ null_vecs

    evals = np.linalg.eigvalsh(L1_proj)
    nz = evals[evals > EPS]

    if len(nz) == 0:
        return 0.0, dim_null

    return float(nz[0]), dim_null


def full_analysis(n_vertices, edges, name=""):
    """Complete spectral analysis."""
    triangles = find_triangles(edges, n_vertices)
    d0 = build_d0(edges, n_vertices)
    d1 = build_d1(edges, triangles) if triangles else np.zeros((0, len(edges)))

    # Verify chain complex
    if d1.shape[0] > 0:
        err = np.max(np.abs(d1 @ d0))
        if err > 1e-10:
            print(f"  WARNING: d1 @ d0 != 0 for {name}: max={err}, V={n_vertices}, E={len(edges)}, T={len(triangles)}")
            # Debug: find the offending triangle
            prod = d1 @ d0
            for t_i in range(len(triangles)):
                row = prod[t_i]
                if np.max(np.abs(row)) > 1e-10:
                    a, b, c = triangles[t_i]
                    print(f"    Triangle {triangles[t_i]}: edges ({a},{b})={'Y' if (a,b) in set(edges) else 'N'}, ({a},{c})={'Y' if (a,c) in set(edges) else 'N'}, ({b},{c})={'Y' if (b,c) in set(edges) else 'N'}")
                    break
            return None  # skip this graph

    b1 = compute_b1(n_vertices, edges, triangles)
    L0_gap = compute_L0_gap(d0)
    stokes_gap, dim_df = compute_stokes_gap(d0, d1, len(edges))

    return {
        'name': name,
        'nV': n_vertices,
        'nE': len(edges),
        'nT': len(triangles),
        'b1': b1,
        'L0_gap': L0_gap,
        'stokes_gap': stokes_gap,
        'dim_divfree': dim_df,
        'edges': edges,
        'triangles': triangles,
    }


# ============================================================
# GRAPH GENERATORS (triangle-rich)
# ============================================================

def complete_graph(n):
    edges = sorted([(i, j) for i in range(n) for j in range(i+1, n)])
    return n, edges, f"K{n}"


def complete_bipartite(m, n):
    """K_{m,n} bipartite. Vertices 0..m-1 and m..m+n-1."""
    edges = sorted([(i, m+j) for i in range(m) for j in range(n)])
    return m+n, edges, f"K{m},{n}"


def cycle_with_all_short_chords(n, max_skip=2):
    """Cycle C_n plus all chords of length <= max_skip."""
    edges = set()
    for i in range(n):
        for s in range(1, max_skip+1):
            j = (i + s) % n
            edges.add(tuple(sorted((i, j))))
    return n, sorted(edges), f"C{n}+skip{max_skip}"


def k5_plus_2anchor():
    """The Fluid-Resonance K5+2anchor graph."""
    n = 7
    edges = sorted(set(
        [(i, j) for i in range(5) for j in range(i+1, 5)]
        + [(0, 5), (3, 5), (5, 6), (2, 6), (4, 6)]
    ))
    return n, edges, "K5+2anc"


def k5_plus_2anchor_full_system():
    """
    The full 2-cluster system from Phase A: hub (K5, vertices 0-4) + spoke (K5, vertices 5-9)
    + connector vertex 10 + anchors 11-15, with 3-clause bridge.
    Based on the Phase A data: 16 vertices, 40 edges, 19 triangles.

    Let me reconstruct the approximate structure:
    Hub K5: {0,1,2,3,4}
    Spoke K5: {5,6,7,8,9}
    Connector: 10
    Anchors: {11,12,13,14,15}
    Bridge edges: connects hub to spoke via connector and anchors.

    Actually, let me build a simpler version that matches the spectral data:
    Two K5 clusters connected by a bridge.
    """
    n = 12
    edges = []
    # Cluster A: K5 on {0,1,2,3,4}
    for i in range(5):
        for j in range(i+1, 5):
            edges.append((i, j))
    # Cluster B: K5 on {5,6,7,8,9}
    for i in range(5, 10):
        for j in range(i+1, 10):
            edges.append((i, j))
    # Bridge: 3 paths through anchors {10, 11}
    edges.extend([(0, 10), (10, 5), (1, 10), (10, 6)])
    edges.extend([(3, 11), (11, 8), (4, 11), (11, 9)])
    edges.append((10, 11))
    edges = sorted(set(edges))
    return n, edges, "2xK5+bridge"


def octahedron():
    """Octahedron: K_{2,2,2} = 6 vertices, 12 edges, 8 triangles."""
    n = 6
    edges = sorted([(i, j) for i in range(n) for j in range(i+1, n) if not (i % 2 == 0 and j == i+1)])
    # Actually octahedron is all edges except 3 opposite pairs
    edges = sorted(set([(i, j) for i in range(6) for j in range(i+1, 6)]) - {(0, 5), (1, 4), (2, 3)})
    return 6, edges, "Octahedron"


def icosahedron():
    """Icosahedron: 12 vertices, 30 edges, 20 triangles."""
    # Standard icosahedron adjacency
    adj = {
        0: [1, 2, 3, 4, 5],
        1: [0, 2, 5, 6, 7],
        2: [0, 1, 3, 7, 8],
        3: [0, 2, 4, 8, 9],
        4: [0, 3, 5, 9, 10],
        5: [0, 1, 4, 10, 6],
        6: [1, 5, 7, 10, 11],
        7: [1, 2, 6, 8, 11],
        8: [2, 3, 7, 9, 11],
        9: [3, 4, 8, 10, 11],
        10: [4, 5, 6, 9, 11],
        11: [6, 7, 8, 9, 10],
    }
    edges = sorted(set(tuple(sorted((u, v))) for u in adj for v in adj[u]))
    return 12, edges, "Icosahedron"


def dense_random(n, p, seed):
    """Dense random graph."""
    rng = np.random.RandomState(seed)
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if rng.random() < p]
    # Ensure connected
    adj = adj_from_edges(edges, n)
    visited = set()
    stack = [0]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        for w in adj[u]:
            if w not in visited:
                stack.append(w)
    for v in range(n):
        if v not in visited:
            edges.append(tuple(sorted((0, v))))
            visited.add(v)
    return n, sorted(set(edges)), f"DR({n},{p:.1f},s{seed})"


def kn_minus_edges(n, remove_count, seed):
    """Start from K_n, remove a few edges (stays triangle-rich)."""
    rng = np.random.RandomState(seed)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    to_remove = set()
    attempts = 0
    while len(to_remove) < remove_count and attempts < 1000:
        idx = rng.randint(len(all_edges))
        candidate = [e for i, e in enumerate(all_edges) if i not in to_remove]
        if not candidate:
            break
        e = candidate[rng.randint(len(candidate))]
        new_edges = [x for x in all_edges if x != e and x not in to_remove]
        if is_connected(n, new_edges):
            to_remove.add(all_edges.index(e))
        attempts += 1
    edges = [e for i, e in enumerate(all_edges) if i not in to_remove]
    return n, sorted(edges), f"K{n}-{len(to_remove)}e(s{seed})"


def wheel(n):
    """Wheel: hub vertex 0 + n-cycle on {1..n}."""
    edges = []
    for i in range(1, n+1):
        edges.append((0, i))
        edges.append(tuple(sorted((i, (i % n) + 1))))
    return n+1, sorted(set(edges)), f"W{n}"


def prism(n):
    """Prism: two n-cycles + rungs. Add diagonal chords for triangles."""
    edges = []
    for i in range(n):
        edges.append(tuple(sorted((i, (i+1) % n))))
        edges.append(tuple(sorted((n+i, n+(i+1) % n))))
        edges.append((i, n+i))
        # Add one diagonal per face to make triangles
        edges.append(tuple(sorted((i, n+(i+1) % n))))
    return 2*n, sorted(set(edges)), f"TPrism{n}"


def petersen():
    outer = list(range(5))
    inner = list(range(5, 10))
    edges = []
    for i in range(5):
        edges.append(tuple(sorted((outer[i], outer[(i+1) % 5]))))
        edges.append(tuple(sorted((inner[i], inner[(i+2) % 5]))))
        edges.append(tuple(sorted((outer[i], inner[i]))))
    return 10, sorted(set(edges)), "Petersen"


# ============================================================
# SURGERY TYPES
# ============================================================

def single_edge_surgeries(n_vertices, edges, name, results):
    """Try removing each edge. Only keep surgeries that reduce simplicial b_1."""
    pre = full_analysis(n_vertices, edges, name)
    if pre is None or pre['b1'] <= 0:
        return 0

    count = 0
    for removed in edges:
        new_edges = [e for e in edges if e != removed]
        if not is_connected(n_vertices, new_edges):
            continue

        new_tri = find_triangles(new_edges, n_vertices)
        new_b1 = compute_b1(n_vertices, new_edges, new_tri)
        if new_b1 >= pre['b1']:
            continue  # b_1 didn't decrease

        delta_b1 = pre['b1'] - new_b1

        post = full_analysis(n_vertices, new_edges, f"{name}\\{removed}")
        if post is None:
            continue

        if pre['stokes_gap'] <= EPS or pre['stokes_gap'] == float('inf'):
            continue
        if post['stokes_gap'] <= EPS or post['stokes_gap'] == float('inf'):
            continue

        # Count how many triangles the removed edge was in
        tri_loss = pre['nT'] - post['nT']

        record_result(pre, post, removed, delta_b1, tri_loss, 'edge', results)
        count += 1

    return count


def vertex_removal_surgeries(n_vertices, edges, name, results):
    """Try removing each vertex (and all its incident edges). Only keep if b_1 decreases."""
    pre = full_analysis(n_vertices, edges, name)
    if pre is None or pre['b1'] <= 0:
        return 0

    count = 0
    for v_remove in range(n_vertices):
        # New vertex set: 0..n-1 minus v_remove, relabel
        vertex_map = {}
        new_idx = 0
        for v in range(n_vertices):
            if v != v_remove:
                vertex_map[v] = new_idx
                new_idx += 1
        new_n = n_vertices - 1
        new_edges = []
        for (u, w) in edges:
            if u != v_remove and w != v_remove:
                new_edges.append(tuple(sorted((vertex_map[u], vertex_map[w]))))
        new_edges = sorted(set(new_edges))

        if not is_connected(new_n, new_edges):
            continue

        new_tri = find_triangles(new_edges, new_n)
        new_b1 = compute_b1(new_n, new_edges, new_tri)
        if new_b1 >= pre['b1']:
            continue

        delta_b1 = pre['b1'] - new_b1
        post = full_analysis(new_n, new_edges, f"{name}\\v{v_remove}")
        if post is None:
            continue

        if pre['stokes_gap'] <= EPS or pre['stokes_gap'] == float('inf'):
            continue
        if post['stokes_gap'] <= EPS or post['stokes_gap'] == float('inf'):
            continue

        tri_loss = pre['nT'] - post['nT']
        record_result(pre, post, f"v{v_remove}", delta_b1, tri_loss, 'vertex', results)
        count += 1

    return count


def record_result(pre, post, removed, delta_b1, tri_loss, surgery_type, results):
    L0_dir = 'DOWN' if post['L0_gap'] < pre['L0_gap'] - EPS else ('UP' if post['L0_gap'] > pre['L0_gap'] + EPS else 'SAME')
    stk_dir = 'UP' if post['stokes_gap'] > pre['stokes_gap'] + EPS else ('DOWN' if post['stokes_gap'] < pre['stokes_gap'] - EPS else 'SAME')
    dual = (L0_dir == 'DOWN' and stk_dir == 'UP')

    results.append({
        'graph': pre['name'],
        'surgery_type': surgery_type,
        'removed': str(removed),
        'delta_b1': delta_b1,
        'b1_pre': pre['b1'],
        'b1_post': post['b1'],
        'L0_pre': pre['L0_gap'],
        'L0_post': post['L0_gap'],
        'L0_dir': L0_dir,
        'stk_pre': pre['stokes_gap'],
        'stk_post': post['stokes_gap'],
        'stk_dir': stk_dir,
        'dual': dual,
        'tri_pre': pre['nT'],
        'tri_post': post['nT'],
        'tri_loss': tri_loss,
        'nV_pre': pre['nV'],
        'nE_pre': pre['nE'],
    })


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 110)
    print("DUAL EFFECT CONJECTURE TEST — v2 (triangle-aware, edge + vertex surgery)")
    print("Conjecture: Surgery reducing b_1 => L0 gap DECREASES, Stokes gap INCREASES")
    print("=" * 110)

    results = []
    graphs = []

    # --- Triangle-rich graphs ---

    # Complete graphs
    for n in [4, 5, 6, 7, 8]:
        graphs.append(complete_graph(n))

    # K5+2anchor (the project graph)
    graphs.append(k5_plus_2anchor())

    # 2-cluster bridge system
    graphs.append(k5_plus_2anchor_full_system())

    # Octahedron and icosahedron
    graphs.append(octahedron())
    graphs.append(icosahedron())

    # Cycles with dense chords (creates many triangles)
    for n in [5, 6, 7, 8, 9, 10]:
        graphs.append(cycle_with_all_short_chords(n, 2))
    for n in [6, 7, 8, 9]:
        graphs.append(cycle_with_all_short_chords(n, 3))

    # Wheels (every spoke creates a triangle with the rim)
    for n in [4, 5, 6, 7, 8]:
        graphs.append(wheel(n))

    # Triangulated prisms
    for n in [3, 4, 5]:
        graphs.append(prism(n))

    # K_n minus a few edges
    for n in [6, 7, 8]:
        for s in range(3):
            graphs.append(kn_minus_edges(n, 2, seed=s*100+n))
            graphs.append(kn_minus_edges(n, 3, seed=s*200+n))

    # Dense random graphs
    for s in range(8):
        graphs.append(dense_random(7, 0.6, seed=s))
        graphs.append(dense_random(8, 0.55, seed=s+50))
        graphs.append(dense_random(9, 0.5, seed=s+100))

    # Petersen
    graphs.append(petersen())

    print(f"Built {len(graphs)} test graphs.\n")

    total_surgeries = 0
    for nv, edges, name in graphs:
        if not is_connected(nv, edges):
            continue

        # Pre-check: need b_1 > 0 and triangles
        tri = find_triangles(edges, nv)
        b1 = compute_b1(nv, edges, tri)
        if b1 <= 0:
            continue

        # Single-edge surgeries
        c1 = single_edge_surgeries(nv, edges, name, results)

        # Vertex-removal surgeries
        c2 = vertex_removal_surgeries(nv, edges, name, results)

        total_surgeries += c1 + c2
        if c1 + c2 > 0:
            print(f"  {name}: V={nv}, E={len(edges)}, T={len(tri)}, b1={b1} => {c1} edge + {c2} vertex surgeries")

    print(f"\nTotal valid surgeries: {total_surgeries}\n")

    if not results:
        print("No valid surgeries found!")
        return

    # Separate results by whether triangles changed
    tri_change = [r for r in results if r['tri_loss'] > 0]
    tri_same = [r for r in results if r['tri_loss'] == 0]

    # ============================================================
    # FULL TABLE
    # ============================================================

    print("=" * 150)
    print(f"{'Graph':<20} {'Type':<6} {'Cut':<10} {'db1':>4} "
          f"{'b1':>3}>{'>b1*':>3} {'dT':>4} "
          f"{'L0':>9} > {'L0*':>9} {'L0':>5} "
          f"{'Stk':>9} > {'Stk*':>9} {'Stk':>5} {'Dual':>5}")
    print("-" * 150)

    for r in results:
        d_str = "YES" if r['dual'] else "no"
        print(f"{r['graph']:<20} {r['surgery_type']:<6} {r['removed']:<10} {r['delta_b1']:>4} "
              f"{r['b1_pre']:>3}>{r['b1_post']:>3} {r['tri_loss']:>4} "
              f"{r['L0_pre']:>9.5f} > {r['L0_post']:>9.5f} {r['L0_dir']:>5} "
              f"{r['stk_pre']:>9.5f} > {r['stk_post']:>9.5f} {r['stk_dir']:>5} {d_str:>5}")

    print("=" * 150)

    # ============================================================
    # STATISTICS
    # ============================================================

    total = len(results)
    dual_yes = sum(1 for r in results if r['dual'])
    dual_no = total - dual_yes

    # Direction counts
    cats = {}
    for r in results:
        key = (r['L0_dir'], r['stk_dir'])
        cats[key] = cats.get(key, 0) + 1

    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS ({total} surgeries)")
    print(f"{'='*70}")
    print(f"  Dual effect (L0 DOWN, Stokes UP):   {dual_yes:>4} ({100*dual_yes/total:.1f}%)")
    print(f"  All other outcomes:                   {dual_no:>4} ({100*dual_no/total:.1f}%)")
    print()
    print(f"  Direction breakdown:")
    for (l0d, sd), cnt in sorted(cats.items()):
        pct = 100*cnt/total
        marker = " <-- conjecture" if l0d == 'DOWN' and sd == 'UP' else ""
        print(f"    L0 {l0d:>4}, Stokes {sd:>4}: {cnt:>4} ({pct:>5.1f}%){marker}")

    # ============================================================
    # SPLIT BY TRIANGLE CHANGE
    # ============================================================

    print(f"\n{'='*70}")
    print("SPLIT BY WHETHER TRIANGLES CHANGED")
    print(f"{'='*70}")

    for label, subset in [("Triangles CHANGED (tri_loss > 0)", tri_change),
                           ("Triangles UNCHANGED (tri_loss = 0)", tri_same)]:
        if not subset:
            print(f"\n  {label}: (no cases)")
            continue
        n = len(subset)
        dy = sum(1 for r in subset if r['dual'])
        print(f"\n  {label}: {n} surgeries")
        print(f"    Dual holds: {dy}/{n} ({100*dy/n:.1f}%)")
        sub_cats = {}
        for r in subset:
            key = (r['L0_dir'], r['stk_dir'])
            sub_cats[key] = sub_cats.get(key, 0) + 1
        for (l0d, sd), cnt in sorted(sub_cats.items()):
            print(f"      L0 {l0d:>4}, Stokes {sd:>4}: {cnt:>4} ({100*cnt/n:>5.1f}%)")

    # ============================================================
    # SPLIT BY SURGERY TYPE
    # ============================================================

    print(f"\n{'='*70}")
    print("SPLIT BY SURGERY TYPE")
    print(f"{'='*70}")

    for stype in ['edge', 'vertex']:
        subset = [r for r in results if r['surgery_type'] == stype]
        if not subset:
            continue
        n = len(subset)
        dy = sum(1 for r in subset if r['dual'])
        print(f"\n  {stype.upper()} surgery: {n} surgeries")
        print(f"    Dual holds: {dy}/{n} ({100*dy/n:.1f}%)")
        sub_cats = {}
        for r in subset:
            key = (r['L0_dir'], r['stk_dir'])
            sub_cats[key] = sub_cats.get(key, 0) + 1
        for (l0d, sd), cnt in sorted(sub_cats.items()):
            print(f"      L0 {l0d:>4}, Stokes {sd:>4}: {cnt:>4} ({100*cnt/n:>5.1f}%)")

    # ============================================================
    # K5+2ANCHOR DETAILED
    # ============================================================

    k5_res = [r for r in results if 'K5+2anc' in r['graph'] and '2xK5' not in r['graph']]
    if k5_res:
        print(f"\n{'='*70}")
        print("K5+2ANCHOR DETAILED RESULTS")
        print(f"{'='*70}")
        for r in k5_res:
            print(f"  {r['surgery_type']:<6} {r['removed']:<10} b1:{r['b1_pre']}->{r['b1_post']} "
                  f"T:{r['tri_pre']}->{r['tri_post']} "
                  f"L0:{r['L0_pre']:.5f}->{r['L0_post']:.5f}({r['L0_dir']}) "
                  f"Stk:{r['stk_pre']:.5f}->{r['stk_post']:.5f}({r['stk_dir']}) "
                  f"Dual:{'YES' if r['dual'] else 'NO'}")

    # ============================================================
    # 2xK5+bridge DETAILED
    # ============================================================

    bridge_res = [r for r in results if '2xK5' in r['graph']]
    if bridge_res:
        print(f"\n{'='*70}")
        print("2xK5+BRIDGE DETAILED RESULTS")
        print(f"{'='*70}")
        for r in bridge_res:
            print(f"  {r['surgery_type']:<6} {r['removed']:<10} b1:{r['b1_pre']}->{r['b1_post']} "
                  f"T:{r['tri_pre']}->{r['tri_post']} "
                  f"L0:{r['L0_pre']:.5f}->{r['L0_post']:.5f}({r['L0_dir']}) "
                  f"Stk:{r['stk_pre']:.5f}->{r['stk_post']:.5f}({r['stk_dir']}) "
                  f"Dual:{'YES' if r['dual'] else 'NO'}")

    # ============================================================
    # COUNTEREXAMPLES
    # ============================================================

    # Show cases where dual DOES hold
    dual_examples = [r for r in results if r['dual']]
    if dual_examples:
        print(f"\n{'='*70}")
        print(f"EXAMPLES WHERE DUAL EFFECT HOLDS ({len(dual_examples)} cases)")
        print(f"{'='*70}")
        for r in dual_examples[:30]:
            print(f"  {r['graph']:<20} {r['surgery_type']:<6} {r['removed']:<10} "
                  f"b1:{r['b1_pre']}->{r['b1_post']} T:{r['tri_pre']}->{r['tri_post']} "
                  f"L0:{r['L0_pre']:.5f}->{r['L0_post']:.5f} "
                  f"Stk:{r['stk_pre']:.5f}->{r['stk_post']:.5f}")

    # Show cases where Stokes gap actually changes (whether up or down)
    stokes_moved = [r for r in results if r['stk_dir'] != 'SAME']
    if stokes_moved:
        print(f"\n{'='*70}")
        print(f"CASES WHERE STOKES GAP ACTUALLY CHANGED ({len(stokes_moved)} cases)")
        print(f"{'='*70}")
        for r in stokes_moved[:40]:
            print(f"  {r['graph']:<20} {r['surgery_type']:<6} {r['removed']:<10} "
                  f"b1:{r['b1_pre']}->{r['b1_post']} dT={r['tri_loss']} "
                  f"L0:{r['L0_pre']:.5f}->{r['L0_post']:.5f}({r['L0_dir']}) "
                  f"Stk:{r['stk_pre']:.5f}->{r['stk_post']:.5f}({r['stk_dir']})")
    else:
        print(f"\n  NOTE: Stokes gap NEVER changed in any surgery!")
        print(f"  This suggests the conjecture applies to a different regime.")

    # ============================================================
    # VERDICT
    # ============================================================

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    pct = 100*dual_yes/total if total > 0 else 0
    n_stk_changed = len(stokes_moved)

    if dual_yes == total:
        print(f"  UNIVERSAL: Dual effect holds in ALL {total} tested surgeries.")
    elif n_stk_changed == 0:
        print(f"  STOKES GAP INVARIANT: In all {total} surgeries, the Stokes gap did not change.")
        print(f"  L0 gap decreased in {sum(1 for r in results if r['L0_dir']=='DOWN')}/{total} cases (first half of conjecture).")
        print(f"  The conjecture's 'Stokes increases' claim cannot be confirmed by single-edge surgery")
        print(f"  on graphs where the removed edge doesn't participate in enough triangles.")
    elif dual_yes == 0:
        print(f"  FALSE: Dual effect holds in 0/{total} cases.")
        if n_stk_changed > 0:
            stk_up = sum(1 for r in stokes_moved if r['stk_dir'] == 'UP')
            stk_down = sum(1 for r in stokes_moved if r['stk_dir'] == 'DOWN')
            print(f"  When Stokes changed: UP {stk_up}, DOWN {stk_down}")
    else:
        print(f"  PARTIAL: Dual effect holds in {dual_yes}/{total} ({pct:.1f}%) surgeries.")
        if tri_change:
            dy_tc = sum(1 for r in tri_change if r['dual'])
            print(f"  Among surgeries that changed triangles: {dy_tc}/{len(tri_change)} ({100*dy_tc/len(tri_change):.1f}%)")


if __name__ == '__main__':
    main()
