"""
Test Dual Effect Conjecture — Version 3 (Focused)

Key insight from v2: Single-edge removal NEVER changes the Stokes gap because
removing one edge from a graph cannot change the "up" Laplacian (d_1^T d_1)
unless that edge was part of a triangle — and even then, removing one edge
can destroy triangles but the resulting d_1^T d_1 often preserves the same
nonzero eigenvalue spectrum on the divergence-free subspace.

The conjecture is really about VERTEX REMOVAL (valve surgery) where we remove
a vertex and all its incident edges, which simultaneously:
  - Destroys multiple triangles (changes the "up" Laplacian)
  - Removes edges (changes the "down" Laplacian)
  - Changes the divergence-free subspace itself

This version focuses exclusively on vertex-removal surgeries on triangle-rich
graphs, with much larger sample size (target: 100+ valid surgeries).

Author: Claude Opus 4.6 (Meridian), 2026-03-12
"""

import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

EPS = 1e-9


# ============================================================
# CORE LINEAR ALGEBRA
# ============================================================

def adj_from_edges(edges, n):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def find_triangles(edges, n):
    adj = adj_from_edges(edges, n)
    tri = set()
    for u, v in edges:
        for w in adj[u] & adj[v]:
            tri.add(tuple(sorted([u, v, w])))
    return sorted(tri)


def build_d0(edges, nV):
    d0 = np.zeros((len(edges), nV))
    for i, (u, v) in enumerate(edges):
        d0[i, u] = -1.0
        d0[i, v] = 1.0
    return d0


def build_d1(edges, triangles):
    edge_idx = {e: i for i, e in enumerate(edges)}
    d1 = np.zeros((len(triangles), len(edges)))
    for t_i, (a, b, c) in enumerate(triangles):
        if (a, b) in edge_idx:
            d1[t_i, edge_idx[(a, b)]] = 1.0
        if (a, c) in edge_idx:
            d1[t_i, edge_idx[(a, c)]] = -1.0
        if (b, c) in edge_idx:
            d1[t_i, edge_idx[(b, c)]] = 1.0
    return d1


def is_connected(n, edges):
    if n <= 1:
        return True
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
    return len(visited) == n


def compute_b1(nV, edges, triangles):
    d0 = build_d0(edges, nV)
    rank_d0 = np.linalg.matrix_rank(d0, tol=1e-10)
    if len(triangles) > 0:
        d1 = build_d1(edges, triangles)
        rank_d1 = np.linalg.matrix_rank(d1, tol=1e-10)
    else:
        rank_d1 = 0
    return len(edges) - rank_d0 - rank_d1


def spectral_analysis(nV, edges):
    """Return (L0_gap, stokes_gap, b1, nT) or None if invalid."""
    triangles = find_triangles(edges, nV)
    d0 = build_d0(edges, nV)
    d1 = build_d1(edges, triangles) if triangles else np.zeros((0, len(edges)))

    # Verify chain complex
    if d1.shape[0] > 0:
        if np.max(np.abs(d1 @ d0)) > 1e-10:
            return None

    b1 = compute_b1(nV, edges, triangles)

    # L0 gap
    L0 = d0.T @ d0
    evals_L0 = np.linalg.eigvalsh(L0)
    nz_L0 = evals_L0[evals_L0 > EPS]
    L0_gap = float(nz_L0[0]) if len(nz_L0) > 0 else 0.0

    # Stokes gap
    nE = len(edges)
    L1_down = d0 @ d0.T
    L1_up = d1.T @ d1 if d1.shape[0] > 0 else np.zeros((nE, nE))
    L1 = L1_down + L1_up

    d0T = d0.T
    U, S, Vt = np.linalg.svd(d0T, full_matrices=True)
    rank = np.sum(S > EPS)
    null_vecs = Vt[rank:].T

    dim_df = null_vecs.shape[1]
    if dim_df == 0:
        stokes_gap = float('inf')
    else:
        L1_proj = null_vecs.T @ L1 @ null_vecs
        evals_stk = np.linalg.eigvalsh(L1_proj)
        nz_stk = evals_stk[evals_stk > EPS]
        stokes_gap = float(nz_stk[0]) if len(nz_stk) > 0 else 0.0

    return L0_gap, stokes_gap, b1, len(triangles), dim_df


def remove_vertex(nV, edges, v_remove):
    """Remove vertex v_remove and relabel remaining vertices."""
    vmap = {}
    idx = 0
    for v in range(nV):
        if v != v_remove:
            vmap[v] = idx
            idx += 1
    new_edges = sorted(set(
        tuple(sorted((vmap[u], vmap[w])))
        for u, w in edges if u != v_remove and w != v_remove
    ))
    return nV - 1, new_edges


# ============================================================
# GRAPH GENERATORS
# ============================================================

def complete_graph(n):
    return n, sorted([(i, j) for i in range(n) for j in range(i+1, n)]), f"K{n}"


def cycle_chorded(n, skip):
    edges = set()
    for i in range(n):
        for s in range(1, skip+1):
            j = (i + s) % n
            edges.add(tuple(sorted((i, j))))
    return n, sorted(edges), f"C{n}s{skip}"


def wheel(n):
    edges = set()
    for i in range(1, n+1):
        edges.add((0, i))
        edges.add(tuple(sorted((i, (i % n) + 1))))
    return n+1, sorted(edges), f"W{n}"


def dense_random(n, p, seed):
    rng = np.random.RandomState(seed)
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if rng.random() < p]
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
    return n, sorted(set(edges)), f"DR{n}_{p:.0%}_s{seed}"


def kn_minus(n, rm, seed):
    rng = np.random.RandomState(seed)
    all_e = [(i, j) for i in range(n) for j in range(i+1, n)]
    removed = set()
    for _ in range(rm * 10):
        if len(removed) >= rm:
            break
        idx = rng.randint(len(all_e))
        if idx in removed:
            continue
        trial = [e for i, e in enumerate(all_e) if i not in removed and i != idx]
        if is_connected(n, trial):
            removed.add(idx)
    edges = [e for i, e in enumerate(all_e) if i not in removed]
    return n, sorted(edges), f"K{n}-{len(removed)}"


def k5_2anchor():
    n = 7
    edges = sorted(set(
        [(i, j) for i in range(5) for j in range(i+1, 5)]
        + [(0, 5), (3, 5), (5, 6), (2, 6), (4, 6)]
    ))
    return n, edges, "K5+2anc"


def octahedron():
    edges = sorted(set([(i, j) for i in range(6) for j in range(i+1, 6)]) - {(0, 5), (1, 4), (2, 3)})
    return 6, edges, "Octahedron"


def icosahedron():
    adj = {
        0: [1,2,3,4,5], 1: [0,2,5,6,7], 2: [0,1,3,7,8],
        3: [0,2,4,8,9], 4: [0,3,5,9,10], 5: [0,1,4,10,6],
        6: [1,5,7,10,11], 7: [1,2,6,8,11], 8: [2,3,7,9,11],
        9: [3,4,8,10,11], 10: [4,5,6,9,11], 11: [6,7,8,9,10],
    }
    edges = sorted(set(tuple(sorted((u, v))) for u in adj for v in adj[u]))
    return 12, edges, "Icosahedron"


def two_k5_shared_edge():
    """Two K5s sharing one edge. 8 vertices."""
    e1 = [(i, j) for i in range(5) for j in range(i+1, 5)]
    # Second K5: shares edge (3,4), new vertices 5,6,7
    e2 = []
    v2 = [3, 4, 5, 6, 7]
    for i in range(5):
        for j in range(i+1, 5):
            e2.append(tuple(sorted((v2[i], v2[j]))))
    edges = sorted(set(e1 + e2))
    return 8, edges, "2xK5_shared"


def two_k4_bridge():
    """Two K4s connected by a single bridge edge."""
    e1 = [(i, j) for i in range(4) for j in range(i+1, 4)]
    e2 = [(i+4, j+4) for i in range(4) for j in range(i+1, 4)]
    edges = sorted(set(e1 + e2 + [(3, 4)]))
    return 8, edges, "2xK4_bridge"


def triangulated_torus(m, n):
    """m x n torus with triangulation."""
    def idx(i, j):
        return (i % m) * n + (j % n)
    edges = set()
    for i in range(m):
        for j in range(n):
            edges.add(tuple(sorted((idx(i,j), idx(i,j+1)))))
            edges.add(tuple(sorted((idx(i,j), idx(i+1,j)))))
            edges.add(tuple(sorted((idx(i,j), idx(i+1,j+1)))))
    return m*n, sorted(edges), f"TriTorus{m}x{n}"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 110)
    print("DUAL EFFECT CONJECTURE — VERTEX SURGERY FOCUS (v3)")
    print("For vertex removal that reduces b_1 and changes triangles:")
    print("  Does L0 gap DECREASE and Stokes gap INCREASE?")
    print("=" * 110)
    print()

    graphs = []

    # Complete graphs (most triangle-rich)
    for n in range(4, 10):
        graphs.append(complete_graph(n))

    # K5+2anchor
    graphs.append(k5_2anchor())

    # Special graphs
    graphs.append(octahedron())
    graphs.append(icosahedron())
    graphs.append(two_k5_shared_edge())
    graphs.append(two_k4_bridge())

    # Chorded cycles
    for n in range(5, 12):
        for skip in [2, 3]:
            if skip < n // 2:
                graphs.append(cycle_chorded(n, skip))

    # Wheels
    for n in range(4, 10):
        graphs.append(wheel(n))

    # Triangulated tori
    for m in [3, 4, 5]:
        for n in [3, 4, 5]:
            graphs.append(triangulated_torus(m, n))

    # K_n minus edges
    for n in [6, 7, 8, 9]:
        for rm in [1, 2, 3]:
            for s in range(3):
                graphs.append(kn_minus(n, rm, seed=n*100+rm*10+s))

    # Dense randoms
    for s in range(20):
        graphs.append(dense_random(7, 0.6, seed=s))
        graphs.append(dense_random(8, 0.55, seed=s+100))
        graphs.append(dense_random(9, 0.5, seed=s+200))
        graphs.append(dense_random(10, 0.45, seed=s+300))
        graphs.append(dense_random(6, 0.7, seed=s+400))

    print(f"Generated {len(graphs)} graphs.\n")

    # Results
    results = []
    skipped_chain = 0
    skipped_no_b1 = 0
    skipped_no_valid = 0

    for nV, edges, name in graphs:
        if not is_connected(nV, edges):
            continue

        pre = spectral_analysis(nV, edges)
        if pre is None:
            skipped_chain += 1
            continue
        L0_pre, stk_pre, b1_pre, nT_pre, df_pre = pre

        if b1_pre <= 0:
            skipped_no_b1 += 1
            continue
        if stk_pre <= EPS or stk_pre == float('inf'):
            skipped_no_valid += 1
            continue

        found_any = False

        for v in range(nV):
            new_nV, new_edges = remove_vertex(nV, edges, v)

            if not is_connected(new_nV, new_edges):
                continue

            post = spectral_analysis(new_nV, new_edges)
            if post is None:
                continue
            L0_post, stk_post, b1_post, nT_post, df_post = post

            if b1_post >= b1_pre:
                continue  # b_1 didn't decrease

            if stk_post <= EPS or stk_post == float('inf'):
                continue

            delta_b1 = b1_pre - b1_post
            tri_loss = nT_pre - nT_post

            L0_dir = 'DOWN' if L0_post < L0_pre - EPS else ('UP' if L0_post > L0_pre + EPS else 'SAME')
            stk_dir = 'UP' if stk_post > stk_pre + EPS else ('DOWN' if stk_post < stk_pre - EPS else 'SAME')
            dual = (L0_dir == 'DOWN' and stk_dir == 'UP')

            results.append({
                'graph': name, 'v': v, 'db1': delta_b1,
                'b1_pre': b1_pre, 'b1_post': b1_post,
                'L0_pre': L0_pre, 'L0_post': L0_post, 'L0_dir': L0_dir,
                'stk_pre': stk_pre, 'stk_post': stk_post, 'stk_dir': stk_dir,
                'dual': dual,
                'nT_pre': nT_pre, 'nT_post': nT_post, 'dT': tri_loss,
                'nV': nV, 'nE': len(edges),
            })
            found_any = True

    total = len(results)
    print(f"Skipped: {skipped_chain} chain failures, {skipped_no_b1} no b1, {skipped_no_valid} invalid gaps")
    print(f"Total valid vertex-removal surgeries: {total}\n")

    if total == 0:
        print("No valid surgeries!")
        return

    # ============================================================
    # TABLE
    # ============================================================

    print(f"{'Graph':<18} {'v':>3} {'db1':>3} {'b1':>3}>{'>':>1}{'b1*':>3} {'dT':>4} "
          f"{'L0':>9}>{'>':>1}{'L0*':>9} {'':>5} "
          f"{'Stk':>9}>{'>':>1}{'Stk*':>9} {'':>5} {'D':>3}")
    print("-" * 120)

    for r in results:
        d = "YES" if r['dual'] else "no"
        print(f"{r['graph']:<18} {r['v']:>3} {r['db1']:>3} {r['b1_pre']:>3}>{r['b1_post']:>3} {r['dT']:>4} "
              f"{r['L0_pre']:>9.5f}>{r['L0_post']:>9.5f} {r['L0_dir']:>5} "
              f"{r['stk_pre']:>9.5f}>{r['stk_post']:>9.5f} {r['stk_dir']:>5} {d:>3}")

    # ============================================================
    # STATISTICS
    # ============================================================

    dual_yes = sum(1 for r in results if r['dual'])
    dual_no = total - dual_yes

    cats = {}
    for r in results:
        key = (r['L0_dir'], r['stk_dir'])
        cats[key] = cats.get(key, 0) + 1

    print(f"\n{'='*70}")
    print(f"OVERALL ({total} surgeries)")
    print(f"{'='*70}")
    for (l, s), c in sorted(cats.items()):
        tag = " *** CONJECTURE ***" if l == 'DOWN' and s == 'UP' else ""
        print(f"  L0 {l:>4}, Stokes {s:>4}: {c:>4} ({100*c/total:>5.1f}%){tag}")

    # Split by triangle change
    tc = [r for r in results if r['dT'] > 0]
    ts = [r for r in results if r['dT'] == 0]

    for label, subset in [("TRIANGLES CHANGED", tc), ("TRIANGLES UNCHANGED", ts)]:
        if not subset:
            continue
        n = len(subset)
        dy = sum(1 for r in subset if r['dual'])
        print(f"\n  {label} ({n} surgeries):")
        print(f"    Dual holds: {dy}/{n} ({100*dy/n:.1f}%)")
        sub_cats = {}
        for r in subset:
            key = (r['L0_dir'], r['stk_dir'])
            sub_cats[key] = sub_cats.get(key, 0) + 1
        for (l, s), c in sorted(sub_cats.items()):
            print(f"      L0 {l:>4}, Stokes {s:>4}: {c:>4} ({100*c/n:>5.1f}%)")

    # Counterexamples: Stokes DOWN when triangles changed and b1 decreased
    stk_down_tc = [r for r in tc if r['stk_dir'] == 'DOWN']
    if stk_down_tc:
        print(f"\n{'='*70}")
        print(f"COUNTEREXAMPLES: Stokes DECREASED despite b1 decrease + triangle loss ({len(stk_down_tc)})")
        print(f"{'='*70}")
        for r in stk_down_tc:
            print(f"  {r['graph']} v{r['v']}: b1 {r['b1_pre']}->{r['b1_post']}, "
                  f"dT={r['dT']}, L0 {r['L0_pre']:.5f}->{r['L0_post']:.5f}({r['L0_dir']}), "
                  f"Stk {r['stk_pre']:.5f}->{r['stk_post']:.5f}({r['stk_dir']})")

    # L0 UP cases
    l0_up_tc = [r for r in tc if r['L0_dir'] == 'UP']
    if l0_up_tc:
        print(f"\n{'='*70}")
        print(f"ANOMALIES: L0 INCREASED despite b1 decrease + triangle loss ({len(l0_up_tc)})")
        print(f"{'='*70}")
        for r in l0_up_tc[:15]:
            print(f"  {r['graph']} v{r['v']}: b1 {r['b1_pre']}->{r['b1_post']}, "
                  f"dT={r['dT']}, V={r['nV']}->V-1, "
                  f"L0 {r['L0_pre']:.5f}->{r['L0_post']:.5f}({r['L0_dir']}), "
                  f"Stk {r['stk_pre']:.5f}->{r['stk_post']:.5f}({r['stk_dir']})")

    # ============================================================
    # VERDICT
    # ============================================================

    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")

    if total == 0:
        print("  No valid surgeries to test.")
        return

    pct_all = 100 * dual_yes / total
    print(f"  Overall dual effect rate: {dual_yes}/{total} ({pct_all:.1f}%)")

    if tc:
        dy_tc = sum(1 for r in tc if r['dual'])
        pct_tc = 100 * dy_tc / len(tc)
        print(f"  Among triangle-changing surgeries: {dy_tc}/{len(tc)} ({pct_tc:.1f}%)")

        stk_up_tc = sum(1 for r in tc if r['stk_dir'] == 'UP')
        stk_dn_tc = sum(1 for r in tc if r['stk_dir'] == 'DOWN')
        stk_sm_tc = sum(1 for r in tc if r['stk_dir'] == 'SAME')
        print(f"  Stokes direction (triangle-changing): UP={stk_up_tc}, DOWN={stk_dn_tc}, SAME={stk_sm_tc}")

        l0_dn_tc = sum(1 for r in tc if r['L0_dir'] == 'DOWN')
        l0_up_tc_n = sum(1 for r in tc if r['L0_dir'] == 'UP')
        l0_sm_tc = sum(1 for r in tc if r['L0_dir'] == 'SAME')
        print(f"  L0 direction (triangle-changing): DOWN={l0_dn_tc}, UP={l0_up_tc_n}, SAME={l0_sm_tc}")

    if stk_down_tc:
        print(f"\n  CONJECTURE STATUS: REFUTED by {len(stk_down_tc)} counterexample(s).")
        print(f"  The Stokes gap can DECREASE under vertex surgery that reduces b_1.")
    elif tc and all(r['stk_dir'] in ('UP', 'SAME') for r in tc):
        if all(r['stk_dir'] == 'UP' for r in tc):
            print(f"\n  STOKES HALF: CONFIRMED across all {len(tc)} triangle-changing surgeries.")
        else:
            stk_up_only = sum(1 for r in tc if r['stk_dir'] == 'UP')
            print(f"\n  STOKES HALF: Stokes never decreases. UP in {stk_up_only}/{len(tc)} "
                  f"({100*stk_up_only/len(tc):.1f}%), SAME in the rest.")

    if l0_up_tc:
        print(f"  L0 HALF: NOT universal — L0 increased in {len(l0_up_tc)}/{len(tc)} cases.")
    elif tc and all(r['L0_dir'] == 'DOWN' for r in tc):
        print(f"  L0 HALF: CONFIRMED — L0 always decreases when triangles change.")


if __name__ == '__main__':
    main()
