"""
Test the Dual Effect Conjecture for Simplicial Surgery:

  Conjecture: For simplicial surgery on a graph that reduces b_1 by 1,
  the graph Laplacian gap (Fiedler value of L_0) DECREASES
  but the Stokes gap (smallest eigenvalue of Hodge 1-Laplacian L_1
  restricted to divergence-free 1-forms) INCREASES.

We build diverse graphs, compute spectral data, try every single-edge
removal that reduces b_1, and track whether the dual effect holds.

Author: Claude Opus 4.6 (Meridian), 2026-03-12
"""

import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.set_printoptions(precision=6, suppress=True)

EPS = 1e-9  # tolerance for zero eigenvalues


# ============================================================
# GRAPH AND SIMPLICIAL COMPLEX CONSTRUCTION
# ============================================================

def edges_from_adj(adj_list):
    """Return sorted list of oriented edges (i,j) with i < j from adjacency list."""
    edges = set()
    for u in adj_list:
        for v in adj_list[u]:
            if u < v:
                edges.add((u, v))
    return sorted(edges)


def adj_from_edges(edges, n_vertices):
    """Build adjacency list from edge list."""
    adj = {i: set() for i in range(n_vertices)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def find_triangles(edges, n_vertices):
    """Find all triangles (3-cliques) in the graph. Return sorted list of (i,j,k) with i<j<k."""
    adj = adj_from_edges(edges, n_vertices)
    triangles = set()
    for u, v in edges:
        common = adj[u] & adj[v]
        for w in common:
            tri = tuple(sorted([u, v, w]))
            triangles.add(tri)
    return sorted(triangles)


def build_incidence_d0(edges, n_vertices):
    """
    Build the vertex-to-edge incidence matrix d_0 (boundary operator B_1^T in some conventions).
    d_0 is |E| x |V|. For edge (i,j) with i<j: d_0[e, i] = -1, d_0[e, j] = +1.
    Then L_0 = d_0^T d_0 is the standard graph Laplacian.
    """
    n_edges = len(edges)
    d0 = np.zeros((n_edges, n_vertices))
    for e_idx, (i, j) in enumerate(edges):
        d0[e_idx, i] = -1.0
        d0[e_idx, j] = 1.0
    return d0


def build_incidence_d1(edges, triangles):
    """
    Build the edge-to-triangle boundary operator d_1.
    d_1 is |T| x |E|. For triangle (i,j,k) with i<j<k:
      faces are (i,j), (i,k), (j,k)
      with boundary signs: (i,j) -> +1, (i,k) -> -1, (j,k) -> +1
    """
    edge_index = {e: idx for idx, e in enumerate(edges)}
    n_edges = len(edges)
    n_tri = len(triangles)
    d1 = np.zeros((n_tri, n_edges))
    for t_idx, (i, j, k) in enumerate(triangles):
        # boundary of (i,j,k) = (j,k) - (i,k) + (i,j)
        e_ij = edge_index.get((i, j))
        e_ik = edge_index.get((i, k))
        e_jk = edge_index.get((j, k))
        if e_ij is not None:
            d1[t_idx, e_ij] = 1.0
        if e_ik is not None:
            d1[t_idx, e_ik] = -1.0
        if e_jk is not None:
            d1[t_idx, e_jk] = 1.0
    return d1


# ============================================================
# SPECTRAL COMPUTATIONS
# ============================================================

def compute_betti_1(n_vertices, edges, triangles):
    """b_1 = dim(ker(d_1^T)) - dim(im(d_0)) = |E| - rank(d0) - rank(d1)
    For connected graph: b_1 = |E| - |V| + 1 - rank(d1... no.
    Actually b_1 = dim(ker(B_1)) - dim(im(B_2)) but let's use the formula:
    b_1 = |E| - |V| + components  for a graph without higher simplices correction.
    More precisely for a simplicial complex:
    b_1 = |E| - rank(d_0) - rank(d_1)
    """
    d0 = build_incidence_d0(edges, n_vertices)
    if len(triangles) > 0:
        d1 = build_incidence_d1(edges, triangles)
        rank_d1 = np.linalg.matrix_rank(d1, tol=1e-10)
    else:
        rank_d1 = 0
    rank_d0 = np.linalg.matrix_rank(d0, tol=1e-10)
    b1 = len(edges) - rank_d0 - rank_d1
    return b1


def compute_graph_betti_1(n_vertices, edges):
    """Graph-only b_1 (ignoring triangles): b_1 = |E| - |V| + connected_components."""
    # BFS to count components
    adj = adj_from_edges(edges, n_vertices)
    visited = set()
    components = 0
    for v in range(n_vertices):
        if v not in visited:
            components += 1
            stack = [v]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                for w in adj[u]:
                    if w not in visited:
                        stack.append(w)
    return len(edges) - n_vertices + components


def compute_L0_gap(d0):
    """Compute the Fiedler value (smallest nonzero eigenvalue of L_0 = d_0^T d_0)."""
    L0 = d0.T @ d0
    evals = np.linalg.eigvalsh(L0)
    # Find smallest eigenvalue > EPS
    nonzero = evals[evals > EPS]
    if len(nonzero) == 0:
        return 0.0
    return float(nonzero[0])


def compute_stokes_gap(d0, d1, n_edges):
    """
    Compute the Stokes gap: smallest eigenvalue of the Hodge 1-Laplacian L_1
    restricted to ker(d_0^T) (divergence-free 1-forms).

    L_1 = d_0 d_0^T + d_1^T d_1  (note: d_0 is |E|x|V|, so d_0 d_0^T is |E|x|E|)

    Wait -- careful with conventions. Let me be explicit:
    - d_0 is |E| x |V|: maps vertices to edges (coboundary / gradient)
    - d_0^T is |V| x |E|: maps edges to vertices (divergence)
    - d_1 is |T| x |E|: maps edges to triangles (curl / coboundary)
    - d_1^T is |E| x |T|: maps triangles to edges

    The Hodge 1-Laplacian is:
    L_1 = d_0 d_0^T + d_1^T d_1   ... no wait.

    Standard convention:
    - B_1 (boundary_1) is |V| x |E| -- maps edges to their boundary vertices
    - B_2 (boundary_2) is |E| x |T| -- maps triangles to their boundary edges

    Then L_1 = B_1^T B_1 + B_2 B_2^T

    In our setup: d_0 = B_1^T (|E| x |V|), d_1 = B_2^T (|T| x |E|)
    So: B_1 = d_0^T, B_2 = d_1^T

    L_1 = (d_0^T)^T (d_0^T) + (d_1^T)(d_1^T)^T = d_0 d_0^T + d_1^T d_1

    Wait, that's not right either. Let me think again.

    Actually the standard Hodge Laplacian on k-forms is:
    Delta_k = delta_{k+1} d_{k+1}^* + d_k^* delta_k

    In the simplicial case with the standard inner product:
    Delta_1 = B_1^T B_1 + B_2 B_2^T  where B_1: edges->vertices, B_2: triangles->edges

    Our d_0 (|E|x|V|) is the coboundary d_0 = B_1^T
    Our d_1 (|T|x|E|) is the coboundary d_1 = B_2^T (but transposed from standard)

    Hmm, let me just use the standard definition directly:

    B_1 = d_0^T (|V| x |E|)
    B_2 = d_1^T (|E| x |T|)

    L_1 = B_1^T B_1 + B_2 B_2^T = d_0 d_0^T + d_1^T d_1... wait no:
    B_1^T = (d_0^T)^T = d_0 ... shape |E|x|V|... B_1^T has shape |E|x|V|
    B_1^T B_1 has shape |E|x|E|? No: B_1 is |V|x|E|, B_1^T is |E|x|V|, so B_1^T B_1 is |E|x|E|?
    No: B_1^T (|E|x|V|) @ B_1 (|V|x|E|) = |E|x|E|. Yes.

    B_2 is |E|x|T|, B_2^T is |T|x|E|. B_2 B_2^T is |E|x|E|. Yes.

    So L_1 = B_1^T B_1 + B_2 B_2^T where both are |E|x|E|.

    Now B_1^T = d_0 (our |E|x|V| matrix), so B_1 = d_0^T.
    B_1^T B_1 = d_0 @ d_0^T ... wait d_0 is |E|x|V|, d_0^T is |V|x|E|.
    d_0 @ d_0^T is |E|x|E|. Hmm but that's not L_0.
    L_0 = d_0^T @ d_0 = |V|x|V|. Yes L_0 is the graph Laplacian.

    And B_2 = d_1^T (|E|x|T|). B_2^T = d_1 (|T|x|E|).
    B_2 B_2^T = d_1^T @ d_1 (|E|x|E|).

    So: L_1 = d_0 @ d_0^T + d_1^T @ d_1
    Both |E|x|E|. This is correct!

    The divergence-free condition: ker(B_1) = ker(d_0^T)
    i.e., d_0^T f = 0 means sum of flows into each vertex = 0.

    Actually wait: B_1 f = d_0^T f. The divergence of f at vertex v is (B_1 f)_v = (d_0^T f)_v.
    So divergence-free means B_1 f = 0, i.e., d_0^T f = 0. But d_0^T is |V|x|E|, so
    ker(d_0^T) = {f in R^|E| : d_0^T f = 0}.

    Wait, I need to be more careful. d_0 is |E|x|V|. d_0^T is |V|x|E|.
    ker(d_0^T) means {f in R^|E| : d_0^T f = 0}. This is the divergence-free subspace.
    """
    # Build L_1
    # d_0 is |E|x|V|, d_1 is |T|x|E|
    L1_lower = d0 @ d0.T  # |E|x|E| -- the "down" Laplacian
    if d1.shape[0] > 0:
        L1_upper = d1.T @ d1   # |E|x|E| -- the "up" Laplacian
    else:
        L1_upper = np.zeros((n_edges, n_edges))
    L1 = L1_lower + L1_upper

    # Find ker(d_0^T) = divergence-free subspace
    # d_0^T is |V|x|E|
    d0T = d0.T  # |V|x|E|

    # SVD of d0T to find its null space
    U, S, Vt = np.linalg.svd(d0T, full_matrices=True)
    # Null space of d0T is the last columns of Vt^T (= rows of Vt) corresponding to zero singular values
    rank = np.sum(S > EPS)
    null_space = Vt[rank:].T  # |E| x dim(null space)

    dim_divfree = null_space.shape[1]
    if dim_divfree == 0:
        return float('inf'), 0  # no divergence-free forms

    # Project L_1 onto the divergence-free subspace
    L1_proj = null_space.T @ L1 @ null_space  # dim_divfree x dim_divfree

    # Eigenvalues of the projected operator
    evals = np.linalg.eigvalsh(L1_proj)

    # The harmonic 1-forms (b_1 of them) will be zero eigenvalues
    nonzero = evals[evals > EPS]

    if len(nonzero) == 0:
        return 0.0, dim_divfree  # all harmonic, no Stokes gap

    return float(nonzero[0]), dim_divfree


def analyze_graph(n_vertices, edges, name=""):
    """Full spectral analysis of a graph/simplicial complex."""
    triangles = find_triangles(edges, n_vertices)

    d0 = build_incidence_d0(edges, n_vertices)
    if len(triangles) > 0:
        d1 = build_incidence_d1(edges, triangles)
    else:
        d1 = np.zeros((0, len(edges)))

    # Verify d^2 = 0 if we have triangles
    if d1.shape[0] > 0:
        chain_check = np.max(np.abs(d1 @ d0))
        if chain_check > 1e-10:
            print(f"  WARNING: d1 @ d0 != 0 (max = {chain_check})")

    b1 = compute_betti_1(n_vertices, edges, triangles)
    graph_b1 = compute_graph_betti_1(n_vertices, edges)
    L0_gap = compute_L0_gap(d0)
    stokes_gap, dim_divfree = compute_stokes_gap(d0, d1, len(edges))

    return {
        'name': name,
        'n_vertices': n_vertices,
        'n_edges': len(edges),
        'n_triangles': len(triangles),
        'b1': b1,
        'graph_b1': graph_b1,
        'L0_gap': L0_gap,
        'stokes_gap': stokes_gap,
        'dim_divfree': dim_divfree,
        'edges': edges,
        'triangles': triangles
    }


def is_connected(n_vertices, edges):
    """Check if the graph is connected."""
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


# ============================================================
# GRAPH GENERATORS
# ============================================================

def make_complete_graph(n):
    """K_n graph."""
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    return n, edges, f"K_{n}"


def make_cycle_graph(n):
    """Cycle C_n."""
    edges = [(i, (i+1) % n) for i in range(n)]
    # Reorient so i < j
    edges = [tuple(sorted(e)) for e in edges]
    edges = sorted(set(edges))
    return n, edges, f"C_{n}"


def make_cycle_with_chords(n, chords):
    """Cycle C_n with extra chord edges."""
    n_v, edges, _ = make_cycle_graph(n)
    for c in chords:
        e = tuple(sorted(c))
        if e not in edges:
            edges.append(e)
    edges = sorted(edges)
    name = f"C_{n}+{len(chords)}chords"
    return n_v, edges, name


def make_k5_plus_anchors():
    """
    The K5+2anchor graph from the Fluid-Resonance project.
    7 vertices: K5 core {0,1,2,3,4}, anchors {5,6}.
    Anchor 5 connects to {0,3,5_self?} no -- connects to core vertices.

    From the Laplacian in RESONANCE_STATE.json (8x8 but vertex 7 has degree 2):
    Actually looking at the 8x8 Laplacian:
    vertex 0: degree 7, connects to 1,2,3,4,5 (and itself row has -1s at 0-5)
    Wait, let me read the Laplacian more carefully.

    L_8x8:
    [[ 7 -1 -1 -1 -1 -1  0  0],   vertex 0: deg 7, connects to 1,2,3,4,5 (6 neighbors but grounded, so +1)
     [-1  6 -1 -1 -1  0  0  0],   vertex 1: deg 6, connects to 0,2,3,4
     [-1 -1  6 -1 -1  0 -1  0],   vertex 2: deg 6, connects to 0,1,3,4,6
     [-1 -1 -1  5 -1 -1  0  0],   vertex 3: deg 5, connects to 0,1,2,4,5
     [-1 -1 -1 -1  6  0 -1  0],   vertex 4: deg 6, connects to 0,1,2,3,6
     [-1  0  0 -1  0  4 -1 -1],   vertex 5: deg 4, connects to 0,3,6,7
     [ 0  0 -1  0 -1 -1  4 -1],   vertex 6: deg 4, connects to 2,4,5,7
     [ 0  0  0  0  0 -1 -1  2]]   vertex 7: deg 2, connects to 5,6

    But this is a grounded Laplacian (one vertex removed). The full graph has 9 vertices
    (0-7 plus a grounded vertex). Actually, the "grounded" Laplacian just means we
    removed the hub vertex. For our test, let's build the 7-vertex version directly:
    K5 on {0,1,2,3,4}, anchors {5,6} with specific bridge edges.

    From the Laplacian, anchor connections:
    - 5 connects to: 0, 3, 6 (and 7 which is beyond our 7-vertex model)
    - 6 connects to: 2, 4, 5 (and 7)

    Let's use 7 vertices (the K5 core + 2 anchors) with the bridge structure.
    """
    n = 7
    # K5 core
    edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
    # Bridge edges from anchors
    edges.extend([(0, 5), (3, 5), (5, 6), (2, 6), (4, 6)])
    edges = sorted(set(edges))
    return n, edges, "K5+2anchor"


def make_k5_plus_extra_edges():
    """K5 with additional vertices and edges creating extra topology."""
    n = 7
    edges = [(i, j) for i in range(5) for j in range(i+1, 5)]  # K5
    # Add two pendant-like structures that create cycles
    edges.extend([(0, 5), (1, 5), (2, 5)])  # vertex 5 connects to 0,1,2
    edges.extend([(3, 6), (4, 6), (0, 6)])  # vertex 6 connects to 3,4,0
    edges = sorted(set(edges))
    return n, edges, "K5+2hubs"


def make_petersen_graph():
    """The Petersen graph: 10 vertices, 15 edges, b_1 = 6."""
    outer = list(range(5))
    inner = list(range(5, 10))
    edges = []
    # Outer pentagon
    for i in range(5):
        edges.append(tuple(sorted((outer[i], outer[(i+1) % 5]))))
    # Inner pentagram
    for i in range(5):
        edges.append(tuple(sorted((inner[i], inner[(i+2) % 5]))))
    # Spokes
    for i in range(5):
        edges.append(tuple(sorted((outer[i], inner[i]))))
    edges = sorted(set(edges))
    return 10, edges, "Petersen"


def make_prism_graph(n):
    """Prism graph: two n-cycles connected by rungs. 2n vertices."""
    edges = []
    # Top cycle
    for i in range(n):
        edges.append(tuple(sorted((i, (i+1) % n))))
    # Bottom cycle
    for i in range(n):
        edges.append(tuple(sorted((n+i, n+(i+1) % n))))
    # Rungs
    for i in range(n):
        edges.append((i, n+i))
    edges = sorted(set(edges))
    return 2*n, edges, f"Prism_{n}"


def make_grid_graph(m, n):
    """m x n grid graph (torus-like if you add wrap edges, but we keep it planar)."""
    def idx(i, j):
        return i * n + j
    edges = []
    for i in range(m):
        for j in range(n):
            if j + 1 < n:
                edges.append(tuple(sorted((idx(i, j), idx(i, j+1)))))
            if i + 1 < m:
                edges.append(tuple(sorted((idx(i, j), idx(i+1, j)))))
    # Add wrap edges to create cycles
    for i in range(m):
        edges.append(tuple(sorted((idx(i, 0), idx(i, n-1)))))
    for j in range(n):
        edges.append(tuple(sorted((idx(0, j), idx(m-1, j)))))
    edges = sorted(set(edges))
    return m*n, edges, f"Torus_{m}x{n}"


def make_random_graph(n, p, seed=None):
    """Erdos-Renyi random graph G(n, p)."""
    rng = np.random.RandomState(seed)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                edges.append((i, j))
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
    if len(visited) < n:
        # Add edges to connect components
        for v in range(n):
            if v not in visited:
                edges.append(tuple(sorted((0, v))))
                visited.add(v)
    edges = sorted(set(edges))
    return n, edges, f"ER({n},{p:.2f},s={seed})"


def make_wheel_graph(n):
    """Wheel graph: hub + n-cycle."""
    # Hub is vertex 0, cycle is 1..n
    edges = []
    for i in range(1, n+1):
        edges.append((0, i))  # spokes
        edges.append(tuple(sorted((i, (i % n) + 1))))  # rim
    edges = sorted(set(edges))
    return n+1, edges, f"Wheel_{n}"


def make_double_cycle(n1, n2, bridge_edges=1):
    """Two cycles sharing some edges or connected by bridges."""
    edges = []
    # First cycle on 0..n1-1
    for i in range(n1):
        edges.append(tuple(sorted((i, (i+1) % n1))))
    # Second cycle on n1..n1+n2-1
    for i in range(n2):
        edges.append(tuple(sorted((n1+i, n1+(i+1) % n2))))
    # Bridge edges
    for b in range(bridge_edges):
        edges.append(tuple(sorted((b % n1, n1 + (b % n2)))))
    edges = sorted(set(edges))
    return n1+n2, edges, f"DblCyc_{n1}_{n2}_b{bridge_edges}"


# ============================================================
# SURGERY: TRY REMOVING EACH EDGE
# ============================================================

def try_all_surgeries(n_vertices, edges, name, results_list):
    """
    For each edge in the graph, remove it and check:
    1. Does the graph stay connected?
    2. Does b_1 decrease by exactly 1?
    3. If so, compute spectral data before and after, record dual effect.
    """
    # Pre-surgery analysis
    pre = analyze_graph(n_vertices, edges, name)

    if pre['b1'] == 0 and pre['graph_b1'] == 0:
        return 0  # No cycles to remove

    surgery_count = 0

    for edge_idx, removed_edge in enumerate(edges):
        new_edges = [e for e in edges if e != removed_edge]

        # Check connectivity
        if not is_connected(n_vertices, new_edges):
            continue

        # Check if b_1 decreased
        new_triangles = find_triangles(new_edges, n_vertices)
        new_b1 = compute_betti_1(n_vertices, new_edges, new_triangles)

        if new_b1 != pre['b1'] - 1:
            continue  # We want surgeries that reduce b_1 by exactly 1

        # This edge removal reduces b_1 by 1 while keeping graph connected
        post = analyze_graph(n_vertices, new_edges, f"{name} \\ {removed_edge}")

        # Only consider cases where both gaps are well-defined
        if pre['stokes_gap'] == 0.0 or pre['stokes_gap'] == float('inf'):
            continue
        if post['stokes_gap'] == 0.0 or post['stokes_gap'] == float('inf'):
            continue
        if pre['L0_gap'] < EPS or post['L0_gap'] < EPS:
            continue

        L0_decreased = post['L0_gap'] < pre['L0_gap'] - EPS
        L0_increased = post['L0_gap'] > pre['L0_gap'] + EPS
        L0_same = not L0_decreased and not L0_increased

        stokes_increased = post['stokes_gap'] > pre['stokes_gap'] + EPS
        stokes_decreased = post['stokes_gap'] < pre['stokes_gap'] - EPS
        stokes_same = not stokes_increased and not stokes_decreased

        dual_effect = L0_decreased and stokes_increased

        results_list.append({
            'graph': name,
            'removed_edge': removed_edge,
            'b1_before': pre['b1'],
            'b1_after': post['b1'],
            'L0_before': pre['L0_gap'],
            'L0_after': post['L0_gap'],
            'L0_direction': 'DOWN' if L0_decreased else ('UP' if L0_increased else 'SAME'),
            'stokes_before': pre['stokes_gap'],
            'stokes_after': post['stokes_gap'],
            'stokes_direction': 'UP' if stokes_increased else ('DOWN' if stokes_decreased else 'SAME'),
            'dual_effect': dual_effect,
            'n_tri_before': pre['n_triangles'],
            'n_tri_after': post['n_triangles'],
        })

        surgery_count += 1

    return surgery_count


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("DUAL EFFECT CONJECTURE TEST")
    print("Conjecture: Edge removal reducing b_1 => L0 gap DECREASES, Stokes gap INCREASES")
    print("=" * 100)
    print()

    results = []

    # Build graph catalog
    graphs = []

    # 1. Complete graphs (b_1 is large for K_n)
    for n in [4, 5, 6, 7]:
        graphs.append(make_complete_graph(n))

    # 2. Cycle graphs with chords
    graphs.append(make_cycle_with_chords(6, [(0, 3)]))
    graphs.append(make_cycle_with_chords(6, [(0, 3), (1, 4)]))
    graphs.append(make_cycle_with_chords(6, [(0, 2), (0, 3), (0, 4)]))
    graphs.append(make_cycle_with_chords(7, [(0, 3), (1, 5)]))
    graphs.append(make_cycle_with_chords(8, [(0, 4), (2, 6)]))
    graphs.append(make_cycle_with_chords(8, [(0, 4), (2, 6), (1, 5)]))
    graphs.append(make_cycle_with_chords(10, [(0, 5), (2, 7), (3, 8)]))

    # 3. K5 + anchor (the project's main graph)
    graphs.append(make_k5_plus_anchors())

    # 4. K5 with extra hubs
    graphs.append(make_k5_plus_extra_edges())

    # 5. Petersen graph
    graphs.append(make_petersen_graph())

    # 6. Prism graphs
    for n in [3, 4, 5, 6]:
        graphs.append(make_prism_graph(n))

    # 7. Wheel graphs
    for n in [4, 5, 6, 7, 8]:
        graphs.append(make_wheel_graph(n))

    # 8. Torus grids
    graphs.append(make_grid_graph(3, 3))
    graphs.append(make_grid_graph(3, 4))
    graphs.append(make_grid_graph(4, 4))

    # 9. Double cycles
    graphs.append(make_double_cycle(4, 4, 1))
    graphs.append(make_double_cycle(5, 5, 1))
    graphs.append(make_double_cycle(4, 4, 2))
    graphs.append(make_double_cycle(5, 5, 2))
    graphs.append(make_double_cycle(3, 3, 1))

    # 10. Random graphs with various densities
    for seed in range(10):
        graphs.append(make_random_graph(8, 0.4, seed=seed))
        graphs.append(make_random_graph(10, 0.35, seed=seed+100))

    # 11. More structured graphs
    # Moebius-Kantor-like: 8-cycle with specific chords
    graphs.append(make_cycle_with_chords(8, [(0, 3), (1, 6), (2, 5), (4, 7)]))
    # Dense cycle with many chords
    graphs.append(make_cycle_with_chords(8, [(0, 2), (0, 4), (1, 3), (1, 5), (2, 6), (3, 7)]))
    # 12-cycle with 4 chords
    graphs.append(make_cycle_with_chords(12, [(0, 6), (3, 9), (1, 7), (4, 10)]))

    print(f"Built {len(graphs)} test graphs.\n")

    total_surgeries = 0
    for nv, edges, name in graphs:
        # Quick pre-check: is graph connected and has b_1 > 0?
        if not is_connected(nv, edges):
            continue
        graph_b1 = compute_graph_betti_1(nv, edges)
        if graph_b1 == 0:
            continue

        count = try_all_surgeries(nv, edges, name, results)
        total_surgeries += count

    print(f"\nTotal valid surgery experiments: {total_surgeries}")
    print(f"(Surgeries where b_1 decreased by 1, graph stayed connected, both gaps well-defined)\n")

    if not results:
        print("No valid surgeries found!")
        return

    # ============================================================
    # SUMMARY TABLE
    # ============================================================

    print("=" * 140)
    print(f"{'Graph':<30} {'Edge':<12} {'b1':>5} {'b1*':>5} {'L0_gap':>10} {'L0*_gap':>10} {'L0':>6} "
          f"{'Stokes':>10} {'Stokes*':>10} {'Stk':>6} {'Dual?':>6}")
    print("-" * 140)

    dual_holds = 0
    dual_fails = 0
    l0_down_stokes_down = 0
    l0_up_stokes_up = 0
    l0_up_stokes_down = 0
    l0_down_stokes_same = 0
    l0_same_stokes_up = 0
    other = 0

    for r in results:
        dual_str = "YES" if r['dual_effect'] else "no"
        print(f"{r['graph']:<30} {str(r['removed_edge']):<12} {r['b1_before']:>5} {r['b1_after']:>5} "
              f"{r['L0_before']:>10.6f} {r['L0_after']:>10.6f} {r['L0_direction']:>6} "
              f"{r['stokes_before']:>10.6f} {r['stokes_after']:>10.6f} {r['stokes_direction']:>6} "
              f"{dual_str:>6}")

        if r['dual_effect']:
            dual_holds += 1
        else:
            dual_fails += 1

        if r['L0_direction'] == 'DOWN' and r['stokes_direction'] == 'DOWN':
            l0_down_stokes_down += 1
        elif r['L0_direction'] == 'UP' and r['stokes_direction'] == 'UP':
            l0_up_stokes_up += 1
        elif r['L0_direction'] == 'UP' and r['stokes_direction'] == 'DOWN':
            l0_up_stokes_down += 1
        elif r['L0_direction'] == 'DOWN' and r['stokes_direction'] == 'SAME':
            l0_down_stokes_same += 1
        elif r['L0_direction'] == 'SAME' and r['stokes_direction'] == 'UP':
            l0_same_stokes_up += 1
        elif not r['dual_effect']:
            other += 1

    print("=" * 140)

    # ============================================================
    # STATISTICS
    # ============================================================

    total = dual_holds + dual_fails
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY ({total} surgeries)")
    print(f"{'='*60}")
    print(f"  Dual effect HOLDS (L0 DOWN, Stokes UP):  {dual_holds:>4} ({100*dual_holds/total:.1f}%)")
    print(f"  Dual effect FAILS:                        {dual_fails:>4} ({100*dual_fails/total:.1f}%)")
    print(f"")
    print(f"  Breakdown of failures:")
    print(f"    L0 DOWN, Stokes DOWN:    {l0_down_stokes_down:>4}")
    print(f"    L0 UP,   Stokes UP:      {l0_up_stokes_up:>4}")
    print(f"    L0 UP,   Stokes DOWN:    {l0_up_stokes_down:>4}")
    print(f"    L0 DOWN, Stokes SAME:    {l0_down_stokes_same:>4}")
    print(f"    L0 SAME, Stokes UP:      {l0_same_stokes_up:>4}")
    print(f"    Other:                    {other:>4}")

    # ============================================================
    # COUNTEREXAMPLE DETAILS
    # ============================================================

    counterexamples = [r for r in results if not r['dual_effect']]
    if counterexamples:
        print(f"\n{'='*60}")
        print("COUNTEREXAMPLES (where dual effect fails)")
        print(f"{'='*60}")
        for r in counterexamples[:20]:  # Show first 20
            print(f"\n  Graph: {r['graph']}")
            print(f"  Removed edge: {r['removed_edge']}")
            print(f"  b_1: {r['b1_before']} -> {r['b1_after']}")
            print(f"  L0 gap: {r['L0_before']:.6f} -> {r['L0_after']:.6f} ({r['L0_direction']})")
            print(f"  Stokes gap: {r['stokes_before']:.6f} -> {r['stokes_after']:.6f} ({r['stokes_direction']})")
            print(f"  Triangles: {r['n_tri_before']} -> {r['n_tri_after']}")

    # ============================================================
    # SPECIFIC K5+2ANCHOR RESULTS
    # ============================================================

    k5_results = [r for r in results if 'K5+2anchor' in r['graph']]
    if k5_results:
        print(f"\n{'='*60}")
        print("K5+2ANCHOR SPECIFIC RESULTS")
        print(f"{'='*60}")
        for r in k5_results:
            print(f"\n  Removed edge: {r['removed_edge']}")
            print(f"  b_1: {r['b1_before']} -> {r['b1_after']}")
            print(f"  L0 gap: {r['L0_before']:.6f} -> {r['L0_after']:.6f} ({r['L0_direction']})")
            print(f"  Stokes gap: {r['stokes_before']:.6f} -> {r['stokes_after']:.6f} ({r['stokes_direction']})")
            print(f"  Dual effect: {'YES' if r['dual_effect'] else 'NO'}")

    # ============================================================
    # ANALYSIS BY GRAPH FAMILY
    # ============================================================

    print(f"\n{'='*60}")
    print("DUAL EFFECT RATE BY GRAPH FAMILY")
    print(f"{'='*60}")

    families = {}
    for r in results:
        # Extract family name
        name = r['graph']
        if name.startswith('K_'):
            family = 'Complete'
        elif name.startswith('C_'):
            family = 'Cycle+chords'
        elif 'K5+' in name:
            family = 'K5+anchors'
        elif name.startswith('Petersen'):
            family = 'Petersen'
        elif name.startswith('Prism'):
            family = 'Prism'
        elif name.startswith('Wheel'):
            family = 'Wheel'
        elif name.startswith('Torus'):
            family = 'Torus'
        elif name.startswith('DblCyc'):
            family = 'DoubleCycle'
        elif name.startswith('ER'):
            family = 'Random'
        else:
            family = 'Other'

        if family not in families:
            families[family] = {'holds': 0, 'fails': 0}
        if r['dual_effect']:
            families[family]['holds'] += 1
        else:
            families[family]['fails'] += 1

    for family in sorted(families.keys()):
        h = families[family]['holds']
        f = families[family]['fails']
        t = h + f
        pct = 100 * h / t if t > 0 else 0
        print(f"  {family:<20}: {h:>3}/{t:>3} hold ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    pct_hold = 100 * dual_holds / total
    if dual_fails == 0:
        print(f"  The dual effect conjecture HOLDS UNIVERSALLY across all {total} tested surgeries.")
    elif pct_hold > 90:
        print(f"  The dual effect holds in {pct_hold:.1f}% of cases ({dual_holds}/{total}).")
        print(f"  There are {dual_fails} counterexamples — the conjecture is NOT universal.")
    elif pct_hold > 50:
        print(f"  The dual effect holds in only {pct_hold:.1f}% of cases ({dual_holds}/{total}).")
        print(f"  The conjecture is a tendency, not a theorem.")
    else:
        print(f"  The dual effect holds in only {pct_hold:.1f}% of cases ({dual_holds}/{total}).")
        print(f"  The conjecture appears to be FALSE in general.")


if __name__ == '__main__':
    main()
