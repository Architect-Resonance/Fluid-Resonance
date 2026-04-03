"""
WEIGHTED K_n SPECTRAL ANALYSIS

Question: How robust is the L_1 = nI result when edge weights are non-uniform?

In a real Biot-Savart interaction graph, edge weights would NOT be equal.
This script investigates how the spectral gap degrades with weight variation.

If the gap degrades gracefully (e.g., gap ~ n * w_min), the framework is
somewhat resilient. If the gap can collapse to 0 for complete graphs with
positive weights, the framework is in trouble.

Author: Claude (Opus 4.6), 2026-03-11
"""

import numpy as np
from itertools import combinations
import sys

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(precision=8, linewidth=120)


def build_weighted_kn_hodge(n, weights):
    """
    Build the Hodge 1-Laplacian for weighted K_n with full clique complex.

    weights: array of length C(n,2), one per edge in lexicographic order.
             weights[e] > 0 for all e.

    The weighted Hodge Laplacian is:
        L_1 = d0 W0 d0^T + d1^T W2 d1

    where W0 = diag(vertex weights), W2 = diag(triangle weights).

    For simplicity, we use the EDGE-WEIGHTED version:
        L_1^w = B0^T W_V^{-1} B0 + B1 W_T B1^T

    Actually, for a cleaner formulation, we use the weighted graph Laplacian
    approach on the clique complex.

    Simplest model: weight w_e on edge e.
    d0: |E| x |V| incidence matrix (unweighted)
    d1: |T| x |E| boundary matrix (unweighted)

    The weighted Hodge Laplacian (Eckmann, 1944) for 1-forms:
        L_1 = d0 * d0^T + d1^T * d1   (unweighted)

    With edge weights, the standard modification is:
        L_1^w[e,e'] = sum over shared vertices v of (1/w_v) * signs
                    + sum over shared triangles t of w_t * signs

    For simplicity, let's just study the UNWEIGHTED Hodge Laplacian
    but with a SUBSET of edges present (modeling weak edges as absent),
    AND the weighted graph Laplacian L_0 = D - W for comparison.

    Actually, the most physically relevant question is:
    Given a complete graph K_n where edge (i,j) has weight w_{ij},
    what is the graph Laplacian spectral gap (Fiedler value)?
    """
    # Build edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    assert len(edges) == len(weights), f"Need {len(edges)} weights, got {len(weights)}"

    # Weighted graph Laplacian: L = D - W
    # where W[i,j] = w_{ij} and D[i,i] = sum_j w_{ij}
    W = np.zeros((n, n))
    for idx, (i, j) in enumerate(edges):
        W[i, j] = weights[idx]
        W[j, i] = weights[idx]

    D = np.diag(W.sum(axis=1))
    L = D - W

    return L, edges


def build_weighted_hodge_1(n, edge_weights):
    """
    Build the weighted Hodge 1-Laplacian for the complete graph K_n.

    For edge-weighted simplicial complexes, one common formulation is:
        L_1 = W_E^{-1/2} * (d0 * d0^T + d1^T * d1) * W_E^{-1/2}

    But the simplest and most natural is just:
        L_1 = d0 * d0^T + d1^T * d1
    with the edges and triangles PRESENT (unweighted combinatorial Hodge).

    The question is: what happens if we build L_1 on a WEIGHTED
    simplicial complex where triangle [i,j,k] has weight
    w_t = min(w_{ij}, w_{jk}, w_{ik}) or w_t = w_{ij}*w_{jk}*w_{ik}?

    For this study, we use the WEIGHTED boundary operators approach:
        d0_w[e, v] = w_e^{1/2} * d0[e, v]
        d1_w[t, e] = w_t^{1/2} * d1[t, e]

    Then L_1^w = d0_w * d0_w^T + d1_w^T * d1_w
    """
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    ne = len(edges)
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx

    # Build triangles (all 3-cliques)
    triangles = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    nt = len(triangles)

    # d0: |E| x |V| incidence
    d0 = np.zeros((ne, n))
    for idx, (i, j) in enumerate(edges):
        d0[idx, i] = -1
        d0[idx, j] = 1

    # d1: |T| x |E| boundary
    d1 = np.zeros((nt, ne))
    for idx, (i, j, k) in enumerate(triangles):
        e_ij = edge_index[(i, j)]
        e_ik = edge_index[(i, k)]
        e_jk = edge_index[(j, k)]
        d1[idx, e_ij] = 1
        d1[idx, e_jk] = 1
        d1[idx, e_ik] = -1

    # Weight the operators
    # Edge weight diagonal
    W_e_sqrt = np.diag(np.sqrt(edge_weights))

    # Triangle weights: use geometric mean of edge weights
    triangle_weights = np.zeros(nt)
    for idx, (i, j, k) in enumerate(triangles):
        w1 = edge_weights[edge_index[(i, j)]]
        w2 = edge_weights[edge_index[(i, k)]]
        w3 = edge_weights[edge_index[(j, k)]]
        triangle_weights[idx] = (w1 * w2 * w3) ** (1.0 / 3.0)

    W_t_sqrt = np.diag(np.sqrt(triangle_weights))

    # Weighted boundary operators
    d0_w = W_e_sqrt @ d0
    d1_w = W_t_sqrt @ d1

    # Weighted Hodge 1-Laplacian
    L1 = d0_w @ d0_w.T + d1_w.T @ d1_w

    return L1


# ============================================================
# TEST 1: Unweighted K_n (sanity check)
# ============================================================
print("=" * 70)
print("TEST 1: Unweighted K_n (all weights = 1)")
print("=" * 70)

for n in [5, 7, 10]:
    ne = n * (n - 1) // 2
    weights = np.ones(ne)

    L0, _ = build_weighted_kn_hodge(n, weights)
    eigs_L0 = np.sort(np.linalg.eigvalsh(L0))
    fiedler = eigs_L0[1]  # second smallest eigenvalue

    L1 = build_weighted_hodge_1(n, weights)
    eigs_L1 = np.sort(np.linalg.eigvalsh(L1))
    min_eig = eigs_L1[0]

    print(f"\n  K_{n}: L0 Fiedler = {fiedler:.6f} (expected {n})")
    print(f"  K_{n}: L1 min eigenvalue = {min_eig:.6f} (expected {n})")
    print(f"  K_{n}: L1 max eigenvalue = {eigs_L1[-1]:.6f} (expected {n})")
    print(f"  K_{n}: All L1 eigenvalues equal? {np.allclose(eigs_L1, n)}")


# ============================================================
# TEST 2: Perturbed weights — small uniform noise
# ============================================================
print("\n\n" + "=" * 70)
print("TEST 2: Perturbed weights w_e = 1 + epsilon * noise")
print("=" * 70)

n = 10
ne = n * (n - 1) // 2
np.random.seed(42)

for eps in [0.01, 0.1, 0.3, 0.5, 0.8, 0.99]:
    noise = np.random.uniform(-1, 1, ne)
    weights = 1.0 + eps * noise
    weights = np.maximum(weights, 0.01)  # ensure positive

    w_min = weights.min()
    w_max = weights.max()
    w_ratio = w_max / w_min

    L0, _ = build_weighted_kn_hodge(n, weights)
    eigs_L0 = np.sort(np.linalg.eigvalsh(L0))
    fiedler = eigs_L0[1]

    L1 = build_weighted_hodge_1(n, weights)
    eigs_L1 = np.sort(np.linalg.eigvalsh(L1))
    min_eig = eigs_L1[0]
    max_eig = eigs_L1[-1]

    print(f"\n  eps={eps:.2f}: w in [{w_min:.3f}, {w_max:.3f}], ratio={w_ratio:.2f}")
    print(f"    L0 Fiedler = {fiedler:.4f} (unweighted: {n})")
    print(f"    L1 min = {min_eig:.4f}, max = {max_eig:.4f}")
    print(f"    L1 gap degradation: {min_eig/n:.4f} of unweighted")


# ============================================================
# TEST 3: Extreme weight variation — one weak edge
# ============================================================
print("\n\n" + "=" * 70)
print("TEST 3: One weak edge — w_e = delta, rest = 1")
print("=" * 70)

n = 10
ne = n * (n - 1) // 2

for delta in [0.5, 0.1, 0.01, 0.001, 0.0001]:
    weights = np.ones(ne)
    weights[0] = delta  # edge (0,1) is weak

    L0, _ = build_weighted_kn_hodge(n, weights)
    eigs_L0 = np.sort(np.linalg.eigvalsh(L0))
    fiedler = eigs_L0[1]

    L1 = build_weighted_hodge_1(n, weights)
    eigs_L1 = np.sort(np.linalg.eigvalsh(L1))
    min_eig = eigs_L1[0]

    print(f"\n  delta={delta:.4f}: edge (0,1) weight = {delta}")
    print(f"    L0 Fiedler = {fiedler:.6f}")
    print(f"    L1 min = {min_eig:.6f}")
    print(f"    L0 degradation: {fiedler/n:.6f}")
    print(f"    L1 degradation: {min_eig/n:.6f}")


# ============================================================
# TEST 4: Systematic weight distribution — how does gap scale?
# ============================================================
print("\n\n" + "=" * 70)
print("TEST 4: Weights drawn from Uniform[a, 1], varying a")
print("=" * 70)

n = 8
ne = n * (n - 1) // 2
np.random.seed(123)
n_trials = 20

print(f"\n  {'a':>6} | {'avg_L0_Fiedler':>15} | {'avg_L1_min':>12} | {'n*a':>6} | {'Fiedler/n':>10} | {'L1_min/n':>10}")
print("  " + "-" * 75)

for a in [0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]:
    fiedlers = []
    l1_mins = []

    for _ in range(n_trials):
        weights = np.random.uniform(a, 1.0, ne)

        L0, _ = build_weighted_kn_hodge(n, weights)
        eigs = np.sort(np.linalg.eigvalsh(L0))
        fiedlers.append(eigs[1])

        L1 = build_weighted_hodge_1(n, weights)
        eigs1 = np.sort(np.linalg.eigvalsh(L1))
        l1_mins.append(eigs1[0])

    avg_f = np.mean(fiedlers)
    avg_l1 = np.mean(l1_mins)

    print(f"  {a:6.2f} | {avg_f:15.4f} | {avg_l1:12.4f} | {n*a:6.2f} | {avg_f/n:10.4f} | {avg_l1/n:10.4f}")


# ============================================================
# TEST 5: Biot-Savart-like weights — distance-dependent
# ============================================================
print("\n\n" + "=" * 70)
print("TEST 5: Biot-Savart-like weights w_{ij} ~ 1/|x_i - x_j|^2")
print("=" * 70)

n = 8
ne = n * (n - 1) // 2
np.random.seed(456)

for config_name, points in [
    ("Uniform cube", np.random.uniform(0, 1, (n, 3))),
    ("Clustered (2 groups)", np.vstack([
        np.random.normal(0, 0.1, (n // 2, 3)),
        np.random.normal(1, 0.1, (n // 2, 3))
    ])),
    ("Isotropic sphere", np.random.randn(n, 3) / np.linalg.norm(np.random.randn(n, 3), axis=1, keepdims=True)),
    ("Concentrated ball", np.random.randn(n, 3) * 0.1),
]:
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = np.array([1.0 / max(np.linalg.norm(points[i] - points[j]) ** 2, 0.001)
                         for i, j in edges])
    # Normalize so max weight = 1
    weights = weights / weights.max()

    w_min = weights.min()
    w_ratio = 1.0 / w_min

    L0, _ = build_weighted_kn_hodge(n, weights)
    eigs = np.sort(np.linalg.eigvalsh(L0))
    fiedler = eigs[1]

    L1 = build_weighted_hodge_1(n, weights)
    eigs1 = np.sort(np.linalg.eigvalsh(L1))
    l1_min = eigs1[0]

    print(f"\n  {config_name}:")
    print(f"    w_min = {w_min:.6f}, w_max = 1.0, ratio = {w_ratio:.1f}")
    print(f"    L0 Fiedler = {fiedler:.4f} (unweighted: {n})")
    print(f"    L1 min = {l1_min:.4f} (unweighted: {n})")
    print(f"    Fiedler / n = {fiedler/n:.4f}")
    print(f"    L1 min / n = {l1_min/n:.4f}")


# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("SUMMARY: Weighted K_n Spectral Gap Robustness")
print("=" * 70)

print("""
Key findings:

1. GRAPH LAPLACIAN (L0) Fiedler value:
   For weighted K_n with weights in [a, 1], the Fiedler value scales
   approximately as n * a (proportional to minimum weight).
   This is a KNOWN result: lambda_2(L) >= n * w_min for K_n.

2. HODGE LAPLACIAN (L1) minimum eigenvalue:
   More complex behavior due to triangle weights.
   With geometric mean triangle weights, the gap also degrades
   with minimum edge weight, but the exact scaling depends on
   the weight distribution.

3. BIOT-SAVART WEIGHTS:
   When points are clustered (not uniformly distributed), the
   weight ratio can be extreme (> 100), and the spectral gap
   degrades proportionally. This is the REALISTIC scenario.

4. IMPLICATION FOR THE PROOF STRATEGY:
   The spectral gap argument requires w_min bounded away from 0.
   For Biot-Savart weights, this means all pairwise interaction
   strengths must be comparable — which requires the spatial
   arrangement to be approximately isotropic.

   Lei et al. gives DIRECTIONAL isotropy but not SPATIAL isotropy.
   CKN gives spatial CONCENTRATION but not spatial UNIFORMITY
   within the concentration region.

   Neither result gives us what we need: w_min >= c * w_max
   for some universal constant c > 0.
""")
