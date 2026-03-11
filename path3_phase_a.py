"""
Path 3, Phase A: Simplicial Complex + Hodge Laplacians

Step 3.1: Build the simplicial complex from the cluster graph
  - 0-simplices (vertices) = variables
  - 1-simplices (edges) = variable pairs sharing a clause
  - 2-simplices (triangles) = clauses (each 3-SAT clause is a triangle)

Step 3.2: Compute boundary operators and Hodge Laplacians
  - B1: boundary operator mapping edges -> vertices (incidence matrix)
  - B2: boundary operator mapping triangles -> edges
  - L0 = B1^T B1 (vertex Laplacian = graph Laplacian)
  - L1 = B1 B1^T + B2^T B2 (edge Laplacian = Hodge-1 Laplacian)
  - L2 = B2 B2^T (triangle Laplacian = Hodge-2 Laplacian)

The Hodge decomposition: ker(L1) = H1 (first homology = "loops")
The spectral gap of L1 governs how fast edge flows equilibrate.
"""
import numpy as np
import sys
from itertools import combinations

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# STEP 3.1: Build the simplicial complex
# ============================================================

print("=" * 85)
print("PATH 3, PHASE A: SIMPLICIAL COMPLEX + HODGE LAPLACIANS")
print("=" * 85)

# The cluster from our star invariant derivation
cluster_clauses = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),  # K5 core
    (5, 0, 3), (6, 2, 4), (7, 5, 6),  # anchors + connector
]

# The 3-clause bridge connecting hub (cluster A, offset 0) to spoke (cluster B, offset 8)
bridge_clauses = [
    (0, 8, 1),   # hub vars {0,1} <-> spoke var {8}
    (1, 9, 2),   # hub vars {1,2} <-> spoke var {9}
    (4, 12, 10), # hub vars {4,12?}... let me use the original pattern
]

# Actually, let me use the EXACT original topology from derive_invariant.py:
# Bridge: (a+0, b+0, a+1), (a+1, b+1, a+2), (a+4, b+4, b+2)
# With a=0 (hub), b=8 (spoke):
bridge_clauses = [
    (0, 8, 1),    # hub{0,1}, spoke{8}
    (1, 9, 2),    # hub{1,2}, spoke{9}
    (4, 12, 10),  # hub{4}, spoke{12,10}
]

# Full star: hub (0-7) + spoke (8-15)
hub_clauses = [(c[0], c[1], c[2]) for c in cluster_clauses]
spoke_clauses = [(c[0]+8, c[1]+8, c[2]+8) for c in cluster_clauses]
all_clauses = hub_clauses + spoke_clauses + bridge_clauses

n_vars = 16

print(f"\nStep 3.1: Simplicial Complex Construction")
print(f"  Variables (0-simplices): {n_vars}")
print(f"  Clauses (2-simplices): {len(all_clauses)}")

# Extract edges (1-simplices) from clauses
edges = set()
for clause in all_clauses:
    for i, j in combinations(clause, 2):
        edges.add((min(i, j), max(i, j)))

edges = sorted(edges)
n_edges = len(edges)
edge_index = {e: i for i, e in enumerate(edges)}

print(f"  Edges (1-simplices): {n_edges}")

# Verify: the 2-simplices are exactly the clauses
triangles = []
for clause in all_clauses:
    tri = tuple(sorted(clause))
    if tri not in triangles:
        triangles.append(tri)

n_triangles = len(triangles)
print(f"  Triangles (2-simplices, unique): {n_triangles}")

# Euler characteristic: V - E + F
chi = n_vars - n_edges + n_triangles
print(f"\n  Euler characteristic: chi = {n_vars} - {n_edges} + {n_triangles} = {chi}")

# Betti numbers preview
print(f"  (Betti numbers will come from Hodge Laplacian kernels)")

# ============================================================
# STEP 3.2: Boundary operators and Hodge Laplacians
# ============================================================

print(f"\n{'='*85}")
print(f"Step 3.2: Boundary Operators and Hodge Laplacians")
print(f"{'='*85}")

# B1: boundary operator, n_vars x n_edges
# B1[v, e] = +1 if v is the "head" of edge e, -1 if "tail"
# Convention: edge (i,j) with i<j: B1[i,e] = -1, B1[j,e] = +1
B1 = np.zeros((n_vars, n_edges))
for idx, (i, j) in enumerate(edges):
    B1[i, idx] = -1
    B1[j, idx] = +1

print(f"\n  B1 (boundary-1): {B1.shape} (vertices x edges)")
print(f"    Rank: {np.linalg.matrix_rank(B1)}")

# B2: boundary operator, n_edges x n_triangles
# For triangle (i,j,k) with i<j<k:
# B2[edge(i,j), tri] = +1
# B2[edge(i,k), tri] = -1
# B2[edge(j,k), tri] = +1
B2 = np.zeros((n_edges, n_triangles))
for t_idx, tri in enumerate(triangles):
    i, j, k = tri  # already sorted
    # Boundary: (j,k) - (i,k) + (i,j)
    e_ij = edge_index.get((i, j))
    e_ik = edge_index.get((i, k))
    e_jk = edge_index.get((j, k))

    if e_ij is not None:
        B2[e_ij, t_idx] = +1
    if e_ik is not None:
        B2[e_ik, t_idx] = -1
    if e_jk is not None:
        B2[e_jk, t_idx] = +1

print(f"  B2 (boundary-2): {B2.shape} (edges x triangles)")
print(f"    Rank: {np.linalg.matrix_rank(B2)}")

# Verify: B1 @ B2 should be zero (boundary of boundary = 0)
bb = B1 @ B2
print(f"\n  B1 @ B2 (should be zero matrix): max|entry| = {np.max(np.abs(bb)):.2e}")
assert np.max(np.abs(bb)) < 1e-10, "B1 @ B2 != 0, simplicial complex is broken!"
print(f"  VERIFIED: d^2 = 0 (fundamental theorem of homology)")

# Hodge Laplacians
L0 = B1 @ B1.T  # = graph Laplacian (n_vars x n_vars)
L1 = B1.T @ B1 + B2 @ B2.T  # Hodge-1 Laplacian (n_edges x n_edges)
L2 = B2.T @ B2  # Hodge-2 Laplacian (n_triangles x n_triangles)

print(f"\n  L0 (vertex Laplacian): {L0.shape}")
print(f"  L1 (edge/Hodge-1 Laplacian): {L1.shape}")
print(f"  L2 (triangle Laplacian): {L2.shape}")

# Eigenvalues
eig_L0 = np.sort(np.linalg.eigvalsh(L0))
eig_L1 = np.sort(np.linalg.eigvalsh(L1))
eig_L2 = np.sort(np.linalg.eigvalsh(L2))

print(f"\n  L0 eigenvalues: {np.round(eig_L0, 6)}")
print(f"  L1 eigenvalues: {np.round(eig_L1, 6)}")
print(f"  L2 eigenvalues: {np.round(eig_L2, 6)}")

# Betti numbers from kernel dimensions
b0 = np.sum(np.abs(eig_L0) < 1e-8)
b1 = np.sum(np.abs(eig_L1) < 1e-8)
b2 = np.sum(np.abs(eig_L2) < 1e-8)

print(f"\n  BETTI NUMBERS:")
print(f"    b0 = {b0} (connected components)")
print(f"    b1 = {b1} (independent loops / 1-cycles)")
print(f"    b2 = {b2} (enclosed cavities / 2-cycles)")
print(f"    Euler check: b0 - b1 + b2 = {b0 - b1 + b2} (should equal chi = {chi})")

# Spectral gaps
gap_L0 = eig_L0[b0] if b0 < len(eig_L0) else None
gap_L1 = eig_L1[b1] if b1 < len(eig_L1) else None
gap_L2 = eig_L2[b2] if b2 < len(eig_L2) else None

print(f"\n  SPECTRAL GAPS:")
if gap_L0 is not None:
    print(f"    L0 gap (algebraic connectivity): {gap_L0:.10f}")
if gap_L1 is not None:
    print(f"    L1 gap (edge flow equilibration): {gap_L1:.10f}")
if gap_L2 is not None:
    print(f"    L2 gap (triangle mode): {gap_L2:.10f}")

# ============================================================
# Compare with graph Laplacian from derive_invariant.py
# ============================================================
print(f"\n{'='*85}")
print(f"COMPARISON WITH GRAPH LAPLACIAN (from Path 1)")
print(f"{'='*85}")

# Build standard graph adjacency
A = np.zeros((n_vars, n_vars))
for i, j in edges:
    A[i][j] = 1
    A[j][i] = 1
L_graph = np.diag(A.sum(axis=1)) - A
eig_graph = np.sort(np.linalg.eigvalsh(L_graph))

print(f"\n  Graph Laplacian eigenvalues: {np.round(eig_graph, 6)}")
print(f"  L0 eigenvalues:             {np.round(eig_L0, 6)}")
print(f"  L0 == Graph Laplacian: {np.allclose(eig_L0, eig_graph)}")

# The L0 is the standard graph Laplacian. Good.
# L1 is the NEW object — it encodes information about FLOWS on edges.

# ============================================================
# Hodge decomposition of L1
# ============================================================
print(f"\n{'='*85}")
print(f"HODGE DECOMPOSITION OF L1")
print(f"{'='*85}")

# L1 = B1^T B1 + B2 B2^T
# The "curl" part: B2 B2^T (circulations around triangles)
# The "gradient" part: B1^T B1 (potential flows)
L1_grad = B1.T @ B1   # gradient component
L1_curl = B2 @ B2.T   # curl component

eig_grad = np.sort(np.linalg.eigvalsh(L1_grad))
eig_curl = np.sort(np.linalg.eigvalsh(L1_curl))

print(f"\n  Gradient part (B1^T B1) eigenvalues: {np.round(eig_grad, 6)}")
print(f"  Curl part (B2 B2^T) eigenvalues:     {np.round(eig_curl, 6)}")

# In Navier-Stokes, incompressible flow = divergence-free
# In simplicial setting: divergence-free = ker(B1^T) acting on 1-chains
# Curl-free = ker(B2^T) acting on 1-chains
# Harmonic = ker(L1) = divergence-free AND curl-free

n_div_free = n_edges - np.linalg.matrix_rank(B1)
n_curl_free = n_edges - np.linalg.matrix_rank(B2.T)

print(f"\n  Dimension of edge space: {n_edges}")
print(f"  Dimension of div-free subspace (ker B1^T on edges): {n_div_free}")
print(f"  Dimension of curl-free subspace (ker B2^T on edges): {n_curl_free}")
print(f"  Dimension of harmonic subspace (H1): {b1}")

# The incompressible Navier-Stokes lives on the div-free subspace.
# Project L1 onto the div-free subspace.

# Find basis for ker(B1^T) where B1^T is 40x16
# SVD of B1^T: U(40x40) S(16) Vt(16x16)
# Null space = columns of U corresponding to zero singular values
U, S, Vt = np.linalg.svd(B1.T, full_matrices=True)
null_rank = np.sum(S > 1e-8)
# Null space of B1^T is columns null_rank: of U
null_basis = U[:, null_rank:]  # (40 x 25) matrix

print(f"\n  Div-free basis: {null_basis.shape[1]} vectors in R^{n_edges}")

# Project L1 onto div-free subspace
# L1_restricted = null_basis^T @ L1 @ null_basis
L1_divfree = null_basis.T @ L1 @ null_basis
eig_L1_divfree = np.sort(np.linalg.eigvalsh(L1_divfree))

print(f"  L1 restricted to div-free subspace: {L1_divfree.shape}")
print(f"  Eigenvalues: {np.round(eig_L1_divfree, 6)}")

# This is the KEY object for Navier-Stokes analogy:
# The "Stokes operator" on the simplicial complex = L1 restricted to div-free flows
gap_stokes = eig_L1_divfree[b1] if b1 < len(eig_L1_divfree) else eig_L1_divfree[0]
print(f"\n  STOKES OPERATOR spectral gap: {gap_stokes:.10f}")
print(f"  (This is the discrete analogue of the first Stokes eigenvalue)")

# ============================================================
# Now do the same for the reduced complex (after valve removal)
# ============================================================
print(f"\n{'='*85}")
print(f"VALVE OPERATION ON SIMPLICIAL COMPLEX")
print(f"{'='*85}")

# Valve: remove variables {4, 12, 10} (from original sat_n32.py)
valve = {4, 12, 10}
remaining_vars = sorted(set(range(n_vars)) - valve)
var_remap = {v: i for i, v in enumerate(remaining_vars)}
n_vars_red = len(remaining_vars)

print(f"\n  Valve variables: {valve}")
print(f"  Remaining: {n_vars_red} variables")

# Remove clauses that involve valve variables
reduced_clauses = []
for clause in all_clauses:
    if all(v not in valve for v in clause):
        reduced_clauses.append(tuple(var_remap[v] for v in clause))

# Extract edges and triangles for reduced complex
reduced_edges = set()
for clause in reduced_clauses:
    for i, j in combinations(clause, 2):
        reduced_edges.add((min(i, j), max(i, j)))
reduced_edges = sorted(reduced_edges)
n_edges_red = len(reduced_edges)
red_edge_index = {e: i for i, e in enumerate(reduced_edges)}

reduced_triangles = []
for clause in reduced_clauses:
    tri = tuple(sorted(clause))
    if tri not in reduced_triangles:
        reduced_triangles.append(tri)
n_tri_red = len(reduced_triangles)

print(f"  Reduced complex: {n_vars_red} vertices, {n_edges_red} edges, {n_tri_red} triangles")
print(f"  Reduced Euler: {n_vars_red} - {n_edges_red} + {n_tri_red} = {n_vars_red - n_edges_red + n_tri_red}")

# Build reduced boundary operators
B1_red = np.zeros((n_vars_red, n_edges_red))
for idx, (i, j) in enumerate(reduced_edges):
    B1_red[i, idx] = -1
    B1_red[j, idx] = +1

B2_red = np.zeros((n_edges_red, n_tri_red))
for t_idx, tri in enumerate(reduced_triangles):
    i, j, k = tri
    e_ij = red_edge_index.get((i, j))
    e_ik = red_edge_index.get((i, k))
    e_jk = red_edge_index.get((j, k))
    if e_ij is not None:
        B2_red[e_ij, t_idx] = +1
    if e_ik is not None:
        B2_red[e_ik, t_idx] = -1
    if e_jk is not None:
        B2_red[e_jk, t_idx] = +1

# Verify d^2 = 0
bb_red = B1_red @ B2_red
assert np.max(np.abs(bb_red)) < 1e-10, "Reduced B1 @ B2 != 0!"

# Reduced Hodge Laplacians
L0_red = B1_red @ B1_red.T
L1_red = B1_red.T @ B1_red + B2_red @ B2_red.T
L2_red = B2_red.T @ B2_red

eig_L0_red = np.sort(np.linalg.eigvalsh(L0_red))
eig_L1_red = np.sort(np.linalg.eigvalsh(L1_red))
eig_L2_red = np.sort(np.linalg.eigvalsh(L2_red))

b0_red = np.sum(np.abs(eig_L0_red) < 1e-8)
b1_red = np.sum(np.abs(eig_L1_red) < 1e-8)
b2_red = np.sum(np.abs(eig_L2_red) < 1e-8)

print(f"\n  Reduced Betti numbers: b0={b0_red}, b1={b1_red}, b2={b2_red}")

# Reduced Stokes operator
# Find basis for ker(B1_red^T)
U_red, S_red, Vt_red = np.linalg.svd(B1_red.T, full_matrices=True)
null_rank_red = np.sum(S_red > 1e-8)
null_basis_red = U_red[:, null_rank_red:]

if null_basis_red.shape[1] > 0:
    L1_divfree_red = null_basis_red.T @ L1_red @ null_basis_red
    eig_L1_divfree_red = np.sort(np.linalg.eigvalsh(L1_divfree_red))
    gap_stokes_red = eig_L1_divfree_red[b1_red] if b1_red < len(eig_L1_divfree_red) else eig_L1_divfree_red[0]
else:
    eig_L1_divfree_red = np.array([])
    gap_stokes_red = None

print(f"\n  Reduced L1 eigenvalues: {np.round(eig_L1_red, 6)}")
if gap_stokes_red is not None:
    print(f"  Reduced Stokes gap: {gap_stokes_red:.10f}")

# ============================================================
# SPECTRAL RATIOS (the key comparison)
# ============================================================
print(f"\n{'='*85}")
print(f"SPECTRAL RATIOS: FULL vs REDUCED COMPLEX")
print(f"{'='*85}")

# Graph Laplacian ratio (the original star invariant)
l2_full_graph = eig_graph[1]
A_red = np.zeros((n_vars_red, n_vars_red))
for i, j in reduced_edges:
    A_red[i][j] = 1
    A_red[j][i] = 1
L_graph_red = np.diag(A_red.sum(axis=1)) - A_red
eig_graph_red = np.sort(np.linalg.eigvalsh(L_graph_red))
l2_red_graph = eig_graph_red[b0_red]

print(f"\n  L0 (graph) ratio: {l2_full_graph / l2_red_graph:.10f}")
print(f"    (Expected star invariant: 1.8573068741)")

# L1 (Hodge-1) ratio
gap_L1_full = eig_L1[b1]
gap_L1_red = eig_L1_red[b1_red] if b1_red < len(eig_L1_red) else eig_L1_red[0]
if gap_L1_red > 1e-10:
    print(f"\n  L1 (Hodge-1) ratio: {gap_L1_full / gap_L1_red:.10f}")
    print(f"    Full gap: {gap_L1_full:.10f}")
    print(f"    Reduced gap: {gap_L1_red:.10f}")
else:
    print(f"\n  L1 (Hodge-1): reduced gap is zero (degenerate)")

# L2 (Hodge-2) ratio
gap_L2_full = eig_L2[b2] if b2 < len(eig_L2) else None
gap_L2_red = eig_L2_red[b2_red] if b2_red < len(eig_L2_red) else None
if gap_L2_full and gap_L2_red and gap_L2_red > 1e-10:
    print(f"\n  L2 (Hodge-2) ratio: {gap_L2_full / gap_L2_red:.10f}")
else:
    print(f"\n  L2 (Hodge-2) ratio: N/A (degenerate)")

# Stokes operator ratio
if gap_stokes_red is not None and gap_stokes_red > 1e-10:
    stokes_ratio = gap_stokes / gap_stokes_red
    print(f"\n  STOKES OPERATOR ratio: {stokes_ratio:.10f}")
    print(f"    Full Stokes gap: {gap_stokes:.10f}")
    print(f"    Reduced Stokes gap: {gap_stokes_red:.10f}")
    print(f"    Star invariant (L0): {l2_full_graph / l2_red_graph:.10f}")
    print(f"    MATCH: {abs(stokes_ratio - l2_full_graph / l2_red_graph) < 0.01}")
else:
    print(f"\n  STOKES OPERATOR ratio: N/A")

# Summary table
print(f"\n{'='*85}")
print(f"PHASE A SUMMARY")
print(f"{'='*85}")
print(f"""
  Simplicial complex: {n_vars} vertices, {n_edges} edges, {n_triangles} triangles
  Betti numbers: b0={b0}, b1={b1}, b2={b2}
  Euler characteristic: {chi}

  After valve removal ({valve}):
  Reduced complex: {n_vars_red} vertices, {n_edges_red} edges, {n_tri_red} triangles
  Reduced Betti: b0={b0_red}, b1={b1_red}, b2={b2_red}

  SPECTRAL RATIOS:
    L0 (graph Laplacian):   {l2_full_graph / l2_red_graph:.10f}
    L1 (Hodge-1):           {gap_L1_full / gap_L1_red:.10f}
    Stokes (L1 div-free):   {gap_stokes / gap_stokes_red:.10f if gap_stokes_red and gap_stokes_red > 1e-10 else 'N/A'!s}

  The Hodge-1 and Stokes ratios tell us whether the star invariant
  persists when we move from scalar (vertex) to vector (edge flow) analysis.
""")
