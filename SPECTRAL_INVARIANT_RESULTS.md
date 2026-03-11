# Spectral Star Invariant — Computational Results

**Date:** 2026-03-11
**Computed by:** Claude (Opus 4.6), independent audit
**Scripts:** `derive_invariant.py`, `factor_polys.py`, `path1_generalization.py`, `path2_random_sat.py`

---

## 1. The Star Invariant (Exact Result)

For a star graph of K5 clusters with 2 anchors, connector, and 3-clause bridges:

```
R = lambda_min(L_8x8) / lambda_min(L_6x6) = 1.8573068741389058...
```

Where:
- **L_8x8** is an integer grounded spoke Laplacian (8x8)
- **L_6x6** is the reduced grounded spoke Laplacian after valve removal (6x6)
- Both are derived from the cluster's internal Laplacian + bridge grounding

### Key Properties
- **Scale-invariant**: Identical from N=32 to N=8192 (star topology)
- **Topology-dependent**: Chain -> 1.636, Binary tree -> 1.327 (different limits)
- **Symmetry-required**: Asymmetric clusters scatter to [1.13, 2.12]
- **Best rational approximation**: 13/7 (within 0.009%)
- **Algebraic degree**: R is degree <= 35 (ratio of irreducible degree 7 and degree 5 algebraic numbers); no simple radical expression

### Characteristic Polynomials (irreducible over Q)

**Numerator eigenvalue (8x8):**
```
P7(t) = t^7 - 33t^6 + 443t^5 - 3097t^4 + 11948t^3 - 24634t^2 + 23588t - 6916
```

**Denominator eigenvalue (6x6):**
```
P5(t) = t^5 - 17t^4 + 104t^3 - 270t^2 + 260t - 52
```

Both confirmed irreducible (no rational roots, no quad/cubic factorization).

### Integer Laplacian Matrices

**8x8 grounded spoke:**
```
[[ 7 -1 -1 -1 -1 -1  0  0]
 [-1  6 -1 -1 -1  0  0  0]
 [-1 -1  6 -1 -1  0 -1  0]
 [-1 -1 -1  5 -1 -1  0  0]
 [-1 -1 -1 -1  6  0 -1  0]
 [-1  0  0 -1  0  4 -1 -1]
 [ 0  0 -1  0 -1 -1  4 -1]
 [ 0  0  0  0  0 -1 -1  2]]
```

**6x6 reduced (after valve removal):**
```
[[ 5 -1 -1 -1  0  0]
 [-1  4 -1  0  0  0]
 [-1 -1  3 -1  0  0]
 [-1  0 -1  4 -1 -1]
 [ 0  0  0 -1  2 -1]
 [ 0  0  0 -1 -1  2]]
```

---

## 2. Path 1 Results: Generalization (88 configurations)

Scanned: K3-K7 cores x 0-3 anchors x 1-5 bridge widths.

### Key Findings

1. **Bridge widths 1-2 are universally degenerate** (valve disconnects spoke)
2. **The invariant depends on bridge width:**

| Bridge Width | Ratio Range   | Mean  |
|-------------|---------------|-------|
| 3           | [1.87, 3.18]  | 2.39  |
| 4           | [1.00, 1.91]  | 1.69  |
| 5           | [1.00, 1.51]  | 1.31  |

3. **Wider bridges -> lower invariant** (channel width effect)
4. **The original 1.857 is geometry-specific** — it requires the exact asymmetric bridge clause pattern of the K5+2anchor cluster
5. **Configurations closest to 1.857:** K6/a2/b4 (1.866), K7/a2/b4 (1.846), K3/a1/b3 (1.872)

### Base Irrationals by Cluster Geometry

| Cluster | Irrational | Note |
|---------|-----------|------|
| K3 + 2 anchors | sqrt(13) | Polynomial factors into quadratic with sqrt(13) |
| K5 + 2 anchors | N/A | P7 and P5 are irreducible over Q (Theorems 3.1-3.2); eigenvalues are degree 7 and 5 algebraic numbers with no simple radical expression |
| K6 + 2 anchors | sqrt(8) | Polynomial factors into quadratic with sqrt(8) |
| K7 + 2 anchors | sqrt(13) | Polynomial factors into quadratic with sqrt(13) |
| K_n + 0 anchors | All integer | Characteristic polynomials split completely over Z |

---

## 3. Path 2 Results: Random 3-SAT (NEGATIVE)

Tested: N={50, 100, 200}, alpha={3.0, 4.0, 4.267, 4.5, 5.0}, 20 instances each.

### Key Findings

1. **No special behavior at alpha=4.267 (critical threshold)**
   - Mean deviation from 1.857 at critical: 0.528
   - Mean deviation from 1.857 off-critical: 0.443
   - Critical is FURTHER from target

2. **Enormous variance** — standard deviations 30-80% of mean
3. **Size scaling diverges** — at N=200, ratios drop to ~1.2-1.5
4. **Recursive decomposition**: some sub-problems hit near 1.857 but with no consistent pattern

### Verdict

The star invariant does NOT appear to govern natural bottlenecks in random 3-SAT instances. The spectral ratio at VIG bisection points shows no concentration around 1.857 and no special behavior at the critical threshold.

---

## 4. Path 3 Phase A: Simplicial Hodge Laplacians

**Script:** `path3_phase_a.py`

### Simplicial Complex (2-cluster system: hub + spoke, 3-clause bridge)

| | Full | Reduced (valve {4,10,12}) |
|---|---|---|
| Vertices (0-simplices) | 16 | 13 |
| Edges (1-simplices) | 40 | 21 |
| Triangles (2-simplices) | 19 | 8 |
| Euler characteristic | -5 | 0 |
| b0 (components) | 1 | 1 |
| b1 (loops) | **6** | **1** |
| b2 (cavities) | 0 | 0 |

**Valve kills 5 of 6 independent loops** (circulation pathways).

### Boundary Operators

```
B1: (16 x 40), rank 15    -- vertices -> edges
B2: (40 x 19), rank 19    -- edges -> triangles
B1 @ B2 = 0               -- d^2 = 0 verified
```

### Hodge Laplacian Eigenvalues

**L0 (vertex/graph Laplacian, 16x16):**
```
[0, 0.672, 1.448, 2.437, 4.043, 4.105, 4.292, 4.646,
 5.952, 6.250, 6.789, 7.000, 7.047, 7.627, 8.394, 9.297]
```

**L1 (edge/Hodge-1, 40x40) — restricted to div-free subspace (25x25):**
```
[0(x6), 0.905, 0.940, 1.296, 1.317, 1.748, 2.000, 2.262,
 2.536, 2.576, 3.000(x2), 3.162, 3.477, 4.000, 4.356, 4.521, 4.575, 5.524, 5.805]
```

**L2 (triangle, 19x19):**
```
[0.905, 0.940, 1.296, 1.317, 1.748, 2.000, 2.262, 2.536,
 2.576, 3.000(x2), 3.162, 3.477, 4.000, 4.356, 4.521, 4.575, 5.524, 5.805]
```

### Spectral Ratios (full / reduced)

| Operator | Full gap | Reduced gap | Ratio | Interpretation |
|---|---|---|---|---|
| L0 (graph) | 0.6724 | 0.2571 | **2.615** | Vertex connectivity drops |
| L1 (Hodge-1) | 0.6724 | 0.2571 | **2.615** | Inherits from L0 gradient |
| Stokes (L1 div-free) | 0.9047 | 1.5188 | **0.596** | Flow equilibration INCREASES |

### Key Finding

The **Stokes operator ratio is inverted** (< 1). Removing valve variables increases the Stokes gap — remaining div-free flows equilibrate faster when circulation pathways are destroyed. The vertex connectivity (L0) and flow equilibration (Stokes) move in **opposite directions** under the valve operation.

---

## 5. Path 3 Phases B-C: Discrete NS + Energy Inequality

**Scripts:** `path3_phase_b.py`, `path3_phase_c.py`

### Discrete Flow Analysis (Phase B)

The Fiedler flow (slowest-decaying div-free mode) lives on the hub cluster's K5 core.

**Identity confirmed:** On div-free modes, Stokes eigenvalue = enstrophy of eigenmode.
(`<f, L1 f> = |curl(f)|^2` when `B1 f = 0`)

| Quantity | Full | Reduced | Effect |
|---|---|---|---|
| Div-free modes | 25 | 9 | -16 modes |
| Harmonic modes (b1) | 6 | 1 | -5 loops destroyed |
| Stokes gap | 0.9047 | 1.5188 | INCREASES |
| Reynolds number | 1.051 | 0.811 | DROPS |
| Energy decay time | 0.553 | 0.329 | 1.68x FASTER |
| Ladyzhenskaya C | 0.582 | 0.549 | Slightly smaller |
| Critical Re | 3.260 | 2.184 | Lower threshold |

### Hodge Decomposition Theorem (Confirmed)

```
L1 spectrum = (L0 non-zero eigenvalues) UNION (L2 eigenvalues) UNION (b1 zeros)
```

Verified numerically: the 34 non-zero L1 eigenvalues are exactly the union of 15 L0 non-zero eigenvalues and 19 L2 eigenvalues. This is the discrete Hodge theorem.

### R = 1.857 in the Hodge Spectrum

19 eigenvalue pairs across L0/L1/L2/Stokes with ratio within 0.01 of 1.857:
- **L0 internal:** 7.627 / 4.105 = 1.858
- **L1 internal:** 1.748 / 0.940 = 1.859, 4.521 / 2.437 = 1.855, 8.394 / 4.521 = 1.857
- **L2 internal:** 1.748 / 0.940 = 1.859
- **Cross-spectrum:** 9.297 / 5.000 = 1.859 (L0_full / L0_red)

R ~ 1.857 appears as an internal spectral ratio within L0, L1, and L2, but NOT as the gap ratio.

### Valve = Topological Regularization

| Level | Ratio (full/red) | Direction |
|---|---|---|
| Vertex connectivity (L0 gap) | 2.615 | WEAKENS |
| Flow dissipation (Stokes gap) | 0.596 | STRENGTHENS |
| Topology (b1) | 6 → 1 | SIMPLIFIES |
| Euler characteristic | -5 → 0 | REGULARIZES |

**NS analogy:** Valve removal destroys circulation modes, increases minimum enstrophy, reduces effective Reynolds number. The complex becomes "more laminar."

---

## 6. Publishable Conclusions

### What IS established:
- R = 1.8573... is an exact, scale-invariant constant of star-cluster graph families
- It arises from the ratio of smallest eigenvalues of two integer Laplacian matrices
- It requires symmetric K5-type clusters and specific bridge geometry
- Both defining polynomials are irreducible over Q
- The invariant is topology-dependent (star vs chain vs tree give different limits)
- The Hodge-1 spectrum of the clause complex satisfies L1 = L0_nz UNION L2 (discrete Hodge theorem)
- R ~ 1.857 appears as internal spectral ratios within L0, L1, and L2
- Valve removal has DUAL effect: weakens vertex connectivity but strengthens flow dissipation
- Valve operation is a topological regularization: b1 drops, Euler char regularizes, Re drops
- **R < 2 for all pure K_n cores (0 anchors) with n >= 5 and bridge width >= 4** (Theorem 9.1, PROVED 2026-03-11). Exact closed forms: lambda_min(L_eff) = (n+2-sqrt(n^2+4n-28))/2, lambda_min(L_red) = (n-sqrt(n^2-16))/2. Proof: inequality reduces to -16 < 16.
- R is monotonically decreasing in bridge width (verified across 432 configuration pairs)
- The R < 2 bound is tight: R(n) -> 2 as n -> infinity, with gap 32/(n^2+2n)

### What is NOT established:
- Connection to 3-SAT phase transition at alpha = 4.267 (Path 2: negative result)
- Universal scaling constant across random SAT instances
- Direct quantitative bridge to Navier-Stokes regularity (Path 3: qualitative only)
- R does NOT appear as a Hodge gap ratio (only as internal eigenvalue pairs)
- **Original Conjecture 9.1 (R < 2 for arbitrary anchors) is REFUTED** (K4+8 anchors, w=4 gives R = 2.014)

### Recommended paper focus:
**Pure spectral graph theory** (Path 1) with a **Hodge-theoretic extension** (Path 3):
1. A new family of exact algebraic invariants for clustered star graphs
2. The valve operation as topological regularization (b1 reduction, Euler normalization)
3. The duality between vertex connectivity and flow dissipation under simplicial surgery
