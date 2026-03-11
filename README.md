# Fluid Resonance — Spectral Invariants of Star-Cluster Graphs

This repository provides computational tools and formal proofs for a family of exact algebraic invariants arising from grounded Laplacian matrices of clustered star graphs. The central result is a geometry-specific spectral constant **R = 1.85731** that governs the ratio of minimum eigenvalues under simplicial surgery (valve removal).

## Scientific Status

| Claim | Status | Evidence |
|-------|--------|----------|
| R is an exact algebraic invariant of star-cluster graphs | **PROVEN** | Irreducible polynomials P7, P5 over Q (Theorems 3.1, 3.2) |
| R is scale-invariant (N=2 to N=8192) | **PROVEN** | Path 1 generalization (Theorem 4.1) |
| R is topology-dependent (star/chain/tree) | **PROVEN** | Fiedler's extremal result (Theorem 5.1) |
| Valve removal has dual effect (weakens connectivity, strengthens dissipation) | **VERIFIED** | Discrete Hodge decomposition (Theorems 8.1-8.3) |
| R < 2.0 for pure K_n cores (0 anchors, n >= 5, w >= 4) | **PROVEN** | Exact closed-form eigenvalues + algebraic inequality → -16 < 16 (Theorem 9.1) |
| R < 2.0 for all star-cluster systems (arbitrary anchors) | **REFUTED** | K4 + 8 anchors at w=4: R = 2.014 (counterexample to original Conjecture 9.1) |
| R is monotonically decreasing in bridge width | **VERIFIED** | 432/432 (core, anchor) pairs confirmed perfectly monotone |
| Star topology is asymptotic limit of vortex stretching | **CONJECTURE** | Supported by CKN dimensional constraint + Fiedler extremal result (Claim 7.1) |
| R bounds enstrophy cascade, preventing NS blow-up | **CONJECTURE** | Would follow if Claim 7.1 proven + pure-core model justified |
| R appears in random 3-SAT phase transitions | **DISPROVEN** | 300 instances tested, no concentration at alpha=4.267 (Path 2) |

## The Constant

```
R = lambda_min(L_8x8) / lambda_min(L_6x6) = 1.8573068741389058
```

- **L_8x8**: 8x8 integer grounded spoke Laplacian of K5+2anchor cluster
- **L_6x6**: 6x6 reduced grounded spoke Laplacian after valve removal
- Defining polynomials: P7 (degree 7) and P5 (degree 5), both irreducible over Q
- Best rational approximation: 13/7 (error < 0.009%)

## Repository Contents

### Formal Documents
- `FORMAL_PROOFS.md` — 12 proven theorems (including Theorem 9.1), open conjectures, full references
- `SIGNAL_6_SUMMARY.md` — Adversarial audit summary (5-face assessment)
- `SPECTRAL_INVARIANT_RESULTS.md` — Complete numerical results archive
- `INCOMPRESSIBLE_STILLNESS_PAPER.md` — Proof strategy for NS regularity (conditional)
- `NAVIER_STOKES_VERIFICATION.md` — Independent verification guide
- `RESONANCE_STATE.json` — Machine-readable canonical data

### Computational Scripts (Python, requires NumPy + SymPy)
- `derive_invariant.py` — Derives R from first principles (integer Laplacians)
- `factor_polys.py` — Proves irreducibility of P7 and P5 over Q
- `path1_generalization.py` — Tests 88 cluster configurations (scale/topology dependence)
- `conjecture91_sweep.py` — 4,128-config parametric sweep (K3-K100, 0-50 anchors, w=4-15)
- `conjecture91_boundary.py` — Boundary analysis + closed-form eigenvalue derivation
- `proof_R_less_than_2.py` — Self-contained algebraic proof of Theorem 9.1
- `path2_random_sat.py` — Random 3-SAT audit (negative result)
- `path3_phase_a.py` — Simplicial Hodge decomposition
- `path3_phase_b.py` — Discrete NS flow analysis
- `path3_phase_c.py` — Hodge spectrum cross-validation

### Computational Scripts (JavaScript, standalone)
- `spectral_invariant_verifier.js` — Rebuilds Laplacian matrices, computes R
- `FLUID_SIMULATOR_HODGE.js` — Full Hodge decomposition (boundary operators, Betti numbers)
- `SAT_resonance_engine.js` — DPLL SAT solver + VIG spectral gap analysis
- `GUARDIAN_STILLNESS.js` — Stability monitor (R < 2.0 threshold)
- `linalg_utils.js` — QR eigenvalue solver (Householder + Wilkinson shifts)
- `verify_resonance.js` — Cross-validates JS against Python benchmarks
- `verify_hodge.js` — Cross-validates Hodge decomposition against Python

### Submission
- `spectral_invariants_flows.tex` — LaTeX source
- `Navier_Stokes_Submission.pdf` — Typeset manuscript

## Verification

Independent verification requires only eigenvalue computation on two explicit integer matrices:

```python
import numpy as np
L8 = np.array([[ 7,-1,-1,-1,-1,-1, 0, 0],
               [-1, 6,-1,-1,-1, 0, 0, 0],
               [-1,-1, 6,-1,-1, 0,-1, 0],
               [-1,-1,-1, 5,-1,-1, 0, 0],
               [-1,-1,-1,-1, 6, 0,-1, 0],
               [-1, 0, 0,-1, 0, 4,-1,-1],
               [ 0, 0,-1, 0,-1,-1, 4,-1],
               [ 0, 0, 0, 0, 0,-1,-1, 2]])
L6 = np.array([[ 5,-1,-1,-1, 0, 0],
               [-1, 4,-1, 0, 0, 0],
               [-1,-1, 3,-1, 0, 0],
               [-1, 0,-1, 4,-1,-1],
               [ 0, 0, 0,-1, 2,-1],
               [ 0, 0, 0,-1,-1, 2]])
R = min(np.linalg.eigvalsh(L8)) / min(np.linalg.eigvalsh(L6))
print(f"R = {R:.16f}")  # 1.8573068741389058
```

## Authors

- **Brendan Siche** — Research design, submission
- Technical assistance from generative AI tools (Claude, Gemini) for numerical verification, adversarial auditing, and typesetting. See manuscript disclosure section for details.

## License

This research is provided for academic review and independent verification.
