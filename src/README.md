# Fluid Resonance — Helical Structure and Enstrophy Constraints in 3D Navier-Stokes

Research into the geometric mechanisms constraining enstrophy growth in three-dimensional incompressible Navier-Stokes flow on the torus. Combines spectral graph theory, helical decomposition, and Fano plane topology to quantify how the Leray projection suppresses nonlinear interactions.

## Key Results

| Result | Status | Reference |
|--------|--------|-----------|
| Cross-helical Leray suppression: alpha = 1 - ln(2) | **PROVEN** | Theorem 12.2, `the_shape.tex` Section 3 |
| Solenoidal coupling: sin^2(theta)/4 | **DNS VERIFIED** | RMS error 0.0017, 30,000 triads |
| Fano plane PG(2,2) governs triadic topology at low k | **PROVEN** | Theorem 14.1 |
| [7,4,3] Hamming code structure of sign frustration | **PROVEN** | Theorem 14.2 |
| Berry holonomy vanishes for NS triads (k+p+q=0) | **PROVEN** | Theorem 15.1, coplanarity proof |
| R < 2 for pure K_n cores (n >= 5, w >= 4) | **PROVEN** | Theorem 9.1, algebraic: -16 < 16 |
| R < 2 for arbitrary anchors | **REFUTED** | K4 + 8 anchors: R = 2.014 |
| Quadrature C-F bridge: Re=1600 gives Holder-1/2 | **OBSERVED** | `scripts/wip/constantin_fefferman_bridge.py` |
| Holder-beta with beta < 1/2 suffices for regularity | **OPEN** | Beirao da Veiga 2019 |
| 22 investigated approaches ruled out | **DOCUMENTED** | `NEGATIVE_RESULTS.md` |

## The Spectral Invariant

```
R = lambda_min(L_8x8) / lambda_min(L_6x6) = 1.8573068741389058
```

- **L_8x8**: Grounded spoke Laplacian of K5+2anchor star cluster
- **L_6x6**: Reduced Laplacian after valve removal
- Defining polynomials P7 (degree 7) and P5 (degree 5), both irreducible over Q

## The Leray Suppression

For cross-helical Lamb vector interactions in the helical basis:

```
Per-triad:  alpha(theta, rho) = 1 - (1+rho)^2(1+cos theta) / [(1+rho^2+2*rho*cos theta)(3-cos theta)]
Same-shell: alpha(theta, rho=1) = (1-cos theta) / (3-cos theta)
Isotropic:  <alpha> = 1 - ln(2) = 0.30685...
```

69.3% of the cross-helical Lamb vector is irrotational and removed by the pressure term.

## Quick Verification

```python
import numpy as np

# Spectral invariant R
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

# Leray suppression isotropic average
from scipy.integrate import quad
alpha_iso = quad(lambda x: (1-x)/(3-x), -1, 1)[0] / 2
print(f"<alpha> = {alpha_iso:.10f}")  # 0.3068528194 = 1 - ln(2)
```

## Repository Structure

```
├── the_shape.tex                    # Paper: helical decomposition + Leray suppression
├── FORMAL_PROOFS.md                 # 15 theorems, gap analysis, references
├── NEGATIVE_RESULTS.md              # 22 confirmed dead ends with refutations
├── SPECTRAL_INVARIANT_RESULTS.md    # Numerical verification archive
│
├── scripts/
│   ├── audit/                       # Automated claim verification pipeline
│   │   ├── tex_parser.py            # Extract claims from .tex
│   │   ├── claim_verifier.py        # Verify against domain rules
│   │   ├── domain_rules.py          # Mathematical consistency rules
│   │   └── report_generator.py      # Audit report output
│   ├── supporting/                  # Scale/topology/Hodge verification
│   └── wip/                         # ~100 computational scripts
│       ├── leray_analytical_formula.py
│       ├── verify_alpha_iso_derivation.py
│       ├── berry_frustration_g1.py
│       ├── check_fano_frustration.py
│       ├── constantin_fefferman_bridge.py
│       ├── mueller_q_prediction.py
│       └── ...
│
├── derive_invariant.py              # Derives R from first principles
├── factor_polys.py                  # Proves P7, P5 irreducible over Q
└── proof_R_less_than_2.py           # Algebraic proof: R < 2
```

## Key References

1. Biferale & Titi 2013 — Same-helicity global regularity (*J. Stat. Phys.* 151)
2. Sahoo & Biferale 2017 — Cross-helical triads drive the forward cascade (*Eur. Phys. J. E* 42)
3. Constantin & Fefferman 1993 — Vorticity direction Holder-1/2 implies regularity (*Indiana Math J.* 42)
4. Miller 2024 — Model equation globally regular (arXiv:2407.02691)
5. Lu & Doering 2019 — Enstrophy exponent 3/2 is sharp (arXiv:1909.00041)
6. Buaria et al. 2024 — Anti-twist regularization (*Science Advances* 10)
7. Bredberg, Keeler, Maloney & Strominger 2012 — NS from Einstein (arXiv:1101.2451)
8. Beirao da Veiga 2019 — Holder-beta < 1/2 open problem (*DCDS-S* 12)
9. Araki 2016 — Arnold sectional curvature for CHW modes (arXiv:1608.05154)

## Requirements

Python 3.8+, `numpy`, `scipy`, `sympy`. No other dependencies.

## Authors

- **Brendan Siche** — Research direction, experimental design
- Technical assistance from generative AI (Claude, Gemini) for computation, verification, and adversarial auditing

## License

This research is provided for academic review and independent verification.
