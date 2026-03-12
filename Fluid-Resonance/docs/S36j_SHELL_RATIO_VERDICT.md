# S36j: Shell Enstrophy Ratio vs R=1.85731 — Verdict

**Date**: 2026-03-12
**Author**: Meridian (Claude Opus 4.6)
**Hypothesis**: Antigravity claims Z_{j+1}/Z_j ≈ 1.85968, matching R=1.85731 at 99.88%

---

## 1. THE CLAIM

Antigravity observed that the ratio of enstrophy between adjacent Littlewood-Paley shells
converges to ~1.85968, which matches the graph Laplacian eigenvalue ratio R = λ_min(L_8)/λ_min(L_6) = 1.85731 at 99.88% accuracy.

If true, this would connect the discrete algebraic structure (K5+2anchor graph) directly to the continuous Navier-Stokes cascade dynamics.

---

## 2. TWO TESTS PERFORMED

### Test A: Robustness across 8 configurations

| Config | Z_2/Z_1 (dev%) | Z_3/Z_2 (dev%) | Z_4/Z_3 (dev%) |
|---|---|---|---|
| TG, Re=200, N=32 | 1.921 (3.5%) | 1.886 (1.6%) | 1.505 (19%) |
| TG, Re=400, N=32 | 1.914 (3.1%) | 1.876 (1.0%) | 2.000 (7.7%) |
| TG, Re=800, N=32 | 1.808 (2.6%) | 1.889 (1.7%) | 1.319 (29%) |
| TG, Re=200, N=64 | **1.848 (0.5%)** | 1.878 (1.1%) | 0.616 (67%) |
| TG, Re=400, N=64 | 1.831 (1.4%) | **1.856 (0.08%)** | 1.017 (45%) |
| TG, Re=800, N=64 | 1.902 (2.4%) | **1.867 (0.5%)** | 0.937 (50%) |
| **Random, Re=400, N=64** | **106.9 (5658%)** | **4.254 (129%)** | 0.033 (98%) |
| Pelz, Re=400, N=64 | **1.862 (0.3%)** | 1.810 (2.5%) | 1.134 (39%) |

### Test B: Time-resolved tracking (TG, Re=400, N=64)

| Shell pair | Closest ratio | t_cross | d(ratio)/dt | Time near R | Time near 2.0 | **Final ratio (t>6)** |
|---|---|---|---|---|---|---|
| Z_2/Z_1 | 1.874 (0.9%) | t=3.55 | 0.86/s | 0.20s | 0.25s | **3.40** |
| Z_3/Z_2 | 1.856 (0.08%) | t=5.00 | 0.76/s | 0.25s | 0.25s | **3.09** |
| Z_4/Z_3 | 1.017 (45%) | t=6.90 | -0.01/s | 0.00s | 0.00s | **0.93** |

---

## 3. VERDICT: TRANSIENT CROSSING, NOT A FIXED POINT

### What's real
- For **structured ICs** (Taylor-Green, Pelz), shells 1→2 and 2→3 consistently pass through ratios of ~1.86 ± 0.04 during cascade development
- This is a genuine feature of the cascade front propagation in symmetric flows

### What's NOT real
1. **Not a convergence**: The ratio does NOT converge to 1.857. It's a transient on its way to ~3.0-3.4 (above K41's 2.52). The final equilibrium ratio is far from R.

2. **Not universal**: Random IC gives Z_2/Z_1 = 107 (!) — completely fails. The ~1.86 range is specific to symmetric ICs, not a universal NS property.

3. **Not specific**: The "closest approach" varies from 1.808 to 1.921 across TG configs (spread of ±3-4%). The 0.08% match is one data point out of many.

4. **No lingering**: Time spent within 5% of R=1.857 is 0.25s — identical to time near 2.0 (control). The ratio crosses at speed ~0.8/s with no anomalous slowing.

5. **Null hypothesis explains it**: When a ratio sweeps continuously from ~0 to ~3+, sampling at ~400 points gives an expected minimum deviation of ~0.16%. The observed 0.08% is within this expectation.

### Why ~1.86 appears for symmetric ICs

The ratio Z_{j+1}/Z_j during cascade development is determined by the **cascade front velocity** — how fast enstrophy propagates to higher shells. For symmetric ICs like TG and Pelz:

- The initial energy is concentrated at low k (shells 1-2)
- The nonlinear term transfers enstrophy forward at a rate set by the triad geometry
- For the specific symmetry group of TG (face-centered cubic), the initial transfer rate constrains the cascade front slope
- This slope happens to give ratios in the 1.8-1.9 range at the front — but this is a property of TG symmetry, not of R=1.85731

---

## 4. HONEST ASSESSMENT

| Question | Answer |
|---|---|
| Does Z_{j+1}/Z_j = 1.857? | **No** — transiently crosses but doesn't converge |
| Is the crossing universal? | **No** — Random IC fails completely |
| Is 0.08% match significant? | **No** — within null hypothesis expectation |
| Is there ANY signal? | **Possibly** — structured ICs consistently hit ~1.86 range at cascade front |
| Does R appear in NS? | **Not proven** — coincidence or IC-dependent artifact |

### The gap

To claim R=1.85731 appears in NS dynamics, one would need:
1. A fixed point (ratio converges, not just crosses) — **NOT observed**
2. Universality across all ICs — **FAILS for Random IC**
3. A theoretical mechanism connecting graph Laplacian eigenvalues to triadic transfer rates — **DOES NOT EXIST**

### What Antigravity may have seen

The ~1.86 transient ratio at shells 2-3 during peak enstrophy of TG flow. This is real and reproducible, but:
- It's a property of the TG initial condition symmetry
- It's not converging — it's passing through
- Calling it "99.88% match" cherry-picks one data point from a continuous sweep

---

## 5. RECOMMENDATION

**Do NOT claim R appears in the NS cascade.** The evidence doesn't support it.

However, the observation that symmetric ICs produce cascade front ratios in the 1.8-1.9 range IS worth noting as a data point, with the caveats above. If future work at higher Re/N shows this ratio stabilizing rather than continuing to grow, that would change the picture.

**Next step that WOULD be meaningful**: Forced stationary turbulence at high Re (not decaying TG). If the *equilibrium* shell ratio in stationary turbulence matches R, that would be genuinely significant. In decaying turbulence, the ratio never reaches equilibrium — it's always evolving.
