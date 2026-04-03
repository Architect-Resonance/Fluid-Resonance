"""
DEPLETION APPROACH — DOCUMENTED NEGATIVE RESULT

Summary of Biot-Savart stretching computations (v1 + v2).
This file records the findings so future sessions don't repeat this work.

Date: 2026-03-11
Author: Claude (Opus 4.6) / Meridian

HYPOTHESIS TESTED:
    If vorticity directions span S^2 (Lei et al. condition, arXiv 2501.08976),
    then the total enstrophy production rate Z_stretch = sum gamma_i^2 (xi_i . S_i . xi_i)
    is depleted by a universal factor c < 1 relative to the unconstrained maximum.

RESULT: HYPOTHESIS FALSE.

EVIDENCE:

1. OPTIMAL DIRECTIONS ALREADY SPAN S^2:
   In 50/50 trials (n=12 filaments in B(0,0.1)), the self-consistent
   optimal directions (each filament picks eigenvector of max eigenvalue
   of its strain tensor, iterated to approximate fixed point) ALWAYS
   satisfy the S^2-spanning condition (min max|xi x e| > 0.5 for 50
   random unit vectors e).

   This means Lei et al.'s constraint is NOT binding:
   the worst-case directions are ALREADY isotropic.

2. NO UNIVERSAL DEPLETION FACTOR:
   |isotropic|/|optimal| for n = 6, 10, 15, 20:
   2.34, 0.66, 0.77, 0.84
   No convergence, no universal constant.

3. CANCELLATION RATIOS IDENTICAL:
   optimal: 0.39 +/- 0.22
   isotropic: 0.40 +/- 0.29
   random: 0.32 +/- 0.24
   No statistically significant difference.

4. SELF-CONSISTENT ITERATION DOESN'T CONVERGE:
   The map (directions -> strain -> optimal response directions) has
   no stable fixed point. Oscillates indefinitely with direction
   changes ~1.2-1.4 per iteration.

5. STRETCHING DOMINATED BY POSITIONS, NOT DIRECTIONS:
   Net stretching std (~8000-11000) >> mean (~1000).
   The sign of total stretching depends more on WHERE filaments are
   than WHICH WAY they point.

WHY THIS MATTERS:
    The depletion of nonlinearity (Constantin 1994, Hou-Li 2006) is real
    in DNS observations: vorticity tends to align with the INTERMEDIATE
    eigenvector of strain, not the most stretching one. But this is a
    dynamical phenomenon — it arises from the TIME EVOLUTION, not from
    a static constraint on directions.

    Lei et al.'s condition is about the SPATIAL DISTRIBUTION of directions,
    not about the DYNAMICAL mechanism that suppresses stretching. The two
    are different things.

    Our computation shows that a static directional constraint (spanning S^2)
    does not, by itself, bound the stretching. The depletion must come from
    the DYNAMICS (how the Navier-Stokes equations evolve the vorticity field
    over time), not from any instantaneous geometric condition.

IMPLICATIONS FOR THE PROOF STRATEGY:
    - Graph-based approaches (L_1 = nI, weighted K_n) remain valid mathematics
      but lack the bridge to PDE dynamics
    - Direct PDE approaches (Grujic's Z_alpha framework, domain shrinking)
      are more promising because they engage with the dynamics
    - Numerical evidence (DNS near-singular solutions) could reveal the
      actual dynamical mechanism

WHAT TO TRY NEXT:
    Vector 1: Grujic's gap (Z_{2/5} -> Z_{1/2})
        - Combines sparseness of super-level sets with Lei et al.
        - Works in the PDE setting directly, no graph discretization needed
        - The 40% reduction (Z_{2/5} vs Z_{1/2}) might be closable with
          the additional geometric information from Lei et al.

    Vector 3: DNS data analysis
        - Study actual near-singular solutions from direct numerical simulation
        - Look for empirical patterns in the interaction network
        - Could suggest new mathematical structures

    Wild card: Constantin's identity + vortex sheet dynamics
        - omega . S omega = |omega|^2 (xi . S xi)
        - Near blow-up, vortex stretching concentrates on sheets/tubes
        - The sheet/tube geometry may provide the missing constraint
        - This is where Xiong-Yang (2024) anti-twist comes in

CAVEATS:
    1. Used regularized Biot-Savart (delta=0.01). The regularization prevents
       the 1/r^3 singularity but also smooths short-range interactions.
    2. Only tested n <= 20 filaments. Large n might behave differently.
    3. Two-filament case is degenerate (on-axis geometry -> zero stretching
       for ALL relative angles, not just parallel).
    4. The "optimal" is only approximate because the self-consistent iteration
       doesn't converge.

FILES:
    - scripts/wip/biot_savart_depletion.py (v1 — wrong baseline, aligned=0)
    - scripts/wip/biot_savart_depletion_v2.py (v2 — corrected, negative result)
    - scripts/wip/weighted_kn_analysis.py (L_1=nI fragility under weights)
    - scripts/wip/honest_assessment.py (adversarial review of full approach)
"""

# No code to run — this is documentation only.
print("This file documents a negative result. See docstring for details.")
print("Vector 2 (Biot-Savart depletion under directional isotropy) is dead.")
print("Promising alternatives: Vector 1 (Grujic's gap) or Vector 3 (DNS data).")
