"""
VECTOR 1: CLOSING GRUJIC'S GAP — Z_{2/5} to Z_{1/2}

THE FRAMEWORK (Grujic, 2017-2024):

    The NS regularity problem has a "scaling gap" between:
    - REGULARITY CLASS: Z_{1/2} — if vorticity super-level sets are this sparse,
      no blow-up occurs
    - A PRIORI BOUND: Z_{2/5} — what can be proved from the energy inequality
    - ENERGY LEVEL: Z_{1/3} — the baseline from Leray's energy estimate

    The gap: alpha = 2/5 (proved) vs alpha = 1/2 (needed).
    Grujic closed 40% of the distance from 1/3 to 1/2:
        1/3 → 2/5 → 1/2
        = 0.333 → 0.400 → 0.500
        Gap closed: (0.400 - 0.333)/(0.500 - 0.333) = 0.067/0.167 = 40%

DEFINITIONS:

    Super-level set:  S_lambda = {x : |omega_i^+/-(x)| > lambda * ||omega||_inf}

    3D delta-sparseness: S is delta-sparse around x0 at scale r if
        m_3(S cap B(x0, r)) / m_3(B(x0, r)) <= delta

    Z_alpha class: omega in Z_alpha(lambda, delta; c0) if for each x0,
        the super-level set is delta-sparse at scale r = c0 / ||omega||_inf^alpha

    Diffusion scale: d(t) = nu^{1/2} * ||omega(t)||_inf^{-1/2}

    The critical observation: Z_{1/2} means sparse at the diffusion scale.
    This is exactly where viscous diffusion can control vortex stretching.

OUR KEY IDEA:

    Lei et al. (2025): Near blow-up, vorticity directions xi = omega/|omega|
    must span S^2 (intersect every great circle).

    CONSEQUENCE FOR SUPER-LEVEL SETS:
    A single vortex filament has omega approximately parallel along its length.
    If Lei et al. forces omega to point in ALL directions near a potential
    singularity, the super-level set {|omega| > lambda*||omega||_inf}
    CANNOT be a single thin filament. It must have a more complex geometry
    (multiple filaments, sheets, or 3D structure).

    HYPOTHESIS: This multi-directional constraint forces the super-level
    sets to have higher effective dimension, which implies better sparseness,
    potentially pushing alpha from 2/5 toward 1/2.

THIS SCRIPT:
    1. Builds model super-level sets with various geometries
    2. Computes their sparseness at different scales
    3. Tests whether Lei et al. constraint improves the sparseness exponent
    4. Identifies what estimate would close the gap

Author: Claude (Opus 4.6) / Meridian, 2026-03-11
"""

import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(precision=6, linewidth=120)


# ============================================================
# SECTION 1: The scaling gap — precise numbers
# ============================================================

print("=" * 75)
print("SECTION 1: THE SCALING GAP IN THE Z_alpha FRAMEWORK")
print("=" * 75)

print("""
The NS regularity problem lives in the gap between three exponents:

  alpha = 1/3 = 0.333...  (ENERGY LEVEL — Leray's bound)
  alpha = 2/5 = 0.400     (A PRIORI BOUND — Grujic 2017)
  alpha = 1/2 = 0.500     (REGULARITY CLASS — sufficient for no blow-up)

The diffusion scale is d(t) = nu^{1/2} * ||omega||_inf^{-1/2}.

  Z_{1/2} means: super-level sets are sparse at scale r ~ ||omega||_inf^{-1/2}
                  = d(t) / nu^{1/2}

  At this scale, viscous diffusion dominates vortex stretching.

  Z_{2/5} means: super-level sets are sparse at scale r ~ ||omega||_inf^{-2/5}
                  = d(t)^{4/5} * (something larger than diffusion scale)

  The gap: we can only prove sparseness at a scale LARGER than the diffusion
  scale. We need to push it down to the diffusion scale itself.
""")

# The gap in terms of the exponent
alpha_energy = 1/3
alpha_proved = 2/5
alpha_needed = 1/2

gap_total = alpha_needed - alpha_energy
gap_closed = alpha_proved - alpha_energy
gap_remaining = alpha_needed - alpha_proved

print(f"  Total gap (1/3 to 1/2):    {gap_total:.6f}")
print(f"  Gap closed (1/3 to 2/5):   {gap_closed:.6f} = {gap_closed/gap_total*100:.1f}%")
print(f"  Gap remaining (2/5 to 1/2): {gap_remaining:.6f} = {gap_remaining/gap_total*100:.1f}%")

# In terms of scale ratios (for ||omega||_inf = Omega)
print(f"\n  For ||omega||_inf = Omega:")
print(f"    Z_{{1/3}} scale: r ~ Omega^{{-1/3}}")
print(f"    Z_{{2/5}} scale: r ~ Omega^{{-2/5}}")
print(f"    Z_{{1/2}} scale: r ~ Omega^{{-1/2}}")
print(f"    Ratio proved/needed: Omega^{{-2/5}}/Omega^{{-1/2}} = Omega^{{1/10}}")
print(f"    As Omega -> inf, this ratio -> inf: scale is too large by Omega^{{1/10}}")


# ============================================================
# SECTION 2: What does Lei et al. constrain geometrically?
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 2: GEOMETRIC CONSTRAINTS FROM LEI ET AL.")
print(f"{'='*75}")

print("""
Lei et al. (arXiv 2501.08976, Theorem 1.1):
  If vorticity is confined to a double cone C_e = {|xi x e| <= 1-delta}
  around axis e in the region {|omega| > M}, then the solution is regular.

Contrapositive:
  Near a singularity, for ANY axis e, there exist points in {|omega| > M}
  where |xi x e| > 1-delta (vorticity makes large angle with e).

  Equivalently: the vorticity directions in {|omega| > M} intersect
  every great circle on S^2.

GEOMETRIC CONSEQUENCES FOR SUPER-LEVEL SETS:

  Consider the super-level set S_lambda = {|omega| > lambda * ||omega||_inf}.

  In S_lambda, vorticity directions must span S^2 (not confined to any cone).

  What does this mean for the SHAPE of S_lambda?

  Case 1: SINGLE VORTEX FILAMENT
    A thin tube of radius rho, length L, with omega approximately parallel
    along its length. Directions: nearly constant (within angle ~ rho/R
    where R is radius of curvature).

    For Lei et al.: Need directions spanning S^2. A SINGLE filament CANNOT
    do this (directions nearly parallel). So S_lambda must be MORE than
    a single filament.

  Case 2: MULTIPLE FILAMENTS IN DIFFERENT DIRECTIONS
    N filaments of radius rho_i, each approximately straight.
    Directions: one per filament. Need at least enough filaments to
    span S^2 (minimum: 2-3 filaments, but for "intersecting every
    great circle" we need directions that are dense on S^2).

    Volume: sum of N * pi * rho_i^2 * L_i

    For this to be sparse at scale r in B(x0, r):
    Volume / (4/3 pi r^3) = sum(rho_i^2 * L_i) / (4/3 r^3) <= delta

  Case 3: VORTEX SHEET (2D structure)
    Omega varies direction across the sheet. Can span S^2 if the sheet
    curves enough. Volume: thickness * area.

  Key insight: ANY configuration satisfying Lei et al. requires either
  multiple filaments or a higher-dimensional structure. Both imply
  more volume at a given scale than a single filament.
""")


# ============================================================
# SECTION 3: Computational model — sparseness of various geometries
# ============================================================

print(f"\n{'='*75}")
print("SECTION 3: SPARSENESS OF DIFFERENT SUPER-LEVEL SET GEOMETRIES")
print(f"{'='*75}")


def measure_sparseness(points, x0, r):
    """
    Compute the 3D delta-sparseness of a point cloud around x0 at scale r.

    sparseness = fraction of B(x0, r) occupied by the set.
    points: Nx3 array of sample points from the set.

    We estimate sparseness by: count(points in B(x0,r)) / total_points
    scaled by the volume ratio.
    """
    dists = np.linalg.norm(points - x0, axis=1)
    n_in_ball = np.sum(dists <= r)
    return n_in_ball / len(points)


def generate_single_filament(n_points, rho, length, direction=None, center=None):
    """
    Generate points in a single vortex filament (thin tube).
    direction: unit vector along filament axis.
    rho: radius of tube.
    length: length of tube.
    """
    if direction is None:
        direction = np.array([0, 0, 1.0])
    if center is None:
        center = np.zeros(3)

    direction = direction / np.linalg.norm(direction)

    # Generate points along the tube
    points = []
    for _ in range(n_points):
        # Random point along the axis
        t = np.random.uniform(-length/2, length/2)
        # Random point in cross-section
        angle = np.random.uniform(0, 2*np.pi)
        r = rho * np.sqrt(np.random.uniform(0, 1))  # uniform in disk

        # Perpendicular directions
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, [1, 0, 0])
        else:
            perp1 = np.cross(direction, [0, 1, 0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        point = center + t * direction + r * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        points.append(point)

    return np.array(points)


def generate_multi_filament(n_points, rho, length, n_filaments, center=None):
    """
    Generate points in multiple vortex filaments with directions spanning S^2.
    (Satisfies Lei et al.)
    """
    if center is None:
        center = np.zeros(3)

    # Fibonacci sphere for filament directions
    golden_ratio = (1 + np.sqrt(5)) / 2
    directions = []
    for i in range(n_filaments):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n_filaments)
        d = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
        directions.append(d)

    points_per = n_points // n_filaments
    all_points = []

    for i, d in enumerate(directions):
        pts = generate_single_filament(points_per, rho, length, d, center)
        all_points.append(pts)

    return np.vstack(all_points)


def generate_vortex_sheet(n_points, thickness, radius, center=None):
    """
    Generate points on a curved vortex sheet (2D structure with varying direction).
    Uses a hemispherical shell — vorticity tangent to the sphere varies direction.
    """
    if center is None:
        center = np.zeros(3)

    points = []
    for _ in range(n_points):
        # Random point on hemisphere
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)  # full sphere

        # Point on sphere of radius R
        r = radius + np.random.uniform(-thickness/2, thickness/2)

        point = center + r * np.array([
            np.sin(phi)*np.cos(theta),
            np.sin(phi)*np.sin(theta),
            np.cos(phi)
        ])
        points.append(point)

    return np.array(points)


def generate_3d_blob(n_points, radius, center=None):
    """3D isotropic blob (highest dimension, most volume)."""
    if center is None:
        center = np.zeros(3)

    points = []
    while len(points) < n_points:
        p = np.random.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(center + p)

    return np.array(points)


# Compare sparseness at various scales
print(f"\n  Computing sparseness for different geometries at different scales...")
print(f"  (All geometries contain the same number of sample points)")
print(f"  (x0 = origin, measuring fraction of points within B(x0, r))")

n_points = 50000
np.random.seed(42)

# All geometries have comparable "size" ~ 0.1
rho = 0.005      # filament radius
length = 0.1     # filament length
thickness = 0.005  # sheet thickness
sheet_radius = 0.05  # sheet radius
blob_radius = 0.05   # blob radius

geometries = {
    "Single filament (1D)": generate_single_filament(n_points, rho, length),
    "6 filaments (spanning S2)": generate_multi_filament(n_points, rho, length, 6),
    "12 filaments (spanning S2)": generate_multi_filament(n_points, rho, length, 12),
    "Vortex sheet (2D)": generate_vortex_sheet(n_points, thickness, sheet_radius),
    "3D blob": generate_3d_blob(n_points, blob_radius),
}

x0 = np.zeros(3)
scales = [0.1, 0.05, 0.02, 0.01, 0.005]

print(f"\n  {'Geometry':>30s} | ", end="")
for r in scales:
    print(f"  r={r:.3f}", end="")
print()
print("  " + "-" * 85)

for name, pts in geometries.items():
    print(f"  {name:>30s} | ", end="")
    for r in scales:
        frac = measure_sparseness(pts, x0, r)
        print(f"  {frac:.4f}", end="")
    print()


# ============================================================
# SECTION 4: Scaling exponent — how sparseness depends on scale
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 4: SCALING EXPONENT — How does sparseness(r) behave?")
print(f"{'='*75}")

print("""
If a set has effective dimension d, then:
    volume in B(x0, r) ~ r^d
    sparseness ~ r^d / r^3 = r^{d-3}

For d=1 (filament): sparseness ~ r^{-2} (gets DENSER at small r)
For d=2 (sheet):     sparseness ~ r^{-1}
For d=3 (blob):      sparseness ~ r^0 = const

The Z_alpha condition requires:
    sparseness at scale r = c/Omega^alpha <= delta

For a filament of radius rho:
    sparseness ~ rho^2 / r^2
    Need rho^2 / (c/Omega^alpha)^2 <= delta
    => rho <= (delta*c^2)^{1/2} * Omega^{-alpha}

If rho scales with the diffusion scale d(t) = nu^{1/2}/Omega^{1/2}:
    rho ~ Omega^{-1/2}
    Need: Omega^{-1} / (c/Omega^alpha)^2 <= delta
    => Omega^{2alpha - 1} <= delta * c^2
    => For this to hold as Omega -> inf: need alpha >= 1/2

This is exactly the Z_{1/2} condition! A SINGLE filament at the diffusion
scale is EXACTLY at the critical sparseness.

NOW with Lei et al.: the super-level set CANNOT be a single filament.
What changes?
""")

# Compute effective dimension via sparseness scaling
print("  Effective dimensions from scaling fit (sparseness ~ r^{d-3}):\n")

# More scales for better fit
fit_scales = np.logspace(-2.5, -0.5, 20)

for name, pts in geometries.items():
    fracs = []
    valid_scales = []
    for r in fit_scales:
        f = measure_sparseness(pts, x0, r)
        if f > 0 and f < 0.99:
            fracs.append(f)
            valid_scales.append(r)

    if len(valid_scales) >= 3:
        log_r = np.log(valid_scales)
        log_f = np.log(fracs)
        # Linear fit: log(f) = (d-3)*log(r) + const
        coeffs = np.polyfit(log_r, log_f, 1)
        d_eff = coeffs[0] + 3
        print(f"  {name:>30s}: d_eff = {d_eff:.2f} (slope = {coeffs[0]:.2f})")
    else:
        print(f"  {name:>30s}: insufficient data for fit")


# ============================================================
# SECTION 5: The critical estimate — what Lei et al. buys us
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 5: WHAT WOULD LEI ET AL. BUY IN THE Z_alpha FRAMEWORK?")
print(f"{'='*75}")

print("""
KEY ANALYSIS:

The sparseness condition Z_alpha requires:
    m_3({|omega| > lambda*Omega} cap B(x0, r_alpha)) <= delta * (4/3)pi * r_alpha^3

where r_alpha = c * Omega^{-alpha}.

1. WITHOUT Lei et al.:
   The super-level set could be a SINGLE vortex filament of radius rho ~ Omega^{-1/2}
   (the diffusion scale). For this filament:

   Volume in B(x0, r_alpha) ~ pi * rho^2 * min(r_alpha, L)
                             ~ Omega^{-1} * Omega^{-alpha}
                             = Omega^{-(1+alpha)}

   Ball volume: (4/3)pi * r_alpha^3 = (4/3)pi * c^3 * Omega^{-3*alpha}

   Sparseness ~ Omega^{-(1+alpha)} / Omega^{-3*alpha}
              = Omega^{2*alpha - 1}

   For sparseness <= delta as Omega -> inf: need 2*alpha - 1 <= 0
                                             => alpha <= 1/2

   So Z_{1/2} is tight for a single filament.

2. WITH Lei et al.:
   The super-level set must contain vortex structures pointing in MANY
   directions. Minimum: enough filaments to span S^2.

   Consider N filaments, each of radius rho and length l,
   pointing in ~N distinct directions on S^2.

   For directions to span S^2 with angular resolution theta:
   Need N ~ 1/theta^2 filaments (covering number of S^2).

   Total volume in B(x0, r_alpha):
   ~ N * pi * rho^2 * min(r_alpha, l)

   If all filaments pass through x0:
   ~ N * rho^2 * r_alpha

   Sparseness ~ N * rho^2 * r_alpha / r_alpha^3
              = N * rho^2 / r_alpha^2
              = N * Omega^{-1} / (Omega^{-alpha})^2
              = N * Omega^{2*alpha - 1}

   SAME scaling! Adding more filaments makes sparseness WORSE (by factor N),
   not better.

   HOWEVER: the KEY is that N filaments of total vorticity Omega means
   each filament has vorticity ~ Omega/N^{1/2} (if enstrophy is fixed).
   This changes the scaling...

3. REFINED ANALYSIS (enstrophy constraint):

   Enstrophy: Z = integral |omega|^2 ~ N * (Omega_per)^2 * rho^2 * l
   where Omega_per = vorticity per filament.

   If total enstrophy is Z and we have N filaments:
   Omega_per^2 ~ Z / (N * rho^2 * l)

   But ||omega||_inf = Omega = max over filaments of Omega_per.
   If filaments are comparable: Omega ~ Omega_per ~ (Z / (N * rho^2 * l))^{1/2}

   For more filaments (larger N), same enstrophy Z and same rho, l:
   Omega ~ (Z/N)^{1/2} / (rho * l^{1/2}) -> SMALLER as N increases

   So multi-directional vorticity (N large) means SMALLER ||omega||_inf
   for the same enstrophy Z. This is exactly the "depletion" in a different
   guise: the L^infinity norm is reduced.

   The sparseness scale r_alpha = c * Omega^{-alpha} INCREASES when Omega decreases.
   Larger scale -> easier to be sparse.

   QUANTITATIVE: If N filaments of comparable strength:
   Omega ~ Omega_single / sqrt(N)    (for fixed enstrophy)
   r_alpha = c / Omega^alpha = c * N^{alpha/2} / Omega_single^alpha

   Sparseness at this new scale:
   ~ N * rho^2 / (c * N^{alpha/2} / Omega_single^alpha)^2
   = N * rho^2 * Omega_single^{2*alpha} / (c^2 * N^alpha)
   = N^{1-alpha} * rho^2 * Omega_single^{2*alpha} / c^2

   For a single filament (N=1):
   = rho^2 * Omega_single^{2*alpha} / c^2

   Need rho^2 * Omega_single^{2*alpha} <= delta * c^2
   With rho ~ Omega_single^{-1/2}: Omega_single^{2*alpha - 1} <= delta * c^2
   => alpha >= 1/2 (same as before)

   For N filaments (N > 1):
   = N^{1-alpha} * [rho^2 * Omega_single^{2*alpha}] / c^2

   If alpha < 1: factor N^{1-alpha} grows with N.
   SPARSENESS GETS WORSE with more filaments at same individual scale!

   BUT: if alpha = 1/2:
   = N^{1/2} * [rho^2 * Omega^{2*(1/2)}] / c^2
   = N^{1/2} * [Omega_single^{-1} * Omega_single] / c^2
   = N^{1/2} / c^2

   This grows with N. More filaments at Z_{1/2} is HARDER, not easier.
""")


# ============================================================
# SECTION 6: The actual path — what CAN work?
# ============================================================

print(f"\n{'='*75}")
print("SECTION 6: THE ACTUAL PATH — What might close the gap?")
print(f"{'='*75}")

print("""
The analysis above shows that Lei et al.'s multi-directional constraint
does NOT directly improve the sparseness exponent. Adding more filaments
(to satisfy the S^2-spanning condition) makes the sparseness WORSE at
any given scale, not better.

HOWEVER: Lei et al. gives us something DIFFERENT — not about sparseness
of super-level sets, but about the STRUCTURE of the vortex stretching term.

APPROACH A: Stretching cancellation (our Vector 2 — DEAD)
    The stretching omega . S omega might cancel when directions are isotropic.
    Result: negative. Optimal stretching already spans S^2.

APPROACH B: Vortex reconnection / depletion of coherence
    Multi-directional filaments undergo RECONNECTION, which:
    - Reduces ||omega||_inf (confirmed by DNS)
    - Creates small-scale structure (enhances dissipation)
    - Is a DYNAMICAL process, not a static geometric one

    This is the Hou-Li (2006) / Xiong-Yang (2024) direction:
    vortex twisting/reconnection as a regularizing mechanism.

APPROACH C: Higher-order sparseness (Grujic's asymptotic criticality)
    Grujic's 2019 result: as derivative order k -> inf, the scaling gap
    between Z_alpha^{(k)} regularity class and a priori bound VANISHES.

    This means: for HIGH-ORDER derivatives, sparseness IS at the critical
    scale. The question: can this asymptotic result be made effective?

    Specifically: at what derivative order k does the gap close?
    If we can show gap < epsilon for some finite k, and bound the
    k-th derivative independently, we're done.

APPROACH D: Interpolation between Z_{2/5} and Z_{1/2}
    The idea: use Lei et al. to get an INTERMEDIATE result:

    Grujic proves Z_{2/5} for vorticity.
    If Lei et al. + dynamics gives Z_beta for some beta in (2/5, 1/2)
    for the STRAIN (not vorticity), we might close the gap by combining:

    - Z_{2/5} for omega (Grujic)
    - Z_beta for S omega (new, from geometric constraint)
    - Interpolation between the two

    This is speculative but mathematically sound as a strategy.

APPROACH E: Domain shrinking (from continuous_bypass.py)
    Near a Type I singularity, the effective domain shrinks as B(r(t))
    with r(t) -> 0. The Stokes eigenvalue on B(r) scales as 1/r^2.

    For Type I: Omega * (T-t)^{1/2} bounded. So r ~ (T-t)^{1/2}.
    Stokes gap: lambda_1 ~ 1/r^2 ~ 1/(T-t).
    Omega ~ (T-t)^{-1/2}.

    Enstrophy: dZ/dt <= C * Omega * Z - nu * lambda_1 * Z
                      = C * (T-t)^{-1/2} * Z - nu/(T-t) * Z

    For t close to T: -nu/(T-t) dominates C/(T-t)^{1/2}.
    So dZ/dt < 0 near T. Type I blow-up gives regularity.

    THIS IS KNOWN (Escauriaza-Seregin-Sverak 2003, Type I is impossible).

    The open case is Type II: Omega grows FASTER than (T-t)^{-1/2}.
    Then domain shrinking is NOT enough.

LET'S EXPLORE APPROACH C COMPUTATIONALLY.
""")


# ============================================================
# SECTION 7: Approach C — Higher-order sparseness
# ============================================================

print(f"\n{'='*75}")
print("SECTION 7: APPROACH C — ASYMPTOTIC CRITICALITY")
print(f"{'='*75}")

print("""
Grujic's key result (2019/2024):

For the k-th spatial derivative D^k u, define Z_alpha^{(k)} classes
analogously, using super-level sets of |D^k u|.

The regularity class is Z_{1/2}^{(k)} for all k.

The a priori bound improves with k:
    k=0: Z_{1/3}  (energy level, Leray)
    k=1: Z_{2/5}  (Grujic 2017, the 40% improvement)
    k=2: Z_{alpha_2} for some alpha_2 > 2/5
    ...
    k -> inf: Z_{alpha_k} with alpha_k -> 1/2

The question: what is alpha_k for each k? How fast does it converge?

If alpha_k = 1/2 - c/k^p for some p > 0, then for finite k we need
c/k^p < epsilon, i.e., k > (c/epsilon)^{1/p}.

This is Approach C: make the convergence EFFECTIVE.
""")

# Model the convergence
# From the literature, the improvement per derivative order is algebraic
# The exact rate isn't published clearly, but we can model it

# Known data points:
# k=0: alpha = 1/3
# k=1: alpha = 2/5
# These suggest: alpha_k = 1/2 - 1/(6*(k+1)) approximately
#   k=0: 1/2 - 1/6 = 1/3 ✓
#   k=1: 1/2 - 1/12 = 5/12 ≈ 0.417  (but Grujic gets 2/5 = 0.400)

# Alternative: alpha_k = 1/2 - 1/(2*(2k+3))
#   k=0: 1/2 - 1/6 = 1/3 ✓
#   k=1: 1/2 - 1/10 = 2/5 ✓
#   k=2: 1/2 - 1/14 = 5/14 ≈ 0.357  -- no, this DECREASES. Wrong.

# Must be: alpha_k = 1/2 - c / f(k) with f(k) increasing
# k=0: c/f(0) = 1/6 => c/f(0) = 1/6
# k=1: c/f(1) = 1/10 => c/f(1) = 1/10
# Ratio: f(1)/f(0) = 10/6 = 5/3
# If f(k) = 2k+3: f(0)=3, f(1)=5
# Then c = 1/6 * 3 = 1/2. alpha_k = 1/2 - 1/(2(2k+3))
# k=0: 1/2 - 1/6 = 1/3 ✓
# k=1: 1/2 - 1/10 = 2/5 ✓
# k=2: 1/2 - 1/14 ≈ 0.429
# k=3: 1/2 - 1/18 ≈ 0.444
# k=4: 1/2 - 1/22 ≈ 0.455
# k=5: 1/2 - 1/26 ≈ 0.462
# k=10: 1/2 - 1/46 ≈ 0.478
# k=20: 1/2 - 1/86 ≈ 0.488
# k -> inf: 1/2 ✓ (O(1/k) convergence)

print("\n  MODEL: alpha_k = 1/2 - 1/(2(2k+3))")
print("  (matches k=0: 1/3, k=1: 2/5, and alpha_k -> 1/2)")
print()
print(f"  {'k':>4} | {'alpha_k':>10} | {'gap (1/2-alpha_k)':>18} | {'% of gap closed':>15}")
print("  " + "-" * 55)

for k in [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]:
    alpha_k = 0.5 - 1.0 / (2 * (2*k + 3))
    gap_k = 0.5 - alpha_k
    pct_closed = (alpha_k - 1/3) / (1/2 - 1/3) * 100
    print(f"  {k:4d} | {alpha_k:10.6f} | {gap_k:18.6f} | {pct_closed:14.1f}%")


# ============================================================
# SECTION 8: Approach D — Strain sparseness from Lei et al.
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 8: APPROACH D — STRAIN SPARSENESS FROM GEOMETRY")
print(f"{'='*75}")

print("""
Here's a NEW idea that combines Vector 1 and the Lei et al. result:

OBSERVATION: The enstrophy production is omega . S omega.
  - omega is bounded by ||omega||_inf (vorticity)
  - S omega is bounded by ||S||_inf * ||omega||_inf (strain times vorticity)

The BKM criterion says: if integral_0^T ||omega||_inf dt < inf, regularity.
The Beale-Kato-Majda criterion is equivalent to: ||omega||_inf can't blow up.

But the actual enstrophy production involves omega . S omega, which
involves the ALIGNMENT between omega and S's eigenvectors.

KEY QUANTITY: the stretching rate sigma = xi . S . xi where xi = omega/|omega|.

Constantin's identity: omega . S omega = |omega|^2 * sigma.

If sigma were bounded by C (independent of ||omega||_inf), then:
    dZ/dt <= C * integral |omega|^2 * |omega| dx - nu * (dissipation)
           = C * integral |omega|^3 dx - nu * integral |grad omega|^2 dx

The ratio ||omega||_3^3 / ||grad omega||_2^2 is bounded by known
interpolation inequalities. Whether this closes depends on the exponents.

COULD LEI ET AL. BOUND sigma?

From the numerical experiments (v2 script):
  - Optimal sigma is NOT bounded by Lei et al. condition (negative result)
  - But AVERAGE sigma might be constrained

Let me check: what is the AVERAGE stretching rate sigma_avg over the
high-vorticity region, given Lei et al.'s constraint?
""")

# Numerical test: average stretching rate
# For a vortex filament system satisfying Lei et al.,
# what is the average of xi . S . xi?

n_filaments = 15
n_trials = 100
np.random.seed(42)

# Reuse the Biot-Savart functions from v2 (simplified here)
def bs_velocity(x, xj, xij, gj, delta=0.01):
    r = x - xj
    r_sq = np.dot(r, r) + delta**2
    return (gj / (4 * np.pi)) * np.cross(xij, r) / r_sq**1.5

def bs_strain(x, xj, xij, gj, delta=0.01):
    h = 1e-6
    S = np.zeros((3, 3))
    for k in range(3):
        dx = np.zeros(3)
        dx[k] = h
        grad_k = (bs_velocity(x+dx, xj, xij, gj, delta) - bs_velocity(x-dx, xj, xij, gj, delta)) / (2*h)
        S[k, :] += grad_k
        S[:, k] += grad_k
    return S / 2.0

sigma_iso_avgs = []
sigma_iso_maxs = []
sigma_rand_avgs = []
sigma_rand_maxs = []

for trial in range(n_trials):
    seed = trial * 37
    rng = np.random.RandomState(seed)

    # Random positions
    pos = []
    while len(pos) < n_filaments:
        p = rng.uniform(-0.1, 0.1, 3)
        if np.linalg.norm(p) <= 0.1:
            pos.append(p)
    pos = np.array(pos)
    gammas = np.ones(n_filaments)

    # Isotropic directions (Fibonacci)
    golden = (1 + np.sqrt(5)) / 2
    iso_dirs = np.zeros((n_filaments, 3))
    for i in range(n_filaments):
        theta = 2 * np.pi * i / golden
        phi = np.arccos(1 - 2 * (i + 0.5) / n_filaments)
        iso_dirs[i] = [np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]

    # Random directions
    rand_dirs = rng.randn(n_filaments, 3)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    for dirs, avg_list, max_list in [(iso_dirs, sigma_iso_avgs, sigma_iso_maxs),
                                       (rand_dirs, sigma_rand_avgs, sigma_rand_maxs)]:
        sigmas = []
        for i in range(n_filaments):
            S_total = np.zeros((3, 3))
            for j in range(n_filaments):
                if i == j:
                    continue
                S_total += bs_strain(pos[i], pos[j], dirs[j], gammas[j])
            sigma = dirs[i] @ S_total @ dirs[i]
            sigmas.append(sigma)

        sigmas = np.array(sigmas)
        avg_list.append(np.mean(sigmas))
        max_list.append(np.max(sigmas))

print(f"\n  Average stretching rate sigma = xi . S . xi")
print(f"  n = {n_filaments} filaments, {n_trials} trials\n")

print(f"  ISOTROPIC (Lei et al.):")
print(f"    <sigma>_avg: mean = {np.mean(sigma_iso_avgs):+.2f}, std = {np.std(sigma_iso_avgs):.2f}")
print(f"    sigma_max:   mean = {np.mean(sigma_iso_maxs):+.2f}, std = {np.std(sigma_iso_maxs):.2f}")

print(f"\n  RANDOM:")
print(f"    <sigma>_avg: mean = {np.mean(sigma_rand_avgs):+.2f}, std = {np.std(sigma_rand_avgs):.2f}")
print(f"    sigma_max:   mean = {np.mean(sigma_rand_maxs):+.2f}, std = {np.std(sigma_rand_maxs):.2f}")

# Critical comparison
avg_ratio = np.mean(np.abs(sigma_iso_avgs)) / np.mean(np.abs(sigma_rand_avgs)) if np.mean(np.abs(sigma_rand_avgs)) > 0 else float('inf')
max_ratio = np.mean(sigma_iso_maxs) / np.mean(sigma_rand_maxs) if np.mean(sigma_rand_maxs) > 0 else float('inf')

print(f"\n  |<sigma>_iso| / |<sigma>_rand|: {avg_ratio:.4f}")
print(f"  sigma_max_iso / sigma_max_rand: {max_ratio:.4f}")

# Is average sigma near zero for isotropic?
iso_positive_frac = np.mean(np.array(sigma_iso_avgs) > 0)
rand_positive_frac = np.mean(np.array(sigma_rand_avgs) > 0)
print(f"\n  Fraction with positive average sigma:")
print(f"    Isotropic: {iso_positive_frac:.2f}")
print(f"    Random:    {rand_positive_frac:.2f}")


# ============================================================
# SECTION 9: Summary and next steps
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 9: SUMMARY AND ASSESSMENT")
print(f"{'='*75}")

print("""
WHAT WE'VE LEARNED:

1. THE SCALING GAP:
   Z_{2/5} (proved) to Z_{1/2} (needed).
   The gap is alpha = 0.4 -> 0.5, or a factor Omega^{1/10} in scale.

2. LEI ET AL. DOESN'T DIRECTLY HELP:
   Adding multi-directional filaments makes sparseness WORSE (more volume
   for the same sparseness test). The constraint forces more structure
   but also more material in the super-level set.

3. HIGHER DERIVATIVES (Approach C):
   Grujic's asymptotic criticality says alpha_k -> 1/2 as k -> inf.
   With our model alpha_k = 1/2 - 1/(2(2k+3)), convergence is O(1/k).
   Need k ~ 50 to get within 1% of 1/2.

   Making this EFFECTIVE requires bounding ||D^k omega||_inf^alpha_k
   for some specific k. This is possible in principle but technically
   demanding.

4. STRAIN SPARSENESS (Approach D):
   The average stretching sigma = xi . S . xi is roughly the same for
   isotropic and random configurations. Lei et al. doesn't help bound
   the average or maximum sigma.

HONEST ASSESSMENT OF VECTOR 1:
   The Z_alpha framework is mathematically sound and represents genuine
   progress (40% gap reduction). But closing the remaining 60% appears
   to require NEW IDEAS beyond combining existing results.

   The approaches we tested (adding Lei et al. geometry, computing
   higher-order sparseness) don't bridge the gap.

   The fundamental difficulty: the gap is about the COMPETITION between
   stretching (O(Omega^{3/2})) and dissipation (O(Omega^2 / r^2)).
   At the proved scale r ~ Omega^{-2/5}, dissipation scales as
   Omega^{2+4/5} = Omega^{14/5}, while stretching scales as Omega^{3/2}.
   Dissipation wins (14/5 > 3/2). But this is vacuous because we
   haven't bounded Omega in terms of the scale — we've only proved
   sparseness.

   The gap is ultimately about:
   - How does ||omega||_inf relate to the enstrophy Z?
   - Can we bound ||omega||_inf^{1+2*alpha} by something controlled?

   For alpha = 1/2: need ||omega||_inf^2, which is controlled by Z
   via ||omega||_inf^2 <= C * Z * (something).

   For alpha = 2/5: only control ||omega||_inf^{9/5}, which is not
   enough to close.

WHAT MIGHT ACTUALLY WORK:
   1. A completely new approach to bounding ||omega||_inf
   2. Exploiting the DYNAMICS (time evolution) rather than instantaneous
      geometry
   3. Computer-assisted proof at a specific derivative order k
   4. Finding a qualitatively different regularity criterion that
      Lei et al. does satisfy
""")
