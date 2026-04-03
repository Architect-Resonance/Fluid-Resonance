"""
ANTI-TWIST LITERATURE DEEP READ -- APPROACH (g1b)
==================================================
S100-M1b: Can Buaria anti-twist + Berry frustration close the 3/28 gap?

PAPERS READ:
  1. Buaria, Lawson & Wilczek 2024 (Science Advances eado1969)
     "Twisting vortex lines regularize Navier-Stokes turbulence"
  2. Jimenez, Wray, Saffman & Rogallo 1993 (JFM 255, 65-90)
     "The structure of intense vorticity in isotropic turbulence"
  3. Tube width consensus from DNS literature (Re_lambda = 94 to 1934)
  4. She & Leveque 1994 (PRL 72, 336) — filament model

EXECUTIVE SUMMARY:
  Option A (anti-twist + Berry synergy) is DEAD.

  Reason: Buaria 2024 identifies the anti-twist MECHANISM but provides NO
  quantitative bounds on tube width. Meanwhile, the DNS consensus (Jimenez 1993
  and many confirmations) shows tube radius R0 ~ 5*eta INDEPENDENT of Re.
  This means tube angular width delta(k) ~ eta*k = k/k_d, which is < 1 for
  ALL inertial-range k. Berry frustration never activates in the inertial range.

  The anti-twist prevents singularity (tubes don't narrow to zero width) but
  the equilibrium width is the Kolmogorov scale — exactly where viscosity
  already kills everything. The anti-twist IS the viscous regularization,
  viewed from the vortex-aligned frame. It's not a new mechanism.

==========================================================================
"""

# =====================================================================
# SECTION 1: What Buaria 2024 Actually Shows
# =====================================================================
"""
BUARIA, LAWSON & WILCZEK 2024 — Key Results

1. FRAMEWORK: Conditionally Averaged Vorticity (CAV)
   - Observer aligned with vorticity vector omega_hat
   - Cylindrical coordinates (rho, theta, z) with e_z = omega_hat
   - Vorticity decomposed: omega_bar = omega_z e_z + omega_rho e_rho + omega_theta e_theta
   - The AZIMUTHAL component omega_theta encodes twist

2. THE ANTI-TWIST EQUATION:
   <(omega_hat . grad u) . omega_hat | Omega> = integral(3*rho^2*z/r^5 * omega_theta dρ dz)

   - Vorticity amplification depends SOLELY on omega_theta (twist)
   - When omega_theta > 0 everywhere: stretching increases monotonically
   - DISCOVERY: At high conditioning (Omega/mean >> 1), omega_theta becomes
     NEGATIVE in the core region ("anti-twist")
   - This negative omega_theta ATTENUATES further amplification

3. WHAT THEY MEASURED:
   - DNS at Re_lambda = 140, 200, 240, 390, 650, 1300
   - Grid sizes up to 12288^3 (!)
   - Conditioning at Omega/<Omega> = 1, 30, 1000
   - Experimental validation at Re_lambda ~ 200

4. WHAT THEY DID NOT PROVIDE:
   - NO scaling law for tube width vs Re or k
   - NO quantitative bound on maximum vorticity
   - NO functional form for anti-twist magnitude vs Omega
   - NO proof that anti-twist prevents blowup
   - Explicit statement: "an important future task would be to leverage this
     physical mechanism to establish rigorous bounds on vorticity amplification"

5. CONCLUSION FOR OUR PURPOSES:
   The anti-twist is a QUALITATIVE mechanism. It explains WHY tubes don't
   blow up, but doesn't give QUANTITATIVE bounds on tube geometry.
"""

# =====================================================================
# SECTION 2: Universal Tube Width — The Consensus
# =====================================================================
"""
VORTEX TUBE RADIUS SCALING — DNS CONSENSUS

The most established result in turbulence structure theory:

   R0 ~ 5*eta    (independent of Re)

where eta = (nu^3/epsilon)^(1/4) is the Kolmogorov microscale.

Key references:
  - Jimenez et al. 1993 (JFM 255): R0 ~ 4-6 eta, Re_lambda = 35-170
  - Experimental (Re_lambda = 332-1934): R0 ~ 5 eta, INDEPENDENT of Re
  - DNS (Re_lambda = 94-1445): radius ~ O(eta) universally
  - She & Leveque 1994 (PRL 72): filaments codimension-2, beta = 2/3

Physical picture:
  - Tube radius set by BALANCE between stretching (narrows tube) and
    viscous diffusion (widens tube)
  - This balance gives R0 ~ eta regardless of large-scale forcing
  - Length scales with integral scale L (NOT eta)
  - Aspect ratio ~ L/eta ~ Re^(3/4) grows with Re

In wavenumber space:
  - Tube core corresponds to k ~ k_d = eta^(-1) ~ (epsilon/nu^3)^(1/4)
  - k_d ~ Re^(3/4) / L
  - Tube angular width at wavenumber k: delta(k) ~ 1/(k*R0) ~ 1/(k*eta) = k_d/k * (1/k_d*eta)

  More precisely:
  - A tube of radius R0 ~ eta and orientation n_hat concentrates energy
    in a cone of angular width delta ~ 1/(k*R0) around n_hat
  - At wavenumber k: delta(k) ~ 1/(k*eta) = k_d/k (since eta ~ 1/k_d)

  For k in inertial range (k << k_d):
    delta(k) ~ k_d/k >> 1  ... wait, this needs care.

  CORRECTION: The angular width of a tube's Fourier signature at
  wavenumber k depends on how k compares to 1/R0:
    - k << 1/R0 (k << k_d): tube looks like a LINE in Fourier space,
      concentrated along a 1D subspace. Angular width ~ pi (broad).
    - k ~ 1/R0 (k ~ k_d): angular width ~ 1 (transition)
    - k >> 1/R0 (k >> k_d): exponential decay, tube invisible.

  So for INERTIAL range k << k_d, tubes are actually NARROW in Fourier
  space (concentrated on a line), not broad!

  This REVERSES the naive pencil picture from g1:
  - In g1 we said delta ~ k0/k where k0 = 1/L (tube orientation scale)
  - Actually, tube RADIUS R0 ~ eta means the angular concentration
    in Fourier space depends on which wavevector directions contribute
  - A straight tube of length L and radius R0: Fourier transform is
    a disk of radius ~1/R0 in the direction perpendicular to tube,
    and extent ~1/L along tube axis
  - At |k| = k: the intersection with the k-shell is an ARC of
    angular extent ~ min(1, 1/(k*R0))

  Let me be precise:
"""

import numpy as np

# =====================================================================
# SECTION 3: Precise Fourier Geometry of a Vortex Tube
# =====================================================================
"""
A cylindrical vortex tube with:
  - Axis along z-hat
  - Length L (integral scale)
  - Core radius R0 ~ eta (Kolmogorov scale)
  - Gaussian vorticity profile: omega ~ exp(-rho^2/(2*R0^2))

Fourier transform:
  omega_hat(k) ~ exp(-k_perp^2 * R0^2 / 2) * sinc(k_z * L / 2)

where k_perp = sqrt(kx^2 + ky^2), k_z = k . z_hat.

On a shell |k| = k:
  k_perp = k * sin(gamma), k_z = k * cos(gamma)
  where gamma = angle between k and tube axis

  omega_hat(k) ~ exp(-(k*sin(gamma))^2 * R0^2 / 2) * sinc(k*cos(gamma) * L / 2)

The Gaussian factor exp(-k^2*R0^2*sin^2(gamma)/2) concentrates the signal
within gamma < 1/(k*R0) of the tube axis.

So the ANGULAR WIDTH on the k-shell is:
  delta_gamma(k) ~ 1/(k*R0) = 1/(k*eta) = k_d/(k*something)

More carefully: R0 = c*eta with c ~ 5, and k_d = 1/eta, so:
  delta_gamma(k) ~ 1/(k * c * eta) = k_d/(c * k) = (1/c) * k_d/k

For k in the inertial range:
  - k_d/k >> 1 (since k << k_d)
  - delta_gamma >> 1/c ~ 0.2
  - The tube signal is spread over a LARGE angular region!

For k near k_d:
  - delta_gamma ~ 1/c ~ 0.2 radians ~ 12 degrees
  - Still fairly broad

KEY INSIGHT: Vortex tubes have their Fourier signal concentrated near
the tube axis direction, but at inertial-range scales (k << k_d),
the angular spread is LARGE — the pencil effect is WEAK.

This is OPPOSITE to what Berry frustration needs!
  - Berry frustration is defeated by NARROW pencils (delta << 1)
  - But at inertial-range k, pencils are WIDE (delta >> 1)
  - Berry frustration IS active at inertial-range k!

Wait — this REVERSES the g1 conclusion. Let me re-examine.
"""

# =====================================================================
# SECTION 4: Corrected Angular Width Analysis
# =====================================================================
"""
THE g1 PENCIL PICTURE WAS WRONG about delta(k).

In g1 (Section 5), I wrote:
  "A vortex tube of angular width delta ~ k0/k"
  with k0 = 1/L = large-scale wavenumber.

This was the angular width of the TUBE IN PHYSICAL SPACE as seen from
wavenumber k. But the relevant quantity is the angular width of the
FOURIER AMPLITUDE on the k-shell.

For a tube of radius R0 and length L along axis n_hat:
  - Physical-space angular extent: Omega_phys ~ R0/L ~ eta/L ~ Re^(-3/4)
  - Fourier-space angular width at |k|=k:
    delta(k) ~ 1/(k*R0) = 1/(k*eta)

At different scales:
  k = k_L (integral): delta ~ L/eta ~ Re^(3/4) >> 1 (isotropic)
  k = k_inertial:      delta ~ k_d/k (intermediate, could be > or < 1)
  k = k_d:             delta ~ 1/c ~ 0.2 (concentrated)
  k = k_* (CLT scale): delta ~ k_d/k_* = k_d/(k_d * Re^(3/28))
                        = Re^(-3/28) ~ 0.3 at Re=10^6

The TRANSITION from isotropic (delta >> 1) to pencil (delta << 1)
occurs at k_transition where delta = 1:
  1/(k_tr * R0) = 1
  k_tr = 1/R0 ~ k_d/c ~ k_d/5

So the ENTIRE INERTIAL RANGE has delta > 1 (tubes are isotropic
in Fourier space). Berry frustration IS active throughout the
inertial range!

The pencil effect only appears at k ~ k_d (dissipation range),
where viscosity dominates anyway.
"""

# Numerical table: angular width at various scales
print("=" * 70)
print("ANGULAR WIDTH delta(k) = 1/(k * R0) FOR R0 = 5*eta")
print("=" * 70)
print(f"{'Scale':>20} {'k/k_d':>10} {'delta(k)':>10} {'Berry active?':>15}")
print("-" * 70)

scales = [
    ("Integral (k_L)", 1e-4, None),
    ("Inertial (mid)", 0.01, None),
    ("Inertial (deep)", 0.1, None),
    ("CLT scale (k_*)", None, "Re-dependent"),  # special
    ("Transition", 0.2, None),  # k = k_d/5
    ("Kolmogorov (k_d)", 1.0, None),
    ("Super-Kolmogorov", 2.0, None),
]

c = 5.0  # R0/eta

for name, k_ratio, note in scales:
    if k_ratio is not None:
        delta = 1.0 / (k_ratio * c)
        active = "YES" if delta > 1 else "PARTIAL" if delta > 0.3 else "NO"
        print(f"{name:>20} {k_ratio:>10.4f} {delta:>10.2f} {active:>15}")
    else:
        # CLT scale: k_*/k_d = Re^(3/28)
        for Re_lam in [100, 1000, 10000, 100000]:
            Re = Re_lam  # approximate
            k_ratio_clt = Re**(3/28)
            delta_clt = 1.0 / (k_ratio_clt * c)
            active = "YES" if delta_clt > 1 else "PARTIAL" if delta_clt > 0.3 else "NO"
            print(f"{'CLT (Re='+str(Re_lam)+')':>20} {k_ratio_clt:>10.4f} {delta_clt:>10.4f} {active:>15}")

print()
print("=" * 70)
print("KEY FINDING: delta > 1 throughout the inertial range!")
print("Berry frustration IS active where it matters.")
print("The g1 pencil loophole was based on incorrect delta(k).")
print("=" * 70)

# =====================================================================
# SECTION 5: Re-assessment of Berry Frustration with Correct delta(k)
# =====================================================================
"""
WITH CORRECTED delta(k), the Berry frustration picture changes:

g1 ORIGINAL CLAIM (WRONG):
  "Pencil angular width delta ~ k0/k << 1 at high k"
  "Berry frustration weak in pencils"
  "Total stretching delta-independent (constant-sum trade-off)"
  "Berry does NOT close the gap"

CORRECTED PICTURE:
  Tube FOURIER angular width delta(k) = 1/(k*R0) = 1/(k*eta)
  At inertial-range k: delta >> 1 (tubes look isotropic in Fourier space)
  Berry frustration IS active throughout inertial range
  The pencil loophole applies only at k ~ k_d (dissipation range)

HOWEVER — this doesn't automatically close the gap either. Why?

The issue is more subtle. Even though tubes are "isotropic" at each k
in the inertial range (delta >> 1), the CORRELATIONS between different
k-shells are NOT isotropic. A tube creates phase correlations:

  arg(a_+(k)) = Berry phase of k_hat relative to tube axis

These phases are CORRELATED across k-shells because they all reference
the same tube axis. The question is whether this correlation is enough
to defeat the topological frustration.

REFINED ANALYSIS:
  - Single tube: phases perfectly correlated (all aligned with tube axis)
  - But the tube only contains O(1) modes at each k (a LINE in Fourier space)
  - Total modes on k-shell: N ~ k^2
  - Fraction in tube: f_tube ~ delta^2 / (4*pi) if delta < 1,
    or ~ 1 if delta > 1
  - At inertial k: f_tube ~ 1 (tube fills the shell) -- NO PENCIL EFFECT
  - At k ~ k_d: f_tube ~ delta^2 ~ 1/c^2 ~ 0.04 -- SMALL

So at inertial scales, a single tube's Fourier modes span the entire
k-shell, and Berry frustration applies fully. The q >= 3/2 bound holds.

At dissipation scales (k ~ k_d), only 4% of modes are in each tube's
pencil, allowing local coherence. But viscosity dominates there anyway.

VERDICT: The g1 pencil loophole was a PHYSICAL-SPACE vs FOURIER-SPACE
confusion. In Fourier space, tubes are isotropic at inertial scales.
Berry frustration IS active throughout the inertial range.

But this still doesn't close the 3/28 gap, because:
1. Berry frustration gives q >= 3/2 = q_CLT (matches CLT, doesn't exceed it)
2. The gap is between k_* ~ Re^(6/7) and k_d ~ Re^(3/4)
3. Both scales are in the DISSIPATION range (k ~ k_d), not inertial
4. At those scales, delta(k) ~ 1/c ~ 0.2 -- Berry frustration is PARTIAL
"""

# =====================================================================
# SECTION 6: The Real Picture — Where the 3/28 Gap Lives
# =====================================================================
"""
THE 3/28 GAP IN FOURIER-ANGULAR COORDINATES

The gap is between:
  k_* = k_d * Re^(3/28)    (CLT regularity scale)
  k_d = 1/eta               (Kolmogorov dissipation scale)

At k_*: delta = 1/(k_* * R0) = 1/(k_d * Re^(3/28) * c*eta)
       = 1/(c * Re^(3/28))

For Re_lambda = 10^4: Re^(3/28) ~ 10^(12/28) ~ 10^0.43 ~ 2.7
  delta(k_*) ~ 1/(5 * 2.7) ~ 0.074

For Re_lambda = 10^6: Re^(3/28) ~ 10^(18/28) ~ 10^0.64 ~ 4.4
  delta(k_*) ~ 1/(5 * 4.4) ~ 0.045

So at the CLT scale k_*, tubes ARE pencils (delta < 0.1).
Berry frustration IS weak there.
The pencil loophole from g1 applies AT THE RIGHT SCALE.

Wait -- but above I said delta(k) > 1 throughout the inertial range.
Let me reconcile:

  k_* is NOT in the inertial range. k_* > k_d (it's in the dissipation range).
  The 3/28 gap is ABOVE k_d: k_d < k < k_* = k_d * Re^(3/28).

  At k = k_d: delta ~ 1/c ~ 0.2
  At k = k_*: delta ~ 1/(c * Re^(3/28)) ~ 0.05-0.07

So the gap lives in the DISSIPATION RANGE where:
  1. Tubes ARE pencils (delta ~ 0.05-0.2)
  2. Berry frustration IS weak (pencil loophole applies)
  3. BUT viscosity is also active

The question becomes: does viscosity close the gap before Berry
frustration becomes irrelevant?

ANSWER: This is exactly the CLT vs viscous dissipation race that
defines the 3/28 gap. Berry frustration adds nothing new here.
  - CLT says: random phases give convergent enstrophy up to k_*
  - Viscosity says: kills everything above k_d
  - Gap: k_d to k_* where neither CLT nor viscosity fully controls

Berry frustration at those scales: WEAK (pencil loophole active).
Anti-twist at those scales: just says tubes have width ~ eta = 1/k_d,
which we already knew.

NO NEW INFORMATION from either anti-twist or Berry at the gap scales.
"""

# Final quantitative summary
print()
print("=" * 70)
print("FINAL ASSESSMENT: ANTI-TWIST + BERRY SYNERGY (OPTION A)")
print("=" * 70)
print()
print("STATUS: DEAD")
print()
print("REASON:")
print("  1. Buaria 2024: anti-twist is qualitative, no bounds on delta(k)")
print("  2. DNS consensus: R0 ~ 5*eta, independent of Re")
print("  3. Tube angular width delta(k) = 1/(k*R0):")
print("     - Inertial range (k << k_d): delta >> 1 (isotropic)")
print("     - Berry frustration active but only gives q >= 3/2 = q_CLT")
print("     - Gap scales (k_d < k < k_*): delta ~ 0.05-0.2 (pencils)")
print("     - Pencil loophole active, Berry frustration weak")
print("  4. Anti-twist = viscous regularization in vortex-aligned frame")
print("     - Not a new mechanism independent of viscosity")
print("     - Cannot provide bounds beyond what viscosity gives")
print()
print("WHAT WE LEARNED:")
print("  - g1 pencil loophole had wrong delta(k) in inertial range")
print("  - Corrected: tubes are ISOTROPIC at inertial k (delta >> 1)")
print("  - Berry frustration IS active in inertial range (confirms q >= 3/2)")
print("  - But the gap lives at k > k_d where pencils form AND viscosity acts")
print("  - Neither Berry nor anti-twist gives information beyond CLT + viscosity")
print()
print("OPTION A REQUIREMENT: delta >= k^(-1/2+eps) with eps > 3/56")
print("REALITY: delta(k) = 1/(c*eta*k) = k_d/(c*k)")
print("  At k_*: delta = k_d/(c*k_*) = 1/(c*Re^(3/28))")
print("  This is LESS than k^(-1/2+eps) for any eps > 0 at high Re.")
print("  Option A is impossible given the observed tube width scaling.")
print()
print("REMAINING VIABLE APPROACHES:")
print("  - Phase coherence control (Approach a): direct measurement")
print("  - Structural identity (Approach b): Miller decomposition")
print("  - Something completely different (Approach d): CKN, Lean4")
print("  - Optics/Mueller (Approach e): Universal Depolarization Theorem")
print("  - Berry frustration in DISSIPATION range: new question")
print("    (Does Berry interact with viscosity at k ~ k_d?)")
print()

# =====================================================================
# SECTION 7: One New Idea — Berry + Viscosity at k_d
# =====================================================================
"""
SPECULATIVE: Berry frustration in the viscous range

At k ~ k_d, delta ~ 1/c ~ 0.2. Berry frustration is partial but not zero.
Viscosity provides exponential decay: E(k) ~ exp(-(k/k_d)^2) for k > k_d.

Question: Does Berry frustration ACCELERATE the viscous decay?

If Berry frustration provides an ADDITIONAL decoherence mechanism at k ~ k_d,
the effective dissipation might increase, pushing k_d higher and narrowing
the 3/28 gap. This is qualitatively different from Option A.

Rough estimate:
  - Berry decoherence at k ~ k_d: phase spread ~ 2*pi * (1 - cos(delta))
    For delta = 0.2: phase spread ~ 2*pi * 0.02 ~ 0.13 rad
    This is small (< 1) — Berry is PERTURBATIVE at k_d

  - Correction to viscous decay rate: nu_eff ~ nu * (1 + Berry correction)
    Berry correction ~ (phase spread)^2 ~ 0.02
    2% correction to viscosity — NEGLIGIBLE

CONCLUSION: Even Berry + viscosity doesn't help. The 3/28 gap stands.

FINAL HONEST ASSESSMENT:
  The 3/28 gap is a SCALING gap, not a constant gap. No mechanism that
  improves constants (Berry, anti-twist, sin^2(theta)/4) can close it.
  Only a mechanism that changes the EXPONENT of k_* or k_d can close it.
  This requires either:
  (a) A proof that coherent structures cannot form (equivalent to Millennium)
  (b) A new symmetry/conservation law that constrains the exponent
  (c) A completely different approach to regularity
"""

# =====================================================================
# SECTION 8: Summary Table
# =====================================================================
print()
print("=" * 70)
print("APPROACH (g) COMPLETE SUMMARY")
print("=" * 70)
print()
print(f"{'Sub-approach':>25} {'Status':>12} {'Gap impact':>15}")
print("-" * 55)
results = [
    ("g1: Berry frustration", "NEGATIVE", "None (pencils)"),
    ("g1b: Anti-twist synergy", "DEAD", "None (no bounds)"),
    ("g1b: Corrected pencils", "REVISED", "Berry active in IR"),
    ("g1b: Berry + viscosity", "NEGLIGIBLE", "~2% correction"),
    ("g1: p_crit improvement", "CONFIRMED", "p=1 isotropic only"),
    ("g1: Three-level picture", "STRUCTURAL", "Clarifies taxonomy"),
]
for name, status, impact in results:
    print(f"{name:>25} {status:>12} {impact:>15}")

print()
print("APPROACH (g) VERDICT: Closed. Berry phase of helical basis is a")
print("genuinely novel computation (Chern number 2) with structural value,")
print("but provides no path to closing the 3/28 gap. The gap is a scaling")
print("gap; all mechanisms explored improve constants only.")
print()
print("KEY CORRECTION from g1b: The pencil loophole in g1 used wrong delta(k).")
print("Tubes are ISOTROPIC in Fourier space at inertial k. Berry frustration")
print("IS active there (confirms q >= 3/2). But at gap scales (k > k_d),")
print("tubes become pencils AND viscosity dominates. No new information.")
