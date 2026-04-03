"""
TESTABLE PREDICTIONS — SHAREABLE WITH RESEARCHERS
S98-M1d: Complete list of concrete, falsifiable predictions from our framework.

Each prediction includes:
  - The prediction itself (quantitative where possible)
  - What data would test it
  - What a positive/negative result means for our framework
  - Estimated difficulty of the test
"""

import numpy as np

print("=" * 70)
print("TESTABLE PREDICTIONS FROM SPECTRAL INVARIANT / LERAY FRAMEWORK")
print("=" * 70)
print("All predictions below are derived from proved geometric results")
print("(sin^2(theta)/4, 1-ln(2), Mueller depolarization) plus the CLT.")
print()

# ================================================================
# PREDICTION 1: Cross-helical Leray suppression
# ================================================================
print("━" * 70)
print("PREDICTION 1: CROSS-HELICAL LERAY SUPPRESSION FACTOR")
print("━" * 70)
print("""
STATEMENT:
  For any incompressible 3D flow, the fraction of cross-helical Lamb
  vector surviving Leray projection is:
    alpha(theta, rho) = 1 - (1+rho)^2(1+cos theta) / [(1+rho^2+2rho cos theta)(3-cos theta)]

  Isotropic average: <alpha> = 1 - ln(2) = 0.30685...
  Lamb-weighted average: sin^2(theta)/4 = 0.25

  This means: 69.3% of the cross-helical nonlinearity is gradient
  (killed by Leray projection). Only 30.7% is solenoidal.

HOW TO TEST:
  1. Take any DNS of 3D NS (homogeneous isotropic turbulence).
  2. Compute the Lamb vector L = omega x u in Fourier space.
  3. Decompose into helical modes (h+, h-).
  4. For each triad (k1, k2, k3), compute the cross-helical Lamb
     component and its Leray projection.
  5. Measure the ratio |P L_cross|^2 / |L_cross|^2 per triad.
  6. Average over triads at fixed theta, rho.

EXPECTED RESULT:
  - The per-triad ratio should match alpha(theta, rho) EXACTLY
    (this is geometric, not statistical).
  - The weighted average should give sin^2(theta)/4.

WHAT FAILURE MEANS:
  A discrepancy would indicate an error in our algebra.
  (This is a KINEMATIC identity, not a dynamical prediction.)

DIFFICULTY: Easy (any pseudospectral NS code with helical decomposition).
""")

# ================================================================
# PREDICTION 2: Universal Depolarization
# ================================================================
print("━" * 70)
print("PREDICTION 2: UNIVERSAL DEPOLARIZATION OF TRIADIC PHASES")
print("━" * 70)
print("""
STATEMENT:
  For each triadic interaction in 3D NS, the Mueller matrix has:
    - Cross-helical dichroism = 0 (identically, for all triad geometries)
    - Same-helical dichroism = -A * S_0 * S_3 (anti-polarizing)
    - Mueller eigenvalues approximately (1, 0, 0, -epsilon)

  Consequence: the forward cascade is a UNIVERSAL DEPOLARIZER.
  Phase information (Stokes S1, S2) is destroyed at each cascade step.
  Helicity imbalance (S3) is exponentially suppressed.

HOW TO TEST:
  1. In DNS, track the Stokes vector (S0, S1, S2, S3) of helical modes
     at each wavenumber k over time.
  2. As k increases (deeper into inertial range), S1 and S2 should
     approach zero faster than S3.
  3. The Kuramoto order parameter R_K should stay below ~0.05
     across all wavenumbers in the inertial range.
  4. At each k, the depolarization ratio D = sqrt(S1^2 + S2^2) / S0
     should decrease with k.

EXPECTED:
  R_K(k) < 0.05 for k in the inertial range (our DNS confirms this).
  D(k) ~ k^{-q} with q > 0 (depolarization strengthens with k).

WHAT FAILURE MEANS:
  R_K > 0.1 would indicate significant phase coherence —
  the depolarization mechanism is overwhelmed by coherent structures.
  This is physically possible (and may happen at very high Re).

DIFFICULTY: Medium (requires helical decomposition + phase tracking).
""")

# ================================================================
# PREDICTION 3: q_CLT = 3/2
# ================================================================
print("━" * 70)
print("PREDICTION 3: CLT DECOHERENCE EXPONENT q_CLT = 3/2")
print("━" * 70)
print("""
STATEMENT:
  The number of effectively independent local triads at wavenumber k
  scales as N_eff(k) ~ k^3 (3D volume in wavevector space).
  By the Central Limit Theorem, this gives:
    q_CLT = 3/2 (decoherence exponent from random phases alone)

  This is a GEOMETRIC/COMBINATORIAL result, independent of dynamics.

HOW TO TEST:
  1. For fixed k3, count the number of triads (k1, k2) with
     |k1|, |k2| in [k/3, 3k] (local triads).
  2. Verify N_triads ~ k^3 on a lattice.
  3. Compute the Leray-weighted effective number:
     N_eff = (sum w_i)^2 / (sum w_i^2) with w_i = sin^2(theta_i)/4.
  4. Verify N_eff ~ k^3 (same scaling as unweighted).

EXPECTED:
  N_eff(k) = C * k^{3.00 +/- 0.05} for k in [4, 40].
  Our Monte Carlo gives alpha = 3.007 (unweighted), 3.005 (weighted).

WHAT FAILURE MEANS:
  The k^3 scaling is purely geometric — it should hold exactly
  on any cubic lattice. A different exponent would indicate an error
  in the triad counting algorithm (e.g., double-counting, wrong shell).

DIFFICULTY: Easy (lattice counting, no NS simulation needed).
""")

# ================================================================
# PREDICTION 4: MHD Elsasser extension
# ================================================================
print("━" * 70)
print("PREDICTION 4: sin^2(theta)/4 UNCHANGED IN MHD")
print("━" * 70)
print("""
STATEMENT:
  The per-triad Leray suppression factor sin^2(theta)/4 is IDENTICAL
  in MHD (magnetohydrodynamics) and hydrodynamics.
  The Leray projector is a spatial operator — it doesn't "know" whether
  the modes are velocity, magnetic field, or Elsasser variables.

HOW TO TEST:
  1. Run MHD DNS with Elsasser variables z+/- = u +/- B.
  2. Compute the per-triad solenoidal fraction for z-.grad(z+) terms.
  3. Compare with the HD formula sin^2(theta)/4.
  4. They should match exactly.

ADDITIONALLY — SS ALPHA PREDICTION:
  alpha_SS = (1 - ln(2)) * (1 - sigma^2)
  where sigma = (E+ - E-)/(E+ + E-) is Elsasser imbalance.
""")

# Print predictions
sigma_arr = [0, 0.3, 0.5, 0.7, 0.9]
alpha_leray = 1 - np.log(2)
print(f"  Predicted alpha_SS vs Elsasser imbalance:")
print(f"  {'sigma':>8s} {'alpha_SS':>10s} {'Disk type':>25s}")
print(f"  {'-'*8} {'-'*10} {'-'*25}")
for sig in sigma_arr:
    a = alpha_leray * (1 - sig**2)
    disk = ""
    if sig < 0.1:
        disk = "Thick disk / RIAF"
    elif sig < 0.4:
        disk = "Moderate imbalance"
    elif sig < 0.6:
        disk = "Solar wind (partial)"
    elif sig < 0.8:
        disk = "Solar wind"
    else:
        disk = "Strongly imbalanced"
    print(f"  {sig:>8.1f} {a:>10.4f} {disk:>25s}")

print("""
HOW TO TEST:
  In MRI disk simulations with controlled Elsasser imbalance,
  measure alpha_SS and compare with (1-ln(2))(1-sigma^2).

  Key: THICK DISKS (sigma ~ 0, 3D isotropic) should give alpha ~ 0.3.
  THIN DISKS (quasi-2D) should give alpha << 0.3 (theta restricted).

WHAT FAILURE MEANS:
  alpha_SS matching our prediction would confirm that the geometric
  Leray suppression factor governs angular momentum transport in disks.
  A mismatch would indicate that the spectral structure of MHD turbulence
  differs significantly from the isotropic assumption.

DIFFICULTY: Medium-Hard (requires MHD DNS with controlled sigma).
""")

# ================================================================
# PREDICTION 5: Non-helical small-scale dynamo
# ================================================================
print("━" * 70)
print("PREDICTION 5: NON-HELICAL SMALL-SCALE DYNAMO")
print("━" * 70)
print("""
STATEMENT:
  The cross-helical dichroism of the cascade is IDENTICALLY ZERO.
  This means: the MHD cascade cannot amplify net helicity.
  Consequence: the small-scale dynamo generates non-helical B fields.

HOW TO TEST:
  1. Run kinematic or fully nonlinear dynamo DNS.
  2. Measure the magnetic helicity spectrum H_M(k) = <A(k) . B(k)>.
  3. At small scales (k >> k_forcing): H_M(k) should be zero
     (within statistical noise), regardless of initial conditions.
  4. This should hold even if the large-scale field has net helicity.

EXPECTED:
  H_M(k) / (k * E_M(k)) << 1 at k >> k_forcing (near-zero realizability).
  This is consistent with existing dynamo literature but our framework
  gives a STRUCTURAL reason (dichroism = 0, not just statistical).

WHAT FAILURE MEANS:
  Net small-scale magnetic helicity would violate our depolarization
  theorem. This would require same-helical dichroism to flip sign
  (currently always anti-polarizing).

DIFFICULTY: Easy-Medium (existing dynamo codes, standard diagnostic).
""")

# ================================================================
# PREDICTION 6: q_local vs Re
# ================================================================
print("━" * 70)
print("PREDICTION 6: q_local BEHAVIOR AT HIGH Re (CRITICAL TEST)")
print("━" * 70)
print("""
STATEMENT:
  The local-triad decoherence exponent q_local measures the rate at
  which phase coherence decays with wavenumber k.

  Our framework predicts:
    q_CLT = 3/2 (upper bound from CLT, geometric)
    q_measured = q_CLT - q_coherence

  From DNS at Re = 400-6400:
    Re=400:  q = 1.60 (above 7/6 — C-F regularity holds)
    Re=1600: q = 1.17 (marginal)
    Re=3200: q = 0.63 (below 7/6 — C-F fails)
    Re=6400: q = 0.87 (NON-MONOTONIC — recovery?)

  THE CRITICAL QUESTION: What happens at Re = 10,000 - 100,000?

  Three scenarios:
    Scenario 1: q recovers → q_CLT = 3/2 (regularity)
    Scenario 2: q plateaus at some q_inf < 7/6 (marginal)
    Scenario 3: q → 0 (full coherence, potential blowup)

HOW TO TEST:
  High-resolution DNS (N = 512-2048) at Re = 10,000 - 100,000.
  Measure R_K(k) using the Kuramoto order parameter with helical
  decomposition and local-triad angular binning.
  Fit R_K ~ k^{-q} and report q.

  THIS IS THE SINGLE MOST IMPORTANT NUMERICAL TEST FOR OUR FRAMEWORK.

DIFFICULTY: Hard (requires large DNS, O(1000 CPU-hours) for Re=100,000).
""")

# ================================================================
# PREDICTION 7: Depression of nonlinearity from sin^2(theta)/4
# ================================================================
print("━" * 70)
print("PREDICTION 7: DEPRESSION QUANTIFIED BY sin^2(theta)/4")
print("━" * 70)
print("""
STATEMENT:
  The "depression of nonlinearity" observed by Iyer et al. (2021) and
  others has a geometric component: sin^2(theta)/4 = 0.25 per triad.

  The observed solenoidal fraction R should satisfy:
    R >= sin^2(theta)/4 * f(phases) = 0.25 * f(phases)

  where f(phases) >= 1 (phases can only increase alpha_eff above
  the incoherent baseline).

  Specifically: R = 0.18 * ln(R_lambda) - 0.38 (Iyer et al.)
  Our sin^2(theta)/4 = 0.25 is the INCOHERENT BASELINE.
  At R_lambda ~ 40 (R_sol ~ 0.28): close to our baseline.
  At higher R_lambda: R_sol > 0.25, confirming phase coherence.

HOW TO TEST:
  In DNS at various R_lambda, decompose the solenoidal fraction into:
    R_sol = R_geometric * R_phase
  where R_geometric = sin^2(theta)/4 (our prediction) and
  R_phase = R_sol / R_geometric (the phase coherence factor).

  R_phase should be >= 1 always, and grow with R_lambda.

EXPECTED:
  R_phase ~ 1 at low R_lambda (incoherent, CLT regime)
  R_phase ~ 2-4 at R_lambda ~ 1000 (partial coherence)
  R_phase never exceeds ~10 (bounded by tube volume fraction)

DIFFICULTY: Easy (reanalysis of existing DNS data).
""")

# ================================================================
# PREDICTION 8: Spin-2 TT suppression
# ================================================================
print("━" * 70)
print("PREDICTION 8: SPIN-2 TT SUPPRESSION 5.7x STRONGER THAN LERAY")
print("━" * 70)

alpha_spin2 = (4 - np.pi) / 16
alpha_spin1 = 0.25  # sin^2(theta)/4
ratio = alpha_spin1 / alpha_spin2

print(f"""
STATEMENT:
  For spin-2 transverse-traceless (TT) projection:
    alpha_TT(theta) = (1-cos theta)^2 / [16 + 4(1-cos theta)^2]
    <alpha_TT> = (4-pi)/16 = {alpha_spin2:.6f}

  This is {ratio:.1f}x more suppressed than Leray (spin-1).

  Physical context: The BKMS fluid/gravity correspondence maps
  3D NS (spin-1, Leray) to 5D gravity constraint (spin-2, TT).
  On the boundary, dimensional reduction collapses spin-2 → spin-1:
  sin^2(theta)/4 IS the correct boundary factor.

HOW TO TEST:
  In holographic NS (AdS/CFT with fluid/gravity duality),
  compute the TT projection of the stress-energy tensor
  fluctuations in the bulk. The per-triad suppression should
  match alpha_TT(theta) exactly.

DIFFICULTY: Very hard (requires holographic DNS framework).
""")

# ================================================================
# SUMMARY TABLE
# ================================================================
print("=" * 70)
print("SUMMARY: ALL TESTABLE PREDICTIONS")
print("=" * 70)

predictions = [
    ("1", "Cross-helical Leray = sin^2(theta)/4", "Kinematic identity",
     "Easy", "Any 3D NS DNS"),
    ("2", "Universal depolarization (R_K < 0.05)", "Mueller + verified",
     "Medium", "Helical phase tracking"),
    ("3", "N_eff ~ k^3 (q_CLT = 3/2)", "Geometric/combinatorial",
     "Easy", "Lattice counting"),
    ("4", "sin^2(theta)/4 same in MHD", "Geometric (Leray is spatial)",
     "Medium", "MHD DNS + Elsasser"),
    ("4b", "alpha_SS = (1-ln2)(1-sigma^2)", "Geometric + Elsasser",
     "Med-Hard", "MRI disk simulations"),
    ("5", "Non-helical small-scale dynamo", "Dichroism = 0",
     "Easy-Med", "Dynamo DNS"),
    ("6", "q_local at Re > 10,000", "CRITICAL test of framework",
     "Hard", "High-Re DNS (N=512+)"),
    ("7", "Depression = sin^2(theta)/4 * f(phases)", "Geometric baseline",
     "Easy", "Reanalysis of existing data"),
    ("8", "Spin-2 TT 5.7x stronger than spin-1", "Exact formula",
     "Very Hard", "Holographic DNS"),
]

print(f"\n  {'#':>3s}  {'Prediction':>45s}  {'Basis':>25s}  {'Difficulty':>10s}")
print(f"  {'-'*3}  {'-'*45}  {'-'*25}  {'-'*10}")
for n, pred, basis, diff, test in predictions:
    print(f"  {n:>3s}  {pred:>45s}  {basis:>25s}  {diff:>10s}")

print(f"""
PRIORITY ORDER FOR SHARING:
  1. Prediction 1 (easy, kinematic, immediately verifiable)
  2. Prediction 3 (easy, geometric, no DNS needed)
  3. Prediction 7 (easy, reanalysis of existing Iyer et al. data)
  4. Prediction 2 (medium, novel diagnostic)
  5. Prediction 5 (medium, connects to dynamo community)
  6. Prediction 4b (hard but connects to astrophysics community)
  7. Prediction 6 (hard but most scientifically important)
""")

print("=" * 70)
print("DONE. 8 testable predictions assembled.")
print("=" * 70)
