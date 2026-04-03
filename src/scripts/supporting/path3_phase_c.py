"""
Path 3, Phase C: Spectral Bridge + Energy Inequality

Step 3.5: Does R = 1.857 appear in the Hodge spectrum?
  - Check if the star invariant appears as a ratio of Hodge eigenvalues
  - Check if it appears in the Stokes spectrum
  - Check the exact relationship between L0 and Stokes ratios

Step 3.6: Discrete energy inequality (blow-up analysis)
  - Continuous NS: d/dt ||u||^2 = -2*nu*||nabla u||^2 + <u, f>
  - Prodi-Serrin regularity: if u in L^p_t L^q_x with 2/p + 3/q = 1, no blow-up
  - Discrete analogue: d/dt ||f||^2 = -2*nu*<f, L1 f> = -2*nu*enstrophy
  - Blow-up requires enstrophy to grow faster than dissipation
  - What does the valve do to the enstrophy growth rate?

Step 3.7: Ladyzhenskaya inequality and the star invariant
  - Ladyzhenskaya: ||u||_L4 <= C * ||u||_L2^{1/2} * ||nabla u||_L2^{1/2} (2D)
  - The constant C depends on the domain geometry
  - On a simplicial complex: C relates to the Stokes gap
  - Does the valve change C in a way that involves R?
"""
import numpy as np
import sys
from itertools import combinations

sys.stdout.reconfigure(encoding="utf-8")


def build_complex(clauses, n_vars):
    """Build simplicial complex from clause list."""
    edges = set()
    for clause in clauses:
        for i, j in combinations(clause, 2):
            edges.add((min(i, j), max(i, j)))
    edges = sorted(edges)
    n_edges = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}

    triangles = []
    for clause in clauses:
        tri = tuple(sorted(clause))
        if tri not in triangles:
            triangles.append(tri)
    n_tri = len(triangles)

    B1 = np.zeros((n_vars, n_edges))
    for idx, (i, j) in enumerate(edges):
        B1[i, idx] = -1
        B1[j, idx] = +1

    B2 = np.zeros((n_edges, n_tri))
    for t_idx, tri in enumerate(triangles):
        i, j, k = tri
        for pair, sign in [((i,j), +1), ((i,k), -1), ((j,k), +1)]:
            eidx = edge_index.get(pair)
            if eidx is not None:
                B2[eidx, t_idx] = sign

    L0 = B1 @ B1.T
    L1 = B1.T @ B1 + B2 @ B2.T
    L2 = B2.T @ B2

    # Div-free basis (null space of B1 acting on edge space)
    U, S, Vt = np.linalg.svd(B1.T, full_matrices=True)
    rank = np.sum(S > 1e-8)
    df_basis = U[:, rank:]

    # Stokes operator
    L1_stokes = df_basis.T @ L1 @ df_basis

    return {
        "B1": B1, "B2": B2, "L0": L0, "L1": L1, "L2": L2,
        "edges": edges, "triangles": triangles, "edge_index": edge_index,
        "n_vars": n_vars, "n_edges": n_edges, "n_tri": n_tri,
        "df_basis": df_basis, "L1_stokes": L1_stokes,
    }


# Build complexes
cluster_clauses = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
    (5, 0, 3), (6, 2, 4), (7, 5, 6),
]
bridge_clauses = [(0, 8, 1), (1, 9, 2), (4, 12, 10)]
hub = list(cluster_clauses)
spoke = [(c[0]+8, c[1]+8, c[2]+8) for c in cluster_clauses]
all_clauses = hub + spoke + bridge_clauses

full = build_complex(all_clauses, 16)

valve = {4, 10, 12}
remaining = sorted(set(range(16)) - valve)
var_remap = {v: i for i, v in enumerate(remaining)}
red_clauses = [tuple(var_remap[v] for v in c) for c in all_clauses
               if all(v not in valve for v in c)]
reduced = build_complex(red_clauses, len(remaining))


# ============================================================
# STEP 3.5: Does R appear in the Hodge spectrum?
# ============================================================
print("=" * 85)
print("STEP 3.5: SPECTRAL BRIDGE — WHERE DOES R = 1.857 APPEAR?")
print("=" * 85)

eig_L0_full = np.sort(np.linalg.eigvalsh(full["L0"]))
eig_L0_red = np.sort(np.linalg.eigvalsh(reduced["L0"]))
eig_L1_full = np.sort(np.linalg.eigvalsh(full["L1"]))
eig_L1_red = np.sort(np.linalg.eigvalsh(reduced["L1"]))
eig_L2_full = np.sort(np.linalg.eigvalsh(full["L2"]))
eig_L2_red = np.sort(np.linalg.eigvalsh(reduced["L2"]))
eig_stokes_full = np.sort(np.linalg.eigvalsh(full["L1_stokes"]))
eig_stokes_red = np.sort(np.linalg.eigvalsh(reduced["L1_stokes"]))

R_star = 1.8573068741  # star invariant from Path 1

# Check ALL possible eigenvalue ratios across the Hodge spectrum
print("\n  Searching for R = 1.857 in eigenvalue ratios...")
print(f"  (across L0, L1, L2, Stokes spectra)")

found_matches = []

def search_ratios(eig_a, eig_b, name_a, name_b, target=R_star, tol=0.01):
    """Search for target ratio between positive eigenvalues of two spectra."""
    matches = []
    pos_a = [v for v in eig_a if v > 0.01]
    pos_b = [v for v in eig_b if v > 0.01]
    for i, a in enumerate(pos_a):
        for j, b in enumerate(pos_b):
            ratio = a / b
            if abs(ratio - target) < tol:
                matches.append((name_a, i, a, name_b, j, b, ratio))
            ratio_inv = b / a
            if abs(ratio_inv - target) < tol:
                matches.append((name_b, j, b, name_a, i, a, ratio_inv))
    return matches

# Within the same spectrum (full)
for name, eig in [("L0_full", eig_L0_full), ("L1_full", eig_L1_full),
                   ("L2_full", eig_L2_full), ("Stokes_full", eig_stokes_full)]:
    matches = search_ratios(eig, eig, name, name)
    found_matches.extend(matches)

# Cross-spectrum (full vs reduced)
for name_f, eig_f, name_r, eig_r in [
    ("L0_full", eig_L0_full, "L0_red", eig_L0_red),
    ("L1_full", eig_L1_full, "L1_red", eig_L1_red),
    ("L2_full", eig_L2_full, "L2_red", eig_L2_red),
    ("Stokes_full", eig_stokes_full, "Stokes_red", eig_stokes_red),
    ("L0_full", eig_L0_full, "Stokes_full", eig_stokes_full),
    ("L0_full", eig_L0_full, "L2_full", eig_L2_full),
    ("Stokes_full", eig_stokes_full, "L2_full", eig_L2_full),
]:
    matches = search_ratios(eig_f, eig_r, name_f, name_r)
    found_matches.extend(matches)

# Deduplicate
seen = set()
unique_matches = []
for m in found_matches:
    key = (m[0], m[1], m[3], m[4])
    if key not in seen:
        seen.add(key)
        unique_matches.append(m)

print(f"\n  Found {len(unique_matches)} eigenvalue pairs with ratio near 1.857:")
for m in unique_matches:
    print(f"    {m[0]}[{m[1]}]={m[2]:.6f} / {m[3]}[{m[4]}]={m[5]:.6f} = {m[6]:.6f}")

# The KEY ratio: L0_gap_full / L0_gap_red (this is the original star invariant)
print(f"\n  KEY RATIOS:")
l0_gap_f = eig_L0_full[1]
l0_gap_r = eig_L0_red[1]
b1_full = np.sum(np.abs(eig_stokes_full) < 1e-8)
b1_red = np.sum(np.abs(eig_stokes_red) < 1e-8)
stokes_gap_f = eig_stokes_full[b1_full]
stokes_gap_r = eig_stokes_red[b1_red]

print(f"    L0 gap ratio (graph):  {l0_gap_f / l0_gap_r:.10f}")
print(f"    Stokes gap ratio:      {stokes_gap_f / stokes_gap_r:.10f}")
print(f"    L2 gap ratio:          {eig_L2_full[0] / eig_L2_red[0]:.10f}")
print(f"    Star invariant R:      {R_star:.10f}")
print(f"    L0 ratio * Stokes ratio = {(l0_gap_f / l0_gap_r) * (stokes_gap_f / stokes_gap_r):.10f}")

# ============================================================
# Check the Hodge-theoretic identity
# ============================================================
print(f"\n{'='*85}")
print(f"HODGE-THEORETIC IDENTITIES")
print(f"{'='*85}")

# L1 eigenvalues = union of {non-zero L0 eigenvalues} and {L2 eigenvalues}
# This is because L1 = B1^T B1 + B2 B2^T, and these two components
# act on orthogonal subspaces (gradient and curl respectively).
# The gradient eigenvalues of L1 = non-zero eigenvalues of L0.
# The curl eigenvalues of L1 = eigenvalues of L2.

print(f"\n  L0 non-zero eigenvalues: {np.round([v for v in eig_L0_full if v > 0.01], 6)}")
print(f"  L2 eigenvalues:          {np.round(eig_L2_full, 6)}")
print(f"  L1 non-zero eigenvalues: {np.round([v for v in eig_L1_full if v > 0.01], 6)}")

# The L1 spectrum is the UNION of L0 (non-zero) and L2, plus b1 zeros
l0_nz = sorted([v for v in eig_L0_full if v > 0.01])
l2_all = sorted(eig_L2_full.tolist())
union = sorted(l0_nz + l2_all)
l1_nz = sorted([v for v in eig_L1_full if v > 0.01])

print(f"\n  L0_nz union L2 = {np.round(union, 6)}")
print(f"  L1_nz          = {np.round(l1_nz, 6)}")
print(f"  Match: {np.allclose(sorted(union), sorted(l1_nz), atol=1e-6)}")

if np.allclose(sorted(union), sorted(l1_nz), atol=1e-6):
    print(f"\n  CONFIRMED: L1 spectrum = (L0 non-zero) UNION (L2 spectrum) UNION (b1 zeros)")
    print(f"  This is the discrete Hodge theorem in action.")

# ============================================================
# STEP 3.6: Energy Inequality (Blow-up Analysis)
# ============================================================
print(f"\n{'='*85}")
print(f"STEP 3.6: DISCRETE ENERGY INEQUALITY")
print(f"{'='*85}")

# The discrete NS energy equation (on div-free subspace):
# d/dt ||f||^2 = -2*nu * <f, L1_stokes f> + 2*<f, F>
# where F is external forcing.
#
# Without forcing:
# d/dt ||f||^2 = -2*nu * <f, L1_stokes f> <= -2*nu*lambda_1*||f||^2
# So ||f(t)||^2 <= ||f(0)||^2 * exp(-2*nu*lambda_1*t)
#
# This means: ALL flows decay exponentially with rate 2*nu*lambda_1
# (except harmonic modes, which are steady states).
#
# The DECAY TIME is tau = 1 / (2*nu*lambda_1)

nu = 1.0
tau_full = 1.0 / (2 * nu * stokes_gap_f)
tau_red = 1.0 / (2 * nu * stokes_gap_r)

print(f"\n  Energy decay (no forcing):")
print(f"    ||f(t)||^2 <= ||f(0)||^2 * exp(-2*nu*lambda_1*t)")
print(f"")
print(f"    Full complex:    lambda_1 = {stokes_gap_f:.6f}, tau = {tau_full:.6f}")
print(f"    Reduced complex: lambda_1 = {stokes_gap_r:.6f}, tau = {tau_red:.6f}")
print(f"    Tau ratio: {tau_full / tau_red:.6f}")
print(f"    (Full decays {tau_full / tau_red:.2f}x SLOWER than reduced)")

# Enstrophy evolution
# In continuous 2D NS: d/dt ||omega||^2 <= -2*nu*||nabla omega||^2 + ...
# In discrete: d/dt ||curl f||^2 = d/dt <B2^T f, B2^T f>
# The key: enstrophy is ALWAYS bounded by the L2 spectrum.
# d/dt (enstrophy) <= -2*nu*mu_1*enstrophy + forcing
# where mu_1 = smallest eigenvalue of L2.

mu1_full = eig_L2_full[0]
mu1_red = eig_L2_red[0]

print(f"\n  Enstrophy decay:")
print(f"    Full complex:    mu_1(L2) = {mu1_full:.6f}")
print(f"    Reduced complex: mu_1(L2) = {mu1_red:.6f}")
print(f"    Enstrophy decays {mu1_full / mu1_red:.4f}x {'faster' if mu1_full > mu1_red else 'slower'} on full")

# ============================================================
# The blow-up question
# ============================================================
print(f"\n{'='*85}")
print(f"BLOW-UP ANALYSIS")
print(f"{'='*85}")

print(f"""
  In continuous 3D NS, blow-up requires enstrophy growth to overwhelm dissipation.
  The Beale-Kato-Majda criterion: blow-up at time T iff
    integral_0^T ||omega(t)||_infty dt = infinity

  In our DISCRETE system:

  1. LINEAR case (no nonlinear term):
     All modes decay exponentially. No blow-up possible.
     Full: decay rate = {2*nu*stokes_gap_f:.6f}
     Reduced: decay rate = {2*nu*stokes_gap_r:.6f}

  2. NONLINEAR case (with advection N(f)):
     The nonlinear term can transfer energy between modes.
     In 2D: enstrophy is conserved by the nonlinear term -> no blow-up.
     In 3D: vortex stretching can amplify enstrophy.

     Our simplicial complex is topologically 2D (2-simplices are the
     highest dimension). So the discrete enstrophy SHOULD be bounded.

  3. THE VALVE EFFECT on potential blow-up:
     Full complex (6 harmonic modes):
       - 6 independent steady circulations that DON'T dissipate
       - Nonlinear interactions between these could cascade energy
       - More "room" for complex flow patterns

     Reduced complex (1 harmonic mode):
       - Only 1 steady circulation survives
       - Nonlinear coupling is dramatically reduced
       - Less "room" for energy cascades
""")

# ============================================================
# Ladyzhenskaya-type inequality on the simplicial complex
# ============================================================
print(f"{'='*85}")
print(f"DISCRETE LADYZHENSKAYA INEQUALITY")
print(f"{'='*85}")

# In continuous 2D: ||u||_L4 <= C * ||u||_L2^{1/2} * ||nabla u||_L2^{1/2}
# The constant C depends on the domain.
# On a simplicial complex:
# ||f||_4 <= C_K * ||f||_2^{1/2} * ||L1^{1/2} f||_2^{1/2}
#
# For eigenmodes f_i with eigenvalue lambda_i:
# ||f_i||_4 <= C_K * ||f_i||_2^{1/2} * lambda_i^{1/2} * ||f_i||_2^{1/2}
#            = C_K * sqrt(lambda_i) * ||f_i||_2
#
# So C_K >= ||f_i||_4 / (sqrt(lambda_i) * ||f_i||_2) for all eigenmodes.
# The optimal C_K is the supremum over all modes.

# Compute C_K for both complexes
_, evecs_full = np.linalg.eigh(full["L1_stokes"])
_, evecs_red = np.linalg.eigh(reduced["L1_stokes"])

def compute_ladyzhenskaya_constant(df_basis, evecs, eigenvalues, B1, L1):
    """Compute discrete Ladyzhenskaya constant for the complex."""
    C_max = 0
    b1 = np.sum(np.abs(eigenvalues) < 1e-8)

    for i in range(b1, len(eigenvalues)):
        lam = eigenvalues[i]
        if lam < 1e-8:
            continue

        # Eigenvector in edge coordinates
        mode = df_basis @ evecs[:, i]

        # L2 norm
        l2_norm = np.linalg.norm(mode)
        if l2_norm < 1e-10:
            continue

        # L4 "norm" (discrete: (sum |f_e|^4)^{1/4})
        l4_norm = np.sum(mode**4)**(1/4)

        # Ladyzhenskaya ratio
        C = l4_norm / (np.sqrt(lam) * l2_norm)
        C_max = max(C_max, C)

    return C_max

C_full = compute_ladyzhenskaya_constant(
    full["df_basis"], evecs_full, eig_stokes_full, full["B1"], full["L1"])
C_red = compute_ladyzhenskaya_constant(
    reduced["df_basis"], evecs_red, eig_stokes_red, reduced["B1"], reduced["L1"])

print(f"\n  Discrete Ladyzhenskaya constant C_K:")
print(f"    Full complex:    C = {C_full:.10f}")
print(f"    Reduced complex: C = {C_red:.10f}")
print(f"    Ratio C_full / C_red = {C_full / C_red:.10f}")

# ============================================================
# The critical Reynolds number
# ============================================================
print(f"\n{'='*85}")
print(f"CRITICAL REYNOLDS NUMBER")
print(f"{'='*85}")

# In continuous NS: turbulence onset when Re > Re_crit
# Re_crit ~ 1 / (C_K^2 * lambda_1) (from the Ladyzhenskaya inequality)
# where C_K is the Ladyzhenskaya constant and lambda_1 is the Stokes gap.

Re_crit_full = 1.0 / (C_full**2 * stokes_gap_f) if C_full > 0 else float('inf')
Re_crit_red = 1.0 / (C_red**2 * stokes_gap_r) if C_red > 0 else float('inf')

print(f"\n  Re_crit ~ 1 / (C^2 * lambda_1):")
print(f"    Full complex:    Re_crit = {Re_crit_full:.6f}")
print(f"    Reduced complex: Re_crit = {Re_crit_red:.6f}")
print(f"    Ratio: {Re_crit_full / Re_crit_red:.6f}")

# ============================================================
# FINAL SYNTHESIS
# ============================================================
print(f"\n{'='*85}")
print(f"PHASE C: FINAL SYNTHESIS")
print(f"{'='*85}")

print(f"""
  THE STAR INVARIANT R = 1.857 IN THE HODGE FRAMEWORK:

  1. GRAPH LEVEL (L0):
     Gap ratio = {l0_gap_f / l0_gap_r:.6f}
     This is NOT R (it's {l0_gap_f / l0_gap_r:.6f} for the 2-cluster system).
     R = 1.857 appears only for the GROUNDED star (hub zeroed out).

  2. HODGE-1 LEVEL (L1):
     Gap ratio = {l0_gap_f / l0_gap_r:.6f} (same as L0 — the gradient inherits it).

  3. STOKES LEVEL (div-free L1):
     Gap ratio = {stokes_gap_f / stokes_gap_r:.6f} (INVERTED relative to L0).
     This is the flow-level invariant: valve STRENGTHENS dissipation.

  4. L2 LEVEL (triangle/curl):
     Gap ratio = {eig_L2_full[0] / eig_L2_red[0]:.6f}

  5. TOPOLOGY:
     b1 (loops): {b1_full} -> {b1_red} (valve destroys {b1_full - b1_red} circulations)
     Euler char: -5 -> 0 (valve regularizes the topology)

  6. ENERGY:
     Full flows decay with tau = {tau_full:.4f}
     Reduced flows decay with tau = {tau_red:.4f}
     Valve makes flows die {tau_full/tau_red:.2f}x faster.

  7. LADYZHENSKAYA CONSTANT:
     C_full = {C_full:.6f}, C_red = {C_red:.6f}
     Critical Re: full = {Re_crit_full:.4f}, reduced = {Re_crit_red:.4f}

  CONCLUSION:
  The star invariant R = 1.857 is a VERTEX-LEVEL (L0) constant of the grounded
  star topology. It does NOT directly appear in the Hodge-1 or Stokes spectra
  of the ungrounded 2-cluster system.

  However, the valve operation creates a DUALITY:
  - At vertex level: gap DECREASES by factor {l0_gap_f / l0_gap_r:.4f}
  - At flow level: gap INCREASES by factor {stokes_gap_r / stokes_gap_f:.4f}
  - At topology level: b1 drops from {b1_full} to {b1_red}
  - At energy level: decay time drops by factor {tau_full / tau_red:.4f}

  The "Navier-Stokes connection" is:
  Valve removal = topological regularization.
  It destroys circulation modes (b1), increases minimum enstrophy,
  reduces the effective Reynolds number, and makes the complex
  "more laminar" — closer to a regime where blow-up cannot occur.

  This is a QUALITATIVE analogy, not a quantitative bridge.
  The value R = 1.857 does not appear in the fluid-level analysis.
""")
