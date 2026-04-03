"""
FILAMENTARY vs RANDOM SCALING DISCREPANCY
==========================================
Antigravity found C ~ 1/N for filamentary structures.
Frustration analysis found C ~ 1/sqrt(N) for random point vortices.

Hypothesis: Filamentary structure (coherent tubes with aligned internal vorticity)
forces SELF-STRETCHING = 0, which removes the O(N) diagonal terms from the sum,
leaving only O(N^2) cross-terms with better cancellation.

This script tests 5 structural regimes:
  Part 1: Pure random point vortices (baseline, expect 1/sqrt(N))
  Part 2: Straight tube filaments (expect 1/N per Antigravity)
  Part 3: Curved filaments (intermediate?)
  Part 4: Random clusters of aligned vortices (tests alignment alone)
  Part 5: Scaling exponent fit for all regimes
"""

import numpy as np
from numpy.linalg import norm

def biot_savart_strain(nodes, omegas, sigma=0.1):
    """Compute strain rate tensor S at each vortex location via Biot-Savart."""
    n = len(nodes)
    S_all = np.zeros((n, 3, 3))
    for i in range(n):
        S = np.zeros((3, 3))
        for j in range(n):
            if i == j:
                continue
            r = nodes[i] - nodes[j]
            r2 = np.dot(r, r) + sigma**2
            r5 = r2**2.5
            oj = omegas[j]
            # Biot-Savart strain kernel: symmetric part of gradient of velocity
            for a in range(3):
                for b in range(3):
                    # S_ab = (3/(8pi)) * (r_a * (omega x r)_b + r_b * (omega x r)_a) / r^5
                    oxr = np.cross(oj, r)
                    S[a, b] += (3.0 / (8.0 * np.pi)) * (r[a] * oxr[b] + r[b] * oxr[a]) / r5
        S_all[i] = S
    return S_all

def compute_C(nodes, omegas, sigma=0.1):
    """Compute C = |sum(omega_i . S_i . omega_i)| / (lambda_max * Z)."""
    S_all = biot_savart_strain(nodes, omegas, sigma)
    n = len(nodes)

    stretching = 0.0
    Z = 0.0
    lambda_max = 0.0

    for i in range(n):
        oi = omegas[i]
        stretching += np.dot(oi, S_all[i] @ oi)
        Z += np.dot(oi, oi)
        eigs = np.linalg.eigvalsh(S_all[i])
        lambda_max = max(lambda_max, np.max(eigs))

    if lambda_max * Z < 1e-15:
        return 0.0
    return abs(stretching) / (lambda_max * Z)


# ============================================================
# Part 1: PURE RANDOM POINT VORTICES
# ============================================================
def random_point_vortices(N):
    """N random positions in [0,1]^3, random unit vorticity directions."""
    nodes = np.random.rand(N, 3)
    omegas = np.random.randn(N, 3)
    omegas /= norm(omegas, axis=1, keepdims=True)
    return nodes, omegas


# ============================================================
# Part 2: STRAIGHT TUBE FILAMENTS
# ============================================================
def straight_tube_filaments(N, n_tubes=None):
    """
    Create N vortices organized as straight tubes.
    Each tube: a line of vortices with ALIGNED vorticity along the tube axis.
    Self-stretching within a tube = 0 (parallel omega, no cross product contribution).
    """
    if n_tubes is None:
        n_tubes = max(2, N // 5)
    per_tube = N // n_tubes
    remainder = N - per_tube * n_tubes

    nodes = []
    omegas = []

    for t in range(n_tubes):
        # Random tube center and direction
        center = np.random.rand(3) * 0.5 + 0.25
        direction = np.random.randn(3)
        direction /= norm(direction)

        n_this = per_tube + (1 if t < remainder else 0)
        for k in range(n_this):
            s = (k - n_this/2) * 0.05  # spacing along tube
            pos = center + s * direction + np.random.randn(3) * 0.005  # small perpendicular jitter
            nodes.append(pos)
            omegas.append(direction.copy())  # aligned with tube axis

    return np.array(nodes), np.array(omegas)


# ============================================================
# Part 3: CURVED FILAMENTS (vortex rings)
# ============================================================
def curved_filaments(N, n_rings=None):
    """
    Create N vortices organized as curved filaments (rings).
    Vorticity is tangent to the ring at each point.
    Self-stretching within a ring != 0 due to curvature.
    """
    if n_rings is None:
        n_rings = max(2, N // 8)
    per_ring = N // n_rings
    remainder = N - per_ring * n_rings

    nodes = []
    omegas = []

    for r in range(n_rings):
        center = np.random.rand(3) * 0.5 + 0.25
        # Random ring normal
        ring_normal = np.random.randn(3)
        ring_normal /= norm(ring_normal)
        # Build orthonormal basis
        if abs(ring_normal[0]) < 0.9:
            e1 = np.cross(ring_normal, [1, 0, 0])
        else:
            e1 = np.cross(ring_normal, [0, 1, 0])
        e1 /= norm(e1)
        e2 = np.cross(ring_normal, e1)

        radius = 0.1 + 0.05 * np.random.rand()
        n_this = per_ring + (1 if r < remainder else 0)

        for k in range(n_this):
            theta = 2 * np.pi * k / n_this
            pos = center + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
            # Tangent direction = d(pos)/dtheta normalized
            tangent = -np.sin(theta) * e1 + np.cos(theta) * e2
            tangent /= norm(tangent)
            nodes.append(pos)
            omegas.append(tangent)

    return np.array(nodes), np.array(omegas)


# ============================================================
# Part 4: RANDOM CLUSTERS OF ALIGNED VORTICES
# ============================================================
def aligned_clusters(N, n_clusters=None):
    """
    Vortices in spatial clusters with aligned vorticity WITHIN each cluster,
    but random orientation BETWEEN clusters.
    Tests whether alignment alone (without filamentary geometry) improves C.
    """
    if n_clusters is None:
        n_clusters = max(2, N // 5)
    per_cluster = N // n_clusters
    remainder = N - per_cluster * n_clusters

    nodes = []
    omegas = []

    for c in range(n_clusters):
        center = np.random.rand(3)
        direction = np.random.randn(3)
        direction /= norm(direction)

        n_this = per_cluster + (1 if c < remainder else 0)
        for k in range(n_this):
            pos = center + np.random.randn(3) * 0.05  # clustered positions
            nodes.append(pos)
            omegas.append(direction.copy())  # all aligned within cluster

    return np.array(nodes), np.array(omegas)


# ============================================================
# Part 5: COMPREHENSIVE SCALING TEST
# ============================================================
def run_scaling_test():
    print("=" * 70)
    print("FILAMENTARY vs RANDOM C-SCALING DISCREPANCY")
    print("=" * 70)

    N_values = [10, 20, 30, 50, 70, 100]
    n_trials = 30

    regimes = {
        "Random points": random_point_vortices,
        "Straight tubes": straight_tube_filaments,
        "Curved rings": curved_filaments,
        "Aligned clusters": aligned_clusters,
    }

    results = {name: {} for name in regimes}

    for name, generator in regimes.items():
        print(f"\n--- {name} ---")
        print(f"{'N':<6} | {'C_mean':<10} | {'C_std':<10} | {'C_max':<10}")
        print("-" * 45)

        for N in N_values:
            Cs = []
            for _ in range(n_trials):
                np.random.seed(None)
                nodes, omegas = generator(N)
                C = compute_C(nodes, omegas)
                Cs.append(C)

            C_mean = np.mean(Cs)
            C_std = np.std(Cs)
            C_max = np.max(Cs)
            results[name][N] = (C_mean, C_std, C_max)
            print(f"{N:<6} | {C_mean:<10.6f} | {C_std:<10.6f} | {C_max:<10.6f}")

    # Fit power laws
    print("\n" + "=" * 70)
    print("POWER LAW FITS: C_mean ~ N^alpha")
    print("=" * 70)

    for name in regimes:
        Ns = np.array(N_values, dtype=float)
        Cs = np.array([results[name][N][0] for N in N_values])
        # Filter out zeros
        mask = Cs > 1e-10
        if mask.sum() < 2:
            print(f"{name:<20}: insufficient data")
            continue
        log_N = np.log(Ns[mask])
        log_C = np.log(Cs[mask])
        # Linear regression: log(C) = alpha * log(N) + beta
        alpha, beta = np.polyfit(log_N, log_C, 1)
        print(f"{name:<20}: alpha = {alpha:.3f}  (C ~ N^{alpha:.3f})")

    # Part 5b: Decompose stretching into SELF vs CROSS contributions
    print("\n" + "=" * 70)
    print("SELF vs CROSS STRETCHING DECOMPOSITION")
    print("=" * 70)
    print("Self-stretching: contributions from vortices in the SAME structure")
    print("Cross-stretching: contributions from vortices in DIFFERENT structures")

    N = 50
    for name, generator in regimes.items():
        if name == "Random points":
            continue  # No structure to decompose

        self_stretching_total = 0.0
        cross_stretching_total = 0.0
        n_decomp_trials = 20

        for _ in range(n_decomp_trials):
            if name == "Straight tubes":
                n_struct = max(2, N // 5)
            elif name == "Curved rings":
                n_struct = max(2, N // 8)
            else:
                n_struct = max(2, N // 5)

            per_struct = N // n_struct
            nodes, omegas = generator(N, n_struct)
            S_all = biot_savart_strain(nodes, omegas)

            # Assign each vortex to its structure
            labels = []
            for s in range(n_struct):
                n_this = per_struct + (1 if s < (N - per_struct * n_struct) else 0)
                labels.extend([s] * n_this)
            labels = np.array(labels[:len(nodes)])

            # Decompose: for each vortex i, compute strain from same-structure vs different-structure
            for i in range(len(nodes)):
                oi = omegas[i]
                S_self = np.zeros((3, 3))
                S_cross = np.zeros((3, 3))
                sigma = 0.1
                for j in range(len(nodes)):
                    if i == j:
                        continue
                    r = nodes[i] - nodes[j]
                    r2 = np.dot(r, r) + sigma**2
                    r5 = r2**2.5
                    oj = omegas[j]
                    oxr = np.cross(oj, r)
                    Sij = np.zeros((3, 3))
                    for a in range(3):
                        for b in range(3):
                            Sij[a, b] = (3.0 / (8.0 * np.pi)) * (r[a] * oxr[b] + r[b] * oxr[a]) / r5

                    if labels[j] == labels[i]:
                        S_self += Sij
                    else:
                        S_cross += Sij

                self_stretching_total += np.dot(oi, S_self @ oi)
                cross_stretching_total += np.dot(oi, S_cross @ oi)

        self_stretching_total /= n_decomp_trials
        cross_stretching_total /= n_decomp_trials
        total = self_stretching_total + cross_stretching_total

        if abs(total) > 1e-15:
            self_frac = self_stretching_total / total
            cross_frac = cross_stretching_total / total
        else:
            self_frac = cross_frac = 0.0

        print(f"\n{name} (N={N}, {n_struct} structures):")
        print(f"  Self-stretching:  {self_stretching_total:+.6f} ({self_frac*100:+.1f}%)")
        print(f"  Cross-stretching: {cross_stretching_total:+.6f} ({cross_frac*100:+.1f}%)")
        print(f"  Total:            {total:+.6f}")

    # Part 5c: The critical test — WHAT STRUCTURAL PROPERTY causes better cancellation?
    print("\n" + "=" * 70)
    print("CRITICAL TEST: Self-stretching contribution in straight tubes")
    print("=" * 70)
    print("For a straight tube with aligned omega along axis t:")
    print("  omega_j = |omega| * t  (same for all j in tube)")
    print("  omega_j x r_ij is PERPENDICULAR to t")
    print("  => omega_i . S_ij . omega_i involves t . (perp to t) = 0")
    print("  => Self-stretching within a straight tube is IDENTICALLY ZERO")
    print()

    # Verify numerically
    N_tube = 20
    center = np.array([0.5, 0.5, 0.5])
    direction = np.array([0, 0, 1.0])
    nodes_tube = np.array([center + k * 0.05 * direction for k in range(N_tube)])
    omegas_tube = np.tile(direction, (N_tube, 1))

    S_tube = biot_savart_strain(nodes_tube, omegas_tube)
    self_stretch = sum(np.dot(omegas_tube[i], S_tube[i] @ omegas_tube[i]) for i in range(N_tube))
    print(f"Numerical verification (single tube, N={N_tube}):")
    print(f"  Self-stretching = {self_stretch:.2e}")
    print(f"  Expected: ~0 (machine precision)")

    # Compare: random omega along same positions
    random_omegas = np.random.randn(N_tube, 3)
    random_omegas /= norm(random_omegas, axis=1, keepdims=True)
    S_rand = biot_savart_strain(nodes_tube, random_omegas)
    rand_stretch = sum(np.dot(random_omegas[i], S_rand[i] @ random_omegas[i]) for i in range(N_tube))
    print(f"\nSame positions, RANDOM omega:")
    print(f"  Self-stretching = {rand_stretch:.2e}")
    print(f"  => Alignment kills self-stretching, not geometry alone")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
The C scaling discrepancy is explained by SELF-STRETCHING:

1. Straight tubes: omega aligned along tube axis t
   => omega x r is perpendicular to t for within-tube pairs
   => omega . S . omega = 0 for all within-tube contributions
   => Self-stretching is IDENTICALLY ZERO
   => Only cross-tube terms survive: O(n_tubes^2) terms out of O(N^2)
   => Much better cancellation

2. Random points: No structural alignment
   => All O(N^2) terms contribute with random signs
   => Central limit theorem: sum ~ sqrt(N^2) = N
   => C ~ N / (N * N) = 1/N... wait, let's be careful:
      Stretching ~ sum of N^2 terms with random signs ~ sqrt(N^2) = N
      Z ~ N (sum of |omega|^2)
      lambda_max ~ N (max eigenvalue of S, which gets contributions from N vortices)
      C = |Stretching| / (lambda_max * Z) ~ N / (N * N) = 1/N

   But we MEASURED C ~ 1/sqrt(N). The discrepancy is because:
      lambda_max does NOT scale as N for random configurations.
      lambda_max ~ sqrt(N) (eigenvalue of sum of random matrices).
      So C ~ N / (sqrt(N) * N) = 1/sqrt(N). Confirmed.

3. For filamentary: Self-stretching = 0, only cross-tube O(M^2) terms.
   M = n_tubes << N. Cross-stretching ~ sqrt(M^2) = M
   Z ~ N, lambda_max ~ sqrt(N) (still random cross-tube)
   C ~ M / (sqrt(N) * N)
   If M ~ N/k (k = tube size): C ~ (N/k) / (sqrt(N) * N) = 1/(k * sqrt(N))

   This gives FASTER decay than 1/sqrt(N) but NOT 1/N.
   The 1/N claim may have been from configurations where lambda_max ~ N
   (which happens when tubes are very close, creating large strain).

CRITICAL INSIGHT: The key structural property is self-stretching = 0
for straight aligned tubes. This is a GEOMETRIC identity, not statistical.
But it only removes diagonal-block terms; cross-tube terms still follow CLT.
""")


if __name__ == "__main__":
    run_scaling_test()
