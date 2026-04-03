"""
trace_free_C_bound.py

OPTION A: Find the maximum stretching ratio C for divergence-free flows.

Key question: for omega in R^3 and S = strain(Biot-Savart(omega)),
what is the supremum of:

    C = |integral omega . S . omega dx| / (||S||_inf * Z)

where Z = integral |omega|^2 dx?

The constraint tr(S) = 0 (from div u = 0) means strain eigenvalues
satisfy alpha + beta + gamma = 0. This limits how much stretching
can happen.

CRITICAL INSIGHT: omega and S are NOT independent. S is determined
by omega through Biot-Savart. A straight vortex tube has ZERO stretching
(Biot-Savart velocity is azimuthal, perpendicular to tube axis).
This coupling forces C < 1.

Approach:
1. Test specific vorticity configurations (tubes, rings, helices)
2. Adversarial search for maximum C
3. Analytic bound derivation

Relates to: Tao's insight that pressure-velocity coupling is essential.
The trace-free constraint IS the pressure constraint.
"""

import numpy as np
from scipy.linalg import eigh


# ============================================================
# PART 1: POINTWISE BOUND
# For a single 3x3 trace-free symmetric matrix S and unit vector xi,
# what is the maximum of |xi . S . xi| / ||S||_op ?
# ============================================================

def pointwise_bound_analysis():
    """
    For trace-free S with eigenvalues alpha >= beta >= gamma,
    alpha + beta + gamma = 0:

    The stretching rate is xi . S . xi.
    Maximum over xi: alpha (when xi = alpha-eigenvector)

    The operator norm ||S||_op = max(|alpha|, |gamma|).

    Question: is alpha / max(|alpha|, |gamma|) always < 1?

    Answer: YES, unless |alpha| = max(|alpha|, |gamma|), i.e., alpha >= |gamma|.
    Since gamma = -(alpha + beta) and beta <= alpha:
    |gamma| = alpha + beta >= 0 (since beta >= -alpha for trace-free with beta being intermediate)

    Actually: alpha >= beta >= gamma, alpha + beta + gamma = 0.
    So gamma = -(alpha + beta).
    |gamma| = alpha + beta (when beta >= 0) or = alpha + beta (could be < 0 if beta < 0).

    Case 1: beta >= 0. Then gamma = -(alpha + beta) <= -alpha. |gamma| = alpha + beta >= alpha.
    So |gamma| >= alpha, meaning ||S||_op = |gamma| >= alpha.
    Ratio: alpha / |gamma| = alpha / (alpha + beta) <= 1 (equality iff beta = 0).

    Case 2: beta < 0. Then gamma = -(alpha + beta), |gamma| = |alpha + beta|.
    If alpha + beta > 0: |gamma| = alpha + beta < alpha (since beta < 0).
    Then ||S||_op = alpha. Ratio = alpha/alpha = 1.
    If alpha + beta < 0: |gamma| = -(alpha + beta) > alpha. Ratio < 1.
    If alpha + beta = 0: gamma = 0, ||S||_op = alpha. Ratio = 1.

    So the pointwise ratio alpha / ||S||_op can reach 1.
    The trace-free constraint does NOT help at the pointwise level!

    BUT: the INTEGRAL may still be < 1 because of the Biot-Savart coupling.
    """
    print("=" * 70)
    print("PART 1: POINTWISE ANALYSIS")
    print("=" * 70)

    # Sample random trace-free symmetric matrices
    n_samples = 100000
    max_ratio = 0

    for _ in range(n_samples):
        # Random eigenvalues with alpha + beta + gamma = 0
        alpha = np.random.uniform(0, 10)
        beta = np.random.uniform(-alpha, alpha)  # beta in [-alpha, alpha]
        gamma = -(alpha + beta)

        # Stretching ratio: alpha / max(|alpha|, |gamma|)
        S_norm = max(abs(alpha), abs(gamma))
        if S_norm > 1e-10:
            ratio = alpha / S_norm
            max_ratio = max(max_ratio, ratio)

    print(f"  Pointwise max(alpha / ||S||_op) = {max_ratio:.6f}")
    print(f"  -> Pointwise bound: C_pointwise = 1.0 (trace-free doesn't help)")
    print(f"  -> The bound C < 1 must come from the INTEGRAL + Biot-Savart coupling")


# ============================================================
# PART 2: BIOT-SAVART COUPLING
# For specific vorticity fields, compute the stretching ratio C.
# ============================================================

def biot_savart_kernel_gradient(x, y, epsilon=0.05):
    """
    Gradient of regularized Biot-Savart kernel at x due to source at y.

    u_i(x) = -(1/4pi) epsilon_ijk omega_j(y) * (x-y)_k / |x-y|^3

    Returns the 3x3 tensor d(kernel)/dx for the velocity gradient.
    """
    r = x - y
    r_norm = np.linalg.norm(r)
    r_reg = np.sqrt(r_norm**2 + epsilon**2)

    if r_norm < 1e-12:
        return np.zeros((3, 3, 3))  # i, j, k tensor: contribution to du_i/dx_k from omega_j

    # d/dx_k [epsilon_ijm r_m / r_reg^3]
    # = epsilon_ijm [delta_mk / r_reg^3 - 3 r_m r_k / r_reg^5]
    result = np.zeros((3, 3, 3))
    eps_tensor = np.zeros((3, 3, 3))
    eps_tensor[0, 1, 2] = eps_tensor[1, 2, 0] = eps_tensor[2, 0, 1] = 1
    eps_tensor[0, 2, 1] = eps_tensor[2, 1, 0] = eps_tensor[1, 0, 2] = -1

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    e = eps_tensor[i, j, m]
                    if e == 0:
                        continue
                    term1 = (1 if m == k else 0) / r_reg**3
                    term2 = -3 * r[m] * r[k] / r_reg**5
                    result[i, j, k] += e * (term1 + term2) / (4 * np.pi)

    return result


def compute_stretching_ratio(positions, vorticities, epsilon=0.05):
    """
    Compute C = |sum omega_i . S_i . omega_i| / (max_eigenvalue * Z)

    For discrete point vortices with positions and vorticity vectors.
    """
    N = len(positions)

    # Compute enstrophy
    Z = sum(np.dot(vorticities[i], vorticities[i]) for i in range(N))

    # Compute stretching at each point
    stretching_total = 0
    max_strain_eigenvalue = 0

    for i in range(N):
        # Velocity gradient at point i due to all other points
        grad_u = np.zeros((3, 3))
        for j in range(N):
            if i == j:
                continue
            K = biot_savart_kernel_gradient(positions[i], positions[j], epsilon)
            # du_i/dx_k = sum_j K[i,j,k] * omega_j(y)
            for a in range(3):
                for k in range(3):
                    for b in range(3):
                        grad_u[a, k] += K[a, b, k] * vorticities[j][b]

        # Strain tensor
        S = 0.5 * (grad_u + grad_u.T)

        # Stretching at this point
        stretch_i = vorticities[i] @ S @ vorticities[i]
        stretching_total += stretch_i

        # Track max eigenvalue
        evals = np.sort(np.linalg.eigvalsh(S))[::-1]
        max_strain_eigenvalue = max(max_strain_eigenvalue, evals[0])

    if max_strain_eigenvalue < 1e-12 or Z < 1e-12:
        return 0.0, 0.0, 0.0

    C = abs(stretching_total) / (max_strain_eigenvalue * Z)
    return C, stretching_total, max_strain_eigenvalue


def test_specific_configurations():
    """Test C for physically meaningful vorticity configurations."""

    print("\n" + "=" * 70)
    print("PART 2: BIOT-SAVART COUPLING — SPECIFIC CONFIGURATIONS")
    print("=" * 70)

    results = []

    # --- Config 1: Straight vortex tube (should have C near 0) ---
    print("\n--- Config 1: Straight vortex tube ---")
    N = 50
    positions = np.zeros((N, 3))
    positions[:, 2] = np.linspace(-2, 2, N)  # Along z-axis
    vorticities = np.zeros((N, 3))
    vorticities[:, 2] = 1.0  # omega = z-hat

    C, S_tot, S_max = compute_stretching_ratio(positions, vorticities)
    print(f"  C = {C:.6f}, Stretching = {S_tot:.6f}, S_max = {S_max:.6f}")
    results.append(("Straight tube", C))

    # --- Config 2: Vortex ring (has self-induced stretching) ---
    print("\n--- Config 2: Vortex ring ---")
    N = 60
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    R = 1.0  # Ring radius
    positions = np.column_stack([R * np.cos(theta), R * np.sin(theta), np.zeros(N)])
    # Vorticity tangent to ring
    vorticities = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(N)])

    C, S_tot, S_max = compute_stretching_ratio(positions, vorticities)
    print(f"  C = {C:.6f}, Stretching = {S_tot:.6f}, S_max = {S_max:.6f}")
    results.append(("Vortex ring", C))

    # --- Config 3: Two anti-parallel tubes (reconnection geometry) ---
    print("\n--- Config 3: Two anti-parallel tubes ---")
    N_per = 30
    positions_list = []
    vorticities_list = []
    for z in np.linspace(-2, 2, N_per):
        positions_list.append([0.3, 0, z])
        vorticities_list.append([0, 0, 1.0])
        positions_list.append([-0.3, 0, z])
        vorticities_list.append([0, 0, -1.0])
    positions = np.array(positions_list)
    vorticities = np.array(vorticities_list)

    C, S_tot, S_max = compute_stretching_ratio(positions, vorticities)
    print(f"  C = {C:.6f}, Stretching = {S_tot:.6f}, S_max = {S_max:.6f}")
    results.append(("Anti-parallel tubes", C))

    # --- Config 4: Trefoil knot (topologically nontrivial) ---
    print("\n--- Config 4: Trefoil knot ---")
    N = 80
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Trefoil parametrization
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)
    positions = np.column_stack([x, y, z]) * 0.3
    # Tangent vectors as vorticity
    dx = np.cos(t) + 4*np.cos(2*t)
    dy = -np.sin(t) + 4*np.sin(2*t)
    dz = -3*np.cos(3*t)
    vorticities = np.column_stack([dx, dy, dz])
    # Normalize to unit vorticity
    norms = np.linalg.norm(vorticities, axis=1, keepdims=True)
    vorticities /= norms

    C, S_tot, S_max = compute_stretching_ratio(positions, vorticities)
    print(f"  C = {C:.6f}, Stretching = {S_tot:.6f}, S_max = {S_max:.6f}")
    results.append(("Trefoil knot", C))

    # --- Config 5: Random filament cloud (turbulence-like) ---
    print("\n--- Config 5: Random filament cloud ---")
    np.random.seed(42)
    N = 60
    positions = np.random.randn(N, 3) * 0.5
    vorticities = np.random.randn(N, 3)
    vorticities /= np.linalg.norm(vorticities, axis=1, keepdims=True)

    C, S_tot, S_max = compute_stretching_ratio(positions, vorticities)
    print(f"  C = {C:.6f}, Stretching = {S_tot:.6f}, S_max = {S_max:.6f}")
    results.append(("Random cloud", C))

    # --- Config 6: Adversarial — all omega aligned with z, positions in xy-plane ---
    print("\n--- Config 6: Adversarial (omega=z, positions in xy-plane) ---")
    N = 40
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    positions = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(N)])
    vorticities = np.tile([0, 0, 1.0], (N, 1))

    C, S_tot, S_max = compute_stretching_ratio(positions, vorticities)
    print(f"  C = {C:.6f}, Stretching = {S_tot:.6f}, S_max = {S_max:.6f}")
    results.append(("Adversarial z-ring", C))

    # --- Config 7: Burgers vortex (known exact solution) ---
    print("\n--- Config 7: Burgers vortex (stretched tube) ---")
    N = 50
    # Vortex along z, with points distributed along z
    positions = np.zeros((N, 3))
    positions[:, 2] = np.linspace(-3, 3, N)
    # Add radial spread (simulating Burgers stretching)
    r_spread = np.exp(-positions[:, 2]**2)
    positions[:, 0] = 0.1 * r_spread
    vorticities = np.zeros((N, 3))
    vorticities[:, 2] = np.exp(-positions[:, 2]**2 / 2)  # Gaussian profile

    C, S_tot, S_max = compute_stretching_ratio(positions, vorticities)
    print(f"  C = {C:.6f}, Stretching = {S_tot:.6f}, S_max = {S_max:.6f}")
    results.append(("Burgers vortex", C))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: STRETCHING RATIO C BY CONFIGURATION")
    print("=" * 70)
    print(f"  {'Configuration':<25} | C")
    print(f"  {'-'*25}-+--------")
    max_C = 0
    for name, C in results:
        print(f"  {name:<25} | {C:.6f}")
        max_C = max(max_C, C)
    print(f"\n  Maximum C found: {max_C:.6f}")
    if max_C < 1.0:
        print(f"  -> C < 1 holds for all tested configurations!")
        print(f"  -> The Biot-Savart coupling DOES suppress stretching")
    else:
        print(f"  -> WARNING: C >= 1 found! The bound does NOT hold universally")

    return results


# ============================================================
# PART 3: ADVERSARIAL SEARCH
# Try to MAXIMIZE C over vorticity configurations.
# ============================================================

def adversarial_C_search(n_trials=200, N=30):
    """
    Randomly generate vorticity configs and find the maximum C.
    This is the adversarial test: can we BREAK the C < 1 bound?
    """
    print("\n" + "=" * 70)
    print("PART 3: ADVERSARIAL SEARCH FOR MAXIMUM C")
    print(f"({n_trials} random configs, {N} points each)")
    print("=" * 70)

    max_C = 0
    max_config = None
    C_values = []

    for trial in range(n_trials):
        np.random.seed(trial * 31 + 7)

        # Random positions and vorticities
        scale = np.random.uniform(0.2, 2.0)
        positions = np.random.randn(N, 3) * scale

        # Various vorticity patterns
        pattern = trial % 5
        if pattern == 0:
            # Random directions
            vorticities = np.random.randn(N, 3)
        elif pattern == 1:
            # All aligned
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            vorticities = np.tile(direction, (N, 1)) * np.random.uniform(0.5, 2, (N, 1))
        elif pattern == 2:
            # Radial (pointing away from center)
            vorticities = positions / (np.linalg.norm(positions, axis=1, keepdims=True) + 0.1)
        elif pattern == 3:
            # Tangential (perpendicular to radial)
            z_hat = np.array([0, 0, 1.0])
            vorticities = np.cross(positions, z_hat)
            norms = np.linalg.norm(vorticities, axis=1, keepdims=True)
            vorticities = np.where(norms > 0.1, vorticities / norms, np.random.randn(N, 3))
        else:
            # Two opposing groups
            vorticities = np.zeros((N, 3))
            vorticities[:N//2, 2] = 1.0
            vorticities[N//2:, 2] = -1.0

        C, _, S_max = compute_stretching_ratio(positions, vorticities)
        C_values.append(C)

        if C > max_C:
            max_C = C
            max_config = (positions.copy(), vorticities.copy(), trial)

    C_arr = np.array(C_values)
    print(f"\n  Distribution of C across {n_trials} configs:")
    print(f"    Mean:   {np.mean(C_arr):.6f}")
    print(f"    Median: {np.median(C_arr):.6f}")
    print(f"    Max:    {np.max(C_arr):.6f}")
    print(f"    Min:    {np.min(C_arr):.6f}")
    print(f"    Std:    {np.std(C_arr):.6f}")
    print(f"    % with C > 0.5: {100*np.mean(C_arr > 0.5):.1f}%")
    print(f"    % with C > 0.8: {100*np.mean(C_arr > 0.8):.1f}%")
    print(f"    % with C > 1.0: {100*np.mean(C_arr > 1.0):.1f}%")

    if max_C >= 1.0:
        print(f"\n  *** C >= 1 FOUND at trial {max_config[2]} ***")
        print(f"  -> The C < 1 bound is REFUTED")
    else:
        print(f"\n  C < 1 holds across ALL {n_trials} trials")
        print(f"  Maximum C = {max_C:.6f}")

    return max_C, C_arr


if __name__ == "__main__":
    pointwise_bound_analysis()
    test_specific_configurations()
    max_C, C_dist = adversarial_C_search()

    print("\n" + "=" * 70)
    print("OVERALL CONCLUSION")
    print("=" * 70)
    if max_C < 1.0:
        print(f"""
  The stretching ratio C = |Stretching| / (||S||_inf * Z) is bounded below 1
  across all tested configurations (max = {max_C:.6f}).

  This means: the Biot-Savart coupling between omega and S prevents
  perfect alignment. Vorticity cannot fully exploit its own strain field.

  Physical reason: a vortex tube's Biot-Savart velocity is perpendicular
  to its own axis. Self-stretching requires geometric twisting, which
  introduces cancellations.

  If this bound can be proved analytically, it provides a new inequality
  for the enstrophy equation: dZ/dt <= (C-1) * ||S||_inf * Z + lower_order
  For C < 1, this gives DECAY (not growth) of enstrophy.
""")
    else:
        print(f"  C can reach {max_C:.6f} >= 1. No universal bound.")
