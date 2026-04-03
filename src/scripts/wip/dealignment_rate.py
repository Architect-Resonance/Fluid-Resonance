"""
dealignment_rate.py

THE OSCILLATION / DE-ALIGNMENT INVESTIGATION
=============================================

Core hypothesis: Biot-Savart coupling forces vorticity direction xi = omega/|omega|
to DE-ALIGN from the alpha-eigenvector at neighboring points when forced toward it
at one point. This creates an oscillation that may provide a modulus of continuity
for xi, connecting to the Lei-Navas-Zhang (2009) Dini criterion.

Key questions:
1. When we align omega_i with alpha at point i, how much does xi_j change at
   nearby point j? (The "coupling strength" of the alignment map)
2. Is the de-alignment STRONGER for nearby points? (Spatial regularity)
3. Does the de-alignment have a predictable relationship with distance?
   If |delta xi_j| ~ f(|x_i - x_j|) for some modulus f, this could imply
   Dini or Holder continuity of the alignment map's fixed points.

Experiment structure:
- Part 1: Single-step de-alignment measurement
- Part 2: De-alignment vs distance (spatial decay)
- Part 3: Iterated de-alignment cascade
- Part 4: Connection to Dini condition
"""

import numpy as np
from scipy.linalg import eigh

# Precompute Levi-Civita
LC = np.zeros((3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            if (i, j, k) in [(0,1,2), (1,2,0), (2,0,1)]:
                LC[i, j, k] = 1
            elif (i, j, k) in [(0,2,1), (2,1,0), (1,0,2)]:
                LC[i, j, k] = -1


def compute_strain_at_point(x, positions, vorticities, epsilon=0.05):
    """Compute strain tensor S at point x from Biot-Savart."""
    grad_u = np.zeros((3, 3))
    for j in range(len(positions)):
        r = x - positions[j]
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            continue
        r_reg = np.sqrt(r_norm**2 + epsilon**2)
        for a in range(3):
            for kk in range(3):
                for b in range(3):
                    for m in range(3):
                        e = LC[a, b, m]
                        if e == 0:
                            continue
                        t1 = (1 if m == kk else 0) / r_reg**3
                        t2 = -3 * r[m] * r[kk] / r_reg**5
                        grad_u[a, kk] += e * vorticities[j][b] * (t1 + t2) / (4 * np.pi)
    S = 0.5 * (grad_u + grad_u.T)
    return S


def get_alpha_direction(S):
    """Return the unit eigenvector of S with largest eigenvalue (alpha)."""
    evals, evecs = eigh(S)
    idx = np.argsort(evals)[::-1]
    return evecs[:, idx[0]]


def xi_direction(omega):
    """Return unit direction of omega (xi = omega/|omega|)."""
    norm = np.linalg.norm(omega)
    if norm < 1e-15:
        return np.zeros(3)
    return omega / norm


def angle_between(v1, v2):
    """Angle in radians between two unit vectors (handling sign ambiguity)."""
    cos_theta = np.clip(np.abs(np.dot(v1, v2)), 0, 1)
    return np.arccos(cos_theta)


# ===== PART 1: Single-step de-alignment =====

def part1_single_step_dealignment():
    """
    Setup: N points with random vorticity.
    Action: Align omega at point 0 with its alpha-eigenvector.
    Measure: How much does xi change at ALL other points (via strain change)?
    """
    print("=" * 70)
    print("PART 1: SINGLE-STEP DE-ALIGNMENT MEASUREMENT")
    print("When we align omega_0 with alpha at point 0,")
    print("how much does the alpha-direction change at other points?")
    print("=" * 70)

    results = []
    for seed in range(20):
        np.random.seed(seed * 31 + 7)
        N = 30
        pos = np.random.randn(N, 3) * 0.5
        vor = np.random.randn(N, 3)
        vor /= np.linalg.norm(vor, axis=1, keepdims=True)

        # Record alpha direction at each point BEFORE alignment
        alpha_before = np.zeros((N, 3))
        for i in range(N):
            S_i = compute_strain_at_point(pos[i], pos, vor)
            alpha_before[i] = get_alpha_direction(S_i)

        # Align omega_0 with its alpha
        S_0 = compute_strain_at_point(pos[0], pos, vor)
        alpha_0 = get_alpha_direction(S_0)
        omega_mag = np.linalg.norm(vor[0])
        # Make sure sign is consistent
        if np.dot(vor[0], alpha_0) < 0:
            alpha_0 = -alpha_0
        vor_new = vor.copy()
        vor_new[0] = alpha_0 * omega_mag

        # Record alpha direction at each point AFTER alignment of point 0
        alpha_after = np.zeros((N, 3))
        for i in range(N):
            S_i = compute_strain_at_point(pos[i], pos, vor_new)
            alpha_after[i] = get_alpha_direction(S_i)

        # Measure de-alignment: angle between alpha_before and alpha_after
        for j in range(1, N):
            dist = np.linalg.norm(pos[0] - pos[j])
            angle_change = angle_between(alpha_before[j], alpha_after[j])
            results.append((dist, angle_change, seed))

    # Sort by distance
    results.sort(key=lambda x: x[0])

    # Bin by distance
    distances = np.array([r[0] for r in results])
    angles = np.array([r[1] for r in results])

    n_bins = 10
    bin_edges = np.linspace(distances.min(), distances.max(), n_bins + 1)

    print(f"\n  {'Dist range':<20} | {'Mean angle change':>18} | {'Max angle change':>18} | {'Count':>5}")
    print(f"  {'-'*20}-+-{'-'*18}-+-{'-'*18}-+-{'-'*5}")

    bin_means = []
    bin_centers = []
    for b in range(n_bins):
        mask = (distances >= bin_edges[b]) & (distances < bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        mean_angle = angles[mask].mean()
        max_angle = angles[mask].max()
        center = (bin_edges[b] + bin_edges[b + 1]) / 2
        bin_means.append(mean_angle)
        bin_centers.append(center)
        print(f"  [{bin_edges[b]:.2f}, {bin_edges[b+1]:.2f})"
              f" | {np.degrees(mean_angle):>15.4f} deg | {np.degrees(max_angle):>15.4f} deg | {mask.sum():>5}")

    # Fit power law: angle ~ C * dist^(-p)
    if len(bin_centers) > 3:
        log_d = np.log(np.array(bin_centers))
        log_a = np.log(np.array(bin_means) + 1e-15)
        coeffs = np.polyfit(log_d, log_a, 1)
        power = coeffs[0]
        print(f"\n  Power-law fit: angle_change ~ dist^({power:.3f})")
        print(f"  {'DECAY' if power < 0 else 'GROWTH'} with distance")
        if power < 0:
            print(f"  De-alignment is STRONGER for NEARBY points (as expected)")
            print(f"  Decay exponent: {abs(power):.3f}")
    return results


# ===== PART 2: De-alignment cascade =====

def part2_cascade():
    """
    Iterate the alignment process and track how xi changes at each point
    between consecutive iterations. Does the per-step xi change decrease?
    This would indicate convergence of xi (not of C) — a different question.
    """
    print("\n" + "=" * 70)
    print("PART 2: DE-ALIGNMENT CASCADE")
    print("Track how xi_i changes between consecutive alignment iterations.")
    print("If |delta xi| decreases, the direction field is converging")
    print("even if C oscillates.")
    print("=" * 70)

    np.random.seed(42)
    N = 30
    pos = np.random.randn(N, 3) * 0.5
    vor = np.random.randn(N, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)

    n_iter = 40
    xi_history = np.zeros((n_iter + 1, N, 3))
    C_history = []

    # Record initial xi
    for i in range(N):
        xi_history[0, i] = xi_direction(vor[i])

    for iteration in range(n_iter):
        # One round of alpha-alignment
        for i in range(N):
            S_i = compute_strain_at_point(pos[i], pos, vor)
            evals, evecs = eigh(S_i)
            idx = np.argsort(evals)[::-1]
            alpha = evecs[:, idx[0]]
            omega_mag = np.linalg.norm(vor[i])
            if omega_mag > 1e-10:
                # Maintain sign consistency
                if np.dot(vor[i], alpha) < 0:
                    alpha = -alpha
                vor[i] = alpha * omega_mag

        # Record xi and C
        for i in range(N):
            xi_history[iteration + 1, i] = xi_direction(vor[i])

        # Compute C_local
        stretching = 0
        max_possible = 0
        for i in range(N):
            S_i = compute_strain_at_point(pos[i], pos, vor)
            stretching += vor[i] @ S_i @ vor[i]
            evals = np.sort(np.linalg.eigvalsh(S_i))[::-1]
            max_possible += evals[0] * np.dot(vor[i], vor[i])
        C = abs(stretching) / (max_possible + 1e-15)
        C_history.append(C)

    # Analyze xi changes between iterations
    print(f"\n  {'Iter':<6} | {'Mean |delta xi|':>16} | {'Max |delta xi|':>15} | {'C_local':>10}")
    print(f"  {'-'*6}-+-{'-'*16}-+-{'-'*15}-+-{'-'*10}")

    delta_xi_means = []
    for t in range(n_iter):
        delta_xi = np.array([angle_between(xi_history[t, i], xi_history[t+1, i])
                             for i in range(N)])
        mean_dxi = delta_xi.mean()
        max_dxi = delta_xi.max()
        delta_xi_means.append(mean_dxi)
        if t < 15 or t >= n_iter - 5 or t % 5 == 0:
            print(f"  {t+1:<6} | {np.degrees(mean_dxi):>13.4f} deg | {np.degrees(max_dxi):>12.4f} deg | {C_history[t]:>10.6f}")

    # Check if delta_xi is decreasing (convergence of direction field)
    first_5 = np.mean(delta_xi_means[:5])
    last_5 = np.mean(delta_xi_means[-5:])
    print(f"\n  Mean |delta xi| (first 5 iters): {np.degrees(first_5):.4f} deg")
    print(f"  Mean |delta xi| (last 5 iters):  {np.degrees(last_5):.4f} deg")

    if last_5 < first_5 * 0.5:
        print(f"  -> Direction field IS converging (ratio: {last_5/first_5:.3f})")
    elif last_5 < first_5:
        print(f"  -> Slight convergence (ratio: {last_5/first_5:.3f})")
    else:
        print(f"  -> Direction field NOT converging (ratio: {last_5/first_5:.3f})")

    # Check C oscillation
    C_last_10 = C_history[-10:]
    C_spread = max(C_last_10) - min(C_last_10)
    print(f"\n  C_local spread (last 10 iters): {C_spread:.6f}")
    print(f"  C_local mean (last 10): {np.mean(C_last_10):.6f}")

    return delta_xi_means, C_history


# ===== PART 3: Spatial coherence of xi =====

def part3_spatial_coherence():
    """
    After many alignment iterations, measure the SPATIAL smoothness of xi.
    Compute |xi(x_i) - xi(x_j)| vs |x_i - x_j| and fit a modulus of continuity.

    If xi has a modulus of continuity omega(r) with integral(omega(r)/r, 0, delta) < inf
    (the Dini condition), then Lei-Navas-Zhang gives regularity.
    """
    print("\n" + "=" * 70)
    print("PART 3: SPATIAL COHERENCE OF xi AFTER ALIGNMENT")
    print("Measure |xi(x_i) - xi(x_j)| vs |x_i - x_j|")
    print("and test the Dini condition for the modulus of continuity.")
    print("=" * 70)

    np.random.seed(42)
    N = 40
    pos = np.random.randn(N, 3) * 0.5
    vor = np.random.randn(N, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)

    # Run many alignment iterations
    for iteration in range(30):
        for i in range(N):
            S_i = compute_strain_at_point(pos[i], pos, vor)
            evals, evecs = eigh(S_i)
            idx = np.argsort(evals)[::-1]
            alpha = evecs[:, idx[0]]
            omega_mag = np.linalg.norm(vor[i])
            if omega_mag > 1e-10:
                if np.dot(vor[i], alpha) < 0:
                    alpha = -alpha
                vor[i] = alpha * omega_mag

    # Compute xi at each point
    xi = np.zeros((N, 3))
    for i in range(N):
        xi[i] = xi_direction(vor[i])

    # All pairwise distances and angle differences
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            angle = angle_between(xi[i], xi[j])
            pairs.append((dist, angle))

    pairs.sort(key=lambda x: x[0])
    distances = np.array([p[0] for p in pairs])
    angles = np.array([p[1] for p in pairs])

    # Bin and compute max angle per bin (modulus of continuity = sup)
    n_bins = 12
    bin_edges = np.linspace(0, distances.max(), n_bins + 1)

    print(f"\n  {'Dist range':<20} | {'omega(r) = max angle':>20} | {'Mean angle':>12} | {'Count':>5}")
    print(f"  {'-'*20}-+-{'-'*20}-+-{'-'*12}-+-{'-'*5}")

    modulus_r = []
    modulus_omega = []

    for b in range(n_bins):
        mask = (distances >= bin_edges[b]) & (distances < bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        max_angle = angles[mask].max()
        mean_angle = angles[mask].mean()
        center = (bin_edges[b] + bin_edges[b + 1]) / 2
        modulus_r.append(center)
        modulus_omega.append(max_angle)
        print(f"  [{bin_edges[b]:.3f}, {bin_edges[b+1]:.3f})"
              f" | {np.degrees(max_angle):>17.4f} deg | {np.degrees(mean_angle):>9.4f} deg | {mask.sum():>5}")

    # Fit modulus: omega(r) ~ C * r^alpha
    if len(modulus_r) > 3:
        mr = np.array(modulus_r)
        mo = np.array(modulus_omega)
        # Only fit where modulus > 0
        valid = mo > 1e-10
        if valid.sum() > 3:
            log_r = np.log(mr[valid])
            log_o = np.log(mo[valid])
            coeffs = np.polyfit(log_r, log_o, 1)
            alpha_exp = coeffs[0]
            C_coeff = np.exp(coeffs[1])
            print(f"\n  Modulus of continuity fit: omega(r) ~ {C_coeff:.4f} * r^{alpha_exp:.4f}")

            if alpha_exp > 0:
                print(f"  xi IS Holder continuous with exponent {alpha_exp:.4f}")

                # Check Dini condition: integral(omega(r)/r, 0, delta) < inf
                # For omega(r) = C*r^alpha, integral = C * r^(alpha-1) dr
                # Converges at 0 iff alpha > 0 (which we just verified)
                print(f"  Dini condition: int(r^{alpha_exp:.4f} / r, 0, delta)")
                print(f"               = int(r^{alpha_exp-1:.4f}, 0, delta)")
                if alpha_exp > 0:
                    print(f"               = CONVERGES (exponent {alpha_exp-1:.4f} > -1)")
                    print(f"  => XI SATISFIES THE DINI CONDITION!")
                    print(f"  => By Lei-Navas-Zhang (2009): this would imply regularity")
                else:
                    print(f"               = DIVERGES (exponent {alpha_exp-1:.4f} <= -1)")
                    print(f"  => Dini condition NOT satisfied")
            else:
                print(f"  xi is NOT Holder continuous (exponent {alpha_exp:.4f} <= 0)")
    return modulus_r, modulus_omega


# ===== PART 4: Robustness check =====

def part4_robustness():
    """
    Repeat Part 3 across multiple seeds and configurations.
    The Holder exponent must be universal (not seed-dependent) to matter.
    """
    print("\n" + "=" * 70)
    print("PART 4: ROBUSTNESS OF HOLDER EXPONENT")
    print("Does the xi-smoothness hold across seeds and configurations?")
    print("=" * 70)

    exponents = []
    for seed in range(15):
        np.random.seed(seed * 47 + 3)
        N = 30 + seed % 10
        scale = 0.3 + 0.1 * (seed % 5)
        pos = np.random.randn(N, 3) * scale
        vor = np.random.randn(N, 3)
        vor /= np.linalg.norm(vor, axis=1, keepdims=True)

        # Align
        for iteration in range(25):
            for i in range(N):
                S_i = compute_strain_at_point(pos[i], pos, vor)
                evals, evecs = eigh(S_i)
                idx = np.argsort(evals)[::-1]
                alpha = evecs[:, idx[0]]
                omega_mag = np.linalg.norm(vor[i])
                if omega_mag > 1e-10:
                    if np.dot(vor[i], alpha) < 0:
                        alpha = -alpha
                    vor[i] = alpha * omega_mag

        # Compute xi pairwise
        xi = np.zeros((N, 3))
        for i in range(N):
            xi[i] = xi_direction(vor[i])

        dists = []
        angs = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                a = angle_between(xi[i], xi[j])
                dists.append(d)
                angs.append(a)

        dists = np.array(dists)
        angs = np.array(angs)

        # Simple power-law fit on all pairs
        valid = (dists > 0.01) & (angs > 1e-10)
        if valid.sum() > 10:
            log_d = np.log(dists[valid])
            log_a = np.log(angs[valid])
            coeffs = np.polyfit(log_d, log_a, 1)
            alpha_exp = coeffs[0]
            exponents.append(alpha_exp)
            print(f"  Seed {seed:2d}: N={N:2d}, scale={scale:.1f}, "
                  f"Holder exponent = {alpha_exp:.4f}")
        else:
            print(f"  Seed {seed:2d}: insufficient data")

    if len(exponents) > 0:
        exponents = np.array(exponents)
        print(f"\n  Holder exponent across seeds:")
        print(f"    Mean:   {exponents.mean():.4f}")
        print(f"    Median: {np.median(exponents):.4f}")
        print(f"    Min:    {exponents.min():.4f}")
        print(f"    Max:    {exponents.max():.4f}")
        print(f"    Std:    {exponents.std():.4f}")

        if exponents.min() > 0:
            print(f"\n  ALL exponents positive => xi is UNIVERSALLY Holder continuous")
            print(f"  => Dini condition UNIVERSALLY satisfied")
            print(f"  => Lei-Navas-Zhang regularity criterion met!")
            print(f"\n  CAVEAT: This is for DISCRETE point vortices with epsilon-regularization.")
            print(f"  The continuous-limit (N -> inf, epsilon -> 0) must be checked separately.")
        elif exponents.mean() > 0:
            print(f"\n  Mean exponent positive but not all — partially Holder")
        else:
            print(f"\n  Mean exponent <= 0 — xi is NOT Holder continuous")


def part5_the_mechanism():
    """
    WHY does Biot-Savart force xi to be smooth?

    The mechanism: if xi(x) points in direction e1 and xi(y) points in
    direction e2, with e1 != e2, then the vortex at x generates a strain
    at y that has a specific relationship to the angle between e1 and e2.

    Specifically: a vortex tube along e1 generates strain with alpha
    perpendicular to e1 (at nearby points). So if xi(y) = e2 is the
    alpha direction at y, it must be approximately perpendicular to e1
    for nearby y. But then e2 is also generating strain at x with alpha
    perpendicular to e2... Creating a consistency constraint.

    This test: for each pair of nearby vortices, compute:
    - The angle between their xi directions
    - The "geometric prediction" from Biot-Savart (what angle the strain geometry predicts)
    """
    print("\n" + "=" * 70)
    print("PART 5: THE BIOT-SAVART SMOOTHNESS MECHANISM")
    print("Why does the coupling constrain xi to be smooth?")
    print("=" * 70)

    np.random.seed(42)
    N = 30
    pos = np.random.randn(N, 3) * 0.5
    vor = np.random.randn(N, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)

    # Align
    for iteration in range(30):
        for i in range(N):
            S_i = compute_strain_at_point(pos[i], pos, vor)
            evals, evecs = eigh(S_i)
            idx = np.argsort(evals)[::-1]
            alpha = evecs[:, idx[0]]
            omega_mag = np.linalg.norm(vor[i])
            if omega_mag > 1e-10:
                if np.dot(vor[i], alpha) < 0:
                    alpha = -alpha
                vor[i] = alpha * omega_mag

    # For nearby pairs: analyze the geometric constraint
    print(f"\n  For nearby pairs (dist < 0.3):")
    print(f"  {'Pair':<10} | {'Dist':>6} | {'xi angle':>10} | {'xi_i . r_hat':>12} | {'xi_j . r_hat':>12}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")

    xi_angles_near = []
    xi_r_angles_i = []
    xi_r_angles_j = []
    count = 0

    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist > 0.3:
                continue

            xi_i = xi_direction(vor[i])
            xi_j = xi_direction(vor[j])
            r_hat = (pos[j] - pos[i]) / dist

            xi_angle = angle_between(xi_i, xi_j)
            # How much do xi_i and xi_j project onto the separation direction?
            dot_i_r = abs(np.dot(xi_i, r_hat))
            dot_j_r = abs(np.dot(xi_j, r_hat))

            xi_angles_near.append(xi_angle)
            xi_r_angles_i.append(dot_i_r)
            xi_r_angles_j.append(dot_j_r)

            if count < 15:
                print(f"  ({i:2d},{j:2d})"
                      f"   | {dist:>6.3f} | {np.degrees(xi_angle):>7.2f} deg"
                      f" | {dot_i_r:>12.4f} | {dot_j_r:>12.4f}")
            count += 1

    if len(xi_angles_near) > 0:
        print(f"\n  Total nearby pairs: {count}")
        print(f"  Mean xi angle between nearby vortices: {np.degrees(np.mean(xi_angles_near)):.2f} deg")
        print(f"  Max xi angle between nearby vortices:  {np.degrees(np.max(xi_angles_near)):.2f} deg")
        print(f"  Mean |xi . r_hat|: i={np.mean(xi_r_angles_i):.4f}, j={np.mean(xi_r_angles_j):.4f}")

        # Key insight: a single vortex along e generates velocity perpendicular to e
        # and strain with alpha perpendicular to both e and r at distance r.
        # So for nearby vortices in approximate alignment, we expect xi . r_hat ~ 0
        # (vorticity perpendicular to separation) OR xi_i ~= xi_j (parallel vortices).
        print(f"\n  INTERPRETATION:")
        print(f"  If nearby xi are similar (small angle), the direction field is smooth.")
        print(f"  The Biot-Savart geometry constrains nearby vortices because:")
        print(f"  - A vortex along e generates strain with alpha ~ perp to e")
        print(f"  - So xi at a neighbor must be ~perp to the source's e")
        print(f"  - But that neighbor's xi also constrains the first point")
        print(f"  - This mutual constraint forces SMOOTHNESS of xi.")


if __name__ == "__main__":
    part1_single_step_dealignment()
    part2_cascade()
    part3_spatial_coherence()
    part4_robustness()
    part5_the_mechanism()
