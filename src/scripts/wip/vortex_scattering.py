"""
vortex_scattering.py

VORTEX SCATTERING AMPLITUDE ANALYSIS
=====================================

Analogy from quantum scattering theory applied to vortex interactions.

In QM: unitarity (probability conservation) bounds the scattering amplitude.
In NS: energy conservation bounds the total deflection of vortex tubes.

The key question: does the energy-conservation constraint on vortex scattering
give a BETTER bound on the bilinear form <B(u,u), Au> than the standard
Ladyzhenskaya inequality?

Standard bound: |<B(u,u), Au>| <= C * ||u||^{1/2} * ||nabla u||^{1/2} * ||Au||^{3/2}

The exponent 3/2 on ||Au|| is the critical obstacle. If we can improve it to
3/2 - delta for ANY delta > 0, regularity follows.

Experiment:
- Part 1: Compute actual bilinear form for Biot-Savart structured fields
- Part 2: Compare to theoretical worst case (ratio = depletion factor)
- Part 3: "Scattering cross-section" — how much do interacting tubes deflect?
- Part 4: Energy conservation constraint on total scattering
- Part 5: Scale dependence — does the depletion factor improve at small scales?
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


def biot_savart_velocity(x, positions, vorticities, epsilon=0.05):
    """Compute velocity at x from Biot-Savart."""
    vel = np.zeros(3)
    for j in range(len(positions)):
        r = x - positions[j]
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            continue
        r_reg = np.sqrt(r_norm**2 + epsilon**2)
        # u = (1/4pi) * (omega x r) / |r|^3
        vel += np.cross(vorticities[j], r) / (4 * np.pi * r_reg**3)
    return vel


def compute_velocity_gradient(x, positions, vorticities, epsilon=0.05):
    """Compute full velocity gradient tensor at x."""
    grad_u = np.zeros((3, 3))
    for j in range(len(positions)):
        r = x - positions[j]
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            continue
        r_reg = np.sqrt(r_norm**2 + epsilon**2)
        for a in range(3):
            for k in range(3):
                for b in range(3):
                    for m in range(3):
                        e = LC[a, b, m]
                        if e == 0:
                            continue
                        t1 = (1 if m == k else 0) / r_reg**3
                        t2 = -3 * r[m] * r[k] / r_reg**5
                        grad_u[a, k] += e * vorticities[j][b] * (t1 + t2) / (4 * np.pi)
    return grad_u


def compute_strain(grad_u):
    """Strain tensor from velocity gradient."""
    return 0.5 * (grad_u + grad_u.T)


def part1_bilinear_form():
    """
    Compute the actual bilinear form <B(u,u), Au> for Biot-Savart fields.

    For point vortices, B(u,u) = P((u.nabla)u) involves the velocity gradient.
    The enstrophy growth rate is: sum_i omega_i . S_i . omega_i

    Compare:
    - Actual stretching: sum_i omega_i . S_i . omega_i
    - Ladyzhenskaya bound: C * E^{1/2} * Z^{1/2} * P^{3/2}
    - Modified bound: C' * max_lambda * Z (stretching majorant)

    The ratio (actual / bound) measures the depletion factor.
    """
    print("=" * 70)
    print("PART 1: BILINEAR FORM DEPLETION")
    print("Compare actual stretching to theoretical worst case")
    print("=" * 70)

    results = []

    for seed in range(50):
        np.random.seed(seed * 37 + 5)
        N = 20 + seed % 20
        scale = 0.3 + 0.1 * (seed % 5)
        pos = np.random.randn(N, 3) * scale
        vor = np.random.randn(N, 3)
        # Normalize to unit vorticity per point
        vor /= np.linalg.norm(vor, axis=1, keepdims=True)

        # Compute stretching, strain norms, enstrophy
        stretching = 0
        Z = 0  # enstrophy proxy
        max_lambda = 0
        S_norm_sq_sum = 0

        for i in range(N):
            S_i = compute_strain(compute_velocity_gradient(pos[i], pos, vor))
            stretching += vor[i] @ S_i @ vor[i]

            evals = np.sort(np.linalg.eigvalsh(S_i))[::-1]
            max_lambda = max(max_lambda, evals[0])

            omega_sq = np.dot(vor[i], vor[i])
            Z += omega_sq
            S_norm_sq_sum += np.sum(S_i**2)

        # Stretching majorant bound: |stretching| <= max_lambda * Z
        majorant_bound = max_lambda * Z
        ratio_majorant = abs(stretching) / (majorant_bound + 1e-15)

        results.append({
            'N': N,
            'stretching': stretching,
            'majorant_bound': majorant_bound,
            'ratio_majorant': ratio_majorant,
            'max_lambda': max_lambda,
            'Z': Z,
        })

    # Analyze
    ratios = [r['ratio_majorant'] for r in results]
    Ns = [r['N'] for r in results]

    print(f"\n  Stretching / (max_lambda * Z) across 50 random configs:")
    print(f"  Mean ratio:   {np.mean(ratios):.6f}")
    print(f"  Max ratio:    {np.max(ratios):.6f}")
    print(f"  Min ratio:    {np.min(ratios):.6f}")
    print(f"  Std ratio:    {np.std(ratios):.6f}")

    # Bin by N
    print(f"\n  {'N range':<15} | {'Mean ratio':>10} | {'Max ratio':>10} | {'Count':>5}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}")
    for n_lo in range(20, 40, 5):
        n_hi = n_lo + 5
        mask = [(n_lo <= r['N'] < n_hi) for r in results]
        if sum(mask) == 0:
            continue
        ratios_bin = [r['ratio_majorant'] for r, m in zip(results, mask) if m]
        print(f"  [{n_lo}, {n_hi})"
              f"        | {np.mean(ratios_bin):>10.6f} | {np.max(ratios_bin):>10.6f} | {sum(mask):>5}")

    return results


def part2_scattering_cross_section():
    """
    TWO-TUBE SCATTERING EXPERIMENT

    Two parallel vortex tubes approach at various impact parameters.
    Measure the deflection (change in vorticity direction) as a function
    of impact parameter b.

    The "scattering cross-section" sigma = integral theta^2(b) * b * db
    measures the total deflection capacity of the interaction.
    """
    print("\n" + "=" * 70)
    print("PART 2: VORTEX SCATTERING CROSS-SECTION")
    print("Two tubes at impact parameter b: measure deflection theta(b)")
    print("=" * 70)

    # Tube 1: along z-axis at origin
    # Tube 2: along z-axis at (b, 0, 0) — parallel, offset by b
    n_seg = 20
    z_vals = np.linspace(-2, 2, n_seg)

    impact_params = np.linspace(0.1, 2.0, 20)
    deflections = []

    print(f"\n  {'b':>6} | {'Mean deflection':>16} | {'Max deflection':>16} | {'theta^2*b':>10}")
    print(f"  {'-'*6}-+-{'-'*16}-+-{'-'*16}-+-{'-'*10}")

    for b in impact_params:
        # Build two-tube configuration
        pos_list = []
        vor_list = []

        for z in z_vals:
            pos_list.append([0, 0, z])
            vor_list.append([0, 0, 1.0])  # Tube 1
            pos_list.append([b, 0, z])
            vor_list.append([0, 0, 1.0])  # Tube 2 (parallel)

        pos = np.array(pos_list)
        vor = np.array(vor_list)

        # Compute strain at each point of tube 2 due to tube 1
        # Measure how much the alpha-direction deviates from z-axis
        tube2_defl = []
        for idx in range(n_seg):
            j = idx * 2 + 1  # Tube 2 points are at odd indices
            S = compute_strain(compute_velocity_gradient(pos[j], pos, vor))
            alpha = np.zeros(3)
            evals, evecs = eigh(S)
            order = np.argsort(evals)[::-1]
            alpha = evecs[:, order[0]]
            # Angle between alpha and z-axis
            angle = np.arccos(np.clip(abs(alpha[2]), 0, 1))
            tube2_defl.append(angle)

        mean_defl = np.mean(tube2_defl)
        max_defl = np.max(tube2_defl)
        theta_sq_b = mean_defl**2 * b  # integrand for cross-section
        deflections.append((b, mean_defl, max_defl, theta_sq_b))
        print(f"  {b:>6.2f} | {np.degrees(mean_defl):>13.4f} deg | {np.degrees(max_defl):>13.4f} deg | {theta_sq_b:>10.6f}")

    # Total scattering cross-section
    bs = np.array([d[0] for d in deflections])
    integrand = np.array([d[3] for d in deflections])
    db = bs[1] - bs[0]
    sigma_total = 2 * np.pi * np.sum(integrand) * db  # integrate over b

    print(f"\n  Total scattering cross-section: sigma = {sigma_total:.6f}")

    # Energy of the two-tube system
    # E ~ n_seg * Gamma^2 * log(L/epsilon) ~ N (rough scaling)
    E_approx = 2 * n_seg  # very rough
    print(f"  Approximate energy: E ~ {E_approx}")
    print(f"  Ratio sigma/E: {sigma_total / E_approx:.6f}")

    # Fit deflection vs distance
    bs_arr = np.array([d[0] for d in deflections])
    defl_arr = np.array([d[1] for d in deflections])
    valid = defl_arr > 1e-6
    if valid.sum() > 5:
        coeffs = np.polyfit(np.log(bs_arr[valid]), np.log(defl_arr[valid]), 1)
        print(f"\n  Deflection power law: theta(b) ~ b^({coeffs[0]:.3f})")
        print(f"  (Biot-Savart predicts b^(-2) for point vortices)")

    return deflections


def part3_perpendicular_scattering():
    """
    Perpendicular tube scattering — the adversarial case.
    Tube 1 along z, Tube 2 along x at distance b in y.
    """
    print("\n" + "=" * 70)
    print("PART 3: PERPENDICULAR TUBE SCATTERING (ADVERSARIAL)")
    print("Maximum stretching geometry: tubes at 90 degrees")
    print("=" * 70)

    n_seg = 20
    z_vals = np.linspace(-2, 2, n_seg)
    x_vals = np.linspace(-2, 2, n_seg)

    impact_params = np.linspace(0.1, 2.0, 20)

    print(f"\n  {'b':>6} | {'Stretching':>12} | {'Max lambda':>12} | {'Z':>8} | {'C = S/(lam*Z)':>14}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*14}")

    c_values = []

    for b in impact_params:
        pos_list = []
        vor_list = []

        for z in z_vals:
            pos_list.append([0, 0, z])
            vor_list.append([0, 0, 1.0])
        for x in x_vals:
            pos_list.append([x, b, 0])
            vor_list.append([1.0, 0, 0])

        pos = np.array(pos_list)
        vor = np.array(vor_list)

        stretching = 0
        Z = 0
        max_lambda = 0

        for i in range(len(pos)):
            S_i = compute_strain(compute_velocity_gradient(pos[i], pos, vor))
            stretching += vor[i] @ S_i @ vor[i]
            evals = np.sort(np.linalg.eigvalsh(S_i))[::-1]
            max_lambda = max(max_lambda, evals[0])
            Z += np.dot(vor[i], vor[i])

        C = abs(stretching) / (max_lambda * Z + 1e-15)
        c_values.append((b, C))
        print(f"  {b:>6.2f} | {stretching:>12.4f} | {max_lambda:>12.6f} | {Z:>8.1f} | {C:>14.6f}")

    # Maximum C at closest approach
    max_C = max(c[1] for c in c_values)
    print(f"\n  Maximum C (stretching majorant) across impact params: {max_C:.6f}")


def part4_energy_unitarity_bound():
    """
    The "unitarity bound" from energy conservation.

    Energy E = (1/2) integral |u|^2 is conserved (in Euler) or decreasing (in NS).
    This LIMITS the total stretching that can occur.

    For N vortex tubes of circulation Gamma, the energy is:
    E ~ sum_{i != j} Gamma_i * Gamma_j * log(|x_i - x_j| / epsilon)

    The stretching is:
    S = sum_i omega_i . S_i . omega_i

    Key question: does energy conservation provide a bound on S that is
    BETTER than the Ladyzhenskaya inequality?

    Test: for N tubes at various configurations, compute S/E and check
    if S/E has a universal upper bound that scales better than sqrt(Z/E).
    """
    print("\n" + "=" * 70)
    print("PART 4: ENERGY-UNITARITY BOUND")
    print("Does E conservation give a better bound on stretching than Ladyzhenskaya?")
    print("=" * 70)

    results = []

    for seed in range(100):
        np.random.seed(seed * 53 + 11)
        N = 15 + seed % 25
        scale = 0.2 + 0.3 * (seed % 4)
        pos = np.random.randn(N, 3) * scale
        vor = np.random.randn(N, 3)
        magnitude = 0.5 + 2.0 * (seed % 5) / 4
        vor = vor / (np.linalg.norm(vor, axis=1, keepdims=True) + 1e-10) * magnitude

        # Compute stretching
        stretching = 0
        Z = 0
        max_lambda = 0
        for i in range(N):
            grad_u = compute_velocity_gradient(pos[i], pos, vor)
            S_i = compute_strain(grad_u)
            stretching += vor[i] @ S_i @ vor[i]
            evals = np.sort(np.linalg.eigvalsh(S_i))[::-1]
            max_lambda = max(max_lambda, evals[0])
            Z += np.dot(vor[i], vor[i])

        # Compute energy (approximate: sum of pairwise log interactions)
        E = 0
        epsilon = 0.05
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                r_reg = np.sqrt(dist**2 + epsilon**2)
                Gi = np.linalg.norm(vor[i])
                Gj = np.linalg.norm(vor[j])
                E += Gi * Gj / (4 * np.pi * r_reg)

        # Various ratios
        C_majorant = abs(stretching) / (max_lambda * Z + 1e-15)
        S_over_E = abs(stretching) / (E + 1e-15)
        S_over_EZ = abs(stretching) / (np.sqrt(E * Z) + 1e-15)

        results.append({
            'N': N,
            'C_majorant': C_majorant,
            'S_over_E': S_over_E,
            'S_over_EZ': S_over_EZ,
            'stretching': abs(stretching),
            'E': E,
            'Z': Z,
        })

    # Analyze which bound is tightest
    C_maj = [r['C_majorant'] for r in results]
    S_E = [r['S_over_E'] for r in results]
    S_EZ = [r['S_over_EZ'] for r in results]

    print(f"\n  Ratio analysis across 100 random configs:")
    print(f"  {'Bound type':<30} | {'Mean':>8} | {'Max':>8} | {'Std':>8}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'|S| / (lambda_max * Z)':<30} | {np.mean(C_maj):>8.5f} | {np.max(C_maj):>8.5f} | {np.std(C_maj):>8.5f}")
    print(f"  {'|S| / E':<30} | {np.mean(S_E):>8.5f} | {np.max(S_E):>8.5f} | {np.std(S_E):>8.5f}")
    print(f"  {'|S| / sqrt(E * Z)':<30} | {np.mean(S_EZ):>8.5f} | {np.max(S_EZ):>8.5f} | {np.std(S_EZ):>8.5f}")

    # Check scaling with N
    print(f"\n  N-dependence of stretching majorant C:")
    for n_lo in range(15, 40, 5):
        n_hi = n_lo + 5
        c_bin = [r['C_majorant'] for r in results if n_lo <= r['N'] < n_hi]
        if len(c_bin) > 0:
            print(f"    N in [{n_lo},{n_hi}): mean C = {np.mean(c_bin):.5f}, max C = {np.max(c_bin):.5f}, n={len(c_bin)}")

    # The key test: does S/sqrt(E*Z) remain bounded?
    # In Ladyzhenskaya: |S| <= C * E^{1/4} * Z^{1/4} * P^{3/4}
    # If S/sqrt(E*Z) is bounded, and P ~ Z/volume, then
    # S ~ C * sqrt(E*Z) which gives dZ/dt <= C*sqrt(E)*Z - nu*P
    # This is SUBCRITICAL (Z grows at most as Z, not Z^{3/2})
    max_ratio = np.max(S_EZ)
    print(f"\n  CRITICAL TEST: max(|S| / sqrt(E*Z)) = {max_ratio:.6f}")
    if max_ratio < 10:
        print(f"  If this bound is universal, it would give:")
        print(f"  dZ/dt <= C*sqrt(E)*Z - nu*dissip")
        print(f"  which is LINEAR in Z (subcritical!) -> REGULARITY")
        print(f"  BUT: this needs proof for continuous fields, not just point vortices")


def part5_scale_dependence():
    """
    Does the depletion factor improve at finer scales?
    If C(epsilon) -> 0 as epsilon -> 0, we might have regularity.
    """
    print("\n" + "=" * 70)
    print("PART 5: SCALE DEPENDENCE OF DEPLETION")
    print("Does the stretching majorant C decrease at finer scales?")
    print("=" * 70)

    np.random.seed(42)
    N = 30
    pos = np.random.randn(N, 3) * 0.5
    vor = np.random.randn(N, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)

    epsilons = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    print(f"\n  {'epsilon':>8} | {'C_majorant':>12} | {'Max lambda':>12} | {'Stretching':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for eps in epsilons:
        stretching = 0
        Z = 0
        max_lambda = 0
        for i in range(N):
            S_i = compute_strain(compute_velocity_gradient(pos[i], pos, vor, epsilon=eps))
            stretching += vor[i] @ S_i @ vor[i]
            evals = np.sort(np.linalg.eigvalsh(S_i))[::-1]
            max_lambda = max(max_lambda, evals[0])
            Z += np.dot(vor[i], vor[i])

        C = abs(stretching) / (max_lambda * Z + 1e-15)
        print(f"  {eps:>8.4f} | {C:>12.6f} | {max_lambda:>12.4f} | {stretching:>12.4f}")


if __name__ == "__main__":
    part1_bilinear_form()
    part2_scattering_cross_section()
    part3_perpendicular_scattering()
    part4_energy_unitarity_bound()
    part5_scale_dependence()
