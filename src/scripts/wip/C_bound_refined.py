"""
C_bound_refined.py

Refined analysis of C = |Stretching| / (max_possible_stretching)

Two metrics:
  C_global = |sum omega_i . S_i . omega_i| / (||S||_inf * Z)
  C_local  = |sum omega_i . S_i . omega_i| / sum(lambda_max_i * |omega_i|^2)

C_local = 1 means EVERY omega is perfectly aligned with its local alpha-eigenvector.
C_local < 1 means the Biot-Savart coupling prevents perfect self-alignment.

Also: adversarial configurations designed to MAXIMIZE stretching.
"""

import numpy as np
from scipy.linalg import eigh


def levi_civita(i, j, k):
    if (i, j, k) in [(0,1,2), (1,2,0), (2,0,1)]:
        return 1
    elif (i, j, k) in [(0,2,1), (2,1,0), (1,0,2)]:
        return -1
    return 0


def compute_strain_at_point(x, positions, vorticities, epsilon=0.05):
    """Compute velocity gradient and strain at point x from all vortices."""
    grad_u = np.zeros((3, 3))
    for j in range(len(positions)):
        r = x - positions[j]
        r_norm = np.linalg.norm(r)
        r_reg = np.sqrt(r_norm**2 + epsilon**2)
        if r_norm < 1e-12:
            continue
        for a in range(3):
            for k in range(3):
                for b in range(3):
                    for m in range(3):
                        eps = levi_civita(a, b, m)
                        if eps == 0:
                            continue
                        t1 = (1 if m == k else 0) / r_reg**3
                        t2 = -3 * r[m] * r[k] / r_reg**5
                        grad_u[a, k] += eps * vorticities[j][b] * (t1 + t2) / (4 * np.pi)
    S = 0.5 * (grad_u + grad_u.T)
    return S


def compute_C_metrics(positions, vorticities, epsilon=0.05):
    """Compute both C_global and C_local."""
    N = len(positions)
    Z = sum(np.dot(vorticities[i], vorticities[i]) for i in range(N))

    stretching_total = 0
    max_possible_local = 0  # sum of lambda_max_i * |omega_i|^2
    max_S_eigenvalue = 0

    for i in range(N):
        S_i = compute_strain_at_point(positions[i], positions, vorticities, epsilon)
        stretch_i = vorticities[i] @ S_i @ vorticities[i]
        stretching_total += stretch_i

        evals = np.sort(np.linalg.eigvalsh(S_i))[::-1]
        lambda_max_i = evals[0]
        max_S_eigenvalue = max(max_S_eigenvalue, lambda_max_i)

        omega_sq_i = np.dot(vorticities[i], vorticities[i])
        max_possible_local += lambda_max_i * omega_sq_i

    C_global = abs(stretching_total) / (max_S_eigenvalue * Z + 1e-15)
    C_local = abs(stretching_total) / (max_possible_local + 1e-15)

    # Also compute the alignment: what fraction of max-possible is achieved?
    # Signed version (positive = stretching, negative = compression)
    C_signed = stretching_total / (max_possible_local + 1e-15)

    return C_global, C_local, C_signed, stretching_total


def run_refined_analysis():
    print("=" * 70)
    print("REFINED C-BOUND ANALYSIS")
    print("Two metrics: C_global (biased low) and C_local (fair)")
    print("=" * 70)

    configs = []

    # 1. Perpendicular interacting tubes (adversarial: one stretches the other)
    print("\n--- Perpendicular tubes (adversarial) ---")
    positions_list = []
    vorticities_list = []
    for z in np.linspace(-2, 2, 30):
        positions_list.append([0, 0, z])
        vorticities_list.append([0, 0, 1.0])  # Tube 1: along z
    for x in np.linspace(-2, 2, 30):
        positions_list.append([x, 0.3, 0])
        vorticities_list.append([1.0, 0, 0])  # Tube 2: along x, offset in y
    pos = np.array(positions_list)
    vor = np.array(vorticities_list)
    cg, cl, cs, st = compute_C_metrics(pos, vor)
    print(f"  C_global={cg:.6f}, C_local={cl:.6f}, C_signed={cs:.6f}")
    configs.append(("Perpendicular tubes", cg, cl, cs))

    # 2. Orthogonal triple (3 tubes along x, y, z)
    print("\n--- Orthogonal triple ---")
    positions_list = []
    vorticities_list = []
    for val in np.linspace(-2, 2, 20):
        positions_list.append([val, 0, 0])
        vorticities_list.append([1, 0, 0])
        positions_list.append([0, val, 0])
        vorticities_list.append([0, 1, 0])
        positions_list.append([0, 0, val])
        vorticities_list.append([0, 0, 1])
    pos = np.array(positions_list)
    vor = np.array(vorticities_list)
    cg, cl, cs, st = compute_C_metrics(pos, vor)
    print(f"  C_global={cg:.6f}, C_local={cl:.6f}, C_signed={cs:.6f}")
    configs.append(("Orthogonal triple", cg, cl, cs))

    # 3. Converging flow (all omega pointing inward)
    print("\n--- Converging flow ---")
    np.random.seed(123)
    N = 40
    pos = np.random.randn(N, 3)
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)  # On unit sphere
    vor = -pos  # Pointing inward (convergence)
    cg, cl, cs, st = compute_C_metrics(pos, vor)
    print(f"  C_global={cg:.6f}, C_local={cl:.6f}, C_signed={cs:.6f}")
    configs.append(("Converging flow", cg, cl, cs))

    # 4. Concentrated vortices (near-singularity test)
    print("\n--- Concentrated vortices (near-singularity) ---")
    N = 20
    pos = np.random.randn(N, 3) * 0.1  # Very close together
    vor = np.random.randn(N, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)
    vor *= 10  # Strong vorticity
    cg, cl, cs, st = compute_C_metrics(pos, vor)
    print(f"  C_global={cg:.6f}, C_local={cl:.6f}, C_signed={cs:.6f}")
    configs.append(("Concentrated", cg, cl, cs))

    # 5. Aligned with strain (try to cheat by pre-aligning)
    print("\n--- Pre-aligned with strain (iterative) ---")
    np.random.seed(99)
    N = 25
    pos = np.random.randn(N, 3) * 0.5
    vor = np.random.randn(N, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)

    # Iterate: compute strain, align omega with alpha, repeat
    for iteration in range(5):
        for i in range(N):
            S_i = compute_strain_at_point(pos[i], pos, vor)
            evals, evecs = eigh(S_i)
            idx = np.argsort(evals)[::-1]
            evecs = evecs[:, idx]
            # Align omega with most-stretching eigenvector
            omega_mag = np.linalg.norm(vor[i])
            vor[i] = evecs[:, 0] * omega_mag

    cg, cl, cs, st = compute_C_metrics(pos, vor)
    print(f"  C_global={cg:.6f}, C_local={cl:.6f}, C_signed={cs:.6f}")
    print(f"  (after 5 iterations of alpha-alignment)")
    configs.append(("Pre-aligned (5 iter)", cg, cl, cs))

    # 6. More alignment iterations
    print("\n--- Pre-aligned with strain (20 iterations) ---")
    np.random.seed(99)
    N = 25
    pos = np.random.randn(N, 3) * 0.5
    vor = np.random.randn(N, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)

    for iteration in range(20):
        for i in range(N):
            S_i = compute_strain_at_point(pos[i], pos, vor)
            evals, evecs = eigh(S_i)
            idx = np.argsort(evals)[::-1]
            evecs = evecs[:, idx]
            omega_mag = np.linalg.norm(vor[i])
            vor[i] = evecs[:, 0] * omega_mag

    cg, cl, cs, st = compute_C_metrics(pos, vor)
    print(f"  C_global={cg:.6f}, C_local={cl:.6f}, C_signed={cs:.6f}")
    configs.append(("Pre-aligned (20 iter)", cg, cl, cs))

    # 7. Adversarial random search
    print("\n--- Adversarial random search (100 trials) ---")
    best_C_local = 0
    for trial in range(100):
        np.random.seed(trial * 71 + 13)
        N = np.random.randint(10, 40)
        scale = np.random.uniform(0.1, 2.0)
        pos = np.random.randn(N, 3) * scale
        vor = np.random.randn(N, 3)
        mag = np.random.uniform(0.5, 5.0)
        vor *= mag / (np.linalg.norm(vor, axis=1, keepdims=True) + 1e-10)

        _, cl, _, _ = compute_C_metrics(pos, vor)
        if cl > best_C_local:
            best_C_local = cl
            best_trial = trial
    print(f"  Best C_local = {best_C_local:.6f} (trial {best_trial})")
    configs.append(("Random search best", 0, best_C_local, 0))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<25} | {'C_global':>8} | {'C_local':>8} | {'C_signed':>8}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    max_cl = 0
    for name, cg, cl, cs in configs:
        print(f"  {name:<25} | {cg:>8.4f} | {cl:>8.4f} | {cs:>+8.4f}")
        max_cl = max(max_cl, cl)

    print(f"\n  Maximum C_local found: {max_cl:.6f}")
    if max_cl < 1.0:
        print(f"  C_local < 1 HOLDS for all configs tested!")
        print(f"  Even with iterative alpha-alignment, C cannot reach 1.")
    else:
        print(f"  C_local >= 1 FOUND — bound is REFUTED")


if __name__ == "__main__":
    run_refined_analysis()
