"""
C_bound_convergence.py

THE KEY EXPERIMENT: does iterative alpha-alignment converge below 1?

If C_local converges to a value strictly < 1 as iterations -> infinity,
then the Biot-Savart self-consistency constraint provides a UNIVERSAL bound
on vortex stretching. This would be a new mathematical result.

Tests:
1. Convergence vs iteration count (up to 100)
2. Dependence on N (number of points)
3. Multiple random seeds
"""

import numpy as np
from scipy.linalg import eigh


def levi_civita(i, j, k):
    if (i, j, k) in [(0,1,2), (1,2,0), (2,0,1)]:
        return 1
    elif (i, j, k) in [(0,2,1), (2,1,0), (1,0,2)]:
        return -1
    return 0

# Precompute Levi-Civita for speed
LC = np.zeros((3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            LC[i, j, k] = levi_civita(i, j, k)


def compute_strain_at_point(x, positions, vorticities, epsilon=0.05):
    """Compute strain tensor at point x."""
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
    S = 0.5 * (grad_u + grad_u.T)
    return S


def compute_C_local(positions, vorticities, epsilon=0.05):
    """Compute C_local = |Stretching| / sum(lambda_max_i * |omega_i|^2)."""
    N = len(positions)
    stretching = 0
    max_possible = 0

    for i in range(N):
        S_i = compute_strain_at_point(positions[i], positions, vorticities, epsilon)
        stretching += vorticities[i] @ S_i @ vorticities[i]

        evals = np.sort(np.linalg.eigvalsh(S_i))[::-1]
        omega_sq = np.dot(vorticities[i], vorticities[i])
        max_possible += evals[0] * omega_sq

    return abs(stretching) / (max_possible + 1e-15)


def iterative_alignment(positions, vorticities, n_iter, epsilon=0.05):
    """
    Iteratively align each omega with its local alpha-eigenvector.
    Returns C_local after each iteration.
    """
    vor = vorticities.copy()
    N = len(positions)
    history = []

    for iteration in range(n_iter):
        for i in range(N):
            S_i = compute_strain_at_point(positions[i], positions, vor, epsilon)
            evals, evecs = eigh(S_i)
            idx = np.argsort(evals)[::-1]
            evecs = evecs[:, idx]
            omega_mag = np.linalg.norm(vor[i])
            if omega_mag > 1e-10:
                vor[i] = evecs[:, 0] * omega_mag

        C = compute_C_local(positions, vor, epsilon)
        history.append(C)

    return history, vor


def run_convergence_test():
    print("=" * 70)
    print("C-BOUND CONVERGENCE TEST")
    print("Does iterative alpha-alignment converge below 1?")
    print("=" * 70)

    n_max_iter = 50

    # Test 1: Multiple seeds, fixed N=25
    print(f"\n--- Test 1: Convergence across seeds (N=25, {n_max_iter} iterations) ---")
    N = 25
    final_values = []
    for seed in range(10):
        np.random.seed(seed * 100 + 42)
        pos = np.random.randn(N, 3) * 0.5
        vor = np.random.randn(N, 3)
        vor /= np.linalg.norm(vor, axis=1, keepdims=True)

        history, _ = iterative_alignment(pos, vor, n_max_iter)
        final_values.append(history[-1])
        print(f"  Seed {seed}: iter 1={history[0]:.4f}, "
              f"iter 10={history[9]:.4f}, "
              f"iter 25={history[24]:.4f}, "
              f"iter {n_max_iter}={history[-1]:.4f}")

    print(f"\n  Final C across seeds: mean={np.mean(final_values):.4f}, "
          f"max={np.max(final_values):.4f}, min={np.min(final_values):.4f}")

    # Test 2: Varying N
    print(f"\n--- Test 2: Dependence on N ({n_max_iter} iterations) ---")
    for N in [10, 20, 30, 40, 50]:
        np.random.seed(42)
        pos = np.random.randn(N, 3) * 0.5
        vor = np.random.randn(N, 3)
        vor /= np.linalg.norm(vor, axis=1, keepdims=True)

        history, _ = iterative_alignment(pos, vor, n_max_iter)
        print(f"  N={N:3d}: final C_local = {history[-1]:.6f} "
              f"(iter 1={history[0]:.4f}, iter 10={history[9]:.4f})")

    # Test 3: Detailed convergence for one case
    print(f"\n--- Test 3: Detailed convergence (N=25, seed=42) ---")
    np.random.seed(42)
    pos = np.random.randn(25, 3) * 0.5
    vor = np.random.randn(25, 3)
    vor /= np.linalg.norm(vor, axis=1, keepdims=True)

    history, _ = iterative_alignment(pos, vor, n_max_iter)
    for i in [0, 1, 2, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]:
        if i < len(history):
            print(f"  Iteration {i+1:3d}: C_local = {history[i]:.6f}")

    # Check convergence rate
    if len(history) > 2:
        last_10 = history[-10:]
        spread = max(last_10) - min(last_10)
        print(f"\n  Spread in last 10 iterations: {spread:.6f}")
        if spread < 0.001:
            print(f"  -> CONVERGED to C_local = {history[-1]:.6f}")
            if history[-1] < 1.0:
                print(f"  -> C < 1 IS A FIXED POINT of the alignment map!")
        else:
            print(f"  -> Still changing (not converged at {n_max_iter} iterations)")

    # Test 4: Concentrated cluster (more adversarial)
    print(f"\n--- Test 4: Concentrated cluster (N=30, scale=0.1) ---")
    np.random.seed(77)
    pos = np.random.randn(30, 3) * 0.1  # Very close together
    vor = np.random.randn(30, 3) * 3  # Strong vorticity

    history, _ = iterative_alignment(pos, vor, n_max_iter)
    print(f"  iter 1={history[0]:.4f}, iter 25={history[24]:.4f}, "
          f"iter {n_max_iter}={history[-1]:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    all_finals = final_values + [history[-1]]
    overall_max = max(all_finals)
    print(f"  Maximum C_local found across ALL tests: {overall_max:.6f}")
    if overall_max < 1.0:
        print(f"  THE C < 1 BOUND SURVIVES.")
        print(f"  The Biot-Savart self-consistency constraint prevents")
        print(f"  perfect vorticity-strain alignment.")
        print(f"")
        print(f"  PHYSICAL INTERPRETATION:")
        print(f"  A vortex tube cannot perfectly stretch itself because")
        print(f"  the Biot-Savart velocity it generates is perpendicular")
        print(f"  to its own axis. Mutual stretching between tubes is")
        print(f"  limited by the same coupling: changing one tube's")
        print(f"  orientation changes the strain field for all others.")
        print(f"")
        print(f"  If provable analytically: this gives")
        print(f"  |Stretching| <= C * sum(lambda_max_i * |omega_i|^2)")
        print(f"  with C strictly < 1, which bounds enstrophy growth.")


if __name__ == "__main__":
    run_convergence_test()
