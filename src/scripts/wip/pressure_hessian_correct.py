"""
pressure_hessian_correct.py

CORRECT implementation of the Restricted Euler model with
Chevillard-Meneveau (2006) "Recent Fluid Deformation" (RFD) closure.

Physics:
  The velocity gradient tensor A = nabla(u) evolves as:
    dA/dt = -A^2 - H + viscous
  where H = (1/rho) * Hessian(p) is the pressure Hessian.

  Incompressibility: tr(A) = 0, so tr(H) = -tr(A^2).

  Mode A (Restricted Euler): H = -(tr(A^2)/3)*I (isotropic only)
    -> dA/dt = -A^2 + (tr(A^2)/3)*I
    -> Known to blow up in finite time (Vieillefosse 1984)
    -> Vorticity aligns with MOST STRETCHING eigenvector (alpha)

  Mode B (RFD closure): H = -tr(A^2) * C/tr(C)
    where C is the Cauchy-Green deformation tensor tracking recent fluid deformation:
      dC/dt = C*A^T + A*C - (1/T)(C - I)  [with viscous relaxation]
    C(0) = I

    -> dA/dt = -A^2 + tr(A^2) * C/tr(C) + viscous
    -> When C = I: reduces to restricted Euler (isotropic pressure)
    -> When C anisotropic: pressure opposes most-stretched direction
    -> Expected: INTERMEDIATE alignment (beta), no blow-up

  Verification: tr(dA/dt) = -tr(A^2) + tr(A^2)*tr(C)/tr(C) = 0  (always)

Key question: Does Mode B show intermediate (beta) alignment while Mode A
shows alpha alignment? If yes -> pressure Hessian is the depletion mechanism.

Reference: Chevillard & Meneveau (2006), PRL 97:174501
           Chevillard et al. (2008), Physics of Fluids 20:101504
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh


def restricted_euler_rfd(t, state, mode='A', T_relax=1.0, T_visc=10.0):
    """
    ODE right-hand side for the velocity gradient + Cauchy-Green system.

    state = [A_flat (9), C_flat (9)] = 18 components
    A: velocity gradient tensor (3x3, trace-free)
    C: Cauchy-Green deformation tensor (3x3, symmetric positive-definite)

    Parameters:
        mode: 'A' (restricted Euler) or 'B' (RFD closure)
        T_relax: viscous relaxation timescale for C (Kolmogorov-scale proxy)
        T_visc: viscous damping timescale for A
    """
    A = state[:9].reshape((3, 3))
    C = state[9:].reshape((3, 3))

    A2 = A @ A
    tr_A2 = np.trace(A2)

    if mode == 'A':
        # Restricted Euler: H = -(tr(A^2)/3)*I (isotropic pressure only)
        # dA/dt = -A^2 + (tr(A^2)/3)*I - A/T_visc
        pressure_term = (tr_A2 / 3.0) * np.eye(3)
    else:
        # RFD closure: H = -tr(A^2) * C/tr(C)
        # dA/dt = -A^2 + tr(A^2) * C/tr(C) - A/T_visc
        tr_C = np.trace(C)
        if tr_C < 1e-12:
            tr_C = 1e-12  # Safety
        pressure_term = tr_A2 * (C / tr_C)

    # Velocity gradient evolution
    dA = -A2 + pressure_term - A / T_visc

    # Cauchy-Green evolution (always evolves, even in Mode A for consistency)
    # dC/dt = C*A^T + A*C - (1/T_relax)*(C - I)
    dC = C @ A.T + A @ C - (1.0 / T_relax) * (C - np.eye(3))

    return np.concatenate([dA.flatten(), dC.flatten()])


def compute_alignment_from_A(A):
    """
    Extract vorticity and strain from A, compute alignment with all 3 eigenvectors.

    Returns: (cos_alpha, cos_beta, cos_gamma, omega_mag, evals)
    """
    # Strain = symmetric part of A
    S = 0.5 * (A + A.T)

    # Vorticity vector from antisymmetric part: omega_i = epsilon_ijk * W_jk
    # W = (A - A^T)/2
    omega = np.array([
        A[2, 1] - A[1, 2],
        A[0, 2] - A[2, 0],
        A[1, 0] - A[0, 1]
    ])
    omega_mag = np.linalg.norm(omega)

    if omega_mag < 1e-12:
        return 0.0, 0.0, 0.0, 0.0, np.zeros(3)

    xi = omega / omega_mag

    # Strain eigendecomposition (ascending order from eigh)
    evals, evecs = eigh(S)
    # Sort descending: alpha >= beta >= gamma
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    cos_alpha = abs(np.dot(xi, evecs[:, 0]))  # Most stretching
    cos_beta = abs(np.dot(xi, evecs[:, 1]))   # Intermediate
    cos_gamma = abs(np.dot(xi, evecs[:, 2]))   # Most compressing

    return cos_alpha, cos_beta, cos_gamma, omega_mag, evals


def run_single_trial(mode, T_relax=1.0, T_visc=10.0, t_end=20.0, seed=None):
    """
    Run one trial: random trace-free initial A, C = I.
    Returns time series of alignment + blow-up flag.
    """
    if seed is not None:
        np.random.seed(seed)

    # Random initial A (trace-free)
    A0 = np.random.randn(3, 3)
    A0 -= (np.trace(A0) / 3.0) * np.eye(3)

    # Initial C = identity (no prior deformation)
    C0 = np.eye(3)

    state0 = np.concatenate([A0.flatten(), C0.flatten()])

    # Integration with event detection for blow-up
    t_eval = np.linspace(0, t_end, 500)

    try:
        sol = solve_ivp(
            restricted_euler_rfd, (0, t_end), state0,
            args=(mode, T_relax, T_visc),
            method='RK45', t_eval=t_eval,
            max_step=0.02,
            rtol=1e-8, atol=1e-10
        )

        if not sol.success:
            return None, True  # Blow-up

        # Check for numerical blow-up
        if np.any(np.abs(sol.y) > 1e8) or np.any(~np.isfinite(sol.y)):
            return None, True

        # Compute alignment at each timestep
        n_steps = sol.y.shape[1]
        alignments = np.zeros((n_steps, 5))  # ca, cb, cg, omega_mag, time

        for i in range(n_steps):
            A = sol.y[:9, i].reshape((3, 3))
            ca, cb, cg, om, evals = compute_alignment_from_A(A)
            alignments[i] = [ca, cb, cg, om, sol.t[i]]

        return alignments, False

    except Exception:
        return None, True  # Blow-up or numerical failure


def run_experiment_stabilized(n_trials=200):
    """
    REVISED experiment: use strong enough viscous damping to prevent ALL blow-ups.
    This isolates the alignment question from the blow-up question.

    Key insight: we want to ask "given the same survival conditions, does
    the pressure Hessian change WHERE vorticity points?" — not WHETHER it blows up.
    """

    print("=" * 72)
    print("PRESSURE HESSIAN TEST v2: ALIGNMENT UNDER STABILIZED DYNAMICS")
    print("Chevillard-Meneveau (2006) RFD closure, strong damping")
    print("=" * 72)

    # Sweep viscous damping to find regime where both modes survive
    # but dynamics are still nontrivial
    configs = [
        (0.1, 1.0, "Fast C-relax, moderate visc"),
        (0.5, 2.0, "Medium C-relax, moderate visc"),
        (1.0, 2.0, "Slow C-relax, moderate visc"),
        (0.1, 5.0, "Fast C-relax, slow visc"),
        (1.0, 5.0, "Slow C-relax, slow visc"),
    ]
    t_end = 15.0

    for T_relax, T_visc, config_label in configs:
        print(f"\n{'-' * 72}")
        print(f"CONFIG: T_relax={T_relax}, T_visc={T_visc} ({config_label})")
        print(f"{'-' * 72}")

        for mode, label in [('A', 'Restricted Euler'),
                            ('B', 'RFD Closure')]:

            blow_ups = 0
            avg_alpha = []
            avg_beta = []
            avg_gamma = []

            # Also track early alignment (first 25%) vs late (last 25%)
            early_alpha = []
            early_beta = []
            late_alpha = []
            late_beta = []

            for trial in range(n_trials):
                alignments, blew_up = run_single_trial(
                    mode, T_relax, T_visc, t_end, seed=trial * 137 + 42
                )

                if blew_up:
                    blow_ups += 1
                    continue

                n_pts = len(alignments)
                q1 = n_pts // 4
                q3 = n_pts * 3 // 4

                # Time-averaged alignment (full)
                avg_alpha.append(np.mean(alignments[:, 0]))
                avg_beta.append(np.mean(alignments[:, 1]))
                avg_gamma.append(np.mean(alignments[:, 2]))

                # Early vs late
                early_alpha.append(np.mean(alignments[:q1, 0]))
                early_beta.append(np.mean(alignments[:q1, 1]))
                late_alpha.append(np.mean(alignments[q3:, 0]))
                late_beta.append(np.mean(alignments[q3:, 1]))

            n_survived = n_trials - blow_ups

            if n_survived < 10:
                print(f"  Mode {mode} ({label}): {blow_ups}/{n_trials} blew up — skipping")
                continue

            aa = np.mean(avg_alpha)
            ab = np.mean(avg_beta)
            ag = np.mean(avg_gamma)

            ea = np.mean(early_alpha)
            eb = np.mean(early_beta)
            la = np.mean(late_alpha)
            lb = np.mean(late_beta)

            dominant = 'alpha' if aa > ab and aa > ag else ('beta' if ab > ag else 'gamma')
            marker = " <<<" if dominant == 'beta' else ""

            print(f"  Mode {mode} ({label}): survived {n_survived}/{n_trials}")
            print(f"    Overall:  alpha={aa:.4f}  beta={ab:.4f}  gamma={ag:.4f}  -> {dominant}{marker}")
            print(f"    Early:    alpha={ea:.4f}  beta={eb:.4f}")
            print(f"    Late:     alpha={la:.4f}  beta={lb:.4f}")

            # Shift: does beta alignment INCREASE over time?
            shift = lb - eb
            print(f"    beta shift (late-early): {shift:+.4f} "
                  f"({'beta growing' if shift > 0.01 else 'stable' if abs(shift) < 0.01 else 'beta declining'})")

    # Trace-free verification
    print(f"\n{'=' * 72}")
    print("TRACE-FREE CHECK:")
    A_test = np.random.randn(3, 3)
    A_test -= (np.trace(A_test) / 3.0) * np.eye(3)
    C_test = np.eye(3) + 0.1 * np.random.randn(3, 3)
    C_test = 0.5 * (C_test + C_test.T)

    state_test = np.concatenate([A_test.flatten(), C_test.flatten()])
    dstate_A = restricted_euler_rfd(0, state_test, mode='A')
    dstate_B = restricted_euler_rfd(0, state_test, mode='B')
    print(f"  Mode A: tr(dA/dt) = {np.trace(dstate_A[:9].reshape(3,3)):.2e}")
    print(f"  Mode B: tr(dA/dt) = {np.trace(dstate_B[:9].reshape(3,3)):.2e}")


if __name__ == "__main__":
    run_experiment_stabilized()
