"""
xi_evolution_depletion.py

Vector 3: Prove that vortex stretching depletion is dynamically FORCED.

The vorticity direction xi = omega/|omega| evolves under:
  D(xi)/Dt = (S - (xi.S.xi)I).xi + viscous_correction

Key question: Does xi always align with the INTERMEDIATE eigenvector of S
(not the most stretching one), regardless of initial conditions?

If YES -> depletion is universal -> path to regularity.
If NO -> depletion is not forced -> need different approach.

Physics:
- S is the strain rate tensor (symmetric, trace-free because div-free)
- trace(S) = 0 IS the pressure constraint (Tao's key insight)
- S has eigenvalues alpha >= beta >= gamma, alpha + beta + gamma = 0
- Most stretching: alpha direction. Intermediate: beta. Most compressing: gamma.
- Depletion = xi aligning with beta (intermediate), not alpha (most stretching)

Approach:
1. N filaments with positions and directions
2. Biot-Savart strain computation (automatically trace-free)
3. Time evolution of xi under inviscid + viscous dynamics
4. Track alignment with all 3 strain eigenvectors
5. Multiple initial conditions (parallel, random, adversarial)
6. Statistics over many trials
"""

import numpy as np
from scipy.linalg import eigh

# Regularized Biot-Savart kernel
EPSILON = 0.1  # Regularization (filament core size)


def biot_savart_velocity_gradient(x, filaments_pos, filaments_omega):
    """
    Compute velocity gradient tensor at point x due to all filaments.

    The Biot-Savart law: u(x) = -(1/4pi) integral omega(y) x (x-y)/|x-y|^3 dy

    For point vortices: u(x) = sum_j (omega_j x r_j) / (4pi |r_j|^3)
    where r_j = x - y_j, regularized with epsilon.

    Returns the 3x3 velocity gradient tensor du_i/dx_j.
    """
    grad_u = np.zeros((3, 3))

    for j in range(len(filaments_pos)):
        r = x - filaments_pos[j]
        r_norm = np.linalg.norm(r)
        r_reg = np.sqrt(r_norm**2 + EPSILON**2)

        if r_norm < 1e-12:
            continue  # Skip self-interaction

        omega_j = filaments_omega[j]

        # Gradient of (omega x r) / |r|^3 (regularized)
        # d/dx_k [ eps_ilm omega_l r_m / r_reg^3 ]
        # = eps_ilm omega_l [ delta_mk / r_reg^3 - 3 r_m r_k / r_reg^5 ]

        for i in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        # Levi-Civita
                        eps = levi_civita(i, l, m)
                        if eps == 0:
                            continue

                        term1 = (1 if m == k else 0) / r_reg**3
                        term2 = -3 * r[m] * r[k] / r_reg**5

                        grad_u[i, k] += eps * omega_j[l] * (term1 + term2) / (4 * np.pi)

    return grad_u


def levi_civita(i, j, k):
    """Levi-Civita symbol."""
    if (i, j, k) in [(0,1,2), (1,2,0), (2,0,1)]:
        return 1
    elif (i, j, k) in [(0,2,1), (2,1,0), (1,0,2)]:
        return -1
    return 0


def extract_strain(grad_u):
    """
    Extract strain rate tensor S = (grad_u + grad_u^T) / 2.
    This is automatically trace-free for divergence-free flows.
    """
    S = 0.5 * (grad_u + grad_u.T)
    return S


def strain_eigenvectors(S):
    """
    Compute eigenvalues and eigenvectors of strain tensor.
    Returns (eigenvalues, eigenvectors) sorted: alpha >= beta >= gamma.
    eigenvalues[0] = alpha (most stretching)
    eigenvalues[1] = beta (intermediate)
    eigenvalues[2] = gamma (most compressing)
    """
    evals, evecs = eigh(S)
    # eigh returns ascending order, we want descending
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


def compute_alignment(xi, S):
    """
    Compute alignment of xi with all 3 strain eigenvectors.
    Returns (cos_alpha, cos_beta, cos_gamma, stretching_rate).
    """
    evals, evecs = strain_eigenvectors(S)

    cos_alpha = abs(np.dot(xi, evecs[:, 0]))  # Most stretching
    cos_beta = abs(np.dot(xi, evecs[:, 1]))   # Intermediate
    cos_gamma = abs(np.dot(xi, evecs[:, 2]))   # Most compressing

    stretching = xi @ S @ xi  # = |omega| * (xi.S.xi) = effective stretching rate

    return cos_alpha, cos_beta, cos_gamma, stretching, evals


def evolve_xi_inviscid(xi, S, dt):
    """
    Evolve xi under inviscid dynamics:
    d(xi)/dt = (S - (xi.S.xi)I).xi = S.xi - (xi.S.xi)*xi

    This rotates xi toward the most stretching eigenvector of S.
    """
    sigma = xi @ S @ xi
    dxi = S @ xi - sigma * xi
    xi_new = xi + dt * dxi
    norm = np.linalg.norm(xi_new)
    if norm < 1e-12 or not np.isfinite(norm):
        return xi  # Return unchanged if degenerate
    xi_new /= norm
    return xi_new


def evolve_xi_with_viscosity(xi, S, omega_mag, grad_omega_mag, laplacian_omega_vec, nu, dt):
    """
    Evolve xi with viscous correction:
    d(xi)/dt = (S - (xi.S.xi)I).xi + nu * (Delta_omega / |omega| - xi * Delta|omega| / |omega|)

    The viscous term pushes xi AWAY from the most stretching direction.
    """
    sigma = xi @ S @ xi
    dxi_inviscid = S @ xi - sigma * xi

    # Viscous correction (simplified model)
    if omega_mag > 1e-10:
        visc_term = nu * (laplacian_omega_vec / omega_mag - xi * np.dot(xi, laplacian_omega_vec) / omega_mag)
    else:
        visc_term = np.zeros(3)

    xi_new = xi + dt * (dxi_inviscid + visc_term)
    norm = np.linalg.norm(xi_new)
    if norm < 1e-12 or not np.isfinite(norm):
        return xi
    xi_new /= norm
    return xi_new


def model_viscous_correction(xi, S, omega_mag, nu, filament_width):
    """
    Model the viscous correction for a Gaussian filament.
    For a filament of width sigma, Delta_omega ~ -omega/sigma^2 (transverse)
    + curvature corrections.

    Returns approximate laplacian_omega vector.
    """
    # Dominant contribution: transverse diffusion kills the filament
    # Delta_omega ~ -(2/sigma^2) * omega (for Gaussian profile)
    laplacian_omega = -(2.0 / filament_width**2) * omega_mag * xi

    # Add a RANDOM perturbation representing curvature effects
    # This is where the pressure Hessian enters implicitly
    perturb = np.random.randn(3) * omega_mag / (filament_width * 5)
    laplacian_omega += perturb

    return laplacian_omega


def run_single_trial(n_filaments, init_type, nu, n_steps, dt, filament_width=0.2):
    """
    Run one trial of xi evolution.

    init_type: 'parallel', 'random', 'adversarial_alpha', 'adversarial_gamma'

    Returns time series of alignment statistics.
    """
    # Generate filament positions (random in unit ball)
    positions = np.random.randn(n_filaments, 3) * 0.5

    # Generate initial directions based on init_type
    if init_type == 'parallel':
        # All filaments point in z-direction
        directions = np.tile([0, 0, 1.0], (n_filaments, 1))
    elif init_type == 'random':
        # Random directions on S^2
        directions = np.random.randn(n_filaments, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    elif init_type == 'adversarial_alpha':
        # Start aligned with MOST stretching direction (worst case for depletion)
        # We'll set initial directions after computing strain
        directions = np.random.randn(n_filaments, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    elif init_type == 'adversarial_gamma':
        # Start aligned with MOST compressing direction
        directions = np.random.randn(n_filaments, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Initial vorticity magnitudes (log-normal for intermittency)
    omega_mags = np.exp(np.random.randn(n_filaments) * 0.5 + 1.0)

    # Build initial omega vectors
    omegas = directions * omega_mags[:, None]

    # For adversarial init: align with strain eigenvectors
    if init_type in ['adversarial_alpha', 'adversarial_gamma']:
        for i in range(n_filaments):
            S_i = extract_strain(biot_savart_velocity_gradient(
                positions[i], positions, omegas))
            evals, evecs = strain_eigenvectors(S_i)
            if init_type == 'adversarial_alpha':
                directions[i] = evecs[:, 0]  # Most stretching
            else:
                directions[i] = evecs[:, 2]  # Most compressing
            omegas[i] = directions[i] * omega_mags[i]

    # Time evolution
    history = {
        'cos_alpha': np.zeros((n_steps, n_filaments)),
        'cos_beta': np.zeros((n_steps, n_filaments)),
        'cos_gamma': np.zeros((n_steps, n_filaments)),
        'stretching': np.zeros((n_steps, n_filaments)),
        'evals': np.zeros((n_steps, n_filaments, 3)),
    }

    for step in range(n_steps):
        for i in range(n_filaments):
            # Compute strain at filament i
            grad_u = biot_savart_velocity_gradient(positions[i], positions, omegas)
            S_i = extract_strain(grad_u)

            # Record alignment
            xi = directions[i]
            ca, cb, cg, stretch, evals_i = compute_alignment(xi, S_i)
            history['cos_alpha'][step, i] = ca
            history['cos_beta'][step, i] = cb
            history['cos_gamma'][step, i] = cg
            history['stretching'][step, i] = stretch
            history['evals'][step, i] = evals_i

            # Evolve xi
            if nu > 0:
                lap_omega = model_viscous_correction(
                    xi, S_i, omega_mags[i], nu, filament_width)
                directions[i] = evolve_xi_with_viscosity(
                    xi, S_i, omega_mags[i], None, lap_omega, nu, dt)
            else:
                directions[i] = evolve_xi_inviscid(xi, S_i, dt)

            # Update omega vector (direction changes, magnitude FIXED)
            # We keep |omega| fixed to study alignment dynamics without blow-up
            omegas[i] = directions[i] * omega_mags[i]

        # Also evolve positions (advection by local velocity - simplified)
        for i in range(n_filaments):
            # Simple model: filaments move slowly
            grad_u = biot_savart_velocity_gradient(positions[i], positions, omegas)
            u_local = np.zeros(3)  # Would need full velocity, not just gradient
            # Skip advection for now — focus on xi alignment dynamics

    return history


def run_experiment():
    """Main experiment: test depletion across many configurations."""

    print("=" * 70)
    print("VECTOR 3: VORTICITY DIRECTION EVOLUTION — DEPLETION TEST")
    print("=" * 70)
    print()

    n_filaments = 8
    n_steps = 100
    dt = 0.002
    n_trials = 30  # Per configuration type

    configs = [
        ('random', 0.0, 'Random, inviscid'),
        ('adversarial_alpha', 0.0, 'Adversarial alpha, inviscid'),
        ('random', 0.1, 'Random, viscous (nu=0.1)'),
        ('random', 1.0, 'Random, viscous (nu=1.0)'),
        ('random', 10.0, 'Random, viscous (nu=10.0)'),
        ('adversarial_alpha', 0.1, 'Adversarial alpha, viscous (nu=0.1)'),
        ('adversarial_alpha', 1.0, 'Adversarial alpha, viscous (nu=1.0)'),
        ('adversarial_alpha', 10.0, 'Adversarial alpha, viscous (nu=10.0)'),
    ]

    for init_type, nu, label in configs:
        print(f"\n--- {label} ({n_trials} trials, {n_filaments} filaments) ---")

        final_alpha = []
        final_beta = []
        final_gamma = []
        final_stretch = []
        initial_alpha = []
        initial_beta = []

        for trial in range(n_trials):
            np.random.seed(trial * 1000 + hash(init_type) % 10000)
            history = run_single_trial(n_filaments, init_type, nu, n_steps, dt)

            # Initial alignment (step 0)
            initial_alpha.append(np.mean(history['cos_alpha'][0]))
            initial_beta.append(np.mean(history['cos_beta'][0]))

            # Final alignment (last 10 steps average for stability)
            final_alpha.append(np.mean(history['cos_alpha'][-10:]))
            final_beta.append(np.mean(history['cos_beta'][-10:]))
            final_gamma.append(np.mean(history['cos_gamma'][-10:]))
            final_stretch.append(np.mean(history['stretching'][-10:]))

        ia = np.mean(initial_alpha)
        ib = np.mean(initial_beta)
        fa = np.mean(final_alpha)
        fb = np.mean(final_beta)
        fg = np.mean(final_gamma)
        fs = np.mean(final_stretch)

        print(f"  Initial:  |cos(alpha)| = {ia:.4f}, |cos(beta)| = {ib:.4f}")
        print(f"  Final:    |cos(alpha)| = {fa:.4f}, |cos(beta)| = {fb:.4f}, |cos(gamma)| = {fg:.4f}")
        print(f"  Stretch:  mean = {fs:.6f}")

        # Depletion test: does beta alignment increase?
        if fb > fa and fb > fg:
            print(f"  DEPLETION: YES — beta (intermediate) dominant ({fb:.4f} > {fa:.4f}, {fg:.4f})")
        elif fa > fb:
            print(f"  DEPLETION: NO — alpha (most stretching) dominant ({fa:.4f} > {fb:.4f})")
        else:
            print(f"  DEPLETION: NO — gamma (most compressing) dominant ({fg:.4f} > {fb:.4f})")

    print()
    print("=" * 70)
    print("TRACE-FREE CHECK (pressure constraint)")
    print("=" * 70)

    # Verify that Biot-Savart strain is always trace-free
    positions = np.random.randn(10, 3) * 0.5
    directions = np.random.randn(10, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    omegas = directions * 3.0

    traces = []
    for i in range(10):
        grad_u = biot_savart_velocity_gradient(positions[i], positions, omegas)
        S = extract_strain(grad_u)
        traces.append(np.trace(S))

    print(f"  tr(S) values: min={min(traces):.2e}, max={max(traces):.2e}")
    print(f"  Trace-free (pressure constraint) verified: {all(abs(t) < 1e-10 for t in traces)}")

    print()
    print("=" * 70)
    print("KEY QUESTION ANSWERED:")
    print("Does the Biot-Savart strain (with pressure/trace-free constraint)")
    print("dynamically force vorticity toward intermediate alignment?")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
