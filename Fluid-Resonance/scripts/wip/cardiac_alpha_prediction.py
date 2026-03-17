"""
CARDIAC FLOW PREDICTION — Leray suppression alpha for organized vortex flows
========================================================================
S100-M1c: Testing the biology prediction.

PREDICTION: alpha(organized vortex) << 0.307 (isotropic average 1-ln(2))
Healthy cardiac flow is an organized vortex ring → should have low alpha.
Disease pushes flow toward isotropic → alpha → 0.307.

METHOD:
  1. Create vortex ring IC in SpectralNS (cardiac-like organized flow)
  2. Create other organized ICs: TG, ABC (Beltrami)
  3. Compare to random IC (isotropic)
  4. For each: compute alpha by measuring Leray suppression of cross-helical Lamb
  5. Track alpha(t) as flow evolves: organized → turbulent

The cross-helical Leray suppression alpha is:
  alpha = |P_sol[L_cross]|² / |L_cross|²
where L_cross = ω_+ × u_- + ω_- × u_+ (cross-helical Lamb vector)
and P_sol is the Leray (divergence-free) projector.

For a Beltrami flow (ω = λu): L = ω × u = λ(u × u) = 0 → alpha = 0.
For isotropic turbulence: alpha → 1-ln(2) ≈ 0.307.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_algebraic_structure import SpectralNS


def vortex_ring_ic(solver, R0=2.0, a=0.5, Gamma=1.0):
    """Create a vortex ring initial condition on periodic domain [0, 2π]³.

    A vortex ring centered at (π, π, π) in the (x,y) plane:
    - Ring radius R0 (major radius)
    - Core radius a (minor radius, Gaussian profile)
    - Circulation Gamma

    The vorticity is concentrated in a torus of major radius R0 and
    core radius a, directed along the azimuthal direction in the torus.

    Args:
        solver: SpectralNS instance
        R0: major radius of the ring
        a: core radius (Gaussian width)
        Gamma: circulation

    Returns:
        u_hat: velocity field in Fourier space, shape (3, N, N, N)
    """
    N = solver.N
    X, Y, Z = solver.X, solver.Y, solver.Z

    # Center at (π, π, π)
    cx, cy, cz = np.pi, np.pi, np.pi

    # Cylindrical coords relative to ring axis (z-axis through center)
    dx = X - cx
    dy = Y - cy
    dz = Z - cz

    # Distance from ring axis in (x,y) plane
    rho = np.sqrt(dx**2 + dy**2)
    rho_safe = np.maximum(rho, 1e-10)

    # Distance from ring centerline (torus core)
    # Ring is at (rho=R0, z=0) → distance = sqrt((rho-R0)² + dz²)
    r_core = np.sqrt((rho - R0)**2 + dz**2)

    # Gaussian vorticity profile in core cross-section
    # Vorticity is in the azimuthal direction of the torus
    omega_mag = (Gamma / (np.pi * a**2)) * np.exp(-r_core**2 / a**2)

    # Azimuthal direction in the torus (tangent to ring centerline)
    # At point (x,y,z), the closest point on the ring is at angle phi = atan2(dy, dx)
    # The tangent to the ring at that point is (-sin(phi), cos(phi), 0)
    phi = np.arctan2(dy, dx)
    omega = np.zeros((3, N, N, N))
    omega[0] = -np.sin(phi) * omega_mag
    omega[1] = np.cos(phi) * omega_mag
    omega[2] = 0.0

    # Solve for velocity via Biot-Savart: u_hat = (ik × omega_hat) / k²
    omega_hat = np.array([fftn(omega[i]) for i in range(3)])

    # u_hat_i = -i * epsilon_ijk * k_j * omega_hat_k / k²
    # Actually: curl(psi) = omega → -k² psi_hat = omega_hat → psi_hat = -omega_hat/k²
    # Then u = curl(psi) → u_hat = ik × psi_hat
    # Simpler: use the vector potential approach
    # u_hat = (ik × omega_hat) / k²  ... wait, that's not right either.
    # Correct: omega_hat = ik × u_hat, so u_hat = (ik × omega_hat) / k² is wrong.
    # Instead: u = curl^{-1}(omega) via the stream function.
    # In Fourier: omega_hat = ik × u_hat, and div(u) = 0 (ik · u_hat = 0)
    # So u_hat = -ik × omega_hat / k²  (Biot-Savart in Fourier)
    k2_safe = solver.k2_safe
    kx, ky, kz = solver.kx, solver.ky, solver.kz

    u_hat = np.zeros((3, N, N, N), dtype=complex)
    # u_hat = -i(k × omega_hat) / k²
    u_hat[0] = -1j * (ky * omega_hat[2] - kz * omega_hat[1]) / k2_safe
    u_hat[1] = -1j * (kz * omega_hat[0] - kx * omega_hat[2]) / k2_safe
    u_hat[2] = -1j * (kx * omega_hat[1] - ky * omega_hat[0]) / k2_safe

    # Zero mean
    u_hat[:, 0, 0, 0] = 0.0

    # Project to ensure divergence-free
    u_hat = solver.project_leray(u_hat)

    # Normalize energy
    u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    E = 0.5 * np.mean(np.sum(u**2, axis=0))
    u_hat *= np.sqrt(0.5 / max(E, 1e-15))

    return u_hat


def abc_flow_ic(solver, A=1.0, B=1.0, C=1.0):
    """Arnold-Beltrami-Childress flow: exact Beltrami (ω = u).

    This is the gold standard for alpha = 0 (Beltrami ⟹ Lamb = 0).
    """
    N = solver.N
    X, Y, Z = solver.X, solver.Y, solver.Z

    u = np.zeros((3, N, N, N))
    u[0] = A * np.sin(Z) + C * np.cos(Y)
    u[1] = B * np.sin(X) + A * np.cos(Z)
    u[2] = C * np.sin(Y) + B * np.cos(X)

    u_hat = np.array([fftn(u[i]) for i in range(3)])
    u_hat = solver.project_leray(u_hat)

    # Normalize
    u_phys = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    E = 0.5 * np.mean(np.sum(u_phys**2, axis=0))
    u_hat *= np.sqrt(0.5 / max(E, 1e-15))
    return u_hat


def measure_cross_helical_alpha(solver, u_hat):
    """Measure the cross-helical Leray suppression factor alpha.

    alpha = |P_sol[L_cross]|² / |L_cross|²

    where L_cross is the cross-helical Lamb vector and P_sol is Leray projection.

    Returns:
        alpha: the suppression factor (0 = Beltrami, 0.307 = isotropic)
        E_lamb_cross: total cross-helical Lamb power
        E_lamb_sol: solenoidal (surviving) cross-helical Lamb power
    """
    # Decompose into helical modes
    u_p, u_m = solver.helical_decompose(u_hat)

    # Reconstruct h+ and h- velocity fields
    u_hat_plus = u_p[np.newaxis] * solver.h_plus
    u_hat_minus = u_m[np.newaxis] * solver.h_minus

    # Compute vorticities: ω± = ik × u±
    omega_hat_plus = solver.compute_vorticity_hat(u_hat_plus)
    omega_hat_minus = solver.compute_vorticity_hat(u_hat_minus)

    # Cross-helical Lamb vector: L_cross = ω+ × u- + ω- × u+
    # Compute in physical space (dealiased)
    u_plus = np.array([np.real(ifftn(u_hat_plus[i])) for i in range(3)])
    u_minus = np.array([np.real(ifftn(u_hat_minus[i])) for i in range(3)])
    om_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
    om_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

    # L_cross = ω+ × u- + ω- × u+
    lamb_cross = np.zeros((3,) + u_plus.shape[1:])
    # ω+ × u-
    lamb_cross[0] += om_plus[1] * u_minus[2] - om_plus[2] * u_minus[1]
    lamb_cross[1] += om_plus[2] * u_minus[0] - om_plus[0] * u_minus[2]
    lamb_cross[2] += om_plus[0] * u_minus[1] - om_plus[1] * u_minus[0]
    # ω- × u+
    lamb_cross[0] += om_minus[1] * u_plus[2] - om_minus[2] * u_plus[1]
    lamb_cross[1] += om_minus[2] * u_plus[0] - om_minus[0] * u_plus[2]
    lamb_cross[2] += om_minus[0] * u_plus[1] - om_minus[1] * u_plus[0]

    # Transform to Fourier and dealias
    lamb_cross_hat = np.array([fftn(lamb_cross[i]) for i in range(3)])
    for i in range(3):
        lamb_cross_hat[i] *= solver.dealias_mask

    # Total cross-helical Lamb power
    E_lamb_cross = np.sum(np.abs(lamb_cross_hat)**2)

    if E_lamb_cross < 1e-30:
        return 0.0, 0.0, 0.0

    # Leray project → solenoidal part
    lamb_sol_hat = solver.project_leray(lamb_cross_hat)
    E_lamb_sol = np.sum(np.abs(lamb_sol_hat)**2)

    alpha = E_lamb_sol / E_lamb_cross
    return alpha, E_lamb_cross, E_lamb_sol


def run_cardiac_prediction():
    """Test: does organized vortex flow have alpha << 0.307?"""
    print("=" * 70)
    print("CARDIAC FLOW PREDICTION — Leray alpha for organized vs random flows")
    print("=" * 70)
    print()
    print("PREDICTION: alpha(organized vortex) << 0.307 = 1-ln(2)")
    print("            alpha(Beltrami) = 0")
    print("            alpha(isotropic) = 0.307")
    print()

    N = 48
    Re = 400
    dt = 0.005
    solver = SpectralNS(N=N, Re=Re)

    # ==========================================
    # 1. ABC flow (exact Beltrami — alpha should ≈ 0)
    # ==========================================
    print("--- ABC FLOW (Beltrami: ω = u) ---")
    u_hat_abc = abc_flow_ic(solver)
    alpha_abc, E_cross, E_sol = measure_cross_helical_alpha(solver, u_hat_abc)
    print(f"  alpha = {alpha_abc:.6f}  (expected: ~0)")
    print(f"  E_cross = {E_cross:.4e}, E_sol = {E_sol:.4e}")
    print()

    # ==========================================
    # 2. Vortex ring (cardiac-like — alpha should be low)
    # ==========================================
    print("--- VORTEX RING (cardiac-like organized flow) ---")
    u_hat_vr = vortex_ring_ic(solver, R0=2.0, a=0.5)
    alpha_vr, E_cross, E_sol = measure_cross_helical_alpha(solver, u_hat_vr)
    print(f"  alpha = {alpha_vr:.6f}  (expected: << 0.307)")
    print(f"  E_cross = {E_cross:.4e}, E_sol = {E_sol:.4e}")
    print()

    # ==========================================
    # 3. Taylor-Green (organized but not Beltrami)
    # ==========================================
    print("--- TAYLOR-GREEN (organized vortex, not Beltrami) ---")
    u_hat_tg = solver.taylor_green_ic()
    alpha_tg, E_cross, E_sol = measure_cross_helical_alpha(solver, u_hat_tg)
    print(f"  alpha = {alpha_tg:.6f}  (expected: ~0.10 from b2)")
    print(f"  E_cross = {E_cross:.4e}, E_sol = {E_sol:.4e}")
    print()

    # ==========================================
    # 4. Random IC (isotropic — alpha should → 0.307)
    # ==========================================
    print("--- RANDOM IC (isotropic, no organization) ---")
    u_hat_rand = solver.random_ic(seed=42)
    alpha_rand, E_cross, E_sol = measure_cross_helical_alpha(solver, u_hat_rand)
    print(f"  alpha = {alpha_rand:.6f}  (expected: ~0.307)")
    print(f"  E_cross = {E_cross:.4e}, E_sol = {E_sol:.4e}")
    print()

    # ==========================================
    # 5. Track alpha(t) for vortex ring as it evolves → turbulence
    # ==========================================
    print("=" * 70)
    print("TIME EVOLUTION: alpha(t) for vortex ring IC")
    print("=" * 70)
    print()
    print(f"{'t':>6} {'alpha':>10} {'E_total':>12} {'interpretation':>20}")
    print("-" * 52)

    u_hat = vortex_ring_ic(solver, R0=2.0, a=0.5)
    t = 0.0
    snapshots = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    snap_idx = 0

    while snap_idx < len(snapshots):
        if t >= snapshots[snap_idx] - 1e-6:
            alpha, _, _ = measure_cross_helical_alpha(solver, u_hat)
            u_phys = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
            E = 0.5 * np.mean(np.sum(u_phys**2, axis=0))

            if alpha < 0.10:
                interp = "near-Beltrami"
            elif alpha < 0.20:
                interp = "organized"
            elif alpha < 0.28:
                interp = "transitional"
            else:
                interp = "~isotropic"
            print(f"{t:>6.2f} {alpha:>10.6f} {E:>12.6f} {interp:>20}")

            snap_idx += 1

        if snap_idx < len(snapshots):
            u_hat = solver.step_rk4(u_hat, dt)
            t += dt

    # ==========================================
    # 6. Same for TG (comparison)
    # ==========================================
    print()
    print("=" * 70)
    print("TIME EVOLUTION: alpha(t) for Taylor-Green IC")
    print("=" * 70)
    print()
    print(f"{'t':>6} {'alpha':>10} {'E_total':>12}")
    print("-" * 32)

    u_hat = solver.taylor_green_ic()
    t = 0.0
    snap_idx = 0

    while snap_idx < len(snapshots):
        if t >= snapshots[snap_idx] - 1e-6:
            alpha, _, _ = measure_cross_helical_alpha(solver, u_hat)
            u_phys = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
            E = 0.5 * np.mean(np.sum(u_phys**2, axis=0))
            print(f"{t:>6.2f} {alpha:>10.6f} {E:>12.6f}")
            snap_idx += 1

        if snap_idx < len(snapshots):
            u_hat = solver.step_rk4(u_hat, dt)
            t += dt

    # ==========================================
    # Summary
    # ==========================================
    print()
    print("=" * 70)
    print("SUMMARY — BIOLOGY PREDICTION TEST")
    print("=" * 70)
    print()
    print(f"  ABC (Beltrami):     alpha = {alpha_abc:.4f}  (ω = λu → Lamb ≈ 0)")
    print(f"  Vortex ring (t=0):  alpha = {alpha_vr:.4f}  (cardiac-like)")
    print(f"  Taylor-Green (t=0): alpha = {alpha_tg:.4f}  (organized vortex)")
    print(f"  Random (t=0):       alpha = {alpha_rand:.4f}  (isotropic)")
    print(f"  Isotropic theory:   alpha = 0.3069  (1 - ln(2))")
    print()
    print("INTERPRETATION:")
    if alpha_vr < 0.20:
        print(f"  Vortex ring alpha = {alpha_vr:.4f} << 0.307 — PREDICTION CONFIRMED")
        print("  Organized vortex flows suppress the Lamb vector.")
        print("  Biology's convergence on helical/vortex flows = convergence")
        print("  toward low-alpha attractor = maximum Leray suppression.")
    else:
        print(f"  Vortex ring alpha = {alpha_vr:.4f} — not significantly below 0.307")
        print("  The vortex ring may not be sufficiently 'Beltrami-like'.")
        print("  Need higher R0/a ratio (thinner ring) or different model.")
    print()
    if alpha_abc < 0.05:
        print(f"  ABC (Beltrami) alpha = {alpha_abc:.4f} confirms alpha → 0 for ω ∥ u.")
    print()
    print("CLINICAL PREDICTION:")
    print("  Healthy heart (organized vortex ring): alpha << 0.307")
    print("  Dilated cardiomyopathy (disordered):   alpha → 0.307")
    print("  Measurable with 4D MRI + helical decomposition + our formula.")


if __name__ == "__main__":
    run_cardiac_prediction()
