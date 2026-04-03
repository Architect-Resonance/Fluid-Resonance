"""
SPECTRAL WINDING NUMBER (C4) -- M2 TASK ASSIGNMENT
====================================================
S100-M1b: New measurable prediction (P9) for DNS verification.

ASSIGNED TO: Meridian 2
PRIORITY: Medium (after current tasks complete)
ESTIMATED DIFFICULTY: Moderate (requires phase extraction + topological counting)

BACKGROUND:
  The helical mode amplitude a+(k) is a complex scalar field on each
  wavenumber shell |k| = k. Its phase arg(a+(k)) defines a map:
    phi: S^2 --> S^1   (from k-hat to phase angle)

  By the Chern number 2 of the Berry connection, this map must have
  at least 2 phase vortices (zeros of a+(k)) on every k-shell where
  a+(k) is nonzero almost everywhere.

  QUESTION: How does the number of phase vortices N_vortex(k) scale
  with k? If N_vortex ~ k^alpha with alpha > 0, each vortex creates
  a branch cut that disrupts phase alignment, providing ADDITIONAL
  decoherence beyond the CLT.

MEASUREMENT PROTOCOL:
  1. Set up HIT DNS (any Re, N >= 64 preferred)
  2. Compute helical decomposition: u_hat(k) = a+(k)*h+(k) + a-(k)*h-(k)
  3. For each shell k = 1, 2, ..., k_max:
     a. Extract phi(k_hat) = arg(a+(k_hat)) on a grid of points on S^2
        (Use HEALPix or icosahedral grid with N_side >= 16 for accuracy)
     b. Count phase vortices: points where phi winds by +/-2*pi
        Method: compute winding number of phi around each pixel
        A vortex exists where sum of phase differences around pixel = +/-2*pi
     c. Record N_vortex(k) = total number of vortices on shell
     d. Also record the winding number distribution (how many +1 vs -1)
  4. Repeat at multiple times T (initial, developed, late turbulence)
  5. Repeat for a-(k) (opposite helicity)

EXPECTED RESULTS:
  - Minimum N_vortex(k) >= 2 (topological constraint from Chern number)
  - If turbulence is nearly isotropic: N_vortex should grow with k
    (more modes, more phase freedom, more vortices)
  - PREDICTION P9a: N_vortex(k) ~ k^alpha with alpha > 0
  - PREDICTION P9b: Net winding = sum of all vortex charges = Chern number = 2
    (This is TOPOLOGICALLY EXACT if a+ is continuous on the shell)
  - PREDICTION P9c: At developed turbulence, N_vortex correlates with
    Kuramoto order parameter R_K (more vortices = lower R_K)

DIAGNOSTIC CHECKS:
  - At k = 1 (largest scale): N_vortex should be small (2-4)
  - At k = k_max/3: N_vortex should be much larger
  - For Taylor-Green IC at t=0: a+(k) has specific symmetries that
    constrain vortex locations (check consistency)
  - For random IC: vortices should be approximately uniformly distributed

IMPLEMENTATION SKETCH (pseudo-Python):
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Pseudo-code for the measurement (M2 should adapt to SpectralNS framework)
def count_phase_vortices_on_shell(a_plus_shell, theta_grid, phi_grid):
    """Count phase vortices of a+(k) on a wavenumber shell.

    Args:
        a_plus_shell: complex array of a+(k_hat) sampled on (theta, phi) grid
        theta_grid: 1D array of polar angles
        phi_grid: 1D array of azimuthal angles

    Returns:
        n_vortices: total number of phase vortices
        charges: list of (theta, phi, charge) for each vortex
    """
    phase = np.angle(a_plus_shell)  # shape (n_theta, n_phi)
    n_theta, n_phi = phase.shape

    vortices = []

    for i in range(n_theta - 1):
        for j in range(n_phi):
            # Compute winding number around this pixel
            # Four corners: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            j_next = (j + 1) % n_phi

            corners = [
                phase[i, j],
                phase[i+1, j],
                phase[i+1, j_next],
                phase[i, j_next],
            ]

            # Sum of phase differences (wrapped to [-pi, pi])
            winding = 0.0
            for c in range(4):
                dphi = corners[(c+1) % 4] - corners[c]
                # Wrap to [-pi, pi]
                dphi = (dphi + np.pi) % (2*np.pi) - np.pi
                winding += dphi

            # Winding number = winding / (2*pi), should be integer
            charge = round(winding / (2*np.pi))

            if charge != 0:
                theta_v = 0.5 * (theta_grid[i] + theta_grid[i+1])
                phi_v = phi_grid[j]
                vortices.append((theta_v, phi_v, charge))

    n_vortices = len(vortices)
    charges = vortices
    return n_vortices, charges


def demo_random_phase_field():
    """Demo: count vortices in a random phase field on S^2."""
    n_theta = 64
    n_phi = 128
    theta = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

    # Random complex field with some smoothness
    # Use spherical harmonics up to l_max to create smooth random field
    from scipy.special import sph_harm

    l_max = 10  # smoothness scale
    a_plus = np.zeros((n_theta, n_phi), dtype=complex)

    rng = np.random.default_rng(42)
    for l in range(1, l_max + 1):
        for m in range(-l, l + 1):
            # Random coefficient
            c = rng.standard_normal() + 1j * rng.standard_normal()
            for i, th in enumerate(theta):
                a_plus[i, :] += c * sph_harm(m, l, phi, th)

    n_vortex, charges = count_phase_vortices_on_shell(a_plus, theta, phi)

    total_charge = sum(c[2] for c in charges)
    n_positive = sum(1 for c in charges if c[2] > 0)
    n_negative = sum(1 for c in charges if c[2] < 0)

    print(f"Random field (l_max={l_max}):")
    print(f"  N_vortex = {n_vortex}")
    print(f"  Positive: {n_positive}, Negative: {n_negative}")
    print(f"  Net charge: {total_charge}")
    print(f"  Expected net charge: 0 (random field, no topological constraint)")
    print()

    # Now with Berry monopole constraint (net charge = 2)
    # Add a background field with Chern number 2
    print("With Berry monopole (Chern=2) background:")
    print("  M2 should compute this from actual DNS helical decomposition")
    print("  The h+(k_hat) basis already carries the monopole")
    print("  So a+(k) in this basis will have net winding = 2")


if __name__ == "__main__":
    print("=" * 70)
    print("SPECTRAL WINDING NUMBER (C4) -- DEMO")
    print("=" * 70)
    print()

    try:
        demo_random_phase_field()
    except ImportError:
        print("scipy not available -- skipping demo")
        print("M2 should implement using SpectralNS framework")

    print()
    print("=" * 70)
    print("M2 TASK SUMMARY")
    print("=" * 70)
    print()
    print("1. In SpectralNS, compute helical decomposition at each timestep")
    print("2. For each k-shell, extract phase of a+(k_hat) on S^2 grid")
    print("3. Count phase vortices using winding number algorithm")
    print("4. Record N_vortex(k) for k = 1 to k_max/3")
    print("5. Check: net charge = 2 (Chern number constraint)")
    print("6. Fit: N_vortex(k) ~ k^alpha, report alpha")
    print("7. Correlate with R_K(k) from Kuramoto analysis")
    print()
    print("PREDICTIONS TO TEST:")
    print("  P9a: N_vortex(k) ~ k^alpha, alpha > 0")
    print("  P9b: Net winding charge = 2 on every shell")
    print("  P9c: N_vortex inversely correlates with R_K")
