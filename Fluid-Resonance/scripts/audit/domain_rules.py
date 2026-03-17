"""NS-specific domain rules and tolerances for claim verification.

These encode physical and mathematical constants, standard scaling laws,
and tolerance thresholds used across multiple verifiers.
"""

import math


# ---------------------------------------------------------------------------
# Physical / mathematical constants
# ---------------------------------------------------------------------------

# Kolmogorov scaling
def kolmogorov_velocity(epsilon: float, K: float) -> float:
    """u_K = (ε/K)^{1/3} — velocity at wavenumber K."""
    return (epsilon / K) ** (1 / 3)


def kolmogorov_turnover(tau_0: float, lam: float, n: int) -> float:
    """τ_n = τ₀ · λ^{-2n/3} — eddy turnover time at shell n."""
    return tau_0 * lam ** (-2 * n / 3)


def kolmogorov_dissipation(epsilon: float, nu: float) -> float:
    """k_d = (ε/ν³)^{1/4} — Kolmogorov dissipation wavenumber."""
    return (epsilon / nu ** 3) ** 0.25


# Bessel function asymptotics
def bessel_i0_asymptotic(kappa: float) -> float:
    """I₀(κ) ~ e^κ / √(2πκ) for large κ."""
    return math.exp(kappa) / math.sqrt(2 * math.pi * kappa)


def von_mises_self_info_exact(kappa: float) -> float:
    """Exact: -ln p(π/2) = -κ + ln I₀(κ) + ln(2π)."""
    from scipy.special import iv
    return -kappa + math.log(iv(0, kappa)) + math.log(2 * math.pi)


def von_mises_self_info_asymptotic(kappa: float) -> float:
    """Asymptotic: (1/2) ln(2π/κ)."""
    return 0.5 * math.log(2 * math.pi / kappa)


# Graph Laplacian
def complete_graph_laplacian_eigenvalues(n: int, weight: float) -> list[float]:
    """Eigenvalues of L = w(nI - J) for K_n with edge weight w.

    Returns sorted list: [0, w*n, w*n, ..., w*n] with multiplicity (1, n-1).
    """
    return [0.0] + [weight * n] * (n - 1)


# Viscosity crossover
def viscosity_crossover_ratio() -> float:
    """K_crit / k_d = √2/4 ≈ 0.3536 (from (1/4)Ku = νK²)."""
    return math.sqrt(2) / 4


# ---------------------------------------------------------------------------
# Tolerance thresholds
# ---------------------------------------------------------------------------

class Tolerance:
    """Standard tolerances for different verification types."""

    # Exact identities (eigenvalues, algebraic equalities)
    EXACT = 1e-12

    # Numerical approximations (rounded values in paper)
    APPROX_REL = 0.01  # 1% relative error

    # Asymptotic convergence (checked at large parameter values)
    ASYMPTOTIC_REL = 0.05  # 5% at κ=10, should be tighter at κ=50

    # Crossover locations (where sign changes)
    CROSSOVER_ABS = 0.3  # Absolute error in κ for crossover point
