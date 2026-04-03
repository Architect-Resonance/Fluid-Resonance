"""
HELICITY H1/L2 BOUND CORRECTION
=================================
S92 — Meridian 2

PROBLEM:
  The Approach A bound chain uses: E- <= E - H/2
  This is WRONG — off by 66x. The issue: helicity involves H^1 norms
  (wavenumber-weighted) while energy involves L^2 norms (unweighted).

CORRECT RELATIONSHIPS:
  Helicity: H = sum_k |k| * (|u_p(k)|^2 - |u_m(k)|^2)
  Energy:   2E = sum_k (|u_p(k)|^2 + |u_m(k)|^2)
  Enstrophy: 2Z = sum_k |k|^2 * (|u_p(k)|^2 + |u_m(k)|^2)

  Define helical norms:
    ||v+||^2_{L2} = sum_k |u_p(k)|^2       (L^2 energy in + sector)
    ||v-||^2_{L2} = sum_k |u_m(k)|^2       (L^2 energy in - sector)
    ||v+||^2_{H1} = sum_k |k|^2 |u_p(k)|^2 (H^1 norm in + sector)
    ||v-||^2_{H1} = sum_k |k|^2 |u_m(k)|^2 (H^1 norm in - sector)

  Then:
    H = ||v+||^2_{H1/2} - ||v-||^2_{H1/2}  (H^{1/2} norms, not H^1!)
    where ||f||^2_{H1/2} = sum_k |k| |f_hat(k)|^2

  Key bounds:
    (a) H <= ||v+||^2_{H1/2}  (since ||v-||^2_{H1/2} >= 0)
    (b) ||v-||^2_{H1/2} = ||v+||^2_{H1/2} - H
    (c) Poincare: ||v-||_{L2} <= (1/k_min) * ||v-||_{H1/2}  ... NO, wrong dimension!

    Actually: H^{1/2} norm to L^2 norm:
    ||v-||^2_{L2} = sum_k |u_m(k)|^2
    ||v-||^2_{H1/2} = sum_k |k| |u_m(k)|^2

    By Cauchy-Schwarz on the spectral sum:
    ||v-||^2_{L2} = sum_k |u_m(k)|^2 * 1
                  = sum_k (|k|^{1/2} |u_m(k)|) * (|k|^{-1/2} |u_m(k)|)
    This doesn't give a clean bound without knowing the spectral distribution.

    However, for k >= k_min = 1 (periodic domain):
    ||v-||^2_{H1/2} = sum_k |k| |u_m(k)|^2 >= k_min * ||v-||^2_{L2}
    So: ||v-||^2_{L2} <= (1/k_min) * ||v-||^2_{H1/2}

    And: ||v-||^2_{H1/2} = ||v+||^2_{H1/2} - H
    And: ||v+||^2_{H1/2} + ||v-||^2_{H1/2} = sum_k |k| * (|u_p|^2 + |u_m|^2)

    Now: sum_k |k| * (|u_p|^2 + |u_m|^2) is related to enstrophy?
    Actually: 2Z = sum_k |k|^2 * (|u_p|^2 + |u_m|^2)
    And: 2E = sum_k (|u_p|^2 + |u_m|^2)

    By C-S: (sum |k| |f_k|^2)^2 <= (sum |f_k|^2)(sum |k|^2 |f_k|^2)
    So: (||v||^2_{H1/2})^2 <= 2E * 2Z
    Hence: ||v||_{H1/2} <= (2E * 2Z)^{1/4} ... hmm, dimensions are off.

    Let me just work with the clean bound:
    ||v-||^2_{L2} <= ||v-||^2_{H1/2} / k_min
                   = (||v||^2_{H1/2} - H) / (2 * k_min)

    where ||v||^2_{H1/2} = ||v+||^2_{H1/2} + ||v-||^2_{H1/2}

    And ||v||^2_{H1/2} = sum_k |k| (|u_p|^2 + |u_m|^2)

    This is a "half-derivative" norm. By interpolation:
    ||v||^2_{H1/2} <= ||v||_{L2} * ||v||_{H1} = sqrt(2E) * sqrt(2Z)

    So: ||v-||^2_{L2} <= (sqrt(4EZ) - H) / (2 * k_min)

    This is the CORRECT bound. Let's test it.

THIS SCRIPT:
  1. Derives the correct H1/2 -> L2 bound chain
  2. Tests it against actual NS solutions
  3. Compares with the wrong bound (E - H/2)
  4. Integrates into the Approach A bound chain
"""

import sys
import os
import numpy as np
from numpy.fft import fftn, ifftn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS


class HelicityBoundTester(SpectralNS):
    """Test the corrected helicity bound."""

    def abc_flow_ic(self, A=1.0, B=1.0, C=1.0):
        """ABC flow: exact Beltrami with curl(u) = u."""
        u = np.zeros((3,) + self.X.shape)
        u[0] = A * np.sin(self.Z) + C * np.cos(self.Y)
        u[1] = B * np.sin(self.X) + A * np.cos(self.Z)
        u[2] = C * np.sin(self.Y) + B * np.cos(self.X)
        return self.project_leray(np.array([fftn(u[i]) for i in range(3)]))

    def compute_all_norms(self, u_hat):
        """Compute all relevant norms for the helicity bound."""
        u_p, u_m = self.helical_decompose(u_hat)

        # L^2 norms (energy-like, no wavenumber weighting)
        E_plus_L2 = np.sum(np.abs(u_p)**2).real / self.N**3
        E_minus_L2 = np.sum(np.abs(u_m)**2).real / self.N**3

        # H^{1/2} norms (helicity-like, |k| weighting)
        E_plus_H12 = np.sum(self.kmag * np.abs(u_p)**2).real / self.N**3
        E_minus_H12 = np.sum(self.kmag * np.abs(u_m)**2).real / self.N**3

        # H^1 norms (enstrophy-like, |k|^2 weighting)
        E_plus_H1 = np.sum(self.k2 * np.abs(u_p)**2).real / self.N**3
        E_minus_H1 = np.sum(self.k2 * np.abs(u_m)**2).real / self.N**3

        # Physical quantities
        H = E_plus_H12 - E_minus_H12  # Helicity (H^{1/2} difference)
        E = 0.5 * (E_plus_L2 + E_minus_L2)  # Energy (L^2)
        Z = 0.5 * (E_plus_H1 + E_minus_H1)  # Enstrophy (H^1)

        # H^{1/2} total
        H12_total = E_plus_H12 + E_minus_H12

        # k_min for the periodic domain (smallest nonzero wavenumber)
        k_min = 1.0

        return {
            'E_plus_L2': E_plus_L2,
            'E_minus_L2': E_minus_L2,
            'E_plus_H12': E_plus_H12,
            'E_minus_H12': E_minus_H12,
            'E_plus_H1': E_plus_H1,
            'E_minus_H1': E_minus_H1,
            'H': H, 'E': E, 'Z': Z,
            'H12_total': H12_total,
            'k_min': k_min,
        }

    def test_bounds(self, u_hat, label=""):
        """Test all helicity -> E- bound formulas."""
        n = self.compute_all_norms(u_hat)

        print(f"\n{'='*65}")
        print(f"HELICITY BOUND TEST: {label}")
        print(f"{'='*65}")

        print(f"\n  Physical quantities:")
        print(f"    H (helicity) = {n['H']:.6e}")
        print(f"    E (energy)   = {n['E']:.6e}")
        print(f"    Z (enstrophy)= {n['Z']:.6e}")

        print(f"\n  Helical sector norms:")
        print(f"    ||v+||^2_L2   = {n['E_plus_L2']:.6e}")
        print(f"    ||v-||^2_L2   = {n['E_minus_L2']:.6e}  <-- THIS is what we want to bound")
        print(f"    ||v+||^2_H1/2 = {n['E_plus_H12']:.6e}")
        print(f"    ||v-||^2_H1/2 = {n['E_minus_H12']:.6e}")
        print(f"    ||v+||^2_H1   = {n['E_plus_H1']:.6e}")
        print(f"    ||v-||^2_H1   = {n['E_minus_H1']:.6e}")

        print(f"\n  Verification: H = ||v+||^2_H1/2 - ||v-||^2_H1/2 = {n['E_plus_H12'] - n['E_minus_H12']:.6e} (should = {n['H']:.6e})")

        # === BOUND 1 (WRONG): E- <= E - H/2 ===
        bound1 = max(0, n['E'] - abs(n['H']) / 2)
        ratio1 = n['E_minus_L2'] / max(bound1, 1e-30)
        valid1 = n['E_minus_L2'] <= bound1 * 1.01

        # === BOUND 2 (CORRECT): ||v-||^2_L2 <= ||v-||^2_H1/2 / k_min ===
        # where ||v-||^2_H1/2 = ||v||^2_H1/2 - H  (when H > 0, otherwise swap)
        if n['H'] >= 0:
            vminus_H12_bound = n['H12_total'] - n['H']  # = 2 * ||v-||^2_H1/2
            # Actually: ||v-||^2_H1/2 = (H12_total - H) / 2
            # No wait: H12_total = E_plus_H12 + E_minus_H12
            # H = E_plus_H12 - E_minus_H12
            # So E_minus_H12 = (H12_total - H) / 2
            vminus_H12_exact = (n['H12_total'] - n['H']) / 2
        else:
            vminus_H12_exact = (n['H12_total'] + abs(n['H'])) / 2

        bound2 = vminus_H12_exact / n['k_min']
        ratio2 = n['E_minus_L2'] / max(bound2, 1e-30)
        valid2 = n['E_minus_L2'] <= bound2 * 1.01

        # === BOUND 3 (TIGHTER): Use interpolation ===
        # ||v||^2_H1/2 <= sqrt(2E) * sqrt(2Z) (by Cauchy-Schwarz on spectrum)
        # Actually: (sum |k| |f|^2)^2 <= (sum |f|^2)(sum |k|^2 |f|^2)
        # So H12_total^2 <= (2E)(2Z)
        # H12_total <= 2*sqrt(E*Z)
        H12_interp = 2 * np.sqrt(n['E'] * n['Z'])
        if n['H'] >= 0:
            vminus_H12_interp = max(0, (H12_interp - n['H']) / 2)
        else:
            vminus_H12_interp = max(0, (H12_interp + abs(n['H'])) / 2)
        bound3 = vminus_H12_interp / n['k_min']
        ratio3 = n['E_minus_L2'] / max(bound3, 1e-30)
        valid3 = n['E_minus_L2'] <= bound3 * 1.01

        print(f"\n  {'Bound':<45} {'Value':>12} {'Actual':>12} {'Ratio':>8} {'Valid':>6}")
        print(f"  {'-'*83}")
        print(f"  {'WRONG: E - H/2':<45} {bound1:12.4e} {n['E_minus_L2']:12.4e} {ratio1:8.4f} {'YES' if valid1 else 'NO':>6}")
        print(f"  {'EXACT H1/2: ||v-||^2_H1/2 / k_min':<45} {bound2:12.4e} {n['E_minus_L2']:12.4e} {ratio2:8.4f} {'YES' if valid2 else 'NO':>6}")
        print(f"  {'INTERP: (sqrt(4EZ) - H) / (2*k_min)':<45} {bound3:12.4e} {n['E_minus_L2']:12.4e} {ratio3:8.4f} {'YES' if valid3 else 'NO':>6}")

        # === What bound 2 means for the full chain ===
        # ||P_sol(L)|| <= ||omega+||*||v-|| + ||omega-||*||v+||
        # Now ||v-||_{L2} <= sqrt(bound2) = sqrt(||v-||^2_H1/2 / k_min)
        # And ||omega+|| = ||v+||_{H1}
        # So the chain becomes:
        # ||P_sol(L)|| <= ||v+||_{H1} * sqrt(||v-||^2_H1/2 / k_min) + ||v-||_{H1} * ||v+||_{L2} / ...
        # This is getting circular. The real question is: are all quantities bounded
        # in terms of H, E, Z (which are the conserved/controlled quantities)?

        print(f"\n  Full chain implications:")
        # For the cross-helical bound:
        omega_hat = self.compute_vorticity_hat(u_hat)
        u_p_coeff, u_m_coeff = self.helical_decompose(u_hat)
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p_coeff), u_m_coeff)
        omega_hat_plus = self.compute_vorticity_hat(
            self.helical_reconstruct(u_p_coeff, np.zeros_like(u_m_coeff)))

        norm_omega_plus = np.sqrt(np.sum(np.abs(omega_hat_plus)**2).real / self.N**3)
        norm_v_minus = np.sqrt(n['E_minus_L2'])
        norm_v_minus_bound = np.sqrt(bound2)

        actual_cross_bound = norm_omega_plus * norm_v_minus
        predicted_cross_bound = norm_omega_plus * norm_v_minus_bound

        print(f"    ||omega+|| * ||v-||_actual    = {actual_cross_bound:.6e}")
        print(f"    ||omega+|| * ||v-||_bounded   = {predicted_cross_bound:.6e}")
        print(f"    Overestimate factor:           = {predicted_cross_bound / max(actual_cross_bound, 1e-30):.4f}")

        # The controlled quantities
        print(f"\n  Conservation laws (what controls the bound):")
        print(f"    H (conserved inviscidly)  = {n['H']:.6e}")
        print(f"    E (decreasing: dE/dt=-2vZ)= {n['E']:.6e}")
        print(f"    Z (bounded by E and H)    = {n['Z']:.6e}")
        print(f"    H^2/(2E) <= Z:            {n['H']**2/(2*n['E']):.6e} <= {n['Z']:.6e} -> {'YES' if n['H']**2/(2*n['E']) <= n['Z']*1.01 else 'NO'}")

        return {
            'bound_wrong': bound1,
            'bound_exact': bound2,
            'bound_interp': bound3,
            'actual': n['E_minus_L2'],
            'valid_exact': valid2,
            'valid_interp': valid3,
            'norms': n,
        }


def test_evolution_bounds(solver, u_hat_0, n_steps=200, dt=0.005, label=""):
    """Track bounds over time evolution."""
    print(f"\n{'='*80}")
    print(f"TIME EVOLUTION: {label}")
    print(f"N={solver.N}, Re={1/solver.nu:.0f}, dt={dt}, steps={n_steps}")
    print(f"{'='*80}")
    print(f"{'step':>5} {'||v-||^2_L2':>13} {'Exact bnd':>13} {'Interp bnd':>13} {'Ratio(ex)':>10} {'Ratio(int)':>10} {'H':>12}")

    u_hat = u_hat_0.copy()
    for step in range(n_steps + 1):
        if step % 20 == 0:
            n = solver.compute_all_norms(u_hat)

            if n['H'] >= 0:
                vminus_H12 = (n['H12_total'] - n['H']) / 2
            else:
                vminus_H12 = (n['H12_total'] + abs(n['H'])) / 2

            bound_exact = vminus_H12 / n['k_min']

            H12_interp = 2 * np.sqrt(n['E'] * n['Z'])
            if n['H'] >= 0:
                vminus_interp = max(0, (H12_interp - n['H']) / 2)
            else:
                vminus_interp = max(0, (H12_interp + abs(n['H'])) / 2)
            bound_interp = vminus_interp / n['k_min']

            r_ex = n['E_minus_L2'] / max(bound_exact, 1e-30)
            r_int = n['E_minus_L2'] / max(bound_interp, 1e-30)

            print(f"{step:5d} {n['E_minus_L2']:13.6e} {bound_exact:13.6e} {bound_interp:13.6e} {r_ex:10.4f} {r_int:10.4f} {n['H']:12.6e}")

        if step < n_steps:
            u_hat = solver.step_rk4(u_hat, dt, mode='full')


def main():
    print("=" * 65)
    print("HELICITY H1/L2 BOUND CORRECTION")
    print("S92 -- Meridian 2")
    print("=" * 65)

    solver = HelicityBoundTester(N=32, Re=400)

    # Test 1: Taylor-Green (achiral)
    print("\n\n" + "#" * 65)
    print("# TEST 1: Taylor-Green (achiral, H ~ 0)")
    print("#" * 65)
    u_tg = solver.taylor_green_ic()
    solver.test_bounds(u_tg, "TG t=0")

    # Test 2: Chiral IC (ABC + 10% TG)
    print("\n\n" + "#" * 65)
    print("# TEST 2: Chiral IC (ABC + 10% TG)")
    print("#" * 65)
    u_abc = solver.abc_flow_ic()
    u_perturb = solver.taylor_green_ic()
    E_abc = 0.5 * np.sum(np.abs(u_abc)**2).real / solver.N**3
    E_pert = 0.5 * np.sum(np.abs(u_perturb)**2).real / solver.N**3
    scale = 0.1 * np.sqrt(E_abc / max(E_pert, 1e-30))
    u_chiral = solver.project_leray(u_abc + scale * u_perturb)
    solver.test_bounds(u_chiral, "Chiral t=0")

    # Test 3: Pelz (achiral, complex)
    print("\n\n" + "#" * 65)
    print("# TEST 3: Pelz (achiral, complex)")
    print("#" * 65)
    u_pelz = solver.pelz_ic()
    solver.test_bounds(u_pelz, "Pelz t=0")

    # Test 4: Time evolution (chiral, most interesting for bounds)
    print("\n\n" + "#" * 65)
    print("# TEST 4: Time Evolution -- Chiral IC")
    print("#" * 65)
    test_evolution_bounds(solver, u_chiral, n_steps=200, dt=0.005, label="Chiral full NS")

    # Test 5: Time evolution (achiral)
    print("\n\n" + "#" * 65)
    print("# TEST 5: Time Evolution -- TG")
    print("#" * 65)
    test_evolution_bounds(solver, u_tg, n_steps=200, dt=0.005, label="TG full NS")


if __name__ == "__main__":
    main()
