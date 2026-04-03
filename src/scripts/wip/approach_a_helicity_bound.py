"""
APPROACH A: HELICITY-SECTOR BOUND ON SOLENOIDAL LAMB VECTOR
=============================================================
S92 — Meridian

THEOREM ATTEMPT:
  If H = int(u . omega) > 0 is conserved, then
  ||P_sol(omega x v)|| <= C(H, E) * ||v-|| * ||omega||
  where v- is the negative-helicity component and C stays finite.

PROOF STRATEGY (Waleffe 1992 framework):
  1. Decompose v = v+ + v-, omega = omega+ + omega-
     where curl(v±) = ±|k| v± (helical modes)
  2. Lamb vector L = omega x v decomposes into 4 sectors:
     L++ = omega+ x v+  (co-helical, positive)
     L-- = omega- x v-  (co-helical, negative)
     L+- = omega+ x v-  (cross-helical)
     L-+ = omega- x v+  (cross-helical)
  3. KEY CLAIM: P_sol(L++) = 0 and P_sol(L--) = 0
     (co-helical Lamb is pure gradient for single-wavenumber interactions)
  4. Therefore P_sol(L) = P_sol(L+- + L-+)
  5. ||P_sol(L+- + L-+)|| <= ||L+-|| + ||L-+||
                          <= ||omega+|| * ||v-|| + ||omega-|| * ||v+||
  6. Helicity: H = ||v+||^2_H1 - ||v-||^2_H1
     If H > 0, then ||v-||_H1 < ||v+||_H1
  7. Energy + helicity trap ||v-|| in terms of H and E

THIS SCRIPT:
  - Tests claim (3) numerically: is co-helical Lamb really solenoidal-free?
  - Measures the 4-sector contribution to P_sol(L)
  - Tests the bound chain (5)-(7) against actual NS evolution
  - Reports HONESTLY what works and what breaks

HONEST TEST: We report what the numbers say, not what we hope.
"""

import sys
import os
import numpy as np
from numpy.fft import fftn, ifftn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS


class HelicitySectorAnalyzer(SpectralNS):
    """Extended solver for helicity-sector Lamb vector analysis."""

    def abc_flow_ic(self, A=1.0, B=1.0, C=1.0):
        """ABC flow: exact Beltrami with curl(u) = u."""
        u = np.zeros((3,) + self.X.shape)
        u[0] = A * np.sin(self.Z) + C * np.cos(self.Y)
        u[1] = B * np.sin(self.X) + A * np.cos(self.Z)
        u[2] = C * np.sin(self.Y) + B * np.cos(self.X)
        return self.project_leray(np.array([fftn(u[i]) for i in range(3)]))

    def decompose_lamb_4sectors(self, u_hat):
        """Decompose Lamb vector into 4 helicity sectors.

        Returns dict with keys '++', '--', '+-', '-+',
        each containing the Fourier-space Lamb vector for that sector.
        """
        # Decompose u into helical components
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)

        # Compute vorticity for each
        omega_hat_plus = self.compute_vorticity_hat(u_hat_plus)
        omega_hat_minus = self.compute_vorticity_hat(u_hat_minus)

        # Physical space
        u_plus = np.array([np.real(ifftn(u_hat_plus[i])) for i in range(3)])
        u_minus = np.array([np.real(ifftn(u_hat_minus[i])) for i in range(3)])
        omega_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
        omega_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

        def cross(a, b):
            return np.array([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])

        def to_hat(field):
            fh = np.array([fftn(field[i]) for i in range(3)])
            for i in range(3):
                fh[i] *= self.dealias_mask
            return fh

        # 4 sectors: omega_X x v_Y
        # Note: Lamb = omega x v (not v x omega), so L = omega x u
        sectors = {
            '++': to_hat(cross(omega_plus, u_plus)),
            '--': to_hat(cross(omega_minus, u_minus)),
            '+-': to_hat(cross(omega_plus, u_minus)),
            '-+': to_hat(cross(omega_minus, u_plus)),
        }
        return sectors

    def l2_norm_hat(self, f_hat):
        """L2 norm from Fourier coefficients: ||f||^2 = sum |f_hat|^2 / N^3."""
        return np.sqrt(np.sum(np.abs(f_hat)**2).real / self.N**3)

    def compute_helicity(self, u_hat):
        """Compute helicity H = int u . omega dx."""
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        H = np.sum(u * omega) * (2 * np.pi / self.N)**3
        return H

    def compute_energy(self, u_hat):
        """Compute kinetic energy E = 0.5 * int |u|^2 dx."""
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        E = 0.5 * np.sum(u**2) * (2 * np.pi / self.N)**3
        return E

    def compute_enstrophy(self, u_hat):
        """Compute enstrophy Z = 0.5 * int |omega|^2 dx."""
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        Z = 0.5 * np.sum(omega**2) * (2 * np.pi / self.N)**3
        return Z

    def helical_energies(self, u_hat):
        """Compute E+ and E- (energy in each helical sector)."""
        u_p, u_m = self.helical_decompose(u_hat)
        E_plus = 0.5 * np.sum(np.abs(u_p)**2).real / self.N**3
        E_minus = 0.5 * np.sum(np.abs(u_m)**2).real / self.N**3
        return E_plus, E_minus


def test_cohelical_solenoidal_claim(solver, u_hat, label=""):
    """TEST CLAIM (3): Is P_sol(L++) ~ 0?

    For a single Fourier mode k, the co-helical Lamb vector omega+ x v+
    should be pure gradient (killed by Leray projection).

    For a general flow with multiple modes, this is only approximate due to
    triadic interactions. This test measures how good the approximation is.
    """
    sectors = solver.decompose_lamb_4sectors(u_hat)

    # Apply Leray projection to each sector
    norms = {}
    psol_norms = {}
    for key in ['++', '--', '+-', '-+']:
        norms[key] = solver.l2_norm_hat(sectors[key])
        psol = solver.project_leray(sectors[key])
        psol_norms[key] = solver.l2_norm_hat(psol)

    # Total Lamb and its solenoidal part
    lamb_total = sum(sectors.values())
    psol_total = solver.project_leray(lamb_total)
    norm_total = solver.l2_norm_hat(lamb_total)
    psol_norm_total = solver.l2_norm_hat(psol_total)

    # Cross-helical contribution to P_sol
    lamb_cross = sectors['+-'] + sectors['-+']
    psol_cross = solver.project_leray(lamb_cross)
    psol_norm_cross = solver.l2_norm_hat(psol_cross)

    print(f"\n{'='*60}")
    print(f"HELICITY-SECTOR LAMB ANALYSIS {label}")
    print(f"{'='*60}")
    print(f"{'Sector':<10} {'||L||':>12} {'||P_sol(L)||':>14} {'Sol. frac':>10}")
    print(f"{'-'*46}")
    for key in ['++', '--', '+-', '-+']:
        frac = psol_norms[key] / max(norms[key], 1e-30)
        tag = " <-- co-helical" if key in ['++', '--'] else " <-- CROSS"
        print(f"  L_{key:<6} {norms[key]:12.6e} {psol_norms[key]:14.6e} {frac:10.4f}{tag}")

    print(f"{'-'*46}")
    print(f"  Total    {norm_total:12.6e} {psol_norm_total:14.6e} {psol_norm_total/max(norm_total, 1e-30):10.4f}")
    print(f"  Cross    {'':12s} {psol_norm_cross:14.6e} {psol_norm_cross/max(psol_norm_total, 1e-30):10.4f} of total P_sol")

    # Verdict on Claim (3)
    co_helical_sol = psol_norms['++'] + psol_norms['--']
    cross_helical_sol = psol_norms['+-'] + psol_norms['-+']
    cross_fraction = cross_helical_sol / max(co_helical_sol + cross_helical_sol, 1e-30)

    print(f"\n  CLAIM (3) test: ||P_sol(L++)||+||P_sol(L--)|| = {co_helical_sol:.6e}")
    print(f"                  ||P_sol(L+-)||+||P_sol(L-+)|| = {cross_helical_sol:.6e}")
    print(f"                  Cross-helical fraction of P_sol = {cross_fraction:.4f}")

    if cross_fraction > 0.8:
        print(f"  VERDICT: CLAIM (3) SUPPORTED — cross-helical dominates P_sol ({cross_fraction:.1%})")
    elif cross_fraction > 0.5:
        print(f"  VERDICT: CLAIM (3) PARTIAL — cross-helical majority but co-helical non-negligible")
    else:
        print(f"  VERDICT: CLAIM (3) REFUTED — co-helical contributes {1-cross_fraction:.1%} to P_sol")

    return {
        'norms': norms,
        'psol_norms': psol_norms,
        'cross_fraction': cross_fraction,
        'solenoidal_fraction': psol_norm_total / max(norm_total, 1e-30),
    }


def test_bound_chain(solver, u_hat, label=""):
    """TEST BOUND CHAIN (5)-(7).

    Chain:
      ||P_sol(L)|| <= ||P_sol(L_cross)|| (if claim 3 holds)
                   <= ||L+-|| + ||L-+||   (triangle inequality)
                   <= ||omega+||*||v-|| + ||omega-||*||v+||  (Cauchy-Schwarz)

    And helicity constrains:
      H = E+ - E- (in terms of helical energies, simplified)
      If H > 0: E- < E+ so ||v-|| is bounded
    """
    sectors = solver.decompose_lamb_4sectors(u_hat)
    psol_total = solver.project_leray(sum(sectors.values()))
    psol_total_norm = solver.l2_norm_hat(psol_total)

    # Helical decomposition for norms
    u_p, u_m = solver.helical_decompose(u_hat)
    u_hat_plus = solver.helical_reconstruct(u_p, np.zeros_like(u_m))
    u_hat_minus = solver.helical_reconstruct(np.zeros_like(u_p), u_m)

    omega_hat_plus = solver.compute_vorticity_hat(u_hat_plus)
    omega_hat_minus = solver.compute_vorticity_hat(u_hat_minus)

    norm_v_plus = solver.l2_norm_hat(u_hat_plus)
    norm_v_minus = solver.l2_norm_hat(u_hat_minus)
    norm_omega_plus = solver.l2_norm_hat(omega_hat_plus)
    norm_omega_minus = solver.l2_norm_hat(omega_hat_minus)

    # Cross Lamb norms
    norm_L_pm = solver.l2_norm_hat(sectors['+-'])
    norm_L_mp = solver.l2_norm_hat(sectors['-+'])

    # Bound chain
    bound_triangle = norm_L_pm + norm_L_mp
    bound_cauchy = norm_omega_plus * norm_v_minus + norm_omega_minus * norm_v_plus

    # Helicity and energy
    H = solver.compute_helicity(u_hat)
    E = solver.compute_energy(u_hat)
    Z = solver.compute_enstrophy(u_hat)
    E_plus, E_minus = solver.helical_energies(u_hat)

    print(f"\n{'='*60}")
    print(f"BOUND CHAIN ANALYSIS {label}")
    print(f"{'='*60}")
    print(f"  ||P_sol(L)||           = {psol_total_norm:.6e}  (actual)")
    print(f"  ||L+-|| + ||L-+||      = {bound_triangle:.6e}  (triangle bound)")
    print(f"  ||w+||*||v-||+||w-||*||v+|| = {bound_cauchy:.6e}  (Cauchy-Schwarz)")
    print(f"")
    print(f"  Chain valid: {psol_total_norm:.6e} <= {bound_triangle:.6e} <= {bound_cauchy:.6e}")
    print(f"  Triangle tight: actual/bound = {psol_total_norm/max(bound_triangle, 1e-30):.4f}")
    print(f"  C-S tight:      actual/bound = {psol_total_norm/max(bound_cauchy, 1e-30):.4f}")
    print(f"")
    print(f"  Helicity H = {H:.6e}")
    print(f"  Energy   E = {E:.6e}")
    print(f"  Enstrophy Z = {Z:.6e}")
    print(f"  E+ = {E_plus:.6e},  E- = {E_minus:.6e},  ratio E-/E+ = {E_minus/max(E_plus, 1e-30):.4f}")
    print(f"  ||v+|| = {norm_v_plus:.6e},  ||v-|| = {norm_v_minus:.6e}")
    print(f"  ||omega+|| = {norm_omega_plus:.6e},  ||omega-|| = {norm_omega_minus:.6e}")

    # The key question: does bounding ||v-|| via H actually give a useful bound?
    # H ~ 2*(E+ - E-) in the simplified Waleffe framework
    # E- ~ (E - H/2) / 2 when H is not too large
    if H > 0:
        E_minus_bound = max(0, (E - abs(H) / 2))
        print(f"\n  Helicity bound: E- <= E - H/2 = {E_minus_bound:.6e}")
        print(f"  Actual E-:                      {E_minus:.6e}")
        print(f"  Bound ratio: {E_minus/max(E_minus_bound, 1e-30):.4f}")

    chain_valid = (psol_total_norm <= bound_triangle * 1.01) and (bound_triangle <= bound_cauchy * 1.01)
    print(f"\n  VERDICT: Chain {'VALID' if chain_valid else 'BROKEN'}")

    return {
        'psol_norm': psol_total_norm,
        'triangle_bound': bound_triangle,
        'cauchy_bound': bound_cauchy,
        'H': H, 'E': E, 'Z': Z,
        'E_plus': E_plus, 'E_minus': E_minus,
        'chain_valid': chain_valid,
    }


def test_evolution(solver, u_hat_0, n_steps=200, dt=0.005, label=""):
    """Evolve under full NS and track the bound chain over time.

    KEY QUESTION: Does the bound stay finite? Does the chain tighten or loosen?
    """
    print(f"\n{'='*60}")
    print(f"TIME EVOLUTION {label}")
    print(f"N={solver.N}, Re={1/solver.nu:.0f}, dt={dt}, steps={n_steps}")
    print(f"{'='*60}")
    print(f"{'step':>5} {'||P_sol(L)||':>13} {'triangle':>13} {'C-S':>13} {'E-/E+':>8} {'H':>12} {'E':>12}")

    u_hat = u_hat_0.copy()
    results = []

    for step in range(n_steps + 1):
        if step % 20 == 0:
            sectors = solver.decompose_lamb_4sectors(u_hat)
            psol = solver.project_leray(sum(sectors.values()))
            psol_norm = solver.l2_norm_hat(psol)

            norm_L_pm = solver.l2_norm_hat(sectors['+-'])
            norm_L_mp = solver.l2_norm_hat(sectors['-+'])
            tri_bound = norm_L_pm + norm_L_mp

            u_p, u_m = solver.helical_decompose(u_hat)
            u_plus = solver.helical_reconstruct(u_p, np.zeros_like(u_m))
            u_minus = solver.helical_reconstruct(np.zeros_like(u_p), u_m)
            o_plus = solver.compute_vorticity_hat(u_plus)
            o_minus = solver.compute_vorticity_hat(u_minus)

            cs_bound = (solver.l2_norm_hat(o_plus) * solver.l2_norm_hat(u_minus)
                        + solver.l2_norm_hat(o_minus) * solver.l2_norm_hat(u_plus))

            H = solver.compute_helicity(u_hat)
            E = solver.compute_energy(u_hat)
            Ep, Em = solver.helical_energies(u_hat)
            ratio = Em / max(Ep, 1e-30)

            print(f"{step:5d} {psol_norm:13.6e} {tri_bound:13.6e} {cs_bound:13.6e} {ratio:8.4f} {H:12.6e} {E:12.6e}")
            results.append({
                'step': step, 'psol': psol_norm, 'triangle': tri_bound,
                'cauchy': cs_bound, 'ratio': ratio, 'H': H, 'E': E,
            })

        if step < n_steps:
            u_hat = solver.step_rk4(u_hat, dt, mode='full')

    # Check: did the bound chain hold throughout?
    all_valid = all(r['psol'] <= r['triangle'] * 1.05 for r in results)
    print(f"\nChain valid throughout: {all_valid}")

    # Check: did ||P_sol(L)|| stay bounded?
    max_psol = max(r['psol'] for r in results)
    final_psol = results[-1]['psol']
    print(f"Max ||P_sol(L)||: {max_psol:.6e}")
    print(f"Final ||P_sol(L)||: {final_psol:.6e}")
    print(f"Grew by factor: {max_psol / max(results[0]['psol'], 1e-30):.2f}x")

    return results


def main():
    print("=" * 60)
    print("APPROACH A: HELICITY-SECTOR BOUND ON P_sol(omega x v)")
    print("S92 — Meridian")
    print("=" * 60)

    solver = HelicitySectorAnalyzer(N=32, Re=400)

    # ---- Test 1: ABC flow (exact Beltrami, L=0) ----
    print("\n\n" + "#" * 60)
    print("# TEST 1: ABC Flow (exact Beltrami — Lamb=0, sanity check)")
    print("#" * 60)
    u_abc = solver.abc_flow_ic()
    test_cohelical_solenoidal_claim(solver, u_abc, "ABC")

    # ---- Test 2: Taylor-Green (achiral, H~0) ----
    print("\n\n" + "#" * 60)
    print("# TEST 2: Taylor-Green (achiral — H ~ 0)")
    print("#" * 60)
    u_tg = solver.taylor_green_ic()
    result_tg = test_cohelical_solenoidal_claim(solver, u_tg, "TG t=0")
    test_bound_chain(solver, u_tg, "TG t=0")

    # ---- Test 3: Chiral IC (strong helicity) ----
    print("\n\n" + "#" * 60)
    print("# TEST 3: Chiral IC (dominant positive helicity)")
    print("#" * 60)
    # ABC + small TG perturbation → mostly positive helicity
    u_chiral = solver.abc_flow_ic(A=1.0, B=1.0, C=1.0)
    # Add small achiral perturbation to break exact Beltrami
    u_perturb = solver.taylor_green_ic()
    E_abc = solver.compute_energy(u_chiral)
    E_perturb = solver.compute_energy(u_perturb)
    # Scale perturbation to 10% of ABC energy
    scale = 0.1 * np.sqrt(E_abc / max(E_perturb, 1e-30))
    u_chiral = u_chiral + scale * u_perturb
    u_chiral = solver.project_leray(u_chiral)

    H_chiral = solver.compute_helicity(u_chiral)
    E_chiral = solver.compute_energy(u_chiral)
    print(f"Chiral IC: H = {H_chiral:.4f}, E = {E_chiral:.4f}, H/E = {H_chiral/E_chiral:.4f}")

    result_chiral = test_cohelical_solenoidal_claim(solver, u_chiral, "Chiral t=0")
    test_bound_chain(solver, u_chiral, "Chiral t=0")

    # ---- Test 4: Pelz IC (achiral, more complex) ----
    print("\n\n" + "#" * 60)
    print("# TEST 4: Pelz IC (complex achiral)")
    print("#" * 60)
    u_pelz = solver.pelz_ic()
    result_pelz = test_cohelical_solenoidal_claim(solver, u_pelz, "Pelz t=0")
    test_bound_chain(solver, u_pelz, "Pelz t=0")

    # ---- Test 5: Time evolution (TG, achiral worst case) ----
    print("\n\n" + "#" * 60)
    print("# TEST 5: Time Evolution — Taylor-Green (worst case: H~0)")
    print("#" * 60)
    test_evolution(solver, u_tg, n_steps=200, dt=0.005, label="TG full NS")

    # ---- Test 6: Time evolution (Chiral IC, should be better) ----
    print("\n\n" + "#" * 60)
    print("# TEST 6: Time Evolution — Chiral IC (H > 0)")
    print("#" * 60)
    test_evolution(solver, u_chiral, n_steps=200, dt=0.005, label="Chiral full NS")

    # ---- Summary ----
    print("\n\n" + "=" * 60)
    print("APPROACH A SUMMARY")
    print("=" * 60)
    print(f"Claim (3) — co-helical P_sol ~ 0:")
    print(f"  TG cross-fraction:     {result_tg['cross_fraction']:.4f}")
    print(f"  Chiral cross-fraction: {result_chiral['cross_fraction']:.4f}")
    print(f"  Pelz cross-fraction:   {result_pelz['cross_fraction']:.4f}")
    print(f"\nIf all > 0.8: Approach A is viable.")
    print(f"If any < 0.5: Approach A is dead — try Approach B.")


if __name__ == "__main__":
    main()
