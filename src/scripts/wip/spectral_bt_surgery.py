"""
SPECTRAL NS WITH DYNAMIC BIFERALE-TITI SURGERY
================================================
Extends Antigravity's spectral_ns_solver.py with CORRECT BT surgery:
the h- projection is applied to the NONLINEAR TERM at each RK4 substep,
not just to the initial condition.

Antigravity's S47 applied surgery only to the IC, which trivially halves
energy and enstrophy (kinematic artifact = Failure 18). The CORRECT BT
surgery removes cross-helicity triadic TRANSFERS at each time step.

Key difference:
  Wrong: u_hat = project_h+(u_hat) at t=0, then evolve normally
  Right: at each dt, nonlinear_hat = project_h+(u x omega), then apply

This script runs 3 cases:
  A) Full NS (baseline)
  B) IC-only surgery (Antigravity's version — should show kinematic halving)
  C) Dynamic surgery (correct BT — surgery on nonlinear term every step)

If B and C give the same result: Antigravity was right, it's all kinematic.
If C differs from B: dynamic surgery has a genuine nonlinear effect.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import time as clock


class SpectralNS:
    """Minimal pseudo-spectral NS solver with helical decomposition."""

    def __init__(self, N=64, Re=400):
        self.N = N
        self.nu = 1.0 / Re
        L = 2.0 * np.pi

        # Physical grid
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')

        # Wavevectors
        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0, 0] = 1.0

        # Leray projector components
        self.P = {}
        K = [self.kx, self.ky, self.kz]
        for i in range(3):
            for j in range(3):
                self.P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / self.k2_safe

        # Dealiasing mask (2/3 rule)
        kmax = N // 3
        self.dealias_mask = (
            (np.abs(self.kx) <= kmax) &
            (np.abs(self.ky) <= kmax) &
            (np.abs(self.kz) <= kmax)
        )

        # Build helical basis (once)
        self._build_helical_basis()

    def _build_helical_basis(self):
        """Craya-Herring helical basis."""
        N = self.N
        kmag = np.sqrt(self.k2_safe)
        khat = np.array([self.kx / kmag, self.ky / kmag, self.kz / kmag])

        # e1 = z_ref x khat
        e1 = np.array([khat[1] * 0 - 1.0 * khat[1],  # wrong, redo properly
                        1.0 * khat[0] - khat[0] * 0,
                        khat[0] * 0 - khat[1] * 0])

        # Proper cross product: (0,0,1) x khat
        e1 = np.array([
            0.0 * khat[2] - 1.0 * khat[1],  # 0*kz - 1*ky = -ky
            1.0 * khat[0] - 0.0 * khat[2],   # 1*kx - 0*kz = kx
            0.0 * khat[1] - 0.0 * khat[0],   # 0*ky - 0*kx = 0
        ])
        e1_mag = np.sqrt(np.sum(e1**2, axis=0))

        # Fallback for k parallel to z
        parallel = e1_mag < 1e-10
        if np.any(parallel):
            # (1,0,0) x khat
            e1_alt = np.array([
                0.0 * khat[2] - 0.0 * khat[1],
                0.0 * khat[0] - 1.0 * khat[2],
                1.0 * khat[1] - 0.0 * khat[0],
            ])
            for i in range(3):
                e1[i] = np.where(parallel, e1_alt[i], e1[i])
            e1_mag = np.sqrt(np.sum(e1**2, axis=0))

        e1_mag = np.maximum(e1_mag, 1e-15)
        e1 /= e1_mag

        # e2 = khat x e1
        e2 = np.array([
            khat[1]*e1[2] - khat[2]*e1[1],
            khat[2]*e1[0] - khat[0]*e1[2],
            khat[0]*e1[1] - khat[1]*e1[0],
        ])

        self.h_plus = (e1 + 1j * e2) / np.sqrt(2.0)
        self.h_minus = (e1 - 1j * e2) / np.sqrt(2.0)
        self.h_plus[:, 0, 0, 0] = 0.0
        self.h_minus[:, 0, 0, 0] = 0.0

    def project_leray(self, f_hat):
        """Apply Leray projection to enforce divergence-free."""
        result = np.zeros_like(f_hat)
        for i in range(3):
            for j in range(3):
                result[i] += self.P[(i, j)] * f_hat[j]
        return result

    def project_h_plus(self, f_hat):
        """Remove h- component, keep only h+."""
        # Project onto helical modes
        f_p = np.sum(np.conj(self.h_plus) * f_hat, axis=0)
        f_m = np.sum(np.conj(self.h_minus) * f_hat, axis=0)
        # Reconstruct with only h+
        return f_p[np.newaxis] * self.h_plus

    def compute_vorticity_hat(self, u_hat):
        """omega_hat = i * k x u_hat"""
        return np.array([
            1j * (self.ky * u_hat[2] - self.kz * u_hat[1]),
            1j * (self.kz * u_hat[0] - self.kx * u_hat[2]),
            1j * (self.kx * u_hat[1] - self.ky * u_hat[0]),
        ])

    def compute_rhs(self, u_hat, surgery_mode='none'):
        """
        RHS of NS in rotational form.
        surgery_mode:
          'none' — full NS
          'dynamic' — project nonlinear term onto h+ only
        """
        # Vorticity in Fourier space
        omega_hat = self.compute_vorticity_hat(u_hat)

        # Physical space
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

        # Lamb vector: u x omega
        lamb = np.array([
            u[1]*omega[2] - u[2]*omega[1],
            u[2]*omega[0] - u[0]*omega[2],
            u[0]*omega[1] - u[1]*omega[0],
        ])
        lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])

        # Dealiasing
        for i in range(3):
            lamb_hat[i] *= self.dealias_mask

        # DYNAMIC SURGERY: project nonlinear term onto h+ only
        if surgery_mode == 'dynamic':
            lamb_hat = self.project_h_plus(lamb_hat)

        # Leray projection (remove pressure gradient)
        rhs = self.project_leray(lamb_hat)

        # Viscous term
        rhs -= self.nu * self.k2[np.newaxis] * u_hat

        # Zero mean
        rhs[:, 0, 0, 0] = 0.0

        return rhs

    def step_rk4(self, u_hat, dt, surgery_mode='none'):
        """RK4 time step."""
        k1 = self.compute_rhs(u_hat, surgery_mode)
        k2 = self.compute_rhs(u_hat + 0.5*dt*k1, surgery_mode)
        k3 = self.compute_rhs(u_hat + 0.5*dt*k2, surgery_mode)
        k4 = self.compute_rhs(u_hat + dt*k3, surgery_mode)
        return u_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def taylor_green_ic(self):
        """Taylor-Green vortex initial condition."""
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        u[2] = 0.0
        u_hat = np.array([fftn(u[i]) for i in range(3)])
        return self.project_leray(u_hat)

    def compute_diagnostics(self, u_hat):
        """Full diagnostic suite."""
        N = self.N
        K = [self.kx, self.ky, self.kz]
        omega_hat = self.compute_vorticity_hat(u_hat)

        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

        # Energy and enstrophy
        E = 0.5 * np.mean(np.sum(u**2, axis=0))
        Z = np.mean(np.sum(omega**2, axis=0))

        # Strain tensor
        S = np.zeros((3, 3) + u.shape[1:])
        for i in range(3):
            for j in range(i, 3):
                S_hat = 0.5 * (1j * K[j] * u_hat[i] + 1j * K[i] * u_hat[j])
                S[i, j] = np.real(ifftn(S_hat))
                if j != i:
                    S[j, i] = S[i, j]

        # Stretching
        stretch = np.zeros(u.shape[1:])
        for i in range(3):
            for j in range(3):
                stretch += omega[i] * S[i, j] * omega[j]
        stretching = np.mean(stretch)

        # Lambda max (Frobenius upper bound)
        S_frob_sq = np.zeros(u.shape[1:])
        for i in range(3):
            for j in range(3):
                S_frob_sq += S[i, j]**2
        lambda_max = np.sqrt(np.max(S_frob_sq))

        # C ratio
        C = abs(stretching) / (lambda_max * Z) if lambda_max * Z > 1e-15 else 0.0

        # Dissipation
        dissipation = 0.0
        for i in range(3):
            for j in range(3):
                dw_hat = 1j * K[j] * omega_hat[i]
                dw = np.real(ifftn(dw_hat))
                dissipation += np.mean(dw**2)

        # Helicity
        helicity = np.mean(np.sum(u * omega, axis=0))

        # Helical energy fractions
        u_p = np.sum(np.conj(self.h_plus) * u_hat, axis=0)
        u_m = np.sum(np.conj(self.h_minus) * u_hat, axis=0)
        E_p = np.sum(np.abs(u_p)**2) / N**3
        E_m = np.sum(np.abs(u_m)**2) / N**3
        E_hel = E_p + E_m
        hp_frac = E_p / E_hel if E_hel > 1e-15 else 0.5

        return {
            'E': E, 'Z': Z, 'stretching': stretching,
            'lambda_max': lambda_max, 'C': C,
            'dissipation': dissipation, 'helicity': helicity,
            'hp_frac': hp_frac,
        }


def run_case(solver, u_hat_init, dt, T_final, surgery_mode, label, diag_every=20):
    """Run a single simulation case."""
    print(f"\n--- {label} (surgery={surgery_mode}) ---")
    print(f"{'t':<7} | {'E':<9} | {'Z':<11} | {'Stretch':<11} | {'C':<9} | {'H':<9} | {'h+%':<5}")
    print("-" * 72)

    u_hat = u_hat_init.copy()
    times, diags = [], []
    step = 0
    t = 0.0

    while t <= T_final + 1e-10:
        if step % diag_every == 0:
            d = solver.compute_diagnostics(u_hat)
            times.append(t)
            diags.append(d)
            print(f"{t:<7.2f} | {d['E']:<9.2e} | {d['Z']:<11.4e} | "
                  f"{d['stretching']:<11.4e} | {d['C']:<9.5f} | "
                  f"{d['helicity']:<9.2e} | {d['hp_frac']*100:<5.1f}")

        u_hat = solver.step_rk4(u_hat, dt, surgery_mode)
        t += dt
        step += 1

    return times, diags


def run_comparison():
    """Three-way comparison: Full NS vs IC-surgery vs Dynamic surgery."""
    N = 64
    Re = 400
    T = 6.0
    dt = 0.005

    print("=" * 72)
    print("DYNAMIC vs KINEMATIC HELICAL SURGERY — 3-WAY COMPARISON")
    print(f"N={N}, Re={Re}, T={T}, dt={dt}")
    print("=" * 72)
    print()
    print("Case A: Full NS (all interactions)")
    print("Case B: IC-only surgery (remove h- at t=0, evolve full NS)")
    print("        -> Antigravity's test. Should show kinematic halving.")
    print("Case C: Dynamic surgery (remove h- from nonlinear term every step)")
    print("        -> Correct BT test. If different from B, it's genuine.")
    print()

    solver = SpectralNS(N=N, Re=Re)
    wall_start = clock.time()

    # Full IC
    u_hat_full = solver.taylor_green_ic()

    # IC with h- removed (Antigravity's version)
    u_hat_ic_surgery = solver.project_h_plus(u_hat_full.copy())

    # Case A: Full NS
    tA, dA = run_case(solver, u_hat_full, dt, T, 'none', 'A: FULL NS')
    t_a = clock.time()
    print(f"  [{t_a - wall_start:.1f}s elapsed]")

    # Case B: IC-only surgery (evolve with full NS dynamics)
    tB, dB = run_case(solver, u_hat_ic_surgery, dt, T, 'none', 'B: IC-ONLY SURGERY')
    t_b = clock.time()
    print(f"  [{t_b - wall_start:.1f}s elapsed]")

    # Case C: Dynamic surgery (surgery on nonlinear term every step)
    tC, dC = run_case(solver, u_hat_full, dt, T, 'dynamic', 'C: DYNAMIC SURGERY')
    t_c = clock.time()
    print(f"  [{t_c - wall_start:.1f}s elapsed]")

    # ============================================================
    # COMPARISON
    # ============================================================
    print("\n" + "=" * 72)
    print("COMPARISON RESULTS")
    print("=" * 72)

    # Energy at t=0
    EA0 = dA[0]['E']
    EB0 = dB[0]['E']
    EC0 = dC[0]['E']
    print(f"\nInitial energy:")
    print(f"  A (full):        E0 = {EA0:.6e}")
    print(f"  B (IC surgery):  E0 = {EB0:.6e} (ratio: {EB0/EA0:.4f})")
    print(f"  C (dynamic):     E0 = {EC0:.6e} (ratio: {EC0/EA0:.4f})")

    # Enstrophy peaks
    ZA = [d['Z'] for d in dA]
    ZB = [d['Z'] for d in dB]
    ZC = [d['Z'] for d in dC]
    ZA_max, ZB_max, ZC_max = max(ZA), max(ZB), max(ZC)
    tA_peak = tA[ZA.index(ZA_max)]
    tB_peak = tB[ZB.index(ZB_max)]
    tC_peak = tC[ZC.index(ZC_max)]

    print(f"\nEnstrophy peaks:")
    print(f"  A (full):        Z_max = {ZA_max:.4e} at t = {tA_peak:.2f}")
    print(f"  B (IC surgery):  Z_max = {ZB_max:.4e} at t = {tB_peak:.2f} (ratio: {ZB_max/ZA_max:.4f})")
    print(f"  C (dynamic):     Z_max = {ZC_max:.4e} at t = {tC_peak:.2f} (ratio: {ZC_max/ZA_max:.4f})")

    # C ratio peaks
    CA = [d['C'] for d in dA]
    CB = [d['C'] for d in dB]
    CC = [d['C'] for d in dC]

    print(f"\nC = |Stretching|/(lambda_max * Z) peaks:")
    print(f"  A (full):        C_max = {max(CA):.6f}")
    print(f"  B (IC surgery):  C_max = {max(CB):.6f}")
    print(f"  C (dynamic):     C_max = {max(CC):.6f}")

    # THE KEY TEST
    print("\n" + "=" * 72)
    print("THE KEY TEST: Is dynamic surgery different from IC-only?")
    print("=" * 72)

    # Compare B and C
    ratio_BC = ZC_max / ZB_max if ZB_max > 0 else float('inf')
    print(f"\n  Z_peak ratio C/B = {ratio_BC:.4f}")

    if abs(ratio_BC - 1.0) < 0.05:
        print("  SAME: Dynamic surgery ~ IC-only surgery.")
        print("  The effect is purely kinematic (energy halving).")
        print("  Antigravity's Failure 18 assessment was CORRECT.")
    else:
        print(f"  DIFFERENT: Dynamic surgery differs from IC-only by {abs(ratio_BC-1)*100:.1f}%.")
        if ZC_max < ZB_max * 0.8:
            print("  Dynamic surgery SUPPRESSES enstrophy MORE than kinematic prediction.")
            print("  This suggests cross-helicity triadic transfers genuinely drive enstrophy growth.")
            print("  The BT mechanism has a REAL dynamic effect!")
        elif ZC_max > ZB_max * 1.2:
            print("  Dynamic surgery allows MORE enstrophy than IC-only!")
            print("  Removing h- from nonlinear term changes the dynamics unfavorably.")
        else:
            print("  Moderate difference — needs larger N or longer T to be conclusive.")

    # Energy evolution comparison
    print(f"\nEnergy at final time (t={T}):")
    EA_f = dA[-1]['E']
    EB_f = dB[-1]['E']
    EC_f = dC[-1]['E']
    print(f"  A: {EA_f:.6e} ({EA_f/EA0*100:.1f}% of initial)")
    print(f"  B: {EB_f:.6e} ({EB_f/EB0*100:.1f}% of initial)")
    print(f"  C: {EC_f:.6e} ({EC_f/EC0*100:.1f}% of initial)")

    # h+ fraction evolution (does h- regenerate in case B?)
    print(f"\nh+ energy fraction at final time:")
    print(f"  A (full):       {dA[-1]['hp_frac']*100:.1f}%")
    print(f"  B (IC surgery): {dB[-1]['hp_frac']*100:.1f}%  (started at ~100%)")
    print(f"  C (dynamic):    {dC[-1]['hp_frac']*100:.1f}%")

    if dB[-1]['hp_frac'] < 0.9:
        print(f"  -> In Case B, h- regenerated to {(1-dB[-1]['hp_frac'])*100:.1f}%!")
        print(f"     The nonlinear term creates cross-helicity content dynamically.")
        print(f"     IC-only surgery cannot maintain single-helicity flow.")

    # Stretching comparison at matching times
    print(f"\nStretching term time series (selected times):")
    print(f"{'t':<7} | {'S_A':<11} | {'S_B':<11} | {'S_C':<11} | {'C/B ratio':<10}")
    print("-" * 55)
    for i in range(0, len(tA), max(1, len(tA)//8)):
        if i < len(dA) and i < len(dB) and i < len(dC):
            sa = dA[i]['stretching']
            sb = dB[i]['stretching']
            sc = dC[i]['stretching']
            ratio = sc / sb if abs(sb) > 1e-15 else float('nan')
            print(f"{tA[i]:<7.2f} | {sa:<11.4e} | {sb:<11.4e} | {sc:<11.4e} | {ratio:<10.4f}")


if __name__ == "__main__":
    run_comparison()
