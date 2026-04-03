"""
MHD SPECTRAL SOLVER — S36i INVESTIGATION
==========================================
Full magnetohydrodynamics: NS coupled with the induction equation.

Equations (incompressible MHD in rotational form):
  du/dt = (u x omega) + (J x B) - grad(p + B^2/2) + nu * Laplacian(u)
  dB/dt = curl(u x B) + eta * Laplacian(B)
  div(u) = 0, div(B) = 0

where J = curl(B) is the current density.

In Fourier space:
  Nonlinear term = Leray[ u x omega + J x B ]
  Induction term = Leray[ curl(u x B) ] = Leray[ ik x FFT(u x B) ]

The previous attempt (bt_robustness_sweep.py probe_mhd) used a UNIFORM B0
background field, which was trivially advection neutralized by periodic BCs.
This solver uses a DYNAMICAL magnetic field that evolves self-consistently.

Key question: Does magnetic field suppress enstrophy growth similarly to
BT surgery or Coriolis rotation?

Physical mechanism:
  - Alfven waves propagate along B field lines
  - These waves carry energy away from vortex stretching regions
  - At strong B, the flow becomes quasi-2D (perpendicular to B)
  - 2D incompressible NS is globally regular

Reference: Sermange-Temam (1983) — global regularity for 2D MHD.
In 3D: regularity is OPEN (same status as NS).
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import time as clock


class SpectralMHD:
    """Pseudo-spectral MHD solver: NS + induction equation."""

    def __init__(self, N=32, Re=400, Rm=400):
        self.N = N
        self.nu = 1.0 / Re      # kinematic viscosity
        self.eta = 1.0 / Rm     # magnetic diffusivity
        L = 2.0 * np.pi

        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')

        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0, 0] = 1.0

        K = [self.kx, self.ky, self.kz]
        self.P = {}
        for i in range(3):
            for j in range(3):
                self.P[(i, j)] = (1.0 if i == j else 0.0) - K[i]*K[j]/self.k2_safe

        kmax = N // 3
        self.dealias_mask = (
            (np.abs(self.kx) <= kmax) &
            (np.abs(self.ky) <= kmax) &
            (np.abs(self.kz) <= kmax)
        )

    def project_leray(self, f_hat):
        result = np.zeros_like(f_hat)
        for i in range(3):
            for j in range(3):
                result[i] += self.P[(i, j)] * f_hat[j]
        return result

    def curl_hat(self, f_hat):
        """curl(f) in Fourier space = i k x f_hat"""
        return np.array([
            1j*(self.ky*f_hat[2] - self.kz*f_hat[1]),
            1j*(self.kz*f_hat[0] - self.kx*f_hat[2]),
            1j*(self.kx*f_hat[1] - self.ky*f_hat[0]),
        ])

    def cross_product_physical(self, a_hat, b_hat):
        """Compute a x b in physical space, return dealiased Fourier coefficients."""
        a = np.array([np.real(ifftn(a_hat[i])) for i in range(3)])
        b = np.array([np.real(ifftn(b_hat[i])) for i in range(3)])
        cross = np.array([
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        ])
        cross_hat = np.array([fftn(cross[i]) for i in range(3)])
        for i in range(3):
            cross_hat[i] *= self.dealias_mask
        return cross_hat

    def compute_rhs(self, state):
        """
        Compute RHS for both u and B equations.
        state = (u_hat, B_hat) — tuple of [3, N, N, N] complex arrays.
        Returns (rhs_u, rhs_B).
        """
        u_hat, B_hat = state

        # Vorticity: omega = curl(u)
        omega_hat = self.curl_hat(u_hat)

        # Current: J = curl(B)
        J_hat = self.curl_hat(B_hat)

        # Lamb vector: u x omega
        lamb_hat = self.cross_product_physical(u_hat, omega_hat)

        # Lorentz force: J x B
        lorentz_hat = self.cross_product_physical(J_hat, B_hat)

        # Momentum RHS: Leray[ u x omega + J x B ] - nu * k^2 * u
        nonlinear_hat = lamb_hat + lorentz_hat
        rhs_u = self.project_leray(nonlinear_hat)
        rhs_u -= self.nu * self.k2[np.newaxis] * u_hat
        rhs_u[:, 0, 0, 0] = 0.0

        # Induction: curl(u x B) - eta * k^2 * B
        uxB_hat = self.cross_product_physical(u_hat, B_hat)
        curl_uxB = self.curl_hat(uxB_hat)
        rhs_B = self.project_leray(curl_uxB)  # Ensure div(B) = 0
        rhs_B -= self.eta * self.k2[np.newaxis] * B_hat
        rhs_B[:, 0, 0, 0] = 0.0

        return (rhs_u, rhs_B)

    def step_rk4(self, state, dt):
        """RK4 for the coupled (u, B) system."""
        def add_states(s, ds, c):
            return (s[0] + c*ds[0], s[1] + c*ds[1])

        k1 = self.compute_rhs(state)
        k2 = self.compute_rhs(add_states(state, k1, 0.5*dt))
        k3 = self.compute_rhs(add_states(state, k2, 0.5*dt))
        k4 = self.compute_rhs(add_states(state, k3, dt))

        u_new = state[0] + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        B_new = state[1] + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        return (u_new, B_new)

    def taylor_green_ic(self):
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        u[2] = 0.0
        return self.project_leray(np.array([fftn(u[i]) for i in range(3)]))

    def ot_mhd_ic(self):
        """Orszag-Tang vortex MHD IC (extended to 3D).
        Classic MHD benchmark with cross-scale interactions."""
        u = np.zeros((3,) + self.X.shape)
        u[0] = -np.sin(self.Y)
        u[1] = np.sin(self.X)
        u[2] = 0.0

        B = np.zeros((3,) + self.X.shape)
        B[0] = -np.sin(self.Y)
        B[1] = np.sin(2*self.X)
        B[2] = 0.0

        u_hat = self.project_leray(np.array([fftn(u[i]) for i in range(3)]))
        B_hat = self.project_leray(np.array([fftn(B[i]) for i in range(3)]))
        return u_hat, B_hat

    def solenoidal_b_ic(self, B0_strength=1.0):
        """Solenoidal magnetic field + TG velocity.
        B = B0 * (cos(z), 0, 0) is divergence-free (dBx/dx = 0).
        Uses TG velocity IC for fair comparison with pure NS."""
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        u[2] = 0.0

        B = np.zeros((3,) + self.X.shape)
        # B_x = B0*cos(z): divergence-free, single mode k=(0,0,1)
        # perpendicular to k -> survives Leray projection
        B[0] = B0_strength * np.cos(self.Z)

        u_hat = self.project_leray(np.array([fftn(u[i]) for i in range(3)]))
        B_hat = self.project_leray(np.array([fftn(B[i]) for i in range(3)]))
        return u_hat, B_hat

    def compute_diagnostics(self, state):
        u_hat, B_hat = state
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        B = np.array([np.real(ifftn(B_hat[i])) for i in range(3)])

        omega_hat = self.curl_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

        J_hat = self.curl_hat(B_hat)
        J = np.array([np.real(ifftn(J_hat[i])) for i in range(3)])

        E_kin = 0.5 * np.mean(np.sum(u**2, axis=0))
        E_mag = 0.5 * np.mean(np.sum(B**2, axis=0))
        Z = np.mean(np.sum(omega**2, axis=0))
        Z_mag = np.mean(np.sum(J**2, axis=0))  # "magnetic enstrophy"

        # Cross helicity: u . B
        H_cross = np.mean(np.sum(u * B, axis=0))

        # Kinetic helicity: u . omega
        H_kin = np.mean(np.sum(u * omega, axis=0))

        # Stretching
        K = [self.kx, self.ky, self.kz]
        S = np.zeros((3, 3) + u.shape[1:])
        for i in range(3):
            for j in range(i, 3):
                S_hat = 0.5 * (1j * K[j] * u_hat[i] + 1j * K[i] * u_hat[j])
                S[i, j] = np.real(ifftn(S_hat))
                if j != i:
                    S[j, i] = S[i, j]

        stretch = np.zeros(u.shape[1:])
        for i in range(3):
            for j in range(3):
                stretch += omega[i] * S[i, j] * omega[j]
        stretching = np.mean(stretch)

        return {
            'E_kin': E_kin, 'E_mag': E_mag, 'E_tot': E_kin + E_mag,
            'Z': Z, 'Z_mag': Z_mag,
            'stretching': stretching,
            'H_cross': H_cross, 'H_kin': H_kin,
        }


# ============================================================
# EXPERIMENT 1: MHD vs pure NS — enstrophy comparison
# ============================================================
def experiment_1_mhd_vs_ns():
    """Compare enstrophy growth: pure NS vs MHD with various B0 strengths."""
    print("=" * 72)
    print("EXPERIMENT 1: MHD vs PURE NS — ENSTROPHY COMPARISON")
    print("=" * 72)

    N = 32
    Re = 400
    Rm = 400  # Magnetic Reynolds number
    T = 4.0
    dt = 0.005

    # Case A: Pure NS (B = 0)
    print("\n--- Case A: Pure NS (no magnetic field) ---")
    solver_ns = SpectralMHD(N=N, Re=Re, Rm=Rm)
    u_hat = solver_ns.taylor_green_ic()
    B_hat = np.zeros_like(u_hat)
    state = (u_hat, B_hat)

    print(f"{'t':<6} | {'E_kin':<10} | {'E_mag':<10} | {'Z':<11} | {'Stretch':<11}")
    print("-" * 55)

    ns_Z_peak = 0
    t, step = 0.0, 0
    diag_every = max(1, int(0.5 / dt))
    while t <= T + 1e-10:
        if step % diag_every == 0:
            d = solver_ns.compute_diagnostics(state)
            ns_Z_peak = max(ns_Z_peak, d['Z'])
            print(f"{t:<6.2f} | {d['E_kin']:<10.4e} | {d['E_mag']:<10.4e} | "
                  f"{d['Z']:<11.4e} | {d['stretching']:<11.4e}")
        state = solver_ns.step_rk4(state, dt)
        t += dt
        step += 1

    # Cases B-D: MHD with increasing B0 (same TG velocity IC)
    B0_values = [0.5, 1.0, 2.0, 5.0]
    mhd_results = []

    for B0 in B0_values:
        print(f"\n--- Case: MHD with B0={B0} (TG velocity + solenoidal B) ---")
        solver = SpectralMHD(N=N, Re=Re, Rm=Rm)
        u_hat, B_hat = solver.solenoidal_b_ic(B0_strength=B0)
        state = (u_hat, B_hat)

        print(f"{'t':<6} | {'E_kin':<10} | {'E_mag':<10} | {'Z':<11} | {'Z_mag':<11} | {'Stretch':<11}")
        print("-" * 70)

        Z_peak = 0
        t, step = 0.0, 0
        while t <= T + 1e-10:
            if step % diag_every == 0:
                d = solver.compute_diagnostics(state)
                Z_peak = max(Z_peak, d['Z'])
                print(f"{t:<6.2f} | {d['E_kin']:<10.4e} | {d['E_mag']:<10.4e} | "
                      f"{d['Z']:<11.4e} | {d['Z_mag']:<11.4e} | {d['stretching']:<11.4e}")
            state = solver.step_rk4(state, dt)
            t += dt
            step += 1

        mhd_results.append({'B0': B0, 'Z_peak': Z_peak})

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY: Z_peak comparison")
    print(f"{'=' * 72}")
    print(f"  Pure NS:       Z_peak = {ns_Z_peak:.4e}")
    for r in mhd_results:
        ratio = r['Z_peak'] / ns_Z_peak if ns_Z_peak > 1e-15 else 0
        print(f"  MHD B0={r['B0']:<4}: Z_peak = {r['Z_peak']:<10.4e}  (ratio = {ratio:.3f})")


# ============================================================
# EXPERIMENT 2: Orszag-Tang MHD benchmark
# ============================================================
def experiment_2_orszag_tang():
    """Orszag-Tang vortex — classic MHD benchmark.
    Has both velocity and magnetic field at comparable strength."""
    print(f"\n{'=' * 72}")
    print("EXPERIMENT 2: ORSZAG-TANG MHD VORTEX")
    print(f"{'=' * 72}")

    N = 32
    Re = 400
    Rm = 400
    T = 4.0
    dt = 0.005

    solver = SpectralMHD(N=N, Re=Re, Rm=Rm)
    u_hat, B_hat = solver.ot_mhd_ic()

    # Also run pure NS with same velocity IC for comparison
    state_mhd = (u_hat.copy(), B_hat.copy())
    state_ns = (u_hat.copy(), np.zeros_like(u_hat))

    print(f"\n{'t':<6} | {'Z_mhd':<11} | {'Z_ns':<11} | {'E_mag':<10} | {'Z_ratio':<8} | {'H_cross':<10}")
    print("-" * 65)

    t, step = 0.0, 0
    diag_every = max(1, int(0.5 / dt))
    while t <= T + 1e-10:
        if step % diag_every == 0:
            d_mhd = solver.compute_diagnostics(state_mhd)
            d_ns = solver.compute_diagnostics(state_ns)
            ratio = d_mhd['Z'] / d_ns['Z'] if d_ns['Z'] > 1e-15 else 0
            print(f"{t:<6.2f} | {d_mhd['Z']:<11.4e} | {d_ns['Z']:<11.4e} | "
                  f"{d_mhd['E_mag']:<10.4e} | {ratio:<8.3f} | {d_mhd['H_cross']:<10.4e}")
        state_mhd = solver.step_rk4(state_mhd, dt)
        state_ns = solver.step_rk4(state_ns, dt)
        t += dt
        step += 1


# ============================================================
# EXPERIMENT 3: Magnetic Prandtl number sweep
# ============================================================
def experiment_3_pm_sweep():
    """How does the magnetic Prandtl number Pm = nu/eta affect enstrophy?
    Pm < 1: magnetic field diffuses faster (liquid metals)
    Pm = 1: equal diffusion
    Pm > 1: magnetic field persists longer (plasma)"""
    print(f"\n{'=' * 72}")
    print("EXPERIMENT 3: MAGNETIC PRANDTL NUMBER SWEEP")
    print(f"{'=' * 72}")

    N = 32
    Re = 400
    T = 3.0
    dt = 0.005
    B0 = 1.0

    Pm_values = [0.1, 0.5, 1.0, 2.0, 10.0]

    print(f"\n{'Pm':<6} | {'Rm':<6} | {'Z_peak_mhd':<12} | {'Z_peak_ns':<12} | {'Z_ratio':<8}")
    print("-" * 52)

    for Pm in Pm_values:
        Rm = Re * Pm

        solver = SpectralMHD(N=N, Re=Re, Rm=Rm)
        u_hat, B_hat = solver.solenoidal_b_ic(B0_strength=B0)
        state_mhd = (u_hat.copy(), B_hat.copy())
        state_ns = (u_hat.copy(), np.zeros_like(u_hat))

        Z_peak_mhd, Z_peak_ns = 0, 0
        t, step = 0.0, 0
        while t <= T:
            d_mhd = solver.compute_diagnostics(state_mhd)
            d_ns = solver.compute_diagnostics(state_ns)
            Z_peak_mhd = max(Z_peak_mhd, d_mhd['Z'])
            Z_peak_ns = max(Z_peak_ns, d_ns['Z'])

            state_mhd = solver.step_rk4(state_mhd, dt)
            state_ns = solver.step_rk4(state_ns, dt)
            t += dt
            step += 1

        ratio = Z_peak_mhd / Z_peak_ns if Z_peak_ns > 1e-15 else 0
        print(f"{Pm:<6.1f} | {Rm:<6.0f} | {Z_peak_mhd:<12.4e} | {Z_peak_ns:<12.4e} | {ratio:<8.3f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    wall_start = clock.time()

    experiment_1_mhd_vs_ns()
    experiment_2_orszag_tang()
    experiment_3_pm_sweep()

    print(f"\n\nTotal wall time: {clock.time() - wall_start:.1f}s")
    print(f"\n{'=' * 72}")
    print("MHD INVESTIGATION COMPLETE")
    print("Key question: Does magnetic field suppress enstrophy like BT surgery?")
    print(f"{'=' * 72}")
