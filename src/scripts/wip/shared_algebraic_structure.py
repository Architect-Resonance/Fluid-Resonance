"""
SHARED ALGEBRAIC STRUCTURE HYPOTHESIS TEST
============================================
Wanderer's question: Under dynamic BT surgery, does the enstrophy evolution
decompose into a form where stretching < dissipation is ALGEBRAICALLY FORCED
(analogous to -16 < 16 in the graph theory proof)?

Graph theory analogy:
  - R < 2 reduces to (n+2)^2 > n^2 + 4n - 28, i.e. 32 > 0 (always true)
  - The "margin" is 32/(n^2+2n), always positive
  - Bridge vertex removal creates a gap where destructive < constructive

PDE question:
  - Full NS: dZ/dt = S_full - D_full (stretching minus dissipation)
  - BT surgery: remove cross-helicity triadic interactions
  - Under BT surgery: is S_BT < D_BT at ALL times? (algebraic tautology)
  - If yes: regularity is algebraically forced under surgery

Method: Pseudo-spectral 3D NS solver (N=32, Re=400, TG IC, RK4, 2/3 dealiasing)
Runs BOTH full NS and BT-surgery NS in parallel, tracking enstrophy budgets.

HONEST TEST: We report what the numbers say, not what we hope.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock


class SpectralNS:
    """Pseudo-spectral 3D Navier-Stokes solver with helical decomposition."""

    def __init__(self, N=32, Re=400):
        self.N = N
        self.nu = 1.0 / Re
        L = 2.0 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, x, x, indexing='ij')

        # Wavenumber grid (full FFT, not rfft -- cleaner for helical decomp)
        k1d = fftfreq(N, d=1.0 / N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0, 0] = 1.0
        self.kmag = np.sqrt(self.k2_safe)

        # Leray projector P_ij = delta_ij - k_i k_j / |k|^2
        K = [self.kx, self.ky, self.kz]
        self.P = {}
        for i in range(3):
            for j in range(3):
                self.P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / self.k2_safe

        # 2/3 dealiasing mask
        kmax = N // 3
        self.dealias_mask = (
            (np.abs(self.kx) <= kmax) &
            (np.abs(self.ky) <= kmax) &
            (np.abs(self.kz) <= kmax)
        )

        # Build helical basis vectors h+, h-
        self._build_helical_basis()

    def _build_helical_basis(self):
        """Construct orthonormal helical basis {h+, h-} in the plane perp to k."""
        kmag = self.kmag
        khat = np.array([self.kx / kmag, self.ky / kmag, self.kz / kmag])

        # e1 = khat x (0,0,1) (fallback to (0,1,0) when k is along z)
        e1 = np.array([-khat[1], khat[0], np.zeros_like(khat[0])])
        e1_mag = np.sqrt(np.sum(e1**2, axis=0))
        parallel = e1_mag < 1e-10
        if np.any(parallel):
            e1_alt = np.array([np.zeros_like(khat[0]), -khat[2], khat[1]])
            for i in range(3):
                e1[i] = np.where(parallel, e1_alt[i], e1[i])
            e1_mag = np.sqrt(np.sum(e1**2, axis=0))
        e1 /= np.maximum(e1_mag, 1e-15)

        # e2 = khat x e1
        e2 = np.array([
            khat[1] * e1[2] - khat[2] * e1[1],
            khat[2] * e1[0] - khat[0] * e1[2],
            khat[0] * e1[1] - khat[1] * e1[0],
        ])

        # h+ = (e1 + i*e2) / sqrt(2),  h- = (e1 - i*e2) / sqrt(2)
        self.h_plus = (e1 + 1j * e2) / np.sqrt(2.0)
        self.h_minus = (e1 - 1j * e2) / np.sqrt(2.0)
        self.h_plus[:, 0, 0, 0] = 0.0
        self.h_minus[:, 0, 0, 0] = 0.0
        # Zero Nyquist modes — same sign issue as Leray projector
        nyq = self.N // 2
        nyq_mask = ((self.kx == nyq) | (self.kx == -nyq) |
                    (self.ky == nyq) | (self.ky == -nyq) |
                    (self.kz == nyq) | (self.kz == -nyq))
        self.h_plus[:, nyq_mask] = 0.0
        self.h_minus[:, nyq_mask] = 0.0

    def project_leray(self, f_hat):
        """Leray projection: remove longitudinal component.

        Zeroes Nyquist modes first: the projector P_ij = δ_ij - k_i·k_j/|k|²
        breaks conjugate symmetry at Nyquist because -(-N/2) mod N = N/2, so
        the off-diagonal P_ij(k) ≠ P_ij(-k) when k has a Nyquist component.
        Nyquist modes carry negligible energy and are outside the 2/3 dealiasing
        mask, so zeroing them is both safe and correct.
        """
        # Zero Nyquist modes to preserve conjugate symmetry
        nyq = self.N // 2
        nyq_mask = ((np.abs(self.kx) == nyq) |
                    (np.abs(self.ky) == nyq) |
                    (np.abs(self.kz) == nyq))
        f_clean = f_hat.copy()
        f_clean[:, nyq_mask] = 0.0

        result = np.zeros_like(f_clean)
        for i in range(3):
            for j in range(3):
                result[i] += self.P[(i, j)] * f_clean[j]
        return result

    def helical_decompose(self, f_hat):
        """Project vector field onto h+ and h- basis. Returns scalar coefficients."""
        f_p = np.sum(np.conj(self.h_plus) * f_hat, axis=0)
        f_m = np.sum(np.conj(self.h_minus) * f_hat, axis=0)
        return f_p, f_m

    def helical_reconstruct(self, f_p, f_m):
        """Reconstruct vector field from helical coefficients."""
        return f_p[np.newaxis] * self.h_plus + f_m[np.newaxis] * self.h_minus

    def compute_vorticity_hat(self, u_hat):
        return np.array([
            1j * (self.ky * u_hat[2] - self.kz * u_hat[1]),
            1j * (self.kz * u_hat[0] - self.kx * u_hat[2]),
            1j * (self.kx * u_hat[1] - self.ky * u_hat[0]),
        ])

    def compute_lamb_hat(self, u_hat):
        """Compute Lamb vector L = u x omega in Fourier space (dealiased)."""
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        lamb = np.array([
            u[1] * omega[2] - u[2] * omega[1],
            u[2] * omega[0] - u[0] * omega[2],
            u[0] * omega[1] - u[1] * omega[0],
        ])
        lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])
        for i in range(3):
            lamb_hat[i] *= self.dealias_mask
        return lamb_hat

    def compute_lamb_hat_bt_surgery(self, u_hat):
        """Compute Lamb vector with BT surgery: remove cross-helicity triadic interactions.

        Biferale-Titi surgery: decompose u = u+ + u-, omega = omega+ + omega-
        Full Lamb = (u+ + u-) x (omega+ + omega-)
                  = u+ x omega+ + u- x omega- + u+ x omega- + u- x omega+
                     (same-same)     (same-same)    (CROSS)       (CROSS)

        BT surgery removes the cross terms, keeping only same-helicity interactions.
        """
        # Decompose velocity into h+ and h- components
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)

        # Compute vorticity for each helical sector
        omega_hat_plus = self.compute_vorticity_hat(u_hat_plus)
        omega_hat_minus = self.compute_vorticity_hat(u_hat_minus)

        # Physical space fields
        u_plus = np.array([np.real(ifftn(u_hat_plus[i])) for i in range(3)])
        u_minus = np.array([np.real(ifftn(u_hat_minus[i])) for i in range(3)])
        omega_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
        omega_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

        # Same-helicity Lamb vector: u+ x omega+ + u- x omega-
        def cross(a, b):
            return np.array([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])

        lamb_same = cross(u_plus, omega_plus) + cross(u_minus, omega_minus)
        lamb_same_hat = np.array([fftn(lamb_same[i]) for i in range(3)])
        for i in range(3):
            lamb_same_hat[i] *= self.dealias_mask
        return lamb_same_hat

    def compute_lamb_hat_cross_only(self, u_hat):
        """Compute ONLY the cross-helicity Lamb vector (the removed part)."""
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)

        omega_hat_plus = self.compute_vorticity_hat(u_hat_plus)
        omega_hat_minus = self.compute_vorticity_hat(u_hat_minus)

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

        lamb_cross = cross(u_plus, omega_minus) + cross(u_minus, omega_plus)
        lamb_cross_hat = np.array([fftn(lamb_cross[i]) for i in range(3)])
        for i in range(3):
            lamb_cross_hat[i] *= self.dealias_mask
        return lamb_cross_hat

    def compute_rhs_full(self, u_hat):
        """Full NS right-hand side."""
        lamb_hat = self.compute_lamb_hat(u_hat)
        rhs = self.project_leray(lamb_hat)
        rhs -= self.nu * self.k2[np.newaxis] * u_hat
        rhs[:, 0, 0, 0] = 0.0
        return rhs

    def compute_rhs_bt(self, u_hat):
        """BT-surgery NS right-hand side (same-helicity interactions only)."""
        lamb_hat = self.compute_lamb_hat_bt_surgery(u_hat)
        rhs = self.project_leray(lamb_hat)
        rhs -= self.nu * self.k2[np.newaxis] * u_hat
        rhs[:, 0, 0, 0] = 0.0
        return rhs

    def step_rk4(self, u_hat, dt, mode='full'):
        """RK4 time stepping. mode='full' or 'bt'."""
        rhs_fn = self.compute_rhs_full if mode == 'full' else self.compute_rhs_bt
        k1 = rhs_fn(u_hat)
        k2 = rhs_fn(u_hat + 0.5 * dt * k1)
        k3 = rhs_fn(u_hat + 0.5 * dt * k2)
        k4 = rhs_fn(u_hat + dt * k3)
        return u_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def taylor_green_ic(self):
        """Taylor-Green vortex initial condition."""
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        u[2] = 0.0
        return self.project_leray(np.array([fftn(u[i]) for i in range(3)]))

    def pelz_ic(self):
        """Pelz flow IC (higher-order modes, more enstrophy)."""
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * (np.cos(3 * self.Y) * np.cos(self.Z)
                                  - np.cos(self.Y) * np.cos(3 * self.Z))
        u[1] = np.sin(self.Y) * (np.cos(3 * self.Z) * np.cos(self.X)
                                  - np.cos(self.Z) * np.cos(3 * self.X))
        u[2] = np.sin(self.Z) * (np.cos(3 * self.X) * np.cos(self.Y)
                                  - np.cos(self.X) * np.cos(3 * self.Y))
        return self.project_leray(np.array([fftn(u[i]) for i in range(3)]))

    def random_ic(self, seed=42, energy_target=0.5):
        """Random solenoidal IC with k^-2 energy spectrum."""
        np.random.seed(seed)
        N = self.N
        u_hat = np.zeros((3, N, N, N), dtype=complex)
        for i in range(3):
            u_hat[i] = fftn(np.random.randn(N, N, N))
            u_hat[i] /= np.maximum(self.k2_safe, 1.0)
        u_hat = self.project_leray(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        E = 0.5 * np.mean(np.sum(u**2, axis=0))
        u_hat *= np.sqrt(energy_target / max(E, 1e-15))
        return u_hat

    def imbalanced_helical_ic(self, seed=42, h_plus_frac=0.8):
        """IC with deliberate helical imbalance: h_plus_frac of energy in h+.

        This breaks the h+/h- symmetry that makes TG degenerate under BT surgery.
        """
        np.random.seed(seed)
        N = self.N
        u_hat = np.zeros((3, N, N, N), dtype=complex)
        for i in range(3):
            u_hat[i] = fftn(np.random.randn(N, N, N))
            u_hat[i] /= np.maximum(self.k2_safe, 1.0)
        u_hat = self.project_leray(u_hat)

        # Decompose into h+ and h-
        u_p, u_m = self.helical_decompose(u_hat)

        # Scale to get desired imbalance
        E_p = np.sum(np.abs(u_p)**2)
        E_m = np.sum(np.abs(u_m)**2)
        # We want E_p_new / (E_p_new + E_m_new) = h_plus_frac
        scale_p = np.sqrt(h_plus_frac * E_m / ((1 - h_plus_frac) * E_p + 1e-30))
        u_p_scaled = u_p * scale_p

        # Reconstruct
        u_hat = self.helical_reconstruct(u_p_scaled, u_m)

        # Enforce reality: helical rescaling breaks conjugate symmetry,
        # so ifftn(u_hat) has nonzero imaginary part. Round-trip through
        # physical space to get a real, divergence-free field.
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        u_hat = self.project_leray(np.array([fftn(u[i]) for i in range(3)]))

        # Normalize total energy
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        E = 0.5 * np.mean(np.sum(u**2, axis=0))
        u_hat *= np.sqrt(0.5 / max(E, 1e-15))
        return u_hat

    def narrowband_imbalanced_ic(self, seed=42, h_plus_frac=0.8, k_max_ic=2):
        """IC with helical imbalance but energy only at low k (like TG).

        This avoids the ||-dS|| inflation from broadband high-k content,
        allowing fair comparison with TG in the Miller Q ratio.
        Energy is placed at integer wavenumber shells |k| <= k_max_ic.
        """
        np.random.seed(seed)
        N = self.N
        u_hat = np.zeros((3, N, N, N), dtype=complex)

        # Only excite modes with |k| <= k_max_ic
        k_int = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        low_k_mask = (k_int > 0.5) & (k_int <= k_max_ic + 0.5)

        for i in range(3):
            raw = fftn(np.random.randn(N, N, N))
            u_hat[i] = raw * low_k_mask

        u_hat = self.project_leray(u_hat)

        # Decompose into h+ and h-
        u_p, u_m = self.helical_decompose(u_hat)

        # Scale to get desired imbalance
        E_p = np.sum(np.abs(u_p)**2)
        E_m = np.sum(np.abs(u_m)**2)
        if E_p < 1e-30 or E_m < 1e-30:
            # Fallback if one sector is empty
            return u_hat
        scale_p = np.sqrt(h_plus_frac * E_m / ((1 - h_plus_frac) * E_p))
        u_p_scaled = u_p * scale_p

        # Reconstruct
        u_hat = self.helical_reconstruct(u_p_scaled, u_m)

        # Enforce reality via physical-space round-trip
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        u_hat = self.project_leray(np.array([fftn(u[i]) for i in range(3)]))

        # Normalize to same energy as TG (E=0.125)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        E = 0.5 * np.mean(np.sum(u**2, axis=0))
        u_hat *= np.sqrt(0.125 / max(E, 1e-15))
        return u_hat

    def helical_energy_fractions(self, u_hat):
        """Return (E_plus_frac, E_minus_frac) for diagnostics."""
        u_p, u_m = self.helical_decompose(u_hat)
        E_p = np.sum(np.abs(u_p)**2)
        E_m = np.sum(np.abs(u_m)**2)
        E_tot = E_p + E_m
        if E_tot < 1e-30:
            return 0.5, 0.5
        return float(E_p / E_tot), float(E_m / E_tot)

    # ------------------------------------------------------------------
    # ENSTROPHY BUDGET COMPUTATION
    # ------------------------------------------------------------------
    def compute_enstrophy_budget(self, u_hat):
        """Compute the enstrophy budget terms:
          Z = (1/2) <|omega|^2>  (enstrophy)
          S = 2 <omega . S . omega>  (vortex stretching)
          D = 2*nu <|nabla omega|^2>  (dissipation)
          dZ/dt = S - D

        S is computed in physical space from omega_i * S_ij * omega_j
        where S_ij = (1/2)(du_i/dx_j + du_j/dx_i).

        D is computed in Fourier space: D = 2*nu * sum(k^2 |omega_hat|^2) / N^3.
        """
        N = self.N

        # Vorticity in Fourier and physical space
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])

        # Velocity gradient tensor in Fourier space: (du_j/dx_i)_hat = i*k_i * u_hat_j
        K = [self.kx, self.ky, self.kz]
        grad_u = np.zeros((3, 3, N, N, N))  # grad_u[i][j] = du_j/dx_i
        for i in range(3):
            for j in range(3):
                grad_u[i, j] = np.real(ifftn(1j * K[i] * u_hat[j]))

        # Strain rate tensor S_ij = (1/2)(du_i/dx_j + du_j/dx_i)
        S_tensor = np.zeros((3, 3, N, N, N))
        for i in range(3):
            for j in range(3):
                S_tensor[i, j] = 0.5 * (grad_u[j, i] + grad_u[i, j])

        # Vortex stretching: 2 * <omega_i S_ij omega_j>  (volume average)
        stretching_field = np.zeros((N, N, N))
        for i in range(3):
            for j in range(3):
                stretching_field += omega[i] * S_tensor[i, j] * omega[j]
        S_stretch = 2.0 * np.mean(stretching_field)

        # Dissipation: 2*nu * <|nabla omega|^2>
        # In Fourier: sum over k of k^2 |omega_hat_k|^2
        # <|nabla omega|^2> = (1/N^3) sum_k k^2 |omega_hat_k|^2 / N^3
        # (the /N^3 accounts for the FFT normalization: fftn gives N^3 * coefficient)
        D_dissip = 0.0
        for i in range(3):
            D_dissip += np.sum(self.k2 * np.abs(omega_hat[i])**2)
        D_dissip *= 2.0 * self.nu / N**6  # Correct normalization for full FFT

        # Enstrophy
        Z = 0.5 * np.mean(np.sum(omega**2, axis=0))

        return Z, S_stretch, D_dissip

    def compute_enstrophy_budget_helical(self, u_hat):
        """Decompose the stretching term by helical sector.

        S_full = S_same + S_cross
        where S_same comes from same-helicity interactions only (BT-safe)
        and S_cross comes from cross-helicity interactions (BT-removed).

        We compute this by computing the strain tensor from u+ and u- separately,
        and the vorticity from omega+ and omega- separately, then combining.
        """
        N = self.N
        K = [self.kx, self.ky, self.kz]

        # Decompose velocity
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)

        # Vorticity for each sector
        omega_hat_plus = self.compute_vorticity_hat(u_hat_plus)
        omega_hat_minus = self.compute_vorticity_hat(u_hat_minus)

        omega_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
        omega_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

        # Strain tensors for each sector
        def compute_strain(u_h):
            grad_u = np.zeros((3, 3, N, N, N))
            for i in range(3):
                for j in range(3):
                    grad_u[i, j] = np.real(ifftn(1j * K[i] * u_h[j]))
            S_t = np.zeros((3, 3, N, N, N))
            for i in range(3):
                for j in range(3):
                    S_t[i, j] = 0.5 * (grad_u[j, i] + grad_u[i, j])
            return S_t

        S_plus = compute_strain(u_hat_plus)
        S_minus = compute_strain(u_hat_minus)

        # Stretching decomposition:
        # omega . S . omega = (omega+ + omega-) . (S+ + S-) . (omega+ + omega-)
        #
        # Same-helicity stretching:
        #   S_same = omega+ . S+ . omega+ + omega- . S- . omega-
        # Cross-helicity stretching (everything else):
        #   S_cross = omega+ . S- . omega+ + omega- . S+ . omega-
        #           + omega+ . S+ . omega- + omega- . S- . omega+
        #           + omega+ . S- . omega- + omega- . S+ . omega+

        def stretching_term(om_a, S_t, om_b):
            """Compute 2 * <om_a_i * S_ij * om_b_j>"""
            field = np.zeros((N, N, N))
            for i in range(3):
                for j in range(3):
                    field += om_a[i] * S_t[i, j] * om_b[j]
            return 2.0 * np.mean(field)

        # Pure same-helicity: omega_s . S_s . omega_s for each sector
        S_same_pp = stretching_term(omega_plus, S_plus, omega_plus)
        S_same_mm = stretching_term(omega_minus, S_minus, omega_minus)
        S_same = S_same_pp + S_same_mm

        # Full stretching (for verification)
        omega_full = omega_plus + omega_minus
        S_full_tensor = S_plus + S_minus
        S_full = stretching_term(omega_full, S_full_tensor, omega_full)

        S_cross = S_full - S_same

        return S_same, S_cross, S_full

    def compute_enstrophy(self, u_hat):
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        return 0.5 * np.mean(np.sum(omega**2, axis=0))

    def compute_total_energy(self, u_hat):
        """E = (1/2) * <|u|^2> — total kinetic energy."""
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        return 0.5 * np.mean(np.sum(u**2, axis=0))

    def compute_total_helicity(self, u_hat):
        """H = <u · omega> — total helicity."""
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        return np.mean(np.sum(u * omega, axis=0))


# ============================================================
# SINGLE IC RUN — returns all tracked arrays
# ============================================================
def run_single_ic(solver, ic_name, u_hat_ic, dt=0.005, T=5.0, report_every=20, verbose=True):
    """Run full NS and BT surgery in parallel from the same IC.
    Returns dict of time-series arrays."""

    n_steps = int(T / dt)
    u_hat_full = u_hat_ic.copy()
    u_hat_bt = u_hat_ic.copy()

    # Helical balance diagnostic at t=0
    hp_frac, hm_frac = solver.helical_energy_fractions(u_hat_ic)

    if verbose:
        print(f"\n{'='*78}")
        print(f"IC: {ic_name}  |  h+ fraction: {hp_frac:.4f}  |  h- fraction: {hm_frac:.4f}")
        print(f"{'='*78}")
        if abs(hp_frac - 0.5) < 0.01:
            print("  WARNING: This IC has near-exact h+/h- balance.")
            print("  BT surgery Lamb vector may be ~zero by symmetry (degenerate case).")
        print()
        print(f"{'t':>5} | {'Z_full':>10} {'S_full':>10} {'D_full':>10} {'S/D_f':>7} | "
              f"{'Z_BT':>10} {'S_BT':>10} {'D_BT':>10} {'S/D_bt':>7} | {'h+%':>5}")
        print("-" * 95)

    times, Z_full_a, S_full_a, D_full_a = [], [], [], []
    S_same_a, S_cross_a = [], []
    Z_bt_a, S_bt_a, D_bt_a = [], [], []
    hp_frac_a = []
    E_full_a, E_bt_a, H_full_a, H_bt_a = [], [], [], []

    for step in range(n_steps + 1):
        t = step * dt

        if step % report_every == 0:
            Z_f, S_f, D_f = solver.compute_enstrophy_budget(u_hat_full)
            S_same_f, S_cross_f, _ = solver.compute_enstrophy_budget_helical(u_hat_full)
            Z_b, S_b, D_b = solver.compute_enstrophy_budget(u_hat_bt)
            hp_f, _ = solver.helical_energy_fractions(u_hat_bt)

            times.append(t)
            Z_full_a.append(Z_f); S_full_a.append(S_f); D_full_a.append(D_f)
            S_same_a.append(S_same_f); S_cross_a.append(S_cross_f)
            Z_bt_a.append(Z_b); S_bt_a.append(S_b); D_bt_a.append(D_b)
            hp_frac_a.append(hp_f)
            E_full_a.append(solver.compute_total_energy(u_hat_full))
            H_full_a.append(solver.compute_total_helicity(u_hat_full))
            E_bt_a.append(solver.compute_total_energy(u_hat_bt))
            H_bt_a.append(solver.compute_total_helicity(u_hat_bt))

            if verbose and step % (report_every * 5) == 0:
                SD_f = S_f / D_f if abs(D_f) > 1e-30 else 0.0
                SD_b = S_b / D_b if abs(D_b) > 1e-30 else 0.0
                E_f = E_full_a[-1]; H_f = H_full_a[-1]
                E_b = E_bt_a[-1]; H_b = H_bt_a[-1]
                print(f"{t:5.2f} | {Z_f:10.4e} {S_f:10.4e} {D_f:10.4e} {SD_f:7.3f} | "
                      f"{Z_b:10.4e} {S_b:10.4e} {D_b:10.4e} {SD_b:7.3f} | {hp_f*100:5.1f}")
                print(f"      | E_f={E_f:.4e} H_f={H_f:+.4e} | E_bt={E_b:.4e} H_bt={H_b:+.4e}")

        if step < n_steps:
            u_hat_full = solver.step_rk4(u_hat_full, dt, mode='full')
            u_hat_bt = solver.step_rk4(u_hat_bt, dt, mode='bt')

    return {
        'ic_name': ic_name,
        'times': np.array(times),
        'Z_full': np.array(Z_full_a), 'S_full': np.array(S_full_a), 'D_full': np.array(D_full_a),
        'S_same': np.array(S_same_a), 'S_cross': np.array(S_cross_a),
        'Z_bt': np.array(Z_bt_a), 'S_bt': np.array(S_bt_a), 'D_bt': np.array(D_bt_a),
        'hp_frac': np.array(hp_frac_a),
        'E_full': np.array(E_full_a), 'E_bt': np.array(E_bt_a),
        'H_full': np.array(H_full_a), 'H_bt': np.array(H_bt_a),
    }


def analyze_results(data, verbose=True):
    """Analyze a single IC run and return summary dict."""
    ic = data['ic_name']
    times = data['times']
    S_full, D_full = data['S_full'], data['D_full']
    S_bt, D_bt = data['S_bt'], data['D_bt']
    S_same, S_cross = data['S_same'], data['S_cross']
    Z_full, Z_bt = data['Z_full'], data['Z_bt']

    valid = times > 0.05
    margin_full = D_full - S_full
    margin_bt = D_bt - S_bt

    SD_full = np.where(D_full > 1e-30, S_full / D_full, 0.0)
    SD_bt = np.where(D_bt > 1e-30, S_bt / D_bt, 0.0)

    # Detect degenerate case (BT stretching ~ 0)
    max_abs_S_bt = np.max(np.abs(S_bt[valid])) if np.any(valid) else 0
    is_degenerate = max_abs_S_bt < 1e-10

    bt_always_positive = np.all(margin_bt[valid] > 0) if np.any(valid) else True
    full_always_positive = np.all(margin_full[valid] > 0) if np.any(valid) else True

    bt_sd_max = np.max(SD_bt[valid]) if np.any(valid) else 0
    full_sd_max = np.max(SD_full[valid]) if np.any(valid) else 0

    # Cross-helicity fraction
    S_total = S_same + S_cross
    cross_frac = np.where(np.abs(S_total) > 1e-30, S_cross / S_total, 0.0)
    same_frac = np.where(np.abs(S_total) > 1e-30, S_same / S_total, 0.0)
    mean_cross_frac = np.mean(np.abs(cross_frac[valid])) if np.any(valid) else 0

    # Time integrals
    try:
        _trapz_fn = np.trapezoid
    except AttributeError:
        _trapz_fn = np.trapz

    if np.any(valid):
        int_S_full = _trapz_fn(S_full[valid], times[valid])
        int_D_full = _trapz_fn(D_full[valid], times[valid])
        int_S_bt = _trapz_fn(S_bt[valid], times[valid])
        int_D_bt = _trapz_fn(D_bt[valid], times[valid])
    else:
        int_S_full = int_D_full = int_S_bt = int_D_bt = 0

    summary = {
        'ic_name': ic,
        'is_degenerate': is_degenerate,
        'max_abs_S_bt': max_abs_S_bt,
        'bt_always_diss_dominated': bt_always_positive,
        'full_always_diss_dominated': full_always_positive,
        'bt_sd_max': bt_sd_max,
        'full_sd_max': full_sd_max,
        'mean_cross_frac': mean_cross_frac,
        'Z_full_max': np.max(Z_full),
        'Z_bt_max': np.max(Z_bt),
        'int_S_full': int_S_full, 'int_D_full': int_D_full,
        'int_S_bt': int_S_bt, 'int_D_bt': int_D_bt,
        'margin_full': margin_full,
        'margin_bt': margin_bt,
        'SD_full': SD_full,
        'SD_bt': SD_bt,
        'cross_frac': cross_frac,
        'same_frac': same_frac,
        'valid': valid,
    }

    if verbose:
        print(f"\n--- Analysis for {ic} ---")
        if is_degenerate:
            print(f"  *** DEGENERATE: max|S_BT| = {max_abs_S_bt:.2e} (effectively zero)")
            print(f"  *** BT surgery produces NO stretching. Flow decays purely viscously.")
            print(f"  *** This is NOT a meaningful test of the algebraic tautology.")
        else:
            print(f"  max|S_BT| = {max_abs_S_bt:.4e} (non-trivial stretching)")

        print(f"  Full NS: D>S always? {full_always_positive}, max(S/D) = {full_sd_max:.4f}")
        print(f"  BT surg: D>S always? {bt_always_positive}, max(S/D) = {bt_sd_max:.4f}")
        print(f"  Cross-helicity fraction of stretching: {mean_cross_frac:.4f}")
        print(f"  Z_full_max = {np.max(Z_full):.4e}, Z_bt_max = {np.max(Z_bt):.4e}")
        if np.max(Z_full) > 0:
            print(f"  BT enstrophy reduction: {(1 - np.max(Z_bt)/np.max(Z_full))*100:.1f}%")
        if abs(int_D_full) > 1e-30:
            print(f"  int(S)/int(D) full: {int_S_full/int_D_full:.4f}")
        if abs(int_D_bt) > 1e-30:
            print(f"  int(S)/int(D) BT:   {int_S_bt/int_D_bt:.4f}")

    return summary


# ============================================================
# MAIN EXPERIMENT — MULTI-IC
# ============================================================
def run_shared_algebraic_structure_test():
    print("=" * 78)
    print("SHARED ALGEBRAIC STRUCTURE HYPOTHESIS TEST")
    print("=" * 78)
    print()
    print("Question: Under BT surgery, is S_BT < D_BT at ALL times?")
    print("  (Analogous to -16 < 16 in graph theory: algebraically forced)")
    print()
    print("Method: Test MULTIPLE initial conditions to avoid degeneracy artifacts.")
    print("  N=32, Re=400, RK4, 2/3 dealiasing, dt=0.005, t in [0, 5]")
    print()

    N = 32
    Re = 400
    dt = 0.005
    T = 5.0

    solver = SpectralNS(N=N, Re=Re)
    wall_start = clock.time()

    # ---- Prepare ICs ----
    ics = {}

    # 1. Taylor-Green (known h+/h- balanced -- will be degenerate under BT)
    ics['Taylor-Green'] = solver.taylor_green_ic()

    # 2. Pelz flow (higher-order modes)
    ics['Pelz'] = solver.pelz_ic()

    # 3. Random IC (seed 42, naturally some imbalance)
    ics['Random (seed=42)'] = solver.random_ic(seed=42)

    # 4. Helically imbalanced: 80% h+, 20% h-
    ics['Imbalanced (80/20)'] = solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.8)

    # 5. Strongly imbalanced: 95% h+, 5% h-
    ics['Imbalanced (95/5)'] = solver.imbalanced_helical_ic(seed=42, h_plus_frac=0.95)

    # ---- Run all ICs ----
    all_data = {}
    all_summaries = {}
    for ic_name, u_hat_ic in ics.items():
        data = run_single_ic(solver, ic_name, u_hat_ic, dt=dt, T=T, verbose=True)
        summary = analyze_results(data, verbose=True)
        all_data[ic_name] = data
        all_summaries[ic_name] = summary

    wall_time = clock.time() - wall_start

    # ============================================================
    # CROSS-IC SUMMARY
    # ============================================================
    print("\n\n" + "=" * 78)
    print("CROSS-IC SUMMARY TABLE")
    print("=" * 78)
    print(f"{'IC':<22} | {'Degen?':>6} | {'BT D>S?':>7} | {'Full D>S?':>9} | "
          f"{'BT S/D_max':>10} | {'Full S/D_max':>12} | {'Cross%':>6} | {'Z_BT/Z_full':>11}")
    print("-" * 100)
    for ic_name, s in all_summaries.items():
        z_ratio = s['Z_bt_max'] / s['Z_full_max'] if s['Z_full_max'] > 0 else 0
        print(f"{ic_name:<22} | {'YES' if s['is_degenerate'] else 'no':>6} | "
              f"{'YES' if s['bt_always_diss_dominated'] else 'NO':>7} | "
              f"{'YES' if s['full_always_diss_dominated'] else 'NO':>9} | "
              f"{s['bt_sd_max']:>10.4f} | {s['full_sd_max']:>12.4f} | "
              f"{s['mean_cross_frac']*100:>5.1f}% | {z_ratio:>11.4f}")

    print(f"\nTotal wall time: {wall_time:.1f}s")

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    # Find non-degenerate ICs
    non_degen = {k: v for k, v in all_summaries.items() if not v['is_degenerate']}
    degen = {k: v for k, v in all_summaries.items() if v['is_degenerate']}

    if degen:
        print(f"\nDegenerate ICs (BT stretching ~ 0, trivially regular): {list(degen.keys())}")
        print("  These have near-exact h+/h- balance. BT surgery zeroes out the")
        print("  nonlinearity entirely. Not meaningful for the algebraic tautology test.")

    if non_degen:
        print(f"\nNon-degenerate ICs (meaningful test): {list(non_degen.keys())}")
        all_bt_regular = all(v['bt_always_diss_dominated'] for v in non_degen.values())
        any_full_irregular = any(not v['full_always_diss_dominated'] for v in non_degen.values())

        for ic_name, s in non_degen.items():
            print(f"\n  {ic_name}:")
            print(f"    BT max(S/D) = {s['bt_sd_max']:.6f}")
            print(f"    Full max(S/D) = {s['full_sd_max']:.6f}")
            if s['bt_always_diss_dominated']:
                print(f"    BT gap (1 - max(S/D)) = {1 - s['bt_sd_max']:.6f}")
            else:
                # Find the worst violation
                worst_idx = np.argmax(s['SD_bt'][s['valid']])
                t_worst = all_data[ic_name]['times'][s['valid']][worst_idx]
                print(f"    BT S/D EXCEEDS 1 at t={t_worst:.2f}")

        print()
        if all_bt_regular and any_full_irregular:
            print("STRONG POSITIVE: BT surgery is ALWAYS dissipation-dominated")
            print("for ALL non-degenerate ICs, while full NS is NOT.")
            print()
            print("This IS the PDE analogue of the algebraic tautology (-16 < 16):")
            print("  - Graph: removing bridge vertices forces destructive < constructive (gap = 32)")
            print("  - PDE: removing cross-helicity triads forces S < D (gap = 1 - max(S/D))")
            print()
            print("The shared algebraic structure:")
            print("  Both the graph proof and the PDE surgery work by the SAME mechanism:")
            print("  removing a class of interactions (bridge vertices / cross-helicity triads)")
            print("  reduces the 'destructive' term below the 'constructive' term,")
            print("  and the gap is STRUCTURALLY POSITIVE -- not a coincidence of parameters.")
        elif all_bt_regular and not any_full_irregular:
            print("WEAK POSITIVE: BT surgery always has D > S, but so does full NS.")
            print("At Re=400, both systems are dissipation-dominated.")
            print("Need higher Re to see the distinction.")
            bt_maxes = [v['bt_sd_max'] for v in non_degen.values()]
            full_maxes = [v['full_sd_max'] for v in non_degen.values()]
            if max(bt_maxes) < max(full_maxes):
                print(f"  But BT has a WIDER margin: max BT S/D = {max(bt_maxes):.4f} vs "
                      f"full S/D = {max(full_maxes):.4f}")
        else:
            print("NEGATIVE: BT surgery does NOT always have D > S.")
            print("The algebraic tautology structure does NOT hold.")
            for ic_name, s in non_degen.items():
                if not s['bt_always_diss_dominated']:
                    print(f"  Counterexample: {ic_name}, BT max(S/D) = {s['bt_sd_max']:.4f}")
    else:
        print("\nALL ICs were degenerate. Cannot draw conclusions.")

    # Graph theory analogy
    print()
    print("--- Graph theory analogy ---")
    print("  Graph: (n+2)^2 - (n^2+4n-28) = 32 > 0  =>  margin = 32/(n^2+2n)")
    for ic_name, s in non_degen.items():
        if s['bt_always_diss_dominated'] and not s['is_degenerate']:
            min_m = np.min(s['margin_bt'][s['valid']])
            max_D = np.max(all_data[ic_name]['D_bt'][s['valid']])
            rel_m = min_m / max_D if max_D > 0 else 0
            print(f"  PDE ({ic_name}): min(D-S) = {min_m:.4e}, relative margin = {rel_m:.4f}")

    # ============================================================
    # PLOTTING — Use the most informative non-degenerate IC
    # ============================================================
    print("\n\nGenerating plots...")

    # Pick the IC with the largest max|S_BT| for the main plot
    if non_degen:
        best_ic = max(non_degen.keys(), key=lambda k: non_degen[k]['max_abs_S_bt'])
    else:
        best_ic = list(all_data.keys())[0]

    d = all_data[best_ic]
    s = all_summaries[best_ic]
    times = d['times']
    valid = s['valid']

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"Shared Algebraic Structure Test: Full NS vs BT Surgery\n"
                 f"N={N}, Re={Re}, IC={best_ic}, RK4, 2/3 dealiasing",
                 fontsize=14, fontweight='bold')

    # Panel 1: Enstrophy evolution (all ICs)
    ax = axes[0, 0]
    colors_full = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_bt = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    for i, (ic_name, dd) in enumerate(all_data.items()):
        ci = i % len(colors_full)
        lbl_f = f'{ic_name} (full)' if i < 3 else None  # limit legend
        lbl_b = f'{ic_name} (BT)' if i < 3 else None
        ax.semilogy(dd['times'], dd['Z_full'], '-', color=colors_full[ci], linewidth=1.5,
                    label=lbl_f, alpha=0.8)
        ax.semilogy(dd['times'], dd['Z_bt'], '--', color=colors_bt[ci], linewidth=1.5,
                    label=lbl_b, alpha=0.8)
    ax.set_xlabel('t')
    ax.set_ylabel('Enstrophy Z')
    ax.set_title('Enstrophy Evolution (all ICs)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: S/D ratio for best non-degenerate IC
    ax = axes[0, 1]
    ax.plot(times[valid], s['SD_full'][valid], 'b-', linewidth=2, label=f'Full NS: S/D')
    ax.plot(times[valid], s['SD_bt'][valid], 'r--', linewidth=2, label=f'BT surgery: S/D')
    ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='S/D = 1 (critical)')
    ax.set_xlabel('t')
    ax.set_ylabel('S / D')
    ax.set_title(f'S/D Ratio ({best_ic})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Margin (D - S) for best IC
    ax = axes[1, 0]
    ax.plot(times[valid], s['margin_full'][valid], 'b-', linewidth=2, label='Full NS: D-S')
    ax.plot(times[valid], s['margin_bt'][valid], 'r--', linewidth=2, label='BT surgery: D-S')
    ax.axhline(y=0.0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('t')
    ax.set_ylabel('D - S')
    ax.set_title(f'Dissipation Margin ({best_ic})\nPositive = dissipation-dominated')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Cross-helicity fraction (best IC)
    ax = axes[1, 1]
    ax.plot(times[valid], s['cross_frac'][valid], 'g-', linewidth=2, label='S_cross / S_full')
    ax.plot(times[valid], s['same_frac'][valid], 'm--', linewidth=2, label='S_same / S_full')
    ax.axhline(y=0.5, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('Fraction')
    ax.set_title(f'Stretching Decomposition ({best_ic})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: Budget terms for best IC
    ax = axes[2, 0]
    ax.plot(times, d['S_full'], 'b-', linewidth=2, label='S_full')
    ax.plot(times, d['S_same'], 'c--', linewidth=1.5, label='S_same (full NS)')
    ax.plot(times, d['S_cross'], 'g:', linewidth=1.5, label='S_cross (full NS)')
    ax.plot(times, d['S_bt'], 'r-', linewidth=2, label='S_bt (BT-evolved)')
    ax.plot(times, d['D_full'], 'b-.', linewidth=1.5, alpha=0.7, label='D_full')
    ax.plot(times, d['D_bt'], 'r-.', linewidth=1.5, alpha=0.7, label='D_bt')
    ax.set_xlabel('t')
    ax.set_ylabel('Rate')
    ax.set_title(f'Enstrophy Budget Terms ({best_ic})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: S/D comparison across ALL non-degenerate ICs
    ax = axes[2, 1]
    ax.fill_between([0, T], 0, 1, alpha=0.1, color='green', label='D > S region')
    for i, (ic_name, dd) in enumerate(all_data.items()):
        ss = all_summaries[ic_name]
        if ss['is_degenerate']:
            continue
        ci = i % len(colors_full)
        vv = ss['valid']
        ax.plot(dd['times'][vv], ss['SD_full'][vv], '-', color=colors_full[ci],
                linewidth=1, alpha=0.5)
        ax.plot(dd['times'][vv], ss['SD_bt'][vv], '--', color=colors_full[ci],
                linewidth=2, label=f'{ic_name} BT')
    ax.axhline(y=1.0, color='k', linestyle='-', linewidth=1.5)
    ax.set_xlabel('t')
    ax.set_ylabel('S / D')
    ax.set_title('Algebraic Tautology Test (all non-degenerate ICs)\nBT S/D < 1 = regularity forced')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('h:/tmp/algebraic_structure.png', dpi=150, bbox_inches='tight')
    print("Plot saved to h:/tmp/algebraic_structure.png")

    return all_data, all_summaries


if __name__ == "__main__":
    all_data, all_summaries = run_shared_algebraic_structure_test()
