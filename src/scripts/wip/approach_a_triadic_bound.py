"""
APPROACH A — TRIADIC CAUCHY-SCHWARZ TIGHTENING
================================================
S92 task: Meridian 2 => Meridian 1 (2026-03-12)

PROBLEM:
  The global Cauchy-Schwarz bound on ||P_sol(omega x v)|| is 600x too loose.
  Global C-S: ||P_sol(L_cross)|| <= ||omega+||*||v-|| + ||omega-||*||v+||
  Actual tightness: 0.16% (ratio 0.0016).

IDEA:
  Replace global C-S with a triad-by-triad bound. The Lamb vector at output
  wavenumber k3 comes from triadic interactions k1 + k2 = k3. If only a few
  triads contribute significantly at each k3, the Bessel inequality is tighter:

    ||P_sol(L_cross)(k3)||^2 <= |P(k3)|^2 * sum_triads |omega+(k1)|^2 * |v-(k2)|^2

  Summing over k3:
    ||P_sol(L_cross)||^2 <= sum_k3 |P(k3)|^2 * sum_{k1+k2=k3} |omega_x(k1)|^2 * |v_y(k2)|^2

  where (x,y) range over the cross-helical pairs (+,-) and (-,+).

  The "effective dimension" d_eff(k3) = number of triads contributing >1%
  of the total at each k3. If d_eff << N^3, we get a tighter bound.

HONEST TEST: We report what the numbers say, not what we hope.
"""

import sys
import os
import numpy as np
from numpy.fft import fftn, ifftn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS
from approach_a_helicity_bound import HelicitySectorAnalyzer


class TriadicBoundAnalyzer(HelicitySectorAnalyzer):
    """Extends HelicitySectorAnalyzer with triad-by-triad bound computation."""

    def _shell_index(self, kmag_val):
        """Map |k| to integer shell index."""
        return int(np.round(kmag_val))

    def compute_triadic_bound_direct(self, u_hat, verbose=True):
        """Compute the triad-by-triad Bessel bound on ||P_sol(L_cross)||.

        For a 3D periodic domain, convolution in Fourier space:
          L_cross_hat(k3) = sum_{k1+k2=k3} omega_x(k1) x v_y(k2)

        where (x,y) are cross-helical pairs.

        Instead of the global bound:
          ||L_cross|| <= ||omega+||*||v-|| + ||omega-||*||v+||

        We compute mode-by-mode:
          |L_cross_hat(k3)|^2 <= (sum_{k1+k2=k3} |omega_x(k1)| * |v_y(k2)|)^2

        And then the Bessel bound:
          |L_cross_hat(k3)|^2 <= N_triads(k3) * sum_{k1+k2=k3} |omega_x(k1)|^2 * |v_y(k2)|^2

        This uses C-S on the inner sum, with the looseness factor = N_triads(k3).
        The key question: is N_triads(k3) much smaller than the global N^3?
        """
        N = self.N

        # Helical decomposition
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)

        omega_hat_plus = self.compute_vorticity_hat(u_hat_plus)
        omega_hat_minus = self.compute_vorticity_hat(u_hat_minus)

        # Compute actual P_sol(L_cross) for comparison
        u_plus = np.array([np.real(ifftn(u_hat_plus[i])) for i in range(3)])
        u_minus = np.array([np.real(ifftn(u_hat_minus[i])) for i in range(3)])
        omega_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
        omega_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

        def cross_phys(a, b):
            return np.array([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])

        # Cross-helical Lamb vector in physical space, then to Fourier
        lamb_cross_phys = cross_phys(omega_plus, u_minus) + cross_phys(omega_minus, u_plus)
        lamb_cross_hat = np.array([fftn(lamb_cross_phys[i]) for i in range(3)])
        for i in range(3):
            lamb_cross_hat[i] *= self.dealias_mask

        # Leray projection
        psol_cross_hat = self.project_leray(lamb_cross_hat)

        # Actual ||P_sol(L_cross)||^2
        actual_norm_sq = np.sum(np.abs(psol_cross_hat)**2).real / N**3

        # --- TRIADIC BOUND ---
        # The convolution theorem tells us:
        #   L_cross_hat(k3) = FFT(omega_x * v_y) for the two cross pairs
        # This IS the sum over all triads k1+k2=k3.
        #
        # To get the triad-by-triad bound, we need to measure the actual
        # convolution spectrum mode by mode. In 3D, explicit triad enumeration
        # is O(N^6) and impractical. Instead, we use shell-averaged bounds.
        #
        # SHELL APPROACH:
        #   Group modes by |k| into shells. For each output shell K3:
        #   |L_cross_hat|^2(K3) <= N_triads(K3) * sum_{K1+K2~K3} E_omega(K1) * E_v(K2)
        #   where E_omega(K) = sum_{|k|~K} |omega_hat(k)|^2 is the shell spectrum.

        kmax = N // 3  # dealiased range
        n_shells = kmax + 1

        # Shell spectra for omega+, omega-, v+, v-
        kmag_grid = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        shell_idx = np.round(kmag_grid).astype(int)

        # Compute shell spectra (vector fields: sum over 3 components)
        def shell_spectrum(f_hat, max_shell):
            """Shell-averaged energy spectrum: E(K) = sum_{|k|~K} |f_hat(k)|^2."""
            spec = np.zeros(max_shell + 1)
            for i in range(3):
                contrib = np.abs(f_hat[i])**2
                for K in range(max_shell + 1):
                    mask = (shell_idx == K) & self.dealias_mask
                    spec[K] += np.sum(contrib[mask]).real
            return spec

        # Scalar shell spectrum (for individual components)
        def shell_spectrum_scalar(f_hat_scalar, max_shell):
            spec = np.zeros(max_shell + 1)
            contrib = np.abs(f_hat_scalar)**2
            for K in range(max_shell + 1):
                mask = (shell_idx == K) & self.dealias_mask
                spec[K] += np.sum(contrib[mask]).real
            return spec

        E_omega_plus = shell_spectrum(omega_hat_plus, kmax)
        E_omega_minus = shell_spectrum(omega_hat_minus, kmax)
        E_v_plus = shell_spectrum(u_hat_plus, kmax)
        E_v_minus = shell_spectrum(u_hat_minus, kmax)

        # Actual P_sol(L_cross) shell spectrum
        E_psol = shell_spectrum(psol_cross_hat, kmax)

        # L_cross shell spectrum (before Leray)
        E_lamb = shell_spectrum(lamb_cross_hat, kmax)

        # --- CONVOLUTION BOUND ---
        # For cross-helical: L_cross = omega+ x v- + omega- x v+
        # In Fourier: L_cross(k3) = sum_{k1+k2=k3} [omega+(k1) x v-(k2) + omega-(k1) x v+(k2)]
        #
        # Shell bound:
        #   E_lamb(K3) <= [sum_{K1+K2>=K3, |K1-K2|<=K3} sqrt(E_omega+(K1)*E_v-(K2))
        #                 + sqrt(E_omega-(K1)*E_v+(K2))]^2
        #
        # But this is still a C-S type bound. The TIGHTER approach:
        #   E_lamb(K3) <= sum_{K1,K2 in triadic range} [E_omega+(K1)*E_v-(K2) + E_omega-(K1)*E_v+(K2)]
        #                 * n_modes(K3)
        # where n_modes(K3) = number of mode pairs in the shell.

        # Count modes per shell
        n_modes = np.zeros(n_shells)
        for K in range(n_shells):
            n_modes[K] = np.sum((shell_idx == K) & self.dealias_mask)

        # Triadic shell bound (convolution of shell spectra)
        # For each output shell K3, sum over all shell pairs (K1, K2)
        # satisfying triangle inequality: |K1-K2| <= K3 <= K1+K2
        conv_bound = np.zeros(n_shells)
        for K3 in range(n_shells):
            for K1 in range(n_shells):
                for K2 in range(n_shells):
                    if abs(K1 - K2) <= K3 <= K1 + K2:
                        # Cross-helical: omega+ x v- and omega- x v+
                        conv_bound[K3] += (E_omega_plus[K1] * E_v_minus[K2]
                                           + E_omega_minus[K1] * E_v_plus[K2])

        # Leray projector factor: |P(k)|^2 <= 1, so Leray doesn't change the bound
        # But we can measure how much Leray helps
        leray_factor = np.zeros(n_shells)
        for K in range(n_shells):
            if E_lamb[K] > 1e-30:
                leray_factor[K] = E_psol[K] / E_lamb[K]

        # Effective number of triads contributing to each shell
        # d_eff(K3) = (sum |a_k|)^2 / (N * sum |a_k|^2) where a_k are triad amplitudes
        # We approximate this from the shell spectra
        n_triads = np.zeros(n_shells)
        for K3 in range(n_shells):
            for K1 in range(n_shells):
                for K2 in range(n_shells):
                    if abs(K1 - K2) <= K3 <= K1 + K2:
                        contrib = (E_omega_plus[K1] * E_v_minus[K2]
                                   + E_omega_minus[K1] * E_v_plus[K2])
                        if contrib > 0:
                            n_triads[K3] += 1

        # Global bounds for comparison
        global_cs_sq = (np.sqrt(np.sum(E_omega_plus)) * np.sqrt(np.sum(E_v_minus))
                        + np.sqrt(np.sum(E_omega_minus)) * np.sqrt(np.sum(E_v_plus)))**2 / N**3

        triadic_bound_sq = np.sum(conv_bound) / N**3
        actual_sq = actual_norm_sq

        # Tightness ratios
        tightness_global = actual_sq / max(global_cs_sq, 1e-30)
        tightness_triadic = actual_sq / max(triadic_bound_sq, 1e-30)

        if verbose:
            print(f"\n{'='*70}")
            print(f"TRIADIC CAUCHY-SCHWARZ BOUND ANALYSIS")
            print(f"{'='*70}")
            print(f"  N = {N}, kmax = {kmax}")
            print(f"\n  ||P_sol(L_cross)||^2     = {actual_sq:.6e}  (actual)")
            print(f"  Global C-S bound^2       = {global_cs_sq:.6e}")
            print(f"  Triadic shell bound^2    = {triadic_bound_sq:.6e}")
            print(f"\n  Tightness (actual/global)  = {tightness_global:.6f}  ({tightness_global*100:.4f}%)")
            print(f"  Tightness (actual/triadic) = {tightness_triadic:.6f}  ({tightness_triadic*100:.4f}%)")
            print(f"  Improvement factor         = {tightness_triadic / max(tightness_global, 1e-30):.2f}x")

            # Shell-by-shell analysis
            print(f"\n  {'Shell':>5} {'E_psol':>12} {'Conv_bound':>12} {'Leray_f':>9} {'n_triads':>9} {'Tightness':>10}")
            print(f"  {'-'*57}")
            for K in range(min(n_shells, 15)):
                if E_psol[K] > 1e-30 or conv_bound[K] > 1e-30:
                    tight = E_psol[K] / max(conv_bound[K], 1e-30)
                    print(f"  {K:5d} {E_psol[K]:12.4e} {conv_bound[K]:12.4e} "
                          f"{leray_factor[K]:9.4f} {n_triads[K]:9.0f} {tight:10.6f}")

            # Effective dimension analysis
            print(f"\n  EFFECTIVE DIMENSION ANALYSIS:")
            active_shells = conv_bound > 1e-30 * np.max(conv_bound)
            if np.any(active_shells):
                mean_triads = np.mean(n_triads[active_shells])
                max_triads = np.max(n_triads)
                print(f"    Mean triads per active shell: {mean_triads:.1f}")
                print(f"    Max triads at any shell: {max_triads:.0f}")
                print(f"    Total active shells: {np.sum(active_shells)}")
                print(f"    Total possible shell pairs: {n_shells**2}")
                print(f"    Effective dimension ratio: {mean_triads / max(n_shells**2, 1):.4f}")

        return {
            'actual_sq': actual_sq,
            'global_cs_sq': global_cs_sq,
            'triadic_bound_sq': triadic_bound_sq,
            'tightness_global': tightness_global,
            'tightness_triadic': tightness_triadic,
            'improvement': tightness_triadic / max(tightness_global, 1e-30),
            'E_psol': E_psol,
            'conv_bound': conv_bound,
            'leray_factor': leray_factor,
            'n_triads': n_triads,
            'n_modes': n_modes,
        }

    def compute_refined_triadic_bound(self, u_hat, verbose=True):
        """REFINED bound: use spectral localization of omega and v.

        Key insight: if omega is concentrated at high k and v at low k,
        then the triads that actually contribute are restricted. We can
        exploit this spectral non-uniformity.

        Method: Weight each triad (K1, K2) -> K3 by:
          w(K1,K2) = E_omega(K1)/||omega||^2 * E_v(K2)/||v||^2
        This is the "probability" that a random mode pair hits this triad.

        The effective number of triads = 1 / sum w(K1,K2)^2  (inverse participation ratio)
        This can be MUCH smaller than the total count.
        """
        N = self.N
        kmax = N // 3
        n_shells = kmax + 1

        # Helical decomposition
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)
        omega_hat_plus = self.compute_vorticity_hat(u_hat_plus)
        omega_hat_minus = self.compute_vorticity_hat(u_hat_minus)

        kmag_grid = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        shell_idx = np.round(kmag_grid).astype(int)

        def shell_spectrum(f_hat, max_shell):
            spec = np.zeros(max_shell + 1)
            for i in range(3):
                contrib = np.abs(f_hat[i])**2
                for K in range(max_shell + 1):
                    mask = (shell_idx == K) & self.dealias_mask
                    spec[K] += np.sum(contrib[mask]).real
            return spec

        E_op = shell_spectrum(omega_hat_plus, kmax)
        E_om = shell_spectrum(omega_hat_minus, kmax)
        E_vp = shell_spectrum(u_hat_plus, kmax)
        E_vm = shell_spectrum(u_hat_minus, kmax)

        # Normalize to get probability distributions
        p_op = E_op / max(np.sum(E_op), 1e-30)
        p_om = E_om / max(np.sum(E_om), 1e-30)
        p_vp = E_vp / max(np.sum(E_vp), 1e-30)
        p_vm = E_vm / max(np.sum(E_vm), 1e-30)

        # Inverse participation ratio for each cross pair
        # Cross pair 1: omega+ x v-
        ipr_1 = 0.0
        total_w_sq_1 = 0.0
        for K1 in range(n_shells):
            for K2 in range(n_shells):
                w = p_op[K1] * p_vm[K2]
                total_w_sq_1 += w**2

        # Cross pair 2: omega- x v+
        total_w_sq_2 = 0.0
        for K1 in range(n_shells):
            for K2 in range(n_shells):
                w = p_om[K1] * p_vp[K2]
                total_w_sq_2 += w**2

        d_eff_1 = 1.0 / max(total_w_sq_1, 1e-30)
        d_eff_2 = 1.0 / max(total_w_sq_2, 1e-30)

        # Spectral entropy for each field (how spread is the energy?)
        def spectral_entropy(p):
            p_pos = p[p > 1e-30]
            return -np.sum(p_pos * np.log(p_pos))

        H_op = spectral_entropy(p_op)
        H_om = spectral_entropy(p_om)
        H_vp = spectral_entropy(p_vp)
        H_vm = spectral_entropy(p_vm)

        # The refined bound uses the fact that the triadic interaction
        # is concentrated in a subspace of dimension d_eff << N^2
        # This gives a factor of d_eff/N^2 improvement over global C-S

        # Compute actual vs bound for validation
        results = self.compute_triadic_bound_direct(u_hat, verbose=False)

        if verbose:
            print(f"\n{'='*70}")
            print(f"REFINED TRIADIC BOUND — SPECTRAL LOCALIZATION")
            print(f"{'='*70}")
            print(f"\n  Spectral entropy (higher = more spread):")
            print(f"    omega+: H = {H_op:.3f}  (max = {np.log(n_shells):.3f})")
            print(f"    omega-: H = {H_om:.3f}")
            print(f"    v+:     H = {H_vp:.3f}")
            print(f"    v-:     H = {H_vm:.3f}")
            print(f"\n  Effective triad dimension (inverse participation ratio):")
            print(f"    omega+ x v-:  d_eff = {d_eff_1:.1f}  (of {n_shells**2} possible)")
            print(f"    omega- x v+:  d_eff = {d_eff_2:.1f}")
            print(f"    Compression:  {d_eff_1/n_shells**2:.4f} and {d_eff_2/n_shells**2:.4f}")
            print(f"\n  Expected improvement from localization: ~{n_shells**2/max(d_eff_1+d_eff_2,1)*2:.1f}x")

        return {
            'd_eff_1': d_eff_1,
            'd_eff_2': d_eff_2,
            'total_shells_sq': n_shells**2,
            'compression_1': d_eff_1 / n_shells**2,
            'compression_2': d_eff_2 / n_shells**2,
            'H_op': H_op, 'H_om': H_om, 'H_vp': H_vp, 'H_vm': H_vm,
            'max_entropy': np.log(n_shells),
            'base_results': results,
        }

    def compute_mode_resolved_tightness(self, u_hat, verbose=True):
        """The REAL test: measure actual vs C-S at each output wavenumber.

        For each k3 in Fourier space, we know:
          L_cross_hat(k3) = FFT[omega_x * v_y](k3) = sum_{k1+k2=k3} omega_x_hat(k1) * v_y_hat(k2)

        The actual value |L_cross_hat(k3)| can be compared to:
          Bound(k3) = sum_{k1+k2=k3} |omega_x_hat(k1)| * |v_y_hat(k2)|

        This measures the PHASE CANCELLATION: how much the complex phases
        in the convolution sum cancel each other out.

        Phase cancellation IS the tightening mechanism. If phases are random,
        we expect |sum a_k e^{i phi_k}| ~ sqrt(N) * rms(a_k) instead of N * rms(a_k).
        That's a sqrt(N) improvement.
        """
        N = self.N

        # Get helical components
        u_p, u_m = self.helical_decompose(u_hat)
        u_hat_plus = self.helical_reconstruct(u_p, np.zeros_like(u_m))
        u_hat_minus = self.helical_reconstruct(np.zeros_like(u_p), u_m)
        omega_hat_plus = self.compute_vorticity_hat(u_hat_plus)
        omega_hat_minus = self.compute_vorticity_hat(u_hat_minus)

        # Cross-helical Lamb in Fourier space (via physical space convolution)
        u_plus = np.array([np.real(ifftn(u_hat_plus[i])) for i in range(3)])
        u_minus = np.array([np.real(ifftn(u_hat_minus[i])) for i in range(3)])
        omega_plus = np.array([np.real(ifftn(omega_hat_plus[i])) for i in range(3)])
        omega_minus = np.array([np.real(ifftn(omega_hat_minus[i])) for i in range(3)])

        def cross_phys(a, b):
            return np.array([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])

        # Actual cross Lamb (with phases)
        L_cross_hat = np.array([fftn(cross_phys(omega_plus, u_minus)[i]
                                      + cross_phys(omega_minus, u_plus)[i])
                                 for i in range(3)])
        for i in range(3):
            L_cross_hat[i] *= self.dealias_mask

        # Absolute-value convolution bound (no phase cancellation)
        # |omega+| x |v-| + |omega-| x |v+| in physical space, then FFT
        abs_omega_plus = np.sqrt(np.sum(omega_plus**2, axis=0))
        abs_u_minus = np.sqrt(np.sum(u_minus**2, axis=0))
        abs_omega_minus = np.sqrt(np.sum(omega_minus**2, axis=0))
        abs_u_plus = np.sqrt(np.sum(u_plus**2, axis=0))

        # The "no-phase" bound: replace vector cross product with scalar product of magnitudes
        # This measures the GEOMETRIC cancellation (cross product direction averaging)
        # plus the PHASE cancellation (Fourier coefficient phases)
        no_phase_field = abs_omega_plus * abs_u_minus + abs_omega_minus * abs_u_plus
        no_phase_hat = fftn(no_phase_field) * self.dealias_mask

        # Now compare mode-by-mode
        # Actual: |L_cross_hat(k)| (vector norm across 3 components)
        actual_per_mode = np.sqrt(np.sum(np.abs(L_cross_hat)**2, axis=0))
        # Bound: |no_phase_hat(k)| (scalar, already summed)
        bound_per_mode = np.abs(no_phase_hat)

        # Shell average
        kmax = N // 3
        n_shells = kmax + 1
        kmag_grid = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        shell_idx = np.round(kmag_grid).astype(int)

        actual_shell = np.zeros(n_shells)
        bound_shell = np.zeros(n_shells)
        n_modes_shell = np.zeros(n_shells)
        for K in range(n_shells):
            mask = (shell_idx == K) & self.dealias_mask
            actual_shell[K] = np.sum(actual_per_mode[mask]**2).real
            bound_shell[K] = np.sum(bound_per_mode[mask]**2).real
            n_modes_shell[K] = np.sum(mask)

        # Also measure after Leray projection
        psol_hat = self.project_leray(L_cross_hat)
        psol_per_mode = np.sqrt(np.sum(np.abs(psol_hat)**2, axis=0))
        psol_shell = np.zeros(n_shells)
        for K in range(n_shells):
            mask = (shell_idx == K) & self.dealias_mask
            psol_shell[K] = np.sum(psol_per_mode[mask]**2).real

        # Tightness by shell
        tightness_shell = np.zeros(n_shells)
        for K in range(n_shells):
            if bound_shell[K] > 1e-30:
                tightness_shell[K] = actual_shell[K] / bound_shell[K]

        # Global tightness
        total_actual = np.sum(actual_shell)
        total_bound = np.sum(bound_shell)
        total_psol = np.sum(psol_shell)

        if verbose:
            print(f"\n{'='*70}")
            print(f"MODE-RESOLVED TIGHTNESS (PHASE CANCELLATION MEASUREMENT)")
            print(f"{'='*70}")
            print(f"\n  ||L_cross|| actual  = {np.sqrt(total_actual/N**3):.6e}")
            print(f"  ||L_cross|| no-phase = {np.sqrt(total_bound/N**3):.6e}")
            print(f"  ||P_sol(L_cross)||   = {np.sqrt(total_psol/N**3):.6e}")
            print(f"\n  Phase cancellation factor (actual/no-phase): {total_actual/max(total_bound,1e-30):.6f}")
            print(f"  Leray factor (P_sol/actual): {total_psol/max(total_actual,1e-30):.6f}")
            print(f"  Combined tightening: {total_psol/max(total_bound,1e-30):.6f}")
            print(f"\n  Expected for random phases: ~1/sqrt(N_modes) ~ {1/np.sqrt(N**3):.6f}")

            print(f"\n  {'Shell':>5} {'Actual':>12} {'Bound':>12} {'P_sol':>12} {'Tightness':>10} {'n_modes':>8}")
            print(f"  {'-'*60}")
            for K in range(min(n_shells, 15)):
                if bound_shell[K] > 1e-30:
                    tight = actual_shell[K] / bound_shell[K]
                    psol_tight = psol_shell[K] / max(bound_shell[K], 1e-30)
                    print(f"  {K:5d} {actual_shell[K]:12.4e} {bound_shell[K]:12.4e} "
                          f"{psol_shell[K]:12.4e} {tight:10.6f} {n_modes_shell[K]:8.0f}")

        return {
            'total_actual': total_actual,
            'total_bound': total_bound,
            'total_psol': total_psol,
            'phase_factor': total_actual / max(total_bound, 1e-30),
            'leray_factor': total_psol / max(total_actual, 1e-30),
            'combined_factor': total_psol / max(total_bound, 1e-30),
            'actual_shell': actual_shell,
            'bound_shell': bound_shell,
            'psol_shell': psol_shell,
            'tightness_shell': tightness_shell,
            'n_modes_shell': n_modes_shell,
        }


def main():
    print("=" * 70)
    print("APPROACH A — TRIADIC CAUCHY-SCHWARZ TIGHTENING")
    print("Task: Meridian 2 => Meridian 1 (S92)")
    print("=" * 70)
    print()
    print("Goal: Close the 600x gap between global C-S and actual ||P_sol(L_cross)||")
    print("Method: Triad-by-triad Bessel inequality + phase cancellation measurement")
    print()

    solver = TriadicBoundAnalyzer(N=32, Re=400)

    # ================================================================
    # TEST 1: Taylor-Green (achiral, worst case)
    # ================================================================
    print("\n" + "#" * 70)
    print("# TEST 1: Taylor-Green at t=0 (achiral)")
    print("#" * 70)
    u_tg = solver.taylor_green_ic()
    r1a = solver.compute_triadic_bound_direct(u_tg)
    r1b = solver.compute_refined_triadic_bound(u_tg)
    r1c = solver.compute_mode_resolved_tightness(u_tg)

    # ================================================================
    # TEST 2: Chiral IC (ABC + perturbation)
    # ================================================================
    print("\n\n" + "#" * 70)
    print("# TEST 2: Chiral IC (ABC + 10% TG perturbation)")
    print("#" * 70)
    u_abc = solver.abc_flow_ic()
    u_perturb = solver.taylor_green_ic()
    E_abc = solver.compute_energy(u_abc)
    E_p = solver.compute_energy(u_perturb)
    scale = 0.1 * np.sqrt(E_abc / max(E_p, 1e-30))
    u_chiral = solver.project_leray(u_abc + scale * u_perturb)

    H = solver.compute_helicity(u_chiral)
    E = solver.compute_energy(u_chiral)
    print(f"  H = {H:.4f}, E = {E:.4f}, H/E = {H/E:.4f}")

    r2a = solver.compute_triadic_bound_direct(u_chiral)
    r2b = solver.compute_refined_triadic_bound(u_chiral)
    r2c = solver.compute_mode_resolved_tightness(u_chiral)

    # ================================================================
    # TEST 3: Pelz IC (complex achiral)
    # ================================================================
    print("\n\n" + "#" * 70)
    print("# TEST 3: Pelz IC (complex achiral)")
    print("#" * 70)
    u_pelz = solver.pelz_ic()
    r3a = solver.compute_triadic_bound_direct(u_pelz)
    r3b = solver.compute_refined_triadic_bound(u_pelz)
    r3c = solver.compute_mode_resolved_tightness(u_pelz)

    # ================================================================
    # TEST 4: Evolved Taylor-Green (t=0.5, more complex spectrum)
    # ================================================================
    print("\n\n" + "#" * 70)
    print("# TEST 4: Evolved Taylor-Green (t=0.5, Re=400)")
    print("#" * 70)
    u_evolved = u_tg.copy()
    for step in range(100):
        u_evolved = solver.step_rk4(u_evolved, 0.005, mode='full')
    print(f"  After 100 steps at dt=0.005 (t=0.5):")
    E_ev = solver.compute_energy(u_evolved)
    Z_ev = solver.compute_enstrophy(u_evolved)
    print(f"  E = {E_ev:.6f}, Z = {Z_ev:.6f}")

    r4a = solver.compute_triadic_bound_direct(u_evolved)
    r4b = solver.compute_refined_triadic_bound(u_evolved)
    r4c = solver.compute_mode_resolved_tightness(u_evolved)

    # ================================================================
    # TEST 5: Higher resolution (N=64) if feasible
    # ================================================================
    print("\n\n" + "#" * 70)
    print("# TEST 5: Higher resolution (N=64, TG)")
    print("#" * 70)
    try:
        solver64 = TriadicBoundAnalyzer(N=64, Re=400)
        u_tg64 = solver64.taylor_green_ic()
        r5a = solver64.compute_triadic_bound_direct(u_tg64)
        r5c = solver64.compute_mode_resolved_tightness(u_tg64)
    except MemoryError:
        print("  N=64 too large for available memory. Skipping.")
        r5a = r5c = None

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n\n" + "=" * 70)
    print("TRIADIC BOUND SUMMARY")
    print("=" * 70)

    print(f"\n{'IC':<25} {'Global C-S':>12} {'Triadic':>12} {'Phase':>12} {'Combined':>12}")
    print(f"{'':25} {'tightness':>12} {'tightness':>12} {'cancel':>12} {'tightness':>12}")
    print(f"{'-'*73}")

    results = [
        ("TG t=0 (N=32)", r1a, r1c),
        ("Chiral (N=32)", r2a, r2c),
        ("Pelz (N=32)", r3a, r3c),
        ("TG evolved t=0.5", r4a, r4c),
    ]
    if r5a is not None and r5c is not None:
        results.append(("TG t=0 (N=64)", r5a, r5c))

    for label, ra, rc in results:
        print(f"  {label:<23} {ra['tightness_global']:12.6f} {ra['tightness_triadic']:12.6f} "
              f"{rc['phase_factor']:12.6f} {rc['combined_factor']:12.6f}")

    print(f"\n  KEY METRICS:")
    print(f"    Triadic vs Global improvement: {r1a['improvement']:.1f}x (TG), "
          f"{r2a['improvement']:.1f}x (Chiral), {r3a['improvement']:.1f}x (Pelz)")
    print(f"    Phase cancellation factor: {r1c['phase_factor']:.4f} (TG), "
          f"{r2c['phase_factor']:.4f} (Chiral), {r3c['phase_factor']:.4f} (Pelz)")
    print(f"    Combined (triadic + phase + Leray): {r1c['combined_factor']:.6f} (TG), "
          f"{r2c['combined_factor']:.6f} (Chiral), {r3c['combined_factor']:.6f} (Pelz)")

    # Verdict
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    best_tightness = max(r1a['tightness_triadic'], r2a['tightness_triadic'],
                         r3a['tightness_triadic'], r4a['tightness_triadic'])
    best_combined = max(r1c['combined_factor'], r2c['combined_factor'],
                        r3c['combined_factor'], r4c['combined_factor'])

    if best_tightness >= 0.1:
        print(f"  STRONG: Triadic bound achieves tightness >= 0.1 ({best_tightness:.4f})")
        print(f"  The Bessel inequality closes the gap sufficiently.")
    elif best_tightness >= 0.01:
        print(f"  MODERATE: Triadic bound achieves tightness in [0.01, 0.1] ({best_tightness:.4f})")
        print(f"  10x+ improvement but still not enough to close the chain.")
    else:
        print(f"  WEAK: Triadic bound tightness < 0.01 ({best_tightness:.4f})")
        print(f"  Shell averaging loses too much information.")

    if best_combined >= 0.1:
        print(f"\n  Phase cancellation helps significantly: combined = {best_combined:.4f}")
        print(f"  Phase cancellation is the REAL tightening mechanism.")
    elif best_combined >= 0.01:
        print(f"\n  Phase cancellation gives moderate help: combined = {best_combined:.4f}")
    else:
        print(f"\n  Phase cancellation insufficient: combined = {best_combined:.6f}")

    print(f"\n  BOTTOM LINE:")
    print(f"    Global C-S gap: ~600x loose")
    print(f"    Best triadic improvement: {1/max(best_tightness, 1e-30):.0f}x loose")
    print(f"    Best combined: {1/max(best_combined, 1e-30):.0f}x loose")

    if best_combined >= 0.1:
        print(f"    => Triadic + phase approach VIABLE — gap reduced to <10x")
        print(f"    => Next step: prove phase cancellation holds for all solenoidal fields")
    elif best_combined >= 0.01:
        print(f"    => Partial success — gap reduced but still O(100x)")
        print(f"    => Need additional structure (e.g., Beltrami decomposition, BT surgery)")
    else:
        print(f"    => Triadic approach INSUFFICIENT — try Approach B (different bound strategy)")


if __name__ == "__main__":
    main()
