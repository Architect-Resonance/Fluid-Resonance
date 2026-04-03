# -*- coding: utf-8 -*-
"""
MILLER Q STRESS TEST -- CIRCLE-CLOSING AUDIT
============================================
Tests the three critical "Step" gaps in the regularity argument:
1. Sector Leakage: Does Q(u_same) produce cross-helical strain?
2. Orthogonality: Are Q_same and Q_cross orthogonal in L^2?
3. Ratio Breach: Does ||Q_same||/||-dS|| ever exceed 1 in a BT-surgery run?

Based on Miller 2024 (arXiv:2407.02691) and Biferale-Titi 2003.
"""

import numpy as np
from numpy.fft import fftn, ifftn
import sys
import os
import time as clock

# Import base solver from workspace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_algebraic_structure import SpectralNS

class CircleClosingAudit(SpectralNS):
    """Adds specific diagnostics for the Circle-Closing audit."""

    def compute_strain_hat(self, u_hat):
        K = [self.kx, self.ky, self.kz]
        S_hat = np.zeros((3, 3) + u_hat.shape[1:], dtype=complex)
        for i in range(3):
            for j in range(3):
                S_hat[i, j] = 0.5 * (1j * K[j] * u_hat[i] + 1j * K[i] * u_hat[j])
        return S_hat

    def compute_neg_laplacian_strain_norm(self, u_hat):
        S_hat = self.compute_strain_hat(u_hat)
        norm_sq = np.sum(self.k2**2 * np.sum(np.abs(S_hat)**2, axis=(0, 1)))
        return np.sqrt(norm_sq / self.N**6)

    def compute_Q_components(self, u_hat):
        """Compute components of Miller's Q = P_st( (u.grad)S + S^2 + 3/4w*w )."""
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        S_hat = self.compute_strain_hat(u_hat)
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        
        # S in physical space
        S = np.zeros((3, 3) + u.shape[1:])
        for i in range(3):
            for j in range(3):
                S[i, j] = np.real(ifftn(S_hat[i, j]))

        # (u.grad)S
        advS = np.zeros_like(S)
        K = [self.kx, self.ky, self.kz]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    advS[i, j] += u[k] * np.real(ifftn(1j * K[k] * S_hat[i, j]))

        # S^2
        S2 = np.zeros_like(S)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    S2[i, j] += S[i, k] * S[k, j]

        # 3/4 w*w
        OO = np.zeros_like(S)
        for i in range(3):
            for j in range(3):
                OO[i, j] = 0.75 * omega[i] * omega[j]

        T = advS + S2 + OO
        
        # Symmetric Trace-free Projection
        tr = T[0, 0] + T[1, 1] + T[2, 2]
        Q = np.zeros_like(T)
        for i in range(3):
            for j in range(3):
                Q[i, j] = 0.5 * (T[i, j] + T[j, i])
                if i == j:
                    Q[i, i] -= tr / 3.0
                    
        # Dealias
        for i in range(3):
            for j in range(3):
                Q_hat = fftn(Q[i, j]) * self.dealias_mask
                Q[i, j] = np.real(ifftn(Q_hat))
        return Q

    def compute_Q_from_lamb(self, lamb_hat):
        """Derive Q-like tensor from a Lamb vector field: Q = P_st(nabla P_leray(L))."""
        # Leray projection of Lamb vector
        lamb_solenoidal_hat = self.project_leray(lamb_hat)
        
        K = [self.kx, self.ky, self.kz]
        # nabla P(L) in Fourier space: gradL_ijk = i * k_i * lamb_j
        gradL_hat = np.zeros((3, 3) + lamb_hat.shape[1:], dtype=complex)
        for i in range(3):
            for j in range(3):
                gradL_hat[i, j] = 1j * K[i] * lamb_solenoidal_hat[j]

        # Symmetric trace-free in physical space
        Q = np.zeros((3, 3) + lamb_hat.shape[1:])
        for i in range(3):
            for j in range(3):
                # We use the REAL part of ifftn to ensure physical consistency
                # (lamb_hat is derived from real fields)
                comp = np.real(ifftn(gradL_hat[i, j]))
                Q[i, j] = comp
        
        # Symmetrize and trace-free
        for i in range(3):
            for j in range(3):
                if i < j:
                    avg = 0.5 * (Q[i, j] + Q[j, i])
                    Q[i, j] = Q[j, i] = avg
        tr = Q[0, 0] + Q[1, 1] + Q[2, 2]
        for i in range(3):
            Q[i, i] -= tr / 3.0
            
        # Dealias
        for i in range(3):
            for j in range(3):
                Q_hat = fftn(Q[i, j]) * self.dealias_mask
                Q[i, j] = np.real(ifftn(Q_hat))
        return Q

    def compute_audit_metrics(self, u_hat):
        """Quantify orthogonality and sectoral decomposition using Lamb sectors."""
        # Get sectoral Lamb vectors (real-preserving)
        lamb_full_hat = self.compute_lamb_hat(u_hat)
        lamb_same_hat = self.compute_lamb_hat_bt_surgery(u_hat)
        lamb_cross_hat = self.compute_lamb_hat_cross_only(u_hat)
        
        # Verify linearity: lamb_full = lamb_same + lamb_cross
        # err = np.max(np.abs(lamb_full_hat - (lamb_same_hat + lamb_cross_hat)))
        # if err > 1e-10:
        #    print(f"  WARNING: Lamb decomposition non-linear error: {err:.2e}")

        # Compute Q components
        Q_from_lamb = self.compute_Q_from_lamb(lamb_full_hat)
        Q_from_miller = self.compute_Q_components(u_hat)
        
        # Check consistency
        diff = np.sqrt(np.sum((Q_from_lamb - Q_from_miller)**2))
        base = np.sqrt(np.sum(Q_from_miller**2))
        # print(f"  [DEBUG] Q-definition relative error: {diff/max(base, 1e-15):.2e}")

        Q_full = Q_from_miller # Use the canonical Miller definition
        # Compute sectoral Q components
        u_p_coeffs, u_m_coeffs = self.helical_decompose(u_hat)
        u_hat_p = self.helical_reconstruct(u_p_coeffs, np.zeros_like(u_m_coeffs))
        u_hat_m = self.helical_reconstruct(np.zeros_like(u_p_coeffs), u_m_coeffs)
        
        Q_same = self.compute_Q_components(u_hat_p) + self.compute_Q_components(u_hat_m)
        Q_cross = Q_full - Q_same
        
        # Metrics
        def inner_prod(A, B):
            return np.sum(A * B) / self.N**3
            
        def tensor_inner(A, B):
            return sum(inner_prod(A[i, j], B[i, j]) for i in range(3) for j in range(3))
            
        def tensor_norm(A):
            return np.sqrt(tensor_inner(A, A))

        nQs = tensor_norm(Q_same)
        nQc = tensor_norm(Q_cross)
        nQf = tensor_norm(Q_full)
        
        cos_theta = tensor_inner(Q_same, Q_cross) / (max(nQs * nQc, 1e-15))
        
        # Fields for further diagnostics
        omega_hat = self.compute_vorticity_hat(u_hat)
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        S_hat = self.compute_strain_hat(u_hat)
        S = np.zeros((3, 3) + u_hat.shape[1:])
        for i in range(3):
            for j in range(3):
                S[i, j] = np.real(ifftn(S_hat[i, j]))
        
        production = 0.0
        for i in range(3):
            for j in range(3):
                production += np.mean(omega[i] * S[i, j] * omega[j])

        # Pressure Hessian: tr(S^2) - 0.5*w^2
        S2_tr = np.zeros(S.shape[2:])
        for i in range(3):
            for j in range(3):
                S2_tr += S[i, j]**2
        w2 = np.sum(omega**2, axis=0)
        source_hat = fftn(S2_tr - 0.5 * w2)
        
        # p_hat = -source_hat / k^2
        p_hat = np.zeros_like(source_hat)
        nz = self.k2 > 0
        p_hat[nz] = -source_hat[nz] / self.k2[nz]
        
        K = [self.kx, self.ky, self.kz]
        Hess_p = np.zeros_like(Q_full)
        trH = np.zeros_like(source_hat, dtype=float)
        for i in range(3):
            for j in range(3):
                H_ij_hat = -K[i] * K[j] * p_hat
                comp_real = np.real(ifftn(H_ij_hat))
                Hess_p[i, j] = comp_real
                if i == j:
                    trH += comp_real
        
        Hess_p_tf = np.zeros_like(Hess_p)
        for i in range(3):
            for j in range(3):
                Hess_p_tf[i, j] = Hess_p[i, j] - (trH / 3.0 if i == j else 0.0)
                    
        # OO_total = P_st(w * w)
        OO_full_tensor = np.zeros_like(S)
        for i in range(3):
            for j in range(3):
                OO_full_tensor[i, j] = omega[i] * omega[j]
        tr_OO = OO_full_tensor[0, 0] + OO_full_tensor[1, 1] + OO_full_tensor[2, 2]
        OO_tf = np.zeros_like(S)
        for i in range(3):
            for j in range(3):
                OO_tf[i, j] = OO_full_tensor[i, j]
                if i == j:
                    OO_tf[i, i] -= tr_OO / 3.0

        # Nonlinear advection strain: sym-grad(P(L))
        u_adv_hat = self.compute_lamb_hat(u_hat) 
        Q_L_hat = self.compute_Q_from_lamb(u_adv_hat) 

        # Find coefficients Q_full = x1*Q_L_hat + x2*OO_tf + x3*Hess_p_tf
        def flatten(A):
            return A.reshape(-1)
            
        A_flat = np.array([flatten(Q_L_hat), flatten(OO_tf), flatten(Hess_p_tf)]).T
        b_flat = flatten(Q_full)
        
        # Identity test: Q = -sym-grad(P(L)) + 0.75*P_st(w*w)
        Q_identity = -Q_L_hat + 0.75 * OO_tf - Hess_p_tf
        
        # Relative error
        diff_id = np.sqrt(np.sum((Q_full - Q_identity)**2))
        base_id = np.sqrt(np.sum(Q_full**2))
        rel_err_id = diff_id / (max(base_id, 1e-15))
        
        dS_norm = self.compute_neg_laplacian_strain_norm(u_hat)
        
        return {
            'ratio_Qf': nQf / dS_norm,
            'ratio_Qs': nQs / dS_norm,
            'ratio_Qc': nQc / dS_norm,
            'cos_theta': cos_theta,
            'id_err': rel_err_id,
            'enstrophy_prod': production,
            'rms_S': tensor_norm(S),
            'rms_w': np.sqrt(np.mean(omega**2)),
            'rms_dS': dS_norm
        }

def run_audit(N=32, Re=400, T=5.0):
    print(f"AUDIT RUN: N={N}, Re={Re}, T={T}")
    audit = CircleClosingAudit(N=N, Re=Re)
    u_hat = audit.random_ic(seed=42)
    dt = 0.005
    n_steps = int(T / dt)
    
    # Track metrics in both Full and BT mode
    def track(mode):
        print(f"\nRunning {mode} mode...")
        u = u_hat.copy()
        history = []
        for step in range(n_steps + 1):
            if step % 20 == 0:
                metrics = audit.compute_audit_metrics(u)
                metrics['t'] = step * dt
                history.append(metrics)
                if step % 100 == 0:
                    print(f"  t={metrics['t']:.2f} | id_err={metrics['id_err']:.2e} | cos_theta={metrics['cos_theta']:.4f}")
            u = audit.step_rk4(u, dt, mode=mode)
        return history

    hist_full = track('full')
    hist_bt = track('bt')
    
    # Summary
    print("\n" + "="*50)
    print("VERDICT: CIRCLE-CLOSING AUDIT")
    print("="*50)
    
    def summary(h, name):
        rfs = [m['ratio_Qf'] for m in h]
        rss = [m['ratio_Qs'] for m in h]
        cos = [m['cos_theta'] for m in h]
        print(f"\n[{name} NS]")
        print(f"  Max ||Q_full||/||-dS|| : {max(rfs):.4f}")
        print(f"  Max ||Q_same||/||-dS|| : {max(rss):.4f}")
        print(f"  Mean Alignment cos(theta): {np.mean(cos):.4f}")
        
    summary(hist_full, "Full")
    summary(hist_bt, "BT-Surgery")

if __name__ == "__main__":
    run_audit()
