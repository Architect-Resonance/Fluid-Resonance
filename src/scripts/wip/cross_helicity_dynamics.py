import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import time as clock

class HelicalDecompositionNS:
    """Spectral NS with helical triadic decomposition of stretching."""

    def __init__(self, N=32, Re=400):
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

        # Leray projector
        self.P = {}
        K = [self.kx, self.ky, self.kz]
        for i in range(3):
            for j in range(3):
                self.P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / self.k2_safe

        # Dealiasing mask
        kmax = N // 3
        self.dealias_mask = (
            (np.abs(self.kx) <= kmax) &
            (np.abs(self.ky) <= kmax) &
            (np.abs(self.kz) <= kmax)
        )

        self._build_helical_basis()

    def _build_helical_basis(self):
        """Craya-Herring helical basis."""
        kmag = np.sqrt(self.k2_safe)
        khat = np.array([self.kx / kmag, self.ky / kmag, self.kz / kmag])

        # e1 = (0,0,1) x khat
        e1 = np.array([-khat[1], khat[0], np.zeros_like(khat[0])])
        e1_mag = np.sqrt(np.sum(e1**2, axis=0))

        # Fallback for k || z
        parallel = e1_mag < 1e-10
        if np.any(parallel):
            e1_alt = np.array([np.zeros_like(khat[0]), -khat[2], khat[1]])
            for i in range(3):
                e1[i] = np.where(parallel, e1_alt[i], e1[i])
            e1_mag = np.sqrt(np.sum(e1**2, axis=0))

        e1 /= np.maximum(e1_mag, 1e-15)

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

    def compute_vorticity_hat(self, u_hat):
        return np.array([
            1j * (self.ky * u_hat[2] - self.kz * u_hat[1]),
            1j * (self.kz * u_hat[0] - self.kx * u_hat[2]),
            1j * (self.kx * u_hat[1] - self.ky * u_hat[0]),
        ])

    def decompose_helical(self, f_hat):
        """Project field into h+ and h- components."""
        f_p_amp = np.sum(np.conj(self.h_plus) * f_hat, axis=0)
        f_m_amp = np.sum(np.conj(self.h_minus) * f_hat, axis=0)
        f_plus = f_p_amp[np.newaxis] * self.h_plus
        f_minus = f_m_amp[np.newaxis] * self.h_minus
        return f_plus, f_minus

    def compute_rhs(self, u_hat):
        omega_hat = self.compute_vorticity_hat(u_hat)
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        omega = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        
        lamb = np.zeros_like(u)
        lamb[0] = u[1]*omega[2] - u[2]*omega[1]
        lamb[1] = u[2]*omega[0] - u[0]*omega[2]
        lamb[2] = u[0]*omega[1] - u[1]*omega[0]

        lamb_hat = np.array([fftn(lamb[i]) for i in range(3)])
        for i in range(3):
            lamb_hat[i] *= self.dealias_mask
            
        rhs = np.zeros_like(u_hat)
        for i in range(3):
            for j in range(3):
                rhs[i] += self.P[(i, j)] * lamb_hat[j]
        
        rhs -= self.nu * self.k2[np.newaxis] * u_hat
        rhs[:, 0, 0, 0] = 0.0
        return rhs

    def step_rk4(self, u_hat, dt):
        k1 = self.compute_rhs(u_hat)
        k2 = self.compute_rhs(u_hat + 0.5*dt*k1)
        k3 = self.compute_rhs(u_hat + 0.5*dt*k2)
        k4 = self.compute_rhs(u_hat + dt*k3)
        return u_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _get_frob(self, field_hat):
        """Compute frog sqrt of S_ij S_ij in physical space."""
        K = [self.kx, self.ky, self.kz]
        S_frob_sq = np.zeros(self.X.shape)
        # S_ij = 0.5 * (dui/dxj + duj/dxi)
        # We need the full grad u for the Frob norm of strain
        u_p = np.array([np.real(ifftn(field_hat[i])) for i in range(3)])
        for i in range(3):
            for j in range(3):
                du_hat = 1j * K[j] * field_hat[i]
                S_ij = 0.5 * (np.real(ifftn(du_hat)) + np.real(ifftn(1j * K[i] * field_hat[j])))
                S_frob_sq += S_ij**2
        return np.sqrt(np.max(S_frob_sq))

    def analyze_stretching(self, u_hat):
        """Decompose stretching into same-helicity and cross-helicity."""
        omega_hat = self.compute_vorticity_hat(u_hat)
        Z_full = np.mean(np.sum(np.real(ifftn(omega_hat))**2, axis=0))
        lambda_max_full = self._get_frob(u_hat)

        up_hat, um_hat = self.decompose_helical(u_hat)
        wp_hat, wm_hat = self.decompose_helical(omega_hat)
        
        # Physical fields
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        up = np.array([np.real(ifftn(up_hat[i])) for i in range(3)])
        um = np.array([np.real(ifftn(um_hat[i])) for i in range(3)])
        
        w = np.array([np.real(ifftn(omega_hat[i])) for i in range(3)])
        wp = np.array([np.real(ifftn(wp_hat[i])) for i in range(3)])
        wm = np.array([np.real(ifftn(wm_hat[i])) for i in range(3)])
        
        # Stretching function
        def get_stretch(u_field, w_field):
            # Grad u_field
            S = np.zeros((3, 3) + u_field.shape[1:])
            K = [self.kx, self.ky, self.kz]
            u_hat_f = np.array([fftn(u_field[i]) for i in range(3)])
            for i in range(3):
                for j in range(3):
                    du_hat = 1j * K[j] * u_hat_f[i]
                    # Sy metrical strain
                    S[i, j] = 0.5 * (np.real(ifftn(du_hat)) + np.real(ifftn(1j * K[i] * u_hat_f[j])))
            
            # w . S . w
            res = np.zeros_like(w_field[0])
            for i in range(3):
                for j in range(3):
                    res += w_field[i] * S[i, j] * w_field[j]
            return np.mean(res)

        # Total stretching
        S_total = get_stretch(u, w)
        C_total = abs(S_total) / (lambda_max_full * Z_full + 1e-15)
        
        # Helical channels
        S_pp = get_stretch(up, wp)
        S_mm = get_stretch(um, wm)
        S_same = S_pp + S_mm
        
        S_cross = S_total - S_same
        # For C_cross, we look at how efficiently the cross-interaction uses total resource
        C_cross = abs(S_cross) / (lambda_max_full * Z_full + 1e-15)
        
        return {
            'total': S_total,
            'C_total': C_total,
            'same': S_same,
            'cross': S_cross,
            'C_cross': C_cross,
            'fraction_cross': abs(S_cross) / (abs(S_same) + abs(S_cross) + 1e-15)
        }

def run_investigation(N_val=32, T_val=2.0, log_data=False):
    Re = 400
    dt = 0.01
    
    solver = HelicalDecompositionNS(N=N_val, Re=Re)
    
    # Taylor-Green IC
    x = solver.X
    u = np.zeros((3,) + x.shape)
    u[0] = np.sin(x) * np.cos(solver.Y) * np.cos(solver.Z)
    u[1] = -np.cos(x) * np.sin(solver.Y) * np.cos(solver.Z)
    u_hat = np.array([fftn(u[i]) for i in range(3)])
    u_hat[:, 0, 0, 0] = 0.0

    print(f"\n--- Resolution: N={N_val}, Re={Re} ---")
    print(f"{'t':<5} | {'Z':<10} | {'C_total':<8} | {'C_cross':<8} | {'Cross%':<6}")
    print("-" * 55)

    t = 0.0
    step = 0
    results = []
    while t <= T_val:
        if step % 20 == 0:
            stats = solver.analyze_stretching(u_hat)
            omega_hat = solver.compute_vorticity_hat(u_hat)
            Z = np.mean(np.sum(np.real(ifftn(omega_hat))**2, axis=0))
            print(f"{t:<5.2f} | {Z:<10.4f} | {stats['C_total']:<8.4f} | "
                  f"{stats['C_cross']:<8.4f} | {stats['fraction_cross']*100:<6.1f}")
            
            if log_data:
                results.append({
                    't': t, 'Z': float(Z), 
                    'C_total': float(stats['C_total']), 
                    'C_cross': float(stats['C_cross']),
                    'CrossFrac': float(stats['fraction_cross'])
                })
            
        u_hat = solver.step_rk4(u_hat, dt)
        t += dt
        step += 1
    
    return results

if __name__ == "__main__":
    import json
    sweep_results = {}
    for N in [32, 48, 64]:
        sweep_results[N] = run_investigation(N_val=N, log_data=True)
    
    with open("cross_helicity_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    print("\nSweep complete. Results saved to cross_helicity_sweep.json")
