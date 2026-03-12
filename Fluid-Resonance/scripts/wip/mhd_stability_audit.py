import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

class SpectralMHD:
    def __init__(self, N=32, Re=400, Rm=400):
        self.N = N
        self.nu = 1.0 / Re
        self.eta = 1.0 / Rm
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
        self.dealias_mask = (np.abs(self.kx) <= kmax) & (np.abs(self.ky) <= kmax) & (np.abs(self.kz) <= kmax)

    def project_leray(self, f_hat):
        result = np.zeros_like(f_hat)
        for i in range(3):
            for j in range(3):
                result[i] += self.P[(i, j)] * f_hat[j]
        return result

    def curl_hat(self, f_hat):
        return np.array([
            1j*(self.ky*f_hat[2] - self.kz*f_hat[1]),
            1j*(self.kz*f_hat[0] - self.kx*f_hat[2]),
            1j*(self.kx*f_hat[1] - self.ky*f_hat[0]),
        ])

    def cross_product_physical(self, a_hat, b_hat):
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
        u_hat, B_hat = state
        omega_hat = self.curl_hat(u_hat)
        J_hat = self.curl_hat(B_hat)
        lamb_hat = self.cross_product_physical(u_hat, omega_hat)
        lorentz_hat = self.cross_product_physical(J_hat, B_hat)
        nonlinear_hat = lamb_hat + lorentz_hat
        rhs_u = self.project_leray(nonlinear_hat)
        rhs_u -= self.nu * self.k2[np.newaxis] * u_hat
        rhs_u[:, 0, 0, 0] = 0.0
        uxB_hat = self.cross_product_physical(u_hat, B_hat)
        curl_uxB = self.curl_hat(uxB_hat)
        rhs_B = self.project_leray(curl_uxB)
        rhs_B -= self.eta * self.k2[np.newaxis] * B_hat
        rhs_B[:, 0, 0, 0] = 0.0
        return (rhs_u, rhs_B)

    def step_rk4(self, state, dt):
        def add_states(s, ds, c):
            return (s[0] + c*ds[0], s[1] + c*ds[1])
        k1 = self.compute_rhs(state)
        k2 = self.compute_rhs(add_states(state, k1, 0.5*dt))
        k3 = self.compute_rhs(add_states(state, k2, 0.5*dt))
        k4 = self.compute_rhs(add_states(state, k3, dt))
        u_new = state[0] + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        B_new = state[1] + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        return (u_new, B_new)

    def solenoidal_b_ic(self, B0_strength=1.0):
        u = np.zeros((3,) + self.X.shape)
        u[0] = np.sin(self.X) * np.cos(self.Y) * np.cos(self.Z)
        u[1] = -np.cos(self.X) * np.sin(self.Y) * np.cos(self.Z)
        u_hat = self.project_leray(np.array([fftn(u[i]) for i in range(3)]))
        B = np.zeros((3,) + self.X.shape)
        B[0] = B0_strength * np.cos(self.Z)
        B_hat = self.project_leray(np.array([fftn(B[i]) for i in range(3)]))
        return u_hat, B_hat

    def ot_mhd_ic(self):
        u = np.zeros((3,) + self.X.shape)
        u[0] = -np.sin(self.Y)
        u[1] = np.sin(self.X)
        u_hat = self.project_leray(np.array([fftn(u[i]) for i in range(3)]))
        B = np.zeros((3,) + self.X.shape)
        B[0] = -np.sin(self.Y)
        B[1] = np.sin(2*self.X)
        B_hat = self.project_leray(np.array([fftn(B[i]) for i in range(3)]))
        return u_hat, B_hat

    def get_energy(self, state):
        u_hat, B_hat = state
        u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
        B = np.array([np.real(ifftn(B_hat[i])) for i in range(3)])
        E_kin = 0.5 * np.mean(np.sum(u**2, axis=0))
        E_mag = 0.5 * np.mean(np.sum(B**2, axis=0))
        return E_kin, E_mag

def run_stability_audit():
    print("--- MHD STABILITY AUDIT (S63) ---")
    N = 32
    Re = 400
    Rm = 400
    T_max = 4.0
    dt_values = [0.005, 0.002, 0.001]
    
    # Test Orszag-Tang IC
    print("\n--- Testing Orszag-Tang IC (Benchmark) ---")
    for dt in dt_values:
        print(f"\nTesting dt = {dt}")
        solver = SpectralMHD(N=N, Re=Re, Rm=Rm)
        state = solver.ot_mhd_ic()
        E0_k, E0_m = solver.get_energy(state)
        E0 = E0_k + E0_m
        print(f"t=0.00 | E_tot = {E0:.6e}")
        
        stable = True
        steps = int(T_max / dt)
        for step in range(1, steps + 1):
            state = solver.step_rk4(state, dt)
            if step % int(0.5/dt) == 0:
                Ek, Em = solver.get_energy(state)
                Etot = Ek + Em
                print(f"t={step*dt:.3f} | E_tot = {Etot:.6e} | Z = {Ek*Re:.2f}")
                if Etot > E0 * 1.5:
                    print(f"UNSTABLE at t={step*dt:.3f}. Energy explosion.")
                    stable = False
                    break
        if stable:
            print("STABLE for this duration.")

if __name__ == "__main__":
    run_stability_audit()
