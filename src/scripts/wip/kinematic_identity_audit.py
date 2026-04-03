import numpy as np
from spectral_ns_solver import SpectralNSSolver

def audit_kinematic_identity():
    print(f"--- S49: KINEMATIC IDENTITY AUDIT ---")
    N = 32
    solver = SpectralNSSolver(N=N)
    
    def get_C(u_hat):
        u = np.real(np.fft.ifftn(u_hat, axes=(1,2,3)))
        w_hat = np.zeros_like(u_hat)
        w_hat[0] = 1j * (solver.ky*u_hat[2] - solver.kz*u_hat[1])
        w_hat[1] = 1j * (solver.kz*u_hat[0] - solver.kx*u_hat[2])
        w_hat[2] = 1j * (solver.kx*u_hat[1] - solver.ky*u_hat[0])
        omega = np.real(np.fft.ifftn(w_hat, axes=(1,2,3)))
        
        grad_u_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i, ki in enumerate([solver.kx, solver.ky, solver.kz]):
            for j in range(3):
                grad_u_hat[i, j] = 1j * ki * u_hat[j]
        grad_u = np.real(np.fft.ifftn(grad_u_hat, axes=(2,3,4)))
        
        stretching_field = np.zeros((N, N, N))
        for i in range(3):
            for j in range(3):
                stretching_field += omega[i] * grad_u[i, j] * omega[j]
        
        avg_stretch_norm = np.mean(np.abs(stretching_field))
        avg_z = np.mean(np.sum(omega**2, axis=0))
        return avg_stretch_norm / (avg_z + 1e-9)

    # Test 100 random Gaussian fields
    ratios = []
    for _ in range(100):
        u_hat_mixed = solver.initialize_random_field(energy_scale=1.0)
        C_mixed = get_C(u_hat_mixed)
        
        solver.zero_helical_mode(sign='+')
        C_minus = get_C(solver.u_hat)
        
        ratios.append(C_minus / C_mixed)
    
    avg_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    print(f"Average Ratio (Single/Mixed) over 100 trials: {avg_ratio:.4f} +/- {std_ratio:.4f}")
    
    if np.isclose(avg_ratio, 0.5, atol=0.05):
        print("RESULT: The 50% reduction is a KINEMATIC IDENTITY for Gaussian random fields.")
        print("It does NOT prove dynamic stability of the Navier-Stokes equations.")
    else:
        print("RESULT: The 50% reduction is NOT a trivial kinematic identity.")

if __name__ == "__main__":
    audit_kinematic_identity()
