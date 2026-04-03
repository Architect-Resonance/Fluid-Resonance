import numpy as np
from spectral_ns_solver import SpectralNSSolver

def run_convergence_audit():
    print(f"--- S47: CONTINUUM CONVERGENCE AUDIT ---")
    print(f"{'Resolution N':<15} | {'Enstrophy Z':<15} | {'Max Stretching C'}")
    print("-" * 55)
    
    # 1. Create a baseline field at N=32
    solver32 = SpectralNSSolver(N=32, nu=0.01)
    u_hat_32 = solver32.initialize_random_field(energy_scale=1.0)
    
    def get_C(solver_inst, u_hat):
        N = solver_inst.N
        # Compute real fields
        u = np.real(np.fft.ifftn(u_hat, axes=(1,2,3)))
        w_hat = np.zeros_like(u_hat)
        w_hat[0] = 1j * (solver_inst.ky*u_hat[2] - solver_inst.kz*u_hat[1])
        w_hat[1] = 1j * (solver_inst.kz*u_hat[0] - solver_inst.kx*u_hat[2])
        w_hat[2] = 1j * (solver_inst.kx*u_hat[1] - solver_inst.ky*u_hat[0])
        omega = np.real(np.fft.ifftn(w_hat, axes=(1,2,3)))
        
        grad_u_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i, ki in enumerate([solver_inst.kx, solver_inst.ky, solver_inst.kz]):
            for j in range(3):
                grad_u_hat[i, j] = 1j * ki * u_hat[j]
        grad_u = np.real(np.fft.ifftn(grad_u_hat, axes=(2,3,4)))
        
        stretching_field = np.zeros((N, N, N))
        for i in range(3):
            for j in range(3):
                stretching_field += omega[i] * grad_u[i, j] * omega[j]
        
        max_stretch = np.max(np.abs(stretching_field))
        avg_z = np.mean(np.sum(omega**2, axis=0))
        return max_stretch / (avg_z + 1e-9), avg_z

    C32, Z32 = get_C(solver32, u_hat_32)
    print(f"{32:<15} | {Z32:<15.4f} | {C32:.6f}")
    
    # 2. Interpolate to N=64
    def interpolate(u_hat_old, N_new):
        # We zero-pad the high frequencies to increase resolution
        N_old = u_hat_old.shape[1]
        u_hat_new = np.zeros((3, N_new, N_new, N_new), dtype=complex)
        mid = N_old // 2
        # Use simple slice-and-dice for periodic Fourier padding
        # Top-left, Top-right, Bot-left, Bot-right in 3D... 
        # Actually easier to just map u_hat_old frequencies to u_hat_new
        u_hat_new[:, :mid, :mid, :mid] = u_hat_old[:, :mid, :mid, :mid]
        u_hat_new[:, -mid:, :mid, :mid] = u_hat_old[:, -mid:, :mid, :mid]
        u_hat_new[:, :mid, -mid:, :mid] = u_hat_old[:, :mid, -mid:, :mid]
        u_hat_new[:, :mid, :mid, -mid:] = u_hat_old[:, :mid, :mid, -mid:]
        u_hat_new[:, -mid:, -mid:, :mid] = u_hat_old[:, -mid:, -mid:, :mid]
        u_hat_new[:, -mid:, :mid, -mid:] = u_hat_old[:, -mid:, :mid, -mid:]
        u_hat_new[:, :mid, -mid:, -mid:] = u_hat_old[:, :mid, -mid:, -mid:]
        u_hat_new[:, -mid:, -mid:, -mid:] = u_hat_old[:, -mid:, -mid:, -mid:]
        
        # Power scaling (FFT normalization)
        return u_hat_new * (N_new/N_old)**3

    for N_next in [64, 128]:
        solver_next = SpectralNSSolver(N=N_next, nu=0.01)
        u_hat_next = interpolate(u_hat_32, N_next)
        C_next, Z_next = get_C(solver_next, u_hat_next)
        print(f"{N_next:<15} | {Z_next:<15.4f} | {C_next:.6f}")

    print("-" * 55)
    print("INSIGHT: If C converges as N increases, the metric is physical.")
    print("If C continues to drop, the field is smoother than the grid limit.")

if __name__ == "__main__":
    run_convergence_audit()
