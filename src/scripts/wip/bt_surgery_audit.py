import numpy as np
from spectral_ns_solver import SpectralNSSolver

def run_bt_surgery_audit():
    print(f"--- S47: BIFERALE-TITI (BT) SURGERY AUDIT ---")
    print(f"{'Mode':<15} | {'Enstrophy Z':<15} | {'Stretching C'}")
    print("-" * 55)
    
    N = 32
    solver = SpectralNSSolver(N=N, nu=0.01)
    
    # 1. Natural Mixed Helicity
    solver.initialize_random_field(energy_scale=1.0)
    u, omega = solver.get_real_fields()
    
    def get_C(solver_inst):
        u_local, w_local = solver_inst.get_real_fields()
        grad_u_hat = np.zeros((3, 3, N, N, N), dtype=complex)
        for i, ki in enumerate([solver_inst.kx, solver_inst.ky, solver_inst.kz]):
            for j in range(3):
                grad_u_hat[i, j] = 1j * ki * solver_inst.u_hat[j]
        grad_u = np.real(np.fft.ifftn(grad_u_hat, axes=(2,3,4)))
        stretching_field = np.zeros((N, N, N))
        for i in range(3):
            for j in range(3):
                stretching_field += w_local[i] * grad_u[i, j] * w_local[j]
        max_stretch = np.max(np.abs(stretching_field))
        avg_z = np.mean(np.sum(w_local**2, axis=0))
        return max_stretch / (avg_z + 1e-9), avg_z

    C_mixed, Z_mixed = get_C(solver)
    print(f"{'Mixed (Natural)':<15} | {Z_mixed:<15.4f} | {C_mixed:.6f}")
    
    # 2. Helical Surgery (Keep ONLY Minus)
    solver.zero_helical_mode(sign='+')
    C_minus, Z_minus = get_C(solver)
    print(f"{'Single (Minus)':<15} | {Z_minus:<15.4f} | {C_minus:.6f}")
    
    # 3. Helical Surgery (Keep ONLY Plus)
    solver.initialize_random_field(energy_scale=1.0) # Reset to random
    solver.zero_helical_mode(sign='-')
    C_plus, Z_plus = get_C(solver)
    print(f"{'Single (Plus)':<15} | {Z_plus:<15.4f} | {C_plus:.6f}")

    print("-" * 55)
    print(f"BT RATIO (Single/Mixed): {C_minus/C_mixed:.4f}")
    print("INSIGHT: If BT Ratio < 1, helical interactions drive the nonlinearity.")

if __name__ == "__main__":
    run_bt_surgery_audit()
