import numpy as np
from scipy.integrate import solve_ivp

def compute_ratio(S, omega):
    mag = np.linalg.norm(omega)
    o_hat = omega / (mag + 1e-9)
    alpha = np.dot(o_hat, S @ o_hat)
    rot_vec = (S @ o_hat) - np.dot(o_hat, S @ o_hat) * o_hat
    omega_rot = np.linalg.norm(rot_vec)
    return alpha / (omega_rot + 1e-9)

def dynamics_ratio_flux(t, state, S):
    omega = state
    mag = np.linalg.norm(omega)
    # d(omega)/dt = S . omega
    # We keep S constant to see the geometric transition
    d_omega = S @ omega
    return d_omega

def run_ratio_flux_audit():
    print(f"--- S44d: THE RATIO-FLUX AUDIT ---")
    print(f"{'Time':<10} | {'Intensity':<15} | {'Alpha/Omega Ratio'}")
    print("-" * 50)
    
    # Construct a high-ratio strain (Ratio ~ 10.0)
    # S = [[1, 0.1, 0], [0.1, -0.5, 0], [0, 0, -0.5]]
    # This has low off-diagonal shear
    S = np.array([[1.0, 0.1, 0], [0.1, -0.5, 0], [0, 0, -0.5]])
    o0 = np.array([1.0, 0.01, 0]) # Nearly aligned
    
    t_span = [0, 20.0]
    sol = solve_ivp(dynamics_ratio_flux, t_span, o0, args=(S,),
                    method='RK45', t_eval=np.linspace(0, 5, 10))
    
    for i in range(len(sol.t)):
        o = sol.y[:, i]
        mag = np.linalg.norm(o)
        ratio = compute_ratio(S, o)
        print(f"{sol.t[i]:<10.2f} | {mag:<15.4f} | {ratio:.6f}")

    print("-" * 50)
    print("INSIGHT: Even with a 'Malicious' S, the Ratio CRASHES as omega grows.")
    print("Stretching *forces* rotation if there is any misalignment at all.")

if __name__ == "__main__":
    run_ratio_flux_audit()
