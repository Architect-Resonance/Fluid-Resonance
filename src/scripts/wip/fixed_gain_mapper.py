import numpy as np
from scipy.integrate import solve_ivp

def fixed_gain_dynamics(t, state, nu, S_const):
    """
    Simulates the evolution of vorticity magnitude |omega| and 
    unit vector \hat{omega} in a fixed (or intensity-scaled) strain S.
    
    Fixed-Gain Theory: S = k * |omega| (Self-induced)
    """
    omega_vec = state[:3]
    mag = np.linalg.norm(omega_vec)
    o_hat = omega_vec / (mag + 1e-9)
    
    # Self-induced strain: proportional to intensity
    # S = S_const * mag
    k = 0.5
    S = S_const * k * mag
    
    # d(omega)/dt = S . omega - nu * omega
    d_omega = S @ omega_vec - nu * omega_vec
    
    return d_omega

def map_gain_pass(intensity_0, S_const, nu=0.0):
    # Initial state: aligned at 45 deg to extensional axis to ensure a "Pass"
    # Extensional axis of S_const usually e1
    o0 = np.array([1.0, 1.0, 0.0])
    o0 = (o0 / np.linalg.norm(o0)) * intensity_0
    
    t_span = [0, 1.0 / intensity_0] # Scaled time span
    sol = solve_ivp(fixed_gain_dynamics, t_span, o0, args=(nu, S_const),
                    method='RK45', rtol=1e-8)
    
    z_start = intensity_0**2
    z_end = np.linalg.norm(sol.y[:, -1])**2
    
    gain = np.log(z_end / z_start)
    return gain

def run_gain_mapper():
    print(f"--- S44: DYNAMIC GAIN-PASS MAPPER ---")
    print(f"{'Initial Z':<15} | {'Log-Gain Delta(ln Z)':<20} | {'Status'}")
    print("-" * 55)
    
    # S_const defines the "Shape" of the strain
    S_const = np.array([[1, 0, 0], [0, -0.5, 0], [0, 0, -0.5]])
    
    for z0_mag in [1, 10, 100, 1000, 10000]:
        z0 = z0_mag**2
        gain = map_gain_pass(z0_mag, S_const)
        print(f"{z0:<15.0f} | {gain:<20.6f} | {'STABLE' if np.abs(gain) < 10 else 'BLOWUP'}")

    print("-" * 55)
    print("ANALYSIS: If Log-Gain is constant, the Fixed-Gain Conjecture holds.")
    print("This means enstrophy doubling occurs at a fixed rate PER ROTATION.")

if __name__ == "__main__":
    run_gain_mapper()
