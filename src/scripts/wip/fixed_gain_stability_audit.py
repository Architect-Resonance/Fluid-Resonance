import numpy as np

def compute_ratio(S, omega):
    mag = np.linalg.norm(omega)
    o_hat = omega / (mag + 1e-9)
    
    # Stretching rate alpha = <o_hat, S o_hat>
    alpha = np.dot(o_hat, S @ o_hat)
    
    # Rotation rate = |(S - o_hat o_hat^T S) o_hat|
    rot_vec = (S @ o_hat) - np.dot(o_hat, S @ o_hat) * o_hat
    omega_rot = np.linalg.norm(rot_vec)
    
    return alpha, omega_rot

def run_stability_audit():
    print(f"--- S44: GAIN-PASS STABILITY AUDIT ---")
    print(f"{'Config':<15} | {'Alpha/Omega Ratio':<20} | {'Max Gain (exp)'}")
    print("-" * 55)
    
    for i in range(5):
        # Random trace-free strain
        S = np.random.randn(3, 3)
        S = (S + S.T) / 2
        S -= np.trace(S) / 3.0 * np.eye(3)
        
        # Random vorticity
        omega = np.random.randn(3)
        
        alpha, omega_rot = compute_ratio(S, omega)
        ratio = alpha / (omega_rot + 1e-9)
        
        # Max gain estimated as pi/2 * ratio (for a 90 deg rotation)
        max_gain = np.exp(np.pi/2 * ratio)
        
        print(f"Random {i+1:<7} | {ratio:<20.6f} | {max_gain:.4f}")

    # Specific "Malicious" case: omega aligned with e1, S stretching e1
    # but with very small off-diagonal to slow down rotation
    S_malice = np.array([[1.0, 0.01, 0], [0.01, -0.5, 0], [0, 0, -0.5]])
    o_malice = np.array([1.0, 0, 0])
    a_m, w_m = compute_ratio(S_malice, o_malice)
    r_m = a_m / (w_m + 1e-9)
    print(f"{'Malicious':<15} | {r_m:<20.6f} | {np.exp(np.pi/2 * r_m):.4e}")

    print("-" * 55)
    print("INSIGHT: Malice requires Alpha/Omega -> infinity.")
    print("This means stretching but NOT rotating. But Biot-Savart ensures that")
    print("if you have stretching, you HAVE shear (off-diagonal) which drives rotation.")

if __name__ == "__main__":
    run_stability_audit()
