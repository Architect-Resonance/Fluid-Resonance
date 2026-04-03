import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

def compute_fields(nodes, omegas, sigma=0.05):
    N = len(nodes)
    Strains = np.zeros((N, 3, 3))
    for i in range(N):
        S_local = np.zeros((3, 3))
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            for a in range(3):
                for b in range(3):
                    val = 0
                    for c in range(3):
                        for d in range(3):
                            eps = 0
                            if (a,c,d) in [(0,1,2), (1,2,0), (2,0,1)]: eps = 1
                            elif (a,c,d) in [(0,2,1), (2,1,0), (1,0,2)]: eps = -1
                            if eps == 0: continue
                            kernel = (3.0 * r_vec[b] * r_vec[c] / (r**5)) - ((1.0 if b==c else 0.0) / (r**3))
                            val += eps * omegas[j][d] * kernel
                    S_local[a, b] += (1.0 / (4.0 * np.pi)) * val
        Strains[i] = S_local
    return Strains

def run_anti_twist_audit():
    print(f"--- S40: ANTI-TWIST & LAGRANGIAN MEMORY AUDIT ---")
    print(f"{'Z Intensity':<15} | {'Stretching':<15} | {'Anti-Twist (Curl)'}")
    print("-" * 55)
    
    # We simulate a "Twisting Collision"
    n = np.array([[0,0,0], [0.01, 0, 0], [0, 0.01, 0]], dtype=float)
    o = np.array([[0,0,1], [0,1,0], [1,0,0]], dtype=float) * 100.0
    
    # Track the "Curl" of the stretching vector: Curl(S . omega)
    # This represents the generation of new rotation that opposes the alignment
    for intensity in [1, 10, 100, 1000]:
        o_curr = o * intensity
        Strains = compute_fields(n, o_curr)
        
        # Stretching vector for segment 0: sigma_0 = S_0 . omega_0
        stretch_vecs = []
        for i in range(len(n)):
            stretch_vecs.append(Strains[i] @ o_curr[i])
            
        # "Anti-Twist" proxy: The degree to which stretching vectors create a new 
        # rotation field that is misaligned with the current vorticity.
        # We look at the curl-like interaction: CrossProduct(omega, stretch)
        total_stretching = np.sum([np.dot(o_curr[i], stretch_vecs[i]) for i in range(len(n))])
        
        # The anti-twist is the "Torque" that rotates omega away from alignment
        torques = [np.linalg.norm(np.cross(o_curr[i], stretch_vecs[i])) for i in range(len(n))]
        avg_torque = np.mean(torques)
        
        print(f"{intensity*100:<15.0f} | {total_stretching:<15.4e} | {avg_torque:.4e}")

if __name__ == "__main__":
    run_anti_twist_audit()
