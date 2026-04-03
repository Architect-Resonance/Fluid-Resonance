import numpy as np
from scipy.linalg import eigh

def compute_physical_fields(nodes, omegas, sigma=0.05):
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

def compute_pressure_hessian(nodes, Strains, omegas, sigma=0.1):
    N = len(nodes)
    Pi = np.zeros((N, 3, 3))
    Q = np.zeros(N)
    for j in range(N):
        S_sym = (Strains[j] + Strains[j].T) / 2.0
        # Q = Tr(S^2) - 0.5|omega|^2
        Q[j] = np.trace(S_sym @ S_sym) - 0.5 * np.linalg.norm(omegas[j])**2

    for i in range(N):
        Pi_local = np.zeros((3, 3))
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            for a in range(3):
                for b in range(3):
                    kernel = (3.0 * r_vec[a] * r_vec[b] / (r**5)) - ((1.0 if a==b else 0.0) / (r**3))
                    Pi_local[a, b] += (1.0 / (4.0 * np.pi)) * Q[j] * kernel
        Pi[i] = Pi_local
    return Pi

def run_global_coordinator_audit():
    print(f"--- S38: GLOBAL COORDINATOR AUDIT (Eigenvalue Derivatives) ---")
    print(f"{'Structure':<20} | {'Max Lambda':<12} | {'d(Lambda)/dt':<15} | {'P-Effect'}")
    print("-" * 75)
    
    def audit_case(name, nodes, omegas):
        Strains = compute_physical_fields(nodes, omegas)
        Pi = compute_pressure_hessian(nodes, Strains, omegas)
        
        avg_lambda = 0
        avg_d_lambda = 0
        avg_p_effect = 0
        
        for i in range(len(nodes)):
            # Symmetric Strain S
            S = (Strains[i] + Strains[i].T) / 2.0
            evals, evecs = eigh(S)
            l_max = evals[-1]
            v_max = evecs[:, -1]
            
            # Restricted Euler growth: -(S^2)_max + 1/3 Tr(S^2)
            # Actually, the eqn is: dS/dt = -S^2 - Pi + (vorticity terms)
            # The growth of the max eigenvalue l_max is approx:
            # d(l_max)/dt = v_max^T . dS/dt . v_max
            #             = - v_max^T . S^2 . v_max - v_max^T . Pi . v_max 
            #             = - l_max^2 - (v_max^T . Pi . v_max)
            
            # We look at the TRACELESS part to center it:
            # q = Tr(S^2)
            # dS/dt_traceless = -(S^2 - 1/3 q I) - (Pi - 1/3 Tr(Pi) I)
            
            q = np.trace(S @ S)
            tr_pi = np.trace(Pi[i])
            
            s_term = -(l_max**2 - (1.0/3.0)*q)
            p_effect = -(np.dot(v_max, Pi[i] @ v_max) - (1.0/3.0)*tr_pi)
            
            d_lambda = s_term + p_effect
            
            avg_lambda += l_max
            avg_d_lambda += d_lambda
            avg_p_effect += p_effect
            
        N = len(nodes)
        print(f"{name:<20} | {avg_lambda/N:<12.4f} | {avg_d_lambda/N:<15.4f} | {avg_p_effect/N:.4f}")

    # Case 1: Tilted Collision
    n1 = np.array([[0,0,0], [0.05, 0.05, 0]], dtype=float)
    o1 = np.array([[1,1,0], [-1,1,0]], dtype=float) * 50.0
    audit_case("Tilted Collision", n1, o1)
    
    # Case 2: Parallel Pair
    n2 = np.array([[0,0,0], [0, 0.1, 0]], dtype=float)
    o2 = np.array([[0,1,0], [0,1,0]], dtype=float) * 50.0
    audit_case("Parallel Pair", n2, o2)

    # Case 3: Random Cluster 
    np.random.seed(42)
    n3 = np.random.uniform(-0.1, 0.1, (10, 3))
    o3 = np.random.normal(0, 1, (10, 3)) * 20.0
    audit_case("Random Cluster", n3, o3)

if __name__ == "__main__":
    run_global_coordinator_audit()
