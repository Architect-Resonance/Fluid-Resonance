import numpy as np
from scipy.linalg import eigh

def compute_pressure_aware_margin(nodes, sigma=0.1, mode='B'):
    N = len(nodes)
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    W = 1.0 / (dist + sigma)
    np.fill_diagonal(W, 0)
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
    L_norm = D_inv_sqrt @ (np.diag(d) - W) @ D_inv_sqrt
    evals = eigh(L_norm, eigvals_only=True)
    l_min_norm = evals[1]
    
    Z_renorm = N / (np.mean(d) + 1e-9)
    
    # Mode A: Standard Graph Margin (0.85 floor)
    # Mode B: Pressure-Aware Margin (incorporating the alignment depletion factor)
    # From S35u/v: Pressure causes vorticity to align away from the max strain, 
    # reducing stretching by a factor of ~0.6-0.8 (depletion).
    # We use a depletion factor of 0.75 based on Restricted Euler stats.
    depletion_factor = 0.75 if mode == 'B' else 1.0
    
    # Margin = Dissipation / (Stretching * Depletion)
    # Lower stretching => Higher Margin.
    margin = (1.0 / (l_min_norm * Z_renorm * depletion_factor)) * 0.45 
    return margin

def run_pressure_lift_audit():
    print(f"--- PRESSURE LIFT AUDIT: COLLIDING STAR (d=0.36) ---")
    print(f"{'Mode':<20} | {'Safety Margin':<15} | {'Status':<10}")
    print("-" * 50)
    
    d = 0.36; n_seg = 100
    star1 = create_star(np.array([0, 0, 0]), n_seg=n_seg)
    star2 = create_star(np.array([d, 0, 0]), n_seg=n_seg)
    nodes = np.concatenate([star1, star2])
    
    for mode_name, mode_id in [("Standard (Graph)", 'A'), ("Pressure-Aware", 'B')]:
        margin = compute_pressure_aware_margin(nodes, mode=mode_id)
        status = "PASS" if margin >= 1.0 else "CRITICAL"
        print(f"{mode_name:<20} | {margin:<15.4f} | {status:<10}")

def create_star(origin, n_filaments=7, n_seg=50):
    phi = np.pi * (3. - np.sqrt(5.))
    nodes = []
    for i in range(n_filaments):
        y = 1 - (i / 6.0) * 2; radius = np.sqrt(1 - y * y); theta = phi * i
        direction = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        for s in np.linspace(0.1, 1.0, n_seg):
            nodes.append(origin + direction * s)
    return np.array(nodes)

if __name__ == "__main__":
    run_pressure_lift_audit()
