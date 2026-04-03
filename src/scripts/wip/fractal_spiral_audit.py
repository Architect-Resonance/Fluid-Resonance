import numpy as np

def compute_physical_stretching(nodes, omegas, sigma=0.05):
    N = len(nodes)
    total_stretch = 0
    for i in range(N):
        local_stretch = 0
        for j in range(N):
            if i == j: continue
            r_vec = nodes[i] - nodes[j]
            r2 = np.dot(r_vec, r_vec) + sigma**2
            r = np.sqrt(r2)
            det = np.dot(np.cross(omegas[i], omegas[j]), r_vec)
            term = (3.0 / (4.0 * np.pi)) * np.dot(omegas[i], r_vec) * det / (r**5)
            local_stretch += term
        total_stretch += np.abs(local_stretch)
    return total_stretch

def create_logarithmic_spiral(n_points=100, a=0.1, b=0.2, h=0.1):
    """
    Creates a helical logarithmic spiral:
    x = a * exp(b*theta) * cos(theta)
    y = a * exp(b*theta) * sin(theta)
    z = h * theta
    """
    nodes = []
    omegas = []
    
    for theta in np.linspace(0, 4*np.pi, n_points):
        r = a * np.exp(b * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h * theta
        nodes.append(np.array([x, y, z]))
        
        # Tangent vector (vorticity direction)
        dx = a * b * np.exp(b * theta) * np.cos(theta) - r * np.sin(theta)
        dy = a * b * np.exp(b * theta) * np.sin(theta) + r * np.cos(theta)
        dz = h
        tangent = np.array([dx, dy, dz])
        omegas.append(tangent / np.linalg.norm(tangent))
        
    return np.array(nodes), np.array(omegas)

def create_multiscale_vortex_cloud(levels=3):
    """
    Creates a cloud of spirals at different scales.
    """
    all_nodes = []
    all_omegas = []
    
    for lvl in range(levels):
        scale = 1.0 / (2**lvl)
        n_p = 50 * (lvl + 1)
        # Randomize center and axis
        center = np.random.normal(0, 1, 3) * scale
        nodes, omegas = create_logarithmic_spiral(n_points=n_p, a=0.1*scale, b=0.2, h=0.1*scale)
        all_nodes.append(nodes + center)
        all_omegas.append(omegas)
        
    return np.concatenate(all_nodes), np.concatenate(all_omegas)

def run_spiral_audit():
    print(f"--- FRACTAL SPIRAL AUDIT: VORTEX BREAKDOWN ---")
    print(f"{'Structure':<25} | {'N':<10} | {'Total Stretching':<15} | {'S/N'}")
    print("-" * 65)
    
    # 1. Single Spiral
    nodes, omegas = create_logarithmic_spiral(n_points=100)
    S = compute_physical_stretching(nodes, omegas)
    print(f"{'Single Spiral':<25} | {len(nodes):<10} | {S:<15.4f} | {S/len(nodes):.4f}")
    
    # 2. Dual Spirals (Counter-rotating / Breakdown)
    n1, o1 = create_logarithmic_spiral(n_points=50, a=0.1, b=0.1, h=0.2)
    n2, o2 = create_logarithmic_spiral(n_points=50, a=0.1, b=0.1, h=-0.2)
    nodes = np.concatenate([n1, n2 + [0, 0.5, 0]])
    omegas = np.concatenate([o1, o2])
    S = compute_physical_stretching(nodes, omegas)
    print(f"{'Dual Spirals (C-Rot)':<25} | {len(nodes):<10} | {S:<15.4f} | {S/len(nodes):.4f}")
    
    # 3. Multi-scale Cloud (3 levels)
    nodes, omegas = create_multiscale_vortex_cloud(levels=3)
    S = compute_physical_stretching(nodes, omegas)
    print(f"{'Multi-scale Cloud':<25} | {len(nodes):<10} | {S:<15.4f} | {S/len(nodes):.4f}")

if __name__ == "__main__":
    run_spiral_audit()
