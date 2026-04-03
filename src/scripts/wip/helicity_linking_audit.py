import numpy as np
from scipy.linalg import eigh

def compute_physical_stretching(nodes, omegas, sigma=0.1):
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

def create_circular_ring(center, axis, radius=1.0, n_seg=50):
    center = np.array(center, dtype=np.float64)
    axis = np.array(axis, dtype=np.float64)
    nodes = []
    omegas = []
    # Orthogonal basis for the ring plane
    if np.abs(axis[0]) < 0.9:
        v1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    else:
        v1 = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis, v1)
    
    for theta in np.linspace(0, 2*np.pi, n_seg, endpoint=False):
        pos = center + radius * (np.cos(theta) * v1 + np.sin(theta) * v2)
        tangent = -np.sin(theta) * v1 + np.cos(theta) * v2
        nodes.append(pos)
        omegas.append(tangent)
    return np.array(nodes), np.array(omegas)

def run_helicity_audit():
    print(f"--- HELICITY & LINKING AUDIT ---")
    print(f"{'Topology':<25} | {'Stretching':<15} | {'Stretching/N':<15}")
    print("-" * 65)

    # 1. Unlinked Parallel Rings
    ring1_nodes, ring1_omegas = create_circular_ring([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    ring2_nodes, ring2_omegas = create_circular_ring([0.0, 2.0, 0.0], [0.0, 0.0, 1.0])
    nodes = np.concatenate([ring1_nodes, ring2_nodes])
    omegas = np.concatenate([ring1_omegas, ring2_omegas])
    S = compute_physical_stretching(nodes, omegas)
    print(f"{'Unlinked Rings':<25} | {S:<15.4f} | {S/len(nodes):<15.4f}")

    # 2. Linked Hopf Link (Distance = 1.0)
    # Ring 1 in XY plane, Ring 2 in XZ plane
    ring1_nodes, ring1_omegas = create_circular_ring([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    ring2_nodes, ring2_omegas = create_circular_ring([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    nodes = np.concatenate([ring1_nodes, ring2_nodes])
    omegas = np.concatenate([ring1_omegas, ring2_omegas])
    S = compute_physical_stretching(nodes, omegas)
    print(f"{'Hopf Link':<25} | {S:<15.4f} | {S/len(nodes):<15.4f}")

    # 3. Borromean Rings (approximate)
    r1_n, r1_o = create_circular_ring([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    r2_n, r2_o = create_circular_ring([0.5, 0.5, 0.0], [0.0, 1.0, 0.0])
    r3_n, r3_o = create_circular_ring([0.5, 0.0, 0.5], [1.0, 0.0, 0.0])
    nodes = np.concatenate([r1_n, r2_n, r3_n])
    omegas = np.concatenate([r1_o, r2_o, r3_o])
    S = compute_physical_stretching(nodes, omegas)
    print(f"{'Borromean Rings':<25} | {S:<15.4f} | {S/len(nodes):<15.4f}")

if __name__ == "__main__":
    run_helicity_audit()
