"""
VECTOR 3: DNS VORTEX INTERACTION TOPOLOGY ANALYSIS

Uses the Johns Hopkins Turbulence Database (JHTDB) to empirically study
the topology of vortex interactions in regions of intense vorticity.

THE QUESTION:
    In real turbulent flows, when vorticity is very intense in a region,
    what does the "interaction network" look like?
    - Do vorticity directions span S^2 (Lei et al. condition)?
    - What is the topology of the high-vorticity point cloud?
    - Is there a natural graph structure? Does it resemble K_n? Star?
    - What is the stretching rate distribution in intense regions?

DATASET:
    JHTDB Forced Isotropic Turbulence, 1024^3 grid, Re_lambda ~ 433
    Queries velocity gradient tensor (9 components) over a subvolume.

METHOD:
    1. Query a 3D subvolume of the velocity gradient field
    2. Compute vorticity omega = curl(u) and strain S = (grad_u + grad_u^T)/2
    3. Identify high-vorticity points (|omega| > threshold * rms)
    4. Build interaction network: connect nearby high-vorticity points
    5. Analyze: directions on S^2, stretching statistics, network topology

Author: Claude (Opus 4.6) / Meridian, 2026-03-11
"""

import numpy as np
from scipy.spatial import cKDTree
import sys
import time as time_module

sys.stdout.reconfigure(encoding="utf-8")
np.set_printoptions(precision=6, linewidth=120)

# ============================================================
# JHTDB Access via SOAP (zeep) — works on Windows
# ============================================================

JHTDB_WSDL = "http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL"
# Default demo token (limited queries, sufficient for our analysis)
AUTH_TOKEN = "edu.jhu.pha.turbulence.testing-201406"
DATASET = "isotropic1024coarse"


def query_velocity_gradient(x_coords, y_coords, z_coords, time_val=0.364,
                             spatial_interp="Fd4Lag4", temporal_interp="None"):
    """
    Query velocity gradient tensor from JHTDB at specified points.

    Returns: Nx9 array of [du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz]
    for N = len(x_coords) * len(y_coords) * len(z_coords) grid points.
    """
    from zeep import Client

    client = Client(JHTDB_WSDL)

    # Build point array (JHTDB expects flat list of x,y,z coordinates)
    points = []
    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                points.append({"x": float(x), "y": float(y), "z": float(z)})

    n_points = len(points)
    print(f"  Querying {n_points} points from JHTDB...")

    # JHTDB has a limit per request (~4096 points). Batch if needed.
    batch_size = 4096
    all_results = []

    for i in range(0, n_points, batch_size):
        batch = points[i:i+batch_size]
        print(f"    Batch {i//batch_size + 1}: points {i}-{min(i+batch_size, n_points)-1}...")

        # Build the SOAP request
        point_array = client.get_type("ns0:ArrayOfPoint3")
        point3_type = client.get_type("ns0:Point3")
        soap_points = point_array([point3_type(**p) for p in batch])

        # Try multiple enum names — JHTDB SOAP is picky about casing
        last_err = None
        result = None
        for si in [spatial_interp, "FD4Lag4", "Fd4Lag4", "FD4NoInt",
                   "Fd4NoInt", "None_Fd4"]:
            try:
                result = client.service.GetVelocityGradient(
                    authToken=AUTH_TOKEN,
                    dataset=DATASET,
                    time=float(time_val),
                    spatialInterpolation=si,
                    temporalInterpolation=temporal_interp,
                    points=soap_points
                )
                if i == 0:
                    print(f"    (using spatialInterpolation='{si}')")
                break
            except Exception as e:
                last_err = e
                continue
        if result is None:
            raise last_err

        for r in result:
            all_results.append([
                r["duxdx"], r["duxdy"], r["duxdz"],
                r["duydx"], r["duydy"], r["duydz"],
                r["duzdx"], r["duzdy"], r["duzdz"]
            ])

        time_module.sleep(0.5)  # Be polite to the server

    return np.array(all_results)


def compute_vorticity_and_strain(grad_u):
    """
    From velocity gradient tensor, compute vorticity and strain.

    grad_u: Nx9 array [du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz]

    Returns:
        omega: Nx3 vorticity vector
        strain_eigenvalues: Nx3 eigenvalues of strain tensor (sorted)
        omega_magnitude: N array of |omega|
    """
    n = len(grad_u)

    # Reshape to Nx3x3 tensor
    G = grad_u.reshape(n, 3, 3)

    # Vorticity: omega_i = epsilon_{ijk} duj/dxk
    omega = np.zeros((n, 3))
    omega[:, 0] = G[:, 2, 1] - G[:, 1, 2]  # dw/dy - dv/dz
    omega[:, 1] = G[:, 0, 2] - G[:, 2, 0]  # du/dz - dw/dx
    omega[:, 2] = G[:, 1, 0] - G[:, 0, 1]  # dv/dx - du/dy

    omega_mag = np.linalg.norm(omega, axis=1)

    # Strain tensor: S = (G + G^T) / 2
    S = (G + G.transpose(0, 2, 1)) / 2

    # Eigenvalues of strain tensor
    strain_evals = np.linalg.eigvalsh(S)  # sorted ascending

    return omega, strain_evals, omega_mag


# ============================================================
# SYNTHETIC TURBULENCE (fallback if JHTDB is unavailable)
# ============================================================

def generate_synthetic_turbulence(n_grid, L=2*np.pi, n_modes=200, seed=42):
    """
    Generate synthetic incompressible turbulence velocity gradient field
    WITH INTERMITTENCY (heavy-tailed vorticity distribution).

    Real turbulence at Re_lambda~433 has |omega|_max / |omega|_rms ~ 10-20.
    Plain Fourier modes give ~2-3 (Gaussian). We add intermittency by:
    1. Superposing localized vortex structures (Burgers-like tubes)
    2. Using log-normal amplitude modulation (Kolmogorov 1962 refined theory)
    """
    rng = np.random.RandomState(seed)

    # Grid
    dx = L / n_grid
    coords = np.linspace(0, L - dx, n_grid)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    n_points = len(X_flat)

    # Initialize gradient tensor
    grad_u = np.zeros((n_points, 3, 3))

    # PART 1: Background Fourier modes (bulk turbulence)
    for _ in range(n_modes):
        k = rng.randn(3) * 3  # wider k-space sampling
        k_mag = np.linalg.norm(k)
        if k_mag < 0.1:
            continue

        amplitude = rng.exponential(0.3) / (1 + k_mag**2)**(5.0/6.0)
        phase = rng.uniform(0, 2*np.pi)

        rand_dir = rng.randn(3)
        u_hat = rand_dir - np.dot(rand_dir, k) / k_mag**2 * k
        u_hat_mag = np.linalg.norm(u_hat)
        if u_hat_mag < 1e-10:
            continue
        u_hat = u_hat / u_hat_mag * amplitude

        kdotx = k[0]*X_flat + k[1]*Y_flat + k[2]*Z_flat + phase
        sin_kdotx = np.sin(kdotx)
        for i in range(3):
            for j in range(3):
                grad_u[:, i, j] += -u_hat[i] * k[j] * sin_kdotx

    # PART 2: Localized intense vortex tubes (intermittency)
    # These create the heavy tails in the vorticity PDF
    n_tubes = 15
    for t in range(n_tubes):
        # Random center and direction
        center = rng.uniform(0, L, 3)
        direction = rng.randn(3)
        direction /= np.linalg.norm(direction)

        # Tube radius and intensity (log-normal for intermittency)
        tube_radius = 0.05 + rng.exponential(0.03)
        intensity = rng.lognormal(mean=1.5, sigma=1.0)

        # Perpendicular directions
        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, [1, 0, 0])
        else:
            perp1 = np.cross(direction, [0, 1, 0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        # Distance from tube axis (with periodic wrapping)
        rel = np.column_stack([X_flat, Y_flat, Z_flat]) - center
        # Periodic wrapping
        for dim in range(3):
            rel[:, dim] = rel[:, dim] - L * np.round(rel[:, dim] / L)

        along = rel @ direction
        perp_vec = rel - along[:, np.newaxis] * direction
        r_perp = np.linalg.norm(perp_vec, axis=1)

        # Gaussian vortex tube: omega = intensity * exp(-r^2/(2*sigma^2)) * direction
        gauss = intensity * np.exp(-r_perp**2 / (2 * tube_radius**2))

        # The velocity gradient from a Burgers vortex tube:
        # omega along 'direction', velocity is azimuthal
        # grad(omega) contributes to the velocity gradient tensor
        # Simplified: add vorticity-like contribution to antisymmetric part
        for i in range(3):
            for j in range(3):
                # Antisymmetric part: (G - G^T)/2 contributes to vorticity
                # omega_k = epsilon_{kij} * G_{ij}
                # We add contributions that create omega along 'direction'
                if i != j:
                    # Levi-Civita contribution
                    for k_idx in range(3):
                        # epsilon_{k,i,j} * direction[k] * gauss
                        eps = 0
                        if (k_idx, i, j) in [(0,1,2), (1,2,0), (2,0,1)]:
                            eps = 1
                        elif (k_idx, i, j) in [(0,2,1), (1,0,2), (2,1,0)]:
                            eps = -1
                        if eps != 0:
                            grad_u[:, i, j] += 0.5 * eps * direction[k_idx] * gauss

        # Add strain associated with vortex stretching
        # Axial strain along tube direction, compression perpendicular
        strain_rate = intensity * 0.3 * np.exp(-r_perp**2 / (2 * tube_radius**2))
        for i in range(3):
            for j in range(3):
                # Stretching along direction: S_ij += rate * (d_i*d_j - delta_ij/3)
                contrib = direction[i] * direction[j]
                if i == j:
                    contrib -= 1.0/3.0
                grad_u[:, i, j] += strain_rate * contrib

    return coords, grad_u.reshape(n_points, 9)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def find_intense_regions(positions, omega, omega_mag, threshold_factor=3.0):
    """
    Find points where |omega| > threshold_factor * omega_rms.
    Returns indices, positions, and vorticity vectors of intense points.
    """
    omega_rms = np.sqrt(np.mean(omega_mag**2))
    threshold = threshold_factor * omega_rms

    intense_idx = np.where(omega_mag > threshold)[0]

    print(f"  omega_rms = {omega_rms:.4f}")
    print(f"  Threshold = {threshold_factor} * omega_rms = {threshold:.4f}")
    print(f"  Intense points: {len(intense_idx)} / {len(omega_mag)} "
          f"({100*len(intense_idx)/len(omega_mag):.2f}%)")

    return intense_idx, omega_rms, threshold


def build_interaction_network(positions, omega, omega_mag, intense_idx,
                                connection_radius):
    """
    Build a graph connecting intense vorticity points that are within
    connection_radius of each other.

    Returns: adjacency list, edge list with weights
    """
    intense_pos = positions[intense_idx]
    n_intense = len(intense_idx)

    if n_intense < 2:
        print("  Too few intense points for network analysis.")
        return [], []

    # KD-tree for fast neighbor search
    tree = cKDTree(intense_pos)
    pairs = tree.query_pairs(r=connection_radius)

    edges = []
    for i, j in pairs:
        # Edge weight: product of vorticity magnitudes
        weight = omega_mag[intense_idx[i]] * omega_mag[intense_idx[j]]
        # Directional alignment
        xi_i = omega[intense_idx[i]] / omega_mag[intense_idx[i]]
        xi_j = omega[intense_idx[j]] / omega_mag[intense_idx[j]]
        alignment = np.dot(xi_i, xi_j)
        edges.append((i, j, weight, alignment))

    return edges


def analyze_direction_coverage(omega, omega_mag, intense_idx, n_checks=200):
    """
    Test whether vorticity directions in intense region span S^2
    (Lei et al. condition).

    For each random axis e, compute max |xi_i x e| over intense points.
    If this is always close to 1, directions span S^2.
    """
    xi = omega[intense_idx] / omega_mag[intense_idx, np.newaxis]
    n = len(intense_idx)

    if n < 3:
        return 0.0, False

    min_max_cross = 1.0
    rng = np.random.RandomState(0)

    for _ in range(n_checks):
        e = rng.randn(3)
        e /= np.linalg.norm(e)

        # For each direction xi_i, compute |xi_i x e|
        crosses = np.linalg.norm(np.cross(xi, e), axis=1)
        max_cross = np.max(crosses)
        min_max_cross = min(min_max_cross, max_cross)

    spans_s2 = min_max_cross > 0.5
    return min_max_cross, spans_s2


def analyze_stretching_alignment(omega, strain_evals, omega_mag, intense_idx, grad_u_full):
    """
    Analyze the alignment between vorticity and strain eigenvectors
    in intense regions (Constantin's identity: omega.S.omega = |omega|^2 * sigma).
    """
    G = grad_u_full.reshape(-1, 3, 3)
    S = (G + G.transpose(0, 2, 1)) / 2

    sigmas = []
    alignments = []  # cosine of angle between omega and each strain eigenvector

    for idx in intense_idx:
        xi = omega[idx] / omega_mag[idx]
        sigma = xi @ S[idx] @ xi
        sigmas.append(sigma)

        # Strain eigenvectors
        evals, evecs = np.linalg.eigh(S[idx])
        # evals sorted ascending: lambda_1 <= lambda_2 <= lambda_3
        # Alignment with each eigenvector
        align = [abs(np.dot(xi, evecs[:, k])) for k in range(3)]
        alignments.append(align)

    sigmas = np.array(sigmas)
    alignments = np.array(alignments)

    return sigmas, alignments


def network_topology_analysis(edges, n_nodes):
    """
    Analyze the topology of the vortex interaction network.
    """
    if len(edges) == 0:
        return {}

    # Degree distribution
    degrees = np.zeros(n_nodes, dtype=int)
    for i, j, w, a in edges:
        degrees[i] += 1
        degrees[j] += 1

    # Alignment distribution
    alignments = np.array([a for _, _, _, a in edges])

    # Connected components (simple union-find)
    parent = list(range(n_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j, w, a in edges:
        union(i, j)

    components = {}
    for i in range(n_nodes):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    comp_sizes = sorted([len(c) for c in components.values()], reverse=True)

    # Is it star-like? Check if one node has much higher degree
    max_degree = np.max(degrees) if len(degrees) > 0 else 0
    mean_degree = np.mean(degrees) if len(degrees) > 0 else 0

    # Is it clique-like? Check density
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    density = len(edges) / max_edges if max_edges > 0 else 0

    return {
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "density": density,
        "max_degree": max_degree,
        "mean_degree": mean_degree,
        "degree_std": np.std(degrees),
        "n_components": len(components),
        "largest_component": comp_sizes[0] if comp_sizes else 0,
        "component_sizes": comp_sizes[:5],
        "alignment_mean": np.mean(alignments),
        "alignment_std": np.std(alignments),
        "frac_parallel": np.mean(np.abs(alignments) > 0.7),
        "frac_perpendicular": np.mean(np.abs(alignments) < 0.3),
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 75)
print("DNS VORTEX INTERACTION TOPOLOGY ANALYSIS")
print("=" * 75)

# Try JHTDB first, fall back to synthetic
use_jhtdb = False
n_grid = 48  # 48^3 = 110K points — manageable for JHTDB query

try:
    print("\n  Attempting JHTDB connection...")
    from zeep import Client
    client = Client(JHTDB_WSDL)

    # Test with a single point first
    point3_type = client.get_type("ns0:Point3")
    point_array = client.get_type("ns0:ArrayOfPoint3")
    test_pt = point_array([point3_type(x=0.1, y=0.1, z=0.1)])

    result = client.service.GetVelocityGradient(
        authToken=AUTH_TOKEN,
        dataset=DATASET,
        time=0.364,
        spatialInterpolation="FD4Lag4",
        temporalInterpolation="NoTInt",
        points=test_pt
    )
    print(f"  JHTDB connection successful! Test gradient: du/dx = {result[0]['duxdx']:.6f}")
    use_jhtdb = True
except Exception as e:
    print(f"  JHTDB unavailable: {e}")
    print(f"  Falling back to synthetic turbulence...")
    use_jhtdb = False


if use_jhtdb:
    # Query a subvolume from JHTDB
    # Domain is [0, 2*pi], grid spacing = 2*pi/1024 ~ 0.00614
    L = 2 * np.pi
    dx = L / 1024
    # Pick a random subvolume
    x0, y0, z0 = 1.0, 1.0, 1.0  # Start point
    sub_L = n_grid * dx  # Subvolume size

    coords = np.linspace(x0, x0 + sub_L - dx, n_grid)

    print(f"\n  Querying {n_grid}^3 = {n_grid**3} velocity gradient points...")
    print(f"  Subvolume: [{x0:.3f}, {x0+sub_L:.3f}]^3, dx = {dx:.5f}")

    grad_u = query_velocity_gradient(coords, coords, coords, time_val=0.364)

    # Build position arrays
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

else:
    # Synthetic turbulence
    print(f"\n  Generating synthetic turbulence ({n_grid}^3 grid, 100 Fourier modes)...")
    coords, grad_u = generate_synthetic_turbulence(n_grid, n_modes=100)

    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    print(f"  Generated {len(positions)} points.")


# ============================================================
# SECTION 1: Compute vorticity and strain
# ============================================================

print(f"\n{'='*75}")
print("SECTION 1: VORTICITY AND STRAIN STATISTICS")
print(f"{'='*75}")

omega, strain_evals, omega_mag = compute_vorticity_and_strain(grad_u)

print(f"\n  Vorticity statistics:")
print(f"    |omega|_rms  = {np.sqrt(np.mean(omega_mag**2)):.4f}")
print(f"    |omega|_max  = {np.max(omega_mag):.4f}")
print(f"    |omega|_mean = {np.mean(omega_mag):.4f}")
print(f"    |omega|_max / |omega|_rms = {np.max(omega_mag)/np.sqrt(np.mean(omega_mag**2)):.2f}")

print(f"\n  Strain eigenvalue statistics (lambda_1 <= lambda_2 <= lambda_3):")
print(f"    <lambda_1> = {np.mean(strain_evals[:,0]):.4f}  (compression)")
print(f"    <lambda_2> = {np.mean(strain_evals[:,1]):.4f}  (intermediate)")
print(f"    <lambda_3> = {np.mean(strain_evals[:,2]):.4f}  (extension)")
print(f"    <trace> = {np.mean(np.sum(strain_evals, axis=1)):.6f}  (should be ~0 for incompressible)")

# Enstrophy
enstrophy_density = omega_mag**2
print(f"\n  Enstrophy density:")
print(f"    Mean: {np.mean(enstrophy_density):.4f}")
print(f"    Max:  {np.max(enstrophy_density):.4f}")
print(f"    Max/Mean: {np.max(enstrophy_density)/np.mean(enstrophy_density):.2f}")


# ============================================================
# SECTION 2: Find intense vorticity regions
# ============================================================

print(f"\n{'='*75}")
print("SECTION 2: INTENSE VORTICITY REGIONS")
print(f"{'='*75}")

for threshold_factor in [2.0, 3.0, 5.0, 7.0]:
    intense_idx, omega_rms, threshold = find_intense_regions(
        positions, omega, omega_mag, threshold_factor)

    if len(intense_idx) < 3:
        print(f"  (too few points at {threshold_factor}x threshold)\n")
        continue

    # Direction coverage (Lei et al. test)
    min_max_cross, spans = analyze_direction_coverage(omega, omega_mag, intense_idx)
    print(f"  Direction coverage: min max|xi x e| = {min_max_cross:.4f} "
          f"({'SPANS S^2' if spans else 'does NOT span S^2'})")

    # Stretching alignment
    sigmas, aligns = analyze_stretching_alignment(
        omega, strain_evals, omega_mag, intense_idx, grad_u)

    print(f"  Stretching rate sigma = xi.S.xi:")
    print(f"    Mean: {np.mean(sigmas):+.4f}")
    print(f"    Std:  {np.std(sigmas):.4f}")
    print(f"    Fraction positive: {np.mean(sigmas > 0):.2f}")

    print(f"  Alignment with strain eigenvectors:")
    print(f"    |cos(omega, e_1)| (compression): {np.mean(aligns[:,0]):.3f}")
    print(f"    |cos(omega, e_2)| (intermediate): {np.mean(aligns[:,1]):.3f}")
    print(f"    |cos(omega, e_3)| (extension):    {np.mean(aligns[:,2]):.3f}")
    print(f"    (Depletion of nonlinearity: omega aligns with INTERMEDIATE, not extension)")
    print()


# ============================================================
# SECTION 3: Build and analyze interaction network
# ============================================================

print(f"\n{'='*75}")
print("SECTION 3: VORTEX INTERACTION NETWORK TOPOLOGY")
print(f"{'='*75}")

# Use 3x threshold for "intense" regions
intense_idx, omega_rms, threshold = find_intense_regions(
    positions, omega, omega_mag, threshold_factor=3.0)

if len(intense_idx) >= 5:
    # Connection radius: a few grid spacings
    grid_dx = coords[1] - coords[0] if len(coords) > 1 else 0.1
    connection_radii = [3*grid_dx, 5*grid_dx, 10*grid_dx]

    for conn_r in connection_radii:
        print(f"\n  Connection radius = {conn_r:.4f} ({conn_r/grid_dx:.1f} grid spacings):")
        edges = build_interaction_network(
            positions, omega, omega_mag, intense_idx, conn_r)

        if len(edges) > 0:
            stats = network_topology_analysis(edges, len(intense_idx))

            print(f"    Nodes: {stats['n_nodes']}, Edges: {stats['n_edges']}")
            print(f"    Density: {stats['density']:.4f} (1.0 = complete graph K_n)")
            print(f"    Mean degree: {stats['mean_degree']:.2f}")
            print(f"    Max degree: {stats['max_degree']} (star-like if >> mean)")
            print(f"    Degree std: {stats['degree_std']:.2f}")
            print(f"    Components: {stats['n_components']}, largest: {stats['largest_component']}")
            print(f"    Component sizes: {stats['component_sizes']}")
            print(f"    Edge alignment: mean |cos| = {stats['alignment_mean']:.3f}")
            print(f"    Parallel edges (|cos|>0.7): {stats['frac_parallel']*100:.1f}%")
            print(f"    Perpendicular edges (|cos|<0.3): {stats['frac_perpendicular']*100:.1f}%")

            # Is it more star-like or clique-like?
            if stats['max_degree'] > 3 * stats['mean_degree'] and stats['mean_degree'] > 0:
                print(f"    TOPOLOGY: STAR-LIKE (hub-and-spoke)")
            elif stats['density'] > 0.5:
                print(f"    TOPOLOGY: DENSE/CLIQUE-LIKE")
            elif stats['density'] > 0.1:
                print(f"    TOPOLOGY: INTERMEDIATE")
            else:
                print(f"    TOPOLOGY: SPARSE")
        else:
            print(f"    No edges found (points too far apart)")


    # ============================================================
    # SECTION 4: Cluster analysis of intense regions
    # ============================================================

    print(f"\n\n{'='*75}")
    print("SECTION 4: CLUSTERING OF INTENSE VORTICITY (spatial structure)")
    print(f"{'='*75}")

    intense_pos = positions[intense_idx]
    intense_omega = omega[intense_idx]
    intense_mag = omega_mag[intense_idx]

    # Find clusters using simple distance threshold
    cluster_radius = 5 * grid_dx
    tree = cKDTree(intense_pos)

    # Connected components
    n_intense = len(intense_idx)
    parent = list(range(n_intense))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    pairs = tree.query_pairs(r=cluster_radius)
    for i, j in pairs:
        union(i, j)

    clusters = {}
    for i in range(n_intense):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    cluster_list = sorted(clusters.values(), key=len, reverse=True)
    print(f"\n  Cluster radius: {cluster_radius:.4f} ({cluster_radius/grid_dx:.1f} grid spacings)")
    print(f"  Number of clusters: {len(cluster_list)}")
    print(f"  Cluster sizes: {[len(c) for c in cluster_list[:10]]}")

    # Analyze each cluster
    for ci, cluster_indices in enumerate(cluster_list[:5]):
        if len(cluster_indices) < 3:
            continue

        c_omega = intense_omega[cluster_indices]
        c_mag = intense_mag[cluster_indices]
        c_pos = intense_pos[cluster_indices]

        # Direction coverage within cluster
        c_xi = c_omega / c_mag[:, np.newaxis]
        min_max_cross = 1.0
        rng = np.random.RandomState(ci)
        for _ in range(100):
            e = rng.randn(3)
            e /= np.linalg.norm(e)
            crosses = np.linalg.norm(np.cross(c_xi, e), axis=1)
            min_max_cross = min(min_max_cross, np.max(crosses))

        spans = min_max_cross > 0.5

        # Spatial extent
        extent = np.max(c_pos, axis=0) - np.min(c_pos, axis=0)

        # Mean alignment between pairs
        pair_alignments = []
        for ii in range(min(len(cluster_indices), 20)):
            for jj in range(ii+1, min(len(cluster_indices), 20)):
                pair_alignments.append(abs(np.dot(c_xi[ii], c_xi[jj])))

        print(f"\n  Cluster {ci+1}: {len(cluster_indices)} points")
        print(f"    Extent: [{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}]")
        print(f"    Max |omega|: {np.max(c_mag):.4f}")
        print(f"    Direction coverage: {min_max_cross:.4f} "
              f"({'SPANS S^2' if spans else 'DOES NOT span S^2'})")
        if pair_alignments:
            print(f"    Mean pairwise |cos|: {np.mean(pair_alignments):.3f} "
                  f"(1=parallel, 0=perpendicular)")
            print(f"    Parallel fraction:   {np.mean(np.array(pair_alignments)>0.7)*100:.1f}%")

else:
    print("  Insufficient intense points for network analysis.")


# ============================================================
# SECTION 5: Summary
# ============================================================

print(f"\n\n{'='*75}")
print("SECTION 5: SUMMARY")
print(f"{'='*75}")

data_source = "JHTDB (Re_lambda~433)" if use_jhtdb else "Synthetic (100 Fourier modes)"
print(f"""
DATA SOURCE: {data_source}
GRID: {n_grid}^3 = {n_grid**3} points

KEY QUESTIONS AND FINDINGS:

1. DO INTENSE VORTICITY DIRECTIONS SPAN S^2 (Lei et al.)?
   (See Section 2 — "Direction coverage" results above)

2. DOES VORTICITY ALIGN WITH INTERMEDIATE STRAIN EIGENVECTOR?
   (See Section 2 — "Alignment with strain eigenvectors" results above)
   Expected: |cos(omega, e_2)| > |cos(omega, e_3)| > |cos(omega, e_1)|
   This would confirm depletion of nonlinearity.

3. WHAT TOPOLOGY DOES THE INTERACTION NETWORK HAVE?
   (See Section 3 — density, degree distribution, star vs clique)

4. DO INDIVIDUAL CLUSTERS HAVE ISOTROPIC DIRECTIONS?
   (See Section 4 — per-cluster direction coverage)

INTERPRETATION GUIDE:
   - If clusters have PARALLEL directions: filamentary structure (single tube)
   - If clusters SPAN S^2: multi-directional (consistent with Lei et al.)
   - If network is STAR-LIKE: one dominant tube with branches
   - If network is CLIQUE-LIKE: dense K_n-like structure (would support our conjecture)
   - If network is SPARSE: isolated structures, no strong interaction
""")
