"""
Strategy C: Berry phase measurement on Fano triad graph.

Three measurements:
1. Triad phase holonomy around closed loops on the Fano plane
2. Maximum phase coherence (frustration bound)
3. c_1 = 2 test: total holonomy = 4*pi?

Uses DNS-evolved Taylor-Green fields at t=1,2,3.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
from itertools import product, combinations
import time
import sys

def taylor_green_ic(N, A=1.0):
    """Taylor-Green vortex initial condition."""
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    Y, X, Z = np.meshgrid(x, x, x, indexing='ij')
    u = np.zeros((3, N, N, N))
    u[0] = A * np.sin(X) * np.cos(Y) * np.cos(Z)
    u[1] = -A * np.cos(X) * np.sin(Y) * np.cos(Z)
    u[2] = 0.0
    return u

def leray_project(u_hat, kx, ky, kz):
    """Project velocity field to be divergence-free in Fourier space."""
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1
    k_dot_u = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    u_hat[0] -= kx * k_dot_u / k2
    u_hat[1] -= ky * k_dot_u / k2
    u_hat[2] -= kz * k_dot_u / k2
    u_hat[:, 0, 0, 0] = 0
    return u_hat

def evolve_ns(u, nu, dt, t_target, N, print_interval=500):
    """Evolve NS with RK4, dealiased."""
    kx = np.fft.fftfreq(N, d=1.0/(2*np.pi*N)).astype(np.float64)
    ky = kx.copy()
    kz = kx.copy()
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1
    dealias = np.ones_like(K2)
    kmax = N // 3
    dealias[np.sqrt(KX**2 + KY**2 + KZ**2) > kmax] = 0

    def rhs(u_hat):
        u_hat_d = u_hat * dealias
        u = np.array([np.real(ifftn(u_hat_d[i])) for i in range(3)])
        omega = np.zeros_like(u)
        omega[0] = np.real(ifftn(1j * (KY * u_hat_d[2] - KZ * u_hat_d[1])))
        omega[1] = np.real(ifftn(1j * (KZ * u_hat_d[0] - KX * u_hat_d[2])))
        omega[2] = np.real(ifftn(1j * (KX * u_hat_d[1] - KY * u_hat_d[0])))
        nl = np.zeros_like(u)
        nl[0] = u[1] * omega[2] - u[2] * omega[1]
        nl[1] = u[2] * omega[0] - u[0] * omega[2]
        nl[2] = u[0] * omega[1] - u[1] * omega[0]
        nl_hat = np.array([fftn(nl[i]) for i in range(3)])
        nl_hat = leray_project(nl_hat, KX, KY, KZ)
        return nl_hat * dealias - nu * K2 * u_hat

    u_hat = np.array([fftn(u[i]) for i in range(3)])
    u_hat = leray_project(u_hat, KX, KY, KZ)

    t = 0.0
    step = 0
    t_start = time.time()
    snapshots = {}

    while t < t_target - dt/2:
        k1 = rhs(u_hat)
        k2r = rhs(u_hat + 0.5 * dt * k1)
        k3r = rhs(u_hat + 0.5 * dt * k2r)
        k4r = rhs(u_hat + dt * k3r)
        u_hat += (dt / 6) * (k1 + 2*k2r + 2*k3r + k4r)
        u_hat = leray_project(u_hat, KX, KY, KZ)
        t += dt
        step += 1

        if step % print_interval == 0:
            elapsed = time.time() - t_start
            print(f"  [step {step}, t={t:.3f}, elapsed={elapsed:.0f}s]", flush=True)

        # Save snapshots at t=1, 2, 3
        for t_snap in [1.0, 2.0, 3.0]:
            if abs(t - t_snap) < dt/2 and t_snap not in snapshots:
                snapshots[t_snap] = u_hat.copy()
                print(f"  ** Snapshot at t={t_snap:.1f} **", flush=True)

    if t_target not in snapshots:
        snapshots[t_target] = u_hat.copy()

    return snapshots

# ============================================================
# FANO PLANE STRUCTURE
# ============================================================

# 7 points of the Fano plane = nonzero elements of GF(2)^3
FANO_POINTS = [
    (1,0,0), (0,1,0), (0,0,1),
    (1,1,0), (1,0,1), (0,1,1),
    (1,1,1)
]

# 7 lines: each line = 3 points where a XOR b XOR c = 0
FANO_LINES = []
for i, a in enumerate(FANO_POINTS):
    for j, b in enumerate(FANO_POINTS):
        if j <= i:
            continue
        c = (a[0]^b[0], a[1]^b[1], a[2]^b[2])
        if c in FANO_POINTS:
            line = tuple(sorted([a, b, c]))
            if line not in FANO_LINES:
                FANO_LINES.append(line)

print(f"Fano lines ({len(FANO_LINES)}):")
for l in FANO_LINES:
    print(f"  {l[0]} + {l[1]} + {l[2]} = 0 (mod 2)")

# Find all triangles (3 non-collinear points = 3 lines forming a cycle)
# A triangle: 3 points A, B, C where no line contains all 3
line_set = set(FANO_LINES)
TRIANGLES = []
for combo in combinations(FANO_POINTS, 3):
    triple = tuple(sorted(combo))
    if triple not in line_set:
        # Check that each pair lies on a Fano line
        a, b, c = combo
        # Each pair defines a unique line in the Fano plane
        # (any 2 points determine a line in PG(2,2))
        TRIANGLES.append(combo)

print(f"Triangles (non-collinear triples): {len(TRIANGLES)}")

# Find all minimal CYCLES in the line graph
# A cycle = sequence of lines where consecutive lines share a point
# Minimal cycle length in the Fano plane = 3 (triangle of lines sharing points)
LINE_CYCLES = []
for i, l1 in enumerate(FANO_LINES):
    s1 = set(l1)
    for j, l2 in enumerate(FANO_LINES):
        if j <= i:
            continue
        s2 = set(l2)
        shared12 = s1 & s2
        if not shared12:
            continue
        for k_idx, l3 in enumerate(FANO_LINES):
            if k_idx <= j:
                continue
            s3 = set(l3)
            shared23 = s2 & s3
            shared31 = s3 & s1
            if shared23 and shared31:
                LINE_CYCLES.append((i, j, k_idx))

print(f"Line 3-cycles: {len(LINE_CYCLES)}")

# ============================================================
# PHASE MEASUREMENT FUNCTIONS
# ============================================================

def get_mode(u_hat, kx, ky, kz, N):
    """Extract Fourier coefficient u_hat(k) as a 3-component complex vector."""
    ix = kx % N
    iy = ky % N
    iz = kz % N
    return np.array([u_hat[c][ix, iy, iz] for c in range(3)])

def triad_phase(u_hat, k, p, q, N):
    """Compute triad phase: arg(u_k . (u_p x u_q))"""
    uk = get_mode(u_hat, *k, N)
    up = get_mode(u_hat, *p, N)
    uq = get_mode(u_hat, *q, N)

    # Triple product: u_k . (u_p x u_q)
    cross = np.cross(up, uq)
    triple = np.dot(uk, cross)  # complex number

    if abs(triple) < 1e-30:
        return 0.0, 0.0  # phase, magnitude

    return np.angle(triple), abs(triple)

def gf2_class(k):
    """Map wavevector to GF(2)^3 class."""
    return (abs(k[0]) % 2, abs(k[1]) % 2, abs(k[2]) % 2)

def find_representative_modes(modes_in_class, N):
    """Pick representative modes for each GF(2)^3 class."""
    # Use lowest-|k| modes as representatives
    sorted_modes = sorted(modes_in_class, key=lambda m: sum(x**2 for x in m))
    return sorted_modes[:min(5, len(sorted_modes))]

# ============================================================
# MAIN COMPUTATION
# ============================================================

print("\n" + "=" * 72)
print("  BERRY PHASE MEASUREMENT ON FANO TRIAD GRAPH")
print("=" * 72)

for N, Re, dt in [(32, 400, 0.001), (32, 1600, 0.0005)]:
    nu = 1.0 / Re
    print(f"\n--- N={N}, Re={Re} ---")

    u = taylor_green_ic(N)
    snapshots = evolve_ns(u, nu, dt, 3.0, N)

    # Build mode catalog
    k_max = 3
    all_modes = []
    class_modes = {p: [] for p in FANO_POINTS}

    for kx, ky, kz in product(range(-k_max, k_max+1), repeat=3):
        k2 = kx**2 + ky**2 + kz**2
        if 0 < k2 <= k_max**2:
            c = gf2_class((kx, ky, kz))
            if c != (0, 0, 0) and c in class_modes:
                all_modes.append((kx, ky, kz))
                class_modes[c].append((kx, ky, kz))

    print(f"  Fano modes: {len(all_modes)}")
    for p in FANO_POINTS:
        print(f"    Class {p}: {len(class_modes[p])} modes")

    # For each time snapshot
    for t_snap in [1.0, 2.0, 3.0]:
        if t_snap not in snapshots:
            continue
        u_hat = snapshots[t_snap]

        print(f"\n  === t = {t_snap:.1f} ===")

        # --------------------------------------------------------
        # MEASUREMENT 1: Triad phase per Fano line
        # --------------------------------------------------------
        # For each Fano line (a, b, c), find actual triads k+p+q=0
        # where k in class a, p in class b, q in class c
        line_phases = {}
        line_phase_details = {}

        for line_idx, line in enumerate(FANO_LINES):
            a, b, c = line
            phases = []
            magnitudes = []

            for k in class_modes[a]:
                for p in class_modes[b]:
                    q = (-(k[0]+p[0]), -(k[1]+p[1]), -(k[2]+p[2]))
                    q_class = gf2_class(q)
                    q2 = q[0]**2 + q[1]**2 + q[2]**2
                    if q_class == c and 0 < q2 <= k_max**2:
                        ph, mag = triad_phase(u_hat, k, p, q, N)
                        phases.append(ph)
                        magnitudes.append(mag)

            if phases:
                phases = np.array(phases)
                magnitudes = np.array(magnitudes)
                # Magnitude-weighted circular mean
                weights = magnitudes / (np.sum(magnitudes) + 1e-30)
                mean_sin = np.sum(weights * np.sin(phases))
                mean_cos = np.sum(weights * np.cos(phases))
                mean_phase = np.arctan2(mean_sin, mean_cos)
                resultant = np.sqrt(mean_sin**2 + mean_cos**2)

                line_phases[line_idx] = mean_phase
                line_phase_details[line_idx] = {
                    'mean_phase': mean_phase,
                    'resultant': resultant,
                    'n_triads': len(phases),
                    'unweighted_mean': np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases))),
                    'unweighted_R': np.sqrt(np.mean(np.sin(phases))**2 + np.mean(np.cos(phases))**2)
                }

        print(f"  Measurement 1: Triad phases per Fano line")
        for li in range(len(FANO_LINES)):
            if li in line_phase_details:
                d = line_phase_details[li]
                line = FANO_LINES[li]
                print(f"    Line {li} {line[0]}+{line[1]}+{line[2]}: "
                      f"phi={d['mean_phase']:+.4f} R={d['resultant']:.4f} "
                      f"(unwt: phi={d['unweighted_mean']:+.4f} R={d['unweighted_R']:.4f}) "
                      f"n={d['n_triads']}")

        # --------------------------------------------------------
        # MEASUREMENT 2: Holonomy around line 3-cycles
        # --------------------------------------------------------
        print(f"\n  Measurement 2: Holonomy around Fano line cycles")
        holonomies = []
        for cycle_idx, (i, j, k_idx) in enumerate(LINE_CYCLES):
            if i in line_phases and j in line_phases and k_idx in line_phases:
                phi = line_phases[i] + line_phases[j] + line_phases[k_idx]
                # Wrap to [-pi, pi]
                phi_wrapped = np.arctan2(np.sin(phi), np.cos(phi))
                holonomies.append(phi_wrapped)
                if cycle_idx < 10:  # Print first 10
                    print(f"    Cycle ({i},{j},{k_idx}): "
                          f"Phi = {line_phases[i]:+.4f} + {line_phases[j]:+.4f} + {line_phases[k_idx]:+.4f} "
                          f"= {phi:+.4f} -> {phi_wrapped:+.4f}")

        if holonomies:
            holonomies = np.array(holonomies)
            total_holonomy = np.sum(holonomies)
            print(f"\n    Total holonomy (sum of all cycles): {total_holonomy:+.6f}")
            print(f"    4*pi = {4*np.pi:.6f}")
            print(f"    2*pi = {2*np.pi:.6f}")
            print(f"    Ratio to 4pi: {total_holonomy / (4*np.pi):.4f}")
            print(f"    Ratio to 2pi: {total_holonomy / (2*np.pi):.4f}")
            print(f"    Mean |holonomy|: {np.mean(np.abs(holonomies)):.4f}")
            print(f"    Std holonomy: {np.std(holonomies):.4f}")

        # --------------------------------------------------------
        # MEASUREMENT 3: Maximum phase coherence (frustration)
        # --------------------------------------------------------
        # For each line, check sin(mean_phase) > 0 or < 0
        # Try all 2^7 = 128 sign flips on the 7 points
        # (flipping a point's sign flips all 3 lines through it)
        # Count max simultaneous sin(theta) > 0

        print(f"\n  Measurement 3: Phase coherence / frustration")

        # Get the sin of each line's phase
        line_sin = {}
        for li in range(len(FANO_LINES)):
            if li in line_phase_details:
                line_sin[li] = np.sin(line_phase_details[li]['mean_phase'])

        n_forward = sum(1 for s in line_sin.values() if s > 0)
        print(f"    Forward lines (sin phi > 0): {n_forward} / {len(line_sin)}")

        # Exhaustive search: flip signs on points, maximize forward lines
        # A sign flip on point p changes the phase of all lines through p by pi
        # This flips sin(phi) for those lines

        # Build incidence: which lines contain which point
        point_to_lines = {p: [] for p in FANO_POINTS}
        for li, line in enumerate(FANO_LINES):
            for p in line:
                point_to_lines[p].append(li)

        best_forward = 0
        best_config = None

        for config in range(128):  # 2^7 sign assignments
            # Each bit = sign of one point (0 = +, 1 = -)
            n_flips_per_line = {}
            for li, line in enumerate(FANO_LINES):
                n_flips = 0
                for pi, p in enumerate(FANO_POINTS):
                    if (config >> pi) & 1:
                        if p in line:
                            n_flips += 1
                n_flips_per_line[li] = n_flips

            # A line's phase flips by n_flips * pi
            # sin(phi + n*pi) = sin(phi)*cos(n*pi) = sin(phi)*(-1)^n
            forward = 0
            for li in line_sin:
                effective_sin = line_sin[li] * ((-1) ** n_flips_per_line[li])
                if effective_sin > 0:
                    forward += 1

            if forward > best_forward:
                best_forward = forward
                best_config = config

        print(f"    Max simultaneous forward lines: {best_forward} / {len(FANO_LINES)}")
        print(f"    Frustration: {len(FANO_LINES) - best_forward} lines cannot be made forward")
        print(f"    Coherence fraction: {best_forward / len(FANO_LINES):.4f}")

        if best_forward < len(FANO_LINES):
            print(f"    ** FRUSTRATION DETECTED: Berry phase prevents full coherence **")

        # Also compute: what's the THEORETICAL max for Fano plane?
        # Fano plane is a balanced incomplete block design
        # Each point is on 3 lines. Flipping a point flips 3 lines.
        # Starting from any configuration, each flip changes 3 lines.
        # The maximum independent set in the Fano line graph...

        # Actually: count over ALL possible initial phase assignments (not just DNS)
        # The point is whether the Fano GEOMETRY limits coherence
        print(f"\n    Combinatorial frustration test (independent of DNS):")
        print(f"    For each of 2^7 = 128 sign assignments to 7 points,")
        print(f"    count how many of 7 lines have positive product:")

        max_positive = 0
        distribution = np.zeros(8, dtype=int)

        for config in range(128):
            signs = np.array([1 if (config >> i) & 1 else -1 for i in range(7)])
            n_pos = 0
            for line in FANO_LINES:
                indices = [FANO_POINTS.index(p) for p in line]
                sign_prod = signs[indices[0]] * signs[indices[1]] * signs[indices[2]]
                if sign_prod > 0:
                    n_pos += 1
            distribution[n_pos] += 1
            max_positive = max(max_positive, n_pos)

        print(f"    Distribution of positive-product lines:")
        for n in range(8):
            if distribution[n] > 0:
                print(f"      {n} lines positive: {distribution[n]} configurations")
        print(f"    Maximum achievable: {max_positive} / 7")
        print(f"    Theoretical coherence bound: {max_positive / 7:.4f}")
        if max_positive < 7:
            print(f"    ** COMBINATORIAL PROOF: Fano plane has intrinsic frustration **")
            print(f"    ** At most {max_positive}/7 triads can simultaneously transfer forward **")

print("\n" + "=" * 72)
print("  DONE")
print("=" * 72)
