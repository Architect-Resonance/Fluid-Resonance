"""
S110-M2 CRITICAL: Berry holonomy with continuous phases.

Re-run holonomy measurement with:
1. RANDOM initial conditions (break all TG discrete symmetries)
2. ABC flow (Arnold-Beltrami-Childress)

Test: does continuous holonomy / 4*pi converge to R ~ 1.8573
instead of the Z_2-quantized 15/8 = 1.875?
"""

import numpy as np
from numpy.fft import fftn, ifftn
from itertools import product as iproduct
import time


def leray_project(u_hat, KX, KY, KZ):
    """Project velocity field to be divergence-free."""
    K2 = KX**2 + KY**2 + KZ**2
    K2_safe = K2.copy()
    K2_safe[0, 0, 0] = 1
    k_dot_u = KX * u_hat[0] + KY * u_hat[1] + KZ * u_hat[2]
    u_hat[0] -= KX * k_dot_u / K2_safe
    u_hat[1] -= KY * k_dot_u / K2_safe
    u_hat[2] -= KZ * k_dot_u / K2_safe
    u_hat[:, 0, 0, 0] = 0
    return u_hat


def random_ic(N, seed=42, k_max_ic=5):
    """Random divergence-free initial condition with energy in shells 1..k_max_ic.
    Properly enforces Hermitian symmetry for real velocity field."""
    rng = np.random.RandomState(seed)

    # Build random REAL velocity field in physical space
    u = np.zeros((3, N, N, N))
    for c in range(3):
        u[c] = rng.randn(N, N, N)

    # Project to divergence-free in Fourier space and filter to low k
    # Use physical wavenumbers for Leray projection
    kx = np.fft.fftfreq(N, d=1.0/(2*np.pi*N)).astype(np.float64)
    KZ, KY, KX = np.meshgrid(kx, kx, kx, indexing='ij')

    u_hat = np.array([fftn(u[i]) for i in range(3)])
    u_hat = leray_project(u_hat, KX, KY, KZ)

    # Use INTEGER wavenumbers for filtering (k_max_ic is in integer units)
    kx_int = np.fft.fftfreq(N, d=1.0/N).astype(np.float64)
    KZi, KYi, KXi = np.meshgrid(kx_int, kx_int, kx_int, indexing='ij')
    K2_int = KXi**2 + KYi**2 + KZi**2
    mask_kill = K2_int > k_max_ic**2
    u_hat[:, mask_kill] = 0

    # Back to physical, normalize
    u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    E = 0.5 * np.mean(np.sum(u**2, axis=0))
    if E > 0:
        u *= np.sqrt(0.5 / E)

    return u


def abc_flow(N, A=1.0, B=0.8, C=0.6):
    """Arnold-Beltrami-Childress flow — eigenfunction of curl, fully 3D."""
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    Y, X, Z = np.meshgrid(x, x, x, indexing='ij')
    u = np.zeros((3, N, N, N))
    u[0] = A * np.sin(Z) + C * np.cos(Y)
    u[1] = B * np.sin(X) + A * np.cos(Z)
    u[2] = C * np.sin(Y) + B * np.cos(X)
    return u


def taylor_green_ic(N, A=1.0):
    """Taylor-Green vortex (for comparison)."""
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    Y, X, Z = np.meshgrid(x, x, x, indexing='ij')
    u = np.zeros((3, N, N, N))
    u[0] = A * np.sin(X) * np.cos(Y) * np.cos(Z)
    u[1] = -A * np.cos(X) * np.sin(Y) * np.cos(Z)
    u[2] = 0.0
    return u


def evolve_ns(u, nu, dt, t_target, N, snap_times=[1.0, 2.0, 3.0], print_interval=500):
    """Evolve NS with RK4, dealiased. Return snapshots."""
    kx = np.fft.fftfreq(N, d=1.0/(2*np.pi*N)).astype(np.float64)
    KZ, KY, KX = np.meshgrid(kx, kx, kx, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2_safe = K2.copy()
    K2_safe[0, 0, 0] = 1
    dealias = np.ones_like(K2)
    kmax = N // 3
    dealias[np.sqrt(K2) > kmax] = 0

    def rhs(u_hat):
        u_hat_d = u_hat * dealias
        u_phys = np.array([np.real(ifftn(u_hat_d[i])) for i in range(3)])
        omega = np.zeros_like(u_phys)
        omega[0] = np.real(ifftn(1j * (KY * u_hat_d[2] - KZ * u_hat_d[1])))
        omega[1] = np.real(ifftn(1j * (KZ * u_hat_d[0] - KX * u_hat_d[2])))
        omega[2] = np.real(ifftn(1j * (KX * u_hat_d[1] - KY * u_hat_d[0])))
        nl = np.zeros_like(u_phys)
        nl[0] = u_phys[1] * omega[2] - u_phys[2] * omega[1]
        nl[1] = u_phys[2] * omega[0] - u_phys[0] * omega[2]
        nl[2] = u_phys[0] * omega[1] - u_phys[1] * omega[0]
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
            E = 0.5 * np.sum(np.abs(u_hat)**2) / N**6
            print(f"  [step {step}, t={t:.3f}, E={E:.4f}, elapsed={elapsed:.0f}s]", flush=True)

        for t_snap in snap_times:
            if abs(t - t_snap) < dt/2 and t_snap not in snapshots:
                snapshots[t_snap] = u_hat.copy()
                print(f"  ** Snapshot at t={t_snap:.1f} **", flush=True)

    if t_target not in snapshots:
        snapshots[t_target] = u_hat.copy()

    return snapshots


# ============================================================
# FANO PLANE
# ============================================================

FANO_POINTS = [
    (1,0,0), (0,1,0), (0,0,1),
    (1,1,0), (1,0,1), (0,1,1),
    (1,1,1)
]

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

# Find all line 3-cycles
LINE_CYCLES = []
for i in range(len(FANO_LINES)):
    s1 = set(FANO_LINES[i])
    for j in range(i+1, len(FANO_LINES)):
        s2 = set(FANO_LINES[j])
        if not (s1 & s2):
            continue
        for k in range(j+1, len(FANO_LINES)):
            s3 = set(FANO_LINES[k])
            if (s2 & s3) and (s3 & s1):
                LINE_CYCLES.append((i, j, k))


def gf2_class(k):
    return (abs(k[0]) % 2, abs(k[1]) % 2, abs(k[2]) % 2)


def get_mode(u_hat, kx, ky, kz, N):
    return np.array([u_hat[c][kx % N, ky % N, kz % N] for c in range(3)])


def triad_phase(u_hat, k, p, q, N):
    uk = get_mode(u_hat, *k, N)
    up = get_mode(u_hat, *p, N)
    uq = get_mode(u_hat, *q, N)
    cross = np.cross(up, uq)
    triple = np.dot(uk, cross)
    if abs(triple) < 1e-30:
        return 0.0, 0.0
    return np.angle(triple), abs(triple)


def measure_holonomy(u_hat, N, k_max=3):
    """Full holonomy measurement. Returns dict with all results."""
    # Build mode catalog
    class_modes = {p: [] for p in FANO_POINTS}
    for kx, ky, kz in iproduct(range(-k_max, k_max+1), repeat=3):
        k2 = kx**2 + ky**2 + kz**2
        if 0 < k2 <= k_max**2:
            c = gf2_class((kx, ky, kz))
            if c != (0,0,0) and c in class_modes:
                class_modes[c].append((kx, ky, kz))

    # Line phases
    # IMPORTANT: only count canonical triads to avoid conjugate-pair cancellation
    # Canonical = first mode k has first nonzero component positive
    def is_canonical(k):
        """Return True if k is in the canonical half (avoid double-counting conjugates)."""
        for ki in k:
            if ki > 0:
                return True
            if ki < 0:
                return False
        return False  # k = (0,0,0) — shouldn't happen

    line_phases = {}
    line_details = {}
    for li, line in enumerate(FANO_LINES):
        a, b, c = line
        phases = []
        mags = []
        for k in class_modes[a]:
            if not is_canonical(k):
                continue  # skip conjugate half
            for p in class_modes[b]:
                q = (-(k[0]+p[0]), -(k[1]+p[1]), -(k[2]+p[2]))
                qc = gf2_class(q)
                q2 = q[0]**2 + q[1]**2 + q[2]**2
                if qc == c and 0 < q2 <= k_max**2:
                    ph, mag = triad_phase(u_hat, k, p, q, N)
                    phases.append(ph)
                    mags.append(mag)
        if phases:
            phases = np.array(phases)
            mags = np.array(mags)
            # Magnitude-weighted circular mean
            w = mags / (np.sum(mags) + 1e-30)
            ms = np.sum(w * np.sin(phases))
            mc = np.sum(w * np.cos(phases))
            mean_ph = np.arctan2(ms, mc)
            R = np.sqrt(ms**2 + mc**2)
            # Unweighted
            ms_u = np.mean(np.sin(phases))
            mc_u = np.mean(np.cos(phases))
            mean_ph_u = np.arctan2(ms_u, mc_u)
            R_u = np.sqrt(ms_u**2 + mc_u**2)

            line_phases[li] = mean_ph
            line_details[li] = {
                'phase_w': mean_ph, 'R_w': R,
                'phase_u': mean_ph_u, 'R_u': R_u,
                'n': len(phases),
                'phase_std': np.std(phases)
            }

    # Holonomy around cycles
    holonomies = []
    for i, j, k in LINE_CYCLES:
        if i in line_phases and j in line_phases and k in line_phases:
            phi = line_phases[i] + line_phases[j] + line_phases[k]
            holonomies.append(phi)  # NOT wrapped — keep continuous sum

    total_holonomy = np.sum(holonomies) if holonomies else 0.0
    ratio_4pi = total_holonomy / (4 * np.pi)

    # Phase coherence
    n_forward = sum(1 for li in line_phases if np.sin(line_phases[li]) > 0)

    return {
        'line_phases': line_phases,
        'line_details': line_details,
        'holonomies': np.array(holonomies) if holonomies else np.array([]),
        'total_holonomy': total_holonomy,
        'ratio_4pi': ratio_4pi,
        'n_forward': n_forward
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("  CRITICAL TEST: Continuous Berry Holonomy")
print("  Target: holonomy / 4pi = R ~ 1.8573 (vs Z_2 quantized 15/8 = 1.875)")
print("=" * 72)
print(f"  Fano: {len(FANO_LINES)} lines, {len(LINE_CYCLES)} 3-cycles")

N = 32
Re = 400
nu = 1.0 / Re
dt = 0.001
R_target = 13.0 / 7.0  # 1.857142...

experiments = [
    ("Taylor-Green (control)", taylor_green_ic(N)),
    ("ABC flow (A=1,B=0.8,C=0.6)", abc_flow(N, 1.0, 0.8, 0.6)),
    ("Random seed=42", random_ic(N, seed=42)),
    ("Random seed=137", random_ic(N, seed=137)),
    ("Random seed=2718", random_ic(N, seed=2718)),
]

results_summary = []

for ic_name, u0 in experiments:
    print(f"\n{'='*72}")
    print(f"  {ic_name}, Re={Re}")
    print(f"{'='*72}")

    snapshots = evolve_ns(u0, nu, dt, 3.0, N, snap_times=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    for t_snap in sorted(snapshots.keys()):
        u_hat = snapshots[t_snap]
        res = measure_holonomy(u_hat, N)

        # Print line phases
        print(f"\n  t={t_snap:.1f}:")
        any_continuous = False
        for li in sorted(res['line_details'].keys()):
            d = res['line_details'][li]
            phase_deg = np.degrees(d['phase_w'])
            # Check if phase is NOT 0 or 180
            if abs(abs(d['phase_w']) - np.pi) > 0.1 and abs(d['phase_w']) > 0.1:
                any_continuous = True
            print(f"    L{li}: phi={d['phase_w']:+7.4f} ({phase_deg:+7.1f}deg) "
                  f"R_w={d['R_w']:.4f} R_u={d['R_u']:.4f} std={d['phase_std']:.3f} n={d['n']}")

        ratio = res['ratio_4pi']
        print(f"\n    Total holonomy: {res['total_holonomy']:+.4f}")
        print(f"    Ratio to 4pi: {ratio:+.6f}")
        print(f"    Target R = 13/7 = {R_target:.6f}")
        print(f"    15/8 = {15/8:.6f}")
        print(f"    Error vs R: {abs(ratio) - R_target:+.6f} ({(abs(ratio)/R_target - 1)*100:+.2f}%)")
        print(f"    Error vs 15/8: {abs(ratio) - 15/8:+.6f} ({(abs(ratio)/(15/8) - 1)*100:+.2f}%)")
        print(f"    Forward lines: {res['n_forward']}/7")
        print(f"    Continuous phases: {'YES' if any_continuous else 'NO (Z_2 locked)'}")

        results_summary.append({
            'ic': ic_name, 't': t_snap,
            'ratio': ratio, 'continuous': any_continuous,
            'n_forward': res['n_forward']
        })

# Final summary
print(f"\n{'='*72}")
print(f"  SUMMARY")
print(f"{'='*72}")
print(f"{'IC':35s} {'t':>4s} {'ratio':>10s} {'vs R':>8s} {'vs 15/8':>8s} {'cont?':>6s}")
for r in results_summary:
    err_R = abs(r['ratio']) - R_target
    err_158 = abs(r['ratio']) - 15/8
    print(f"{r['ic']:35s} {r['t']:4.1f} {r['ratio']:+10.6f} {err_R:+8.4f} {err_158:+8.4f} {'YES' if r['continuous'] else 'no':>6s}")

print(f"\nR = 13/7 = {R_target:.6f}")
print(f"15/8    = {15/8:.6f}")
print(f"Diff    = 1/56 = {1/56:.6f}")
