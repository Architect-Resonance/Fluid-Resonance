"""
M1 ENSTROPHY CANCELLATION TEST ON NS-EVOLVED FIELDS
=====================================================
Three questions:
1. Does signed enstrophy cancellation hold on NS-evolved fields (not just random)?
2. Do cross-helical triads cancel MORE than same-helical? (M1 prediction)
3. What is f_same in the Fano shell (k <= sqrt(3))?

Previous result on random solenoidal fields: 94-98% cancellation, dZ/dt / Z^{3/2} ~ K^{-1.6}
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import time as clock


def P(*args, **kwargs):
    """Unbuffered print."""
    print(*args, **kwargs, flush=True)


# ============================================================
# SOLVER
# ============================================================

class SpectralNS:
    def __init__(self, N, nu):
        self.N = N
        self.nu = nu
        k1d = fftfreq(N, d=1.0/N)
        self.kx, self.ky, self.kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0,0,0] = 1.0
        self.kmag = np.sqrt(self.k2_safe)

        kmax = N // 3
        self.mask = ((np.abs(self.kx) <= kmax) &
                     (np.abs(self.ky) <= kmax) &
                     (np.abs(self.kz) <= kmax)).astype(float)

    def leray_project(self, fx, fy, fz):
        kdotf = self.kx*fx + self.ky*fy + self.kz*fz
        fx = fx - self.kx * kdotf / self.k2_safe
        fy = fy - self.ky * kdotf / self.k2_safe
        fz = fz - self.kz * kdotf / self.k2_safe
        return fx, fy, fz

    def taylor_green_ic(self):
        N = self.N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        ux = np.sin(X) * np.cos(Y) * np.cos(Z)
        uy = -np.cos(X) * np.sin(Y) * np.cos(Z)
        uz = np.zeros_like(X)
        ux_hat = fftn(ux) * self.mask
        uy_hat = fftn(uy) * self.mask
        uz_hat = fftn(uz) * self.mask
        ux_hat, uy_hat, uz_hat = self.leray_project(ux_hat, uy_hat, uz_hat)
        return ux_hat, uy_hat, uz_hat

    def nonlinear(self, ux_hat, uy_hat, uz_hat):
        m = self.mask
        ux = np.real(ifftn(ux_hat * m))
        uy = np.real(ifftn(uy_hat * m))
        uz = np.real(ifftn(uz_hat * m))

        ikx = 1j * self.kx
        iky = 1j * self.ky
        ikz = 1j * self.kz

        dux_dx = np.real(ifftn(ikx * ux_hat * m))
        dux_dy = np.real(ifftn(iky * ux_hat * m))
        dux_dz = np.real(ifftn(ikz * ux_hat * m))
        duy_dx = np.real(ifftn(ikx * uy_hat * m))
        duy_dy = np.real(ifftn(iky * uy_hat * m))
        duy_dz = np.real(ifftn(ikz * uy_hat * m))
        duz_dx = np.real(ifftn(ikx * uz_hat * m))
        duz_dy = np.real(ifftn(iky * uz_hat * m))
        duz_dz = np.real(ifftn(ikz * uz_hat * m))

        nlx = ux*dux_dx + uy*dux_dy + uz*dux_dz
        nly = ux*duy_dx + uy*duy_dy + uz*duy_dz
        nlz = ux*duz_dx + uy*duz_dy + uz*duz_dz

        nlx_hat = fftn(nlx) * m
        nly_hat = fftn(nly) * m
        nlz_hat = fftn(nlz) * m

        nlx_hat, nly_hat, nlz_hat = self.leray_project(nlx_hat, nly_hat, nlz_hat)
        return -nlx_hat, -nly_hat, -nlz_hat

    def rhs(self, ux, uy, uz):
        nlx, nly, nlz = self.nonlinear(ux, uy, uz)
        visc = -self.nu * self.k2
        return nlx + visc*ux, nly + visc*uy, nlz + visc*uz

    def rk4_step(self, ux, uy, uz, dt):
        k1x, k1y, k1z = self.rhs(ux, uy, uz)
        k2x, k2y, k2z = self.rhs(ux+dt/2*k1x, uy+dt/2*k1y, uz+dt/2*k1z)
        k3x, k3y, k3z = self.rhs(ux+dt/2*k2x, uy+dt/2*k2y, uz+dt/2*k2z)
        k4x, k4y, k4z = self.rhs(ux+dt*k3x, uy+dt*k3y, uz+dt*k3z)
        return (ux + dt/6*(k1x + 2*k2x + 2*k3x + k4x),
                uy + dt/6*(k1y + 2*k2y + 2*k3y + k4y),
                uz + dt/6*(k1z + 2*k2z + 2*k3z + k4z))


# ============================================================
# DIAGNOSTICS
# ============================================================

def compute_energy_enstrophy(ns, ux, uy, uz):
    N = ns.N
    norm = 1.0 / N**6
    E = np.sum(np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2) * norm
    Z = np.sum(ns.k2 * (np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2)) * norm
    return E, Z


def compute_enstrophy_production(ns, ux_hat, uy_hat, uz_hat):
    """
    Nonlinear enstrophy production per mode:
      dZ/dt|_NL(k) = |k|^2 * 2 Re[ u_hat*(k) . N_hat(k) ]
    """
    nlx, nly, nlz = ns.nonlinear(ux_hat, uy_hat, uz_hat)
    norm = 1.0 / ns.N**6
    prod_per_mode = ns.k2 * 2 * np.real(
        np.conj(ux_hat) * nlx +
        np.conj(uy_hat) * nly +
        np.conj(uz_hat) * nlz
    ) * norm
    signed_total = np.sum(prod_per_mode)
    unsigned_total = np.sum(np.abs(prod_per_mode))
    return signed_total, unsigned_total, prod_per_mode


def helical_decomposition(ns, ux_hat, uy_hat, uz_hat):
    """
    Decompose each Fourier mode into helical components.
    sigma(k) = (|a+|^2 - |a-|^2) / (|a+|^2 + |a-|^2)
    """
    khat_x = ns.kx / ns.kmag
    khat_y = ns.ky / ns.kmag
    khat_z = ns.kz / ns.kmag

    # e1 = khat x (0,0,1), fallback to khat x (0,1,0)
    e1x = khat_y * 1.0
    e1y = -khat_x * 1.0
    e1z = np.zeros_like(khat_z)
    e1_mag = np.sqrt(e1x**2 + e1y**2 + e1z**2)

    par = e1_mag < 1e-10
    e1x[par] = -khat_z[par]
    e1y[par] = 0.0
    e1z[par] = khat_x[par]
    e1_mag = np.sqrt(e1x**2 + e1y**2 + e1z**2)
    e1_mag[e1_mag < 1e-30] = 1.0
    e1x /= e1_mag; e1y /= e1_mag; e1z /= e1_mag

    # e2 = khat x e1
    e2x = khat_y * e1z - khat_z * e1y
    e2y = khat_z * e1x - khat_x * e1z
    e2z = khat_x * e1y - khat_y * e1x

    u_dot_e1 = ux_hat * e1x + uy_hat * e1y + uz_hat * e1z
    u_dot_e2 = ux_hat * e2x + uy_hat * e2y + uz_hat * e2z

    a_plus  = (u_dot_e1 - 1j * u_dot_e2) / np.sqrt(2)
    a_minus = (u_dot_e1 + 1j * u_dot_e2) / np.sqrt(2)

    ep = np.abs(a_plus)**2
    em = np.abs(a_minus)**2
    total = ep + em
    total_safe = np.where(total < 1e-30, 1.0, total)
    sigma = np.where(total < 1e-30, 0.0, (ep - em) / total_safe)

    return a_plus, a_minus, sigma


def shell_analysis(ns, prod_per_mode, sigma, max_k=None):
    """Per-shell cancellation and helicity structure."""
    kmag = ns.kmag.copy()
    kmag[0,0,0] = 0.0
    if max_k is None:
        max_k = ns.N // 3

    results = []
    for k_target in range(1, max_k + 1):
        shell = (kmag >= k_target - 0.5) & (kmag < k_target + 0.5)
        if not np.any(shell):
            continue

        sp = prod_per_mode[shell]
        signed = np.sum(sp)
        unsigned = np.sum(np.abs(sp))
        cancel = 1.0 - abs(signed) / unsigned if unsigned > 1e-30 else 0.0

        s = sigma[shell]
        mean_abs_sigma = np.mean(np.abs(s))
        energies = np.abs(sp)
        total_e = np.sum(energies)
        f_same = np.sum(energies[np.abs(s) > 0.5]) / total_e if total_e > 1e-30 else 0.0

        same_mask = np.abs(s) > 0.5
        cross_mask = ~same_mask
        same_signed = np.sum(sp[same_mask]) if np.any(same_mask) else 0.0
        same_unsigned = np.sum(np.abs(sp[same_mask])) if np.any(same_mask) else 0.0
        cross_signed = np.sum(sp[cross_mask]) if np.any(cross_mask) else 0.0
        cross_unsigned = np.sum(np.abs(sp[cross_mask])) if np.any(cross_mask) else 0.0

        same_cancel = 1.0 - abs(same_signed)/same_unsigned if same_unsigned > 1e-30 else 0.0
        cross_cancel = 1.0 - abs(cross_signed)/cross_unsigned if cross_unsigned > 1e-30 else 0.0

        results.append({
            'k': k_target, 'signed': signed, 'unsigned': unsigned,
            'cancel': cancel, 'mean_sigma': mean_abs_sigma, 'f_same': f_same,
            'same_cancel': same_cancel, 'cross_cancel': cross_cancel,
            'n_modes': int(np.sum(shell)),
            'n_same': int(np.sum(same_mask)), 'n_cross': int(np.sum(cross_mask)),
        })
    return results


def generate_random_solenoidal(ns, target_E):
    """Random solenoidal field with reality condition and given energy."""
    N = ns.N
    ux_phys = np.random.randn(N, N, N)
    uy_phys = np.random.randn(N, N, N)
    uz_phys = np.random.randn(N, N, N)
    rx = fftn(ux_phys) * ns.mask
    ry = fftn(uy_phys) * ns.mask
    rz = fftn(uz_phys) * ns.mask
    rx, ry, rz = ns.leray_project(rx, ry, rz)
    E_r = np.sum(np.abs(rx)**2 + np.abs(ry)**2 + np.abs(rz)**2) / N**6
    if E_r > 1e-30:
        sc = np.sqrt(target_E / E_r)
        rx *= sc; ry *= sc; rz *= sc
    return rx, ry, rz


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(N, Re, snapshots, dt, n_random=100):
    nu = 1.0 / Re
    ns = SpectralNS(N, nu)
    ux, uy, uz = ns.taylor_green_ic()

    E0, Z0 = compute_energy_enstrophy(ns, ux, uy, uz)
    scale = 1.0 / np.sqrt(E0)
    ux *= scale; uy *= scale; uz *= scale
    E0, Z0 = compute_energy_enstrophy(ns, ux, uy, uz)
    P(f"  IC: E={E0:.6f}, Z={Z0:.6f}")

    t = 0.0
    snap_idx = 0
    step = 0
    t_start = clock.time()
    all_results = []

    while snap_idx < len(snapshots):
        if t >= snapshots[snap_idx] - dt/2:
            E_now, Z_now = compute_energy_enstrophy(ns, ux, uy, uz)
            signed, unsigned, prod = compute_enstrophy_production(ns, ux, uy, uz)
            cancel = 1.0 - abs(signed)/unsigned if unsigned > 1e-30 else 0.0
            ratio_Z = abs(signed) / Z_now**1.5 if Z_now > 1e-30 else 0.0

            _, _, sigma = helical_decomposition(ns, ux, uy, uz)
            shells = shell_analysis(ns, prod, sigma)

            P(f"\n  t={t:.2f}: E={E_now:.6f}, Z={Z_now:.6f}")
            P(f"    Signed dZ/dt_NL   = {signed:.8f}")
            P(f"    Unsigned |dZ/dt|   = {unsigned:.8f}")
            P(f"    Cancellation      = {cancel*100:.1f}%")
            P(f"    |dZ/dt|/Z^(3/2)   = {ratio_Z:.8f}")

            P(f"    {'k':>3} {'cancel%':>8} {'|s|':>7} {'f_same':>7} {'same_c%':>8} {'cross_c%':>9} {'n':>5}")
            for s in shells[:12]:
                P(f"    {s['k']:3d} {s['cancel']*100:7.1f}% {s['mean_sigma']:7.3f} {s['f_same']:7.3f} "
                  f"{s['same_cancel']*100:7.1f}% {s['cross_cancel']*100:8.1f}% {s['n_modes']:5d}")

            # Random solenoidal comparison
            rc_list, rr_list = [], []
            for _ in range(n_random):
                rx, ry, rz = generate_random_solenoidal(ns, E_now)
                s_r, u_r, _ = compute_enstrophy_production(ns, rx, ry, rz)
                _, Z_r = compute_energy_enstrophy(ns, rx, ry, rz)
                c_r = 1.0 - abs(s_r)/u_r if u_r > 1e-30 else 0.0
                r_r = abs(s_r) / Z_r**1.5 if Z_r > 1e-30 else 0.0
                rc_list.append(c_r)
                rr_list.append(r_r)

            rc = np.array(rc_list); rr = np.array(rr_list)
            P(f"\n    RANDOM ({n_random} samples, same E={E_now:.6f}):")
            P(f"      Cancel: {rc.mean()*100:.1f}% +/- {rc.std()*100:.1f}%  [min {rc.min()*100:.1f}%, max {rc.max()*100:.1f}%]")
            P(f"      |dZ/dt|/Z^1.5: {rr.mean():.8f} +/- {rr.std():.8f}")
            P(f"    NS-evolved:")
            P(f"      Cancel: {cancel*100:.1f}%")
            P(f"      |dZ/dt|/Z^1.5: {ratio_Z:.8f}")
            diff_sigma = (cancel - rc.mean()) / rc.std() if rc.std() > 1e-10 else 0.0
            P(f"      Deviation from random: {diff_sigma:+.2f} sigma")

            all_results.append({
                't': t, 'E': E_now, 'Z': Z_now,
                'signed': signed, 'unsigned': unsigned,
                'cancel': cancel, 'ratio_Z': ratio_Z,
                'random_cancel_mean': rc.mean(), 'random_cancel_std': rc.std(),
                'random_ratio_mean': rr.mean(), 'random_ratio_std': rr.std(),
                'shells': shells,
            })
            snap_idx += 1

        if snap_idx >= len(snapshots):
            break

        ux, uy, uz = ns.rk4_step(ux, uy, uz, dt)
        t += dt
        step += 1

        if step % 500 == 0:
            elapsed = clock.time() - t_start
            P(f"    [step {step}, t={t:.3f}, elapsed {elapsed:.0f}s]")

    elapsed = clock.time() - t_start
    P(f"\n  Total time: {elapsed:.1f}s, {step} steps")
    return all_results


# ============================================================
# RUN
# ============================================================

P("=" * 80)
P("M1 TEST: SIGNED ENSTROPHY CANCELLATION ON NS-EVOLVED FIELDS")
P("=" * 80)

all_experiments = {}

# --- N=32, Re=400 ---
P("\n" + "=" * 80)
P("EXPERIMENT 1: N=32, Re=400")
P("=" * 80)
all_experiments['N32_Re400'] = run_experiment(
    N=32, Re=400,
    snapshots=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    dt=0.002, n_random=100
)

# --- N=32, Re=1600 ---
P("\n" + "=" * 80)
P("EXPERIMENT 2: N=32, Re=1600")
P("=" * 80)
all_experiments['N32_Re1600'] = run_experiment(
    N=32, Re=1600,
    snapshots=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    dt=0.001, n_random=100
)

# --- N=64, Re=400 (fewer snapshots and samples for speed) ---
P("\n" + "=" * 80)
P("EXPERIMENT 3: N=64, Re=400")
P("=" * 80)
all_experiments['N64_Re400'] = run_experiment(
    N=64, Re=400,
    snapshots=[0.0, 1.0, 2.0, 3.0],
    dt=0.001, n_random=30
)


# ============================================================
# SUMMARY
# ============================================================

P("\n\n" + "=" * 80)
P("GRAND SUMMARY")
P("=" * 80)

for label, results in all_experiments.items():
    P(f"\n--- {label} ---")
    P(f"  {'t':>5} {'NS_cancel%':>11} {'Rnd_cancel%':>12} {'dev_sigma':>10} {'NS_ratio':>12} {'Rnd_ratio':>12}")
    for r in results:
        dev = (r['cancel']-r['random_cancel_mean'])/r['random_cancel_std'] if r['random_cancel_std']>0 else 0
        P(f"  {r['t']:5.1f} {r['cancel']*100:10.1f}% {r['random_cancel_mean']*100:10.1f}% "
          f"{dev:+10.2f} {r['ratio_Z']:12.8f} {r['random_ratio_mean']:12.8f}")

# Fano shell summary
P("\n\nFANO SHELL (k<=3) SUMMARY:")
P(f"  {'Expt':>12} {'t':>5} {'k':>3} {'cancel%':>8} {'f_same':>7} {'same_c%':>8} {'cross_c%':>9}")
for label, results in all_experiments.items():
    for r in results:
        for s in r['shells']:
            if s['k'] <= 3:
                P(f"  {label:>12} {r['t']:5.1f} {s['k']:3d} {s['cancel']*100:7.1f}% {s['f_same']:7.3f} "
                  f"{s['same_cancel']*100:7.1f}% {s['cross_cancel']*100:8.1f}%")

# Cross vs same helical cancellation summary
P("\n\nCROSS vs SAME HELICAL CANCELLATION (all shells aggregated):")
for label, results in all_experiments.items():
    P(f"\n  --- {label} ---")
    for r in results:
        all_same_c = [s['same_cancel'] for s in r['shells'] if s['n_same'] > 0]
        all_cross_c = [s['cross_cancel'] for s in r['shells'] if s['n_cross'] > 0]
        if all_same_c and all_cross_c:
            P(f"  t={r['t']:.1f}: same_cancel={np.mean(all_same_c)*100:.1f}%, "
              f"cross_cancel={np.mean(all_cross_c)*100:.1f}%, "
              f"cross > same: {np.mean(all_cross_c) > np.mean(all_same_c)}")

P("\n\nDONE.")
