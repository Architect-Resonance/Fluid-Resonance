"""
WANDERER S107-W: Three Information-Theoretic Measurements
=========================================================
1. Energy flux Pi_K (proxy for I_K) — does it scale as 1/tau_K?
2. Fano bottleneck — is k<=3 anomalously LOW in information transfer?
3. <sin(theta)> across shells — Von Mises prediction test

Based on Tanogami & Araki 2025 (arXiv:2408.03635):
  I_K = sigma_dot_K (information flow = phase-space contraction rate)
  I_K <= C * K * <|u_K|^p>^{1/p}

And Benavides & Bustamante (arXiv:2507.03397):
  Triad phase theta_n controls cascade direction: Pi ~ sin(theta)
  Von Mises statistics: P(theta) ~ exp[kappa * sin(theta)]
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock


def P(*args, **kwargs):
    print(*args, **kwargs, flush=True)


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
        return self.leray_project(ux_hat, uy_hat, uz_hat)

    def nonlinear(self, ux_hat, uy_hat, uz_hat):
        m = self.mask
        ux = np.real(ifftn(ux_hat * m))
        uy = np.real(ifftn(uy_hat * m))
        uz = np.real(ifftn(uz_hat * m))
        ikx, iky, ikz = 1j*self.kx, 1j*self.ky, 1j*self.kz
        dux_dx = np.real(ifftn(ikx*ux_hat*m))
        dux_dy = np.real(ifftn(iky*ux_hat*m))
        dux_dz = np.real(ifftn(ikz*ux_hat*m))
        duy_dx = np.real(ifftn(ikx*uy_hat*m))
        duy_dy = np.real(ifftn(iky*uy_hat*m))
        duy_dz = np.real(ifftn(ikz*uy_hat*m))
        duz_dx = np.real(ifftn(ikx*uz_hat*m))
        duz_dy = np.real(ifftn(iky*uz_hat*m))
        duz_dz = np.real(ifftn(ikz*uz_hat*m))
        nlx = -(ux*dux_dx + uy*dux_dy + uz*dux_dz)
        nly = -(ux*duy_dx + uy*duy_dy + uz*duy_dz)
        nlz = -(ux*duz_dx + uy*duz_dy + uz*duz_dz)
        nlx_hat = fftn(nlx) * m
        nly_hat = fftn(nly) * m
        nlz_hat = fftn(nlz) * m
        return self.leray_project(nlx_hat, nly_hat, nlz_hat)

    def rk4_step(self, ux, uy, uz, dt):
        def rhs(ux, uy, uz):
            nlx, nly, nlz = self.nonlinear(ux, uy, uz)
            return (nlx - self.nu*self.k2*ux,
                    nly - self.nu*self.k2*uy,
                    nlz - self.nu*self.k2*uz)
        k1x, k1y, k1z = rhs(ux, uy, uz)
        k2x, k2y, k2z = rhs(ux+dt/2*k1x, uy+dt/2*k1y, uz+dt/2*k1z)
        k3x, k3y, k3z = rhs(ux+dt/2*k2x, uy+dt/2*k2y, uz+dt/2*k2z)
        k4x, k4y, k4z = rhs(ux+dt*k3x, uy+dt*k3y, uz+dt*k3z)
        ux_new = ux + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
        uy_new = uy + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
        uz_new = uz + dt/6*(k1z + 2*k2z + 2*k3z + k4z)
        return self.leray_project(ux_new, uy_new, uz_new)


def compute_shell_energy_flux(ns, ux_hat, uy_hat, uz_hat, max_k=None):
    """
    Compute energy flux Pi_K through each shell K.
    Pi_K = sum over modes k with |k| in shell K of Re[u*(k) . NL(k)]
    This is the energy transfer INTO shell K from all other shells.

    Also compute phase-space contraction proxy:
    sigma_dot_K ~ sum of |partial F_i / partial x_i| for modes at shell K
    """
    nlx, nly, nlz = ns.nonlinear(ux_hat, uy_hat, uz_hat)
    norm = 1.0 / ns.N**6

    # Per-mode energy transfer: Re[u*(k) . NL(k)]
    transfer_per_mode = 2 * np.real(
        np.conj(ux_hat)*nlx + np.conj(uy_hat)*nly + np.conj(uz_hat)*nlz
    ) * norm

    # Per-mode energy
    E_per_mode = (np.abs(ux_hat)**2 + np.abs(uy_hat)**2 + np.abs(uz_hat)**2) * norm

    kmag = ns.kmag.copy()
    kmag[0,0,0] = 0.0
    if max_k is None:
        max_k = ns.N // 3

    results = []
    for k_target in range(1, max_k + 1):
        shell = (kmag >= k_target - 0.5) & (kmag < k_target + 0.5)
        if not np.any(shell):
            results.append({'k': k_target, 'Pi': 0, 'E_k': 0, 'u_k': 0, 'tau_k': np.inf, 'n': 0})
            continue

        Pi_k = np.sum(transfer_per_mode[shell])  # energy flux into shell
        E_k = np.sum(E_per_mode[shell])
        u_k = np.sqrt(E_k) if E_k > 0 else 0  # characteristic velocity
        tau_k = 1.0 / (k_target * u_k) if u_k > 1e-30 else np.inf  # eddy turnover time
        n_modes = int(np.sum(shell))

        results.append({
            'k': k_target, 'Pi': Pi_k, 'E_k': E_k, 'u_k': u_k,
            'tau_k': tau_k, 'n': n_modes
        })

    return results


def compute_cumulative_flux(ns, ux_hat, uy_hat, uz_hat, max_k=None):
    """
    Cumulative energy flux Pi(K) = rate of energy transfer from modes |k|<K to |k|>K.
    This is the standard energy flux used in Kolmogorov theory.
    Pi(K) = -sum_{|k|<K} Re[u*(k) . NL(k)]
    """
    nlx, nly, nlz = ns.nonlinear(ux_hat, uy_hat, uz_hat)
    norm = 1.0 / ns.N**6

    transfer = 2 * np.real(
        np.conj(ux_hat)*nlx + np.conj(uy_hat)*nly + np.conj(uz_hat)*nlz
    ) * norm

    kmag = ns.kmag.copy()
    kmag[0,0,0] = 0.0
    if max_k is None:
        max_k = ns.N // 3

    fluxes = []
    for K in range(1, max_k + 1):
        low = kmag < K + 0.5
        Pi_K = -np.sum(transfer[low])  # minus sign: flux OUT of low modes
        fluxes.append((K, Pi_K))

    return fluxes


def compute_triad_phases(ns, ux_hat, uy_hat, uz_hat, max_k=8):
    """
    Compute triad phases for triads k + p = q.
    The "triad phase" theta = arg(T(k,p,q)) where T is the triple correlation.

    For scalar simplification: project velocity onto k-direction,
    theta(k,p,q) = phase(u_k) + phase(u_p) - phase(u_q)

    We use the helical projection: a(k) = u_hat(k) . e_+(k)
    Then theta = arg(a_k) + arg(a_p) - arg(a_q)

    Collect sin(theta) statistics per shell.
    """
    N = ns.N
    kmag = ns.kmag.copy()
    kmag[0,0,0] = 0.0

    # Helical projection: a(k) = u . (e1 + i*e2)/sqrt(2)
    khat_x = ns.kx / ns.kmag
    khat_y = ns.ky / ns.kmag
    khat_z = ns.kz / ns.kmag

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

    e2x = khat_y*e1z - khat_z*e1y
    e2y = khat_z*e1x - khat_x*e1z
    e2z = khat_x*e1y - khat_y*e1x

    # Helical amplitude (positive helicity)
    a_plus = (ux_hat*e1x + uy_hat*e1y + uz_hat*e1z
              - 1j*(ux_hat*e2x + uy_hat*e2y + uz_hat*e2z)) / np.sqrt(2)

    # Phase of helical amplitude
    phase = np.angle(a_plus)
    amplitude = np.abs(a_plus)

    # Build shell lists
    kmax = min(max_k, N//3)
    shell_modes = {}
    k1d = fftfreq(N, d=1.0/N)
    for ix in range(N):
        for iy in range(N):
            for iz in range(N):
                kv = np.array([k1d[ix], k1d[iy], k1d[iz]])
                km = np.sqrt(np.sum(kv**2))
                if km < 0.5 or km > kmax + 0.5:
                    continue
                shell = int(round(km))
                if shell < 1 or shell > kmax:
                    continue
                if shell not in shell_modes:
                    shell_modes[shell] = []
                shell_modes[shell].append((ix, iy, iz, kv, phase[ix,iy,iz], amplitude[ix,iy,iz]))

    # Sample triads k + p = q within nearby shells
    # For each shell pair (s1, s2), find triads where k in s1, p in s2, q = k+p
    triad_phases = {s: [] for s in range(1, kmax+1)}

    for s1 in range(1, min(kmax//2+1, kmax)):
        modes1 = shell_modes.get(s1, [])
        if len(modes1) == 0:
            continue
        # Sample subset to keep computation tractable
        if len(modes1) > 50:
            idx1 = np.random.choice(len(modes1), 50, replace=False)
            modes1_sample = [modes1[i] for i in idx1]
        else:
            modes1_sample = modes1

        for s2 in range(s1, min(s1+3, kmax)):  # nearby shells only (local triads)
            modes2 = shell_modes.get(s2, [])
            if len(modes2) == 0:
                continue
            if len(modes2) > 50:
                idx2 = np.random.choice(len(modes2), 50, replace=False)
                modes2_sample = [modes2[i] for i in idx2]
            else:
                modes2_sample = modes2

            for ix1, iy1, iz1, kv1, ph1, amp1 in modes1_sample:
                if amp1 < 1e-20:
                    continue
                for ix2, iy2, iz2, kv2, ph2, amp2 in modes2_sample:
                    if amp2 < 1e-20:
                        continue
                    # q = k + p
                    qv = kv1 + kv2
                    qm = np.sqrt(np.sum(qv**2))
                    sq = int(round(qm))
                    if sq < 1 or sq > kmax:
                        continue

                    # Get phase at q
                    iq = (int(round(qv[0])) % N, int(round(qv[1])) % N, int(round(qv[2])) % N)
                    ph3 = phase[iq]
                    amp3 = amplitude[iq]
                    if amp3 < 1e-20:
                        continue

                    # Triad phase: theta = phi_k + phi_p - phi_q
                    theta = ph1 + ph2 - ph3
                    # Weight by amplitude product
                    weight = amp1 * amp2 * amp3

                    # Assign to the output shell (q)
                    triad_phases[sq].append((theta, weight))

    # Compute statistics per shell
    results = {}
    for s in range(1, kmax+1):
        phases = triad_phases.get(s, [])
        if len(phases) == 0:
            results[s] = {'n_triads': 0, 'mean_sin': 0, 'mean_cos': 0,
                          'R': 0, 'mean_sin_weighted': 0}
            continue

        thetas = np.array([p[0] for p in phases])
        weights = np.array([p[1] for p in phases])
        weights /= np.sum(weights) if np.sum(weights) > 0 else 1.0

        results[s] = {
            'n_triads': len(thetas),
            'mean_sin': np.mean(np.sin(thetas)),
            'mean_cos': np.mean(np.cos(thetas)),
            'R': np.sqrt(np.mean(np.sin(thetas))**2 + np.mean(np.cos(thetas))**2),
            'mean_sin_weighted': np.sum(weights * np.sin(thetas)),
        }

    return results


# ============================================================
# MAIN
# ============================================================

P("="*72)
P("  WANDERER S107-W: Information Flow Measurements")
P("="*72)

experiments = [
    ('N32_Re400', 32, 400, 0.001, 3.0),
    ('N32_Re1600', 32, 1600, 0.0005, 3.0),
    ('N64_Re400', 64, 400, 0.001, 3.0),
]

all_results = {}

for label, N, Re, dt, t_final in experiments:
    P(f"\n--- {label} ---")
    nu = 1.0 / Re
    ns = SpectralNS(N, nu)
    ux, uy, uz = ns.taylor_green_ic()

    n_steps = int(t_final / dt)
    snapshots = [1.0, 2.0, 3.0]
    snap_steps = {int(t/dt): t for t in snapshots}

    t0 = clock.time()
    exp_results = {}

    for step in range(1, n_steps + 1):
        ux, uy, uz = ns.rk4_step(ux, uy, uz, dt)

        if step in snap_steps:
            t = snap_steps[step]
            elapsed = clock.time() - t0

            # Measurement 1: Shell energy flux
            shell_flux = compute_shell_energy_flux(ns, ux, uy, uz)

            # Measurement 2: Cumulative flux
            cum_flux = compute_cumulative_flux(ns, ux, uy, uz)

            # Measurement 3: Triad phases
            max_k_phase = min(N//3, 12)
            triad_stats = compute_triad_phases(ns, ux, uy, uz, max_k=max_k_phase)

            exp_results[t] = {
                'shell_flux': shell_flux,
                'cum_flux': cum_flux,
                'triad_stats': triad_stats,
            }
            P(f"  t={t:.1f}: measured (elapsed={elapsed:.0f}s)")

        if step % 500 == 0 and step not in snap_steps:
            elapsed = clock.time() - t0
            P(f"  [step {step}, t={step*dt:.3f}, elapsed={elapsed:.0f}s]")

    all_results[label] = exp_results

# ============================================================
# ANALYSIS
# ============================================================

P("\n" + "="*72)
P("  MEASUREMENT 1: Energy Flux vs 1/tau_K")
P("="*72)

for label, exp_results in all_results.items():
    t_key = max(exp_results.keys())
    shell_flux = exp_results[t_key]['shell_flux']
    P(f"\n  {label} at t={t_key:.1f}:")
    P(f"  {'k':>4}  {'Pi_k':>12}  {'E_k':>12}  {'u_k':>10}  {'tau_k':>10}  {'|Pi|*tau':>10}  {'n':>6}")
    for sf in shell_flux:
        k = sf['k']
        if sf['E_k'] < 1e-30:
            continue
        Pi_tau = abs(sf['Pi']) * sf['tau_k'] if sf['tau_k'] < 1e10 else 0
        P(f"  {k:4d}  {sf['Pi']:12.6f}  {sf['E_k']:12.6f}  {sf['u_k']:10.6f}  "
          f"{sf['tau_k']:10.4f}  {Pi_tau:10.6f}  {sf['n']:6d}")

    # Tanogami prediction: I_K ~ C/tau_K, so |Pi_K| * tau_K should be roughly constant
    P(f"\n  Tanogami test: if |Pi_K| ~ 1/tau_K, then |Pi_K|*tau_K = const")

P("\n" + "="*72)
P("  MEASUREMENT 2: Cumulative Flux Pi(K) — Fano Bottleneck Test")
P("="*72)

for label, exp_results in all_results.items():
    t_key = max(exp_results.keys())
    cum_flux = exp_results[t_key]['cum_flux']
    P(f"\n  {label} at t={t_key:.1f}:")
    P(f"  {'K':>4}  {'Pi(K)':>12}  {'note':>20}")
    for K, Pi_K in cum_flux:
        note = "<-- FANO" if K <= 3 else ""
        P(f"  {K:4d}  {Pi_K:12.8f}  {note:>20}")

P("\n" + "="*72)
P("  MEASUREMENT 3: Triad Phases <sin(theta)> — Von Mises Test")
P("="*72)

for label, exp_results in all_results.items():
    t_key = max(exp_results.keys())
    triad_stats = exp_results[t_key]['triad_stats']
    P(f"\n  {label} at t={t_key:.1f}:")
    P(f"  {'k':>4}  {'n_triads':>10}  {'<sin>':>10}  {'<cos>':>10}  {'R':>10}  {'<sin>_w':>10}")
    for k in sorted(triad_stats.keys()):
        ts = triad_stats[k]
        if ts['n_triads'] == 0:
            continue
        P(f"  {k:4d}  {ts['n_triads']:10d}  {ts['mean_sin']:10.4f}  {ts['mean_cos']:10.4f}  "
          f"{ts['R']:10.4f}  {ts['mean_sin_weighted']:10.4f}")

    # Von Mises prediction: <sin(theta)> > 0 for forward cascade
    P(f"\n  Von Mises test: <sin(theta)> > 0 means forward cascade")
    P(f"  <sin(theta)> < 1 means partial alignment (not saturated)")

# ============================================================
# PLOTS
# ============================================================

P("\n  Generating plots...")
outdir = 'Fluid-Resonance/scripts/wip'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: |Pi_k| vs k with 1/tau_k reference
ax = axes[0, 0]
for label, exp_results in all_results.items():
    t_key = max(exp_results.keys())
    sf = exp_results[t_key]['shell_flux']
    ks = [s['k'] for s in sf if s['E_k'] > 1e-30]
    pis = [abs(s['Pi']) for s in sf if s['E_k'] > 1e-30]
    inv_tau = [1.0/s['tau_k'] if s['tau_k'] < 1e10 else 0 for s in sf if s['E_k'] > 1e-30]
    ax.loglog(ks, pis, 'o-', label=f'{label} |Pi_k|', markersize=4)
    ax.loglog(ks, inv_tau, 's--', label=f'{label} 1/tau_k', markersize=3, alpha=0.5)
ax.set_xlabel('Shell k')
ax.set_ylabel('Value')
ax.set_title('Shell Energy Flux vs 1/tau_K (Tanogami test)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 2: Cumulative flux Pi(K)
ax = axes[0, 1]
for label, exp_results in all_results.items():
    t_key = max(exp_results.keys())
    cf = exp_results[t_key]['cum_flux']
    ks = [c[0] for c in cf]
    pis = [c[1] for c in cf]
    ax.plot(ks, pis, 'o-', label=label, markersize=4)
ax.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='Fano boundary')
ax.set_xlabel('K')
ax.set_ylabel('Pi(K)')
ax.set_title('Cumulative Energy Flux (Fano bottleneck)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: <sin(theta)> vs k
ax = axes[1, 0]
for label, exp_results in all_results.items():
    t_key = max(exp_results.keys())
    ts = exp_results[t_key]['triad_stats']
    ks = [k for k in sorted(ts.keys()) if ts[k]['n_triads'] > 10]
    sines = [ts[k]['mean_sin'] for k in ks]
    ax.plot(ks, sines, 'o-', label=label, markersize=4)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axhline(y=1, color='red', linestyle='--', alpha=0.3, label='saturation')
ax.axvline(x=3, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Shell k')
ax.set_ylabel('<sin(theta)>')
ax.set_title('Triad Phase Alignment (Von Mises test)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: |Pi_k| * tau_k vs k (should be constant if Tanogami holds)
ax = axes[1, 1]
for label, exp_results in all_results.items():
    t_key = max(exp_results.keys())
    sf = exp_results[t_key]['shell_flux']
    ks = [s['k'] for s in sf if s['E_k'] > 1e-30 and s['tau_k'] < 1e10]
    pi_tau = [abs(s['Pi'])*s['tau_k'] for s in sf if s['E_k'] > 1e-30 and s['tau_k'] < 1e10]
    ax.plot(ks, pi_tau, 'o-', label=label, markersize=4)
ax.axvline(x=3, color='red', linestyle='--', alpha=0.3, label='Fano boundary')
ax.set_xlabel('Shell k')
ax.set_ylabel('|Pi_k| * tau_k')
ax.set_title('Tanogami Constancy Test (should be ~const)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("Information Flow Measurements (Wanderer S107-W)", fontsize=14)
plt.tight_layout()
plt.savefig(f'{outdir}/information_flow_test.png', dpi=150, bbox_inches='tight')
P(f"  Plot saved: {outdir}/information_flow_test.png")

# Time evolution plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for label, exp_results in all_results.items():
    times = sorted(exp_results.keys())
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(times)))

    # <sin(theta)> evolution
    ax = axes[0]
    for i, t in enumerate(times):
        ts = exp_results[t]['triad_stats']
        ks = [k for k in sorted(ts.keys()) if ts[k]['n_triads'] > 10]
        sines = [ts[k]['mean_sin'] for k in ks]
        if ks:
            ax.plot(ks, sines, 'o-', color=cmap[i], label=f'{label} t={t:.0f}', markersize=3)

axes[0].set_xlabel('Shell k')
axes[0].set_ylabel('<sin(theta)>')
axes[0].set_title('Phase Alignment Evolution')
axes[0].legend(fontsize=6)
axes[0].grid(True, alpha=0.3)

# Cumulative flux evolution
ax = axes[1]
for label, exp_results in all_results.items():
    times = sorted(exp_results.keys())
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(times)))
    for i, t in enumerate(times):
        cf = exp_results[t]['cum_flux']
        ks = [c[0] for c in cf]
        pis = [c[1] for c in cf]
        ax.plot(ks, pis, '-', color=cmap[i], label=f'{label} t={t:.0f}', linewidth=1)
ax.axvline(x=3, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('K')
ax.set_ylabel('Pi(K)')
ax.set_title('Cumulative Flux Evolution')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# |Pi_k|*tau_k evolution
ax = axes[2]
for label, exp_results in all_results.items():
    times = sorted(exp_results.keys())
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(times)))
    for i, t in enumerate(times):
        sf = exp_results[t]['shell_flux']
        ks = [s['k'] for s in sf if s['E_k'] > 1e-30 and s['tau_k'] < 1e10]
        pt = [abs(s['Pi'])*s['tau_k'] for s in sf if s['E_k'] > 1e-30 and s['tau_k'] < 1e10]
        if ks:
            ax.plot(ks, pt, 'o-', color=cmap[i], label=f'{label} t={t:.0f}', markersize=3)
ax.axvline(x=3, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Shell k')
ax.set_ylabel('|Pi_k| * tau_k')
ax.set_title('Tanogami Constancy Evolution')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{outdir}/information_flow_evolution.png', dpi=150, bbox_inches='tight')
P(f"  Plot saved: {outdir}/information_flow_evolution.png")

P("\n  DONE.")
