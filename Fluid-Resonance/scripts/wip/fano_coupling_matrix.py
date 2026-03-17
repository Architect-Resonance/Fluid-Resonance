"""
TASK 2: Extract the Fano Coupling Matrix from NS-evolved fields
================================================================

Restrict to modes with |k| <= 3.
For each pair of modes (i,j) that share a triad k=p+q, compute the
effective coupling strength from the actual NS B(k,p,q) coefficients
weighted by current amplitudes.

Build the resulting coupling matrix. Compute eigenvalues.
Compare with the star cluster Laplacian prediction (lambda_min ~ 0.4950).

The key question: does the ACTUAL NS coupling at the Fano shell look
like the star cluster Laplacian? If lambda_min ~ 0.4950, the R argument
from S107-W2 has empirical support.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as clock
from itertools import combinations

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

    def rhs(self, ux, uy, uz):
        nlx, nly, nlz = self.nonlinear(ux, uy, uz)
        return (nlx - self.nu*self.k2*ux,
                nly - self.nu*self.k2*uy,
                nlz - self.nu*self.k2*uz)

    def rk4_step(self, ux, uy, uz, dt):
        k1x, k1y, k1z = self.rhs(ux, uy, uz)
        k2x, k2y, k2z = self.rhs(ux+dt/2*k1x, uy+dt/2*k1y, uz+dt/2*k1z)
        k3x, k3y, k3z = self.rhs(ux+dt/2*k2x, uy+dt/2*k2y, uz+dt/2*k2z)
        k4x, k4y, k4z = self.rhs(ux+dt*k3x, uy+dt*k3y, uz+dt*k3z)
        ux_new = ux + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
        uy_new = uy + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
        uz_new = uz + dt/6*(k1z + 2*k2z + 2*k3z + k4z)
        return self.leray_project(ux_new, uy_new, uz_new)

    def get_khat(self, ix, iy, iz):
        """Get wavevector (kx, ky, kz) for grid index (ix, iy, iz)"""
        return (self.kx[ix, iy, iz], self.ky[ix, iy, iz], self.kz[ix, iy, iz])

    def leray_projector_matrix(self, kx, ky, kz):
        """3x3 Leray projector P_ij = delta_ij - ki*kj/|k|^2"""
        k2 = kx**2 + ky**2 + kz**2
        if k2 == 0:
            return np.eye(3)
        k = np.array([kx, ky, kz])
        return np.eye(3) - np.outer(k, k) / k2


def get_fano_modes(ns, k_max=3):
    """
    Get all modes with 0 < |k| <= k_max (Fano shell).
    Returns list of (ix, iy, iz, kx, ky, kz) tuples.
    """
    N = ns.N
    modes = []
    k1d = fftfreq(N, d=1.0/N)
    for ix in range(N):
        for iy in range(N):
            for iz in range(N):
                kx, ky, kz = k1d[ix], k1d[iy], k1d[iz]
                kmag = np.sqrt(kx**2 + ky**2 + kz**2)
                if 0 < kmag <= k_max + 0.5 and ns.mask[ix, iy, iz] > 0:
                    modes.append((ix, iy, iz, int(kx), int(ky), int(kz)))
    return modes


def find_triads(modes, N):
    """
    Find all triads k = p + q among the Fano modes.
    Returns list of (mode_k_idx, mode_p_idx, mode_q_idx).
    """
    # Build lookup: (kx,ky,kz) -> mode index
    k_to_idx = {}
    for i, (ix, iy, iz, kx, ky, kz) in enumerate(modes):
        k_to_idx[(kx, ky, kz)] = i

    triads = []
    n = len(modes)
    for ip in range(n):
        _, _, _, px, py, pz = modes[ip]
        for iq in range(ip, n):  # avoid double counting
            _, _, _, qx, qy, qz = modes[iq]
            # k = p + q
            kx, ky, kz = px + qx, py + qy, pz + qz
            if (kx, ky, kz) in k_to_idx:
                ik = k_to_idx[(kx, ky, kz)]
                triads.append((ik, ip, iq))
    return triads


def compute_coupling_coefficient(ns, mode_k, mode_p, mode_q, ux, uy, uz):
    """
    Compute the NS triadic coupling coefficient B(k,p,q).

    For the NS equation du_k/dt = sum_{p+q=k} B(k,p,q) * u_p * u_q - nu*k^2*u_k

    B(k,p,q) is a tensor: B(k,p,q)_{alpha,beta,gamma} = -i * P_k_{alpha,mu} * q_mu * delta_{beta,gamma}
                                                          - i * P_k_{alpha,mu} * p_mu * delta_{gamma,beta}
    (symmetrized in p,q).

    The effective scalar coupling for the triad is:
    C_{pq->k} = |B(k,p,q) * u_p * u_q|^2 / (|u_p|^2 * |u_q|^2)

    For the coupling matrix, we want the ENERGY transfer rate between modes,
    weighted by current amplitudes.
    """
    ix_k, iy_k, iz_k, kx, ky, kz = mode_k
    ix_p, iy_p, iz_p, px, py, pz = mode_p
    ix_q, iy_q, iz_q, qx, qy, qz = mode_q

    norm = 1.0 / ns.N**3

    # Leray projector at k
    P_k = ns.leray_projector_matrix(kx, ky, kz)

    # Mode amplitudes (normalized)
    u_p = np.array([ux[ix_p, iy_p, iz_p],
                     uy[ix_p, iy_p, iz_p],
                     uz[ix_p, iy_p, iz_p]]) * norm
    u_q = np.array([ux[ix_q, iy_q, iz_q],
                     uy[ix_q, iy_q, iz_q],
                     uz[ix_q, iy_q, iz_q]]) * norm
    u_k = np.array([ux[ix_k, iy_k, iz_k],
                     uy[ix_k, iy_k, iz_k],
                     uz[ix_k, iy_k, iz_k]]) * norm

    # Coupling: B(k,p,q) * u_p * u_q
    # = -i * P_k . [(q . u_p) u_q + (p . u_q) u_p]  (symmetrized)
    q_vec = np.array([qx, qy, qz], dtype=float)
    p_vec = np.array([px, py, pz], dtype=float)

    q_dot_up = np.dot(q_vec, u_p)
    p_dot_uq = np.dot(p_vec, u_q)

    # The nonlinear term contribution from this triad
    nl_contribution = -1j * P_k @ (q_dot_up * u_q + p_dot_uq * u_p)

    # Energy transfer: Re[u_k* . NL_contribution]
    energy_transfer = np.real(np.dot(np.conj(u_k), nl_contribution))

    # Coupling strength (amplitude-independent, geometric):
    # |B(k,p,q)|_F = Frobenius norm of the coupling tensor
    # This is the "geometric" coupling, independent of current amplitudes
    # For the Leray-weighted coupling: B_{alpha} = -i P_k_{alpha,mu} q_mu (for one ordering)
    B_pq = -1j * P_k @ np.diag(q_vec)  # 3x3 tensor (simplified)
    B_qp = -1j * P_k @ np.diag(p_vec)
    coupling_strength = np.sqrt(np.sum(np.abs(B_pq)**2) + np.sum(np.abs(B_qp)**2))

    # Amplitude-weighted coupling
    amp_p = np.sqrt(np.sum(np.abs(u_p)**2))
    amp_q = np.sqrt(np.sum(np.abs(u_q)**2))
    weighted_coupling = coupling_strength * amp_p * amp_q

    return {
        'energy_transfer': energy_transfer,
        'coupling_strength': coupling_strength,
        'weighted_coupling': weighted_coupling,
        'amp_p': amp_p,
        'amp_q': amp_q,
        'nl_contribution': nl_contribution,
    }


def build_coupling_matrix(ns, ux, uy, uz, k_max=3):
    """
    Build the Fano coupling matrix.

    For each pair of modes (i,j) that share a triad, the matrix entry
    L_{ij} = sum over triads connecting i and j of the coupling strength.

    The diagonal L_{ii} = -sum_j L_{ij} (Laplacian convention).
    """
    P(f"  Finding Fano modes (|k| <= {k_max})...")
    modes = get_fano_modes(ns, k_max)
    n_modes = len(modes)
    P(f"  Found {n_modes} modes")

    P(f"  Finding triads...")
    triads = find_triads(modes, ns.N)
    P(f"  Found {len(triads)} triads")

    # Build adjacency matrix (coupling between mode pairs)
    # Method 1: Geometric coupling (independent of amplitudes)
    L_geom = np.zeros((n_modes, n_modes))
    # Method 2: Amplitude-weighted coupling
    L_weighted = np.zeros((n_modes, n_modes))
    # Method 3: Energy transfer matrix
    L_energy = np.zeros((n_modes, n_modes))

    P(f"  Computing coupling coefficients for {len(triads)} triads...")
    t0 = clock.time()

    for t_idx, (ik, ip, iq) in enumerate(triads):
        if t_idx % 500 == 0 and t_idx > 0:
            P(f"    [{t_idx}/{len(triads)}, elapsed={clock.time()-t0:.1f}s]")

        mode_k = modes[ik]
        mode_p = modes[ip]
        mode_q = modes[iq]

        c = compute_coupling_coefficient(ns, mode_k, mode_p, mode_q, ux, uy, uz)

        # Geometric coupling (off-diagonal, symmetric)
        L_geom[ip, iq] += c['coupling_strength']
        L_geom[iq, ip] += c['coupling_strength']
        L_geom[ik, ip] += c['coupling_strength']
        L_geom[ip, ik] += c['coupling_strength']
        L_geom[ik, iq] += c['coupling_strength']
        L_geom[iq, ik] += c['coupling_strength']

        # Weighted coupling
        L_weighted[ip, iq] += c['weighted_coupling']
        L_weighted[iq, ip] += c['weighted_coupling']
        L_weighted[ik, ip] += c['weighted_coupling']
        L_weighted[ip, ik] += c['weighted_coupling']
        L_weighted[ik, iq] += c['weighted_coupling']
        L_weighted[iq, ik] += c['weighted_coupling']

        # Energy transfer (antisymmetric contribution)
        L_energy[ik, ip] += abs(c['energy_transfer'])
        L_energy[ip, ik] += abs(c['energy_transfer'])
        L_energy[ik, iq] += abs(c['energy_transfer'])
        L_energy[iq, ik] += abs(c['energy_transfer'])

    P(f"  Coupling computation done ({clock.time()-t0:.1f}s)")

    # Convert to Laplacian: L_ii = -sum_j L_ij
    for L in [L_geom, L_weighted, L_energy]:
        for i in range(n_modes):
            L[i, i] = -np.sum(L[i, :])

    return modes, triads, L_geom, L_weighted, L_energy


def analyze_matrix(name, L, n_show=15):
    """Compute and display eigenvalues of coupling Laplacian."""
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L)))

    P(f"\n  === {name} ===")
    P(f"  Matrix size: {L.shape[0]}x{L.shape[0]}")
    P(f"  First {n_show} eigenvalues:")
    for i in range(min(n_show, len(eigenvalues))):
        P(f"    lambda_{i} = {eigenvalues[i]:.6f}")

    if len(eigenvalues) > 1:
        # lambda_0 should be ~0 (Laplacian nullspace)
        P(f"  lambda_0 (nullspace) = {eigenvalues[0]:.2e}")
        P(f"  lambda_1 (algebraic connectivity) = {eigenvalues[1]:.6f}")
        P(f"  lambda_max = {eigenvalues[-1]:.6f}")
        P(f"  Spectral gap: lambda_1/lambda_max = {eigenvalues[1]/eigenvalues[-1]:.6f}")

        # Compare with star cluster prediction
        P(f"\n  Star cluster comparison:")
        P(f"    Predicted lambda_min = 0.4950")
        P(f"    Measured  lambda_1   = {eigenvalues[1]:.6f}")
        if eigenvalues[1] != 0:
            P(f"    Ratio: measured/predicted = {eigenvalues[1]/0.4950:.6f}")

        # Compare with 7/4 (Fano algebraic connectivity with weight 1/4)
        P(f"    Fano prediction (7/4) = 1.75")
        P(f"    Measured lambda_1     = {eigenvalues[1]:.6f}")

    return eigenvalues


def build_gf2_laplacian(ns, ux, uy, uz, k_max=3):
    """
    WANDERER'S CORRECTION: Group modes by GF(2)^3 class.

    Each wavevector k = (kx, ky, kz) maps to (kx mod 2, ky mod 2, kz mod 2).
    This gives 7 nonzero classes = the 7 points of the Fano plane PG(2,2).

    Build a 7x7 coupling matrix by summing coupling weights within each class.
    THIS is the object to compare against the star cluster Laplacian.
    """
    P(f"\n  Building GF(2)^3 Laplacian (Wanderer's correction)...")
    modes = get_fano_modes(ns, k_max)
    norm = 1.0 / ns.N**3

    # Map each mode to its GF(2)^3 class
    # Classes: all nonzero elements of GF(2)^3 = {(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)}
    gf2_classes = [
        (0,0,1), (0,1,0), (0,1,1),
        (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    ]
    class_idx = {c: i for i, c in enumerate(gf2_classes)}
    class_names = [f"({c[0]},{c[1]},{c[2]})" for c in gf2_classes]

    mode_class = {}
    class_modes = {i: [] for i in range(7)}
    for mi, (ix, iy, iz, kx, ky, kz) in enumerate(modes):
        c = (int(kx) % 2, int(ky) % 2, int(kz) % 2)
        if c == (0, 0, 0):
            continue  # skip zero class (shouldn't happen since |k|>0)
        ci = class_idx.get(c)
        if ci is not None:
            mode_class[mi] = ci
            class_modes[ci].append(mi)

    P(f"  GF(2)^3 classes:")
    for i, c in enumerate(gf2_classes):
        P(f"    {class_names[i]}: {len(class_modes[i])} modes")

    # Find triads and compute coupling between GF(2)^3 classes
    triads = find_triads(modes, ns.N)
    P(f"  Triads: {len(triads)}")

    # Three coupling matrices: geometric, weighted, energy
    L_geom = np.zeros((7, 7))
    L_weighted = np.zeros((7, 7))
    L_energy = np.zeros((7, 7))

    for ik, ip, iq in triads:
        if ik not in mode_class or ip not in mode_class or iq not in mode_class:
            continue

        ck = mode_class[ik]
        cp = mode_class[ip]
        cq = mode_class[iq]

        c = compute_coupling_coefficient(ns, modes[ik], modes[ip], modes[iq], ux, uy, uz)

        # Geometric: count triads between class pairs
        pairs = set()
        pairs.add((min(ck, cp), max(ck, cp)))
        pairs.add((min(ck, cq), max(ck, cq)))
        pairs.add((min(cp, cq), max(cp, cq)))

        for i, j in pairs:
            if i != j:
                L_geom[i, j] += c['coupling_strength']
                L_geom[j, i] += c['coupling_strength']
                L_weighted[i, j] += c['weighted_coupling']
                L_weighted[j, i] += c['weighted_coupling']
                L_energy[i, j] += abs(c['energy_transfer'])
                L_energy[j, i] += abs(c['energy_transfer'])

    # Laplacian convention: diagonal = -sum of row
    for L in [L_geom, L_weighted, L_energy]:
        for i in range(7):
            L[i, i] = -np.sum(L[i, :])

    P(f"\n  GF(2)^3 Geometric Laplacian (7x7):")
    header = "        " + "  ".join(f"{class_names[j]:>10}" for j in range(7))
    P(header)
    for i in range(7):
        row = f"  {class_names[i]:>6}"
        for j in range(7):
            row += f"  {L_geom[i,j]:10.1f}"
        P(row)

    eig_geom = analyze_matrix("GF(2)^3 Geometric Laplacian", L_geom)
    eig_weighted = analyze_matrix("GF(2)^3 Amplitude-Weighted Laplacian", L_weighted)
    eig_energy = analyze_matrix("GF(2)^3 Energy Transfer Laplacian", L_energy)

    # Normalize: divide by max eigenvalue magnitude to get relative structure
    P(f"\n  Normalized eigenvalues (divide by |lambda_max|):")
    for name, eigs in [("Geometric", eig_geom), ("Weighted", eig_weighted), ("Energy", eig_energy)]:
        if abs(eigs[0]) > 0:
            normed = eigs / abs(eigs[0])
            P(f"    {name}: {[f'{v:.4f}' for v in normed]}")

    return gf2_classes, L_geom, L_weighted, L_energy, eig_geom, eig_weighted, eig_energy


def build_shell_laplacian(ns, ux, uy, uz, k_max=3):
    """
    Alternative approach: build coupling matrix between SHELLS (not individual modes).

    Group modes by |k| (shell 1, 2, 3) and compute inter-shell coupling.
    This gives a smaller matrix more directly comparable to the 7x7 or 8x8
    star cluster Laplacian from S107-W2.
    """
    P(f"\n  Building SHELL Laplacian (shells 1..{k_max})...")
    modes = get_fano_modes(ns, k_max)

    # Assign each mode to a shell
    shell_of = {}
    for i, (ix, iy, iz, kx, ky, kz) in enumerate(modes):
        kmag = np.sqrt(kx**2 + ky**2 + kz**2)
        shell = int(np.round(kmag))
        shell_of[i] = shell

    # Find unique shells
    shells = sorted(set(shell_of.values()))
    n_shells = len(shells)
    shell_idx = {s: i for i, s in enumerate(shells)}

    P(f"  Shells: {shells}")
    P(f"  Modes per shell: {[(s, sum(1 for v in shell_of.values() if v == s)) for s in shells]}")

    # Find triads and compute inter-shell coupling
    triads = find_triads(modes, ns.N)
    P(f"  Triads: {len(triads)}")

    L_shell = np.zeros((n_shells, n_shells))

    for ik, ip, iq in triads:
        sk = shell_idx[shell_of[ik]]
        sp = shell_idx[shell_of[ip]]
        sq = shell_idx[shell_of[iq]]

        # Each triad connects 3 shells (or fewer if some coincide)
        # Coupling weight = 1 (geometric, unweighted)
        pairs = set()
        pairs.add((min(sk, sp), max(sk, sp)))
        pairs.add((min(sk, sq), max(sk, sq)))
        pairs.add((min(sp, sq), max(sp, sq)))

        for i, j in pairs:
            if i != j:
                L_shell[i, j] += 1
                L_shell[j, i] += 1

    # Laplacian: diagonal = -sum of row
    for i in range(n_shells):
        L_shell[i, i] = -np.sum(L_shell[i, :])

    return shells, L_shell


def run_experiment(N, Re, t_final=3.0, k_max=3):
    nu = 1.0 / Re
    ns = SpectralNS(N, nu)

    P(f"\n{'='*72}")
    P(f"  Experiment: N={N}, Re={Re}, k_max={k_max}")
    P(f"{'='*72}")

    ux, uy, uz = ns.taylor_green_ic()

    # Timestep: proven stable in previous runs
    dt = 0.001 if Re <= 400 else 0.0005
    t = 0.0
    step = 0
    t0 = clock.time()

    while t < t_final - dt/2:
        ux, uy, uz = ns.rk4_step(ux, uy, uz, dt)
        t += dt
        step += 1
        if step % 500 == 0:
            P(f"  [step {step}, t={t:.3f}, elapsed={clock.time()-t0:.0f}s]")

    P(f"  Evolved to t={t:.3f} (elapsed={clock.time()-t0:.0f}s)")

    norm = 1.0 / N**6
    E = 0.5 * np.sum(np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2) * norm
    Z = 0.5 * np.sum(ns.k2 * (np.abs(ux)**2 + np.abs(uy)**2 + np.abs(uz)**2)) * norm
    P(f"  E={E:.6f}, Z={Z:.4f}")

    # Build mode-by-mode coupling matrix
    modes, triads, L_geom, L_weighted, L_energy = build_coupling_matrix(ns, ux, uy, uz, k_max)

    # Analyze all three versions
    eig_geom = analyze_matrix("Geometric Coupling Laplacian", L_geom)
    eig_weighted = analyze_matrix("Amplitude-Weighted Coupling Laplacian", L_weighted)
    eig_energy = analyze_matrix("Energy Transfer Laplacian", L_energy)

    # Build GF(2)^3 Laplacian (Wanderer's correction — the RIGHT grouping)
    gf2_classes, L_gf2_geom, L_gf2_w, L_gf2_e, eig_gf2_g, eig_gf2_w, eig_gf2_e = \
        build_gf2_laplacian(ns, ux, uy, uz, k_max)

    # Build shell-level Laplacian (for reference, not the right comparison)
    shells, L_shell = build_shell_laplacian(ns, ux, uy, uz, k_max)
    eig_shell = analyze_matrix("Shell-Level Laplacian", L_shell)

    return {
        'modes': modes, 'triads': triads,
        'L_geom': L_geom, 'L_weighted': L_weighted, 'L_energy': L_energy,
        'eig_geom': eig_geom, 'eig_weighted': eig_weighted, 'eig_energy': eig_energy,
        'gf2_classes': gf2_classes,
        'eig_gf2_geom': eig_gf2_g, 'eig_gf2_weighted': eig_gf2_w, 'eig_gf2_energy': eig_gf2_e,
        'L_gf2_geom': L_gf2_geom, 'L_gf2_weighted': L_gf2_w, 'L_gf2_energy': L_gf2_e,
        'shells': shells, 'L_shell': L_shell, 'eig_shell': eig_shell,
    }


if __name__ == '__main__':
    P("=" * 72)
    P("  TASK 2: Fano Coupling Matrix — Eigenvalue Extraction")
    P("=" * 72)
    P()
    P("  Goal: Build the actual NS coupling matrix at the Fano shell (|k|<=3)")
    P("  and compare eigenvalues with the star cluster Laplacian prediction.")
    P("  Star cluster lambda_min = 0.4950, Fano algebraic connectivity = 7/4 = 1.75")
    P()

    results = {}
    for N, Re in [(32, 400), (32, 1600)]:
        label = f"N{N}_Re{Re}"
        results[label] = run_experiment(N, Re, t_final=3.0, k_max=3)

    # Summary
    P("\n" + "=" * 72)
    P("  SUMMARY: Eigenvalue Comparison with Star Cluster")
    P("=" * 72)
    P()

    for label, res in results.items():
        P(f"  {label}:")
        P(f"    178-mode Geometric:  lambda_1 = {res['eig_geom'][1]:.6f}")
        P(f"    178-mode Weighted:   lambda_1 = {res['eig_weighted'][1]:.6f}")
        P(f"    178-mode Energy:     lambda_1 = {res['eig_energy'][1]:.6f}")
        P(f"    --- GF(2)^3 GROUPING (7x7, Wanderer's correction) ---")
        P(f"    GF2 Geometric:  lambda_1 = {res['eig_gf2_geom'][1]:.6f}")
        P(f"    GF2 Weighted:   lambda_1 = {res['eig_gf2_weighted'][1]:.6f}")
        P(f"    GF2 Energy:     lambda_1 = {res['eig_gf2_energy'][1]:.6f}")
        P()

    P("  Star cluster prediction: lambda_min = 0.4950")
    P("  Fano prediction: lambda_2 = 7/4 = 1.75 (K_7 with weight 1/4)")
    P()

    # Generate plots
    P("  Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (label, res) in enumerate(results.items()):
        if idx >= 2:
            break
        ax = axes[idx]
        # Plot eigenvalue spectra for all 3 methods
        for name, eigs, color in [
            ('Geometric', res['eig_geom'], 'blue'),
            ('Weighted', res['eig_weighted'], 'red'),
            ('Energy', res['eig_energy'], 'green'),
        ]:
            # Normalize to show relative structure
            if len(eigs) > 1 and eigs[-1] != 0:
                eigs_norm = eigs / abs(eigs[-1])
            else:
                eigs_norm = eigs
            ax.plot(range(min(30, len(eigs_norm))), eigs_norm[:30], 'o-',
                    label=name, color=color, markersize=3, alpha=0.7)
        ax.axhline(y=0, color='k', ls='--', alpha=0.3)
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('Eigenvalue (normalized)')
        ax.set_title(f'{label}\nCoupling Laplacian Eigenvalues')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot 3: GF(2)^3 Laplacian eigenvalues (THE key comparison)
    ax = axes[2]
    for label, res in results.items():
        for name, key, marker in [
            ('Geom', 'eig_gf2_geom', 'o'),
            ('Weighted', 'eig_gf2_weighted', 's'),
            ('Energy', 'eig_gf2_energy', '^'),
        ]:
            eigs = res[key]
            # Normalize: divide by |lambda_0| so we see relative structure
            if abs(eigs[0]) > 0:
                eigs_n = eigs / abs(eigs[0])
            else:
                eigs_n = eigs
            ax.plot(range(len(eigs_n)), eigs_n, marker+'-',
                    label=f'{name} {label}', markersize=5, alpha=0.7)
    # Star cluster normalized prediction: eigenvalues of K_7 with weight 1/4
    # lambda = 0 (1x) and 7/4 (6x). Normalized: 0 and 1.0
    ax.axhline(y=-1.0, color='red', ls='--', alpha=0.7, label='K_7 degenerate (all equal)')
    ax.axhline(y=0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue / |lambda_0|')
    ax.set_title('GF(2)^3 Laplacian (7x7)\nNormalized Eigenvalues')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = 'Fluid-Resonance/scripts/wip/fano_coupling_matrix.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    P(f"  Plot saved: {outpath}")

    P("\n  DONE.")
