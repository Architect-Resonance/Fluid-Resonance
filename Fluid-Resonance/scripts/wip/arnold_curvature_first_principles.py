"""
arnold_curvature_first_principles.py
=====================================
First-principles computation of Arnold sectional curvature on SDiff(T^3)
for helical Fourier modes.

PURPOSE: Answer Wanderer's audit (S101b-W2) -- is K_base = m^2 sin^2(phi)/8 correct?

APPROACH: Use the Levi-Civita connection on SDiff(T^3) with L^2 metric,
computed via the Koszul formula. For helical eigenmodes of curl, the
connection simplifies because omega_u = sigma|k|u.

Connection formula (derived from Koszul formula for right-invariant metric):
  nabla_u v = (1/2)[curl(u x v) - P_sol(omega_v x u) - P_sol(omega_u x v)]

For curl eigenmodes omega_u = sigma|k|u, omega_v = tau|p|v:
  nabla_u v = (1/2)[curl(u x v) + (tau|p| - sigma|k|) P_sol(u x v)]

Key simplification: nabla_v v = 0 (single helical mode = Beltrami = steady Euler).
So the curvature reduces to:
  <R(u,v)v, u> = -<nabla_v(nabla_u v), u> - <nabla_{[u,v]} v, u>
"""

import numpy as np


# ================================================================
# Fourier mode operations on T^3
# ================================================================

def cross_product_fields(modes_a, modes_b):
    """
    Cross product of two vector fields in Fourier space.
    (a x b)(x) = sum_{k,p} (a_hat_k x b_hat_p) e^{i(k+p).x}
    """
    result = {}
    for ka, a_hat in modes_a.items():
        for kb, b_hat in modes_b.items():
            q = tuple(np.array(ka) + np.array(kb))
            cross = np.cross(a_hat, b_hat)
            if q in result:
                result[q] = result[q] + cross
            else:
                result[q] = cross.copy()
    return {k: v for k, v in result.items() if np.linalg.norm(v) > 1e-15}


def curl_field(modes):
    """curl in Fourier space: curl(f_hat e^{iq.x}) = iq x f_hat e^{iq.x}"""
    result = {}
    for q, f_hat in modes.items():
        q_vec = np.array(q, dtype=float)
        curled = 1j * np.cross(q_vec, f_hat)
        if np.linalg.norm(curled) > 1e-15:
            result[q] = curled
    return result


def leray_project_field(modes):
    """Leray projection: remove gradient part at each wavevector."""
    result = {}
    for q, f_hat in modes.items():
        q_vec = np.array(q, dtype=float)
        q_sq = np.dot(q_vec, q_vec)
        if q_sq < 1e-20:
            continue  # skip zero mode
        q_hat = q_vec / np.sqrt(q_sq)
        projected = f_hat - q_hat * np.dot(q_hat, f_hat)
        if np.linalg.norm(projected) > 1e-15:
            result[q] = projected
    return result


def scale_field(modes, scalar):
    """Multiply all modes by a scalar."""
    return {k: scalar * v for k, v in modes.items()}


def add_fields(*fields_and_signs):
    """Add multiple fields: add_fields((f1, 1.0), (f2, -1.0), ...)"""
    result = {}
    for modes, sign in fields_and_signs:
        for k, v in modes.items():
            if k in result:
                result[k] = result[k] + sign * v
            else:
                result[k] = sign * v.copy()
    return {k: v for k, v in result.items() if np.linalg.norm(v) > 1e-15}


def L2_inner(modes_a, modes_b, volume=(2*np.pi)**3):
    """L^2 inner product: V * sum_k a_k . conj(b_k)"""
    result = 0.0 + 0.0j
    for k, a_hat in modes_a.items():
        if k in modes_b:
            result += np.dot(a_hat, np.conj(modes_b[k]))
    return volume * result


def L2_norm_sq(modes, volume=(2*np.pi)**3):
    """L^2 norm squared."""
    return np.real(L2_inner(modes, modes, volume))


def advection(u_modes, v_modes):
    """
    Compute P_sol[(u . nabla)v] in Fourier space.
    (u . nabla)v at wavevector k'+q: i(u_hat_{k'} . q) v_hat_q
    Then Leray project.
    """
    result = {}
    for kp, u_hat in u_modes.items():
        for q, v_hat in v_modes.items():
            out_k = tuple(np.array(kp) + np.array(q))
            dot_uq = np.dot(u_hat, np.array(q, dtype=complex))
            contrib = 1j * dot_uq * v_hat
            if out_k in result:
                result[out_k] = result[out_k] + contrib
            else:
                result[out_k] = contrib.copy()
    return leray_project_field(result)


# ================================================================
# Connection on SDiff(T^3)
# ================================================================

def connection(u_modes, v_modes, omega_u_modes, omega_v_modes):
    """
    Levi-Civita connection on SDiff(T^3) with L^2 metric:
    nabla_u v = (1/2)[curl(u x v) - P_sol(omega_v x u) - P_sol(omega_u x v)]

    omega_u = curl(u), omega_v = curl(v)
    """
    # Term 1: curl(u x v)
    uxv = cross_product_fields(u_modes, v_modes)
    curl_uxv = curl_field(uxv)

    # Term 2: -P_sol(omega_v x u)
    omv_x_u = cross_product_fields(omega_v_modes, u_modes)
    term2 = leray_project_field(omv_x_u)

    # Term 3: -P_sol(omega_u x v)
    omu_x_v = cross_product_fields(omega_u_modes, v_modes)
    term3 = leray_project_field(omu_x_v)

    # nabla_u v = (1/2)(curl_uxv - term2 - term3)
    return scale_field(
        add_fields((curl_uxv, 1.0), (term2, -1.0), (term3, -1.0)),
        0.5
    )


# ================================================================
# Helical mode construction
# ================================================================

def make_helical_mode(k_vec, sigma):
    """
    Create a real helical mode on T^3 as Fourier modes dict.
    u(x) = Re[xi e^{ik.x}] with curl(u) = sigma|k|u.

    Returns: (u_modes, omega_modes) where omega = curl(u) = sigma|k|u
    """
    k = np.array(k_vec, dtype=float)
    knorm = np.linalg.norm(k)
    khat = k / knorm

    # Build frame: e1 perp k, e2 = khat x e1
    axes = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]
    dots = [abs(np.dot(khat, a)) for a in axes]
    best = np.argmin(dots)
    e1 = axes[best] - np.dot(axes[best], khat) * khat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(khat, e1)

    xi = (e1 + 1j * sigma * e2) / np.sqrt(2)

    # Verify curl eigenvalue: k x xi = -i sigma |k| xi
    check = np.cross(k, xi) - (-1j * sigma * knorm * xi)
    assert np.max(np.abs(check)) < 1e-10, f"Curl check failed: {np.max(np.abs(check))}"

    k_tuple = tuple(k.astype(int))
    mk_tuple = tuple((-k).astype(int))

    u_modes = {k_tuple: xi / 2, mk_tuple: np.conj(xi) / 2}

    # omega = curl(u) = sigma|k| u
    omega_modes = {k_tuple: sigma * knorm * xi / 2,
                   mk_tuple: sigma * knorm * np.conj(xi) / 2}

    return u_modes, omega_modes


def make_helical_pair(k_vec, p_vec, sigma, tau):
    """
    Create two helical modes with consistent frame choice.
    Place both in the plane spanned by k and p.
    """
    k = np.array(k_vec, dtype=float)
    p = np.array(p_vec, dtype=float)
    knorm = np.linalg.norm(k)
    pnorm = np.linalg.norm(p)

    cos_phi = np.clip(np.dot(k, p) / (knorm * pnorm), -1, 1)
    phi = np.arccos(cos_phi)
    sin_phi = np.sin(phi)

    if sin_phi < 1e-12:
        # Parallel -- use independent frames
        u_modes, omega_u = make_helical_mode(k_vec, sigma)
        v_modes, omega_v = make_helical_mode(p_vec, tau)
        return u_modes, omega_u, v_modes, omega_v, phi

    # Build consistent frames
    khat = k / knorm
    phat = p / pnorm

    # y = (k x p) / |k x p|  (normal to k-p plane)
    y_dir = np.cross(k, p)
    y_dir = y_dir / np.linalg.norm(y_dir)

    # At k: e1 = y x khat (in k-p plane, perp to k), e2 = y
    e1_k = np.cross(y_dir, khat)
    e1_k = e1_k / np.linalg.norm(e1_k)
    e2_k = y_dir.copy()

    xi_k = (e1_k + 1j * sigma * e2_k) / np.sqrt(2)

    # Verify
    check_k = np.cross(k, xi_k) - (-1j * sigma * knorm * xi_k)
    assert np.max(np.abs(check_k)) < 1e-10, f"Curl k failed"

    # At p: f1 = y x phat, f2 = y
    f1_p = np.cross(y_dir, phat)
    f1_p = f1_p / np.linalg.norm(f1_p)
    f2_p = y_dir.copy()

    xi_p = (f1_p + 1j * tau * f2_p) / np.sqrt(2)

    check_p = np.cross(p, xi_p) - (-1j * tau * pnorm * xi_p)
    assert np.max(np.abs(check_p)) < 1e-10, f"Curl p failed"

    k_t = tuple(k.astype(int))
    mk_t = tuple((-k).astype(int))
    p_t = tuple(p.astype(int))
    mp_t = tuple((-p).astype(int))

    u_modes = {k_t: xi_k / 2, mk_t: np.conj(xi_k) / 2}
    omega_u = {k_t: sigma * knorm * xi_k / 2,
               mk_t: sigma * knorm * np.conj(xi_k) / 2}

    v_modes = {p_t: xi_p / 2, mp_t: np.conj(xi_p) / 2}
    omega_v = {p_t: tau * pnorm * xi_p / 2,
               mp_t: tau * pnorm * np.conj(xi_p) / 2}

    return u_modes, omega_u, v_modes, omega_v, phi


# ================================================================
# Curvature computation
# ================================================================

def compute_curvature(k_vec, p_vec, sigma, tau, debug=False):
    """
    Compute sectional curvature K(u,v) on SDiff(T^3) for helical modes.

    Uses: nabla_v v = 0 (Beltrami = steady Euler), so:
    <R(u,v)v, u> = <nabla_u v, nabla_u v> - <nabla_u v, nabla_v u>
                   - <nabla_{[u,v]} v, u>

    Wait, let me derive this more carefully.

    For constant (right-invariant) vector fields on a Lie group:
    R(u,v)w = nabla_u nabla_v w - nabla_v nabla_u w - nabla_{[u,v]} w

    With nabla_v v = 0:
    R(u,v)v = nabla_u(0) - nabla_v(nabla_u v) - nabla_{[u,v]} v
            = -nabla_v(nabla_u v) - nabla_{[u,v]} v

    <R(u,v)v, u> = -<nabla_v(nabla_u v), u> - <nabla_{[u,v]} v, u>

    Metric compatibility (constant metric on Lie group):
    <nabla_v w, u> = -<w, nabla_v u>

    So: -<nabla_v(nabla_u v), u> = <nabla_u v, nabla_v u>

    For the second term, we compute nabla_{[u,v]} v directly.
    """
    u, om_u, v, om_v, phi = make_helical_pair(k_vec, p_vec, sigma, tau)

    norm_u_sq = L2_norm_sq(u)
    norm_v_sq = L2_norm_sq(v)
    uv_inner = np.real(L2_inner(u, v))

    if debug:
        print(f"  ||u||^2 = {norm_u_sq:.6f}, ||v||^2 = {norm_v_sq:.6f}, <u,v> = {uv_inner:.6f}")

    # Step 1: nabla_u v
    nav_uv = connection(u, v, om_u, om_v)

    if debug:
        print(f"  nabla_u v modes: {len(nav_uv)}")
        for q, c in sorted(nav_uv.items()):
            print(f"    {q}: {c}")

    # Step 2: nabla_v u
    nav_vu = connection(v, u, om_v, om_u)

    # Step 3: <nabla_u v, nabla_v u>
    term1 = np.real(L2_inner(nav_uv, nav_vu))

    if debug:
        print(f"  <nabla_u v, nabla_v u> = {term1:.10f}")
        print(f"  ||nabla_u v||^2 = {L2_norm_sq(nav_uv):.10f}")

    # Step 4: Lie bracket [u,v] = curl(u x v)  (for SDiff right-invariant)
    # Actually [u,v] on SDiff = P_sol[(v.nabla)u - (u.nabla)v]
    # = advection(v,u) - advection(u,v)   (with Leray projection)
    bracket = add_fields(
        (advection(v, u), 1.0),
        (advection(u, v), -1.0)
    )

    # omega of bracket: curl of bracket
    om_bracket = curl_field(bracket)

    if debug:
        print(f"  [u,v] modes: {len(bracket)}")
        print(f"  ||[u,v]||^2 = {L2_norm_sq(bracket):.10f}")

    # Step 5: nabla_{[u,v]} v
    nav_bracket_v = connection(bracket, v, om_bracket, om_v)

    # Step 6: -<nabla_{[u,v]} v, u>
    term2 = -np.real(L2_inner(nav_bracket_v, u))

    if debug:
        print(f"  -<nabla_[u,v] v, u> = {term2:.10f}")

    # Curvature numerator: <R(u,v)v, u> = term1 + term2
    numerator = term1 + term2
    denominator = norm_u_sq * norm_v_sq - uv_inner**2

    K = numerator / denominator

    if debug:
        print(f"  numerator = {numerator:.10f}")
        print(f"  denominator = {denominator:.10f}")
        print(f"  K = {K:.10f}")

    return K, phi


# ================================================================
# MAIN
# ================================================================

print("=" * 70)
print("ARNOLD CURVATURE ON SDiff(T^3) -- FIRST PRINCIPLES")
print("Using Levi-Civita connection via Koszul formula")
print("No assumptions about K_base")
print("=" * 70)

# Test 1: k=(1,0,0), p=(0,1,0), phi=pi/2, |k|=|p|=1
print("\n--- Test 1: k=(1,0,0), p=(0,1,0) [phi=pi/2, m=1] ---")
K_same, phi = compute_curvature([1,0,0], [0,1,0], +1, +1, debug=True)
K_cross, _ = compute_curvature([1,0,0], [0,1,0], +1, -1, debug=True)

K_base = (K_same + K_cross) / 2
splitting = (K_cross - K_same) / 2
m = 1.0
claimed_Kb = m**2 * np.sin(phi)**2 / 8
claimed_split = m * m * np.sin(phi)**2 / 2

print(f"\nRESULTS:")
print(f"  K_same  = {K_same:.8f}  (claimed: {-3/8:.8f})")
print(f"  K_cross = {K_cross:.8f}  (claimed: {5/8:.8f})")
print(f"  K_base  = {K_base:.8f}  (claimed: {claimed_Kb:.8f})")
print(f"  splitting = {splitting:.8f}  (claimed: {claimed_split:.8f})")

# Test 2: different magnitudes
print("\n--- Test 2: k=(1,0,0), p=(1,1,0) [phi=pi/4, |p|=sqrt(2)] ---")
K_same2, phi2 = compute_curvature([1,0,0], [1,1,0], +1, +1)
K_cross2, _ = compute_curvature([1,0,0], [1,1,0], +1, -1)
K_base2 = (K_same2 + K_cross2) / 2
splitting2 = (K_cross2 - K_same2) / 2
mk2, mp2 = 1.0, np.sqrt(2)
print(f"  K_same  = {K_same2:.8f}")
print(f"  K_cross = {K_cross2:.8f}")
print(f"  K_base  = {K_base2:.8f}  (formula: {mk2**2*mp2**2*np.sin(phi2)**2/8:.8f})")
print(f"  splitting = {splitting2:.8f}  (formula: {mk2*mp2*np.sin(phi2)**2/2:.8f})")

# Test 3: higher wavenumber
print("\n--- Test 3: k=(2,0,0), p=(0,2,0) [phi=pi/2, m=2] ---")
K_same3, phi3 = compute_curvature([2,0,0], [0,2,0], +1, +1)
K_cross3, _ = compute_curvature([2,0,0], [0,2,0], +1, -1)
K_base3 = (K_same3 + K_cross3) / 2
splitting3 = (K_cross3 - K_same3) / 2
m3 = 2.0
print(f"  K_same  = {K_same3:.8f}")
print(f"  K_cross = {K_cross3:.8f}")
print(f"  K_base  = {K_base3:.8f}  (formula: {m3**2*m3**2*np.sin(phi3)**2/8:.8f})")
print(f"  splitting = {splitting3:.8f}  (formula: {m3*m3*np.sin(phi3)**2/2:.8f})")

# Test 4: Fano edge (1,1,0) vs (0,1,1)
print("\n--- Test 4: k=(1,1,0), p=(0,1,1) [phi=pi/3, |k|=|p|=sqrt(2)] ---")
K_same4, phi4 = compute_curvature([1,1,0], [0,1,1], +1, +1)
K_cross4, _ = compute_curvature([1,1,0], [0,1,1], +1, -1)
K_base4 = (K_same4 + K_cross4) / 2
splitting4 = (K_cross4 - K_same4) / 2
m4 = np.sqrt(2)
print(f"  K_same  = {K_same4:.8f}")
print(f"  K_cross = {K_cross4:.8f}")
print(f"  K_base  = {K_base4:.8f}  (formula: {m4**2*m4**2*np.sin(phi4)**2/8:.8f})")
print(f"  splitting = {splitting4:.8f}  (formula: {m4*m4*np.sin(phi4)**2/2:.8f})")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

tests = [
    ([1,0,0], [0,1,0], "pi/2, m=1"),
    ([1,0,0], [1,1,0], "pi/4, mixed"),
    ([2,0,0], [0,2,0], "pi/2, m=2"),
    ([1,1,0], [0,1,1], "pi/3, sqrt2"),
    ([1,0,0], [0,1,1], "pi/2, mixed"),
    ([1,0,0], [1,1,1], "~55deg, mixed"),
]

print(f"{'k':>12} {'p':>12} {'K_same':>12} {'K_cross':>12} {'K_base':>12} {'split':>12}")
print("-" * 78)

for kv, pv, label in tests:
    Ks, ph = compute_curvature(kv, pv, +1, +1)
    Kc, _ = compute_curvature(kv, pv, +1, -1)
    Kb = (Ks + Kc) / 2
    sp = (Kc - Ks) / 2
    print(f"  {str(kv):>10} {str(pv):>10} {Ks:>12.6f} {Kc:>12.6f} {Kb:>12.6f} {sp:>12.6f}")
