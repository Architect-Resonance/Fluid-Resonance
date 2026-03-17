"""
REAL AORTIC FLOW: Leray suppression measurement
=================================================
Measures alpha (cross-helical Leray suppression) on REAL aortic CFD data
from the 4DFlowNet dataset (Ferdian et al., Frontiers in Physics 2020).

Data: aorta03_HR.h5 — single time frame, 328x88x112 grid, binary mask.
DOI: 10.17608/k6.auckland.24424888.v1, CC BY 4.0.

Pipeline: load u/v/w -> mask -> Hann window -> zero-pad to cubic -> FFT
-> helical decompose -> measure alpha.

Prediction: alpha ~ 0.05-0.10 for organized aortic flow (near-Beltrami).

S100-M2e, Meridian 2.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import h5py


def build_helical_basis(N):
    """Build helical basis vectors h+, h- for a cubic N^3 grid."""
    k1d = fftfreq(N, d=1.0 / N)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2_safe = k2.copy()
    k2_safe[0, 0, 0] = 1.0
    kmag = np.sqrt(k2_safe)

    khat = np.array([kx / kmag, ky / kmag, kz / kmag])

    # e1 = khat x (0,0,1), fallback to (0,1,0) when k || z
    e1 = np.array([-khat[1], khat[0], np.zeros_like(khat[0])])
    e1_mag = np.sqrt(np.sum(e1**2, axis=0))
    parallel = e1_mag < 1e-10
    if np.any(parallel):
        e1_alt = np.array([np.zeros_like(khat[0]), -khat[2], khat[1]])
        for i in range(3):
            e1[i] = np.where(parallel, e1_alt[i], e1[i])
        e1_mag = np.sqrt(np.sum(e1**2, axis=0))
    e1 /= np.maximum(e1_mag, 1e-15)

    e2 = np.array([
        khat[1] * e1[2] - khat[2] * e1[1],
        khat[2] * e1[0] - khat[0] * e1[2],
        khat[0] * e1[1] - khat[1] * e1[0],
    ])

    h_plus = (e1 + 1j * e2) / np.sqrt(2.0)
    h_minus = (e1 - 1j * e2) / np.sqrt(2.0)
    h_plus[:, 0, 0, 0] = 0.0
    h_minus[:, 0, 0, 0] = 0.0

    return kx, ky, kz, k2, k2_safe, kmag, h_plus, h_minus


def measure_alpha_from_fields(u_hat, kx, ky, kz, k2, k2_safe, h_plus, h_minus, dealias_mask):
    """Measure cross-helical Leray suppression factor from Fourier velocity field."""
    # Helical decomposition
    f_p = np.sum(np.conj(h_plus) * u_hat, axis=0)
    f_m = np.sum(np.conj(h_minus) * u_hat, axis=0)

    u_hat_plus = f_p[np.newaxis] * h_plus
    u_hat_minus = f_m[np.newaxis] * h_minus

    # Vorticity for each sector
    def vorticity(uh):
        return np.array([
            1j * (ky * uh[2] - kz * uh[1]),
            1j * (kz * uh[0] - kx * uh[2]),
            1j * (kx * uh[1] - ky * uh[0]),
        ])

    om_plus_hat = vorticity(u_hat_plus)
    om_minus_hat = vorticity(u_hat_minus)

    N = u_hat.shape[1]
    u_plus = np.array([np.real(ifftn(u_hat_plus[i])) for i in range(3)])
    u_minus = np.array([np.real(ifftn(u_hat_minus[i])) for i in range(3)])
    om_plus = np.array([np.real(ifftn(om_plus_hat[i])) for i in range(3)])
    om_minus = np.array([np.real(ifftn(om_minus_hat[i])) for i in range(3)])

    # Cross-helical Lamb vector: u+ x omega- + u- x omega+
    def cross(a, b):
        return np.array([
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0],
        ])

    lamb_cross = cross(u_plus, om_minus) + cross(u_minus, om_plus)
    lamb_cross_hat = np.array([fftn(lamb_cross[i]) for i in range(3)])
    for i in range(3):
        lamb_cross_hat[i] *= dealias_mask

    E_total = np.sum(np.abs(lamb_cross_hat)**2)
    if E_total < 1e-30:
        return 0.0, 0.0, 0.0

    # Leray projection
    P = {}
    K = [kx, ky, kz]
    for i in range(3):
        for j in range(3):
            P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / k2_safe

    lamb_sol_hat = np.zeros_like(lamb_cross_hat)
    for i in range(3):
        for j in range(3):
            lamb_sol_hat[i] += P[(i, j)] * lamb_cross_hat[j]

    E_sol = np.sum(np.abs(lamb_sol_hat)**2)
    alpha = E_sol / E_total

    # Also compute helicity ratio |H|/(E*Z)^{1/2}
    u = np.array([np.real(ifftn(u_hat[i])) for i in range(3)])
    om_hat = vorticity(u_hat)
    om = np.array([np.real(ifftn(om_hat[i])) for i in range(3)])
    E = 0.5 * np.mean(np.sum(u**2, axis=0))
    Z = 0.5 * np.mean(np.sum(om**2, axis=0))
    H = np.mean(np.sum(u * om, axis=0))
    h_ratio = abs(H) / np.sqrt(max(E * Z, 1e-30))

    return alpha, h_ratio, E


def hann_3d(shape):
    """3D Hann window."""
    windows = []
    for n in shape:
        windows.append(np.hanning(n))
    w = windows[0][:, None, None] * windows[1][None, :, None] * windows[2][None, None, :]
    return w


def main():
    print("=" * 70)
    print("  REAL AORTIC FLOW: Leray Suppression Measurement")
    print("  Data: 4DFlowNet aorta03_HR.h5 (Ferdian et al. 2020)")
    print("=" * 70)

    h5_path = os.path.join(os.path.dirname(__file__), 'aorta03_HR.h5')
    if not os.path.exists(h5_path):
        print(f"  ERROR: {h5_path} not found. Download from Figshare first.")
        return

    # Load data
    print("\n  Loading data...")
    with h5py.File(h5_path, 'r') as f:
        u_raw = f['u'][0]  # (328, 88, 112)
        v_raw = f['v'][0]
        w_raw = f['w'][0]
        mask = f['mask'][0]

    shape_orig = u_raw.shape
    n_masked = np.sum(mask > 0.5)
    print(f"  Original shape: {shape_orig}")
    print(f"  Masked voxels: {n_masked} ({100*n_masked/mask.size:.1f}%)")
    print(f"  u range: [{np.min(u_raw[mask>0.5]):.4f}, {np.max(u_raw[mask>0.5]):.4f}]")
    print(f"  v range: [{np.min(v_raw[mask>0.5]):.4f}, {np.max(v_raw[mask>0.5]):.4f}]")
    print(f"  w range: [{np.min(w_raw[mask>0.5]):.4f}, {np.max(w_raw[mask>0.5]):.4f}]")

    # Strategy: Extract a cubic sub-volume around the aorta center of mass,
    # apply mask + Hann window, then FFT.
    # Also try the full non-cubic grid with zero-padding to cubic.

    results = {}

    # ================================================================
    # METHOD 1: Full grid, Hann windowed, zero-padded to cubic
    # ================================================================
    print("\n  METHOD 1: Full grid -> Hann window -> pad to N^3")
    print("  " + "-" * 55)

    # Apply mask (zero outside aorta)
    u_masked = u_raw * mask
    v_masked = v_raw * mask
    w_masked = w_raw * mask

    # Hann window (reduces spectral leakage from non-periodicity)
    hann = hann_3d(shape_orig)
    u_win = u_masked * hann
    v_win = v_masked * hann
    w_win = w_masked * hann

    # Zero-pad to cubic (use max dimension)
    N_cube = max(shape_orig)  # 328
    # That's too large for full helical decomposition. Use a smaller cube.
    # Downsample: take every 2nd point -> 164x44x56, pad to 164^3
    # Still large. Let's use every 4th -> 82x22x28, pad to 82... not great.
    # Better: extract the bounding box of the mask and pad that to cubic.

    # Find mask bounding box
    nz = np.nonzero(mask > 0.5)
    bbox = [(nz[i].min(), nz[i].max() + 1) for i in range(3)]
    print(f"  Mask bounding box: {bbox}")
    bbox_size = tuple(b[1] - b[0] for b in bbox)
    print(f"  Bounding box size: {bbox_size}")

    # Extract bounding box
    sl = tuple(slice(b[0], b[1]) for b in bbox)
    u_box = u_masked[sl]
    v_box = v_masked[sl]
    w_box = w_masked[sl]
    mask_box = mask[sl]

    print(f"  Box shape: {u_box.shape}, mask fill: {100*np.mean(mask_box > 0.5):.1f}%")

    # Hann window on the box
    hann_box = hann_3d(u_box.shape)
    u_bw = u_box * hann_box
    v_bw = v_box * hann_box
    w_bw = w_box * hann_box

    # Pad to cubic (next power of 2 or just max dim)
    N_pad = max(u_box.shape)
    # Round up to even for FFT efficiency
    if N_pad % 2 != 0:
        N_pad += 1
    print(f"  Padding to cubic N={N_pad}")

    u_cube = np.zeros((N_pad, N_pad, N_pad))
    v_cube = np.zeros((N_pad, N_pad, N_pad))
    w_cube = np.zeros((N_pad, N_pad, N_pad))

    # Center the data in the cube
    off = [(N_pad - s) // 2 for s in u_box.shape]
    sl_dst = tuple(slice(off[i], off[i] + u_box.shape[i]) for i in range(3))
    u_cube[sl_dst] = u_bw
    v_cube[sl_dst] = v_bw
    w_cube[sl_dst] = w_bw

    print(f"  FFT + helical decomposition on {N_pad}^3 grid...")

    # Build Fourier infrastructure
    kx, ky, kz, k2, k2_safe, kmag, h_plus, h_minus = build_helical_basis(N_pad)

    # 2/3 dealiasing mask
    kmax = N_pad // 3
    dealias_mask = (
        (np.abs(kx) <= kmax) &
        (np.abs(ky) <= kmax) &
        (np.abs(kz) <= kmax)
    )

    # FFT
    u_hat = np.array([fftn(u_cube), fftn(v_cube), fftn(w_cube)])

    # Leray project to ensure solenoidal (CFD data should already be, but windowing breaks it)
    P = {}
    K = [kx, ky, kz]
    for i in range(3):
        for j in range(3):
            P[(i, j)] = (1.0 if i == j else 0.0) - K[i] * K[j] / k2_safe

    u_hat_sol = np.zeros_like(u_hat)
    for i in range(3):
        for j in range(3):
            u_hat_sol[i] += P[(i, j)] * u_hat[j]

    # Measure divergence before/after Leray
    div_before = np.sqrt(np.mean(np.abs(1j*kx*u_hat[0] + 1j*ky*u_hat[1] + 1j*kz*u_hat[2])**2))
    div_after = np.sqrt(np.mean(np.abs(1j*kx*u_hat_sol[0] + 1j*ky*u_hat_sol[1] + 1j*kz*u_hat_sol[2])**2))
    print(f"  Divergence RMS: before Leray = {div_before:.2e}, after = {div_after:.2e}")

    # Measure alpha
    alpha_m1, h_ratio_m1, E_m1 = measure_alpha_from_fields(
        u_hat_sol, kx, ky, kz, k2, k2_safe, h_plus, h_minus, dealias_mask)

    print(f"\n  METHOD 1 RESULTS:")
    print(f"    alpha (cross-helical Leray suppression) = {alpha_m1:.6f}")
    print(f"    Helicity ratio |H|/(EZ)^{1/2}          = {h_ratio_m1:.6f}")
    print(f"    Energy E                                = {E_m1:.6e}")
    results['Method 1: Full box + Hann'] = alpha_m1

    # ================================================================
    # METHOD 2: Subvolumes along the aorta
    # ================================================================
    print("\n\n  METHOD 2: Local sub-cubes along the aorta")
    print("  " + "-" * 55)

    # Extract multiple small cubes along the aorta axis (dimension 0 = long axis)
    sub_N = 64  # cube size
    n_subs = 0
    alphas_local = []
    h_ratios_local = []

    # Build infrastructure once for sub_N
    kx_s, ky_s, kz_s, k2_s, k2s_safe, kmag_s, hp_s, hm_s = build_helical_basis(sub_N)
    kmax_s = sub_N // 3
    dm_s = (np.abs(kx_s) <= kmax_s) & (np.abs(ky_s) <= kmax_s) & (np.abs(kz_s) <= kmax_s)

    # Slide along dimension 0 of the bounding box
    step = sub_N // 2  # 50% overlap
    centers = []

    for i0 in range(bbox[0][0], bbox[0][1] - sub_N + 1, step):
        # Check if this slice has enough mask coverage
        sl_check = (slice(i0, i0 + sub_N),
                     slice(bbox[1][0], min(bbox[1][0] + sub_N, shape_orig[1])),
                     slice(bbox[2][0], min(bbox[2][0] + sub_N, shape_orig[2])))

        mask_sub = mask[sl_check]
        # Pad to sub_N^3 if needed
        actual_shape = mask_sub.shape
        fill = np.sum(mask_sub > 0.5) / sub_N**3

        if fill < 0.01:  # Skip nearly empty cubes
            continue

        # Extract and zero-pad
        u_sub = np.zeros((sub_N, sub_N, sub_N))
        v_sub = np.zeros((sub_N, sub_N, sub_N))
        w_sub = np.zeros((sub_N, sub_N, sub_N))

        s0 = actual_shape[0]
        s1 = min(actual_shape[1], sub_N)
        s2 = min(actual_shape[2], sub_N)

        u_sub[:s0, :s1, :s2] = u_masked[i0:i0+s0, bbox[1][0]:bbox[1][0]+s1, bbox[2][0]:bbox[2][0]+s2]
        v_sub[:s0, :s1, :s2] = v_masked[i0:i0+s0, bbox[1][0]:bbox[1][0]+s1, bbox[2][0]:bbox[2][0]+s2]
        w_sub[:s0, :s1, :s2] = w_masked[i0:i0+s0, bbox[1][0]:bbox[1][0]+s1, bbox[2][0]:bbox[2][0]+s2]

        # Hann window
        hann_s = hann_3d((sub_N, sub_N, sub_N))
        u_sub *= hann_s
        v_sub *= hann_s
        w_sub *= hann_s

        # FFT + Leray project
        uh = np.array([fftn(u_sub), fftn(v_sub), fftn(w_sub)])

        P_s = {}
        K_s = [kx_s, ky_s, kz_s]
        for ii in range(3):
            for jj in range(3):
                P_s[(ii, jj)] = (1.0 if ii == jj else 0.0) - K_s[ii] * K_s[jj] / k2s_safe
        uh_sol = np.zeros_like(uh)
        for ii in range(3):
            for jj in range(3):
                uh_sol[ii] += P_s[(ii, jj)] * uh[jj]

        a, hr, _ = measure_alpha_from_fields(uh_sol, kx_s, ky_s, kz_s, k2_s, k2s_safe, hp_s, hm_s, dm_s)
        alphas_local.append(a)
        h_ratios_local.append(hr)
        centers.append(i0 + sub_N // 2)
        n_subs += 1

    alphas_local = np.array(alphas_local)
    h_ratios_local = np.array(h_ratios_local)
    centers = np.array(centers)

    print(f"  Analyzed {n_subs} sub-cubes (N={sub_N}, step={step})")
    print(f"\n  {'Position':>10}  {'alpha':>10}  {'|H|/(EZ)^1/2':>14}")
    print("  " + "-" * 40)
    for i in range(len(alphas_local)):
        print(f"  {centers[i]:>10d}  {alphas_local[i]:>10.6f}  {h_ratios_local[i]:>14.6f}")

    if len(alphas_local) > 0:
        print(f"\n  METHOD 2 RESULTS:")
        print(f"    alpha mean = {np.mean(alphas_local):.6f}")
        print(f"    alpha std  = {np.std(alphas_local):.6f}")
        print(f"    alpha range: [{np.min(alphas_local):.6f}, {np.max(alphas_local):.6f}]")
        print(f"    helicity ratio mean = {np.mean(h_ratios_local):.6f}")
        results['Method 2: Local cubes'] = np.mean(alphas_local)

    # ================================================================
    # Comparison with model predictions
    # ================================================================
    print("\n\n  COMPARISON WITH MODEL PREDICTIONS")
    print("  " + "-" * 55)
    print(f"  Model prediction (cardiac_healthy_vs_diseased.py):")
    print(f"    Healthy vortex ring:  alpha ~ 0.066")
    print(f"    Taylor-Green (achiral): alpha ~ 0.087")
    print(f"    Random (diseased):    alpha ~ 0.393")
    print(f"    Isotropic average:    alpha = 1 - ln2 = 0.307")
    print(f"    Beltrami (ABC):       alpha ~ 0.000")
    print(f"")
    for name, a in results.items():
        category = "ORGANIZED" if a < 0.15 else "MODERATE" if a < 0.25 else "DISORDERED"
        print(f"    {name}: alpha = {a:.6f}  [{category}]")

    if alpha_m1 < 0.15:
        print(f"\n  >>> PREDICTION CONFIRMED: Real aortic flow shows LOW alpha ({alpha_m1:.3f})")
        print(f"  >>> Organized helical flow -> strong Leray suppression")
    elif alpha_m1 < 0.25:
        print(f"\n  >>> MODERATE alpha ({alpha_m1:.3f}) — partially organized")
    else:
        print(f"\n  >>> SURPRISE: alpha = {alpha_m1:.3f} higher than predicted. Investigate.")

    # ================================================================
    # 4-panel figure
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Real Aortic Flow: Leray Suppression (4DFlowNet aorta03)', fontsize=14)

    # Panel 1: Mid-slice velocity magnitude
    ax = axes[0, 0]
    mid = shape_orig[0] // 2
    speed = np.sqrt(u_raw[mid]**2 + v_raw[mid]**2 + w_raw[mid]**2)
    speed[mask[mid] < 0.5] = np.nan
    im = ax.imshow(speed.T, origin='lower', cmap='hot')
    ax.set_title(f'Velocity magnitude (slice z={mid})')
    plt.colorbar(im, ax=ax, label='|u| (m/s)')

    # Panel 2: Alpha along aorta (Method 2)
    ax = axes[0, 1]
    if len(alphas_local) > 0:
        ax.plot(centers, alphas_local, 'bo-', lw=2, ms=6, label='alpha (local)')
        ax.axhline(0.066, color='g', ls='--', lw=1, label='Model: healthy vortex ring')
        ax.axhline(0.307, color='r', ls='--', lw=1, label='Isotropic: 1-ln2')
        ax.axhline(alpha_m1, color='purple', ls=':', lw=2, label=f'Method 1 global: {alpha_m1:.3f}')
        ax.set_xlabel('Position along aorta')
        ax.set_ylabel('alpha')
        ax.set_title('Leray suppression along aorta')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No sub-cubes', ha='center', va='center', transform=ax.transAxes)

    # Panel 3: Helicity ratio along aorta
    ax = axes[1, 0]
    if len(h_ratios_local) > 0:
        ax.plot(centers, h_ratios_local, 'rs-', lw=2, ms=6)
        ax.set_xlabel('Position along aorta')
        ax.set_ylabel('|H| / (E*Z)^{1/2}')
        ax.set_title('Helicity ratio along aorta (1=Beltrami)')
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'No sub-cubes', ha='center', va='center', transform=ax.transAxes)

    # Panel 4: Comparison bar chart
    ax = axes[1, 1]
    labels = ['Beltrami\n(ABC)', 'Healthy\nvortex ring', 'REAL\nAORTA', 'Taylor-\nGreen', 'Isotropic\n1-ln2', 'Random\n(diseased)']
    values = [0.0, 0.066, alpha_m1, 0.087, 0.307, 0.393]
    colors_bar = ['#2ecc71', '#27ae60', '#364FC7', '#f39c12', '#e74c3c', '#c0392b']
    bars = ax.bar(labels, values, color=colors_bar, edgecolor='black', lw=0.5)
    ax.set_ylabel('alpha (Leray suppression)')
    ax.set_title('Aortic flow in context')
    ax.axhline(0.25, color='gray', ls=':', lw=1, label='alpha_E = 1/4')
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'real_aortic_alpha.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved to {out_path}")
    print("\n  DONE.")


if __name__ == '__main__':
    main()
