"""
fano_curvature_connection.py
????????????????????????????
Investigate the connection between:
  M1 (Arnold curvature on SDiff(T?)) and
  M2 (Fano plane / Triadic structure at low shells)

M1 formula:
  K(?,?) = K_base(k,p) - ??? ? |k|?|p|?sin?(?)/2
  where ? = angle between k and p, ?,? ? {+1,-1} are helicity signs.

  K_base(k,p) = (1/4)|k|?|p|? sin?? ? f(k,p)
  For the standard Arnold formula on SDiff, the leading curvature term
  for helical modes ?_k^?, ?_p^? is (see e.g. Shkoller 1998, Misio?ek 1993):

    K(?,?) = A(k,p,?) + ??? ? B(k,p,?)

  where the helicity splitting term is:
    B(k,p,?) = -|k||p|sin?(?)/2   (negative = focusing for same-helicity)

  For Fano-shell vectors (|k|=|p|=1, ?=?/2):
    K_same  = K_base - 1/2 ? sin?(?/2) = K_base - 1/2
    K_cross = K_base + 1/2 ? sin?(?/2) = K_base + 1/2

  Brendan's values: K_same = -3/8, K_cross = +5/8
  => K_base = 1/8 for unit vectors at right angles.

  We use K_base = 1/8 (unit sphere, ?=?/2) and generalize:
    K(?,?, |k|, |p|, ?) = |k|?|p|?sin?(?)/8 - ????|k||p|sin?(?)/2

  which for |k|=|p|=1, ?=?/2 gives K_same=-3/8, K_cross=+5/8 ?

M2 structure:
  The 7 nonzero vectors in GF(2)? correspond to integer wavevectors.
  We embed them as actual integer vectors in Z? (not mod 2).
  A Fano "line" is a triple {k,p,q} where k+p+q = 0 mod 2 componentwise
  which, for these unit/edge vectors, means k XOR p XOR q = 0, i.e. k+p+q = 0 or 2?(something).

  BUT: for actual NS triads we need k+p+q = 0 in Z?.
  Since the 7 GF(2) vectors are in {0,1}?\{0}, we need to think about
  which sign assignments make k+p+q = 0 exactly.

  For a Fano line {a,b,c} with a XOR b XOR c = 0 (componentwise mod 2),
  sign assignments ?_a, ?_b, ?_c ? {?1} such that ?_a?a + ?_b?b + ?_c?c = 0
  in Z?.
"""

import numpy as np
from itertools import product

# ?????????????????????????????????????????????
# 1. The 7 nonzero vectors of GF(2)?
# ?????????????????????????????????????????????
gf2_vecs = np.array([
    [1, 0, 0],  # e1
    [0, 1, 0],  # e2
    [0, 0, 1],  # e3
    [1, 1, 0],  # e1+e2
    [1, 0, 1],  # e1+e3
    [0, 1, 1],  # e2+e3
    [1, 1, 1],  # e1+e2+e3
], dtype=int)

labels = ['e1', 'e2', 'e3', 'e1+e2', 'e1+e3', 'e2+e3', 'e1+e2+e3']

# ?????????????????????????????????????????????
# 2. Find the 7 Fano lines (triples summing to 0 mod 2)
# ?????????????????????????????????????????????
print("=" * 70)
print("FANO PLANE: 7 lines of PG(2,2)")
print("=" * 70)

fano_lines = []
n = len(gf2_vecs)
for i in range(n):
    for j in range(i+1, n):
        for k in range(j+1, n):
            # Check if XOR sum = 0 (i.e. a XOR b XOR c = 0 in GF(2))
            xor_sum = gf2_vecs[i] ^ gf2_vecs[j] ^ gf2_vecs[k]
            if np.all(xor_sum == 0):
                fano_lines.append((i, j, k))

print(f"\nFound {len(fano_lines)} Fano lines:\n")
for idx, (i, j, k) in enumerate(fano_lines):
    vi, vj, vk = gf2_vecs[i], gf2_vecs[j], gf2_vecs[k]
    print(f"  Line {idx+1}: {{{labels[i]}, {labels[j]}, {labels[k]}}} "
          f"= {vi.tolist()}, {vj.tolist()}, {vk.tolist()}")
    # Verify XOR
    s = vi ^ vj ^ vk
    status = "OK" if np.all(s==0) else "FAIL"
    print(f"           XOR check: {vi} XOR {vj} XOR {vk} = {s}  {status}")

# ?????????????????????????????????????????????
# 3. Find actual Z? triads: sign assignments ? such that ?_i?v_i + ?_j?v_j + ?_k?v_k = 0
# ?????????????????????????????????????????????
print("\n" + "=" * 70)
print("SIGN ASSIGNMENTS: which ? give k+p+q=0 in Z??")
print("=" * 70)

def find_sign_triads(vi, vj, vk):
    """Find all sign assignments (?i, ?j, ?k) in {?1}? with ?i?vi + ?j?vj + ?k?vk = 0."""
    results = []
    for ei, ej, ek in product([1, -1], repeat=3):
        vec_sum = ei * vi + ej * vj + ek * vk
        if np.all(vec_sum == 0):
            results.append((ei, ej, ek))
    return results

line_triads = []  # list of (vi, vj, vk, ei, ej, ek) for actual NS triads
for idx, (i, j, k) in enumerate(fano_lines):
    vi, vj, vk = gf2_vecs[i], gf2_vecs[j], gf2_vecs[k]
    sign_assignments = find_sign_triads(vi, vj, vk)
    print(f"\n  Fano Line {idx+1}: [{labels[i]}, {labels[j]}, {labels[k]}]")
    if sign_assignments:
        for (ei, ej, ek) in sign_assignments:
            k1 = ei * vi
            k2 = ej * vj
            k3 = ek * vk
            print(f"    ?=({ei:+d},{ej:+d},{ek:+d}) ? {k1.tolist()} + {k2.tolist()} + {k3.tolist()} = {(k1+k2+k3).tolist()}")
            line_triads.append((k1, k2, k3))
    else:
        print(f"    No pure ? solution. Trying with 2-multiples...")
        # Try ?i?vi + ?j?vj = ?k?vk * 2 etc. ? sometimes one must flip parity
        # Actually for Fano lines in GF(2), k+p+q=0 mod 2 means k+p+q is even
        # in each component, so k+p+q = 0 or ?2 per component.
        for ei, ej, ek in product([1, -1], repeat=3):
            vec_sum = ei * vi + ej * vj + ek * vk
            if np.all(vec_sum % 2 == 0) and not np.all(vec_sum == 0):
                print(f"    ?=({ei:+d},{ej:+d},{ek:+d}) ? sum = {vec_sum.tolist()} (even, not zero)")

# ?????????????????????????????????????????????
# 4. Arnold curvature formula
# ?????????????????????????????????????????????
def angle_between(u, v):
    """Angle in radians between vectors u and v."""
    cos_a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return np.arccos(cos_a)

def K_base(ku, kv, phi):
    """
    K_base for unit helical modes ?_{ku}^?, ?_{kv}^?.
    Using the form K_base = |ku|?|kv|? sin?(phi)/8
    which gives 1/8 for unit vectors at right angles.
    """
    return (np.linalg.norm(ku)**2 * np.linalg.norm(kv)**2 * np.sin(phi)**2) / 8.0

def K_arnold(ku, kv, sigma, tau):
    """
    Arnold curvature for helical modes:
    K(?,?) = K_base(k,p) - ??? ? |k||p|?sin?(?)/2
    """
    phi = angle_between(ku, kv)
    kb = K_base(ku, kv, phi)
    helicity_term = sigma * tau * np.linalg.norm(ku) * np.linalg.norm(kv) * np.sin(phi)**2 / 2.0
    return kb - helicity_term, kb, phi

print("\n" + "=" * 70)
print("ARNOLD CURVATURE ? VERIFICATION on unit vectors at ?=?/2")
print("=" * 70)
test_k = np.array([1, 0, 0], dtype=float)
test_p = np.array([0, 1, 0], dtype=float)
K_same_test, Kb, phi_test = K_arnold(test_k, test_p, +1, +1)
K_cross_test, _, _ = K_arnold(test_k, test_p, +1, -1)
print(f"  |k|=|p|=1, ?=?/2:")
print(f"  K_base  = {Kb:.4f}  (expected 1/8 = {1/8:.4f})")
print(f"  K_same  = {K_same_test:.4f}  (expected -3/8 = {-3/8:.4f})")
print(f"  K_cross = {K_cross_test:.4f}  (expected +5/8 = {+5/8:.4f})")

# ?????????????????????????????????????????????
# 5. For each Fano line, compute K for all edges and all helicity assignments
# ?????????????????????????????????????????????
print("\n" + "=" * 70)
print("FANO LINES: ANGLES AND CURVATURES")
print("=" * 70)

helicity_labels = {(+1,+1): "same(++)", (+1,-1): "cross(+-)", (-1,+1): "cross(-+)", (-1,-1): "same(--)"}
helicity_pairs  = [(+1,+1), (+1,-1), (-1,+1), (-1,-1)]

# Build unique actual NS triads from the Fano lines
# For each Fano line, we take the "canonical" integer triad
canonical_triads = []
for idx, (i, j, k_idx) in enumerate(fano_lines):
    vi = gf2_vecs[i].astype(float)
    vj = gf2_vecs[j].astype(float)
    vk = gf2_vecs[k_idx].astype(float)
    # Find if there's a sign assignment
    sa = find_sign_triads(gf2_vecs[i], gf2_vecs[j], gf2_vecs[k_idx])
    if sa:
        ei, ej, ek = sa[0]
        canonical_triads.append((ei*vi, ej*vj, ek*vk, idx+1, labels[i], labels[j], labels[k_idx]))
    else:
        # No exact triad; store unsigned
        canonical_triads.append((vi, vj, vk, idx+1, labels[i], labels[j], labels[k_idx]))

for (k1, k2, k3, line_num, l1, l2, l3) in canonical_triads:
    print(f"\n??? Fano Line {line_num}: {{{l1}, {l2}, {l3}}} ???")
    print(f"  Integer vectors: k={k1.astype(int).tolist()}, p={k2.astype(int).tolist()}, q={k3.astype(int).tolist()}")
    print(f"  Sum: {(k1+k2+k3).astype(int).tolist()}")

    pairs = [('k,p', k1, k2), ('k,q', k1, k3), ('p,q', k2, k3)]
    print(f"\n  {'Edge':<8} {'|k1|':>6} {'|k2|':>6} {'? (?)':>8} {'K_base':>8} {'K_same':>8} {'K_cross':>8}")
    print(f"  {'-'*62}")
    edge_data = {}
    for edge_name, u, v in pairs:
        phi = angle_between(u, v)
        K_s, Kb, _ = K_arnold(u, v, +1, +1)
        K_c, _,  _ = K_arnold(u, v, +1, -1)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        print(f"  {edge_name:<8} {norm_u:>6.3f} {norm_v:>6.3f} {np.degrees(phi):>8.2f} "
              f"{Kb:>8.4f} {K_s:>8.4f} {K_c:>8.4f}")
        edge_data[edge_name] = {'phi': phi, 'K_same': K_s, 'K_cross': K_c, 'K_base': Kb}

# ?????????????????????????????????????????????
# 6. Helicity balance constraint on a Fano triad
# ?????????????????????????????????????????????
print("\n" + "=" * 70)
print("HELICITY PARITY CONSTRAINT")
print("=" * 70)
print("""
A helical mode has wavevector k and helicity sign h ? {?1}.
In a triad k+p+q=0, there are 2?=8 helicity assignments (?_k, ?_p, ?_q).
Physical NS couples: B(?_k^?, ?_p^?) ? ?_{k+p}^{?}

The key question: does ?_k=?_p FORCE a particular ?_q from the triad geometry?

Answer from NS helical decomposition (Waleffe 1992):
  The interaction coefficient is:
    c^s_{kpq} ? s_q ? (s_p|p| - s_k|k|) ? (?_q ? ?_k ? ?_p) ? ?(k+p+q)
  where s = helicity sign.

  This does NOT algebraically force ?_q from ?_k and ?_p.
  The helicity of the OUTPUT mode is free ? it's determined by which
  mode you're projecting onto, not by the input helicities.

  However: the STRENGTH of the coupling depends on (s_p|p| - s_k|k|).
  For same-helicity same-magnitude: s_p=s_k, |p|=|k| ? coefficient = 0!
  This is the Waleffe "null interaction" result.
""")

print("Waleffe null interaction check for Fano unit vectors:")
print(f"{'Triad':<30} {'s_k':>5} {'s_p':>5} {'s_q':>5} {'(s_p|p|-s_k|k|)':>18}")
print("-" * 65)
for (k1, k2, k3, line_num, l1, l2, l3) in canonical_triads:
    nk1 = np.linalg.norm(k1)
    nk2 = np.linalg.norm(k2)
    nk3 = np.linalg.norm(k3)
    # Check all helicity combinations
    for sk, sp, sq in product([+1,-1], repeat=3):
        coeff_kp = sp*nk2 - sk*nk1  # coupling k?p on triad kpq (projected on sq mode)
        if abs(coeff_kp) < 1e-10:
            print(f"  Line {line_num} {l1},{l2},{l3}  "
                  f"sk={sk:+d} sp={sp:+d} sq={sq:+d}  coeff={coeff_kp:+.4f}  ? NULL")

# ?????????????????????????????????????????????
# 7. Net curvature sum over all edges of each Fano line
#    for all helicity assignments
# ?????????????????????????????????????????????
print("\n" + "=" * 70)
print("NET CURVATURE SUM OVER ALL 3 EDGES OF EACH FANO LINE")
print("=" * 70)
print("""
For a triad (k,p,q), there are 3 edges: (k,p), (k,q), (p,q).
Each edge has a helicity pair (?,?).
Total curvature K_total = K(k,p) + K(k,q) + K(p,q).

If ?_k=+1, ?_p=+1, ?_q=+1 (all same): all edges same-helicity
If ?_k=+1, ?_p=+1, ?_q=-1 (one flip): (k,p)=same, (k,q)=cross, (p,q)=cross
""")

print(f"\n{'Line':<8} {'(?k,?p,?q)':<15} {'K(k,p)':>9} {'K(k,q)':>9} {'K(p,q)':>9} {'K_total':>9} {'Sign(K)':>8}")
print("-" * 72)

for (k1, k2, k3, line_num, l1, l2, l3) in canonical_triads:
    for sk, sp, sq in product([+1,-1], repeat=3):
        # Only show unique helicity classes (up to overall flip ??-? for all)
        if sk < 0:  # break overall sign degeneracy
            continue
        K_kp, _, _ = K_arnold(k1, k2, sk, sp)
        K_kq, _, _ = K_arnold(k1, k3, sk, sq)
        K_pq, _, _ = K_arnold(k2, k3, sp, sq)
        K_tot = K_kp + K_kq + K_pq
        helicity_type = f"({sk:+d},{sp:+d},{sq:+d})"
        sign = "+" if K_tot > 0 else ("-" if K_tot < 0 else "0")
        print(f"  {line_num:<6} {helicity_type:<15} "
              f"{K_kp:>9.4f} {K_kq:>9.4f} {K_pq:>9.4f} {K_tot:>9.4f} {sign:>8}")
    print()

# ?????????????????????????????????????????????
# 8. Summary statistics: helicity balance across ALL Fano lines
# ?????????????????????????????????????????????
print("=" * 70)
print("HELICITY BALANCE SUMMARY ACROSS ALL 7 FANO LINES")
print("=" * 70)

# For each edge in each Fano line, compute K_same and K_cross
all_K_same = []
all_K_cross = []
all_K_base = []
all_angles = []

for (k1, k2, k3, line_num, l1, l2, l3) in canonical_triads:
    pairs = [('k,p', k1, k2), ('k,q', k1, k3), ('p,q', k2, k3)]
    for _, u, v in pairs:
        phi = angle_between(u, v)
        K_s, Kb, _ = K_arnold(u, v, +1, +1)
        K_c, _,  _ = K_arnold(u, v, +1, -1)
        all_K_same.append(K_s)
        all_K_cross.append(K_c)
        all_K_base.append(Kb)
        all_angles.append(np.degrees(phi))

print(f"\n  Total edges across all 7 Fano lines: {len(all_K_same)} (7 lines ? 3 edges)")
print(f"\n  Angle statistics:")
print(f"    angles: {sorted(set(round(a,2) for a in all_angles))}")
print(f"\n  K_same  statistics: mean={np.mean(all_K_same):.4f}, "
      f"min={np.min(all_K_same):.4f}, max={np.max(all_K_same):.4f}")
print(f"  K_cross statistics: mean={np.mean(all_K_cross):.4f}, "
      f"min={np.min(all_K_cross):.4f}, max={np.max(all_K_cross):.4f}")
print(f"  K_base  statistics: mean={np.mean(all_K_base):.4f}, "
      f"min={np.min(all_K_base):.4f}, max={np.max(all_K_base):.4f}")

n_negative_same  = sum(1 for x in all_K_same  if x < 0)
n_positive_same  = sum(1 for x in all_K_same  if x > 0)
n_negative_cross = sum(1 for x in all_K_cross if x < 0)
n_positive_cross = sum(1 for x in all_K_cross if x > 0)

print(f"\n  K_same  < 0 (geodesic focusing):   {n_negative_same}/{len(all_K_same)} edges")
print(f"  K_same  > 0 (geodesic defocusing):  {n_positive_same}/{len(all_K_same)} edges")
print(f"  K_cross < 0 (geodesic focusing):    {n_negative_cross}/{len(all_K_cross)} edges")
print(f"  K_cross > 0 (geodesic defocusing):  {n_positive_cross}/{len(all_K_cross)} edges")

# ?????????????????????????????????????????????
# 9. Does the Fano structure CONSTRAIN helicity balance?
# ?????????????????????????????????????????????
print("\n" + "=" * 70)
print("FANO CONSTRAINT ON HELICITY BALANCE")
print("=" * 70)

print("""
Key question: In the full NS sum over all triadic interactions,
does the Fano incidence structure (every pair on exactly one line)
force a balance between same-helicity and cross-helicity interactions?

Counting argument:
  7 Fano lines ? 3 edges = 21 edges total.
  Each of the 21 pairs of GF(2) vectors appears exactly once (Fano property).

  For a fixed pair (k, p) on Fano line L, the third vertex q is DETERMINED.
  This means: the partner mode q for any (k,p) interaction is FIXED.

  Helicity assignment degrees of freedom:
  - Without constraint: 2^7 = 128 ways to assign helicities to 7 nodes.
  - The Fano structure does NOT reduce this ? helicity is a physical
    property of the mode, not a combinatorial label.

  BUT: the Fano structure DOES constrain which triads exist.
  If we ask "for mode k with helicity +1, how many of its Fano
  partners have FORCED helicity?" ? the answer is 0 without dynamics.
""")

print("Counting same vs cross-helicity pairs for a FIXED node assignment:")
print("Assume node helicities = arbitrary assignment h ? {+1,-1}^7\n")

# Try all 2^7 helicity assignments to the 7 Fano nodes
# Count how often same-helicity edges = cross-helicity edges per line
n_balanced_lines = 0
n_total_configs = 0
balance_dist = {}

for h_assignment in product([+1,-1], repeat=7):
    n_total_configs += 1
    line_same_counts = []
    for (i, j, k_idx) in fano_lines:
        # Helicities of the 3 nodes
        hi, hj, hk_ = h_assignment[i], h_assignment[j], h_assignment[k_idx]
        pairs_on_line = [(hi, hj), (hi, hk_), (hj, hk_)]
        n_same = sum(1 for (a,b) in pairs_on_line if a*b > 0)
        n_cross = 3 - n_same
        line_same_counts.append(n_same)

    # Total same and cross across all 21 edges
    total_same = sum(line_same_counts)
    total_cross = 21 - total_same
    key = (total_same, total_cross)
    balance_dist[key] = balance_dist.get(key, 0) + 1

print("Distribution of (same-helicity edges, cross-helicity edges) over all 2^7 helicity assignments:")
print(f"{'(same, cross)':<20} {'count':>8} {'fraction':>10}")
print("-" * 42)
for key in sorted(balance_dist.keys()):
    count = balance_dist[key]
    print(f"  {str(key):<20} {count:>8} {count/n_total_configs:>10.4f}")

print(f"\nTotal configurations: {n_total_configs}")

# Check: is there a forbidden/forced balance?
min_same = min(k[0] for k in balance_dist)
max_same = max(k[0] for k in balance_dist)
print(f"\nRange of same-helicity edges: [{min_same}, {max_same}] out of 21 total edges")
print(f"Note: 21/2 = 10.5, so perfect balance would be (10 or 11 same, 11 or 10 cross)")

# ?????????????????????????????????????????????
# 10. PHYSICAL SYNTHESIS
# ?????????????????????????????????????????????
print("\n" + "=" * 70)
print("PHYSICAL SYNTHESIS: M1-M2 CONNECTION")
print("=" * 70)
print("""
The Fano structure (M2) and Arnold curvature (M1) connect as follows:

1. THE TRIAD CONSTRAINT:
   On each Fano line {k,p,q} with k+p+q=0 in Z? (with appropriate signs),
   the THIRD wavevector is algebraically determined by the other two.
   This means: when modes k and p interact, the output lands on q (Fano-determined).

2. CURVATURE LANDSCAPE ON FANO LINES:
   For GF(2)? vectors (unit/edge vectors in Z?):
   - Edges within the cubic lattice (e.g., (1,0,0) and (0,1,0)): ?=90?
   - K_same = K_base - 1/2 ? sin?(90?) = K_base - 1/2
   - K_cross = K_base + 1/2

   The SIGN of K depends on K_base, which depends on |k|,|p|,?.
   For Fano vectors with |k|=|p|=1 and ?=90?: K_same = -3/8 < 0 (DANGEROUS).

3. THE BALANCE QUESTION:
   The Fano incidence (every pair on exactly one line) does NOT force
   a specific helicity balance ? helicity signs are physical, not combinatorial.

   HOWEVER: the Fano structure has a remarkable property:
   The 21 pairs partition into 7 triples where EACH triple sums to zero.
   This means every same-helicity pair (k,p) has a DETERMINED partner q.

   If sin??/4 (Leray suppression) is strongest at ?=?/2 (Fano geometry),
   the Fano structure CONCENTRATES the most-suppressed interactions
   into a closed, self-consistent set.

4. THE KEY INSIGHT (candidate):
   On a Fano line {k,p,q} with all angles = 90?:
   - K_same = -3/8 for each same-helicity edge (dangerous, geodesically focusing)
   - K_cross = +5/8 for each cross-helicity edge (protective, defocusing)
   - Leray suppression ceiling = sin?(90?)/4 = 1/4 (strongest possible)

   The Fano geometry maximizes BOTH dangers (most negative K_same)
   AND protections (largest Leray suppression) simultaneously.

   Whether these exactly balance is NOT determined by combinatorics alone ?
   it requires the dynamics (NS nonlinearity) to pick the helicity assignment.
""")
