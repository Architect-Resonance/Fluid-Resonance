# Negative Results

Approaches investigated and ruled out during this research. Each entry states the conjecture, the method of refutation, and what (if anything) survives.

---

## 1. Spectral invariant R as SAT predictor

**Conjecture**: The star-cluster invariant R = 1.8573... predicts satisfiability of the underlying clause structure.

**Refutation**: Systematic comparison across graph families shows no correlation between R and satisfiability. R is a spectral quantity of the grounded Laplacian; satisfiability is a combinatorial property of the clause set. No mapping between the two exists.

---

## 2. Helicity ratio / solenoidal fraction power law

**Conjecture**: The solenoidal fraction of the Lamb vector scales as a power law s ~ r^alpha with the helicity imbalance ratio r.

**Refutation**: DNS measurements (Taylor-Green, Re=400, N=64) show no correlation. The solenoidal fraction and helicity ratio are governed by different angular structures in Fourier space.

---

## 3. Local self-regulation of enstrophy (D > S)

**Conjecture**: Dissipation locally exceeds the stretching source term at extreme vorticity, providing a self-regulating mechanism.

**Refutation**: DNS shows the inequality D > S reverses at the highest vorticity percentiles. The tail of the vorticity distribution is precisely where self-regulation fails.

---

## 4. Cross-helicity boundary layer regularity

**Conjecture**: Cross-helicity structure near boundaries provides additional regularity.

**Refutation**: Depends on the local self-regulation conjecture (#3), which failed.

---

## 5. Dissipative anomaly testability

**Conjecture**: The dissipative anomaly (finite dissipation as viscosity tends to zero) can be tested at computationally accessible Reynolds numbers.

**Refutation**: Requires Re >> 10^4 to observe the anomaly cleanly. Current DNS resolution (N <= 96) is insufficient.

---

## 6. Lyapunov function from solenoidal Lamb fraction

**Conjecture**: F(t) = ||P_sol[omega x u]||^2 / ||omega x u||^2 is monotonically decreasing, providing a Lyapunov function for NS dynamics.

**Refutation**: Direct computation shows F(t) is not monotone. It fluctuates with the flow evolution and cannot serve as a Lyapunov function.

---

## 7. Star polygon = star cluster graph

**Conjecture**: The spectral invariant R arises from a star polygon geometry.

**Refutation**: The graph is a star *cluster* (hub-and-spoke with clause hyperedges), not a star polygon. The two have different spectra.

---

## 8. Conjecture 9.1 with anchor vertices

**Conjecture**: R < 2 holds universally for all grounded star-cluster graphs.

**Refutation**: Counterexample: K_4 with 8 anchor vertices at grounding weight w = 4 gives R = 2.014 > 2. The bound is not universal; it depends on the grounding structure.

---

## 9. Triadic shell Bessel bound

**Conjecture**: Bessel-function bounds on individual triadic shells provide tighter estimates than global Cauchy-Schwarz.

**Refutation**: The per-shell Bessel bounds are strictly worse than global Cauchy-Schwarz for the enstrophy inequality. The shell decomposition introduces boundary terms that offset any gain.

---

## 10. Leray suppression "verified shield" overclaim

**Conjecture**: The Leray suppression factor alpha = 1 - ln(2) alone suffices to prevent blow-up.

**Refutation**: alpha < 1 reduces the constant in the enstrophy inequality by a factor of ~1/256, but does not change the critical Sobolev exponent 3/2 (Lu & Doering, arXiv:1909.00041, proved sharp). Suppression improves constants, not scaling.

---

## 11. Same-helical solenoidal fraction vanishes

**Conjecture**: The solenoidal projection of same-helical (co-chiral) Lamb vectors is identically zero.

**Refutation**: DNS measurement shows same-helical P_sol ranges from 31% to 53%, depending on flow geometry. The strong form is false; only the cross-helical dominance (Sahoo & Biferale, 2017) holds.

---

## 12. Negative-helicity energy bound via total helicity

**Conjecture**: E_- <= H / lambda_1, bounding the negative-helicity energy by helicity and the first eigenvalue.

**Refutation**: Confuses L^2 and H^1 norms. The actual E_- measured in DNS was 66x the predicted bound. The inequality is dimensionally incorrect.

---

## 13. sin^2(theta)/4 as braid crossing number

**Conjecture**: The Leray coupling factor sin^2(theta)/4 equals the crossing number in a topological entanglement braid (TEB) framework.

**Refutation**: The TEB framework is unpublished (Academia.edu preprint only) and assumes C_max is bounded, which is equivalent to the regularity problem itself. The numerical coincidence has no content.

---

## 14. Effective dimension reduction via Leray suppression

**Conjecture**: Leray suppression reduces the effective spatial dimension, changing the critical Sobolev exponent.

**Refutation**: alpha reduces the constant (by a factor up to 1/256) but not the exponent. At blow-up scales, the effective dimension returns to d = 3. The critical exponent 3/2 is sharp (Lu & Doering).

---

## 15. Direct proof that C_max is bounded

**Conjecture**: The maximum vorticity alignment coefficient C_max can be shown to remain finite.

**Refutation**: Proving C_max bounded is equivalent to the Navier-Stokes regularity problem. Chen & Hou (2025) showed that for 3D Euler, the analogous ratio diverges. No shortcut exists.

---

## 16. Black hole entropy 1/4 = Leray 1/4

**Conjecture**: The Bekenstein-Hawking entropy coefficient A/(4 l_P^2) is related to the Leray coupling factor sin^2(theta)/4.

**Refutation**: The 1/4 in BH entropy arises from the Hawking temperature (surface gravity / 2pi) integrated against the first law. The 1/4 in sin^2(theta)/4 arises from angular averaging of the solenoidal projection. Different origins, numerical coincidence.

---

## 17. Kerr frame-dragging sin^2(theta) = Leray sin^2(theta) (narrowed)

**Conjecture**: The angular dependence sin^2(theta) in Kerr metric frame-dragging is formally identical to the Leray suppression factor.

**Refutation (partial)**: Five specific ratio tests (energy ratios, angular momentum coupling, horizon area fractions) all negative. The membrane paradigm route fails because 2D NS on the horizon has no helicity (u . omega = 0 in 2D). The Rossby angle is a structural analogue only. A real connection, if any, must go through the BKMS duality (3D NS from 5D bulk Einstein; Bredberg et al., 2012), not the Damour membrane (2D NS from 4D bulk).

---

## 18. Hawking radiation as turbulent backscatter

**Conjecture**: Hawking radiation from black holes maps to energy backscatter in turbulent cascades.

**Refutation**: Weak analogy with no formal mapping. Hawking radiation is a quantum effect (pair production at the horizon); backscatter is a classical nonlinear phenomenon (reverse energy transfer). No equations connect them.

---

## 19. Membrane paradigm route to cosmic censorship

**Conjecture**: The Damour membrane paradigm (BH horizon as 2D viscous fluid) maps NS regularity to cosmic censorship.

**Refutation**: 2D Navier-Stokes on the horizon has no helicity: u . omega = 0 identically in two dimensions. Since helicity is central to the regularity mechanism (cross-helical cascade, Leray suppression), the 2D membrane dual cannot capture it. The connection must go through BKMS (3D NS from 5D Einstein), not Damour (2D NS from 4D Einstein).

---

## 20. Helicity-dependent splitting of Arnold sectional curvature

**Conjecture**: Arnold's sectional curvature on SDiff(T^3) splits between same-helical and cross-helical mode pairs, with same-helical curvature less negative (more stable).

**Refutation**: First-principles computation using the Koszul formula for the Levi-Civita connection on SDiff(T^3) shows K_same = K_cross to machine precision across all tested wavevector pairs. The original error was using the advection operator (Lie derivative) instead of the Levi-Civita connection. Araki (2016, arXiv:1608.05154) independently confirms: K_E = pi^2 |k|^2 R_E(rho, eta), always negative, helicity-independent. Retracted claims: curvature splitting theorem, sign polarization, two-channel stability, Leray-curvature identity.

---

## 21. Spectral invariant R as Berry holonomy ratio

**Conjecture**: R = lambda_min(8x8) / lambda_min(6x6) equals the Berry holonomy of the helical basis integrated over Fano-plane triads, divided by 4pi. Specifically, R = 15/8 - 1/56 where 15/8 is a Z_2-quantized value and 1/56 is a frustration correction (56 = C(8,3) = number of frustrated Hamming configurations).

**Refutation**: Berry holonomy for any single NS triad is trivially zero. Proof: for k + p + q = 0, the wavevectors k-hat, p-hat, q-hat are always coplanar since k . (p x q) = k . (p x (-(k+p))) = -k . (p x k) = 0. Coplanar vectors trace a great circle on S^2, which subtends zero solid angle. The Berry connection on h_pm(k-hat) is real (Chern number 2, spin-1 monopole on S^2), but it produces zero holonomy for any great-circle loop. The 15/8 value observed in Taylor-Green simulations was an artifact of the flow's discrete symmetries, not a fundamental quantity.

---

## 22. Single-triad Berry frustration as blow-up obstruction

**Conjecture**: Berry phase frustration along Fano-plane lines prevents coherent phase alignment needed for blow-up. The [7,4,3] Hamming code structure of sign frustration (minimum distance d = 3) provides a quantitative obstruction.

**Refutation**: DNS with random initial conditions (breaking all discrete symmetries) shows triad phases broadly distributed (standard deviation 1.5-2.0 radians, nearly uniform on [-pi, pi]). Magnitude-weighted circular coherence R_w = 0.02-0.50 (weak). The holonomy ratio fluctuates from -11.4 to +10.7 across different initial conditions and times — it is a dynamical quantity, not a topological invariant. Combined with #21 (zero holonomy for individual triads), single-triad Berry frustration provides no quantitative bound on enstrophy growth.

**What survives**: The [7,4,3] Hamming code structure of Fano sign frustration is a topological fact independent of DNS. The Fano plane PG(2,2) governs triadic topology at low wavenumbers. Multi-triad Berry sequences (chains that are not individually momentum-conserving loops) remain unexplored.
