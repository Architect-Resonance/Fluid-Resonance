# Beltrami Flows in Migdal's Loop Space

**Status: WORKING DRAFT — analytical calculation in progress**
**Date: 2026-03-12**
**Authors: The Wanderer (Claude), with Meridian audit pending**

## 0. Summary of the Key Insight

The original hypothesis (Meridian's Q1) was: "Beltrami flows are eigenstates of the loop Hamiltonian."

**This is wrong in the naive sense.** The Wilson loop of a Beltrami flow is NOT an eigenfunction of L_C in the standard linear sense (the eigenvalue would depend on the loop C).

**What IS true — and more interesting:**

Beltrami flows define the **kernel of the nonlinear loop operator**. The Lamb vector omega x u vanishes identically for Beltrami flows, so they see only the dissipative part of the loop Hamiltonian. The NS dynamics decompose into:

```
L_C = L^diss_C + N_C
```

where N_C is the nonlinear (Lamb vector) operator. Beltrami flows live in ker(N_C). The anti-twist mechanism (Buaria 2024) is the dynamical approach to this kernel at extreme vorticity.

This is NOT an eigenstate problem. It's a **kernel problem**: the nonlinearity has a zero set, and the dynamics are attracted toward it.

---

## 1. Beltrami Flows as Exact NS Solutions (Classical)

**Definition.** A Beltrami flow satisfies nabla x u = lambda u with nabla . u = 0.

**Proposition 1.1.** If u(0) is Beltrami with eigenvalue lambda, then u(t) = u(0) e^{-nu lambda^2 t} is the unique smooth NS solution.

**Proof.**

The NS equation: d_t u + (u . nabla)u = -nabla p + nu nabla^2 u

For Beltrami, the nonlinear term decomposes:
```
(u . nabla)u = omega x u + nabla(|u|^2/2)
```
Since omega x u = lambda u x u = 0:
```
d_t u = -nabla(p + |u|^2/2) + nu nabla^2 u
```

The Laplacian of a Beltrami field:
```
nabla^2 u = nabla(nabla . u) - nabla x (nabla x u) = 0 - lambda(nabla x u) = -lambda^2 u
```

So: d_t u = -nabla P - nu lambda^2 u

where P = p + |u|^2/2. Taking the curl (since nabla x nabla P = 0):
```
d_t omega = -nu lambda^2 omega
```

Since omega = lambda u: d_t u = -nu lambda^2 u.

**Solution:** u(t) = u(0) exp(-nu lambda^2 t). QED.

**Note:** This is well-known (Dombre et al. 1986, Enciso & Peralta-Salas 2015). The key point for us is what it implies in loop space.

---

## 2. Beltrami Flows in Loop Space

### 2.1 The Wilson Loop

For a velocity field u, the Wilson loop (Migdal 2024, Eq. 8) is:
```
Psi_u[C] = exp(i gamma Gamma[C] / nu)
```
where Gamma[C] = oint_C u . dl is the velocity circulation around closed loop C.

For a Beltrami flow, by Stokes' theorem:
```
Gamma[C] = oint_C u . dl = int_S (nabla x u) . dA = lambda int_S u . dA
```

So the circulation equals lambda times the velocity flux through any surface S bounded by C.

### 2.2 Time Evolution of the Wilson Loop

Since u(t) = u(0) e^{-nu lambda^2 t}:
```
Gamma[C, t] = e^{-nu lambda^2 t} Gamma[C, 0] = e^{-nu lambda^2 t} Gamma_0[C]
```

The Wilson loop evolves as:
```
Psi[C, t] = exp(i gamma Gamma_0[C] e^{-nu lambda^2 t} / nu)
```

This is **separable** (in the logarithmic sense): ln Psi = f[C] . g(t), where:
- f[C] = i gamma Gamma_0[C] / nu (spatial: depends on loop shape and initial field)
- g(t) = e^{-nu lambda^2 t} (temporal: universal for all Beltrami flows with same lambda)

**The circulation decays uniformly.** ALL loops C decay at the same rate nu lambda^2. No loop can escape. This is the loop-space expression of Beltrami regularity.

### 2.3 Verification Against the Loop Equation

The loop equation (Migdal 2024-2025):
```
d_t Psi[C, t] = nu L_C Psi[C, t]
```

Left side:
```
d_t Psi = (i gamma / nu) (-nu lambda^2) Gamma_0 e^{-nu lambda^2 t} . Psi
        = -i gamma lambda^2 Gamma[C,t] . Psi
```

So the loop equation requires:
```
nu L_C Psi = -i gamma lambda^2 Gamma[C,t] . Psi
```

This tells us: L_C Psi_Beltrami = -(i gamma lambda^2 / nu) Gamma[C,t] . Psi_Beltrami

**NOT an eigenvalue equation** (the coefficient depends on C through Gamma[C,t]).

### 2.4 Why It's Not an Eigenstate (and Why That's OK)

In quantum mechanics, eigenstates of H satisfy H|n> = E_n|n> with E_n constant.

Here, L_C Psi = f(C) . Psi where f(C) = -(i gamma lambda^2 / nu) Gamma[C,t]. Since f depends on the loop C, this is NOT a standard eigenvalue equation.

**Physical reason:** The Wilson loop Psi = exp(i gamma Gamma/nu) is an exponential of a LINEAR functional (the circulation). Eigenstates of a differential operator are typically NOT exponentials of linear functionals.

**What Beltrami flows ARE in loop space:** They are **coherent states** — states with a definite phase (circulation) that evolves by simple exponential decay. In quantum optics, a coherent state |alpha> satisfies a|alpha> = alpha|alpha> (eigenstate of the annihilation operator, not the Hamiltonian). Similarly, Beltrami flows are eigenstates of the **circulation operator**, not the loop Hamiltonian.

---

## 3. The Beltrami Kernel of the Nonlinear Loop Operator

### 3.1 Decomposition of the Loop Hamiltonian

From Migdal 2024 (Eq. 46), the loop operator at each vertex:
```
L-hat = -nu nabla-hat x omega-hat  +  omega-hat x v-hat
         (dissipative term)           (Lamb vector term)
```

Define:
- **Dissipative operator:** L^diss = -nu nabla-hat x omega-hat
- **Nonlinear operator:** N-hat = omega-hat x v-hat (the Lamb vector in loop space)

The full loop Laplacian decomposes:
```
L_C = L^diss_C + N_C
```

### 3.2 The Lamb Vector Vanishes for Beltrami

For a Beltrami flow, omega x u = lambda u x u = 0 pointwise.

In loop space, this means the nonlinear operator N-hat acting on Psi_Beltrami produces terms proportional to the Lamb vector, which is identically zero.

**Therefore:** N_C Psi_Beltrami = 0

Beltrami flows are in the **kernel** of the nonlinear loop operator.

### 3.3 Evolution on the Kernel

On ker(N_C), the loop equation reduces to:
```
d_t Psi = nu L^diss_C Psi
```

The evolution is PURELY DISSIPATIVE — no stretching, no nonlinear cascade, no energy transfer between scales. The only dynamics is exponential decay at rate nu lambda^2.

**This is the strongest possible regularity statement in loop space:**

If a flow is Beltrami, its Wilson loop evolves under a purely dissipative operator. No blow-up is possible because there's no mechanism to amplify circulation — only to damp it.

### 3.4 The Kernel Structure

The kernel ker(N_C) is infinite-dimensional. All Beltrami flows with ANY eigenvalue lambda are in it:
- ABC flows (lambda = 1)
- Helical pipe flows (various lambda)
- Arbitrary superpositions within a single helicity sector

**Key constraint:** Different lambda values give different decay rates. A superposition of Beltrami modes with lambda_1, lambda_2 decays at DIFFERENT rates, causing the flow to evolve within ker(N_C) toward the smallest |lambda| mode (slowest decay).

---

## 4. Helicity as a Loop-Space Invariant

### 4.1 Helicity-Energy Relation for Beltrami

For Beltrami with nabla x u = lambda u:
```
H = int u . omega d^3x = lambda int |u|^2 d^3x = lambda . 2E
```

So H/E = 2 lambda. The ratio H/E determines the Beltrami eigenvalue, which determines the decay rate.

### 4.2 Helicity Conservation in BT Surgery

In single-helicity NS (BT surgery), helicity is exactly conserved inviscidly (Waleffe 1992, Biferale-Titi 2013).

With viscosity: dH/dt = -2nu int omega . (nabla x omega) d^3x = -2nu lambda int |omega|^2 = -2nu lambda Z

And dE/dt = -2nu Z (standard energy dissipation).

So: d(H/E)/dt = d(2lambda)/dt needs to be checked...

Actually, for a Beltrami flow: H = lambda . 2E, so dH/dt = lambda . 2 dE/dt = lambda . 2(-2nu Z) = -4nu lambda Z
And separately: dH/dt = -2nu int omega . (nabla x omega) = -2nu lambda int |omega|^2 = -2nu lambda . 2Z = -4nu lambda Z. Check!

So H/E = 2lambda is exactly conserved for Beltrami flows. The helicity-to-energy ratio is a constant of motion.

### 4.3 Helicity in Loop Space (TO BE DEVELOPED)

The helicity H = int u . omega is a quadratic functional of u. In loop space, it should correspond to a second-order operator acting on Psi.

**Conjecture:** There exists a loop-space operator H-hat such that:
- H-hat commutes with L^diss_C (the dissipative loop Laplacian) on the Beltrami sector
- H-hat Psi_Beltrami = lambda . (something) . Psi_Beltrami
- This commutation is WHY helicity is conserved in BT surgery

**Status:** This conjecture is OPEN. Computing H-hat explicitly in Migdal's framework requires translating int u . omega into a functional of Wilson loops.

One approach: Note that oint_C u . dl = Gamma[C], and omega . dA = lambda u . dA. So the helicity density u . omega evaluated at a point x can be recovered from the Wilson loop by shrinking C to a point around x. But this requires a limit and may be singular.

---

## 5. The Anti-Twist Mechanism as Approach to ker(N_C)

### 5.1 Buaria 2024 in Loop-Space Language

Buaria, Lawson & Wilczek (2024, Science Advances) showed:
1. At extreme vorticity, a spontaneous negative anti-twist emerges in vortex cores
2. This anti-twist reverses vortex stretching
3. The mechanism is INVISCID — works for Euler (nu=0)
4. It forces the flow toward u || omega (Beltrami alignment)

**Translation to loop space:**

The anti-twist mechanism reduces the Lamb vector omega x u locally at extreme vorticity. In loop space, this means:
```
N_C Psi -> 0  at extreme vorticity regions
```

The flow dynamically approaches ker(N_C). The extreme regions "want to become Beltrami."

### 5.2 Why This Is Inviscid

The anti-twist works at nu = 0 (Euler). In Migdal's framework, nu = hbar. So the anti-twist is a CLASSICAL (nu -> 0) phenomenon.

In loop-space language: the approach to ker(N_C) is driven by the nonlinear operator N_C ITSELF. The Lamb vector is not killed by dissipation — it self-destructs through its own dynamics. This is a property of the classical (Euler) Hamiltonian, not the quantum (viscous) correction.

### 5.3 Connection to Q2 (D > S)

Our Q2 test showed: S >> D at extreme vorticity INSTANTANEOUSLY.
Buaria 2024 showed: the anti-twist reverses stretching TEMPORALLY (same vortex tube first amplifies, then attenuates).

These are consistent. At any frozen instant:
- Extreme points have S >> D (they're currently being stretched)
- But those SAME points will later have the anti-twist develop, reversing S
- The volume-averaged D > S holds because the temporal self-attenuation keeps the total stretching budget bounded

In loop space: the Wilson loop at extreme points is being driven away from ker(N_C) by local stretching, but the anti-twist pulls it back. The volume average stays near ker(N_C) because the excursions are self-limiting.

---

## 6. The Three-Thread Convergence

| Thread | Source | Loop-Space Translation |
|--------|--------|----------------------|
| **Migdal** | 2024-2025 | NS = linear loop equation. Loop Hamiltonian = dissipative + Lamb vector |
| **Buaria** | 2020 + 2024 | Anti-twist = dynamical approach to ker(Lamb vector) at extreme vorticity |
| **Our BT surgery** | S36 | Restricts to single-helicity sector, closer to Beltrami, D > S globally |

**The synthesis:**

1. The nonlinear loop operator N_C has a kernel: Beltrami flows (omega x u = 0)
2. The anti-twist mechanism (inviscid) drives extreme vorticity toward ker(N_C)
3. BT surgery restricts dynamics to single-helicity sector, where helicity conservation further constrains the approach to ker(N_C)
4. On ker(N_C), evolution is purely dissipative: no blow-up possible
5. The global D > S (Investigation 4) is the volume-integrated consequence of the dynamics being near ker(N_C) in the BT sector
6. D > S is global, not local (Q2): because the anti-twist is temporal (excursions happen, but self-correct), not instantaneous

**What remains to prove:**
1. ker(N_C) IS an attractor for the NS dynamics (under what conditions?)
2. The approach rate is fast enough to prevent blow-up (regularity)
3. Migdal's No Explosion Theorem extends from stochastic to deterministic ICs
4. The helicity operator H-hat commutes with L^diss_C on the Beltrami sector

---

## 7. Honest Gaps

1. **NOT an eigenstate problem.** The original framing was wrong. Beltrami flows are in ker(N_C), not eigenstates of L_C. This is actually more interesting, but the "paper to write" is different from what Meridian proposed.

2. **Helicity operator in loop space is not computed.** We conjecture it commutes with L^diss, but haven't proved it.

3. **The anti-twist as approach to ker(N_C) is interpretive.** Buaria describes it in physical space; our translation to loop space is a reinterpretation, not a derivation.

4. **Global D > S is numerical, not proven.** Investigation 4 confirmed D > S for all tested ICs, but it's not a theorem.

5. **The continuum limit is open.** Migdal's framework is proven at discrete (finite N) level; the N -> infinity limit is what Brue and De Lellis are working on.

6. **Star polygons != Beltrami.** Migdal's attractor is star polygons in momentum loop space. Beltrami flows in position space may have no direct connection to star polygons.

---

## 8. Numerical Validation Plan

1. **ABC flow test:** Verify u(t) = u(0)e^{-nu lambda^2 t} exactly, compute Wilson loops, verify Lamb vector = 0
2. **Approach to Beltrami:** Start from non-Beltrami IC, measure |omega x u| / (|omega| |u|) over time at extreme vorticity
3. **Temporal anti-twist:** Track individual high-vorticity regions, measure stretching reversal
4. **Wilson loop decay:** Compare decay rates of Wilson loops for Beltrami vs general ICs
