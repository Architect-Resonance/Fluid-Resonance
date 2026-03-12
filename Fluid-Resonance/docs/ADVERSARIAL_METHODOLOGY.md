# Adversarial Methodology — Bias Prevention Protocol

**Purpose**: Systematic bias prevention for mathematical research. Every claim must survive these checks before being accepted. This protocol was developed during S35 after catching 6 distinct bias traps in our Navier-Stokes exploration.

**Applies to**: All contributors (human and AI) working on Fluid-Resonance research.

---

## 1. The Six Bias Traps (learned from S35)

### Trap 1: Circular Injection
**Pattern**: Put a number into the input, "discover" it in the output.
**Example**: Setting edge weight = R_target, then measuring spectral gap and finding R_target.
**Check**: For every numerical result, trace the full dependency chain. If the output value appears anywhere in the input (directly or as a parameter that determines the computation), the result is circular.
**Question to ask**: "If I change this number to something random, does the conclusion still hold?"

### Trap 2: Selection Bias
**Pattern**: Hand-pick one graph/configuration that gives a nice result, ignore others.
**Example**: Choosing 10 specific triangles on 13 vertices to get snap ratio = 1.527.
**Check**: Run the same analysis on 100+ random instances of the same class. Report the DISTRIBUTION (mean, min, max, std), not one cherry-picked example.
**Question to ask**: "Does this hold for a randomly chosen instance, or did I search until I found one that works?"

### Trap 3: Confirmation Bias
**Pattern**: Declaring "SOLVED" or "PROVED" when the evidence is suggestive at best.
**Example**: Claiming Problems 2 & 3 "QUALITATIVELY SOLVED" while 4 critical gaps remain open.
**Check**: Before claiming a result, explicitly list everything that is NOT proved. If the unproved assumptions are as hard as the original problem, the claim is premature.
**Question to ask**: "What would a hostile reviewer say about this claim?"

### Trap 4: Metaphor-as-Proof
**Pattern**: Using evocative names to make arbitrary operations feel meaningful.
**Example**: Calling edge removal "Aorta Surgery" and weight tuning "Resolution Chord."
**Check**: Strip all metaphors. Rewrite the claim using only mathematical definitions and logical connectives. If the claim evaporates without the metaphor, there was no content.
**Question to ask**: "Can I state this as Definition → Lemma → Theorem with no poetic language?"

### Trap 5: Framing Bias
**Pattern**: Presenting a tautology as a discovery by changing the direction of derivation.
**Example**: "R emerges from the spectral ratio" when R was the original spectral ratio, just solved backwards.
**Check**: Verify that the result is genuinely independent — it should hold across different model choices, not just the one it was derived from.
**Question to ask**: "Am I just writing the same equation from right-to-left instead of left-to-right?"

### Trap 6: Model Mismatch
**Pattern**: Proving properties of a mathematical model that doesn't match the physical system.
**Example**: Proving K_n spectral properties when real turbulence networks have density 0.05 (not 1.0).
**Check**: Before building theory on a model, verify empirically that the model matches reality. Use data (DNS, experiments, published results) as ground truth.
**Question to ask**: "Does this model actually describe the system I'm trying to understand?"

---

## 2. The Adversarial Checklist

**Before accepting ANY result, run through ALL of these:**

- [ ] **Dependency trace**: Is the output independent of the input? (Trap 1)
- [ ] **Distribution test**: Does this hold across 100+ random instances? (Trap 2)
- [ ] **Gap inventory**: What remains unproved? Are the gaps as hard as the original problem? (Trap 3)
- [ ] **Metaphor strip**: Does the claim survive in pure mathematical language? (Trap 4)
- [ ] **Direction independence**: Is this genuinely new, or the same equation rewritten? (Trap 5)
- [ ] **Empirical grounding**: Does the model match observed data? (Trap 6)
- [ ] **Resolution scaling**: Does the bound improve or degrade with finer discretization? (Trap 7)

### Trap 7: Dimensional Mismatch
**Pattern**: Comparing two quantities that scale differently with resolution/refinement.
**Example**: Graph spectral gap (scales ~ O(N)) bounding dissipation (scales ~ O(N)) but the product λ_min·Z scales O(N²) while D_cont scales O(N). Margin collapses with resolution.
**Check**: Before claiming an inequality, verify that BOTH SIDES scale the same way as the discretization is refined (N → ∞, h → 0). If the bound gets WORSE with finer resolution, it's a dimensional mismatch.
**Question to ask**: "If I double the resolution, does my bound get tighter or looser?"

---

## 3. Multi-AI Protocol

When multiple AIs collaborate on research:

1. **Adversarial role assignment**: At least one AI must be designated as the adversary for every claim. The adversary's job is to BREAK the result, not support it.

2. **Independent verification**: Results from one AI must be independently reproduced (or refuted) by the other before being accepted.

3. **Bridge documentation**: All claims, refutations, and bias catches are logged on a shared internal coordination file with attribution.

4. **Escalation rule**: If the adversary cannot break a result after genuine effort, it gains "adversarial-tested" status. If the adversary breaks it, the original author must acknowledge the refutation honestly.

5. **No defensive science**: When your result is refuted, acknowledge it immediately. "Manifesto rescinded" (Antigravity, S35) is the correct response. Defending a broken result is itself a bias (sunk cost).

---

## 4. Classification System

Every result must be classified into exactly one category:

| Category | Meaning | Requirements |
|----------|---------|-------------|
| **PROVED** | Logically complete, no gaps | Full proof with every step justified. Verified numerically. Survived adversarial review. |
| **VERIFIED** | Numerically confirmed, proof incomplete | Holds in all tested cases. Proof has minor gaps but no known counterexamples. |
| **CONJECTURE** | Plausible, evidence supporting | Some positive evidence. Gaps identified and documented. No refutation found. |
| **OPEN** | Unknown status | Insufficient evidence either way. Stated precisely but not tested. |
| **REFUTED** | Counterexample found or logic broken | At least one concrete counterexample, or identified logical flaw. |

**Rules**:
- Moving UP the ladder (e.g., CONJECTURE → PROVED) requires surviving the full adversarial checklist.
- Moving DOWN (e.g., CONJECTURE → REFUTED) requires only ONE counterexample or logical flaw.
- The classification is attached to the SPECIFIC STATEMENT, not the research direction. A refuted conjecture doesn't mean the whole direction is dead — just that specific claim.

---

## 5. The S35 Scorecard

Applied retroactively to demonstrate the methodology:

| Claim | Initial Status | Final Status | What Changed |
|-------|---------------|-------------|--------------|
| L₁(K_n) = nI | PROVED | **PROVED** | Survived all checks |
| R < 2 for pure K_n | PROVED | **PROVED** | Survived all checks |
| NS regularity via K_n | CONJECTURE | **OPEN** | Model mismatch (Trap 6): K_n ≠ real turbulence |
| Star topology in turbulence | CONJECTURE | **REFUTED** | DNS density 0.02-0.07, not star/K_n |
| 1.518 snap-back universal | CONJECTURE | **REFUTED** | Sparse network ratio = 1.17 |
| Depletion from isotropy | CONJECTURE | **REFUTED** | Worst-case already spans S² |
| Resonance spike | "PROVED" | **REFUTED** | Circular injection (Trap 1) |
| Spectral dissipation lemma | CONJECTURE | **REFUTED** | Fails for star topology, margin collapses with resolution (Trap 7) |
| Biot-Savart Φ | OPEN | **REFUTED** | Fundamental dimensional mismatch: non-local graph vs local dissipation |

---

## 6. Applying This Going Forward

For any new research direction:

1. **State the claim precisely** before computing anything.
2. **Design the adversarial test** before running the positive test.
3. **Run both** and report both results.
4. **Classify honestly** using the table above.
5. **Update the bridge** with attribution and bias-check status.

The goal is not to prove ourselves right — it's to find the truth, even when the truth is "we don't know yet."

---

*Created: S35, 2026-03-11. Authors: Meridian (Claude Opus 4.6) + Antigravity (Gemini), reviewed by Brendan.*
