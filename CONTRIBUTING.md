# Contributing to Fluid-Resonance

## Push Policy (effective 2026-03-12)

### Who can push directly to `main`?
**Nobody.** All changes must go through pull requests.

Once branch protection is enabled on GitHub:
- All commits require a PR with at least 1 approving review
- Brendan (Architect-Resonance owner) reviews and merges

### For Gemini / Antigravity
Gemini does **NOT** have permission to push directly to the GitHub repository.

**Workflow:**
1. Write code/docs locally or propose changes on the CLAUDE_BRIDGE
2. Brendan or Meridian (Claude) will review for scientific accuracy
3. Only scientifically verified content gets committed and pushed
4. All claims must be clearly labeled: PROVEN, VERIFIED (numerically), CONJECTURE, or OPEN

**What will be rejected:**
- Claims that go beyond what the math supports (e.g., "BYPASSED", "IMPOSSIBLE" for unproven results)
- Metaphorical/motivational documents (manifestos, protocols, guides)
- "Audits" of problems we haven't worked on (BSD, RH, Yang-Mills, etc.)
- Files that reference non-existent files
- Data that cannot be reproduced by running the referenced script

### For Meridian (Claude)
- Can commit and push via the clean repo at `/tmp/fluid-resonance-clean/`
- Must verify all claims before pushing (run scripts, check outputs)
- Negative results are welcome — honesty over optimism

### Scientific standards for all contributors
1. Every claim needs a status label: PROVEN / VERIFIED / CONJECTURE / OPEN / REFUTED
2. Every numerical result needs a script that reproduces it
3. The gap between discrete graph results and PDE results must be stated explicitly
4. No file should reference files that don't exist in the repo

## Repository Structure

```
├── *.py                    # Core proof/verification scripts (root level)
├── spectral_invariants_flows.tex   # LaTeX manuscript
├── FORMAL_PROOFS.md        # Theorems with proofs and status
├── README.md               # Overview with status table
├── CONTRIBUTING.md          # This file
├── docs/                   # Supporting documentation + data
│   ├── RESONANCE_STATE.json    # Single source of truth for results
│   └── *.md / *.txt            # Verification reports, results archives
└── scripts/
    ├── supporting/         # Verified path scripts (Paths 1-3)
    └── wip/                # Work in progress (experiments, solvers)
```
