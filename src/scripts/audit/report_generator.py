"""Generate audit reports in markdown and JSON format."""

import json
from datetime import datetime
from pathlib import Path

from claim_verifier import Status, VerificationResult
from tex_parser import Claim


def generate_markdown(
    claims: list[Claim],
    results: list[VerificationResult],
    tex_file: str,
    section: int | None = None,
) -> str:
    """Generate a markdown audit report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    scope = f"Section {section}" if section else "Full paper"

    lines.append(f"# Math Audit Report — {Path(tex_file).name}")
    lines.append(f"**Generated**: {now}  ")
    lines.append(f"**Scope**: {scope}  ")
    lines.append("")

    # Summary counts
    counts = {}
    for r in results:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    summary_parts = [f"{v} {k.lower()}" for k, v in sorted(counts.items())]
    lines.append(f"**Summary**: {', '.join(summary_parts)}")
    lines.append("")

    # Table
    lines.append("| # | Claim | Label | Status | Method | Notes |")
    lines.append("|---|-------|-------|--------|--------|-------|")

    for i, (claim, result) in enumerate(zip(claims, results), 1):
        status_icon = {
            Status.CONFIRMED: "CONFIRMED",
            Status.REFUTED: "**REFUTED**",
            Status.INCONCLUSIVE: "INCONCLUSIVE",
            Status.SKIPPED: "skipped",
        }[result.status]

        title = claim.title or claim.env_type
        label = f"`{claim.label}`" if not claim.label.startswith("unlabeled") else "—"
        notes = result.notes[:80] + ("..." if len(result.notes) > 80 else "")

        lines.append(
            f"| {i} | {title} | {label} | {status_icon} | {result.method} | {notes} |"
        )

    lines.append("")

    # Details for non-SKIPPED, non-CONFIRMED
    issues = [(c, r) for c, r in zip(claims, results)
              if r.status in (Status.REFUTED, Status.INCONCLUSIVE)]
    if issues:
        lines.append("## Issues Found")
        lines.append("")
        for claim, result in issues:
            lines.append(f"### {claim.title or claim.label} (L{claim.line_number})")
            lines.append(f"**Status**: {result.status.value}  ")
            lines.append(f"**Method**: {result.method}  ")
            lines.append(f"**Notes**: {result.notes}")
            if result.details:
                lines.append(f"```json\n{json.dumps(result.details, indent=2, default=str)}\n```")
            lines.append("")

    return "\n".join(lines)


def generate_json(
    claims: list[Claim],
    results: list[VerificationResult],
    tex_file: str,
    section: int | None = None,
) -> str:
    """Generate a JSON audit report."""
    records = []
    for claim, result in zip(claims, results):
        records.append({
            "label": claim.label,
            "env_type": claim.env_type,
            "title": claim.title,
            "line_number": claim.line_number,
            "has_proof": claim.has_proof,
            "status": result.status.value,
            "method": result.method,
            "notes": result.notes,
            "details": result.details,
        })

    report = {
        "file": tex_file,
        "section": section,
        "timestamp": datetime.now().isoformat(),
        "summary": {s.value: sum(1 for r in results if r.status == s) for s in Status},
        "claims": records,
    }
    return json.dumps(report, indent=2, default=str)
