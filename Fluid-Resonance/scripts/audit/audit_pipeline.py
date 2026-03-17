#!/usr/bin/env python
"""Paper audit pipeline — extract claims from .tex and verify each one.

Usage:
    python audit_pipeline.py <tex_file> [--section N] [--output report.md] [--json]

Examples:
    python audit_pipeline.py ../../the_shape.tex --section 5
    python audit_pipeline.py ../../the_shape.tex --output audit_report.md --json
"""

import argparse
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode math symbols
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tex_parser import extract_claims
from claim_verifier import verify_claim
from report_generator import generate_markdown, generate_json


def main():
    parser = argparse.ArgumentParser(description="Audit mathematical claims in a .tex file")
    parser.add_argument("tex_file", help="Path to the .tex file")
    parser.add_argument("--section", type=int, default=None, help="Only audit this section number")
    parser.add_argument("--output", "-o", default=None, help="Write markdown report to file")
    parser.add_argument("--json", action="store_true", help="Also output JSON report")
    args = parser.parse_args()

    tex_path = Path(args.tex_file)
    if not tex_path.exists():
        print(f"Error: {tex_path} not found", file=sys.stderr)
        sys.exit(1)

    # Step 1: Extract claims
    claims = extract_claims(str(tex_path), section_filter=args.section)
    scope = f"section {args.section}" if args.section else "full paper"
    print(f"Extracted {len(claims)} claims from {tex_path.name} ({scope})")

    # Step 2: Verify each claim
    results = []
    for claim in claims:
        print(f"  Verifying: {claim.title or claim.label}...", end=" ", flush=True)
        result = verify_claim(claim)
        results.append(result)
        print(result.status.value)

    # Step 3: Generate reports
    md_report = generate_markdown(claims, results, str(tex_path), args.section)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(md_report, encoding="utf-8")
        print(f"\nMarkdown report written to {out_path}")

        if args.json:
            json_path = out_path.with_suffix(".json")
            json_report = generate_json(claims, results, str(tex_path), args.section)
            json_path.write_text(json_report, encoding="utf-8")
            print(f"JSON report written to {json_path}")
    else:
        print()
        print(md_report)

    # Summary
    from claim_verifier import Status
    confirmed = sum(1 for r in results if r.status == Status.CONFIRMED)
    refuted = sum(1 for r in results if r.status == Status.REFUTED)
    skipped = sum(1 for r in results if r.status == Status.SKIPPED)

    if refuted:
        print(f"\n*** {refuted} REFUTED claims — review required ***")
        sys.exit(1)

    print(f"\n{confirmed} confirmed, {skipped} skipped (no verifier), 0 refuted")


if __name__ == "__main__":
    main()
