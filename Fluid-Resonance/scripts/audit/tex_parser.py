"""Extract mathematical claims from .tex files using TexSoup.

Each claim is a proposition, theorem, lemma, corollary, conjecture, or
definition environment, with its label, math expressions, and line number.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

CLAIM_ENVS = {
    "proposition", "theorem", "lemma", "corollary", "conjecture", "definition",
}


@dataclass
class Claim:
    label: str              # e.g., "prop:fano-lap"
    env_type: str           # "proposition", "theorem", etc.
    title: str              # Optional title in brackets, e.g., "Fano Laplacian"
    raw_latex: str          # Full LaTeX body (between \begin and \end)
    math_expressions: list  # Extracted display math blocks
    line_number: int        # Line number of \begin{...}
    has_proof: bool = False # Whether a \begin{proof} follows


@dataclass
class Section:
    number: int             # Section index (1-based)
    title: str
    line_number: int
    claims: list = field(default_factory=list)


def extract_claims(tex_path: str, section_filter: int | None = None) -> list[Claim]:
    """Parse a .tex file and return all mathematical claims.

    Args:
        tex_path: Path to the .tex file.
        section_filter: If set, only return claims from this section number.

    Returns:
        List of Claim objects.
    """
    text = Path(tex_path).read_text(encoding="utf-8")
    lines = text.split("\n")

    # Build section map
    sections: list[Section] = []
    section_re = re.compile(r"\\section\{(.+?)\}")
    for i, line in enumerate(lines, 1):
        m = section_re.search(line)
        if m:
            sections.append(Section(
                number=len(sections) + 1,
                title=m.group(1),
                line_number=i,
            ))

    # Build claim list
    claims: list[Claim] = []
    env_re = re.compile(
        r"\\begin\{(" + "|".join(CLAIM_ENVS) + r")\}"
        r"(?:\[([^\]]*)\])?"  # optional title
    )
    label_re = re.compile(r"\\label\{([^}]+)\}")
    display_math_re = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
    proof_re = re.compile(r"\\begin\{proof\}")

    i = 0
    while i < len(lines):
        m = env_re.search(lines[i])
        if m:
            env_type = m.group(1)
            title = m.group(2) or ""
            start_line = i + 1  # 1-based

            # Collect body until \end{env_type}
            end_re = re.compile(r"\\end\{" + env_type + r"\}")
            body_lines = []
            j = i + 1
            while j < len(lines) and not end_re.search(lines[j]):
                body_lines.append(lines[j])
                j += 1
            body = "\n".join(body_lines)

            # Extract label
            lm = label_re.search(body) or label_re.search(lines[i])
            label = lm.group(1) if lm else f"unlabeled:{env_type}:{start_line}"

            # Extract display math
            math_exprs = display_math_re.findall(body)
            math_exprs = [m.strip() for m in math_exprs]

            # Check if proof follows
            has_proof = False
            for k in range(j + 1, min(j + 5, len(lines))):
                if proof_re.search(lines[k]):
                    has_proof = True
                    break

            claims.append(Claim(
                label=label,
                env_type=env_type,
                title=title,
                raw_latex=body,
                math_expressions=math_exprs,
                line_number=start_line,
                has_proof=has_proof,
            ))
            i = j + 1
        else:
            i += 1

    # Assign claims to sections
    for claim in claims:
        for s_idx, sec in enumerate(sections):
            next_line = sections[s_idx + 1].line_number if s_idx + 1 < len(sections) else float("inf")
            if sec.line_number <= claim.line_number < next_line:
                sec.claims.append(claim)
                break

    # Filter by section if requested
    if section_filter is not None:
        filtered = []
        for sec in sections:
            if sec.number == section_filter:
                filtered = sec.claims
                break
        return filtered

    return claims


if __name__ == "__main__":
    import sys
    tex_file = sys.argv[1] if len(sys.argv) > 1 else "../../the_shape.tex"
    section = int(sys.argv[2]) if len(sys.argv) > 2 else None

    claims = extract_claims(tex_file, section_filter=section)
    print(f"Found {len(claims)} claims" + (f" in section {section}" if section else ""))
    for c in claims:
        proof_tag = " [has proof]" if c.has_proof else ""
        math_tag = f" [{len(c.math_expressions)} equations]" if c.math_expressions else ""
        print(f"  L{c.line_number}: {c.env_type} [{c.title}] ({c.label}){math_tag}{proof_tag}")
        for expr in c.math_expressions:
            print(f"    \\[ {expr[:80]}{'...' if len(expr) > 80 else ''} \\]")
