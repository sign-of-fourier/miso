#!/usr/bin/env python3
"""
Underwriter Notes Parser
========================
Parses life insurance underwriter co_summary notes to isolate the impact of
labs (blood, urine, saliva, oral swab) and APS (attending physician statements)
on risk classification.

Excludes information from build charts, income, policy details, applicant
statements, MVR, MIB, and other non-lab/APS sources.

Output:
  - parsed_findings.csv   — structured findings from labs/APS clauses
  - unknown_terms.csv     — abbreviations/clauses that need manual review
"""

import re
import pandas as pd
import spacy
from typing import Optional

# =============================================================================
# === CONFIGURABLE DICTIONARIES — EDIT THESE ===
# =============================================================================

# Sources to INCLUDE — labs and APS variants
# Keys are lowercase patterns to match; values are the canonical source name.
LAB_AND_APS_SOURCE_TERMS = {
    "bld": "blood",
    "blood": "blood",
    "ur": "urine",
    "urine": "urine",
    "saliva": "saliva",
    "os": "oral swab",
    "oral swab": "oral swab",
    "oral fluid": "oral swab",
    "aps": "APS",
    "attending physician statement": "APS",
    "attending physician": "APS",
    "labs": "labs-general",
    "lab": "labs-general",
    "lab results": "labs-general",
    "lab work": "labs-general",
}

# Risk class terms — keys are lowercase abbreviations; values are canonical names.
RISK_CLASS_TERMS = {
    "elite": "elite",
    "preferred plus": "preferred plus",
    "pp": "preferred plus",
    "ppnt": "preferred plus non-tobacco",
    "pref plus": "preferred plus",
    "preferred": "preferred",
    "pref": "preferred",
    "standard": "standard",
    "std": "standard",
    "smoker": "smoker",
    "smkr": "smoker",
    "skr": "smoker",
    "non-smoker": "non-smoker",
    "nonsmoker": "non-smoker",
    "non smoker": "non-smoker",
    "n/s": "non-smoker",
    "ns": "non-smoker",
    "nsmkr": "non-smoker",
    "table rated": "table rated",
    "tbr": "table rated",
    "tbl": "table rated",
    "substandard": "substandard",
    "sub std": "substandard",
    "substd": "substandard",
    "decline": "decline",
    "declined": "decline",
}

# Sources to EXCLUDE — if a clause is primarily about one of these, skip it.
EXCLUDE_SOURCE_TERMS = {
    "build",
    "bmi",
    "ht/wt",
    "ht wt",
    "height/weight",
    "height weight",
    "height",
    "weight",
    "income",
    "income verified",
    "mvr",
    "motor vehicle",
    "mib",
    "inspection",
    "inspection report",
    "app statement",
    "applicant statement",
    "application",
    "financial",
    "policy amt",
    "policy amount",
    "face amount",
    "face amt",
}

# Regex patterns for rating impacts (e.g., +25, +50, table D)
RATING_PATTERNS = [
    r"[+-]\d+",              # e.g. +25, +50, +125, -25
    r"table\s*[a-zA-Z0-9]+", # e.g. table D, table 4, table H
    r"flat extra\s*\$?\d+",  # e.g. flat extra $5
]

# Medical abbreviations — keys are lowercase; values are expansions (for reference).
MEDICAL_ABBREVIATIONS = {
    "hx": "history",
    "tx": "treatment",
    "rx": "prescription",
    "dx": "diagnosis",
    "sx": "symptoms",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "d/t": "due to",
    "wnl": "within normal limits",
    "w/": "with",
    "w/o": "without",
    "w/in": "within",
    "chol": "cholesterol",
    "gluc": "glucose",
    "a1c": "hemoglobin a1c",
    "hba1c": "hemoglobin a1c",
    "nic": "nicotine",
    "cot": "cotinine",
    "elev": "elevated",
    "neg": "negative",
    "pos": "positive",
    "abn": "abnormal",
    "nml": "normal",
    "fbs": "fasting blood sugar",
    "lipids": "lipids",
    "ldl": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "trig": "triglycerides",
    "ast": "AST (liver enzyme)",
    "alt": "ALT (liver enzyme)",
    "ggt": "GGT (liver enzyme)",
    "bun": "blood urea nitrogen",
    "creat": "creatinine",
    "psa": "prostate-specific antigen",
    "ekg": "electrocardiogram",
    "ecg": "electrocardiogram",
    "bp": "blood pressure",
    "hr": "heart rate",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "cvd": "cardiovascular disease",
    "copd": "chronic obstructive pulmonary disease",
    "ca": "cancer",
    "fx": "fracture",
    "unremarkable": "unremarkable (no significant findings)",
    "tentative": "tentative (preliminary)",
    "tracking": "tracking (classifying as)",
    "elevated": "elevated",
    "liver": "liver",
    "enzymes": "liver enzymes",
    "enzyme": "enzyme",
    "positive": "positive",
    "negative": "negative",
    "oral": "oral (part of oral swab)",
    "swab": "swab (part of oral swab)",
    "fluid": "fluid",
    "lisinopril": "lisinopril (ACE inhibitor)",
    "metformin": "metformin (diabetes medication)",
    "statin": "statin (cholesterol medication)",
}

# Combined set of ALL known terms (for unknown-term detection)
ALL_KNOWN_TERMS = (
    set(LAB_AND_APS_SOURCE_TERMS.keys())
    | set(RISK_CLASS_TERMS.keys())
    | EXCLUDE_SOURCE_TERMS
    | set(MEDICAL_ABBREVIATIONS.keys())
)


# =============================================================================
# === PARSER LOGIC ===
# =============================================================================

# Load spaCy model
try:
    nlp = spacy.load("en_core_sci_md")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# Add sentencizer as fallback if model lacks it
if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def split_into_clauses(text: str) -> list[str]:
    """
    Split underwriter notes into clauses using both spaCy sentence detection
    and period/semicolon splitting, since notes are often terse fragments.
    """
    # First pass: split on periods, semicolons, and newlines
    # But be careful not to split inside abbreviations like "d/t" or "w/"
    raw_splits = re.split(r'(?<=[.;])\s+|\n+', text.strip())

    clauses = []
    for segment in raw_splits:
        segment = segment.strip()
        if not segment:
            continue
        # Use spaCy for further sentence splitting if the segment is long
        if len(segment) > 80:
            doc = nlp(segment)
            for sent in doc.sents:
                s = sent.text.strip()
                if s:
                    clauses.append(s)
        else:
            clauses.append(segment)

    # If nothing was split, return the whole text as one clause
    if not clauses:
        clauses = [text.strip()]

    return clauses


def detect_source(clause_lower: str) -> tuple[Optional[str], str]:
    """
    Determine the information source for a clause.

    Returns:
        (source_type, confidence) where source_type is:
        - A canonical lab/APS source name if detected
        - "EXCLUDE" if it's an excluded source
        - None if ambiguous
    """
    # Check for excluded sources first
    for excl_term in sorted(EXCLUDE_SOURCE_TERMS, key=len, reverse=True):
        # Use word boundary matching for short terms to avoid false positives
        if len(excl_term) <= 3:
            pattern = r'(?<![a-z])' + re.escape(excl_term) + r'(?![a-z])'
        else:
            pattern = r'\b' + re.escape(excl_term) + r'\b'
        if re.search(pattern, clause_lower):
            # Check if there's ALSO a lab/APS source in the same clause
            lab_source = _detect_lab_aps_source(clause_lower)
            if lab_source:
                # Both present — ambiguous. But if the clause starts with
                # the lab/APS source or uses "d/t labs" / "per APS", keep it.
                if _clause_primarily_about_lab_aps(clause_lower, lab_source):
                    return lab_source, "medium"
                else:
                    return "EXCLUDE", "high"
            return "EXCLUDE", "high"

    # Check for lab/APS sources
    lab_source = _detect_lab_aps_source(clause_lower)
    if lab_source:
        return lab_source, "high"

    # Check if the clause references risk class terms alongside "d/t labs",
    # "per labs", "d/t aps", etc. — these are lab/APS-driven risk decisions
    for trigger in [r'd/t\s+labs', r'd/t\s+lab', r'per\s+labs', r'per\s+lab',
                    r'd/t\s+aps', r'per\s+aps', r'd/t\s+bld', r'd/t\s+blood',
                    r'd/t\s+ur', r'd/t\s+urine']:
        if re.search(trigger, clause_lower):
            # Determine the specific lab source from the trigger
            for src_key, src_val in LAB_AND_APS_SOURCE_TERMS.items():
                if src_key in trigger:
                    return src_val, "high"
            return "labs-general", "medium"

    return None, "low"


def _detect_lab_aps_source(clause_lower: str) -> Optional[str]:
    """Check if clause references a lab or APS source. Return canonical name or None."""
    # Check multi-word terms first (longer terms first to avoid partial matches)
    for term in sorted(LAB_AND_APS_SOURCE_TERMS.keys(), key=len, reverse=True):
        if ' ' in term or '/' in term:
            # Multi-word or slash terms: direct substring match
            if term in clause_lower:
                return LAB_AND_APS_SOURCE_TERMS[term]
        else:
            # Single-word terms: use boundary matching
            # For very short terms (2 chars like "ur", "os"), be strict
            if len(term) <= 2:
                pattern = r'(?<![a-z])' + re.escape(term) + r'(?![a-z/])'
            else:
                pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, clause_lower):
                return LAB_AND_APS_SOURCE_TERMS[term]
    return None


def _clause_primarily_about_lab_aps(clause_lower: str, lab_source: str) -> bool:
    """Determine if a clause with mixed sources is primarily about labs/APS."""
    # If clause starts with a lab/APS term, it's primarily about that
    for term in LAB_AND_APS_SOURCE_TERMS:
        if clause_lower.startswith(term):
            return True
    # If "d/t" or "per" points to labs/APS, it's primarily lab-driven
    if re.search(r'(?:d/t|per)\s+(?:labs?|aps|bld|blood|ur|urine)', clause_lower):
        return True
    return False


def extract_risk_impact(clause_lower: str) -> Optional[str]:
    """Extract risk class or rating impact from a clause."""
    impacts = []

    # Check for rating patterns (+25, table D, etc.)
    for pattern in RATING_PATTERNS:
        match = re.search(pattern, clause_lower)
        if match:
            impacts.append(match.group(0))

    # Check for risk class terms — collect ALL that appear.
    # Sort by length descending to match longer terms first and avoid
    # double-counting (e.g., "preferred plus" before "preferred").
    # Track position in the clause so we can preserve original order.
    already_matched_canonical = set()
    risk_hits = []  # list of (position_in_clause, canonical_name)
    working = clause_lower
    for term in sorted(RISK_CLASS_TERMS.keys(), key=len, reverse=True):
        canonical = RISK_CLASS_TERMS[term]
        if canonical in already_matched_canonical:
            continue
        if ' ' in term or '/' in term:
            idx = working.find(term)
            if idx != -1:
                risk_hits.append((idx, canonical))
                already_matched_canonical.add(canonical)
                working = working[:idx] + ' ' * len(term) + working[idx + len(term):]
        else:
            if len(term) <= 2:
                pat = r'(?<![a-z])' + re.escape(term) + r'(?![a-z/])'
            else:
                pat = r'\b' + re.escape(term) + r'\b'
            m = re.search(pat, working)
            if m:
                risk_hits.append((m.start(), canonical))
                already_matched_canonical.add(canonical)
                working = working[:m.start()] + ' ' * (m.end() - m.start()) + working[m.end():]

    # Sort by position in the clause to preserve original order
    risk_hits.sort(key=lambda x: x[0])
    impacts.extend([canonical for _, canonical in risk_hits])

    if not impacts:
        return None

    # Join with space ("preferred plus non-smoker") rather than comma
    # when the terms form a compound risk class
    return " ".join(impacts)


def extract_finding(clause: str, clause_lower: str, source_type: str) -> Optional[str]:
    """
    Extract the medical finding from a clause by removing the source reference,
    rating/risk terms, and connecting words, leaving the core medical content.
    """
    finding = clause_lower

    # Remove the source reference itself
    for term in sorted(LAB_AND_APS_SOURCE_TERMS.keys(), key=len, reverse=True):
        if ' ' in term:
            finding = finding.replace(term, ' ')
        else:
            if len(term) <= 2:
                finding = re.sub(r'(?<![a-z])' + re.escape(term) + r'(?![a-z/])', ' ', finding)
            else:
                finding = re.sub(r'\b' + re.escape(term) + r'\b', ' ', finding)

    # Remove rating patterns
    for pattern in RATING_PATTERNS:
        finding = re.sub(pattern, ' ', finding)

    # Remove risk class terms
    for term in sorted(RISK_CLASS_TERMS.keys(), key=len, reverse=True):
        if ' ' in term or '/' in term:
            finding = finding.replace(term, ' ')
        else:
            if len(term) <= 2:
                finding = re.sub(r'(?<![a-z])' + re.escape(term) + r'(?![a-z/])', ' ', finding)
            else:
                finding = re.sub(r'\b' + re.escape(term) + r'\b', ' ', finding)

    # Remove common connecting/structural words
    noise_words = [
        r'\bshows\b', r'\breveals\b', r'\bindicates\b', r'\bconfirms\b',
        r'\bd/t\b', r'\bper\b', r'\btracking\b', r'\btentative\b',
        r'\bverified\b', r'\bok\b', r'\bfor\b', r'\bthe\b', r'\ba\b',
        r'\bof\b', r'\bhx\b',
    ]
    for nw in noise_words:
        finding = re.sub(nw, ' ', finding)

    # Clean up punctuation and whitespace
    finding = re.sub(r'[.,;:]+', ' ', finding)
    finding = re.sub(r'\s+', ' ', finding).strip()

    # If what's left is empty or too short to be meaningful, return None
    if not finding or len(finding) < 2:
        return None

    return finding


def detect_unknown_terms(clause: str, clause_lower: str) -> list[dict]:
    """
    Find abbreviation-like tokens in the clause that aren't in any dictionary.
    Returns list of {unknown_term, surrounding_context}.
    """
    unknowns = []
    # Look for short tokens (2-5 chars) that look like abbreviations
    # (all lowercase, possibly with slashes or periods)
    tokens = re.findall(r'\b([a-z][a-z/\.]{1,5})\b', clause_lower)

    for token in tokens:
        token_clean = token.rstrip('.')
        if token_clean in ALL_KNOWN_TERMS:
            continue
        # Skip very common English words
        common_words = {
            'the', 'and', 'for', 'with', 'from', 'that', 'this', 'has',
            'was', 'are', 'not', 'but', 'all', 'can', 'had', 'her',
            'his', 'one', 'our', 'out', 'day', 'get', 'may', 'new',
            'now', 'old', 'see', 'two', 'way', 'who', 'did', 'its',
            'let', 'say', 'she', 'too', 'use', 'per', 'due', 'via',
            'ok', 'no', 'yes', 'shows', 'show', 'by', 'at', 'to',
            'in', 'of', 'on', 'or', 'an', 'it', 'is', 'be', 'as',
            'do', 'if', 'so', 'up', 'he', 'we', 'my', 'am',
        }
        if token_clean in common_words:
            continue
        # Get surrounding context (10 chars each side)
        idx = clause_lower.find(token_clean)
        start = max(0, idx - 15)
        end = min(len(clause_lower), idx + len(token_clean) + 15)
        context = clause[start:end].strip()

        unknowns.append({
            "unknown_term": token_clean,
            "surrounding_context": f"...{context}..."
        })

    return unknowns


def parse_notes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main parsing function.

    Args:
        df: DataFrame with columns pol_key_id and co_summary

    Returns:
        (findings_df, unknowns_df)
    """
    findings_rows = []
    unknown_rows = []

    for _, row in df.iterrows():
        pol_key_id = row["pol_key_id"]
        co_summary = row["co_summary"]

        if pd.isna(co_summary) or not str(co_summary).strip():
            continue

        text = str(co_summary).strip()
        clauses = split_into_clauses(text)

        for clause in clauses:
            clause_lower = clause.lower().strip()

            # --- Detect unknown terms (do this for ALL clauses) ---
            unknowns = detect_unknown_terms(clause, clause_lower)
            for u in unknowns:
                unknown_rows.append({
                    "pol_key_id": pol_key_id,
                    "unknown_term": u["unknown_term"],
                    "surrounding_context": u["surrounding_context"],
                })

            # --- Determine source ---
            source_type, source_confidence = detect_source(clause_lower)

            # Skip excluded sources
            if source_type == "EXCLUDE":
                continue

            # If no source detected, check if this is a risk-tracking clause
            # referencing labs (e.g., "Tracking pp nsmkr" in a context where
            # surrounding clauses are about labs)
            if source_type is None:
                # Check if the clause has risk class terms and references labs
                # via "d/t labs" etc. — already handled in detect_source.
                # Otherwise, check for "tracking" + risk class which often
                # follows a lab clause contextually.
                if re.search(r'\btracking\b', clause_lower):
                    risk = extract_risk_impact(clause_lower)
                    if risk:
                        # This is likely a risk decision clause — check if
                        # it explicitly references labs/APS
                        if re.search(r'd/t\s+(?:labs?|bld|blood|ur|urine)', clause_lower):
                            source_type = "labs-general"
                            source_confidence = "high"
                        else:
                            # "Tracking [risk class]" without explicit source.
                            # In underwriter notes, "tracking" clauses that
                            # immediately follow lab results are typically
                            # lab-driven risk decisions. Include as labs-general
                            # with medium confidence.
                            source_type = "labs-general"
                            source_confidence = "medium"
                else:
                    continue

            # --- Extract findings ---
            risk_impact = extract_risk_impact(clause_lower)
            finding = extract_finding(clause, clause_lower, source_type)

            # Determine confidence
            confidence = source_confidence

            findings_rows.append({
                "pol_key_id": pol_key_id,
                "source_type": source_type,
                "finding": finding if finding else None,
                "risk_impact": risk_impact,
                "raw_text": clause,
                "confidence": confidence,
            })

    findings_df = pd.DataFrame(findings_rows, columns=[
        "pol_key_id", "source_type", "finding", "risk_impact",
        "raw_text", "confidence"
    ])

    unknowns_df = pd.DataFrame(unknown_rows, columns=[
        "pol_key_id", "unknown_term", "surrounding_context"
    ])

    # Deduplicate unknowns
    if not unknowns_df.empty:
        unknowns_df = unknowns_df.drop_duplicates()

    return findings_df, unknowns_df


# =============================================================================
# === TEST DATA & EXECUTION ===
# =============================================================================

if __name__ == "__main__":
    # Few-shot test examples
    test_data = [
        {"pol_key_id": 1, "co_summary": "Blood +25 d/t elevated cholesterol. Build ok. APS shows hx of htn tx w/ lisinopril."},
        {"pol_key_id": 2, "co_summary": "Labs wnl. Tracking pp nsmkr. Income verified."},
        {"pol_key_id": 3, "co_summary": "Urine positive nicotine, tracking smkr d/t labs. Bld wnl. APS unremarkable."},
        {"pol_key_id": 4, "co_summary": "Tentative std d/t htn per APS. Labs wnl. Oral swab negative."},
        {"pol_key_id": 5, "co_summary": "Bld +50 elevated liver enzymes. Ur wnl. Build tbr +75."},
    ]

    df_input = pd.DataFrame(test_data)

    print("=" * 70)
    print("INPUT DATA")
    print("=" * 70)
    for _, r in df_input.iterrows():
        print(f"  [{r['pol_key_id']}] {r['co_summary']}")
    print()

    findings_df, unknowns_df = parse_notes(df_input)

    print("=" * 70)
    print("PARSED FINDINGS")
    print("=" * 70)
    print(findings_df.to_string(index=False))
    print()

    print("=" * 70)
    print("UNKNOWN TERMS")
    print("=" * 70)
    if unknowns_df.empty:
        print("  (none)")
    else:
        print(unknowns_df.to_string(index=False))
    print()

    # Save to CSV
    findings_df.to_csv("/home/user/workspace/parsed_findings.csv", index=False)
    unknowns_df.to_csv("/home/user/workspace/unknown_terms.csv", index=False)
    print("Saved: parsed_findings.csv, unknown_terms.csv")
