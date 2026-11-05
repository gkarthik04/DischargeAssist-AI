"""
Confidence scoring and hallucination detection for generated discharge content.
Uses source-grounded rules so the score reflects faithfulness to the note
instead of speculative "standard practice" expectations.
"""

import re

from models import ConfidenceReport


_DOSAGE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|mL|ml|g|mmol|mEq|units?)\b", re.I)
_DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
    re.I,
)
_VALUE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*mmHg\b|"
    r"\b\d+(?:\.\d+)?\s*(?:Â°|Ã‚Â°)?[CF]\b|"
    r"\b\d+(?:\.\d+)?\s*%\b",
    re.I,
)
_MED_WITH_DOSE_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z-]{2,}(?:\s+[A-Z][a-zA-Z-]{2,})?)\b(?=\s+\d+(?:\.\d+)?\s*(?:mg|mcg|mL|ml|g|units?)\b)"
)
_BRAND_NAME_PATTERN = re.compile(r"\b[A-Z][a-zA-Z-]{3,}\s*\([A-Z][a-zA-Z-]{3,}\)")


def _normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("Ã‚Â°", "Â°")
        .replace("â€“", "-")
        .replace("â€”", "-")
    )


def _extract_high_salience_terms(text: str) -> list[str]:
    extracted: list[str] = []
    for pattern in (
        _DOSAGE_PATTERN,
        _DATE_PATTERN,
        _VALUE_PATTERN,
        _MED_WITH_DOSE_PATTERN,
        _BRAND_NAME_PATTERN,
    ):
        for match in pattern.findall(text):
            if isinstance(match, tuple):
                extracted.extend(part for part in match if part)
            else:
                extracted.append(match)
    return extracted


def _sanitize_hallucinated_terms(terms: list[str]) -> list[str]:
    cleaned: list[str] = []
    for term in terms:
        text = str(term).strip()
        if not text:
            continue
        if len(text) > 60:
            continue
        if len(text.split()) > 6:
            continue
        if any(punct in text for punct in ".!?") and len(text.split()) > 3:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def _sanitize_warnings(warnings: list[str]) -> list[str]:
    cleaned: list[str] = []
    for warning in warnings:
        text = str(warning).strip()
        if not text:
            continue
        if text.lower() in {"none", "none documented", "null"}:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def _build_rule_warnings(
    original_note: str,
    generated_text: str,
    hallucinated_terms: list[str],
) -> list[str]:
    warnings: list[str] = []

    generated_words = re.findall(r"[a-zA-Z0-9']+", generated_text)
    if len(generated_words) < 35:
        warnings.append("The patient summary is brief and may leave out useful context.")

    original_normalized = _normalize_text(original_note)
    generated_normalized = _normalize_text(generated_text)

    severity_markers = (
        "shortness of breath",
        "wheezing",
        "accessory muscles",
        "spo2",
        "oxygen",
    )
    missing_markers = [
        marker for marker in severity_markers
        if marker in original_normalized and marker not in generated_normalized
    ]
    if len(missing_markers) >= 2:
        warnings.append("The summary may understate severity by omitting multiple documented clinical details.")

    if hallucinated_terms:
        warnings.append("Some clinical details in the summary could not be directly matched to the source note.")

    return warnings


def _rule_based_check(original_note: str, generated_text: str) -> tuple[float, list[str], list[str]]:
    """
    Score confidence using only source-grounded checks.
    Clean summaries should score high; unsupported details should lower confidence.
    """
    original_normalized = _normalize_text(original_note)
    hallucinated: list[str] = []

    if not generated_text or len(generated_text.strip()) < 10:
        return 0.25, [], ["The generated summary is too short to judge reliably."]

    score = 0.86

    salience_terms = _extract_high_salience_terms(generated_text)
    for term in salience_terms:
        normalized_term = _normalize_text(term).strip()
        if len(normalized_term) < 3:
            continue
        if normalized_term not in original_normalized:
            hallucinated.append(term.strip())

    hallucinated = sorted(set(hallucinated), key=str.lower)

    if hallucinated:
        score -= min(len(hallucinated) * 0.08, 0.30)
    else:
        score += 0.05

    if not salience_terms:
        score -= 0.10
    else:
        score += 0.03

    warnings = _build_rule_warnings(original_note, generated_text, hallucinated)
    score = round(max(0.35, min(0.96, score)), 3)
    return score, hallucinated, warnings


def score_confidence(
    original_note: str,
    generated_text: str,
    output_type: str = "patient summary",
) -> ConfidenceReport:
    """
    Run a source-grounded confidence check.
    The score reflects faithfulness to documented source content rather than
    whether the system guessed additional standard discharge advice.
    """
    rule_score, rule_hallucinated, rule_warnings = _rule_based_check(original_note, generated_text)

    print(f"[DEBUG] Rule-based score: {rule_score}, hallucinated: {len(rule_hallucinated)}, warnings: {len(rule_warnings)}")

    all_hallucinated = _sanitize_hallucinated_terms(rule_hallucinated)
    meaningful_warnings = _sanitize_warnings(rule_warnings)

    overall_score = max(0.0, min(1.0, rule_score))

    if meaningful_warnings:
        overall_score -= min(0.03 * len(meaningful_warnings), 0.09)

    if all_hallucinated:
        overall_score -= min(0.05 * len(all_hallucinated), 0.20)

    if not all_hallucinated and len(meaningful_warnings) <= 1:
        overall_score = max(overall_score, 0.84)
    elif not all_hallucinated and len(meaningful_warnings) == 2:
        overall_score = max(overall_score, 0.78)

    overall_score = round(max(0.0, min(1.0, overall_score)), 3)
    is_safe = overall_score >= 0.75 and len(all_hallucinated) == 0

    print(f"[DEBUG] Final score: {overall_score}, is_safe: {is_safe}")

    return ConfidenceReport(
        overall_score=overall_score,
        hallucinated_terms=all_hallucinated,
        warnings=meaningful_warnings,
        is_safe_to_use=is_safe,
    )
