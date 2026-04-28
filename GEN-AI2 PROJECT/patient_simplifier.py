"""
modules/patient_simplifier.py
──────────────────────────────
Generates a plain-language patient summary from SOAP note + medications.
"""

import json
import re

from models import SOAPNote, MedicationExplanation
from llm_engine import call_llm
from prompt_templates import patient_summary_prompt_v2


_UNSUPPORTED_PATIENT_PHRASES = (
    "this means",
    "this happened because",
    "tests showed",
)

_PATIENT_TERM_REPLACEMENTS = (
    ("water pill", "medicine to remove extra fluid"),
    ("a&e", "the emergency department"),
    ("ed", "the emergency department"),
    ("yellow spit", "yellow mucus"),
    ("through a vein", "through an IV"),
    ("in your arm", "through an IV"),
    ("through a tube in your arm", "through an IV"),
)

_REDUNDANT_FRAGMENT_PREFIXES = (
    "home medications",
    "medications",
    "diagnoses",
    "diagnosis",
    "problems",
    "encourage fluid intake",
)

_LIKELY_FRAGMENT_TERMS = {
    "albuterol",
    "inhaler",
    "ceftriaxone",
    "azithromycin",
    "pneumonia",
    "asthma",
    "community-acquired",
    "community",
    "acquired",
    "diagnosis",
    "diagnoses",
    "medication",
    "medications",
}

_COMMON_SENTENCE_VERBS = {
    "am", "is", "are", "was", "were", "be", "being", "been",
    "have", "has", "had",
    "do", "does", "did",
    "go", "goes", "went",
    "give", "gives", "gave",
    "take", "takes", "taking",
    "keep", "keeps",
    "use", "uses",
    "need", "needs",
    "start", "starts",
    "watch", "watches",
    "test", "tests",
    "come", "comes",
    "help", "helps",
    "move", "moving",
    "treat", "treats",
}


def _normalize_sentence_key(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _token_set(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) > 2
    }


def _is_redundant_sentence(candidate: str, prior_sentences: list[str]) -> bool:
    candidate_key = _normalize_sentence_key(candidate)
    if not candidate_key:
        return True

    candidate_tokens = _token_set(candidate)
    if not candidate_tokens:
        return False

    for previous in prior_sentences:
        previous_key = _normalize_sentence_key(previous)
        if candidate_key == previous_key:
            return True

        previous_tokens = _token_set(previous)
        if not previous_tokens:
            continue

        overlap = len(candidate_tokens & previous_tokens) / max(1, len(candidate_tokens))
        if overlap >= 0.8:
            return True

    return False


def _looks_like_fragment(sentence: str) -> bool:
    tokens = re.findall(r"[a-zA-Z-']+", sentence.lower())
    if len(tokens) < 4:
        return False

    token_set = set(tokens)
    has_common_verb = any(token in _COMMON_SENTENCE_VERBS for token in token_set)
    fragment_term_count = sum(token in _LIKELY_FRAGMENT_TERMS for token in tokens)

    if not has_common_verb and fragment_term_count >= 2:
        return True

    if ":" in sentence and fragment_term_count >= 2:
        return True

    return False


def sanitize_patient_summary(summary: str) -> str:
    summary = summary.replace("\r", "\n")
    summary = re.sub(r"^[\-\*\u2022]+\s*", "", summary.strip(), flags=re.M)
    summary = re.sub(r"\s*\*\s*", " ", summary.strip())
    summary = re.sub(r"\n{2,}", "\n", summary)
    summary = re.sub(r"\s{2,}", " ", summary)

    chunks: list[str] = []
    for block in re.split(r"\n+", summary.strip()):
        chunks.extend(re.split(r"(?<=[.!?])\s+", block.strip()))

    kept: list[str] = []
    for sentence in chunks:
        cleaned = sentence.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(phrase in lowered for phrase in _UNSUPPORTED_PATIENT_PHRASES):
            continue
        for source, replacement in _PATIENT_TERM_REPLACEMENTS:
            cleaned = re.sub(rf"\b{re.escape(source)}\b", replacement, cleaned, flags=re.I)
        lowered = cleaned.lower().strip(" .")
        if any(lowered.startswith(prefix) for prefix in _REDUNDANT_FRAGMENT_PREFIXES) and kept:
            continue
        if _looks_like_fragment(cleaned):
            continue
        if _is_redundant_sentence(cleaned, kept):
            continue
        kept.append(cleaned)
    return " ".join(kept).strip()


def generate_patient_summary(
    original_note: str,
    soap_note:    SOAPNote,
    medications:  MedicationExplanation,
    target_grade: int = 6,
) -> tuple[str, str]:
    """
    Generate a patient-friendly English summary.

    Returns:
        (summary_text, prompt_hash) tuple
    """
    soap_json = soap_note.model_dump_json(indent=2)
    med_payload = {
        "home_medications": [med.model_dump() for med in medications.home_medications],
        "inpatient_medications": [med.model_dump() for med in medications.inpatient_medications],
    }
    med_json = json.dumps(med_payload, indent=2)

    system, user = patient_summary_prompt_v2(original_note, soap_json, med_json, target_grade)
    summary, prompt_hash = call_llm(system, user, expect_json=False, use_cache=False)

    return sanitize_patient_summary(summary), prompt_hash
