"""
modules/medication_explainer.py
────────────────────────────────
Extracts and classifies medications found in the clinical note.
"""

import re

from models import MedicationExplanation, MedicationEntry
from llm_engine import call_llm, parse_json_response
from prompt_templates import medication_prompt

_SECTION_STOP_MARKERS = (
    "allergies:",
    "vitals:",
    "exam:",
    "labs:",
    "ecg:",
    "cxr:",
    "a/p:",
    "assessment:",
    "plan:",
    "pmh:",
    "hpi:",
    "cc:",
)


def _normalize_med_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return cleaned.split()[0] if cleaned else ""


def _extract_section(text: str, heading_pattern: str) -> str:
    match = re.search(rf"(?is)\b{heading_pattern}\b\s*:\s*", text)
    if not match:
        return ""
    remaining = text[match.end():]
    lowered = remaining.lower()
    cut_points = [lowered.find(marker) for marker in _SECTION_STOP_MARKERS if lowered.find(marker) != -1]
    end = min(cut_points) if cut_points else len(remaining)
    return remaining[:end].strip()


def _normalize_missing_text(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = str(value).strip()
    if cleaned.lower() in {"not documented", "not specified", "none documented", "null", "unknown"}:
        return fallback
    return cleaned


def _purpose_is_explicit(normalized_note: str, med_name: str, purpose: str) -> bool:
    med_key = _normalize_med_name(med_name)
    purpose_key = str(purpose).strip().lower()
    if not med_key or not purpose_key or purpose_key == "not documented":
        return False

    fragments = re.split(r"[\n.;]", normalized_note)
    for fragment in fragments:
        lowered = fragment.lower()
        if med_key in lowered and purpose_key in lowered:
            return True
    return False


def _classify_medications(normalized_note: str, entries: list[MedicationEntry]) -> tuple[list[MedicationEntry], list[MedicationEntry], list[MedicationEntry]]:
    home_section = _extract_section(normalized_note, r"medications?")
    home_section_normalized = home_section.lower()

    categorized_entries: list[MedicationEntry] = []
    home_medications: list[MedicationEntry] = []
    inpatient_medications: list[MedicationEntry] = []

    for med in entries:
        med_key = _normalize_med_name(med.name)
        in_home_section = med_key and med_key in home_section_normalized
        categorized = med.model_copy(update={
            "category": "home_discharge" if in_home_section else "inpatient_plan"
        })
        categorized_entries.append(categorized)
        if in_home_section:
            home_medications.append(categorized)
        else:
            inpatient_medications.append(categorized)

    return categorized_entries, home_medications, inpatient_medications


def explain_medications(normalized_note: str) -> tuple[MedicationExplanation, str]:
    """
    Extract medications from a normalized note and explain each one.

    Returns:
        (MedicationExplanation, prompt_hash) tuple
    """
    system, user = medication_prompt(normalized_note)
    response, prompt_hash = call_llm(system, user, expect_json=True, use_cache=False)
    data = parse_json_response(response)

    entries: list[MedicationEntry] = []
    for med in data.get("medications", []):
        name = med.get("name", "Unknown")
        purpose = _normalize_missing_text(med.get("purpose"), "Not documented")
        if purpose != "Not documented" and not _purpose_is_explicit(normalized_note, name, purpose):
            purpose = "Not documented"
        entries.append(MedicationEntry(
            name=name,
            dose=_normalize_missing_text(med.get("dose"), "Not specified"),
            frequency=_normalize_missing_text(med.get("frequency"), "Not specified"),
            purpose=purpose,
            warnings=med.get("warnings", None),
        ))

    categorized_entries, home_medications, inpatient_medications = _classify_medications(normalized_note, entries)

    return MedicationExplanation(
        medications=categorized_entries,
        home_medications=home_medications,
        inpatient_medications=inpatient_medications,
    ), prompt_hash
