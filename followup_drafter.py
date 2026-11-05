"""
modules/followup_drafter.py
────────────────────────────
Generates and sanitizes structured follow-up instructions from the SOAP note.
"""

from models import SOAPNote, FollowUpInstruction
from llm_engine import call_llm, parse_json_response
from prompt_templates import followup_prompt


_INPATIENT_MONITORING_TERMS = (
    "telemetry",
    "troponin",
    "ccu",
    "icu",
    "admit",
    "intravenous",
    "iv",
    "inpatient",
)

_HOME_MONITORING_TERMS = (
    "blood pressure",
    "blood sugar",
    "glucose",
    "weight",
    "swelling",
    "breathing",
    "shortness of breath",
    "symptoms",
    "pulse",
    "temperature",
)


def _clean_optional_text(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.lower() in {"not documented", "not specified", "none documented", "null"}:
        return None
    return cleaned


def _sanitize_list(items: list[str], allowed_terms: tuple[str, ...] | None = None) -> list[str]:
    cleaned: list[str] = []
    for item in items or []:
        text = item.strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"not documented", "not specified", "none documented", "null"}:
            continue
        if any(term in lowered for term in _INPATIENT_MONITORING_TERMS):
            if allowed_terms is None:
                continue
        if allowed_terms is not None and not any(term in lowered for term in allowed_terms):
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def draft_followup(soap_note: SOAPNote) -> tuple[FollowUpInstruction, str]:
    """
    Draft follow-up instructions based on the SOAP note.

    Returns:
        (FollowUpInstruction, prompt_hash) tuple
    """
    soap_json = soap_note.model_dump_json(indent=2)
    system, user = followup_prompt(soap_json)
    response, prompt_hash = call_llm(system, user, expect_json=True, use_cache=False)
    data = parse_json_response(response)

    return FollowUpInstruction(
        appointment_date=_clean_optional_text(data.get("appointment_date")),
        specialist=_clean_optional_text(data.get("specialist")),
        red_flag_symptoms=_sanitize_list(data.get("red_flag_symptoms", [])),
        lifestyle_changes=_sanitize_list(data.get("lifestyle_changes", [])),
        monitoring=_sanitize_list(data.get("monitoring", []), allowed_terms=_HOME_MONITORING_TERMS),
    ), prompt_hash
