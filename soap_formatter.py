"""
modules/soap_formatter.py
─────────────────────────
Generates a structured SOAP note from a preprocessed clinical note.
"""

from models import SOAPNote
from llm_engine import call_llm, parse_json_response
from prompt_templates import soap_prompt


def generate_soap_note(normalized_note: str) -> tuple[SOAPNote, str]:
    """
    Generate a SOAP note from a normalized clinical note.
    
    Returns:
        (SOAPNote, prompt_hash) tuple
    """
    system, user = soap_prompt(normalized_note)
    response, prompt_hash = call_llm(system, user, expect_json=True, use_cache=False)
    data = parse_json_response(response)

    return SOAPNote(
        subjective = data.get("subjective", "Not documented"),
        objective  = data.get("objective",  "Not documented"),
        assessment = data.get("assessment", "Not documented"),
        plan       = data.get("plan",       "Not documented"),
    ), prompt_hash
