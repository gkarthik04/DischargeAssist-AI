"""
novelty/multilingual.py
────────────────────────
Novelty Feature 3 — Multilingual Patient Summary

Translates the patient-friendly English summary into regional Indian
languages using a dedicated LLM call with a medical translation prompt.

Supported: Hindi (hi), Telugu (te), Tamil (ta), Kannada (kn),
           Marathi (mr), Bengali (bn), Gujarati (gu)
"""

from llm_engine import call_llm
from prompt_templates import translation_prompt, LANGUAGE_NAMES


def translate_summary(english_summary: str, target_lang: str) -> str | None:
    """
    Translate the patient summary into the target language.

    Args:
        english_summary: The plain-English patient summary
        target_lang:     ISO 639-1 language code (e.g., 'hi', 'te')

    Returns:
        Translated text, or None if target_lang is 'en' or unsupported
    """
    if target_lang == "en":
        return None

    if target_lang not in LANGUAGE_NAMES:
        raise ValueError(
            f"Language '{target_lang}' not supported. "
            f"Supported: {list(LANGUAGE_NAMES.keys())}"
        )

    system, user   = translation_prompt(english_summary, target_lang)
    translated, _  = call_llm(system, user, expect_json=False, use_cache=False)
    return translated.strip()
