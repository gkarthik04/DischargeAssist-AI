"""
novelty/readability.py
───────────────────────
Novelty Feature 1 — Adaptive Readability Scoring

Computes Flesch-Kincaid Grade Level of patient summaries.
If the reading level exceeds the target, it automatically re-prompts
the LLM to simplify the text, up to MAX_PASSES refinement rounds.
"""

import re
from models import ReadabilityReport
from llm_engine import call_llm
from prompt_templates import simplify_further_prompt

MAX_PASSES = 3


# ── Flesch-Kincaid Grade Level (no external library needed) ───────────────────

def _count_syllables(word: str) -> int:
    """Estimate syllable count using vowel-group heuristic."""
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count  = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent 'e' at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_kincaid_grade(text: str) -> float:
    """
    Compute Flesch-Kincaid Grade Level.
    FK Grade = 0.39 × (words/sentences) + 11.8 × (syllables/words) − 15.59
    """
    # Split into sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(1, len(sentences))

    # Split into words
    words = re.findall(r"[a-zA-Z']+", text)
    n_words = max(1, len(words))

    # Count syllables
    n_syllables = sum(_count_syllables(w) for w in words)

    grade = (
        0.39  * (n_words / n_sentences)
        + 11.8 * (n_syllables / n_words)
        - 15.59
    )
    return round(max(0.0, grade), 2)


# ── Main adaptive readability function ────────────────────────────────────────

def adapt_readability(
    summary:      str,
    target_grade: int = 6,
) -> tuple[str, ReadabilityReport]:
    """
    Check readability of summary and iteratively simplify if above target.

    Args:
        summary:      The initial patient-facing summary text
        target_grade: Target Flesch-Kincaid grade level (default 6)

    Returns:
        (final_summary, ReadabilityReport) tuple
    """
    original_grade = _flesch_kincaid_grade(summary)
    current_text   = summary
    current_grade  = original_grade
    passes         = 0

    while current_grade > target_grade and passes < MAX_PASSES:
        system, user = simplify_further_prompt(current_text, current_grade, target_grade)
        simplified, _ = call_llm(system, user, expect_json=False, use_cache=False)
        current_text  = simplified.strip()
        current_grade = _flesch_kincaid_grade(current_text)
        passes += 1

    report = ReadabilityReport(
        original_grade    = original_grade,
        final_grade       = current_grade,
        passes_target     = current_grade <= target_grade,
        refinement_passes = passes,
    )

    return current_text, report
