"""
evaluation.py
──────────────
Evaluation pipeline for DischargeAssist outputs.
Computes ROUGE-L against MTS-Dialog reference summaries.
BERTScore is optional (heavy dependency — install separately).

Usage:
    from evaluation import evaluate_summary
    scores = evaluate_summary(generated_text, reference_text)
"""

import re
from dataclasses import dataclass


@dataclass
class EvaluationScores:
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    bert_score: float | None = None


# ── Minimal ROUGE implementation (no external dependency) ─────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _rouge_n(hypothesis: str, reference: str, n: int) -> float:
    """Compute ROUGE-N F1 score."""
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    hyp_ngrams = _ngrams(hyp_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)

    if not ref_ngrams:
        return 0.0

    overlap = sum(
        min(hyp_ngrams.get(gram, 0), ref_ngrams[gram])
        for gram in ref_ngrams
    )

    precision = overlap / max(1, sum(hyp_ngrams.values()))
    recall    = overlap / max(1, sum(ref_ngrams.values()))

    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 4)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest Common Subsequence length via DP."""
    m, n = len(a), len(b)
    # Space-optimised O(min(m,n))
    if m < n:
        a, b, m, n = b, a, n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    return prev[n]


def _rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 score based on LCS."""
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)

    if not hyp or not ref:
        return 0.0

    lcs   = _lcs_length(hyp, ref)
    prec  = lcs / len(hyp)
    rec   = lcs / len(ref)

    if prec + rec == 0:
        return 0.0

    return round(2 * prec * rec / (prec + rec), 4)


# ── BERTScore (optional) ───────────────────────────────────────────────────────

def _bert_score(hypothesis: str, reference: str) -> float | None:
    """
    Compute BERTScore F1. Returns None if bert-score is not installed.
    Install with: pip install bert-score
    """
    try:
        from bert_score import score as bs_score
        P, R, F1 = bs_score([hypothesis], [reference], lang="en", verbose=False)
        return round(float(F1[0]), 4)
    except ImportError:
        return None


# ── Main evaluation function ───────────────────────────────────────────────────

def evaluate_summary(generated: str, reference: str) -> EvaluationScores:
    """
    Evaluate a generated summary against a reference summary.

    Args:
        generated:  The model-generated discharge summary
        reference:  The ground-truth summary (from MTS-Dialog dataset)

    Returns:
        EvaluationScores with ROUGE-1, ROUGE-2, ROUGE-L, and optionally BERTScore
    """
    return EvaluationScores(
        rouge1_f   = _rouge_n(generated, reference, 1),
        rouge2_f   = _rouge_n(generated, reference, 2),
        rougeL_f   = _rouge_l(generated, reference),
        bert_score = _bert_score(generated, reference),
    )


def print_scores(scores: EvaluationScores, label: str = ""):
    header = f"── Evaluation Scores {label} ".ljust(50, "─")
    print(header)
    print(f"  ROUGE-1 F1 : {scores.rouge1_f:.4f}")
    print(f"  ROUGE-2 F1 : {scores.rouge2_f:.4f}")
    print(f"  ROUGE-L F1 : {scores.rougeL_f:.4f}")
    if scores.bert_score is not None:
        print(f"  BERTScore  : {scores.bert_score:.4f}")
    else:
        print("  BERTScore  : (install bert-score package to enable)")
    print("─" * 50)
