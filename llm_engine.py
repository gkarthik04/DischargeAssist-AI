"""
llm_engine.py
─────────────
Centralized LLM API caller for DischargeAssist.
Backend: Google Gemini (FREE tier — 1,500 requests/day on gemini-1.5-flash)

Setup:
    pip install google-generativeai
    export GOOGLE_API_KEY="your-key-here"

Get a free key at: https://aistudio.google.com/app/apikey
"""

import os
import json
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

import google.generativeai as genai

# ── Client setup (API key from env GOOGLE_API_KEY) ────────────────────────────

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Default to available Gemini models and fail over when one is rate-limited.
DEFAULT_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]
ENV_MODEL = os.getenv("GEMINI_MODEL")
MODEL_CANDIDATES = [ENV_MODEL] if ENV_MODEL else DEFAULT_MODELS
_MODELS: dict[str, genai.GenerativeModel] = {}
REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "45"))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
ENABLE_CACHE = os.getenv("ENABLE_LLM_CACHE", "false").lower() == "true"
_RESPONSE_CACHE: dict[str, str] = {}


# ── Core call function ─────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_prompt:   str,
    expect_json:   bool = True,
    retries:       int  = 4,
    use_cache:     bool = True,
) -> tuple[str, str]:
    """
    Call Gemini and return (response_text, prompt_hash).

    Args:
        system_prompt: System-level instructions for the model
        user_prompt:   The user turn content
        expect_json:   If True, validate that response is parseable JSON
        retries:       Number of retry attempts on failure
        use_cache:     If False, bypass response cache (useful for quality checks)

    Returns:
        (response_text, prompt_hash) tuple
    """
    prompt_hash = _hash_prompt(system_prompt + user_prompt)

    # Gemini combines system + user into a single prompt
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    cache_key = hashlib.sha256(f"{expect_json}|{full_prompt}".encode()).hexdigest()
    if ENABLE_CACHE and use_cache and cache_key in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[cache_key], prompt_hash

    generation_config = genai.types.GenerationConfig(
        temperature      = DEFAULT_TEMPERATURE,
        max_output_tokens= 2048,
        # Ask Gemini for JSON output when expected
        response_mime_type = "application/json" if expect_json else "text/plain",
    )

    last_json_error = None
    for attempt in range(retries + 1):
        try:
            response = _generate_content(
                full_prompt,
                generation_config=generation_config,
            )

            text = response.text.strip()

            if expect_json:
                text = _normalize_json_response(text)

            if ENABLE_CACHE and use_cache:
                _RESPONSE_CACHE[cache_key] = text
            return text, prompt_hash

        except (json.JSONDecodeError, ValueError) as e:
            last_json_error = e
            if attempt < retries:
                time.sleep(1.5 ** attempt)
                continue
            raise RuntimeError(
                f"Gemini returned invalid JSON after {retries+1} attempts: {last_json_error}"
            )

        except Exception as e:
            # Covers quota errors, network errors, etc.
            retry_delay = _extract_retry_delay_seconds(str(e))
            if retry_delay is not None and attempt < retries:
                time.sleep(retry_delay + 1)
                continue
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini API error: {e}")


# ── JSON parsing helper ────────────────────────────────────────────────────────

def parse_json_response(response_text: str) -> dict[str, Any]:
    """Parse and return JSON from LLM response text."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM response as JSON: {e}\nResponse: {response_text[:200]}")


def clear_llm_cache() -> None:
    """Clear the LLM response cache. Useful for testing."""
    global _RESPONSE_CACHE
    _RESPONSE_CACHE.clear()
    print("[INFO] LLM response cache cleared")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that LLMs sometimes add."""
    text = text.strip()
    if text.startswith("```"):
        lines  = text.splitlines()
        start  = 1 if lines[0].startswith("```") else 0
        end    = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text   = "\n".join(lines[start:end]).strip()
    return text


def _normalize_json_response(text: str) -> str:
    """Clean and repair model JSON output when needed."""
    text = _strip_json_fences(text)
    candidates = [text]

    extracted = _extract_json_candidate(text)
    if extracted and extracted != text:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    repaired = _repair_json_response(candidates[-1])
    json.loads(repaired)
    return repaired


def _extract_json_candidate(text: str) -> str | None:
    """Trim leading or trailing commentary around a JSON object or array."""
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [pos for pos in (object_start, array_start) if pos != -1]
    if not starts:
        return None

    start = min(starts)
    object_end = text.rfind("}")
    array_end = text.rfind("]")
    end = max(object_end, array_end)
    if end <= start:
        return None

    return text[start:end + 1].strip()


def _repair_json_response(text: str) -> str:
    """Ask the model to rewrite malformed JSON as valid JSON only."""
    repair_prompt = (
        "Fix the malformed JSON below.\n"
        "Return valid JSON only.\n"
        "Do not add commentary, markdown, or extra keys.\n\n"
        f"MALFORMED JSON:\n{text}"
    )
    repair_config = genai.types.GenerationConfig(
        temperature=0,
        max_output_tokens=2048,
        response_mime_type="application/json",
    )
    repaired = _generate_content(
        repair_prompt,
        generation_config=repair_config,
    ).text.strip()
    return _strip_json_fences(repaired)


def _extract_retry_delay_seconds(error_text: str) -> float | None:
    """Read provider-suggested retry delays from quota errors."""
    match = re.search(r"Please retry in\s+(\d+(?:\.\d+)?)s", error_text)
    if match:
        return float(match.group(1))
    return None


def _is_quota_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return "resourceexhausted" in lowered or "quota exceeded" in lowered


def _get_model(model_name: str) -> genai.GenerativeModel:
    if model_name not in _MODELS:
        _MODELS[model_name] = genai.GenerativeModel(model_name)
    return _MODELS[model_name]


def _generate_content(prompt: str, generation_config: Any):
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _get_model(model_name).generate_content,
                    prompt,
                    generation_config=generation_config,
                )
                return future.result(timeout=REQUEST_TIMEOUT_SECONDS)
        except FutureTimeoutError as exc:
            last_error = RuntimeError(
                f"Gemini request timed out after {REQUEST_TIMEOUT_SECONDS} seconds for model {model_name}"
            )
            continue
        except Exception as exc:
            last_error = exc
            if _is_quota_error(str(exc)):
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("No Gemini models are configured.")


def _validate_json(text: str) -> None:
    """Raise ValueError if text is not valid JSON."""
    json.loads(text)


def _hash_prompt(prompt: str) -> str:
    """Create a short hash of the prompt for audit logging."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]
