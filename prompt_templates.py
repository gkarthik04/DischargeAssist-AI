"""
prompt_templates.py
───────────────────
Domain-specific prompt templates for each DischargeAssist module.
Each template:
  - Injects a strict JSON schema so the LLM outputs structured data
  - Uses clinical framing to reduce hallucination
  - Includes a role preamble for context grounding
"""

import json

# ── Shared system preamble ─────────────────────────────────────────────────────

CLINICAL_SYSTEM_PROMPT = """You are a board-certified clinical documentation specialist 
with 15 years of experience writing discharge summaries for teaching hospitals.

Rules you MUST follow:
1. Only extract information explicitly present in the provided clinical note.
2. If a field cannot be determined from the note, set it to "Not documented".
3. Never invent drug names, dosages, dates, or clinical findings.
4. Always respond with valid JSON only — no markdown fences, no extra text.
5. Use plain, professional medical English.
"""

# ── SOAP Note template ─────────────────────────────────────────────────────────

SOAP_SCHEMA = {
    "subjective":  "string — chief complaint and patient-reported symptoms",
    "objective":   "string — vital signs, physical exam, lab results, imaging",
    "assessment":  "string — primary diagnosis, differential diagnoses",
    "plan":        "string — treatment plan, medications, procedures ordered",
}

def soap_prompt(normalized_note: str) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for SOAP generation."""
    schema_str = json.dumps(SOAP_SCHEMA, indent=2)
    user = f"""Convert the following clinical note into a structured SOAP note.

CLINICAL NOTE:
{normalized_note}

OUTPUT FORMAT (valid JSON only):
{schema_str}
"""
    return CLINICAL_SYSTEM_PROMPT, user


# ── Medication explainer template ──────────────────────────────────────────────

MEDICATION_SCHEMA = {
    "medications": [
        {
            "name":      "string — medication name exactly as documented, without adding brand aliases",
            "dose":      "string — e.g., 500 mg",
            "frequency": "string — e.g., twice daily",
            "purpose":   "string — reason stated in the note, or 'Not documented'",
            "warnings":  "string — only warnings explicitly documented in the note, or null",
        }
    ]
}

def medication_prompt(normalized_note: str) -> tuple[str, str]:
    schema_str = json.dumps(MEDICATION_SCHEMA, indent=2)
    user = f"""Extract all medications from the clinical note and explain each one clearly.

CLINICAL NOTE:
{normalized_note}

OUTPUT FORMAT (valid JSON only):
{schema_str}

Important: Only list medications explicitly mentioned in the note.
Do not add brand names, side effects, interactions, or counseling points unless they are explicitly documented.
If the reason for a medication is not explicitly stated in the note, use "Not documented".
"""
    return CLINICAL_SYSTEM_PROMPT, user


# ── Follow-up instruction template ─────────────────────────────────────────────

FOLLOWUP_SCHEMA = {
    "appointment_date":  "string — date or timeframe, or null if not specified",
    "specialist":        "string — type of specialist or department, or null",
    "red_flag_symptoms": ["string — symptom that requires immediate medical attention"],
    "lifestyle_changes": ["string — diet, activity, or habit modification"],
    "monitoring":        ["string — vitals, lab values, or symptoms to track at home"],
}

def followup_prompt(soap_note_json: str) -> tuple[str, str]:
    schema_str = json.dumps(FOLLOWUP_SCHEMA, indent=2)
    user = f"""Based on the SOAP note below, generate clear follow-up instructions 
for both the clinician and the patient.

SOAP NOTE:
{soap_note_json}

OUTPUT FORMAT (valid JSON only):
{schema_str}

Rules:
- Only include follow-up details that are explicitly stated in the SOAP note.
- Do not convert presenting symptoms into return precautions unless the note explicitly gives return precautions.
- If no appointment date, specialist, lifestyle change, or home monitoring instruction is documented, return null or an empty list as appropriate.
"""
    return CLINICAL_SYSTEM_PROMPT, user


# ── Patient simplifier template ────────────────────────────────────────────────

def patient_summary_prompt(
    original_note: str,
    soap_note_json: str,
    medications_json: str,
    target_grade: int = 6,
) -> tuple[str, str]:
    system = """You are a health literacy specialist who explains medical information 
to patients and their families in simple, compassionate language.

Rules:
1. Write at approximately a US grade {grade} reading level.
2. Avoid all medical jargon. If a medical term is unavoidable, explain it in parentheses.
3. Use short sentences (under 20 words each).
4. Use only facts that are explicitly present in the source note, SOAP note, or medication list.
5. Do not add mechanisms, side effects, reasons for medicines, return precautions, or follow-up details unless they are explicitly stated.
6. Do not add brand names or expand medication names beyond what appears in the source materials.
7. Be calm and clear, but do not add generic reassurance or advice that is not documented.
8. Do not explain what a diagnosis "means" unless the explanation is explicitly documented.
9. Do not say "tests showed", "this means", "through a vein", "in your arm", "through a tube in your arm", "for more care", or similar explanatory phrases unless those exact facts are in the source.
10. Respond with a plain paragraph of text only — no JSON, no bullet points.
""".format(grade=target_grade)

    user = f"""Write a patient-friendly explanation of this hospital visit.
Tell the patient: what was wrong, what was done, what medications they are taking and why,
and what they need to do after they leave.

If a detail is not documented, leave it out rather than guessing.

ORIGINAL NOTE:
{original_note}

SOAP NOTE:
{soap_note_json}

MEDICATIONS:
{medications_json}

Write in simple English at a grade {target_grade} reading level.
"""
    return system, user


def patient_summary_prompt_v2(
    original_note: str,
    soap_note_json: str,
    medications_json: str,
    target_grade: int = 6,
) -> tuple[str, str]:
    system = """You are a health literacy specialist who explains medical information
to patients and their families in simple, compassionate language.

Rules:
1. Write at approximately a US grade {grade} reading level.
2. Avoid all medical jargon. If a medical term is unavoidable, explain it in parentheses.
3. Use short sentences (under 20 words each).
4. Use only facts that are explicitly present in the source note, SOAP note, or medication list.
5. Do not add mechanisms, side effects, reasons for medicines, return precautions, or follow-up details unless they are explicitly stated.
6. Do not add brand names or expand medication names beyond what appears in the source materials.
7. Be calm and clear, but do not add generic reassurance or advice that is not documented.
8. Do not explain what a diagnosis "means" unless the explanation is explicitly documented.
9. Do not say "tests showed", "this means", "through a vein", "for more care", or similar explanatory phrases unless those exact facts are in the source.
10. If a medication appears only as an inpatient treatment or plan item, describe it only as something given in the hospital.
11. Do not present inpatient-only medications as home, discharge, or ongoing medicines unless the source explicitly says that.
12. Do not use bullet markers, asterisks, or list formatting in the response.
13. Do not repeat the same diagnosis, medication, or instruction in a separate trailing sentence or list.
14. Do not append a summary list of diagnoses or medications after the paragraph.
15. Respond with a plain paragraph of text only - no JSON, no bullet points.
""".format(grade=target_grade)

    user = f"""Write a patient-friendly explanation of this hospital visit.
Tell the patient: what was wrong, what was done in the hospital, which medicines were home medicines,
and which medicines were only given during this hospital stay if that is explicitly documented.

If a detail is not documented, leave it out rather than guessing.

ORIGINAL NOTE:
{original_note}

SOAP NOTE:
{soap_note_json}

MEDICATIONS:
{medications_json}

Write in simple English at a grade {target_grade} reading level.
"""
    return system, user


# ── Readability refinement template ───────────────────────────────────────────

def simplify_further_prompt(text: str, current_grade: float, target_grade: int) -> tuple[str, str]:
    system = """You are a health literacy expert. Simplify the following medical text 
so patients can easily understand it. Use shorter words and sentences."""

    user = f"""This text is written at a grade {current_grade:.1f} reading level.
Rewrite it at a grade {target_grade} reading level or lower.
Keep all the medical facts correct. Do not add new information.
Do not add repeated sentences, list fragments, medication lists, diagnosis lists, or extra trailing instructions.
Do not add body-part details for IV access such as "in your arm" unless that exact detail is in the source.
Return one clean paragraph only.

TEXT TO SIMPLIFY:
{text}

Respond with only the rewritten text, no explanation.
"""
    return system, user


# ── Translation template ───────────────────────────────────────────────────────

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
}

def translation_prompt(english_summary: str, target_lang_code: str) -> tuple[str, str]:
    lang_name = LANGUAGE_NAMES.get(target_lang_code, target_lang_code)
    system = f"""You are a certified medical translator specializing in {lang_name}.
Translate patient health information accurately and compassionately.
Maintain the simple reading level of the original.
Do not translate drug names — keep them in English."""

    user = f"""Translate the following patient discharge summary into {lang_name}.

ENGLISH SUMMARY:
{english_summary}

Provide only the {lang_name} translation. Do not include the English original.
"""
    return system, user


# ── Hallucination check template ───────────────────────────────────────────────

def hallucination_check_prompt(
    original_note: str,
    generated_output: str,
    output_type: str,
) -> tuple[str, str]:
    system = """You are an EXTREMELY CRITICAL clinical quality reviewer. Your job is to find problems 
and quality gaps — not to rubber-stamp content. Be SKEPTICAL, CONSERVATIVE, and TOUGH.

Identify real issues when they exist, but do not invent problems just to be critical.
If the generated content is faithful to the source, it is acceptable to return few or no issues."""

    user = f"""Compare the GENERATED {output_type.upper()} against the ORIGINAL NOTE.

ORIGINAL CLINICAL NOTE:
{original_note}

GENERATED {output_type.upper()}:
{generated_output}

YOUR EVALUATION:

1. HALLUCINATION CHECK: Information NOT explicitly supported
   - Flag any drug names, dosages, dates, lab values NOT in original
   - Flag interpretations or explanations not documented
   - Wording changes ARE OK if they preserve facts

2. COMPLETENESS CHECK: What's MISSING?
   - Are critical clinical findings omitted?
   - Are important medications missing?
   - Is follow-up info incomplete?

3. CLARITY & SPECIFICITY CHECK: Is this just generic boilerplate?
   - Does it say "patient has X condition" without detail?
   - Are specific vital signs, lab values, findings omitted?
   - Is the clinical reasoning clear and specific?

4. STRUCTURAL CHECK: Is it well-organized?
   - Are sections logical and complete?
   - Is there redundancy or confusion?

Respond with valid JSON only:
{{
  "hallucinated_terms": ["terms not in original"],
  "warnings": ["problematic sentences or notable risks"],
  "missing_info": ["important info missing from summary"],
  "quality_issues": ["generic, vague, unclear, or poorly structured sections"],
  "overall_confidence": confidence score (0.0 to 1.0)
}}

SCORING RULES:
- Start around 0.75 for a faithful, useful summary.
- Lower the score for unsupported facts, major omissions, or misleading wording.
- Use 0.85 to 0.95 for strong outputs with only minor issues.
- Use below 0.60 only when there are serious safety, accuracy, or completeness concerns.
- It is acceptable for warnings, missing_info, and quality_issues to be empty when no meaningful issues are present."""

    return system, user
