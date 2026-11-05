"""
main.py
────────
DischargeAssist AI — FastAPI Application

Endpoints:
  POST /generate          → full pipeline: preprocess → SOAP → meds → follow-up
                            → patient summary → readability → confidence → translate
  GET  /summary/{id}      → retrieve a stored discharge summary
  GET  /audit/{id}        → retrieve audit log for a patient
  GET  /patients          → list all patients in the database
  GET  /health            → service health check

Run with:
  uvicorn main:app --reload
"""

import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse

from models import ClinicalNoteInput, DischargeSummary
from preprocessor import ClinicalNotePreprocessor
from soap_formatter import generate_soap_note
from medication_explainer import explain_medications
from followup_drafter import draft_followup
from patient_simplifier import generate_patient_summary, sanitize_patient_summary
from readability import adapt_readability
from confidence_scorer import score_confidence
from multilingual import translate_summary
from database import AuditDB
from evaluation import evaluate_summary, print_scores

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DischargeAssist AI",
    description = "Clinical Summary Architect — Converts raw clinical notes into structured discharge summaries",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

preprocessor = ClinicalNotePreprocessor()
db           = AuditDB()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc)},
    )


# ── Main pipeline endpoint ─────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.post("/generate", response_model=DischargeSummary, tags=["Pipeline"])
async def generate_discharge_summary(note_input: ClinicalNoteInput):
    """
    Full DischargeAssist pipeline.

    Takes a raw clinical note and returns a complete discharge summary
    including SOAP note, medication explanations, follow-up instructions,
    a patient-friendly summary (with readability adaptation), confidence
    scoring, and optional multilingual translation.
    """
    pid = note_input.patient_id

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    preprocessed = preprocessor.process(note_input.raw_note)

    # ── Step 2: Generate SOAP Note ────────────────────────────────────────────
    soap_note, soap_hash = generate_soap_note(preprocessed.normalized_text)
    db.log_call(pid, "soap_formatter", soap_hash, soap_note.model_dump_json(), confidence=None)

    # ── Step 3: Explain Medications ───────────────────────────────────────────
    medications, med_hash = explain_medications(preprocessed.normalized_text)
    db.log_call(pid, "medication_explainer", med_hash, medications.model_dump_json())

    # ── Step 4: Draft Follow-up Instructions ──────────────────────────────────
    follow_up, fu_hash = draft_followup(soap_note)
    db.log_call(pid, "followup_drafter", fu_hash, follow_up.model_dump_json())

    # ── Step 5: Generate Patient Summary ──────────────────────────────────────
    patient_summary_raw, ps_hash = generate_patient_summary(
        preprocessed.normalized_text,
        soap_note,
        medications,
        note_input.target_grade,
    )
    db.log_call(pid, "patient_simplifier", ps_hash, patient_summary_raw)

    # ── Step 6 [NOVELTY]: Adaptive Readability ────────────────────────────────
    patient_summary_en, readability = adapt_readability(
        patient_summary_raw, note_input.target_grade
    )
    patient_summary_en = sanitize_patient_summary(patient_summary_en)

    # ── Step 7 [NOVELTY]: Confidence Scoring ─────────────────────────────────
    patient_summary_lang = None

    def _format_medication_entry(med) -> str:
        return "; ".join(
            part for part in [
                med.name,
                med.dose,
                med.frequency,
                med.purpose,
                med.warnings or "",
            ] if part and part not in {"Not documented", "Not specified"}
        )

    home_medication_entries = [
        _format_medication_entry(med) for med in medications.home_medications
    ]
    inpatient_medication_entries = [
        _format_medication_entry(med) for med in medications.inpatient_medications
    ]
    home_medication_text = " | ".join(entry for entry in home_medication_entries if entry)
    inpatient_medication_text = " | ".join(entry for entry in inpatient_medication_entries if entry)
    followup_text = " ".join(
        part for part in [
            follow_up.appointment_date or "",
            follow_up.specialist or "",
            " ".join(follow_up.red_flag_symptoms),
            " ".join(follow_up.lifestyle_changes),
            " ".join(follow_up.monitoring),
        ] if part and part != "Not documented"
    )
    confidence_review_text = "\n".join(
        part for part in [
            patient_summary_en,
            f"Home/discharge medications: {home_medication_text}" if home_medication_text else "",
            followup_text,
        ] if part
    )
    confidence = score_confidence(
        original_note  = preprocessed.normalized_text,
        generated_text = confidence_review_text,
        output_type    = "patient summary",
    )

    # Log confidence against SOAP entry
    db.log_call(
        pid, "confidence_scorer", soap_hash,
        json.dumps(confidence.model_dump()),
        confidence = confidence.overall_score,
    )

    # ── Step 8 [NOVELTY]: Multilingual Translation ────────────────────────────
    if note_input.language != "en":
        patient_summary_lang = translate_summary(patient_summary_en, note_input.language)
        db.log_call(pid, f"translator_{note_input.language}", ps_hash, patient_summary_lang or "")

    # ── Assemble final summary ────────────────────────────────────────────────
    summary = DischargeSummary(
        patient_id           = pid,
        generated_at         = datetime.utcnow(),
        soap_note            = soap_note,
        medications          = medications,
        follow_up            = follow_up,
        patient_summary_en   = patient_summary_en,
        patient_summary_lang = patient_summary_lang,
        language             = note_input.language,
        confidence           = confidence,
        readability          = readability,
    )

    # ── Persist to database ───────────────────────────────────────────────────
    db.save_summary(pid, summary.model_dump())

    return summary


# ── Retrieval endpoints ────────────────────────────────────────────────────────

@app.get("/summary/{patient_id}", tags=["Retrieval"])
async def get_summary(patient_id: str):
    """Retrieve the most recent discharge summary for a patient."""
    result = db.get_summary(patient_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No summary found for patient {patient_id}")
    return result


@app.get("/audit/{patient_id}", tags=["Audit"])
async def get_audit_log(patient_id: str):
    """Retrieve the full audit log (all LLM calls) for a patient."""
    entries = db.get_audit_log(patient_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"No audit entries for patient {patient_id}")
    return {"patient_id": patient_id, "entries": entries, "total": len(entries)}


@app.get("/patients", tags=["Retrieval"])
async def list_patients():
    """List all patient IDs that have discharge summaries."""
    return {"patients": db.list_patients()}


@app.get("/history", tags=["Retrieval"])
async def list_recent_history(limit: int = 8):
    """Return the most recent saved discharge summaries for the history drawer."""
    return {"items": db.list_recent_summaries(limit)}


# ── Evaluation endpoint (for research/demo use) ────────────────────────────────

@app.post("/evaluate", tags=["Evaluation"])
async def evaluate(generated: str, reference: str):
    """
    Evaluate a generated summary against a reference summary.
    Use with MTS-Dialog ground-truth summaries.
    """
    scores = evaluate_summary(generated, reference)
    return {
        "rouge1_f":   scores.rouge1_f,
        "rouge2_f":   scores.rouge2_f,
        "rougeL_f":   scores.rougeL_f,
        "bert_score": scores.bert_score,
    }


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "DischargeAssist AI", "version": "1.0.0"}
