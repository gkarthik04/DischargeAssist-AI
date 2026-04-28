from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Input ──────────────────────────────────────────────────────────────────────

class ClinicalNoteInput(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    raw_note: str  = Field(..., description="Raw clinical note text")
    language: str  = Field(default="en", description="Target output language: en | hi | te")
    target_grade: int = Field(default=6, description="Target Flesch-Kincaid grade level for patient summary")


# ── SOAP Structure ─────────────────────────────────────────────────────────────

class SOAPNote(BaseModel):
    subjective:  str = Field(..., description="Chief complaint and patient-reported symptoms")
    objective:   str = Field(..., description="Vital signs, exam findings, lab results")
    assessment:  str = Field(..., description="Diagnosis and clinical impression")
    plan:        str = Field(..., description="Treatment, medications, follow-up plan")


# ── Medication ─────────────────────────────────────────────────────────────────

class MedicationEntry(BaseModel):
    name:      str
    dose:      str
    frequency: str
    purpose:   str
    warnings:  Optional[str] = None
    category:  Optional[str] = None


class MedicationExplanation(BaseModel):
    medications: list[MedicationEntry]
    home_medications: list[MedicationEntry] = Field(default_factory=list)
    inpatient_medications: list[MedicationEntry] = Field(default_factory=list)


# ── Follow-up ──────────────────────────────────────────────────────────────────

class FollowUpInstruction(BaseModel):
    appointment_date:  Optional[str]
    specialist:        Optional[str]
    red_flag_symptoms: list[str]
    lifestyle_changes: list[str]
    monitoring:        list[str]


# ── Confidence / Novelty ───────────────────────────────────────────────────────

class ConfidenceReport(BaseModel):
    overall_score:    float = Field(..., ge=0.0, le=1.0)
    hallucinated_terms: list[str]
    warnings:         list[str]
    is_safe_to_use:   bool


class ReadabilityReport(BaseModel):
    original_grade:    float
    final_grade:       float
    passes_target:     bool
    refinement_passes: int


# ── Full Discharge Summary ─────────────────────────────────────────────────────

class DischargeSummary(BaseModel):
    patient_id:           str
    generated_at:         datetime
    soap_note:            SOAPNote
    medications:          MedicationExplanation
    follow_up:            FollowUpInstruction
    patient_summary_en:   str
    patient_summary_lang: Optional[str]
    language:             str
    confidence:           ConfidenceReport
    readability:          ReadabilityReport


# ── Audit Log Entry ────────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    patient_id:   str
    prompt_hash:  str
    module:       str
    llm_response: str
    confidence:   float
    rouge_score:  Optional[float] = None
    created_at:   datetime = Field(default_factory=datetime.utcnow)
