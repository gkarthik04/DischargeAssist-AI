"""
demo.py
────────
Standalone demo script for DischargeAssist AI.
Run directly with: python demo.py

Does NOT require the FastAPI server to be running.
Calls the pipeline directly and prints a formatted report.
"""

import json
import sys
from datetime import datetime

from preprocessor import ClinicalNotePreprocessor
from soap_formatter import generate_soap_note
from medication_explainer import explain_medications
from followup_drafter import draft_followup
from patient_simplifier import generate_patient_summary
from readability import adapt_readability
from confidence_scorer import score_confidence
from multilingual import translate_summary
from evaluation import evaluate_summary, print_scores


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ── Sample clinical note (MTS-Dialog style) ────────────────────────────────────

SAMPLE_NOTE = """
Chief Complaint: Worsening shortness of breath and leg swelling for 3 days.

History of Present Illness:
Mr. Ramesh Kumar, a 68-year-old male with known HTN, DM2, and CAD, presents
to the emergency department with a 3-day history of worsening SOB and bilateral
leg swelling. He reports orthopnea requiring 3 pillows to sleep and a 5 kg weight
gain over the past week. He denies any chest pain, fever, or productive cough.

Vital Signs:
BP 168/96 mmhg, HR 102, RR 22, T 37.1°C, O2 88% on room air.

Physical Exam:
Bilateral crackles at lung bases. 3+ pitting edema to the knees bilaterally.
Elevated JVP. S3 gallop on auscultation.

Lab Results:
BNP 1850 pg/mL (elevated), Cr 1.6 mg/dL (baseline 1.1), Na 132 mEq/L.
Hgb 10.2 g/dL.

Imaging:
Chest X-ray: cardiomegaly with bilateral pulmonary edema.
ECG: sinus tachycardia, no ST changes.

Past Medical History: HTN, DM2, CAD, previous MI in 2019.

Medications on Admission:
Lasix 40 mg qd, Glucophage 500 mg bid, Lisinopril 10 mg qd,
Atorvastatin 40 mg qd, Aspirin 81 mg qd.

Assessment:
Acute decompensated CHF exacerbation secondary to medication non-compliance.
Mild acute kidney injury secondary to poor forward flow.
DM2 — glucose 210 mg/dL, management continued.
Anemia — likely anaemia of chronic disease.

Plan:
- IV Furosemide (Lasix) 80 mg bolus, then 20 mg/hr infusion
- Strict fluid restriction 1.5 L/day, low sodium diet
- Daily weights, strict I&O monitoring
- Repeat BMP in 6 hours
- Cardiology consult
- Follow up with cardiologist in 2 weeks
- Restart home Lisinopril once Cr improves
"""

# ── Reference summary (from MTS-Dialog for evaluation) ────────────────────────

REFERENCE_SUMMARY = """
Patient presented with acute decompensated heart failure with worsening shortness 
of breath and bilateral leg edema. Treatment included intravenous diuretics and 
fluid restriction. Patient was counseled on medication compliance and low sodium diet. 
Follow-up with cardiology scheduled in two weeks.
"""


def run_demo():
    print("\n" + "═" * 60)
    print("  DischargeAssist AI — Demo Run")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("═" * 60)

    pid          = "PT-DEMO-001"
    target_grade = 6
    language     = "hi"   # Hindi output

    preprocessor = ClinicalNotePreprocessor()

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    print("\n[1/7] Preprocessing clinical note...")
    preprocessed = preprocessor.process(SAMPLE_NOTE)
    print(f"  ✓ Sections detected: {list(preprocessed.sections.keys())}")
    print(f"  ✓ Drugs identified: {preprocessed.detected_drugs}")
    print(f"  ✓ Word count: {preprocessed.word_count}")

    # ── Step 2: SOAP Note ─────────────────────────────────────────────────────
    print("\n[2/7] Generating SOAP note...")
    soap_note, _ = generate_soap_note(preprocessed.normalized_text)
    print("  ✓ SOAP note generated")
    print(f"\n  SUBJECTIVE:\n  {soap_note.subjective[:120]}...")
    print(f"\n  ASSESSMENT:\n  {soap_note.assessment[:120]}...")

    # ── Step 3: Medications ───────────────────────────────────────────────────
    print("\n[3/7] Extracting and explaining medications...")
    medications, _ = explain_medications(preprocessed.normalized_text)
    print(f"  ✓ {len(medications.medications)} medication(s) found")
    for m in medications.medications:
        print(f"     • {m.name} {m.dose} {m.frequency} — {m.purpose}")

    # ── Step 4: Follow-up ─────────────────────────────────────────────────────
    print("\n[4/7] Drafting follow-up instructions...")
    follow_up, _ = draft_followup(soap_note)
    print(f"  ✓ Appointment: {follow_up.appointment_date}")
    print(f"  ✓ Specialist:  {follow_up.specialist}")
    print(f"  ✓ Red flags:   {follow_up.red_flag_symptoms[:2]}")

    # ── Step 5: Patient summary ───────────────────────────────────────────────
    print("\n[5/7] Generating patient-friendly summary...")
    raw_summary, _ = generate_patient_summary(
        preprocessed.normalized_text,
        soap_note,
        medications,
        target_grade,
    )
    print(f"  ✓ Initial summary ({len(raw_summary.split())} words)")

    # ── Step 6: Readability adaptation ───────────────────────────────────────
    print(f"\n[6/7] Adapting readability to grade {target_grade}...")
    final_summary, readability = adapt_readability(raw_summary, target_grade)
    print(f"  ✓ Original grade: {readability.original_grade}")
    print(f"  ✓ Final grade:    {readability.final_grade}")
    print(f"  ✓ Passes target:  {readability.passes_target}")
    print(f"  ✓ Passes made:    {readability.refinement_passes}")
    print(f"\n  PATIENT SUMMARY (English):\n  {final_summary[:300]}...")

    # ── Step 7: Confidence scoring ────────────────────────────────────────────
    print("\n[7/7] Running confidence scoring + hallucination check...")
    confidence = score_confidence(
        preprocessed.normalized_text,
        soap_note.model_dump_json() + " " + final_summary,
        output_type="discharge summary",
    )
    print(f"  ✓ Overall confidence: {confidence.overall_score:.2%}")
    print(f"  ✓ Safe to use:        {confidence.is_safe_to_use}")
    if confidence.hallucinated_terms:
        print(f"  ⚠ Flagged terms:     {confidence.hallucinated_terms}")
    if confidence.warnings:
        print(f"  ⚠ Warnings:          {confidence.warnings[:1]}")

    # ── Multilingual translation ──────────────────────────────────────────────
    if language != "en":
        print(f"\n[+] Translating to Hindi...")
        hindi_summary = translate_summary(final_summary, "hi")
        if hindi_summary:
            print(f"\n  PATIENT SUMMARY (Hindi):\n  {hindi_summary[:200]}...")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n── Evaluation Against Reference Summary ─────────────────────────")
    eval_text = f"{soap_note.subjective} {soap_note.assessment} {final_summary}"
    scores = evaluate_summary(eval_text, REFERENCE_SUMMARY)
    print_scores(scores, "(Full Summary vs Reference)")

    print("\n✅ Demo complete.\n")


if __name__ == "__main__":
    run_demo()
