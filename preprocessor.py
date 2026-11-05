"""
preprocessor.py
───────────────
Cleans and structures raw clinical notes before they hit the LLM.
Steps:
  1. Noise removal        — strip irrelevant headers, duplicate whitespace
  2. Section tagging      — detect SOAP / clinical section headings
  3. Entity normalization — standardize drug names, units, abbreviations
"""

import re
from dataclasses import dataclass


# ── Known section aliases ──────────────────────────────────────────────────────

SECTION_MAP = {
    # Subjective synonyms
    r"(chief complaint|cc|presenting complaint|reason for visit)": "CHIEF_COMPLAINT",
    r"(history of present illness|hpi|history of presenting illness)":  "HPI",
    r"(review of systems|ros)": "REVIEW_OF_SYSTEMS",
    r"(past medical history|pmh|past history)": "PAST_MEDICAL_HISTORY",
    r"(medications?|current medications?|med list)": "MEDICATIONS",
    r"(allergies|allergy)": "ALLERGIES",
    r"(social history|sh|social hx)": "SOCIAL_HISTORY",
    r"(family history|fh|family hx)": "FAMILY_HISTORY",
    # Objective synonyms
    r"(physical exam|pe|examination|exam findings?)": "PHYSICAL_EXAM",
    r"(vital signs?|vitals?|vs)": "VITAL_SIGNS",
    r"(lab(?:oratory)? results?|labs?|investigations?)": "LAB_RESULTS",
    r"(imaging|radiology|x-?ray|ct scan|mri)": "IMAGING",
    # Assessment + Plan
    r"(assessment|diagnosis|impression|dx)": "ASSESSMENT",
    r"(plan|management|treatment|disposition)": "PLAN",
    r"(follow[- ]?up|follow up instructions?)": "FOLLOW_UP",
}

# ── Abbreviation expansion ─────────────────────────────────────────────────────

ABBREV_MAP = {
    r"\bbid\b": "twice daily",
    r"\btid\b": "three times daily",
    r"\bqid\b": "four times daily",
    r"\bqd\b":  "once daily",
    r"\bprn\b": "as needed",
    r"\bpo\b":  "by mouth",
    r"\biv\b":  "intravenous",
    r"\bim\b":  "intramuscular",
    r"\bsc\b":  "subcutaneous",
    r"\bSOB\b": "shortness of breath",
    r"\bCP\b":  "chest pain",
    r"\bHTN\b": "hypertension",
    r"\bDM\b":  "diabetes mellitus",
    r"\bDM2\b": "type 2 diabetes mellitus",
    r"\bCAD\b": "coronary artery disease",
    r"\bCHF\b": "congestive heart failure",
    r"\bCOPD\b":"chronic obstructive pulmonary disease",
    r"\bCVA\b": "cerebrovascular accident (stroke)",
    r"\bMI\b":  "myocardial infarction",
    r"\bECG\b": "electrocardiogram",
    r"\bHR\b":  "heart rate",
    r"\bBP\b":  "blood pressure",
    r"\bRR\b":  "respiratory rate",
    r"\bT\b(?=\s*[:=]\s*\d)": "temperature",  # T: 98.6
    r"\bO2\b":  "oxygen saturation",
    r"\bWBC\b": "white blood cell count",
    r"\bRBC\b": "red blood cell count",
    r"\bHgb\b": "hemoglobin",
    r"\bHct\b": "hematocrit",
    r"\bBUN\b": "blood urea nitrogen",
    r"\bCr\b":  "creatinine",
    r"\bNa\b":  "sodium",
    r"\bK\b(?=\s*[:=])": "potassium",
    r"\bGlu\b": "glucose",
}

# ── Drug name normalization ────────────────────────────────────────────────────

DRUG_ALIASES = {
    "tylenol":    "Acetaminophen (Tylenol)",
    "paracetamol":"Acetaminophen (Paracetamol)",
    "motrin":     "Ibuprofen (Motrin)",
    "advil":      "Ibuprofen (Advil)",
    "lasix":      "Furosemide (Lasix)",
    "glucophage": "Metformin (Glucophage)",
    "zocor":      "Simvastatin (Zocor)",
    "lipitor":    "Atorvastatin (Lipitor)",
    "norvasc":    "Amlodipine (Norvasc)",
    "zithromax":  "Azithromycin (Zithromax)",
    "augmentin":  "Amoxicillin-Clavulanate (Augmentin)",
}


@dataclass
class PreprocessedNote:
    raw_text:      str
    cleaned_text:  str
    sections:      dict[str, str]
    normalized_text: str
    detected_drugs:  list[str]
    word_count:    int


# ── Main preprocessor class ────────────────────────────────────────────────────

class ClinicalNotePreprocessor:

    def process(self, raw_text: str) -> PreprocessedNote:
        cleaned      = self._clean_noise(raw_text)
        sections     = self._tag_sections(cleaned)
        normalized   = self._normalize_abbreviations(cleaned)
        normalized   = self._normalize_units(normalized)
        drugs        = self._detect_drugs(normalized)
        normalized   = self._normalize_drug_names(normalized)

        return PreprocessedNote(
            raw_text        = raw_text,
            cleaned_text    = cleaned,
            sections        = sections,
            normalized_text = normalized,
            detected_drugs  = drugs,
            word_count      = len(normalized.split()),
        )

    # ── Step 1: Noise removal ────────────────────────────────────────────────

    def _clean_noise(self, text: str) -> str:
        # Remove page headers/footers (e.g., "Page 1 of 3", "CONFIDENTIAL")
        text = re.sub(r"page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"confidential|draft|for internal use only", "", text, flags=re.IGNORECASE)

        # Normalize line endings and excessive whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Remove special characters that add no meaning
        text = re.sub(r"[_=*]{3,}", "", text)

        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    # ── Step 2: Section tagging ──────────────────────────────────────────────

    def _tag_sections(self, text: str) -> dict[str, str]:
        sections: dict[str, str] = {}

        # Build pattern: "SectionName:" followed by content until next section
        for pattern, tag in SECTION_MAP.items():
            match = re.search(
                rf"(?i)(?:^|\n)\s*{pattern}\s*[:\-]?\s*(.+?)(?=\n\s*(?:{'|'.join(SECTION_MAP.keys())})\s*[:\-]|$)",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                sections[tag] = match.group(1).strip()

        # Fallback: if no sections detected, treat entire text as HPI
        if not sections:
            sections["HPI"] = text.strip()

        return sections

    # ── Step 3: Abbreviation expansion ──────────────────────────────────────

    def _normalize_abbreviations(self, text: str) -> str:
        for pattern, expansion in ABBREV_MAP.items():
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text

    # ── Step 4: Unit standardization ────────────────────────────────────────

    def _normalize_units(self, text: str) -> str:
        # Blood pressure: 120/80mmhg → 120/80 mmHg
        text = re.sub(r"(\d+/\d+)\s*mmhg", r"\1 mmHg", text, flags=re.IGNORECASE)
        # Temperature: 98.6F → 98.6°F, 37C → 37°C
        text = re.sub(r"(\d+\.?\d*)\s*°?\s*([CF])\b", r"\1°\2", text)
        # Dosage: 500mg → 500 mg
        text = re.sub(r"(\d+)\s*(mg|mcg|ml|mL|g|kg|mmol|mEq)\b", r"\1 \2", text, flags=re.IGNORECASE)
        return text

    # ── Step 5: Drug detection ───────────────────────────────────────────────

    def _detect_drugs(self, text: str) -> list[str]:
        found = []
        for alias in DRUG_ALIASES:
            if re.search(rf"\b{alias}\b", text, flags=re.IGNORECASE):
                found.append(alias.capitalize())
        return list(set(found))

    # ── Step 6: Drug name normalization ─────────────────────────────────────

    def _normalize_drug_names(self, text: str) -> str:
        for alias, canonical in DRUG_ALIASES.items():
            text = re.sub(rf"\b{alias}\b", canonical, text, flags=re.IGNORECASE)
        return text


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
    Chief Complaint: SOB and CP for 2 days
    HPI: 65yo male with HTN, DM2 presenting with worsening SOB.
         BP 160/95 mmhg, HR 98, O2 92%.
    Medications: Lasix 40mg qd, Glucophage 500mg bid, Lisinopril 10mg qd
    Assessment: CHF exacerbation, DM2 uncontrolled
    Plan: IV Lasix, cardiac monitoring, follow up with cardiology in 2 weeks
    """

    preprocessor = ClinicalNotePreprocessor()
    result = preprocessor.process(sample)
    print("Sections:", list(result.sections.keys()))
    print("Detected drugs:", result.detected_drugs)
    print("Normalized:\n", result.normalized_text[:300])
