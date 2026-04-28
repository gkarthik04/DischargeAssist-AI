# DischargeAssist AI

DischargeAssist AI converts raw clinical notes into structured discharge-ready outputs with a FastAPI backend, a single-page HTML frontend, and a Gemini-powered LLM pipeline.

The project takes a raw note and produces:

- A structured SOAP note
- Extracted and categorized medications
- Follow-up instructions
- A patient-friendly summary
- Readability scoring and automatic simplification
- Confidence scoring based on source-grounded checks
- Optional multilingual translation
- SQLite-backed audit and summary history

## What This Project Contains

This repository is a self-contained prototype for clinical summarization workflows. It includes:

- A backend API in `main.py`
- A browser UI in `dischargeassist.html`
- Modular processing components for each generation step
- A local SQLite database for saved summaries and audit logs
- A standalone demo script for quick testing
- Lightweight evaluation utilities for ROUGE and optional BERTScore

## End-to-End Flow

The `/generate` pipeline follows this sequence:

1. Preprocess the raw clinical note
2. Generate a SOAP note
3. Extract and explain medications
4. Draft follow-up instructions
5. Generate a patient-friendly summary
6. Simplify readability if the reading grade is too high
7. Score confidence using source-grounded checks
8. Translate the patient summary if a non-English language is requested
9. Save the final summary and audit trail to SQLite

## Tech Stack

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic v2
- Google Gemini via `google-generativeai`
- SQLite
- Plain HTML, CSS, and JavaScript frontend

## Repository Layout

The current project is arranged as a flat Python repository rather than a package-based folder structure.

```text
GEN-AI2 PROJECT/
|-- main.py
|-- models.py
|-- preprocessor.py
|-- prompt_templates.py
|-- llm_engine.py
|-- database.py
|-- soap_formatter.py
|-- medication_explainer.py
|-- followup_drafter.py
|-- patient_simplifier.py
|-- readability.py
|-- confidence_scorer.py
|-- multilingual.py
|-- evaluation.py
|-- demo.py
|-- dischargeassist.html
|-- requirements.txt
|-- dischargeassist.db
|-- README.md
|-- *.log / *.err.log
|-- __pycache__/
`-- .venv/
```

## File-By-File Guide

### Core backend

- `main.py`
  FastAPI application entry point. Defines the API, wires together the pipeline, stores outputs in SQLite, and exposes retrieval, history, evaluation, and health endpoints.

- `models.py`
  Pydantic models for input payloads and all generated structures, including SOAP notes, medications, follow-up instructions, readability reports, confidence reports, and full discharge summaries.

- `database.py`
  SQLite persistence layer. Creates tables, stores audit rows, saves full discharge summaries, and supports summary retrieval, patient listing, and recent-history queries.

- `llm_engine.py`
  Shared Gemini client wrapper. Handles model selection, retries, JSON cleanup, timeout handling, optional caching, and prompt hashing for audit logging.

- `prompt_templates.py`
  Central library of prompt templates used by each module. Contains the system prompts, JSON schemas, readability simplification prompts, and multilingual translation prompts.

### Processing and generation modules

- `preprocessor.py`
  Cleans the raw note, removes noise, detects section headings, expands abbreviations, normalizes units, detects drug aliases, and returns a structured `PreprocessedNote`.

- `soap_formatter.py`
  Uses the LLM to convert the normalized note into a structured SOAP note.

- `medication_explainer.py`
  Extracts medications from the note, normalizes missing fields, validates whether purposes are explicitly supported, and classifies medications into home/discharge versus inpatient-plan groups.

- `followup_drafter.py`
  Generates structured follow-up instructions from the SOAP note and filters out inpatient-only monitoring language that should not appear as home guidance.

- `patient_simplifier.py`
  Creates the patient-friendly English summary and then sanitizes the wording to remove redundancy, unsupported phrasing, and low-quality fragments.

- `readability.py`
  Computes Flesch-Kincaid grade level and can re-prompt the LLM up to three times to simplify the patient summary until it meets the target grade or the retry limit.

- `confidence_scorer.py`
  Performs source-grounded confidence scoring. It checks whether high-salience details in the generated text can be matched back to the original note and returns a confidence score, warnings, and flagged terms.

- `multilingual.py`
  Translates the patient summary into supported regional languages when requested.

- `evaluation.py`
  Provides evaluation helpers for ROUGE-1, ROUGE-2, ROUGE-L, and optional BERTScore.

### Demo and interface

- `demo.py`
  Standalone command-line demo. Runs the pipeline directly on an embedded sample note without starting the API server.

- `dischargeassist.html`
  Browser UI for the app. Lets a user paste a clinical note, choose a summary language, submit to the backend, inspect results across tabs, and view saved history.

### Configuration and environment

- `requirements.txt`
  Python dependencies required for the project.

- `.venv/`
  Local virtual environment directory.

### Generated and runtime artifacts

- `dischargeassist.db`
  SQLite database created and used by the app. Stores saved summaries and the audit log.

- `api8000.log`, `uvicorn.log`, `frontend.log`, `*.err.log`
  Runtime logs produced while starting the backend or local static frontend server.

- `__pycache__/`
  Python bytecode cache files.

## Architecture Notes

### Input model

The main request body is defined by `ClinicalNoteInput` in `models.py`:

- `patient_id`: unique identifier for the run
- `raw_note`: unstructured clinical note text
- `language`: output language, default `en`
- `target_grade`: target patient-summary reading grade, default `6`

### Output model

The `/generate` endpoint returns a `DischargeSummary` object containing:

- Patient ID and generation timestamp
- SOAP note
- Medication explanation lists
- Follow-up instructions
- English patient summary
- Optional translated patient summary
- Readability report
- Confidence report

### Supported languages

`multilingual.py` and `prompt_templates.py` currently support:

- English: `en`
- Hindi: `hi`
- Telugu: `te`
- Tamil: `ta`
- Kannada: `kn`
- Marathi: `mr`
- Bengali: `bn`
- Gujarati: `gu`

Note: the frontend dropdown currently exposes `en`, `hi`, `te`, `ta`, and `kn`.

## Setup

### 1. Create a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
py -m pip install -r requirements.txt
```

### 3. Set environment variables

The backend expects a Gemini API key:

```powershell
$env:GOOGLE_API_KEY="your-api-key"
```

Optional environment variables used by `llm_engine.py`:

- `GEMINI_MODEL`
- `LLM_REQUEST_TIMEOUT_SECONDS`
- `LLM_TEMPERATURE`
- `ENABLE_LLM_CACHE`

## Running the Project

### Option 1. Run the API backend

```powershell
py -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Option 2. Open the frontend

You can either open `dischargeassist.html` directly or serve it locally.

Example local static server:

```powershell
py -m http.server 8080
```

Then open:

- `http://127.0.0.1:8080/dischargeassist.html`

The frontend expects the FastAPI backend to be reachable at `http://127.0.0.1:8000`.

### Option 3. Run the standalone demo

```powershell
py demo.py
```

This runs the pipeline with the built-in sample note and prints the generated outputs in the terminal.

## API Endpoints

### `GET /`

Redirects to Swagger docs.

### `POST /generate`

Runs the full generation pipeline.

Example request body:

```json
{
  "patient_id": "PT-001",
  "raw_note": "CC: SOB. HPI: 65yo male with HTN...",
  "language": "hi",
  "target_grade": 6
}
```

### `GET /summary/{patient_id}`

Returns the most recent saved discharge summary for a patient.

### `GET /audit/{patient_id}`

Returns the LLM audit log entries for that patient.

### `GET /patients`

Lists all patient IDs stored in the summaries table.

### `GET /history`

Returns the most recent saved summaries for the frontend history drawer.

Query parameter:

- `limit`: default `8`

### `POST /evaluate`

Evaluates generated text against a reference summary.

### `GET /health`

Returns backend health metadata.

## Frontend Features

The `dischargeassist.html` page provides:

- Clinical note input area
- Summary language selector
- Patient ID tracking
- Progress indicator for generation steps
- Tabs for SOAP note, medications, follow-up, patient summary, and quality
- Confidence and readability display
- History drawer powered by `/history`
- Copy and print actions for summary outputs

## Database Details

`database.py` creates two SQLite tables:

- `audit_log`
  Stores module-level outputs, prompt hashes, confidence, evaluation scores, and timestamps.

- `discharge_summaries`
  Stores the full final summary JSON for each patient run.

The database file is:

- `dischargeassist.db`

## Evaluation

`evaluation.py` includes:

- ROUGE-1 F1
- ROUGE-2 F1
- ROUGE-L F1
- Optional BERTScore

`demo.py` shows how evaluation can be run against a reference summary.

To enable BERTScore if needed:

```powershell
py -m pip install bert-score
```

## Important Notes

- The app depends on a valid Gemini API key before generation calls will work.
- The repository currently includes generated database and log files; those are runtime artifacts, not source code.
- The confidence scorer is rule-based and source-grounded. It is helpful for review, but it is not a substitute for clinical validation.
- The readability simplifier aims for a grade-level target, but outputs should still be manually reviewed for patient safety and wording quality.
- This is a prototype/research-style project, not a production medical system.

## Suggested Cleanup for Future Improvement

If you continue evolving the repo, a good next step would be to reorganize it into folders such as:

- `app/` for backend code
- `modules/` for generation components
- `frontend/` for HTML/CSS/JS
- `data/` for the SQLite database and sample files
- `logs/` for runtime logs

That is not required for the project to work today, but it would make the codebase easier to maintain as it grows.
