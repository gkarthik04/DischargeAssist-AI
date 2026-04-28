"""
database.py
────────────
SQLite audit log for every LLM call made by DischargeAssist.
Stores: patient_id, module name, prompt hash, LLM response,
        confidence score, evaluation scores, timestamp.

Also stores full discharge summaries for retrieval.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("dischargeassist.db")


# ── Schema ─────────────────────────────────────────────────────────────────────

CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT    NOT NULL,
    module       TEXT    NOT NULL,
    prompt_hash  TEXT    NOT NULL,
    llm_response TEXT    NOT NULL,
    confidence   REAL,
    rouge_score  REAL,
    bert_score   REAL,
    created_at   TEXT    NOT NULL
);
"""

CREATE_SUMMARIES_TABLE = """
CREATE TABLE IF NOT EXISTS discharge_summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT    NOT NULL,
    summary_json TEXT    NOT NULL,
    created_at   TEXT    NOT NULL
);
"""


# ── Database manager ───────────────────────────────────────────────────────────

class AuditDB:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(CREATE_AUDIT_TABLE)
            conn.execute(CREATE_SUMMARIES_TABLE)
            conn.commit()

    # ── Audit log ──────────────────────────────────────────────────────────────

    def log_call(
        self,
        patient_id:   str,
        module:       str,
        prompt_hash:  str,
        llm_response: str,
        confidence:   float  = None,
        rouge_score:  float  = None,
        bert_score:   float  = None,
    ) -> int:
        """Insert an audit log entry. Returns the row id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO audit_log
                   (patient_id, module, prompt_hash, llm_response,
                    confidence, rouge_score, bert_score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patient_id,
                    module,
                    prompt_hash,
                    llm_response[:4000],  # truncate very long responses
                    confidence,
                    rouge_score,
                    bert_score,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_audit_log(self, patient_id: str) -> list[dict]:
        """Return all audit entries for a patient."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE patient_id = ? ORDER BY created_at DESC",
                (patient_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Discharge summary store ────────────────────────────────────────────────

    def save_summary(self, patient_id: str, summary_dict: dict) -> int:
        """Persist a full discharge summary as JSON."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO discharge_summaries (patient_id, summary_json, created_at)
                   VALUES (?, ?, ?)""",
                (
                    patient_id,
                    json.dumps(summary_dict, default=str),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_summary(self, patient_id: str) -> dict | None:
        """Retrieve the most recent discharge summary for a patient."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT summary_json FROM discharge_summaries
                   WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1""",
                (patient_id,),
            ).fetchone()
        return json.loads(row["summary_json"]) if row else None

    def list_patients(self) -> list[str]:
        """Return list of all patient IDs with summaries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT patient_id FROM discharge_summaries ORDER BY patient_id"
            ).fetchall()
        return [r["patient_id"] for r in rows]

    def list_recent_summaries(self, limit: int = 8) -> list[dict]:
        """Return the most recent saved summaries for history views."""
        safe_limit = max(1, min(int(limit), 50))
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT summary_json
                   FROM discharge_summaries
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (safe_limit,),
            ).fetchall()
        return [json.loads(row["summary_json"]) for row in rows]

    # ── Evaluation score update ────────────────────────────────────────────────

    def update_scores(self, log_id: int, rouge: float, bert: float):
        """Update ROUGE and BERTScore for an existing audit entry."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE audit_log SET rouge_score = ?, bert_score = ? WHERE id = ?",
                (rouge, bert, log_id),
            )
            conn.commit()
