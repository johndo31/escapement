"""
Local storage layer for Escapement traces using SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from escapement.config import get_config
from escapement.tracer import Trace, TraceMetadata, TraceStep


class TraceStorage:
    """SQLite-based storage for traces."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_config().get_db_path()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    name TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    created_at TEXT NOT NULL,
                    ended_at TEXT,
                    total_steps INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost_usd REAL DEFAULT 0.0,
                    tags TEXT,  -- JSON array
                    environment TEXT,  -- JSON object
                    agent_info TEXT  -- JSON object
                );

                CREATE TABLE IF NOT EXISTS trace_steps (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    step_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    duration_ms REAL,
                    request_hash TEXT,
                    data TEXT NOT NULL,  -- JSON blob of full step
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id),
                    UNIQUE(trace_id, step_number)
                );

                CREATE INDEX IF NOT EXISTS idx_trace_steps_trace_id
                    ON trace_steps(trace_id);
                CREATE INDEX IF NOT EXISTS idx_trace_steps_request_hash
                    ON trace_steps(request_hash);
                CREATE INDEX IF NOT EXISTS idx_traces_status
                    ON traces(status);
                CREATE INDEX IF NOT EXISTS idx_traces_created_at
                    ON traces(created_at);
            """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_trace(self, trace: Trace) -> None:
        """Save or update a trace."""
        with self._connect() as conn:
            # Upsert trace metadata
            conn.execute(
                """
                INSERT INTO traces (
                    trace_id, name, status, created_at, ended_at,
                    total_steps, total_tokens, total_cost_usd,
                    tags, environment, agent_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    name = excluded.name,
                    status = excluded.status,
                    ended_at = excluded.ended_at,
                    total_steps = excluded.total_steps,
                    total_tokens = excluded.total_tokens,
                    total_cost_usd = excluded.total_cost_usd,
                    tags = excluded.tags,
                    environment = excluded.environment,
                    agent_info = excluded.agent_info
                """,
                (
                    trace.metadata.trace_id,
                    trace.metadata.name,
                    trace.metadata.status,
                    trace.metadata.created_at.isoformat(),
                    trace.metadata.ended_at.isoformat() if trace.metadata.ended_at else None,
                    trace.metadata.total_steps,
                    trace.metadata.total_tokens,
                    trace.metadata.total_cost_usd,
                    json.dumps(trace.metadata.tags),
                    json.dumps(trace.metadata.environment),
                    json.dumps(trace.metadata.agent_info),
                ),
            )

    def save_step(self, trace_id: str, step: TraceStep) -> None:
        """Save a single step to the database."""
        with self._connect() as conn:
            step_data = step.model_dump(mode="json")
            conn.execute(
                """
                INSERT INTO trace_steps (
                    id, trace_id, step_number, step_type, timestamp,
                    duration_ms, request_hash, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id, step_number) DO UPDATE SET
                    data = excluded.data,
                    duration_ms = excluded.duration_ms
                """,
                (
                    step.id,
                    trace_id,
                    step.step_number,
                    step.step_type.value,
                    step.timestamp.isoformat(),
                    step.duration_ms,
                    step.request_hash,
                    json.dumps(step_data),
                ),
            )

    def load_trace(self, trace_id: str) -> Trace | None:
        """Load a complete trace by ID."""
        with self._connect() as conn:
            # Load metadata
            row = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()

            if not row:
                return None

            metadata = TraceMetadata(
                trace_id=row["trace_id"],
                name=row["name"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
                ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                total_steps=row["total_steps"],
                total_tokens=row["total_tokens"],
                total_cost_usd=row["total_cost_usd"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                environment=json.loads(row["environment"]) if row["environment"] else {},
                agent_info=json.loads(row["agent_info"]) if row["agent_info"] else {},
            )

            # Load steps
            step_rows = conn.execute(
                """
                SELECT data FROM trace_steps
                WHERE trace_id = ?
                ORDER BY step_number
                """,
                (trace_id,),
            ).fetchall()

            steps = [TraceStep.model_validate(json.loads(row["data"])) for row in step_rows]

            return Trace(metadata=metadata, steps=steps)

    def list_traces(
        self,
        limit: int = 50,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[TraceMetadata]:
        """List trace metadata with optional filtering."""
        with self._connect() as conn:
            query = "SELECT * FROM traces"
            params: list[Any] = []
            conditions = []

            if status:
                conditions.append("status = ?")
                params.append(status)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

            traces = []
            for row in rows:
                meta = TraceMetadata(
                    trace_id=row["trace_id"],
                    name=row["name"],
                    status=row["status"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                    total_steps=row["total_steps"],
                    total_tokens=row["total_tokens"],
                    total_cost_usd=row["total_cost_usd"],
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                )

                # Filter by tags if specified
                if tags:
                    if not all(t in meta.tags for t in tags):
                        continue

                traces.append(meta)

            return traces

    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace and all its steps."""
        with self._connect() as conn:
            conn.execute("DELETE FROM trace_steps WHERE trace_id = ?", (trace_id,))
            result = conn.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
            return result.rowcount > 0

    def get_step_by_hash(self, request_hash: str, trace_id: str | None = None) -> TraceStep | None:
        """Find a step by its request hash, optionally within a specific trace."""
        with self._connect() as conn:
            if trace_id:
                row = conn.execute(
                    """
                    SELECT data FROM trace_steps
                    WHERE request_hash = ? AND trace_id = ?
                    LIMIT 1
                    """,
                    (request_hash, trace_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT data FROM trace_steps
                    WHERE request_hash = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (request_hash,),
                ).fetchone()

            if row:
                return TraceStep.model_validate(json.loads(row["data"]))
            return None

    def export_trace(self, trace_id: str, output_path: Path) -> bool:
        """Export a trace to a JSON file."""
        trace = self.load_trace(trace_id)
        if not trace:
            return False

        output_path.write_text(json.dumps(trace.to_dict(), indent=2))
        return True

    def import_trace(self, input_path: Path) -> Trace | None:
        """Import a trace from a JSON file."""
        try:
            data = json.loads(input_path.read_text())
            trace = Trace.from_dict(data)

            # Save to database
            self.save_trace(trace)
            for step in trace.steps:
                self.save_step(trace.trace_id, step)

            return trace
        except (json.JSONDecodeError, ValueError):
            return None


# Singleton storage instance
_storage: TraceStorage | None = None


def get_storage() -> TraceStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = TraceStorage()
    return _storage
