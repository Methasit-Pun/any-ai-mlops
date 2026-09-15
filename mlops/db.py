"""Read-only access to the any-ai-backend Postgres database.

This module never writes to the database — any-ai-backend (Prisma) remains
the single source of truth and owner of the schema.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras

from .config import Config


@contextmanager
def get_connection():
    conn = psycopg2.connect(Config.DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def _dict_cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def get_active_agent_configs(conn) -> list[dict[str, Any]]:
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT id, name, type, model, prompt, temperature, voice,
                   language, max_duration, updated_at
            FROM agent_configs
            WHERE is_active = true
            ORDER BY id
            """
        )
        return cur.fetchall()


def get_conversation_logs(conn, config_id: str, since: datetime) -> list[dict[str, Any]]:
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT call_id, start_time, end_time, duration, metadata
            FROM conversation_logs
            WHERE config_id = %s AND start_time >= %s
            ORDER BY start_time
            """,
            (config_id, since),
        )
        return cur.fetchall()


def get_call_logs_by_ids(conn, call_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not call_ids:
        return {}
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT call_id, status, duration, cost, transcription
            FROM call_logs
            WHERE call_id = ANY(%s)
            """,
            (call_ids,),
        )
        return {row["call_id"]: row for row in cur.fetchall()}


def get_call_summaries_by_ids(conn, call_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not call_ids:
        return {}
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT call_id, sentiment
            FROM call_summaries
            WHERE call_id = ANY(%s)
            """,
            (call_ids,),
        )
        return {row["call_id"]: row for row in cur.fetchall()}


def get_appointment_count(conn, agent_config_id: str, since: datetime) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM appointments
            WHERE agent_config_id = %s AND source = 'ai_call' AND created_at >= %s
            """,
            (agent_config_id, since),
        )
        return cur.fetchone()[0]


def get_agent_config_by_id(conn, agent_id: str) -> dict[str, Any] | None:
    with _dict_cursor(conn) as cur:
        cur.execute(
            "SELECT id, name FROM agent_configs WHERE id = %s",
            (agent_id,),
        )
        return cur.fetchone()


def get_recent_transcriptions(conn, agent_id: str, limit: int) -> list[dict[str, Any]]:
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT call_id, transcription
            FROM call_logs
            WHERE agent_id = %s AND transcription IS NOT NULL AND transcription != ''
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (agent_id, limit),
        )
        return cur.fetchall()
