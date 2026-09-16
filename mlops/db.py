"""Read-only access to the any-ai-backend Postgres database.

This module never writes to the database — any-ai-backend (Prisma) remains
the single source of truth and owner of the schema.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import psycopg2
import psycopg2.extras

from .config import Config


def _libpq_dsn(database_url: str) -> str:
    """Strip query params libpq doesn't understand (e.g. Prisma's `pgbouncer=true`)."""
    parsed = urlparse(database_url)
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query) if k != "pgbouncer"])
    return urlunparse(parsed._replace(query=query))


@contextmanager
def get_connection():
    conn = psycopg2.connect(_libpq_dsn(Config.DATABASE_URL))
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


def _format_transcript(messages: list[dict[str, Any]] | None) -> str:
    """Flattens conversation_logs.messages ([{role, content, timestamp}, ...]) into
    the plain-text transcript the Gemini judge prompt expects.
    """
    if not messages:
        return ""
    return "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages)


def get_call_logs_by_ids(conn, call_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Joins call_logs (telephony status/cost, written by the LiveKit/SIP layer) with
    conversation_logs (the actual transcript, written by the call-ended webhook into
    `messages` — call_logs.transcription itself is never populated by any-ai-backend).
    Starts from conversation_logs since that's the side guaranteed to exist for every
    tracked call; call_logs may be absent for calls outside the SIP/telephony path.
    """
    if not call_ids:
        return {}
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT conv.call_id, cl.status, cl.duration, cl.cost, conv.messages
            FROM conversation_logs conv
            LEFT JOIN call_logs cl ON cl.call_id = conv.call_id
            WHERE conv.call_id = ANY(%s)
            """,
            (call_ids,),
        )
        return {
            row["call_id"]: {
                "status": row["status"],
                "duration": row["duration"],
                "cost": row["cost"],
                "transcription": _format_transcript(row["messages"]),
            }
            for row in cur.fetchall()
        }


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


def get_appointment_count_by_call_ids(conn, call_ids: list[str]) -> int:
    """Counts appointments booked from a specific set of calls (via call_logs.call_id),
    not just any appointment created in a time window — an appointment_config_id +
    created_at filter can't tell whether a booking actually came from one of these calls.
    """
    if not call_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM appointments a
            JOIN call_logs cl ON cl.id = a.call_log_id
            WHERE cl.call_id = ANY(%s) AND a.source = 'ai_call'
            """,
            (call_ids,),
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
    """Reads transcripts from conversation_logs.messages, keyed by config_id (the
    AgentConfig.id) — call_logs.agent_id/.transcription are never populated by
    any-ai-backend, so call_logs can't be the source for this.
    """
    with _dict_cursor(conn) as cur:
        cur.execute(
            """
            SELECT call_id, messages
            FROM conversation_logs
            WHERE config_id = %s AND messages IS NOT NULL AND jsonb_array_length(messages) > 0
            ORDER BY start_time DESC
            LIMIT %s
            """,
            (agent_id, limit),
        )
        return [
            {"call_id": row["call_id"], "transcription": _format_transcript(row["messages"])}
            for row in cur.fetchall()
        ]
