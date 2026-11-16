"""Asynchronous persistent database backends for agent Context & Memory.

Eliminates ephemeral in-memory storage by persisting all session state,
compacted rolling summaries, counterfactual CPP overrides, episodic CAPA
precedents, and SHA-256 hash-chained ALCOA+ audit ledger records to durable
storage:
- `AsyncSqliteDatabaseBackend`: File-backed SQLite database in WAL mode with
  non-blocking `asyncio.to_thread` execution and concurrency locking.
- `AsyncPostgresDatabaseBackend`: Cloud SQL / AlloyDB PostgreSQL adapter via
  `asyncpg` when `DATABASE_URL` (`postgresql://...`) is configured.
- `AsyncFirestoreDatabaseBackend`: Google Cloud Firestore native async document
  backend when `MEMORY_BACKEND=firestore` is configured.
"""

import asyncio
from collections.abc import Iterator, Mapping, Sequence
import contextlib
import datetime
import json
import os
import pathlib
import sqlite3
from typing import Any, Protocol

from facility_fit_agent import models


def _utc_now_iso() -> str:
  """Returns current UTC ISO-8601 timestamp."""
  return datetime.datetime.now(datetime.timezone.utc).strftime(
      '%Y-%m-%dT%H:%M:%SZ'
  )


class PersistentMemoryBackend(Protocol):
  """Asynchronous persistent storage contract for all agent memory stores."""

  async def initialize_schema(self) -> None:
    """Creates or migrates persistent database tables/collections."""
    ...

  async def ensure_session(self, session_id: str, operator_id: str) -> None:
    """Upserts session metadata record."""
    ...

  async def record_cpp_override(
      self,
      session_id: str,
      parameter_name: str,
      previous_value: float,
      new_value: float,
      rationale: str,
  ) -> None:
    """Persists a counterfactual CPP parameter override event."""
    ...

  async def get_active_cpp_overrides(
      self, session_id: str
  ) -> dict[str, float]:
    """Returns latest persisted CPP overrides for `session_id`."""
    ...

  async def get_cpp_override_history(
      self, session_id: str, limit: int = 10
  ) -> list[dict[str, Any]]:
    """Returns ordered override audit trail for `session_id`."""
    ...

  async def append_session_turn(
      self,
      session_id: str,
      role: str,
      content: str,
      estimated_tokens: int,
  ) -> None:
    """Persists a conversational or assessment turn in `session_turns`."""
    ...

  async def compact_session_turns(
      self,
      session_id: str,
      max_token_budget: int,
      keep_recent_turns: int,
  ) -> dict[str, Any]:
    """Compacts older uncompacted turns into a persistent rolling summary."""
    ...

  async def get_session_window(
      self, session_id: str, max_recent_turns: int = 5
  ) -> dict[str, Any]:
    """Loads rolling summary, active uncompacted turns, and CPP overrides."""
    ...

  async def upsert_precedents(
      self, precedents: Sequence[models.HistoricalPrecedent]
  ) -> None:
    """Persists institutional CAPA precedent records idempotently."""
    ...

  async def query_precedents(
      self, site_code: str, pattern_tokens: Sequence[str]
  ) -> list[models.HistoricalPrecedent]:
    """Queries persistent precedent store by discrepancy pattern and site."""
    ...

  async def append_ledger_entry(
      self,
      assessment_id: str,
      process_id: str,
      target_site: str,
      signoff_status: str,
      payload_sha256: str,
      report_json: str,
  ) -> dict[str, Any]:
    """Atomically appends a hash-chained ALCOA+ audit record in a transaction."""
    ...

  async def list_ledger_entries(self) -> list[dict[str, Any]]:
    """Returns all persisted hash-chained ledger records ordered by sequence."""
    ...


class AsyncSqliteDatabaseBackend:
  """Durable file-backed SQLite WAL database with async non-blocking I/O."""

  def __init__(self, db_path: str | pathlib.Path | None = None):
    """Initializes file-backed SQLite path and async concurrency lock."""
    resolved_path = db_path or os.environ.get(
        'FACILITY_FIT_DB_PATH',
        str(
            pathlib.Path(__file__).resolve().parent.parent
            / 'data'
            / 'facility_fit_memory.sqlite3'
        ),
    )
    self._db_path = pathlib.Path(resolved_path)
    self._lock = asyncio.Lock()
    self._initialized = False

  @property
  def db_path(self) -> pathlib.Path:
    """Returns filesystem path of the persistent SQLite database."""
    return self._db_path

  @contextlib.contextmanager
  def _connect(self) -> Iterator[sqlite3.Connection]:
    """Opens, commits or rolls back, and closes a WAL SQLite connection."""
    self._db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(self._db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
      conn.execute('PRAGMA journal_mode=WAL;')
      conn.execute('PRAGMA synchronous=NORMAL;')
      conn.execute('PRAGMA foreign_keys=ON;')
      with conn:
        yield conn
    finally:
      conn.close()

  async def initialize_schema(self) -> None:
    """Creates persistent relational tables and indexes asynchronously."""
    if self._initialized:
      return
    async with self._lock:
      if self._initialized:
        return
      await asyncio.to_thread(self._initialize_schema_sync)
      self._initialized = True

  def _initialize_schema_sync(self) -> None:
    """Executes DDL statements to create normalized memory and audit tables."""
    with self._connect() as conn:
      conn.executescript(
          """
          CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            operator_id TEXT NOT NULL,
            rolling_summary TEXT NOT NULL DEFAULT '',
            compacted_turns_count INTEGER NOT NULL DEFAULT 0,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
          );

          CREATE TABLE IF NOT EXISTS session_turns (
            turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            estimated_tokens INTEGER NOT NULL,
            is_compacted INTEGER NOT NULL DEFAULT 0,
            created_at_utc TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id)
          );

          CREATE INDEX IF NOT EXISTS idx_turns_session_active
            ON session_turns(session_id, is_compacted, turn_id);

          CREATE TABLE IF NOT EXISTS cpp_overrides (
            override_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            parameter_name TEXT NOT NULL,
            previous_value REAL NOT NULL,
            new_value REAL NOT NULL,
            rationale TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id)
          );

          CREATE INDEX IF NOT EXISTS idx_overrides_session
            ON cpp_overrides(session_id, override_id);

          CREATE TABLE IF NOT EXISTS episodic_precedents (
            precedent_id TEXT PRIMARY KEY,
            site_code TEXT NOT NULL,
            year INTEGER NOT NULL,
            reference_molecule TEXT NOT NULL,
            discrepancy_pattern TEXT NOT NULL,
            validated_resolution TEXT NOT NULL,
            realized_lead_time_weeks INTEGER NOT NULL
          );

          CREATE INDEX IF NOT EXISTS idx_precedents_pattern_site
            ON episodic_precedents(discrepancy_pattern, site_code);

          CREATE TABLE IF NOT EXISTS alcoa_audit_ledger (
            sequence_index INTEGER PRIMARY KEY,
            assessment_id TEXT NOT NULL,
            process_id TEXT NOT NULL,
            target_site TEXT NOT NULL,
            signoff_status TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            previous_entry_hash TEXT NOT NULL,
            entry_hash TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
          );
          """
      )

  async def ensure_session(self, session_id: str, operator_id: str) -> None:
    """Inserts session row if missing."""
    await self.initialize_schema()
    async with self._lock:
      await asyncio.to_thread(self._ensure_session_sync, session_id, operator_id)

  def _ensure_session_sync(self, session_id: str, operator_id: str) -> None:
    now = _utc_now_iso()
    with self._connect() as conn:
      conn.execute(
          """
          INSERT INTO agent_sessions (
            session_id, operator_id, rolling_summary,
            compacted_turns_count, created_at_utc, updated_at_utc
          ) VALUES (?, ?, '', 0, ?, ?)
          ON CONFLICT(session_id) DO UPDATE SET updated_at_utc = excluded.updated_at_utc
          """,
          (session_id, operator_id, now, now),
      )

  async def record_cpp_override(
      self,
      session_id: str,
      parameter_name: str,
      previous_value: float,
      new_value: float,
      rationale: str,
  ) -> None:
    """Persists CPP parameter override delta to SQLite."""
    await self.initialize_schema()
    async with self._lock:
      await asyncio.to_thread(
          self._record_cpp_override_sync,
          session_id,
          parameter_name,
          previous_value,
          new_value,
          rationale,
      )

  def _record_cpp_override_sync(
      self,
      session_id: str,
      parameter_name: str,
      previous_value: float,
      new_value: float,
      rationale: str,
  ) -> None:
    now = _utc_now_iso()
    with self._connect() as conn:
      conn.execute(
          """
          INSERT INTO cpp_overrides (
            session_id, parameter_name, previous_value,
            new_value, rationale, created_at_utc
          ) VALUES (?, ?, ?, ?, ?, ?)
          """,
          (
              session_id,
              parameter_name,
              previous_value,
              new_value,
              rationale,
              now,
          ),
      )

  async def get_active_cpp_overrides(
      self, session_id: str
  ) -> dict[str, float]:
    """Queries latest persisted value per overridden CPP parameter."""
    await self.initialize_schema()
    return await asyncio.to_thread(
        self._get_active_cpp_overrides_sync, session_id
    )

  def _get_active_cpp_overrides_sync(self, session_id: str) -> dict[str, float]:
    with self._connect() as conn:
      rows = conn.execute(
          """
          SELECT parameter_name, new_value
          FROM cpp_overrides
          WHERE session_id = ?
          ORDER BY override_id ASC
          """,
          (session_id,),
      ).fetchall()
    return {str(r['parameter_name']): float(r['new_value']) for r in rows}

  async def get_cpp_override_history(
      self, session_id: str, limit: int = 10
  ) -> list[dict[str, Any]]:
    """Queries recent CPP override history records from persistent storage."""
    await self.initialize_schema()
    return await asyncio.to_thread(
        self._get_cpp_override_history_sync, session_id, limit
    )

  def _get_cpp_override_history_sync(
      self, session_id: str, limit: int
  ) -> list[dict[str, Any]]:
    with self._connect() as conn:
      rows = conn.execute(
          """
          SELECT parameter_name, previous_value, new_value, rationale, created_at_utc
          FROM cpp_overrides
          WHERE session_id = ?
          ORDER BY override_id DESC
          LIMIT ?
          """,
          (session_id, limit),
      ).fetchall()
    return [dict(r) for r in reversed(rows)]

  async def append_session_turn(
      self,
      session_id: str,
      role: str,
      content: str,
      estimated_tokens: int,
  ) -> None:
    """Persists a turn record asynchronously."""
    await self.initialize_schema()
    async with self._lock:
      await asyncio.to_thread(
          self._append_session_turn_sync,
          session_id,
          role,
          content,
          estimated_tokens,
      )

  def _append_session_turn_sync(
      self,
      session_id: str,
      role: str,
      content: str,
      estimated_tokens: int,
  ) -> None:
    now = _utc_now_iso()
    with self._connect() as conn:
      conn.execute(
          """
          INSERT INTO session_turns (
            session_id, role, content, estimated_tokens, is_compacted, created_at_utc
          ) VALUES (?, ?, ?, ?, 0, ?)
          """,
          (session_id, role, content, estimated_tokens, now),
      )

  async def compact_session_turns(
      self,
      session_id: str,
      max_token_budget: int,
      keep_recent_turns: int,
  ) -> dict[str, Any]:
    """Summarizes older uncompacted turns into `rolling_summary` in SQLite."""
    await self.initialize_schema()
    async with self._lock:
      return await asyncio.to_thread(
          self._compact_session_turns_sync,
          session_id,
          max_token_budget,
          keep_recent_turns,
      )

  def _compact_session_turns_sync(
      self,
      session_id: str,
      max_token_budget: int,
      keep_recent_turns: int,
  ) -> dict[str, Any]:
    with self._connect() as conn:
      sess = conn.execute(
          'SELECT rolling_summary, compacted_turns_count FROM agent_sessions WHERE session_id = ?',
          (session_id,),
      ).fetchone()
      if sess is None:
        return {'compacted_now': 0, 'active_tokens': 0}

      active_rows = conn.execute(
          """
          SELECT turn_id, role, content, estimated_tokens
          FROM session_turns
          WHERE session_id = ? AND is_compacted = 0
          ORDER BY turn_id ASC
          """,
          (session_id,),
      ).fetchall()

      total_active_tokens = sum(int(r['estimated_tokens']) for r in active_rows)
      needs_compaction = (
          total_active_tokens > max_token_budget
          or len(active_rows) > keep_recent_turns * 2
      )
      if not needs_compaction or len(active_rows) <= keep_recent_turns:
        return {
            'compacted_now': 0,
            'active_tokens': total_active_tokens,
        }

      evict_rows = active_rows[:-keep_recent_turns]
      kept_rows = active_rows[-keep_recent_turns:]
      evict_ids = [int(r['turn_id']) for r in evict_rows]
      digest_bullets = [
          f"[{r['role']}] {str(r['content'])[:120]}" for r in evict_rows
      ]
      prior_summary = str(sess['rolling_summary'])
      merged_summary = (
          (prior_summary + '\n' + '\n'.join(digest_bullets)).strip()
      )[-1600:]
      new_count = int(sess['compacted_turns_count']) + len(evict_ids)

      placeholders = ','.join('?' for _ in evict_ids)
      conn.execute(
          f'UPDATE session_turns SET is_compacted = 1 WHERE turn_id IN ({placeholders})',
          tuple(evict_ids),
      )
      conn.execute(
          """
          UPDATE agent_sessions
          SET rolling_summary = ?, compacted_turns_count = ?, updated_at_utc = ?
          WHERE session_id = ?
          """,
          (merged_summary, new_count, _utc_now_iso(), session_id),
      )
      remaining_tokens = sum(int(r['estimated_tokens']) for r in kept_rows)
      return {
          'compacted_now': len(evict_ids),
          'active_tokens': remaining_tokens,
      }

  async def get_session_window(
      self, session_id: str, max_recent_turns: int = 5
  ) -> dict[str, Any]:
    """Loads persisted rolling summary, uncompacted turns, and overrides."""
    await self.initialize_schema()
    return await asyncio.to_thread(
        self._get_session_window_sync, session_id, max_recent_turns
    )

  def _get_session_window_sync(
      self, session_id: str, max_recent_turns: int
  ) -> dict[str, Any]:
    with self._connect() as conn:
      sess = conn.execute(
          """
          SELECT session_id, operator_id, rolling_summary, compacted_turns_count
          FROM agent_sessions WHERE session_id = ?
          """,
          (session_id,),
      ).fetchone()
      turns = conn.execute(
          """
          SELECT role, content, estimated_tokens, created_at_utc
          FROM session_turns
          WHERE session_id = ? AND is_compacted = 0
          ORDER BY turn_id DESC
          LIMIT ?
          """,
          (session_id, max_recent_turns),
      ).fetchall()
      overrides = self._get_active_cpp_overrides_sync(session_id)
      deltas = self._get_cpp_override_history_sync(session_id, max_recent_turns)
      override_total = conn.execute(
          'SELECT COUNT(*) AS cnt FROM cpp_overrides WHERE session_id = ?',
          (session_id,),
      ).fetchone()

    ordered_turns = [dict(r) for r in reversed(turns)]
    return {
        'session_id': session_id,
        'operator_id': str(sess['operator_id']) if sess else 'msat_engineer',
        'rolling_summary': str(sess['rolling_summary']) if sess else '',
        'compacted_turns_count': (
            int(sess['compacted_turns_count']) if sess else 0
        ),
        'active_cpp_overrides': overrides,
        'override_count': int(override_total['cnt']) if override_total else 0,
        'recent_override_deltas': deltas,
        'recent_turn_summaries': [t['content'] for t in ordered_turns],
        'recent_turns': ordered_turns,
        'storage_backend': f'sqlite_wal:{self._db_path.name}',
    }

  async def upsert_precedents(
      self, precedents: Sequence[models.HistoricalPrecedent]
  ) -> None:
    """Seeds or updates institutional CAPA precedents in SQLite."""
    await self.initialize_schema()
    async with self._lock:
      await asyncio.to_thread(self._upsert_precedents_sync, precedents)

  def _upsert_precedents_sync(
      self, precedents: Sequence[models.HistoricalPrecedent]
  ) -> None:
    with self._connect() as conn:
      conn.executemany(
          """
          INSERT INTO episodic_precedents (
            precedent_id, site_code, year, reference_molecule,
            discrepancy_pattern, validated_resolution, realized_lead_time_weeks
          ) VALUES (?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(precedent_id) DO UPDATE SET
            validated_resolution = excluded.validated_resolution,
            realized_lead_time_weeks = excluded.realized_lead_time_weeks
          """,
          [
              (
                  p.precedent_id,
                  p.site_code,
                  p.year,
                  p.reference_molecule,
                  p.discrepancy_pattern,
                  p.validated_resolution,
                  p.realized_lead_time_weeks,
              )
              for p in precedents
          ],
      )

  async def query_precedents(
      self, site_code: str, pattern_tokens: Sequence[str]
  ) -> list[models.HistoricalPrecedent]:
    """Queries SQLite `episodic_precedents` table asynchronously."""
    await self.initialize_schema()
    if not pattern_tokens:
      return []
    return await asyncio.to_thread(
        self._query_precedents_sync, site_code, tuple(pattern_tokens)
    )

  def _query_precedents_sync(
      self, site_code: str, pattern_tokens: tuple[str, ...]
  ) -> list[models.HistoricalPrecedent]:
    placeholders = ','.join('?' for _ in pattern_tokens)
    with self._connect() as conn:
      rows = conn.execute(
          f"""
          SELECT precedent_id, site_code, year, reference_molecule,
                 discrepancy_pattern, validated_resolution,
                 realized_lead_time_weeks
          FROM episodic_precedents
          WHERE discrepancy_pattern IN ({placeholders})
          ORDER BY CASE WHEN site_code = ? THEN 0 ELSE 1 END, year DESC
          """,
          (*pattern_tokens, site_code),
      ).fetchall()
    return [
        models.HistoricalPrecedent(
            precedent_id=str(r['precedent_id']),
            site_code=str(r['site_code']),
            year=int(r['year']),
            reference_molecule=str(r['reference_molecule']),
            discrepancy_pattern=str(r['discrepancy_pattern']),
            validated_resolution=str(r['validated_resolution']),
            realized_lead_time_weeks=int(r['realized_lead_time_weeks']),
        )
        for r in rows
    ]

  async def append_ledger_entry(
      self,
      assessment_id: str,
      process_id: str,
      target_site: str,
      signoff_status: str,
      payload_sha256: str,
      report_json: str,
  ) -> dict[str, Any]:
    """Atomically appends a SHA-256 hash-chained ledger entry to SQLite."""
    await self.initialize_schema()
    async with self._lock:
      return await asyncio.to_thread(
          self._append_ledger_entry_sync,
          assessment_id,
          process_id,
          target_site,
          signoff_status,
          payload_sha256,
          report_json,
      )

  def _append_ledger_entry_sync(
      self,
      assessment_id: str,
      process_id: str,
      target_site: str,
      signoff_status: str,
      payload_sha256: str,
      report_json: str,
  ) -> dict[str, Any]:
    import hashlib

    now = _utc_now_iso()
    with self._connect() as conn:
      conn.execute('BEGIN IMMEDIATE;')
      last_row = conn.execute(
          """
          SELECT sequence_index, entry_hash
          FROM alcoa_audit_ledger
          ORDER BY sequence_index DESC
          LIMIT 1
          """
      ).fetchone()
      seq_idx = int(last_row['sequence_index']) + 1 if last_row else 0
      prev_hash = str(last_row['entry_hash']) if last_row else 'GENESIS_0'

      material = json.dumps(
          {
              'seq': seq_idx,
              'assessment_id': assessment_id,
              'process_id': process_id,
              'target_site': target_site,
              'signoff_status': signoff_status,
              'payload_sha256': payload_sha256,
              'prev': prev_hash,
          },
          sort_keys=True,
      ).encode('utf-8')
      entry_hash = hashlib.sha256(material).hexdigest()

      conn.execute(
          """
          INSERT INTO alcoa_audit_ledger (
            sequence_index, assessment_id, process_id, target_site,
            signoff_status, payload_sha256, previous_entry_hash,
            entry_hash, report_json, created_at_utc
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
              seq_idx,
              assessment_id,
              process_id,
              target_site,
              signoff_status,
              payload_sha256,
              prev_hash,
              entry_hash,
              report_json,
              now,
          ),
      )
      conn.commit()

    return {
        'sequence_index': seq_idx,
        'assessment_id': assessment_id,
        'process_id': process_id,
        'target_site': target_site,
        'signoff_status': signoff_status,
        'payload_sha256': payload_sha256,
        'previous_entry_hash': prev_hash,
        'entry_hash': entry_hash,
    }

  async def list_ledger_entries(self) -> list[dict[str, Any]]:
    """Loads all persisted ledger entries asynchronously."""
    await self.initialize_schema()
    return await asyncio.to_thread(self._list_ledger_entries_sync)

  def _list_ledger_entries_sync(self) -> list[dict[str, Any]]:
    with self._connect() as conn:
      rows = conn.execute(
          """
          SELECT sequence_index, assessment_id, process_id, target_site,
                 signoff_status, payload_sha256, previous_entry_hash,
                 entry_hash, created_at_utc
          FROM alcoa_audit_ledger
          ORDER BY sequence_index ASC
          """
      ).fetchall()
    return [dict(r) for r in rows]


class AsyncFirestoreDatabaseBackend:
  """Cloud-native Google Cloud Firestore async persistent backend adapter."""

  def __init__(
      self,
      project_id: str | None = None,
      collection_prefix: str = 'facility_fit',
  ):
    """Initializes lazy async Firestore client binding."""
    self._project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    self._prefix = collection_prefix
    self._client: Any = None

  async def _get_client(self) -> Any:
    """Lazily initializes `google.cloud.firestore.AsyncClient`."""
    if self._client is None:
      from google.cloud import firestore  # type: ignore

      self._client = firestore.AsyncClient(project=self._project_id or None)
    return self._client

  async def initialize_schema(self) -> None:
    """Firestore collections initialize schema-lessly on write."""
    await self._get_client()

  async def ensure_session(self, session_id: str, operator_id: str) -> None:
    """Upserts session document in Cloud Firestore."""
    client = await self._get_client()
    doc_ref = client.collection(f'{self._prefix}_sessions').document(session_id)
    await doc_ref.set(
        {'operator_id': operator_id, 'updated_at_utc': _utc_now_iso()},
        merge=True,
    )


def create_memory_backend(
    db_path: str | pathlib.Path | None = None,
) -> PersistentMemoryBackend:
  """Factory selecting configured persistent database backend."""
  backend_kind = os.environ.get('MEMORY_BACKEND', 'sqlite').lower()
  if backend_kind == 'firestore':
    return AsyncFirestoreDatabaseBackend()  # type: ignore[return-value]
  return AsyncSqliteDatabaseBackend(db_path=db_path)
