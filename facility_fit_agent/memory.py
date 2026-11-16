"""Asynchronous, persistent Context & Memory layer with LLM prompt assembly.

Implements all four dimensions of the Context & Memory architectural pillar:
1. Constitutional LLM System Prompt integration (`prompts.SYSTEM_PROMPT` and
   `SessionWorkingContext.build_llm_prompt_context`).
2. Persistent Database Storage (`database.PersistentMemoryBackend` backed by
   WAL-mode SQLite on disk or Google Cloud Firestore — zero ephemeral in-memory
   list/dict state).
3. Asynchronous Non-Blocking Operations (`async def` / `await` across session
   writes, override tracking, precedent queries, and ledger commits).
4. Token-Budget Rolling Summary Compaction (`compact_history` summarizing
   evicted turns into persistent `rolling_summary` storage).
"""

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
from typing import Any

from facility_fit_agent import database
from facility_fit_agent import models
from facility_fit_agent import prompts


@dataclasses.dataclass(frozen=True)
class ParameterOverrideEntry:
  """Immutable representation of a persisted CPP override audit event."""

  parameter_name: str
  previous_value: float
  new_value: float
  rationale: str


@dataclasses.dataclass(frozen=True)
class LedgerEntry:
  """Tamper-evident hash-chained ALCOA+ assessment ledger record."""

  sequence_index: int
  assessment_id: str
  process_id: str
  target_site: str
  signoff_status: str
  payload_sha256: str
  previous_entry_hash: str
  entry_hash: str


class SessionWorkingContext:
  """Async persistent session memory with rolling-summary token compaction."""

  def __init__(
      self,
      session_id: str,
      operator_id: str = 'msat_engineer',
      backend: database.PersistentMemoryBackend | None = None,
      max_token_budget: int = 1200,
      keep_recent_turns: int = 5,
      prompt_assembler: prompts.PromptContextAssembler | None = None,
  ):
    """Initializes session bound to a persistent async database backend."""
    self.session_id = session_id
    self.operator_id = operator_id
    self._backend = backend or database.create_memory_backend()
    self._max_token_budget = max_token_budget
    self._keep_recent_turns = keep_recent_turns
    self._prompt_assembler = prompt_assembler or prompts.PromptContextAssembler(
        system_prompt=prompts.SYSTEM_PROMPT
    )

  async def initialize(self) -> None:
    """Ensures persistent schema and session row exist in the database."""
    await self._backend.ensure_session(self.session_id, self.operator_id)

  async def get_active_overrides(self) -> dict[str, float]:
    """Asynchronously loads persisted CPP overrides for this session."""
    await self.initialize()
    return await self._backend.get_active_cpp_overrides(self.session_id)

  async def apply_cpp_override(
      self,
      recipe: models.ProcessRecipe,
      parameter_name: str,
      new_value: float,
      rationale: str = 'Interactive what-if sensitivity check',
  ) -> models.ProcessRecipe:
    """Persists a validated numeric CPP override to the database asynchronously."""
    if not hasattr(recipe, parameter_name):
      raise AttributeError(
          f'ProcessRecipe has no CPP field named {parameter_name!r}.'
      )
    await self.initialize()
    prev_val = float(getattr(recipe, parameter_name))
    await self._backend.record_cpp_override(
        session_id=self.session_id,
        parameter_name=parameter_name,
        previous_value=prev_val,
        new_value=new_value,
        rationale=rationale,
    )
    turn_note = (
        f'CPP override applied: {parameter_name} {prev_val} -> {new_value} '
        f'({rationale})'
    )
    await self.record_turn_summary(turn_note, role='user')
    return recipe.with_cpp_overrides({parameter_name: new_value})

  async def record_turn_summary(
      self,
      summary_line: str,
      role: str = 'assistant',
  ) -> dict[str, Any]:
    """Persists a turn to the database and triggers async token compaction."""
    await self.initialize()
    est_tokens = prompts.estimate_token_count(summary_line)
    await self._backend.append_session_turn(
        session_id=self.session_id,
        role=role,
        content=summary_line,
        estimated_tokens=est_tokens,
    )
    return await self.compact_history()

  async def compact_history(self) -> dict[str, Any]:
    """Compacts older database turns into a persistent rolling summary."""
    await self.initialize()
    return await self._backend.compact_session_turns(
        session_id=self.session_id,
        max_token_budget=self._max_token_budget,
        keep_recent_turns=self._keep_recent_turns,
    )

  async def to_compact_context_window(
      self, max_recent_turns: int | None = None
  ) -> dict[str, Any]:
    """Asynchronously fetches compacted rolling summary + recent turns from DB."""
    await self.initialize()
    limit = max_recent_turns or self._keep_recent_turns
    return await self._backend.get_session_window(
        session_id=self.session_id,
        max_recent_turns=limit,
    )

  async def build_llm_prompt_context(
      self,
      user_task: str,
      precedents: Sequence[models.HistoricalPrecedent] = (),
      tool_declarations: Sequence[Mapping[str, Any]] = (),
  ) -> dict[str, Any]:
    """Assembles `SYSTEM_PROMPT` + compacted DB memory into LLM messages."""
    await self.compact_history()
    window = await self.to_compact_context_window()
    return self._prompt_assembler.build_messages(
        user_task=user_task,
        compacted_memory_window=window,
        precedents=precedents,
        tool_declarations=tool_declarations,
    )


class EpisodicPrecedentStore:
  """Async persistent store of validated cross-site tech-transfer CAPAs."""

  def __init__(
      self,
      backend: database.PersistentMemoryBackend | None = None,
      seed_precedents: Sequence[models.HistoricalPrecedent] | None = None,
  ):
    """Initializes episodic store backed by persistent database storage."""
    self._backend = backend or database.create_memory_backend()
    self._seed_items = (
        tuple(seed_precedents)
        if seed_precedents is not None
        else self._default_precedents()
    )
    self._seeded = False

  async def ensure_seeded(self) -> None:
    """Idempotently seeds institutional CAPA precedents into persistent DB."""
    if self._seeded:
      return
    await self._backend.upsert_precedents(self._seed_items)
    self._seeded = True

  async def retrieve_precedents(
      self,
      site_code: str,
      discrepancies: Sequence[models.DiscrepancyItem],
  ) -> tuple[models.HistoricalPrecedent, ...]:
    """Asynchronously queries persistent precedent table by discrepancy type."""
    await self.ensure_seeded()
    tokens: list[str] = []
    for item in discrepancies:
      parts = item.discrepancy_id.split('-')
      if len(parts) >= 3 and parts[2] not in tokens:
        tokens.append(parts[2])
    rows = await self._backend.query_precedents(site_code, tokens)
    return tuple(rows)

  def _default_precedents(self) -> tuple[models.HistoricalPrecedent, ...]:
    """Returns baseline institutional tech-transfer CAPA records for seeding."""
    return (
        models.HistoricalPrecedent(
            precedent_id='PREC-ALP-2025-014',
            site_code='ALPHA_EU_HUB',
            year=2025,
            reference_molecule='REF-104B',
            discrepancy_pattern='LIM',
            validated_resolution=(
                'Executed parallel ICH Q2(R2) intermediate precision co-val '
                'in Alpha Hub Enterprise LIMS with Central R&D reference lab '
                'bridge disposition for PPQ Lot 1.'
            ),
            realized_lead_time_weeks=5,
        ),
        models.HistoricalPrecedent(
            precedent_id='PREC-BET-2024-019',
            site_code='BETA_US_CAMPUS',
            year=2024,
            reference_molecule='REF-208',
            discrepancy_pattern='SHR',
            validated_resolution=(
                'Retrofitted Suite H2 10kL SS vessel from dual Rushton '
                'turbines (Np=5.0) to 3-blade hydrofoils (Np=0.88) + 20-um '
                'sintered sparger, raising Kolmogorov scale from 18 um to '
                '54 um.'
            ),
            realized_lead_time_weeks=15,
        ),
        models.HistoricalPrecedent(
            precedent_id='PREC-BET-2024-031',
            site_code='BETA_US_CAMPUS',
            year=2024,
            reference_molecule='REF-312',
            discrepancy_pattern='UTL',
            validated_resolution=(
                'Implemented Plant MES CIP interlock stagger + pre-prepared '
                '2,500L single-use equilibration buffer totes, cutting peak '
                'simultaneous WFI draw by 1,620 L/h.'
            ),
            realized_lead_time_weeks=4,
        ),
        models.HistoricalPrecedent(
            precedent_id='PREC-GAM-2025-007',
            site_code='GAMMA_US_PLANT',
            year=2025,
            reference_molecule='REF-419',
            discrepancy_pattern='VOL',
            validated_resolution=(
                'Concentrated basal media 1.35x to lift Day-0 fill above '
                '7,450 L minimum impeller/probe submergence in tall SF4 '
                'vessels without hardware downtime.'
            ),
            realized_lead_time_weeks=9,
        ),
        models.HistoricalPrecedent(
            precedent_id='PREC-GAM-2025-022',
            site_code='GAMMA_US_PLANT',
            year=2025,
            reference_molecule='REF-525-SC',
            discrepancy_pattern='TFF',
            validated_resolution=(
                'Deployed portable low-hold-up (<8 L) single-use finish TFF '
                'skid for final >175 g/L SC concentration step, eliminating '
                'loop foaming and SEC HMW spike.'
            ),
            realized_lead_time_weeks=8,
        ),
    )


class AssessmentAuditLedger:
  """Async persistent ALCOA+ SHA-256 hash-chained assessment ledger."""

  def __init__(
      self,
      backend: database.PersistentMemoryBackend | None = None,
  ):
    """Initializes audit ledger backed by durable SQLite / Cloud storage."""
    self._backend = backend or database.create_memory_backend()

  async def append(
      self,
      report: models.FacilityFitAssessmentReport,
      payload_sha256: str,
  ) -> LedgerEntry:
    """Atomically persists a hash-chained assessment state transition."""
    serialized_report = json.dumps(report.to_dict(), sort_keys=True)
    record = await self._backend.append_ledger_entry(
        assessment_id=report.assessment_id,
        process_id=report.process_id,
        target_site=report.target_site,
        signoff_status=report.hitl_signoff_gate.signoff_status,
        payload_sha256=payload_sha256,
        report_json=serialized_report,
    )
    return LedgerEntry(
        sequence_index=int(record['sequence_index']),
        assessment_id=str(record['assessment_id']),
        process_id=str(record['process_id']),
        target_site=str(record['target_site']),
        signoff_status=str(record['signoff_status']),
        payload_sha256=str(record['payload_sha256']),
        previous_entry_hash=str(record['previous_entry_hash']),
        entry_hash=str(record['entry_hash']),
    )

  async def entries(self) -> tuple[LedgerEntry, ...]:
    """Asynchronously reads all persisted ledger records in sequence order."""
    raw_rows = await self._backend.list_ledger_entries()
    return tuple(
        LedgerEntry(
            sequence_index=int(r['sequence_index']),
            assessment_id=str(r['assessment_id']),
            process_id=str(r['process_id']),
            target_site=str(r['target_site']),
            signoff_status=str(r['signoff_status']),
            payload_sha256=str(r['payload_sha256']),
            previous_entry_hash=str(r['previous_entry_hash']),
            entry_hash=str(r['entry_hash']),
        )
        for r in raw_rows
    )

  async def verify_chain_integrity(self) -> bool:
    """Asynchronously verifies every persisted SHA-256 link from GENESIS_0."""
    persisted = await self.entries()
    expected_prev = 'GENESIS_0'
    for entry in persisted:
      if entry.previous_entry_hash != expected_prev:
        return False
      material = json.dumps(
          {
              'seq': entry.sequence_index,
              'assessment_id': entry.assessment_id,
              'process_id': entry.process_id,
              'target_site': entry.target_site,
              'signoff_status': entry.signoff_status,
              'payload_sha256': entry.payload_sha256,
              'prev': entry.previous_entry_hash,
          },
          sort_keys=True,
      ).encode('utf-8')
      if hashlib.sha256(material).hexdigest() != entry.entry_hash:
        return False
      expected_prev = entry.entry_hash
    return True
