"""Context & Memory layer for short-term session state and episodic CAPAs.

Implements the Context & Memory architectural pillar:
- `SessionWorkingContext`: Short-term conversational and "what-if" CPP
  simulation state with bounded token-budget context compaction.
- `EpisodicPrecedentStore`: Long-term institutional memory of validated Biopharma
  cross-site tech-transfer precedents and CAPAs to ground mitigations.
- `AssessmentAuditLedger`: Append-only, SHA-256 hash-chained ledger of
  facility fit evaluations and human sponsor HITL signoffs (ALCOA+ compliant).
"""

from collections.abc import Sequence
import dataclasses
import hashlib
import json
from typing import Any

from facility_fit_agent import models


@dataclasses.dataclass(frozen=True)
class ParameterOverrideEntry:
  """Records a single user 'what-if' CPP parameter override in session state."""

  parameter_name: str
  previous_value: float
  new_value: float
  rationale: str


class SessionWorkingContext:
  """Manages short-term working state and context-window compaction."""

  def __init__(self, session_id: str, operator_id: str = 'msat_engineer'):
    """Initializes short-term working memory for an interactive session."""
    self.session_id = session_id
    self.operator_id = operator_id
    self._cpp_overrides: dict[str, float] = {}
    self._override_history: list[ParameterOverrideEntry] = []
    self._recent_summaries: list[str] = []

  @property
  def active_overrides(self) -> dict[str, float]:
    """Returns copy of active CPP parameter overrides."""
    return dict(self._cpp_overrides)

  def apply_cpp_override(
      self,
      recipe: models.ProcessRecipe,
      parameter_name: str,
      new_value: float,
      rationale: str = 'Interactive what-if sensitivity check',
  ) -> models.ProcessRecipe:
    """Applies a typed numeric CPP override and records delta in memory."""
    if not hasattr(recipe, parameter_name):
      raise AttributeError(
          f'ProcessRecipe has no CPP field named {parameter_name!r}.'
      )
    prev_val = float(getattr(recipe, parameter_name))
    self._cpp_overrides[parameter_name] = new_value
    self._override_history.append(
        ParameterOverrideEntry(
            parameter_name=parameter_name,
            previous_value=prev_val,
            new_value=new_value,
            rationale=rationale,
        )
    )
    return recipe.with_cpp_overrides({parameter_name: new_value})

  def record_turn_summary(self, summary_line: str) -> None:
    """Appends a concise turn outcome to short-term working memory."""
    self._recent_summaries.append(summary_line)

  def to_compact_context_window(
      self, max_recent_turns: int = 5
  ) -> dict[str, Any]:
    """Compacts working context to a bounded token-safe summary dictionary."""
    return {
        'session_id': self.session_id,
        'operator_id': self.operator_id,
        'active_cpp_overrides': dict(self._cpp_overrides),
        'override_count': len(self._override_history),
        'recent_override_deltas': [
            dataclasses.asdict(entry)
            for entry in self._override_history[-max_recent_turns:]
        ],
        'recent_turn_summaries': self._recent_summaries[-max_recent_turns:],
    }


class EpisodicPrecedentStore:
  """Long-term institutional memory of cross-site Biopharma tech-transfer CAPAs."""

  def __init__(
      self,
      precedents: Sequence[models.HistoricalPrecedent] | None = None,
  ):
    """Initializes store with verified historical Biopharma CMC precedents."""
    self._precedents = (
        tuple(precedents)
        if precedents is not None
        else self._default_precedents()
    )

  def retrieve_precedents(
      self,
      site_code: str,
      discrepancies: Sequence[models.DiscrepancyItem],
  ) -> tuple[models.HistoricalPrecedent, ...]:
    """Retrieves historical CAPA precedents matching discrepancy categories."""
    tokens: set[str] = set()
    for item in discrepancies:
      parts = item.discrepancy_id.split('-')
      if len(parts) >= 3:
        tokens.add(parts[2])

    matched: list[models.HistoricalPrecedent] = []
    for prec in self._precedents:
      pattern_match = prec.discrepancy_pattern in tokens
      site_match = prec.site_code in (site_code, 'NETWORK_WIDE')
      if pattern_match and site_match:
        matched.append(prec)
      elif pattern_match and prec not in matched:
        matched.append(prec)
    return tuple(matched)

  def _default_precedents(self) -> tuple[models.HistoricalPrecedent, ...]:
    """Returns curated Biopharma GBO/BRDS historical tech-transfer precedents."""
    return (
        models.HistoricalPrecedent(
            precedent_id='PREC-ALP-2025-014',
            site_code='ALPHA_EU_HUB',
            year=2025,
            reference_molecule='REF-104B',
            discrepancy_pattern='LIM',
            validated_resolution=(
                'Executed parallel ICH Q2(R2) intermediate precision co-val '
                'in Alpha Hub Enterprise LIMS with Central R&D reference lab bridge '
                'disposition for PPQ Lot 1.'
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


class AssessmentAuditLedger:
  """Append-only cryptographic ledger persisting assessments and signoffs."""

  def __init__(self):
    """Initializes an empty hash-chained ledger."""
    self._entries: list[LedgerEntry] = []

  def append(
      self,
      report: models.FacilityFitAssessmentReport,
      payload_sha256: str,
  ) -> LedgerEntry:
    """Appends an assessment state transition and computes chain hash."""
    prev_hash = self._entries[-1].entry_hash if self._entries else 'GENESIS_0'
    seq_idx = len(self._entries)
    material = json.dumps(
        {
            'seq': seq_idx,
            'assessment_id': report.assessment_id,
            'process_id': report.process_id,
            'target_site': report.target_site,
            'signoff_status': report.hitl_signoff_gate.signoff_status,
            'payload_sha256': payload_sha256,
            'prev': prev_hash,
        },
        sort_keys=True,
    ).encode('utf-8')
    entry_hash = hashlib.sha256(material).hexdigest()
    entry = LedgerEntry(
        sequence_index=seq_idx,
        assessment_id=report.assessment_id,
        process_id=report.process_id,
        target_site=report.target_site,
        signoff_status=report.hitl_signoff_gate.signoff_status,
        payload_sha256=payload_sha256,
        previous_entry_hash=prev_hash,
        entry_hash=entry_hash,
    )
    self._entries.append(entry)
    return entry

  def entries(self) -> tuple[LedgerEntry, ...]:
    """Returns immutable snapshot of all ledger entries."""
    return tuple(self._entries)

  def verify_chain_integrity(self) -> bool:
    """Recomputes every SHA-256 link to verify zero ledger tampering."""
    expected_prev = 'GENESIS_0'
    for entry in self._entries:
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
