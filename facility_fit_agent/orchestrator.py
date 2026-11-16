"""Orchestration & Logic layer executing the 5-stage GxP reasoning DAG.

Implements the Orchestration & Logic architectural pillar:
- Explicit 5-stage deterministic state machine (`OrchestrationStage`)
coordinating
  Enterprise PKM ingestion, Plant MES / LIMS discovery, physics tool execution,
  episodic CAPA precedent recall, and GxP Human-in-the-Loop gating.
- Multi-site portfolio fan-out orchestration (`orchestrate_network_comparison`).
- Stateful counterfactual sensitivity simulation (`simulate_what_if`).
"""

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import enum
from typing import Any

from facility_fit_agent import agent
from facility_fit_agent import memory
from facility_fit_agent import models
from facility_fit_agent import observability
from facility_fit_agent import tools

_MANDATORY_REPORT_KEYS = frozenset({
    'assessment_id',
    'process_id',
    'target_site',
    'overall_fit_status',
    'discrepancy_matrix',
    'mitigation_recommendations',
})


class OrchestrationStage(str, enum.Enum):
  """Canonical state-machine nodes of the Facility Fit workflow DAG."""

  STAGE_1_INGEST_AND_NORMALIZE = 'STAGE_1_INGEST_AND_NORMALIZE'
  STAGE_2_ASSET_AND_LIMS_DISCOVERY = 'STAGE_2_ASSET_AND_LIMS_DISCOVERY'
  STAGE_3_DETERMINISTIC_PHYSICS = 'STAGE_3_DETERMINISTIC_PHYSICS'
  STAGE_4_RISK_AND_PRECEDENT_RECALL = 'STAGE_4_RISK_AND_PRECEDENT_RECALL'
  STAGE_5_GXP_HITL_AND_AUDIT_COMMIT = 'STAGE_5_GXP_HITL_AND_AUDIT_COMMIT'


def validate_report_contract(payload: Mapping[str, Any]) -> None:
  """Verifies strict compliance with `FacilityFitAssessmentReport` schema."""
  missing = _MANDATORY_REPORT_KEYS - set(payload.keys())
  if missing:
    missing_str = ', '.join(sorted(missing))
    raise models.SchemaContractError(
        'Output violates FacilityFitAssessmentReport contract; '
        f'missing keys: {missing_str}.'
    )


class FacilityFitOrchestrator:
  """Stateful DAG orchestrator integrating Tools, Memory, and Tracing."""

  def __init__(
      self,
      tool_registry: tools.ToolRegistry | None = None,
      precedent_store: memory.EpisodicPrecedentStore | None = None,
      audit_ledger: memory.AssessmentAuditLedger | None = None,
      metrics: observability.MetricsRegistry | None = None,
      timestamp_provider: datetime.datetime | None = None,
  ):
    """Initializes orchestrator with injectable collaborators."""
    self._tools = tool_registry or tools.ToolRegistry()
    self._precedents = precedent_store or memory.EpisodicPrecedentStore()
    self._ledger = (
        audit_ledger
        if audit_ledger is not None
        else memory.AssessmentAuditLedger()
    )
    self._metrics = (
        metrics if metrics is not None else observability.MetricsRegistry()
    )
    self._agent = agent.BiopharmaFacilityFitAgent(
        timestamp_provider=timestamp_provider
    )

  @property
  def tool_registry(self) -> tools.ToolRegistry:
    """Returns the active `ToolRegistry`."""
    return self._tools

  @property
  def audit_ledger(self) -> memory.AssessmentAuditLedger:
    """Returns the ALCOA+ hash-chained `AssessmentAuditLedger`."""
    return self._ledger

  @property
  def metrics(self) -> observability.MetricsRegistry:
    """Returns the shared `MetricsRegistry`."""
    return self._metrics

  def execute_assessment(
      self,
      process_id: str,
      target_site: str,
      session_context: memory.SessionWorkingContext | None = None,
      sponsor_signoff: str = '',
  ) -> models.FacilityFitAssessmentReport:
    """Executes all 5 DAG stages with full span tracing and memory recall."""
    tracer = observability.AgentTracer(metrics=self._metrics)
    self._metrics.increment('assessments_executed_total')

    with tracer.start_span(
        span_name='workflow:facility_fit_assessment',
        stage='ORCHESTRATOR_ROOT',
        attributes={'process_id': process_id, 'target_site': target_site},
    ) as root_span:
      with tracer.start_span(
          span_name='stage_1_ingest_and_normalize',
          stage=OrchestrationStage.STAGE_1_INGEST_AND_NORMALIZE.value,
      ):
        recipe: models.ProcessRecipe = self._tools.invoke(
            'ingest_enterprise_pkm_recipe',
            tracer=tracer,
            process_id=process_id,
        )
        if session_context is not None and session_context.active_overrides:
          recipe = recipe.with_cpp_overrides(session_context.active_overrides)

      with tracer.start_span(
          span_name='stage_2_asset_and_lims_discovery',
          stage=OrchestrationStage.STAGE_2_ASSET_AND_LIMS_DISCOVERY.value,
      ):
        facility: models.FacilityRegistry = self._tools.invoke(
            'discover_plant_mes_site_envelope',
            tracer=tracer,
            site_code=target_site,
        )

      with tracer.start_span(
          span_name='stage_3_deterministic_physics',
          stage=OrchestrationStage.STAGE_3_DETERMINISTIC_PHYSICS.value,
      ):
        self._tools.invoke(
            'compute_bioreactor_hydrodynamics',
            tracer=tracer,
            recipe=recipe,
            bioreactor=facility.bioreactors[0],
        )
        self._tools.invoke(
            'compute_downstream_sizing',
            tracer=tracer,
            recipe=recipe,
            chrom_skid=facility.chromatography_skids[0],
        )
        base_report = self._agent.evaluate(recipe, facility)

      with tracer.start_span(
          span_name='stage_4_risk_and_precedent_recall',
          stage=OrchestrationStage.STAGE_4_RISK_AND_PRECEDENT_RECALL.value,
      ) as recall_span:
        matched_precedents = self._precedents.retrieve_precedents(
            site_code=target_site,
            discrepancies=base_report.discrepancy_matrix,
        )
        recall_span.set_attribute(
            'memory.matched_precedents', len(matched_precedents)
        )
        enriched_report = dataclasses.replace(
            base_report,
            matched_precedents=matched_precedents,
        )

      with tracer.start_span(
          span_name='stage_5_gxp_hitl_and_audit_commit',
          stage=OrchestrationStage.STAGE_5_GXP_HITL_AND_AUDIT_COMMIT.value,
      ):
        if sponsor_signoff:
          enriched_report = self._agent.record_sponsor_signoff(
              enriched_report, sponsor_signoff
          )
        if enriched_report.hitl_signoff_gate.persistence_locked:
          self._metrics.increment('hitl_locks_triggered_total')
        preliminary_dict = enriched_report.to_dict()
        validate_report_contract(preliminary_dict)

      root_span.set_attribute(
          'fit.normalized_index', enriched_report.normalized_fit_index
      )
      root_span.set_attribute(
          'fit.overall_status', enriched_report.overall_fit_status.value
      )

    obs_bundle = tracer.build_bundle(enriched_report.to_dict())
    final_report = dataclasses.replace(
        enriched_report, observability=obs_bundle
    )
    self._ledger.append(final_report, obs_bundle.payload_sha256)

    if session_context is not None:
      session_context.record_turn_summary(
          f'{process_id}@{target_site} -> '
          f'index={final_report.normalized_fit_index:.2f} '
          f'({final_report.overall_fit_status.value})'
      )
    return final_report

  def orchestrate_network_comparison(
      self,
      process_id: str,
      site_codes: Sequence[str] = ('ALPHA_EU_HUB', 'BETA_US_CAMPUS', 'GAMMA_US_PLANT'),
      session_context: memory.SessionWorkingContext | None = None,
      sponsor_signoff: str = '',
  ) -> tuple[models.FacilityFitAssessmentReport, ...]:
    """Fans out assessments across multiple receiving sites and ranks fit."""
    reports = [
        self.execute_assessment(
            process_id=process_id,
            target_site=site_code,
            session_context=session_context,
            sponsor_signoff=sponsor_signoff,
        )
        for site_code in site_codes
    ]
    return tuple(reports)

  def simulate_what_if(
      self,
      session_context: memory.SessionWorkingContext,
      process_id: str,
      target_site: str,
      cpp_overrides: Mapping[str, float],
  ) -> dict[str, Any]:
    """Runs stateful before/after counterfactual CPP sensitivity simulation."""
    baseline = self.execute_assessment(
        process_id=process_id,
        target_site=target_site,
    )
    base_recipe = self._tools.invoke(
        'ingest_enterprise_pkm_recipe',
        process_id=process_id,
    )
    for param_name, new_val in cpp_overrides.items():
      base_recipe = session_context.apply_cpp_override(
          base_recipe,
          parameter_name=param_name,
          new_value=new_val,
      )
    simulated = self.execute_assessment(
        process_id=process_id,
        target_site=target_site,
        session_context=session_context,
    )
    return {
        'session_window': session_context.to_compact_context_window(),
        'baseline_fit_index': baseline.normalized_fit_index,
        'simulated_fit_index': simulated.normalized_fit_index,
        'fit_index_delta': round(
            simulated.normalized_fit_index - baseline.normalized_fit_index, 4
        ),
        'baseline_discrepancy_ids': [
            d.discrepancy_id for d in baseline.discrepancy_matrix
        ],
        'simulated_discrepancy_ids': [
            d.discrepancy_id for d in simulated.discrepancy_matrix
        ],
        'simulated_report': simulated.to_dict(),
    }
