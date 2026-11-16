"""Asynchronous 5-stage GxP orchestration DAG with persistent memory & prompts."""

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
import dataclasses
import datetime
import enum
from typing import Any, TypeVar

from facility_fit_agent import agent
from facility_fit_agent import database
from facility_fit_agent import memory
from facility_fit_agent import models
from facility_fit_agent import observability
from facility_fit_agent import prompts
from facility_fit_agent import tools

_T = TypeVar('_T')

_MANDATORY_REPORT_KEYS = frozenset({
    'assessment_id',
    'process_id',
    'target_site',
    'overall_fit_status',
    'discrepancy_matrix',
    'mitigation_recommendations',
})


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
  """Runs an async workflow safely from synchronous CLI or HTTP entrypoints."""
  try:
    asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(coro)
  loop = asyncio.new_event_loop()
  try:
    return loop.run_until_complete(coro)
  finally:
    loop.close()


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
  """Async DAG orchestrator integrating Tools, Persistent DB Memory, & Prompts."""

  def __init__(
      self,
      backend: database.PersistentMemoryBackend | None = None,
      tool_registry: tools.ToolRegistry | None = None,
      precedent_store: memory.EpisodicPrecedentStore | None = None,
      audit_ledger: memory.AssessmentAuditLedger | None = None,
      metrics: observability.MetricsRegistry | None = None,
      prompt_assembler: prompts.PromptContextAssembler | None = None,
      timestamp_provider: datetime.datetime | None = None,
  ):
    """Initializes orchestrator sharing a persistent database backend."""
    shared_backend = backend or database.create_memory_backend()
    self._backend = shared_backend
    self._tools = tool_registry or tools.ToolRegistry()
    self._precedents = precedent_store or memory.EpisodicPrecedentStore(
        backend=shared_backend
    )
    self._ledger = audit_ledger or memory.AssessmentAuditLedger(
        backend=shared_backend
    )
    self._metrics = (
        metrics if metrics is not None else observability.MetricsRegistry()
    )
    self._prompt_assembler = prompt_assembler or prompts.PromptContextAssembler(
        system_prompt=prompts.SYSTEM_PROMPT
    )
    self._advisor = prompts.GeminiConstitutionalAdvisor()
    self._agent = agent.BiopharmaFacilityFitAgent(
        timestamp_provider=timestamp_provider
    )

  @property
  def backend(self) -> database.PersistentMemoryBackend:
    """Returns the shared persistent database backend."""
    return self._backend

  @property
  def tool_registry(self) -> tools.ToolRegistry:
    """Returns the active `ToolRegistry`."""
    return self._tools

  @property
  def audit_ledger(self) -> memory.AssessmentAuditLedger:
    """Returns the persistent ALCOA+ `AssessmentAuditLedger`."""
    return self._ledger

  @property
  def metrics(self) -> observability.MetricsRegistry:
    """Returns the shared `MetricsRegistry`."""
    return self._metrics

  async def execute_assessment_async(
      self,
      process_id: str,
      target_site: str,
      session_context: memory.SessionWorkingContext | None = None,
      sponsor_signoff: str = '',
  ) -> models.FacilityFitAssessmentReport:
    """Asynchronously executes all 5 DAG stages with persistent DB memory."""
    tracer = observability.AgentTracer(metrics=self._metrics)
    self._metrics.increment('assessments_executed_total')

    async with tracer.start_async_span(
        span_name='workflow:facility_fit_assessment',
        stage='ORCHESTRATOR_ROOT',
        attributes={'process_id': process_id, 'target_site': target_site},
    ) as root_span:
      async with tracer.start_async_span(
          span_name='stage_1_ingest_and_normalize',
          stage=OrchestrationStage.STAGE_1_INGEST_AND_NORMALIZE.value,
      ):
        if session_context is not None:
          recipe, overrides = await asyncio.gather(
              self._tools.invoke_async(
                  'ingest_enterprise_pkm_recipe',
                  tracer=tracer,
                  process_id=process_id,
              ),
              session_context.get_active_overrides(),
          )
          if overrides:
            recipe = recipe.with_cpp_overrides(overrides)
        else:
          recipe = await self._tools.invoke_async(
              'ingest_enterprise_pkm_recipe',
              tracer=tracer,
              process_id=process_id,
          )

      async with tracer.start_async_span(
          span_name='stage_2_asset_and_lims_discovery',
          stage=OrchestrationStage.STAGE_2_ASSET_AND_LIMS_DISCOVERY.value,
      ):
        facility, _ = await asyncio.gather(
            self._tools.invoke_async(
                'discover_plant_mes_site_envelope',
                tracer=tracer,
                site_code=target_site,
            ),
            self._precedents.ensure_seeded(),
        )

      async with tracer.start_async_span(
          span_name='stage_3_deterministic_physics',
          stage=OrchestrationStage.STAGE_3_DETERMINISTIC_PHYSICS.value,
      ):
        await asyncio.gather(
            self._tools.invoke_async(
                'compute_bioreactor_hydrodynamics',
                tracer=tracer,
                recipe=recipe,
                bioreactor=facility.bioreactors[0],
            ),
            self._tools.invoke_async(
                'compute_downstream_sizing',
                tracer=tracer,
                recipe=recipe,
                chrom_skid=facility.chromatography_skids[0],
            ),
        )
        base_report = await asyncio.to_thread(
            self._agent.evaluate, recipe, facility
        )

      async with tracer.start_async_span(
          span_name='stage_4_risk_and_precedent_recall',
          stage=OrchestrationStage.STAGE_4_RISK_AND_PRECEDENT_RECALL.value,
      ) as recall_span:
        matched_precedents = await self._precedents.retrieve_precedents(
            site_code=target_site,
            discrepancies=base_report.discrepancy_matrix,
        )
        task_desc = (
            f'Evaluate {process_id} at {target_site} and synthesize GxP '
            'mitigation guidance grounded in deterministic tool outputs.'
        )
        if session_context is not None:
          llm_prompt = await session_context.build_llm_prompt_context(
              user_task=task_desc,
              precedents=matched_precedents,
              tool_declarations=self._tools.export_declarations(),
          )
        else:
          llm_prompt = self._prompt_assembler.build_messages(
              user_task=task_desc,
              compacted_memory_window={'session_id': 'stateless'},
              precedents=matched_precedents,
              tool_declarations=self._tools.export_declarations(),
          )
        advisory_note = await asyncio.to_thread(
            self._advisor.generate_sync, llm_prompt
        )
        recall_span.set_attribute(
            'memory.matched_precedents', len(matched_precedents)
        )
        recall_span.set_attribute(
            'llm.system_prompt_enforced', True
        )
        recall_span.set_attribute(
            'llm.prompt_tokens',
            llm_prompt['token_usage_estimate']['total_tokens'],
        )
        recall_span.set_attribute('llm.advisory_note', advisory_note)
        enriched_report = dataclasses.replace(
            base_report,
            matched_precedents=matched_precedents,
        )

      async with tracer.start_async_span(
          span_name='stage_5_gxp_hitl_and_audit_commit',
          stage=OrchestrationStage.STAGE_5_GXP_HITL_AND_AUDIT_COMMIT.value,
      ):
        if sponsor_signoff:
          enriched_report = self._agent.record_sponsor_signoff(
              enriched_report, sponsor_signoff
          )
        if enriched_report.hitl_signoff_gate.persistence_locked:
          self._metrics.increment('hitl_locks_triggered_total')
        validate_report_contract(enriched_report.to_dict())

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
    await self._ledger.append(final_report, obs_bundle.payload_sha256)

    if session_context is not None:
      await session_context.record_turn_summary(
          f'{process_id}@{target_site} -> '
          f'index={final_report.normalized_fit_index:.2f} '
          f'({final_report.overall_fit_status.value})'
      )
    return final_report

  async def orchestrate_network_comparison_async(
      self,
      process_id: str,
      site_codes: Sequence[str] = (
          'ALPHA_EU_HUB',
          'BETA_US_CAMPUS',
          'GAMMA_US_PLANT',
      ),
      session_context: memory.SessionWorkingContext | None = None,
      sponsor_signoff: str = '',
  ) -> tuple[models.FacilityFitAssessmentReport, ...]:
    """Concurrently evaluates multiple receiving sites via `asyncio.gather`."""
    tasks = [
        self.execute_assessment_async(
            process_id=process_id,
            target_site=site_code,
            session_context=session_context,
            sponsor_signoff=sponsor_signoff,
        )
        for site_code in site_codes
    ]
    results = await asyncio.gather(*tasks)
    return tuple(results)

  async def simulate_what_if_async(
      self,
      session_context: memory.SessionWorkingContext,
      process_id: str,
      target_site: str,
      cpp_overrides: Mapping[str, float],
  ) -> dict[str, Any]:
    """Runs async stateful counterfactual CPP simulation backed by SQLite/DB."""
    baseline = await self.execute_assessment_async(
        process_id=process_id,
        target_site=target_site,
    )
    base_recipe: models.ProcessRecipe = await self._tools.invoke_async(
        'ingest_enterprise_pkm_recipe',
        process_id=process_id,
    )
    for param_name, new_val in cpp_overrides.items():
      base_recipe = await session_context.apply_cpp_override(
          base_recipe,
          parameter_name=param_name,
          new_value=new_val,
      )
    simulated = await self.execute_assessment_async(
        process_id=process_id,
        target_site=target_site,
        session_context=session_context,
    )
    llm_context = await session_context.build_llm_prompt_context(
        user_task=f'Explain sensitivity delta for {process_id} at {target_site}',
        precedents=simulated.matched_precedents,
        tool_declarations=self._tools.export_declarations(),
    )
    return {
        'session_window': await session_context.to_compact_context_window(),
        'llm_prompt_telemetry': llm_context['token_usage_estimate'],
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

  def execute_assessment(
      self,
      process_id: str,
      target_site: str,
      session_context: memory.SessionWorkingContext | None = None,
      sponsor_signoff: str = '',
  ) -> models.FacilityFitAssessmentReport:
    """Synchronous convenience wrapper around `execute_assessment_async`."""
    return _run_sync(
        self.execute_assessment_async(
            process_id=process_id,
            target_site=target_site,
            session_context=session_context,
            sponsor_signoff=sponsor_signoff,
        )
    )

  def orchestrate_network_comparison(
      self,
      process_id: str,
      site_codes: Sequence[str] = (
          'ALPHA_EU_HUB',
          'BETA_US_CAMPUS',
          'GAMMA_US_PLANT',
      ),
      session_context: memory.SessionWorkingContext | None = None,
      sponsor_signoff: str = '',
  ) -> tuple[models.FacilityFitAssessmentReport, ...]:
    """Synchronous wrapper around `orchestrate_network_comparison_async`."""
    return _run_sync(
        self.orchestrate_network_comparison_async(
            process_id=process_id,
            site_codes=site_codes,
            session_context=session_context,
            sponsor_signoff=sponsor_signoff,
        )
    )

  def simulate_what_if(
      self,
      session_context: memory.SessionWorkingContext,
      process_id: str,
      target_site: str,
      cpp_overrides: Mapping[str, float],
  ) -> dict[str, Any]:
    """Synchronous wrapper around `simulate_what_if_async`."""
    return _run_sync(
        self.simulate_what_if_async(
            session_context=session_context,
            process_id=process_id,
            target_site=target_site,
            cpp_overrides=cpp_overrides,
        )
    )
