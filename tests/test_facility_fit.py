"""Async unit, persistent SQLite DB, prompt assembly, and HTTP API tests."""

import datetime
import http.server
import json
import pathlib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from facility_fit_agent import calculations
from facility_fit_agent import catalog
from facility_fit_agent import cicd
from facility_fit_agent import database
from facility_fit_agent import memory
from facility_fit_agent import models
from facility_fit_agent import observability
from facility_fit_agent import orchestrator
from facility_fit_agent import prompts
import main


class AsyncContextAndMemoryPersistenceTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self) -> None:
    self._tmp_dir = tempfile.TemporaryDirectory()
    self.db_file = pathlib.Path(self._tmp_dir.name) / 'test_memory.sqlite3'
    self.backend = database.AsyncSqliteDatabaseBackend(db_path=self.db_file)
    self.orch = orchestrator.FacilityFitOrchestrator(
        backend=self.backend,
        timestamp_provider=datetime.datetime(
            2026, 11, 16, 3, 0, 0, tzinfo=datetime.timezone.utc
        ),
    )

  async def asyncTearDown(self) -> None:
    self._tmp_dir.cleanup()

  async def test_system_prompt_and_persistent_sqlite_survival_across_restarts(
      self,
  ) -> None:
    session_1 = memory.SessionWorkingContext(
        session_id='SES-PERSIST-01',
        operator_id='msat_lead',
        backend=self.backend,
        max_token_budget=120,
        keep_recent_turns=3,
    )
    recipe = catalog.get_process_recipes()['PKM-AB4821-DS-V4.2']
    await session_1.apply_cpp_override(
        recipe,
        parameter_name='target_batch_volume_liters',
        new_value=8000.0,
    )
    for idx in range(7):
      await session_1.record_turn_summary(
          f'Historical sensitivity iteration #{idx}: verified shear and kLa '
          'margins across candidate vessel geometries.'
      )

    # Instantiate a brand-new backend and session pointing to the same disk DB
    reopened_backend = database.AsyncSqliteDatabaseBackend(db_path=self.db_file)
    session_reopened = memory.SessionWorkingContext(
        session_id='SES-PERSIST-01',
        backend=reopened_backend,
        max_token_budget=120,
        keep_recent_turns=3,
    )
    overrides = await session_reopened.get_active_overrides()
    self.assertEqual(overrides['target_batch_volume_liters'], 8000.0)

    window = await session_reopened.to_compact_context_window()
    self.assertGreater(window['compacted_turns_count'], 0)
    self.assertTrue(window['rolling_summary'])
    self.assertLessEqual(len(window['recent_turns']), 3)

    llm_payload = await session_reopened.build_llm_prompt_context(
        user_task='Assess 8,000L down-scaled batch at Beta Campus.',
        tool_declarations=self.orch.tool_registry.export_declarations(),
    )
    self.assertIn(
        'BIOPHARMA CMC CONTINUOUS FACILITY FIT AGENT',
        llm_payload['system_instruction']['content'],
    )
    self.assertIn(
        'FacilityFitAssessmentReport',
        llm_payload['system_instruction']['content'],
    )
    self.assertTrue(llm_payload['token_usage_estimate']['within_budget'])

  async def test_async_orchestrator_what_if_and_persistent_ledger_chain(
      self,
  ) -> None:
    session = memory.SessionWorkingContext(
        session_id='SES-WHATIF-02',
        backend=self.backend,
    )
    sim = await self.orch.simulate_what_if_async(
        session_context=session,
        process_id='PKM-AB4821-DS-V4.2',
        target_site='BETA_US_CAMPUS',
        cpp_overrides={'target_batch_volume_liters': 8000.0},
    )
    self.assertGreater(sim['simulated_fit_index'], sim['baseline_fit_index'])
    self.assertIn('llm_prompt_telemetry', sim)

    reports = await self.orch.orchestrate_network_comparison_async(
        'PKM-AB4821-DS-V4.2'
    )
    self.assertEqual(len(reports), 3)
    self.assertEqual(reports[0].normalized_fit_index, 0.92)
    assert reports[0].observability is not None
    self.assertEqual(
        reports[0].observability.payload_sha256,
        observability.compute_canonical_sha256(reports[0].to_dict()),
    )

    reopened_ledger = memory.AssessmentAuditLedger(
        backend=database.AsyncSqliteDatabaseBackend(db_path=self.db_file)
    )
    entries = await reopened_ledger.entries()
    self.assertEqual(len(entries), 5)
    self.assertTrue(await reopened_ledger.verify_chain_integrity())

  async def test_async_tool_validation_and_cicd_gates(self) -> None:
    with self.assertRaises(models.GxpBoundaryViolationError):
      await self.orch.tool_registry.invoke_async(
          'actuate_shop_floor_mes_setpoint',
          mes_tag='EU-ALP-B1-BRX1501',
          proposed_setpoint=55.0,
      )
    with self.assertRaises(ValueError):
      await self.orch.tool_registry.invoke_async(
          'ingest_enterprise_pkm_recipe',
          undeclared_arg='bad',
      )
    gates = await cicd.run_cicd_gates_async()
    self.assertTrue(gates['release_ready'])
    self.assertEqual(gates['gates_passed'], 4)


class PhysicsAndHttpServerTest(unittest.TestCase):

  def test_deterministic_physics_equations(self) -> None:
    re_val = calculations.calculate_reynolds_number(
        density_kg_m3=1030.0,
        rpm=60.0,
        impeller_diameter_m=0.80,
        viscosity_pa_s=0.0015,
    )
    tip_val = calculations.calculate_impeller_tip_speed(
        rpm=60.0,
        impeller_diameter_m=0.80,
    )
    self.assertAlmostEqual(re_val, 439466.67, places=1)
    self.assertAlmostEqual(tip_val, 2.5133, places=3)
    self.assertGreater(prompts.estimate_token_count(prompts.SYSTEM_PROMPT), 100)

  def test_http_server_endpoints(self) -> None:
    srv = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), main.FacilityFitRequestHandler
    )
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
      with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz') as resp:
        hz = json.loads(resp.read())
        self.assertEqual(hz['status'], 'ok')
      with urllib.request.urlopen(
          f'http://127.0.0.1:{port}/api/system-prompt'
      ) as resp:
        sp = json.loads(resp.read())
        self.assertIn('FacilityFitAssessmentReport', sp['system_prompt'])
      req = urllib.request.Request(
          f'http://127.0.0.1:{port}/api/redteam-mes-write',
          data=b'{"mes_tag":"EU-ALP-B1-BRX1501","proposed_setpoint":60}',
          headers={'Content-Type': 'application/json'},
          method='POST',
      )
      with self.assertRaises(urllib.error.HTTPError) as ctx:
        urllib.request.urlopen(req)
      self.assertEqual(ctx.exception.code, 403)
    finally:
      srv.shutdown()
      srv.server_close()


if __name__ == '__main__':
  unittest.main()
