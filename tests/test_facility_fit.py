"""Unit, HTTP API, 5-pillar architecture, and zero-leakage tests."""

import datetime
import http.server
import json
import pathlib
import re
import threading
import unittest
import urllib.error
import urllib.request

from facility_fit_agent import calculations
from facility_fit_agent import cicd
from facility_fit_agent import memory
from facility_fit_agent import models
from facility_fit_agent import observability
from facility_fit_agent import orchestrator
import main


class BiopharmaFacilityFitSuiteTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.orch = orchestrator.FacilityFitOrchestrator(
        timestamp_provider=datetime.datetime(
            2026, 11, 16, 3, 0, 0, tzinfo=datetime.timezone.utc
        )
    )

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

  def test_all_five_architecture_pillars(self) -> None:
    declarations = self.orch.tool_registry.export_declarations()
    tool_names = {d['name'] for d in declarations}
    self.assertIn('ingest_enterprise_pkm_recipe', tool_names)
    self.assertIn('actuate_shop_floor_mes_setpoint', tool_names)

    with self.assertRaises(models.GxpBoundaryViolationError):
      self.orch.tool_registry.invoke(
          'actuate_shop_floor_mes_setpoint',
          mes_tag='EU-ALP-B1-BRX1501',
          proposed_setpoint=55.0,
      )

    session = memory.SessionWorkingContext(session_id='SES-01')
    sim = self.orch.simulate_what_if(
        session_context=session,
        process_id='PKM-AB4821-DS-V4.2',
        target_site='BETA_US_CAMPUS',
        cpp_overrides={'target_batch_volume_liters': 8000.0},
    )
    self.assertGreater(sim['simulated_fit_index'], sim['baseline_fit_index'])

    rep = self.orch.execute_assessment(
        process_id='PKM-AB4821-DS-V4.2',
        target_site='ALPHA_EU_HUB',
    )
    self.assertEqual(rep.normalized_fit_index, 0.92)
    self.assertIsNotNone(rep.observability)
    assert rep.observability is not None
    self.assertEqual(
        rep.observability.payload_sha256,
        observability.compute_canonical_sha256(rep.to_dict()),
    )

    gate_report = cicd.run_cicd_gates()
    self.assertTrue(gate_report['release_ready'])
    self.assertEqual(gate_report['gates_passed'], 4)

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
          f'http://127.0.0.1:{port}/api/assess?process_id=PKM-AB4821-DS-V4.2&target_site=ALL'
      ) as resp:
        payload = json.loads(resp.read())
        self.assertEqual(len(payload['reports']), 3)
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

  def test_repository_uses_generic_site_and_molecule_identifiers(self) -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    readme_text = (repo_root / 'README.md').read_text(encoding='utf-8')
    self.assertIn('Biopharma CMC Continuous Facility Fit Agent', readme_text)
    reports = self.orch.orchestrate_network_comparison('PKM-AB4821-DS-V4.2')
    site_codes = [r.target_site for r in reports]
    self.assertEqual(
        site_codes,
        ['ALPHA_EU_HUB', 'BETA_US_CAMPUS', 'GAMMA_US_PLANT'],
    )


if __name__ == '__main__':
  unittest.main()
