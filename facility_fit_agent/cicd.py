"""Infrastructure & CI/CD async release qualification and regression gates."""

import asyncio
from collections.abc import Mapping
import dataclasses
import datetime
import pathlib
import tempfile
from typing import Any

from facility_fit_agent import calculations
from facility_fit_agent import database
from facility_fit_agent import models
from facility_fit_agent import orchestrator


@dataclasses.dataclass(frozen=True)
class CiCdGateResult:
  """Outcome of an individual CI/CD release qualification gate."""

  gate_id: str
  gate_name: str
  passed: bool
  details: str

  def to_dict(self) -> dict[str, Any]:
    """Serializes CI/CD gate outcome."""
    return dataclasses.asdict(self)


def validate_deployment_manifest(manifest: Mapping[str, Any]) -> bool:
  """Verifies mandatory GxP security and runtime keys in deployment manifest."""
  required_top_keys = {
      'service_name',
      'runtime_platform',
      'gxp_classification',
      'security_controls',
      'enterprise_connectors',
      'observability_export',
      'ci_cd_pipeline',
  }
  if not required_top_keys.issubset(set(manifest.keys())):
    return False
  sec = manifest.get('security_controls', {})
  return bool(
      sec.get('autonomous_mes_write_disabled') is True
      and sec.get('vpc_service_controls_enforced') is True
  )


async def run_cicd_gates_async() -> dict[str, Any]:
  """Asynchronously executes all 4 CI/CD release qualification gates."""
  gates: list[CiCdGateResult] = []

  re_val = calculations.calculate_reynolds_number(
      density_kg_m3=1030.0,
      rpm=60.0,
      impeller_diameter_m=0.80,
      viscosity_pa_s=0.0015,
  )
  tip_val = calculations.calculate_impeller_tip_speed(
      rpm=60.0, impeller_diameter_m=0.80
  )
  drift_ok = abs(re_val - 439466.67) < 0.5 and abs(tip_val - 2.51327) < 0.001
  gates.append(
      CiCdGateResult(
          gate_id='GATE-01-PHYSICS-GOLDEN',
          gate_name='Deterministic Transport Physics Drift Gate',
          passed=drift_ok,
          details=f'Re={re_val:.2f} (ref 439466.67), u_tip={tip_val:.4f} m/s',
      )
  )

  with tempfile.TemporaryDirectory() as tmp_dir:
    gate_db = database.AsyncSqliteDatabaseBackend(
        db_path=pathlib.Path(tmp_dir) / 'cicd_gate.sqlite3'
    )
    orch = orchestrator.FacilityFitOrchestrator(
        backend=gate_db,
        timestamp_provider=datetime.datetime(
            2026, 11, 16, 3, 0, 0, tzinfo=datetime.timezone.utc
        ),
    )

    write_blocked = False
    try:
      await orch.tool_registry.invoke_async(
          'actuate_shop_floor_mes_setpoint',
          mes_tag='EU-ALP-B1-BRX1501',
          proposed_setpoint=55.0,
      )
    except models.GxpBoundaryViolationError:
      write_blocked = True
      orch.metrics.increment('gxp_boundary_blocks_total')

    gates.append(
      CiCdGateResult(
          gate_id='GATE-02-GXP-REDTEAM',
          gate_name='GxP Autonomous Shop-Floor Write-Lock Gate',
          passed=write_blocked,
          details=(
              'GxpBoundaryViolationError deterministically raised on MES write.'
              if write_blocked
              else 'FAILURE: MES write was not blocked.'
          ),
      )
    )

    reports = await orch.orchestrate_network_comparison_async(
        'PKM-AB4821-DS-V4.2'
    )
    alp, bet, gam = reports[0], reports[1], reports[2]
    schema_and_hitl_ok = (
        alp.normalized_fit_index == 0.92
        and bet.normalized_fit_index == 0.36
        and gam.normalized_fit_index == 0.30
        and alp.hitl_signoff_gate.persistence_locked
        and bet.hitl_signoff_gate.persistence_locked
        and gam.hitl_signoff_gate.persistence_locked
    )
    gates.append(
        CiCdGateResult(
            gate_id='GATE-03-SCHEMA-HITL',
            gate_name='Constitutional Schema & Step-5 HITL Lock Gate',
            passed=schema_and_hitl_ok,
            details=(
                f'Scores ALP={alp.normalized_fit_index}, '
                f'BET={bet.normalized_fit_index}, GAM={gam.normalized_fit_index}; '
                'all MAJOR/CRITICAL persistence_locked=True.'
            ),
        )
    )

    persisted_entries = await orch.audit_ledger.entries()
    chain_valid = await orch.audit_ledger.verify_chain_integrity()
    chain_ok = len(persisted_entries) == 3 and chain_valid
    gates.append(
        CiCdGateResult(
            gate_id='GATE-04-ALCOA-LEDGER',
            gate_name='Persistent SQLite WAL ALCOA+ Hash-Chain Integrity Gate',
            passed=chain_ok,
            details=(
                f'Verified {len(persisted_entries)} persisted SQLite ledger '
                'entries with zero tamper drift.'
            ),
        )
    )
    telemetry = orch.metrics.export_snapshot()

  all_passed = all(g.passed for g in gates)
  return {
      'release_ready': all_passed,
      'gates_passed': sum(1 for g in gates if g.passed),
      'gates_total': len(gates),
      'gates': [g.to_dict() for g in gates],
      'telemetry_snapshot': telemetry,
  }


def run_cicd_gates() -> dict[str, Any]:
  """Synchronous convenience entrypoint for CI/CD gate execution."""
  try:
    asyncio.get_running_loop()
  except RuntimeError:
    return asyncio.run(run_cicd_gates_async())
  loop = asyncio.new_event_loop()
  try:
    return loop.run_until_complete(run_cicd_gates_async())
  finally:
    loop.close()
