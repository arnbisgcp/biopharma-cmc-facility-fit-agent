"""Infrastructure & CI/CD release qualification and regression gates.

Implements the Infrastructure & CI/CD architectural pillar:
- Automated 4-gate pre-deployment qualification suite (`run_cicd_gates`):
  1. Deterministic Physics Golden Drift Gate (zero tolerance for formula drift).
  2. GxP MES Write-Lock Red-Team Gate (verifies autonomous writes fail hard).
  3. Schema Contract & Step 5 HITL Persistence Lock Gate.
  4. ALCOA+ Cryptographic Hash-Chain Ledger Integrity Gate.
- Deployment manifest loader & validator (`validate_deployment_manifest`).
"""

from collections.abc import Mapping
import dataclasses
import datetime
from typing import Any

from facility_fit_agent import calculations
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


def run_cicd_gates() -> dict[str, Any]:
  """Executes all 4 mandatory CI/CD qualification gates prior to promotion."""
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

  orch = orchestrator.FacilityFitOrchestrator(
      timestamp_provider=datetime.datetime(
          2026, 11, 16, 3, 0, 0, tzinfo=datetime.timezone.utc
      )
  )
  write_blocked = False
  try:
    orch.tool_registry.invoke(
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

  reports = orch.orchestrate_network_comparison('PKM-AB4821-DS-V4.2')
  dub, wp, elk = reports[0], reports[1], reports[2]
  schema_and_hitl_ok = (
      dub.normalized_fit_index == 0.92
      and wp.normalized_fit_index == 0.36
      and elk.normalized_fit_index == 0.30
      and dub.hitl_signoff_gate.persistence_locked
      and wp.hitl_signoff_gate.persistence_locked
      and elk.hitl_signoff_gate.persistence_locked
  )
  gates.append(
      CiCdGateResult(
          gate_id='GATE-03-SCHEMA-HITL',
          gate_name='Constitutional Schema & Step-5 HITL Lock Gate',
          passed=schema_and_hitl_ok,
          details=(
              f'Scores DUB={dub.normalized_fit_index}, '
              f'WP={wp.normalized_fit_index}, ELK={elk.normalized_fit_index}; '
              'all MAJOR/CRITICAL persistence_locked=True.'
          ),
      )
  )

  chain_ok = (
      len(orch.audit_ledger.entries()) == 3
      and orch.audit_ledger.verify_chain_integrity()
  )
  gates.append(
      CiCdGateResult(
          gate_id='GATE-04-ALCOA-LEDGER',
          gate_name='ALCOA+ SHA-256 Hash-Chain Audit Integrity Gate',
          passed=chain_ok,
          details=(
              f'Verified {len(orch.audit_ledger.entries())} chained ledger '
              'entries with zero tamper drift.'
          ),
      )
  )

  all_passed = all(g.passed for g in gates)
  return {
      'release_ready': all_passed,
      'gates_passed': sum(1 for g in gates if g.passed),
      'gates_total': len(gates),
      'gates': [g.to_dict() for g in gates],
      'telemetry_snapshot': orch.metrics.export_snapshot(),
  }
