"""Data models and strict schemas for the Biopharma CMC Facility Fit Agent."""

from collections.abc import Mapping
import dataclasses
import enum
from typing import Any


class Severity(str, enum.Enum):
  """Constitutional discrepancy severity levels."""

  CRITICAL = 'CRITICAL'
  MAJOR = 'MAJOR'
  MINOR = 'MINOR'


class OverallFitStatus(str, enum.Enum):
  """Normalized facility fit determination statuses."""

  QUALIFIED_FIT = 'QUALIFIED_FIT'
  CONDITIONAL_FIT_CHANGE_CONTROL = 'CONDITIONAL_FIT_CHANGE_CONTROL'
  NOT_FIT_WITHOUT_MODIFICATION = 'NOT_FIT_WITHOUT_MODIFICATION'
  CRITICAL_DISCREPANCIES_IDENTIFIED = 'CRITICAL_DISCREPANCIES_IDENTIFIED'


class ImpellerType(str, enum.Enum):
  """Bioreactor agitation impeller geometries."""

  RUSHTON_TURBINE = 'RUSHTON_TURBINE'
  PITCHED_BLADE = 'PITCHED_BLADE'
  LOW_SHEAR_HYDROFOIL = 'LOW_SHEAR_HYDROFOIL'


class SpargerType(str, enum.Enum):
  """Bioreactor gas sparging configurations."""

  SINTERED_MICRO = 'SINTERED_MICRO'
  DRILLED_PIPE = 'DRILLED_PIPE'
  OPEN_PIPE = 'OPEN_PIPE'


@dataclasses.dataclass(frozen=True)
class CriticalQualityAttribute:
  """Represents a target CQA linked to unit operation CPPs."""

  cqa_id: str
  attribute_name: str
  acceptance_range: str
  sensitive_to: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ProcessRecipe:
  """Normalized process recipe ingested from Enterprise PKM and R&D packages."""

  process_id: str
  molecule_code: str
  molecule_modality: str
  pkm_version: str
  source_scale_liters: float
  target_batch_volume_liters: float
  initial_seed_volume_liters: float
  culture_duration_days: float
  broth_density_kg_m3: float
  broth_viscosity_pa_s: float
  harvest_viscosity_cp: float
  peak_viable_cell_density_e6_ml: float
  specific_o2_uptake_mmol_cell_hr: float
  target_kla_per_hr: float
  max_impeller_tip_speed_m_s: float
  min_kolmogorov_scale_um: float
  target_p_v_min_w_m3: float
  target_p_v_max_w_m3: float
  max_aspect_ratio_h_d: float
  operating_temp_c: float
  cip_sip_temp_c: float
  harvest_titer_g_l: float
  clarification_yield: float
  capture_target_linear_velocity_cm_hr: float
  capture_bed_height_cm: float
  capture_resin_dbc_g_l: float
  capture_max_cycles: int
  max_harvest_hold_hours: float
  downstream_overall_yield: float
  ufdf_start_conc_g_l: float
  ufdf_target_ds_conc_g_l: float
  ufdf_diavolumes: float
  ufdf_avg_flux_lmh: float
  ufdf_max_process_hours: float
  peak_buffer_wfi_lph: float
  required_raw_materials: tuple[tuple[str, str], ...]
  required_lims_methods: tuple[tuple[str, str], ...]
  cqas: tuple[CriticalQualityAttribute, ...]

  def with_cpp_overrides(
      self,
      overrides: Mapping[str, float],
  ) -> 'ProcessRecipe':
    """Returns a copy with validated numeric CPP overrides applied."""
    allowed_fields = {
        f.name
        for f in dataclasses.fields(self)
        if f.type is float or f.type == 'float'
    }
    for key in overrides:
      if key not in allowed_fields:
        raise ValueError(f'Unsupported numeric CPP override field: {key!r}.')
    typed_kwargs: dict[str, Any] = dict(overrides)
    return dataclasses.replace(self, **typed_kwargs)


@dataclasses.dataclass(frozen=True)
class BioreactorAsset:
  """Plant MES bioreactor capability envelope and physical geometry."""

  equipment_id: str
  mes_tag: str
  vessel_type: str
  nominal_volume_liters: float
  min_working_volume_liters: float
  max_working_volume_liters: float
  vessel_diameter_m: float
  impeller_type: ImpellerType
  impeller_count: int
  impeller_diameter_m: float
  power_number_np: float
  min_rpm: float
  max_rpm: float
  sparger_type: SpargerType
  max_aeration_vvm: float
  max_o2_slpm: float
  jacket_area_full_m2: float
  overall_htc_w_m2_k: float
  coolant_supply_temp_c: float
  sip_max_rated_temp_c: float


@dataclasses.dataclass(frozen=True)
class ChromatographySkidAsset:
  """Plant MES chromatography column and skid capability envelope."""

  equipment_id: str
  mes_tag: str
  column_diameter_cm: float
  min_bed_height_cm: float
  max_bed_height_cm: float
  max_linear_velocity_cm_hr: float
  max_operating_pressure_bar: float
  skid_piping_delta_p_bar: float


@dataclasses.dataclass(frozen=True)
class TffSkidAsset:
  """Plant MES tangential flow filtration skid capability envelope."""

  equipment_id: str
  mes_tag: str
  max_membrane_area_m2: float
  min_recirculation_volume_liters: float
  max_transmembrane_pressure_bar: float


@dataclasses.dataclass(frozen=True)
class CleanUtilityEnvelope:
  """Receiving site clean utility peak loop capacities."""

  wfi_loop_tag: str
  available_peak_wfi_lph: float
  concurrent_cip_wfi_draw_lph: float
  clean_steam_available_kg_hr: float
  sip_steam_demand_kg_hr: float


@dataclasses.dataclass(frozen=True)
class LimsMethodStatus:
  """Enterprise LIMS analytical method qualification record at site."""

  method_id: str
  method_name: str
  status: str
  completion_pct: int


@dataclasses.dataclass(frozen=True)
class MaterialQualificationStatus:
  """Site ERP/MES raw material and consumable qualification status."""

  material_code: str
  material_name: str
  qualified: bool
  lead_time_weeks: int
  notes: str


@dataclasses.dataclass(frozen=True)
class SupplyScheduleSlot:
  """Enterprise Supply Planner supply planning PPQ campaign window availability."""

  schedule_id: str
  next_open_window: str
  slot_conflict: bool
  conflict_notes: str


@dataclasses.dataclass(frozen=True)
class FacilityRegistry:
  """Integrated MES, LIMS, and Enterprise Supply Planner snapshot for a receiving site."""

  site_code: str
  site_display_name: str
  location: str
  regulatory_classifications: tuple[str, ...]
  bioreactors: tuple[BioreactorAsset, ...]
  chromatography_skids: tuple[ChromatographySkidAsset, ...]
  tff_skids: tuple[TffSkidAsset, ...]
  utilities: CleanUtilityEnvelope
  lims_methods: tuple[LimsMethodStatus, ...]
  materials: tuple[MaterialQualificationStatus, ...]
  supply_schedule: SupplyScheduleSlot


@dataclasses.dataclass(frozen=True)
class BioprocessPhysicsResult:
  """Deterministic physical engineering values computed for a site match."""

  selected_bioreactor_id: str
  selected_chrom_id: str
  selected_tff_id: str
  selected_rpm: float
  reynolds_number: float
  impeller_tip_speed_m_s: float
  gassed_power_per_volume_w_m3: float
  kolmogorov_scale_um: float
  achievable_kla_per_hr: float
  oxygen_demand_mmol_l_hr: float
  oxygen_transfer_capacity_mmol_l_hr: float
  liquid_height_m: float
  aspect_ratio_h_d: float
  metabolic_plus_agitation_heat_kw: float
  jacket_cooling_capacity_kw: float
  chrom_bed_volume_liters: float
  chrom_operating_velocity_cm_hr: float
  chrom_pressure_drop_bar: float
  chrom_cycles_required: int
  chrom_total_duration_hours: float
  tff_required_area_m2: float
  tff_final_retentate_liters: float
  total_peak_wfi_demand_lph: float

  def to_dict(self) -> dict[str, Any]:
    """Serializes deterministic engineering outputs to a JSON dictionary."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DiscrepancyItem:
  """Single structured discrepancy record in the Discrepancy Matrix."""

  discrepancy_id: str
  unit_operation: str
  fit_dimension: str
  parameter_name: str
  recipe_demand: str
  facility_capability: str
  calculated_value: str
  severity: Severity
  cqa_impact: str
  source_system_tags: tuple[str, ...]

  def to_dict(self) -> dict[str, Any]:
    """Serializes the discrepancy item with string enum values."""
    payload = dataclasses.asdict(self)
    payload['severity'] = self.severity.value
    payload['source_system_tags'] = list(self.source_system_tags)
    return payload


@dataclasses.dataclass(frozen=True)
class MitigationRecommendation:
  """Actionable engineering or quality mitigation linked to discrepancies."""

  mitigation_id: str
  linked_discrepancy_ids: tuple[str, ...]
  category: str
  action_title: str
  engineering_rationale: str
  change_control_required: bool
  estimated_lead_time_weeks: int
  owner_function: str

  def to_dict(self) -> dict[str, Any]:
    """Serializes the mitigation item to a JSON dictionary."""
    payload = dataclasses.asdict(self)
    payload['linked_discrepancy_ids'] = list(self.linked_discrepancy_ids)
    return payload


@dataclasses.dataclass(frozen=True)
class UnitOperationAssessment:
  """Step-by-step fit summary for a discrete unit operation."""

  unit_operation: str
  equipment_id: str
  mes_tag: str
  fit_score: float
  status: str
  checks_passed: tuple[str, ...]
  discrepancy_ids: tuple[str, ...]

  def to_dict(self) -> dict[str, Any]:
    """Serializes unit operation fit details."""
    payload = dataclasses.asdict(self)
    payload['checks_passed'] = list(self.checks_passed)
    payload['discrepancy_ids'] = list(self.discrepancy_ids)
    return payload


class GxpBoundaryViolationError(PermissionError):
  """Raised when any caller attempts autonomous shop-floor MES/SOP writes."""


class SchemaContractError(ValueError):
  """Raised when an output violates the FacilityFitAssessmentReport schema."""


@dataclasses.dataclass(frozen=True)
class HistoricalPrecedent:
  """Institutional memory record of a validated Biopharma tech-transfer CAPA."""

  precedent_id: str
  site_code: str
  year: int
  reference_molecule: str
  discrepancy_pattern: str
  validated_resolution: str
  realized_lead_time_weeks: int

  def to_dict(self) -> dict[str, Any]:
    """Serializes historical precedent record."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TraceSpanRecord:
  """OpenTelemetry-aligned execution span for auditability and tracing."""

  trace_id: str
  span_id: str
  parent_span_id: str
  span_name: str
  stage: str
  status: str
  duration_ms: float
  attributes: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    """Serializes trace span metadata."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ObservabilityBundle:
  """Aggregated tracing, metrics, and ALCOA+ digest for an assessment run."""

  trace_id: str
  total_duration_ms: float
  tool_invocations_count: int
  deterministic_calc_ratio: float
  payload_sha256: str
  spans: tuple[TraceSpanRecord, ...]

  def to_dict(self) -> dict[str, Any]:
    """Serializes observability telemetry."""
    return {
        'trace_id': self.trace_id,
        'total_duration_ms': round(self.total_duration_ms, 3),
        'tool_invocations_count': self.tool_invocations_count,
        'deterministic_calc_ratio': self.deterministic_calc_ratio,
        'payload_sha256': self.payload_sha256,
        'spans': [span.to_dict() for span in self.spans],
    }


@dataclasses.dataclass(frozen=True)
class HitlSignoffGate:
  """Human-in-the-Loop governance gate for GxP decision support."""

  requires_sponsor_signoff: bool
  persistence_locked: bool
  signoff_status: str
  required_approvers: tuple[str, ...]
  governance_rationale: str
  signed_by: str = ''
  signed_timestamp_utc: str = ''

  def to_dict(self) -> dict[str, Any]:
    """Serializes the HITL signoff gate state."""
    payload = dataclasses.asdict(self)
    payload['required_approvers'] = list(self.required_approvers)
    return payload


@dataclasses.dataclass(frozen=True)
class FacilityFitAssessmentReport:
  """Strict constitutional output schema `FacilityFitAssessmentReport`."""

  assessment_id: str
  process_id: str
  target_site: str
  overall_fit_status: OverallFitStatus
  normalized_fit_index: float
  discrepancy_matrix: tuple[DiscrepancyItem, ...]
  mitigation_recommendations: tuple[MitigationRecommendation, ...]
  physics_calculations: BioprocessPhysicsResult
  unit_operation_assessments: tuple[UnitOperationAssessment, ...]
  reasoning_protocol_trace: tuple[str, ...]
  hitl_signoff_gate: HitlSignoffGate
  gxp_boundary_statement: str
  generated_timestamp_utc: str
  matched_precedents: tuple[HistoricalPrecedent, ...] = ()
  observability: ObservabilityBundle | None = None

  def to_dict(self) -> dict[str, Any]:
    """Serializes report strictly adhering to `FacilityFitAssessmentReport`."""
    result: dict[str, Any] = {
        'schema_name': 'FacilityFitAssessmentReport',
        'assessment_id': self.assessment_id,
        'process_id': self.process_id,
        'target_site': self.target_site,
        'overall_fit_status': self.overall_fit_status.value,
        'normalized_fit_index': round(self.normalized_fit_index, 4),
        'severity_counts': {
            'CRITICAL': sum(
                1
                for item in self.discrepancy_matrix
                if item.severity == Severity.CRITICAL
            ),
            'MAJOR': sum(
                1
                for item in self.discrepancy_matrix
                if item.severity == Severity.MAJOR
            ),
            'MINOR': sum(
                1
                for item in self.discrepancy_matrix
                if item.severity == Severity.MINOR
            ),
        },
        'discrepancy_matrix': [
            item.to_dict() for item in self.discrepancy_matrix
        ],
        'mitigation_recommendations': [
            rec.to_dict() for rec in self.mitigation_recommendations
        ],
        'physics_calculations': self.physics_calculations.to_dict(),
        'unit_operation_assessments': [
            op.to_dict() for op in self.unit_operation_assessments
        ],
        'reasoning_protocol_trace': list(self.reasoning_protocol_trace),
        'hitl_signoff_gate': self.hitl_signoff_gate.to_dict(),
        'gxp_boundary_statement': self.gxp_boundary_statement,
        'generated_timestamp_utc': self.generated_timestamp_utc,
        'matched_precedents': [
            prec.to_dict() for prec in self.matched_precedents
        ],
    }
    if self.observability is not None:
      result['observability'] = self.observability.to_dict()
    return result
