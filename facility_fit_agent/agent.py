"""Biopharma CMC Continuous Facility Fit Agent reasoning and assessment engine."""

import dataclasses
import datetime

from facility_fit_agent import calculations
from facility_fit_agent import models

_GXP_BOUNDARY_NOTICE = (
    'NON-GxP DECISION SUPPORT ONLY (FDA CDER/CBER & EU Annex 11 Framework): '
    'This report provides deterministic engineering fit scores and '
    'discrepancy logs. Autonomous shop-floor MES actuation and SOP overrides '
    'are strictly prohibited.'
)


def _select_best_bioreactor(
    recipe: models.ProcessRecipe,
    facility: models.FacilityRegistry,
) -> models.BioreactorAsset:
  """Selects the closest volumetric bioreactor match from Plant MES."""
  best_asset = facility.bioreactors[0]
  best_penalty = float('inf')
  for asset in facility.bioreactors:
    within_bounds = (
        asset.min_working_volume_liters
        <= recipe.target_batch_volume_liters
        <= asset.max_working_volume_liters
    )
    vol_diff = abs(
        asset.max_working_volume_liters - recipe.target_batch_volume_liters
    )
    penalty = vol_diff if within_bounds else vol_diff + 100000.0
    if penalty < best_penalty:
      best_penalty = penalty
      best_asset = asset
  return best_asset


def _select_best_chrom_skid(
    recipe: models.ProcessRecipe,
    facility: models.FacilityRegistry,
) -> models.ChromatographySkidAsset:
  """Selects the chromatography skid closest to target cycle count."""
  best_skid = facility.chromatography_skids[0]
  best_score = float('inf')
  for skid in facility.chromatography_skids:
    sizing = calculations.calculate_chromatography_sizing(recipe, skid)
    cycle_gap = abs(sizing.cycles_required - recipe.capture_max_cycles)
    if cycle_gap < best_score:
      best_score = cycle_gap
      best_skid = skid
  return best_skid


class BiopharmaFacilityFitAgent:
  """Executes the 5-step constitutional facility fit reasoning protocol."""

  def __init__(
      self,
      timestamp_provider: datetime.datetime | None = None,
  ):
    """Initializes the agent with an optional deterministic timestamp."""
    self._fixed_timestamp = timestamp_provider

  def evaluate(
      self,
      recipe: models.ProcessRecipe,
      facility: models.FacilityRegistry,
  ) -> models.FacilityFitAssessmentReport:
    """Runs the 5-step facility fit evaluation for a recipe and site."""
    now_utc = self._fixed_timestamp or datetime.datetime.now(
        datetime.timezone.utc
    )
    iso_now = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

    bioreactor = _select_best_bioreactor(recipe, facility)
    chrom_skid = _select_best_chrom_skid(recipe, facility)
    tff_skid = facility.tff_skids[0]

    physics = self._compute_physics(recipe, facility, bioreactor, chrom_skid)
    discrepancies: list[models.DiscrepancyItem] = []
    mitigations: list[models.MitigationRecommendation] = []

    usp_checks, usp_disc = self._assess_upstream(
        recipe, facility, bioreactor, physics, discrepancies, mitigations
    )
    dsp_checks, dsp_disc = self._assess_downstream(
        recipe,
        facility,
        chrom_skid,
        tff_skid,
        physics,
        discrepancies,
        mitigations,
    )
    util_checks, util_disc = self._assess_utilities_and_readiness(
        recipe, facility, physics, discrepancies, mitigations
    )

    unit_ops = (
        models.UnitOperationAssessment(
            unit_operation='Upstream Seed & Production Bioreactor',
            equipment_id=bioreactor.equipment_id,
            mes_tag=bioreactor.mes_tag,
            fit_score=self._score_subset(discrepancies, usp_disc),
            status='PASS' if not usp_disc else 'DISCREPANCY_FLAGGED',
            checks_passed=tuple(usp_checks),
            discrepancy_ids=tuple(usp_disc),
        ),
        models.UnitOperationAssessment(
            unit_operation='Downstream Capture & UF/DF Purification',
            equipment_id=f'{chrom_skid.equipment_id} / {tff_skid.equipment_id}',
            mes_tag=f'{chrom_skid.mes_tag} | {tff_skid.mes_tag}',
            fit_score=self._score_subset(discrepancies, dsp_disc),
            status='PASS' if not dsp_disc else 'DISCREPANCY_FLAGGED',
            checks_passed=tuple(dsp_checks),
            discrepancy_ids=tuple(dsp_disc),
        ),
        models.UnitOperationAssessment(
            unit_operation='Clean Utilities, Raw Materials & Enterprise LIMS',
            equipment_id=facility.utilities.wfi_loop_tag,
            mes_tag=facility.supply_schedule.schedule_id,
            fit_score=self._score_subset(discrepancies, util_disc),
            status='PASS' if not util_disc else 'DISCREPANCY_FLAGGED',
            checks_passed=tuple(util_checks),
            discrepancy_ids=tuple(util_disc),
        ),
    )

    fit_index, overall_status = self._score_facility(discrepancies)
    hitl_gate = self._build_hitl_gate(discrepancies, facility.site_code)

    trace = (
        (
            f'STEP 1 [Ingest & Normalize]: Parsed {recipe.process_id}'
            f' ({recipe.molecule_code},'
            f' {recipe.target_batch_volume_liters:.0f}L target batch,'
            f' {len(recipe.cqas)} CQAs) from {recipe.pkm_version}.'
        ),
        (
            'STEP 2 [Asset Discovery]: Matched Plant MES assets '
            f'{bioreactor.equipment_id} ({bioreactor.mes_tag}), '
            f'{chrom_skid.equipment_id}, and {tff_skid.equipment_id} at '
            f'{facility.site_display_name}.'
        ),
        (
            'STEP 3 [Unit Op Physics Match]: Executed deterministic transport '
            f'equations -> Re={physics.reynolds_number:,.0f}, '
            f'u_tip={physics.impeller_tip_speed_m_s:.2f} m/s, '
            f'lambda_K={physics.kolmogorov_scale_um:.1f} um, '
            f'kLa={physics.achievable_kla_per_hr:.1f} h^-1, '
            f'Chrom dP={physics.chrom_pressure_drop_bar:.2f} bar.'
        ),
        (
            'STEP 4 [Dynamic Risk Scoring]: Computed normalized fit index '
            f'{fit_index:.2f} ({overall_status.value}) across '
            f'{len(discrepancies)} discrepancies.'
        ),
        (
            f'STEP 5 [HITL Signoff Gate]: {hitl_gate.signoff_status} '
            f'(persistence_locked={hitl_gate.persistence_locked}).'
        ),
    )

    assessment_id = (
        f'FFA-{recipe.molecule_code.replace("-", "")}-'
        f'{facility.site_code[:3]}-20261110'
    )
    return models.FacilityFitAssessmentReport(
        assessment_id=assessment_id,
        process_id=recipe.process_id,
        target_site=facility.site_code,
        overall_fit_status=overall_status,
        normalized_fit_index=fit_index,
        discrepancy_matrix=tuple(discrepancies),
        mitigation_recommendations=tuple(mitigations),
        physics_calculations=physics,
        unit_operation_assessments=unit_ops,
        reasoning_protocol_trace=trace,
        hitl_signoff_gate=hitl_gate,
        gxp_boundary_statement=_GXP_BOUNDARY_NOTICE,
        generated_timestamp_utc=iso_now,
    )

  def record_sponsor_signoff(
      self,
      report: models.FacilityFitAssessmentReport,
      sponsor_name_and_role: str,
  ) -> models.FacilityFitAssessmentReport:
    """Records technical sponsor HITL signoff to unlock report persistence."""
    if not sponsor_name_and_role.strip():
      raise ValueError('Sponsor name and role must be non-empty for signoff.')
    now_utc = self._fixed_timestamp or datetime.datetime.now(
        datetime.timezone.utc
    )
    unlocked_gate = dataclasses.replace(
        report.hitl_signoff_gate,
        persistence_locked=False,
        signoff_status='APPROVED_BY_TECHNICAL_SPONSOR',
        signed_by=sponsor_name_and_role.strip(),
        signed_timestamp_utc=now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
    )
    return dataclasses.replace(report, hitl_signoff_gate=unlocked_gate)

  def _compute_physics(
      self,
      recipe: models.ProcessRecipe,
      facility: models.FacilityRegistry,
      bioreactor: models.BioreactorAsset,
      chrom_skid: models.ChromatographySkidAsset,
  ) -> models.BioprocessPhysicsResult:
    """Invokes deterministic engineering calculators across all unit ops."""
    effective_vol = min(
        recipe.target_batch_volume_liters,
        bioreactor.max_working_volume_liters,
    )
    rpm = calculations.select_optimal_agitation_rpm(recipe, bioreactor)
    re_num = calculations.calculate_reynolds_number(
        density_kg_m3=recipe.broth_density_kg_m3,
        rpm=rpm,
        impeller_diameter_m=bioreactor.impeller_diameter_m,
        viscosity_pa_s=recipe.broth_viscosity_pa_s,
    )
    tip_speed = calculations.calculate_impeller_tip_speed(
        rpm=rpm,
        impeller_diameter_m=bioreactor.impeller_diameter_m,
    )
    p_v = calculations.calculate_gassed_power_per_volume(
        impeller_count=bioreactor.impeller_count,
        power_number_np=bioreactor.power_number_np,
        density_kg_m3=recipe.broth_density_kg_m3,
        rpm=rpm,
        impeller_diameter_m=bioreactor.impeller_diameter_m,
        working_volume_liters=effective_vol,
    )
    kolmogorov_um = calculations.calculate_kolmogorov_microscale_um(
        power_per_volume_w_m3=p_v,
        density_kg_m3=recipe.broth_density_kg_m3,
        viscosity_pa_s=recipe.broth_viscosity_pa_s,
        impeller_type=bioreactor.impeller_type,
    )
    kla = calculations.calculate_kla_per_hour(
        power_per_volume_w_m3=p_v,
        working_volume_liters=effective_vol,
        vessel_diameter_m=bioreactor.vessel_diameter_m,
        aeration_vvm=bioreactor.max_aeration_vvm,
        sparger_type=bioreactor.sparger_type,
    )
    our_demand, otr_cap = calculations.calculate_oxygen_balance(
        peak_vcd_e6_ml=recipe.peak_viable_cell_density_e6_ml,
        specific_o2_mmol_cell_hr=recipe.specific_o2_uptake_mmol_cell_hr,
        kla_per_hr=kla,
        sparger_type=bioreactor.sparger_type,
    )
    liq_h, aspect_ratio = calculations.calculate_liquid_geometry(
        working_volume_liters=effective_vol,
        vessel_diameter_m=bioreactor.vessel_diameter_m,
    )
    heat_kw, cooling_kw = calculations.calculate_thermal_balance_kw(
        working_volume_liters=effective_vol,
        max_working_volume_liters=bioreactor.max_working_volume_liters,
        our_demand_mmol_l_hr=our_demand,
        power_per_volume_w_m3=p_v,
        jacket_area_full_m2=bioreactor.jacket_area_full_m2,
        overall_htc_w_m2_k=bioreactor.overall_htc_w_m2_k,
        operating_temp_c=recipe.operating_temp_c,
        coolant_supply_temp_c=bioreactor.coolant_supply_temp_c,
    )
    chrom_out = calculations.calculate_chromatography_sizing(recipe, chrom_skid)
    tff_out = calculations.calculate_tff_sizing(recipe)
    total_wfi = (
        recipe.peak_buffer_wfi_lph
        + facility.utilities.concurrent_cip_wfi_draw_lph
    )

    return models.BioprocessPhysicsResult(
        selected_bioreactor_id=bioreactor.equipment_id,
        selected_chrom_id=chrom_skid.equipment_id,
        selected_tff_id=facility.tff_skids[0].equipment_id,
        selected_rpm=round(rpm, 1),
        reynolds_number=round(re_num, 1),
        impeller_tip_speed_m_s=round(tip_speed, 3),
        gassed_power_per_volume_w_m3=round(p_v, 2),
        kolmogorov_scale_um=round(kolmogorov_um, 2),
        achievable_kla_per_hr=round(kla, 2),
        oxygen_demand_mmol_l_hr=round(our_demand, 2),
        oxygen_transfer_capacity_mmol_l_hr=round(otr_cap, 2),
        liquid_height_m=round(liq_h, 2),
        aspect_ratio_h_d=round(aspect_ratio, 2),
        metabolic_plus_agitation_heat_kw=round(heat_kw, 2),
        jacket_cooling_capacity_kw=round(cooling_kw, 2),
        chrom_bed_volume_liters=chrom_out.bed_volume_liters,
        chrom_operating_velocity_cm_hr=chrom_out.operating_velocity_cm_hr,
        chrom_pressure_drop_bar=chrom_out.pressure_drop_bar,
        chrom_cycles_required=chrom_out.cycles_required,
        chrom_total_duration_hours=chrom_out.total_duration_hours,
        tff_required_area_m2=tff_out.required_area_m2,
        tff_final_retentate_liters=tff_out.final_retentate_liters,
        total_peak_wfi_demand_lph=round(total_wfi, 1),
    )

  def _assess_upstream(
      self,
      recipe: models.ProcessRecipe,
      facility: models.FacilityRegistry,
      bioreactor: models.BioreactorAsset,
      physics: models.BioprocessPhysicsResult,
      discrepancies: list[models.DiscrepancyItem],
      mitigations: list[models.MitigationRecommendation],
  ) -> tuple[list[str], list[str]]:
    """Evaluates Volumetric, Agitation/Aeration, and Thermal upstream fit."""
    passed: list[str] = []
    disc_ids: list[str] = []
    prefix = facility.site_code[:3]

    if recipe.target_batch_volume_liters > bioreactor.max_working_volume_liters:
      did = f'DISC-{prefix}-VOL-01'
      disc_ids.append(did)
      util_pct = (
          recipe.target_batch_volume_liters
          / bioreactor.max_working_volume_liters
      ) * 100.0
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Production Bioreactor',
              fit_dimension='Volumetric Fit',
              parameter_name='Maximum Working Volume',
              recipe_demand=f'{recipe.target_batch_volume_liters:,.0f} L',
              facility_capability=(
                  f'{bioreactor.max_working_volume_liters:,.0f} L '
                  f'({bioreactor.equipment_id})'
              ),
              calculated_value=f'{util_pct:.1f}% vessel utilization',
              severity=models.Severity.CRITICAL,
              cqa_impact=(
                  'Cannot accommodate full commercial harvest without '
                  'splitting into dual bioreactors and re-validating lot '
                  'pooling comparability.'
              ),
              source_system_tags=(recipe.process_id, bioreactor.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-VOL-01',
              linked_discrepancy_ids=(did,),
              category='Process Scale / Change Control',
              action_title=(
                  'Down-scale batch to 8,000L or execute 2x6,000L dual-vessel '
                  'harvest pooling under Comparability Protocol'
              ),
              engineering_rationale=(
                  'Maintains <=85% working volume headroom for foam/gas '
                  'holdup in Suite H2 vessels.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=12,
              owner_function='MS&T Upstream & Regulatory CMC',
          )
      )
    else:
      passed.append(
          f'Max working volume fit ({recipe.target_batch_volume_liters:,.0f} L '
          f'<= {bioreactor.max_working_volume_liters:,.0f} L)'
      )

    if recipe.initial_seed_volume_liters < bioreactor.min_working_volume_liters:
      did = f'DISC-{prefix}-VOL-02'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Production Bioreactor (Day-0 Inoculation)',
              fit_dimension='Volumetric Fit',
              parameter_name='Minimum Impeller & Probe Submergence Volume',
              recipe_demand=(
                  f'{recipe.initial_seed_volume_liters:,.0f} L Day-0 volume'
              ),
              facility_capability=(
                  f'>= {bioreactor.min_working_volume_liters:,.0f} L min '
                  f'submergence ({bioreactor.equipment_id})'
              ),
              calculated_value=(
                  'Deficit of'
                  f' {bioreactor.min_working_volume_liters - recipe.initial_seed_volume_liters:,.0f}'
                  ' L at Day-0'
              ),
              severity=models.Severity.CRITICAL,
              cqa_impact=(
                  'Impeller gas-flooding, vortex foam shear, and upper pH/DO '
                  'probe exposure on Days 0-3 (CQA-CHG-02, CQA-AGG-04).'
              ),
              source_system_tags=(recipe.process_id, bioreactor.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-VOL-02',
              linked_discrepancy_ids=(did,),
              category='Recipe / Equipment Modification',
              action_title=(
                  'Concentrate nutrient feeds 1.4x to raise Day-0 basal fill '
                  'to >=7,500L or lower bottom impeller hub'
              ),
              engineering_rationale=(
                  'Ensures full lower impeller and redundant pH/DO sensor '
                  'submergence prior to feed initiation.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=10,
              owner_function='R&D Upstream Development & Site Engineering',
          )
      )
    else:
      passed.append(
          f'Day-0 submergence fit ({recipe.initial_seed_volume_liters:,.0f} L '
          f'>= {bioreactor.min_working_volume_liters:,.0f} L)'
      )

    if (
        physics.impeller_tip_speed_m_s > recipe.max_impeller_tip_speed_m_s
        or physics.kolmogorov_scale_um < recipe.min_kolmogorov_scale_um
    ):
      did = f'DISC-{prefix}-SHR-01'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Production Bioreactor',
              fit_dimension='Agitation & Hydrodynamic Shear Fit',
              parameter_name='Impeller Tip Speed & Kolmogorov Eddy Scale',
              recipe_demand=(
                  f'u_tip <= {recipe.max_impeller_tip_speed_m_s:.2f} m/s & '
                  f'lambda_K >= {recipe.min_kolmogorov_scale_um:.1f} um'
              ),
              facility_capability=(
                  f'{bioreactor.impeller_type.value} (Np='
                  f'{bioreactor.power_number_np}, D_i='
                  f'{bioreactor.impeller_diameter_m} m)'
              ),
              calculated_value=(
                  f'u_tip={physics.impeller_tip_speed_m_s:.2f} m/s, '
                  f'lambda_K={physics.kolmogorov_scale_um:.1f} um at '
                  f'{physics.selected_rpm:.0f} RPM (P/V='
                  f'{physics.gassed_power_per_volume_w_m3:.1f} W/m^3)'
              ),
              severity=models.Severity.CRITICAL,
              cqa_impact=(
                  'CHO cell membrane shear damage, elevated LDH/HCP load, and '
                  'reductase-mediated disulfide reduction (CQA-IMP-03, '
                  'CQA-CHG-02).'
              ),
              source_system_tags=(recipe.process_id, bioreactor.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-SHR-01',
              linked_discrepancy_ids=(did,),
              category='Capital Equipment Retrofit',
              action_title=(
                  'Replace Rushton turbines with low-shear 3-blade hydrofoil '
                  'impellers and install sintered micro-sparger'
              ),
              engineering_rationale=(
                  'Reduces Np from 5.0 to ~0.9, cutting local energy '
                  'dissipation by >60% while maintaining kLa >=24 h^-1.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=16,
              owner_function='Site Reliability Engineering & MS&T',
          )
      )
    else:
      passed.append(
          f'Shear envelope safe (u_tip={physics.impeller_tip_speed_m_s:.2f} '
          f'm/s, lambda_K={physics.kolmogorov_scale_um:.1f} um)'
      )

    if physics.achievable_kla_per_hr < recipe.target_kla_per_hr:
      did = f'DISC-{prefix}-KLA-01'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Production Bioreactor',
              fit_dimension='Aeration & Mass Transfer Fit',
              parameter_name='Volumetric O2 Transfer Coefficient (kLa)',
              recipe_demand=f'kLa >= {recipe.target_kla_per_hr:.1f} h^-1',
              facility_capability=(
                  f'{bioreactor.sparger_type.value} @ max '
                  f'{bioreactor.max_aeration_vvm:.3f} vvm'
              ),
              calculated_value=(
                  f'kLa={physics.achievable_kla_per_hr:.1f} h^-1 '
                  f'(OTR={physics.oxygen_transfer_capacity_mmol_l_hr:.2f} vs '
                  f'OUR={physics.oxygen_demand_mmol_l_hr:.2f} mmol/L/h)'
              ),
              severity=models.Severity.CRITICAL,
              cqa_impact=(
                  'Hypoxic DO collapse at peak VCD causing lactate spike and '
                  'aberrant N-glycosylation (CQA-GLY-01).'
              ),
              source_system_tags=(recipe.process_id, bioreactor.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-KLA-01',
              linked_discrepancy_ids=(did,),
              category='Equipment Sparger Upgrade',
              action_title=(
                  'Retrofit 20-um sintered dual-sparger assembly (micro + '
                  'open pipe for CO2 stripping) and upgrade O2 MFC'
              ),
              engineering_rationale=(
                  'Raises mass transfer coefficient by ~1.9x without '
                  'increasing impeller tip speed.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=14,
              owner_function='Site Process Engineering',
          )
      )
    else:
      passed.append(
          f'O2 mass transfer fit (kLa={physics.achievable_kla_per_hr:.1f} '
          f'h^-1 >= {recipe.target_kla_per_hr:.1f} h^-1)'
      )

    if physics.aspect_ratio_h_d > recipe.max_aspect_ratio_h_d:
      did = f'DISC-{prefix}-GEO-01'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Production Bioreactor',
              fit_dimension='Vessel Geometry Fit',
              parameter_name='Liquid Aspect Ratio (H_L / D_T)',
              recipe_demand=f'H_L / D_T <= {recipe.max_aspect_ratio_h_d:.2f}',
              facility_capability=(
                  f'D_T = {bioreactor.vessel_diameter_m:.2f} m '
                  f'({bioreactor.equipment_id})'
              ),
              calculated_value=(
                  f'H_L / D_T = {physics.aspect_ratio_h_d:.2f} '
                  f'(H_L = {physics.liquid_height_m:.2f} m)'
              ),
              severity=models.Severity.MAJOR,
              cqa_impact=(
                  'Elevated hydrostatic pressure and dissolved pCO2 (>140 '
                  'mmHg) shifting galactosylation and osmolality (CQA-GLY-01).'
              ),
              source_system_tags=(recipe.process_id, bioreactor.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-GEO-01',
              linked_discrepancy_ids=(did,),
              category='Process Characterization / Change Control',
              action_title=(
                  'Execute air-sweep CO2 stripping protocol and qualify pH '
                  'deadband adjustment via satellite scale-down model'
              ),
              engineering_rationale=(
                  'Offsets hydrostatic CO2 retention in tall-aspect vessel.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=8,
              owner_function='MS&T Upstream',
          )
      )
    else:
      passed.append(
          f'Aspect ratio fit (H/D={physics.aspect_ratio_h_d:.2f} <= '
          f'{recipe.max_aspect_ratio_h_d:.2f})'
      )

    if recipe.cip_sip_temp_c > bioreactor.sip_max_rated_temp_c:
      did = f'DISC-{prefix}-THM-01'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Production Bioreactor (CIP/SIP)',
              fit_dimension='Thermal & Sterilization Fit',
              parameter_name='SIP Sterilization Temperature Rating',
              recipe_demand=f'{recipe.cip_sip_temp_c:.1f} C SIP hold',
              facility_capability=(
                  f'Max {bioreactor.sip_max_rated_temp_c:.1f} C '
                  f'({bioreactor.equipment_id})'
              ),
              calculated_value=(
                  f'+{recipe.cip_sip_temp_c - bioreactor.sip_max_rated_temp_c:.1f}'
                  ' C above validated SIP envelope'
              ),
              severity=(
                  models.Severity.MINOR
                  if bioreactor.sip_max_rated_temp_c >= 121.0
                  else models.Severity.CRITICAL
              ),
              cqa_impact=(
                  'Requires F0-equivalent SIP hold time extension at 121.5 C '
                  'to maintain sterility assurance without seal degradation.'
              ),
              source_system_tags=(recipe.process_id, bioreactor.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-THM-01',
              linked_discrepancy_ids=(did,),
              category='Procedural Automation Adjustment',
              action_title=(
                  'Adjust SIP phase setpoint to 121.2 C and extend hold by '
                  '8 minutes to preserve equivalent lethality F0 >= 30 min'
              ),
              engineering_rationale=(
                  'Stays within validated elastomer/elastomeric diaphragm '
                  'rating while achieving identical F0.'
              ),
              change_control_required=False,
              estimated_lead_time_weeks=2,
              owner_function='Site Sterility Assurance & Automation',
          )
      )
    elif physics.metabolic_plus_agitation_heat_kw > (
        physics.jacket_cooling_capacity_kw
    ):
      did = f'DISC-{prefix}-THM-02'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Production Bioreactor',
              fit_dimension='Thermal Dynamics Fit',
              parameter_name='Jacket Heat Removal Capacity',
              recipe_demand=(
                  f'{physics.metabolic_plus_agitation_heat_kw:.1f} kW peak load'
              ),
              facility_capability=(
                  f'{physics.jacket_cooling_capacity_kw:.1f} kW jacket capacity'
              ),
              calculated_value='Cooling deficit at peak metabolic activity',
              severity=models.Severity.CRITICAL,
              cqa_impact='Temperature excursion >37.2 C increasing acidic %.',
              source_system_tags=(recipe.process_id, bioreactor.mes_tag),
          )
      )
    else:
      passed.append(
          f'Jacket thermal margin ({physics.jacket_cooling_capacity_kw:.1f} kW '
          f'capacity vs {physics.metabolic_plus_agitation_heat_kw:.1f} kW load)'
      )

    return passed, disc_ids

  def _assess_downstream(
      self,
      recipe: models.ProcessRecipe,
      facility: models.FacilityRegistry,
      chrom_skid: models.ChromatographySkidAsset,
      tff_skid: models.TffSkidAsset,
      physics: models.BioprocessPhysicsResult,
      discrepancies: list[models.DiscrepancyItem],
      mitigations: list[models.MitigationRecommendation],
  ) -> tuple[list[str], list[str]]:
    """Evaluates Protein A Chromatography and UF/DF TFF fit."""
    passed: list[str] = []
    disc_ids: list[str] = []
    prefix = facility.site_code[:3]

    if physics.chrom_pressure_drop_bar > chrom_skid.max_operating_pressure_bar:
      did = f'DISC-{prefix}-CHR-01'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Protein A Capture Chromatography',
              fit_dimension='Downstream Skid Pressure Fit',
              parameter_name='Column + Skid Pressure Drop (Darcy dP)',
              recipe_demand=(
                  f'{recipe.capture_target_linear_velocity_cm_hr:.0f} cm/h '
                  f'across {recipe.capture_bed_height_cm:.0f} cm bed'
              ),
              facility_capability=(
                  f'Max {chrom_skid.max_operating_pressure_bar:.2f} bar '
                  f'({chrom_skid.equipment_id})'
              ),
              calculated_value=(
                  f'dP = {physics.chrom_pressure_drop_bar:.2f} bar at '
                  f'{physics.chrom_operating_velocity_cm_hr:.0f} cm/h'
              ),
              severity=models.Severity.MAJOR,
              cqa_impact=(
                  'High-pressure interlock trip and resin bed compression '
                  'risking channelling and yield loss (CQA-AGG-04).'
              ),
              source_system_tags=(recipe.process_id, chrom_skid.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-CHR-01',
              linked_discrepancy_ids=(did,),
              category='Downstream Parameter / Skid Upgrade',
              action_title=(
                  'Down-rate capture linear flow velocity to <=185 cm/h or '
                  'qualify higher-rating 3.0 bar AxiChrom skid'
              ),
              engineering_rationale=(
                  'Maintains total system dP <= 1.80 bar (90% of skid limit).'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=8,
              owner_function='MS&T Downstream Purification',
          )
      )
    else:
      passed.append(
          f'Chromatography pressure fit ({physics.chrom_pressure_drop_bar:.2f} '
          f'bar <= {chrom_skid.max_operating_pressure_bar:.2f} bar)'
      )

    if physics.chrom_total_duration_hours > recipe.max_harvest_hold_hours:
      did = f'DISC-{prefix}-CHR-02'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Protein A Capture Chromatography',
              fit_dimension='Resin Capacity & Hold-Time Fit',
              parameter_name='Capture Cycle Count & Harvest Hold Duration',
              recipe_demand=(
                  f'<= {recipe.capture_max_cycles} cycles & <= '
                  f'{recipe.max_harvest_hold_hours:.0f} h clarified hold'
              ),
              facility_capability=(
                  f'{chrom_skid.column_diameter_cm:.0f} cm column '
                  f'({physics.chrom_bed_volume_liters:.1f} L resin bed)'
              ),
              calculated_value=(
                  f'{physics.chrom_cycles_required} cycles requiring '
                  f'{physics.chrom_total_duration_hours:.1f} h total run time'
              ),
              severity=models.Severity.MAJOR,
              cqa_impact=(
                  'Exceeds validated clarified harvest stability window, '
                  'increasing proteolysis and acidic variants (CQA-CHG-02).'
              ),
              source_system_tags=(recipe.process_id, chrom_skid.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-CHR-02',
              linked_discrepancy_ids=(did,),
              category='Equipment / Stability Extension',
              action_title=(
                  'Pack 140 cm column or execute chilled (2-8 C) clarified '
                  'harvest hold extension study to 42 hours'
              ),
              engineering_rationale=(
                  'Prevents proteolytic clipping during multi-cycle capture.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=9,
              owner_function='MS&T Downstream & Analytical Sciences',
          )
      )
    elif physics.chrom_cycles_required > recipe.capture_max_cycles:
      did = f'DISC-{prefix}-CHR-03'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Protein A Capture Chromatography',
              fit_dimension='Resin Cycling Fit',
              parameter_name='Capture Cycles per Batch',
              recipe_demand=f'<= {recipe.capture_max_cycles} cycles target',
              facility_capability=(
                  f'{chrom_skid.column_diameter_cm:.0f} cm column '
                  f'({physics.chrom_bed_volume_liters:.1f} L bed)'
              ),
              calculated_value=(
                  f'{physics.chrom_cycles_required} cycles '
                  f'({physics.chrom_total_duration_hours:.1f} h <= '
                  f'{recipe.max_harvest_hold_hours:.0f} h hold limit)'
              ),
              severity=models.Severity.MINOR,
              cqa_impact=(
                  'Within validated harvest stability window and resin '
                  'lifetime; adds ~4.0 hours to suite occupancy.'
              ),
              source_system_tags=(recipe.process_id, chrom_skid.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-CHR-03',
              linked_discrepancy_ids=(did,),
              category='Procedural / Batch Record Parameter',
              action_title=(
                  f'Update MES phase recipe to {physics.chrom_cycles_required} '
                  'equal-mass capture cycles and increase equilibration '
                  'buffer tote volume by 22%'
              ),
              engineering_rationale=(
                  'Total processing remains safely inside validated harvest '
                  'hold time.'
              ),
              change_control_required=False,
              estimated_lead_time_weeks=2,
              owner_function='Site Manufacturing Automation',
          )
      )
    else:
      passed.append(
          f'Capture cycling fit ({physics.chrom_cycles_required} cycles, '
          f'{physics.chrom_total_duration_hours:.1f} h)'
      )

    if physics.tff_required_area_m2 > tff_skid.max_membrane_area_m2:
      did = f'DISC-{prefix}-TFF-01'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='UF/DF Tangential Flow Filtration',
              fit_dimension='TFF Cassette Surface Area Fit',
              parameter_name='Required TFF Membrane Area',
              recipe_demand=(
                  f'{physics.tff_required_area_m2:.1f} m^2 for <= '
                  f'{recipe.ufdf_max_process_hours:.1f} h UF/DF window'
              ),
              facility_capability=(
                  f'Max {tff_skid.max_membrane_area_m2:.1f} m^2 holder '
                  f'({tff_skid.equipment_id})'
              ),
              calculated_value=(
                  'Shortfall of '
                  f'{physics.tff_required_area_m2 - tff_skid.max_membrane_area_m2:.1f} m^2'
              ),
              severity=models.Severity.MAJOR,
              cqa_impact=(
                  'Extends shear pump passes by >38%, elevating sub-visible '
                  'particles and HMW aggregation risk (CQA-AGG-04).'
              ),
              source_system_tags=(recipe.process_id, tff_skid.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-TFF-01',
              linked_discrepancy_ids=(did,),
              category='Process Split / Cassette Holder Expansion',
              action_title=(
                  'Install secondary Pellicon manifold deck (+15 m^2) or '
                  'process CEX eluate in 2 UF/DF sub-lots'
              ),
              engineering_rationale=(
                  'Keeps pump pass count within R&D shear qualification limit.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=7,
              owner_function='MS&T Downstream & Site Engineering',
          )
      )
    else:
      passed.append(
          f'TFF area fit ({physics.tff_required_area_m2:.1f} m^2 <= '
          f'{tff_skid.max_membrane_area_m2:.1f} m^2)'
      )

    if (
        physics.tff_final_retentate_liters
        < tff_skid.min_recirculation_volume_liters
    ):
      did = f'DISC-{prefix}-TFF-02'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='UF/DF Tangential Flow Filtration',
              fit_dimension='Minimum Working Volume Fit',
              parameter_name='Final DS Retentate Volume vs Skid Hold-Up',
              recipe_demand=(
                  f'{physics.tff_final_retentate_liters:.1f} L final pool @ '
                  f'{recipe.ufdf_target_ds_conc_g_l:.0f} g/L'
              ),
              facility_capability=(
                  f'>= {tff_skid.min_recirculation_volume_liters:.1f} L min '
                  f'recirculation ({tff_skid.equipment_id})'
              ),
              calculated_value=(
                  f'{physics.tff_final_retentate_liters:.1f} L < '
                  f'{tff_skid.min_recirculation_volume_liters:.1f} L min loop'
              ),
              severity=models.Severity.CRITICAL,
              cqa_impact=(
                  'Severe air entrainment, foaming, and protein aggregation '
                  'during final over-concentration step (CQA-AGG-04).'
              ),
              source_system_tags=(recipe.process_id, tff_skid.mes_tag),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-TFF-02',
              linked_discrepancy_ids=(did,),
              category='Equipment Selection',
              action_title=(
                  'Deploy portable low-hold-up single-use TFF finish skid '
                  '(<10 L minimum working volume) for final concentration'
              ),
              engineering_rationale=(
                  'Eliminates vortex air ingestion at >160 g/L viscosity.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=10,
              owner_function='MS&T Downstream',
          )
      )
    else:
      passed.append(
          'TFF min retentate volume fit '
          f'({physics.tff_final_retentate_liters:.1f} L >= '
          f'{tff_skid.min_recirculation_volume_liters:.1f} L)'
      )

    return passed, disc_ids

  def _assess_utilities_and_readiness(
      self,
      recipe: models.ProcessRecipe,
      facility: models.FacilityRegistry,
      physics: models.BioprocessPhysicsResult,
      discrepancies: list[models.DiscrepancyItem],
      mitigations: list[models.MitigationRecommendation],
  ) -> tuple[list[str], list[str]]:
    """Evaluates Clean Utilities, Raw Materials, Enterprise LIMS, & Enterprise Supply Planner."""
    passed: list[str] = []
    disc_ids: list[str] = []
    prefix = facility.site_code[:3]

    if (
        physics.total_peak_wfi_demand_lph
        > facility.utilities.available_peak_wfi_lph
    ):
      did = f'DISC-{prefix}-UTL-01'
      disc_ids.append(did)
      deficit = (
          physics.total_peak_wfi_demand_lph
          - facility.utilities.available_peak_wfi_lph
      )
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Clean Utilities (WFI Generation & Loop)',
              fit_dimension='Utility Load Fit',
              parameter_name='Peak Simultaneous WFI Draw',
              recipe_demand=(
                  f'{physics.total_peak_wfi_demand_lph:,.0f} L/h '
                  '(Buffer + Concurrent CIP)'
              ),
              facility_capability=(
                  f'{facility.utilities.available_peak_wfi_lph:,.0f} L/h '
                  f'({facility.utilities.wfi_loop_tag})'
              ),
              calculated_value=f'Peak deficit of {deficit:,.0f} L/h',
              severity=models.Severity.MAJOR,
              cqa_impact=(
                  'Loop pressure drop interlock halting inline chromatography '
                  'buffer dilution during capture run.'
              ),
              source_system_tags=(
                  recipe.process_id,
                  facility.utilities.wfi_loop_tag,
              ),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-UTL-01',
              linked_discrepancy_ids=(did,),
              category='Utility Scheduling / Pre-Make Totes',
              action_title=(
                  'Implement MES CIP interlock stagger and pre-prepare '
                  'Protein A equilibration buffer 12h ahead in 2,500L SU totes'
              ),
              engineering_rationale=(
                  'Shaves 1,550 L/h off simultaneous WFI loop peak draw.'
              ),
              change_control_required=True,
              estimated_lead_time_weeks=5,
              owner_function='Site Utilities & Manufacturing Operations',
          )
      )
    else:
      passed.append(
          f'Peak WFI capacity fit ({physics.total_peak_wfi_demand_lph:,.0f} '
          f'L/h <= {facility.utilities.available_peak_wfi_lph:,.0f} L/h)'
      )

    mat_map = {m.material_code: m for m in facility.materials}
    for mat_code, mat_name in recipe.required_raw_materials:
      site_mat = mat_map.get(mat_code)
      if site_mat is None or not site_mat.qualified:
        did = f'DISC-{prefix}-MAT-{mat_code[-3:]}'
        disc_ids.append(did)
        lead_wks = site_mat.lead_time_weeks if site_mat else 16
        discrepancies.append(
            models.DiscrepancyItem(
                discrepancy_id=did,
                unit_operation='Raw Materials & Single-Use Consumables',
                fit_dimension='Material Compatibility & Qualification',
                parameter_name=f'{mat_code} ({mat_name})',
                recipe_demand='Qualified GMP Vendor & Site Material Master',
                facility_capability=(
                    site_mat.notes if site_mat else 'Missing from site ERP/MES'
                ),
                calculated_value=f'Unqualified ({lead_wks}-week lead time)',
                severity=models.Severity.MAJOR,
                cqa_impact=(
                    'Blocks GMP batch release and raw material incoming QC '
                    'disposition (CQA-IMP-03).'
                ),
                source_system_tags=(recipe.process_id, f'ERP-{mat_code}'),
            )
        )
        mitigations.append(
            models.MitigationRecommendation(
                mitigation_id=f'MIT-{prefix}-MAT-{mat_code[-3:]}',
                linked_discrepancy_ids=(did,),
                category='Quality Raw Material Qualification',
                action_title=(
                    f'Initiate Global Change Control for {mat_code} site '
                    'specification, vendor audit, and incoming ID testing'
                ),
                engineering_rationale=(
                    'Establishes compendial/in-house release spec in SAP/MES.'
                ),
                change_control_required=True,
                estimated_lead_time_weeks=lead_wks,
                owner_function='Site Quality Control & Global Procurement',
            )
        )

    lims_map = {m.method_id: m for m in facility.lims_methods}
    for meth_id, meth_name in recipe.required_lims_methods:
      site_meth = lims_map.get(meth_id)
      if site_meth is None or site_meth.status != 'QUALIFIED':
        did = f'DISC-{prefix}-LIM-{meth_id[-3:]}'
        disc_ids.append(did)
        pct = site_meth.completion_pct if site_meth else 0
        status_label = site_meth.status if site_meth else 'MISSING'
        discrepancies.append(
            models.DiscrepancyItem(
                discrepancy_id=did,
                unit_operation='Analytical QC Release & IPC (Enterprise LIMS)',
                fit_dimension='Analytical Method Readiness',
                parameter_name=f'{meth_id} ({meth_name})',
                recipe_demand='QUALIFIED (100% Co-Validation Complete)',
                facility_capability=f'{status_label} ({pct}% complete)',
                calculated_value=f'{100 - pct}% transfer protocol remaining',
                severity=models.Severity.MAJOR,
                cqa_impact=(
                    'Prevents in-site GMP lot disposition without reference '
                    'lab sample shipment waiver (CQA-IMP-03).'
                ),
                source_system_tags=(recipe.process_id, meth_id),
            )
        )
        mitigations.append(
            models.MitigationRecommendation(
                mitigation_id=f'MIT-{prefix}-LIM-{meth_id[-3:]}',
                linked_discrepancy_ids=(did,),
                category='Enterprise LIMS Analytical Transfer',
                action_title=(
                    f'Complete {meth_id} intermediate precision runs in '
                    'Enterprise LIMS or authorize Central R&D Reference Lab bridge '
                    'testing for PPQ Lot 1'
                ),
                engineering_rationale=(
                    'Closes ICH Q2(R2) method transfer package prior to PPQ.'
                ),
                change_control_required=True,
                estimated_lead_time_weeks=6 if pct >= 50 else 11,
                owner_function='Analytical Method Transfer Lead & Site QC',
            )
        )

    if facility.supply_schedule.slot_conflict:
      did = f'DISC-{prefix}-SCH-01'
      disc_ids.append(did)
      discrepancies.append(
          models.DiscrepancyItem(
              discrepancy_id=did,
              unit_operation='Campaign Supply Planning (Enterprise Supply Planner)',
              fit_dimension='Schedule & Suite Turnaround Fit',
              parameter_name='PPQ Campaign Slot Window',
              recipe_demand='Unconstrained 2027-Q1 PPQ campaign window',
              facility_capability=facility.supply_schedule.next_open_window,
              calculated_value=facility.supply_schedule.conflict_notes,
              severity=models.Severity.MINOR,
              cqa_impact='No direct CQA impact; shifts clinical/PPQ delivery.',
              source_system_tags=(
                  recipe.process_id,
                  facility.supply_schedule.schedule_id,
              ),
          )
      )
      mitigations.append(
          models.MitigationRecommendation(
              mitigation_id=f'MIT-{prefix}-SCH-01',
              linked_discrepancy_ids=(did,),
              category='Supply Chain Planning',
              action_title=(
                  'Re-level Suite H2 changeover window in Enterprise Supply Planner by +14 '
                  'days with Clinical Supply Operations'
              ),
              engineering_rationale=(
                  'Eliminates changeover CIP/SIP collision between campaigns.'
              ),
              change_control_required=False,
              estimated_lead_time_weeks=2,
              owner_function='Enterprise Supply Planner Supply Chain Planner',
          )
      )
    else:
      passed.append(
          'Enterprise Supply Planner campaign window open'
          f' ({facility.supply_schedule.next_open_window})'
      )

    if len(disc_ids) == 0:
      passed.append('All raw materials and Enterprise LIMS methods qualified')
    return passed, disc_ids

  def _score_subset(
      self,
      all_discrepancies: list[models.DiscrepancyItem],
      subset_ids: list[str],
  ) -> float:
    """Computes normalized 0.0-1.0 fit score for a single unit operation."""
    id_set = set(subset_ids)
    penalty = 0.0
    for item in all_discrepancies:
      if item.discrepancy_id in id_set:
        if item.severity == models.Severity.CRITICAL:
          penalty += 0.35
        elif item.severity == models.Severity.MAJOR:
          penalty += 0.14
        else:
          penalty += 0.04
    return round(max(0.0, 1.0 - penalty), 2)

  def _score_facility(
      self,
      discrepancies: list[models.DiscrepancyItem],
  ) -> tuple[float, models.OverallFitStatus]:
    """Computes normalized facility fit index (0.0-1.0) and status."""
    crit = sum(
        1 for d in discrepancies if d.severity == models.Severity.CRITICAL
    )
    major = sum(1 for d in discrepancies if d.severity == models.Severity.MAJOR)
    minor = sum(1 for d in discrepancies if d.severity == models.Severity.MINOR)

    penalty = (crit * 0.18) + (major * 0.06) + (minor * 0.02)
    fit_index = round(max(0.05, 1.0 - penalty), 2) if discrepancies else 1.0

    if crit >= 2:
      status = models.OverallFitStatus.CRITICAL_DISCREPANCIES_IDENTIFIED
    elif crit == 1 or major >= 3:
      status = models.OverallFitStatus.NOT_FIT_WITHOUT_MODIFICATION
    elif major >= 1:
      status = models.OverallFitStatus.CONDITIONAL_FIT_CHANGE_CONTROL
    else:
      status = models.OverallFitStatus.QUALIFIED_FIT
    return fit_index, status

  def _build_hitl_gate(
      self,
      discrepancies: list[models.DiscrepancyItem],
      site_code: str,
  ) -> models.HitlSignoffGate:
    """Enforces Constitutional Step 5 HITL signoff on MAJOR/CRITICAL items."""
    has_blocking = any(
        d.severity in (models.Severity.CRITICAL, models.Severity.MAJOR)
        for d in discrepancies
    )
    if has_blocking:
      return models.HitlSignoffGate(
          requires_sponsor_signoff=True,
          persistence_locked=True,
          signoff_status='PENDING_TECHNICAL_SPONSOR_SIGNOFF',
          required_approvers=(
              f'{site_code} MS&T Technical Sponsor',
              f'{site_code} Site Quality Head (GxP Oversight)',
              'R&D Bioprocess Tech Transfer Lead',
          ),
          governance_rationale=(
              'Constitutional Protocol Step 5: Report contains MAJOR or '
              'CRITICAL discrepancies and cannot be persisted to PKM/MES '
              'planning records without documented human technical sponsor '
              'adjudication.'
          ),
      )
    return models.HitlSignoffGate(
        requires_sponsor_signoff=False,
        persistence_locked=False,
        signoff_status='AUTO_CLEARED_MINOR_OR_CLEAN',
        required_approvers=(f'{site_code} MS&T Engineer',),
        governance_rationale=(
            'Zero CRITICAL or MAJOR discrepancies detected; eligible for '
            'standard MS&T procedural review.'
        ),
    )
