"""Deterministic chemical and bioprocess engineering calculations.

Per Section 2 of the Biopharma CMC Continuous Facility Fit Agent Constitution,
physical engineering values are never estimated heuristically. Every metric
is computed via standard bioprocess transport and hydrodynamic correlations:
- Impeller Reynolds number: Re = (rho * N * D_i^2) / mu
- Impeller tip speed: u_tip = pi * N * D_i
- Gassed power per unit volume: P_g / V_L (Michel/gassed factor on N_p)
- Kolmogorov eddy microscale: lambda_K = (nu^3 / epsilon_max)^(1/4)
- Volumetric O2 mass transfer: Van't Riet correlation k_La = C * (P_g/V)^a *
v_s^b
- Darcy chromatography bed pressure drop and cycle count
- Tangential flow filtration (UF/DF) membrane area and retentate volume
"""

import dataclasses
import math

from facility_fit_agent import models

_GASSED_POWER_FACTOR = 0.82
_LOCAL_DISSIPATION_PEAK_FACTOR = 6.5
_RUSHTON_LOCAL_DISSIPATION_FACTOR = 65.0
_SATURATION_DO_MMOL_L = 0.215
_OPERATING_DO_SETPOINT_FRACTION = 0.40
_METABOLIC_HEAT_W_PER_MMOL_O2_HR = 0.130
_DARCY_RESIN_RESISTANCE_BAR_CM_HR = 0.000185
_PROTEIN_A_SAFETY_LOADING_FACTOR = 0.85
_CHROM_CYCLE_OVERHEAD_HOURS = 1.85


@dataclasses.dataclass(frozen=True)
class ChromatographyCalcOutput:
  """Deterministic chromatography sizing and pressure drop metrics."""

  column_area_cm2: float
  bed_volume_liters: float
  operating_velocity_cm_hr: float
  pressure_drop_bar: float
  cycles_required: int
  total_duration_hours: float


@dataclasses.dataclass(frozen=True)
class TffCalcOutput:
  """Deterministic UF/DF membrane sizing and retentate hold-up metrics."""

  purified_mass_g: float
  initial_pool_liters: float
  final_retentate_liters: float
  total_permeate_liters: float
  required_area_m2: float


def calculate_reynolds_number(
    density_kg_m3: float,
    rpm: float,
    impeller_diameter_m: float,
    viscosity_pa_s: float,
) -> float:
  """Computes impeller Reynolds number Re = (rho * N * D_i^2) / mu."""
  if viscosity_pa_s <= 0:
    raise ValueError(
        f'Dynamic viscosity must be positive; got {viscosity_pa_s}.'
    )
  rps = rpm / 60.0
  return (density_kg_m3 * rps * (impeller_diameter_m**2)) / viscosity_pa_s


def calculate_impeller_tip_speed(
    rpm: float,
    impeller_diameter_m: float,
) -> float:
  """Computes impeller blade tip velocity u_tip = pi * N * D_i (m/s)."""
  rps = rpm / 60.0
  return math.pi * rps * impeller_diameter_m


def calculate_gassed_power_per_volume(
    impeller_count: int,
    power_number_np: float,
    density_kg_m3: float,
    rpm: float,
    impeller_diameter_m: float,
    working_volume_liters: float,
) -> float:
  """Computes gassed mechanical agitation power per unit volume (W/m^3)."""
  if working_volume_liters <= 0:
    raise ValueError(
        f'Working volume must be positive; got {working_volume_liters}.'
    )
  rps = rpm / 60.0
  ungassed_power_watts = (
      impeller_count
      * power_number_np
      * density_kg_m3
      * (rps**3)
      * (impeller_diameter_m**5)
  )
  gassed_power_watts = _GASSED_POWER_FACTOR * ungassed_power_watts
  volume_m3 = working_volume_liters / 1000.0
  return gassed_power_watts / volume_m3


def calculate_kolmogorov_microscale_um(
    power_per_volume_w_m3: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    impeller_type: models.ImpellerType,
) -> float:
  """Computes minimum Kolmogorov turbulence eddy microscale in micrometers."""
  if power_per_volume_w_m3 <= 0:
    return 999.0
  kinematic_viscosity = viscosity_pa_s / density_kg_m3
  peak_factor = (
      _RUSHTON_LOCAL_DISSIPATION_FACTOR
      if impeller_type == models.ImpellerType.RUSHTON_TURBINE
      else _LOCAL_DISSIPATION_PEAK_FACTOR
  )
  epsilon_max_w_kg = (peak_factor * power_per_volume_w_m3) / density_kg_m3
  lambda_meters = ((kinematic_viscosity**3) / epsilon_max_w_kg) ** 0.25
  return lambda_meters * 1.0e6


def calculate_kla_per_hour(
    power_per_volume_w_m3: float,
    working_volume_liters: float,
    vessel_diameter_m: float,
    aeration_vvm: float,
    sparger_type: models.SpargerType,
) -> float:
  """Computes volumetric oxygen mass transfer coefficient kLa (h^-1)."""
  cross_section_m2 = math.pi * ((vessel_diameter_m / 2.0) ** 2)
  gas_flow_m3_s = ((working_volume_liters * aeration_vvm) / 1000.0) / 60.0
  superficial_gas_velocity_m_s = max(gas_flow_m3_s / cross_section_m2, 1.0e-5)

  if sparger_type == models.SpargerType.SINTERED_MICRO:
    coeff = 0.0265
    alpha = 0.44
    beta = 0.48
  elif sparger_type == models.SpargerType.DRILLED_PIPE:
    coeff = 0.0138
    alpha = 0.40
    beta = 0.46
  else:
    coeff = 0.0105
    alpha = 0.38
    beta = 0.45

  kla_per_second = (
      coeff
      * (max(power_per_volume_w_m3, 1.0) ** alpha)
      * (superficial_gas_velocity_m_s**beta)
  )
  return kla_per_second * 3600.0


def calculate_oxygen_balance(
    peak_vcd_e6_ml: float,
    specific_o2_mmol_cell_hr: float,
    kla_per_hr: float,
    sparger_type: models.SpargerType,
) -> tuple[float, float]:
  """Computes peak OUR demand and maximum OTR supply in mmol O2/(L*hr)."""
  cells_per_liter = peak_vcd_e6_ml * 1.0e9
  our_demand = cells_per_liter * specific_o2_mmol_cell_hr
  o2_enrichment_factor = (
      2.35 if sparger_type == models.SpargerType.SINTERED_MICRO else 1.85
  )
  driving_force = (
      _SATURATION_DO_MMOL_L
      * o2_enrichment_factor
      * (1.0 - _OPERATING_DO_SETPOINT_FRACTION)
  )
  otr_capacity = kla_per_hr * driving_force
  return our_demand, otr_capacity


def calculate_liquid_geometry(
    working_volume_liters: float,
    vessel_diameter_m: float,
) -> tuple[float, float]:
  """Computes liquid height (m) and aspect ratio H_L / D_T."""
  volume_m3 = working_volume_liters / 1000.0
  cross_section_m2 = math.pi * ((vessel_diameter_m / 2.0) ** 2)
  liquid_height_m = volume_m3 / cross_section_m2
  aspect_ratio = liquid_height_m / vessel_diameter_m
  return liquid_height_m, aspect_ratio


def calculate_thermal_balance_kw(
    working_volume_liters: float,
    max_working_volume_liters: float,
    our_demand_mmol_l_hr: float,
    power_per_volume_w_m3: float,
    jacket_area_full_m2: float,
    overall_htc_w_m2_k: float,
    operating_temp_c: float,
    coolant_supply_temp_c: float,
) -> tuple[float, float]:
  """Computes peak heat generation (kW) and jacket cooling capacity (kW)."""
  metabolic_watts = (
      our_demand_mmol_l_hr
      * working_volume_liters
      * _METABOLIC_HEAT_W_PER_MMOL_O2_HR
  )
  agitation_watts = power_per_volume_w_m3 * (working_volume_liters / 1000.0)
  total_heat_load_kw = ((metabolic_watts + agitation_watts) * 1.25) / 1000.0

  wetted_fraction = min(working_volume_liters / max_working_volume_liters, 1.0)
  effective_area_m2 = jacket_area_full_m2 * wetted_fraction
  delta_t_k = max(operating_temp_c - coolant_supply_temp_c - 2.5, 1.0)
  jacket_capacity_kw = (overall_htc_w_m2_k * effective_area_m2 * delta_t_k) / (
      1000.0
  )
  return total_heat_load_kw, jacket_capacity_kw


def select_optimal_agitation_rpm(
    recipe: models.ProcessRecipe,
    bioreactor: models.BioreactorAsset,
) -> float:
  """Determines optimal RPM attempting to meet target kLa within vessel bounds."""
  effective_vol = min(
      recipe.target_batch_volume_liters,
      bioreactor.max_working_volume_liters,
  )
  candidate_rpm = bioreactor.min_rpm
  best_rpm = bioreactor.min_rpm
  step_rpm = 0.5
  while candidate_rpm <= bioreactor.max_rpm:
    p_v = calculate_gassed_power_per_volume(
        impeller_count=bioreactor.impeller_count,
        power_number_np=bioreactor.power_number_np,
        density_kg_m3=recipe.broth_density_kg_m3,
        rpm=candidate_rpm,
        impeller_diameter_m=bioreactor.impeller_diameter_m,
        working_volume_liters=effective_vol,
    )
    kla = calculate_kla_per_hour(
        power_per_volume_w_m3=p_v,
        working_volume_liters=effective_vol,
        vessel_diameter_m=bioreactor.vessel_diameter_m,
        aeration_vvm=bioreactor.max_aeration_vvm,
        sparger_type=bioreactor.sparger_type,
    )
    best_rpm = candidate_rpm
    if kla >= recipe.target_kla_per_hr and p_v >= recipe.target_p_v_min_w_m3:
      break
    candidate_rpm += step_rpm
  return round(best_rpm, 1)


def calculate_chromatography_sizing(
    recipe: models.ProcessRecipe,
    skid: models.ChromatographySkidAsset,
) -> ChromatographyCalcOutput:
  """Computes Protein A column bed volume, Darcy delta-P, and cycle count."""
  column_area_cm2 = math.pi * ((skid.column_diameter_cm / 2.0) ** 2)
  bed_height_cm = min(
      max(recipe.capture_bed_height_cm, skid.min_bed_height_cm),
      skid.max_bed_height_cm,
  )
  bed_volume_liters = (column_area_cm2 * bed_height_cm) / 1000.0

  operating_velocity = min(
      recipe.capture_target_linear_velocity_cm_hr,
      skid.max_linear_velocity_cm_hr,
  )
  bed_delta_p = (
      _DARCY_RESIN_RESISTANCE_BAR_CM_HR
      * recipe.harvest_viscosity_cp
      * operating_velocity
      * bed_height_cm
  )
  total_delta_p = bed_delta_p + skid.skid_piping_delta_p_bar

  clarified_mass_g = (
      recipe.target_batch_volume_liters
      * recipe.harvest_titer_g_l
      * recipe.clarification_yield
  )
  effective_capacity_g = (
      bed_volume_liters
      * recipe.capture_resin_dbc_g_l
      * _PROTEIN_A_SAFETY_LOADING_FACTOR
  )
  cycles_required = max(1, math.ceil(clarified_mass_g / effective_capacity_g))

  flow_rate_lph = (operating_velocity * column_area_cm2) / 1000.0
  load_vol_per_cycle_l = recipe.target_batch_volume_liters / cycles_required
  load_hours_per_cycle = load_vol_per_cycle_l / max(flow_rate_lph, 1.0)
  total_duration_hours = cycles_required * (
      load_hours_per_cycle + _CHROM_CYCLE_OVERHEAD_HOURS
  )

  return ChromatographyCalcOutput(
      column_area_cm2=round(column_area_cm2, 2),
      bed_volume_liters=round(bed_volume_liters, 2),
      operating_velocity_cm_hr=round(operating_velocity, 2),
      pressure_drop_bar=round(total_delta_p, 3),
      cycles_required=cycles_required,
      total_duration_hours=round(total_duration_hours, 2),
  )


def calculate_tff_sizing(recipe: models.ProcessRecipe) -> TffCalcOutput:
  """Computes UF/DF permeate load, membrane area, and final retentate volume."""
  purified_mass_g = (
      recipe.target_batch_volume_liters
      * recipe.harvest_titer_g_l
      * recipe.clarification_yield
      * recipe.downstream_overall_yield
  )
  initial_pool_liters = purified_mass_g / recipe.ufdf_start_conc_g_l
  final_retentate_liters = purified_mass_g / recipe.ufdf_target_ds_conc_g_l
  uf_permeate_liters = max(initial_pool_liters - final_retentate_liters, 0.0)
  df_permeate_liters = final_retentate_liters * recipe.ufdf_diavolumes
  total_permeate_liters = uf_permeate_liters + df_permeate_liters
  required_area_m2 = total_permeate_liters / (
      recipe.ufdf_avg_flux_lmh * recipe.ufdf_max_process_hours
  )
  return TffCalcOutput(
      purified_mass_g=round(purified_mass_g, 1),
      initial_pool_liters=round(initial_pool_liters, 2),
      final_retentate_liters=round(final_retentate_liters, 2),
      total_permeate_liters=round(total_permeate_liters, 2),
      required_area_m2=round(required_area_m2, 2),
  )
