"""Tool & Interface Design layer for the Biopharma CMC Facility Fit Agent.

Implements the Tool & Interface Design architectural pillar:
- Strictly typed connector `Protocol` contracts for Enterprise PKM, Plant MES,
  Enterprise LIMS, and Enterprise Supply Planner.
- Schema-validated `ToolDeclaration` metadata exportable as Vertex AI / Gemini
  ADK `FunctionDeclaration` contracts.
- Hard GxP Read-Only Operational Boundary guardrails (`PlantMesReadOnlyTool`)
  that deterministically block any shop-floor setpoint write or SOP override.
- Deterministic Bioprocess Physics Tool wrappers around `calculations.py`.
"""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, Protocol

from facility_fit_agent import calculations
from facility_fit_agent import catalog
from facility_fit_agent import models
from facility_fit_agent import observability


class PkmConnector(Protocol):
  """Protocol interface for read-only Enterprise PKM recipe ingestion."""

  def fetch_recipe(self, process_id: str) -> models.ProcessRecipe:
    """Returns normalized `ProcessRecipe` for `process_id`."""
    ...


class PlantMesConnector(Protocol):
  """Protocol interface for read-only Plant MES & LIMS site discovery."""

  def fetch_site_registry(self, site_code: str) -> models.FacilityRegistry:
    """Returns verified `FacilityRegistry` for `site_code`."""
    ...


@dataclasses.dataclass(frozen=True)
class ToolDeclaration:
  """Vertex AI / ADK-compatible tool contract specification."""

  name: str
  description: str
  read_only_gxp_safe: bool
  parameters_json_schema: dict[str, Any]
  returns_description: str

  def to_dict(self) -> dict[str, Any]:
    """Serializes the tool interface contract to JSON."""
    return dataclasses.asdict(self)


class CatalogPkmConnector:
  """Default in-memory Enterprise PKM connector backed by verified catalog."""

  def __init__(
      self,
      recipes: Mapping[str, models.ProcessRecipe] | None = None,
  ):
    """Initializes connector with optional injected recipe map."""
    self._recipes = (
        dict(recipes) if recipes is not None else catalog.get_process_recipes()
    )

  def fetch_recipe(self, process_id: str) -> models.ProcessRecipe:
    """Retrieves a normalized PKM recipe or raises KeyError."""
    if process_id not in self._recipes:
      available = ', '.join(sorted(self._recipes.keys()))
      raise KeyError(
          f'Unknown Enterprise PKM process_id {process_id!r}; '
          f'available: {available}.'
      )
    return self._recipes[process_id]


class CatalogMesConnector:
  """Default read-only Plant MES / Enterprise LIMS / Enterprise Supply Planner site connector."""

  def __init__(
      self,
      sites: Mapping[str, models.FacilityRegistry] | None = None,
  ):
    """Initializes connector with optional injected site map."""
    self._sites = (
        dict(sites) if sites is not None else catalog.get_facility_registries()
    )

  def fetch_site_registry(self, site_code: str) -> models.FacilityRegistry:
    """Retrieves verified receiving facility capabilities."""
    if site_code not in self._sites:
      available = ', '.join(sorted(self._sites.keys()))
      raise KeyError(
          f'Unknown receiving site_code {site_code!r}; available: {available}.'
      )
    return self._sites[site_code]


class ToolRegistry:
  """Central registry enforcing schemas, tracing, and GxP write-locks."""

  def __init__(
      self,
      pkm_connector: PkmConnector | None = None,
      mes_connector: PlantMesConnector | None = None,
  ):
    """Initializes the tool registry and registers canonical CMC tools."""
    self._pkm = pkm_connector or CatalogPkmConnector()
    self._mes = mes_connector or CatalogMesConnector()
    self._declarations: dict[str, ToolDeclaration] = {}
    self._handlers: dict[str, Callable[..., Any]] = {}
    self._register_builtin_tools()

  def register_tool(
      self,
      declaration: ToolDeclaration,
      handler: Callable[..., Any],
  ) -> None:
    """Registers a typed tool contract and execution handler."""
    self._declarations[declaration.name] = declaration
    self._handlers[declaration.name] = handler

  def export_declarations(self) -> list[dict[str, Any]]:
    """Exports all registered tool schemas for ADK / Gemini function-calling."""
    return [
        self._declarations[name].to_dict()
        for name in sorted(self._declarations.keys())
    ]

  def invoke(
      self,
      tool_name: str,
      tracer: observability.AgentTracer | None = None,
      **kwargs: Any,
  ) -> Any:
    """Invokes a registered tool inside an observability trace span."""
    if tool_name not in self._handlers:
      raise KeyError(f'Unregistered tool {tool_name!r}.')
    handler = self._handlers[tool_name]

    if tracer is None:
      return handler(**kwargs)

    tracer.metrics.increment('tool_invocations_total')
    tracer.metrics.increment(f'tool.{tool_name}.calls')
    with tracer.start_span(
        span_name=f'tool:{tool_name}',
        stage='TOOL_INVOCATION',
        attributes={'tool.name': tool_name, 'tool.arg_keys': sorted(kwargs)},
    ):
      return handler(**kwargs)

  def _register_builtin_tools(self) -> None:
    """Registers PKM, MES, LIMS, Physics, and GxP Write-Lock guard tools."""
    self.register_tool(
        ToolDeclaration(
            name='ingest_enterprise_pkm_recipe',
            description=(
                'Fetches and normalizes unit operations, CPPs, and CQAs from '
                'Enterprise Process Knowledge Management (PKM).'
            ),
            read_only_gxp_safe=True,
            parameters_json_schema={
                'type': 'object',
                'properties': {'process_id': {'type': 'string'}},
                'required': ['process_id'],
            },
            returns_description='Normalized ProcessRecipe dataclass.',
        ),
        self._pkm.fetch_recipe,
    )

    self.register_tool(
        ToolDeclaration(
            name='discover_plant_mes_site_envelope',
            description=(
                'Retrieves verified bioreactor, chromatography, TFF, clean '
                'utility, Enterprise LIMS, and Enterprise Supply Planner envelopes for a site.'
            ),
            read_only_gxp_safe=True,
            parameters_json_schema={
                'type': 'object',
                'properties': {'site_code': {'type': 'string'}},
                'required': ['site_code'],
            },
            returns_description='Verified FacilityRegistry snapshot.',
        ),
        self._mes.fetch_site_registry,
    )

    self.register_tool(
        ToolDeclaration(
            name='compute_bioreactor_hydrodynamics',
            description=(
                'Deterministically computes impeller Reynolds number, tip '
                'speed, gassed P/V, Kolmogorov eddy microscale, and Van t Riet '
                'kLa without heuristic estimation.'
            ),
            read_only_gxp_safe=True,
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'recipe': {'type': 'object'},
                    'bioreactor': {'type': 'object'},
                },
                'required': ['recipe', 'bioreactor'],
            },
            returns_description='Dictionary of deterministic upstream metrics.',
        ),
        self._compute_upstream_hydrodynamics,
    )

    self.register_tool(
        ToolDeclaration(
            name='compute_downstream_sizing',
            description=(
                'Deterministically computes Protein A Darcy pressure drop, '
                'cycle count, and UF/DF TFF membrane area and hold-up volume.'
            ),
            read_only_gxp_safe=True,
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'recipe': {'type': 'object'},
                    'chrom_skid': {'type': 'object'},
                },
                'required': ['recipe', 'chrom_skid'],
            },
            returns_description=(
                'Tuple of ChromatographyCalcOutput, TffCalcOutput.'
            ),
        ),
        self._compute_downstream_sizing,
    )

    self.register_tool(
        ToolDeclaration(
            name='actuate_shop_floor_mes_setpoint',
            description=(
                'GUARDED WRITE-LOCK ENDPOINT: Always blocks autonomous MES or '
                'SOP modifications per GxP Constitution Section 2.'
            ),
            read_only_gxp_safe=False,
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'mes_tag': {'type': 'string'},
                    'proposed_setpoint': {'type': 'number'},
                },
                'required': ['mes_tag', 'proposed_setpoint'],
            },
            returns_description='Always raises GxpBoundaryViolationError.',
        ),
        self._block_mes_actuation,
    )

  def _compute_upstream_hydrodynamics(
      self,
      recipe: models.ProcessRecipe,
      bioreactor: models.BioreactorAsset,
  ) -> dict[str, float]:
    """Executes deterministic upstream hydrodynamic transport calculations."""
    effective_vol = min(
        recipe.target_batch_volume_liters,
        bioreactor.max_working_volume_liters,
    )
    rpm = calculations.select_optimal_agitation_rpm(recipe, bioreactor)
    reynolds = calculations.calculate_reynolds_number(
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
    kolmogorov = calculations.calculate_kolmogorov_microscale_um(
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
    return {
        'rpm': rpm,
        'reynolds_number': round(reynolds, 1),
        'impeller_tip_speed_m_s': round(tip_speed, 3),
        'gassed_power_per_volume_w_m3': round(p_v, 2),
        'kolmogorov_scale_um': round(kolmogorov, 2),
        'achievable_kla_per_hr': round(kla, 2),
    }

  def _compute_downstream_sizing(
      self,
      recipe: models.ProcessRecipe,
      chrom_skid: models.ChromatographySkidAsset,
  ) -> tuple[calculations.ChromatographyCalcOutput, calculations.TffCalcOutput]:
    """Executes deterministic chromatography and UF/DF sizing equations."""
    chrom = calculations.calculate_chromatography_sizing(recipe, chrom_skid)
    tff = calculations.calculate_tff_sizing(recipe)
    return chrom, tff

  def _block_mes_actuation(
      self,
      mes_tag: str,
      proposed_setpoint: float,
  ) -> None:
    """Deterministically rejects any shop-floor write attempt."""
    raise models.GxpBoundaryViolationError(
        'GxP Operational Boundary Violation: Autonomous shop-floor write to '
        f'Plant MES tag {mes_tag!r} (setpoint={proposed_setpoint}) is '
        'strictly prohibited. Agent operates in Non-GxP Decision-Support Mode.'
    )
