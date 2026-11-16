"""Constitutional LLM system prompt and dynamic context window assembly.

Provides the authoritative `SYSTEM_PROMPT` governing LLM reasoning, GxP
operational boundaries, deterministic tool-use enforcement, and token-bounded
prompt construction integrating compacted session memory and episodic CAPAs.
"""

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any
import urllib.request

from facility_fit_agent import models

SYSTEM_PROMPT = """# ROLE & CONSTITUTION: BIOPHARMA CMC CONTINUOUS FACILITY FIT AGENT

## 1. IDENTITY & MISSION
You are the Biopharma CMC Continuous Facility Fit Agent, an autonomous, GxP-aware
decision-support engine embedded within Global Biomanufacturing Operations (GBO)
and Bioprocess R&D Sciences (BRDS). Your mission is to evaluate process recipe
parameters from Enterprise Process Knowledge Management (PKM) packages against
the verified physical, hydrodynamic, thermal, clean-utility, analytical (LIMS),
and supply-planning constraints of target receiving manufacturing facilities
prior to commercial or PPQ campaign execution.

## 2. GxP COMPLIANCE & OPERATIONAL BOUNDARIES
- Non-GxP Decision Support Only: Provide normalized risk scores, fit analyses,
  and discrepancy matrices. NEVER execute autonomous shop-floor Plant MES
  setpoint writes or override validated Standard Operating Procedures (SOPs).
- Deterministic Calculation Priority: NEVER estimate, interpolate, or
  hallucinate physical engineering values (Reynolds numbers, impeller tip
  speeds, gassed P/V, Kolmogorov eddy microscales, Van't Riet kLa, jacket heat
  transfer duties, chromatography Darcy pressure drops, or UF/DF membrane areas).
  ALWAYS invoke the designated deterministic calculation tools and cite their
  exact outputs.
- Strict Discrepancy Classification:
  * CRITICAL: Process-breaking, cell-viability, or sterility/safety failure risk.
  * MAJOR: Yield, CQA, analytical readiness, or filing impact requiring formal
    Global Change Control (GCC).
  * MINOR: Procedural or automation phase parameter adjustment within validated
    envelopes.

## 3. MULTI-STEP REASONING PROTOCOL
1. Ingest & Normalize: Parse unit operations, CPPs, and CQAs from Enterprise PKM
   plus any active session counterfactual CPP overrides.
2. Asset & Readiness Discovery: Retrieve verified vessel/skid capabilities,
   utility capacities, LIMS transfer statuses, and supply slots.
3. Deterministic Unit-Operation Fit: Evaluate Volumetric Fit -> Hydrodynamic
   Shear & Aeration Fit -> Thermal Balance -> Clean Utility Load -> Material &
   Analytical Readiness.
4. Dynamic Risk Scoring & Precedent Grounding: Compute normalized fit index
   (0.00 to 1.00) and ground mitigations in retrieved historical CAPA precedents.
5. Human-in-the-Loop (HITL) Governance Gate: Whenever MAJOR or CRITICAL
   discrepancies exist, enforce `persistence_locked: true` until human Technical
   Sponsor adjudication is recorded.

## 4. STRICT OUTPUT SCHEMA CONTRACT
All structured outputs MUST conform strictly to the JSON schema
`FacilityFitAssessmentReport` containing mandatory keys: `assessment_id`,
`process_id`, `target_site`, `overall_fit_status`, `discrepancy_matrix`, and
`mitigation_recommendations`.
"""


def estimate_token_count(text: str) -> int:
  """Estimates LLM token count conservatively (~4 characters per token)."""
  if not text:
    return 0
  return max(1, (len(text) + 3) // 4)


class PromptContextAssembler:
  """Builds token-bounded LLM message arrays from persistent session memory."""

  def __init__(
      self,
      system_prompt: str = SYSTEM_PROMPT,
      max_context_tokens: int = 4096,
  ):
    """Initializes the prompt assembler with system prompt and token budget."""
    self._system_prompt = system_prompt
    self._max_context_tokens = max_context_tokens

  @property
  def system_prompt(self) -> str:
    """Returns the constitutional LLM system prompt."""
    return self._system_prompt

  def build_messages(
      self,
      user_task: str,
      compacted_memory_window: Mapping[str, Any],
      precedents: Sequence[models.HistoricalPrecedent] = (),
      tool_declarations: Sequence[Mapping[str, Any]] = (),
  ) -> dict[str, Any]:
    """Assembles structured system instruction, memory context, and user turn."""
    precedent_block = [prec.to_dict() for prec in precedents]
    memory_context_str = json.dumps(
        {
            'session_memory': dict(compacted_memory_window),
            'episodic_capa_precedents': precedent_block,
            'registered_tool_names': [t.get('name') for t in tool_declarations],
        },
        indent=2,
        sort_keys=True,
    )

    sys_tokens = estimate_token_count(self._system_prompt)
    mem_tokens = estimate_token_count(memory_context_str)
    usr_tokens = estimate_token_count(user_task)
    total_tokens = sys_tokens + mem_tokens + usr_tokens

    return {
        'system_instruction': {
            'role': 'system',
            'content': self._system_prompt,
        },
        'messages': [
            {
                'role': 'user',
                'name': 'persistent_memory_context',
                'content': (
                    '<retrieved_context_and_memory>\n'
                    f'{memory_context_str}\n'
                    '</retrieved_context_and_memory>'
                ),
            },
            {
                'role': 'user',
                'content': user_task,
            },
        ],
        'token_usage_estimate': {
            'system_tokens': sys_tokens,
            'memory_context_tokens': mem_tokens,
            'user_task_tokens': usr_tokens,
            'total_tokens': total_tokens,
            'max_budget_tokens': self._max_context_tokens,
            'within_budget': total_tokens <= self._max_context_tokens,
        },
    }


class GeminiConstitutionalAdvisor:
  """Optional async Gemini LLM narrative synthesizer grounded in SYSTEM_PROMPT."""

  def __init__(self, model_name: str = 'gemini-2.5-flash'):
    """Initializes Gemini advisor reading `GEMINI_API_KEY` if configured."""
    self._model_name = model_name
    self._api_key = os.environ.get('GEMINI_API_KEY', '')

  def generate_sync(self, assembled_prompt: Mapping[str, Any]) -> str:
    """Calls Gemini REST endpoint if configured, else returns grounded summary."""
    if not self._api_key:
      usage = assembled_prompt.get('token_usage_estimate', {})
      return (
          'Deterministic GxP Advisory Mode active (SYSTEM_PROMPT enforced; '
          f'context_tokens={usage.get("total_tokens", 0)}).'
      )
    url = (
        'https://generativelanguage.googleapis.com/v1beta/models/'
        f'{self._model_name}:generateContent?key={self._api_key}'
    )
    sys_text = assembled_prompt['system_instruction']['content']
    contents = [
        {'role': 'user', 'parts': [{'text': m['content']}]}
        for m in assembled_prompt['messages']
    ]
    body = json.dumps({
        'systemInstruction': {'parts': [{'text': sys_text}]},
        'contents': contents,
        'generationConfig': {'temperature': 0.0, 'maxOutputTokens': 512},
    }).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
      data = json.loads(resp.read())
      candidates = data.get('candidates', [])
      if candidates:
        parts = candidates[0].get('content', {}).get('parts', [])
        if parts:
          return str(parts[0].get('text', ''))
    return 'Gemini synthesis completed.'
