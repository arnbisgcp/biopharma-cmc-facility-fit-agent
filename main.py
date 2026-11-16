"""Cloud Run HTTP server backed by persistent async SQLite WAL memory."""

import asyncio
import http.server
import json
import os
import pathlib
from typing import Any
import urllib.parse

from facility_fit_agent import cicd
from facility_fit_agent import database
from facility_fit_agent import memory
from facility_fit_agent import models
from facility_fit_agent import orchestrator
from facility_fit_agent import prompts

_STATIC_INDEX = pathlib.Path(__file__).parent / 'static' / 'index.html'
_DB_BACKEND = database.create_memory_backend()
_ORCHESTRATOR = orchestrator.FacilityFitOrchestrator(backend=_DB_BACKEND)


def _get_session(session_id: str) -> memory.SessionWorkingContext:
  """Returns a session handle bound to the shared persistent DB backend."""
  return memory.SessionWorkingContext(
      session_id=session_id,
      backend=_DB_BACKEND,
  )


class FacilityFitRequestHandler(http.server.BaseHTTPRequestHandler):
  """Serves the Interactive Cockpit UI and REST API endpoints on Cloud Run."""

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    """Handles UI, health, assessment, tools, prompt, CI/CD, & telemetry."""
    parsed = urllib.parse.urlparse(self.path)
    route = parsed.path
    query = urllib.parse.parse_qs(parsed.query)

    if route in ('/', '/index.html'):
      html_bytes = _STATIC_INDEX.read_bytes()
      self.send_response(200)
      self.send_header('Content-Type', 'text/html; charset=utf-8')
      self.send_header('Content-Length', str(len(html_bytes)))
      self.end_headers()
      self.wfile.write(html_bytes)
      return

    if route == '/healthz':
      chain_ok = asyncio.run(_ORCHESTRATOR.audit_ledger.verify_chain_integrity())
      self._send_json(
          200,
          {
              'status': 'ok',
              'service': 'biopharma-cmc-continuous-facility-fit-agent',
              'gxp_mode': 'NON_GXP_DECISION_SUPPORT',
              'persistent_backend': type(_DB_BACKEND).__name__,
              'ledger_chain_valid': chain_ok,
          },
      )
      return

    if route == '/api/system-prompt':
      self._send_json(
          200,
          {
              'system_prompt': prompts.SYSTEM_PROMPT,
              'estimated_tokens': prompts.estimate_token_count(
                  prompts.SYSTEM_PROMPT
              ),
          },
      )
      return

    if route == '/api/tools':
      self._send_json(
          200,
          {
              'tool_declarations': (
                  _ORCHESTRATOR.tool_registry.export_declarations()
              )
          },
      )
      return

    if route == '/api/cicd-gates':
      self._send_json(200, cicd.run_cicd_gates())
      return

    if route == '/api/telemetry':
      entries = asyncio.run(_ORCHESTRATOR.audit_ledger.entries())
      chain_ok = asyncio.run(_ORCHESTRATOR.audit_ledger.verify_chain_integrity())
      self._send_json(
          200,
          {
              'telemetry': _ORCHESTRATOR.metrics.export_snapshot(),
              'ledger_entries_count': len(entries),
              'ledger_chain_valid': chain_ok,
          },
      )
      return

    if route == '/api/assess':
      process_id = query.get('process_id', ['PKM-AB4821-DS-V4.2'])[0]
      target_site = query.get('target_site', ['ALL'])[0]
      session_id = query.get('session_id', [''])[0]
      sess = _get_session(session_id) if session_id else None
      try:
        if target_site.upper() == 'ALL':
          reports = _ORCHESTRATOR.orchestrate_network_comparison(
              process_id=process_id,
              session_context=sess,
          )
          chain_ok = asyncio.run(
              _ORCHESTRATOR.audit_ledger.verify_chain_integrity()
          )
          self._send_json(
              200,
              {
                  'process_id': process_id,
                  'reports': [rep.to_dict() for rep in reports],
                  'ledger_chain_valid': chain_ok,
              },
          )
        else:
          report = _ORCHESTRATOR.execute_assessment(
              process_id=process_id,
              target_site=target_site,
              session_context=sess,
          )
          self._send_json(200, report.to_dict())
      except (KeyError, ValueError) as err:
        self._send_json(400, {'error': str(err)})
      return

    self._send_json(404, {'error': f'Unknown endpoint: {route}'})

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Handles counterfactual what-if, sponsor signoff, and red-team routes."""
    parsed = urllib.parse.urlparse(self.path)
    route = parsed.path
    body = self._read_json_body()

    if route == '/api/simulate-what-if':
      session_id = str(body.get('session_id', 'cloudrun-default-session'))
      process_id = str(body.get('process_id', 'PKM-AB4821-DS-V4.2'))
      target_site = str(body.get('target_site', 'BETA_US_CAMPUS'))
      raw_overrides = body.get('cpp_overrides', {})
      overrides = {str(k): float(v) for k, v in raw_overrides.items()}
      try:
        session = _get_session(session_id)
        result = _ORCHESTRATOR.simulate_what_if(
            session_context=session,
            process_id=process_id,
            target_site=target_site,
            cpp_overrides=overrides,
        )
        self._send_json(200, result)
      except (KeyError, ValueError, AttributeError) as err:
        self._send_json(400, {'error': str(err)})
      return

    if route == '/api/signoff':
      process_id = str(body.get('process_id', 'PKM-AB4821-DS-V4.2'))
      target_site = str(body.get('target_site', 'ALPHA_EU_HUB'))
      sponsor = str(body.get('sponsor', '')).strip()
      if not sponsor:
        self._send_json(400, {'error': 'Field "sponsor" is required.'})
        return
      try:
        signed_report = _ORCHESTRATOR.execute_assessment(
            process_id=process_id,
            target_site=target_site,
            sponsor_signoff=sponsor,
        )
        self._send_json(200, signed_report.to_dict())
      except (KeyError, ValueError) as err:
        self._send_json(400, {'error': str(err)})
      return

    if route == '/api/redteam-mes-write':
      mes_tag = str(body.get('mes_tag', 'EU-ALP-B1-BRX1501'))
      setpoint = float(body.get('proposed_setpoint', 55.0))
      try:
        _ORCHESTRATOR.tool_registry.invoke(
            'actuate_shop_floor_mes_setpoint',
            mes_tag=mes_tag,
            proposed_setpoint=setpoint,
        )
      except models.GxpBoundaryViolationError as exc:
        _ORCHESTRATOR.metrics.increment('gxp_boundary_blocks_total')
        self._send_json(
            403,
            {
                'blocked': True,
                'error_type': 'GxpBoundaryViolationError',
                'message': str(exc),
                'gxp_boundary_blocks_total': _ORCHESTRATOR.metrics.get_counter(
                    'gxp_boundary_blocks_total'
                ),
            },
        )
        return

    self._send_json(404, {'error': f'Unknown POST endpoint: {route}'})

  def log_message(self, fmt: str, *args: Any) -> None:
    del fmt, args

  def _read_json_body(self) -> dict[str, Any]:
    content_len = int(self.headers.get('Content-Length', '0'))
    if content_len <= 0:
      return {}
    raw = self.rfile.read(content_len).decode('utf-8')
    return json.loads(raw) if raw.strip() else {}

  def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2).encode('utf-8')
    self.send_response(status_code)
    self.send_header('Content-Type', 'application/json; charset=utf-8')
    self.send_header('Content-Length', str(len(encoded)))
    self.end_headers()
    self.wfile.write(encoded)


def main() -> None:
  """Starts the Cloud Run HTTP server on 0.0.0.0:$PORT (default 8080)."""
  port = int(os.environ.get('PORT', '8080'))
  server = http.server.ThreadingHTTPServer(
      ('0.0.0.0', port), FacilityFitRequestHandler
  )
  print(
      'Biopharma CMC Continuous Facility Fit Agent listening on '
      f'0.0.0.0:{port}',
      flush=True,
  )
  server.serve_forever()


if __name__ == '__main__':
  main()
