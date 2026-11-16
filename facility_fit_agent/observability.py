"""Observability, OpenTelemetry-style tracing, and ALCOA+ digest utilities.

Implements the Observability & Tracing architectural pillar:
- Hierarchical distributed span tracing (`AgentTracer`) across every
  orchestration node and deterministic tool invocation.
- Structured operational counters and latency telemetry (`MetricsRegistry`).
- Deterministic SHA-256 canonical payload hashing for 21 CFR Part 11 / ALCOA+
  non-GxP decision-support auditability.
"""

import collections
from collections.abc import Iterator, Mapping
import contextlib
import hashlib
import json
import time
from typing import Any
import uuid

import logging

from facility_fit_agent import models


def compute_canonical_sha256(payload: Mapping[str, Any]) -> str:
  """Computes a deterministic SHA-256 hex digest over canonical JSON."""
  filtered = {k: v for k, v in payload.items() if k != 'observability'}
  encoded = json.dumps(filtered, sort_keys=True, separators=(',', ':')).encode(
      'utf-8'
  )
  return hashlib.sha256(encoded).hexdigest()


class MetricsRegistry:
  """In-memory structured counter and latency collector for agent telemetry."""

  def __init__(self):
    """Initializes zeroed operational counters and latency buckets."""
    self._counters: collections.Counter[str] = collections.Counter()
    self._latencies_ms: dict[str, list[float]] = collections.defaultdict(list)

  def increment(self, metric_name: str, delta: int = 1) -> None:
    """Increments a named operational counter."""
    self._counters[metric_name] += delta

  def record_latency_ms(self, operation_name: str, duration_ms: float) -> None:
    """Appends an operation duration sample in milliseconds."""
    self._latencies_ms[operation_name].append(round(duration_ms, 3))

  def get_counter(self, metric_name: str) -> int:
    """Returns current value for `metric_name` (0 if unset)."""
    return self._counters.get(metric_name, 0)

  def export_snapshot(self) -> dict[str, Any]:
    """Exports counters and p50/max latency summaries as JSON-ready dict."""
    latency_summary: dict[str, dict[str, float]] = {}
    for op_name, samples in self._latencies_ms.items():
      ordered = sorted(samples)
      mid_idx = len(ordered) // 2
      latency_summary[op_name] = {
          'count': float(len(ordered)),
          'p50_ms': ordered[mid_idx],
          'max_ms': ordered[-1],
      }
    return {
        'counters': dict(self._counters),
        'latencies_ms': latency_summary,
    }


class ActiveSpanHandle:
  """Mutable handle yielded while an execution span is in flight."""

  def __init__(self, attributes: dict[str, Any]):
    """Initializes the active span handle."""
    self._attributes = attributes

  def set_attribute(self, key: str, value: Any) -> None:
    """Attaches a structured key-value attribute to the current span."""
    self._attributes[key] = value


class AgentTracer:
  """Records hierarchical OpenTelemetry-compatible execution spans."""

  def __init__(
      self,
      metrics: MetricsRegistry | None = None,
      trace_id: str = '',
  ):
    """Initializes tracer with optional shared `MetricsRegistry`."""
    self._trace_id = trace_id or uuid.uuid4().hex[:16]
    self._metrics = metrics if metrics is not None else MetricsRegistry()
    self._spans: list[models.TraceSpanRecord] = []
    self._active_stack: list[str] = []

  @property
  def trace_id(self) -> str:
    """Returns the correlation trace identifier."""
    return self._trace_id

  @property
  def metrics(self) -> MetricsRegistry:
    """Returns the underlying metrics collector."""
    return self._metrics

  @contextlib.contextmanager
  def start_span(
      self,
      span_name: str,
      stage: str,
      attributes: Mapping[str, Any] | None = None,
  ) -> Iterator[ActiveSpanHandle]:
    """Records a timed execution span with parent correlation and status."""
    span_id = uuid.uuid4().hex[:12]
    parent_id = self._active_stack[-1] if self._active_stack else 'root'
    span_attrs: dict[str, Any] = {
        'gxp.mode': 'NON_GXP_DECISION_SUPPORT',
        'gxp.mes_write_lock': True,
    }
    if attributes is not None:
      span_attrs.update(attributes)

    handle = ActiveSpanHandle(span_attrs)
    self._active_stack.append(span_id)
    start_ns = time.perf_counter_ns()
    status = 'OK'
    try:
      yield handle
    except Exception as exc:
      status = f'ERROR:{type(exc).__name__}'
      span_attrs['error.type'] = type(exc).__name__
      logging.warning(
          'Trace %s span %s failed with %s',
          self._trace_id,
          span_name,
          type(exc).__name__,
      )
      raise
    finally:
      elapsed_ms = (time.perf_counter_ns() - start_ns) / 1.0e6
      self._active_stack.pop()
      self._metrics.record_latency_ms(span_name, elapsed_ms)
      self._spans.append(
          models.TraceSpanRecord(
              trace_id=self._trace_id,
              span_id=span_id,
              parent_span_id=parent_id,
              span_name=span_name,
              stage=stage,
              status=status,
              duration_ms=round(elapsed_ms, 3),
              attributes=span_attrs,
          )
      )

  def build_bundle(
      self,
      report_dict: Mapping[str, Any],
  ) -> models.ObservabilityBundle:
    """Builds the immutable `ObservabilityBundle` with ALCOA+ SHA-256 digest."""
    total_ms = sum(
        s.duration_ms for s in self._spans if s.parent_span_id == 'root'
    )
    if total_ms == 0.0:
      total_ms = sum(s.duration_ms for s in self._spans)
    tool_calls = sum(1 for s in self._spans if s.stage == 'TOOL_INVOCATION')
    digest = compute_canonical_sha256(report_dict)
    return models.ObservabilityBundle(
        trace_id=self._trace_id,
        total_duration_ms=round(total_ms, 3),
        tool_invocations_count=tool_calls,
        deterministic_calc_ratio=1.0,
        payload_sha256=digest,
        spans=tuple(self._spans),
    )
