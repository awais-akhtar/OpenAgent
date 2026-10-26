"""Telemetry Exporter — structured metrics collection and Prometheus export.

Collects metrics from all Phase 1/2/3 components and exposes them in:
  - Prometheus text format (``/metrics`` endpoint)
  - JSON snapshot (for internal dashboards)
  - Structured log lines (for log aggregation pipelines)

Components
----------
``MetricType``            -- COUNTER / GAUGE / HISTOGRAM / SUMMARY
``LabelSet``              -- immutable key-value label dict for a metric series
``MetricSample``          -- one time-stamped observation
``LatencyHistogram``      -- fixed-bucket latency histogram with percentile query
``TelemetryRegistry``     -- thread-safe store for all metric families
``PrometheusTextFormatter``-- renders registry to Prometheus exposition format
``Phase3TelemetryCollector``-- scrapes Phase 1/2/3 metrics dicts on a schedule
                              and feeds them into the registry
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetricType
# ---------------------------------------------------------------------------

class MetricType(str, Enum):
    COUNTER   = "counter"
    GAUGE     = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY   = "summary"


# ---------------------------------------------------------------------------
# LabelSet
# ---------------------------------------------------------------------------

class LabelSet:
    """Immutable sorted label dictionary, hashable for use as dict key."""

    def __init__(self, **labels: str) -> None:
        self._labels: Tuple[Tuple[str, str], ...] = tuple(
            sorted(labels.items()))

    def __hash__(self) -> int:
        return hash(self._labels)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LabelSet) and self._labels == other._labels

    def to_dict(self) -> Dict[str, str]:
        return dict(self._labels)

    def prometheus_suffix(self) -> str:
        if not self._labels:
            return ""
        pairs = ",".join(f'{k}="{v}"' for k, v in self._labels)
        return "{" + pairs + "}"

    @classmethod
    def empty(cls) -> "LabelSet":
        return cls()


# ---------------------------------------------------------------------------
# MetricSample
# ---------------------------------------------------------------------------

@dataclass
class MetricSample:
    """One observed value for a metric series."""
    name:      str
    value:     float
    labels:    LabelSet = field(default_factory=LabelSet.empty)
    timestamp: float    = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE


# ---------------------------------------------------------------------------
# LatencyHistogram
# ---------------------------------------------------------------------------

class LatencyHistogram:
    """Fixed-bucket histogram for latency measurements (milliseconds).

    Default buckets: 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000 ms.
    """

    DEFAULT_BUCKETS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0,
                       250.0, 500.0, 1000.0, 2500.0, 5000.0)

    def __init__(self, buckets: Optional[Tuple[float, ...]] = None) -> None:
        self._buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts  = [0] * (len(self._buckets) + 1)  # +1 for +Inf
        self._sum_ms  = 0.0
        self._total   = 0
        self._lock    = threading.Lock()

    def observe(self, latency_ms: float) -> None:
        with self._lock:
            self._sum_ms += latency_ms
            self._total  += 1
            for i, bound in enumerate(self._buckets):
                if latency_ms <= bound:
                    self._counts[i] += 1
                    return
            self._counts[-1] += 1   # +Inf bucket

    def percentile(self, p: float) -> float:
        """Estimate the p-th percentile (0-100) via linear interpolation."""
        with self._lock:
            if self._total == 0:
                return 0.0
            target = self._total * p / 100.0
            cumulative = 0
            for i, bound in enumerate(self._buckets):
                cumulative += self._counts[i]
                if cumulative >= target:
                    lower = self._buckets[i - 1] if i > 0 else 0.0
                    bucket_count = self._counts[i]
                    if bucket_count == 0:
                        return bound
                    prev_cum = cumulative - bucket_count
                    frac = (target - prev_cum) / bucket_count
                    return lower + frac * (bound - lower)
            return float("inf")

    @property
    def mean_ms(self) -> float:
        with self._lock:
            return self._sum_ms / max(1, self._total)

    @property
    def count(self) -> int:
        with self._lock:
            return self._total

    def prometheus_lines(self, name: str, labels: LabelSet) -> List[str]:
        """Render histogram in Prometheus text format."""
        lines: List[str] = []
        lstr  = labels.prometheus_suffix()
        with self._lock:
            cumulative = 0
            for i, bound in enumerate(self._buckets):
                cumulative += self._counts[i]
                lines.append(
                    f'{name}_bucket{{le="{bound:.1f}"{", " if lstr else ""}'
                    f'{lstr[1:-1] if lstr else ""}}} {cumulative}')
            cumulative += self._counts[-1]
            lines.append(f'{name}_bucket{{le="+Inf"{", " if lstr else ""}'
                         f'{lstr[1:-1] if lstr else ""}}} {cumulative}')
            lines.append(f'{name}_sum{lstr} {self._sum_ms:.4f}')
            lines.append(f'{name}_count{lstr} {self._total}')
        return lines


# ---------------------------------------------------------------------------
# MetricFamily
# ---------------------------------------------------------------------------

@dataclass
class MetricFamily:
    """A named metric with its type and per-label-set values."""
    name:    str
    mtype:   MetricType
    help:    str = ""
    # label_set → (value, histogram)
    series:  Dict[LabelSet, Any] = field(default_factory=dict)
    _lock:   threading.Lock      = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float, labels: Optional[LabelSet] = None) -> None:
        ls = labels or LabelSet.empty()
        with self._lock:
            self.series[ls] = value

    def increment(self, delta: float = 1.0, labels: Optional[LabelSet] = None) -> None:
        ls = labels or LabelSet.empty()
        with self._lock:
            self.series[ls] = self.series.get(ls, 0.0) + delta

    def observe_histogram(
        self, value_ms: float, labels: Optional[LabelSet] = None
    ) -> None:
        ls = labels or LabelSet.empty()
        with self._lock:
            if ls not in self.series:
                self.series[ls] = LatencyHistogram()
            self.series[ls].observe(value_ms)


# ---------------------------------------------------------------------------
# TelemetryRegistry
# ---------------------------------------------------------------------------

class TelemetryRegistry:
    """Thread-safe store for all metric families."""

    def __init__(self) -> None:
        self._families: Dict[str, MetricFamily] = {}
        self._lock = threading.RLock()

    def _get_or_create(
        self, name: str, mtype: MetricType, help_text: str = ""
    ) -> MetricFamily:
        with self._lock:
            if name not in self._families:
                self._families[name] = MetricFamily(
                    name=name, mtype=mtype, help=help_text)
            return self._families[name]

    def counter(
        self, name: str, value: float = 1.0,
        labels: Optional[LabelSet] = None, help: str = "",
    ) -> None:
        fam = self._get_or_create(name, MetricType.COUNTER, help)
        fam.increment(value, labels)

    def gauge(
        self, name: str, value: float,
        labels: Optional[LabelSet] = None, help: str = "",
    ) -> None:
        fam = self._get_or_create(name, MetricType.GAUGE, help)
        fam.set(value, labels)

    def histogram(
        self, name: str, value_ms: float,
        labels: Optional[LabelSet] = None, help: str = "",
    ) -> None:
        fam = self._get_or_create(name, MetricType.HISTOGRAM, help)
        fam.observe_histogram(value_ms, labels)

    def snapshot(self) -> List[MetricSample]:
        samples: List[MetricSample] = []
        with self._lock:
            for fam in self._families.values():
                with fam._lock:
                    for ls, val in fam.series.items():
                        if isinstance(val, (int, float)):
                            samples.append(MetricSample(
                                name=fam.name, value=float(val),
                                labels=ls, metric_type=fam.mtype))
        return samples

    def to_json(self) -> List[dict]:
        return [
            {
                "name":   s.name,
                "value":  s.value,
                "labels": s.labels.to_dict(),
                "type":   s.metric_type.value,
                "ts":     s.timestamp,
            }
            for s in self.snapshot()
        ]

    def all_families(self) -> List[MetricFamily]:
        with self._lock:
            return list(self._families.values())


# ---------------------------------------------------------------------------
# PrometheusTextFormatter
# ---------------------------------------------------------------------------

class PrometheusTextFormatter:
    """Renders a ``TelemetryRegistry`` to Prometheus text exposition format."""

    def render(self, registry: TelemetryRegistry) -> str:
        lines: List[str] = []
        for fam in registry.all_families():
            lines.append(f"# HELP {fam.name} {fam.help or fam.name}")
            lines.append(f"# TYPE {fam.name} {fam.mtype.value}")
            with fam._lock:
                for ls, val in fam.series.items():
                    lstr = ls.prometheus_suffix()
                    if isinstance(val, LatencyHistogram):
                        lines.extend(val.prometheus_lines(fam.name, ls))
                    elif isinstance(val, (int, float)):
                        lines.append(f"{fam.name}{lstr} {val}")
        lines.append("")   # trailing newline required by Prometheus
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase3TelemetryCollector
# ---------------------------------------------------------------------------

class Phase3TelemetryCollector:
    """Scrapes metrics dicts from registered sources and feeds TelemetryRegistry.

    Sources are any callable ``() → dict`` that returns flat or nested
    key-value pairs; the collector flattens nested dicts with ``__`` separator
    and infers metric type from value type.

    Usage::

        collector = Phase3TelemetryCollector(registry=TelemetryRegistry())
        collector.add_source("phase2_service", service.metrics)
        collector.add_source("memory_manager", mem_mgr.metrics)
        collector.start(interval_s=5.0)
    """

    def __init__(
        self,
        registry: Optional[TelemetryRegistry] = None,
        scrape_prefix: str = "openagent",
    ) -> None:
        self._registry  = registry or TelemetryRegistry()
        self._prefix    = scrape_prefix
        self._sources:  Dict[str, Callable[[], dict]] = {}
        self._lock      = threading.Lock()
        self._stop_ev   = threading.Event()
        self._thread:   Optional[threading.Thread] = None
        self._scrapes   = 0
        self._errors    = 0

    def add_source(self, name: str, fn: Callable[[], dict]) -> None:
        with self._lock:
            self._sources[name] = fn

    def remove_source(self, name: str) -> None:
        with self._lock:
            self._sources.pop(name, None)

    # ---- scrape logic -------------------------------------------------------

    def scrape_once(self) -> None:
        with self._lock:
            sources = dict(self._sources)
        for src_name, fn in sources.items():
            try:
                data = fn()
                self._ingest(src_name, data)
                self._scrapes += 1
            except Exception as exc:
                logger.debug("telemetry: scrape error for %s: %s", src_name, exc)
                self._errors += 1

    def _ingest(self, source: str, data: dict, prefix: str = "") -> None:
        for key, val in data.items():
            metric_name = f"{self._prefix}_{source}_{prefix}{key}".replace("-", "_")
            if isinstance(val, dict):
                self._ingest(source, val, f"{prefix}{key}__")
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                # Heuristic: keys ending in _total or _count → counter; else gauge
                if key.endswith("_total") or key.endswith("_count") or key.endswith("_calls"):
                    self._registry.counter(metric_name, float(val))
                elif "latency" in key or "ms" in key:
                    self._registry.histogram(metric_name, float(val))
                else:
                    self._registry.gauge(metric_name, float(val))
            elif isinstance(val, bool):
                self._registry.gauge(metric_name, 1.0 if val else 0.0)

    # ---- lifecycle ----------------------------------------------------------

    def start(self, interval_s: float = 5.0) -> None:
        self._interval = interval_s
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="telemetry-collector")
        self._stop_ev.clear()
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_ev.wait(timeout=self._interval):
            self.scrape_once()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_ev.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("telemetry collector did not stop before deadline")

    # ---- export -------------------------------------------------------------

    @property
    def registry(self) -> TelemetryRegistry:
        return self._registry

    def prometheus_text(self) -> str:
        return PrometheusTextFormatter().render(self._registry)

    def json_snapshot(self) -> List[dict]:
        return self._registry.to_json()

    def metrics(self) -> dict:
        with self._lock:
            return {
                "telemetry_sources":       len(self._sources),
                "telemetry_scrapes_total": self._scrapes,
                "telemetry_errors_total":  self._errors,
            }


__all__ = [
    "MetricType",
    "LabelSet",
    "MetricSample",
    "LatencyHistogram",
    "MetricFamily",
    "TelemetryRegistry",
    "PrometheusTextFormatter",
    "Phase3TelemetryCollector",
]
