"""Lightweight in-process metrics for agent-level observability (Req 7).

Tracks counters and latencies in memory during a session.  On exit, main.py
calls write_snapshot() to persist a JSON summary to data/metrics_snapshot.json.

Monitor live logs with:
    tail -f data/agent.log | python -m json.tool
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_counters: dict[str, int] = defaultdict(int)
_latencies: list[dict] = []
_events: list[dict] = []


def increment(metric: str, value: int = 1) -> None:
    """Increment a named counter."""
    _counters[metric] += value


def record_latency(seconds: float, name: str = "request_latency_s") -> None:
    """Record a latency sample."""
    _latencies.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "latency_s": round(seconds, 4),
    })


def event(name: str, **kwargs) -> None:
    """Record a structured event (e.g. errors, user switches)."""
    _events.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "name": name,
        **kwargs,
    })
    logger.info("metrics.event", extra={"event_name": name, **kwargs})


def summary() -> dict:
    """Return a dict snapshot of all metrics for this session."""
    avg_latency = (
        sum(x["latency_s"] for x in _latencies) / len(_latencies)
        if _latencies else 0.0
    )
    return {
        "counters": dict(_counters),
        "avg_latency_s": round(avg_latency, 4),
        "latency_samples": len(_latencies),
        "recent_events": _events[-20:],
    }


def write_snapshot(path: str | Path) -> None:
    """Persist the current metrics summary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = summary()
    path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    logger.info("Metrics snapshot written", extra={"path": str(path), "counters": snap["counters"]})
