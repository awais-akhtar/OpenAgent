"""Convert a scientific run into a reviewable plain-text report."""

from __future__ import annotations

import json
from typing import Any


def build_report(*, goal: str, plan: dict[str, Any], result: dict[str, Any], verification: dict[str, Any]) -> str:
    lines = [f"Scientific task: {goal}", "", f"Workflow: {plan.get('workflow', 'unknown')}", "", "Plan:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(plan.get("steps", []), start=1))
    if plan.get("assumptions"):
        lines.extend(["", "Assumptions:"])
        lines.extend(f"- {item}" for item in plan["assumptions"])
    execution = result.get("execution") if isinstance(result.get("execution"), dict) else None
    if execution:
        lines.extend(["", "Execution and self-correction:"])
        for event in execution.get("lifecycle", []):
            attempt = f" attempt {event['attempt']}" if event.get("attempt") else ""
            lines.append(f"- {event.get('stage', 'step')}{attempt}: {event.get('message', '')}")
        lines.extend([
            f"- Attempts: {execution.get('attempts', 0)}",
            f"- Repairs: {execution.get('repairs', 0)}",
            f"- Execution time: {float(execution.get('elapsed_ms', 0.0)):.3f} ms",
            f"- Peak process memory: {int(execution.get('peak_memory_bytes', 0))} bytes",
        ])
    if result.get("memory_used"):
        lines.extend(["", "Scientific memory used:"])
        lines.extend(f"- {item.get('name')}: {item.get('value')} {item.get('unit', '')}".rstrip() for item in result["memory_used"])
    lines.extend(["", "Verification:", f"- Status: {'passed' if verification.get('passed') else 'needs review'}", f"- Score: {float(verification.get('score', 0.0)):.2f}"])
    for finding in verification.get("findings", []):
        lines.append(f"- {finding.get('check', 'check')}: {finding.get('message', '')}")
    lines.extend(["", "Result:", json.dumps(result, indent=2, sort_keys=True, default=str)])
    return "\n".join(lines)
