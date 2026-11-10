"""Validate the small, local CPU product observation without mixing route evidence."""

from __future__ import annotations

import hashlib
import json
import ast
from pathlib import Path
from typing import Any

from .cpu_model_admission import CPUModelAdmissionError, admit_cpu_openvino_bundle


SOURCE_PATHS = (
    "openagent/cpu_product_gate.py",
    "openagent/cpu_model_admission.py",
    "openagent/cpu_budget_job.py",
    "openagent/model_gateway.py",
    "openagent/adaptive_engine/device_runtime.py",
    "openagent/adaptive_engine/hardware.py",
    "openagent/chat.py",
    "openagent/agents.py",
    "openagent/codebase/service.py",
    "openagent/routes/codebase_routes.py",
    "openagent/routes/runtime.py",
    "openagent/workbench.py",
    "openagent/desktop.py",
    "frontend-second/src/screens/WorkspaceScreens.tsx",
    "phase1_verify.py",
    "scripts/verify_cpu_baseline.py",
    "scripts/verify_cpu_budget.py",
)
REQUIRED_CALLERS = (
    "chat", "chat_followup", "coding", "research", "executive",
    "codebase_ask", "codebase_edit_proposal",
)


def source_pins(project_root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


def bundle_pin(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(model_dir.rglob("*")):
        if not file.is_file():
            continue
        digest.update(file.relative_to(model_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_cpu_product_observation(
    project_root: Path, receipt_path: Path, model_dir: Path, *,
    require_host_max_bytes: int | None = None,
    require_job_max_bytes: int | None = None,
) -> dict[str, Any]:
    """Check a captured run against the current bounded CPU source and model."""
    issues: list[str] = []
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("observation must be an object")
    except (OSError, ValueError) as exc:
        return {"valid": False, "issues": [f"observation unavailable: {exc}"]}
    if receipt.get("kind") != "local-cpu-product-smoke-not-family-route-receipt":
        issues.append("wrong observation kind")
    if receipt.get("backend") != "openvino-direct" or receipt.get("device") != "CPU":
        issues.append("observation is not the direct CPU backend")
    if receipt.get("model_type") != "qwen2":
        issues.append("observation is not the admitted small Qwen2 model")
    try:
        if receipt.get("source_sha256") != source_pins(project_root):
            issues.append("source pins do not match current production code")
        if receipt.get("model_bundle_sha256") != bundle_pin(model_dir):
            issues.append("model bundle pin does not match current local files")
        actual_model_bytes = admit_cpu_openvino_bundle(model_dir)
        if receipt.get("model_bytes") != actual_model_bytes:
            issues.append("recorded model bytes do not match the current local bundle")
    except (OSError, CPUModelAdmissionError) as exc:
        issues.append(f"source or model bundle could not be inspected: {exc}")
    model_bytes = receipt.get("model_bytes")
    rss = receipt.get("sampled_process_tree_peak_rss_bytes")
    app_rss = receipt.get("sampled_app_process_tree_peak_rss_bytes")
    host_ram = receipt.get("host_total_ram_bytes")
    if not isinstance(model_bytes, int) or not 0 < model_bytes <= 4 * 1024**3:
        issues.append("model file budget is missing or exceeded")
    if not isinstance(rss, int) or not 0 < rss <= 8 * 1024**3:
        issues.append("sampled worker memory budget is missing or exceeded")
    if not isinstance(app_rss, int) or not 0 < app_rss <= 8 * 1024**3:
        issues.append("sampled whole-app memory budget is missing or exceeded")
    elif isinstance(rss, int) and app_rss < rss:
        issues.append("whole-app memory cannot be below its worker memory")
    if not isinstance(host_ram, int) or host_ram <= 0:
        issues.append("host RAM observation is missing")
    elif require_host_max_bytes is not None and host_ram > require_host_max_bytes:
        issues.append("observation was not captured on the required low-RAM host")
    if require_job_max_bytes is not None:
        job_limit = receipt.get("job_memory_limit_bytes")
        job_peak = receipt.get("job_peak_commit_bytes")
        if (receipt.get("budget_method") != "windows-job-object" or
                receipt.get("job_limit_enforced") is not True or
                not isinstance(job_limit, int) or not 0 < job_limit <= require_job_max_bytes or
                not isinstance(job_peak, int) or not 0 < job_peak <= job_limit):
            issues.append("enforced Windows job memory budget is missing or exceeded")
        if isinstance(app_rss, int) and isinstance(job_limit, int) and app_rss > job_limit:
            issues.append("sampled app memory exceeds the enforced job budget")
    if not receipt.get("complete_text") or not receipt.get("stream_text"):
        issues.append("completion or streaming output is missing")
    if receipt.get("cancelled_worker_retired") is not True or receipt.get("post_cancel_gateway_reload") is not True:
        issues.append("cancellation or recovery observation is missing")
    callers = receipt.get("task_callers")
    if not isinstance(callers, dict):
        callers = {}
    for name in REQUIRED_CALLERS:
        item = callers.get(name)
        if not isinstance(item, dict) or item.get("provider") != "openvino-direct":
            issues.append(f"{name} did not record exact local CPU routing")
            continue
        elapsed = item.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not 0 < elapsed <= 15:
            issues.append(f"{name} latency was missing or above 15 seconds")
        if not isinstance(item.get("response_text"), str) or not item["response_text"].strip():
            issues.append(f"{name} response is missing")
    codebase = callers.get("codebase_ask")
    if not isinstance(codebase, dict) or codebase.get("source_unchanged") is not True:
        issues.append("read-only CodeQ source integrity is missing")
    chat = callers.get("chat")
    if isinstance(chat, dict) and "cpu" not in str(chat.get("response_text", "")).lower():
        issues.append("chat did not answer the local CPU question")
    followup = callers.get("chat_followup")
    if isinstance(followup, dict) and (
        followup.get("same_conversation") is not True or
        "blue otter" not in str(followup.get("response_text", "")).lower()
    ):
        issues.append("chat did not retain the two-turn codeword")
    coding = callers.get("coding")
    if isinstance(coding, dict):
        source = str(coding.get("response_text", ""))
        if "```python" in source:
            source = source.split("```python", 1)[1].split("```", 1)[0]
        try:
            tree = ast.parse(source)
            matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                       node.name == "add" and
                       [arg.arg for arg in node.args.args] == ["a", "b"]]
            if not matches:
                issues.append("coding did not provide the requested add(a, b) function")
            elif not any(isinstance(stmt, ast.Return) and
                         isinstance(stmt.value, ast.BinOp) and
                         isinstance(stmt.value.op, ast.Add) and
                         isinstance(stmt.value.left, ast.Name) and stmt.value.left.id == "a" and
                         isinstance(stmt.value.right, ast.Name) and stmt.value.right.id == "b"
                         for stmt in matches[0].body):
                issues.append("coding add(a, b) did not return a + b")
        except SyntaxError:
            issues.append("coding response is not valid Python")
    research = callers.get("research")
    if isinstance(research, dict):
        response = str(research.get("response_text", "")).lower()
        if "limitation" not in response or not all(f"{step}." in response for step in (1, 2)):
            issues.append("research did not provide two steps and a limitation")
    executive = callers.get("executive")
    if isinstance(executive, dict):
        response = str(executive.get("response_text", "")).lower()
        if "16" not in response or not any(word in response for word in ("test", "benchmark", "measure")):
            issues.append("executive did not identify a 16 GB test")
    if isinstance(codebase, dict):
        response = str(codebase.get("response_text", "")).lower()
        if "alice" not in response or "hello" not in response:
            issues.append("CodeQ did not explain the selected greet source")
    edit = callers.get("codebase_edit_proposal")
    if isinstance(edit, dict):
        proposed = str(edit.get("response_text", ""))
        if edit.get("source_unchanged_before_review") is not True or edit.get("saved_after_review") is not True:
            issues.append("CodeQ proposal/review/save lifecycle is missing")
        try:
            tree = ast.parse(proposed)
            if not any(isinstance(node, ast.FunctionDef) and node.name == "greet" and
                       any(isinstance(stmt, ast.Return) and
                           isinstance(stmt.value, ast.BinOp) and
                           isinstance(stmt.value.right, ast.Constant) and stmt.value.right.value == "!"
                           for stmt in node.body)
                       for node in tree.body):
                issues.append("CodeQ proposal did not add the requested exclamation mark")
        except SyntaxError:
            issues.append("CodeQ proposal was not valid Python")
    return {
        "valid": not issues,
        "issues": issues,
        "host_total_ram_bytes": host_ram,
        "sampled_process_tree_peak_rss_bytes": rss,
        "sampled_app_process_tree_peak_rss_bytes": app_rss,
        "model_bytes": model_bytes,
        "caller_count": len(REQUIRED_CALLERS),
    }
