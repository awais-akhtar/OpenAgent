"""Run the bundled small OpenVINO model through OpenAgent's real CPU gateway.

This is a local product smoke check, not a Phase 1 family-route receipt. It
never downloads or converts a model and never enables an external fallback.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from tempfile import TemporaryDirectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] /
                        "models/openvino/qwen2.5-1.5b-instruct-int4-ov")
    parser.add_argument("--runtime-python", type=Path, default=Path(__file__).resolve().parents[1] /
                        ".accelerator-venv/Scripts/python.exe")
    parser.add_argument("--max-model-gib", type=float, default=4.0)
    parser.add_argument("--max-sampled-rss-gib", type=float, default=8.0)
    parser.add_argument("--tasks", action="store_true", help="also exercise production chat, coding, research and agent callers")
    parser.add_argument("--receipt", type=Path, help="write the measured local CPU product observation as JSON")
    parser.add_argument("--wait-for-budget-parent", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    runtime_python = args.runtime_python.resolve()
    if not runtime_python.is_file():
        parser.error("an existing OpenVINO GenAI Python runtime is required; nothing is installed")
    if args.max_model_gib <= 0 or args.max_sampled_rss_gib <= 0:
        parser.error("positive model and sampled-memory budgets are required")
    from openagent.cpu_model_admission import CPUModelAdmissionError, admit_cpu_openvino_bundle
    try:
        model_bytes = admit_cpu_openvino_bundle(
            model_dir, max_bytes=int(args.max_model_gib * 1024**3))
    except CPUModelAdmissionError as exc:
        parser.error(str(exc))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2":
        parser.error("this baseline checks a Qwen2 OpenVINO model, not a MoE route")
    initial_source_pins = None
    initial_bundle_pin = None
    if args.receipt is not None:
        from openagent.cpu_product_gate import bundle_pin, source_pins
        project_root = Path(__file__).resolve().parents[1]
        initial_source_pins = source_pins(project_root)
        initial_bundle_pin = bundle_pin(model_dir)
    if args.wait_for_budget_parent and sys.stdin.readline().strip() != "START":
        raise RuntimeError("CPU budget parent did not admit the run")

    import psutil
    from openagent.adaptive_engine.device_runtime import DirectHardwareClient
    from openagent.adaptive_engine.hardware import HardwareProfiler
    from openagent.model_gateway import ModelGateway

    os.environ["OPENAGENT_MODEL_FALLBACK"] = "none"
    os.environ["OPENAGENT_MODEL_ROUTING"] = "cpu-baseline"
    profiler = HardwareProfiler(model_dir.parent, accelerator_python=str(runtime_python))
    client = DirectHardwareClient(profiler)
    gateway = ModelGateway(direct_hardware=client)
    gateway.providers["openvino-direct"].configure(
        model_path=str(model_dir), device="CPU", model_id="phase1-cpu-small")
    stop = threading.Event()
    peak_rss = 0
    app_peak_rss = 0

    def sample() -> None:
        nonlocal peak_rss, app_peak_rss
        while not stop.wait(0.02):
            process = client._process  # diagnostic only; owned by DirectHardwareClient
            try:
                app = psutil.Process(os.getpid())
                app_processes = [app, *app.children(recursive=True)]
                app_peak_rss = max(app_peak_rss, sum(
                    item.memory_info().rss for item in app_processes if item.is_running()))
                if process is None:
                    continue
                root = psutil.Process(process.pid)
                processes = [root, *root.children(recursive=True)]
                rss = sum(item.memory_info().rss for item in processes if item.is_running())
                peak_rss = max(peak_rss, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    started = time.perf_counter()
    task_report: dict[str, dict[str, object]] = {}
    try:
        result = gateway.complete(
            [{"role": "user", "content": "Say hello in one short sentence."}],
            provider_id="openvino-direct", max_output_tokens=16)
        chunks = list(gateway.stream_complete(
            [{"role": "user", "content": "Say the word ready."}],
            provider_id="openvino-direct", max_output_tokens=16))
        streamed = "".join(chunk.content for chunk in chunks)
        if (result.provider != "openvino-direct" or not result.content.strip() or
                not chunks or not chunks[-1].done or not streamed.strip() or
                any(chunk.provider != "openvino-direct" for chunk in chunks)):
            raise RuntimeError("CPU gateway completion or streaming was incomplete")
        # The resident worker must be retired when a long request is cancelled;
        # the next ordinary gateway request must load into a fresh worker.
        old_worker = client._process
        cancel_request = threading.Event()
        timer = threading.Timer(0.25, cancel_request.set)
        timer.start()
        try:
            try:
                client.generate(
                    "phase1-cpu-small",
                    messages=[{"role": "user", "content": "Count upward slowly from one."}],
                    generation={"max_new_tokens": 256, "do_sample": False},
                    cancelled=cancel_request.is_set)
            except InterruptedError:
                pass
            else:
                raise RuntimeError("long CPU generation completed without observing cancellation")
        finally:
            timer.cancel()
        if client._process is old_worker:
            raise RuntimeError("cancelled CPU worker was not retired")
        recovered = gateway.complete(
            [{"role": "user", "content": "After cancellation, say recovered."}],
            provider_id="openvino-direct", max_output_tokens=16)
        if recovered.provider != "openvino-direct" or not recovered.content.strip():
            raise RuntimeError("CPU gateway did not recover after worker retirement")
        if args.tasks:
            from openagent.agents import AgentRegistry
            from openagent.chat import ChatService
            from openagent.codebase.service import CodebaseService
            from openagent.storage import OpenAgentStore

            with TemporaryDirectory(prefix="openagent-cpu-smoke-") as data_dir:
                store = OpenAgentStore(data_dir)
                chat = ChatService(store, gateway)
                agents = AgentRegistry(store, gateway)
                task_started = time.perf_counter()
                turn = chat.send("Remember my codeword blue otter. Reply briefly about local CPU inference.",
                                 project="cpu-smoke", provider="auto")
                if turn.get("provider") != "openvino-direct" or not str(turn["messages"][-1]["content"]).strip():
                    raise RuntimeError("production chat did not use the local CPU model")
                if "cpu" not in str(turn["messages"][-1]["content"]).lower():
                    raise RuntimeError("production chat missed the CPU question")
                task_report["chat"] = {
                    "elapsed_seconds": round(time.perf_counter() - task_started, 3),
                    "response_characters": len(str(turn["messages"][-1]["content"])),
                    "response_text": str(turn["messages"][-1]["content"]),
                    "provider": turn["provider"],
                }
                task_started = time.perf_counter()
                followup = chat.send(
                    "What codeword did I give you? Reply with only the codeword.",
                    conversation_id=str(turn["conversation_id"]),
                    project="cpu-smoke", provider="auto",
                )
                followup_text = str(followup["messages"][-1]["content"])
                if (followup.get("provider") != "openvino-direct" or
                        followup.get("conversation_id") != turn.get("conversation_id") or
                        "blue otter" not in followup_text.lower()):
                    raise RuntimeError("production chat failed the two-turn CPU context check")
                task_report["chat_followup"] = {
                    "elapsed_seconds": round(time.perf_counter() - task_started, 3),
                    "response_characters": len(followup_text),
                    "response_text": followup_text,
                    "provider": followup["provider"],
                    "same_conversation": True,
                }
                prompts = {
                    "coding": "Write a short Python add(a, b) function and one usage example.",
                    "research": "Give two steps to evaluate a local CPU model offline and name one limitation.",
                    "executive": "The CPU model works here but has not been checked on a 16 GB laptop. State the next concrete test in one sentence.",
                }
                for agent_id, prompt in prompts.items():
                    task_started = time.perf_counter()
                    run = agents.run(agent_id, prompt, project="cpu-smoke")
                    metadata = run.get("metadata") or {}
                    if (metadata.get("provider") != "openvino-direct" or metadata.get("degraded") or
                            not str(run.get("response", "")).strip()):
                        raise RuntimeError(f"production {agent_id} agent did not use the local CPU model")
                    response_text = str(run["response"])
                    if agent_id == "coding":
                        source = response_text.split("```python", 1)[-1].split("```", 1)[0].strip()
                        try:
                            parsed = ast.parse(source)
                        except SyntaxError as exc:
                            raise RuntimeError("coding agent returned invalid Python") from exc
                        if not any(isinstance(node, ast.FunctionDef) and node.name == "add"
                                   for node in parsed.body):
                            raise RuntimeError("coding agent omitted the requested add function")
                    elif agent_id == "research":
                        if not all(marker in response_text.lower() for marker in ("1.", "2.", "limitation")):
                            raise RuntimeError("research agent omitted steps or limitation")
                    elif "16" not in response_text:
                        raise RuntimeError("executive agent omitted the 16 GB qualification task")
                    task_report[agent_id] = {
                        "elapsed_seconds": round(time.perf_counter() - task_started, 3),
                        "response_characters": len(str(run["response"])),
                        "response_text": response_text,
                        "provider": metadata["provider"],
                    }
                source = Path(data_dir) / "source"
                source.mkdir()
                original = 'def greet(name):\n    return "Hello, " + name\n'
                (source / "hello.py").write_text(original, encoding="utf-8")
                codebase = CodebaseService(store, gateway)
                project = codebase.create_project(name="CPU smoke", source_path=str(source))
                task_started = time.perf_counter()
                explanation = codebase.ask_project(
                    project["id"], "Explain what greet returns.", paths=["hello.py"])
                answer = str(explanation["answer"])
                if (explanation["provider"] != "openvino-direct" or "Hello" not in answer or
                        (source / "hello.py").read_text(encoding="utf-8") != original or
                        codebase.list_runs(project=project["project"])):
                    raise RuntimeError("read-only codebase question did not complete safely on CPU")
                task_report["codebase_ask"] = {
                    "elapsed_seconds": round(time.perf_counter() - task_started, 3),
                    "response_characters": len(answer),
                    "response_text": answer,
                    "provider": explanation["provider"],
                    "source_unchanged": True,
                }
                task_started = time.perf_counter()
                current_hash = str(codebase.read_file(project["id"], "hello.py")["sha256"])
                proposal = codebase.propose_file_edit(
                    project["id"], "hello.py",
                    'Replace return "Hello, " + name with return "Hello, " + name + "!".',
                    expected_sha256=current_hash,
                )
                if ((source / "hello.py").read_text(encoding="utf-8") != original or
                        proposal["provider"] != "openvino-direct" or
                        'name + "!"' not in proposal["proposed_content"]):
                    raise RuntimeError("local edit proposal changed source or missed the requested edit")
                codebase.write_file(
                    project["id"], "hello.py", str(proposal["proposed_content"]),
                    expected_sha256=str(proposal["expected_sha256"]),
                )
                if (source / "hello.py").read_text(encoding="utf-8") != proposal["proposed_content"]:
                    raise RuntimeError("reviewed CodeQ edit did not save to selected source")
                task_report["codebase_edit_proposal"] = {
                    "elapsed_seconds": round(time.perf_counter() - task_started, 3),
                    "response_characters": len(proposal["proposed_content"]),
                    "response_text": proposal["proposed_content"],
                    "provider": proposal["provider"],
                    "source_unchanged_before_review": True,
                    "saved_after_review": True,
                }
    finally:
        stop.set()
        watcher.join(timeout=2)
        client.close()

    report = {
        "kind": "local-cpu-product-smoke-not-family-route-receipt",
        "model_type": config["model_type"],
        "backend": "openvino-direct",
        "device": "CPU",
        "model_bytes": model_bytes,
        "host_total_ram_bytes": psutil.virtual_memory().total,
        "sampled_process_tree_peak_rss_bytes": peak_rss,
        "sampled_app_process_tree_peak_rss_bytes": app_peak_rss,
        "sampled_rss_limit_bytes": int(args.max_sampled_rss_gib * 1024**3),
        "complete_text": result.content,
        "stream_text": streamed,
        "cancelled_worker_retired": True,
        "post_cancel_gateway_reload": True,
        "task_callers": task_report,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if (peak_rss == 0 or peak_rss > report["sampled_rss_limit_bytes"] or
            app_peak_rss < peak_rss or app_peak_rss > report["sampled_rss_limit_bytes"]):
        print(json.dumps(report, sort_keys=True))
        return 1
    if args.receipt is not None:
        from openagent.cpu_product_gate import bundle_pin, source_pins

        if (source_pins(project_root) != initial_source_pins or
                bundle_pin(model_dir) != initial_bundle_pin):
            raise RuntimeError("CPU observation source or model bundle changed during capture")
        report["model_bundle_sha256"] = initial_bundle_pin
        report["source_sha256"] = initial_source_pins
        report["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
        receipt = args.receipt.resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        pending = receipt.with_suffix(receipt.suffix + ".pending")
        pending.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pending.replace(receipt)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
