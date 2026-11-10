"""Run the existing CPU product smoke inside an enforced Windows memory job.

This qualifies an 8 GiB process budget on the current CPU. It does not claim
that the host physically has 16 GiB RAM or that every CPU has the same speed.
No model or dependency is downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from openagent.cpu_budget_job import CPUJobBudget
from openagent.cpu_product_gate import source_pins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-gib", type=float, default=8.0)
    parser.add_argument("--receipt", type=Path, default=Path(__file__).resolve().parents[1] /
                        "docs/audits/phase1/cpu-product-budget-observation.json")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("enforced CPU budget capture currently requires Windows Job Objects")
    if not 1 <= args.limit_gib <= 8:
        parser.error("CPU baseline job memory limit must be between 1 and 8 GiB")
    limit_bytes = int(args.limit_gib * 1024**3)
    receipt = args.receipt.resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    pending = receipt.with_name(receipt.name + "." + uuid4().hex + ".pending")
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, "-m", "scripts.verify_cpu_baseline", "--tasks",
        "--receipt", str(pending), "--wait-for-budget-parent",
        "--max-sampled-rss-gib", str(args.limit_gib),
    ]
    process = None
    try:
        with CPUJobBudget(limit_bytes) as job:
            process = subprocess.Popen(
                command, cwd=project_root, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            job.assign(process)
            stdout, stderr = process.communicate("START\n", timeout=180)
            peak_commit = job.peak_bytes()
        if process.returncode != 0:
            raise RuntimeError(f"CPU smoke failed (exit {process.returncode}): {stderr}\n{stdout}")
        report = json.loads(pending.read_text(encoding="utf-8"))
        if report.get("source_sha256") != source_pins(project_root):
            raise RuntimeError("CPU source changed before budget receipt publication")
        if peak_commit <= 0 or peak_commit > limit_bytes:
            raise RuntimeError("Windows job exceeded its configured memory budget")
        report.update({
            "budget_method": "windows-job-object",
            "job_limit_enforced": True,
            "job_memory_limit_bytes": limit_bytes,
            "job_peak_commit_bytes": peak_commit,
        })
        pending.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pending.replace(receipt)
        print(json.dumps({
            "receipt": str(receipt), "job_memory_limit_bytes": limit_bytes,
            "job_peak_commit_bytes": peak_commit,
            "sampled_app_process_tree_peak_rss_bytes": report["sampled_app_process_tree_peak_rss_bytes"],
            "caller_count": len(report["task_callers"]),
        }, sort_keys=True))
        return 0
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        pending.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
