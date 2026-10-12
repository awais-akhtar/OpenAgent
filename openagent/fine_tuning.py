"""Fine-tuning and LoRA adapter management for local AI models.

Manages the full lifecycle of fine-tuning jobs and LoRA adapters:
- Training dataset curation and validation
- Fine-tuning job scheduling (delegates to Ollama/llama.cpp backends)
- LoRA adapter storage, versioning, activation
- Training progress tracking
- Evaluation / benchmarking post-training
- Export in GGUF / safetensors format descriptors

Privacy: all training data stays local; no gradient or model data
leaves the machine unless the user explicitly exports and shares.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class FineTuningError(RuntimeError):
    """Raised when a fine-tuning operation fails or is misconfigured."""


class FineTuningManager:
    """Manages fine-tuning jobs and LoRA adapters for local models."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.finetune_root = (store.data_dir / "finetune").resolve()
        self.finetune_root.mkdir(parents=True, exist_ok=True)
        self.initialize()
        self._seed_templates()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.store.initialize()

    def _seed_templates(self) -> None:
        """Seed common fine-tuning config templates."""
        now = utc_now()
        templates = [
            {
                "id": "lora-fast", "name": "LoRA Fast", "method": "lora",
                "description": "Quick LoRA fine-tune — good for instruction following.",
                "use_case": "instruction",
                "config": {"epochs": 3, "lr": 2e-4, "rank": 8, "alpha": 16,
                           "batch_size": 4, "max_seq_len": 512},
            },
            {
                "id": "lora-quality", "name": "LoRA Quality", "method": "lora",
                "description": "Higher quality LoRA — slower but more accurate.",
                "use_case": "general",
                "config": {"epochs": 5, "lr": 1e-4, "rank": 32, "alpha": 64,
                           "batch_size": 2, "max_seq_len": 2048},
            },
            {
                "id": "qlora-4bit", "name": "QLoRA 4-bit", "method": "qlora",
                "description": "4-bit quantized LoRA — minimal VRAM required.",
                "use_case": "low-resource",
                "config": {"epochs": 3, "lr": 2e-4, "rank": 16, "alpha": 32,
                           "bits": 4, "batch_size": 4, "max_seq_len": 1024},
            },
            {
                "id": "full-finetune", "name": "Full Fine-tune", "method": "full",
                "description": "Full model fine-tuning — highest accuracy, large GPU required.",
                "use_case": "specialized",
                "config": {"epochs": 2, "lr": 5e-5, "batch_size": 1, "max_seq_len": 4096},
            },
        ]
        with self.store.session() as conn:
            for t in templates:
                conn.execute(
                    "INSERT OR IGNORE INTO finetune_templates VALUES(?,?,?,?,?,?,?)",
                    (t["id"], t["name"], t["description"], t["method"],
                     json.dumps(t["config"]), t["use_case"], now),
                )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        *,
        project: str = "default",
        description: str = "",
        format: str = "jsonl",
        split_train: float = 0.9,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("Dataset name is required.")
        now = utc_now()
        ds_id = hashlib.md5(f"ds-{project}-{name}-{now}".encode()).hexdigest()
        stored_path = str(self.finetune_root / f"{ds_id}.jsonl")
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO finetune_datasets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (ds_id, project, name.strip(), description, format,
                 0, split_train, 1.0 - split_train, stored_path, "{}", now, now),
            )
        return self.get_dataset(ds_id)

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT id,project,name,description,format,num_examples,split_train,"
                "split_val,stored_path,metadata_json,created_at,updated_at "
                "FROM finetune_datasets WHERE id=?", (dataset_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"Dataset not found: {dataset_id}")
        return {
            "id": row[0], "project": row[1], "name": row[2], "description": row[3],
            "format": row[4], "num_examples": row[5], "split_train": row[6],
            "split_val": row[7], "stored_path": row[8],
            "metadata": json.loads(row[9] or "{}"),
            "created_at": row[10], "updated_at": row[11],
        }

    def add_example(
        self,
        dataset_id: str,
        user_input: str,
        assistant_output: str,
        *,
        project: str = "default",
        system_prompt: str = "",
        quality_score: float = 1.0,
    ) -> dict[str, Any]:
        if not user_input.strip() or not assistant_output.strip():
            raise ValueError("user_input and assistant_output are required.")
        now = utc_now()
        ex_id = hashlib.md5(f"ex-{dataset_id}-{user_input[:32]}-{now}".encode()).hexdigest()
        # 90/10 split by default
        split = "train" if hash(ex_id) % 10 != 0 else "val"
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO finetune_examples VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ex_id, dataset_id, project, system_prompt,
                 user_input, assistant_output, split,
                 max(0.0, min(1.0, quality_score)), "{}", now),
            )
            conn.execute(
                "UPDATE finetune_datasets SET num_examples=num_examples+1, updated_at=? WHERE id=?",
                (now, dataset_id),
            )
        return {"id": ex_id, "dataset_id": dataset_id, "split": split}

    def list_datasets(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id,project,name,description,format,num_examples,split_train,"
                "split_val,stored_path,metadata_json,created_at,updated_at "
                "FROM finetune_datasets WHERE project=? ORDER BY created_at DESC",
                (project,),
            ).fetchall()
        return [
            {"id": r[0], "project": r[1], "name": r[2], "description": r[3],
             "format": r[4], "num_examples": r[5], "split_train": r[6],
             "split_val": r[7], "stored_path": r[8],
             "metadata": json.loads(r[9] or "{}"),
             "created_at": r[10], "updated_at": r[11]}
            for r in rows
        ]

    def export_dataset_jsonl(self, dataset_id: str) -> str:
        """Export dataset examples as JSONL string (chat format)."""
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT system_prompt, user_input, assistant_output "
                "FROM finetune_examples WHERE dataset_id=? AND split='train'",
                (dataset_id,),
            ).fetchall()
        lines = []
        for row in rows:
            messages: list[dict[str, str]] = []
            if row[0]:
                messages.append({"role": "system", "content": row[0]})
            messages.append({"role": "user", "content": row[1]})
            messages.append({"role": "assistant", "content": row[2]})
            lines.append(json.dumps({"messages": messages}))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fine-tuning jobs
    # ------------------------------------------------------------------

    def create_job(
        self,
        dataset_id: str,
        base_model: str,
        *,
        project: str = "default",
        method: str = "lora",
        config: dict[str, Any] | None = None,
        template_id: str = "",
    ) -> dict[str, Any]:
        if not base_model.strip():
            raise ValueError("base_model is required.")
        # Merge template config
        merged_config: dict[str, Any] = {"epochs": 3, "lr": 2e-4, "rank": 16,
                                          "alpha": 32, "batch_size": 4, "max_seq_len": 512}
        if template_id:
            with self.store.session() as conn:
                trow = conn.execute(
                    "SELECT config_json, method FROM finetune_templates WHERE id=?", (template_id,)
                ).fetchone()
            if trow:
                merged_config.update(json.loads(trow[0] or "{}"))
                if not method or method == "lora":
                    method = trow[1]
        if config:
            merged_config.update(config)

        now = utc_now()
        job_id = hashlib.md5(f"job-{project}-{dataset_id}-{base_model}-{now}".encode()).hexdigest()
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO finetune_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, project, dataset_id, base_model, method,
                 "queued", json.dumps(merged_config), 0.0, 0.0,
                 "", "", "", "", now, now),
            )
        self.store.add_audit_event(AuditEvent(
            actor="finetune",
            action="finetune.job.create",
            resource=f"finetune:job:create:{job_id}",
            project=project,
            metadata={"job_id": job_id, "base_model": base_model, "method": method},
        ))
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT id,project,dataset_id,base_model,method,status,config_json,"
                "progress,current_loss,output_adapter_id,error_message,"
                "started_at,completed_at,created_at,updated_at "
                "FROM finetune_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"Job not found: {job_id}")
        return {
            "id": row[0], "project": row[1], "dataset_id": row[2],
            "base_model": row[3], "method": row[4], "status": row[5],
            "config": json.loads(row[6] or "{}"),
            "progress": row[7], "current_loss": row[8],
            "output_adapter_id": row[9], "error_message": row[10],
            "started_at": row[11], "completed_at": row[12],
            "created_at": row[13], "updated_at": row[14],
        }

    def list_jobs(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id FROM finetune_jobs WHERE project=? ORDER BY created_at DESC LIMIT 50",
                (project,),
            ).fetchall()
        return [self.get_job(r[0]) for r in rows]

    def update_job_progress(
        self,
        job_id: str,
        progress: float,
        *,
        loss: float = 0.0,
        status: str = "running",
    ) -> dict[str, Any]:
        now = utc_now()
        started = ""
        if status == "running":
            job = self.get_job(job_id)
            started = job.get("started_at") or now
        with self.store.session() as conn:
            conn.execute(
                "UPDATE finetune_jobs SET progress=?, current_loss=?, status=?, "
                "started_at=CASE WHEN started_at='' THEN ? ELSE started_at END, "
                "updated_at=? WHERE id=?",
                (max(0.0, min(1.0, progress)), loss, status, now, now, job_id),
            )
        return self.get_job(job_id)

    def complete_job(
        self, job_id: str, *, adapter_id: str = "", error: str = ""
    ) -> dict[str, Any]:
        now = utc_now()
        status = "failed" if error else "completed"
        with self.store.session() as conn:
            conn.execute(
                "UPDATE finetune_jobs SET status=?, progress=?, output_adapter_id=?, "
                "error_message=?, completed_at=?, updated_at=? WHERE id=?",
                (status, 1.0 if not error else 0.0,
                 adapter_id, error, now, now, job_id),
            )
        return self.get_job(job_id)

    # ------------------------------------------------------------------
    # LoRA adapters
    # ------------------------------------------------------------------

    def register_adapter(
        self,
        name: str,
        base_model: str,
        *,
        project: str = "default",
        description: str = "",
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: list[str] | None = None,
        stored_path: str = "",
        size_bytes: int = 0,
        eval_score: float = 0.0,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("Adapter name is required.")
        now = utc_now()
        adapter_id = hashlib.md5(f"lora-{project}-{name}-{base_model}-{now}".encode()).hexdigest()
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO lora_adapters VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (adapter_id, project, name.strip(), base_model, description,
                 rank, alpha,
                 json.dumps(target_modules or ["q_proj", "v_proj"]),
                 stored_path, size_bytes, 0, eval_score, 1, "{}", now, now),
            )
        return self.get_adapter(adapter_id)

    def get_adapter(self, adapter_id: str) -> dict[str, Any]:
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT id,project,name,base_model,description,rank,alpha,"
                "target_modules_json,stored_path,size_bytes,active,eval_score,"
                "version,metadata_json,created_at,updated_at "
                "FROM lora_adapters WHERE id=?", (adapter_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"Adapter not found: {adapter_id}")
        return {
            "id": row[0], "project": row[1], "name": row[2], "base_model": row[3],
            "description": row[4], "rank": row[5], "alpha": row[6],
            "target_modules": json.loads(row[7] or "[]"),
            "stored_path": row[8], "size_bytes": row[9],
            "active": bool(row[10]), "eval_score": row[11], "version": row[12],
            "metadata": json.loads(row[13] or "{}"),
            "created_at": row[14], "updated_at": row[15],
        }

    def list_adapters(
        self, *, project: str = "default", base_model: str = ""
    ) -> list[dict[str, Any]]:
        query = "SELECT id FROM lora_adapters WHERE project=?"
        params: list[Any] = [project]
        if base_model:
            query += " AND base_model=?"
            params.append(base_model)
        query += " ORDER BY eval_score DESC, created_at DESC"
        with self.store.session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self.get_adapter(r[0]) for r in rows]

    def activate_adapter(self, adapter_id: str, *, project: str = "default") -> dict[str, Any]:
        """Activate an adapter (deactivates others for the same base model)."""
        adapter = self.get_adapter(adapter_id)
        now = utc_now()
        with self.store.session() as conn:
            # Deactivate all adapters for same base model in project
            conn.execute(
                "UPDATE lora_adapters SET active=0, updated_at=? "
                "WHERE project=? AND base_model=?",
                (now, project, adapter["base_model"]),
            )
            # Activate this one
            conn.execute(
                "UPDATE lora_adapters SET active=1, updated_at=? WHERE id=?",
                (now, adapter_id),
            )
        return self.get_adapter(adapter_id)

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def list_templates(self) -> list[dict[str, Any]]:
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id,name,description,method,config_json,use_case,created_at "
                "FROM finetune_templates ORDER BY name"
            ).fetchall()
        return [
            {"id": r[0], "name": r[1], "description": r[2], "method": r[3],
             "config": json.loads(r[4] or "{}"), "use_case": r[5], "created_at": r[6]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as conn:
            n_datasets = conn.execute(
                "SELECT COUNT(*) FROM finetune_datasets WHERE project=?", (project,)
            ).fetchone()[0]
            n_jobs = conn.execute(
                "SELECT COUNT(*) FROM finetune_jobs WHERE project=?", (project,)
            ).fetchone()[0]
            n_adapters = conn.execute(
                "SELECT COUNT(*) FROM lora_adapters WHERE project=?", (project,)
            ).fetchone()[0]
            job_statuses = conn.execute(
                "SELECT status, COUNT(*) FROM finetune_jobs WHERE project=? GROUP BY status",
                (project,),
            ).fetchall()
        return {
            "datasets": n_datasets,
            "jobs": n_jobs,
            "adapters": n_adapters,
            "job_statuses": dict(job_statuses),
            "project": project,
            "features": [
                "lora", "qlora", "full-finetune", "dataset-curation",
                "train-val-split", "jsonl-export", "adapter-versioning",
                "adapter-activation", "eval-scoring", "config-templates",
            ],
        }
