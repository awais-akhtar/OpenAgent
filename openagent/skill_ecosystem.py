"""Local skill editor, validator, package signer, and supply-chain scanner."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4
import zipfile

from .models import AuditEvent, utc_now
from .security_privacy import SecurityPrivacyManager
from .storage import OpenAgentStore


class SkillEcosystem:
    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.package_root = store.data_dir / "skill_packages"
        self.package_root.mkdir(parents=True, exist_ok=True)
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS developer_skills (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    manifest_json TEXT NOT NULL DEFAULT '{}',
                    source_url TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_developer_skills_project ON developer_skills(project, updated_at DESC);
                CREATE TABLE IF NOT EXISTS developer_skill_tests (
                    id TEXT PRIMARY KEY,
                    skill_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS developer_skill_marketplace (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    skill_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    package_path TEXT NOT NULL,
                    package_sha256 TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    visibility TEXT NOT NULL DEFAULT 'local',
                    status TEXT NOT NULL DEFAULT 'published',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_developer_skill_marketplace
                    ON developer_skill_marketplace(project, status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS developer_skill_installations (
                    id TEXT PRIMARY KEY,
                    listing_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    installed_skill_id TEXT NOT NULL,
                    installed_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'installed',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS developer_skill_sandbox_runs (
                    id TEXT PRIMARY KEY,
                    skill_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )

    def save(self, *, name: str, markdown: str, project: str = "default", version: str = "0.1.0", manifest: dict[str, Any] | None = None, skill_id: str = "", source_url: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name or not markdown.strip():
            raise ValueError("Skill name and Markdown are required.")
        skill_id = skill_id or uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO developer_skills (id, project, name, version, markdown, manifest_json, source_url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name, version=excluded.version,
                    markdown=excluded.markdown, manifest_json=excluded.manifest_json,
                    source_url=excluded.source_url, updated_at=excluded.updated_at
                """,
                (skill_id, project, name, version, markdown, json.dumps(manifest or {}, sort_keys=True), source_url, now, now),
            )
        self._audit("skill.save", skill_id, project, {"version": version})
        return self.get(skill_id)

    def get(self, skill_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM developer_skills WHERE id = ?", (skill_id,)).fetchone()
        if row is None:
            raise KeyError(skill_id)
        result = dict(row)
        result["manifest"] = json.loads(result.pop("manifest_json") or "{}")
        return result

    def list(self, *, project: str = "default", limit: int = 100) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT id FROM developer_skills WHERE project = ? ORDER BY updated_at DESC LIMIT ?", (project, max(1, min(limit, 500)))).fetchall()
        return [self.get(row["id"]) for row in rows]

    def import_url(self, url: str, *, allow_network: bool, project: str = "default") -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError("Skill imports require an HTTPS URL.")
        if not allow_network:
            raise ValueError("Skill URL import requires allow_network=true.")
        request = Request(url, headers={"Accept": "text/markdown,text/plain", "User-Agent": "OpenAgent/skill-import"})
        with urlopen(request, timeout=30) as response:
            content = response.read(2_000_001)
        if len(content) > 2_000_000:
            raise ValueError("Skill import exceeds the 2 MB limit.")
        markdown = content.decode("utf-8", errors="strict")
        name = Path(parsed.path).stem or "Imported skill"
        return self.save(name=name, markdown=markdown, project=project, source_url=url)

    def test(self, skill_id: str) -> dict[str, Any]:
        skill = self.get(skill_id)
        markdown = skill["markdown"]
        manifest = skill["manifest"]
        findings = []
        if not re.search(r"^#\s+.+", markdown, re.MULTILINE):
            findings.append({"severity": "error", "message": "Skill Markdown needs a level-one title."})
        if len(markdown) > 500_000:
            findings.append({"severity": "error", "message": "Skill Markdown exceeds 500 KB."})
        permissions = manifest.get("permissions", [])
        if not isinstance(permissions, list):
            findings.append({"severity": "error", "message": "manifest.permissions must be a list."})
        scan = self.dependency_scan(skill_id)
        findings.extend(scan["findings"])
        status = "passed" if not any(item["severity"] == "error" for item in findings) else "failed"
        report = {"skill_id": skill_id, "status": status, "findings": findings, "permission_preview": self.permission_preview(skill_id)}
        test_id = uuid4().hex
        with self.store.session() as connection:
            connection.execute("INSERT INTO developer_skill_tests (id, skill_id, status, report_json, created_at) VALUES (?, ?, ?, ?, ?)", (test_id, skill_id, status, json.dumps(report, sort_keys=True), utc_now()))
            connection.execute("UPDATE developer_skills SET status = ?, updated_at = ? WHERE id = ?", (status, utc_now(), skill_id))
        self._audit("skill.test", skill_id, skill["project"], {"status": status})
        return {"test_id": test_id, **report}

    def permission_preview(self, skill_id: str) -> dict[str, Any]:
        manifest = self.get(skill_id)["manifest"]
        requested = [str(item) for item in manifest.get("permissions", [])] if isinstance(manifest.get("permissions"), list) else []
        high_risk = [item for item in requested if item in {"network", "desktop", "shell", "secrets", "filesystem:write"}]
        return {"requested": requested, "high_risk": high_risk, "approval_required": bool(high_risk), "default": "deny"}

    def dependency_scan(self, skill_id: str) -> dict[str, Any]:
        manifest = self.get(skill_id)["manifest"]
        dependencies = manifest.get("dependencies", {})
        findings = []
        if dependencies and not isinstance(dependencies, dict):
            findings.append({"severity": "error", "message": "manifest.dependencies must be an object of pinned packages."})
            dependencies = {}
        for name, version in dependencies.items() if isinstance(dependencies, dict) else []:
            text = str(version)
            if not re.search(r"(?:==|===)\s*[0-9]", text):
                findings.append({"severity": "warning", "message": f"Dependency '{name}' is not exactly pinned."})
            if str(name).lower() in {"requests2", "python3-dateutil", "jeilyfish", "colourama"}:
                findings.append({"severity": "error", "message": f"Dependency '{name}' matches a known typo-squatting pattern."})
        return {"skill_id": skill_id, "dependencies": dependencies, "findings": findings}

    def package(self, skill_id: str, *, signing_key: str) -> dict[str, Any]:
        if len(signing_key) < 16:
            raise ValueError("A signing key of at least 16 characters is required.")
        skill = self.get(skill_id)
        report = self.test(skill_id)
        if report["status"] != "passed":
            raise ValueError("Skill tests must pass before packaging.")
        manifest = {**skill["manifest"], "id": skill_id, "name": skill["name"], "version": skill["version"]}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("SKILL.md", skill["markdown"])
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        payload = buffer.getvalue()
        signature = hmac.new(signing_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        path = self.package_root / f"{skill_id}-{skill['version']}.zip"
        path.write_bytes(payload)
        signature_path = path.with_suffix(".zip.sig")
        signature_path.write_text(signature, encoding="ascii")
        self._audit("skill.package", skill_id, skill["project"], {"sha256": hashlib.sha256(payload).hexdigest()})
        return {"skill_id": skill_id, "path": str(path), "signature_path": str(signature_path), "signature": signature, "algorithm": "HMAC-SHA256", "size_bytes": len(payload)}

    def export_markdown(self, skill_id: str) -> dict[str, Any]:
        skill = self.get(skill_id)
        path = self.package_root / f"{skill_id}.md"
        path.write_text(skill["markdown"], encoding="utf-8")
        return {"skill_id": skill_id, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def run_sandbox(
        self,
        skill_id: str,
        *,
        input_data: dict[str, Any] | None = None,
        approved_permissions: list[str] | None = None,
    ) -> dict[str, Any]:
        skill = self.get(skill_id)
        preview = self.permission_preview(skill_id)
        approved = set(str(item) for item in (approved_permissions or []))
        security = SecurityPrivacyManager(self.store)
        profile = security.create_sandbox_profile(
            name=f"skill-{skill_id}-sandbox",
            permissions=sorted(approved),
            network_allowed=False,
            filesystem={"read": [], "write": []},
            env={},
            metadata={"skill_id": skill_id, "declarative_only": True},
        )
        evaluation = security.evaluate_sandbox_request(
            profile_id=profile["id"], permissions=preview["requested"], network=False,
        )
        status = "completed" if evaluation["allowed"] else "blocked"
        input_data = dict(input_data or {})
        result = {
            "skill": {"id": skill_id, "name": skill["name"], "version": skill["version"]},
            "input": input_data,
            "sandbox": evaluation,
            "output": {
                "instruction_digest": hashlib.sha256(skill["markdown"].encode("utf-8")).hexdigest(),
                "preview": skill["markdown"][:1000],
                "declarative": True,
            } if status == "completed" else {},
            "privacy_boundary": "The skill sandbox never executes package code, shell commands, network requests, or filesystem writes.",
        }
        run_id = uuid4().hex
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO developer_skill_sandbox_runs (id, skill_id, project, status, input_json, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, skill_id, skill["project"], status, json.dumps(input_data, sort_keys=True), json.dumps(result, sort_keys=True), utc_now()),
            )
        self._audit("skill.sandbox.run", skill_id, skill["project"], {"run_id": run_id, "status": status})
        return {"id": run_id, "status": status, "result": result}

    def publish(self, skill_id: str, *, signing_key: str, visibility: str = "local") -> dict[str, Any]:
        skill = self.get(skill_id)
        visibility = visibility.strip().lower()
        if visibility not in {"local", "private", "community"}:
            raise ValueError("visibility must be local, private, or community.")
        package = self.package(skill_id, signing_key=signing_key)
        listing_id = hashlib.sha256(f"{skill_id}:{skill['version']}:{visibility}".encode("utf-8")).hexdigest()
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO developer_skill_marketplace (
                    id, project, skill_id, name, version, package_path, package_sha256,
                    signature, visibility, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'published', ?, ?)
                ON CONFLICT(id) DO UPDATE SET package_path=excluded.package_path,
                    package_sha256=excluded.package_sha256, signature=excluded.signature,
                    visibility=excluded.visibility, status='published', updated_at=excluded.updated_at
                """,
                (
                    listing_id, skill["project"], skill_id, skill["name"], skill["version"], package["path"],
                    hashlib.sha256(Path(package["path"]).read_bytes()).hexdigest(), package["signature"], visibility, now, now,
                ),
            )
        self._audit("skill.marketplace.publish", skill_id, skill["project"], {"listing_id": listing_id, "visibility": visibility})
        return self.get_listing(listing_id)

    def marketplace(self, *, project: str = "default", include_removed: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM developer_skill_marketplace WHERE project = ?"
        values: list[Any] = [project]
        if not include_removed:
            query += " AND status != 'removed'"
        query += " ORDER BY updated_at DESC"
        with self.store.session() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._listing_row(row) for row in rows]

    def get_listing(self, listing_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM developer_skill_marketplace WHERE id = ?", (listing_id,)).fetchone()
        if row is None:
            raise KeyError(listing_id)
        return self._listing_row(row)

    def install(self, listing_id: str, *, signing_key: str, project: str = "default") -> dict[str, Any]:
        listing = self.get_listing(listing_id)
        if listing["status"] != "published":
            raise ValueError("Only published skills can be installed.")
        payload = Path(listing["package_path"]).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        signature = hmac.new(signing_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(digest, listing["package_sha256"]) or not hmac.compare_digest(signature, listing["signature"]):
            raise ValueError("Skill package integrity or signature verification failed.")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if set(archive.namelist()) != {"SKILL.md", "manifest.json"}:
                raise ValueError("Skill package contains unexpected files.")
            markdown = archive.read("SKILL.md").decode("utf-8", errors="strict")
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        installed = self.save(
            name=listing["name"], markdown=markdown, project=project, version=listing["version"],
            manifest=manifest, skill_id=uuid4().hex, source_url=f"marketplace:{listing_id}",
        )
        installation_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO developer_skill_installations (id, listing_id, project, installed_skill_id, installed_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'installed', ?, ?)",
                (installation_id, listing_id, project, installed["id"], listing["version"], now, now),
            )
        self._audit("skill.marketplace.install", installed["id"], project, {"listing_id": listing_id, "installation_id": installation_id})
        return self.installation(installation_id)

    def update_installation(self, installation_id: str, *, listing_id: str, signing_key: str) -> dict[str, Any]:
        previous = self.installation(installation_id)
        installed = self.install(listing_id, signing_key=signing_key, project=previous["project"])
        self.uninstall(installation_id)
        return {"previous": previous, "current": installed}

    def uninstall(self, installation_id: str) -> dict[str, Any]:
        installation = self.installation(installation_id)
        with self.store.session() as connection:
            connection.execute("DELETE FROM developer_skill_tests WHERE skill_id = ?", (installation["installed_skill_id"],))
            connection.execute("DELETE FROM developer_skills WHERE id = ?", (installation["installed_skill_id"],))
            connection.execute(
                "UPDATE developer_skill_installations SET status = 'uninstalled', updated_at = ? WHERE id = ?",
                (utc_now(), installation_id),
            )
        self._audit("skill.marketplace.uninstall", installation["installed_skill_id"], installation["project"], {"installation_id": installation_id})
        return self.installation(installation_id)

    def remove_listing(self, listing_id: str) -> dict[str, Any]:
        listing = self.get_listing(listing_id)
        with self.store.session() as connection:
            connection.execute("UPDATE developer_skill_marketplace SET status = 'removed', updated_at = ? WHERE id = ?", (utc_now(), listing_id))
        self._audit("skill.marketplace.remove", listing["skill_id"], listing["project"], {"listing_id": listing_id})
        return self.get_listing(listing_id)

    def installations(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM developer_skill_installations WHERE project = ? ORDER BY updated_at DESC", (project,)
            ).fetchall()
        return [dict(row) for row in rows]

    def installation(self, installation_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM developer_skill_installations WHERE id = ?", (installation_id,)).fetchone()
        if row is None:
            raise KeyError(installation_id)
        return dict(row)

    @staticmethod
    def _listing_row(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"], "project": row["project"], "skill_id": row["skill_id"],
            "name": row["name"], "version": row["version"], "package_path": row["package_path"],
            "package_sha256": row["package_sha256"], "signature": row["signature"],
            "visibility": row["visibility"], "status": row["status"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def _audit(self, action: str, skill_id: str, project: str, metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(AuditEvent(actor="skill-ecosystem", action=action, resource=f"skill:{skill_id}", project=project, metadata=metadata))
