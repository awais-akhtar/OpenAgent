"""Permission-aware tools exposed to the software-engineering controller."""

from __future__ import annotations

import difflib
import hashlib
import inspect
import json
from pathlib import Path
import re
import shlex
import shutil
from urllib.parse import urlparse
from typing import Any, Callable
from uuid import uuid4

from .repository import RepositoryIntelligence
from .sandbox import SandboxManager
from ..tools.browser import BrowserRuntime


RISK = {"automatic": 0, "sandbox": 1, "approval": 2, "denied": 3}
DENIED_COMMANDS = {"format", "mkfs", "shutdown", "reboot", "diskpart", "reg", "regedit", "net", "netsh", "sc", "sudo", "su"}
MUTATING_GIT = {"push", "reset", "clean", "rebase", "merge", "cherry-pick", "tag"}
SAFE_RUNNERS = {
    "python", "python.exe", "pytest", "ruff", "mypy", "node", "npm", "npm.cmd", "npx", "pnpm", "yarn",
    "git", "cargo", "rustc", "go", "dotnet", "java", "javac", "mvn", "gradle", "gradlew", "gradlew.bat",
    "cmake", "ctest", "ninja", "make", "zig", "ruby", "bundle", "php", "composer", "elixir", "mix",
    "dart", "flutter", "swift", "xcodebuild", "forge", "solc", "godot", "godot4",
}
SECRET_RE = re.compile(r"(?i)(token|secret|password|api[_-]?key|private[_-]?key)\s*[:=]\s*([^\s\"']+)")
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}


class ApprovalRequired(PermissionError):
    def __init__(self, tool: str, reason: str, payload: dict[str, Any]) -> None:
        super().__init__(reason)
        self.tool = tool
        self.reason = reason
        self.payload = payload


class ToolPolicy:
    """Classify tool operations before they reach a workspace backend."""

    def classify(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool in {"list_directory", "read_file", "read_files", "grep", "search_symbol", "find_references", "project_map", "git_status", "git_diff", "git_log", "process_status", "inspect_logs"}:
            return {"risk": 0, "label": "automatic", "reason": "read-only workspace operation"}
        if tool == "install_dependency":
            raw_command = arguments.get("command", [])
            command = shlex.split(raw_command, posix=False) if isinstance(raw_command, str) else [str(item) for item in raw_command]
            executable = Path(command[0]).name.lower() if command else ""
            package_managers = {
                "python", "python.exe", "pip", "pip.exe", "uv", "npm", "npm.cmd", "pnpm", "yarn", "poetry",
                "cargo", "go", "dotnet", "mvn", "gradle", "gradlew", "gradlew.bat", "bundle", "gem", "composer",
                "mix", "dart", "flutter", "swift", "forge",
            }
            if executable not in package_managers:
                return {"risk": 3, "label": "denied", "reason": "dependency setup must use a recognized package manager"}
            return {"risk": 1, "label": "sandbox", "reason": "dependency setup inside protected workspace"}
        if tool in {
            "write_file", "patch_file", "delete_file", "move_file", "run_tests", "run_linter",
            "run_typechecker", "run_build", "browser_test", "stop_process",
            "git_commit", "git_checkout", "git_revert", "git_reset", "restore_snapshot",
        }:
            return {"risk": 1, "label": "sandbox", "reason": "bounded sandbox operation"}
        if tool in {"browser_inspect", "browser_automate"}:
            raw_url = str(arguments.get("url", "")).strip()
            parsed = urlparse(raw_url)
            if not raw_url or parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return {
                    "risk": 3,
                    "label": "denied",
                    "reason": f"{tool} requires a complete HTTP URL; create the page and start its local server before inspecting it",
                }
            host = (parsed.hostname or "").lower()
            risk = 1 if host in {"127.0.0.1", "localhost", "::1"} else 2
            return {"risk": risk, "label": "sandbox" if risk == 1 else "approval", "reason": "local browser inspection" if risk == 1 else "external browser navigation requires approval"}
        if tool == "start_process":
            return self.classify("run_shell", {"command": arguments.get("command", [])})
        if tool == "run_shell":
            raw_command = arguments.get("command", [])
            if isinstance(raw_command, str):
                source = raw_command.strip().lower()
                inline_prefixes = ("def ", "async def ", "class ", "import ", "from ", "print(", "@dataclass", "if __name__")
                if source.startswith(inline_prefixes) or ("\n" in source and not source.startswith(tuple(f"{runner} " for runner in SAFE_RUNNERS))):
                    return {
                        "risk": 3,
                        "label": "denied",
                        "reason": "inline source code is not an executable command; use write_file to create a source file, then run it with a command such as python file.py",
                    }
            command = shlex.split(raw_command, posix=False) if isinstance(raw_command, str) else [str(item) for item in raw_command]
            executable = Path(command[0]).name.lower() if command else ""
            if executable in DENIED_COMMANDS:
                return {"risk": 3, "label": "denied", "reason": "system administration commands are never autonomous"}
            if executable == "git" and len(command) > 1 and command[1].lower() in MUTATING_GIT:
                return {"risk": 2, "label": "approval", "reason": "mutating Git operations require approval"}
            if executable not in SAFE_RUNNERS:
                return {"risk": 2, "label": "approval", "reason": "unrecognized executable requires approval"}
            if executable in {"npm", "npm.cmd", "npx", "pnpm", "yarn"} and any(item.lower() in {"install", "add", "publish"} for item in command[1:]):
                if any(item.lower() == "publish" for item in command[1:]):
                    return {"risk": 3, "label": "denied", "reason": "package publishing is outside the protected autonomous workspace"}
                return {"risk": 1, "label": "sandbox", "reason": "dependency setup inside protected workspace"}
            return {"risk": 1, "label": "sandbox", "reason": "recognized command in isolated workspace"}
        if tool in {"git_push", "deploy", "drop_database", "send_email", "purchase_service", "change_dns"}:
            return {"risk": 3, "label": "denied", "reason": "operation is outside autonomous scope"}
        return {"risk": 2, "label": "approval", "reason": "unknown tool requires approval"}


class CodebaseTools:
    """Filesystem, search, command, Git, and validation tools for one workspace."""

    def __init__(self, sandbox: SandboxManager, workspace_id: str, *, backend: str = "local", event: Callable[..., None] | None = None) -> None:
        self.sandbox = sandbox
        self.workspace_id = workspace_id
        self.backend = backend
        self.root = sandbox.path(workspace_id)
        self.repo = RepositoryIntelligence(self.root)
        self.policy = ToolPolicy()
        self.event = event or (lambda *args, **kwargs: None)

    def execute(self, tool: str, arguments: dict[str, Any], *, approved: bool = False) -> dict[str, Any]:
        arguments = self._normalize_arguments(tool, arguments)
        if tool == "browser_automate":
            arguments["approved"] = approved
        handler = getattr(self, f"tool_{tool}", None)
        if handler is None:
            raise ValueError(f"unknown codebase tool: {tool}")
        accepted = set(inspect.signature(handler).parameters)
        arguments = {key: value for key, value in arguments.items() if key in accepted}
        decision = self.policy.classify(tool, arguments)
        if decision["risk"] == 3:
            raise PermissionError(decision["reason"])
        if decision["risk"] == 2 and not approved:
            raise ApprovalRequired(tool, decision["reason"], arguments)
        self.event("tool.started", tool=tool, arguments=self._safe_arguments(arguments), risk=decision)
        try:
            result = handler(**arguments)
        except Exception as exc:
            self.event("tool.failed", tool=tool, error=str(exc))
            raise
        self.event("tool.completed", tool=tool, result=self._bounded_result(result))
        return result

    @staticmethod
    def _normalize_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Accept the small vocabulary differences produced by local models."""
        normalized = dict(arguments or {})
        aliases = {
            "read_file": {"file": "path", "filename": "path"},
            "read_files": {"files": "paths", "filenames": "paths"},
            "write_file": {"file": "path", "filename": "path", "text": "content", "body": "content", "code": "content"},
            "patch_file": {"file": "path", "filename": "path", "old": "search", "new": "replace"},
            "delete_file": {"file": "path", "filename": "path"},
            "grep": {"pattern": "query", "text": "query"},
            "search_symbol": {"query": "name", "symbol": "name"},
            "find_references": {"query": "symbol", "name": "symbol", "text": "symbol"},
            "run_shell": {"cmd": "command", "argv": "command"},
            "start_process": {"cmd": "command", "argv": "command"},
        }.get(tool, {})
        for alias, canonical in aliases.items():
            if canonical not in normalized and alias in normalized:
                normalized[canonical] = normalized.pop(alias)
        return normalized

    def tool_list_directory(self, path: str = "", recursive: bool = False) -> dict[str, Any]:
        target = self.repo.resolve(path, must_exist=True)
        iterator = target.rglob("*") if recursive else target.iterdir()
        entries = []
        for item in iterator:
            rel = item.relative_to(self.root).as_posix()
            if any(part in {".git", "node_modules", ".venv", "__pycache__"} for part in item.relative_to(self.root).parts):
                continue
            if self._is_sensitive_path(rel):
                continue
            entries.append({"path": rel, "type": "directory" if item.is_dir() else "file", "size": item.stat().st_size if item.is_file() else 0})
            if len(entries) >= 5000:
                break
        return {"entries": entries, "truncated": len(entries) >= 5000}

    def tool_read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> dict[str, Any]:
        self._assert_not_sensitive(path)
        target = self.repo.resolve(path, must_exist=True)
        if not target.is_file() or target.stat().st_size > 2_000_000:
            raise ValueError("file is not readable text or exceeds 2 MB")
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line) if int(end_line or 0) > 0 else start + 1000)
        content = "\n".join(lines[start - 1:end])
        return {"path": path, "start_line": start, "end_line": end, "content": content, "sha256": hashlib.sha256(text.encode()).hexdigest()}

    def tool_read_files(self, paths: list[str]) -> dict[str, Any]:
        return {"files": [self.tool_read_file(str(path)) for path in paths[:30]]}

    def tool_write_file(self, path: str, content: str, expected_sha256: str = "") -> dict[str, Any]:
        self._assert_not_sensitive(path)
        target = self.repo.resolve(path)
        old = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
        if expected_sha256 and hashlib.sha256(old.encode()).hexdigest() != expected_sha256:
            raise ValueError("file changed since it was read")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8", newline="\n")
        return self._change_result(path, old, str(content))

    def tool_patch_file(self, path: str, search: str, replace: str, expected_occurrences: int = 1) -> dict[str, Any]:
        self._assert_not_sensitive(path)
        target = self.repo.resolve(path, must_exist=True)
        old = target.read_text(encoding="utf-8", errors="replace")
        occurrences = old.count(search)
        if occurrences != int(expected_occurrences):
            raise ValueError(f"patch expected {expected_occurrences} occurrence(s), found {occurrences}")
        new = old.replace(search, replace)
        target.write_text(new, encoding="utf-8", newline="\n")
        return self._change_result(path, old, new)

    def tool_delete_file(self, path: str) -> dict[str, Any]:
        self._assert_not_sensitive(path)
        target = self.repo.resolve(path, must_exist=True)
        if target.is_dir():
            raise ValueError("directory deletion is not supported by this tool")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        target.unlink()
        return {"path": path, "deleted": True, "old_sha256": digest, "new_sha256": "", "patch": f"deleted {path}"}

    def tool_move_file(self, source: str, destination: str) -> dict[str, Any]:
        self._assert_not_sensitive(source)
        self._assert_not_sensitive(destination)
        old = self.repo.resolve(source, must_exist=True)
        new = self.repo.resolve(destination)
        digest = hashlib.sha256(old.read_bytes()).hexdigest() if old.is_file() else ""
        new.parent.mkdir(parents=True, exist_ok=True)
        old.replace(new)
        return {"path": destination, "source": source, "destination": destination, "old_sha256": digest, "new_sha256": digest, "patch": f"moved {source} to {destination}"}

    def tool_grep(self, query: str, limit: int = 100) -> dict[str, Any]:
        return {"matches": self.repo.search(query, limit=limit)}

    def tool_search_symbol(self, name: str) -> dict[str, Any]:
        project_map = self.repo.scan()
        needle = name.lower()
        return {"symbols": [item for item in project_map["symbols"] if needle in item["name"].lower()][:200]}

    def tool_find_references(self, symbol: str, limit: int = 200) -> dict[str, Any]:
        return self.tool_grep(symbol, limit=limit)

    def tool_project_map(self) -> dict[str, Any]:
        return self.repo.scan()

    def tool_run_shell(self, command: list[str] | str, cwd: str = "", timeout_seconds: int = 120) -> dict[str, Any]:
        parts = shlex.split(command, posix=False) if isinstance(command, str) else [str(item) for item in command]
        if not parts:
            raise ValueError("command is required")
        result = self.sandbox.run(self.workspace_id, parts, cwd=cwd, backend=self.backend, timeout_seconds=timeout_seconds)
        self.event("command.stdout", command=parts, chunk=self._redact(result.stdout[-20_000:]))
        if result.stderr:
            self.event("command.stderr", command=parts, chunk=self._redact(result.stderr[-20_000:]))
        return self._redact_result(result.to_dict())

    def tool_run_tests(self, command: list[str] | str | None = None) -> dict[str, Any]:
        return self._run_named("test", command)

    def tool_run_linter(self, command: list[str] | str | None = None) -> dict[str, Any]:
        return self._run_named("lint", command)

    def tool_run_typechecker(self, command: list[str] | str | None = None) -> dict[str, Any]:
        return self._run_named("typecheck", command)

    def tool_run_build(self, command: list[str] | str | None = None) -> dict[str, Any]:
        return self._run_named("build", command)

    def tool_browser_test(self, command: list[str] | str | None = None) -> dict[str, Any]:
        value = command or ["npx", "playwright", "test"]
        return self.tool_run_shell(value, timeout_seconds=600)

    def tool_start_process(self, command: list[str] | str, cwd: str = "") -> dict[str, Any]:
        parts = shlex.split(command, posix=False) if isinstance(command, str) else [str(item) for item in command]
        if not parts:
            raise ValueError("command is required")
        result = self.sandbox.start_process(self.workspace_id, parts, cwd=cwd, backend=self.backend)
        self.event("process.started", process=result)
        return result

    def tool_process_status(self, process_id: str) -> dict[str, Any]:
        return self.sandbox.process_status(self.workspace_id, process_id)

    def tool_stop_process(self, process_id: str) -> dict[str, Any]:
        result = self.sandbox.stop_process(self.workspace_id, process_id)
        self.event("process.stopped", process=result)
        return result

    def tool_inspect_logs(self, process_id: str, limit: int = 100_000) -> dict[str, Any]:
        return self._redact_result(self.sandbox.process_logs(self.workspace_id, process_id, limit=limit))

    def tool_browser_inspect(self, url: str, actions: list[dict[str, Any]] | None = None, timeout_seconds: int = 120) -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("browser URL must be HTTP or HTTPS")
        probe_dir = self.root / ".openagent" / "browser-artifacts"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_id = uuid4().hex
        script_path = probe_dir / f"probe-{probe_id}.cjs"
        payload_path = probe_dir / f"probe-{probe_id}.json"
        screenshot_path = probe_dir / f"probe-{probe_id}.png"
        payload_path.write_text(json.dumps({"url": url, "actions": (actions or [])[:30], "screenshot": str(screenshot_path)}), encoding="utf-8")
        script_path.write_text(
            """const fs=require('fs'); const {chromium}=require('playwright');
const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
(async()=>{const browser=await chromium.launch({headless:true}); const page=await browser.newPage();
const consoleErrors=[]; page.on('console',m=>{if(m.type()==='error') consoleErrors.push(m.text())}); page.on('pageerror',e=>consoleErrors.push(e.message));
const evidence=[]; await page.goto(input.url,{waitUntil:'domcontentloaded'}); evidence.push({action:'goto',url:page.url(),title:await page.title()});
for(const step of input.actions){if(step.action==='wait'){await page.waitForTimeout(Number(step.ms||250)); evidence.push({action:'wait',ms:Number(step.ms||250),url:page.url()}); continue;}
const locator=step.test_id?page.getByTestId(step.test_id):step.role?page.getByRole(step.role,{name:step.name}):page.locator(step.selector);
if(step.action==='click') await locator.click(); else if(step.action==='fill') await locator.fill(String(step.value||'')); else if(step.action==='press') await locator.press(String(step.key||'Enter')); else throw new Error('Unsupported browser action: '+step.action);
evidence.push({action:step.action,selector:step.selector||step.test_id||step.name||'',url:page.url()});}
await page.screenshot({path:input.screenshot,fullPage:true}); const result={url:page.url(),title:await page.title(),console_errors:consoleErrors,screenshot:input.screenshot,evidence};
await browser.close(); process.stdout.write(JSON.stringify(result));})().catch(error=>{console.error(error.stack||String(error));process.exit(1)});""",
            encoding="utf-8",
        )
        try:
            command = ["node", str(script_path.relative_to(self.root)), str(payload_path.relative_to(self.root))]
            node_modules = Path(__file__).resolve().parents[2] / "node_modules"
            command_result = self.sandbox.run(
                self.workspace_id,
                command,
                backend=self.backend,
                timeout_seconds=timeout_seconds,
                environment={"NODE_PATH": str(node_modules)} if self.backend == "local" and node_modules.is_dir() else {},
            )
            result = self._redact_result(command_result.to_dict())
            self.event("command.stdout", command=command, chunk=self._redact(result.get("stdout", "")[-20_000:]))
            if result.get("stderr"):
                self.event("command.stderr", command=command, chunk=self._redact(result.get("stderr", "")[-20_000:]))
            evidence: dict[str, Any] = {}
            if result.get("exit_code") == 0:
                try:
                    evidence = json.loads(result.get("stdout", "{}"))
                except json.JSONDecodeError:
                    evidence = {"parse_error": "browser probe returned invalid JSON"}
            return {**result, "browser": evidence, "artifact": screenshot_path.relative_to(self.root).as_posix() if screenshot_path.exists() else ""}
        finally:
            script_path.unlink(missing_ok=True)
            payload_path.unlink(missing_ok=True)

    def tool_browser_automate(
        self,
        url: str,
        actions: list[dict[str, Any]] | None = None,
        timeout_seconds: int = 120,
        approved: bool = False,
    ) -> dict[str, Any]:
        runtime = BrowserRuntime(self.root / ".openagent")
        result = runtime.run(url, actions=actions, approved=approved, timeout_seconds=timeout_seconds)
        artifact = str(result.get("artifact", ""))
        if artifact:
            try:
                result["artifact"] = Path(artifact).relative_to(self.root).as_posix()
            except ValueError:
                pass
        return result

    def tool_install_dependency(self, command: list[str] | str) -> dict[str, Any]:
        return self.tool_run_shell(command, timeout_seconds=900)

    def tool_git_status(self) -> dict[str, Any]:
        return self.tool_run_shell(["git", "status", "--short"])

    def tool_git_diff(self, staged: bool = False) -> dict[str, Any]:
        command = ["git", "diff", "--cached"] if staged else ["git", "diff"]
        return self.tool_run_shell(command)

    def tool_git_log(self, limit: int = 20) -> dict[str, Any]:
        return self.tool_run_shell(["git", "log", f"-{max(1, min(int(limit), 100))}", "--oneline", "--decorate"])

    def tool_git_commit(self, message: str) -> dict[str, Any]:
        self.tool_run_shell(["git", "add", "-A"])
        return self.tool_run_shell(["git", "commit", "-m", message])

    def tool_git_checkout(self, branch: str) -> dict[str, Any]:
        return self.tool_run_shell(["git", "checkout", self.sandbox.safe_branch(branch)])

    def tool_git_revert(self, revision: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Fa-f0-9]{7,40}", revision):
            raise ValueError("revision must be a Git commit hash")
        return self.tool_run_shell(["git", "revert", "--no-edit", revision])

    def tool_git_reset(self, revision: str = "HEAD") -> dict[str, Any]:
        if revision != "HEAD" and not re.fullmatch(r"[A-Fa-f0-9]{7,40}", revision):
            raise ValueError("revision must be HEAD or a Git commit hash")
        return self.tool_run_shell(["git", "reset", "--hard", revision])

    def _run_named(self, name: str, command: list[str] | str | None) -> dict[str, Any]:
        if command is None:
            command = self.repo.scan().get("commands", {}).get(name, "")
        if not command:
            return {"skipped": True, "reason": f"no {name} command detected", "exit_code": -1}
        return self.tool_run_shell(command, timeout_seconds=900)

    @staticmethod
    def _change_result(path: str, old: str, new: str) -> dict[str, Any]:
        patch = "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""))
        return {"path": path, "old_sha256": hashlib.sha256(old.encode()).hexdigest(), "new_sha256": hashlib.sha256(new.encode()).hexdigest(), "patch": patch[-100_000:]}

    @staticmethod
    def _redact(value: str) -> str:
        return SECRET_RE.sub(lambda match: f"{match.group(1)}=<redacted>", str(value))

    def _redact_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {key: self._redact(value) if key in {"stdout", "stderr"} else value for key, value in result.items()}

    def _safe_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self._redact(json.dumps(arguments, default=str)))

    @staticmethod
    def _bounded_result(result: dict[str, Any]) -> dict[str, Any]:
        return {key: (value[-20_000:] if isinstance(value, str) else value) for key, value in result.items()}

    @staticmethod
    def _is_sensitive_path(path: str) -> bool:
        name = Path(path).name.lower()
        if name == ".env" or (name.startswith(".env.") and name not in {".env.example", ".env.sample", ".env.template"}):
            return True
        if Path(name).suffix in SENSITIVE_SUFFIXES:
            return True
        return name in {"credentials", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}

    def _assert_not_sensitive(self, path: str) -> None:
        if self._is_sensitive_path(path):
            raise PermissionError("secret-bearing files are not exposed to codebase tools; use a vault-backed process binding")
