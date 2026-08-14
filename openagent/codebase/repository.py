"""Bounded repository discovery and context selection."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any, Iterable


IGNORED_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "dist", "build",
    "coverage", ".next", ".cache", "__pycache__", ".pytest_cache", ".mypy_cache",
}
TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".rb", ".swift", ".sql", ".html",
    ".css", ".scss", ".sass", ".vue", ".svelte", ".md", ".toml", ".yaml", ".yml",
    ".json", ".xml", ".sh", ".ps1", ".bat", ".cmd", ".ini", ".cfg", ".txt", ".prisma",
    ".sol", ".dart", ".ex", ".exs", ".gd", ".zig", ".scala", ".lua",
}
LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".kt": "kotlin",
    ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".sql": "sql",
    ".sol": "solidity", ".dart": "dart", ".ex": "elixir", ".exs": "elixir",
    ".gd": "gdscript", ".zig": "zig", ".scala": "scala", ".lua": "lua",
}
SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|interface|type|enum|struct|trait)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
IMPORT_RE = re.compile(
    r"(?:from\s+([\w.]+)\s+import|import\s+([\w.@/\-]+)|require\([\"']([^\"']+)|from\s+[\"']([^\"']+))"
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


class RepositoryBoundaryError(ValueError):
    pass


class RepositoryIntelligence:
    """Build and query a compact, persisted project map."""

    def __init__(self, root: str | Path, *, max_files: int = 50_000, max_file_bytes: int = 1_000_000) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_files = max(100, min(int(max_files), 100_000))
        self.max_file_bytes = max(8_192, min(int(max_file_bytes), 5_000_000))
        if not self.root.is_dir():
            raise RepositoryBoundaryError("repository root must be an existing directory")

    def resolve(self, relative_path: str, *, must_exist: bool = False) -> Path:
        value = str(relative_path).replace("\\", "/").strip().lstrip("/")
        candidate = (self.root / value).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise RepositoryBoundaryError("path must remain inside the repository workspace")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(value)
        return candidate

    def files(self) -> list[Path]:
        found: list[Path] = []
        for path in self.root.rglob("*"):
            if any(part in IGNORED_NAMES for part in path.relative_to(self.root).parts):
                continue
            if path.is_symlink():
                continue
            if path.is_file():
                found.append(path)
                if len(found) >= self.max_files:
                    break
        return found

    def scan(self) -> dict[str, Any]:
        files = self.files()
        languages: Counter[str] = Counter()
        symbols: list[dict[str, Any]] = []
        imports: dict[str, list[str]] = defaultdict(list)
        routes: list[dict[str, str]] = []
        database_models: list[dict[str, Any]] = []
        call_graph: list[dict[str, Any]] = []
        important: list[str] = []
        total_bytes = 0
        for path in files:
            rel = path.relative_to(self.root).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                continue
            total_bytes += size
            language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
            if language:
                languages[language] += 1
            if path.name.lower() in {
                "package.json", "pyproject.toml", "requirements.txt", "cargo.toml", "go.mod",
                "pom.xml", "build.gradle", "docker-compose.yml", "docker-compose.yaml", "dockerfile",
                ".env.example", "readme.md", "manage.py", "vite.config.ts", "next.config.js",
                "composer.json", "gemfile", "mix.exs", "pubspec.yaml", "cmakelists.txt",
                "build.zig", "foundry.toml", "project.godot", "settings.gradle", "settings.gradle.kts",
                "hardhat.config.js", "hardhat.config.ts", "artisan", "package.swift",
            }:
                important.append(rel)
            if size > self.max_file_bytes or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = self._read_text(path)
            if text is None:
                continue
            for name, kind, line in self._symbols(path, text):
                symbols.append({"name": name, "kind": kind, "path": rel, "line": line})
            imports[rel] = sorted({part for match in IMPORT_RE.findall(text) for part in match if part})[:200]
            routes.extend(self._routes(rel, text))
            database_models.extend(self._database_models(rel, text))
            call_graph.extend(self._calls(path, rel, text))
        package = self._json_file("package.json")
        framework, database = self._detect_stack(files, package)
        commands = self._commands(package, files)
        return {
            "root": str(self.root),
            "file_count": len(files),
            "total_bytes": total_bytes,
            "languages": [name for name, _ in languages.most_common()],
            "language_counts": dict(languages),
            "framework": framework,
            "database": database,
            "entrypoints": self._entrypoints(files),
            "important_files": sorted(set(important))[:100],
            "commands": commands,
            "dependencies": self._dependencies(package),
            "configuration_files": self._configuration_files(files),
            "test_files": self._test_files(files),
            "symbols": symbols[:20_000],
            "imports": dict(list(imports.items())[:10_000]),
            "dependency_graph": [{"file": path, "imports": values} for path, values in list(imports.items())[:10_000]],
            "call_graph": call_graph[:20_000],
            "database_models": database_models[:5_000],
            "routes": routes[:5_000],
            "git": self._git_state(),
            "truncated": len(files) >= self.max_files,
        }

    def select_context(self, query: str, *, limit: int = 20, max_chars: int = 100_000) -> list[dict[str, Any]]:
        terms = {word.lower() for word in TOKEN_RE.findall(query)}
        scored: list[tuple[float, Path, str]] = []
        for path in self.files():
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            rel = path.relative_to(self.root).as_posix()
            text = self._read_text(path)
            if text is None:
                continue
            name_words = {word.lower() for word in TOKEN_RE.findall(rel)}
            content_words = Counter(word.lower() for word in TOKEN_RE.findall(text[:250_000]))
            lowered_text = text[:250_000].lower()
            score = (
                sum(8 for term in terms if term in name_words)
                + sum(min(content_words[term], 12) * 2 for term in terms)
                + sum(2 for term in terms if term in lowered_text)
            )
            if path.name.lower() in {"package.json", "pyproject.toml", "readme.md"}:
                score += 2
            if score:
                scored.append((float(score), path, text))
        selected: list[dict[str, Any]] = []
        used = 0
        for score, path, text in sorted(scored, key=lambda item: (-item[0], item[1].as_posix()))[: max(1, limit)]:
            remaining = max_chars - used
            if remaining <= 0:
                break
            excerpt = self._relevant_excerpt(text, terms, min(12_000, remaining))
            used += len(excerpt)
            selected.append({"path": path.relative_to(self.root).as_posix(), "score": score, "content": excerpt})
        return selected

    def search(self, query: str, *, limit: int = 100) -> list[dict[str, Any]]:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        results: list[dict[str, Any]] = []
        for path in self.files():
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = self._read_text(path)
            if text is None:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    results.append({"path": path.relative_to(self.root).as_posix(), "line": line_no, "text": line[:500]})
                    if len(results) >= max(1, min(limit, 500)):
                        return results
        return results

    def _read_text(self, path: Path) -> str | None:
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        if b"\x00" in raw[:8192]:
            return None
        return raw.decode("utf-8", errors="replace")

    def _symbols(self, path: Path, text: str) -> Iterable[tuple[str, str, int]]:
        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield node.name, node.__class__.__name__.lower(), int(node.lineno)
                return
            except SyntaxError:
                pass
        for match in SYMBOL_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            prefix = text[match.start():match.end()].strip().split()[0]
            yield match.group(1), prefix, line

    def _routes(self, rel: str, text: str) -> list[dict[str, str]]:
        patterns = [
            re.compile(r"(?:@\w+\.(get|post|put|patch|delete)|\b(app|router)\.(get|post|put|patch|delete))\s*\(\s*[\"']([^\"']+)", re.I),
            re.compile(r"\(\s*[\"'](GET|POST|PUT|PATCH|DELETE)[\"']\s*,\s*[\"']([^\"']+)", re.I),
        ]
        output: list[dict[str, str]] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                groups = [group for group in match.groups() if group]
                method = next((item.upper() for item in groups if item.lower() in {"get", "post", "put", "patch", "delete"}), "ANY")
                route = groups[-1]
                output.append({"method": method, "path": route, "file": rel})
        return output

    def _calls(self, path: Path, rel: str, text: str) -> list[dict[str, Any]]:
        if path.suffix.lower() != ".py":
            return []
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        output: list[dict[str, Any]] = []
        for owner in ast.walk(tree):
            if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls: set[str] = set()
            for node in ast.walk(owner):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
            if calls:
                output.append({"file": rel, "symbol": owner.name, "line": int(owner.lineno), "calls": sorted(calls)[:200]})
        return output

    @staticmethod
    def _database_models(rel: str, text: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for match in re.finditer(r"(?m)^\s*model\s+([A-Za-z_]\w*)\s*\{", text):
            output.append({"name": match.group(1), "kind": "prisma", "file": rel, "line": text.count("\n", 0, match.start()) + 1})
        for match in re.finditer(r"(?m)^\s*class\s+([A-Za-z_]\w*)\s*\([^\n]*(?:Model|Base)[^\n]*\)\s*:", text):
            output.append({"name": match.group(1), "kind": "orm", "file": rel, "line": text.count("\n", 0, match.start()) + 1})
        return output

    def _dependencies(self, package: dict[str, Any]) -> dict[str, list[str]]:
        dependencies: dict[str, list[str]] = {}
        if package:
            dependencies["npm"] = sorted({*package.get("dependencies", {}), *package.get("devDependencies", {})})[:5_000]
        requirements = self.root / "requirements.txt"
        if requirements.is_file():
            values = []
            for line in requirements.read_text(encoding="utf-8", errors="replace").splitlines():
                value = re.split(r"[<>=!~;\[]", line.strip(), maxsplit=1)[0]
                if value and not value.startswith("#"):
                    values.append(value)
            dependencies["python"] = sorted(set(values))[:5_000]
        pyproject = self.root / "pyproject.toml"
        if pyproject.is_file():
            try:
                parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                values = parsed.get("project", {}).get("dependencies", [])
                dependencies.setdefault("python", []).extend(
                    re.split(r"[<>=!~;\[]", str(value), maxsplit=1)[0] for value in values
                )
                dependencies["python"] = sorted(set(filter(None, dependencies["python"])))[:5_000]
            except (OSError, tomllib.TOMLDecodeError):
                pass
        return dependencies

    def _configuration_files(self, files: list[Path]) -> list[str]:
        markers = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
        return sorted(
            path.relative_to(self.root).as_posix() for path in files
            if path.suffix.lower() in markers or path.name.lower() in {"dockerfile", "makefile", ".env.example"}
        )[:1_000]

    def _test_files(self, files: list[Path]) -> list[str]:
        output = []
        for path in files:
            rel = path.relative_to(self.root).as_posix()
            name = path.name.lower()
            if "test" in path.parts or "tests" in path.parts or name.startswith("test_") or any(token in name for token in (".test.", ".spec.")):
                output.append(rel)
        return sorted(output)[:5_000]

    def _git_state(self) -> dict[str, Any]:
        if not (self.root / ".git").exists():
            return {"repository": False, "branch": "", "commit": "", "dirty": False, "changes": []}
        def run(*arguments: str) -> str:
            try:
                result = subprocess.run(
                    ["git", *arguments], cwd=str(self.root), capture_output=True, text=True,
                    errors="replace", timeout=10, check=False, shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError):
                return ""
            return result.stdout.strip() if result.returncode == 0 else ""
        changes = run("status", "--short").splitlines()
        return {
            "repository": True,
            "branch": run("branch", "--show-current"),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(changes),
            "changes": changes[:500],
        }

    def _detect_stack(self, files: list[Path], package: dict[str, Any]) -> tuple[str, str]:
        names = {path.relative_to(self.root).as_posix().lower() for path in files}
        deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})} if package else {}
        framework = ""
        for dependency, label in (("next", "nextjs"), ("@angular/core", "angular"), ("vue", "vue"), ("svelte", "svelte"), ("react", "react")):
            if dependency in deps:
                framework = label
                break
        if not framework and "manage.py" in names:
            framework = "django"
        manifest_stacks = (
            (("foundry.toml", "hardhat.config.js", "hardhat.config.ts"), "solidity"),
            (("config/application.rb",), "rails"),
            (("artisan",), "laravel"),
            (("mix.exs",), "phoenix"),
            (("pubspec.yaml",), "flutter"),
            (("project.godot",), "godot"),
            (("cmakelists.txt",), "cmake"),
            (("build.zig",), "zig"),
            (("cargo.toml",), "rust"),
            (("go.mod",), "go"),
        )
        if not framework:
            framework = next(
                (label for markers, label in manifest_stacks if any(marker in names for marker in markers)),
                "",
            )
        if not framework and any(name.endswith((".csproj", ".sln")) for name in names):
            framework = "dotnet"
        if not framework and any(name.endswith(".uproject") for name in names):
            framework = "unreal"
        if not framework and "projectsettings/projectversion.txt" in names:
            framework = "unity"
        if not framework and any(name.endswith(".xcodeproj/project.pbxproj") for name in names):
            framework = "ios"
        if not framework and any(name in names for name in ("settings.gradle", "settings.gradle.kts")):
            framework = "android" if any("androidmanifest.xml" in name for name in names) else "spring"
        if not framework and any(name.endswith("app.py") for name in names):
            framework = "python"
        corpus = " ".join(names) + " " + json.dumps(deps)
        database = next((label for needle, label in (("prisma", "postgresql/prisma"), ("psycopg", "postgresql"), ("postgres", "postgresql"), ("mysql", "mysql"), ("mongodb", "mongodb"), ("sqlite", "sqlite")) if needle in corpus), "")
        return framework, database

    def _commands(self, package: dict[str, Any], files: list[Path]) -> dict[str, str]:
        scripts = package.get("scripts", {}) if package else {}
        commands: dict[str, str] = {}
        for name in ("install", "dev", "test", "build", "lint", "typecheck"):
            if name == "install" and package:
                commands[name] = "npm install"
            elif name in scripts:
                commands[name] = f"npm run {name}"
        for candidate in ("test:e2e", "e2e", "test:browser"):
            if candidate in scripts:
                commands["browser"] = f"npm run {candidate}"
                break
        names = {path.name.lower() for path in files}
        python_dependencies = set(self._dependencies(package).get("python", []))
        pyproject_text = ""
        pyproject = self.root / "pyproject.toml"
        if pyproject.is_file():
            pyproject_text = pyproject.read_text(encoding="utf-8", errors="replace").lower()
        if "pytest" in python_dependencies or "pytest.ini" in names or "conftest.py" in names or "[tool.pytest" in pyproject_text:
            commands.setdefault("test", "python -m pytest")
        elif any(path.name.lower().startswith("test_") and path.suffix.lower() == ".py" for path in files):
            commands.setdefault("test", "python -m unittest discover -v")
        if "build" in python_dependencies or "[build-system]" in pyproject_text:
            commands.setdefault("build", "python -m build")
        if "ruff" in python_dependencies or "[tool.ruff" in pyproject_text or "ruff.toml" in names or ".ruff.toml" in names:
            commands.setdefault("lint", "python -m ruff check .")
        if "mypy" in python_dependencies or "[tool.mypy" in pyproject_text or "mypy.ini" in names:
            commands.setdefault("typecheck", "python -m mypy .")
        if "cargo.toml" in names:
            commands.update({"test": "cargo test", "build": "cargo build", "lint": "cargo clippy"})
        if "go.mod" in names:
            commands.update({"test": "go test ./...", "build": "go build ./..."})
        if "build.zig" in names:
            commands.update({"test": "zig build test", "build": "zig build"})
        if "foundry.toml" in names:
            commands.update({"test": "forge test", "build": "forge build"})
        if "mix.exs" in names:
            commands.update({"test": "mix test", "build": "mix compile --warnings-as-errors"})
        if "artisan" in names:
            commands.setdefault("test", "php artisan test")
        if "gemfile" in names and "config/application.rb" in {path.relative_to(self.root).as_posix().lower() for path in files}:
            commands.setdefault("test", "bundle exec rails test")
        if "pubspec.yaml" in names:
            commands.update({"test": "flutter test", "build": "flutter build bundle"})
        if "project.godot" in names:
            commands.setdefault("runtime", "godot --headless --path . --quit")
        return commands

    def _entrypoints(self, files: list[Path]) -> list[str]:
        preferred = {
            "main.py", "app.py", "manage.py", "server.py", "index.js", "index.ts", "src/index.ts",
            "src/main.ts", "src/main.tsx", "src/app/page.tsx", "src/app.tsx", "cmd/main.go",
            "lib/main.dart", "src/main.rs", "src/main.zig", "package.swift", "project.godot",
        }
        rels = [path.relative_to(self.root).as_posix() for path in files]
        return [rel for rel in rels if rel.lower() in preferred or Path(rel).name.lower() in preferred][:50]

    def _json_file(self, relative: str) -> dict[str, Any]:
        path = self.root / relative
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _relevant_excerpt(text: str, terms: set[str], limit: int) -> str:
        lines = text.splitlines()
        hits = [index for index, line in enumerate(lines) if any(term in line.lower() for term in terms)]
        if not hits:
            return text[:limit]
        indexes: set[int] = set()
        for hit in hits[:40]:
            indexes.update(range(max(0, hit - 4), min(len(lines), hit + 7)))
        excerpt = "\n".join(f"{index + 1}: {lines[index]}" for index in sorted(indexes))
        return excerpt[:limit]
