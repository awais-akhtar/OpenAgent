"""Project intent and delivery contracts for CodeQ runs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


STACK_HINTS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("next.js", "nextjs", "next js"), "nextjs", "typescript"),
    (("react native",), "react-native", "typescript"),
    (("react",), "react", "javascript"),
    (("vue", "vue.js", "vuejs"), "vue", "javascript"),
    (("svelte", "sveltekit"), "svelte", "typescript"),
    (("angular",), "angular", "typescript"),
    (("django",), "django", "python"),
    (("fastapi", "fast api"), "fastapi", "python"),
    (("flask",), "flask", "python"),
    (("python",), "python", "python"),
    (("express", "node.js", "nodejs", "node js"), "node", "javascript"),
    (("spring boot", "spring", "java"), "spring", "java"),
    ((".net", "dotnet", "asp.net", "c#"), "dotnet", "csharp"),
    (("flutter", "dart"), "flutter", "dart"),
    (("electron",), "electron", "javascript"),
    (("tauri",), "tauri", "rust"),
    (("rust", "cargo"), "rust", "rust"),
    (("golang", "go service", " go "), "go", "go"),
    (("kotlin", "android"), "android", "kotlin"),
    (("swiftui", "swift", "ios"), "ios", "swift"),
    (("unity",), "unity", "csharp"),
    (("unreal",), "unreal", "cpp"),
)

FUZZY_STACK_WORDS: dict[str, tuple[str, str]] = {
    "django": ("django", "python"),
    "fastapi": ("fastapi", "python"),
    "flask": ("flask", "python"),
    "react": ("react", "javascript"),
    "angular": ("angular", "typescript"),
    "svelte": ("svelte", "typescript"),
    "flutter": ("flutter", "dart"),
    "electron": ("electron", "javascript"),
}


def _fuzzy_stack(text: str) -> tuple[str, str] | None:
    """Recognize a clearly misspelled framework name without guessing prose."""

    tokens = re.findall(r"[a-z][a-z0-9.+#-]{3,}", text)
    for token in tokens:
        for canonical, stack in FUZZY_STACK_WORDS.items():
            if abs(len(token) - len(canonical)) <= 2 and _transposition_distance(token, canonical) <= 2:
                return stack
    return None


def _transposition_distance(left: str, right: str) -> int:
    """Return an optimal-string-alignment distance for short framework names."""

    rows = len(left) + 1
    columns = len(right) + 1
    matrix = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        matrix[row][0] = row
    for column in range(columns):
        matrix[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            cost = 0 if left[row - 1] == right[column - 1] else 1
            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1] + cost,
            )
            if row > 1 and column > 1 and left[row - 1] == right[column - 2] and left[row - 2] == right[column - 1]:
                matrix[row][column] = min(matrix[row][column], matrix[row - 2][column - 2] + 1)
    return matrix[-1][-1]


def classify_request(goal: str, project_map: dict[str, Any]) -> dict[str, Any]:
    """Describe the requested product without coupling the controller to one UI stack."""

    text = f" {str(goal).lower()} "
    existing_framework = str(project_map.get("framework", "")).strip().lower()
    existing_languages = [str(item).lower() for item in project_map.get("languages", [])]
    framework = existing_framework
    language = existing_languages[0] if existing_languages else ""
    explicit_stack = False
    for needles, candidate_framework, candidate_language in STACK_HINTS:
        if any(needle in text for needle in needles):
            framework = candidate_framework
            language = candidate_language
            explicit_stack = True
            break
    if not explicit_stack:
        fuzzy = _fuzzy_stack(text)
        if fuzzy:
            framework, language = fuzzy
            explicit_stack = True

    kind = "application"
    categories = (
        (("mobile app", "android", "ios", "react native", "flutter"), "mobile application"),
        (("desktop app", "desktop application", "electron", "tauri"), "desktop application"),
        (("api", "backend", "microservice", "web service"), "service"),
        (("cli", "command line", "terminal tool"), "command-line application"),
        (("library", "package", "sdk"), "library"),
        (("game", "simulation"), "interactive application"),
        (("dashboard", "website", "web app", "frontend", "page", "portal"), "web application"),
        (("data pipeline", "etl", "machine learning", "ml model"), "data application"),
    )
    for needles, candidate in categories:
        if any(needle in text for needle in needles):
            kind = candidate
            break
    if framework == "django" and any(term in text for term in ("frontend", "page", "website", "web app", "bootstrap")):
        kind = "web application"

    greenfield = int(project_map.get("file_count", 0) or 0) == 0
    backend_terms = ("database", "postgres", "mysql", "mongodb", "backend", "api", "oauth", "google login", "stripe", "webhook", "authentication system")
    architecture = "full-stack" if kind == "web application" and any(term in text for term in backend_terms) else "single-runtime"
    if greenfield and not framework:
        if kind in {"application", "web application"}:
            kind = "web application"
            framework, language = "react", "javascript"
        elif kind in {"service", "command-line application", "library", "data application", "desktop application"}:
            framework, language = "python", "python"
        elif kind == "interactive application":
            framework, language = "browser", "javascript"

    return {
        "mode": "greenfield" if greenfield else "existing",
        "kind": kind,
        "framework": framework or "repository-native",
        "language": language or "repository-native",
        "explicit_stack": explicit_stack,
        "architecture": architecture,
        "requires_ui": kind in {"web application", "mobile application", "desktop application", "interactive application"},
        "requires_runtime": kind != "library",
        "requires_tests": True,
    }


def artifact_contract(profile: dict[str, Any]) -> dict[str, Any]:
    """Return minimum structural evidence expected from a new project."""

    framework = str(profile.get("framework", ""))
    contracts: dict[str, dict[str, Any]] = {
        "react": {"all": ["package.json", "index.html"], "any": ["src/App.jsx", "src/App.tsx"], "minimum_files": 4},
        "nextjs": {"all": ["package.json"], "any": ["app/page.tsx", "app/page.jsx", "pages/index.tsx", "pages/index.jsx"], "minimum_files": 4},
        "vue": {"all": ["package.json"], "any": ["src/App.vue"], "minimum_files": 4},
        "svelte": {"all": ["package.json"], "any": ["src/routes/+page.svelte", "src/App.svelte"], "minimum_files": 4},
        "angular": {"all": ["package.json", "angular.json"], "any": ["src/app/app.component.ts"], "minimum_files": 5},
        "python": {"any_suffix": [".py"], "minimum_files": 2},
        "fastapi": {"any": ["main.py", "app/main.py", "src/main.py"], "minimum_files": 3},
        "flask": {"any": ["app.py", "app/__init__.py", "src/app.py"], "minimum_files": 3},
        "django": {"all": ["manage.py"], "any_suffix": [".py"], "minimum_files": 4},
        "node": {"all": ["package.json"], "any_suffix": [".js", ".ts"], "minimum_files": 3},
        "go": {"all": ["go.mod"], "any": ["main.go", "cmd/main.go"], "minimum_files": 2},
        "rust": {"all": ["Cargo.toml"], "any": ["src/main.rs", "src/lib.rs"], "minimum_files": 2},
        "tauri": {"all": ["package.json", "src-tauri/Cargo.toml"], "minimum_files": 5},
        "dotnet": {"any_suffix": [".csproj"], "minimum_files": 3},
        "spring": {"any": ["pom.xml", "build.gradle", "build.gradle.kts"], "any_suffix": [".java", ".kt"], "minimum_files": 4},
        "flutter": {"all": ["pubspec.yaml"], "any": ["lib/main.dart"], "minimum_files": 3},
        "react-native": {"all": ["package.json"], "any": ["App.tsx", "App.jsx", "app/index.tsx"], "minimum_files": 4},
        "electron": {"all": ["package.json"], "any": ["main.js", "src/main.js", "src/main.ts"], "minimum_files": 4},
        "browser": {"all": ["index.html"], "any_suffix": [".js"], "minimum_files": 3},
    }
    return contracts.get(framework, {"minimum_files": 2})


def check_artifact_contract(root: Path, profile: dict[str, Any]) -> list[str]:
    contract = artifact_contract(profile)
    files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not any(part in {".git", "node_modules", ".venv", "dist", "build"} for part in path.relative_to(root).parts)
    ]
    lowered = {path.lower() for path in files}
    missing: list[str] = []
    for path in contract.get("all", []):
        if path.lower() not in lowered:
            missing.append(path)
    candidates = [str(path).lower() for path in contract.get("any", [])]
    if candidates and not any(path in lowered for path in candidates):
        missing.append("one of: " + ", ".join(contract["any"]))
    suffixes = tuple(str(item).lower() for item in contract.get("any_suffix", []))
    if suffixes and not any(path.endswith(suffixes) for path in lowered):
        missing.append("source file with suffix: " + ", ".join(suffixes))
    minimum = int(contract.get("minimum_files", 1))
    if len(files) < minimum:
        missing.append(f"at least {minimum} project files (found {len(files)})")
    if profile.get("architecture") == "full-stack":
        backend_suffixes = (".py", ".js", ".mjs", ".cjs", ".ts", ".go", ".rs", ".java", ".cs")
        has_backend = any(
            path.lower().endswith(backend_suffixes)
            and any(part in {"api", "backend", "server", "services"} for part in Path(path).parts[:-1])
            for path in files
        )
        integrated_backend = str(profile.get("framework", "")) == "django" and "manage.py" in lowered
        if not has_backend and not integrated_backend:
            missing.append("a server implementation under api/, backend/, server/, or services/")
    return missing


def task_needs_file_change(title: str, agent: str) -> bool:
    text = str(title).lower()
    if str(agent).lower() not in {"programmer", "debugger", "tester"}:
        return False
    return bool(re.search(r"\b(create|implement|build|add|fix|repair|refactor|integrate|scaffold|style|configure)\b", text))
