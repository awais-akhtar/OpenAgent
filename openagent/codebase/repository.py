"""Bounded repository discovery and context selection."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
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
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".java", ".kt", ".kts", ".go", ".rs",
    ".c", ".h", ".cc", ".hh", ".cpp", ".hpp", ".cxx", ".hxx", ".cs", ".php", ".rb", ".swift", ".sql", ".html",
    ".css", ".scss", ".sass", ".vue", ".svelte", ".md", ".toml", ".yaml", ".yml",
    ".json", ".xml", ".sh", ".ps1", ".bat", ".cmd", ".ini", ".cfg", ".txt", ".prisma",
    ".sol", ".dart", ".ex", ".exs", ".gd", ".zig", ".scala", ".lua", ".kts",
}
LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".mts": "typescript", ".cts": "typescript", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
    ".cxx": "cpp", ".hh": "cpp", ".hpp": "cpp", ".hxx": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".sql": "sql", ".vue": "vue", ".svelte": "svelte",
    ".scss": "scss", ".sass": "sass", ".html": "html", ".css": "css", ".json": "json", ".xml": "xml",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".sh": "bash", ".ps1": "powershell",
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
SYMBOL_RETRIEVAL_STOPWORDS = frozenset({
    # Natural-language task vocabulary that occurs in many code identifiers.
    # These terms still participate in lexical retrieval, but alone they are
    # too broad to nominate a declaration as a graph-backed symbol hit.
    "build", "call", "calls", "class", "code", "current", "data", "file", "files",
    "fresh", "freshness", "function", "functions", "hash", "hashes", "line", "lines",
    "index", "indexed", "indexing", "method", "methods", "module", "modules",
    "parser", "path", "paths", "project",
    "range", "read", "repository", "request", "result", "results", "safe", "source",
    "sources", "test", "tests", "use", "used", "using", "valid", "validate",
    "validation", "write",
})


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

    def scan(self, *, previous: dict[str, Any] | None = None, force_reindex: bool = False) -> dict[str, Any]:
        """Build a fresh file inventory and incrementally re-index changed files.

        Directory enumeration and content hashing still run on every call. Parser
        work and per-file relationship extraction are reused only when the source
        hash, scanner schema and locally installed grammar set are unchanged.
        """
        from .scip_index import read_scip_index
        from .tree_sitter_index import TreeSitterIndexer, language_for_suffix

        parser = TreeSitterIndexer()
        schema_version = 2
        prior_meta = (previous or {}).get("index_metadata", {}) if isinstance(previous, dict) else {}
        cache_compatible = (
            not force_reindex
            and isinstance(previous, dict)
            and int(prior_meta.get("scanner_schema", 0) or 0) == schema_version
            and str(prior_meta.get("parser_cache_key", "")) == parser.cache_key
        )
        prior_cache = (previous or {}).get("file_index_cache", {}) if cache_compatible else {}
        files = self.files()
        languages: Counter[str] = Counter()
        symbols: list[dict[str, Any]] = []
        imports: dict[str, list[str]] = defaultdict(list)
        routes: list[dict[str, str]] = []
        database_models: list[dict[str, Any]] = []
        call_graph: list[dict[str, Any]] = []
        important: list[str] = []
        file_inventory: list[dict[str, Any]] = []
        file_index_cache: dict[str, dict[str, Any]] = {}
        tree_sitter_files: dict[str, dict[str, Any]] = {}
        reused_files = 0
        parsed_files = 0
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
            indexable = size <= self.max_file_bytes and path.suffix.lower() in TEXT_SUFFIXES
            content_sha256 = ""
            text = None
            if indexable:
                try:
                    raw_content = path.read_bytes()
                    content_sha256 = hashlib.sha256(raw_content).hexdigest()
                    if b"\x00" not in raw_content[:8192]:
                        text = raw_content.decode("utf-8", errors="replace")
                except OSError:
                    indexable = False
            if text is None:
                file_inventory.append({
                    "path": rel,
                    "size_bytes": size,
                    "language": language or "",
                    "content_sha256": content_sha256,
                    "index_status": "metadata-only",
                })
                continue

            cached = prior_cache.get(rel) if isinstance(prior_cache, dict) else None
            if isinstance(cached, dict) and cached.get("content_sha256") == content_sha256:
                cached_tree = cached.get("tree_sitter", {}) if isinstance(cached.get("tree_sitter"), dict) else {}
                file_symbols = [dict(item) for item in cached.get("symbols", []) if isinstance(item, dict)]
                file_imports = [str(item) for item in cached.get("imports", []) if str(item)]
                file_routes = [dict(item) for item in cached.get("routes", []) if isinstance(item, dict)]
                file_models = [dict(item) for item in cached.get("database_models", []) if isinstance(item, dict)]
                file_calls = [dict(item) for item in cached.get("call_graph", []) if isinstance(item, dict)]
                reused_files += 1
                tree_sitter_files[rel] = cached_tree
            else:
                parsed_files += 1
                parser_language = language_for_suffix(path.suffix.lower())
                parsed = parser.parse(text, parser_language, max_source_bytes=self.max_file_bytes) if parser_language else None
                if parsed and parsed.get("status") == "parsed":
                    # Parser output is preferred. Retain the legacy scanner's
                    # independently extracted declarations only where the
                    # grammar does not expose a declaration, avoiding omissions
                    # in language-specific constructs.
                    parser_symbols = [
                        {"name": str(item["name"]), "kind": str(item.get("kind", "symbol")), "path": rel,
                         "line": int(item.get("line", 1)), "content_sha256": content_sha256,
                         "provenance": f"tree-sitter:{parser_language}@{parser.version}"}
                        for item in parsed.get("symbols", []) if item.get("name")
                    ]
                    fallback_symbols = [
                        {"name": name, "kind": kind, "path": rel, "line": line,
                         "content_sha256": content_sha256, "provenance": self._symbol_provenance(path, text)}
                        for name, kind, line in self._symbols(path, text)
                    ]
                    parser_keys = {(item["name"], item["line"]) for item in parser_symbols}
                    file_symbols = parser_symbols + [item for item in fallback_symbols if (item["name"], item["line"]) not in parser_keys]
                    file_imports = sorted(set(str(item) for item in parsed.get("imports", []) if str(item)))
                    file_imports = sorted(set(file_imports) | {part for match in IMPORT_RE.findall(text) for part in match if part})
                    file_calls = [
                        {"file": rel, **dict(item)}
                        for item in parsed.get("calls", []) if isinstance(item, dict)
                    ]
                    if not file_calls and path.suffix.lower() == ".py":
                        file_calls = self._calls(path, rel, text)
                    tree_sitter_files[rel] = {
                        "language": parser_language,
                        "parser": "tree-sitter-language-pack",
                        "parser_version": parser.version,
                        "status": parsed.get("status", "parsed"),
                        "diagnostic_count": len(parsed.get("diagnostics", [])),
                        "used_for_relationships": True,
                    }
                else:
                    file_symbols = [
                        {"name": name, "kind": kind, "path": rel, "line": line,
                         "content_sha256": content_sha256, "provenance": self._symbol_provenance(path, text)}
                        for name, kind, line in self._symbols(path, text)
                    ]
                    file_imports = sorted({part for match in IMPORT_RE.findall(text) for part in match if part})
                    file_calls = self._calls(path, rel, text) if path.suffix.lower() == ".py" else []
                    tree_sitter_files[rel] = parser.status(parser_language)
                    if parsed:
                        tree_sitter_files[rel].update({
                            "status": parsed.get("status", "unavailable"),
                            "diagnostic_count": len(parsed.get("diagnostics", [])),
                            "used_for_relationships": False,
                        })
                file_routes = self._routes(rel, text, content_sha256=content_sha256)
                file_models = self._database_models(rel, text)
                file_index_cache[rel] = {
                    "content_sha256": content_sha256,
                    "symbols": file_symbols,
                    "imports": file_imports,
                    "routes": file_routes,
                    "database_models": file_models,
                    "call_graph": file_calls,
                    "tree_sitter": tree_sitter_files[rel],
                }
            if rel not in file_index_cache:
                file_index_cache[rel] = {
                    "content_sha256": content_sha256,
                    "symbols": file_symbols,
                    "imports": file_imports,
                    "routes": file_routes,
                    "database_models": file_models,
                    "call_graph": file_calls,
                    "tree_sitter": tree_sitter_files[rel],
                }
            file_inventory.append({
                "path": rel,
                "size_bytes": size,
                "language": language or "",
                "content_sha256": content_sha256,
                "index_status": "indexed",
            })
            symbols.extend(file_symbols)
            imports[rel] = file_imports[:200]
            routes.extend(file_routes)
            database_models.extend(file_models)
            call_graph.extend(file_calls)
        package = self._json_file("package.json")
        framework, database = self._detect_stack(files, package)
        commands = self._commands(package, files)
        scip = read_scip_index(self.root)
        symbols.extend(scip.get("symbols", []))
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
            "file_inventory": file_inventory,
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
            "scip_symbols": scip.get("symbols", []),
            "scip_occurrences": scip.get("occurrences", []),
            "scip_index": {
                key: value for key, value in scip.items()
                if key not in {"symbols", "occurrences", "file_hashes", "indexers_on_path"}
            },
            "tree_sitter_files": tree_sitter_files,
            "file_index_cache": file_index_cache,
            "index_metadata": {
                "scanner_schema": schema_version,
                "parser_cache_key": parser.cache_key,
                "tree_sitter_version": parser.version,
                "installed_grammars": list(parser.loaded_languages),
                "files_seen": len(files),
                "files_reused": reused_files,
                "files_parsed": parsed_files,
                "incremental": cache_compatible,
            },
            "git": self._git_state(),
            "truncated": len(files) >= self.max_files,
        }

    def select_context(self, query: str, *, limit: int = 20, max_chars: int = 100_000) -> list[dict[str, Any]]:
        terms = {word.lower() for word in TOKEN_RE.findall(query)}
        scored: list[tuple[float, Path, str, str]] = []
        for path in self.files():
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            rel = path.relative_to(self.root).as_posix()
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue
            text = raw.decode("utf-8", errors="replace")
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
                scored.append((float(score), path, text, hashlib.sha256(raw).hexdigest()))
        selected: list[dict[str, Any]] = []
        used = 0
        for score, path, text, content_hash in sorted(scored, key=lambda item: (-item[0], item[1].as_posix()))[: max(1, limit)]:
            remaining = max_chars - used
            if remaining <= 0:
                break
            excerpt = self._relevant_excerpt(text, terms, min(12_000, remaining))
            used += len(excerpt)
            selected.append({
                "path": path.relative_to(self.root).as_posix(),
                "score": score,
                "content": excerpt,
                "content_sha256": content_hash,
                **self._excerpt_line_range(excerpt),
                "retrieval_source": "lexical",
            })
        return selected

    def select_task_context(
        self,
        query: str,
        graph: dict[str, Any],
        *,
        seed_paths: list[str] | tuple[str, ...] = (),
        limit: int = 16,
        max_chars: int = 48_000,
        max_file_chars: int = 6_000,
        graph_limit: int = 8,
        fts_candidates_input: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve current, bounded context from task files, lexical hits and graph edges.

        Explicit task files are included even when their names/content do not overlap
        the goal wording. Graph neighbors are included only when the current file hash
        still matches the graph revision. Rows are bounded before prompt serialization.
        """
        from .work_graph import related_code_paths, safe_relative_path

        budget = max(2, min(int(max_chars), 100_000))
        per_file = max(256, min(int(max_file_chars), budget))
        query_terms = {word.lower() for word in TOKEN_RE.findall(query)}
        graph_nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
        graph_file_hashes: dict[str, str] = {}
        for node in graph_nodes:
            if node.get("kind") not in {"file", "test"} or not node.get("path"):
                continue
            path = str(node.get("path", "")).replace("\\", "/")
            digest = str(node.get("content_sha256", ""))
            # A test node is a semantic alias of its file and may not carry a
            # hash; never let that empty alias shadow the file node's hash.
            if digest or path not in graph_file_hashes:
                graph_file_hashes[path] = digest
        rows: list[dict[str, Any]] = []
        selected_paths: set[str] = set()
        serialized_chars = 2  # JSON array brackets
        truncated = False
        stale_graph_paths: list[str] = []
        stale_fts_paths: list[str] = []

        def append_path(
            path_text: str,
            *,
            source: str,
            score: float = 0.0,
            relation: dict[str, Any] | None = None,
            expected_hash: str = "",
            stale_bucket: str = "fts",
        ) -> bool:
            nonlocal serialized_chars, truncated
            relative = safe_relative_path(path_text)
            if not relative or relative in selected_paths:
                return False
            try:
                path = self.resolve(relative, must_exist=True)
                raw = path.read_bytes()
            except (OSError, ValueError):
                return False
            if b"\x00" in raw[:8192]:
                return False
            text = raw.decode("utf-8", errors="replace")
            content = self._relevant_excerpt(text, query_terms, per_file)
            content_hash = hashlib.sha256(raw).hexdigest()
            if expected_hash and content_hash != expected_hash:
                (stale_graph_paths if stale_bucket == "graph" else stale_fts_paths).append(relative)
                return False
            row: dict[str, Any] = {
                "path": relative,
                "score": score,
                "content": content[:per_file],
                "content_sha256": content_hash,
                **self._excerpt_line_range(content[:per_file]),
                "retrieval_source": source,
            }
            if relation:
                relation_row = {
                    "graph_relation": relation.get("relation", ""),
                    "graph_related_to": relation.get("related_to", ""),
                    "graph_provenance": relation.get("provenance", ""),
                    "graph_confidence": relation.get("confidence", ""),
                }
                row.update(relation_row)
                row["graph_relations"] = [{
                    "relation": relation_row["graph_relation"],
                    "related_to": relation_row["graph_related_to"],
                    "provenance": relation_row["graph_provenance"],
                    "confidence": relation_row["graph_confidence"],
                }]
            row_chars = len(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            separator_chars = 1 if rows else 0
            remaining = budget - serialized_chars - separator_chars
            if row_chars > remaining:
                fixed = dict(row)
                fixed["content"] = ""
                overhead = len(json.dumps(fixed, ensure_ascii=False, separators=(",", ":")))
                content_budget = max(0, remaining - overhead - 8)
                if content_budget < 128:
                    truncated = True
                    return False
                row["content"] = row["content"][:content_budget]
                bounds = self._excerpt_line_range(row["content"])
                row.update(bounds)
                row_chars = len(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                if row_chars > remaining:
                    truncated = True
                    return False
                truncated = True
            rows.append(row)
            selected_paths.add(relative)
            serialized_chars += row_chars + separator_chars
            return True

        clean_seeds: list[str] = []
        for raw_path in seed_paths:
            relative = safe_relative_path(raw_path)
            if relative and relative not in clean_seeds:
                clean_seeds.append(relative)
        for relative in clean_seeds:
            append_path(relative, source="task_file", score=1_000_000.0)

        query_identifier_terms = {
            part.lower() for token in query_terms for part in re.findall(r"[A-Za-z0-9]+", token)
            if len(part) >= 3 and part.lower() not in SYMBOL_RETRIEVAL_STOPWORDS
        }

        # Repository routing facts are useful only when the prompt identifies
        # a concrete API/route concept or path segment. Never fan out to every
        # route merely because the repository contains them.
        query_lower = query.lower()
        route_context_terms = {"api", "route", "routes", "endpoint", "endpoints", "handler", "request"}
        route_context_terms.update({"get", "post", "put", "patch", "delete"})
        route_context = bool(query_terms & route_context_terms)
        route_candidates = 0
        for node in graph_nodes:
            if node.get("kind") != "route" or not node.get("path"):
                continue
            route = str(node.get("route", "")).strip()
            method = str(node.get("method", "")).upper()
            route_terms = {
                part.lower() for part in re.findall(r"[A-Za-z0-9]+", route)
                if len(part) >= 3 and part.lower() not in {"api", "http", "https", "get", "post", "put", "patch", "delete"}
            }
            route_hit = bool(route and route.lower() in query_lower)
            route_hit = route_hit or (route_context and bool(route_terms & query_identifier_terms))
            if method and method.lower() in query_terms and route_terms & query_identifier_terms:
                route_hit = True
            if not route_hit:
                continue
            relative = safe_relative_path(str(node.get("path", "")))
            expected_hash = graph_file_hashes.get(relative, "")
            if not expected_hash:
                stale_graph_paths.append(relative)
                continue
            if append_path(
                relative,
                source="route",
                score=850_000.0,
                expected_hash=expected_hash,
                stale_bucket="graph",
                relation={
                    "relation": "matched-route-declaration",
                    "related_to": f"{method} {route}".strip(),
                    "provenance": "repository-route-index",
                    "confidence": "source-pattern",
                },
            ):
                route_candidates += 1

        # Stable symbol identities are a deterministic retrieval source in
        # their own right. Require a fresh hash and a meaningful identifier
        # token overlap; do not let an old graph label nominate current code.
        symbol_candidates = 0
        symbol_seeds: list[str] = []
        for node in graph_nodes:
            if node.get("kind") != "symbol" or not node.get("path"):
                continue
            name = str(node.get("label", ""))
            name_terms = {part.lower() for part in re.findall(r"[A-Za-z0-9]+", name) if len(part) >= 3}
            matched = name_terms & query_identifier_terms
            if not matched:
                continue
            relative = safe_relative_path(str(node.get("path", "")))
            expected_hash = str(node.get("content_sha256", ""))
            try:
                current_hash = hashlib.sha256(self.resolve(relative, must_exist=True).read_bytes()).hexdigest()
            except (OSError, ValueError):
                current_hash = ""
            if not expected_hash or current_hash != expected_hash:
                stale_graph_paths.append(relative)
                continue
            if append_path(relative, source="symbol", score=900_000.0):
                symbol_candidates += 1
                symbol_seeds.append(relative)

        fts_candidates = 0
        for candidate in list(fts_candidates_input or [])[:100]:
            if not isinstance(candidate, dict):
                continue
            relative = safe_relative_path(str(candidate.get("path", "")))
            expected_hash = str(candidate.get("content_sha256", ""))
            if not relative or not expected_hash:
                continue
            if append_path(
                relative,
                source="fts",
                score=float(candidate.get("score", 0) or 0),
                expected_hash=expected_hash,
            ):
                fts_candidates += 1

        lexical_limit = max(0, min(int(limit), 100))
        lexical = self.select_context(
            query,
            limit=lexical_limit,
            max_chars=min(budget, lexical_limit * per_file),
        ) if lexical_limit else []
        lexical_seeds: list[str] = []
        for candidate in lexical:
            relative = safe_relative_path(candidate.get("path", ""))
            if not relative:
                continue
            lexical_seeds.append(relative)
            append_path(relative, source="lexical", score=float(candidate.get("score", 0) or 0))

        fresh_graph_seeds: list[str] = []
        context_hash_by_path = {str(row["path"]): str(row["content_sha256"]) for row in rows}
        for relative in dict.fromkeys([*clean_seeds, *symbol_seeds, *lexical_seeds]):
            expected_hash = graph_file_hashes.get(relative, "")
            current_hash = context_hash_by_path.get(relative, "")
            if expected_hash and current_hash == expected_hash:
                fresh_graph_seeds.append(relative)
            else:
                stale_graph_paths.append(relative)
        neighbors = related_code_paths(
            graph,
            fresh_graph_seeds,
            limit=max(0, min(int(graph_limit), 32)),
        )
        for neighbor in neighbors:
            relative = safe_relative_path(neighbor.get("path", ""))
            if not relative:
                continue
            if relative in selected_paths:
                existing = next((row for row in rows if row["path"] == relative), None)
                if existing is not None:
                    edge = {
                        "relation": neighbor.get("relation", ""),
                        "related_to": neighbor.get("related_to", ""),
                        "provenance": neighbor.get("provenance", ""),
                        "confidence": neighbor.get("confidence", ""),
                    }
                    relations = existing.setdefault("graph_relations", [])
                    if edge not in relations:
                        old_size = len(json.dumps(existing, ensure_ascii=False, separators=(",", ":")))
                        relations.append(edge)
                        new_size = len(json.dumps(existing, ensure_ascii=False, separators=(",", ":")))
                        if serialized_chars + new_size - old_size <= budget:
                            serialized_chars += new_size - old_size
                        else:
                            relations.pop()
                            if not relations:
                                existing.pop("graph_relations", None)
                continue
            expected_hash = graph_file_hashes.get(relative, "")
            if not expected_hash:
                stale_graph_paths.append(relative)
                continue
            try:
                current_path = self.resolve(relative, must_exist=True)
                actual_hash = hashlib.sha256(current_path.read_bytes()).hexdigest()
            except (OSError, ValueError):
                stale_graph_paths.append(relative)
                continue
            if actual_hash != expected_hash:
                stale_graph_paths.append(relative)
                continue
            append_path(relative, source="graph", relation=neighbor)

        manifest = [
            {
                "path": row["path"],
                "source": row["retrieval_source"],
                "content_sha256": row["content_sha256"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "graph_relation": row.get("graph_relation", ""),
                "graph_confidence": row.get("graph_confidence", ""),
            }
            for row in rows
        ]
        return rows, {
            "context_chars": serialized_chars,
            "context_budget_chars": budget,
            "context_file_count": len(rows),
            "lexical_candidates": len(lexical),
            "fts_candidates": fts_candidates,
            "route_candidates": route_candidates,
            "symbol_candidates": symbol_candidates,
            "graph_candidates": len(neighbors),
            "stale_graph_paths": sorted(set(stale_graph_paths))[:32],
            "stale_fts_paths": sorted(set(stale_fts_paths))[:32],
            "truncated_to_budget": truncated,
            "manifest": manifest,
        }

    @staticmethod
    def _excerpt_line_range(excerpt: str) -> dict[str, int]:
        numbered = [int(match) for match in re.findall(r"(?m)^(\d+):", excerpt)]
        if numbered:
            return {"line_start": min(numbered), "line_end": max(numbered)}
        line_count = max(1, len(excerpt.splitlines()))
        return {"line_start": 1, "line_end": line_count}

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

    @staticmethod
    def _symbol_provenance(path: Path, text: str) -> str:
        if path.suffix.lower() != ".py":
            return "declaration-pattern"
        try:
            ast.parse(text)
            return "python-ast"
        except SyntaxError:
            return "declaration-pattern-invalid-python-syntax"

    def _routes(self, rel: str, text: str, *, content_sha256: str = "") -> list[dict[str, Any]]:
        patterns = [
            re.compile(r"(?:@\w+\.(get|post|put|patch|delete)|\b(app|router)\.(get|post|put|patch|delete))\s*\(\s*[\"']([^\"']+)", re.I),
            re.compile(r"\(\s*[\"'](GET|POST|PUT|PATCH|DELETE)[\"']\s*,\s*[\"']([^\"']+)", re.I),
        ]
        output: list[dict[str, Any]] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                groups = [group for group in match.groups() if group]
                method = next((item.upper() for item in groups if item.lower() in {"get", "post", "put", "patch", "delete"}), "ANY")
                route = groups[-1]
                output.append({
                    "method": method, "path": route, "file": rel,
                    "line": text.count("\n", 0, match.start()) + 1,
                    "content_sha256": content_sha256,
                    "provenance": "route-pattern-scan",
                })
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
