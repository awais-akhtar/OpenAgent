"""Offline, bounded Tree-sitter indexing for CodeQ.

The language pack can download grammars on demand. CodeQ deliberately does not
call ``get_parser``/``process`` until a grammar is already present in the local
cache, so opening or scanning a project never causes network activity.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import re
import threading
from typing import Any


_lock = threading.RLock()
_initialized: set[tuple[str, str]] = set()
_DECLARATIONS = {
    "function_definition", "function_declaration", "function_item", "function_signature_item",
    "method_definition", "method_declaration", "method", "constructor_declaration",
    "class_definition", "class_declaration", "struct_item", "struct_declaration",
    "interface_declaration", "trait_item", "enum_declaration", "enum_item", "type_alias_declaration",
    "function_definition_item", "function_decl", "method_decl", "class_specifier", "struct_specifier",
    "enum_specifier", "namespace_definition", "function_signature", "function_signature_item",
    "function_statement",
}
_NAME_NODES = {
    "identifier", "type_identifier", "field_identifier", "namespace_identifier",
    "simple_identifier", "operator_name", "property_identifier", "function_name",
}
_IMPORT_NODES = {
    "import_statement", "import_declaration", "use_declaration", "using_directive",
    "using_declaration", "preproc_include", "include_declaration", "import_from_statement",
    "import_from_clause", "package_import",
}
_CALLS = {
    "call", "call_expression", "method_invocation", "function_call_expression",
    "call_expression_suffix", "invocation_expression", "macro_invocation", "command",
}
_IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")
_SUFFIX_LANGUAGES = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript", ".mts": "typescript",
    ".cts": "typescript", ".tsx": "tsx", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
    ".cxx": "cpp", ".hh": "cpp", ".hpp": "cpp", ".hxx": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".sql": "sql", ".vue": "vue",
    ".svelte": "svelte", ".scss": "scss", ".sol": "solidity", ".dart": "dart",
    ".ex": "elixir", ".exs": "elixir", ".gd": "gdscript", ".zig": "zig", ".scala": "scala",
    ".lua": "lua", ".sh": "bash", ".ps1": "powershell", ".html": "html", ".xml": "xml",
    ".css": "css", ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
}


def _cache_dir() -> Path:
    configured = os.environ.get("OPENAGENT_TREE_SITTER_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "OpenAgent" / "cache" / "tree-sitter"
    return Path.home() / ".cache" / "openagent" / "tree-sitter"


def language_for_suffix(suffix: str) -> str:
    """Map source suffixes to the pack's canonical parser name."""
    return _SUFFIX_LANGUAGES.get(suffix.lower(), "")


def supported_language_names() -> tuple[str, ...]:
    """Languages the repository scanner can route to a grammar adapter."""
    return tuple(sorted(set(_SUFFIX_LANGUAGES.values())))


class TreeSitterIndexer:
    """Use only grammars already installed in the local language-pack cache."""

    def __init__(self, *, pack: Any | None = None, cache_dir: str | Path | None = None) -> None:
        self._pack = pack
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else _cache_dir()
        self._loaded: tuple[str, ...] = ()
        self._version = "unavailable"
        self._status = "optional-dependency-missing"
        self._initialize()

    def _initialize(self) -> None:
        pack = self._pack
        if pack is None:
            try:
                import tree_sitter_language_pack as pack  # type: ignore[no-redef]
            except ImportError:
                return
        try:
            self._version = str(importlib.metadata.version("tree-sitter-language-pack"))
        except importlib.metadata.PackageNotFoundError:
            self._version = "unknown"
        key = (str(self.cache_dir), self._version)
        try:
            with _lock:
                if key not in _initialized:
                    # No languages/groups are supplied: init configures the local
                    # cache and does not download grammars.
                    pack.init(pack.PackConfig(cache_dir=str(self.cache_dir)))
                    _initialized.add(key)
                self._loaded = tuple(sorted(set(str(name) for name in pack.downloaded_languages())))
            self._status = "ready" if self._loaded else "no-local-grammars"
            self._pack = pack
        except Exception:
            # A denied/missing cache must degrade to the existing scanner. Never
            # try a network fallback from a repository scan.
            self._status = "local-cache-unavailable"
            self._loaded = ()

    @property
    def version(self) -> str:
        return self._version

    @property
    def loaded_languages(self) -> tuple[str, ...]:
        return self._loaded

    @property
    def cache_key(self) -> str:
        return f"{self._version}|{','.join(self._loaded)}"

    def status(self, language: str) -> dict[str, Any]:
        available = bool(language and language in self._loaded and self._pack is not None)
        return {
            "language": language,
            "parser": "tree-sitter-language-pack" if available else "fallback-scanner",
            "parser_version": self._version if available else "",
            "status": "available" if available else (self._status if language else "unsupported-language"),
        }

    def parse(self, source: str, language: str, *, max_source_bytes: int = 1_000_000) -> dict[str, Any] | None:
        if not language or language not in self._loaded or self._pack is None:
            return None
        encoded = source.encode("utf-8", errors="replace")
        if len(encoded) > max_source_bytes:
            return {"status": "source-too-large", "language": language, "parser_version": self._version}
        try:
            result = self._pack.process(source, self._pack.ProcessConfig(
                language=language,
                structure=True,
                imports=True,
                symbols=True,
                diagnostics=True,
                max_source_bytes=max_source_bytes,
                parse_timeout_ms=250,
            ))
            parsed_symbols: list[dict[str, Any]] = []
            self._flatten_structure(getattr(result, "structure", []) or [], parsed_symbols)
            for item in getattr(result, "symbols", []) or []:
                name = str(getattr(item, "name", "") or "").strip()
                span = getattr(item, "span", None)
                if name and span is not None:
                    parsed_symbols.append({
                        "name": name,
                        "kind": str(getattr(item, "kind", "symbol")),
                        "line": max(1, int(getattr(span, "start_line", 0)) + 1),
                    })
            imports: list[str] = []
            for item in getattr(result, "imports", []) or []:
                value = str(getattr(item, "source", "") or "").strip().strip("\"'<> ")
                if value:
                    quoted = re.findall(r"[\"'<>]([^\"'<>]+)[\"'>]", value)
                    imports.extend(quoted or [value])
            tree_parser = self._pack.get_parser(language)
            if hasattr(tree_parser, "set_timeout_micros"):
                tree_parser.set_timeout_micros(250_000)
            syntax_tree = tree_parser.parse(encoded)
            syntax_root = getattr(syntax_tree, "root_node", None)
            if syntax_root is not None:
                parsed_symbols.extend(self._syntax_symbols(syntax_root))
                imports = sorted(set(imports) | set(self._syntax_imports(syntax_root)))
            calls = self._calls(syntax_root) if syntax_root is not None else []
            unique_symbols: dict[tuple[str, int], dict[str, Any]] = {}
            for item in parsed_symbols:
                unique_symbols[(item["name"], item["line"])] = item
            diagnostics = [
                {"message": str(getattr(item, "message", "")), "severity": str(getattr(item, "severity", "error"))}
                for item in getattr(result, "diagnostics", []) or []
            ]
            return {
                "status": "parsed-with-errors" if diagnostics else "parsed",
                "language": language,
                "parser_version": self._version,
                "symbols": list(unique_symbols.values())[:20_000],
                "imports": sorted(set(imports))[:200],
                "calls": calls[:20_000],
                "diagnostics": diagnostics[:500],
            }
        except Exception as exc:
            # Parser errors are per-file evidence, not a reason to drop the
            # whole repository scan.
            return {
                "status": "parse-failed",
                "language": language,
                "parser_version": self._version,
                "error_type": type(exc).__name__,
                "symbols": [], "imports": [], "calls": [], "diagnostics": [],
            }

    @staticmethod
    def _span_line(value: Any) -> int:
        span = getattr(value, "span", None)
        return max(1, int(getattr(span, "start_line", 0)) + 1) if span is not None else 1

    def _flatten_structure(self, values: Any, output: list[dict[str, Any]]) -> None:
        for item in values if isinstance(values, (list, tuple)) else []:
            name = str(getattr(item, "name", "") or "").strip()
            kind = str(getattr(item, "kind", "symbol"))
            if name:
                output.append({"name": name, "kind": kind, "line": self._span_line(item)})
            self._flatten_structure(getattr(item, "children", []) or [], output)

    @staticmethod
    def _children(node: Any) -> list[Any]:
        return list(getattr(node, "named_children", None) or getattr(node, "children", ()) or ())

    @classmethod
    def _name_for_declaration(cls, node: Any) -> tuple[str, Any | None]:
        for field in ("name", "declarator"):
            value = node.child_by_field_name(field) if hasattr(node, "child_by_field_name") else None
            if value is None:
                continue
            stack = [value]
            while stack:
                current = stack.pop(0)
                if str(getattr(current, "type", "")) in _NAME_NODES:
                    raw = bytes(getattr(current, "text", b"") or b"")
                    return raw.decode("utf-8", errors="replace").strip(), current
                stack[0:0] = cls._children(current)
        body = node.child_by_field_name("body") if hasattr(node, "child_by_field_name") else None
        body_start = int(getattr(body, "start_byte", getattr(node, "end_byte", 0))) if body is not None else int(getattr(node, "end_byte", 0))
        stack = [node]
        while stack:
            current = stack.pop(0)
            if current is not node and str(getattr(current, "type", "")) in _NAME_NODES:
                if int(getattr(current, "start_byte", body_start)) >= body_start:
                    continue
                raw = bytes(getattr(current, "text", b"") or b"")
                name = raw.decode("utf-8", errors="replace").strip()
                if name:
                    return name, current
            stack[0:0] = cls._children(current)
        return "", None

    @classmethod
    def _syntax_symbols(cls, root: Any) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        stack = [root]
        while stack:
            node = stack.pop()
            children = cls._children(node)
            stack.extend(reversed(children))
            kind = str(getattr(node, "type", ""))
            if kind not in _DECLARATIONS:
                continue
            name, name_node = cls._name_for_declaration(node)
            if name and name_node is not None:
                point = getattr(node, "start_point", None)
                row = getattr(point, "row", point[0] if isinstance(point, tuple) else 0)
                output.append({"name": name, "kind": kind, "line": int(row) + 1})
        return output

    @classmethod
    def _syntax_imports(cls, root: Any) -> list[str]:
        output: list[str] = []
        stack = [root]
        while stack:
            node = stack.pop()
            children = cls._children(node)
            stack.extend(reversed(children))
            if str(getattr(node, "type", "")) not in _IMPORT_NODES:
                continue
            value = None
            if hasattr(node, "child_by_field_name"):
                for field in ("source", "path", "argument", "module_name", "name"):
                    value = node.child_by_field_name(field)
                    if value is not None:
                        break
            if value is None:
                value = node
            raw = bytes(getattr(value, "text", b"") or b"").decode("utf-8", errors="replace").strip()
            if raw:
                found = re.findall(r"[\"'<>]([^\"'<>]+)[\"'>]", raw)
                if found:
                    output.extend(found)
                else:
                    raw = raw.strip("\"'<> ")
                    if raw:
                        output.append(raw)
        return output

    @classmethod
    def _calls(cls, root: Any) -> list[dict[str, Any]]:
        """Extract syntactic call edges; graph resolution remains conservative."""
        try:
            if root is None:
                return []
            output: list[dict[str, Any]] = []
            stack = [root]
            while stack:
                node = stack.pop()
                children = cls._children(node)
                stack.extend(reversed(children))
                if str(getattr(node, "type", "")) not in _CALLS:
                    continue
                function = None
                if hasattr(node, "child_by_field_name"):
                    for field in ("function", "name", "method", "callee"):
                        function = node.child_by_field_name(field)
                        if function is not None:
                            break
                if function is None and children:
                    function = children[0]
                text = bytes(getattr(function, "text", b"") or b"").decode("utf-8", errors="replace").strip() if function else ""
                leaf_matches = [text] if str(getattr(function, "type", "")) in {"command_name", "function_name"} else _IDENTIFIER.findall(text)
                if not leaf_matches:
                    continue
                owner = getattr(node, "parent", None)
                owner_name = ""
                while owner is not None:
                    if str(getattr(owner, "type", "")) in _DECLARATIONS:
                        owner_name, _ = cls._name_for_declaration(owner)
                        break
                    owner = getattr(owner, "parent", None)
                if owner_name:
                    point = getattr(node, "start_point", None)
                    row = getattr(point, "row", point[0] if isinstance(point, tuple) else 0)
                    output.append({"symbol": owner_name, "calls": [leaf_matches[-1]], "line": int(row) + 1})
            # Merge repeated call sites for a function while retaining source lines.
            merged: dict[str, dict[str, Any]] = {}
            for item in output:
                row = merged.setdefault(item["symbol"], {"symbol": item["symbol"], "calls": [], "lines": []})
                row["calls"].extend(item["calls"])
                row["lines"].append(item["line"])
            return [{**row, "calls": sorted(set(row["calls"])), "lines": sorted(set(row["lines"])), "provenance": "tree-sitter-call-syntax"} for row in merged.values()]
        except Exception:
            return []
