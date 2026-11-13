"""Deterministic graph views over CodeQ's persisted repository map.

The graph is deliberately derived from scanner evidence, never from model
claims. Stable node IDs are path/symbol based; source hashes and ranges are
metadata that can be refreshed after edits. A normalized SQLite graph store
persists owner/project-scoped graph revisions alongside the repository map.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import posixpath
from pathlib import PurePosixPath
import re
from typing import Any


SCHEMA_VERSION = 1


def _node_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"{kind}:{digest}"


def file_node_id(path: str) -> str:
    """Return the stable graph identity for a repository-relative file path."""
    normalized = str(path).replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("file graph identity requires a relative path")
    return _node_id("file", normalized)


SCRIPT_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".vue", ".svelte")


def _join(directory: str, leaf: str) -> str:
    return leaf if directory in ("", ".") else f"{directory}/{leaf}"


def _python_module_candidates(target: str, known_paths: set[str]) -> list[str]:
    leafs = (".py", ".pyi", "/__init__.py", "/__init__.pyi")
    return [path for path in (target + leaf for leaf in leafs) if path in known_paths and path.startswith(target)]


def _script_candidates(target: str, known_paths: set[str]) -> list[str]:
    """An exact file wins; otherwise an extension; otherwise a directory's index file. Never a loose prefix."""
    exact = [target] if target in known_paths else []
    with_extension = [target + ext for ext in SCRIPT_EXTENSIONS if target + ext in known_paths]
    index_files = [f"{target}/index{ext}" for ext in SCRIPT_EXTENSIONS if f"{target}/index{ext}" in known_paths]
    for group in (exact, with_extension, index_files):
        if group:
            return group
    return []


def resolve_local_import(source_path: str, name: str, known_paths: set[str]) -> list[str]:
    """Repository files an import names. Empty when it names nothing in the repository.

    Relative imports resolve from the importing file's directory (Python dots count
    package levels). Bare names resolve only to repository files: a Python dotted module
    must match a package path, and a bare name such as `os` or `react` never does.
    Ambiguous results are returned as several candidates for the caller to label.
    """
    normalized = str(name).replace("\\", "/").strip()
    if not normalized:
        return []
    source_dir = posixpath.dirname(source_path)
    python_source = source_path.endswith((".py", ".pyi"))
    if normalized.startswith("."):
        if python_source and not normalized.startswith(("./", "../")):
            level = len(normalized) - len(normalized.lstrip("."))
            base = source_dir or "."
            for _ in range(level - 1):
                base = posixpath.dirname(base) or "."
            remainder = normalized[level:].replace(".", "/")
            target = posixpath.normpath(_join(base, remainder)) if remainder else posixpath.normpath(base)
            return _python_module_candidates(target, known_paths)
        return _script_candidates(posixpath.normpath(posixpath.join(source_dir or ".", normalized)), known_paths)
    if "/" in normalized or normalized.startswith("@"):
        return _script_candidates(posixpath.normpath(normalized), known_paths)
    # A dotted name is a Python module path from the repository root. A plain name is only
    # matched at the root, so `import os` never resolves to some unrelated `os.py`.
    target = normalized.replace(".", "/")
    # A script can import its sibling modules (its own directory is on the import path), then the root.
    found = _python_module_candidates(_join(source_dir or ".", target), known_paths) if python_source else []
    found = found or _python_module_candidates(target, known_paths)
    if not found and "." in normalized:
        found = [path for path in sorted(known_paths) if path.endswith(("/" + target + ".py", "/" + target + "/__init__.py"))]
    return found


def _norm_dependency(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _dependency_root(name: str) -> str:
    """The package an import name belongs to: `@scope/pkg` for scoped npm, first segment otherwise."""
    text = str(name).strip()
    if text.startswith("@"):
        return _norm_dependency("/".join(text.split("/")[:2]))
    return _norm_dependency(re.split(r"[./]", text)[0])


def build_code_knowledge_graph(project_map: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, stable graph backed only by repository scan records."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_node(kind: str, key: str, label: str, *, stable_id: str = "", **data: Any) -> str:
        node_id = stable_id or _node_id(kind, key)
        nodes[node_id] = {"id": node_id, "kind": kind, "key": key, "label": label, **data}
        return node_id

    def add_edge(source: str, target: str, relation: str, provenance: str, confidence: str = "verified", **extra: Any) -> None:
        if not source or not target:
            return
        edge_key = (source, target, relation)
        edges[edge_key] = {
            "id": _node_id("edge", "\0".join(edge_key)),
            "source": source,
            "target": target,
            "relation": relation,
            "provenance": provenance,
            "confidence": confidence,
            **extra,
        }

    dependency_nodes: dict[str, str] = {}

    def ensure_dependency(ecosystem: str, name: str) -> str:
        key = f"{ecosystem}\0{name}"
        if key not in dependency_nodes:
            dependency_nodes[key] = add_node("dependency", key, name, ecosystem=ecosystem)
        return dependency_nodes[key]

    root_key = str(project_map.get("root", ""))
    project_id = add_node("project", root_key or "repository", PurePosixPath(root_key).name or "Repository", root=root_key)

    inventory = project_map.get("file_inventory", [])
    file_nodes: dict[str, str] = {}
    for entry in inventory if isinstance(inventory, list) else []:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", "")).replace("\\", "/").strip("/")
        if not path:
            continue
        node = add_node(
            "file", path, PurePosixPath(path).name,
            path=path,
            content_sha256=str(entry.get("content_sha256", "")),
            freshness=str(entry.get("index_status", "unknown")),
            size_bytes=int(entry.get("size_bytes", 0) or 0),
            language=str(entry.get("language", "")),
        )
        file_nodes[path] = node
        add_edge(project_id, node, "contains", "filesystem-scan")

    def ensure_file(path: str) -> str:
        normalized = str(path).replace("\\", "/").strip("/")
        if normalized not in file_nodes:
            node = add_node("file", normalized, PurePosixPath(normalized).name, path=normalized, content_sha256="", freshness="not-indexed")
            file_nodes[normalized] = node
            add_edge(project_id, node, "contains", "repository-map")
        return file_nodes[normalized]

    symbols_by_name: dict[str, list[str]] = defaultdict(list)
    symbols_by_location: dict[tuple[str, str, int], str] = {}
    symbols_by_scip_id: dict[str, str] = {}
    ordinal: dict[tuple[str, str, str], int] = defaultdict(int)
    symbol_items = project_map.get("symbols", []) if isinstance(project_map.get("symbols"), list) else []
    # SCIP's compiler-backed global identity takes precedence over lexical
    # fallback identity when both describe the same source declaration.
    symbol_items = sorted(symbol_items, key=lambda item: (not bool(item.get("stable_symbol_id")) if isinstance(item, dict) else True))
    for item in symbol_items:
        if not isinstance(item, dict):
            continue
        path, name, kind = str(item.get("path", "")), str(item.get("name", "")), str(item.get("kind", "symbol"))
        if not path or not name:
            continue
        identity = (path.replace("\\", "/"), kind, name)
        line = int(item.get("line", 0) or 0)
        location = (identity[0], name, line)
        scip_symbol = str(item.get("stable_symbol_id", ""))
        if not scip_symbol and location in symbols_by_location:
            symbol_id = symbols_by_location[location]
        else:
            ordinal[identity] += 1
            if scip_symbol:
                scoped_scip_id = f"{identity[0]}\0{scip_symbol}" if scip_symbol.startswith("local ") else scip_symbol
                key = f"scip\0{scoped_scip_id}"
                stable_id = _node_id("symbol", key)
            else:
                scoped_scip_id = ""
                key = f"{identity[0]}\0{kind}\0{name}\0{ordinal[identity]}"
                stable_id = ""
            symbol_id = add_node(
                "symbol", key, name,
                stable_id=stable_id,
                path=identity[0], symbol_kind=kind,
                line=line,
                content_sha256=str(item.get("content_sha256", "")),
                provenance=str(item.get("provenance", "repository-symbol-scan")),
                stable_symbol_id=scip_symbol,
                relationships=item.get("relationships", []),
            )
            symbols_by_location[location] = symbol_id
            if scip_symbol:
                symbols_by_scip_id[scoped_scip_id] = symbol_id
        if symbol_id not in symbols_by_name[name]:
            symbols_by_name[name].append(symbol_id)
        add_edge(ensure_file(path), symbol_id, "defines", str(item.get("provenance", "repository-symbol-scan")))

    # SCIP occurrences are used only after the SCIP reader revalidates their
    # current source range/hash. References never use unverified name matches.
    for occurrence in project_map.get("scip_occurrences", []) if isinstance(project_map.get("scip_occurrences"), list) else []:
        if not isinstance(occurrence, dict):
            continue
        path = str(occurrence.get("path", "")).replace("\\", "/").strip("/")
        scip_symbol = str(occurrence.get("symbol", ""))
        if not path or not scip_symbol:
            continue
        scoped_scip_id = f"{path}\0{scip_symbol}" if scip_symbol.startswith("local ") else scip_symbol
        symbol_id = symbols_by_scip_id.get(scoped_scip_id)
        if not symbol_id:
            key = f"scip\0{scoped_scip_id}"
            symbol_id = add_node(
                "symbol", key, str(occurrence.get("name", scip_symbol)),
                stable_id=_node_id("symbol", key), path="", symbol_kind="external-or-unresolved",
                content_sha256="", provenance=str(occurrence.get("provenance", "scip")),
                stable_symbol_id=scip_symbol, freshness="definition-not-indexed",
            )
            symbols_by_scip_id[scoped_scip_id] = symbol_id
        if bool(occurrence.get("is_definition")):
            add_edge(ensure_file(path), symbol_id, "defines", str(occurrence.get("provenance", "scip")),
                     str(occurrence.get("confidence", "source-range-validated")), line=int(occurrence.get("line", 0) or 0),
                     content_sha256=str(occurrence.get("content_sha256", "")), index_sha256=str(occurrence.get("index_sha256", "")))
        else:
            add_edge(ensure_file(path), symbol_id, "references", str(occurrence.get("provenance", "scip")),
                     str(occurrence.get("confidence", "source-range-validated")), line=int(occurrence.get("line", 0) or 0),
                     content_sha256=str(occurrence.get("content_sha256", "")), index_sha256=str(occurrence.get("index_sha256", "")))
        for relationship in occurrence.get("relationships", []) if isinstance(occurrence.get("relationships"), list) else []:
            if not isinstance(relationship, dict) or not relationship.get("symbol"):
                continue
            target_symbol = str(relationship["symbol"])
            target_scoped = f"{path}\0{target_symbol}" if target_symbol.startswith("local ") else target_symbol
            target_id = symbols_by_scip_id.get(target_scoped)
            if not target_id:
                target_key = f"scip\0{target_scoped}"
                target_id = add_node("symbol", target_key, target_symbol, stable_id=_node_id("symbol", target_key),
                                     path="", symbol_kind="external-or-unresolved", stable_symbol_id=target_symbol,
                                     provenance=str(occurrence.get("provenance", "scip")))
                symbols_by_scip_id[target_scoped] = target_id
            relation = "implements" if relationship.get("is_implementation") else "related-symbol"
            add_edge(symbol_id, target_id, relation, str(occurrence.get("provenance", "scip")),
                     str(occurrence.get("confidence", "source-range-validated")))

    scip_index = project_map.get("scip_index", {}) if isinstance(project_map.get("scip_index"), dict) else {}
    if scip_index.get("index_sha256"):
        tool = scip_index.get("tool", {}) if isinstance(scip_index.get("tool"), dict) else {}
        indexer_key = f"{tool.get('name', 'unknown')}@{tool.get('version', 'unknown')}"
        indexer_id = add_node(
            "indexer", indexer_key, indexer_key,
            index_sha256=str(scip_index.get("index_sha256", "")),
            status=str(scip_index.get("status", "unknown")),
            validated_document_count=int(scip_index.get("validated_document_count", 0) or 0),
            provenance="scip-index-metadata",
        )
        add_edge(project_id, indexer_id, "indexed-by", "scip-index-metadata", "indexer-identified")

    for route in project_map.get("routes", []) if isinstance(project_map.get("routes"), list) else []:
        if not isinstance(route, dict):
            continue
        path = str(route.get("file", ""))
        method, route_path = str(route.get("method", "ANY")), str(route.get("path", ""))
        if not path or not route_path:
            continue
        normalized_route_file = path.replace("\\", "/")
        key = f"{normalized_route_file}\0{method.upper()}\0{route_path}"
        route_id = add_node(
            "route", key, f"{method.upper()} {route_path}", path=path,
            method=method.upper(), route=route_path, line=int(route.get("line", 0) or 0),
            content_sha256=str(route.get("content_sha256", "")),
            provenance=str(route.get("provenance", "route-pattern-scan")),
        )
        add_edge(ensure_file(path), route_id, "declares", str(route.get("provenance", "route-pattern-scan")), "source-pattern")

    test_nodes: dict[str, str] = {}
    for raw_path in project_map.get("test_files", []) if isinstance(project_map.get("test_files"), list) else []:
        path = str(raw_path).replace("\\", "/").strip("/")
        if path:
            test_id = add_node("test", path, PurePosixPath(path).name, path=path)
            test_nodes[path] = test_id
            add_edge(project_id, test_id, "has-test", "test-file-discovery")

    # These are deliberately labeled candidates: naming conventions suggest
    # likely coverage but do not prove that a test exercises a source file.
    for test_path, test_id in test_nodes.items():
        name = PurePosixPath(test_path).name.lower()
        stem = name
        for prefix in ("test_",):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        for suffix in (".test.py", ".spec.py", ".test.tsx", ".test.ts", ".spec.tsx", ".spec.ts", ".test.jsx", ".test.js", ".spec.jsx", ".spec.js", ".py", ".tsx", ".ts", ".jsx", ".js"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        if stem:
            candidates = [path for path in file_nodes if PurePosixPath(path).stem.lower() == stem and path not in test_nodes]
            if len(candidates) == 1:
                add_edge(test_id, file_nodes[candidates[0]], "possible-tests", "filename-convention", "candidate")

    # Imports are references. A local target is linked only when the resolver finds exactly one
    # repository file; otherwise the edge says why it is not linked (ambiguous, or external and
    # unresolved), instead of guessing.
    known_paths = set(file_nodes)
    declared_dependencies: dict[str, tuple[str, str]] = {}
    raw_dependencies = project_map.get("dependencies", {}) if isinstance(project_map.get("dependencies"), dict) else {}
    for ecosystem, names in sorted(raw_dependencies.items()):
        for name in names if isinstance(names, list) else []:
            declared_dependencies.setdefault(_norm_dependency(str(name)), (str(ecosystem), str(name)))
    for source_path, imports in (project_map.get("imports", {}) or {}).items():
        source_path = str(source_path).replace("\\", "/").strip("/")
        source_id = ensure_file(source_path)
        for imported in imports if isinstance(imports, list) else []:
            name = str(imported).strip()
            if not name:
                continue
            candidates = resolve_local_import(source_path, name, known_paths)
            node_resolution = "resolved-local" if len(candidates) == 1 else "ambiguous" if len(candidates) > 1 else "unresolved"
            import_id = add_node("import", name, name, resolution=node_resolution)
            if len(candidates) == 1:
                resolution = "resolved-local"
                add_edge(source_id, file_nodes[candidates[0]], "imports-local", "import-resolution", "unambiguous-path", resolution=resolution)
            elif len(candidates) > 1:
                resolution = "ambiguous"
            elif name.startswith("."):
                resolution = "unresolved-relative"
            else:
                resolution = "external-or-unresolved"
            add_edge(source_id, import_id, "imports", "import-scan", "reference", resolution=resolution)
            declared = declared_dependencies.get(_dependency_root(name))
            if declared is not None:
                add_edge(source_id, ensure_dependency(declared[0], declared[1]), "uses-dependency", "import-name-match", "name-match", resolution="declared-dependency")

    # Configuration, dependency declarations and commands are linked to the project or the
    # manifest that declares them.
    for config_path in project_map.get("configuration_files", []) if isinstance(project_map.get("configuration_files"), list) else []:
        normalized_config = str(config_path).replace("\\", "/").strip("/")
        if normalized_config:
            add_edge(project_id, ensure_file(normalized_config), "configuration-file", "config-scan")
    manifest_for = {"npm": ("package.json",), "python": ("requirements.txt", "pyproject.toml")}
    for ecosystem, names in sorted(raw_dependencies.items()):
        manifest = next((candidate for candidate in manifest_for.get(str(ecosystem), ()) if candidate in file_nodes), "")
        source_node = file_nodes[manifest] if manifest else project_id
        for name in names if isinstance(names, list) else []:
            add_edge(source_node, ensure_dependency(str(ecosystem), str(name)), "declares-dependency", "manifest-scan")
    commands = project_map.get("commands", {}) if isinstance(project_map.get("commands"), dict) else {}
    for command_name, command_text in sorted(commands.items()):
        command_id = add_node("command", f"{command_name}\0{command_text}", str(command_name), command=str(command_text))
        add_edge(project_id, command_id, "defines-command", "command-scan")

    for call in project_map.get("call_graph", []) if isinstance(project_map.get("call_graph"), list) else []:
        if not isinstance(call, dict):
            continue
        owner_name = str(call.get("symbol", ""))
        owner_candidates = symbols_by_name.get(owner_name, [])
        if len(owner_candidates) != 1:
            continue
        owner_id = owner_candidates[0]
        for callee in call.get("calls", []) if isinstance(call.get("calls"), list) else []:
            target_candidates = symbols_by_name.get(str(callee), [])
            if len(target_candidates) == 1:
                provenance = str(call.get("provenance", "python-ast-call-scan"))
                confidence = "unique-symbol-name"
                if provenance.startswith("tree-sitter-call-syntax"):
                    confidence = "syntax-call-unique-symbol-name"
                add_edge(owner_id, target_candidates[0], "calls", provenance, confidence)

    graph_nodes = sorted(nodes.values(), key=lambda item: (item["kind"], item["key"], item["id"]))
    graph_edges = sorted(edges.values(), key=lambda item: (item["relation"], item["source"], item["target"]))
    revision_payload = json.dumps({"nodes": graph_nodes, "edges": graph_edges}, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": hashlib.sha256(revision_payload.encode("utf-8")).hexdigest(),
        "nodes": graph_nodes,
        "edges": graph_edges,
        "summary": {
            "nodes": len(graph_nodes),
            "edges": len(graph_edges),
            "files": sum(node["kind"] == "file" for node in graph_nodes),
            "symbols": sum(node["kind"] == "symbol" for node in graph_nodes),
            "routes": sum(node["kind"] == "route" for node in graph_nodes),
            "tests": sum(node["kind"] == "test" for node in graph_nodes),
        },
    }


def related_code_paths(graph: dict[str, Any], seed_paths: list[str], *, limit: int = 12) -> list[dict[str, Any]]:
    """Return source-backed import, test, call, and SCIP reference neighbors for seed files."""
    nodes = {str(node.get("id")): node for node in graph.get("nodes", []) if isinstance(node, dict)}
    path_by_id = {node_id: str(node.get("path", "")) for node_id, node in nodes.items() if node.get("path")}
    symbol_path_by_id = {node_id: str(node.get("path", "")) for node_id, node in nodes.items() if node.get("kind") == "symbol" and node.get("path")}
    seeds = {str(path).replace("\\", "/").strip("/") for path in seed_paths}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        relation = str(edge.get("relation", ""))
        if relation not in {"imports-local", "possible-tests", "references"}:
            continue
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        source_path, target_path = path_by_id.get(source, ""), path_by_id.get(target, "")
        if relation == "imports-local" and source_path in seeds and target_path not in seeds:
            item = {"path": target_path, "relation": "imported-by-selected-file", "related_to": source_path, "provenance": edge.get("provenance"), "confidence": edge.get("confidence")}
            result[(target_path, item["relation"])] = item
        elif relation == "imports-local" and target_path in seeds and source_path not in seeds:
            item = {"path": source_path, "relation": "imports-selected-file", "related_to": target_path, "provenance": edge.get("provenance"), "confidence": edge.get("confidence")}
            result[(source_path, item["relation"])] = item
        elif relation == "possible-tests" and target_path in seeds:
            item = {"path": source_path, "relation": "possible-test-for-selected-file", "related_to": target_path, "provenance": edge.get("provenance"), "confidence": "candidate"}
            result[(source_path, item["relation"])] = item
        elif relation == "references" and source_path in seeds and target_path and target_path not in seeds:
            item = {"path": target_path, "relation": "symbol-reference-target", "related_to": source_path, "provenance": edge.get("provenance"), "confidence": edge.get("confidence")}
            result[(target_path, item["relation"])] = item
        elif relation == "references" and target_path in seeds and source_path and source_path not in seeds:
            item = {"path": source_path, "relation": "references-selected-symbol", "related_to": target_path, "provenance": edge.get("provenance"), "confidence": edge.get("confidence")}
            result[(source_path, item["relation"])] = item

    # Calls are linked only when the scanner resolved the call to one symbol.
    # Expand one hop in either direction so a task can see a selected function's
    # callees and its callers without trusting ambiguous name-only matches.
    symbols_by_seed_path: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("relation") != "defines":
            continue
        file_path = path_by_id.get(str(edge.get("source", "")), "")
        symbol_id = str(edge.get("target", ""))
        if file_path in seeds and symbol_id in symbol_path_by_id:
            symbols_by_seed_path[file_path].add(symbol_id)
    seed_symbol_ids = {symbol_id for ids in symbols_by_seed_path.values() for symbol_id in ids}
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict) or edge.get("relation") != "calls":
            continue
        source_id, target_id = str(edge.get("source", "")), str(edge.get("target", ""))
        if source_id in seed_symbol_ids and target_id in symbol_path_by_id:
            related_path = symbol_path_by_id[target_id]
            related_to = symbol_path_by_id[source_id]
            relation = "selected-file-calls"
        elif target_id in seed_symbol_ids and source_id in symbol_path_by_id:
            related_path = symbol_path_by_id[source_id]
            related_to = symbol_path_by_id[target_id]
            relation = "calls-selected-file"
        else:
            continue
        if related_path and related_path not in seeds:
            item = {
                "path": related_path,
                "relation": relation,
                "related_to": related_to,
                "provenance": edge.get("provenance"),
                "confidence": edge.get("confidence"),
            }
            result[(related_path, relation)] = item
    return sorted(result.values(), key=lambda item: (item["relation"], item["path"]))[:max(0, int(limit))]


MAX_QUERY_LIMIT = 200


def query_graph(
    graph: dict[str, Any],
    *,
    kind: str | None = None,
    text: str | None = None,
    path_prefix: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Bounded lookup over graph nodes: filter by kind, label/path text, or path prefix.

    `limit` is clamped to MAX_QUERY_LIMIT so large graphs never return unbounded
    results. `truncated` reports whether more matches existed.
    """
    bounded = max(1, min(int(limit), MAX_QUERY_LIMIT))
    needle = (text or "").strip().lower()
    prefix = (path_prefix or "").replace("\\", "/").strip("/")
    matches: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if kind is not None and node.get("kind") != kind:
            continue
        path = str(node.get("path", ""))
        if prefix and not (path == prefix or path.startswith(prefix + "/")):
            continue
        if needle and needle not in str(node.get("label", "")).lower() and needle not in path.lower():
            continue
        matches.append(node)
    return {"nodes": matches[:bounded], "truncated": len(matches) > bounded, "total": len(matches), "limit": bounded}


def safe_relative_path(relative: str) -> str | None:
    """Return a normalized repository-relative path, or None if it would escape the root.

    Rejects absolute paths, drive letters, and any `..` segment.
    """
    text = str(relative or "").replace("\\", "/").strip()
    if not text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        return None
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def prune_stale_references(graph: dict[str, Any], live_paths: set[str]) -> dict[str, Any]:
    """Drop file nodes whose file no longer exists, and every edge that touches them.

    Used on a stored revision before its references reach a prompt. Returns the pruned graph
    and how many nodes were rejected, so the caller can record it.
    """
    live = {str(path).replace("\\", "/").strip("/") for path in live_paths}
    stale_ids = {
        str(node.get("id"))
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("kind") == "file" and str(node.get("path", "")) not in live
    }
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict) and str(node.get("id")) not in stale_ids]
    edges = [
        edge for edge in graph.get("edges", [])
        if isinstance(edge, dict) and str(edge.get("source")) not in stale_ids and str(edge.get("target")) not in stale_ids
    ]
    return {**graph, "nodes": nodes, "edges": edges, "rejected_stale_nodes": len(stale_ids)}
