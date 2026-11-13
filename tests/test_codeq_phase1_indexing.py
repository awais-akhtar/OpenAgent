"""Phase 1 parser/indexer integration and incremental refresh regressions."""

from __future__ import annotations

import hashlib
import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.scip_index import read_scip_index
from openagent.codebase.tree_sitter_index import TreeSitterIndexer, language_for_suffix, supported_language_names
from openagent.codebase.work_graph import build_code_knowledge_graph, related_code_paths
from openagent.codebase import graph_store
from openagent.codebase.service import CodebaseService
from openagent.storage import OpenAgentStore


def _varint(value: int) -> bytes:
    value = int(value)
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _vfield(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _bfield(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _sfield(number: int, value: str) -> bytes:
    return _bfield(number, value.encode("utf-8"))


def _scip_occurrence(symbol: str, line: int, start: int, end: int, *, definition: bool = False) -> bytes:
    packed = b"".join(_varint(value) for value in (line, start, line, end))
    # Current SCIP uses typed multi-line range (field 9); line/columns are
    # zero-based and the end is half-open.
    multiline = _vfield(1, line) + _vfield(2, start) + _vfield(3, line) + _vfield(4, end)
    return (
        _bfield(9, multiline)
        + _sfield(2, symbol)
        + _vfield(3, 1 if definition else 0)
        + _bfield(1, packed)
    )


def _scip_document(path: str, *, occurrences: list[bytes], symbol_info: bytes = b"", encoding: int = 3) -> bytes:
    result = _sfield(1, path) + _sfield(4, "python") + _vfield(6, encoding)
    result += b"".join(_bfield(2, occurrence) for occurrence in occurrences)
    if symbol_info:
        result += _bfield(3, symbol_info)
    return result


def _scip_symbol_info(symbol: str, name: str) -> bytes:
    return _sfield(1, symbol) + _sfield(6, name)


def _write_scip(root: Path, documents: list[bytes]) -> Path:
    tool = _sfield(1, "scip-python") + _sfield(2, "0.4.0")
    metadata = _bfield(2, tool) + _sfield(3, root.as_uri())
    payload = _bfield(1, metadata) + b"".join(_bfield(2, document) for document in documents)
    path = root / "index.scip"
    path.write_bytes(payload)
    return path


class FakeTreeSitterPack:
    class ProcessConfig:
        def __init__(self, **values):
            self.values = values

    class PackConfig:
        def __init__(self, **values):
            self.values = values

    def __init__(self, languages: list[str], result=None) -> None:
        self.languages = languages
        self.result = result or SimpleNamespace(
            structure=[SimpleNamespace(name="parse", kind="function", span=SimpleNamespace(start_line=2), children=[])],
            symbols=[],
            imports=[SimpleNamespace(source="./helper")],
            diagnostics=[],
        )
        self.process_calls = 0

    def init(self, config) -> None:
        return None

    def downloaded_languages(self) -> list[str]:
        return self.languages

    def process(self, source, config):
        self.process_calls += 1
        return self.result

    def get_parser(self, language):
        return SimpleNamespace(
            parse=lambda source: SimpleNamespace(root_node=SimpleNamespace(children=[])),
        )


class PhaseOneIndexingTests(unittest.TestCase):
    def test_language_routing_covers_declared_grammars_without_claiming_all_are_installed(self) -> None:
        self.assertEqual("python", language_for_suffix(".PY"))
        self.assertEqual("typescript", language_for_suffix(".mts"))
        self.assertEqual("csharp", language_for_suffix(".cs"))
        self.assertEqual("rust", language_for_suffix(".rs"))
        self.assertEqual("", language_for_suffix(".unknown"))

    def test_tree_sitter_uses_installed_grammar_and_reports_parser_provenance(self) -> None:
        pack = FakeTreeSitterPack(["python"])
        indexer = TreeSitterIndexer(pack=pack, cache_dir=Path(tempfile.gettempdir()) / "codeq-tree-sitter-test")
        result = indexer.parse("def parse():\n    return 1\n", "python")
        self.assertEqual(1, pack.process_calls)
        self.assertEqual("parsed", result["status"])
        self.assertEqual("parse", result["symbols"][0]["name"])
        self.assertEqual(3, result["symbols"][0]["line"])
        self.assertEqual(["./helper"], result["imports"])

    def test_tree_sitter_does_not_load_or_download_missing_grammars(self) -> None:
        pack = FakeTreeSitterPack([])
        indexer = TreeSitterIndexer(pack=pack, cache_dir=Path(tempfile.gettempdir()) / "codeq-tree-sitter-empty")
        self.assertIsNone(indexer.parse("def f(): pass", "python"))
        self.assertEqual(0, pack.process_calls)
        self.assertEqual("no-local-grammars", indexer.status("python")["status"])

    def test_all_declared_installed_grammars_parse_valid_language_fixtures(self) -> None:
        indexer = TreeSitterIndexer()
        expected = set(indexer.loaded_languages)
        declared = set(supported_language_names())
        if not declared.issubset(expected):
            self.skipTest(f"requires all {len(declared)} optional grammars; currently cached: {len(expected & declared)}")
        samples = {
            "bash": "greet() { echo hello; }\n",
            "c": "int add(int a, int b) { return a + b; }\nint run() { return add(1, 2); }\n",
            "cpp": "int add(int a, int b) { return a + b; }\nint run() { return add(1, 2); }\n",
            "csharp": "public class Demo { public int Add(int a, int b) { return a+b; } }\n",
            "css": ".thing { color: red; }\n",
            "dart": "int add(int a, int b) => a+b;\n",
            "elixir": "defmodule Demo do\n def add(a,b), do: a+b\nend\n",
            "gdscript": "func add(a: int, b: int) -> int:\n return a+b\n",
            "go": "package demo\nfunc Add(a int,b int) int { return a+b }\n",
            "html": "<html><body><p>Hello</p></body></html>\n",
            "java": "class Demo { int add(int a,int b) { return a+b; } }\n",
            "javascript": "export function add(a,b) { return a+b; }\n",
            "json": '{"name":"demo"}\n',
            "kotlin": "fun add(a: Int,b: Int): Int = a+b\n",
            "lua": "function add(a,b) return a+b end\n",
            "php": "<?php function add($a,$b) { return $a+$b; }\n",
            "powershell": "function Add-Numbers {\n param([int]$a, [int]$b)\n return $a + $b\n}\n",
            "python": "def add(a,b):\n return a+b\n",
            "ruby": "def add(a,b)\n a+b\nend\n",
            "rust": "pub fn add(a:i32,b:i32)->i32 { a+b }\n",
            "scala": "def add(a: Int,b: Int): Int = a+b\n",
            "scss": ".thing { color: red; }\n",
            "solidity": "contract Demo { function add(uint a,uint b) public pure returns(uint) { return a+b; } }\n",
            "sql": "CREATE TABLE demo (id INTEGER);\n",
            "svelte": "<script>function add(a,b){return a+b}</script>\n",
            "swift": "func add(_ a:Int,_ b:Int)->Int { a+b }\n",
            "toml": "[project]\nname=\"demo\"\n",
            "tsx": "export function App(){ return <div>Hi</div> }\n",
            "typescript": "export function add(a:number,b:number):number { return a+b }\n",
            "vue": "<script>export default { methods: { greet() { return \"hi\" } } }</script>\n",
            "xml": "<root><item>hello</item></root>\n",
            "yaml": "name: demo\n",
            "zig": "pub fn add(a:i32,b:i32) i32 { return a+b; }\n",
        }
        for language in sorted(declared):
            with self.subTest(language=language):
                result = indexer.parse(samples[language], language)
                self.assertIsNotNone(result)
                self.assertEqual("parsed", result["status"], result.get("diagnostics"))

    def test_real_parser_records_imports_calls_routes_and_tests(self) -> None:
        indexer = TreeSitterIndexer()
        if not {"python", "typescript"}.issubset(indexer.loaded_languages):
            self.skipTest("Python and TypeScript grammar bundles are not locally installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "helper.py").write_text("def calc():\n    return 1\n", encoding="utf-8")
            (root / "main.py").write_text(
                "from helper import calc\nfrom fastapi import FastAPI\napp = FastAPI()\n"
                "@app.get('/calc')\ndef run():\n    return calc()\n",
                encoding="utf-8",
            )
            (root / "app.ts").write_text(
                'import { formatName } from "./ts_helper";\nexport function runApp(){ return formatName("a"); }\n', encoding="utf-8",
            )
            (root / "ts_helper.ts").write_text(
                "export function formatName(value: string) { return value.trim(); }\n", encoding="utf-8",
            )
            (root / "test_main.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
            project_map = RepositoryIntelligence(root).scan()
            by_path = project_map["tree_sitter_files"]
            self.assertTrue(by_path["main.py"]["used_for_relationships"])
            self.assertTrue(next(item["provenance"] for item in project_map["symbols"] if item["path"] == "main.py").startswith("tree-sitter:python@"))
            self.assertIn("helper", project_map["imports"]["main.py"])
            self.assertTrue(any(item["path"] == "/calc" and item["file"] == "main.py" for item in project_map["routes"]))
            self.assertTrue(any(item["symbol"] == "run" and "calc" in item["calls"] for item in project_map["call_graph"]))
            self.assertIn("test_main.py", project_map["test_files"])
            self.assertIn("./ts_helper", project_map["imports"]["app.ts"])
            graph = build_code_knowledge_graph(project_map)
            ts_neighbors = related_code_paths(graph, ["app.ts"])
            self.assertTrue(any(
                item["path"] == "ts_helper.ts"
                and item["relation"] == "selected-file-calls"
                and item["provenance"].startswith("tree-sitter-call-syntax")
                for item in ts_neighbors
            ), ts_neighbors)

    def test_real_tree_sitter_cross_file_graph_and_incremental_refresh_across_languages(self) -> None:
        """Exercise the installed parser through scan, graph construction and refresh.

        Parser-only fixtures and synthetic graph inputs can pass while the production
        scanner-to-graph seam is broken. These paired-file projects cover four
        language grammars through the same path CodeQ uses for project indexing.
        """
        indexer = TreeSitterIndexer()
        fixtures = {
            "python": (
                {"view.py": "def render():\n    return format_value('x')\n",
                 "format.py": "def format_value(value):\n    return value.strip()\n"},
                "view.py", "format.py", "format_value",
                {"view.py": "def render():\n    return 1\n"},
            ),
            "typescript": (
                {"view.ts": 'export function render(){ return formatValue("x"); }\n',
                 "format.ts": "export function formatValue(value: string){ return value.trim(); }\n"},
                "view.ts", "format.ts", "formatValue",
                {"view.ts": "export function render(){ return 1; }\n"},
            ),
            "go": (
                {"view.go": "package demo\nfunc Render() int { return FormatValue() }\n",
                 "format.go": "package demo\nfunc FormatValue() int { return 1 }\n"},
                "view.go", "format.go", "FormatValue",
                {"view.go": "package demo\nfunc Render() int { return 1 }\n"},
            ),
            "java": (
                {"View.java": "class View { int render(){ return formatValue(); } }\n",
                 "Format.java": "class Format { int formatValue(){ return 1; } }\n"},
                "View.java", "Format.java", "formatValue",
                {"View.java": "class View { int render(){ return 1; } }\n"},
            ),
        }
        required = {"python", "typescript", "go", "java"}
        if not required.issubset(indexer.loaded_languages):
            self.skipTest(f"requires local grammars {sorted(required)}; available: {indexer.loaded_languages}")

        for language, (files, caller, callee, symbol, changed_files) in fixtures.items():
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for relative, source in files.items():
                    (root / relative).write_text(source, encoding="utf-8")

                scanner = RepositoryIntelligence(root)
                before_map = scanner.scan()
                before_graph = build_code_knowledge_graph(before_map)
                parser_rows = before_map["tree_sitter_files"]
                self.assertTrue(parser_rows[caller]["used_for_relationships"], parser_rows[caller])
                self.assertTrue(parser_rows[callee]["used_for_relationships"], parser_rows[callee])
                self.assertTrue(any(
                    item["path"] == callee
                    and item["relation"] == "selected-file-calls"
                    and item["provenance"] == "tree-sitter-call-syntax"
                    for item in related_code_paths(before_graph, [caller])
                ), (language, related_code_paths(before_graph, [caller])))

                for relative, source in changed_files.items():
                    (root / relative).write_text(source, encoding="utf-8")
                after_map = scanner.scan(previous=before_map)
                after_graph = build_code_knowledge_graph(after_map)
                self.assertEqual(1, after_map["index_metadata"]["files_reused"])
                self.assertEqual(1, after_map["index_metadata"]["files_parsed"])
                self.assertFalse(any(
                    item["path"] == callee and item["relation"] == "selected-file-calls"
                    for item in related_code_paths(after_graph, [caller])
                ), (language, related_code_paths(after_graph, [caller])))

    def test_tree_sitter_extracts_symbols_imports_and_calls_across_grammars(self) -> None:
        indexer = TreeSitterIndexer()
        samples = {
            "cpp": ("int add(int a,int b){return a+b;} int run(){return add(1,2);}", "run", "add"),
            "dart": ("int helper(int x)=>x; int add(int a,int b)=>helper(a)+b;", "add", None),
            "go": ('package demo\nimport "fmt"\nfunc Add(a int,b int) int{return a+b}\nfunc Run() int{return Add(1,2)}', "Run", "Add"),
            "java": ("import java.util.List; class A { int add(int x){return helper(x);} int helper(int x){return x;} }", "add", "helper"),
            "javascript": ('import { helper } from "./helper"; export function add(a){return helper(a)}', "add", "helper"),
            "kotlin": ("fun helper(x:Int):Int=x\nfun add(a:Int,b:Int):Int=helper(a)+b", "add", "helper"),
            "powershell": ("function Add-Numbers { return 1 }\nfunction Run { Add-Numbers 1 2 }", "Run", "Add-Numbers"),
            "python": ("from helper import add\ndef run(a,b): return add(a,b)", "run", "add"),
        }
        for language, (source, expected_symbol, expected_call) in samples.items():
            with self.subTest(language=language):
                if language not in indexer.loaded_languages:
                    self.skipTest(f"{language} grammar is not locally installed")
                parsed = indexer.parse(source, language)
                self.assertEqual("parsed", parsed["status"])
                self.assertIn(expected_symbol, {item["name"] for item in parsed["symbols"]})
                if expected_call:
                    self.assertTrue(any(expected_call in item["calls"] for item in parsed["calls"]), parsed["calls"])
                if language in {"go", "java", "javascript", "python"}:
                    self.assertTrue(parsed["imports"], parsed["imports"])

    def test_scan_reindexes_only_changed_or_added_files_and_drops_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
            (root / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
            (root / "c.ts").write_text("export function gamma() { return 3; }\n", encoding="utf-8")
            scanner = RepositoryIntelligence(root)
            before = scanner.scan()
            self.assertEqual(3, before["index_metadata"]["files_parsed"])
            (root / "a.py").write_text("def alpha():\n    return 7\n", encoding="utf-8")
            (root / "b.py").unlink()
            (root / "d.rs").write_text("pub fn delta() -> i32 { 4 }\n", encoding="utf-8")
            after = scanner.scan(previous=before)
            self.assertEqual(1, after["index_metadata"]["files_reused"])
            self.assertEqual(2, after["index_metadata"]["files_parsed"])
            self.assertEqual({"a.py", "c.ts", "d.rs"}, {item["path"] for item in after["file_inventory"]})
            self.assertNotIn("b.py", after["file_index_cache"])
            alpha = next(item for item in after["symbols"] if item["name"] == "alpha")
            self.assertEqual(hashlib.sha256((root / "a.py").read_bytes()).hexdigest(), alpha["content_sha256"])

    def test_force_reindex_does_not_reuse_cached_parser_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("def alpha(): pass\n", encoding="utf-8")
            scanner = RepositoryIntelligence(root)
            first = scanner.scan()
            second = scanner.scan(previous=first, force_reindex=True)
            self.assertEqual(0, second["index_metadata"]["files_reused"])
            self.assertEqual(1, second["index_metadata"]["files_parsed"])

    def test_scip_index_requires_matching_root_safe_paths_and_current_source_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "defs.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            (root / "use.py").write_text("from defs import add\nadd(1, 2)\n", encoding="utf-8")
            symbol = "scip-python python 1.0 add()."
            definition = _scip_occurrence(symbol, 0, 4, 7, definition=True)
            reference = _scip_occurrence(symbol, 1, 0, 3)
            docs = [
                _scip_document("defs.py", occurrences=[definition], symbol_info=_scip_symbol_info(symbol, "add")),
                _scip_document("use.py", occurrences=[reference]),
            ]
            index_path = _write_scip(root, docs)
            with patch("openagent.codebase.scip_index.shutil.which", side_effect=lambda name: "C:/tools/scip-python.exe" if name == "scip-python" else None):
                loaded = read_scip_index(root, index_path)
            self.assertEqual("loaded", loaded["status"])
            self.assertEqual(2, len(loaded["occurrences"]))
            self.assertEqual(1, len(loaded["symbols"]))
            self.assertEqual("scip:scip-python@0.4.0", loaded["occurrences"][0]["provenance"])

            with patch("openagent.codebase.scip_index.shutil.which", side_effect=lambda name: "C:/tools/scip-python.exe" if name == "scip-python" else None):
                project_map = RepositoryIntelligence(root).scan()
            graph = build_code_knowledge_graph(project_map)
            reference_paths = related_code_paths(graph, ["use.py"])
            self.assertTrue(any(item["path"] == "defs.py" and item["relation"] == "symbol-reference-target" for item in reference_paths))

            # Source changed under an otherwise valid index; the old range no
            # longer identifies the symbol, so its occurrence is not trusted.
            (root / "use.py").write_text("from defs import add\nsub(1, 2)\n", encoding="utf-8")
            with patch("openagent.codebase.scip_index.shutil.which", side_effect=lambda name: "C:/tools/scip-python.exe" if name == "scip-python" else None):
                stale = read_scip_index(root, index_path)
            self.assertEqual(1, len(stale["occurrences"]))
            self.assertEqual("loaded", stale["status"])

            wrong_root = root.parent / (root.name + "-elsewhere")
            wrong_root.mkdir()
            wrong_index = wrong_root / "index.scip"
            wrong_index.write_bytes(index_path.read_bytes())
            with patch("openagent.codebase.scip_index.shutil.which", side_effect=lambda name: "C:/tools/scip-python.exe" if name == "scip-python" else None):
                wrong = read_scip_index(wrong_root, wrong_index)
            self.assertEqual("rejected", wrong["status"])
            self.assertIn("project_root", wrong["reason"])

    def test_scip_index_rejects_noncanonical_traversal_document_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _scip_document("../outside.py", occurrences=[])
            path = _write_scip(root, [payload])
            with patch("openagent.codebase.scip_index.shutil.which", side_effect=lambda name: "C:/tools/scip-python.exe" if name == "scip-python" else None):
                parsed = read_scip_index(root, path)
            self.assertEqual("loaded-no-source-validated-occurrences", parsed["status"])
            self.assertEqual(["../outside.py"], parsed["rejected_documents"])
            self.assertEqual({}, parsed["file_hashes"])

    def test_scip_gzip_and_document_resource_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "main.py"
            source.write_text("def add(): pass\n", encoding="utf-8")
            symbol = "scip-python python 1.0 add()."
            index_path = _write_scip(root, [
                _scip_document("main.py", occurrences=[_scip_occurrence(symbol, 0, 4, 7, definition=True)], symbol_info=_scip_symbol_info(symbol, "add")),
            ])
            index_path.write_bytes(gzip.compress(index_path.read_bytes()))
            with patch("openagent.codebase.scip_index.shutil.which", side_effect=lambda name: "C:/tools/scip-python.exe" if name == "scip-python" else None):
                loaded = read_scip_index(root, index_path)
            self.assertEqual("loaded", loaded["status"])
            self.assertEqual(1, len(loaded["symbols"]))

            # Keep the test small while exercising the fail-closed resource guard.
            index_path.write_bytes(_write_scip(root, [
                _scip_document("main.py", occurrences=[
                    _scip_occurrence(symbol, 0, 4, 7, definition=True),
                    _scip_occurrence(symbol, 0, 4, 7, definition=True),
                ], symbol_info=_scip_symbol_info(symbol, "add")),
            ]).read_bytes())
            with patch("openagent.codebase.scip_index.MAX_OCCURRENCES_PER_DOCUMENT", 1), patch(
                "openagent.codebase.scip_index.shutil.which", side_effect=lambda name: "C:/tools/scip-python.exe" if name == "scip-python" else None,
            ):
                bounded = read_scip_index(root, index_path)
            self.assertEqual("rejected", bounded["status"])
            self.assertIn("document occurrence count", bounded["reason"])

    def test_scip_unspecified_encoding_only_accepts_unambiguous_ascii_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain.py").write_text("def add(): pass\n", encoding="utf-8")
            (root / "unicode.py").write_text("# 🚀 add\n", encoding="utf-8")
            symbol = "scip-python python 1.0 add()."
            documents = [
                _scip_document("plain.py", occurrences=[_scip_occurrence(symbol, 0, 4, 7, definition=True)], symbol_info=_scip_symbol_info(symbol, "add"), encoding=0),
                _scip_document("unicode.py", occurrences=[_scip_occurrence(symbol, 0, 4, 7, definition=True)], symbol_info=_scip_symbol_info(symbol, "add"), encoding=0),
            ]
            index_path = _write_scip(root, documents)
            with patch("openagent.codebase.scip_index.shutil.which", side_effect=lambda name: "C:/tools/scip-python.exe" if name == "scip-python" else None):
                parsed = read_scip_index(root, index_path)
            self.assertEqual(["plain.py"], [item["path"] for item in parsed["symbols"]])

    def test_old_graph_neighbors_are_refreshed_when_an_import_is_removed(self) -> None:
        before_map = {
            "root": "/repo",
            "file_inventory": [
                {"path": "caller.py", "content_sha256": "1", "language": "python"},
                {"path": "target.py", "content_sha256": "2", "language": "python"},
            ],
            "imports": {"caller.py": ["target"]},
        }
        after_map = {
            "root": "/repo",
            "file_inventory": [
                {"path": "caller.py", "content_sha256": "3", "language": "python"},
                {"path": "target.py", "content_sha256": "2", "language": "python"},
            ],
            "imports": {"caller.py": []},
        }
        before = build_code_knowledge_graph(before_map)
        after = build_code_knowledge_graph(after_map)
        scope = graph_store.refresh_scope(before, after)
        self.assertEqual(["target.py"], scope["neighbour_paths"])

    def test_tree_sitter_call_provenance_flows_into_cross_file_graph_traversal(self) -> None:
        project_map = {
            "root": "/repo",
            "file_inventory": [
                {"path": "src/view.ts", "content_sha256": "a" * 64, "index_status": "indexed"},
                {"path": "src/format.ts", "content_sha256": "b" * 64, "index_status": "indexed"},
            ],
            "symbols": [
                {"name": "render", "kind": "function_declaration", "path": "src/view.ts", "line": 1, "content_sha256": "a" * 64, "provenance": "tree-sitter:typescript@fixture"},
                {"name": "formatValue", "kind": "function_declaration", "path": "src/format.ts", "line": 1, "content_sha256": "b" * 64, "provenance": "tree-sitter:typescript@fixture"},
            ],
            "call_graph": [{
                "file": "src/view.ts", "symbol": "render", "calls": ["formatValue"],
                "provenance": "tree-sitter-call-syntax:typescript@fixture",
            }],
            "imports": {}, "routes": [], "test_files": [],
        }
        graph = build_code_knowledge_graph(project_map)
        related = related_code_paths(graph, ["src/view.ts"])
        edge = next(item for item in related if item["path"] == "src/format.ts" and item["relation"] == "selected-file-calls")
        self.assertEqual("tree-sitter-call-syntax:typescript@fixture", edge["provenance"])
        self.assertEqual("syntax-call-unique-symbol-name", edge["confidence"])

    def test_incremental_cache_is_project_owner_and_workspace_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = OpenAgentStore(root / "data")
            service = CodebaseService(store, object())  # type: ignore[arg-type]
            workspace = root / "workspace"
            workspace.mkdir()
            project_map = {"root": str(workspace.resolve()), "file_index_cache": {"a.py": {"content_sha256": "x"}}}
            service._save_index_cache("p1", "alice", str(workspace), project_map)
            self.assertEqual(project_map, service._load_index_cache("p1", "alice", str(workspace)))
            self.assertIsNone(service._load_index_cache("p1", "bob", str(workspace)))
            self.assertIsNone(service._load_index_cache("p2", "alice", str(workspace)))
            self.assertIsNone(service._load_index_cache("p1", "alice", str(root)))


if __name__ == "__main__":
    unittest.main()
