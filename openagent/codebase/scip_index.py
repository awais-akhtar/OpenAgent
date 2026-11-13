"""Bounded, source-revalidated reader for existing SCIP protobuf indexes.

CodeQ does not run language indexer commands automatically. It reads a local
``index.scip`` only when present, validates the declared project root and
indexer identity, checks every document path, and re-validates indexed ranges
against the current source before emitting any symbol/reference records.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse


MAX_INDEX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_DOCUMENTS = 50_000
MAX_OCCURRENCES_PER_DOCUMENT = 50_000
MAX_OCCURRENCES = 100_000
MAX_SYMBOLS_PER_DOCUMENT = 20_000
MAX_SYMBOLS = 200_000
_IDENTIFIERS = re.compile(r"[A-Za-z_$][\w$]*")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SCIPIndexError(ValueError):
    pass


def _read_varint_stream(stream: BinaryIO) -> int | None:
    result = 0
    shift = 0
    for _ in range(10):
        chunk = stream.read(1)
        if not chunk:
            if shift == 0:
                return None
            raise SCIPIndexError("truncated protobuf varint")
        byte = chunk[0]
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result
        shift += 7
    raise SCIPIndexError("protobuf varint exceeds 64 bits")


def _read_varint_bytes(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(data):
            raise SCIPIndexError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise SCIPIndexError("protobuf varint exceeds 64 bits")


def _fields(data: bytes):
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint_bytes(data, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise SCIPIndexError("invalid protobuf field number")
        if wire == 0:
            value, offset = _read_varint_bytes(data, offset)
        elif wire == 1:
            if offset + 8 > len(data):
                raise SCIPIndexError("truncated fixed64 protobuf field")
            value, offset = data[offset:offset + 8], offset + 8
        elif wire == 2:
            length, offset = _read_varint_bytes(data, offset)
            if length > MAX_MESSAGE_BYTES or offset + length > len(data):
                raise SCIPIndexError("oversized or truncated protobuf field")
            value, offset = data[offset:offset + length], offset + length
        elif wire == 5:
            if offset + 4 > len(data):
                raise SCIPIndexError("truncated fixed32 protobuf field")
            value, offset = data[offset:offset + 4], offset + 4
        else:
            raise SCIPIndexError("unsupported protobuf wire type")
        yield number, wire, value


def _text(value: bytes) -> str:
    return value.decode("utf-8", errors="strict")


def _parse_info(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"symbol": "", "display_name": "", "relationships": []}
    for number, wire, value in _fields(data):
        if wire != 2:
            continue
        if number == 1:
            result["symbol"] = _text(value)
        elif number == 6:
            result["display_name"] = _text(value)
        elif number == 4:
            relation: dict[str, Any] = {"symbol": "", "is_reference": False, "is_implementation": False, "is_definition": False}
            for r_number, r_wire, r_value in _fields(value):
                if r_number == 1 and r_wire == 2:
                    relation["symbol"] = _text(r_value)
                elif r_number in {2, 3, 5} and r_wire == 0:
                    relation[{2: "is_reference", 3: "is_implementation", 5: "is_definition"}[r_number]] = bool(r_value)
            if relation["symbol"]:
                result["relationships"].append(relation)
    return result


def _parse_range(data: bytes) -> tuple[int, int, int, int] | None:
    values: dict[int, int] = {}
    for number, wire, value in _fields(data):
        if wire == 0 and number in {1, 2, 3, 4}:
            values[number] = int(value)
    if {1, 2, 3, 4}.issubset(values):
        return values[1], values[2], values[3], values[4]
    return None


def _parse_single_line_range(data: bytes) -> tuple[int, int, int, int] | None:
    values: dict[int, int] = {}
    for number, wire, value in _fields(data):
        if wire == 0 and number in {1, 2, 3}:
            values[number] = int(value)
    if 1 in values and 2 in values and 3 in values:
        return values[1], values[2], values[1], values[3]
    return None


def _packed_ints(data: bytes) -> list[int]:
    values: list[int] = []
    offset = 0
    while offset < len(data):
        value, offset = _read_varint_bytes(data, offset)
        values.append(value)
    return values


def _parse_occurrence(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"symbol": "", "roles": 0, "range": None}
    typed_range: tuple[int, int, int, int] | None = None
    legacy_range: list[int] = []
    for number, wire, value in _fields(data):
        if number == 1 and wire == 2:
            legacy_range = _packed_ints(value)
        elif number == 1 and wire == 0:
            legacy_range.append(int(value))
        elif number == 2 and wire == 2:
            result["symbol"] = _text(value)
        elif number == 3 and wire == 0:
            result["roles"] = int(value)
        elif number in {8, 9} and wire == 2:
            typed_range = _parse_single_line_range(value) if number == 8 else _parse_range(value)
    if typed_range is not None:
        result["range"] = typed_range
    elif legacy_range:
        values = legacy_range
        if len(values) == 3:
            result["range"] = (values[0], values[1], values[0], values[2])
        elif len(values) == 4:
            result["range"] = tuple(values)
    return result


def _parse_document(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"path": "", "language": "", "text": None, "position_encoding": 0, "symbols": [], "occurrences": []}
    for number, wire, value in _fields(data):
        if number in {1, 4, 5} and wire == 2:
            if number == 1:
                result["path"] = _text(value)
            elif number == 4:
                result["language"] = _text(value)
            else:
                result["text"] = _text(value)
        elif number == 2 and wire == 2:
            result["occurrences"].append(_parse_occurrence(value))
            if len(result["occurrences"]) > MAX_OCCURRENCES_PER_DOCUMENT:
                raise SCIPIndexError("SCIP document occurrence count exceeds the configured bound")
        elif number == 3 and wire == 2:
            result["symbols"].append(_parse_info(value))
            if len(result["symbols"]) > MAX_SYMBOLS_PER_DOCUMENT:
                raise SCIPIndexError("SCIP document symbol count exceeds the configured bound")
        elif number == 6 and wire == 0:
            result["position_encoding"] = int(value)
    return result


def _parse_tool(data: bytes) -> dict[str, Any]:
    result = {"name": "", "version": ""}
    for number, wire, value in _fields(data):
        if number in {1, 2} and wire == 2:
            result[{1: "name", 2: "version"}[number]] = _text(value)
    return result


def _parse_metadata(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"project_root": "", "tool": {"name": "", "version": ""}, "protocol_version": 0}
    for number, wire, value in _fields(data):
        if number == 2 and wire == 2:
            result["tool"] = _parse_tool(value)
        elif number == 3 and wire == 2:
            result["project_root"] = _text(value)
        elif number == 1 and wire == 0:
            result["protocol_version"] = int(value)
    return result


class _LimitedReader:
    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self.stream = stream
        self.limit = limit
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = min(1024 * 1024, self.limit - self.count + 1)
        size = min(size, self.limit - self.count + 1)
        chunk = self.stream.read(size)
        self.count += len(chunk)
        if self.count > self.limit:
            raise SCIPIndexError("SCIP index exceeds the uncompressed size limit")
        return chunk

    def skip(self, count: int) -> None:
        remaining = count
        while remaining:
            chunk = self.read(min(1024 * 1024, remaining))
            if not chunk:
                raise SCIPIndexError("truncated SCIP index")
            remaining -= len(chunk)


def _read_exact(stream: _LimitedReader, count: int) -> bytes:
    if count > MAX_MESSAGE_BYTES:
        raise SCIPIndexError("SCIP metadata or document exceeds the message size limit")
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(count - len(chunks))
        if not chunk:
            raise SCIPIndexError("truncated SCIP index")
        chunks.extend(chunk)
    return bytes(chunks)


def _root_from_uri(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        raise SCIPIndexError("SCIP project_root must be a local file URI or path")
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
        if len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path).resolve()
    return Path(value).expanduser().resolve()


def _line_slice(lines: list[str], span: tuple[int, int, int, int], encoding: int) -> str | None:
    start_line, start_col, end_line, end_col = span
    if min(start_line, start_col, end_line, end_col) < 0 or end_col < start_col or end_line < start_line or end_line >= len(lines) or start_line >= len(lines):
        return None
    if start_line != end_line:
        return None  # identifiers/references must be on one line for exact validation
    line = lines[start_line]
    if encoding == 0:
        # Legacy SCIP producers may leave the encoding field at its protobuf
        # default. For ASCII-only lines, byte/UTF-16/UTF-32 offsets coincide;
        # on any non-ASCII line the encoding is ambiguous and is rejected.
        if not line.isascii():
            return None
        return line[start_col:end_col]
    if encoding == 1:  # UTF-8 bytes
        raw = line.encode("utf-8")
        try:
            return raw[start_col:end_col].decode("utf-8")
        except UnicodeDecodeError:
            return None
    if encoding == 2:  # UTF-16 code units
        units = line.encode("utf-16-le")
        try:
            return units[start_col * 2:end_col * 2].decode("utf-16-le")
        except UnicodeDecodeError:
            return None
    if encoding == 3:  # UTF-32 code points
        return line[start_col:end_col]
    return None  # unspecified encoding cannot be trusted for range validation


def _symbol_leaf(value: str) -> str:
    value = value.strip()
    match = re.search(r"([A-Za-z_$][\w$-]*)(?:\([^)]*\))?[.#:]?$", value)
    return match.group(1) if match else ""


def _indexers_on_path() -> list[dict[str, str]]:
    names = ("scip-python", "scip-typescript", "scip-dotnet", "scip-java", "scip-go", "scip-rust", "scip")
    return [{"name": name, "path": found} for name in names if (found := shutil.which(name))]


def read_scip_index(root: str | Path, index_path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate the standard local SCIP ``Index`` protobuf payload."""
    project_root = Path(root).expanduser().resolve()
    path = Path(index_path).expanduser().resolve() if index_path else project_root / "index.scip"
    result: dict[str, Any] = {
        "status": "not-present",
        "index_path": str(path),
        "indexers_on_path": _indexers_on_path(),
        "index_sha256": "",
        "tool": {"name": "", "version": ""},
        "documents": [], "symbols": [], "occurrences": [], "diagnostics": [],
    }
    try:
        path.relative_to(project_root)
    except ValueError:
        result.update(status="rejected", reason="SCIP index file must remain inside the selected CodeQ workspace")
        return result
    cursor = project_root
    for part in path.relative_to(project_root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            result.update(status="rejected", reason="SCIP index path must not traverse symbolic links")
            return result
    if not path.exists():
        return result
    if path.is_symlink() or not path.is_file():
        result.update(status="rejected", reason="SCIP index must be a regular, non-symlink file")
        return result
    try:
        if path.stat().st_size > MAX_INDEX_UNCOMPRESSED_BYTES:
            result.update(status="rejected", reason="SCIP index exceeds the configured size limit")
            return result
        digest = _sha256_file(path)
        with path.open("rb") as raw_stream:
            prefix = raw_stream.read(2)
            raw_stream.seek(0)
            stream: BinaryIO = gzip.GzipFile(fileobj=raw_stream) if prefix == b"\x1f\x8b" else raw_stream
            limited = _LimitedReader(stream, MAX_INDEX_UNCOMPRESSED_BYTES)
            metadata: dict[str, Any] | None = None
            documents: list[dict[str, Any]] = []
            total_occurrences = 0
            total_symbols = 0
            while True:
                tag = _read_varint_stream(limited)
                if tag is None:
                    break
                number, wire = tag >> 3, tag & 7
                if number == 0:
                    raise SCIPIndexError("invalid SCIP field tag")
                if wire == 2:
                    length = _read_varint_stream(limited)
                    if length is None:
                        raise SCIPIndexError("truncated SCIP field length")
                    if number in {1, 2}:
                        payload = _read_exact(limited, length)
                        if number == 1:
                            if metadata is not None:
                                raise SCIPIndexError("SCIP index contains repeated metadata")
                            metadata = _parse_metadata(payload)
                        else:
                            document = _parse_document(payload)
                            total_occurrences += len(document["occurrences"])
                            total_symbols += len(document["symbols"])
                            documents.append(document)
                            if len(documents) > MAX_DOCUMENTS:
                                raise SCIPIndexError("SCIP document count exceeds the configured bound")
                            if total_occurrences > MAX_OCCURRENCES:
                                raise SCIPIndexError("SCIP total occurrence count exceeds the configured bound")
                            if total_symbols > MAX_SYMBOLS:
                                raise SCIPIndexError("SCIP total symbol count exceeds the configured bound")
                    else:
                        limited.skip(length)
                elif wire == 0:
                    if _read_varint_stream(limited) is None:
                        raise SCIPIndexError("truncated SCIP varint field")
                elif wire == 1:
                    limited.skip(8)
                elif wire == 5:
                    limited.skip(4)
                else:
                    raise SCIPIndexError("unsupported SCIP protobuf wire type")
        if metadata is None:
            raise SCIPIndexError("SCIP index has no metadata header")
        index_root = _root_from_uri(str(metadata.get("project_root", "")))
        tool = metadata.get("tool", {})
        if index_root != project_root:
            raise SCIPIndexError("SCIP project_root does not match the selected CodeQ workspace")
        if int(metadata.get("protocol_version", 0)) != 0:
            raise SCIPIndexError("SCIP protocol version is not supported by this reader")
        if not str(tool.get("name", "")).strip() or not str(tool.get("version", "")).strip():
            raise SCIPIndexError("SCIP index is missing indexer name or version")
        available_indexers = _indexers_on_path()
        if not any(item["name"] == str(tool["name"]) for item in available_indexers):
            raise SCIPIndexError(f"SCIP indexer '{tool['name']}' is not installed on PATH")

        symbol_info: dict[str, dict[str, Any]] = {}
        for doc in documents:
            for info in doc["symbols"]:
                if info.get("symbol"):
                    symbol_info[str(info["symbol"])] = info
        verified_symbols: list[dict[str, Any]] = []
        verified_occurrences: list[dict[str, Any]] = []
        file_hashes: dict[str, str] = {}
        rejected_documents: list[str] = []
        for doc in documents:
            rel = str(doc.get("path", ""))
            parts = PurePosixPath(rel).parts
            raw_parts = rel.split("/")
            if (not rel or "\\" in rel or rel.startswith("/") or "//" in rel
                    or PurePosixPath(rel).is_absolute() or re.match(r"^[A-Za-z]:", rel)
                    or any(part in {"", ".", ".."} for part in raw_parts)):
                rejected_documents.append(rel or "<empty>")
                continue
            source = project_root.joinpath(*parts)
            try:
                cursor = project_root
                has_symlink_parent = False
                for part in parts:
                    cursor = cursor / part
                    if cursor.is_symlink():
                        has_symlink_parent = True
                        break
                if has_symlink_parent or not source.is_file() or project_root not in source.resolve().parents:
                    rejected_documents.append(rel)
                    continue
                raw_source = source.read_bytes()
            except OSError:
                rejected_documents.append(rel)
                continue
            source_hash = hashlib.sha256(raw_source).hexdigest()
            try:
                current_text = raw_source.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                rejected_documents.append(rel)
                continue
            if doc.get("text") is not None and doc["text"] != current_text:
                rejected_documents.append(rel)
                continue
            file_hashes[rel] = source_hash
            lines = current_text.splitlines()
            for occurrence in doc["occurrences"]:
                symbol = str(occurrence.get("symbol", ""))
                span = occurrence.get("range")
                if not symbol or not span:
                    continue
                info = symbol_info.get(symbol, {})
                display = str(info.get("display_name", "")).strip()
                expected = display or _symbol_leaf(symbol)
                excerpt = _line_slice(lines, span, int(doc.get("position_encoding", 0)))
                # Only source ranges which still identify the expected name are
                # admitted. Unspecified encodings and stale/moved spans are skipped.
                if not expected or excerpt != expected:
                    continue
                row = {
                    "symbol": symbol,
                    "name": expected,
                    "path": rel,
                    "line": int(span[0]) + 1,
                    "is_definition": bool(int(occurrence.get("roles", 0)) & 1),
                    "is_reference": not bool(int(occurrence.get("roles", 0)) & 1),
                    "content_sha256": source_hash,
                    "index_sha256": digest,
                    "provenance": f"scip:{tool['name']}@{tool['version']}",
                    "confidence": "source-range-validated",
                    "relationships": info.get("relationships", []),
                }
                verified_occurrences.append(row)
                if row["is_definition"]:
                    verified_symbols.append({
                        "name": expected,
                        "kind": "scip-symbol",
                        "path": rel,
                        "line": row["line"],
                        "content_sha256": source_hash,
                        "provenance": row["provenance"],
                        "stable_symbol_id": symbol,
                        "relationships": info.get("relationships", []),
                    })
        for rel, expected_hash in list(file_hashes.items()):
            if _sha256_file(project_root.joinpath(*PurePosixPath(rel).parts)) != expected_hash:
                file_hashes.pop(rel, None)
                verified_symbols = [item for item in verified_symbols if item["path"] != rel]
                verified_occurrences = [item for item in verified_occurrences if item["path"] != rel]
                rejected_documents.append(rel)
        if _sha256_file(path) != digest:
            raise SCIPIndexError("SCIP index changed while it was being validated")
        result.update({
            "status": "loaded" if verified_occurrences else "loaded-no-source-validated-occurrences",
            "index_sha256": digest,
            "tool": tool,
            "protocol_version": int(metadata.get("protocol_version", 0)),
            "document_count": len(documents),
            "validated_document_count": len(file_hashes),
            "rejected_documents": sorted(set(rejected_documents))[:100],
            "file_hashes": file_hashes,
            "symbols": verified_symbols[:200_000],
            "occurrences": verified_occurrences[:1_000_000],
            "diagnostics": [],
        })
        return result
    except (OSError, EOFError, UnicodeDecodeError, SCIPIndexError, gzip.BadGzipFile) as exc:
        result.update(status="rejected", reason=str(exc), symbols=[], occurrences=[], file_hashes={})
        return result
