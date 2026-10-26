"""Static consistency audit for the native public ABI source boundary.

This audit does not load a shared library or prove ABI compatibility at
runtime. It catches stale ctypes symbol names and exported header declarations
without a corresponding C implementation before build/evidence gates run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_EXPORTED_DECLARATION = re.compile(
    r"(?m)^[ \t]*OA_API\s+([^;{}]*);", re.DOTALL
)
_CTYPES_SYMBOL = re.compile(
    # Native functions are commonly reached through nested owners such as
    # ``self.kernel`` or ``self.native.library``. Matching only variables
    # literally named ``library``/``lib`` silently omitted those ABI edges.
    r"\b(?:[A-Za-z_]\w*\.)+((?:oa|coli)_[A-Za-z0-9_]+)\b"
)


@dataclass(frozen=True)
class PublicABIAudit:
    exported_symbols: tuple[str, ...]
    ctypes_symbols: tuple[str, ...]
    missing_export_definitions: tuple[str, ...]
    missing_ctypes_sources: tuple[str, ...]
    duplicate_export_definitions: tuple[str, ...]
    mismatched_export_parameters: tuple[str, ...]
    mismatched_export_types: tuple[str, ...]
    mismatched_calling_conventions: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing_export_definitions or
                    self.missing_ctypes_sources or
                    self.duplicate_export_definitions or
                    self.mismatched_export_parameters or
                    self.mismatched_export_types or
                    self.mismatched_calling_conventions)

    def require_ok(self) -> "PublicABIAudit":
        if not self.ok:
            details = []
            if self.missing_export_definitions:
                details.append("export declarations without one C definition: " +
                               ", ".join(self.missing_export_definitions))
            if self.missing_ctypes_sources:
                details.append("ctypes symbols without native source: " +
                               ", ".join(self.missing_ctypes_sources))
            if self.duplicate_export_definitions:
                details.append("export symbols with multiple C definitions: " +
                               ", ".join(self.duplicate_export_definitions))
            if self.mismatched_export_parameters:
                details.append("export declarations and C definitions with different parameter counts: " +
                               ", ".join(self.mismatched_export_parameters))
            if self.mismatched_export_types:
                details.append("export declarations and C definitions with different return or parameter types: " +
                               ", ".join(self.mismatched_export_types))
            if self.mismatched_calling_conventions:
                details.append("export declarations and C definitions with different calling conventions: " +
                               ", ".join(self.mismatched_calling_conventions))
            raise ValueError("public ABI source audit failed: " + "; ".join(details))
        return self


def _matching_parenthesis(source: str, opening: int) -> int | None:
    """Find a C parameter-list terminator while honoring nested declarators."""
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == '(':
            depth += 1
        elif source[index] == ')':
            depth -= 1
            if depth == 0:
                return index
    return None


def _parameter_count(parameters: str) -> int | None:
    """Count top-level parameters, including callback declarators with commas."""
    value = parameters.strip()
    if not value or value == 'void':
        return 0
    depth = 0
    brackets = 0
    count = 1
    for character in value:
        if character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
            if depth < 0:
                return None
        elif character == '[':
            brackets += 1
        elif character == ']':
            brackets -= 1
            if brackets < 0:
                return None
        elif character == ',' and depth == 0 and brackets == 0:
            count += 1
    return count if depth == 0 and brackets == 0 else None


def _split_parameters(parameters: str) -> tuple[str, ...] | None:
    """Split top-level C parameters without splitting callback signatures."""
    value = parameters.strip()
    if not value or value == 'void':
        return ()
    depth = 0
    brackets = 0
    start = 0
    parts = []
    for index, character in enumerate(value):
        if character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
        elif character == '[':
            brackets += 1
        elif character == ']':
            brackets -= 1
        elif character == ',' and depth == 0 and brackets == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    if depth != 0 or brackets != 0:
        return None
    parts.append(value[start:].strip())
    return tuple(parts) if all(parts) else None


_C_QUALIFIERS = {'const', 'volatile', 'restrict', '_Atomic'}
_C_BUILTIN_TYPES = {
    'void', 'char', 'short', 'int', 'long', 'float', 'double', 'signed',
    'unsigned', '_Bool', '_Complex',
}


def _normalize_parameter_type(parameter: str) -> str | None:
    """Remove declarator names while preserving ABI-relevant type spelling."""
    value = re.sub(r"/\*.*?\*/|//[^\r\n]*", " ", parameter, flags=re.DOTALL).strip()
    if not value:
        return None
    # Function pointer parameter names occur inside the nested declarator.
    value = re.sub(r"\(\s*(\*+)\s*[A-Za-z_]\w*\s*\)", r"(\1)", value)
    # Array parameters are adjusted to pointers by C. Preserve the extent as
    # part of the source contract while removing only the declarator name.
    value = re.sub(r"\b([A-Za-z_]\w*)\s*(\[[^\]]*\])", r"\2", value)
    identifiers = list(re.finditer(r"[A-Za-z_]\w*", value))
    if not identifiers:
        return None
    # Only a top-level trailing identifier can be the parameter name. Names
    # inside function-pointer callback types are not candidates.
    top_level = []
    depth = 0
    for item in identifiers:
        depth = 0
        for character in value[:item.start()]:
            if character == '(':
                depth += 1
            elif character == ')':
                depth -= 1
        if depth == 0:
            top_level.append(item)
    last = top_level[-1] if top_level else None
    if last is not None:
        words = [item.group(0) for item in top_level]
        last_word = last.group(0)
        preceding = words[:-1]
        preceding_types = [word for word in preceding
                           if word not in _C_QUALIFIERS and
                           word not in ('struct', 'union', 'enum')]
        # A lone typedef name (or struct tag) is a type. Remove only a final
        # identifier following a recognizable type or pointer declarator.
        struct_tag_only = (len(words) == 2 and
                           words[0] in ('struct', 'union', 'enum'))
        if (not value[last.end():].strip() and not struct_tag_only and
                last_word not in _C_QUALIFIERS and
                ('*' in value or preceding_types or
                 any(word in _C_BUILTIN_TYPES for word in preceding))):
            value = value[:last.start()] + value[last.end():]
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*\*\s*", " *", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_return_type(value: str) -> str:
    value = re.sub(r"\b(?:OA_API|OA_CALL|extern)\b", " ", value)
    value = re.sub(r"/\*.*?\*/|//[^\r\n]*", " ", value, flags=re.DOTALL)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*\*\s*", " *", value)
    return re.sub(r"\s+", " ", value).strip()


def _extract_signature(source: str, symbol: str, *, definition: bool) -> tuple[str, tuple[str, ...]] | None:
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\s*\(")
    signatures = []
    for occurrence in pattern.finditer(source):
        opening = source.find('(', occurrence.start())
        closing = _matching_parenthesis(source, opening)
        if closing is None:
            continue
        if definition:
            after = closing + 1
            while after < len(source) and source[after].isspace():
                after += 1
            if after >= len(source) or source[after] != '{':
                continue
        params = _split_parameters(source[opening + 1:closing])
        if params is None:
            continue
        normalized = tuple(_normalize_parameter_type(item) for item in params)
        if any(item is None for item in normalized):
            continue
        prefix = source[:occurrence.start()]
        if definition:
            boundary = max(prefix.rfind(';'), prefix.rfind('}'), prefix.rfind('{'))
            prefix = prefix[boundary + 1:]
            prefix = re.sub(r"(?m)^\s*#.*$", " ", prefix)
            prefix = re.sub(r"(?m)^.*\bOA_PACKED_EXPORT\b.*$", " ", prefix)
        result_type = _normalize_return_type(prefix)
        signatures.append((result_type, normalized))
    if not signatures:
        return None
    return signatures[0]


def _declared_parameter_count(source: str, symbol: str) -> int | None:
    occurrence = re.search(r"\b" + re.escape(symbol) + r"\s*\(", source)
    if occurrence is None:
        return None
    opening = source.find('(', occurrence.start())
    closing = _matching_parenthesis(source, opening)
    if closing is None:
        return None
    return _parameter_count(source[opening + 1:closing])


def _definition_parameter_counts(source: str, symbol: str) -> tuple[int, ...]:
    counts = []
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\s*\(")
    for occurrence in pattern.finditer(source):
        opening = source.find('(', occurrence.start())
        closing = _matching_parenthesis(source, opening)
        if closing is None:
            continue
        after = closing + 1
        while after < len(source) and source[after].isspace():
            after += 1
        if after < len(source) and source[after] == '{':
            count = _parameter_count(source[opening + 1:closing])
            if count is not None:
                counts.append(count)
    return tuple(counts)


_CALLING_CONVENTIONS = r"OA_CALL|__cdecl|__stdcall|WINAPI|CALLBACK"


def _calling_convention(
    source: str, symbol: str, *, definition: bool = False
) -> str | None:
    """Return the explicit convention immediately preceding a function name.

    ``None`` represents the platform's default C convention. Do not erase
    ``OA_CALL`` during signature normalization: on Windows it expands to
    ``__cdecl``, which is ABI-significant for callbacks and exported functions.
    """
    pattern = re.compile(
        r"\b(?:(?P<convention>" + _CALLING_CONVENTIONS + r")\s+)?" +
        re.escape(symbol) + r"\s*\("
    )
    for match in pattern.finditer(source):
        opening = source.find("(", match.start())
        closing = _matching_parenthesis(source, opening)
        if closing is None:
            continue
        if definition:
            after = closing + 1
            while after < len(source) and source[after].isspace():
                after += 1
            if after >= len(source) or source[after] != "{":
                continue
        return match.group("convention")
    return None


def audit_public_abi(root: str | Path | None = None) -> PublicABIAudit:
    """Reconcile exported C headers, C definitions and Python ctypes symbols.

    ``root`` may be the repository root, ``openagent`` directory, or the
    ``adaptive_engine`` source directory. Only source text is read; nothing is
    imported, compiled, or loaded.
    """
    base = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    if (base / "openagent" / "adaptive_engine").is_dir():
        base = base / "openagent" / "adaptive_engine"
    elif (base / "adaptive_engine").is_dir():
        base = base / "adaptive_engine"
    elif base.name != "adaptive_engine":
        raise ValueError("root must identify the repository, openagent, or adaptive_engine")

    native = base / "native"
    headers = sorted(native.rglob("*.h"))
    c_sources = sorted(base.rglob("*.c"))
    py_sources = sorted(base.rglob("*.py"))
    if not headers or not c_sources or not py_sources:
        raise ValueError("native ABI headers, C sources and Python adapters are required")

    header_text = {path: path.read_text(encoding="utf-8") for path in headers}
    c_text = {path: path.read_text(encoding="utf-8") for path in c_sources}
    py_text = {path: path.read_text(encoding="utf-8") for path in py_sources}

    declarations: dict[str, list[Path]] = {}
    declaration_counts: dict[str, list[int]] = {}
    declaration_signatures: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    declaration_conventions: dict[str, list[str | None]] = {}
    for path, source in header_text.items():
        if path.name == "native_exports.h":
            continue  # Export/calling-convention macros are not declarations.
        for match in _EXPORTED_DECLARATION.finditer(source):
            prefix = re.sub(r"\bOA_CALL\b", "", match.group(1))
            opening = prefix.find("(")
            if opening < 0:
                continue
            names = re.findall(r"[A-Za-z_]\w*", prefix[:opening])
            if names:
                declarations.setdefault(names[-1], []).append(path)
                count = _declared_parameter_count(match.group(1), names[-1])
                if count is not None:
                    declaration_counts.setdefault(names[-1], []).append(count)
                signature = _extract_signature(match.group(1), names[-1], definition=False)
                if signature is not None:
                    declaration_signatures.setdefault(names[-1], []).append(signature)
                declaration_conventions.setdefault(names[-1], []).append(
                    _calling_convention(match.group(1), names[-1])
                )

    definitions: dict[str, list[Path]] = {}
    for symbol in declarations:
        pattern = re.compile(
            r"(?m)^[ \t]*(?:[A-Za-z_]\w*[ \t\*]+)+" + re.escape(symbol) +
            r"\s*\([^;{}]*\)\s*\{", re.DOTALL)
        for path, source in c_text.items():
            if pattern.search(source):
                definitions.setdefault(symbol, []).append(path)

    # The packed executor declarations are generated from one C macro with a
    # format suffix; their concrete names cannot occur in ordinary definition
    # syntax, so validate both generator families as a unit.
    generated_executor_templates = (
        "oa_executor_swiglu_##suffix", "oa_executor_apply_##suffix")
    all_c_text = "\n".join(c_text.values())
    generated = {symbol for symbol in declarations
                 if symbol.startswith(("oa_executor_swiglu_", "oa_executor_apply_"))}
    if generated and all(template in all_c_text for template in generated_executor_templates):
        for symbol in generated:
            definitions.setdefault(symbol, []).append(native / "segment_executor.c")

    ctypes_symbols = set()
    for source in py_text.values():
        ctypes_symbols.update(_CTYPES_SYMBOL.findall(source))
    native_text = "\n".join((*header_text.values(), *c_text.values()))
    missing_ctypes = sorted(symbol for symbol in ctypes_symbols
                            if not re.search(r"\b" + re.escape(symbol) + r"\b", native_text))
    missing_exports = sorted(symbol for symbol in declarations
                             if len(definitions.get(symbol, ())) == 0)
    duplicate_exports = sorted(symbol for symbol, paths in definitions.items()
                               if len(paths) > 1)
    mismatched_parameters = []
    mismatched_types = []
    mismatched_calling_conventions = []
    for symbol, declared in declaration_counts.items():
        if symbol in generated:
            continue  # Concrete entries are emitted by one shared C macro.
        implemented_signatures = tuple(signature for source in c_text.values()
            if (signature := _extract_signature(source, symbol, definition=True)) is not None)
        implemented = tuple(len(signature[1]) for signature in implemented_signatures)
        if implemented and (len(set(declared)) != 1 or
                            any(count != declared[0] for count in implemented)):
            mismatched_parameters.append(symbol)
        declared_signatures = declaration_signatures.get(symbol, ())
        if (implemented_signatures and declared_signatures and
                (len(set(declared_signatures)) != 1 or
                 any(signature != declared_signatures[0]
                     for signature in implemented_signatures))):
            mismatched_types.append(symbol)
        declared_conventions = declaration_conventions.get(symbol, ())
        implementation_conventions = tuple(
            _calling_convention(source, symbol, definition=True)
            for source in c_text.values()
            if _extract_signature(source, symbol, definition=True) is not None
        )
        if (declared_conventions and implementation_conventions and
                (len(set(declared_conventions)) != 1 or
                 any(convention != declared_conventions[0]
                     for convention in implementation_conventions))):
            mismatched_calling_conventions.append(symbol)
    return PublicABIAudit(
        tuple(sorted(declarations)), tuple(sorted(ctypes_symbols)),
        tuple(missing_exports), tuple(missing_ctypes), tuple(duplicate_exports),
        tuple(sorted(mismatched_parameters)), tuple(sorted(mismatched_types)),
        tuple(sorted(mismatched_calling_conventions)))


__all__ = ["PublicABIAudit", "audit_public_abi"]
