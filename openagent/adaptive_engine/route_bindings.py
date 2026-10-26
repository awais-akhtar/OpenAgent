"""Exact, source-reviewed bindings for admitted Phase 1 execution tuples.

These bindings close the *catalog coverage* gap between hand-authored route
examples and the canonical P02 matrix. They carry the six-field tuple and its
ordered source owners without claiming that a route was executed or evidenced.
Runtime admission/evidence remains governed by the source matrix and P13 gate.
"""

from dataclasses import dataclass
from functools import lru_cache
import ast
from pathlib import Path
from typing import Tuple

from .source_scope import source_matrix_row, source_matrix_rows


@dataclass(frozen=True)
class SourceReviewedRouteBinding:
    """Canonical source chain for one exact, source-reviewed route tuple."""

    family: str
    model_type: str
    backend: str
    arithmetic: str
    source_format: str
    persisted_format: str
    source_review: str
    chain_order: Tuple[str, ...]
    callable_chain: Tuple[Tuple[str, Tuple[str, ...]], ...]
    runtime_certified: bool = False

    @property
    def route_key(self):
        return (self.family, self.model_type, self.backend, self.arithmetic,
                self.source_format, self.persisted_format)

    def owners_for(self, dimension: str) -> Tuple[str, ...]:
        for name, owners in self.callable_chain:
            if name == dimension:
                return owners
        raise KeyError(dimension)

    def validate(self) -> bool:
        if (type(self.source_review) is not str or
                not self.source_review.startswith("p02-")):
            raise ValueError("exact P02 source review identifier required")
        if self.runtime_certified:
            raise ValueError("source binding cannot claim runtime certification")
        if not self.chain_order or tuple(name for name, _ in self.callable_chain) != self.chain_order:
            raise ValueError("complete ordered P02 callable chain required")
        if any(not owners or any(not owner for owner in owners)
               for _, owners in self.callable_chain):
            raise ValueError("every P02 chain dimension needs source owners")
        canonical = source_matrix_row(
            family=self.family, model_type=self.model_type, backend=self.backend,
            arithmetic=self.arithmetic, source_format=self.source_format,
            persisted_format=self.persisted_format)
        expected_chain = tuple((name, tuple(canonical["callable_chain"][name]))
                               for name in canonical["chain_order"])
        if (canonical["admission"] != "verified-admitted" or
                canonical["combination_status"] != "verified-admitted" or
                canonical["source_review"] != self.source_review or
                tuple(canonical["chain_order"]) != self.chain_order or
                expected_chain != self.callable_chain):
            raise ValueError("binding differs from its exact canonical P02 row")
        return True


@lru_cache(maxsize=1)
def source_reviewed_route_bindings():
    """Return one validated binding per unique source-admitted route tuple.

    Publisher/repository variants collapse only when the canonical matrix's
    exact-row resolver confirms that their source contract agrees.
    """
    bindings = {}
    for row in source_matrix_rows(admitted_only=True):
        key = (row["family"], row["model_type"], row["backend"],
               row["arithmetic"], row["source_format"], row["persisted_format"])
        binding = SourceReviewedRouteBinding(
            family=key[0], model_type=key[1], backend=key[2], arithmetic=key[3],
            source_format=key[4], persisted_format=key[5],
            source_review=row["source_review"],
            chain_order=tuple(row["chain_order"]),
            callable_chain=tuple((name, tuple(row["callable_chain"][name]))
                                 for name in row["chain_order"]))
        previous = bindings.get(key)
        if previous is not None and previous != binding:
            raise ValueError("publisher variants disagree on exact P02 binding")
        bindings[key] = binding
    result = tuple(bindings[key] for key in sorted(bindings))
    for binding in result:
        binding.validate()
    return result


def get_source_reviewed_route_binding(*, family, model_type, backend, arithmetic,
                                      source_format, persisted_format):
    """Resolve and validate one exact source-reviewed six-field route."""
    key = (family, model_type, backend, arithmetic, source_format, persisted_format)
    if any(type(value) is not str or not value for value in key):
        raise ValueError("complete six-field P02 route identity required")
    index = {binding.route_key: binding for binding in source_reviewed_route_bindings()}
    if key in index:
        return index[key]
    raise KeyError("/".join(key))


def validate_source_reviewed_route_bindings():
    """Check unique tuple coverage and exact chain equality; never certify runs."""
    rows = source_matrix_rows(admitted_only=True)
    expected = {(
        row["family"], row["model_type"], row["backend"], row["arithmetic"],
        row["source_format"], row["persisted_format"])
        for row in rows}
    bindings = source_reviewed_route_bindings()
    actual = {binding.route_key for binding in bindings}
    if actual != expected:
        raise ValueError("P02 exact source binding coverage differs from admitted matrix")
    if any(not binding.validate() for binding in bindings):
        raise ValueError("invalid P02 source route binding")
    return True


def _direct_ast_members(scope):
    """Return source declarations directly owned by a module/class scope."""
    body = getattr(scope, "body", ())
    result = {}
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                result[name] = node
    return result


def _resolve_callable_owner_source(reference, repository_root):
    """Resolve a catalog owner to a real project file and declared symbol.

    This is intentionally a static existence check, not an import or proof of
    callability/execution. Slash-qualified native source paths are checked as
    files; dotted Python owners resolve the longest real module prefix and
    then every remaining name through its AST scope.
    """
    root = Path(repository_root).resolve()
    search_roots = (root / "openagent" / "adaptive_engine", root / "openagent")
    if "/" in reference or reference.endswith((".c", ".cc", ".cpp", ".h", ".hpp")):
        return any((base / reference).is_file() for base in search_roots)

    parts = reference.split(".")
    for base in search_roots:
        for split in range(len(parts), 0, -1):
            module = base.joinpath(*parts[:split])
            source = module.with_suffix(".py")
            if not source.is_file():
                package_init = module / "__init__.py"
                if package_init.is_file():
                    source = package_init
                else:
                    continue
            try:
                scope = ast.parse(source.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, SyntaxError):
                return False
            for name in parts[split:]:
                scope = _direct_ast_members(scope).get(name)
                if scope is None:
                    return False
            return True
    return False


def validate_callable_owner_sources(repository_root=None):
    """Require every P02 binding owner to resolve to a project source symbol.

    Unlike the binding-chain check, this catches stale references such as a
    deleted module or misspelled class/method name. It does not certify runtime
    reachability, invocation, ABI compatibility, or numerical behavior.
    """
    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    references = sorted({
        reference
        for binding in source_reviewed_route_bindings()
        for _, owners in binding.callable_chain
        for reference in owners
    })
    missing = [reference for reference in references
               if not _resolve_callable_owner_source(reference, root)]
    if missing:
        raise ValueError("P02 callable owner source references do not resolve: " +
                         ", ".join(missing[:20]))
    return len(references)


__all__ = ["SourceReviewedRouteBinding", "source_reviewed_route_bindings",
           "get_source_reviewed_route_binding",
           "validate_source_reviewed_route_bindings",
           "validate_callable_owner_sources"]
