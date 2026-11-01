"""G25: Advanced Features routes - document processing, RAG, knowledge graphs, reasoning."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from ..document_processor_service import DocumentProcessorService
from ..knowledge_graph_service import KnowledgeGraphService
from ..rag_service import RAGService
from ..advanced_reasoning_service import AdvancedReasoningService
from ..semantic_memory_service import SemanticMemoryService
from ..vector_embeddings_service import VectorEmbeddingsService
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "g25_advanced"
ROUTES = (
    ("POST", "/api/g25/documents/upload"),
    ("GET", "/api/g25/documents"),
    ("GET", "/api/g25/documents/{id}"),
    ("POST", "/api/g25/documents/{id}/analyze"),
    ("POST", "/api/g25/graph/nodes"),
    ("POST", "/api/g25/graph/edges"),
    ("GET", "/api/g25/graph/query"),
    ("GET", "/api/g25/graph/paths"),
    ("POST", "/api/g25/rag/index"),
    ("POST", "/api/g25/rag/search"),
    ("GET", "/api/g25/rag/context"),
    ("POST", "/api/g25/reasoning/chains"),
    ("POST", "/api/g25/reasoning/analyze"),
    ("GET", "/api/g25/reasoning/chains/{id}"),
    ("POST", "/api/g25/memory/sessions"),
    ("POST", "/api/g25/memory/store"),
    ("GET", "/api/g25/memory/history"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class G25RouteDependencies:
    doc_processor: DocumentProcessorService
    knowledge_graph: KnowledgeGraphService
    rag_service: RAGService
    reasoning_service: AdvancedReasoningService
    memory_service: SemanticMemoryService
    embeddings: VectorEmbeddingsService
    read_json: Any
    query_owner: Callable[[dict[str, list[str]]], str]
    auth_scope_owner: Callable[[], str]


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def handle_get(path: str, query: str, deps: G25RouteDependencies) -> JsonRouteResult | None:
    """Handle GET requests for G25 advanced features."""
    params = parse_qs(query)

    project = _first(params, "project") or "default"
    owner = deps.query_owner(params) or "default"

    if path == "/api/g25/documents":
        docs = deps.doc_processor.list_documents(project=project, owner=owner)
        return JsonRouteResult({"documents": docs})

    if path.startswith("/api/g25/documents/") and path.count("/") == 4:
        doc_id = unquote(path.removeprefix("/api/g25/documents/")).strip()
        doc = deps.doc_processor.get_document(doc_id, project=project, owner=owner)
        if not doc:
            return JsonRouteResult({"error": "Document not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(doc)

    if path == "/api/g25/graph/query":
        query_text = _first(params, "q")
        if not query_text:
            return JsonRouteResult({"error": "Query required"}, HTTPStatus.BAD_REQUEST)
        results = deps.knowledge_graph.query_graph(query_text, project=project, owner=owner)
        return JsonRouteResult({"results": results})

    if path == "/api/g25/graph/paths":
        start_label = _first(params, "start")
        end_label = _first(params, "end")
        if not start_label or not end_label:
            return JsonRouteResult({"error": "Start and end labels required"}, HTTPStatus.BAD_REQUEST)
        paths = deps.knowledge_graph.find_paths(start_label, end_label, project=project, owner=owner)
        return JsonRouteResult({"paths": paths})

    if path == "/api/g25/rag/context":
        doc_id = _first(params, "doc_id")
        query = _first(params, "q")
        if not doc_id:
            return JsonRouteResult({"error": "Document ID required"}, HTTPStatus.BAD_REQUEST)
        context = deps.rag_service.get_context(doc_id, query or "", project=project, owner=owner)
        return JsonRouteResult(context)

    if path.startswith("/api/g25/reasoning/chains/"):
        chain_id = unquote(path.removeprefix("/api/g25/reasoning/chains/")).strip()
        chain = deps.reasoning_service.get_chain(chain_id, project=project, owner=owner)
        if not chain:
            return JsonRouteResult({"error": "Chain not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(chain)

    if path == "/api/g25/memory/history":
        session_id = _first(params, "session_id")
        if not session_id:
            return JsonRouteResult({"error": "Session ID required"}, HTTPStatus.BAD_REQUEST)
        history = deps.memory_service.get_conversation_history(session_id, project=project, owner=owner)
        return JsonRouteResult(history)

    return None


def handle_post(path: str, deps: G25RouteDependencies) -> JsonRouteResult | None:
    """Handle POST requests for G25 advanced features."""
    if not path.startswith("/api/g25/"):
        return None

    payload = deps.read_json()

    project = str(payload.get("project", "")).strip() or "default"
    owner = deps.auth_scope_owner() or "default"

    if path == "/api/g25/documents/upload":
        filename = str(payload.get("filename", "document")).strip()
        content_b64 = str(payload.get("content", "")).strip()

        if not content_b64:
            return JsonRouteResult({"error": "Content required"}, HTTPStatus.BAD_REQUEST)

        try:
            content_bytes = base64.b64decode(content_b64)
        except Exception as e:
            return JsonRouteResult({"error": f"Invalid content: {e}"}, HTTPStatus.BAD_REQUEST)

        result = deps.doc_processor.process_document(filename, content_bytes, project=project, owner=owner)
        return JsonRouteResult(result, HTTPStatus.CREATED)

    if path.startswith("/api/g25/documents/") and path.endswith("/analyze"):
        doc_id = unquote(path.removeprefix("/api/g25/documents/").removesuffix("/analyze")).strip()
        analysis = deps.doc_processor.analyze_document(doc_id, project=project, owner=owner)
        if not analysis:
            return JsonRouteResult({"error": "Document not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(analysis)

    if path == "/api/g25/graph/nodes":
        label = str(payload.get("label", "")).strip()
        properties = dict(payload.get("properties", {}))

        if not label:
            return JsonRouteResult({"error": "Label required"}, HTTPStatus.BAD_REQUEST)

        node = deps.knowledge_graph.create_node(label, properties, project=project, owner=owner)
        return JsonRouteResult(node, HTTPStatus.CREATED)

    if path == "/api/g25/graph/edges":
        source_id = str(payload.get("source_id", "")).strip()
        target_id = str(payload.get("target_id", "")).strip()
        relationship = str(payload.get("relationship", "")).strip()

        if not source_id or not target_id or not relationship:
            return JsonRouteResult(
                {"error": "source_id, target_id, relationship required"},
                HTTPStatus.BAD_REQUEST,
            )

        edge = deps.knowledge_graph.create_edge(source_id, target_id, relationship, project=project, owner=owner)
        return JsonRouteResult(edge, HTTPStatus.CREATED)

    if path == "/api/g25/rag/index":
        doc_id = str(payload.get("doc_id", "")).strip()
        chunks = list(payload.get("chunks", []))

        if not doc_id or not chunks:
            return JsonRouteResult({"error": "doc_id and chunks required"}, HTTPStatus.BAD_REQUEST)

        result = deps.rag_service.index_document(doc_id, chunks, project=project, owner=owner)
        return JsonRouteResult(result, HTTPStatus.CREATED)

    if path == "/api/g25/rag/search":
        query = str(payload.get("query", "")).strip()
        if not query:
            return JsonRouteResult({"error": "Query required"}, HTTPStatus.BAD_REQUEST)

        results = deps.rag_service.search(query, project=project, owner=owner)
        return JsonRouteResult({"results": results})

    if path == "/api/g25/reasoning/chains":
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            return JsonRouteResult({"error": "Topic required"}, HTTPStatus.BAD_REQUEST)

        chain = deps.reasoning_service.create_chain(topic, project=project, owner=owner)
        return JsonRouteResult(chain, HTTPStatus.CREATED)

    if path == "/api/g25/reasoning/analyze":
        text = str(payload.get("text", "")).strip()
        if not text:
            return JsonRouteResult({"error": "Text required"}, HTTPStatus.BAD_REQUEST)

        analysis = deps.reasoning_service.analyze_semantics(text, project=project, owner=owner)
        return JsonRouteResult(analysis)

    if path == "/api/g25/memory/sessions":
        session = deps.memory_service.create_session(project=project, owner=owner)
        return JsonRouteResult(session, HTTPStatus.CREATED)

    if path == "/api/g25/memory/store":
        session_id = str(payload.get("session_id", "")).strip()
        role = str(payload.get("role", "")).strip()
        content = str(payload.get("content", "")).strip()

        if not session_id or not role or not content:
            return JsonRouteResult(
                {"error": "session_id, role, content required"},
                HTTPStatus.BAD_REQUEST,
            )

        result = deps.memory_service.store_message(session_id, role, content, project=project, owner=owner)
        return JsonRouteResult(result)

    return None
