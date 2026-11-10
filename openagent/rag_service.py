"""G25: RAG (Retrieval-Augmented Generation) - semantic search + context retrieval."""

from __future__ import annotations

import logging
from typing import Any

from .models import utc_now
from .storage import OpenAgentStore

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation: semantic search with context."""

    def __init__(self, embeddings_service, document_processor, knowledge_graph, store: OpenAgentStore | None = None):
        self.embeddings = embeddings_service
        self.documents = document_processor
        self.knowledge_graph = knowledge_graph
        self.store = store
        # Keyed by the (project, owner, doc_id) triple rather than a joined
        # string: a project or owner name containing a colon would otherwise
        # produce the same key as a different pairing, and one scope's chunks
        # would be searchable from the other.
        self.retrieval_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    def index_document(self, doc_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Index document for semantic search."""
        doc = self.documents.get_document(doc_id, project, owner)
        if not doc:
            return {"error": "Document not found"}

        text = doc["text"]

        # Split into chunks
        chunks = self._chunk_text(text, chunk_size=500)

        # Embed each chunk
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_text(chunk)
            chunk_embeddings.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding,
                "project": project,
                "owner": owner,
            })

        if self.store is not None:
            with self.store.session() as connection:
                connection.execute(
                    """INSERT INTO g25_rag_indexed_documents (doc_id, project, owner, indexed_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(doc_id) DO UPDATE SET indexed_at = excluded.indexed_at""",
                    (doc_id, project, owner, utc_now()),
                )
        self.retrieval_index[(project, owner, doc_id)] = chunk_embeddings
        logger.info(f"Indexed document {doc_id} with {len(chunks)} chunks (project={project}, owner={owner})")

        return {
            "document_id": doc_id,
            "chunks": len(chunks),
            "indexed": True,
        }

    def search(self, query: str, top_k: int = 5, project: str = "default", owner: str = "default",
               doc_id: str = "") -> dict[str, Any]:
        """Search documents semantically, optionally within one document."""
        if self.store is not None:
            with self.store.session() as connection:
                stored = connection.execute(
                    """SELECT doc_id FROM g25_rag_indexed_documents
                       WHERE project = ? AND owner = ? AND (? = '' OR doc_id = ?)""",
                    (project, owner, doc_id, doc_id),
                ).fetchall()
            for row in stored:
                indexed_id = row["doc_id"]
                if (project, owner, indexed_id) not in self.retrieval_index:
                    # The indexed-document marker is durable. Rebuild derived
                    # embeddings lazily from the owner-bound saved text.
                    self.index_document(indexed_id, project=project, owner=owner)
        query_embedding = self.embeddings.embed_text(query)

        results = []
        for (indexed_project, indexed_owner, indexed_doc), chunks in list(self.retrieval_index.items()):
            # Only search documents in this project/owner
            if indexed_project != project or indexed_owner != owner:
                continue
            if doc_id and indexed_doc != doc_id:
                continue
            for chunk_data in chunks:
                similarity = self.embeddings.similarity(
                    query_embedding,
                    chunk_data["embedding"]
                )
                results.append({
                    "doc_id": chunk_data["doc_id"],
                    "chunk_id": chunk_data["chunk_id"],
                    "text": chunk_data["text"],
                    "similarity": float(similarity),
                    "project": project,
                    "owner": owner,
                })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "query": query,
            "results": results[:top_k],
            "total_matches": len(results),
        }

    def get_context(self, query: str, context_size: int = 3, project: str = "default", owner: str = "default",
                    doc_id: str = "") -> dict[str, Any]:
        """Retrieve context for query from documents and knowledge graph.

        ``doc_id`` narrows the document side to a single document; the
        knowledge-graph side is not document scoped.
        """
        # Semantic search
        search_results = self.search(query, top_k=context_size, project=project, owner=owner, doc_id=doc_id)

        # Knowledge graph search
        kg_results = self.knowledge_graph.query_graph(query, project=project, owner=owner)

        return {
            "query": query,
            "document_context": search_results["results"],
            "knowledge_context": kg_results["results"],
            "total_context": len(search_results["results"]) + len(kg_results["results"]),
        }

    def generate_with_context(self, query: str, context_size: int = 3, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Generate response using retrieved context (RAG)."""
        context = self.get_context(query, context_size, project=project, owner=owner)

        # Build context string
        context_text = ""

        # Add document context
        for result in context["document_context"]:
            context_text += f"[Document {result['doc_id'][:8]}]: {result['text']}\n"

        # Add knowledge graph context
        for result in context["knowledge_context"]:
            context_text += f"[Knowledge]: {result['label']} - {result['properties']}\n"

        return {
            "query": query,
            "context": context_text,
            "context_size": context["total_context"],
            "ready_for_generation": True,
            "note": "Context retrieved. Ready for LLM generation.",
        }

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        sentences = text.split(". ")
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len > chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = current_chunk[-overlap//100:] if overlap else []
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_len
        
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
        
        return chunks

    def get_rag_stats(self) -> dict[str, Any]:
        """Get RAG system statistics."""
        total_chunks = sum(len(chunks) for chunks in self.retrieval_index.values())
        
        return {
            "indexed_documents": len(self.retrieval_index),
            "total_chunks": total_chunks,
            "embedding_dimension": self.embeddings.embedding_dim,
            "knowledge_graph_nodes": len(self.knowledge_graph.nodes),
        }
