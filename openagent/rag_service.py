"""G25: RAG (Retrieval-Augmented Generation) - semantic search + context retrieval."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation: semantic search with context."""

    def __init__(self, embeddings_service, document_processor, knowledge_graph):
        self.embeddings = embeddings_service
        self.documents = document_processor
        self.knowledge_graph = knowledge_graph
        self.retrieval_index = {}  # Maps embeddings to documents

    def index_document(self, doc_id: str) -> dict[str, Any]:
        """Index document for semantic search."""
        doc = self.documents.get_document(doc_id)
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
            })
        
        self.retrieval_index[doc_id] = chunk_embeddings
        logger.info(f"Indexed document {doc_id} with {len(chunks)} chunks")
        
        return {
            "document_id": doc_id,
            "chunks": len(chunks),
            "indexed": True,
        }

    def semantic_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Search documents semantically."""
        query_embedding = self.embeddings.embed_text(query)
        
        results = []
        for doc_id, chunks in self.retrieval_index.items():
            for chunk_data in chunks:
                similarity = self.embeddings.similarity(
                    query_embedding,
                    chunk_data["embedding"]
                )
                results.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_data["chunk_id"],
                    "text": chunk_data["text"],
                    "similarity": float(similarity),
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "query": query,
            "results": results[:top_k],
            "total_matches": len(results),
        }

    def retrieve_context(self, query: str, context_size: int = 3) -> dict[str, Any]:
        """Retrieve context for query from documents and knowledge graph."""
        # Semantic search
        search_results = self.semantic_search(query, top_k=context_size)
        
        # Knowledge graph search
        kg_results = self.knowledge_graph.query_graph(query)
        
        return {
            "query": query,
            "document_context": search_results["results"],
            "knowledge_context": kg_results["results"],
            "total_context": len(search_results["results"]) + len(kg_results["results"]),
        }

    def generate_with_context(self, query: str, context_size: int = 3) -> dict[str, Any]:
        """Generate response using retrieved context (RAG)."""
        context = self.retrieve_context(query, context_size)
        
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
