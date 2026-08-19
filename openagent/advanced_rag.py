"""LatticeRAG: local multi-lane retrieval, evidence packing, and grounded answers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
import time
from typing import Any

from .knowledge import KnowledgeEngine
from .model_gateway import ModelGateway
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore
from .vector_store import VectorStore


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
QUESTION_WORDS = {
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "me", "of", "on", "or", "that", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with", "would", "you",
}
NEGATIONS = {"no", "not", "never", "without", "cannot", "can't", "doesn't", "isn't"}


class AdvancedRAGEngine:
    """Grounded local RAG with independent retrieval lanes and auditable fusion."""

    algorithm = "openagent-lattice-rag-v1"

    def __init__(
        self,
        store: OpenAgentStore,
        *,
        knowledge: KnowledgeEngine | None = None,
        vector_store: VectorStore | None = None,
        gateway: ModelGateway | None = None,
    ) -> None:
        self.store = store
        self.vector_store = vector_store or VectorStore(store)
        self.knowledge = knowledge or KnowledgeEngine(store, vector_store=self.vector_store)
        self.gateway = gateway or ModelGateway()
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_query_cache (
                    cache_key TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    knowledge_generation TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_rag_cache_project_expiry
                    ON rag_query_cache(project, expires_at);

                CREATE TABLE IF NOT EXISTS rag_query_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    query TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    source_count INTEGER NOT NULL,
                    generated INTEGER NOT NULL DEFAULT 0,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_rag_runs_project_created
                    ON rag_query_runs(project, created_at DESC);
                """
            )

    def close(self) -> None:
        self.vector_store.close()

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as connection:
            run_count = int(
                connection.execute(
                    "SELECT count(*) FROM rag_query_runs WHERE project = ?", (project,)
                ).fetchone()[0]
            )
            cache_count = int(
                connection.execute(
                    "SELECT count(*) FROM rag_query_cache WHERE project = ?", (project,)
                ).fetchone()[0]
            )
        return {
            "status": "ready",
            "project": project,
            "algorithm": self.algorithm,
            "retrieval_lanes": ["fts5-bm25", "faiss-cosine", "chroma-cosine", "sqlite-fallback"],
            "fusion": ["reciprocal-rank", "backend-consensus", "query-coverage", "mmr-diversity"],
            "grounding": ["source-citations", "evidence-budget", "conflict-flags", "low-confidence-abstention"],
            "query_runs": run_count,
            "cache_entries": cache_count,
            "vector_store": self.vector_store.status(project=project),
            "privacy_boundary": (
                "Documents, queries, embeddings, cache entries, and retrieval traces remain in local "
                "OpenAgent storage. Answer generation crosses a boundary only when the caller explicitly "
                "selects and approves an external provider."
            ),
        }

    def query(
        self,
        query: str,
        *,
        project: str = "default",
        limit: int = 8,
        context_token_budget: int = 1200,
        generate: bool = False,
        provider: str = "auto",
        model: str | None = None,
        allow_external: bool = False,
        use_cache: bool = True,
        embedding_provider: str = "",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        query = " ".join(query.split()).strip()
        if not query:
            raise ValueError("query is required.")
        project = project.strip() or "default"
        limit = max(1, min(int(limit), 20))
        context_token_budget = max(128, min(int(context_token_budget), 8192))
        embedding_provider = (
            embedding_provider or os.getenv("OPENAGENT_EMBEDDING_PROVIDER", "local-hash")
        ).strip() or "local-hash"
        generation = self._knowledge_generation(project)
        cache_key = self._cache_key(
            query=query,
            project=project,
            generation=generation,
            limit=limit,
            context_token_budget=context_token_budget,
            generate=generate,
            provider=provider,
            model=model or "",
            embedding_provider=embedding_provider,
        )
        if use_cache:
            cached = self._get_cache(cache_key)
            if cached is not None:
                cached["cache"] = {"hit": True, "key": cache_key, "knowledge_generation": generation}
                cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                self._record_run(cached, cache_hit=True)
                self.store.add_audit_event(
                    AuditEvent(
                        actor="advanced-rag",
                        action="rag.query.cache_hit",
                        resource=f"rag-query:{cache_key[:16]}",
                        project=project,
                        metadata={
                            "algorithm": self.algorithm,
                            "sources": len(cached.get("sources", [])),
                            "latency_ms": cached["latency_ms"],
                            "cache_hit": True,
                        },
                    )
                )
                return cached

        variants = query_variants(query)
        candidate_limit = max(24, limit * 5)
        lanes: dict[str, list[dict[str, Any]]] = {}
        for index, variant in enumerate(variants):
            lanes[f"lexical:{index}"] = self.knowledge.search(
                variant, project=project, limit=candidate_limit
            )
        lanes["semantic"] = self.knowledge.semantic_search(
            query,
            project=project,
            limit=candidate_limit,
            embedding_provider=embedding_provider,
        )
        fused = fuse_rankings(query, lanes)
        selected = diverse_evidence(query, fused, limit=limit)
        context, citations, packed_tokens = pack_context(
            selected, token_budget=context_token_budget
        )
        packed_evidence = selected[: len(citations)]
        conflicts = detect_conflicts(selected)
        confidence = confidence_metrics(query, selected, conflicts)
        answer = extractive_answer(query, packed_evidence, confidence)
        generation_meta: dict[str, Any] = {"used": False, "provider": "extractive-local", "model": "none"}
        if generate and selected:
            prompt = grounded_prompt(query, context)
            model_result = self.gateway.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the supplied local evidence. Cite sources using [S1], [S2], etc. "
                            "If the evidence is insufficient or conflicting, say so directly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                provider_id=provider,
                model=model,
                allow_external=allow_external,
            )
            answer = model_result.content
            generation_meta = {
                "used": True,
                "provider": model_result.provider,
                "model": model_result.model,
                "metadata": model_result.metadata,
            }

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        result = {
            "project": project,
            "query": query,
            "answer": answer,
            "sources": citations,
            "evidence": [public_evidence(item) for item in packed_evidence],
            "conflicts": conflicts,
            "confidence": confidence,
            "context": {
                "packed_tokens_estimate": packed_tokens,
                "token_budget": context_token_budget,
                "source_count": len(citations),
            },
            "retrieval": {
                "algorithm": self.algorithm,
                "query_variants": variants,
                "embedding_provider": embedding_provider,
                "lanes": {name: len(items) for name, items in lanes.items()},
                "candidate_count": len(fused),
                "selected_count": len(selected),
                "techniques": [
                    "change-sensitive-cache",
                    "multi-query-lexical-retrieval",
                    "faiss-chroma-consensus",
                    "reciprocal-rank-fusion",
                    "coverage-aware-reranking",
                    "maximal-marginal-relevance",
                    "evidence-budget-packing",
                ],
            },
            "generation": generation_meta,
            "cache": {"hit": False, "key": cache_key, "knowledge_generation": generation},
            "latency_ms": latency_ms,
            "privacy_boundary": (
                "Retrieval and caching were local. Generation remained local unless allow_external=true "
                "and an external provider was explicitly selected."
            ),
        }
        if use_cache:
            self._put_cache(cache_key, project, generation, result)
        self._record_run(result, cache_hit=False)
        self.store.add_audit_event(
            AuditEvent(
                actor="advanced-rag",
                action="rag.query",
                resource=f"rag-query:{cache_key[:16]}",
                project=project,
                metadata={
                    "algorithm": self.algorithm,
                    "sources": len(citations),
                    "generated": generate,
                    "provider": generation_meta["provider"],
                    "latency_ms": latency_ms,
                    "cache_hit": False,
                },
            )
        )
        return result

    def benchmark(self, *, project: str = "rag-benchmark", iterations: int = 5) -> dict[str, Any]:
        samples = [
            (
                "Local privacy boundary",
                "OpenAgent stores private documents in local SQLite. FAISS accelerates memory search and Chroma persists a local mirror.",
            ),
            (
                "CPU latency",
                "Adaptive token budgets, warm model sessions, prompt compaction, and response caching reduce CPU inference latency.",
            ),
            (
                "Grounded retrieval",
                "Reciprocal rank fusion combines lexical and semantic evidence while citations preserve source lineage.",
            ),
        ]
        for title, content in samples:
            self.knowledge.ingest_text(title, content, project=project, source="rag-benchmark")
        cold = self.query(
            "How does local retrieval stay private and fast?",
            project=project,
            generate=False,
            use_cache=False,
        )
        warm_latencies = []
        for _ in range(max(1, min(int(iterations), 50))):
            warm = self.query(
                "How does local retrieval stay private and fast?",
                project=project,
                generate=False,
                use_cache=True,
            )
            warm_latencies.append(float(warm["latency_ms"]))
        return {
            "status": "ok",
            "algorithm": self.algorithm,
            "project": project,
            "cold_latency_ms": cold["latency_ms"],
            "warm_latency_ms": {
                "min": round(min(warm_latencies), 3),
                "mean": round(sum(warm_latencies) / len(warm_latencies), 3),
                "max": round(max(warm_latencies), 3),
            },
            "top_source": cold["sources"][0] if cold["sources"] else None,
            "target": "sub-second retrieval on a warm local index; model generation latency is measured separately",
        }

    def _knowledge_generation(self, project: str) -> str:
        with self.store.session() as connection:
            row = connection.execute(
                """
                SELECT count(*) AS item_count, COALESCE(max(created_at), '') AS newest
                FROM document_chunks WHERE project = ?
                """,
                (project,),
            ).fetchone()
            document_hashes = connection.execute(
                "SELECT content_hash FROM documents WHERE project = ? ORDER BY id",
                (project,),
            ).fetchall()
        digest = hashlib.sha256()
        digest.update(f"{row['item_count']}:{row['newest']}".encode("utf-8"))
        for item in document_hashes:
            digest.update(str(item["content_hash"]).encode("ascii", errors="ignore"))
        return digest.hexdigest()[:24]

    def _cache_key(self, **payload: Any) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _get_cache(self, cache_key: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM rag_query_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00"))
        if expires_at <= datetime.now(timezone.utc):
            with self.store.session() as connection:
                connection.execute("DELETE FROM rag_query_cache WHERE cache_key = ?", (cache_key,))
            return None
        return json.loads(row["response_json"])

    def _put_cache(self, cache_key: str, project: str, generation: str, result: dict[str, Any]) -> None:
        created_at = utc_now()
        ttl_seconds = _env_int("OPENAGENT_RAG_CACHE_TTL_SECONDS", 900, minimum=1, maximum=86400)
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO rag_query_cache (
                    cache_key, project, knowledge_generation, response_json, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, project, generation, json.dumps(result, sort_keys=True), created_at, expires_at),
            )

    def _record_run(self, result: dict[str, Any], *, cache_hit: bool) -> None:
        created_at = utc_now()
        run_id = hashlib.sha256(
            f"{result['project']}:{result['query']}:{created_at}:{time.time_ns()}:{cache_hit}".encode("utf-8")
        ).hexdigest()[:32]
        metrics = {
            "confidence": result.get("confidence", {}),
            "retrieval": result.get("retrieval", {}),
            "cache": result.get("cache", {}),
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO rag_query_runs (
                    id, project, query, algorithm, source_count, generated,
                    cache_hit, latency_ms, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result["project"],
                    result["query"],
                    self.algorithm,
                    len(result.get("sources", [])),
                    1 if result.get("generation", {}).get("used") else 0,
                    1 if cache_hit else 0,
                    float(result.get("latency_ms", 0.0)),
                    json.dumps(metrics, sort_keys=True),
                    created_at,
                ),
            )


def query_variants(query: str) -> list[str]:
    normalized = " ".join(query.split())
    terms = [token.lower() for token in TOKEN_RE.findall(normalized)]
    useful = [term for term in terms if term not in QUESTION_WORDS and len(term) > 2]
    variants = [normalized]
    keyword_query = " ".join(dict.fromkeys(useful))
    if keyword_query and keyword_query.lower() != normalized.lower():
        variants.append(keyword_query)
    quoted = re.findall(r'"([^"]{3,})"', normalized)
    if quoted:
        variants.append(" ".join(quoted))
    return list(dict.fromkeys(variants))[:3]


def fuse_rankings(query: str, lanes: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    query_terms = useful_terms(query)
    for lane, items in lanes.items():
        lane_weight = 1.15 if lane == "semantic" else 1.0
        for rank, item in enumerate(items, start=1):
            record_id = str(item["id"])
            current = records.setdefault(
                record_id,
                {
                    **item,
                    "rrf_score": 0.0,
                    "lanes": [],
                    "backend_votes": set(),
                },
            )
            current["rrf_score"] += lane_weight / (60.0 + rank)
            current["lanes"].append({"lane": lane, "rank": rank})
            for backend in item.get("retrieval", []):
                if backend in {"faiss", "chroma", "lexical", "semantic"}:
                    current["backend_votes"].add(backend)
            current["base_score"] = max(
                float(current.get("base_score", 0.0)), float(item.get("score", 0.0))
            )
    for item in records.values():
        content_terms = useful_terms(f"{item.get('title', '')} {item.get('content', '')}")
        coverage = len(query_terms & content_terms) / max(1, len(query_terms))
        phrase_bonus = 1.0 if query.lower() in str(item.get("content", "")).lower() else 0.0
        consensus = min(1.0, len(item["backend_votes"]) / 3.0)
        item["coverage_score"] = coverage
        item["consensus_score"] = consensus
        item["score"] = round(
            (0.44 * min(1.0, float(item["base_score"])))
            + (0.25 * min(1.0, item["rrf_score"] * 30.0))
            + (0.20 * coverage)
            + (0.08 * consensus)
            + (0.03 * phrase_bonus),
            6,
        )
        item["backend_votes"] = sorted(item["backend_votes"])
    return sorted(records.values(), key=lambda item: float(item["score"]), reverse=True)


def diverse_evidence(query: str, candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < limit:
        best_index = 0
        best_value = float("-inf")
        for index, candidate in enumerate(remaining):
            relevance = float(candidate.get("score", 0.0))
            redundancy = max(
                (jaccard(candidate.get("content", ""), item.get("content", "")) for item in selected),
                default=0.0,
            )
            value = (0.82 * relevance) - (0.18 * redundancy)
            if value > best_value:
                best_index = index
                best_value = value
        chosen = remaining.pop(best_index)
        chosen["diversity_score"] = round(best_value, 6)
        selected.append(chosen)
    return selected


def pack_context(items: list[dict[str, Any]], *, token_budget: int) -> tuple[str, list[dict[str, Any]], int]:
    remaining = token_budget
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    used = 0
    for index, item in enumerate(items, start=1):
        header = f"[S{index}] {item.get('title', 'Source')}"
        header_tokens = estimate_tokens(header) + 6
        if remaining <= header_tokens + 16:
            break
        content = " ".join(str(item.get("content", "")).split())
        allowed = min(remaining - header_tokens, max(48, int(token_budget * 0.42)))
        excerpt = truncate_to_tokens(content, allowed)
        excerpt_tokens = estimate_tokens(excerpt)
        blocks.append(f"{header}\n{excerpt}")
        citations.append(
            {
                "citation": f"S{index}",
                "chunk_id": str(item.get("id", "")),
                "document_id": str(item.get("document_id", "")),
                "title": str(item.get("title", "Source")),
                "score": item.get("score", 0.0),
                "retrieval": item.get("backend_votes") or item.get("retrieval", []),
            }
        )
        spent = header_tokens + excerpt_tokens
        used += spent
        remaining -= spent
    return "\n\n".join(blocks), citations, used


def detect_conflicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for left_index, left in enumerate(items):
        left_terms = useful_terms(str(left.get("content", "")))
        left_negated = bool(left_terms & NEGATIONS)
        for right in items[left_index + 1 :]:
            right_terms = useful_terms(str(right.get("content", "")))
            overlap = len((left_terms - NEGATIONS) & (right_terms - NEGATIONS))
            if overlap >= 4 and left_negated != bool(right_terms & NEGATIONS):
                conflicts.append(
                    {
                        "left": str(left.get("id", "")),
                        "right": str(right.get("id", "")),
                        "reason": "Overlapping evidence uses opposite negation polarity; review both sources.",
                    }
                )
    return conflicts[:10]


def confidence_metrics(query: str, items: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"score": 0.0, "level": "insufficient", "coverage": 0.0, "conflict_penalty": 0.0}
    query_terms = useful_terms(query)
    covered: set[str] = set()
    for item in items:
        covered.update(query_terms & useful_terms(str(item.get("content", ""))))
    coverage = len(covered) / max(1, len(query_terms))
    top_score = float(items[0].get("score", 0.0))
    consensus = sum(float(item.get("consensus_score", 0.0)) for item in items[:3]) / max(1, min(3, len(items)))
    conflict_penalty = min(0.3, len(conflicts) * 0.08)
    score = max(0.0, min(1.0, (0.5 * top_score) + (0.32 * coverage) + (0.18 * consensus) - conflict_penalty))
    level = "high" if score >= 0.72 else "medium" if score >= 0.42 else "low"
    return {
        "score": round(score, 6),
        "level": level,
        "coverage": round(coverage, 6),
        "backend_consensus": round(consensus, 6),
        "conflict_penalty": round(conflict_penalty, 6),
    }


def extractive_answer(query: str, items: list[dict[str, Any]], confidence: dict[str, Any]) -> str:
    if not items:
        return "I could not find local evidence for this question. Add or sync relevant sources before relying on an answer."
    if confidence["level"] == "low":
        prefix = "The local evidence is limited, so treat this as a lead rather than a final answer."
    else:
        prefix = "Based on the strongest local evidence:"
    sentences: list[str] = []
    for index, item in enumerate(items[:4], start=1):
        content = " ".join(str(item.get("content", "")).split())
        choices = [sentence.strip() for sentence in SENTENCE_RE.split(content) if sentence.strip()]
        if choices:
            sentences.append(f"{choices[0]} [S{index}]")
    return f"{prefix}\n\n" + " ".join(sentences)


def grounded_prompt(query: str, context: str) -> str:
    return f"Question:\n{query}\n\nLocal evidence:\n{context}\n\nWrite a concise grounded answer with inline [S#] citations."


def public_evidence(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id", "")),
        "document_id": str(item.get("document_id", "")),
        "title": str(item.get("title", "Source")),
        "content": str(item.get("content", "")),
        "score": item.get("score", 0.0),
        "coverage_score": round(float(item.get("coverage_score", 0.0)), 6),
        "consensus_score": round(float(item.get("consensus_score", 0.0)), 6),
        "diversity_score": item.get("diversity_score", 0.0),
        "lanes": item.get("lanes", []),
        "retrieval": item.get("backend_votes") or item.get("retrieval", []),
        "metadata": item.get("metadata", {}),
    }


def useful_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in QUESTION_WORDS
    }


def jaccard(left: str, right: str) -> float:
    left_terms = useful_terms(left)
    right_terms = useful_terms(right)
    union = left_terms | right_terms
    return len(left_terms & right_terms) / len(union) if union else 0.0


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def truncate_to_tokens(text: str, tokens: int) -> str:
    max_chars = max(16, tokens * 4)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    import os

    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))
