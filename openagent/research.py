"""Deep research engine for OpenAgent.

Performs iterative, source-attributed research using the local model gateway
and configured search tools. All processing is local-first; external web
access only happens when explicitly allowed.
"""

from __future__ import annotations

import time
import uuid
import re
from typing import Any

from .core_runtime import smooth_runtime_profile
from .context import ContextEngine
from .external_research import ExternalResearchCollector
from .model_gateway import ModelGateway, ModelResult
from .models import AuditEvent
from .storage import OpenAgentStore


class ResearchEngine:
    """Iterative research engine that plans, collects, analyzes, and synthesizes."""

    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None) -> None:
        self.store = store
        self.context = ContextEngine(store)
        self.gateway = gateway or ModelGateway()

    def run(
        self,
        query: str,
        *,
        project: str = "default",
        max_iterations: int = 3,
        allow_external: bool = False,
        provider: str = "auto",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Execute a deep research run and return structured findings."""

        run_id = str(uuid.uuid4())
        started_at = time.time()
        runtime_profile = smooth_runtime_profile(task="research").to_dict()
        effective_iterations = 1 if max_iterations <= 3 else min(max_iterations, 3)
        context_limit = 6 if runtime_profile["tier"] == "ultra-light-8gb" else 10

        self.store.add_audit_event(
            AuditEvent(
                actor="research-engine",
                action="research.start",
                resource=f"research:{run_id}",
                project=project,
                metadata={"query": query, "max_iterations": effective_iterations, "runtime_profile": runtime_profile},
            )
        )

        # Step 1: Plan locally first so research stays responsive when a CPU
        # model server is cold-starting.
        plan = self._local_plan(query)

        # Step 2: Collect local context evidence
        context_hits = [
            item for item in self.context.search(query, project=project, limit=context_limit * 3)
            if str(item.get("type", "")).casefold() not in {"conversation", "research"}
        ][:context_limit]
        sources: list[dict[str, Any]] = [
            {
                "id": ev.get("id", ""), "title": ev.get("title", ""),
                "snippet": ev.get("content", "")[:500], "type": "local_memory",
                "url": "", "published_at": ev.get("created_at", ""),
                "metadata": {"source": ev.get("source", "local")},
            }
            for ev in context_hits
        ]
        if allow_external:
            sources.extend(ExternalResearchCollector().search(query, limit=self._requested_source_limit(query)))

        # Step 3: Iterative analysis — refine understanding
        # Step 3: Analyze evidence locally, then use one bounded model call for
        # synthesis. This keeps the module usable on CPU-only machines.
        analysis_notes = [
            self._local_analysis(query, plan, sources, iteration)
            for iteration in range(effective_iterations)
        ]

        # Step 4: Synthesize the final report
        synthesis = self._synthesize_result(
            query,
            plan,
            sources,
            analysis_notes,
            provider=provider,
            model=model,
            allow_external=allow_external,
        )
        report = synthesis.content
        external_sources = [source for source in sources if source.get("url")]
        if external_sources:
            report = f"{report.rstrip()}\n\n{self._source_list(sources)}"
        elif self._needs_current_sources(query):
            boundary = (
                "Public source search was authorized but no current sources could be reached. "
                if allow_external else
                "Current public sources were not searched because external research was not authorized for this run. "
            )
            report = f"{boundary}No claim about the latest paper can be verified.\n\n{report}".strip()

        elapsed = round(time.time() - started_at, 2)
        self.store.add_audit_event(
            AuditEvent(
                actor="research-engine",
                action="research.complete",
                resource=f"research:{run_id}",
                project=project,
                metadata={
                    "sources": len(sources),
                    "elapsed": elapsed,
                    "provider": synthesis.provider,
                    "model": synthesis.model,
                    "inference": synthesis.metadata.get("inference", {}),
                },
            )
        )

        # Store research output in context
        self.context.ingest(
            title=f"Research: {query[:80]}",
            content=report,
            type="research",
            source="research-engine",
            project=project,
            metadata={"run_id": run_id, "sources_count": len(sources)},
        )

        return {
            "run_id": run_id,
            "query": query,
            "project": project,
            "plan": plan,
            "sources": sources,
            "report": report,
            "elapsed_seconds": elapsed,
            "iterations": len(analysis_notes),
            "runtime_profile": runtime_profile,
            "provider": synthesis.provider,
            "model": synthesis.model,
            "engine": synthesis.metadata.get("inference", {}),
            "model_metadata": synthesis.metadata,
            "mode": "external-source-backed" if external_sources else "compact-local-synthesis",
        }

    def _local_plan(self, query: str) -> str:
        focus = query.strip().rstrip(".") or "the requested topic"
        return "\n".join(
            [
                f"1. Clarify the exact question: {focus}.",
                "2. Search local memory, knowledge, workspace, and prior research records first.",
                "3. Separate confirmed local evidence from assumptions or missing sources.",
                "4. Synthesize an actionable answer with source notes and next steps.",
            ]
        )

    def _local_analysis(
        self,
        query: str,
        plan: str,
        sources: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        if sources:
            evidence = "\n".join(
                f"- {source.get('title') or source.get('type') or 'Local source'}: {source.get('snippet', '')[:220]}"
                for source in sources[:6]
            )
        else:
            evidence = "No matching local sources were found. Mark conclusions as model-assisted and provisional."
        return (
            f"Iteration {iteration + 1} local analysis for: {query}\n\n"
            f"Plan followed:\n{plan}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Findings: prioritize local evidence, call out gaps, and keep recommendations auditable."
        )

    def _plan(self, query: str, *, provider: str, model: str | None, allow_external: bool) -> str:
        """Generate a research plan for the given query."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research planning assistant. Given a research question, "
                    "produce a concise numbered plan of 3-5 specific sub-questions or investigation steps. "
                    "Focus on what can be determined from local knowledge and context."
                ),
            },
            {"role": "user", "content": f"Plan a research investigation for: {query}"},
        ]
        result = self.gateway.complete(messages, provider_id=provider, model=model, allow_external=allow_external)
        return result.content

    def _analyze(
        self,
        query: str,
        plan: str,
        sources: list[dict[str, Any]],
        iteration: int,
        provider: str,
        model: str | None,
        allow_external: bool,
    ) -> str:
        """Analyze available evidence for one research iteration."""
        source_text = "\n".join(
            f"- [{s['type']}] {s['title']}: {s['snippet']}"
            for s in sources[:8]
        ) or "No local sources found yet."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Analyze the available evidence for the research question. "
                    "Note gaps, patterns, and key findings. If you have enough to finalize, end with [COMPLETE]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {query}\n\n"
                    f"Research plan:\n{plan}\n\n"
                    f"Available evidence (iteration {iteration + 1}):\n{source_text}\n\n"
                    "Analyze the evidence and identify key findings."
                ),
            },
        ]
        result = self.gateway.complete(messages, provider_id=provider, model=model, allow_external=allow_external)
        return result.content

    def _synthesize(
        self,
        query: str,
        plan: str,
        sources: list[dict[str, Any]],
        analysis_notes: list[str],
        provider: str,
        model: str | None,
        allow_external: bool,
    ) -> str:
        """Produce a final synthesized research report."""
        return self._synthesize_result(
            query,
            plan,
            sources,
            analysis_notes,
            provider=provider,
            model=model,
            allow_external=allow_external,
        ).content

    def _synthesize_result(
        self,
        query: str,
        plan: str,
        sources: list[dict[str, Any]],
        analysis_notes: list[str],
        provider: str,
        model: str | None,
        allow_external: bool,
    ) -> ModelResult:
        """Produce a report together with auditable shared-engine evidence."""
        combined_analysis = "\n\n".join(f"Iteration {i+1}:\n{note}" for i, note in enumerate(analysis_notes))
        source_text = "\n".join(
            f"[{index}] {source.get('title') or 'Untitled source'} | "
            f"date={source.get('published_at') or 'unknown'} | url={source.get('url') or 'local'} | "
            f"evidence={source.get('snippet', '')[:700]}"
            for index, source in enumerate(sources[:30], start=1)
        ) or "No verified sources were collected."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research synthesis expert. Produce a clear, well-structured research report "
                    "with an executive summary, key findings, and conclusions. "
                    "Every factual or comparative claim must be grounded in the supplied evidence. "
                    "Use bracket citations such as [1] that match the supplied source numbers. "
                    "Never invent a paper, date, author, URL, experiment, or search action. "
                    "If evidence is missing, say so directly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {query}\n\n"
                    f"Plan followed:\n{plan}\n\n"
                    f"Analysis notes:\n{combined_analysis}\n\n"
                    f"Verified source records:\n{source_text}\n\n"
                    "Write a useful report that answers the question directly, compares the strongest evidence, "
                    "and distinguishes metadata/abstract evidence from full-paper evidence."
                ),
            },
        ]
        return self._stream_model_result(
            messages,
            provider=provider,
            model=model,
            allow_external=allow_external,
            max_output_tokens=768,
        )

    def _stream_model_text(
        self,
        messages: list[dict[str, str]],
        *,
        provider: str,
        model: str | None,
        allow_external: bool,
        max_output_tokens: int,
    ) -> str:
        return self._stream_model_result(
            messages,
            provider=provider,
            model=model,
            allow_external=allow_external,
            max_output_tokens=max_output_tokens,
        ).content

    def _stream_model_result(
        self,
        messages: list[dict[str, str]],
        *,
        provider: str,
        model: str | None,
        allow_external: bool,
        max_output_tokens: int,
    ) -> ModelResult:
        parts: list[str] = []
        fallback_seen = False
        final_provider = provider
        final_model = model or ""
        final_metadata: dict[str, Any] = {}
        for chunk in self.gateway.stream_complete(
            messages,
            provider_id=provider,
            model=model,
            allow_external=allow_external,
            max_output_tokens=max_output_tokens,
        ):
            metadata = dict(chunk.metadata or {})
            if metadata.get("fallback_from") and not fallback_seen:
                parts.clear()
                fallback_seen = True
            final_provider = chunk.provider
            final_model = chunk.model
            final_metadata = metadata
            if chunk.content:
                parts.append(chunk.content)
        return ModelResult(
            provider=final_provider,
            model=final_model,
            content="".join(parts).strip(),
            metadata=final_metadata,
        )

    @staticmethod
    def _needs_current_sources(query: str) -> bool:
        return bool(re.search(r"\b(latest|newest|recent|today|current|paper|study|research)\b", query, re.IGNORECASE))

    @staticmethod
    def _requested_source_limit(query: str) -> int:
        match = re.search(
            r"\b(?:find|review|compare|list)?\s*(\d{1,2})\s+(?:papers?|studies|sources?|options?|cars?|products?)\b",
            query,
            re.IGNORECASE,
        )
        if match:
            return max(8, min(int(match.group(1)), 40))
        return 20 if re.search(r"\b(literature review|systematic review|options?|compare|shortlist)\b", query, re.IGNORECASE) else 14

    @staticmethod
    def _source_list(sources: list[dict[str, Any]]) -> str:
        lines = ["## Sources"]
        for index, source in enumerate(sources, start=1):
            if not source.get("url"):
                continue
            title = str(source.get("title") or source.get("url") or "Source").replace("[", "(").replace("]", ")")
            date = str(source.get("published_at") or "date not supplied")[:10]
            lines.append(f"{index}. [{title}]({source.get('url')}) — {date} · {source.get('type', 'web')}")
        return "\n".join(lines)

    def status(self, *, project: str = "default") -> dict[str, Any]:
        """Return the research engine status."""
        recent_research = self.store.search_events(
            "research", project=project, limit=5
        )
        return {
            "engine": "openagent-research",
            "project": project,
            "recent_research_count": len(recent_research),
            "gateway_provider": "auto",
            "runtime_profile": smooth_runtime_profile(task="research").to_dict(),
            "capabilities": [
                "query_planning",
                "local_context_search",
                "iterative_analysis",
                "source_synthesis",
            ],
        }
