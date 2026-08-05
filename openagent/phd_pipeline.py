"""Automatic local multi-model drafting and verification for PhD materials."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import re
import time
from typing import Any

from .model_gateway import ModelGateway, ModelResult


@dataclass(frozen=True)
class PhDPipelineResult:
    content: str
    provider: str
    model: str
    trace: dict[str, Any]
    metadata: dict[str, Any]


def run_phd_writing_pipeline(
    gateway: ModelGateway,
    messages: list[dict[str, str]],
    *,
    artifact_type: str,
    evidence_text: str,
    target_words: tuple[int, int] = (300, 900),
    provider: str = "auto",
    model: str | None = None,
    allow_external: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> PhDPipelineResult:
    """Draft, audit, revise, and independently proofread local artifacts."""

    routes = _candidate_routes(gateway, provider=provider, model=model, allow_external=allow_external)
    writer = routes[0]
    reviewer = routes[1] if len(routes) > 1 else writer
    proofreader = routes[2] if len(routes) > 2 else reviewer
    stages: list[dict[str, Any]] = []
    document_budget = _document_output_budget(target_words)
    _emit_progress(
        progress,
        stage="routing",
        status="completed",
        percent=5,
        message=f"Selected {len(routes)} local model route{'s' if len(routes) != 1 else ''} for writing and review.",
        route=writer,
    )

    _emit_progress(progress, stage="draft", status="running", percent=10, message="Drafting from verified evidence.", route=writer)
    draft = _complete(
        gateway,
        messages,
        writer,
        allow_external=allow_external,
        max_output_tokens=document_budget,
    )
    stages.append(_stage("draft", draft, writer))
    _emit_progress(progress, stage="draft", status="completed", percent=28, message="Evidence-grounded draft completed.", route=writer, result=draft)
    draft_text = _clean_document(draft.content)
    if _is_fallback(draft) or len(draft_text) < 80:
        _emit_progress(progress, stage="complete", status="degraded", percent=100, message="Model draft was unavailable; local fallback material was retained.")
        return PhDPipelineResult(
            content=draft_text,
            provider=draft.provider,
            model=draft.model,
            trace=_trace(routes, stages, finalized=False),
            metadata=draft.metadata,
        )

    evidence_excerpt = evidence_text[:32_000]
    review_messages = [
        {
            "role": "system",
            "content": (
                "Act as a rigorous academic application reviewer. Audit the draft against the supplied evidence. "
                "Identify unsupported claims, missing position-specific details, weak fit, repetition, vague language, "
                "and document-structure problems. Do not reveal hidden reasoning and do not rewrite the document. "
                "Evidence IDs and source labels are internal metadata and must never appear in the submitted document. "
                "Return a concise revision brief with headings: FACT CHECK, POSITION FIT, STRUCTURE, VOICE, REQUIRED REVISIONS."
            ),
        },
        {
            "role": "user",
            "content": f"DOCUMENT TYPE: {artifact_type}\n\nEVIDENCE:\n{evidence_excerpt}\n\nDRAFT:\n{draft_text[:28_000]}",
        },
    ]
    try:
        _emit_progress(progress, stage="evidence-review", status="running", percent=34, message="A second model is checking claims and position fit.", route=reviewer)
        review = _complete(
            gateway,
            review_messages,
            reviewer,
            allow_external=allow_external,
            max_output_tokens=384,
        )
        review_text = review.content.strip()
        stages.append(_stage("review", review, reviewer))
        _emit_progress(progress, stage="evidence-review", status="completed", percent=46, message="Evidence and fit review completed.", route=reviewer, result=review)
    except Exception as exc:
        review = None
        review_text = f"Reviewer unavailable: {type(exc).__name__}"
        stages.append({"stage": "review", "status": "degraded", "error": type(exc).__name__})
        _emit_progress(progress, stage="evidence-review", status="degraded", percent=46, message="Independent evidence review was unavailable.")

    if review is None or _is_fallback(review) or len(review_text) < 50:
        _emit_progress(progress, stage="complete", status="degraded", percent=100, message="Draft retained because an independent review could not be completed.")
        return PhDPipelineResult(
            content=draft_text,
            provider=draft.provider,
            model=draft.model,
            trace=_trace(routes, stages, finalized=False),
            metadata=draft.metadata,
        )

    final_messages = [
        {
            "role": "system",
            "content": (
                "You are the final academic application editor. Revise the draft using the review brief while remaining "
                "strictly grounded in the evidence. Preserve every accurate personal fact, remove unsupported claims, "
                "make the fit specific, and improve organization and natural professional voice. Never invent missing facts; "
                "use [NEEDS INPUT: ...] only when a required fact cannot be supported. Remove every evidence ID, source ID, "
                "profile marker, and position-URL marker from the prose. End the document cleanly and return only the "
                f"finished {target_words[0]}-{target_words[1]} word document."
            ),
        },
        {
            "role": "user",
            "content": (
                f"DOCUMENT TYPE: {artifact_type}\n\nEVIDENCE:\n{evidence_excerpt}\n\n"
                f"DRAFT:\n{draft_text[:24_000]}\n\nREVIEW BRIEF:\n{review_text[:12_000]}"
            ),
        },
    ]
    try:
        _emit_progress(progress, stage="revision", status="running", percent=52, message="Revising against the independent review.", route=writer)
        final = _complete(
            gateway,
            final_messages,
            writer,
            allow_external=allow_external,
            max_output_tokens=document_budget,
        )
        final_text = _clean_document(final.content)
        stages.append(_stage("finalize", final, writer))
        _emit_progress(progress, stage="revision", status="completed", percent=66, message="Reviewed revision completed.", route=writer, result=final)
    except Exception as exc:
        final = None
        final_text = ""
        stages.append({"stage": "finalize", "status": "degraded", "error": type(exc).__name__})
        _emit_progress(progress, stage="revision", status="degraded", percent=66, message="Revision model was unavailable.")

    if final is not None and not _is_fallback(final) and not _length_in_range(final_text, target_words):
        repair_low = min(target_words[1], max(target_words[0], math.ceil(target_words[0] * 1.08)))
        repair_high = min(target_words[1], max(repair_low, math.ceil(target_words[0] * 1.35)))
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are the final length and completeness editor for an academic application document. "
                    f"Return a complete document between {repair_low} and {repair_high} words. "
                    "Expand only with concrete, evidence-supported explanation when the draft is short; condense repetition "
                    "when it is long. Preserve verified names and facts, never invent achievements, never expose source or "
                    "evidence identifiers, and retain a clean professional ending. Return only the finished document."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT TYPE: {artifact_type}\n\nEVIDENCE:\n{evidence_excerpt}\n\n"
                    f"DOCUMENT TO REPAIR:\n{final_text[:28_000]}"
                ),
            },
        ]
        try:
            _emit_progress(progress, stage="length-check", status="running", percent=70, message="Repairing document length without adding unsupported claims.", route=writer)
            repaired = _complete(
                gateway,
                repair_messages,
                writer,
                allow_external=allow_external,
                max_output_tokens=document_budget,
            )
            repaired_text = _clean_document(repaired.content)
            stages.append(_stage("length-repair", repaired, writer))
            if not _is_fallback(repaired) and _document_is_complete(repaired_text, target_words):
                final = repaired
                final_text = repaired_text
            _emit_progress(progress, stage="length-check", status="completed", percent=78, message="Length and completeness check finished.", route=writer, result=repaired)
        except Exception as exc:
            stages.append({"stage": "length-repair", "status": "degraded", "error": type(exc).__name__})
            _emit_progress(progress, stage="length-check", status="degraded", percent=78, message="Length repair was unavailable.")

    proofread_passed = False
    if final is not None and not _is_fallback(final) and _document_is_complete(final_text, target_words):
        proofread_passed, final, final_text = _proofread_and_repair(
            gateway,
            artifact_type=artifact_type,
            evidence_excerpt=evidence_excerpt,
            content=final_text,
            current=final,
            writer=writer,
            proofreader=proofreader,
            allow_external=allow_external,
            max_output_tokens=document_budget,
            target_words=target_words,
            stages=stages,
            progress=progress,
        )

    if final is None or _is_fallback(final) or not _document_is_complete(final_text, target_words) or not proofread_passed:
        best = final_text if len(final_text.split()) > len(draft_text.split()) else draft_text
        _emit_progress(progress, stage="complete", status="degraded", percent=100, message="A usable draft was saved, but the independent final quality gate did not pass.")
        return PhDPipelineResult(
            content=best,
            provider=draft.provider,
            model=draft.model,
            trace=_trace(routes, stages, finalized=False),
            metadata=draft.metadata,
        )
    _emit_progress(progress, stage="complete", status="completed", percent=100, message="Independent proofreading passed; final document is ready.", route=(final.provider, final.model), result=final)
    return PhDPipelineResult(
        content=final_text,
        provider=final.provider,
        model=final.model,
        trace=_trace(routes, stages, finalized=True),
        metadata=final.metadata,
    )


def _proofread_and_repair(
    gateway: Any,
    *,
    artifact_type: str,
    evidence_excerpt: str,
    content: str,
    current: ModelResult,
    writer: tuple[str, str | None],
    proofreader: tuple[str, str | None],
    allow_external: bool,
    max_output_tokens: int,
    target_words: tuple[int, int],
    stages: list[dict[str, Any]],
    progress: Callable[[dict[str, Any]], None] | None,
) -> tuple[bool, ModelResult, str]:
    review_messages = _proofread_messages(artifact_type, evidence_excerpt, content)
    _emit_progress(
        progress,
        stage="proofread",
        status="running",
        percent=82,
        message="An independent model is proofreading facts, fit, language, and submission readiness.",
        route=proofreader,
    )
    try:
        proofread = _complete(
            gateway,
            review_messages,
            proofreader,
            allow_external=allow_external,
            max_output_tokens=420,
        )
        proofread_text = proofread.content.strip()
        stages.append(_stage("proofread", proofread, proofreader))
    except Exception as exc:
        stages.append({"stage": "proofread", "status": "degraded", "error": type(exc).__name__})
        _emit_progress(progress, stage="proofread", status="degraded", percent=90, message="Independent final proofreading was unavailable.")
        return False, current, content
    if _is_fallback(proofread) or len(proofread_text) < 30:
        _emit_progress(progress, stage="proofread", status="degraded", percent=90, message="The proofreader did not return a reliable quality verdict.")
        return False, current, content
    if not _proofread_requires_revision(proofread_text):
        _emit_progress(progress, stage="proofread", status="completed", percent=96, message="Independent proofreader passed the document.", route=proofreader, result=proofread)
        return True, current, content

    _emit_progress(progress, stage="proofread-repair", status="running", percent=90, message="Applying the proofreader's required corrections.", route=writer)
    repair_messages = [
        {
            "role": "system",
            "content": (
                "Act as the final academic application editor. Apply every valid correction in the independent "
                "proofread brief. Keep the document strictly grounded in supplied evidence, preserve verified facts, "
                "remove unsupported or generic claims, and return only the finished document. Do not add source IDs, "
                "analysis, commentary, or claims about bypassing authorship detection."
            ),
        },
        {
            "role": "user",
            "content": (
                f"DOCUMENT TYPE: {artifact_type}\n\nEVIDENCE:\n{evidence_excerpt}\n\n"
                f"DOCUMENT:\n{content[:28_000]}\n\nINDEPENDENT PROOFREAD:\n{proofread_text[:12_000]}"
            ),
        },
    ]
    try:
        repaired = _complete(
            gateway,
            repair_messages,
            writer,
            allow_external=allow_external,
            max_output_tokens=max_output_tokens,
        )
        repaired_text = _clean_document(repaired.content)
        stages.append(_stage("proofread-repair", repaired, writer))
    except Exception as exc:
        stages.append({"stage": "proofread-repair", "status": "degraded", "error": type(exc).__name__})
        _emit_progress(progress, stage="proofread-repair", status="degraded", percent=95, message="The final correction pass was unavailable.")
        return False, current, content
    if _is_fallback(repaired) or not _document_is_complete(repaired_text, target_words):
        _emit_progress(progress, stage="proofread-repair", status="degraded", percent=95, message="The corrected document did not pass deterministic completeness checks.")
        return False, current, content

    _emit_progress(progress, stage="proofread-confirmation", status="running", percent=96, message="Confirming the repaired document with the independent model.", route=proofreader)
    try:
        confirmation = _complete(
            gateway,
            _proofread_messages(artifact_type, evidence_excerpt, repaired_text),
            proofreader,
            allow_external=allow_external,
            max_output_tokens=320,
        )
        confirmation_text = confirmation.content.strip()
        stages.append(_stage("proofread-confirmation", confirmation, proofreader))
    except Exception as exc:
        stages.append({"stage": "proofread-confirmation", "status": "degraded", "error": type(exc).__name__})
        _emit_progress(progress, stage="proofread-confirmation", status="degraded", percent=98, message="The confirmation review was unavailable.")
        return False, repaired, repaired_text
    passed = not _is_fallback(confirmation) and len(confirmation_text) >= 30 and not _proofread_requires_revision(confirmation_text)
    _emit_progress(
        progress,
        stage="proofread-confirmation",
        status="completed" if passed else "degraded",
        percent=99,
        message="Independent confirmation passed." if passed else "Independent confirmation still found required changes.",
        route=proofreader,
        result=confirmation,
    )
    return passed, repaired, repaired_text


def _proofread_messages(artifact_type: str, evidence_excerpt: str, content: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an independent final proofreader for a doctoral application. Check the submitted document "
                "against the evidence for unsupported claims, incorrect names, weak position specificity, missing "
                "requirements, grammar, repetition, generic phrasing, incomplete endings, and inappropriate source "
                "markers. Do not rewrite the document and do not reveal hidden reasoning. Start with exactly "
                "VERDICT: PASS or VERDICT: REVISE, then give concise sections FACTS, FIT, LANGUAGE, and REQUIRED FIXES. "
                "Judge authentic authorship by evidence and the applicant's stated voice, never by trying to evade an "
                "AI detector."
            ),
        },
        {
            "role": "user",
            "content": f"DOCUMENT TYPE: {artifact_type}\n\nEVIDENCE:\n{evidence_excerpt}\n\nFINAL DOCUMENT:\n{content[:30_000]}",
        },
    ]


def _proofread_requires_revision(content: str) -> bool:
    normalized = " ".join(content.lower().split())
    if re.search(r"\bverdict\s*:\s*revise\b", normalized):
        return True
    if re.search(r"\bverdict\s*:\s*pass\b", normalized):
        return False
    return any(
        marker in normalized
        for marker in (
            "unsupported claim",
            "incorrect name",
            "must be revised",
            "required fixes: revise",
            "required fixes: correct",
            "required fixes: remove",
        )
    )


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str,
    status: str,
    percent: int,
    message: str,
    route: tuple[str, str | None] | None = None,
    result: ModelResult | None = None,
) -> None:
    if callback is None:
        return
    event: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "percent": max(0, min(100, int(percent))),
        "message": message,
        "at_unix": time.time(),
    }
    if route is not None:
        event["requested_provider"] = route[0]
        event["requested_model"] = route[1] or ""
    if result is not None:
        event["provider"] = result.provider
        event["model"] = result.model
        event["latency_ms"] = result.metadata.get("latency_ms")
    try:
        callback(event)
    except Exception:
        return


def _candidate_routes(
    gateway: Any,
    *,
    provider: str,
    model: str | None,
    allow_external: bool,
) -> list[tuple[str, str | None]]:
    requested = provider.strip() or "auto"
    routes: list[tuple[str, str | None]] = []
    if requested != "auto" and model:
        routes.append((requested, model))
    discovered: list[dict[str, Any]] = []
    discover = getattr(gateway, "discover_models", None)
    if callable(discover):
        try:
            discovered = [item for item in discover() if item.get("available")]
        except Exception:
            discovered = []
    priority = {"ollama": 0, "openvino-direct": 1, "llamacpp": 2, "local": 9}
    discovered.sort(key=lambda item: priority.get(str(item.get("provider", "")), 7))
    deterministic_local: tuple[str, str | None] | None = None
    for item in discovered:
        route_provider = str(item.get("provider", "")).strip()
        route_model = str(item.get("model", "")).strip() or None
        if not route_provider or route_provider == "openai-compatible":
            continue
        route = (route_provider, route_model)
        if route_provider == "local" and requested != "local":
            deterministic_local = route
            continue
        if route not in routes:
            routes.append(route)
    if requested != "auto" and not model:
        matching = next((route for route in routes if route[0] == requested), None)
        if matching is not None:
            routes.remove(matching)
            routes.insert(0, matching)
        else:
            routes.insert(0, (requested, None))
    if requested == "openai-compatible" and allow_external and (requested, model) not in routes:
        routes.insert(0, (requested, model))
    if not routes:
        routes.append(deterministic_local or (requested, model))
    return routes[:3]


def _complete(
    gateway: Any,
    messages: list[dict[str, str]],
    route: tuple[str, str | None],
    *,
    allow_external: bool,
    max_output_tokens: int | None = None,
) -> ModelResult:
    route_provider, route_model = route
    stream = getattr(gateway, "stream_complete", None)
    if callable(stream):
        started = time.perf_counter()
        parts: list[str] = []
        result_provider = route_provider
        result_model = route_model or ""
        metadata: dict[str, Any] = {}
        fallback_seen = False
        kwargs = {
            "provider_id": route_provider,
            "model": route_model,
            "allow_external": allow_external and route_provider == "openai-compatible",
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        try:
            chunks = stream(messages, **kwargs)
        except TypeError as exc:
            if "max_output_tokens" not in str(exc):
                raise
            kwargs.pop("max_output_tokens", None)
            chunks = stream(messages, **kwargs)
        for chunk in chunks:
            chunk_metadata = dict(getattr(chunk, "metadata", {}) or {})
            if chunk_metadata.get("fallback_from") and not fallback_seen:
                parts.clear()
                fallback_seen = True
            content = str(getattr(chunk, "content", "") or "")
            if content:
                parts.append(content)
            result_provider = str(getattr(chunk, "provider", result_provider) or result_provider)
            result_model = str(getattr(chunk, "model", result_model) or result_model)
            metadata = chunk_metadata
        metadata.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 3))
        return ModelResult(result_provider, result_model, "".join(parts), metadata)
    kwargs = {
        "provider_id": route_provider,
        "model": route_model,
        "allow_external": allow_external and route_provider == "openai-compatible",
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    try:
        return gateway.complete(messages, **kwargs)
    except TypeError as exc:
        if "max_output_tokens" not in str(exc):
            raise
        kwargs.pop("max_output_tokens", None)
        return gateway.complete(messages, **kwargs)


def _document_output_budget(target_words: tuple[int, int]) -> int:
    minimum = max(120, int(target_words[0]))
    return min(1536, max(512, int(minimum * 1.55)))


def _clean_document(content: str) -> str:
    """Remove internal retrieval notation while preserving author-visible gaps."""

    text = content.strip()
    text = re.sub(r"\s*\((?:evidence_id\s*=\s*[^)]+|\[?profile\]?|source_id\s*=\s*[^)]+)\)", "", text, flags=re.I)
    text = re.sub(
        r"\s*,?\s*as\s+(?:advertised|outlined|described)\s+(?:by|at|in)\s+\[(?:position[ _]?url|source)\]",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\[(?:position[ _]?url|profile|evidence[ _]?id|source[ _]?id)\]", "", text, flags=re.I)
    text = re.sub(r"\bevidence_id\s*=\s*[\w-]+", "", text, flags=re.I)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _document_is_complete(content: str, target_words: tuple[int, int]) -> bool:
    if not _length_in_range(content, target_words):
        return False
    ending = content.rstrip().lower()
    return ending.endswith((".", "!", "?", "sincerely,", "best regards,")) or any(
        marker in ending[-180:] for marker in ("sincerely", "best regards", "kind regards", "respectfully")
    )


def _length_in_range(content: str, target_words: tuple[int, int]) -> bool:
    words = len(re.findall(r"\b[\w'-]+\b", content))
    return max(1, int(target_words[0])) <= words <= max(int(target_words[0]), int(target_words[1]))


def _is_fallback(result: ModelResult) -> bool:
    return bool(result.metadata.get("deterministic_fallback"))


def _stage(name: str, result: ModelResult, route: tuple[str, str | None]) -> dict[str, Any]:
    return {
        "stage": name,
        "status": "completed",
        "requested_provider": route[0],
        "requested_model": route[1] or "",
        "provider": result.provider,
        "model": result.model,
        "latency_ms": result.metadata.get("latency_ms"),
        "cache_hit": bool(result.metadata.get("cache_hit")),
    }


def _trace(
    routes: list[tuple[str, str | None]],
    stages: list[dict[str, Any]],
    *,
    finalized: bool,
) -> dict[str, Any]:
    completed_models = {
        (str(stage.get("provider", "")), str(stage.get("model", "")))
        for stage in stages
        if stage.get("status") == "completed"
    }
    role_routes = {
        "writer": routes[0] if routes else ("", None),
        "evidence_reviewer": routes[1] if len(routes) > 1 else routes[0] if routes else ("", None),
        "final_proofreader": routes[2] if len(routes) > 2 else routes[1] if len(routes) > 1 else routes[0] if routes else ("", None),
    }
    return {
        "strategy": "local-draft-review-revise-proofread-v2",
        "routes": [{"provider": item[0], "model": item[1] or ""} for item in routes],
        "roles": {
            role: {"provider": route[0], "model": route[1] or ""}
            for role, route in role_routes.items()
        },
        "stages": stages,
        "stage_count": len(stages),
        "distinct_models": len(completed_models),
        "multi_model": len(completed_models) > 1,
        "quality_gate": "passed" if finalized else "not-passed",
        "finalized": finalized,
        "privacy_boundary": "All stages use local providers unless an external writer was explicitly approved for this generation run.",
    }
