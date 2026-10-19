"""Digital Twin — Personal Operating Memory and Predictive Intelligence.

The Digital Twin continuously learns from every user interaction and builds a
living model of the user: their preferences, behaviour patterns, goals,
communication style, and predicted next actions.

Key subsystems
--------------
* **Profile** — persistent user model: interests, style, preferred hours, tools.
* **Interaction recorder** — every chat, task, search, and file-open is recorded.
* **Preference learner** — derives and updates weighted preferences from interactions.
* **Behaviour analyser** — detects usage patterns (peak hours, frequent topics, etc.).
* **Intent predictor** — suggests the most likely next action given current context.
* **Proactive engine** — surfaces time-sensitive suggestions before the user asks.
* **Memory consolidator** — periodically compresses episodic memory into semantic facts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hour_bin(iso: str) -> int:
    """Return the hour-of-day (0-23) for an ISO timestamp."""
    try:
        return datetime.fromisoformat(iso).hour
    except ValueError:
        return 12


def _day_of_week(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%A")
    except ValueError:
        return "Unknown"


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse weighted-feature dicts."""
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(v ** 2 for v in a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _extract_topics(text: str) -> list[str]:
    """Extract key topic words from free text (stopword-filtered)."""
    STOP = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "it", "its", "this", "that", "i", "you", "we", "they", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "as", "be", "do",
        "have", "has", "had", "not", "can", "will", "would", "could", "should",
        "please", "help", "what", "how", "when", "where", "why", "who",
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in STOP][:20]


# ---------------------------------------------------------------------------
# DigitalTwin
# ---------------------------------------------------------------------------

class DigitalTwin:
    """Personal operating memory — learns from every interaction."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.initialize()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.store.initialize()

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def get_profile(self, *, project: str = "default") -> dict[str, Any]:
        """Return the digital twin profile for a project, creating it if needed."""
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT id,profile_json,version,created_at,updated_at FROM twin_profiles WHERE project=?",
                (project,),
            ).fetchone()
        if not row:
            return self._create_profile(project)
        return {
            "id": row[0],
            "project": project,
            "profile": json.loads(row[1] or "{}"),
            "version": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }

    def _create_profile(self, project: str) -> dict[str, Any]:
        now = utc_now()
        profile_id = hashlib.md5(f"twin-{project}".encode()).hexdigest()
        default_profile: dict[str, Any] = {
            "name": "User",
            "preferred_language": "en",
            "preferred_hours": {"start": 9, "end": 18},
            "communication_style": "balanced",
            "expertise_areas": [],
            "interests": [],
            "work_mode": "deep-focus",
            "model_preference": "local",
            "privacy_level": "strict",
        }
        with self.store.session() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO twin_profiles VALUES(?,?,?,?,?,?)",
                (profile_id, project, json.dumps(default_profile), 1, now, now),
            )
        return {
            "id": profile_id, "project": project,
            "profile": default_profile, "version": 1,
            "created_at": now, "updated_at": now,
        }

    def update_profile(
        self, updates: dict[str, Any], *, project: str = "default"
    ) -> dict[str, Any]:
        existing = self.get_profile(project=project)
        profile = existing["profile"]
        profile.update(updates)
        now = utc_now()
        with self.store.session() as conn:
            conn.execute(
                "UPDATE twin_profiles SET profile_json=?, version=version+1, updated_at=? WHERE project=?",
                (json.dumps(profile), now, project),
            )
        return self.get_profile(project=project)

    # ------------------------------------------------------------------
    # Interaction recording
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        kind: str,
        content: str = "",
        *,
        project: str = "default",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a user interaction and update learned preferences."""
        now = utc_now()
        interaction_id = uuid4().hex
        topics = _extract_topics(content)
        hour = _hour_bin(now)
        dow = _day_of_week(now)
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO twin_interactions VALUES(?,?,?,?,?,?,?,?,?)",
                (interaction_id, project, kind, content[:1000],
                 json.dumps(topics), hour, dow,
                 json.dumps(context or {}), now),
            )
        # Update topic preferences
        self._update_topic_weights(topics, kind=kind, project=project)
        return {"id": interaction_id, "kind": kind, "topics": topics}

    def list_sources(self, *, project: str = "default", limit: int = 100) -> list[dict[str, Any]]:
        """List provenance event metadata without exposing captured interaction content."""
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT id, kind, topics_json, created_at FROM twin_interactions "
                "WHERE project=? ORDER BY created_at DESC LIMIT ?",
                (project, max(1, min(int(limit), 500))),
            ).fetchall()
        result = []
        for row in rows:
            try:
                topics = json.loads(row[2] or "[]")
            except json.JSONDecodeError:
                topics = []
            result.append({"id": row[0], "kind": row[1], "topics": topics, "created_at": row[3]})
        return result

    def _update_topic_weights(
        self, topics: list[str], *, kind: str, project: str
    ) -> None:
        now = utc_now()
        weight_boost = 1.5 if kind in ("chat", "research", "search") else 1.0
        for topic in topics:
            pref_id = hashlib.md5(f"topic-{project}-{topic}".encode()).hexdigest()
            with self.store.session() as conn:
                existing = conn.execute(
                    "SELECT id, weight FROM twin_preferences WHERE project=? AND category='topic' AND key=?",
                    (project, topic),
                ).fetchone()
                if existing:
                    new_weight = min(10.0, existing[1] + weight_boost * 0.1)
                    conn.execute(
                        "UPDATE twin_preferences SET weight=?, updated_at=? WHERE id=?",
                        (new_weight, now, existing[0]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO twin_preferences VALUES(?,?,?,?,?,?,?,?,?)",
                        (pref_id, project, "topic", topic, topic, weight_boost, "inferred", now, now),
                    )

    # ------------------------------------------------------------------
    # Behaviour analysis
    # ------------------------------------------------------------------

    def analyse_patterns(self, *, project: str = "default") -> dict[str, Any]:
        """Analyse interaction history and extract behaviour patterns."""
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT kind, hour_of_day, day_of_week, topics_json, created_at "
                "FROM twin_interactions WHERE project=? ORDER BY created_at DESC LIMIT 500",
                (project,),
            ).fetchall()

        if not rows:
            return {"patterns": [], "peak_hours": [], "top_topics": [], "activity_by_day": {}}

        hour_counter: Counter[int] = Counter()
        day_counter: Counter[str] = Counter()
        topic_counter: Counter[str] = Counter()
        kind_counter: Counter[str] = Counter()

        for row in rows:
            kind_counter[row[0]] += 1
            hour_counter[row[1]] += 1
            day_counter[row[2]] += 1
            try:
                topics = json.loads(row[3] or "[]")
                topic_counter.update(topics)
            except json.JSONDecodeError:
                pass

        peak_hours = [h for h, _ in hour_counter.most_common(3)]
        top_topics = [t for t, _ in topic_counter.most_common(10)]
        activity_by_day = dict(day_counter.most_common())

        patterns = []
        if peak_hours:
            patterns.append({
                "type": "peak_hours",
                "description": f"Most active at hours {peak_hours}",
                "data": peak_hours,
            })
        if top_topics:
            patterns.append({
                "type": "top_topics",
                "description": f"Frequent topics: {', '.join(top_topics[:5])}",
                "data": top_topics,
            })
        dominant_kind = kind_counter.most_common(1)[0][0] if kind_counter else "chat"
        patterns.append({
            "type": "dominant_activity",
            "description": f"Primary activity: {dominant_kind}",
            "data": dict(kind_counter.most_common(5)),
        })

        # Persist patterns
        now = utc_now()
        pattern_id = hashlib.md5(f"pattern-{project}-{now[:10]}".encode()).hexdigest()
        with self.store.session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO twin_patterns VALUES(?,?,?,?,?,?,?)",
                (pattern_id, project, "behaviour_snapshot",
                 json.dumps({"peak_hours": peak_hours, "top_topics": top_topics,
                             "activity_by_day": activity_by_day}),
                 0.8, now, now),
            )

        return {
            "patterns": patterns,
            "peak_hours": peak_hours,
            "top_topics": top_topics,
            "activity_by_day": activity_by_day,
            "total_interactions": len(rows),
        }

    # ------------------------------------------------------------------
    # Intent prediction
    # ------------------------------------------------------------------

    def predict_intent(
        self, context_text: str, *, project: str = "default"
    ) -> dict[str, Any]:
        """Predict the most likely user intent given the current context text."""
        topics = _extract_topics(context_text)
        # Get user top preferences
        with self.store.session() as conn:
            pref_rows = conn.execute(
                "SELECT key, weight FROM twin_preferences "
                "WHERE project=? AND category='topic' ORDER BY weight DESC LIMIT 20",
                (project,),
            ).fetchall()
            recent_rows = conn.execute(
                "SELECT kind, topics_json FROM twin_interactions "
                "WHERE project=? ORDER BY created_at DESC LIMIT 10",
                (project,),
            ).fetchall()

        pref_weights = {r[0]: r[1] for r in pref_rows}
        recent_kinds: list[str] = []
        recent_topics: list[str] = []
        for row in recent_rows:
            recent_kinds.append(row[0])
            try:
                recent_topics.extend(json.loads(row[1] or "[]"))
            except json.JSONDecodeError:
                pass

        # Score possible intents
        intent_scores: dict[str, float] = {
            "search_knowledge": 0.0,
            "start_chat": 0.1,
            "create_task": 0.0,
            "run_research": 0.0,
            "write_document": 0.0,
            "run_workflow": 0.0,
            "check_calendar": 0.0,
        }
        query_words = {"what", "how", "why", "explain", "find", "search", "show"}
        task_words = {"create", "make", "add", "new", "build", "todo", "task", "plan"}
        research_words = {"research", "investigate", "analyze", "analyse", "deep", "study"}
        doc_words = {"write", "draft", "compose", "document", "article", "report"}
        cal_words = {"schedule", "meeting", "calendar", "appointment", "when", "time", "date"}

        lower = context_text.lower()
        for word in query_words:
            if word in lower:
                intent_scores["search_knowledge"] += 1.5
                intent_scores["start_chat"] += 0.5
        for word in task_words:
            if word in lower:
                intent_scores["create_task"] += 1.5
        for word in research_words:
            if word in lower:
                intent_scores["run_research"] += 2.0
        for word in doc_words:
            if word in lower:
                intent_scores["write_document"] += 1.5
        for word in cal_words:
            if word in lower:
                intent_scores["check_calendar"] += 1.5

        # Boost by user preferences
        topic_overlap = sum(pref_weights.get(t, 0) for t in topics)
        if topic_overlap > 0:
            intent_scores["search_knowledge"] += min(topic_overlap * 0.2, 2.0)

        # Recency bias
        if recent_kinds:
            last_kind = recent_kinds[0]
            boost_map = {
                "chat": "start_chat",
                "search": "search_knowledge",
                "research": "run_research",
                "task": "create_task",
                "workflow": "run_workflow",
            }
            mapped = boost_map.get(last_kind)
            if mapped:
                intent_scores[mapped] += 0.5

        # Normalize
        total = sum(intent_scores.values()) or 1.0
        normalized = {k: round(v / total, 3) for k, v in intent_scores.items()}
        top_intent = max(normalized, key=lambda k: normalized[k])
        confidence = normalized[top_intent]

        # Store prediction
        now = utc_now()
        pred_id = uuid4().hex
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO twin_predictions VALUES(?,?,?,?,?,?,?,?)",
                (pred_id, project, "intent", top_intent, confidence,
                 json.dumps({"topics": topics, "scores": normalized}), 0, now),
            )

        return {
            "predicted_intent": top_intent,
            "confidence": confidence,
            "all_scores": normalized,
            "context_topics": topics,
        }

    # ------------------------------------------------------------------
    # Proactive suggestions
    # ------------------------------------------------------------------

    def generate_proactive_suggestions(
        self, *, project: str = "default"
    ) -> list[dict[str, Any]]:
        """Return a list of proactive suggestions based on user state."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        suggestions: list[dict[str, Any]] = []

        patterns = self.analyse_patterns(project=project)
        peak_hours = patterns.get("peak_hours", [])
        top_topics = patterns.get("top_topics", [])

        # Morning brief
        if 7 <= hour <= 9:
            suggestions.append({
                "type": "morning_brief",
                "title": "Good morning — daily brief ready",
                "description": "Review your tasks, calendar, and overnight messages.",
                "action": "open_daily_brief",
                "priority": "high",
            })

        # Peak productivity window
        if peak_hours and hour in peak_hours:
            suggestions.append({
                "type": "focus_time",
                "title": "Peak focus window detected",
                "description": "This is typically your most productive time. Consider deep work.",
                "action": "start_focus_session",
                "priority": "medium",
            })

        # Topic-based suggestions
        for topic in top_topics[:3]:
            suggestions.append({
                "type": "knowledge_gap",
                "title": f"Continue exploring: {topic}",
                "description": f"Based on your interests, you may want to research more about {topic}.",
                "action": "run_research",
                "query": topic,
                "priority": "low",
            })

        # EOD wrap-up
        if 16 <= hour <= 18:
            suggestions.append({
                "type": "end_of_day",
                "title": "End-of-day wrap-up",
                "description": "Review completed tasks and plan for tomorrow.",
                "action": "daily_review",
                "priority": "medium",
            })

        return suggestions[:8]

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------

    def create_goal(
        self,
        title: str,
        *,
        project: str = "default",
        description: str = "",
        target_date: str = "",
    ) -> dict[str, Any]:
        now = utc_now()
        goal_id = uuid4().hex
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO twin_goals VALUES(?,?,?,?,?,?,?,?,?,?)",
                (goal_id, project, title, description, target_date,
                 "active", 0.0, "{}", now, now),
            )
        return self.get_goal(goal_id)

    def get_goal(self, goal_id: str) -> dict[str, Any]:
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT id,project,title,description,target_date,status,progress,metadata_json,created_at,updated_at "
                "FROM twin_goals WHERE id=?", (goal_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"Goal not found: {goal_id}")
        return {
            "id": row[0], "project": row[1], "title": row[2],
            "description": row[3], "target_date": row[4],
            "status": row[5], "progress": row[6],
            "metadata": json.loads(row[7] or "{}"),
            "created_at": row[8], "updated_at": row[9],
        }

    def list_goals(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id,project,title,description,target_date,status,progress,metadata_json,created_at,updated_at "
                "FROM twin_goals WHERE project=? ORDER BY created_at DESC",
                (project,),
            ).fetchall()
        return [
            {"id": r[0], "project": r[1], "title": r[2], "description": r[3],
             "target_date": r[4], "status": r[5], "progress": r[6],
             "metadata": json.loads(r[7] or "{}"), "created_at": r[8], "updated_at": r[9]}
            for r in rows
        ]

    def update_goal_progress(
        self, goal_id: str, progress: float, *, status: str = ""
    ) -> dict[str, Any]:
        now = utc_now()
        progress = max(0.0, min(1.0, progress))
        with self.store.session() as conn:
            conn.execute(
                "UPDATE twin_goals SET progress=?, updated_at=? WHERE id=?",
                (progress, now, goal_id),
            )
            if status:
                conn.execute(
                    "UPDATE twin_goals SET status=?, updated_at=? WHERE id=?",
                    (status, now, goal_id),
                )
        return self.get_goal(goal_id)

    # ------------------------------------------------------------------
    # Memory consolidation
    # ------------------------------------------------------------------

    def consolidate_memory(self, *, project: str = "default") -> dict[str, Any]:
        """Compress old episodic interactions into summary facts."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT kind, topics_json FROM twin_interactions "
                "WHERE project=? AND created_at < ? LIMIT 500",
                (project, cutoff),
            ).fetchall()
        if not rows:
            return {"consolidated": 0, "facts": []}

        topic_counter: Counter[str] = Counter()
        kind_counter: Counter[str] = Counter()
        for row in rows:
            kind_counter[row[0]] += 1
            try:
                topic_counter.update(json.loads(row[1] or "[]"))
            except json.JSONDecodeError:
                pass

        facts = [
            f"Frequently discussed: {', '.join(t for t, _ in topic_counter.most_common(5))}",
            f"Primary activities: {', '.join(k for k, _ in kind_counter.most_common(3))}",
        ]

        # Update profile with consolidated facts
        self.update_profile({
            "consolidated_interests": [t for t, _ in topic_counter.most_common(10)],
            "consolidated_activities": dict(kind_counter.most_common()),
        }, project=project)

        return {"consolidated": len(rows), "facts": facts}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as conn:
            n_interactions = conn.execute(
                "SELECT COUNT(*) FROM twin_interactions WHERE project=?", (project,)
            ).fetchone()[0]
            n_prefs = conn.execute(
                "SELECT COUNT(*) FROM twin_preferences WHERE project=?", (project,)
            ).fetchone()[0]
            n_goals = conn.execute(
                "SELECT COUNT(*) FROM twin_goals WHERE project=? AND status='active'", (project,)
            ).fetchone()[0]
            n_predictions = conn.execute(
                "SELECT COUNT(*) FROM twin_predictions WHERE project=?", (project,)
            ).fetchone()[0]
        return {
            "interactions_recorded": n_interactions,
            "preferences_learned": n_prefs,
            "active_goals": n_goals,
            "predictions_made": n_predictions,
            "project": project,
            "features": [
                "profile-management", "interaction-recording", "preference-learning",
                "behaviour-analysis", "intent-prediction", "proactive-suggestions",
                "goal-tracking", "memory-consolidation", "long-horizon-planning",
            ],
        }
