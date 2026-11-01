"""G25: Advanced reasoning - multi-turn chains, complex analysis, semantic understanding."""

from __future__ import annotations

import logging
import json
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ReasoningChain:
    """Multi-step reasoning chain."""
    def __init__(self, chain_id: str, topic: str):
        self.id = chain_id
        self.topic = topic
        self.steps = []
        self.conclusions = []

    def add_step(self, step_type: str, input_data: str, output_data: str) -> None:
        """Add reasoning step."""
        self.steps.append({
            "type": step_type,
            "input": input_data,
            "output": output_data,
            "step_num": len(self.steps) + 1,
        })

    def add_conclusion(self, conclusion: str) -> None:
        """Add conclusion."""
        self.conclusions.append(conclusion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "steps": self.steps,
            "conclusions": self.conclusions,
            "depth": len(self.steps),
        }


class AdvancedReasoningService:
    """Multi-turn reasoning and semantic understanding."""

    def __init__(self):
        self.chains: dict[str, ReasoningChain] = {}
        self.memory_store = {}

    def create_reasoning_chain(self, topic: str) -> dict[str, Any]:
        """Create new reasoning chain."""
        chain_id = str(uuid4())
        chain = ReasoningChain(chain_id, topic)
        self.chains[chain_id] = chain
        logger.info(f"Created reasoning chain: {topic}")
        return {"chain_id": chain_id, "topic": topic}

    def add_reasoning_step(self, chain_id: str, step_type: str, input_data: str, output_data: str) -> dict[str, Any]:
        """Add step to reasoning chain."""
        if chain_id not in self.chains:
            return {"error": "Chain not found"}
        
        chain = self.chains[chain_id]
        chain.add_step(step_type, input_data, output_data)
        
        return {
            "chain_id": chain_id,
            "step_num": len(chain.steps),
            "step_type": step_type,
        }

    def analyze_semantically(self, text: str) -> dict[str, Any]:
        """Semantic analysis of text."""
        words = text.split()
        sentences = text.split(".")
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "entities": self._extract_entities(text),
            "concepts": self._extract_concepts(text),
            "sentiment": self._analyze_sentiment(text),
        }

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities."""
        import re
        # Simple: find capitalized words
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
        return list(set(entities))[:5]

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts."""
        words = text.lower().split()
        stop_words = {"the", "a", "an", "and", "or", "in", "is", "to", "of", "for", "with"}
        concepts = [w for w in words if w not in stop_words and len(w) > 4]
        
        from collections import Counter
        counts = Counter(concepts)
        return [c for c, _ in counts.most_common(5)]

    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis."""
        positive_words = {"good", "great", "excellent", "amazing", "wonderful"}
        negative_words = {"bad", "terrible", "awful", "horrible", "poor"}
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def complex_reasoning(self, query: str, context: str) -> dict[str, Any]:
        """Complex multi-step reasoning."""
        chain = ReasoningChain(str(uuid4()), query)
        
        # Step 1: Analyze question
        chain.add_step("question_analysis", query, f"Analyzing: {query}")
        
        # Step 2: Extract context
        chain.add_step("context_extraction", context, f"Context length: {len(context)}")
        
        # Step 3: Semantic analysis
        analysis = self.analyze_semantically(context)
        chain.add_step("semantic_analysis", context, json.dumps(analysis, default=str))
        
        # Step 4: Reasoning
        reasoning = f"Based on {analysis['concept_count'] if 'concept_count' in analysis else 'the'} key concepts..."
        chain.add_step("reasoning", query, reasoning)
        
        # Step 5: Conclusion
        conclusion = f"Comprehensive analysis of '{query}' suggests multiple interpretations."
        chain.add_conclusion(conclusion)
        
        return chain.to_dict()

    def get_chain(self, chain_id: str) -> dict[str, Any] | None:
        """Get reasoning chain."""
        return self.chains.get(chain_id).to_dict() if chain_id in self.chains else None

    def list_chains(self) -> list[dict[str, Any]]:
        """List all reasoning chains."""
        return [chain.to_dict() for chain in self.chains.values()]

    def store_memory(self, key: str, value: Any) -> None:
        """Store in long-term memory."""
        self.memory_store[key] = value
        logger.info(f"Stored memory: {key}")

    def retrieve_memory(self, key: str) -> Any | None:
        """Retrieve from memory."""
        return self.memory_store.get(key)

    def create_chain(self, topic: str) -> dict[str, Any]:
        """Alias: create_reasoning_chain (for route compatibility)."""
        return self.create_reasoning_chain(topic)

    def analyze_semantics(self, text: str) -> dict[str, Any]:
        """Alias: analyze_semantically (for route compatibility)."""
        return self.analyze_semantically(text)

    def get_reasoning_stats(self) -> dict[str, Any]:
        """Get reasoning statistics."""
        total_steps = sum(len(chain.steps) for chain in self.chains.values())
        
        return {
            "total_chains": len(self.chains),
            "total_steps": total_steps,
            "avg_chain_depth": total_steps / max(len(self.chains), 1),
            "memory_items": len(self.memory_store),
        }
