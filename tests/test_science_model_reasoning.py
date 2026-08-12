from __future__ import annotations

from dataclasses import dataclass
import json
import unittest

from openagent.science.agent import ScientificAgent


@dataclass
class _Result:
    content: str
    provider: str = "local-test"
    model: str = "science-test"


class _Gateway:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return _Result(self.content)


class ScientificModelReasoningTests(unittest.TestCase):
    def test_grounded_analysis_accepts_only_available_evidence(self) -> None:
        payload = {
            "summary": "Energy decreased during the calculation.",
            "claims": [{"text": "Final energy is lower.", "evidence_ids": ["result.energy_final"]}],
            "limitations": ["This is a numerical result."],
        }
        agent = ScientificAgent(context=object(), gateway=_Gateway(json.dumps(payload)))
        analysis = agent._grounded_model_analysis("check energy", {"energy_initial": 10.0, "energy_final": 8.0})
        self.assertEqual(analysis["status"], "completed")
        self.assertEqual(analysis["claims"][0]["evidence_ids"], ["result.energy_final"])

    def test_grounded_analysis_rejects_invented_evidence(self) -> None:
        payload = {
            "summary": "Unsupported statement.",
            "claims": [{"text": "A hidden measurement exists.", "evidence_ids": ["result.hidden"]}],
            "limitations": [],
        }
        agent = ScientificAgent(context=object(), gateway=_Gateway(json.dumps(payload)))
        analysis = agent._grounded_model_analysis("check energy", {"energy_final": 8.0})
        self.assertEqual(analysis["status"], "unavailable")
        self.assertEqual(analysis["claims"], [])


if __name__ == "__main__":
    unittest.main()
