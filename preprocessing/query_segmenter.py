"""Compatibility wrapper that exposes QuerySegmenter for tests.

This module delegates rule-based checks to the retrieval-side helpers
and exposes a `_zero_shot_classify` method that tests can patch.
"""
import os
import sys
from typing import Optional, List, Dict, Any
from importlib.machinery import SourceFileLoader

# Dynamically load the retrieval/query_segmenter.py as a distinct module
retrieval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'retrieval', 'query_segmenter.py'))
retr_mod = SourceFileLoader('retrieval_query_segmenter', retrieval_path).load_module()
# import helpers from the loaded module
rule_based_segment = getattr(retr_mod, 'rule_based_segment')
extract_slots = getattr(retr_mod, 'extract_slots')
zero_shot_segment = getattr(retr_mod, 'zero_shot_segment')


class QuerySegmenter:
    def __init__(self, confidence_threshold: float = 0.6, use_gpu: bool = False):
        self.confidence_threshold = confidence_threshold

    def _zero_shot_classify(self, query: str) -> Optional[Dict[str, Any]]:
        """Delegates to retrieval.zero_shot_segment by default.

        Tests can patch this method to simulate low/high confidence outputs.
        """
        try:
            seg, conf = zero_shot_segment(query)
            return {"segment": seg, "confidence": conf}
        except Exception:
            return None

    def segment(self, query: str, chat_history: Optional[List[str]] = None) -> Dict[str, Any]:
        # Rule-based first
        rule = rule_based_segment(query, chat_history)
        slots = extract_slots(query, chat_history)
        if rule:
            return {"original_query": query, "normalized_query": query.strip().lower(), "segment": rule, "confidence": 0.99, "slots": slots}

        # Zero-shot (patchable)
        zs = self._zero_shot_classify(query)
        if zs:
            seg = zs.get("segment")
            conf = zs.get("confidence", 0.0)
            # Normalize legacy label
            if seg == 'factual_qa':
                seg = 'factual'
            if conf < self.confidence_threshold:
                return {"original_query": query, "normalized_query": query.strip().lower(), "segment": "factual", "confidence": conf, "slots": slots}
            return {"original_query": query, "normalized_query": query.strip().lower(), "segment": seg, "confidence": conf, "slots": slots}

        # Heuristic fallback: W-word -> factual
        if any(w in query.lower() for w in ["who","what","where","when","why","how","which"]):
            return {"original_query": query, "normalized_query": query.strip().lower(), "segment": "factual", "confidence": 0.5, "slots": slots}

        return {"original_query": query, "normalized_query": query.strip().lower(), "segment": "explain", "confidence": 0.4, "slots": slots}

