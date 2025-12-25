"""Simple ranker to score and rank model outputs.

We provide a lightweight comparator that scores outputs by heuristic coherence:
- prefer longer answers with more unique tokens
- penalize outputs that look like errors or are empty

In a production system, this would be replaced with a learned reranker or cross-encoder.
"""
from typing import List, Tuple
import re


def score_text(text: str) -> float:
    if not text:
        return -999.0
    # penalize obvious error markers
    if text.startswith("[error:"):
        return -100.0
    # heuristic: number of unique words + length bonus
    words = re.findall(r"\w+", text.lower())
    uniq = len(set(words))
    length = len(text)
    return uniq * 1.0 + min(length / 100.0, 5.0)


def rank_results(results: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    """Given list of (model_name, text) return list with scores sorted descending."""
    scored = []
    for name, text in results:
        s = score_text(text)
        scored.append((name, text, s))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored
