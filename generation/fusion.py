"""Fusion module: combine multiple generated outputs into a final answer.

We provide two simple strategies:
- 'best' => return the top-ranked answer
- 'merge' => attempt to combine unique sentences from top-N answers

This is intentionally simple and deterministic for unit tests.
"""
from typing import List, Tuple
import re


def best_fusion(scored: List[Tuple[str, str, float]]) -> str:
    if not scored:
        return ""
    # scored is expected sorted desc
    return scored[0][1]


def merge_fusion(scored: List[Tuple[str, str, float]], top_n: int = 2) -> str:
    if not scored:
        return ""
    texts = [t for _, t, _ in scored[:top_n]]
    # naive merge: split into sentences and keep unique ones preserving order
    seen = set()
    out_sentences = []
    for txt in texts:
        sents = re.split(r'(?<=[.!?])\s+', txt.strip())
        for s in sents:
            norm = s.strip()
            if not norm:
                continue
            if norm not in seen:
                seen.add(norm)
                out_sentences.append(norm)
    return ' '.join(out_sentences)


def fuse(scored: List[Tuple[str, str, float]], strategy: str = 'best') -> str:
    strategy = strategy.lower()
    if strategy == 'merge':
        return merge_fusion(scored)
    return best_fusion(scored)
