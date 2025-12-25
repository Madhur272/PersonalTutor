"""Demo runner: run the full pipeline for an example query and print per-model outputs,
ranked results, and fused output.

Usage:
    python tools/demo_pipeline.py

This script is offline-safe: it will use LocalEchoGenerator when OpenRouter key is not present,
and will not perform aggressive network calls unless retrieval returns no local docs and
OnlineKGQuery fallback is triggered (which requires requests/bs4/spacy).
"""
import os
import sys
import json

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.append(proj_root)

# pipeline and retrieval
from pipeline import segment_pipeline as sp
from preprocessing.build_retriever import hybrid_search
from retrieval.kg_query import KGQuery

# generation components
from generation.generator_pool import GeneratorPool
from generation.ranker import rank_results
from generation.fusion import fuse


def demo_query(query: str = None):
    query = query or "What is photosynthesis?"
    print(f"\n=== Demo: running pipeline for query: {query}\n")

    # classify
    seg_info = sp.seg_inst.segment(query)
    segment = seg_info.get('segment', 'factual')
    print(f"Segment: {segment} (confidence={seg_info.get('confidence')})")

    # retrieval
    retrieved = []
    try:
        retrieved = hybrid_search(query, k=5) or []
    except Exception as e:
        print(f"Retrieval error: {e}")
    print(f"Retrieved {len(retrieved)} local chunks.")

    # KG augmentation
    kgq = KGQuery()
    entities = sp.extract_entities_from_query(query)
    kg_contexts = []
    if entities and kgq.graph:
        kg_contexts = kgq.expand_context_from_query(entities, depth=1)
    print(f"KG contexts found: {len(kg_contexts)}")

    # normalize contexts -> list of texts
    ctx_texts = []
    for r in (retrieved or []) + (kg_contexts or []):
        if isinstance(r, dict) and 'text' in r:
            ctx_texts.append(r['text'])
        else:
            ctx_texts.append(str(r))

    # generation via generator pool (get per-model outputs)
    cfg = {
        'openrouter_key': os.getenv('OPENROUTER_KEY'),
        'random_seed': 42
    }
    pool = GeneratorPool(config=cfg)
    multi_out = pool.run_all(query, ctx_texts, segment, max_tokens=200)

    print('\n--- Per-model raw outputs ---')
    for name, text in multi_out:
        print(f"\n[{name}]\n{text[:600]}\n")

    # ranking
    scored = rank_results(multi_out)
    print('\n--- Ranked outputs (score descending) ---')
    for name, text, score in scored:
        print(f"{score:.2f}\t{name}\t{text[:200]}")

    # fusion
    strategy = os.getenv('GENERATION_FUSION', 'best')
    fused = fuse(scored, strategy=strategy)
    print('\n--- Fused final answer ---\n')
    print(fused)

    # also show pipeline final output (end-to-end)
    print('\n--- Pipeline.run_pipeline final answer (full flow) ---\n')
    try:
        final = sp.run_pipeline(query, search_func=None)
        print(final)
    except Exception as e:
        print(f"Pipeline execution error: {e}")


if __name__ == '__main__':
    demo_query()
