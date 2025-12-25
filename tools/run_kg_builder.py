"""Small helper to run the KG builder and save the export into retrieval/kg_export.json

This script loads the extractor JSON, builds the KG using retrieval/kg_builder.py,
and writes the node-link JSON to retrieval/kg_export.json so the pipeline can use it.

Usage:
    python tools/run_kg_builder.py
"""
import os
import sys

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.append(proj_root)

from retrieval import kg_builder

EXTRACTOR_PATH = os.path.join(proj_root, 'extractor', 'chunks_class_9_English_Beehive.json')
OUT_PATH = os.path.join(proj_root, 'retrieval', 'kg_export.json')

if __name__ == '__main__':
    print(f"Building KG from {EXTRACTOR_PATH}...")
    G = kg_builder.build_knowledge_graph(EXTRACTOR_PATH)
    if G is None:
        print("KG build failed or extractor file missing.")
        sys.exit(1)
    kg_builder.save_graph(G, OUT_PATH)
    print(f"KG saved to {OUT_PATH}")
