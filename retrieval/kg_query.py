import json
import networkx as nx
from networkx.readwrite import json_graph

import os
from typing import List, Dict, Optional

# By default look for kg_export.json in the same folder as this file
KG_PATH = os.path.join(os.path.dirname(__file__), "kg_export.json")


class KGQuery:
    """Offline KG query helper: loads a prebuilt KG (node-link JSON) and
    exposes traversal and normalized-context extraction functions.

    This class behaviour is unchanged from the repository's original
    implementation but remains intentionally small so the online helper
    (below) can reuse its normalization conventions.
    """

    def __init__(self, kg_path: str = KG_PATH):
        self.graph = None
        try:
            self.graph = self.load_graph(kg_path)
        except FileNotFoundError:
            # Graph file may not exist in all deployments; callers should
            # handle the None case gracefully.
            self.graph = None

    def load_graph(self, path: str) -> nx.DiGraph:
        """Load a saved KG from JSON export."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json_graph.node_link_graph(data, directed=True)

    # --- Core Query Functions ---

    def get_neighbors(self, entity: str, depth: int = 1) -> List[str]:
        """Return neighbors up to a given depth."""
        if not self.graph or entity not in self.graph:
            return []
        nodes = set([entity])
        for _ in range(depth):
            neighbors = []
            for n in list(nodes):
                neighbors += list(self.graph.successors(n)) + list(self.graph.predecessors(n))
            nodes.update(neighbors)
        return list(nodes - {entity})

    def find_relation(self, src: str, tgt: str):
        """Return relation(s) between two entities if exists."""
        if not self.graph:
            return None
        if self.graph.has_edge(src, tgt):
            return self.graph[src][tgt]
        return None

    def get_context_for_entity(self, entity: str) -> List[Dict]:
        """Collect all text chunks associated with an entity (via edges).

        Returns a list of retrieval-like dicts with keys 'text' and 'meta'.
        """
        contexts: List[Dict] = []
        if not self.graph or entity not in self.graph:
            return contexts
        for nbr in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, nbr)
            if not edge_data:
                continue
            # Normalize the returned context so generators can treat it like a retrieval hit
            contexts.append({
                "text": edge_data.get("chunk_text", ""),
                "meta": {
                    "entity": entity,
                    "relation": edge_data.get("relation", ""),
                    "target": nbr,
                    "chunk_id": edge_data.get("chunk_id"),
                    **{k: v for k, v in edge_data.items() if k not in ["relation", "chunk_id", "chunk_text"]}
                }
            })
        return contexts

    def expand_context_from_query(self, query_entities: List[str], depth: int = 1) -> List[Dict]:
        """Given list of entities, expand context with KG traversal.

        Returns a list of normalized retrieval-like dicts containing 'text' and 'meta'.
        """
        context: List[Dict] = []
        for ent in query_entities:
            # include direct contexts
            ctxs = self.get_context_for_entity(ent)
            for c in ctxs:
                context.append(c)
            # include neighbors' contexts as a mild expansion
            neighbors = self.get_neighbors(ent, depth=depth)
            for n in neighbors:
                # include contexts where neighbor is source
                ctxs_n = self.get_context_for_entity(n)
                for c in ctxs_n:
                    # mark expanded context
                    c_exp = dict(c)
                    c_exp['meta'] = dict(c_exp.get('meta', {}))
                    c_exp['meta']['expanded_from'] = ent
                    context.append(c_exp)
        return context


# --- Online (real-time) KG querying helper ---
class OnlineKGQuery:
    """Helper to perform lightweight, online KG-style extraction from live web pages.

    Motivation:
    - When the pipeline detects an "internet-answerable" query but there are no
      pre-chunked local pages, this helper fetches pages (or accepts URLs),
      extracts relations using spaCy dependency rules (same as the builder),
      and returns normalized contexts in the same {'text','meta'} shape as the
      offline KG query.

    Notes:
    - This module performs live HTTP requests using `requests` and parses HTML
      with `beautifulsoup4`. If those packages are missing, the helper will
      raise a clear ImportError with instructions.
    - For production-grade search you should provide a search function (e.g.
      an API wrapper for Bing/Google) to resolve query -> URLs. The helper
      supports a pluggable `search_func` parameter.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        # lazy imports
        try:
            import requests
            from bs4 import BeautifulSoup
        except Exception as e:
            raise ImportError("OnlineKGQuery requires 'requests' and 'beautifulsoup4' packages. Install them with: pip install requests beautifulsoup4") from e

        try:
            import spacy
            # reuse the same lightweight model used in the builder; loading is lazy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise ImportError("OnlineKGQuery requires spaCy and the en_core_web_sm model. Install with: pip install spacy && python -m spacy download en_core_web_sm") from e

        # local imports saved to instance
        self.requests = requests
        self.BeautifulSoup = BeautifulSoup
        # reuse helper functions from kg_builder if available
        try:
            from .kg_builder import extract_svo_relations
            self._extract_relations = extract_svo_relations
        except Exception:
            # fallback: simple sentence-level contexts if builder isn't importable
            self._extract_relations = None

        # simple in-memory cache; optionally write to disk in future
        self._page_cache = {}
        self.cache_dir = cache_dir

    def fetch_page_text(self, url: str, timeout: int = 10) -> str:
        """Fetch a page and return its visible textual content.

        This function attempts to extract <article> content first, then falls
        back to <main> or the full <body> text. It strips scripts/styles and
        returns a cleaned string truncated to a reasonable size.
        """
        if url in self._page_cache:
            return self._page_cache[url]

        resp = self.requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (compatible)"})
        resp.raise_for_status()
        soup = self.BeautifulSoup(resp.text, "html.parser")

        # remove scripts/styles
        for s in soup(['script', 'style', 'noscript', 'iframe']):
            s.extract()

        # prefer article > main > body
        text_blocks = []
        article = soup.find('article')
        if article:
            text_blocks.append(article.get_text(separator=' ', strip=True))
        main = soup.find('main')
        if main:
            text_blocks.append(main.get_text(separator=' ', strip=True))
        body = soup.find('body')
        if body:
            text_blocks.append(body.get_text(separator=' ', strip=True))

        text = '\n'.join([t for t in text_blocks if t])
        # normalize whitespace
        text = ' '.join(text.split())
        # cache and return
        self._page_cache[url] = text
        return text

    def extract_contexts_from_text(self, text: str, source: Optional[str] = None, max_relations: int = 20) -> List[Dict]:
        """Run spaCy parsing on text and return relation contexts in normalized form.

        If the project kg_builder.extract_svo_relations is available we call it
        per document; otherwise, we fall back to returning the top sentences.
        """
        docs = []
        if not text:
            return docs

        doc = self.nlp(text)
        chunk_metadata = {"source_url": source}

        if self._extract_relations is not None:
            relations = self._extract_relations(doc, chunk_metadata)
            # relations is expected to be list of (subj, obj, attrs)
            for i, (src, tgt, attrs) in enumerate(relations):
                if i >= max_relations:
                    break
                contexts = attrs.get('chunk_text') if attrs.get('chunk_text') else ''
                # prefer to build a short sentence around the verb using token.sent
                text_snippet = ''
                if 'chunk_text' in attrs and attrs['chunk_text']:
                    text_snippet = attrs['chunk_text']
                else:
                    # find a sentence containing either subject or object
                    for sent in doc.sents:
                        s = sent.text
                        if src in s or tgt in s:
                            text_snippet = s
                            break
                docs.append({
                    'text': text_snippet or f"{src} {attrs.get('relation','')} {tgt}",
                    'meta': {
                        'source_url': source,
                        'subject': src,
                        'object': tgt,
                        **attrs
                    }
                })
        else:
            # fallback: return top sentences up to max_relations
            for i, sent in enumerate(doc.sents):
                if i >= max_relations:
                    break
                docs.append({'text': sent.text, 'meta': {'source_url': source}})

        return docs

    def query_urls(self, urls: List[str], max_pages: int = 5, max_relations_per_page: int = 20) -> List[Dict]:
        """Given a list of URLs, fetch and extract contexts from each.

        Returns a deduplicated list of normalized {'text','meta'} dicts.
        """
        contexts: List[Dict] = []
        seen_texts = set()
        for i, url in enumerate(urls):
            if i >= max_pages:
                break
            try:
                page_text = self.fetch_page_text(url)
            except Exception as e:
                # skip failures but record minimal meta so callers know
                contexts.append({'text': '', 'meta': {'source_url': url, 'error': str(e)}})
                continue

            extracted = self.extract_contexts_from_text(page_text, source=url, max_relations=max_relations_per_page)
            for ctx in extracted:
                t = ctx.get('text', '').strip()
                if not t:
                    continue
                key = t[:300]
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                contexts.append(ctx)
        return contexts

    def query_with_search(self, query: str, search_func=None, num_urls: int = 5, **kwargs) -> List[Dict]:
        """High-level convenience: resolve `query` to URLs using `search_func`
        (callable) and then query the returned URLs.

        `search_func` should accept (query, num_results) and return a list of URLs.
        If no search_func is provided this method raises an informative error so
        the pipeline can plug in a site-specific or API-backed search implementation.
        """
        if not search_func:
            raise RuntimeError("No search function provided. Provide search_func(query, num_results)->List[urls] or call query_urls(urls=...) directly.")
        urls = search_func(query, num_urls)
        return self.query_urls(urls, **kwargs)


# --- Example offline/online usage ---
if __name__ == "__main__":
    print("KGQuery example (offline) vs OnlineKGQuery (fetch pages)")
    # offline example (will silently handle missing file)
    kgq = KGQuery()
    if kgq.graph:
        print("Neighbors of 'Albert Einstein':", kgq.get_neighbors("Albert Einstein", depth=2))
    else:
        print("No offline KG file found; use OnlineKGQuery for live pages.")

    # online example: only run if requests is installed and internet available
    try:
        okg = OnlineKGQuery()
        # small demo using a couple of well-known static pages (not run here)
        demo_urls = [
            "https://en.wikipedia.org/wiki/Albert_Einstein",
        ]
        print("Fetching demo URL(s)...")
        contexts = okg.query_urls(demo_urls, max_pages=1, max_relations_per_page=10)
        print("Sample online contexts:", contexts[:3])
    except Exception as e:
        print("OnlineKGQuery not available:", e)
