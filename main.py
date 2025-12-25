import os
import sys
import json
from typing import List, Dict

# ensure project modules are importable when running this script directly
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Lightweight chat memory
class ChatMemory:
    """Chat memory with optional embedding-based retrieval, auto-summary snapshot, and persistence.

    Features:
    - persist_path: optional JSON file to load/save turns
    - use_embeddings: lazy-loads SentenceTransformer when first needed
    - summarization_threshold: when number of turns exceeds this, create a short snapshot summary
    - summary_keeps: when summarizing, keep last N turns plus summary
    """

    def __init__(self, persist_path: str = None, use_embeddings: bool = True, embedding_model: str = "all-MiniLM-L6-v2", summarization_threshold: int = 50, summary_keeps: int = 6):
        self.turns: List[Dict] = []
        self.persist_path = persist_path
        self._use_embeddings = use_embeddings
        self._embedding_model_name = embedding_model
        self._embedder = None  # lazy loaded
        self._embeddings = []  # parallel list of numpy arrays for each stored turn (user+assistant combined)
        self.summarization_threshold = summarization_threshold
        self.summary_keeps = summary_keeps

        if persist_path and os.path.exists(persist_path):
            try:
                with open(persist_path, 'r', encoding='utf-8') as f:
                    self.turns = json.load(f)
            except Exception:
                self.turns = []

    # --- Embedding utilities (lazy) ---
    def _load_embedder(self):
        if not self._use_embeddings:
            return None
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                self._np = np
                self._embedder = SentenceTransformer(self._embedding_model_name)
            except Exception as e:
                print(f"⚠️ Embedding model load failed: {e} — embedding features disabled.")
                self._use_embeddings = False
                self._embedder = None
        return self._embedder

    def _text_for_turn(self, turn: Dict) -> str:
        """Create a single text string for a turn suitable for embedding/search."""
        parts = []
        if 'user' in turn:
            parts.append(f"User: {turn['user']}")
        if 'assistant' in turn:
            if 'model' in turn:
                parts.append(f"Assistant ({turn['model']}): {turn['assistant']}")
            else:
                parts.append(f"Assistant: {turn['assistant']}")
        if 'summary' in turn:
            parts.append(f"Summary: {turn['summary']}")
        return "\n".join(parts)

    def _ensure_embeddings_for_all(self):
        """Ensure we have embeddings for all current turns; compute any missing ones."""
        if not self._use_embeddings:
            return
        embedder = self._load_embedder()
        if embedder is None:
            return
        # compute embeddings for any turns missing them
        if len(self._embeddings) == len(self.turns):
            return
        texts = [self._text_for_turn(t) for t in self.turns]
        try:
            embs = embedder.encode(texts, convert_to_numpy=True)
            self._embeddings = [e for e in embs]
        except Exception as e:
            print(f"⚠️ Failed to compute embeddings: {e}")
            self._use_embeddings = False
            self._embeddings = []

    # --- Basic operations ---
    def add_user(self, text: str):
        self.turns.append({'user': text})
        # embeddings will be computed lazily on retrieval
        # possibly trigger summarization
        self._auto_maybe_summarize()

    def add_assistant(self, text: str, model: str = None):
        last = self.turns[-1] if self.turns and 'user' in self.turns[-1] and 'assistant' not in self.turns[-1] else None
        entry = {'assistant': text}
        if model:
            entry['model'] = model
        if last is None:
            self.turns.append(entry)
        else:
            last.update(entry)
        # keep embeddings lazy
        self._auto_maybe_summarize()

    def get_all(self) -> List[Dict]:
        return list(self.turns)

    def get_recent(self, n: int = 10) -> List[Dict]:
        return list(self.turns[-n:])

    def clear(self):
        self.turns = []
        self._embeddings = []

    def save(self):
        if not self.persist_path:
            return
        try:
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(self.turns, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to persist chat memory: {e}")

    # --- Retrieval / summarization features ---
    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return the top_k most relevant turns (as dicts) for the query.

        Falls back to simple recency if embedding support unavailable.
        """
        if not self.turns:
            return []
        if not self._use_embeddings:
            # return latest turns as fallback
            return self.get_recent(top_k)

        self._ensure_embeddings_for_all()
        if not self._embeddings:
            return self.get_recent(top_k)

        try:
            embedder = self._load_embedder()
            q_emb = embedder.encode([query], convert_to_numpy=True)[0]
            # cosine similarity
            vecs = self._embeddings
            import numpy as _np
            sims = [_np.dot(q_emb, v) / (_np.linalg.norm(q_emb) * (_np.linalg.norm(v) + 1e-10)) for v in vecs]
            idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
            return [self.turns[i] for i in idxs]
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return self.get_recent(top_k)

    def summarize_memory(self) -> str:
        """Create a short summary snapshot from the current memory using the project's summarization helper.

        Returns the summary string and inserts it as a special turn {'summary': text}.
        """
        if not self.turns:
            return ""
        try:
            # reuse generation.summarize_context to create friendly student summary
            from generation.generate_answers import summarize_context
            texts = [self._text_for_turn(t) for t in self.turns]
            # chunk long lists into smaller groups to avoid hitting token limits
            summary = summarize_context(texts)
            # attach summary as its own turn (system note)
            self.turns.insert(0, {'summary': summary})
            # trim if memory too large: keep summary + last N turns
            keeps = self.summary_keeps
            self.turns = [t for t in self.turns if 'summary' in t] + self.turns[-keeps:]
            # reset embeddings (they will be recomputed lazily)
            self._embeddings = []
            return summary
        except Exception as e:
            print(f"⚠️ Summarization failed: {e}")
            return ""

    def _auto_maybe_summarize(self):
        """Automatically summarize when memory exceeds threshold."""
        try:
            if self.summarization_threshold and len(self.turns) >= self.summarization_threshold:
                print("🔖 Chat memory large; creating summary snapshot...")
                self.summarize_memory()
                # persist if configured
                self.save()
        except Exception as e:
            print(f"⚠️ Auto-summarize error: {e}")


def initialize_server(persist_path: str = None):
    """Initialize heavy components (retriever, KG) and return a server dict.

    Returns a dict with keys: 'mem', 'run_pipeline', 'kgq', 'gen_mod'.
    """
    print("Loading retriever and knowledge graph (this may take a few seconds)...")
    # lazy init retriever (may download models on first run)
    try:
        from preprocessing.build_retriever import init_retriever
        init_retriever()
        print("✅ Retriever initialized.")
    except Exception as e:
        print(f"⚠️ Retriever init failed: {e}")

    # Try to load KGQuery (optional)
    kgq = None
    try:
        from retrieval.kg_query import KGQuery
        kgq = KGQuery()
        print("✅ Knowledge graph loaded.")
    except Exception as e:
        print(f"⚠️ KG load failed: {e} — continuing without KG augmentation.")
        kgq = None

    # pipeline
    run_pipeline = None
    try:
        from pipeline.segment_pipeline import run_pipeline as rp
        run_pipeline = rp
    except Exception as e:
        print(f"❌ Failed to import pipeline: {e}")

    # generation meta accessor
    try:
        import generation.generate_answers as gen_mod
    except Exception:
        gen_mod = None

    mem_path = os.path.join(ROOT, 'chat_memory.json') if persist_path is None else persist_path
    mem = ChatMemory(persist_path=mem_path)

    return {"mem": mem, "run_pipeline": run_pipeline, "kgq": kgq, "gen_mod": gen_mod}


def handle_query(server: dict, query: str):
    """Handle a single query using the provided server dict and update the ChatMemory.

    Returns (reply_str, meta_dict).
    """
    if not server or 'mem' not in server:
        raise ValueError("server must be initialized via initialize_server()")
    mem = server['mem']
    run_pipeline = server.get('run_pipeline')
    gen_mod = server.get('gen_mod')

    mem.add_user(query)
    try:
        if run_pipeline is not None:
            reply = run_pipeline(query, chat_history=mem.get_all())
        else:
            from pipeline.segment_pipeline import run_pipeline as rp
            reply = rp(query, chat_history=mem.get_all())
    except Exception as e:
        # Log the detailed error server-side but return a concise, user-friendly message
        import traceback as _tb
        tb = _tb.format_exc()
        print(f"⚠️ Pipeline exception while handling query: {e}\n{tb}")
        reply = "⚠️ Generative backend error: external model request failed. Please try again later."
        # include error info in model_info/meta if the caller wants diagnostics
        model_info = {"error": str(e)}

    # Try to capture generation meta if available
    model_info = None
    if gen_mod is not None and hasattr(gen_mod, 'LAST_GENERATION_META'):
        try:
            model_info = gen_mod.LAST_GENERATION_META
        except Exception:
            model_info = None

    mem.add_assistant(reply, model=(model_info.get('model') if isinstance(model_info, dict) and model_info.get('model') else (','.join(model_info.get('models')) if isinstance(model_info, dict) and model_info.get('models') else None)))

    return reply, (model_info if model_info is not None else {})


def main():
    # keep the interactive main behavior but reuse shared initialization
    server = initialize_server()
    if server is None:
        print("❌ Failed to initialize server components; exiting.")
        return

    mem = server.get('mem')
    run_pipeline = server.get('run_pipeline')
    gen_mod = server.get('gen_mod')

    print("\nInteractive assistant ready. Commands: /exit /clear /show /save\n")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Saving memory...")
            mem.save()
            break
        if not q:
            continue
        if q.lower() in {"/exit", "exit", "quit", ":q"}:
            print("Exiting. Saving memory...")
            mem.save()
            break
        if q.lower() in {"/clear", "/clear"}:
            mem.clear()
            print("Chat memory cleared.")
            continue
        if q.lower() in {"/show", "/show"}:
            for i, t in enumerate(mem.get_all()):
                print(f"[{i}] {t}")
            continue
        if q.lower() in {"/save", "/save"}:
            mem.save()
            print(f"Saved chat memory to {mem.persist_path}")
            continue

        # Normal query
        mem.add_user(q)
        # Pass the entire memory to the pipeline so segmenter and LLMs can access it
        try:
            # prefer run_pipeline via server if available
            if run_pipeline is not None:
                reply = run_pipeline(q, chat_history=mem.get_all())
            else:
                from pipeline.segment_pipeline import run_pipeline as rp
                reply = rp(q, chat_history=mem.get_all())
        except Exception as e:
            reply = f"⚠️ Pipeline error: {e}"
        # Try to capture generation meta if available
        model_info = None
        if gen_mod is not None and hasattr(gen_mod, 'LAST_GENERATION_META'):
            try:
                model_info = gen_mod.LAST_GENERATION_META
            except Exception:
                model_info = None
        # record assistant reply and model info
        mem.add_assistant(reply, model=(model_info.get('model') if isinstance(model_info, dict) and model_info.get('model') else (','.join(model_info.get('models')) if isinstance(model_info, dict) and model_info.get('models') else None)))

        print(f"Assistant: {reply}\n")

    print("Bye!")


if __name__ == '__main__':
    main()
