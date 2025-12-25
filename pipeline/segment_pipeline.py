import os
import sys
import spacy
# Import the compatibility QuerySegmenter wrapper from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query_segmenter import QuerySegmenter
# Retriever and generation helpers
# Ensure generation path is available before importing modules that import generate_answers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'generation')))
import generate_answers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing')))
import preprocessing.build_retriever as build_retriever
# KG utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'retrieval')))
from kg_query import KGQuery
# Note: OnlineKGQuery is imported lazily inside retrieve_and_augment to avoid
# import-time heavy dependencies (requests/bs4/spacy) when not needed.

nlp = spacy.load("en_core_web_sm")

# instantiate a single QuerySegmenter to use across the pipeline
seg_inst = QuerySegmenter()

# Example: pass a site-specific adapter (e.g. Byju's) as `search_func` to
# enable online fallback when local retrieval returns no docs. The adapter
# should follow the signature: func(query: str, num_results: int) -> List[str]
#
# from retrieval.search_adapters import byjus_search
# response = run_pipeline("what is photosynthesis", search_func=byjus_search)

# Lazy init KGQuery (some environments may not have the KG file)
try:
    kgq = KGQuery()
except Exception as e:
    print(f"⚠️ KGQuery init failed: {e} — KG augmentation will be disabled.")
    kgq = None

def extract_entities_from_query(query: str):
    """Simple entity extractor using spaCy NER."""
    doc = nlp(query)
    return [ent.text for ent in doc.ents]

def run_pipeline(query, chat_history: list = None, search_func=None):
    """Main pipeline entry.

    Uses the unified QuerySegmenter interface (seg_inst.segment) to decide routing.
    """
    # Step 1: classify query (use unified segmenter)
    seg_info = seg_inst.segment(query, chat_history=chat_history)
    segment = seg_info.get("segment")
    slots = seg_info.get("slots", {})
    confidence = seg_info.get("confidence", 0.0)
    print(f"🧭 Query classified into segment: {segment} (conf={confidence:.2f})")

    # Helper: safe retrieval + KG augmentation
    def retrieve_and_augment(q, k=5, search_func_local=None):
        docs = []
        try:
            # call through to the module attribute so tests can monkeypatch
            docs = build_retriever.hybrid_search(q, k=k) or []
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            docs = []

        # If no docs found locally, optionally fallback to online KG query first
        if not docs:
            # prefer the locally-passed search_func, then pipeline-level search_func
            sf = search_func_local if search_func_local is not None else search_func
            if sf is not None:
                try:
                    # import OnlineKGQuery lazily to avoid heavy deps at module import
                    from retrieval.kg_query import OnlineKGQuery
                    okg = OnlineKGQuery()
                    online_ctxs = okg.query_with_search(q, search_func=sf, num_urls=5)
                    # append normalized contexts
                    for c in online_ctxs:
                        # ensure shape
                        if isinstance(c, dict) and 'text' in c and 'meta' in c:
                            docs.append(c)
                        else:
                            docs.append({'text': str(c), 'meta': {}})
                except Exception as e:
                    print(f"⚠️ Online KG fallback failed: {e}")

        # KG augmentation when entities exist (run after online fallback so tests using search_func get priority)
        if not docs:
            try:
                entities = extract_entities_from_query(q)
                if entities:
                    kg_context = []
                    if kgq and getattr(kgq, 'expand_context_from_query', None):
                        kg_context = kgq.expand_context_from_query(entities, depth=2) or []
                    # Normalize KG contexts into retriever-like dicts: {'text': ..., 'meta': {...}}
                    normalized_kg_docs = []
                    for c in kg_context:
                        if isinstance(c, dict) and 'text' in c and 'meta' in c:
                            normalized_kg_docs.append(c)
                        else:
                            # best-effort conversion
                            normalized_kg_docs.append({
                                'text': c.get('text', '') if isinstance(c, dict) else str(c),
                                'meta': c.get('meta', {}) if isinstance(c, dict) else {}
                            })

                    # Deduplicate by chunk_id then by text
                    seen = set()
                    for d in normalized_kg_docs:
                        cid = d['meta'].get('chunk_id')
                        key = cid if cid is not None else d['text'][:200]
                        if key in seen:
                            continue
                        seen.add(key)
                        docs.append(d)
            except Exception as e:
                print(f"⚠️ KG augmentation error: {e}")

        return docs

    # Step 2: routing
    retrieval_driven = {"factual", "compare", "explain", "explanation", "definition", "procedural", "multi_hop", "translation", "creative", "faq", "faq_lookup"}

    # Handle follow-up queries: convert to factual retrieval by augmenting
    # the query with the previous conversational turn when available.
    query_for_retrieval = query
    if segment == 'follow_up':
        # attempt to recover the prior user/assistant turn from chat_history
        try:
            if chat_history and isinstance(chat_history, list) and len(chat_history) >= 2:
                # the last item is the current user (added by main); take the item before it
                prev = chat_history[-2]
                prev_text = None
                if isinstance(prev, dict):
                    prev_text = prev.get('assistant') or prev.get('user')
                else:
                    prev_text = str(prev)
                if prev_text:
                    query_for_retrieval = prev_text + ' ; follow-up: ' + query
            # downgrade segment to factual for routing purposes
            segment = 'factual'
        except Exception:
            query_for_retrieval = query

    # Summarization
    if segment in {"summary", "summarization"}:
        retrieved_chunks = retrieve_and_augment(query, k=5)
        if not retrieved_chunks:
            return "⚠️ No documents found to summarize."
        # Ask LLM to summarize the retrieved context
        try:
            # use the dedicated summarization helper
            return generate_answers.summarize_context(retrieved_chunks)
        except Exception as e:
            print(f"⚠️ Generation error (summarization): {e}")
            return "⚠️ Summarization failed."

    # Grammar correction
    if segment == "grammar":
        # Use the grammar correction helper if available
        try:
            return generate_answers.grammar_correct(query)
        except Exception as e:
            print(f"⚠️ Grammar correction error: {e}")
            return f"[Grammar correction unavailable] Original: {query}"

    # Image / diagram handling
    if segment in {"image_based", "diagram", "image"}:
        # Try an image/diagram understanding helper which may use retrieved context
        try:
            retrieved_chunks = retrieve_and_augment(query, k=5)
        except Exception:
            retrieved_chunks = []
        try:
            return generate_answers.image_understanding(query, retrieved_chunks)
        except Exception as e:
            print(f"⚠️ Image pipeline error: {e}")
            return "[Image/diagram pipeline not available]"

    # FAQ lookup (simple placeholder)
    if segment == "faq":
        # Try FAQ-specific answer generator that may use retrieved FAQs or docs
        retrieved_chunks = retrieve_and_augment(query, k=5)
        try:
            return generate_answers.faq_answer(query, retrieved_chunks)
        except Exception as e:
            print(f"⚠️ FAQ handler error: {e}")
            return f"[FAQ lookup unavailable] Try searching FAQ for: {query}"

    # Retrieval-driven generation
    if segment in retrieval_driven:
        retrieved_chunks = retrieve_and_augment(query, k=5)
        if not retrieved_chunks:
            return "⚠️ No evidence retrieved for this query."

        # Step 3: generation (with error handling)
        try:
            # Some tests monkeypatch generate_answer with a function that doesn't
            # accept chat_history; detect the callable signature and call
            # accordingly to remain backwards-compatible.
            import inspect
            gen_fn = generate_answers.generate_answer
            try:
                sig = inspect.signature(gen_fn)
                if 'chat_history' in sig.parameters:
                    answer = gen_fn(query, retrieved_chunks, chat_history=chat_history)
                else:
                    answer = gen_fn(query, retrieved_chunks)
            except (ValueError, TypeError):
                # If inspect fails, fall back to calling with chat_history and allow
                # the exception to be caught below.
                answer = gen_fn(query, retrieved_chunks, chat_history=chat_history)

            if not answer or not answer.strip():
                return "⚠️ Generative model returned empty answer."
            return answer
        except TypeError as e:
            # specific handling for unexpected keyword from older test stubs
            print(f"⚠️ Generation error: {e}")
            try:
                # try calling without chat_history as a final fallback
                answer = generate_answers.generate_answer(query, retrieved_chunks)
                if not answer or not answer.strip():
                    return "⚠️ Generative model returned empty answer."
                return answer
            except Exception as e2:
                print(f"⚠️ Generation fallback error: {e2}")
                return "⚠️ Failed to generate an answer."
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return "⚠️ Failed to generate an answer."

    # Default fallback
    return "❌ Sorry, I couldn't handle your request."


if __name__ == "__main__":
    print("\n--- Segment Pipeline Ready ---")
    while True:
        user_query = input("\nAsk me something (or type 'exit'): ")
        if user_query.lower() == "exit":
            break
        response = run_pipeline(user_query)
        print(f"\n🤖 Final Answer: {response}")